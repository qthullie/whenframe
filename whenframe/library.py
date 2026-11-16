"""A video library on disk: which videos were indexed, and their frame embeddings.

Layout of a library folder:

    library.sqlite   one row per video (path, size, mtime, duration, where its frames are)
    frames.f16       all frame embeddings, float16, appended in indexing order
    times.f32        the instant (seconds) each embedding stands for
    offsets.f32      each frame's query-bank offset (see bank.py), filled in at the first search
    config.json      the model and sampling settings the library was built with

Embeddings are only ever appended. A re-indexed or deleted video leaves dead
rows behind, which `compact()` removes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import bank as qbank
from .encoder import DEFAULT_MODEL, Encoder
from .video import RepeatFilter, VideoError, iter_frames, probe, thumbnails


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    fps: float = 1.0
    dedup: float = 4.0  # 0 encodes every sampled frame
    max_gap: int = 4


@dataclass(frozen=True)
class Video:
    id: int
    path: str
    duration: float
    start: int  # first row in frames.f16
    count: int


SCHEMA = """
create table if not exists videos (
    id integer primary key,
    path text unique not null,
    size integer not null,
    mtime real not null,
    duration real not null,
    start integer not null,
    count integer not null,
    indexed_at real not null
);
"""


class Library:
    def __init__(self, folder: str | Path, settings: Settings | None = None, encoder: Encoder | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        cfg = self.folder / "config.json"
        if cfg.exists():
            stored = Settings(**json.loads(cfg.read_text()))
            if settings and settings != stored:
                raise ValueError(f"{self.folder} was built with {stored}; use a new folder for other settings")
            self.settings = stored
        else:
            self.settings = settings or Settings()
            cfg.write_text(json.dumps(asdict(self.settings), indent=1))
        self.db = sqlite3.connect(self.folder / "library.sqlite")
        self.db.executescript(SCHEMA)
        self._encoder = encoder
        self._cache = None

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self.settings.model)
        return self._encoder

    # ------------------------------------------------------------------ writing
    def _rows(self) -> int:
        f = self.folder / "times.f32"
        return f.stat().st_size // 4 if f.exists() else 0

    def needs_indexing(self, path: Path) -> bool:
        st = path.stat()
        row = self.db.execute("select size, mtime from videos where path = ?", (str(path),)).fetchone()
        return row is None or row[0] != st.st_size or abs(row[1] - st.st_mtime) > 1e-6

    def add(self, path: str | Path) -> Video | None:
        """Index one video, unless it is already indexed and unchanged."""
        path = Path(path).resolve()
        if not self.needs_indexing(path):
            return None
        info = probe(path)
        s = self.settings
        repeats = RepeatFilter(s.dedup, s.max_gap) if s.dedup > 0 else None
        embs, kept_times, seen = [], [], 0
        for times, frames in iter_frames(path, s.fps, self.encoder.size):
            seen += len(frames)
            keep = repeats(thumbnails(frames)) if repeats else np.arange(len(frames))
            if len(keep):
                embs.append(self.encoder.encode_images(frames[keep]).astype(np.float16))
                kept_times.append(times[keep].astype(np.float32))
        if not seen:
            raise VideoError(f"{path}: no frame decoded")
        emb, times = np.concatenate(embs), np.concatenate(kept_times)
        start = self._rows()
        with open(self.folder / "frames.f16", "ab") as f:
            f.write(emb.tobytes())
        with open(self.folder / "times.f32", "ab") as f:
            f.write(times.tobytes())
        st = path.stat()
        duration = info.duration or float(seen / s.fps)
        self.db.execute(
            "insert into videos (path, size, mtime, duration, start, count, indexed_at) values (?, ?, ?, ?, ?, ?, ?)"
            " on conflict(path) do update set size=excluded.size, mtime=excluded.mtime, duration=excluded.duration,"
            " start=excluded.start, count=excluded.count, indexed_at=excluded.indexed_at",
            (str(path), st.st_size, st.st_mtime, duration, start, len(times), time.time()),
        )
        self.db.commit()
        self._cache = None
        vid = self.db.execute("select id from videos where path = ?", (str(path),)).fetchone()[0]
        return Video(vid, str(path), duration, start, len(times))

    def add_many(self, paths: Iterable[str | Path], on_done: Callable[[Path, Video | None, Exception | None], None]
                 | None = None) -> None:  # fmt: skip
        for p in paths:
            try:
                v = self.add(p)
                err = None
            except (VideoError, OSError) as e:
                v, err = None, e
            if on_done:
                on_done(Path(p), v, err)

    def remove(self, path: str | Path) -> None:
        self.db.execute("delete from videos where path = ?", (str(Path(path).resolve()),))
        self.db.commit()
        self._cache = None

    def compact(self) -> None:
        """Rewrite the embedding files without rows that no video points to any more."""
        emb, times = self._arrays()
        videos = self.videos()
        new_emb, new_times, moves = [], [], []
        row = 0
        for v in videos:
            new_emb.append(emb[v.start : v.start + v.count])
            new_times.append(times[v.start : v.start + v.count])
            moves.append((row, v.id))
            row += v.count
        (self.folder / "frames.f16").write_bytes(np.concatenate(new_emb).tobytes() if new_emb else b"")
        (self.folder / "times.f32").write_bytes(np.concatenate(new_times).tobytes() if new_times else b"")
        for name in ("offsets.f32", "offsets.json"):  # rows moved: recomputed at the next search
            (self.folder / name).unlink(missing_ok=True)
        self.db.executemany("update videos set start = ? where id = ?", moves)
        self.db.commit()
        self._cache = None

    # ------------------------------------------------------------------ reading
    def videos(self) -> list[Video]:
        rows = self.db.execute("select id, path, duration, start, count from videos order by start").fetchall()
        return [Video(*r) for r in rows]

    def _arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = self._rows()
        if n == 0:
            return np.zeros((0, 0), np.float16), np.zeros(0, np.float32)
        dim = (self.folder / "frames.f16").stat().st_size // (2 * n)
        emb = np.memmap(self.folder / "frames.f16", dtype=np.float16, mode="r", shape=(n, dim))
        times = np.fromfile(self.folder / "times.f32", dtype=np.float32)
        return emb, times

    def snapshot(self):
        """(videos, embeddings, times), loaded once and reused until the library changes."""
        if self._cache is None:
            emb, times = self._arrays()
            self._cache = (self.videos(), np.asarray(emb, dtype=np.float32), times)
        return self._cache

    def offsets(self, bank: np.ndarray, beta: float) -> np.ndarray:
        """The query-bank offset of every row, computed once for rows that do not have one yet."""
        emb, _ = self._arrays()
        path, meta = self.folder / "offsets.f32", self.folder / "offsets.json"
        key = {"bank": qbank.digest(bank), "beta": beta}
        done = np.zeros(0, dtype=np.float32)
        if path.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            done = np.fromfile(path, dtype=np.float32)
        if len(done) > len(emb):
            done = done[:0]
        if len(done) < len(emb):
            done = np.concatenate([done, qbank.offsets(emb[len(done) :], bank, beta)])
            done.tofile(path)
            meta.write_text(json.dumps(key))
        return done

    def stats(self) -> dict:
        videos = self.videos()
        size = sum((self.folder / f).stat().st_size for f in ("frames.f16", "times.f32") if (self.folder / f).exists())
        return {
            "videos": len(videos),
            "hours": sum(v.duration for v in videos) / 3600,
            "embeddings": sum(v.count for v in videos),
            "index_bytes": size,
            "settings": asdict(self.settings),
        }
