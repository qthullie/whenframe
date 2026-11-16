"""A video library on disk: which videos were indexed, and their frame embeddings.

Layout of a library folder:

    library.sqlite        one row per video (path, size, mtime, duration, where its frames are)
    frames.MODEL.f16      the frame embeddings of each model, float16, appended in indexing order
    times.f32             the instant (seconds) each row stands for, shared by the models
    offsets.MODEL.f32     each frame's query-bank offset (see bank.py), filled in at the first search
    config.json           the models and sampling settings the library was built with

Embeddings are only ever appended. A re-indexed or deleted video leaves dead
rows behind, which `compact()` removes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
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
    model: str = DEFAULT_MODEL  # several models joined by "+" are indexed together
    fps: float = 1.0
    dedup: float = 4.0  # 0 encodes every sampled frame
    max_gap: int = 4

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self.model.split("+"))


@dataclass(frozen=True)
class Video:
    id: int
    path: str
    duration: float
    start: int  # first row in the frame files
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
    def __init__(self, folder: str | Path, settings: Settings | None = None, encoder: Encoder | None = None,
                 encoders: dict[str, Encoder] | None = None, backend: str | None = None):  # fmt: skip
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
        # `whenframe serve` reads the library from several threads; writes come from one indexer
        self.db = sqlite3.connect(self.folder / "library.sqlite", check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self.backend = backend  # runtime of the image tower (encoder.BACKENDS); None: the default
        self._encoders = dict(encoders or {})
        if encoder is not None:
            self._encoders[self.settings.models[0]] = encoder
        self._cache = None

    def encoder_for(self, model: str) -> Encoder:
        if model not in self._encoders:
            self._encoders[model] = Encoder(model, backend=self.backend)
        return self._encoders[model]

    @property
    def encoder(self) -> Encoder:
        """The first model's encoder."""
        return self.encoder_for(self.settings.models[0])

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
        encoders = [self.encoder_for(m) for m in s.models]
        sizes = list(dict.fromkeys(e.size for e in encoders))  # one decode per input size
        streams = [iter_frames(path, s.fps, size) for size in sizes]
        repeats = RepeatFilter(s.dedup, s.max_gap) if s.dedup > 0 else None
        embs, kept_times, seen = {m: [] for m in s.models}, [], 0
        try:
            for chunks in zip(*streams, strict=False):
                n = min(len(frames) for _, frames in chunks)
                by_size = {size: frames[:n] for size, (_, frames) in zip(sizes, chunks, strict=True)}
                times = chunks[0][0][:n]
                seen += n
                # the first model's frames decide what is kept, for every model, so the rows stay aligned
                keep = repeats(thumbnails(by_size[encoders[0].size])) if repeats else np.arange(n)
                if len(keep):
                    for model, enc in zip(s.models, encoders, strict=True):
                        embs[model].append(enc.encode_images(by_size[enc.size][keep]).astype(np.float16))
                    kept_times.append(times[keep].astype(np.float32))
        finally:
            for stream in streams:
                stream.close()
        if not seen:
            raise VideoError(f"{path}: no frame decoded")
        times = np.concatenate(kept_times)
        st = path.stat()
        duration = info.duration or float(seen / s.fps)
        with self._lock:  # readers see the video with all its rows, or not at all
            start = self._rows()
            for model in s.models:
                with open(self.folder / f"frames.{model}.f16", "ab") as f:
                    f.write(np.concatenate(embs[model]).tobytes())
            with open(self.folder / "times.f32", "ab") as f:
                f.write(times.tobytes())
            self.db.execute(
                "insert into videos (path, size, mtime, duration, start, count, indexed_at)"
                " values (?, ?, ?, ?, ?, ?, ?)"
                " on conflict(path) do update set size=excluded.size, mtime=excluded.mtime,"
                " duration=excluded.duration, start=excluded.start, count=excluded.count,"
                " indexed_at=excluded.indexed_at",
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
        with self._lock:
            self.db.execute("delete from videos where path = ?", (str(Path(path).resolve()),))
            self.db.commit()
            self._cache = None

    def dead_rows(self) -> int:
        """Rows left behind by re-indexed or removed videos."""
        with self._lock:
            return self._rows() - sum(v.count for v in self.videos())

    def compact(self) -> None:
        """Rewrite the embedding files without rows that no video points to any more."""
        with self._lock:
            self._compact()

    def compact_if_needed(self, share: float = 0.25) -> bool:
        """Compact when more than `share` of the rows are dead; True if it did."""
        with self._lock:
            rows = self._rows()
            if not rows or self.dead_rows() <= share * rows:
                return False
            self._compact()
            return True

    def _compact(self) -> None:
        videos = self.videos()
        keep = np.concatenate([np.arange(v.start, v.start + v.count) for v in videos]) if videos else np.zeros(0, int)
        for model in self.settings.models:
            emb, _ = self._arrays(model)
            (self.folder / f"frames.{model}.f16").write_bytes(np.ascontiguousarray(emb[keep]).tobytes())
        _, times = self._arrays(self.settings.models[0])
        (self.folder / "times.f32").write_bytes(times[keep].tobytes())
        for f in self.folder.glob("offsets.*"):  # rows moved: recomputed at the next search
            f.unlink()
        starts = np.concatenate([[0], np.cumsum([v.count for v in videos])[:-1]]) if videos else []
        self.db.executemany("update videos set start = ? where id = ?",
                            [(int(s), v.id) for s, v in zip(starts, videos, strict=True)])  # fmt: skip
        self.db.commit()
        self._cache = None

    # ------------------------------------------------------------------ reading
    def videos(self) -> list[Video]:
        with self._lock:
            rows = self.db.execute("select id, path, duration, start, count from videos order by start").fetchall()
        return [Video(*r) for r in rows]

    def _arrays(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        n = self._rows()
        if n == 0:
            return np.zeros((0, 0), np.float16), np.zeros(0, np.float32)
        path = self.folder / f"frames.{model}.f16"
        dim = path.stat().st_size // (2 * n)
        emb = np.memmap(path, dtype=np.float16, mode="r", shape=(n, dim))
        times = np.fromfile(self.folder / "times.f32", dtype=np.float32)
        return emb, times

    def snapshot(self):
        """(videos, {model: embeddings}, times), loaded once and reused until the library changes."""
        with self._lock:
            if self._cache is None:
                embs, times = {}, np.zeros(0, np.float32)
                for model in self.settings.models:
                    emb, times = self._arrays(model)
                    embs[model] = np.asarray(emb, dtype=np.float32)
                self._cache = (self.videos(), embs, times)
            return self._cache

    def offsets(self, model: str, bank: np.ndarray, beta: float) -> np.ndarray:
        """The query-bank offset of every row for one model, computed once for rows that lack one."""
        with self._lock:
            return self._offsets(model, bank, beta)

    def _offsets(self, model: str, bank: np.ndarray, beta: float) -> np.ndarray:
        emb, _ = self._arrays(model)
        path, meta = self.folder / f"offsets.{model}.f32", self.folder / f"offsets.{model}.json"
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
        files = [*self.folder.glob("frames.*.f16"), self.folder / "times.f32"]
        return {
            "videos": len(videos),
            "hours": sum(v.duration for v in videos) / 3600,
            "embeddings": sum(v.count for v in videos),
            "index_bytes": sum(f.stat().st_size for f in files if f.exists()),
            "settings": asdict(self.settings),
        }
