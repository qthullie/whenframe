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
from . import speech
from .encoder import DEFAULT_MODEL, Encoder
from .video import RepeatFilter, VideoError, is_photo, iter_frames, probe, read_photo, thumbnails


@dataclass
class Settings:
    model: str = DEFAULT_MODEL  # several models joined by "+" are indexed together
    fps: float = 1.0
    dedup: float = 4.0  # 0 encodes every sampled frame
    max_gap: int = 4
    speech: str = ""  # Whisper model transcribing the videos ("base", "small"), or "" for none
    photos: bool = True  # photos are indexed with the videos, each as a single frame

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
                 encoders: dict[str, Encoder] | None = None, backend: str | None = None,
                 transcriber=None):  # fmt: skip
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
        self.db.executescript(SCHEMA + speech.SCHEMA)
        self._transcriber = transcriber  # speech.Transcriber, made when first needed
        self._lock = threading.RLock()
        self.backend = backend  # runtime of the image tower (encoder.BACKENDS); None: the default
        self._encoders = dict(encoders or {})
        if encoder is not None:
            self._encoders[self.settings.models[0]] = encoder
        self._cache = None

    def enable_speech(self, size: str) -> None:
        """Transcribe this library's videos with Whisper `size` from now on (and those not transcribed yet)."""
        if size not in speech.WHISPER:
            raise ValueError(f"unknown Whisper model {size!r}; one of {', '.join(speech.WHISPER)}")
        if self.settings.speech != size:
            self.settings = Settings(**{**asdict(self.settings), "speech": size})
            (self.folder / "config.json").write_text(json.dumps(asdict(self.settings), indent=1))

    @property
    def transcriber(self):
        if self._transcriber is None:
            self._transcriber = speech.Transcriber(self.settings.speech)
        return self._transcriber

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

    def needs_transcript(self, path: Path) -> bool:
        if not self.settings.speech or is_photo(path):
            return False
        row = self.db.execute("select t.model from videos v left join transcribed t on t.video_id = v.id"
                              " where v.path = ?", (str(path),)).fetchone()  # fmt: skip
        return row is None or row[0] != self.settings.speech

    def add(self, path: str | Path) -> Video | None:
        """Index one video, unless it is already indexed and unchanged; transcribe it too if the library does."""
        path = Path(path).resolve()
        video = self._add_frames(path) if self.needs_indexing(path) else None
        if self.needs_transcript(path):
            video = video or next(v for v in self.videos() if v.path == str(path))
            said = self.transcriber(str(path), video.duration)
            with self._lock:
                self.db.execute("delete from speech where video_id = ?", (video.id,))
                self.db.executemany("insert into speech (video_id, start, end, text) values (?, ?, ?, ?)",
                                    [(video.id, a, b, text) for a, b, text in said])  # fmt: skip
                self.db.execute("insert or replace into transcribed (video_id, model, at) values (?, ?, ?)",
                                (video.id, self.settings.speech, time.time()))  # fmt: skip
                self.db.commit()
        return video

    def speech_hits(self, query: str) -> list[tuple[int, float, float, float, str]]:
        """Segments of speech matching the query (speech.segment_scores)."""
        with self._lock:
            return speech.segment_scores(self.db, query)

    def has_speech(self) -> bool:
        with self._lock:
            return self.db.execute("select 1 from speech limit 1").fetchone() is not None

    def _add_frames(self, path: Path) -> Video:
        if is_photo(path):
            return self._add_photo(path)
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
        return self._store(path, {m: np.concatenate(e) for m, e in embs.items()}, np.concatenate(kept_times),
                           info.duration or float(seen / s.fps))  # fmt: skip

    def _add_photo(self, path: Path) -> Video:
        """A photo is a video of one frame and no duration: its moment is the photo itself."""
        embs = {}
        for model in self.settings.models:
            enc = self.encoder_for(model)
            embs[model] = enc.encode_images(read_photo(path, enc.size)).astype(np.float16)
        return self._store(path, embs, np.zeros(1, dtype=np.float32), 0.0)

    def _store(self, path: Path, embs: dict[str, np.ndarray], times: np.ndarray, duration: float) -> Video:
        s = self.settings
        st = path.stat()
        with self._lock:  # readers see the video with all its rows, or not at all
            start = self._rows()
            for model in s.models:
                with open(self.folder / f"frames.{model}.f16", "ab") as f:
                    f.write(embs[model].astype(np.float16).tobytes())
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
            row = self.db.execute("select id from videos where path = ?", (str(Path(path).resolve()),)).fetchone()
            if row:
                for table in ("speech", "transcribed"):
                    self.db.execute(f"delete from {table} where video_id = ?", row)  # noqa: S608 (fixed names)
                self.db.execute("delete from videos where id = ?", row)
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
            "videos": sum(not is_photo(v.path) for v in videos),
            "photos": sum(is_photo(v.path) for v in videos),
            "hours": sum(v.duration for v in videos) / 3600,
            "embeddings": sum(v.count for v in videos),
            "index_bytes": sum(f.stat().st_size for f in files if f.exists()),
            "speech_segments": self.db.execute("select count(*) from speech").fetchone()[0],
            "settings": asdict(self.settings),
        }
