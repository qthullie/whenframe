"""Ranking moments across a whole library for one text query.

1. One matrix-vector product scores every indexed frame against the query.
2. Frames that match almost any description are marked down (query-bank
   normalisation, see bank.py).
3. Each video's best frame shortlists the videos worth a closer look.
4. In each shortlisted video, moments are placed around its best frames: fixed
   windows by default, or spans grown around the peaks of its curve.
5. All moments are ranked together. SigLIP's sigmoid makes a frame's score an
   absolute "does this match" estimate, so moments from different videos can be
   compared directly.

The defaults are the default search of docs/RESULTS.md, chosen on DiDeMo
training videos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bank as qbank
from .library import Library, Video
from .moments import Span, on_grid, smooth, spans, windows


@dataclass
class SearchConfig:
    moments: str = "windows"  # "windows": a fixed length around the best frames; "spans": grown around peaks
    window: float = 10.0  # seconds, for "windows"
    per_video: int = 3
    bank: float = 1.0  # weight of the query-bank correction; 0 turns it off
    beta: float = 50.0
    shortlist: int = 64  # videos examined in detail
    # for "spans"
    step: float = 0.5  # seconds per grid cell
    sigma: float = 0.0  # smoothing, in grid cells
    alpha: float = 0.6  # where a span stops, between the video's floor (0) and the peak (1)
    contrast: float = 0.0  # weight of (peak - floor) in the ranking score
    min_length: float = 10.0


@dataclass(frozen=True)
class Hit:
    path: str
    start: float
    end: float
    score: float  # ranking score
    confidence: float  # SigLIP's probability that the best frame of the moment matches the query
    video_id: int


class Searcher:
    def __init__(self, library: Library, config: SearchConfig | None = None, bank: np.ndarray | None = None):
        self.library = library
        self.config = config or SearchConfig()
        self.bank = bank if bank is not None else qbank.load(library.settings.model)
        self._offsets = None

    def offsets(self) -> np.ndarray | None:
        if self.config.bank <= 0 or self.bank is None:
            return None
        _, emb, _ = self.library.snapshot()
        if self._offsets is None or len(self._offsets) != len(emb):
            self._offsets = self.library.offsets(self.bank, self.config.beta)
        return self._offsets

    def frame_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of every indexed frame with the query."""
        _, emb, _ = self.library.snapshot()
        q = self.library.encoder.encode_texts([query])[0].astype(np.float32)
        return emb @ q

    def video_moments(self, video: Video, scores: np.ndarray, times: np.ndarray) -> list[Span]:
        c = self.config
        sl = slice(video.start, video.start + video.count)
        t, v = times[sl].astype(np.float64), scores[sl].astype(np.float64)
        if c.moments == "windows":
            return windows(t, v, video.duration, c.window, c.per_video, 1.0 / self.library.settings.fps)
        grid, values = on_grid(t, v, video.duration, c.step)
        return spans(grid, smooth(values, c.sigma), c.step, c.alpha, c.per_video, c.min_length)

    def rank_score(self, span: Span) -> float:
        if self.config.moments == "windows":
            return span.peak
        return span.peak + self.config.contrast * (span.peak - span.floor)

    def search(self, query: str, k: int = 10, scores: np.ndarray | None = None) -> list[Hit]:
        videos, _, times = self.library.snapshot()
        if not videos:
            return []
        raw = self.frame_scores(query) if scores is None else scores
        offsets = self.offsets()
        ranking = raw - self.config.bank * offsets if offsets is not None else raw
        best = np.array([ranking[v.start : v.start + v.count].max() for v in videos])
        prob = self.library.encoder.probability
        hits = []
        for i in np.argsort(-best)[: self.config.shortlist]:
            v = videos[i]
            t, r = times[v.start : v.start + v.count], raw[v.start : v.start + v.count]
            for s in self.video_moments(v, ranking, times):
                inside = (t >= s.start) & (t < s.end)
                top = r[inside].max() if inside.any() else r.max()
                hits.append(Hit(v.path, s.start, s.end, self.rank_score(s), float(prob(np.float64(top))), v.id))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
