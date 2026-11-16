"""Ranking moments across a whole library for one text query.

1. One matrix-vector product scores every indexed frame against the query.
2. Each video's best frame shortlists the videos worth a closer look.
3. In each shortlisted video, moments grow around the peaks of its curve.
4. All moments are ranked together. SigLIP's sigmoid makes a frame's score an
   absolute "does this match" estimate, so moments from different videos can be
   compared directly; `contrast` adds how much the moment stands out from the
   rest of its own video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import Library, Video
from .moments import Span, on_grid, smooth, spans


@dataclass
class SearchConfig:
    step: float = 0.5  # seconds per grid cell
    sigma: float = 2.0  # smoothing, in grid cells
    alpha: float = 0.6  # where a moment stops, between the video's floor (0) and the peak (1)
    per_video: int = 2
    shortlist: int = 64  # videos examined in detail
    contrast: float = 0.5  # weight of (peak - floor) in the ranking score
    min_length: float = 1.0


@dataclass(frozen=True)
class Hit:
    path: str
    start: float
    end: float
    score: float  # ranking score
    confidence: float  # SigLIP's probability that the best frame matches the query
    video_id: int


class Searcher:
    def __init__(self, library: Library, config: SearchConfig | None = None):
        self.library = library
        self.config = config or SearchConfig()

    def frame_scores(self, query: str) -> np.ndarray:
        _, emb, _ = self.library.snapshot()
        q = self.library.encoder.encode_texts([query])[0].astype(np.float32)
        return emb @ q

    def video_spans(self, video: Video, scores: np.ndarray, times: np.ndarray, count: int | None = None) -> list[Span]:
        c = self.config
        sl = slice(video.start, video.start + video.count)
        grid, values = on_grid(times[sl].astype(np.float64), scores[sl].astype(np.float64), video.duration, c.step)
        curve = smooth(values, c.sigma)
        return spans(grid, curve, c.step, c.alpha, count or c.per_video, c.min_length)

    def rank_score(self, span: Span) -> float:
        return span.peak + self.config.contrast * (span.peak - span.floor)

    def search(self, query: str, k: int = 10, scores: np.ndarray | None = None) -> list[Hit]:
        videos, _, times = self.library.snapshot()
        if not videos:
            return []
        scores = self.frame_scores(query) if scores is None else scores
        best = np.array([scores[v.start : v.start + v.count].max() for v in videos])
        order = np.argsort(-best)[: self.config.shortlist]
        prob = self.library.encoder.probability
        hits = []
        for i in order:
            v = videos[i]
            for s in self.video_spans(v, scores, times):
                hits.append(Hit(v.path, s.start, s.end, self.rank_score(s), float(prob(np.float64(s.peak))), v.id))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
