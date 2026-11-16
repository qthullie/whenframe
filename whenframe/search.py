"""Ranking moments across a whole library for one text query.

1. One matrix-vector product per model scores every indexed frame against the query.
2. Frames that match almost any description are marked down (query-bank
   normalisation, see bank.py).
3. When several models were indexed, their scores are standardised over the
   library and added, with weights tuned on DiDeMo training videos.
4. Each video's best frame shortlists the videos worth a closer look.
5. In each shortlisted video, moments are placed around its best frames: fixed
   windows by default, or spans grown around the peaks of its curve.
6. All moments are ranked together.

The defaults are the default search of docs/RESULTS.md, chosen on DiDeMo
training videos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bank as qbank
from .encoder import FUSION, MODELS
from .library import Library, Video
from .moments import Span, on_grid, smooth, spans, windows


@dataclass
class SearchConfig:
    moments: str = "windows"  # "windows": a fixed length around the best frames; "spans": grown around peaks
    window: float = 10.0  # seconds, for "windows"
    per_video: int = 3
    bank: float = 1.0  # strength of the query-bank correction, times each model's tuned strength; 0 turns it off
    beta: float | None = None  # replaces each model's tuned beta when set
    weights: dict[str, float] | None = None  # weight of each model's scores when several are indexed
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
    confidence: float  # the first model's probability that the best frame of the moment matches the query
    video_id: int


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-12)


class Searcher:
    def __init__(self, library: Library, config: SearchConfig | None = None,
                 banks: dict[str, np.ndarray] | None = None):  # fmt: skip
        self.library = library
        self.config = config or SearchConfig()
        self.models = library.settings.models
        banks = banks or {}
        self.banks = {m: banks[m] if m in banks else qbank.load(m) for m in self.models}
        self._offsets = {}

    def _tuned(self, model: str) -> tuple[float, float]:
        spec = MODELS.get(model)
        beta = self.config.beta or (spec.bank_beta if spec else 50.0)
        return beta, (spec.bank_lambda if spec else 1.0)

    def offsets(self, model: str) -> np.ndarray | None:
        if self.config.bank <= 0 or self.banks.get(model) is None:
            return None
        _, embs, _ = self.library.snapshot()
        if model not in self._offsets or len(self._offsets[model]) != len(embs[model]):
            self._offsets[model] = self.library.offsets(model, self.banks[model], self._tuned(model)[0])
        return self._offsets[model]

    def frame_scores(self, query: str) -> dict[str, np.ndarray]:
        """Cosine similarity of every indexed frame with the query, for each model."""
        _, embs, _ = self.library.snapshot()
        out = {}
        for model in self.models:
            q = self.library.encoder_for(model).encode_texts([query])[0].astype(np.float32)
            out[model] = embs[model] @ q
        return out

    def ranking_scores(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        """What moments are ranked by: corrected, and combined when there are several models."""
        corrected = {}
        for model in self.models:
            off = self.offsets(model)
            strength = self.config.bank * self._tuned(model)[1]
            corrected[model] = raw[model] if off is None else raw[model] - strength * off
        if len(self.models) == 1:
            return corrected[self.models[0]]
        weights = self.config.weights or FUSION
        return sum(weights.get(m, 1.0) * standardise(corrected[m]) for m in self.models)

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

    def search(self, query: str, k: int = 10, scores: dict[str, np.ndarray] | None = None) -> list[Hit]:
        videos, _, times = self.library.snapshot()
        if not videos:
            return []
        raw = self.frame_scores(query) if scores is None else scores
        ranking = self.ranking_scores(raw)
        first = raw[self.models[0]]
        prob = self.library.encoder.probability
        best = np.array([ranking[v.start : v.start + v.count].max() for v in videos])
        hits = []
        for i in np.argsort(-best)[: self.config.shortlist]:
            v = videos[i]
            t, r = times[v.start : v.start + v.count], first[v.start : v.start + v.count]
            for s in self.video_moments(v, ranking, times):
                inside = (t >= s.start) & (t < s.end)
                top = r[inside].max() if inside.any() else r.max()
                hits.append(Hit(v.path, s.start, s.end, self.rank_score(s), float(prob(np.float64(top))), v.id))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
