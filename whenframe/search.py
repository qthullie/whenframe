"""Ranking moments across a whole library for one text query; a photo is a moment of its own.

1. One matrix-vector product per model scores every indexed frame against the query.
2. Frames that match almost any description are marked down (query-bank
   normalisation, see bank.py).
3. When several models were indexed, their scores are standardised over the
   library and added, with weights tuned on DiDeMo training videos.
4. When the videos were transcribed, the words of the query that are said
   around a frame add to its score (speech.py); when their sound tracks were
   embedded, so does CLAP's score of what is heard around it (sounds.py); with
   the weights chosen with the library's model (chosen, below).
5. Each video's best frame shortlists the videos worth a closer look.
6. In each shortlisted video, moments are placed around its best frames: fixed
   windows by default, or spans grown around the peaks of its curve.
7. All moments are ranked together.

The defaults are the default search of docs/RESULTS.md, chosen on DiDeMo
training videos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import bank as qbank
from . import sounds
from .encoder import FUSED, FUSION, MODELS
from .half import BLOCK, to_float32
from .library import Library, Video
from .moments import Span, on_grid, smooth, spans, windows
from .sounds import SoundConfig
from .speech import SpeechConfig
from .speech import frame_scores as frame_speech


@dataclass
class SearchConfig:
    # "windows": a fixed length around the best frames; "spans": grown around peaks; "auto": spans when the
    # library's videos last more than AUTO_SPANS seconds (median), windows otherwise
    moments: str = "auto"
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
    # used when the library is transcribed, and when its sound tracks are embedded; None: as chosen with the
    # library's model on the tuning sets (chosen, below)
    speech: SpeechConfig | None = None
    sound: SoundConfig | None = None


def chosen(library: Library) -> tuple[SpeechConfig, SoundConfig]:
    """The speech settings and the sound weight chosen with the library's model on YouCook2's and UnAV-100's tuning
    sets (docs/RESULTS.md, sixth, seventh and tenth rounds): those of its models together when they were measured
    together (encoder.FUSED); its temporal model's, kept in the model's file, when its frames go through one; else
    its image model's (encoder.MODELS); the defaults for other models searched together, for which none were
    chosen."""
    models = library.settings.models
    fused = FUSED.get(library.settings.model)
    if fused:
        return (SpeechConfig(fused.speech_weight, fused.speech_pad, fused.speech_normalise),
                SoundConfig(weight=fused.sound_weight))  # fmt: skip
    if len(models) > 1 or models[0] not in MODELS:
        return SpeechConfig(), SoundConfig()
    if library.settings.temporal:
        try:
            s = library.temporal_model.get("search")
        except ValueError:  # the model is not installed: its frames, already indexed, are searched all the same
            s = None
        if s:
            sp = s["speech"]
            return SpeechConfig(sp["weight"], sp["pad"], sp["normalise"]), SoundConfig(weight=s["sound_weight"])
    spec = MODELS[models[0]]
    return SpeechConfig(pad=spec.speech_pad), SoundConfig(weight=spec.sound_weight)


# Windows did better on DiDeMo's clips of at most 30 seconds, spans on YouCook2's videos of 289 seconds
# (median): any threshold between the two gives each benchmark its best (docs/RESULTS.md, fourth round).
AUTO_SPANS = 60.0


@dataclass(frozen=True)
class Hit:
    path: str
    start: float
    end: float
    score: float  # ranking score
    confidence: float  # the first model's probability that the best frame of the moment matches the query
    video_id: int
    at: float = 0.0  # the instant of the moment's best frame
    said: str = ""  # what is said there that matches the query, when the library is transcribed
    photo: bool = False  # a photo: start and end are 0, the moment is the photo itself
    heard: float = 0.0  # how well the sound around the best frame matches, in standard deviations (--sounds)


HIDDEN_BELOW = 0.005  # shown as 0 %: rarely right (0.8 % of the right moments on DiDeMo) when only frames count
HEARD = 1.0  # a sound this many standard deviations above the library's usual match is worth showing


def worth_showing(hit: Hit) -> bool:
    """Whether a moment is shown by default: the frames match, or what is said or heard there does."""
    return hit.confidence >= HIDDEN_BELOW or bool(hit.said) or hit.heard >= HEARD


def scores(emb: np.ndarray, q: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """emb @ q; float16 embeddings are converted a block of rows at a time, never all at once."""
    if emb.dtype == np.float32:
        return emb @ q
    out = np.empty(len(emb), dtype=np.float32)
    for i in range(0, len(emb), block):
        out[i : i + block] = to_float32(emb[i : i + block]) @ q
    return out


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-12)


class Searcher:
    def __init__(self, library: Library, config: SearchConfig | None = None,
                 banks: dict[str, np.ndarray] | None = None):  # fmt: skip
        self.library = library
        config = config or SearchConfig()
        speech, sound = chosen(library)
        self.config = replace(config, speech=speech if config.speech is None else config.speech,
                              sound=sound if config.sound is None else config.sound)  # fmt: skip
        self.models = library.settings.models
        banks = banks or {}
        self.banks = {m: banks[m] if m in banks else qbank.load(m) for m in self.models}
        self._offsets = {}

    @property
    def moments(self) -> str:
        """The kind of moments placed: as configured, or for "auto" as the library's videos suit."""
        if self.config.moments != "auto":
            return self.config.moments
        videos, _, _ = self.library.snapshot()
        lengths = [v.duration for v in videos if v.duration > 0]  # photos have none
        return "spans" if lengths and float(np.median(lengths)) > AUTO_SPANS else "windows"

    def _tuned(self, model: str) -> tuple[float, float]:
        """The query bank's beta and strength for this model, as tuned for the kind of moments placed."""
        spec = MODELS.get(model)
        beta = self.config.beta or (spec.bank_beta if spec else 50.0)
        if spec is None:
            return beta, 1.0
        if self.moments == "spans" and spec.spans_lambda is not None:
            return beta, spec.spans_lambda
        return beta, spec.bank_lambda

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
            out[model] = scores(embs[model], q)
        return out

    def ranking_scores(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        """What moments are ranked by: corrected, and combined when there are several models."""
        corrected = {}
        for model in self.models:
            off = self.offsets(model)
            strength = self.config.bank * self._tuned(model)[1]
            corrected[model] = raw[model] if off is None else raw[model] - strength * off
        first = corrected[self.models[0]]
        self._first_corrected = first
        if len(self.models) == 1:
            return first
        fused = FUSED.get(self.library.settings.model)
        if fused and self.config.weights is None:  # as measured together, on the first model's scale
            z = standardise(sum(w * standardise(corrected[m]) for m, w in zip(self.models, fused.weights, strict=True)))
            return float(first.mean()) + float(first.std()) * z
        weights = self.config.weights or FUSION
        return sum(weights.get(m, 1.0) * standardise(corrected[m]) for m in self.models)

    @property
    def _on_first_scale(self) -> bool:
        """Whether the ranking scores are on the first model's scale: one model, or models fused as measured."""
        return len(self.models) == 1 or (self.library.settings.model in FUSED and self.config.weights is None)

    def video_moments(self, video: Video, scores: np.ndarray, times: np.ndarray) -> list[Span]:
        c = self.config
        sl = slice(video.start, video.start + video.count)
        t, v = times[sl].astype(np.float64), scores[sl].astype(np.float64)
        if video.duration == 0:  # a photo
            return [Span(0.0, 0.0, float(v.max()), float(v.max()), float(v.max()))]
        if self.moments == "windows":
            return windows(t, v, video.duration, c.window, c.per_video, 1.0 / self.library.settings.fps)
        grid, values = on_grid(t, v, video.duration, c.step)
        return spans(grid, smooth(values, c.sigma), c.step, c.alpha, c.per_video, c.min_length)

    def rank_score(self, span: Span) -> float:
        if self.moments == "windows":
            return span.peak
        return span.peak + self.config.contrast * (span.peak - span.floor)

    def speech_scores(self, query: str, ranking: np.ndarray, videos, times) -> tuple[np.ndarray, list]:
        """What speech adds to each frame's ranking score, and the matching segments."""
        if not self.library.has_speech():
            return np.zeros_like(ranking), []
        hits = self.library.speech_hits(query)
        cfg = self.config.speech
        said = frame_speech(times, {v.id: (v.start, v.count) for v in videos}, hits, cfg)
        # the weight was tuned on one model's corrected scores; several models' are standardised first, unless they
        # were put back on the first model's scale
        scale = 1.0 if self._on_first_scale else 1.0 / max(float(self._first_corrected.std()), 1e-9)
        return cfg.weight * scale * said, hits

    def sound_scores(self, query: str, videos, times, heard: np.ndarray | None = None) -> np.ndarray:
        """For each frame, CLAP's score of the window around it, standardised over the library (0 without sound).
        `heard` gives CLAP's cosine of every window instead of embedding the query (tests)."""
        if not self.library.has_sounds():
            return np.zeros(len(times), dtype=np.float32)
        ids, starts, emb = self.library.sound_windows()
        if heard is None:
            heard = emb @ self.library.listener.clap.encode_texts([query])[0]
        length = self.config.sound.length
        cos = sounds.frame_scores(times, {v.id: (v.start, v.count) for v in videos}, ids, starts, heard, length)
        return sounds.standardised(cos)

    def search(self, query: str, k: int = 10, scores: dict[str, np.ndarray] | None = None,
               heard: np.ndarray | None = None) -> list[Hit]:  # fmt: skip
        videos, _, times = self.library.snapshot()
        if not videos:
            return []
        raw = self.frame_scores(query) if scores is None else scores
        ranking = self.ranking_scores(raw)
        extra, segments = self.speech_scores(query, ranking, videos, times)
        # what is heard and what is seen are added as standardised scores, with the tuned weight
        loud = self.sound_scores(query, videos, times, heard)
        ranking = ranking + extra + self.config.sound.weight * float(ranking.std()) * loud
        first = raw[self.models[0]]
        prob = self.library.encoder.probability
        best = np.array([ranking[v.start : v.start + v.count].max() for v in videos])
        hits = []
        for i in np.argsort(-best)[: self.config.shortlist]:
            v = videos[i]
            t, r = times[v.start : v.start + v.count], first[v.start : v.start + v.count]
            g = ranking[v.start : v.start + v.count]
            for s in self.video_moments(v, ranking, times):
                inside = np.flatnonzero((t >= s.start) & (t < s.end))
                inside = inside if len(inside) else np.arange(len(t))
                top = r[inside].max()
                best = inside[np.argmax(g[inside])]
                at = float(t[best])
                hit = Hit(v.path, s.start, s.end, self.rank_score(s), float(prob(np.float64(top))), v.id, at,
                          photo=v.duration == 0, heard=float(loud[v.start + best]))  # fmt: skip
                hits.append(hit)
        hits.sort(key=lambda h: -h.score)
        return [self._with_quote(h, segments) for h in hits[:k]]

    def _with_quote(self, hit: Hit, segments) -> Hit:
        """The best matching segment said during the moment, if any."""
        pad = self.config.speech.pad
        near = [s for s in segments if s[0] == hit.video_id and s[1] <= hit.end + pad and s[2] >= hit.start - pad]
        return replace(hit, said=max(near, key=lambda s: s[3])[4]) if near else hit
