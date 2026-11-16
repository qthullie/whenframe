"""Shared evaluation code: methods, metrics and tuning on precomputed frame embeddings.

All methods see the same frame embeddings and the same query embeddings; they
only differ in how they turn per-frame scores into ranked moments.

    scene              shots from ffmpeg's scene detector, each scored by its middle
                       frame (in the spirit of tools that index one keyframe per shot)
    frame_window       fixed windows around the best frames of each video
    whenframe          moments grown around peaks of the smoothed curve, ranked by
                       peak score plus contrast with the rest of the video
    whenframe_bank     whenframe on scores corrected for frames that match many
                       descriptions (query-bank normalisation, Bogolin et al. 2022)
    frame_window_bank  frame_window on the same corrected scores
    whenframe_dedup    the default search (the best of the above on the tuning
                       videos) on the frames left after skipping repeats
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field, replace

import numpy as np

from whenframe.moments import on_grid, smooth, spans
from whenframe.video import RepeatFilter

CORPUS_KS = [1, 5, 10, 100]
SHORTLIST = 64
METHODS = ["scene", "frame_window", "whenframe", "whenframe_bank", "frame_window_bank", "whenframe_dedup"]
CANDIDATES = ["frame_window", "whenframe", "whenframe_bank", "frame_window_bank"]  # for the default search
BANKED = {"whenframe_bank", "frame_window_bank"}
PRIMARY = ("corpus", "R@5 IoU>=0.5")


@dataclass
class Frozen:
    names: list
    emb: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    durations: np.ndarray
    keep: np.ndarray  # frames kept by the default repeat filter
    cuts: list
    thumbs: np.ndarray | None = None
    bank: np.ndarray | None = None  # text embeddings of descriptions of other videos
    _offsets: dict = field(default_factory=dict, repr=False)

    def select(self, mask: np.ndarray) -> Frozen:
        """The same videos, with only the frames in `mask` (every video must keep at least one)."""
        idx = np.flatnonzero(mask)
        counts = np.add.reduceat(mask.astype(np.int64), self.starts)
        assert counts.min() > 0, "a video lost all its frames"
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        thumbs = None if self.thumbs is None else self.thumbs[idx]
        return replace(self, emb=self.emb[idx], times=self.times[idx], starts=starts, counts=counts,
                       keep=self.keep[idx], thumbs=thumbs, _offsets={})  # fmt: skip

    def only_kept(self) -> Frozen:
        return self.select(self.keep)

    def without_repeats(self, threshold: float, max_gap: int = 4) -> np.ndarray:
        mask = np.zeros(len(self.times), dtype=bool)
        for s, c in zip(self.starts, self.counts, strict=True):
            mask[s + RepeatFilter(threshold, max_gap)(self.thumbs[s : s + c].astype(np.float32))] = True
        return mask

    def every(self, k: int) -> np.ndarray:
        mask = np.zeros(len(self.times), dtype=bool)
        for s, c in zip(self.starts, self.counts, strict=True):
            mask[s : s + c : k] = True
        return mask

    def offsets(self, beta: float) -> np.ndarray:
        """How well each frame matches descriptions in general: a soft maximum of its scores against the bank.

        In cosine units, (1 / beta) * log(mean_b exp(beta * cos(b, frame))). Subtracting it
        from a query's score is the inverted softmax of query-bank normalisation.
        """
        if beta not in self._offsets:
            out = np.empty(len(self.emb), dtype=np.float32)
            for i in range(0, len(self.emb), 4096):
                z = beta * (self.emb[i : i + 4096] @ self.bank.T)
                top = z.max(axis=1)
                out[i : i + 4096] = (top + np.log(np.exp(z - top[:, None]).mean(axis=1))) / beta
            self._offsets[beta] = out
        return self._offsets[beta]


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- methods
def frame_window(f: Frozen, scores: np.ndarray, v: int, p: dict) -> list[tuple[float, float, float]]:
    """Fixed windows around the best frame, then around the best frame at least a window away, and so on."""
    s, c = f.starts[v], f.counts[v]
    times, left = f.times[s : s + c], scores[s : s + c].astype(np.float64)
    w, d = p["window"], float(f.durations[v])
    out = []
    for _ in range(p.get("per_video", 1)):
        i = int(np.argmax(left))
        if left[i] == -np.inf:
            break
        t = float(times[i])
        start = min(max(0.0, t + 0.5 - w / 2), max(0.0, d - w))
        out.append((start, min(d, start + w), float(scores[s + i])))
        left[np.abs(times - t) < w] = -np.inf
    return out


def scene(f: Frozen, scores: np.ndarray, v: int, p: dict) -> list[tuple[float, float, float]]:
    s, c, d = f.starts[v], f.counts[v], float(f.durations[v])
    bounds = [0.0, *[t for t in f.cuts[v] if 0.0 < t < d], d]
    t = f.times[s : s + c]
    out = []
    for a, b in itertools.pairwise(bounds):
        mid = np.argmin(np.abs(t - (a + b) / 2))
        out.append((a, b, float(scores[s + mid])))
    out.sort(key=lambda x: -x[2])
    return out[: p["per_video"]]


def whenframe(f: Frozen, scores: np.ndarray, v: int, p: dict) -> list[tuple[float, float, float]]:
    s, c = f.starts[v], f.counts[v]
    grid, values = on_grid(f.times[s : s + c].astype(np.float64), scores[s : s + c].astype(np.float64),
                           float(f.durations[v]), p["step"])  # fmt: skip
    curve = smooth(values, p["sigma"])
    found = spans(grid, curve, p["step"], p["alpha"], p["per_video"], p["min_length"])
    return [(x.start, x.end, x.peak + p["contrast"] * (x.peak - x.floor)) for x in found]


PREDICT = {"frame_window": frame_window, "scene": scene, "whenframe": whenframe, "whenframe_bank": whenframe,
           "frame_window_bank": frame_window}  # fmt: skip

# The grids of the second tuning run (see "Changes made before the first test run" in docs/PROTOCOL.md).
SPACE = {
    "frame_window": {"window": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], "per_video": [1, 2, 3]},
    "scene": {"per_video": [1, 2, 3]},
    "whenframe": {"step": [0.5], "sigma": [0.0, 1.0, 2.0], "alpha": [0.5, 0.6, 0.7, 0.8],
                  "contrast": [0.0, 0.5, 1.0], "min_length": [5.0, 7.5, 10.0], "per_video": [2, 3]},
    # tried on top of the tuned parameters of whenframe and of frame_window
    "bank": {"beta": [10.0, 20.0, 50.0, 100.0], "lam": [0.25, 0.5, 1.0]},
}  # fmt: skip


def frame_scores(f: Frozen, text: np.ndarray, method: str, p: dict) -> np.ndarray:
    """frames x queries."""
    scores = f.emb @ text.T
    if method in BANKED:
        scores -= p["lam"] * f.offsets(p["beta"])[:, None]
    return scores


# ---------------------------------------------------------------- evaluation
def run(f: Frozen, queries: list, text: np.ndarray, method: str, p: dict, scores: np.ndarray | None = None) -> dict:
    predict = PREDICT[method]
    if scores is None:
        scores = frame_scores(f, text, method, p)
    best = np.maximum.reduceat(scores, f.starts, axis=0)  # videos x queries
    single_iou, hits = [], {(k, th): 0 for k in CORPUS_KS for th in (0.5, 0.7)}
    video_hits = dict.fromkeys(CORPUS_KS, 0)
    for qi, q in enumerate(queries):
        col = scores[:, qi]
        top = predict(f, col, q["video"], p)
        single_iou.append(iou(top[0][:2], q["moment"]) if top else 0.0)
        ranked = []
        for v in np.argsort(-best[:, qi])[:SHORTLIST]:
            ranked += [(int(v), *x) for x in predict(f, col, int(v), p)]
        ranked.sort(key=lambda x: -x[3])
        ranked = ranked[: max(CORPUS_KS)]
        first_ok = {th: None for th in (0.5, 0.7)}
        for r, (v, a, b, _) in enumerate(ranked):
            if v == q["video"]:
                o = iou((a, b), q["moment"])
                for th in (0.5, 0.7):
                    if first_ok[th] is None and o >= th:
                        first_ok[th] = r
        for th, r in first_ok.items():
            for k in CORPUS_KS:
                hits[(k, th)] += r is not None and r < k
        order = list(dict.fromkeys(v for v, *_ in ranked))
        rank_v = order.index(q["video"]) if q["video"] in order else None
        for k in CORPUS_KS:
            video_hits[k] += rank_v is not None and rank_v < k
    n = len(queries)
    single = np.array(single_iou)
    return {
        "params": p,
        "single": {"R1 IoU>=0.5": float((single >= 0.5).mean()), "R1 IoU>=0.7": float((single >= 0.7).mean()),
                   "mIoU": float(single.mean())},
        "corpus": {f"R@{k} IoU>={th}": hits[(k, th)] / n for k in CORPUS_KS for th in (0.5, 0.7)},
        "video": {f"R@{k}": video_hits[k] / n for k in CORPUS_KS},
    }  # fmt: skip


def latency(f: Frozen, text: np.ndarray, method: str, p: dict, n: int = 200) -> float:
    """Milliseconds per query, one query at a time as in the tool: score every frame, shortlist, rank moments.

    The text encoder is timed separately; the bank offsets are computed when indexing.
    """
    predict = PREDICT[method]
    offsets = p["lam"] * f.offsets(p["beta"]) if method in BANKED else None
    n = min(n, len(text))
    t0 = time.perf_counter()
    for qi in range(n):
        col = f.emb @ text[qi]
        if offsets is not None:
            col = col - offsets
        best = np.maximum.reduceat(col, f.starts)
        ranked = []
        for v in np.argsort(-best)[:SHORTLIST]:
            ranked += predict(f, col, int(v), p)
        ranked.sort(key=lambda x: -x[2])
    return 1000 * (time.perf_counter() - t0) / n


def search_space(method: str, budget: int, rng: random.Random) -> list[dict]:
    space = SPACE[method]
    combos = [dict(zip(space, vals, strict=True)) for vals in itertools.product(*space.values())]
    return rng.sample(combos, budget) if len(combos) > budget else combos


def grid_search(f: Frozen, queries: list, text: np.ndarray, budget: int = 60, seed: int = 0) -> dict:
    """Best parameters of each method for the primary metric. Exhaustive when small, random otherwise."""
    rng = random.Random(seed)
    scores = f.emb @ text.T
    best = {}

    def pick(method, combos, shared):
        scored = []
        for p in combos:
            r = run(f, queries, text, method, p, scores if shared else None)
            scored.append((r[PRIMARY[0]][PRIMARY[1]], p))
            print(f"{method} {p} -> {scored[-1][0]:.4f}", flush=True)
        scored.sort(key=lambda x: -x[0])
        best[method] = {"params": scored[0][1], "train_score": scored[0][0], "tried": len(combos)}

    for method in ("frame_window", "scene", "whenframe"):
        pick(method, search_space(method, budget, rng), shared=True)
    if f.bank is not None:
        bank = search_space("bank", budget, rng)
        for method, base in (("whenframe_bank", "whenframe"), ("frame_window_bank", "frame_window")):
            pick(method, [{**best[base]["params"], **p} for p in bank], shared=False)
    default = max((m for m in CANDIDATES if m in best), key=lambda m: best[m]["train_score"])
    best["default"] = default
    best["whenframe_dedup"] = {**best[default], "method": default}
    return best


def method_of(tuned: dict, method: str) -> str:
    """The method whose moments and scores a row uses: whenframe_dedup runs the default search."""
    return tuned["default"] if method == "whenframe_dedup" else method


def cost_curve(f: Frozen, queries: list, text: np.ndarray, method: str, p: dict) -> dict:
    """The primary metric against the share of frames encoded: skipping repeats, or a lower frame rate."""
    scores = frame_scores(f, text, method, p)
    out = {"repeats": [], "rate": []}
    for th in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0):
        mask = f.without_repeats(th) if th else np.ones(len(f.times), dtype=bool)
        r = run(f.select(mask), queries, text, method, p, scores[mask])
        out["repeats"].append({"threshold": th, "kept": float(mask.mean()), "score": r[PRIMARY[0]][PRIMARY[1]]})
        print(f"repeats {th}: kept {mask.mean():.3f} -> {out['repeats'][-1]['score']:.4f}", flush=True)
    for k in (1, 2, 3, 4, 5):
        mask = f.every(k)
        r = run(f.select(mask), queries, text, method, p, scores[mask])
        out["rate"].append({"every": k, "kept": float(mask.mean()), "score": r[PRIMARY[0]][PRIMARY[1]]})
        print(f"every {k}: kept {mask.mean():.3f} -> {out['rate'][-1]['score']:.4f}", flush=True)
    return out


def evaluate_all(f: Frozen, queries: list, text: np.ndarray, tuned: dict, methods: list[str]) -> dict:
    out = {}
    for method in methods:
        frozen = f.only_kept() if method == "whenframe_dedup" else f
        p = tuned[method]["params"]
        out[method] = run(frozen, queries, text, method_of(tuned, method), p)
        out[method]["query_ms"] = latency(frozen, text, method_of(tuned, method), p)
        print(f"{method}: {out[method]['corpus']}", flush=True)
    return out
