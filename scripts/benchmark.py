"""Shared evaluation code: methods, metrics and tuning on precomputed frame embeddings.

All methods see the same frame embeddings and the same query embeddings; they
only differ in how they turn per-frame scores into ranked moments.

    frame_window   the best frame of each video, with a fixed window around it
    scene          shots from ffmpeg's scene detector, each scored by its middle
                   frame (in the spirit of tools that index one keyframe per shot)
    whenframe      moments grown around peaks of the smoothed curve, ranked by
                   peak score plus contrast with the rest of the video
    whenframe_dedup  the same on the frames left after skipping repeats
"""

from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, replace

import numpy as np

from whenframe.moments import on_grid, smooth, spans

CORPUS_KS = [1, 5, 10, 100]
SHORTLIST = 64
METHODS = ["frame_window", "scene", "whenframe", "whenframe_dedup"]
PRIMARY = ("corpus", "R@5 IoU>=0.5")


@dataclass
class Frozen:
    names: list
    emb: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    durations: np.ndarray
    keep: np.ndarray
    cuts: list

    def only_kept(self) -> Frozen:
        idx = np.flatnonzero(self.keep)
        counts = np.array([self.keep[s : s + c].sum() for s, c in zip(self.starts, self.counts, strict=True)])
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return replace(self, emb=self.emb[idx], times=self.times[idx], starts=starts, counts=counts,
                       keep=np.ones(len(idx), bool))  # fmt: skip


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- methods
def frame_window(f: Frozen, scores: np.ndarray, v: int, p: dict) -> list[tuple[float, float, float]]:
    s, c = f.starts[v], f.counts[v]
    i = int(np.argmax(scores[s : s + c]))
    t, w, d = float(f.times[s + i]), p["window"], float(f.durations[v])
    start = min(max(0.0, t + 0.5 - w / 2), max(0.0, d - w))
    return [(start, min(d, start + w), float(scores[s + i]))]


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


PREDICT = {"frame_window": frame_window, "scene": scene, "whenframe": whenframe, "whenframe_dedup": whenframe}

SPACE = {
    "frame_window": {"window": [2.0, 3.0, 5.0, 7.0, 10.0, 15.0]},
    "scene": {"per_video": [1, 2, 3]},
    "whenframe": {"step": [0.5], "sigma": [0.0, 1.0, 2.0, 4.0], "alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  "contrast": [0.0, 0.5, 1.0, 2.0], "min_length": [2.0, 5.0], "per_video": [1, 2, 3]},
}  # fmt: skip


# ---------------------------------------------------------------- evaluation
def run(f: Frozen, queries: list, text: np.ndarray, method: str, p: dict) -> dict:
    predict = PREDICT[method]
    scores = f.emb @ text.T  # frames x queries
    best = np.maximum.reduceat(scores, f.starts, axis=0)  # videos x queries
    single_iou, hits = [], {(k, th): 0 for k in CORPUS_KS for th in (0.5, 0.7)}
    video_hits = dict.fromkeys(CORPUS_KS, 0)
    t0 = time.time()
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
        "query_ms": 1000 * (time.time() - t0) / n,
    }  # fmt: skip


def grid_search(f: Frozen, queries: list, text: np.ndarray, budget: int = 60, seed: int = 0) -> dict:
    """Best parameters of each method for the primary metric. Exhaustive when small, random otherwise."""
    rng = random.Random(seed)
    best = {}
    for method in ("frame_window", "scene", "whenframe"):
        space = SPACE[method]
        combos = [dict(zip(space, vals, strict=True)) for vals in itertools.product(*space.values())]
        if len(combos) > budget:
            combos = rng.sample(combos, budget)
        scored = []
        for p in combos:
            r = run(f, queries, text, method, p)
            scored.append((r[PRIMARY[0]][PRIMARY[1]], p))
            print(f"{method} {p} -> {scored[-1][0]:.4f}", flush=True)
        scored.sort(key=lambda x: -x[0])
        best[method] = {"params": scored[0][1], "train_score": scored[0][0], "tried": len(combos)}
    best["whenframe_dedup"] = best["whenframe"]
    return best


def evaluate_all(f: Frozen, queries: list, text: np.ndarray, tuned: dict) -> dict:
    out = {}
    for method in METHODS:
        frozen = f.only_kept() if method == "whenframe_dedup" else f
        out[method] = run(frozen, queries, text, method, tuned[method]["params"])
    return out
