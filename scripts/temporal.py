"""Trains whenframe's temporal model (whenframe/temporal.py) on DiDeMo's training videos (fifth round).

    python scripts/train_data.py ...                  # the training videos' frames, embedded
    python scripts/temporal.py texts --annotations didemo_train.json --model-dir ...
    python scripts/temporal.py train --annotations didemo_train.json --name base   # on the training videos,
                                                     # checked on the 500 tuning videos as it goes
    python scripts/temporal.py evaluate ...           # once, after docs/PROTOCOL.md says so

The model reads the frames of a video together and adds to each frame's
SigLIP 2 embedding what its neighbours show (whenframe/temporal.py). It is
trained with three losses, each after a published idea:

- between videos, SigLIP's own sigmoid loss (Zhai et al., 2023) on a video's
  best frame against each description of the batch, so that scores stay
  absolute and comparable across a library, as whenframe ranks them;
- within a video, the share of the frames' softmax that falls inside the
  described moment, so that the best frame is in it (the saliency losses of
  Moment-DETR and QD-DETR, Lei et al. 2021, Moon et al. 2023; the other moments
  of the same video are the hard negatives of ReLoCLNet, Zhang et al. 2021);
- a leash, 1 - cos(new, old), that keeps each frame near its SigLIP 2 embedding,
  so that what DiDeMo's descriptions never mention is still found.

Frames are dropped at random (the tool skips repeats) and a description's
moment is, half the time, one of its four annotators' instead of the most
common one. Needs JAX (pip install jax); the tool runs the model with numpy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from whenframe import temporal  # noqa: E402

CACHE = ROOT / "results" / "cache"
TRAIN = CACHE / "didemo_all_openvino.npz"
TEXTS = CACHE / "didemo_all_text.npy"
QUERIES = CACHE / "didemo_all_queries.json"
OUT = ROOT / "results" / "temporal"
SEGMENT = 5.0  # DiDeMo's moments are counted in 5-second segments


def stem(name: str) -> str:
    return Path(name).stem


def texts(args) -> None:
    """The descriptions of the embedded training videos, embedded by SigLIP 2 base's text tower."""
    from whenframe.encoder import Encoder

    z = np.load(TRAIN)
    have = set(z["names"].tolist())
    ann = [a for a in json.loads(Path(args.annotations).read_text()) if stem(a["video"]) in have]
    queries = [{"video": stem(a["video"]), "text": a["description"], "times": a["times"]} for a in ann]
    enc = Encoder("siglip2-base", folder=args.model_dir)
    t0 = time.time()
    out = np.concatenate([enc.encode_texts([q["text"] for q in queries[i : i + 64]])
                          for i in range(0, len(queries), 64)]).astype(np.float16)  # fmt: skip
    np.save(TEXTS, out)
    QUERIES.write_text(json.dumps(queries))
    print(f"{len(queries)} descriptions of {len(have)} videos embedded in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- the model, in JAX
def init(key, d: int = 256, layers: int = 2, heads: int = 4, dim: int = 768) -> dict:
    import jax

    keys = iter(jax.random.split(key, 8 + 6 * layers))

    def dense(n_in, n_out, scale=None):
        s = scale if scale is not None else 1.0 / math.sqrt(n_in)
        return np.asarray(jax.random.normal(next(keys), (n_in, n_out)) * s, dtype=np.float32)

    p = {"in_ln_g": np.ones(dim, np.float32), "in_ln_b": np.zeros(dim, np.float32), "in_w": dense(dim, d),
         "in_b": np.zeros(d, np.float32), "time_w": dense(2 * len(temporal.PERIODS), d, 0.1),
         "out_ln_g": np.ones(d, np.float32), "out_ln_b": np.zeros(d, np.float32),
         "out_w": np.zeros((d, dim), np.float32), "out_b": np.zeros(dim, np.float32)}  # fmt: skip
    for i in range(layers):
        p |= {f"l{i}_ln1_g": np.ones(d, np.float32), f"l{i}_ln1_b": np.zeros(d, np.float32),
              f"l{i}_qkv_w": dense(d, 3 * d), f"l{i}_qkv_b": np.zeros(3 * d, np.float32),
              f"l{i}_out_w": dense(d, d, 0.5 / math.sqrt(d)), f"l{i}_out_b": np.zeros(d, np.float32),
              f"l{i}_ln2_g": np.ones(d, np.float32), f"l{i}_ln2_b": np.zeros(d, np.float32),
              f"l{i}_fc1_w": dense(d, 2 * d), f"l{i}_fc1_b": np.zeros(2 * d, np.float32),
              f"l{i}_fc2_w": dense(2 * d, d, 0.5 / math.sqrt(2 * d)),
              f"l{i}_fc2_b": np.zeros(d, np.float32)}  # fmt: skip
    p["scale"], p["bias"] = np.float32(math.log(112.6689)), np.float32(-16.7717)  # SigLIP 2 base's own
    # the head, run at search time on a description and the frames of a shortlisted video
    h = d // 2
    p |= {"h_f_ln_g": np.ones(dim, np.float32), "h_f_ln_b": np.zeros(dim, np.float32), "h_f_w": dense(dim, h),
          "h_q_w": dense(dim, h), "h_q_b": np.ones(h, np.float32), "h_m_w": dense(h, h),
          "h_m_b": np.zeros(h, np.float32),
          "h_time_w": dense(2 * len(temporal.PERIODS), h, 0.1), "h_ln1_g": np.ones(h, np.float32),
          "h_ln1_b": np.zeros(h, np.float32), "h_qkv_w": dense(h, 3 * h),
          "h_qkv_b": np.zeros(3 * h, np.float32),
          "h_out_w": dense(h, h, 0.5 / math.sqrt(h)), "h_out_b": np.zeros(h, np.float32),
          "h_ln2_g": np.ones(h, np.float32), "h_ln2_b": np.zeros(h, np.float32), "h_s_w": np.zeros(h, np.float32),
          "h_s_b": np.float32(0.0), "h_d_w": dense(h, 2, 0.01), "h_d_b": np.zeros(2, np.float32)}  # fmt: skip
    return p


def forward(p: dict, x, t, mask, layers: int, heads: int, key=None, dropout: float = 0.0):
    """(B, L, 768) unit embeddings, (B, L) times and valid frames -> (B, L, 768) unit embeddings."""
    import jax
    import jax.numpy as jnp

    def ln(v, g, b):
        mu = v.mean(-1, keepdims=True)
        return (v - mu) / jnp.sqrt(((v - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b

    def drop(v, k):
        if key is None or dropout <= 0:
            return v
        return v * jax.random.bernoulli(k, 1 - dropout, v.shape) / (1 - dropout)

    keys = jax.random.split(key, 2 * layers) if key is not None else [None] * (2 * layers)
    rel = (t - t[:, :1])[..., None] * (2 * jnp.pi / jnp.array(temporal.PERIODS))
    h = ln(x, p["in_ln_g"], p["in_ln_b"]) @ p["in_w"] + p["in_b"]
    h = h + jnp.concatenate([jnp.sin(rel), jnp.cos(rel)], -1) @ p["time_w"]
    B, L, d = h.shape
    allowed = mask[:, None, None, :]  # attend to real frames only
    for i in range(layers):
        q = ln(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        qkv = q @ p[f"l{i}_qkv_w"] + p[f"l{i}_qkv_b"]
        qs, ks, vs = (qkv[..., j * d : (j + 1) * d].reshape(B, L, heads, d // heads).transpose(0, 2, 1, 3)
                      for j in range(3))  # fmt: skip
        att = jnp.where(allowed, qs @ ks.transpose(0, 1, 3, 2) / math.sqrt(d // heads), -1e9)
        att = jax.nn.softmax(att, axis=-1)
        h = h + drop((att @ vs).transpose(0, 2, 1, 3).reshape(B, L, d) @ p[f"l{i}_out_w"] + p[f"l{i}_out_b"],
                     keys[2 * i])  # fmt: skip
        m = ln(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        g = jax.nn.gelu(m @ p[f"l{i}_fc1_w"] + p[f"l{i}_fc1_b"], approximate=True)
        h = h + drop(g @ p[f"l{i}_fc2_w"] + p[f"l{i}_fc2_b"], keys[2 * i + 1])
    out = x + ln(h, p["out_ln_g"], p["out_ln_b"]) @ p["out_w"] + p["out_b"]
    # padded frames are zero: a norm with its square root at zero would make every gradient NaN
    return out / jnp.sqrt(jnp.sum(out**2, axis=-1, keepdims=True) + 1e-12)


def head(p, new, t, mask, q, heads: int):
    """(N, L, 768) adapted frames of N videos, each with the description q (N, 768) it is searched for ->
    foreground logits (N, L) and the moment's reach before and after each frame's middle, in seconds (N, L, 2)."""
    import jax
    import jax.numpy as jnp

    def ln(v, g, b):
        mu = v.mean(-1, keepdims=True)
        return (v - mu) / jnp.sqrt(((v - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b

    u = ln(new, p["h_f_ln_g"], p["h_f_ln_b"]) @ p["h_f_w"]
    c = q @ p["h_q_w"] + p["h_q_b"]
    rel = (t - t[:, :1])[..., None] * (2 * jnp.pi / jnp.array(temporal.PERIODS))
    e = jax.nn.gelu((u * c[:, None, :]) @ p["h_m_w"] + p["h_m_b"], approximate=True)
    e = e + jnp.concatenate([jnp.sin(rel), jnp.cos(rel)], -1) @ p["h_time_w"]
    N, L, h = e.shape
    k = ln(e, p["h_ln1_g"], p["h_ln1_b"]) @ p["h_qkv_w"] + p["h_qkv_b"]
    qs, ks, vs = (k[..., j * h : (j + 1) * h].reshape(N, L, heads, h // heads).transpose(0, 2, 1, 3) for j in range(3))
    att = jnp.where(mask[:, None, None, :], qs @ ks.transpose(0, 1, 3, 2) / math.sqrt(h // heads), -1e9)
    e = e + (jax.nn.softmax(att, -1) @ vs).transpose(0, 2, 1, 3).reshape(N, L, h) @ p["h_out_w"] + p["h_out_b"]
    e = ln(e, p["h_ln2_g"], p["h_ln2_b"])
    a, b = jnp.exp(p["scale"]), p["bias"]
    fg = a * jnp.einsum("nld,nd->nl", new, q) + b + e @ p["h_s_w"] + p["h_s_b"]  # SigLIP's logit, corrected
    reach = jax.nn.softplus(e @ p["h_d_w"] + p["h_d_b"]) * 5.0
    return fg, reach


def head_loss(p, new, t, mask, q, inside, start, end, heads: int, negatives: int, key):
    """Foreground of the description's frames, against the rest of its video and against other videos, and
    the reach of the frames inside the moment to its two ends (L1 and generalised IoU, as in Moment-DETR)."""
    import jax
    import jax.numpy as jnp

    n = q.shape[0]
    fg, reach = head(p, new, t, mask, q, heads)
    pos = jnp.where(inside, 1.0, 0.0)
    frames = jnp.sum(jnp.where(mask, jax.nn.softplus(fg) - pos * fg, 0.0)) / jnp.sum(mask)  # binary cross-entropy
    # other videos, as many per description: their frames are all background for it
    shift = jax.random.randint(key, (negatives,), 1, n)
    others = [(jnp.arange(n) + s) % n for s in shift]
    neg_fg = jnp.stack([head(p, new[o], t[o], mask[o], q, heads)[0] for o in others], 1)  # (n, K, L)
    neg_best = jax.nn.logsumexp(jnp.where(mask[jnp.stack(others, 1)], neg_fg, -1e9), -1)
    own_best = jax.nn.logsumexp(jnp.where(mask, fg, -1e9), -1)
    rank = jnp.mean(jax.nn.softplus(-own_best)) + jnp.mean(jax.nn.softplus(neg_best))
    mid = t + 0.5
    lo, hi = mid - reach[..., 0], mid + reach[..., 1]
    inter = jnp.maximum(0.0, jnp.minimum(hi, end[:, None]) - jnp.maximum(lo, start[:, None]))
    union = jnp.maximum(hi, end[:, None]) - jnp.minimum(lo, start[:, None])
    giou = 1 - inter / jnp.maximum(union, 1e-6)
    l1 = (jnp.abs(lo - start[:, None]) + jnp.abs(hi - end[:, None])) / 10.0
    bounds = jnp.sum((giou + l1) * pos) / jnp.maximum(jnp.sum(pos), 1.0)
    return frames + rank + bounds, (frames, rank, bounds)


def loss(p, batch, layers, heads, key, cfg):
    import jax
    import jax.numpy as jnp

    x, t, mask, q, inside, start, end = batch
    key, sub = jax.random.split(key)
    new = forward(p, x, t, mask, layers, heads, key, cfg["dropout"])
    a, b = jnp.exp(p["scale"]), p["bias"]
    z = a * jnp.einsum("vld,qd->vql", new, q) + b  # every video's frames against every description
    z = jnp.where(mask[:, None, :], z, -1e9)
    best = jax.nn.logsumexp(z, axis=-1)  # a smooth best frame of each video, for each description
    n = q.shape[0]
    y = 2 * jnp.eye(n) - 1
    between = jnp.sum(jax.nn.softplus(-y * best)) / n
    own = jnp.diagonal(z, axis1=0, axis2=1).T  # (n, L): each description against its own video
    within = jnp.mean(jax.nn.logsumexp(own, axis=-1) - jax.nn.logsumexp(jnp.where(inside, own, -1e9), axis=-1))
    leash = jnp.sum((1 - jnp.sum(new * x, -1)) * mask) / jnp.sum(mask)
    total = between + cfg["within"] * within + cfg["leash"] * leash
    found, parts = head_loss(p, new, t, mask, q, inside, start, end, heads, cfg["negatives"], sub)
    return total + cfg["head"] * found, (between, within, leash, *parts)


# ---------------------------------------------------------------- data
class Videos:
    """Frames of the training videos and their descriptions, in batches."""

    def __init__(self, seed: int):
        z = np.load(TRAIN)
        self.names = z["names"].tolist()
        self.emb, self.times, self.keep = z["emb"], z["times"], z["keep"]
        self.starts, self.counts, self.durations = z["starts"], z["counts"], z["durations"]
        index = {n: i for i, n in enumerate(self.names)}
        self.text = np.load(TEXTS)
        self.queries = json.loads(QUERIES.read_text())
        self.by_video: dict[int, list[int]] = {}
        for qi, q in enumerate(self.queries):
            self.by_video.setdefault(index[q["video"]], []).append(qi)
        self.videos = sorted(self.by_video)
        self.rng = random.Random(seed)

    def moment(self, qi: int, v: int) -> tuple[float, float]:
        times = self.queries[qi]["times"]
        if self.rng.random() < 0.5:
            a, b = self.rng.choice(times)  # one annotator's
        else:
            (a, b), _ = Counter(map(tuple, times)).most_common(1)[0]
        return a * SEGMENT, min((b + 1) * SEGMENT, max(float(self.durations[v]), a * SEGMENT + 1.0))

    def batch(self, n: int, length: int = temporal.WINDOW):
        vids = self.rng.sample(self.videos, n)
        x = np.zeros((n, length, 768), np.float32)
        t = np.zeros((n, length), np.float32)
        mask = np.zeros((n, length), bool)
        inside = np.zeros((n, length), bool)
        q = np.zeros((n, 768), np.float32)
        ends = np.zeros((n, 2), np.float32)
        for i, v in enumerate(vids):
            s, c = int(self.starts[v]), int(self.counts[v])
            rows = np.arange(s, s + c)
            if self.rng.random() < 0.5:  # as the tool indexes: repeats skipped
                rows = rows[self.keep[s : s + c]]
            if self.rng.random() < 0.3:  # and fewer frames still, sometimes
                rows = rows[np.sort(self.rng.sample(range(len(rows)), max(1, int(len(rows) * 0.7))))]
            rows = rows[:length]
            qi = self.rng.choice(self.by_video[v])
            a, b = self.moment(qi, v)
            tt = self.times[rows].astype(np.float32)
            inn = (tt + 0.5 >= a) & (tt + 0.5 < b)
            if not inn.any():  # the moment's frames were all dropped: keep the nearest one
                inn[np.argmin(np.abs(tt + 0.5 - (a + b) / 2))] = True
            k = len(rows)
            x[i, :k], t[i, :k], mask[i, :k], inside[i, :k] = self.emb[rows], tt, True, inn
            q[i] = self.text[qi]
            ends[i] = a, b
        return x, t, mask, q, inside, ends[:, 0], ends[:, 1]


# ---------------------------------------------------------------- checking on the tuning videos
class Tuning:
    """The 500 tuning videos as the tool indexes them (OpenVINO, repeats skipped), and the default search."""

    def __init__(self, args):
        import didemo

        frozen, self.queries, _ = didemo.load("train", args.annotations)
        frozen.bank = didemo.bank(args)
        self.text = didemo.text_embeddings(self.queries, "train", args.model_dir)
        ov = np.load(didemo.CACHE / f"didemo_train{didemo.suffix(didemo.MAIN, 'openvino')}.npz")
        base = replace(frozen, emb=ov["emb"][ov["times"] < didemo.SECONDS].astype(np.float32), _offsets={})
        self.kept = base.only_kept()
        tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
        self.method = tuned["default"]
        self.params = tuned[self.method]["params"]

    def adapted(self, p: dict) -> np.ndarray:
        k = self.kept
        out = np.empty_like(k.emb)
        for s, c in zip(k.starts, k.counts, strict=True):
            out[s : s + c] = temporal.apply(p, k.emb[s : s + c], k.times[s : s + c])
        return out

    def score(self, p: dict | None) -> dict:
        from benchmark import run

        emb = None if p is None else self.adapted(p)
        f = self.kept if p is None else replace(self.kept, emb=emb, _offsets={})
        r = run(f, self.queries, self.text, self.method, self.params)
        out = {"R@5": r["corpus"]["R@5 IoU>=0.5"], "R@1": r["corpus"]["R@1 IoU>=0.5"], "video_R@5": r["video"]["R@5"],
               "single_R1@0.5": r["single"]["R1 IoU>=0.5"]}  # fmt: skip
        if p is not None:
            h = search_with_head(p, self.kept, self.queries, self.text, self.params["lam"], self.params["beta"],
                                 self.kept.bank, emb=emb)  # fmt: skip
            out |= {"head_R@5": h["corpus"]["R@5 IoU>=0.5"], "head_R@1": h["corpus"]["R@1 IoU>=0.5"],
                    "head_single_R1@0.5": h["single"]["R1 IoU>=0.5"]}  # fmt: skip
        return out


def search_with_head(
    p: dict,
    f,
    queries: list,
    text: np.ndarray,
    lam: float,
    beta: float,
    bank: np.ndarray,
    emb: np.ndarray | None = None,
    count: int = 3,
) -> dict:
    """The whole search with the model: adapted frames, the query bank, the 64 best videos shortlisted, the
    head placing up to `count` moments in each, all moments ranked by the head's corrected logit. Metrics as
    benchmark.run's."""
    import jax
    import jax.numpy as jnp
    from benchmark import CORPUS_KS, SHORTLIST, iou

    from whenframe.bank import offsets as bank_offsets

    if emb is None:
        emb = np.concatenate([temporal.apply(p, f.emb[s : s + c], f.times[s : s + c])
                              for s, c in zip(f.starts, f.counts, strict=True)])  # fmt: skip
    off = bank_offsets(emb, bank, beta)
    a = float(np.exp(p["scale"]))
    nv, length = len(f.starts), int(f.counts.max())
    X = np.zeros((nv, length, emb.shape[1]), np.float32)
    T = np.zeros((nv, length), np.float32)
    M = np.zeros((nv, length), bool)
    OFF = np.zeros((nv, length), np.float32)
    for v, (s, c) in enumerate(zip(f.starts, f.counts, strict=True)):
        X[v, :c], T[v, :c], M[v, :c], OFF[v, :c] = emb[s : s + c], f.times[s : s + c], True, off[s : s + c]
    jp = {k: jnp.asarray(v) for k, v in p.items() if k not in ("layers", "heads")}
    run_head = jax.jit(lambda xs, ts, ms, qs: head(jp, xs, ts, ms, qs, int(p["heads"])))
    hits = {(k, th): 0 for k in CORPUS_KS for th in (0.5, 0.7)}
    video_hits = dict.fromkeys(CORPUS_KS, 0)
    single = []
    for qi, query in enumerate(queries):
        corrected = emb @ text[qi] - lam * off
        best = np.maximum.reduceat(corrected, f.starts)
        short = np.argsort(-best)[:SHORTLIST]
        vids = np.concatenate([short, [query["video"]]])  # the right video too, for the single-video metrics
        fg, reach = run_head(X[vids], T[vids], M[vids], jnp.repeat(jnp.asarray(text[qi])[None], len(vids), 0))
        fg = np.asarray(fg) - a * lam * OFF[vids]
        reach = np.asarray(reach)
        ranked = []
        for j, v in enumerate(vids[:-1]):
            c = int(f.counts[v])
            for lo, hi, i in temporal.moments(fg[j, :c], reach[j, :c], T[v, :c], float(f.durations[v]), count):
                ranked.append((int(v), lo, hi, float(fg[j, i])))
        ranked.sort(key=lambda m: -m[3])
        ranked = ranked[: max(CORPUS_KS)]
        first_ok = {th: None for th in (0.5, 0.7)}
        for r, (v, lo, hi, _) in enumerate(ranked):
            if v == query["video"]:
                o = iou((lo, hi), query["moment"])
                for th in (0.5, 0.7):
                    if first_ok[th] is None and o >= th:
                        first_ok[th] = r
        for th, r in first_ok.items():
            for k in CORPUS_KS:
                hits[(k, th)] += r is not None and r < k
        order = list(dict.fromkeys(v for v, *_ in ranked))
        rank_v = order.index(query["video"]) if query["video"] in order else None
        for k in CORPUS_KS:
            video_hits[k] += rank_v is not None and rank_v < k
        v, c = query["video"], int(f.counts[query["video"]])
        top = temporal.moments(fg[-1, :c], reach[-1, :c], T[v, :c], float(f.durations[v]), 1)
        single.append(iou(top[0][:2], query["moment"]) if top else 0.0)
    n, single = len(queries), np.array(single)
    return {"corpus": {f"R@{k} IoU>={th}": hits[(k, th)] / n for k in CORPUS_KS for th in (0.5, 0.7)},
            "video": {f"R@{k}": video_hits[k] / n for k in CORPUS_KS},
            "single": {"R1 IoU>=0.5": float((single >= 0.5).mean()), "R1 IoU>=0.7": float((single >= 0.7).mean()),
                       "mIoU": float(single.mean())}}  # fmt: skip


def numpy_params(p: dict, layers: int, heads: int) -> dict:
    out = {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}
    out["layers"], out["heads"] = layers, heads
    return out


def train(args) -> None:
    import jax
    import jax.numpy as jnp

    cfg = {"within": args.within, "leash": args.leash, "dropout": args.dropout, "head": args.head,
           "negatives": args.negatives}  # fmt: skip
    data, tuning = Videos(args.seed), Tuning(args)
    key = jax.random.PRNGKey(args.seed)
    params = jax.tree.map(jnp.asarray, init(key, args.width, args.layers, args.heads))
    adam_m = jax.tree.map(jnp.zeros_like, params)
    adam_v = jax.tree.map(jnp.zeros_like, params)
    decayed = {k: k.endswith("_w") for k in params}

    @jax.jit
    def step(params, adam_m, adam_v, batch, key, lr, i):
        (value, parts), grads = jax.value_and_grad(loss, has_aux=True)(params, batch, args.layers, args.heads, key,
                                                                         cfg)  # fmt: skip
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0 / (norm + 1e-9)), grads)
        adam_m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, adam_m, grads)
        adam_v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2, adam_v, grads)
        c1, c2 = 1 - 0.9 ** (i + 1), 1 - 0.999 ** (i + 1)

        def update(k, p, m, v):
            p = p - lr * (m / c1) / (jnp.sqrt(v / c2) + 1e-8)
            return p * (1 - lr * args.decay) if decayed[k] else p

        params = {k: update(k, params[k], adam_m[k], adam_v[k]) for k in params}
        return params, adam_m, adam_v, value, parts

    base = tuning.score(None)
    log = {"config": vars(args) | {"annotations": None}, "tuning_without": base, "checks": []}
    print(f"tuning videos without the model: R@5 {100 * base['R@5']:.1f}", flush=True)
    best, best_params = -1.0, None
    t0 = time.time()
    for i in range(args.steps):
        warm = min(1.0, (i + 1) / args.warmup)
        lr = args.lr * warm * 0.5 * (1 + math.cos(math.pi * i / args.steps))
        key, sub = jax.random.split(key)
        batch = tuple(jnp.asarray(b) for b in data.batch(args.batch))
        params, adam_m, adam_v, value, parts = step(params, adam_m, adam_v, batch, sub, lr, i)
        if (i + 1) % args.every == 0 or i + 1 == args.steps:
            p = numpy_params(jax.device_get(params), args.layers, args.heads)
            s = tuning.score(p)
            names = ("between", "within", "leash", "frames", "rank", "bounds")
            losses = {k: float(v) for k, v in zip(names, parts, strict=True)}
            log["checks"].append({"step": i + 1, "loss": float(value), **losses, **s, "seconds": time.time() - t0})
            print(f"step {i + 1}: loss {float(value):.3f} ({', '.join(f'{k} {v:.3f}' for k, v in losses.items())}); "
                  f"tuning R@5 {100 * s['R@5']:.1f}, with the head {100 * s['head_R@5']:.1f}, single "
                  f"{100 * s['single_R1@0.5']:.1f} / {100 * s['head_single_R1@0.5']:.1f}; {time.time() - t0:.0f} s",
                  flush=True)  # fmt: skip
            score = max(s["R@5"], s["head_R@5"])
            if score > best:
                best, best_params = score, p
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.name}.npz", **best_params)
    log["best"] = best
    (OUT / f"{args.name}.json").write_text(json.dumps(log, indent=1))
    print(f"best tuning R@5 {100 * best:.1f} (without: {100 * base['R@5']:.1f}); wrote {OUT / args.name}.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["texts", "train"])
    ap.add_argument("--annotations", required=True, help="DiDeMo's didemo_train.json")
    ap.add_argument("--train-annotations", help="the same, for the query bank (default: --annotations)")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--name", default="temporal")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--within", type=float, default=1.0, help="weight of the loss within a video")
    ap.add_argument("--leash", type=float, default=1.0, help="weight of the leash to SigLIP 2's embeddings")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head", type=float, default=1.0, help="weight of the head's losses")
    ap.add_argument("--negatives", type=int, default=4, help="other videos per description, for the head")
    ap.add_argument("--every", type=int, default=500, help="steps between two checks on the tuning videos")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.train_annotations = args.train_annotations or args.annotations
    args.split = "train"
    {"texts": texts, "train": train}[args.stage](args)


if __name__ == "__main__":
    main()
