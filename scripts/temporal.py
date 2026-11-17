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
FIXED = ("scale", "bias", "lam", "beta")  # SigLIP 2's logit scale and bias, the search's bank correction


def stem(name: str) -> str:
    return Path(name).stem


def texts(args) -> None:
    """The descriptions of the training videos (the tuning videos' left out), embedded by SigLIP 2 base's text
    tower; those of videos that could not be read are left out when training."""
    from whenframe.encoder import Encoder

    tuning = {stem(n) for n in (ROOT / "results" / "didemo_train_videos.txt").read_text().split()}
    ann = [a for a in json.loads(Path(args.annotations).read_text()) if stem(a["video"]) not in tuning]
    queries = [{"video": stem(a["video"]), "text": a["description"], "times": a["times"]} for a in ann]
    enc = Encoder("siglip2-base", folder=args.model_dir)
    t0 = time.time()
    out = np.concatenate([enc.encode_texts([q["text"] for q in queries[i : i + 64]])
                          for i in range(0, len(queries), 64)]).astype(np.float16)  # fmt: skip
    np.save(TEXTS, out)
    QUERIES.write_text(json.dumps(queries))
    print(
        f"{len(queries)} descriptions of {len({q['video'] for q in queries})} videos embedded in "
        f"{time.time() - t0:.0f} s"
    )


# ---------------------------------------------------------------- the model, in JAX
def init(
    key, d: int = 256, layers: int = 2, heads: int = 4, dim: int = 768, lam: float = 1.0, beta: float = 50.0
) -> dict:
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
    p["lam"], p["beta"] = np.float32(lam), np.float32(beta)  # the query bank's correction, as the search's
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


def head(p, new, t, mask, q, off, heads: int, u=None):
    """(N, L, 768) adapted frames of N videos and their bank offsets (N, L), each with the description q (N, 768)
    it is searched for -> foreground logits (N, L) and the moment's reach before and after each frame's middle,
    in seconds (N, L, 2). u: the frames' projection, which does not depend on the description, when known."""
    import jax
    import jax.numpy as jnp

    def ln(v, g, b):
        mu = v.mean(-1, keepdims=True)
        return (v - mu) / jnp.sqrt(((v - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b

    if u is None:
        u = frames_projection(p, new)
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
    fg = a * (jnp.einsum("nld,nd->nl", new, q) - p["lam"] * off) + b + e @ p["h_s_w"] + p["h_s_b"]
    reach = jax.nn.softplus(e @ p["h_d_w"] + p["h_d_b"]) * 5.0
    return fg, reach


def frames_projection(p, new):
    import jax.numpy as jnp

    mu = new.mean(-1, keepdims=True)
    v = (new - mu) / jnp.sqrt(((new - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
    return (v * p["h_f_ln_g"] + p["h_f_ln_b"]) @ p["h_f_w"]


def head_loss(p, new, t, mask, q, off, inside, start, end, heads: int, negatives: int, key):
    """Foreground of the description's frames, against the rest of its video and against other videos, and
    the reach of the frames inside the moment to its two ends (L1 and generalised IoU, as in Moment-DETR)."""
    import jax
    import jax.numpy as jnp

    n = q.shape[0]
    u = frames_projection(p, new)
    fg, reach = head(p, new, t, mask, q, off, heads, u)
    pos = jnp.where(inside, 1.0, 0.0)
    frames = jnp.sum(jnp.where(mask, jax.nn.softplus(fg) - pos * fg, 0.0)) / jnp.sum(mask)  # binary cross-entropy
    # other videos, as many per description: their frames are all background for it
    shift = jax.random.randint(key, (negatives,), 1, n)
    others = [(jnp.arange(n) + s) % n for s in shift]
    neg_fg = jnp.stack([head(p, new[o], t[o], mask[o], q, off[o], heads, u[o])[0] for o in others], 1)  # (n, K, L)
    neg_best = jax.nn.logsumexp(jnp.where(mask[jnp.stack(others, 1)], neg_fg, -1e9), -1)
    own_best = jax.nn.logsumexp(jnp.where(mask, fg, -1e9), -1)
    rank = jnp.mean(jax.nn.softplus(-own_best)) + jnp.mean(jax.nn.softplus(neg_best))
    mid = t + 0.5
    lo, hi = mid - reach[..., 0], mid + reach[..., 1]
    inter = jnp.maximum(0.0, jnp.minimum(hi, end[:, None]) - jnp.maximum(lo, start[:, None]))
    union = (hi - lo) + (end - start)[:, None] - inter
    hull = jnp.maximum(hi, end[:, None]) - jnp.minimum(lo, start[:, None])
    giou = 1 - inter / jnp.maximum(union, 1e-6) + (hull - union) / jnp.maximum(hull, 1e-6)  # 1 - GIoU
    l1 = (jnp.abs(lo - start[:, None]) + jnp.abs(hi - end[:, None])) / 10.0
    bounds = jnp.sum((giou + l1) * pos) / jnp.maximum(jnp.sum(pos), 1.0)
    return frames + rank + bounds, (frames, rank, bounds)


def bank_offsets(new, bank, beta):
    """(B, L, 768) frames, (K, 768) bank descriptions -> (B, L): whenframe.bank.offsets, in JAX."""
    import jax
    import jax.numpy as jnp

    return (jax.nn.logsumexp(beta * (new @ bank.T), axis=-1) - jnp.log(bank.shape[0])) / beta


def loss(p, batch, layers, heads, key, cfg):
    import jax
    import jax.numpy as jnp

    x, t, mask, q, inside, start, end, bank = batch
    key, sub = jax.random.split(key)
    new = forward(p, x, t, mask, layers, heads, key, cfg["dropout"])
    off = bank_offsets(new, bank, p["beta"])  # (B, L), from a part of the bank drawn for the batch
    a = jnp.exp(p["scale"])
    # every video's frames against every description, as the search scores them: bank-corrected similarity
    z = a * (jnp.einsum("vld,qd->vql", new, q) - p["lam"] * off[:, None, :])
    z = jnp.where(mask[:, None, :], z, -1e9)
    best = jax.nn.logsumexp(z, axis=-1)  # (V, Q): a smooth best frame of each video, for each description
    # each description's own video, among the batch's (InfoNCE, as CLIP4Clip's): the corpus search's ranking
    between = jnp.mean(jax.nn.logsumexp(best, axis=0) - jnp.diagonal(best))
    own = jnp.diagonal(z, axis1=0, axis2=1).T  # (n, L): each description against its own video
    within = jnp.mean(jax.nn.logsumexp(own, axis=-1) - jax.nn.logsumexp(jnp.where(inside, own, -1e9), axis=-1))
    leash = jnp.sum((1 - jnp.sum(new * x, -1)) * mask) / jnp.sum(mask)
    total = cfg["between"] * between + cfg["within"] * within + cfg["leash"] * leash
    if not cfg["head"]:  # the adapter alone
        zero = jnp.float32(0.0)
        return total, (between, within, leash, zero, zero, zero)
    sg = jax.lax.stop_gradient  # the head learns from the adapter's output; its losses do not reshape it
    found, parts = head_loss(p, sg(new), t, mask, q, sg(off), inside, start, end, heads, cfg["negatives"], sub)
    return total + cfg["head"] * found, (between, within, leash, *parts)


# ---------------------------------------------------------------- data
class Videos:
    """Frames of the training videos and their descriptions, in batches."""

    def __init__(self, seed: int, bank: np.ndarray, bank_sample: int):
        self.bank, self.bank_sample = bank.astype(np.float32), bank_sample
        z = np.load(TRAIN)
        self.names = z["names"].tolist()
        self.emb, self.times, self.keep = z["emb"], z["times"], z["keep"]
        self.starts, self.counts, self.durations = z["starts"], z["counts"], z["durations"]
        index = {n: i for i, n in enumerate(self.names)}
        self.text = np.load(TEXTS)
        self.queries = json.loads(QUERIES.read_text())
        self.by_video: dict[int, list[int]] = {}
        for qi, q in enumerate(self.queries):
            if q["video"] in index:  # the others' videos were not embedded (yet, or could not be read)
                self.by_video.setdefault(index[q["video"]], []).append(qi)
        self.videos = sorted(self.by_video)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

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
            qi = self.rng.choice(self.by_video[v])
            a, b = self.moment(qi, v)
            tt = self.times[rows].astype(np.float32)
            inn = (tt + 0.5 >= a) & (tt + 0.5 < b)
            if inn.any() and self.rng.random() < 0.5:
                # a window that starts or ends elsewhere than the video, as in a long video: no frame of the
                # moment is cut, so what is learnt of a moment's ends is not where DiDeMo's videos start
                first, last = np.flatnonzero(inn)[[0, -1]]
                lo = self.rng.randint(0, min(int(first), 8))
                hi = len(rows) - self.rng.randint(0, min(len(rows) - 1 - int(last), 8))
                rows, tt, inn = rows[lo:hi], tt[lo:hi], inn[lo:hi]
            rows, tt, inn = rows[:length], tt[:length], inn[:length]
            if not inn.any():  # the moment's frames were all dropped: keep the nearest one
                inn[np.argmin(np.abs(tt + 0.5 - (a + b) / 2))] = True
            k = len(rows)
            x[i, :k], t[i, :k], mask[i, :k], inside[i, :k] = self.emb[rows], tt, True, inn
            q[i] = self.text[qi]
            ends[i] = a, b
        part = self.bank[self.np_rng.choice(len(self.bank), self.bank_sample, replace=False)]
        return x, t, mask, q, inside, ends[:, 0], ends[:, 1], part


# ---------------------------------------------------------------- checking on the tuning videos
class DiDeMo:
    """DiDeMo's videos of one split as the tool indexes them (OpenVINO, repeats skipped), and the default search:
    the 500 tuning videos ("train"), or the test videos."""

    def __init__(self, args, split: str = "train"):
        import didemo

        frozen, self.queries, _ = didemo.load(split, args.annotations if split == "train" else args.test_annotations)
        frozen.bank = didemo.bank(args)
        self.text = didemo.text_embeddings(self.queries, split, args.model_dir)
        ov = np.load(didemo.CACHE / f"didemo_{split}{didemo.suffix(didemo.MAIN, 'openvino')}.npz")
        base = replace(frozen, emb=ov["emb"][ov["times"] < didemo.SECONDS].astype(np.float32), _offsets={})
        self.kept = base.only_kept()
        tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
        self.method = tuned["default"]
        self.params = tuned[self.method]["params"]

    def adapted(self, p: dict) -> np.ndarray:
        return adapt(p, self.kept.emb, self.kept.times, self.kept.starts, self.kept.counts)

    def score(self, p: dict | None, with_head: bool = True) -> dict:
        from benchmark import run

        emb = None if p is None else self.adapted(p)
        f = self.kept if p is None else replace(self.kept, emb=emb, _offsets={})
        r = run(f, self.queries, self.text, self.method, self.params)
        out = {"R@5": r["corpus"]["R@5 IoU>=0.5"], "R@1": r["corpus"]["R@1 IoU>=0.5"], "video_R@5": r["video"]["R@5"],
               "single_R1@0.5": r["single"]["R1 IoU>=0.5"]}  # fmt: skip
        if p is not None and with_head:
            h = search_with_head(p, self.kept, self.queries, self.text, self.kept.bank, emb=emb)
            w = search_with_head(p, self.kept, self.queries, self.text, self.kept.bank, emb=emb, place="windows")
            out |= {"head_R@5": h["corpus"]["R@5 IoU>=0.5"], "head_R@1": h["corpus"]["R@1 IoU>=0.5"],
                    "head_single_R1@0.5": h["single"]["R1 IoU>=0.5"], "rescored_R@5": w["corpus"]["R@5 IoU>=0.5"],
                    "rescored_single_R1@0.5": w["single"]["R1 IoU>=0.5"]}  # fmt: skip
        return out


# ---------------------------------------------------------------- the head in numpy, as the search would run it
# Tried in the fifth round and not kept (docs/PROTOCOL.md): neither its moments nor its scores did better on the
# tuning videos than the default search's windows on the adapter's frames.
def numpy_head(p: dict, x: np.ndarray, t: np.ndarray, q: np.ndarray, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """At search time, for one description q and the adapted frames of one video (one window), with their
    query-bank offsets: each frame's foreground logit (SigLIP 2's logit of the bank-corrected similarity, as the
    search scores frames, plus the head's correction) and how far the moment reaches before and after the
    frame's middle, in seconds."""
    n = len(x)
    fg, reach = numpy_head_batch(p, x[None], np.asarray(t, dtype=np.float64)[None], np.ones((1, n), bool), q, off[None])
    return fg[0], reach[0]


def numpy_frames_projection(p: dict, x: np.ndarray) -> np.ndarray:
    """The part of the head that does not depend on the description: computed once for a library's frames."""
    return temporal.layer_norm(x, p["h_f_ln_g"], p["h_f_ln_b"]) @ p["h_f_w"]


def numpy_head_batch(p: dict, x: np.ndarray, t: np.ndarray, mask: np.ndarray, q: np.ndarray, off: np.ndarray,
               u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """numpy_head() for N windows of at most WINDOW frames at once: (N, L, D) frames, (N, L) times and which frames
    are real, (D,) the description, (N, L) offsets; u: frames_projection(x), when already known."""
    u = numpy_frames_projection(p, x) if u is None else u
    c = q @ p["h_q_w"] + p["h_q_b"]
    rel = (t - t[:, :1])[..., None] * (2 * np.pi / np.array(temporal.PERIODS))
    tf = np.concatenate([np.sin(rel), np.cos(rel)], axis=-1).astype(np.float32)
    e = temporal.gelu((u * c) @ p["h_m_w"] + p["h_m_b"]) + tf @ p["h_time_w"]
    n, length, h = e.shape
    heads = p["heads"]
    k = temporal.layer_norm(e, p["h_ln1_g"], p["h_ln1_b"]) @ p["h_qkv_w"] + p["h_qkv_b"]
    qs, ks, vs = (k[..., j * h : (j + 1) * h].reshape(n, length, heads, h // heads).transpose(0, 2, 1, 3)
                  for j in range(3))  # fmt: skip
    att = np.where(mask[:, None, None, :], qs @ ks.transpose(0, 1, 3, 2) / np.sqrt(h // heads), -np.inf)
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    e = e + (att @ vs).transpose(0, 2, 1, 3).reshape(n, length, h) @ p["h_out_w"] + p["h_out_b"]
    e = temporal.layer_norm(e, p["h_ln2_g"], p["h_ln2_b"])
    fg = np.exp(p["scale"]) * (np.einsum("nld,d->nl", x, q) - p["lam"] * off) + p["bias"] + e @ p["h_s_w"] + p["h_s_b"]
    reach = np.logaddexp(0.0, e @ p["h_d_w"] + p["h_d_b"]) * 5.0
    return fg.astype(np.float32), reach.astype(np.float32)


def numpy_head_video(p: dict, x: np.ndarray, t: np.ndarray, q: np.ndarray, off: np.ndarray):
    """numpy_head() over a whole video, window by window as apply(), the windows' outputs blended."""
    n = len(x)
    fg, reach, weight = np.zeros(n, np.float32), np.zeros((n, 2), np.float32), np.zeros(n, np.float32)
    for s, e, w in temporal.windows(n):
        f, r = numpy_head(p, x[s:e], np.asarray(t[s:e], dtype=np.float64), q, off[s:e])
        fg[s:e] += w * f
        reach[s:e] += w[:, None] * r
        weight[s:e] += w
    return fg / weight, reach / weight[:, None]


def numpy_moments(fg: np.ndarray, reach: np.ndarray, t: np.ndarray, duration: float, count: int = 3):
    """Up to `count` moments (start, end, frame index), best foreground first; a frame inside a moment
    already taken starts no other."""
    out = []
    mid = np.asarray(t, dtype=np.float64) + 0.5
    for i in np.argsort(-fg):
        if any(a <= mid[i] <= b for a, b, _ in out):
            continue
        a = max(0.0, mid[i] - float(reach[i, 0]))
        b = min(float(duration), mid[i] + float(reach[i, 1])) if duration else mid[i] + float(reach[i, 1])
        out.append((a, max(b, a + 0.5), int(i)))
        if len(out) == count:
            break
    return out


def search_with_head(
    p: dict,
    f,
    queries: list,
    text: np.ndarray,
    bank: np.ndarray,
    emb: np.ndarray | None = None,
    count: int = 3,
    per_query: bool = False,
    place: str = "reach",
) -> dict:
    """The whole search with the model, as the tool runs it: adapted frames, the query bank, the 64 best
    videos shortlisted, the head scoring their frames for the description (whenframe.temporal, numpy), up to
    `count` moments in each, all moments ranked by the head's logit. Moments as the head places them
    (place="reach"), or 10-second windows around its best frames, as the default search places them
    (place="windows"). Metrics as benchmark.run's."""
    from benchmark import CORPUS_KS, SHORTLIST, iou

    from whenframe.bank import offsets as numpy_offsets
    from whenframe.moments import windows

    if emb is None:
        emb = adapt(p, f.emb, f.times, f.starts, f.counts)
    off = numpy_offsets(emb, bank, float(p["beta"]))
    lam = float(p["lam"])
    hits = {(k, th): 0 for k in CORPUS_KS for th in (0.5, 0.7)}
    video_hits = dict.fromkeys(CORPUS_KS, 0)
    single, found = [], []
    # videos of one window, padded together, so that a description's shortlist goes through the head at once
    nv, w = len(f.starts), temporal.WINDOW
    short = f.counts <= w
    X, T = np.zeros((nv, w, emb.shape[1]), np.float32), np.zeros((nv, w))
    M, OFF = np.zeros((nv, w), bool), np.zeros((nv, w), np.float32)
    for v in np.flatnonzero(short):
        s, c = int(f.starts[v]), int(f.counts[v])
        X[v, :c], T[v, :c], M[v, :c], OFF[v, :c] = emb[s : s + c], f.times[s : s + c], True, off[s : s + c]
    U = numpy_frames_projection(p, X)

    def scored(vids: np.ndarray, q: np.ndarray) -> dict:
        out = {}
        one = vids[short[vids]]
        if len(one):
            fg, reach = numpy_head_batch(p, X[one], T[one], M[one], q, OFF[one], U[one])
            for j, v in enumerate(one):
                c = int(f.counts[v])
                out[int(v)] = fg[j, :c], reach[j, :c]
        for v in vids[~short[vids]]:
            s, c = int(f.starts[v]), int(f.counts[v])
            out[int(v)] = numpy_head_video(p, emb[s : s + c], f.times[s : s + c], q, off[s : s + c])
        return out

    def placed(v: int, fg: np.ndarray, reach: np.ndarray, n: int):
        t = f.times[int(f.starts[v]) : int(f.starts[v]) + int(f.counts[v])]
        if place == "windows":
            ws = windows(t.astype(np.float64), fg.astype(np.float64), float(f.durations[v]), 10.0, n)
            return [(x.start, x.end, x.peak) for x in ws]
        return [(lo, hi, float(fg[i])) for lo, hi, i in numpy_moments(fg, reach, t, float(f.durations[v]), n)]

    for qi, query in enumerate(queries):
        q = text[qi]
        best = np.maximum.reduceat(emb @ q - lam * off, f.starts)
        shortlist = np.argsort(-best)[:SHORTLIST]
        heads = scored(np.unique(np.concatenate([shortlist, [query["video"]]])), q)
        ranked = [(int(v), *m) for v in shortlist for m in placed(int(v), *heads[int(v)], count)]
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
        found.append(first_ok[0.5] is not None and first_ok[0.5] < 5)
        order = list(dict.fromkeys(v for v, *_ in ranked))
        rank_v = order.index(query["video"]) if query["video"] in order else None
        for k in CORPUS_KS:
            video_hits[k] += rank_v is not None and rank_v < k
        top = placed(query["video"], *heads[int(query["video"])], 1)
        single.append(iou(top[0][:2], query["moment"]) if top else 0.0)
    n, single = len(queries), np.array(single)
    out = {"corpus": {f"R@{k} IoU>={th}": hits[(k, th)] / n for k in CORPUS_KS for th in (0.5, 0.7)},
           "video": {f"R@{k}": video_hits[k] / n for k in CORPUS_KS},
           "single": {"R1 IoU>=0.5": float((single >= 0.5).mean()), "R1 IoU>=0.7": float((single >= 0.7).mean()),
                      "mIoU": float(single.mean())}}  # fmt: skip
    return out | ({"found": found} if per_query else {})


def adapt(p: dict, emb: np.ndarray, times: np.ndarray, starts, counts) -> np.ndarray:
    """Every video's frames through the adapter."""
    out = np.empty(emb.shape, np.float32)
    for s, c in zip(starts, counts, strict=True):
        out[s : s + c] = temporal.apply(p, emb[s : s + c], times[s : s + c])
    return out


def numpy_params(p: dict, layers: int, heads: int) -> dict:
    out = {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}
    out["layers"], out["heads"] = layers, heads
    return out


def calibrated_bias(p: dict, batch) -> float:
    """The head's bias before training: halfway between the logits of the frames inside the described moments
    and those of other videos' frames, rather than SigLIP 2's own bias, far below both."""
    from whenframe.bank import offsets

    x, _, mask, q, inside, _, _, bank = batch
    a, lam, beta = float(np.exp(p["scale"])), float(p["lam"]), float(p["beta"])
    off = offsets(x.reshape(-1, x.shape[-1]), bank, beta).reshape(mask.shape)
    own = a * (np.einsum("nld,nd->nl", x, q) - lam * off) + float(p["bias"])
    other = a * (np.einsum("nld,nd->nl", np.roll(x, 1, 0), q) - lam * np.roll(off, 1, 0)) + float(p["bias"])
    return -0.5 * (float(own[inside].mean()) + float(other[np.roll(mask, 1, 0)].mean()))


def train(args) -> None:
    import jax
    import jax.numpy as jnp

    cfg = {"between": args.between, "within": args.within, "leash": args.leash, "dropout": args.dropout,
           "head": args.head, "negatives": args.negatives}  # fmt: skip
    tuning = DiDeMo(args)
    data = Videos(args.seed, tuning.kept.bank, args.bank_sample)
    key = jax.random.PRNGKey(args.seed)
    params = jax.tree.map(jnp.asarray, init(key, args.width, args.layers, args.heads, lam=tuning.params["lam"],
                                            beta=tuning.params["beta"]))  # fmt: skip
    params["h_s_b"] = jnp.float32(calibrated_bias(params, data.batch(args.batch)))
    adam_m = jax.tree.map(jnp.zeros_like, params)
    adam_v = jax.tree.map(jnp.zeros_like, params)
    decayed = {k: k.endswith("_w") for k in params}
    rate = {k: 0.0 if k in FIXED else args.head_lr if k.startswith("h_") else args.lr for k in params}

    @jax.jit
    def step(params, adam_m, adam_v, batch, key, warm, i):
        (value, parts), grads = jax.value_and_grad(loss, has_aux=True)(params, batch, args.layers, args.heads, key,
                                                                         cfg)  # fmt: skip
        norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0 / (norm + 1e-9)), grads)
        adam_m = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, adam_m, grads)
        adam_v = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g**2, adam_v, grads)
        c1, c2 = 1 - 0.9 ** (i + 1), 1 - 0.999 ** (i + 1)

        def update(k, p, m, v):
            lr = rate[k] * warm
            p = p - lr * (m / c1) / (jnp.sqrt(v / c2) + 1e-8)
            return p * (1 - lr * args.decay) if decayed[k] else p

        params = {k: update(k, params[k], adam_m[k], adam_v[k]) for k in params}
        return params, adam_m, adam_v, value, parts

    base = tuning.score(None)
    where = (
        "annotations",
        "train_annotations",
        "model_dir",
        "test_annotations",
        "charades_annotations",
        "youcook2_annotations",
        "unav_annotations",
        "clap_dir",
        "out",
        "model",
    )  # local paths: not logged
    log = {"config": {k: v for k, v in vars(args).items() if k not in where}, "tuning_without": base, "checks": []}
    print(f"tuning videos without the model: R@5 {100 * base['R@5']:.1f}", flush=True)
    best, best_params = -1.0, None
    t0 = time.time()
    for i in range(args.steps):
        warm = min(1.0, (i + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * i / args.steps))
        key, sub = jax.random.split(key)
        batch = tuple(jnp.asarray(b) for b in data.batch(args.batch))
        params, adam_m, adam_v, value, parts = step(params, adam_m, adam_v, batch, sub, warm, i)
        if (i + 1) % args.every == 0 or i + 1 == args.steps:
            p = numpy_params(jax.device_get(params), args.layers, args.heads)
            s = tuning.score(p, with_head=args.head > 0)
            names = ("between", "within", "leash", "frames", "rank", "bounds")
            losses = {k: float(v) for k, v in zip(names, parts, strict=True)}
            log["checks"].append({"step": i + 1, "loss": float(value), **losses, **s, "seconds": time.time() - t0})
            shown = ", ".join(f"{k} {100 * v:.1f}" for k, v in s.items())
            print(f"step {i + 1}: loss {float(value):.3f} ({', '.join(f'{k} {v:.3f}' for k, v in losses.items())}); "
                  f"tuning {shown}; {time.time() - t0:.0f} s", flush=True)  # fmt: skip
            score = max(s[k] for k in ("R@5", "head_R@5", "rescored_R@5") if k in s)
            if score > best:
                best, best_params = score, p
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.name}.npz", **best_params)
    log["best"] = best
    (OUT / f"{args.name}.json").write_text(json.dumps(log, indent=1))
    print(f"best tuning R@5 {100 * best:.1f} (without: {100 * base['R@5']:.1f}); wrote {OUT / args.name}.npz")


def versus(a, b) -> dict:
    """Descriptions found (R@5, IoU >= 0.5) by a and not by b, and the reverse, with the sign test's p."""
    from explore import sign_test

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    win, loss = int((a & ~b).sum()), int((~a & b).sum())
    return {"wins": win, "losses": loss, "p": sign_test(win, loss)}


def evaluate(args) -> None:
    """Fifth round, run once (docs/PROTOCOL.md): the model against SigLIP 2 as it is, on DiDeMo's test videos,
    then on Charades-STA's, YouCook2's and UnAV-100's, each searched as in its own round."""
    from argparse import Namespace

    import charades
    import unav
    import youcook2
    from benchmark import frame_scores, run

    p = temporal.load(Path(args.model))
    res = {"model": str(Path(args.model).relative_to(ROOT)) if Path(args.model).is_relative_to(ROOT) else args.model}

    variants = ("without", "adapter", "head", "rescored")

    def three(f, queries, text, method, params):
        """SigLIP 2 as it is, the adapter, the adapter and the head (its moments, and its scores in the default
        search's windows); R@5 found or not for each description."""
        emb = adapt(p, f.emb, f.times, f.starts, f.counts)
        out = {"without": run(f, queries, text, method, params, per_query=True),
               "adapter": run(replace(f, emb=emb, _offsets={}), queries, text, method, params, per_query=True),
               "head": search_with_head(p, f, queries, text, f.bank, emb=emb, per_query=True),
               "rescored": search_with_head(p, f, queries, text, f.bank, emb=emb, per_query=True,
                                            place="windows")}  # fmt: skip
        found = {k: v.pop("found") for k, v in out.items()}
        for k in variants[1:]:
            out[f"{k}_vs_without"] = versus(found[k], found["without"])
            out[k].pop("params", None)
        out["without"].pop("params", None)
        return out

    def show(name, r):
        print(f"{name}: R@5 " + ", ".join(f"{k} {100 * r[k]['corpus']['R@5 IoU>=0.5']:.1f}" for k in variants),
              flush=True)  # fmt: skip

    test = DiDeMo(args, "test")
    res["didemo"] = {"videos": len(test.kept.starts), "queries": len(test.queries), "method": test.method,
                     **three(test.kept, test.queries, test.text, test.method, test.params)}  # fmt: skip
    show("DiDeMo test", res["didemo"])
    t0, video = time.perf_counter(), test.kept
    adapt(p, video.emb, video.times, video.starts, video.counts)
    res["didemo"]["adapter_ms_per_frame"] = 1000 * (time.perf_counter() - t0) / len(video.times)

    shared = {"model_dir": args.model_dir, "train_annotations": args.annotations}
    f, queries, text = charades.load(Namespace(annotations=args.charades_annotations, **shared))
    res["charades"] = {"videos": len(f.names), "queries": len(queries),
                       **three(f, queries, text, test.method, test.params)}  # fmt: skip
    show("Charades-STA", res["charades"])

    f, queries, text, db, params, _ = youcook2.load(Namespace(annotations=args.youcook2_annotations, **shared))
    which = [i for i, q in enumerate(queries) if q["name"] not in youcook2.split(f.names)]
    adapted = replace(f, emb=adapt(p, f.emb, f.times, f.starts, f.counts), _offsets={})
    best = json.loads((ROOT / "results" / "youcook2_tuned_spans.json").read_text())["best"]
    cfg = youcook2.speech.SpeechConfig(best["weight"], best["pad"], best["normalise"])
    yc = {"test_steps": len(which)}
    for label, g in (("without", f), ("adapter", adapted)):
        for name, method, spoken in (("windows", "frame_window_bank", False), ("speech_spans", "whenframe_bank", True)):
            visual = frame_scores(g, text, method, params[method])
            scores = youcook2.with_speech(g, queries, visual, db, cfg, which) if spoken else visual
            yc[f"{label}_{name}"] = youcook2.measure(g, queries, scores, method, params[method], which)
    for name in ("windows", "speech_spans"):
        yc[f"adapter_vs_without_{name}"] = versus(
            yc[f"adapter_{name}"].pop("found"), yc[f"without_{name}"].pop("found")
        )
    res["youcook2"] = yc
    print("YouCook2: R@5 " + ", ".join(f"{k} {100 * yc[k]['corpus']['R@5 IoU>=0.5']:.1f}" for k in yc
                                       if isinstance(yc[k], dict) and "corpus" in yc[k]), flush=True)  # fmt: skip

    z, labels, events, siglip, clap, bank = unav.prepare(
        Namespace(annotations=args.unav_annotations, model_dir=args.model_dir, clap_dir=args.clap_dir)
    )
    names = z["names"].tolist()
    lib = unav.Library(z, set(names) - unav.tuning_videos(names))
    tuned = json.loads((ROOT / "results" / "unav_tuned.json").read_text())["best"]
    heard = unav.standardise(lib.sound(clap, tuned["length"]))
    ua = {"videos": len(lib.names)}
    for label in ("without", "adapter"):
        if label == "adapter":
            starts = np.array([r for r, _ in lib.ranges])
            counts = np.array([c for _, c in lib.ranges])
            lib.emb = adapt(p, lib.emb, lib.times, starts, counts)
        frames = unav.standardise(lib.frames(siglip, bank))
        ua[f"{label}_frames"] = unav.evaluate_scores(lib, events, labels, frames)
        ua[f"{label}_both"] = unav.evaluate_scores(lib, events, labels, frames + tuned["weight"] * heard)
    for name in ("frames", "both"):
        ua[f"adapter_vs_without_{name}"] = unav.compare(ua[f"adapter_{name}"], ua[f"without_{name}"])
    print("UnAV-100: P@10 " + ", ".join(f"{k} {100 * ua[k]['p_at_k']:.1f}" for k in ua
                                         if isinstance(ua[k], dict) and "p_at_k" in ua[k]), flush=True)  # fmt: skip
    for k in list(ua):
        if isinstance(ua[k], dict) and "labels" in ua[k]:
            ua[k] = {"p_at_k": ua[k]["p_at_k"], "map": ua[k]["map"]}
    res["unav"] = ua
    out = Path(args.out or ROOT / "results" / "temporal_test.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["texts", "train", "evaluate"])
    ap.add_argument("--annotations", required=True, help="DiDeMo's didemo_train.json")
    ap.add_argument("--train-annotations", help="the same, for the query bank (default: --annotations)")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--name", default="temporal")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4, help="the adapter's learning rate (0: SigLIP 2 as it is)")
    ap.add_argument("--head-lr", type=float, default=3e-4, help="the head's")
    ap.add_argument("--bank-sample", type=int, default=1024, help="bank descriptions drawn for each batch")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--between", type=float, default=1.0, help="weight of the loss between videos")
    ap.add_argument("--within", type=float, default=1.0, help="weight of the loss within a video")
    ap.add_argument("--leash", type=float, default=1.0, help="weight of the leash to SigLIP 2's embeddings")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head", type=float, default=1.0, help="weight of the head's losses")
    ap.add_argument("--negatives", type=int, default=4, help="other videos per description, for the head")
    ap.add_argument("--every", type=int, default=500, help="steps between two checks on the tuning videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", help="evaluate: the trained weights (results/temporal/NAME.npz)")
    ap.add_argument("--test-annotations", help="evaluate: DiDeMo's didemo_test.json")
    ap.add_argument("--charades-annotations", help="evaluate: Charades-STA's test parquet")
    ap.add_argument("--youcook2-annotations", help="evaluate: YouCook2's validation parquet")
    ap.add_argument("--unav-annotations", help="evaluate: UnAV-100's unav100_all.json")
    ap.add_argument("--clap-dir", help="evaluate: folder of CLAP's files")
    ap.add_argument("--out", help="evaluate: where to write the results (default: results/temporal_test.json)")
    args = ap.parse_args()
    args.train_annotations = args.train_annotations or args.annotations
    args.split = "train"
    {"texts": texts, "train": train, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
