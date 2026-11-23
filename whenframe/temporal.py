"""whenframe's temporal model: each frame's embedding, rewritten with what the frames around it show.

SigLIP 2 embeds each frame on its own, so "jumps into the pool" or "starts to
dance" must be read from a single image. This model reads the embeddings of a
video's frames together, a window of WINDOW frames at a time, with their times
in seconds, and adds to each frame's embedding what its neighbours say; the sum
is normalised again and searched as before, in SigLIP 2's own space, against the
same text embeddings and query bank. It is small (1.4 million weights, about
half a millisecond of CPU per frame against some 80 for SigLIP 2's image tower)
and was trained on DiDeMo's training videos (scripts/temporal.py,
docs/PROTOCOL.md and docs/RESULTS.md, fifth round).

Two transformer layers over the frames, as CLIP4Clip's temporal transformer
(Luo et al., 2021), with its output projection initialised at zero so that the
untrained model is SigLIP 2 itself (as the zero-initialised residual branches
of CLIP-Adapter and ControlNet). This file runs it with numpy, for indexing; the
training runs the same computation with JAX.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

WINDOW, STRIDE = 32, 16  # frames read together, and how far the next window starts
# the image encoders a temporal model was trained for: v3 (seventh round) and v4 (ninth round)
MODELS = {"siglip2-base": "temporal-siglip2-base", "pe-core-b16": "temporal-pe-core-b16"}
# the libraries whose frames were measured through them: v3 alone, v4 alone, and both together (eleventh round)
MEASURED = ("siglip2-base", "pe-core-b16", "siglip2-base+pe-core-b16")
PERIODS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)  # seconds: the times' sines and cosines at these periods


def time_features(t: np.ndarray) -> np.ndarray:
    """(n, 12) sines and cosines of the frames' times, measured from the window's first frame."""
    rel = (t - t[0])[:, None] * (2 * np.pi / np.array(PERIODS))[None, :]
    return np.concatenate([np.sin(rel), np.cos(rel)], axis=1).astype(np.float32)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def forward(p: dict, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One window: (n, 768) unit embeddings and their times -> (n, 768) what the context adds."""
    h = layer_norm(x, p["in_ln_g"], p["in_ln_b"]) @ p["in_w"] + p["in_b"] + time_features(t) @ p["time_w"]
    heads = p["heads"]
    for i in range(p["layers"]):
        q = layer_norm(h, p[f"l{i}_ln1_g"], p[f"l{i}_ln1_b"])
        qkv = q @ p[f"l{i}_qkv_w"] + p[f"l{i}_qkv_b"]
        n, d = h.shape
        qs, ks, vs = (qkv[:, j * d : (j + 1) * d].reshape(n, heads, d // heads).transpose(1, 0, 2) for j in range(3))
        att = qs @ ks.transpose(0, 2, 1) / np.sqrt(d // heads)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        h = h + (att @ vs).transpose(1, 0, 2).reshape(n, d) @ p[f"l{i}_out_w"] + p[f"l{i}_out_b"]
        m = layer_norm(h, p[f"l{i}_ln2_g"], p[f"l{i}_ln2_b"])
        h = h + gelu(m @ p[f"l{i}_fc1_w"] + p[f"l{i}_fc1_b"]) @ p[f"l{i}_fc2_w"] + p[f"l{i}_fc2_b"]
    return layer_norm(h, p["out_ln_g"], p["out_ln_b"]) @ p["out_w"] + p["out_b"]


def windows(n: int):
    """(start, end, weights) of the windows that cover n frames: WINDOW frames every STRIDE, the last one
    ending with the video; weights rise towards a window's middle, to blend where windows overlap."""
    starts = list(range(0, max(n - WINDOW, 0) + 1, STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    for s in starts:
        e = min(s + WINDOW, n)
        yield s, e, 1.0 + np.minimum(np.arange(e - s), np.arange(e - s)[::-1]).astype(np.float32)


def apply(p: dict, emb: np.ndarray, times: np.ndarray) -> np.ndarray:
    """A whole video's frames, window by window, the windows' additions blended; returns unit vectors."""
    x = np.asarray(emb, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x
    total, weight = np.zeros_like(x), np.zeros((n, 1), dtype=np.float32)
    for s, e, w in windows(n):
        total[s:e] += w[:, None] * forward(p, x[s:e], np.asarray(times[s:e], dtype=np.float64))
        weight[s:e] += w[:, None]
    out = x + total / weight
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def for_library(model: str) -> str:
    """The temporal models a new library of these image models ("+"-joined) goes through, one per model in their
    order, joined by "+": as measured, and only when all are installed; "" otherwise."""
    if model not in MEASURED:
        return ""
    names = [MODELS[m] for m in model.split("+")]
    return "+".join(names) if all(load(name=f"{n}.npz") is not None for n in names) else ""


def load(path=None, name: str = "temporal-siglip2-base.npz") -> dict | None:
    """Trained weights: those shipped with the package (None when there are none), or a file's. When the file has
    them, p["search"] holds the speech settings and the sound weight chosen with the model on the tuning sets, which
    a search of its frames uses (search.chosen)."""
    if path is None:
        path = resources.files("whenframe.data").joinpath(name)
        if not path.is_file():
            return None
    with path.open("rb") as f:
        z = np.load(f)
        p = {k: z[k].astype(np.float32) for k in z.files if k not in ("layers", "heads", "search")}
        p["layers"], p["heads"] = int(z["layers"]), int(z["heads"])
        if "search" in z.files:
            p["search"] = json.loads(str(z["search"]))
    return p
