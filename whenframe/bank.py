"""Correcting for frames that match almost any description.

A few frames (a dim room, a crowd, an empty sky) score well against most
texts and crowd the top of every search. Query-bank normalisation (Bogolin et
al., CVPR 2022) measures how well each frame matches a bank of unrelated
descriptions and takes it off the frame's scores:

    score(query, frame) = cos(query, frame) - lam * offset(frame)
    offset(frame)       = (1 / beta) * log mean_b exp(beta * cos(b, frame))

The offset depends on the frame only, so it is computed once per frame. The
banks shipped with the package hold the text embeddings of 4,096 descriptions
from the DiDeMo training split, one per model (see data/NOTICE).
"""

from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np

BANKS = {"siglip2-base": "bank-siglip2-base.npy", "siglip2-b32": "bank-siglip2-b32.npy"}


def load(model: str) -> np.ndarray | None:
    """The bank for a model's text space, or None when there is none."""
    name = BANKS.get(model)
    if name is None:
        return None
    with resources.files("whenframe.data").joinpath(name).open("rb") as f:
        return np.load(f).astype(np.float32)


def digest(bank: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(bank, dtype=np.float32).tobytes()).hexdigest()[:16]


def offsets(emb: np.ndarray, bank: np.ndarray, beta: float, chunk: int = 4096) -> np.ndarray:
    """offset(frame) for each row of `emb`, in cosine units."""
    out = np.empty(len(emb), dtype=np.float32)
    for i in range(0, len(emb), chunk):
        z = beta * (np.asarray(emb[i : i + chunk], dtype=np.float32) @ bank.T)
        top = z.max(axis=1)
        out[i : i + chunk] = (top + np.log(np.exp(z - top[:, None]).mean(axis=1))) / beta
    return out
