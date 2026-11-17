"""float16 embeddings to float32, a block at a time.

The frames' embeddings are stored as float16. numpy converts float16 one value at a
time, about ten times slower than ONNX Runtime's Cast, which a one-node model shipped
with the package (data/cast16.onnx, written by scripts/cast16.py) runs instead.
"""

from __future__ import annotations

import os
from importlib import resources

import numpy as np

_session = None

BLOCK = 1 << 16  # rows converted at a time: 200 MB of float32 for 768 dimensions


def to_float32(x: np.ndarray) -> np.ndarray:
    """A float32 copy of a 2-D array; float16 goes through ONNX Runtime's Cast."""
    if x.dtype != np.float16 or x.ndim != 2 or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    global _session
    if _session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        model = resources.files("whenframe.data").joinpath("cast16.onnx").read_bytes()
        _session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
    return _session.run(None, {"half": np.ascontiguousarray(x)})[0]


def float32_copy(x: np.ndarray) -> np.ndarray:
    """The whole array as float32, converted a block at a time into one array (no second copy)."""
    out = np.empty(x.shape, dtype=np.float32)
    for i in range(0, len(x), BLOCK):
        out[i : i + BLOCK] = to_float32(x[i : i + BLOCK])
    return out


def memory_budget() -> float:
    """Bytes of float32 embeddings a search may keep in memory: $WHENFRAME_MEMORY gigabytes, else a quarter
    of the machine's memory."""
    if os.environ.get("WHENFRAME_MEMORY"):
        return float(os.environ["WHENFRAME_MEMORY"]) * 1e9
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 4
    except (ValueError, OSError, AttributeError):  # not a POSIX system
        return 2e9
