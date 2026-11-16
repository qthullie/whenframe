"""SigLIP 2 image and text towers, run with ONNX Runtime.

SigLIP scores an image-text pair with a sigmoid, sigmoid(scale * cos + bias),
trained on each pair independently. The score is therefore comparable across
videos, which matters when one query is ranked against a whole library.

The ONNX exports come from onnx-community; the scale and bias are those of the
original google/siglip2 checkpoints, which the exports leave out.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    image_size: int
    logit_scale: float
    logit_bias: float
    text_length: int = 64


MODELS = {
    "siglip2-base": ModelSpec("onnx-community/siglip2-base-patch16-224-ONNX", 224, 112.6689, -16.7717),
}
DEFAULT_MODEL = "siglip2-base"


def cache_dir() -> Path:
    root = os.environ.get("WHENFRAME_CACHE") or Path.home() / ".cache" / "whenframe"
    return Path(root)


def fetch(spec: ModelSpec, quantized: bool = False) -> Path:
    """Download the model files once into the cache and return their folder."""
    folder = cache_dir() / "models" / spec.repo.replace("/", "--")
    vision = "onnx/vision_model_int8.onnx" if quantized else "onnx/vision_model.onnx"
    for name in ("tokenizer.json", vision, "onnx/text_model.onnx"):
        dest = folder / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{spec.repo}/resolve/main/{name}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed host)
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return folder


def _session(path: Path, threads: int | None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # idle threads sleep instead of spinning: as fast here, and less CPU burnt on a shared machine
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, model: str = DEFAULT_MODEL, quantized: bool = False, folder: str | Path | None = None,
                 threads: int | None = None):  # fmt: skip
        self.name = model
        self.spec = MODELS[model]
        self.quantized = quantized
        folder = Path(folder) if folder else fetch(self.spec, quantized)
        vision = "vision_model_int8.onnx" if quantized else "vision_model.onnx"
        self._vision = _session(folder / "onnx" / vision, threads)
        self._text = None
        self._text_path = folder / "onnx" / "text_model.onnx"
        self._threads = threads
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        # SigLIP was trained on lower-cased text padded to a fixed length, ending with <eos>
        self._tokenizer.enable_truncation(self.spec.text_length)
        self._tokenizer.enable_padding(length=self.spec.text_length, pad_id=self._tokenizer.token_to_id("<pad>"))

    @property
    def size(self) -> int:
        return self.spec.image_size

    def encode_images(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        """uint8 (n, size, size, 3) RGB -> unit vectors (n, d)."""
        out = []
        for i in range(0, len(frames), batch):
            x = frames[i : i + batch].astype(np.float32) / 127.5 - 1.0  # rescale 1/255, then mean 0.5, std 0.5
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
            out.append(self._vision.run(["pooler_output"], {"pixel_values": x})[0])
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.concatenate(out))

    def tokenize(self, texts: list[str]) -> np.ndarray:
        return np.array([e.ids for e in self._tokenizer.encode_batch([t.lower() for t in texts])], dtype=np.int64)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self._text is None:  # the text tower is large and only needed at query time
            self._text = _session(self._text_path, self._threads)
        out = self._text.run(["pooler_output"], {"input_ids": self.tokenize(texts)})[0]
        return _normalize(out)

    @property
    def dim(self) -> int:
        return self._vision.get_outputs()[1].shape[-1]

    def probability(self, cos: np.ndarray) -> np.ndarray:
        """SigLIP's own estimate that a frame matches the text."""
        z = self.spec.logit_scale * cos + self.spec.logit_bias
        return 1.0 / (1.0 + np.exp(-z))
