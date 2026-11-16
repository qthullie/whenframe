"""Image and text towers run with ONNX Runtime: SigLIP 2, and MobileCLIP-S2 for the cheaper modes.

SigLIP scores an image-text pair with a sigmoid, sigmoid(scale * cos + bias),
trained on each pair independently, so its score is an absolute "does this
match" that can be compared across videos. The ONNX export leaves the scale and
bias out; they are those of the original google/siglip2 checkpoint.

MobileCLIP-S2 was trained with a softmax and has no such calibration. Its scale
and bias map its cosine onto SigLIP's logit, fitted on DiDeMo training pairs
(`scripts/didemo.py calibrate`), so every mode shows confidences on one scale.
It takes about 2.3 times less CPU per frame (docs/RESULTS.md).
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ONNX Runtime 1.29 sends usage telemetry to Microsoft unless told not to; nothing should leave the machine
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    image_size: int
    logit_scale: float  # probability that a frame matches = sigmoid(logit_scale * cos + logit_bias)
    logit_bias: float
    vision_file: str = "onnx/vision_model.onnx"
    text_file: str = "onnx/text_model.onnx"
    pixels: str = "symmetric"  # "symmetric": [-1, 1]; "unit": [0, 1]
    image_output: str = "pooler_output"
    text_output: str = "pooler_output"
    text_length: int = 64
    pad_token: str | None = "<pad>"  # None pads with id 0
    lowercase: bool = True  # SigLIP was trained on lower-cased text; CLIP tokenizers lower-case by themselves
    bank_beta: float = 50.0  # query-bank correction, as tuned on DiDeMo training videos
    bank_lambda: float = 1.0


MODELS = {
    "siglip2-base": ModelSpec("onnx-community/siglip2-base-patch16-224-ONNX", 224, 112.6689, -16.7717),
    "mobileclip-s2": ModelSpec(
        "Xenova/mobileclip_s2", 256, 66.86, -19.567, pixels="unit", image_output="image_embeds",
        text_output="text_embeds", text_length=77, pad_token=None, lowercase=False,
    ),
}  # fmt: skip
DEFAULT_MODEL = "siglip2-base"
# what `whenframe index --mode` stores; several models are searched together and their scores combined
MODES = {"standard": "siglip2-base", "economy": "mobileclip-s2", "precision": "siglip2-base+mobileclip-s2"}
# weight of each model's scores in a combined search, as tuned on DiDeMo training videos
FUSION = {"siglip2-base": 1.0, "mobileclip-s2": 1.0}


def cache_dir() -> Path:
    root = os.environ.get("WHENFRAME_CACHE") or Path.home() / ".cache" / "whenframe"
    return Path(root)


def fetch(spec: ModelSpec) -> Path:
    """Download the model files once into the cache and return their folder."""
    folder = cache_dir() / "models" / spec.repo.replace("/", "--")
    for name in ("tokenizer.json", spec.vision_file, spec.text_file):
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
    def __init__(self, model: str = DEFAULT_MODEL, folder: str | Path | None = None, threads: int | None = None):
        self.name = model
        self.spec = MODELS[model]
        folder = Path(folder) if folder else fetch(self.spec)
        self._vision = _session(folder / self.spec.vision_file, threads)
        self._text = None
        self._text_path = folder / self.spec.text_file
        self._threads = threads
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        # fixed-length input ending with the end-of-text token, as both models were trained
        n = self.spec.text_length
        pad = self.spec.pad_token
        self._tokenizer.enable_truncation(n)
        self._tokenizer.enable_padding(length=n, pad_id=self._tokenizer.token_to_id(pad) if pad else 0)

    @property
    def size(self) -> int:
        return self.spec.image_size

    def encode_images(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        """uint8 (n, size, size, 3) RGB -> unit vectors (n, d)."""
        out = []
        for i in range(0, len(frames), batch):
            x = frames[i : i + batch].astype(np.float32)
            x = x / 255.0 if self.spec.pixels == "unit" else x / 127.5 - 1.0
            x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
            out.append(self._vision.run([self.spec.image_output], {"pixel_values": x})[0])
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.concatenate(out))

    def tokenize(self, texts: list[str]) -> np.ndarray:
        texts = [t.lower() for t in texts] if self.spec.lowercase else list(texts)
        return np.array([e.ids for e in self._tokenizer.encode_batch(texts)], dtype=np.int64)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self._text is None:  # the text tower is only needed at query time
            self._text = _session(self._text_path, self._threads)
        out = self._text.run([self.spec.text_output], {"input_ids": self.tokenize(texts)})[0]
        return _normalize(out)

    @property
    def dim(self) -> int:
        return next(o for o in self._vision.get_outputs() if o.name == self.spec.image_output).shape[-1]

    def probability(self, cos: np.ndarray) -> np.ndarray:
        """The model's estimate that a frame matches the text."""
        z = self.spec.logit_scale * cos + self.spec.logit_bias
        return 1.0 / (1.0 + np.exp(-z))
