"""Image and text towers run with ONNX Runtime: SigLIP 2 at two sizes, and MobileCLIP-S2 for research use.

SigLIP scores an image-text pair with a sigmoid, sigmoid(scale * cos + bias),
trained on each pair independently, so its score is an absolute "does this
match" that can be compared across videos. The ONNX export leaves the scale and
bias out; they are those of the original google/siglip2 checkpoint.

SigLIP 2 B/32 (256 px, 64 patches instead of 196) is the cheaper model of the
economy and precision modes; its scale and bias are its own checkpoint's.

MobileCLIP-S2 is measured in the benchmark but is not used by any mode: Apple
licenses its weights for non-commercial research only. It was trained with a
softmax; its scale and bias map its cosine onto SigLIP's logit, fitted on DiDeMo
training pairs (`scripts/didemo.py calibrate`).

The image tower runs with OpenVINO when it is installed (`pip install
whenframe[fast]`), which computes in bfloat16 on processors that do so natively
(AMX, AVX512-BF16) and in float32 elsewhere; otherwise with ONNX Runtime. The
text tower always runs with ONNX Runtime in float32.
"""

from __future__ import annotations

import importlib.util
import os
import sys
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
    spans_lambda: float | None = None  # its strength when moments are adaptive spans; None: as bank_lambda


MODELS = {
    "siglip2-base": ModelSpec("onnx-community/siglip2-base-patch16-224-ONNX", 224, 112.6689, -16.7717,
                              spans_lambda=0.5),
    "siglip2-b32": ModelSpec("onnx-community/siglip2-base-patch32-256-ONNX", 256, 111.8326, -16.7669,
                             bank_lambda=0.5),
    # research use only (Apple Machine Learning Research Model licence)
    "mobileclip-s2": ModelSpec(
        "Xenova/mobileclip_s2", 256, 66.86, -19.567, pixels="unit", image_output="image_embeds",
        text_output="text_embeds", text_length=77, pad_token=None, lowercase=False, bank_beta=10.0,
    ),
}  # fmt: skip
DEFAULT_MODEL = "siglip2-base"
# what `whenframe index --mode` stores; several models are searched together and their scores combined
MODES = {"standard": "siglip2-base", "economy": "siglip2-b32", "precision": "siglip2-base+siglip2-b32"}
# weight of each model's scores in a combined search, as tuned on DiDeMo training videos
FUSION = {"siglip2-base": 1.0, "siglip2-b32": 0.25, "mobileclip-s2": 1.0}


def cache_dir() -> Path:
    root = os.environ.get("WHENFRAME_CACHE") or Path.home() / ".cache" / "whenframe"
    return Path(root)


def fetch(spec: ModelSpec) -> Path:
    """Download the model files once into the cache and return their folder."""
    return fetch_files(spec.repo, ("tokenizer.json", spec.vision_file, spec.text_file))


def fetch_files(repo: str, names: tuple[str, ...]) -> Path:
    """Download files of a Hugging Face repository once into the cache and return their folder."""
    folder = cache_dir() / "models" / repo.replace("/", "--")
    for name in names:
        dest = folder / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
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


BACKENDS = ("auto", "onnxruntime", "openvino")


def backend_name(name: str | None = None) -> str:
    """The runtime of the image tower: the one asked for, else $WHENFRAME_BACKEND, else OpenVINO if installed."""
    name = name or os.environ.get("WHENFRAME_BACKEND") or "auto"
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; one of {', '.join(BACKENDS)}")
    if name != "auto":
        return name
    return "openvino" if importlib.util.find_spec("openvino") else "onnxruntime"


def _import_openvino():
    """OpenVINO, without the usage report its model converter sends to Google Analytics on import.

    `import openvino` loads openvino.tools.ovc, which sends an "import" event through
    the openvino_telemetry package; when that package cannot be imported, OpenVINO
    uses a silent stub of its own instead.
    """
    sys.modules.setdefault("openvino_telemetry", None)  # a None entry makes the import fail
    import openvino

    return openvino


class _OnnxRuntimeImages:
    precision = "f32"

    def __init__(self, path: Path, output: str, threads: int | None):
        self._session = _session(path, threads)
        self._output = output
        self.dim = next(o for o in self._session.get_outputs() if o.name == output).shape[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._session.run([self._output], {"pixel_values": x})[0]


class _OpenVINOImages:
    """The image tower compiled by OpenVINO, in the precision OpenVINO picks for this processor."""

    def __init__(self, path: Path, output: str, threads: int | None):
        core = _import_openvino().Core()
        config = {"INFERENCE_NUM_THREADS": threads} if threads else {}
        self._compiled = core.compile_model(core.read_model(str(path)), "CPU", config)
        self._output = self._compiled.output(output)
        self.dim = self._output.get_partial_shape()[-1].get_length()
        self.precision = self._compiled.get_property("INFERENCE_PRECISION_HINT").get_type_name()  # bf16, f32

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._compiled(x)[self._output]


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, model: str = DEFAULT_MODEL, folder: str | Path | None = None, threads: int | None = None,
                 backend: str | None = None):  # fmt: skip
        self.name = model
        self.spec = MODELS[model]
        self.backend = backend_name(backend)
        folder = Path(folder) if folder else fetch(self.spec)
        self._vision_path = folder / self.spec.vision_file
        self._vision = None  # loaded when the first frames come: searching only needs the text tower
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

    @property
    def images(self) -> _OnnxRuntimeImages | _OpenVINOImages:
        """The image tower, loaded on first use."""
        if self._vision is None:
            runtime = _OpenVINOImages if self.backend == "openvino" else _OnnxRuntimeImages
            self._vision = runtime(self._vision_path, self.spec.image_output, self._threads)
        return self._vision

    def encode_images(self, frames: np.ndarray, batch: int = 32) -> np.ndarray:
        """uint8 (n, size, size, 3) RGB -> unit vectors (n, d)."""
        out = []
        for i in range(0, len(frames), batch):
            x = frames[i : i + batch].astype(np.float32)
            x = x / 255.0 if self.spec.pixels == "unit" else x / 127.5 - 1.0
            out.append(self.images(np.ascontiguousarray(x.transpose(0, 3, 1, 2))))
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
        return self.images.dim

    def probability(self, cos: np.ndarray) -> np.ndarray:
        """The model's estimate that a frame matches the text."""
        z = self.spec.logit_scale * cos + self.spec.logit_bias
        return 1.0 / (1.0 + np.exp(-z))
