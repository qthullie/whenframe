"""Image and text towers run with ONNX Runtime: SigLIP 2 at three sizes, and MobileCLIP-S2 for research use.

SigLIP scores an image-text pair with a sigmoid, sigmoid(scale * cos + bias),
trained on each pair independently, so its score is an absolute "does this
match" that can be compared across videos. The ONNX export leaves the scale and
bias out; they are those of the original google/siglip2 checkpoint.

SigLIP 2 B/32 (256 px, 64 patches instead of 196) is the cheaper model of the
economy and precision modes; its scale and bias are its own checkpoint's.

SigLIP 2 So400m (400 million parameters in the image tower, 5 times the base
model's) comes from Immich's ONNX export of the open_clip weights, which takes
one image or one text at a time; its descriptions are lower-cased as the base
model's are, which did as well as open_clip's own cleaning on DiDeMo tuning
videos, and its scale and bias are those of google/siglip2-so400m-patch14-224.

MobileCLIP-S2 is measured in the benchmark but is not used by any mode: Apple
licenses its weights for non-commercial research only. It was trained with a
softmax; its scale and bias map its cosine onto SigLIP's logit, fitted on DiDeMo
training pairs (`scripts/didemo.py calibrate`).

The image tower runs with OpenVINO when it is installed (`pip install
whenframe[fast]`), which computes in bfloat16 on processors that do so natively
(AMX, AVX512-BF16) and in float32 elsewhere; otherwise with ONNX Runtime. The
text tower runs with ONNX Runtime in float32, but So400m's, which OpenVINO runs
ten times faster on this machine (74 ms a text instead of 721) when it runs the
image tower.
"""

from __future__ import annotations

import html
import importlib.util
import json
import os
import string
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
    clean: str = "lower"  # "lower": as SigLIP was trained; "none": CLIP tokenizers clean by themselves; "canonical"
    tokenizer_file: str = "tokenizer.json"
    image_input: str = "pixel_values"
    text_input: str = "input_ids"
    text_dtype: str = "int64"
    weights_folder: str | None = None  # a folder of weights stored next to the ONNX graphs, fetched whole
    text_openvino: bool = False  # the text tower runs with OpenVINO too, when the image tower does
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
        text_output="text_embeds", text_length=77, pad_token=None, clean="none", bank_beta=10.0,
    ),
    "siglip2-so400m": ModelSpec(
        "immich-app/ViT-SO400M-14-SigLIP2__webli", 224, 109.8812, -15.9324, vision_file="visual/model.onnx",
        text_file="textual/model.onnx", tokenizer_file="textual/tokenizer.json", image_input="image",
        text_input="text", text_dtype="int32", image_output="embedding", text_output="embedding",
        clean="lower", weights_folder="textual", text_openvino=True,
    ),
}  # fmt: skip
DEFAULT_MODEL = "siglip2-base"
# what `whenframe index --mode` stores; several models are searched together and their scores combined
MODES = {
    "standard": "siglip2-base",
    "economy": "siglip2-b32",
    "precision": "siglip2-base+siglip2-b32",
    "large": "siglip2-so400m",
}
# weight of each model's scores in a combined search, as tuned on DiDeMo training videos
FUSION = {"siglip2-base": 1.0, "siglip2-b32": 0.25, "mobileclip-s2": 1.0}


def cache_dir() -> Path:
    root = os.environ.get("WHENFRAME_CACHE") or Path.home() / ".cache" / "whenframe"
    return Path(root)


def fetch(spec: ModelSpec) -> Path:
    """Download the model files once into the cache and return their folder."""
    folder = fetch_files(spec.repo, (spec.tokenizer_file, spec.vision_file, spec.text_file))
    if spec.weights_folder:
        fetch_folder(spec.repo, spec.weights_folder)
    return folder


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


def fetch_folder(repo: str, sub: str) -> Path:
    """Download every file of a folder of a Hugging Face repository once, as listed by the Hub."""
    folder = cache_dir() / "models" / repo.replace("/", "--")
    done = folder / sub / ".complete"
    if not done.exists():
        url = f"https://huggingface.co/api/models/{repo}/tree/main/{sub}"
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed host)
            listing = json.load(r)
        fetch_files(repo, tuple(e["path"] for e in listing if e.get("type") == "file"))
        done.touch()
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


class _OnnxRuntime:
    """A tower run by ONNX Runtime, on any number of inputs at once or, if its graph says so, one at a time."""

    precision = "f32"

    def __init__(self, path: Path, input: str, output: str, threads: int | None):
        self._session = _session(path, threads)
        self._input, self._output = input, output
        self.one_at_a_time = next(i for i in self._session.get_inputs() if i.name == input).shape[0] == 1
        self.dim = next(o for o in self._session.get_outputs() if o.name == output).shape[-1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.one_at_a_time:
            return np.concatenate([self._session.run([self._output], {self._input: x[i : i + 1]})[0]
                                   for i in range(len(x))])  # fmt: skip
        return self._session.run([self._output], {self._input: x})[0]


class _OpenVINO:
    """A tower compiled by OpenVINO, in the precision OpenVINO picks for this processor.

    A graph that takes one input at a time runs several at once, each in a request
    of its own (OpenVINO's throughput mode): 321 ms of CPU time per frame instead
    of 426 for SigLIP 2 So400m on 4 cores.
    """

    def __init__(self, path: Path, input: str, output: str, threads: int | None):
        ov = _import_openvino()
        core = ov.Core()
        model = core.read_model(str(path))
        batch = model.input(input).get_partial_shape()[0]
        self.one_at_a_time = batch.is_static and batch.get_length() == 1
        config = {"INFERENCE_NUM_THREADS": threads} if threads else {}
        if self.one_at_a_time:
            config["PERFORMANCE_HINT"] = "THROUGHPUT"
        self._compiled = core.compile_model(model, "CPU", config)
        self._output = self._compiled.output(output)
        self.dim = self._output.get_partial_shape()[-1].get_length()
        self.precision = self._compiled.get_property("INFERENCE_PRECISION_HINT").get_type_name()  # bf16, f32
        if self.one_at_a_time:
            requests = self._compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            self._queue = ov.AsyncInferQueue(self._compiled, requests)
            self._queue.set_callback(self._done)
            self._out: dict[int, np.ndarray] = {}

    def _done(self, request, i: int) -> None:
        self._out[i] = request.get_tensor(self._output).data.copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.one_at_a_time:
            return self._compiled(x)[self._output]
        self._out.clear()
        for i in range(len(x)):
            self._queue.start_async({0: x[i : i + 1]}, i)
        self._queue.wait_all()
        return np.concatenate([self._out[i] for i in range(len(x))])


_PUNCTUATION = str.maketrans("", "", string.punctuation)


def clean(text: str, how: str) -> str:
    """A description as the text tower was trained to read it.

    "canonical" is open_clip's "canonicalize" for SigLIP 2 (after big_vision):
    underscores as spaces, ASCII punctuation removed, lower case, single spaces.
    """
    if how == "lower":
        return text.lower()
    if how == "canonical":
        text = html.unescape(html.unescape(text)).replace("_", " ").translate(_PUNCTUATION).lower()
        return " ".join(text.split())
    return text


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
        self._tokenizer = Tokenizer.from_file(str(folder / self.spec.tokenizer_file))
        # fixed-length input ending with the end-of-text token, as both models were trained
        n = self.spec.text_length
        pad = self.spec.pad_token
        self._tokenizer.enable_truncation(n)
        self._tokenizer.enable_padding(length=n, pad_id=self._tokenizer.token_to_id(pad) if pad else 0)

    @property
    def size(self) -> int:
        return self.spec.image_size

    @property
    def images(self) -> _OnnxRuntime | _OpenVINO:
        """The image tower, loaded on first use."""
        if self._vision is None:
            runtime = _OpenVINO if self.backend == "openvino" else _OnnxRuntime
            self._vision = runtime(self._vision_path, self.spec.image_input, self.spec.image_output, self._threads)
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
        texts = [clean(t, self.spec.clean) for t in texts]
        return np.array([e.ids for e in self._tokenizer.encode_batch(texts)], dtype=self.spec.text_dtype)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self._text is None:  # the text tower is only needed at query time
            openvino = self.backend == "openvino" and self.spec.text_openvino
            runtime = _OpenVINO if openvino else _OnnxRuntime
            self._text = runtime(self._text_path, self.spec.text_input, self.spec.text_output, self._threads)
        return _normalize(self._text(self.tokenize(texts)))

    @property
    def dim(self) -> int:
        return self.images.dim

    def probability(self, cos: np.ndarray) -> np.ndarray:
        """The model's estimate that a frame matches the text."""
        z = self.spec.logit_scale * cos + self.spec.logit_bias
        return 1.0 / (1.0 + np.exp(-z))
