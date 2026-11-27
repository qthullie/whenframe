"""Image and text towers: SigLIP 2 B/16 and Meta's PE core B/16, which a new library is indexed with; SigLIP 2
B/32 and So400m, for libraries built in the modes of earlier versions; MobileCLIP-S2 for research use.

SigLIP scores an image-text pair with a sigmoid, sigmoid(scale * cos + bias),
trained on each pair independently, so its score is an absolute "does this
match" that can be compared across videos. The ONNX export leaves the scale and
bias out; they are those of the original google/siglip2 checkpoint.

SigLIP 2 B/32 (256 px, 64 patches instead of 196) was the cheaper model of the
economy and precision modes; its scale and bias are its own checkpoint's.

SigLIP 2 So400m (400 million parameters in the image tower, 5 times the base
model's) comes from Immich's ONNX export of the open_clip weights, which takes
one image or one text at a time; its descriptions are lower-cased as the base
model's are, which did as well as open_clip's own cleaning on DiDeMo tuning
videos, and its scale and bias are those of google/siglip2-so400m-patch14-224.

MobileCLIP-S2 is measured in the benchmark but is not used by the tool: Apple
licenses its weights for non-commercial research only. It was trained with a
softmax; its scale and bias map its cosine onto SigLIP's logit, fitted on DiDeMo
training pairs (`scripts/didemo.py calibrate`).

The image towers run with OpenVINO, which computes in bfloat16 on processors
that do so natively (AMX, AVX512-BF16) and in float32 elsewhere; SigLIP 2's can
run with ONNX Runtime instead (`--backend onnxruntime`), PE core B/16's, exported
for OpenVINO, cannot. The text towers run with ONNX Runtime in float32, but
So400m's and PE's, which OpenVINO runs faster (So400m's ten times on this
machine, 74 ms a text instead of 721) when it runs the image towers.
"""

from __future__ import annotations

import hashlib
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
    # the files, when they are not all at the repository's main: (path in the model's folder, URL, SHA-256)
    sources: tuple[tuple[str, str, str], ...] = ()
    text_openvino: bool = False  # the text tower runs with OpenVINO too, when the image tower does
    bank_beta: float = 50.0  # query-bank correction, as tuned on DiDeMo training videos
    bank_lambda: float = 1.0
    spans_lambda: float | None = None  # its strength when moments are adaptive spans; None: as bank_lambda
    # what is said and what is heard in a search, as chosen with this model on YouCook2's and UnAV-100's tuning
    # sets (sixth round, results/compare_tuned.json); a temporal model's file brings its own (whenframe/temporal.py)
    speech_pad: float = 0.0  # seconds before and after a frame in which what is said counts
    sound_weight: float = 2.0  # of CLAP's standardised score, added to the frames'


RELEASE = "https://github.com/qthullie/whenframe/releases/download/models-1"  # files no model hub serves
RELEASE2 = "https://github.com/qthullie/whenframe/releases/download/models-2"  # the text towers in 8 bits
SIGLIP2_BASE = (
    "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/"
    "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
)  # the revision measured

MODELS = {
    # SigLIP 2 B/16: onnx-community's image tower, and its text tower with its weights in 8 bits (twelfth round,
    # scripts/text8.py), which whenframe's release models-2 serves
    "siglip2-base": ModelSpec(
        "onnx-community/siglip2-base-patch16-224-ONNX", 224, 112.6689, -16.7717, spans_lambda=0.5,
        text_file="onnx/text_model_int8.xml", text_openvino=True,
        sources=(
            ("tokenizer.json", f"{SIGLIP2_BASE}/tokenizer.json",
             "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
            ("onnx/vision_model.onnx", f"{SIGLIP2_BASE}/onnx/vision_model.onnx",
             "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad"),
            ("onnx/text_model_int8.xml", f"{RELEASE2}/siglip2-base-text-int8.xml",
             "e18ff373d9a8174509f69f9a7de067d99ff116b11f60ebb617fec03a300ba639"),
            ("onnx/text_model_int8.bin", f"{RELEASE2}/siglip2-base-text-int8.bin",
             "11ba698c1abcbbdff6527c6187da8707dcaa6080ee872d18b4b0e161c2ea4f6b"),
        ),
    ),
    "siglip2-b32": ModelSpec("onnx-community/siglip2-base-patch32-256-ONNX", 256, 111.8326, -16.7669,
                             bank_lambda=0.5, speech_pad=5.0, sound_weight=1.5),
    # research use only (Apple Machine Learning Research Model licence)
    "mobileclip-s2": ModelSpec(
        "Xenova/mobileclip_s2", 256, 66.86, -19.567, pixels="unit", image_output="image_embeds",
        text_output="text_embeds", text_length=77, pad_token=None, clean="none", bank_beta=10.0,
    ),
    "siglip2-so400m": ModelSpec(
        "immich-app/ViT-SO400M-14-SigLIP2__webli", 224, 109.8812, -15.9324, vision_file="visual/model.onnx",
        text_file="textual/model.onnx", tokenizer_file="textual/tokenizer.json", image_input="image",
        text_input="text", text_dtype="int32", image_output="embedding", text_output="embedding",
        clean="lower", weights_folder="textual", text_openvino=True, sound_weight=1.5,
    ),
    # Meta's Perception Encoder, PE core B/16 (Apache License 2.0), measured in the ninth to twelfth rounds:
    # timm/PE-Core-B-16's tokenizer, its image tower exported again from timm/PE-Core-B-16's weights, for OpenVINO,
    # one frame at a time (scripts/pe_core.py), and onnx-community's text tower with its weights in 8 bits (twelfth
    # round, scripts/text8.py), which no model hub serves: whenframe's releases models-1 and models-2 do, built there
    # again to the same bytes (.github/workflows/models.yml). CLIP has no bias of its own: the bias only places the
    # probability shown with a result, at even odds for a cosine of 0.2.
    "pe-core-b16": ModelSpec(
        "onnx-community/PE-Core-B16-224-ONNX", 224, 99.9263, -19.9853, vision_file="onnx/vision_model_one.xml",
        text_file="onnx/text_model_int8.xml", image_output="image_embeds", text_output="text_embeds",
        text_length=32, pad_token=None, clean="none", text_openvino=True,
        # its query bank and spans' strength as chosen on the tuning sets (results/round9_bank.json)
        bank_beta=50.0, bank_lambda=1.0, spans_lambda=0.5,
        sources=(
            ("tokenizer.json", "https://huggingface.co/timm/PE-Core-B-16/resolve/"
             "0038414f37721c5eafc1a5e0da802d291c909de3/tokenizer.json",
             "43d4281659ce30b6aca2c7485944b3d73925bb7f87c7234daf3b6b094a6d618c"),
            ("onnx/text_model_int8.xml", f"{RELEASE2}/pe-core-b16-text-int8.xml",
             "a95c1b12b688a33d3a23cf42797eb04e69276da42cdb6c107599bca1408d385e"),
            ("onnx/text_model_int8.bin", f"{RELEASE2}/pe-core-b16-text-int8.bin",
             "ad94692b99865c4eea1aacb25eff76ec9b5098662da32ed1c1ea981f67622166"),
            ("onnx/vision_model_one.xml", f"{RELEASE}/pe-core-b16-vision.xml",
             "8bb5f9908c0011ecd138cb328f1ef7dc862f890c804ff32adf7882402b3bd474"),
            ("onnx/vision_model_one.bin", f"{RELEASE}/pe-core-b16-vision.bin",
             "dde7ec6a300b4fe7a1c4cccfa3f27fe62adbdbc7b6ea9ccea1fbb6db29e00bc1"),
        ),
    ),
    # the thirteenth round's candidates for a small model (docs/PROTOCOL.md, scripts/small.py): PE core S/16 and
    # T/16 (Apache License 2.0), exported here from timm's weights with their image tower at 224 or 384 pixels, both
    # towers' weights in 8 bits; measured from files in whenframe's cache, which no release serves yet. Each one's
    # query bank (beta, strength) and spans' strength as chosen on the tuning sets (results/round13_bank.json)
    **{
        f"pe-core-{size_name}-{pixels}": ModelSpec(
            f"whenframe/pe-core-{size_name}-{pixels}", pixels, 100.0, -20.0, vision_file="onnx/vision_model_int8.xml",
            text_file="onnx/text_model_int8.xml", image_output="image_embeds", text_output="text_embeds",
            text_length=32, pad_token=None, clean="none", text_openvino=True,
            bank_beta=beta, bank_lambda=lam, spans_lambda=spans,
        )
        for (size_name, pixels), (beta, lam, spans) in {
            ("s16", 224): (50.0, 1.0, 0.5), ("s16", 384): (20.0, 0.5, 0.25),
            ("t16", 224): (50.0, 1.0, 0.25), ("t16", 384): (50.0, 0.5, 0.25),
        }.items()
    },
}  # fmt: skip
DEFAULT_MODEL = "siglip2-base"  # the encoder of a text alone (the demo's server) and of the first model's scores
# what a new library is indexed with: SigLIP 2 B/16 and PE core B/16, searched through whenframe's models v3 and v4
# together (docs/RESULTS.md, eleventh round). Libraries built in the modes of earlier versions keep their models:
# SigLIP 2 B/16 alone (standard), B/32 (economy), both (precision), So400m (large until the eleventh round).
DEFAULT_LIBRARY = "siglip2-base+pe-core-b16"
# weight of each model's scores in the precision mode's combined search, as tuned on DiDeMo training videos
FUSION = {"siglip2-base": 1.0, "siglip2-b32": 0.25, "mobileclip-s2": 1.0}


@dataclass(frozen=True)
class Fused:
    """Several models searched as the tenth and eleventh rounds measured them together (scripts/compare.py,
    `fused`): each model's corrected scores standardised over the library and added with its weight, the sum
    standardised again and put back on the first model's scale (its mean and spread), so that the speech weight
    means what it means with one model; with the speech settings and sound weight chosen with them on the tuning
    sets."""

    weights: tuple[float, ...]  # in the order of the models
    speech_weight: float
    speech_pad: float
    speech_normalise: str
    sound_weight: float


FUSED = {"siglip2-base+pe-core-b16": Fused((1.0, 0.75), 0.005, 5.0, "raw", 1.5)}


def cache_dir() -> Path:
    root = os.environ.get("WHENFRAME_CACHE") or Path.home() / ".cache" / "whenframe"
    return Path(root)


def fetch(spec: ModelSpec) -> Path:
    """Download the model files once into the cache and return their folder."""
    if spec.sources:
        folder = cache_dir() / "models" / spec.repo.replace("/", "--")
        for name, url, digest in spec.sources:
            download(url, folder / name, digest)
        return folder
    folder = fetch_files(spec.repo, (spec.tokenizer_file, spec.vision_file, spec.text_file))
    if spec.weights_folder:
        fetch_folder(spec.repo, spec.weights_folder)
    return folder


def download(url: str, dest: Path, sha256: str = "") -> None:
    """A file, once: written next to its place and moved there whole, after its SHA-256 is checked when given."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed hosts)
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink()
        raise OSError(f"{url}: SHA-256 {h.hexdigest()}, expected {sha256}")
    tmp.replace(dest)


def fetch_files(repo: str, names: tuple[str, ...]) -> Path:
    """Download files of a Hugging Face repository once into the cache and return their folder."""
    folder = cache_dir() / "models" / repo.replace("/", "--")
    for name in names:
        download(f"https://huggingface.co/{repo}/resolve/main/{name}", folder / name)
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
            # a tower exported for OpenVINO (.xml) runs with it whatever the backend
            openvino = self.backend == "openvino" or self._vision_path.suffix == ".xml"
            runtime = _OpenVINO if openvino else _OnnxRuntime
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
            # a tower saved for OpenVINO (.xml) runs with it whatever the backend
            openvino = self._text_path.suffix == ".xml" or (self.backend == "openvino" and self.spec.text_openvino)
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
