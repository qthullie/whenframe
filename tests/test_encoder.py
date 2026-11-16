"""The two runtimes of the image tower give the same embeddings, on a tiny model built here."""

import importlib.util

import numpy as np
import pytest

from whenframe import encoder as enc

onnx = pytest.importorskip("onnx")


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A folder holding a 3-channel mean followed by a 3x4 matrix, saved as an ONNX image tower."""
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    weights = np.array([[1, 0, 2, 1], [0, 1, 1, -1], [1, 1, 0, 2]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values", "axes"], ["mean"], keepdims=0),
         helper.make_node("MatMul", ["mean", "weights"], ["pooler_output"])],
        "tiny",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["n", 3, 8, 8])],
        [helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["n", 4])],
        [numpy_helper.from_array(np.array([2, 3], dtype=np.int64), "axes"),
         numpy_helper.from_array(weights, "weights")],
    )  # fmt: skip
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    (tmp_path / "onnx").mkdir()
    onnx.save(model, tmp_path / "onnx" / "vision_model.onnx")
    Tokenizer(WordLevel({"<pad>": 0, "[UNK]": 1}, unk_token="[UNK]")).save(str(tmp_path / "tokenizer.json"))
    monkeypatch.setitem(enc.MODELS, "tiny", enc.ModelSpec("test/tiny", 8, 10.0, 0.0))
    return tmp_path


def frames(n=5):
    return np.random.default_rng(0).integers(0, 256, (n, 8, 8, 3), dtype=np.uint8)


def test_onnxruntime(tiny):
    e = enc.Encoder("tiny", folder=tiny, backend="onnxruntime")
    x = frames()
    mean = (x.astype(np.float32) / 127.5 - 1.0).mean(axis=(1, 2))
    want = mean @ np.array([[1, 0, 2, 1], [0, 1, 1, -1], [1, 1, 0, 2]], dtype=np.float32)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert e.dim == 4 and e.images.precision == "f32"
    np.testing.assert_allclose(e.encode_images(x), want, atol=1e-5)


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="OpenVINO is not installed")
def test_openvino_matches_onnxruntime(tiny):
    a = enc.Encoder("tiny", folder=tiny, backend="onnxruntime").encode_images(frames())
    openvino = enc.Encoder("tiny", folder=tiny, backend="openvino")
    b = openvino.encode_images(frames())
    assert openvino.images.precision in ("f32", "bf16", "f16")
    assert (a * b).sum(axis=1).min() > 0.999  # bfloat16 on processors that have it
    from openvino.tools.ovc import telemetry_utils

    assert telemetry_utils.tm.__name__.endswith("telemetry_stub")  # OpenVINO's silent stand-in


def test_backend_names(monkeypatch):
    monkeypatch.delenv("WHENFRAME_BACKEND", raising=False)
    has_openvino = importlib.util.find_spec("openvino") is not None
    assert enc.backend_name() == ("openvino" if has_openvino else "onnxruntime")
    assert enc.backend_name("onnxruntime") == "onnxruntime"
    monkeypatch.setenv("WHENFRAME_BACKEND", "onnxruntime")
    assert enc.backend_name() == "onnxruntime"
    with pytest.raises(ValueError):
        enc.backend_name("tensorflow")
