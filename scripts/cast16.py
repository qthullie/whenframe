"""Writes whenframe/data/cast16.onnx: one Cast node, float16 to float32, of any (n, d) block.

numpy converts float16 one value at a time (10 ns a value here); ONNX Runtime's Cast
is vectorised (under 1 ns), which whenframe/half.py uses for libraries kept as float16.

    python scripts/cast16.py      # needs onnx (pip install -e ".[dev]")
"""

from pathlib import Path

from onnx import TensorProto, helper, save

OUT = Path(__file__).resolve().parents[1] / "whenframe" / "data" / "cast16.onnx"

graph = helper.make_graph(
    [helper.make_node("Cast", ["half"], ["full"], to=TensorProto.FLOAT)],
    "cast16",
    [helper.make_tensor_value_info("half", TensorProto.FLOAT16, ["n", "d"])],
    [helper.make_tensor_value_info("full", TensorProto.FLOAT, ["n", "d"])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="whenframe")
model.ir_version = 8
save(model, OUT)
print(f"wrote {OUT}: {OUT.stat().st_size} bytes")
