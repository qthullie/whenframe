"""The thirteenth round (docs/PROTOCOL.md): a small model, for phones. PE core S/16 and T/16, exported here.

    python scripts/small.py sample --annotations didemo_train.json --videos DIR --folder CHECK
    TORCH_PYTHON -I scripts/small.py export --model pe-core-s16-224 --folder CHECK --out DIR   # PyTorch, open_clip
    python scripts/small.py check --folder DIR                     # the export against open_clip, before any number
    NNCF_PYTHON -I scripts/small.py compress --folder DIR          # both towers in 8 bits, for OpenVINO
    python scripts/small.py install --folder DIR                   # into whenframe's cache, as MODELS names it

`sample` decodes the ninth round's 16 frames of DiDeMo's tuning videos (the middle cached frame of the first 16
videos with a description) at 224 and at 384 pixels, squashed as the tool squashes frames, with one description of
each and five more sentences for the tokenizer alone (scripts/pe_core.py's).

`export` runs where PyTorch and open_clip are installed (not whenframe's dependencies): open_clip's PE-Core-S-16-384
or T-16-384 with timm's weights at a fixed revision, read into memory, the image tower set to the candidate's size as
timm resizes its positions, its rotary embedding written as scripts/pe_core.py's fast_rope writes it (the same
arithmetic, fewer operations); then the image tower and the text tower exported to ONNX in float32, the reference's
embeddings of the sample, its token ids and its logit scale beside them, and timm's tokenizer at the same revision.

`check` compares, as the protocol fixes it: the token ids whenframe's tokenizer gives must be open_clip's, and each
embedding of the export (ONNX Runtime, float32) at a cosine of at least 0.999 to the reference's.

`compress` saves both towers for OpenVINO with their weights in 8 bits (scripts/text8.py's compression, each tower in
a process of its own); `install` copies them, with the tokenizer, where whenframe's cache keeps the model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# the candidates: open_clip's model and timm's weights at the revision read, and the image size
TIMM = {
    "PE-Core-S-16-384": ("timm/PE-Core-S-16-384", "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"),
    "PE-Core-T-16-384": ("timm/PE-Core-T-16-384", "7fe539ed578ac49a1c2b4f946e4b0747704c825a"),
}
CANDIDATES = {
    "pe-core-s16-224": ("PE-Core-S-16-384", 224),
    "pe-core-s16-384": ("PE-Core-S-16-384", 384),
    "pe-core-t16-224": ("PE-Core-T-16-384", 224),
    "pe-core-t16-384": ("PE-Core-T-16-384", 384),
}
SIZES = (224, 384)
TEXT_LENGTH = 32
MIN_COSINE = 0.999
VISION, TEXT = "onnx/vision_model.onnx", "onnx/text_model.onnx"
VISION8, TEXT8 = "onnx/vision_model_int8.xml", "onnx/text_model_int8.xml"
WEB = "onnx/text_model_web.onnx"  # the text tower for a browser: TEXT8's 8-bit weights, in ONNX


def sample(args) -> None:
    import didemo
    from pe_core import EXTRA

    from whenframe.video import sample_frames

    out = Path(args.folder)
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(ROOT / "results" / "cache" / "didemo_train_openvino.npz")
    names = [str(n) for n in base["names"]]
    frozen, queries, _ = didemo.load("train", args.annotations)
    said = {}
    for q in queries:
        said.setdefault(frozen.names[q["video"]] if isinstance(q["video"], int) else q["video"], []).append(q["text"])
    frames = {size: [] for size in SIZES}
    texts, picked = [], []
    for v, name in enumerate(names):
        if len(texts) == 16:
            break
        path = Path(args.videos) / name
        path = path if path.exists() else path.with_name(f"{name}.mp4")
        if not path.exists() or name not in said:
            continue
        want = base["times"][base["starts"][v] : base["starts"][v] + base["counts"][v]]
        at_size = {}
        for size in SIZES:
            t, f = sample_frames(path, 1.0, size)
            at = np.flatnonzero(np.round(t, 3) == np.round(want[len(want) // 2], 3))
            if len(at):
                at_size[size] = (f[at[0]], float(t[at[0]]))
        if len(at_size) == len(SIZES):
            for size in SIZES:
                frames[size].append(at_size[size][0])
            texts.append(said[name][0])
            picked.append([name, at_size[SIZES[0]][1]])
    for size in SIZES:
        np.save(out / f"frames_{size}.npy", np.stack(frames[size]))
    (out / "texts.json").write_text(json.dumps(texts + EXTRA, indent=1))
    (out / "picked.json").write_text(json.dumps(picked, indent=1))
    print(f"{len(texts)} frames at {SIZES} pixels, {len(texts)} descriptions and {len(EXTRA)} more sentences in {out}")


def load_open_clip(name: str):
    """open_clip's model with timm's weights at the revision fixed above, read into memory (never to disk)."""
    import open_clip
    from safetensors.torch import load

    repo, revision = TIMM[name]
    t0 = time.time()
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/open_clip_model.safetensors") as r:  # noqa: S310
        data = r.read()
    print(f"{repo}: {len(data) / 1e6:.0f} MB read into memory in {time.time() - t0:.0f} s", flush=True)
    model = open_clip.create_model(name, pretrained=None)
    model.load_state_dict(load(data), strict=True)
    return model.eval()


def set_size(visual, size: int) -> None:
    """The image tower at `size` pixels: timm resizes its absolute positions and its rotary embedding's grid."""
    trunk = getattr(visual, "trunk", visual)
    current = trunk.patch_embed.img_size
    if tuple(current) != (size, size):
        trunk.set_input_size(img_size=size)


class TextTower:
    """open_clip's encode_text as a module, for the export: token ids in, the text's embedding out (not normalised,
    as the tool normalises)."""

    def __new__(cls, model):
        import torch

        class Tower(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model.encode_text(input_ids)

        return Tower(model)


def export(args) -> None:
    import open_clip
    import torch
    from pe_core import cosines, fast_rope

    torch.set_num_threads(4)
    name, size = CANDIDATES[args.model]
    repo, revision = TIMM[name]
    here, out = Path(args.folder), Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    model = load_open_clip(name)
    set_size(model.visual, size)
    frames = np.load(here / f"frames_{size}.npy")
    texts = json.loads((here / "texts.json").read_text())
    x = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()  # PE's [-1, 1]
    ids = open_clip.get_tokenizer(name)(texts)
    assert ids.shape[1] == TEXT_LENGTH, ids.shape
    with torch.no_grad():
        ref_images = model.encode_image(x).numpy()
        ref_texts = model.encode_text(ids).numpy()
    layers = fast_rope(model.visual)
    with torch.no_grad():
        fast = model.encode_image(x).numpy()
    info = {"model": args.model, "open_clip": name, "repo": repo, "revision": revision, "size": size,
            "torch": torch.__version__, "open_clip_version": open_clip.__version__,
            "logit_scale": float(model.logit_scale.exp().detach()),
            "fast_rope": {"layers": layers, "cosine_min": float(cosines(fast, ref_images).min())},
            "weights": {"image": sum(p.numel() for p in model.visual.parameters()),
                        "all": sum(p.numel() for p in model.parameters())}}  # fmt: skip
    with torch.no_grad():
        torch.onnx.export(model.visual, (x[:2],), str(out / VISION), input_names=["pixel_values"],
                          output_names=["image_embeds"], opset_version=18, dynamo=False,
                          dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}})  # fmt: skip
        torch.onnx.export(TextTower(model), (ids[:2],), str(out / TEXT), input_names=["input_ids"],
                          output_names=["text_embeds"], opset_version=18, dynamo=False,
                          dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}})  # fmt: skip
    np.save(out / "ref_images.npy", ref_images)
    np.save(out / "ref_texts.npy", ref_texts)
    np.save(out / "ref_ids.npy", ids.numpy())
    shutil.copy(here / f"frames_{size}.npy", out / "frames.npy")
    shutil.copy(here / "texts.json", out / "texts.json")
    with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json") as r:  # noqa: S310
        (out / "tokenizer.json").write_bytes(r.read())
    (out / "export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def check(args) -> None:
    """The export against the reference, as the protocol fixes it; writes check.json in the folder."""
    import onnxruntime as ort

    from whenframe.encoder import Encoder

    folder = Path(args.folder)
    info = json.loads((folder / "export.json").read_text())
    texts = json.loads((folder / "texts.json").read_text())
    ids = Encoder(info["model"], folder=folder).tokenize(texts)  # the tool's own tokenisation (towers not loaded)
    ids_equal = bool(np.array_equal(ids.astype(np.int64), np.load(folder / "ref_ids.npy")))
    from pe_core import cosines

    frames = np.load(folder / "frames.npy")
    x = (frames.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    images = ort.InferenceSession(str(folder / VISION)).run(["image_embeds"], {"pixel_values": x})[0]
    text = ort.InferenceSession(str(folder / TEXT)).run(["text_embeds"], {"input_ids": ids.astype(np.int64)})[0]
    result = {"ids_equal": ids_equal,
              "image_cosine_min": float(cosines(images, np.load(folder / "ref_images.npy")).min()),
              "text_cosine_min": float(cosines(text, np.load(folder / "ref_texts.npy")).min())}  # fmt: skip
    result["passed"] = ids_equal and min(result["image_cosine_min"], result["text_cosine_min"]) >= MIN_COSINE
    (folder / "check.json").write_text(json.dumps(result, indent=1))
    print(info["model"], result)
    if not result["passed"]:
        sys.exit("the export is not the reference: the protocol leaves this size out")


def compress(args) -> None:
    """Both towers for OpenVINO with their weights in 8 bits, each compressed in a process of its own."""
    folder = Path(args.folder)
    for source, dest in ((VISION, VISION8), (TEXT, TEXT8)):
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "text8.py"), "compress", "--source",
                        str(folder / source), "--out", str(folder / dest)], check=True)  # fmt: skip
        sizes = sum(p.stat().st_size for p in (folder / dest, (folder / dest).with_suffix(".bin")))
        print(f"{dest}: {sizes / 1e6:.1f} MB", flush=True)


def install(args) -> None:
    """The 8-bit towers and the tokenizer where whenframe's cache keeps the model (MODELS[model].repo)."""
    from whenframe.encoder import MODELS, cache_dir

    folder = Path(args.folder)
    info = json.loads((folder / "export.json").read_text())
    spec = MODELS[info["model"]]
    dest = cache_dir() / "models" / spec.repo.replace("/", "--")
    for name in ("tokenizer.json", VISION8, VISION8.replace(".xml", ".bin"), TEXT8, TEXT8.replace(".xml", ".bin")):
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / name, dest / name)
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"{info['model']}: installed in {dest}, {total / 1e6:.1f} MB")


def compressed_weights(xml: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Each weight NNCF compressed in an OpenVINO model (INT8_ASYM): its 8-bit values, zero points and scales, by the
    name of the ONNX initializer it was converted from; OpenVINO computes the weight as (q - zero point) * scale."""
    from pe_core import _openvino

    model = _openvino().Core().read_model(str(xml))
    out = {}
    for op in model.get_ordered_ops():
        name = op.get_friendly_name()
        if op.get_type_name() != "Constant" or op.get_output_element_type(0).get_type_name() != "u8":
            continue
        if name.endswith("/zero_point"):
            continue
        (convert,) = [t.get_node() for t in op.output(0).get_target_inputs()]
        (subtract,) = [t.get_node() for t in convert.output(0).get_target_inputs()]
        (multiply,) = [t.get_node() for t in subtract.output(0).get_target_inputs()]
        assert (subtract.get_type_name(), multiply.get_type_name()) == ("Subtract", "Multiply"), name

        def other(node, source):
            inputs = [i.get_source_output().get_node() for i in node.inputs()]
            (x,) = [n for n in inputs if n.get_friendly_name() != source.get_friendly_name()]
            while x.get_type_name() == "Convert":  # the zero point, converted before the subtraction
                x = x.input(0).get_source_output().get_node()
            return x.get_data()

        zero, scale = other(subtract, convert), other(multiply, subtract)
        out[name] = (op.get_data().copy(), zero.copy(), scale.astype(np.float32))
    return out


def web(args) -> None:
    """The text tower for a browser (onnxruntime-web): the float32 export's graph with the 8-bit weights the tool
    runs (TEXT8). Each compressed weight is kept as its 8-bit values, zero points and scales and dequantized before
    its product as OpenVINO dequantizes it, (q - zero point) * scale; the token embeddings after the lookup, only the
    rows of the description's tokens, so that the table stays in 8 bits in memory. Writes onnx/text_model_web.onnx
    and checks it with ONNX Runtime against the 8-bit tower as OpenVINO runs it (web.json)."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from pe_core import cosines

    from whenframe.encoder import Encoder

    folder = Path(args.folder)
    weights = compressed_weights(folder / TEXT8)
    model = onnx.load(str(folder / TEXT))
    graph = model.graph
    floats = {i.name: i for i in graph.initializer}
    first, inits = [], []
    for name, (q, zero, scale) in weights.items():
        w = numpy_helper.to_array(floats[name])
        assert w.shape == q.shape, (name, w.shape, q.shape)
        deq = (q.astype(np.float32) - zero.astype(np.float32)) * scale
        # within a step of its channel (half a step, and the scale's own rounding to 16 bits): this weight's 8 bits
        assert (np.abs(deq - w) <= scale + 1e-7).all(), f"{name}: not this weight's 8 bits"
        axis = 1 if zero.shape[0] == 1 else 0
        z, s = zero.reshape(-1), scale.reshape(-1)
        graph.initializer.remove(floats[name])
        inits += [numpy_helper.from_array(q, f"{name}.q"), numpy_helper.from_array(z, f"{name}.zero"),
                  numpy_helper.from_array(s, f"{name}.scale")]  # fmt: skip
        users = [n for n in graph.node if name in n.input]
        if [n.op_type for n in users] == ["Gather"] and axis == 0:  # the token embeddings: the rows looked up only
            (g,) = users
            ids, out = g.input[1], g.output[0]
            p = f"{name}.rows"
            nodes = [helper.make_node("Gather", [f"{name}.q", ids], [f"{p}.q"], axis=0),
                     helper.make_node("Gather", [f"{name}.zero", ids], [f"{p}.zero"], axis=0),
                     helper.make_node("Gather", [f"{name}.scale", ids], [f"{p}.scale"], axis=0),
                     helper.make_node("Cast", [f"{p}.q"], [f"{p}.qf"], to=TensorProto.FLOAT),
                     helper.make_node("Cast", [f"{p}.zero"], [f"{p}.zf"], to=TensorProto.FLOAT),
                     helper.make_node("Unsqueeze", [f"{p}.zf", f"{p}.last"], [f"{p}.zu"]),
                     helper.make_node("Unsqueeze", [f"{p}.scale", f"{p}.last"], [f"{p}.su"]),
                     helper.make_node("Sub", [f"{p}.qf", f"{p}.zu"], [f"{p}.d"]),
                     helper.make_node("Mul", [f"{p}.d", f"{p}.su"], [out])]  # fmt: skip
            inits.append(numpy_helper.from_array(np.array([-1], np.int64), f"{p}.last"))
            at = list(graph.node).index(g)
            graph.node.remove(g)
            for k, n in enumerate(nodes):
                graph.node.insert(at + k, n)
        else:  # a matrix of a product: dequantized whole, as its operator reads it
            first.append(helper.make_node("DequantizeLinear", [f"{name}.q", f"{name}.scale", f"{name}.zero"], [name],
                                          axis=axis))  # fmt: skip
    for k, n in enumerate(first):
        graph.node.insert(k, n)
    graph.initializer.extend(inits)
    # any length up to 32: the tower is causal and reads the end token's output, so the tokens after it change
    # nothing, and a browser computes only the description's own (checked below)
    length = graph.input[0].type.tensor_type.shape.dim[1]
    length.ClearField("dim_value")
    length.dim_param = "length"
    onnx.checker.check_model(model)
    out = folder / WEB
    onnx.save(model, str(out))
    texts = json.loads((folder / "texts.json").read_text()) + list(args.sentences or [])
    if args.annotations:  # and the 200 descriptions round13.py's stage 1 timed the tool's tower on
        import didemo

        texts += [q["text"] for q in didemo.load("train", args.annotations)[1][:200]]
    enc = Encoder(json.loads((folder / "export.json").read_text())["model"], folder=folder)
    ids = enc.tokenize(texts).astype(np.int64)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = np.concatenate([session.run(["text_embeds"], {"input_ids": ids[i : i + 1]})[0] for i in range(len(ids))])
    ends = ids.argmax(axis=1) + 1  # the end token is the largest id
    short = np.concatenate([session.run(["text_embeds"], {"input_ids": ids[i : i + 1, :n]})[0]
                            for i, n in enumerate(ends)])  # fmt: skip
    # the 8-bit tower as the tool runs it (OpenVINO, normalised), a description at a time as a search embeds it
    tool = np.concatenate([enc.encode_texts([t]) for t in texts])
    ref = np.load(folder / "ref_texts.npy")
    result = {"mb": out.stat().st_size / 1e6, "weights_8bit": len(weights), "texts": len(texts),
              "cosine_to_tool_min": float(cosines(got, tool).min()),
              "own_tokens_only_cosine_min": float(cosines(short, got).min()),
              "cosine_to_reference_min": float(cosines(got[: len(ref)], ref).min()),
              "tool_to_reference_min": float(cosines(tool[: len(ref)], ref).min())}  # fmt: skip
    (folder / "web.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    # for the browser (site/clip.js, scripts/web_text.mjs): CLIP's merges, from which its vocabulary follows, and
    # what the browser must find: the tool's token ids and embeddings
    merges = json.loads((folder / "tokenizer.json").read_text())["model"]["merges"]
    (folder / "merges.txt").write_text("".join(f"{' '.join(m) if isinstance(m, list) else m}\n" for m in merges))
    (folder / "web_check.json").write_text(json.dumps({"texts": texts, "ids": ids.tolist(),
                                                        "tool": tool.astype(float).round(7).tolist()}))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "export", "check", "compress", "install", "web"])
    ap.add_argument("--sentences", nargs="*", help="web: more descriptions to check the tower on")
    ap.add_argument("--annotations", help="sample: DiDeMo's didemo_train.json; web: its 200 tuning descriptions too")
    ap.add_argument("--videos", help="sample: the folder of DiDeMo's tuning videos")
    ap.add_argument("--folder", help="sample: where the sample goes; export: where it is; others: the export's")
    ap.add_argument("--model", choices=list(CANDIDATES), help="export: the candidate")
    ap.add_argument("--out", help="export: the candidate's folder")
    args = ap.parse_args()
    stages = {"sample": sample, "export": export, "check": check, "compress": compress, "install": install, "web": web}
    stages[args.stage](args)


if __name__ == "__main__":
    main()
