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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "export", "check", "compress", "install"])
    ap.add_argument("--annotations", help="sample: DiDeMo's didemo_train.json")
    ap.add_argument("--videos", help="sample: the folder of DiDeMo's tuning videos")
    ap.add_argument("--folder", help="sample: where the sample goes; export: where it is; others: the export's")
    ap.add_argument("--model", choices=list(CANDIDATES), help="export: the candidate")
    ap.add_argument("--out", help="export: the candidate's folder")
    args = ap.parse_args()
    {"sample": sample, "export": export, "check": check, "compress": compress, "install": install}[args.stage](args)


if __name__ == "__main__":
    main()
