"""The ninth round (docs/PROTOCOL.md): Meta's Perception Encoder, PE core B/16, in place of SigLIP 2 B/16.

    python scripts/pe_core.py fetch     --model-folder DIR      # onnx-community's two towers, timm's tokenizer
    python scripts/pe_core.py sample    --folder CHECK --annotations didemo_train.json --videos DIR
    python -I scripts/pe_core.py reference --folder CHECK --export DIR/onnx/vision_model_any_batch.onnx
    python scripts/pe_core.py check     --folder CHECK --model-folder DIR    # results/round9_check.json

`sample` decodes 16 frames of DiDeMo's tuning videos at the standard mode's instants (one per video, its middle
cached frame, squashed to 224 pixels by ffmpeg) and takes one description of each, with five more sentences for the
tokenizer alone. `reference` runs in an environment of its own, with PyTorch and open_clip: it reads
timm/PE-Core-B-16's weights into memory (never to disk), embeds the frames and descriptions with open_clip's
PE-Core-B-16, and exports the image tower again so that it takes any batch size, as onnx-community's, traced with
a batch of 2, does not. `check` embeds the same frames and descriptions as whenframe does, with OpenVINO, and
compares: the token ids must be equal and each embedding at a cosine of at least 0.999 to the reference's, the
image tower in float32 (on this processor OpenVINO computes every tower in bfloat16, SigLIP 2's too, and the
measure keeps that precision for both).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ONNX = "onnx-community/PE-Core-B16-224-ONNX"
TIMM = "timm/PE-Core-B-16"
CHECKED = ROOT / "results" / "round9_check.json"
# sentences that only test the tokenizer: capitals, punctuation, spaces, accents
EXTRA = ["A man's dog jumps over the FENCE!", "the camera pans left   then right", "kids playing in the snow, 2nd time",
         "café au lait — poured slowly", "first time we see the red car's door open"]  # fmt: skip


def fetch(args) -> None:
    """The two towers as onnx-community exported them and timm's tokenizer, into one folder."""
    folder = Path(args.model_folder)
    files = [(ONNX, "onnx/vision_model.onnx"), (ONNX, "onnx/text_model_fp16.onnx")]
    files += [(TIMM, f) for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                  "vocab.json", "merges.txt", "open_clip_config.json")]  # fmt: skip
    for repo, name in files:
        dest = folder / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{name}", timeout=60) as r:  # noqa: S310
            with open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
        print(f"fetched {repo}/{name}")


def sample(args) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import didemo

    from whenframe.video import sample_frames

    out = Path(args.folder)
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(ROOT / "results" / "cache" / "didemo_train_openvino.npz")
    names = [str(n) for n in base["names"]]
    frozen, queries, _ = didemo.load("train", args.annotations)
    said = {}
    for q in queries:
        said.setdefault(frozen.names[q["video"]] if isinstance(q["video"], int) else q["video"], []).append(q["text"])
    frames, texts, picked = [], [], []
    for v, name in enumerate(names):
        if len(frames) == 16:
            break
        path = Path(args.videos) / name
        path = path if path.exists() else path.with_name(f"{name}.mp4")
        if not path.exists() or name not in said:
            continue
        t, f = sample_frames(path, 1.0, 224)
        want = base["times"][base["starts"][v] : base["starts"][v] + base["counts"][v]]
        at = np.flatnonzero(np.round(t, 3) == np.round(want[len(want) // 2], 3))
        if len(at):
            frames.append(f[at[0]])
            texts.append(said[name][0])
            picked.append([name, float(t[at[0]])])
    np.save(out / "frames.npy", np.stack(frames))
    (out / "texts.json").write_text(json.dumps(texts + EXTRA, indent=1))
    (out / "picked.json").write_text(json.dumps(picked, indent=1))
    print(f"{len(frames)} frames, {len(texts)} descriptions and {len(EXTRA)} more sentences in {out}")


def reference(args) -> None:
    """open_clip's PE-Core-B-16, its weights read into memory; and the image tower exported with any batch size."""
    import open_clip
    import torch
    from safetensors.torch import load

    here = Path(args.folder)
    torch.set_num_threads(4)
    t0 = time.time()
    with urllib.request.urlopen(f"https://huggingface.co/{TIMM}/resolve/main/open_clip_model.safetensors") as r:  # noqa: S310
        data = r.read()
    print(f"weights read into memory: {len(data) / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    model = open_clip.create_model("PE-Core-B-16", pretrained=None)
    model.load_state_dict(load(data), strict=True)
    del data
    model.eval()
    frames = np.load(here / "frames.npy")
    texts = json.loads((here / "texts.json").read_text())
    x = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).contiguous()  # PE's [-1, 1]
    ids = open_clip.get_tokenizer("PE-Core-B-16")(texts)
    with torch.no_grad():
        np.save(here / "ref_images.npy", model.encode_image(x).numpy())
        np.save(here / "ref_texts.npy", model.encode_text(ids).numpy())
    np.save(here / "ref_ids.npy", ids.numpy())
    info = {"torch": torch.__version__, "open_clip": open_clip.__version__,
            "logit_scale": float(model.logit_scale.exp().detach())}  # fmt: skip
    (here / "reference.json").write_text(json.dumps(info, indent=1))
    if args.export:
        with torch.no_grad():
            torch.onnx.export(model.visual, (x[:2],), args.export, input_names=["pixel_values"],
                              output_names=["image_embeds"], opset_version=18, dynamo=False,
                              dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}})  # fmt: skip
        print(f"exported the image tower to {args.export}")
    print(json.dumps(info))


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def check(args) -> None:
    """whenframe's embeddings of the check's frames and descriptions against the reference's."""
    import openvino as ov

    sys.path.insert(0, str(ROOT))
    from whenframe.encoder import Encoder

    here, folder = Path(args.folder), Path(args.model_folder)
    frames = np.load(here / "frames.npy")
    texts = json.loads((here / "texts.json").read_text())
    n = len(texts) - len(EXTRA)
    ref_img, ref_txt, ref_ids = (np.load(here / f) for f in ("ref_images.npy", "ref_texts.npy", "ref_ids.npy"))
    enc = Encoder("pe-core-b16", folder=folder, backend="openvino")
    ids = enc.tokenize(texts)
    images = cosines(enc.encode_images(frames), ref_img)
    words = cosines(enc.encode_texts(texts[:n]), ref_txt[:n])
    # onnx-community's own image tower, for the record: it only takes batches of 2 frames
    core = ov.Core()
    two = core.compile_model(core.read_model(str(folder / "onnx" / "vision_model.onnx")), "CPU")
    x = (frames.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    pairs = np.concatenate([two(np.ascontiguousarray(x[i : i + 2]))[two.output(0)] for i in range(0, len(x), 2)])
    # the same image tower computed in float32, and SigLIP 2 B/16's own gap between OpenVINO's bfloat16 and float32
    # on the same frames: whenframe runs every tower in the precision OpenVINO picks for the processor
    f32 = {"INFERENCE_PRECISION_HINT": "f32"}
    full = core.compile_model(core.read_model(str(folder / enc.spec.vision_file)), "CPU", f32)
    in_f32 = cosines(np.concatenate([full(np.ascontiguousarray(x[i : i + 1]))[full.output(0)] for i in range(len(x))]),
                     ref_img)  # fmt: skip
    base = Encoder("siglip2-base", backend="openvino")
    picked = base.encode_images(frames)
    s32 = core.compile_model(core.read_model(str(base._vision_path)), "CPU", f32)
    y = (frames.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    siglip = cosines(picked, s32(np.ascontiguousarray(y))[s32.output(base.spec.image_output)])
    # the export is what the check judges, in float32; the measure runs it in bfloat16, as SigLIP 2's (decided
    # before any benchmark number: docs/PROTOCOL.md)
    ok = bool(np.array_equal(ids, ref_ids) and in_f32.min() >= 0.999 and words.min() >= 0.999)
    out = {
        "reference": json.loads((here / "reference.json").read_text()),
        "frames": [x for x in json.loads((here / "picked.json").read_text())],
        "descriptions": texts[:n], "more_sentences": EXTRA,
        "token_ids_equal": bool(np.array_equal(ids, ref_ids)),
        "image_cosine": {"min": float(images.min()), "mean": float(images.mean())},
        "text_cosine": {"min": float(words.min()), "mean": float(words.mean())},
        "onnx_community_image_tower_in_pairs": {"min": float(cosines(pairs, ref_img).min())},
        "image_precision": enc.images.precision, "text_precision": enc._text.precision, "export_passed": ok,
        "image_cosine_float32": {"min": float(in_f32.min()), "mean": float(in_f32.mean())},
        "siglip2_base_picked_against_float32": {"precision": base.images.precision, "min": float(siglip.min()),
                                                "mean": float(siglip.mean())},
    }  # fmt: skip
    CHECKED.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("frames", "descriptions", "more_sentences")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "sample", "reference", "check"])
    ap.add_argument("--folder", help="the check's folder: its frames, descriptions and the reference's embeddings")
    ap.add_argument("--model-folder", help="PE core B/16's files, as `fetch` and `reference --export` leave them")
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--videos", help="the folder of DiDeMo's training videos")
    ap.add_argument("--export", help="reference: where to export the image tower, taking any batch size")
    args = ap.parse_args()
    {"fetch": fetch, "sample": sample, "reference": reference, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
