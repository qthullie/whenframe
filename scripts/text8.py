"""The text towers of SigLIP 2 B/16 and PE core B/16 with their weights in 8 bits (twelfth round, part 2).

    python scripts/text8.py release --out DIR     # both towers compressed again, checked, for the release models-2

NNCF compresses each tower's weights to 8 bits (`nncf.compress_weights`, INT8_ASYM, data-free, one scale per
output channel, computed with numpy) and OpenVINO saves it; the activations stay in the precision OpenVINO computes
in. The source files
are the ones the eleventh round's text towers were read from, at fixed revisions, their SHA-256 checked:
onnx-community's ONNX exports of SigLIP 2 B/16's text tower (float32) and PE core B/16's (float16).
scripts/round12.py's towers stage compressed the files measured; `release` does it again from the downloaded
sources, where whenframe's release models-2 is published (.github/workflows/models.yml), and checks that the files
are the bytes measured (results/round12_towers.json).

NNCF is not a dependency of whenframe or of its benchmarks (it pins numpy below 2.5): it is installed beside them,
without its dependencies' pins, as in .github/workflows/models.yml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MEASURED = ROOT / "results" / "round12_towers.json"
NOTICE = ROOT / "models" / "text-int8" / "NOTICE"
# the source of each tower, at the revision whenframe read it from, with its SHA-256
SOURCES = {
    "siglip2-base": ("https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/"
                     "ba1f3b0843f24bc5417d38e19c37b287d719b2f4/onnx/text_model.onnx",
                     "baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293"),
    "pe-core-b16": ("https://huggingface.co/onnx-community/PE-Core-B16-224-ONNX/resolve/"
                    "7668f1df203c393f95310563ff999bfc8fd627b4/onnx/text_model_fp16.onnx",
                    "d837a6e29246892cafb195c92265899defd2ca89166a692f61e0c5b6a57ff65f"),
}  # fmt: skip
ASSET = "{}-text-int8"  # the release's file names: ASSET.xml and ASSET.bin, by model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def compress(source: Path, dest: Path) -> None:
    """One tower's weights in 8 bits, saved for OpenVINO at dest (.xml, with its .bin), written whole. NNCF computes
    the scales with numpy, not with OpenVINO's kernels, whose arithmetic can depend on the processor: the files are
    then the same bytes on any machine."""
    import os

    from whenframe.encoder import _import_openvino

    os.environ["NNCF_DISABLE_OPTIMIZED_COMPRESSION"] = "1"

    ov = _import_openvino()  # first: NNCF then finds no usage-report package, and reports nothing
    import nncf

    model = nncf.compress_weights(ov.Core().read_model(str(source)), mode=nncf.CompressWeightsMode.INT8_ASYM)
    tmp = dest.with_name(dest.stem + ".tmp.xml")
    ov.save_model(model, str(tmp), compress_to_fp16=False)
    tmp.with_suffix(".bin").replace(dest.with_suffix(".bin"))
    tmp.replace(dest)


def release(args) -> None:
    """Both towers compressed again from their sources, and checked against the towers measured: export.json, and
    the release's notes.md when each file has the bytes measured."""
    from importlib.metadata import version

    from whenframe.encoder import download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measured = json.loads(MEASURED.read_text())
    record = {"versions": {p: version(p) for p in ("openvino", "nncf", "numpy")}, "towers": {}}
    for model, (url, digest) in SOURCES.items():
        assert measured[model]["source_sha256"] == digest, f"{model}: not the source measured"
        source = out / "source" / model / Path(url).name
        download(url, source, digest)
        dest = out / f"{ASSET.format(model)}.xml"
        compress(source, dest)
        got = {"text_model_int8.xml": sha256(dest), "text_model_int8.bin": sha256(dest.with_suffix(".bin"))}
        same = got == measured[model]["sha256"]
        record["towers"][model] = {"source": url, "sha256": got, "same_bytes_as_measured": same}
        print(f"{model}: {got}, {'the bytes measured' if same else 'NOT the bytes measured'}")
    (out / "export.json").write_text(json.dumps(record, indent=1))
    if not all(t["same_bytes_as_measured"] for t in record["towers"].values()):
        sys.exit("the towers built are not the ones measured: nothing to publish")
    lines = ["The text towers of SigLIP 2 B/16 and PE core B/16 with their weights in 8 bits, as whenframe",
             "searches with them since the twelfth round (docs/PROTOCOL.md; docs/RESULTS.md, twelfth round).", "",
             "Built by `scripts/text8.py release` from:"]  # fmt: skip
    lines += [f"- {model}: {url} (SHA-256 {digest})" for model, (url, digest) in SOURCES.items()]
    v = record["versions"]
    lines += ["", f"with OpenVINO {v['openvino']} and NNCF {v['nncf']}; each file has the bytes of the tower measured:",
              ""]  # fmt: skip
    lines += [f"- {ASSET.format(m)}.{ext}: {t['sha256'][f'text_model_int8.{ext}']}"
              for m, t in record["towers"].items() for ext in ("xml", "bin")]  # fmt: skip
    (out / "notes.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["release"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"release": release}[args.stage](args)


if __name__ == "__main__":
    main()
