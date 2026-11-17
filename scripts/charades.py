"""Charades-STA test (Gao et al., 2017): the parameters tuned on DiDeMo, applied without any change.

Charades videos (Sigurdsson et al., 2016) show people doing everyday things at
home, about 30 seconds each; Charades-STA locates 3,720 sentences in 1,334 of
them. The usual metrics are those of a given video (R1 at IoU 0.5 and 0.7, mean
IoU); the whole-library search is reported as well.

    python scripts/charades.py extract  --videos DIR --annotations test.parquet
    python scripts/charades.py evaluate --annotations test.parquet
    python scripts/charades.py prior    --annotations test.parquet   # a reference added after the run

The annotations and an mp4 copy of the videos are on the Hugging Face Hub
(lmms-eval/charades_sta). Charades is licensed for non-commercial research.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import CORPUS_KS, Frozen, iou, latency, run  # noqa: E402

from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
METHODS = ["frame_window", "frame_window_bank", "whenframe", "whenframe_bank"]


def annotations(path: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return [{"video": r["video"], "text": r["caption"].strip(), "moment": (float(r["timestamp"][0]),
             float(r["timestamp"][1]))} for r in rows]  # fmt: skip


def extract(args) -> None:
    ann = annotations(args.annotations)
    names = sorted({a["video"] for a in ann})
    enc = didemo.make_encoder(args.model_dir)
    cols = {k: [] for k in ("emb", "times", "keep", "starts", "counts", "durations")}
    done, row, t0 = [], 0, time.time()
    for name in names:
        p = Path(args.videos) / name
        try:
            info = probe(p)
            t, frames = sample_frames(p, 1.0, enc.size)
            e = enc.encode_images(frames)
        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
            print(f"skip {name}: {ex}", file=sys.stderr)
            continue
        if not len(frames):
            continue
        keep = np.zeros(len(frames), dtype=bool)
        keep[drop_repeats(frames)] = True
        for k, v in (("emb", e.astype(np.float16)), ("times", t.astype(np.float32)), ("keep", keep), ("starts", row),
                     ("counts", len(frames)), ("durations", info.duration or len(frames))):  # fmt: skip
            cols[k].append(v)
        done.append(name)
        row += len(frames)
        if len(done) % 100 == 0:
            print(f"{len(done)}/{len(names)} videos, {row} frames, {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(CACHE / "charades_test.npz", names=np.array(done), **{
        k: np.concatenate(v) if k in ("emb", "times", "keep") else np.array(v) for k, v in cols.items()})  # fmt: skip
    print(f"wrote {len(done)} videos, {row} frames in {time.time() - t0:.0f} s")


def load(args):
    """The test videos' frames (every frame, as in the evaluation), the descriptions and their embeddings."""
    z = np.load(CACHE / "charades_test.npz")
    names = z["names"].tolist()
    frozen = Frozen(names=names, emb=z["emb"].astype(np.float32), times=z["times"], starts=z["starts"],
                    counts=z["counts"], durations=z["durations"], keep=z["keep"], cuts=[[] for _ in names])  # fmt: skip
    index = {n: i for i, n in enumerate(names)}
    queries = []
    for a in annotations(args.annotations):
        if a["video"] in index:
            v = index[a["video"]]
            start, end = a["moment"]
            queries.append({"text": a["text"], "video": v,
                            "moment": (max(0.0, start), min(end, float(frozen.durations[v])))})  # fmt: skip
    path = CACHE / "charades_test_text.npy"
    if not path.exists():
        np.save(path, didemo.encode([q["text"] for q in queries], args.model_dir))
    text = np.load(path)
    frozen.bank = didemo.bank(args)  # the DiDeMo bank, unchanged
    return frozen, queries, text


def evaluate(args) -> None:
    frozen, queries, text = load(args)
    names = frozen.names
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    res = {"queries": len(queries), "videos": len(names), "corpus_ks": CORPUS_KS, "methods": METHODS}
    for m in METHODS:
        res[m] = run(frozen, queries, text, m, tuned[m]["params"])
        res[m]["query_ms"] = latency(frozen, text, m, tuned[m]["params"])
        print(m, res[m]["single"], res[m]["corpus"]["R@5 IoU>=0.5"], flush=True)
    (ROOT / "results" / "charades_test.json").write_text(json.dumps(res, indent=1))


def prior(args) -> None:
    """What the window alone scores: 10 seconds in the middle of the video, whatever the description.

    Not in the protocol; added after the evaluation, as a reference for its numbers. Charades-STA
    moments last about 8 seconds in videos of about 30, so a fixed window already overlaps many.
    """
    z = np.load(CACHE / "charades_test.npz")
    durations = dict(zip(z["names"].tolist(), z["durations"].astype(float), strict=True))
    ious = []
    for a in annotations(args.annotations):
        if a["video"] in durations:
            d = durations[a["video"]]
            start = max(0.0, (d - 10.0) / 2)
            ious.append(iou((start, min(d, start + 10.0)), (max(0.0, a["moment"][0]), min(a["moment"][1], d))))
    ious = np.array(ious)
    out = ROOT / "results" / "charades_test.json"
    res = json.loads(out.read_text())
    single = {"R1 IoU>=0.5": float((ious >= 0.5).mean()), "R1 IoU>=0.7": float((ious >= 0.7).mean()),
              "mIoU": float(ious.mean())}  # fmt: skip
    res["middle_window"] = {"single": single}
    out.write_text(json.dumps(res, indent=1))
    print("middle_window", res["middle_window"]["single"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "evaluate", "prior"])
    ap.add_argument("--videos")
    ap.add_argument("--annotations", required=True, help="lmms-eval/charades_sta data/test-00000-of-00001.parquet")
    ap.add_argument("--train-annotations", help="DiDeMo training annotations, for the query bank")
    ap.add_argument("--model-dir")
    a = ap.parse_args()
    {"extract": extract, "evaluate": evaluate, "prior": prior}[a.stage](a)


if __name__ == "__main__":
    main()
