"""DiDeMo: personal Flickr videos with moments described in plain English.

Each video is cut into 5-second segments and each description was located by
four annotators as a run of segments ("times": [[first, last], ...]). The
reference moment used here is the run most annotators chose.

    python scripts/didemo.py extract --split test --videos DIR --annotations didemo_test.json
    python scripts/didemo.py tune    --split train
    python scripts/didemo.py evaluate --split test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import CORPUS_KS, METHODS, Frozen, evaluate_all, grid_search  # noqa: E402

from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
SEGMENT = 5.0


def reference(times: list[list[int]], duration: float) -> tuple[float, float]:
    (a, b), _ = Counter(map(tuple, times)).most_common(1)[0]
    return a * SEGMENT, min((b + 1) * SEGMENT, max(duration, a * SEGMENT + 1.0))


def scene_cuts(path: Path, threshold: float = 0.3) -> list[float]:
    """Shot boundaries as ffmpeg's scene detector sees them, at full frame rate."""
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-i", str(path), "-an", "-vf",
           f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]  # fmt: skip
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    return [float(tok.split(":")[1]) for line in err.splitlines() if "pts_time:" in line
            for tok in line.split() if tok.startswith("pts_time:")]  # fmt: skip


def extract(args) -> None:
    ann = json.loads(Path(args.annotations).read_text())
    folder = Path(args.videos)
    names = sorted({a["video"] for a in ann if (folder / a["video"]).exists()})
    if args.limit:
        names = names[: args.limit]
    enc = Encoder(folder=args.model_dir) if args.model_dir else Encoder()
    out = CACHE / f"didemo_{args.split}.npz"
    emb, times, keep, starts, counts, durations, cuts, decode_s, encode_s = [], [], [], [], [], [], [], [], []
    row = 0
    t_all = time.time()
    for i, name in enumerate(names):
        p = folder / name
        try:
            info = probe(p)
            t0 = time.time()
            t, frames = sample_frames(p, 1.0, enc.size)
            t1 = time.time()
            e = enc.encode_images(frames)
            t2 = time.time()
        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
            print(f"skip {name}: {ex}", file=sys.stderr)
            names[i] = None
            continue
        emb.append(e.astype(np.float16))
        times.append(t.astype(np.float32))
        k = np.zeros(len(frames), dtype=bool)
        k[drop_repeats(frames)] = True
        keep.append(k)
        starts.append(row)
        counts.append(len(frames))
        durations.append(info.duration or len(frames))
        cuts.append(scene_cuts(p))
        decode_s.append(t1 - t0)
        encode_s.append(t2 - t1)
        row += len(frames)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(names)} videos, {row} frames, {time.time() - t_all:.0f} s", flush=True)
    names = [n for n in names if n]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, emb=np.concatenate(emb), times=np.concatenate(times), keep=np.concatenate(keep),
        starts=np.array(starts), counts=np.array(counts), durations=np.array(durations, dtype=np.float32),
        decode_s=np.array(decode_s), encode_s=np.array(encode_s), names=np.array(names),
        cuts=np.array(json.dumps(cuts)), model=np.array(enc.name),
    )  # fmt: skip
    print(f"wrote {out}: {len(names)} videos, {row} frames")


def load(split: str, annotations: str | None = None):
    z = np.load(CACHE / f"didemo_{split}.npz")
    names = z["names"].tolist()
    frozen = Frozen(
        names=names, emb=z["emb"].astype(np.float32), times=z["times"], starts=z["starts"], counts=z["counts"],
        durations=z["durations"], keep=z["keep"], cuts=json.loads(str(z["cuts"])),
    )  # fmt: skip
    ann_path = Path(annotations) if annotations else CACHE.parent.parent / "data" / f"didemo_{split}.json"
    ann = json.loads(ann_path.read_text())
    index = {n: i for i, n in enumerate(names)}
    queries = []
    for a in ann:
        if a["video"] in index:
            v = index[a["video"]]
            queries.append({"text": a["description"], "video": v, "moment": reference(a["times"], frozen.durations[v])})
    cost = {
        "videos": len(names),
        "hours": float(frozen.durations.sum() / 3600),
        "frames": int(len(frozen.times)),
        "decode_s": float(z["decode_s"].sum()),
        "encode_s": float(z["encode_s"].sum()),
        "kept_fraction": float(frozen.keep.mean()),
    }
    return frozen, queries, cost


def text_embeddings(queries, split: str, model_dir: str | None):
    path = CACHE / f"didemo_{split}_text.npy"
    if path.exists():
        return np.load(path)
    enc = Encoder(folder=model_dir) if model_dir else Encoder()
    t = np.concatenate([enc.encode_texts([q["text"] for q in queries[i : i + 64]]) for i in range(0, len(queries), 64)])
    np.save(path, t.astype(np.float32))
    return t


def tune(args) -> None:
    frozen, queries, _ = load(args.split, args.annotations)
    text = text_embeddings(queries, args.split, args.model_dir)
    best = grid_search(frozen, queries, text)
    out = ROOT / "results" / "didemo_tuned.json"
    out.write_text(json.dumps(best, indent=1))
    print(json.dumps(best, indent=1))


def evaluate(args) -> None:
    frozen, queries, cost = load(args.split, args.annotations)
    text = text_embeddings(queries, args.split, args.model_dir)
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    res = evaluate_all(frozen, queries, text, tuned)
    res["cost"] = cost
    res["queries"] = len(queries)
    res["corpus_ks"] = CORPUS_KS
    res["methods"] = list(METHODS)
    out = ROOT / "results" / f"didemo_{args.split}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({m: res[m]["corpus"] for m in METHODS}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "tune", "evaluate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos")
    ap.add_argument("--annotations")
    ap.add_argument("--model-dir")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"extract": extract, "tune": tune, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
