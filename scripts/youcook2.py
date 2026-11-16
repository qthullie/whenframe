"""YouCook2: long cooking videos with every recipe step described and timed (docs/PROTOCOL.md, third round).

    python scripts/youcook2.py extract    --videos DIR --annotations val.parquet   # frames, with OpenVINO
    python scripts/youcook2.py transcribe --videos DIR --annotations val.parquet   # Whisper base
    python scripts/youcook2.py tune       --annotations val.parquet   # the speech weight, on a third of the videos
    python scripts/youcook2.py evaluate   --annotations val.parquet   # on the other two thirds

The annotations are lmms-eval/YouCook2's data/val-00000-of-00001.parquet; the
videos, nguyenminh04/youcook2-videos on the Hugging Face Hub (videos/ID.mp4).
Every search runs over all the videos found; the steps of a third of them are
for tuning, those of the others for the test.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import MAIN, PREDICT, SHORTLIST, Frozen, frame_scores, iou  # noqa: E402
from explore import sign_test  # noqa: E402

from whenframe import speech  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
FRAMES = CACHE / "youcook2.npz"
SPEECH = CACHE / "youcook2_speech.json"
METHODS = ["frame_window_bank", "whenframe_bank"]
GRID = [speech.SpeechConfig(w, pad, norm) for w in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05) for pad in (0.0, 5.0)
        for norm in ("raw", "best")]  # fmt: skip


def cpu() -> float:
    """CPU seconds of this process and of the ffmpeg processes it waited for."""
    me, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def steps(path: str) -> list[dict]:
    import pyarrow.parquet as pq

    return [{"video": r["youtube_id"], "text": r["sentence"].strip(), "moment": (float(r["segment"][0]),
             float(r["segment"][1]))} for r in pq.read_table(path).to_pylist()]  # fmt: skip


def video_names(args) -> list[str]:
    wanted = {s["video"] for s in steps(args.annotations)}
    return sorted(p.stem for p in Path(args.videos).glob("*.mp4") if p.stem in wanted)


def split(names: list[str]) -> set[str]:
    """The tuning videos: a third of them, drawn once."""
    return set(random.Random(0).sample(sorted(names), len(names) // 3))


def extract(args) -> None:
    """One frame per second of every video, repeats skipped, embedded with OpenVINO; resumes where it stopped."""
    enc = didemo.make_encoder(args.model_dir, MAIN, "openvino")
    parts = CACHE / "youcook2_parts"
    parts.mkdir(parents=True, exist_ok=True)
    names, t0 = video_names(args), time.time()
    for n, name in enumerate(names, 1):
        part = parts / f"{name}.npz"
        if part.exists():
            continue
        p = Path(args.videos) / f"{name}.mp4"
        c0 = cpu()
        info = probe(p)
        t, frames = sample_frames(p, 1.0, enc.size)
        keep = drop_repeats(frames)  # as the tool indexes: repeated frames are not encoded
        c1 = cpu()
        e = enc.encode_images(frames[keep])
        c2 = cpu()
        cost = {"duration": info.duration or len(frames), "decode_cpu": c1 - c0, "encode_cpu": c2 - c1,
                "seen": len(frames)}  # fmt: skip
        np.savez(part.with_suffix(".tmp.npz"), emb=e.astype(np.float16), times=t[keep].astype(np.float32), **cost)
        part.with_suffix(".tmp.npz").replace(part)  # a video is either done or not started
        if n % 25 == 0:
            print(f"{n}/{len(names)} videos, {time.time() - t0:.0f} s", flush=True)
    z = [np.load(parts / f"{name}.npz") for name in names]
    counts = np.array([len(x["times"]) for x in z])
    np.savez_compressed(
        FRAMES, names=np.array(names), precision=np.array(enc.images.precision),
        emb=np.concatenate([x["emb"] for x in z]), times=np.concatenate([x["times"] for x in z]),
        starts=np.concatenate([[0], np.cumsum(counts)[:-1]]), counts=counts,
        **{k: np.array([float(x[k]) for x in z]) for k in ("duration", "decode_cpu", "encode_cpu", "seen")},
    )  # fmt: skip
    print(f"wrote {FRAMES}: {len(names)} videos, {counts.sum()} frames")


def transcribe(args) -> None:
    done = json.loads(SPEECH.read_text()) if SPEECH.exists() else {"model": "base", "videos": {}}
    model = speech.Transcriber("base", threads=4, folder=args.whisper_dir)
    t0 = time.time()
    for name in video_names(args):
        if name in done["videos"]:
            continue
        p = Path(args.videos) / f"{name}.mp4"
        duration = probe(p).duration
        c0 = cpu()
        segments = model(str(p), duration)
        done["videos"][name] = {"duration": duration, "cpu": cpu() - c0,
                                "segments": [[round(a, 2), round(b, 2), s] for a, b, s in segments]}  # fmt: skip
        SPEECH.write_text(json.dumps(done))
        n = len(done["videos"])
        if n % 10 == 0:
            print(f"{n} videos transcribed, {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {SPEECH}: {len(done['videos'])} videos")


def load(args):
    z = np.load(FRAMES)
    names = z["names"].tolist()
    f = Frozen(names=names, emb=z["emb"].astype(np.float32), times=z["times"], starts=z["starts"], counts=z["counts"],
               durations=z["duration"].astype(np.float64), keep=np.ones(len(z["times"]), bool),
               cuts=[[] for _ in names])  # fmt: skip
    f.bank = didemo.bank(args)  # the DiDeMo query bank, unchanged
    index = {n: i for i, n in enumerate(names)}
    queries = [{**s, "video": index[s["video"]], "name": s["video"]} for s in steps(args.annotations)
               if s["video"] in index]  # fmt: skip
    path = CACHE / "youcook2_text.npy"
    if not path.exists():
        np.save(path, didemo.encode([q["text"] for q in queries], args.model_dir))
    text = np.load(path)
    db = sqlite3.connect(":memory:")
    db.executescript(speech.SCHEMA)
    said = json.loads(SPEECH.read_text())["videos"] if SPEECH.exists() else {}
    rows = [(index[n], a, b, s) for n, v in said.items() if n in index for a, b, s in v["segments"]]
    db.executemany("insert into speech (video_id, start, end, text) values (?, ?, ?, ?)", rows)
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    return f, queries, text, db, {m: tuned[m]["params"] for m in METHODS}, z


def measure(f: Frozen, queries: list, scores: np.ndarray, method: str, p: dict, which: list[int]) -> dict:
    """The protocol's metrics over the queries `which`; scores is frames x queries."""
    predict = PREDICT[method]
    single, rank, inside = [], {0.3: [], 0.5: []}, []
    for qi in which:
        q, col = queries[qi], scores[:, qi]
        top = predict(f, col, q["video"], p)
        single.append(iou(top[0][:2], q["moment"]) if top else 0.0)
        best = np.maximum.reduceat(col, f.starts)
        ranked = []
        for v in np.argsort(-best)[:SHORTLIST]:
            s, c = f.starts[v], f.counts[v]
            t, g = f.times[s : s + c], col[s : s + c]
            for a, b, peak in predict(f, col, int(v), p):
                near = np.flatnonzero((t >= a) & (t < b))
                near = near if len(near) else np.arange(len(t))
                ranked.append((int(v), a, b, peak, float(t[near[np.argmax(g[near])]])))
        ranked = sorted(ranked, key=lambda x: -x[3])[:10]
        for th in rank:
            hit = [r for r, (v, a, b, _, _) in enumerate(ranked) if v == q["video"] and iou((a, b), q["moment"]) >= th]
            rank[th].append(hit[0] if hit else None)
        lo, hi = q["moment"]
        hit = [r for r, (v, _, _, _, at) in enumerate(ranked) if v == q["video"] and lo <= at <= hi]
        inside.append(hit[0] if hit else None)
    single = np.array(single)
    out = {
        "single": {
            "R1 IoU>=0.3": float((single >= 0.3).mean()),
            "R1 IoU>=0.5": float((single >= 0.5).mean()),
            "R1 IoU>=0.7": float((single >= 0.7).mean()),
            "mIoU": float(single.mean()),
        },
        "corpus": {
            f"R@{k} IoU>={th}": float(np.mean([r is not None and r < k for r in rank[th]]))
            for th in rank
            for k in (1, 5, 10)
        },
        "corpus_inside": {"R@5": float(np.mean([r is not None and r < 5 for r in inside]))},
    }
    out["found"] = [r is not None and r < 5 for r in rank[0.5]]
    return out


def with_speech(f: Frozen, queries: list, visual: np.ndarray, db, cfg: speech.SpeechConfig, which) -> np.ndarray:
    ranges = {v: (int(f.starts[v]), int(f.counts[v])) for v in range(len(f.names))}
    out = visual.copy()
    for qi in which:
        hits = speech.segment_scores(db, queries[qi]["text"])
        out[:, qi] += cfg.weight * speech.frame_scores(f.times, ranges, hits, cfg)
    return out


def tune(args) -> None:
    f, queries, text, db, params, _ = load(args)
    tuning = split(f.names)
    which = [i for i, q in enumerate(queries) if q["name"] in tuning]
    visual = frame_scores(f, text, "frame_window_bank", params["frame_window_bank"])
    base = measure(f, queries, visual, "frame_window_bank", params["frame_window_bank"], which)
    rows = []
    for cfg in GRID:
        r = measure(f, queries, with_speech(f, queries, visual, db, cfg, which), "frame_window_bank",
                    params["frame_window_bank"], which)  # fmt: skip
        rows.append({"weight": cfg.weight, "pad": cfg.pad, "normalise": cfg.normalise,
                     "score": r["corpus"]["R@5 IoU>=0.5"]})  # fmt: skip
        print(f"{cfg}  R@5 {100 * rows[-1]['score']:.1f}", flush=True)
    best = max(rows, key=lambda r: r["score"])  # the first of equals, in the grid's order
    out = {"tuning_videos": len(tuning), "tuning_steps": len(which), "without_speech": base["corpus"]["R@5 IoU>=0.5"],
           "grid": rows, "best": best}  # fmt: skip
    (ROOT / "results" / "youcook2_tuned.json").write_text(json.dumps(out, indent=1))
    print(f"best: {best}, against {100 * out['without_speech']:.1f} without speech")


def evaluate(args) -> None:
    f, queries, text, db, params, z = load(args)
    tuning = split(f.names)
    which = [i for i, q in enumerate(queries) if q["name"] not in tuning]
    best = json.loads((ROOT / "results" / "youcook2_tuned.json").read_text())["best"]
    cfg = speech.SpeechConfig(best["weight"], best["pad"], best["normalise"])
    res = {"videos": len(f.names), "test_videos": len(f.names) - len(tuning), "test_steps": len(which),
           "hours": float(z["duration"].sum() / 3600), "speech": {"weight": cfg.weight, "pad": cfg.pad,
                                                                   "normalise": cfg.normalise}}  # fmt: skip
    for m in METHODS:
        res[m] = measure(f, queries, frame_scores(f, text, m, params[m]), m, params[m], which)
    visual = frame_scores(f, text, "frame_window_bank", params["frame_window_bank"])
    res["speech_bank"] = measure(f, queries, with_speech(f, queries, visual, db, cfg, which), "frame_window_bank",
                                 params["frame_window_bank"], which)  # fmt: skip
    a, b = np.array(res["speech_bank"].pop("found")), np.array(res["frame_window_bank"].pop("found"))
    res["whenframe_bank"].pop("found")
    res["speech_vs_default"] = {"wins": int((a & ~b).sum()), "losses": int((~a & b).sum()),
                                "p": sign_test(int((a & ~b).sum()), int((~a & b).sum()))}  # fmt: skip
    hours = res["hours"]
    said = json.loads(SPEECH.read_text())["videos"]
    said_hours = sum(v["duration"] for v in said.values()) / 3600
    res["cost"] = {"precision": str(z["precision"]), "frames_seen": int(z["seen"].sum()), "frames_kept": len(f.times),
                   "decode_cpu_min_per_hour": float(z["decode_cpu"].sum() / 60 / hours),
                   "encode_cpu_min_per_hour": float(z["encode_cpu"].sum() / 60 / hours),
                   "transcribe_cpu_min_per_hour": sum(v["cpu"] for v in said.values()) / 60 / said_hours}  # fmt: skip
    (ROOT / "results" / "youcook2_test.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in (*METHODS, "speech_bank")}, indent=1))
    print(json.dumps(res["cost"], indent=1), res["speech_vs_default"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "transcribe", "tune", "evaluate"])
    ap.add_argument("--videos", help="folder of ID.mp4 files")
    ap.add_argument("--annotations", required=True, help="lmms-eval/YouCook2 data/val-00000-of-00001.parquet")
    ap.add_argument("--train-annotations", help="DiDeMo training annotations, for the query bank")
    ap.add_argument("--model-dir", help="folder of SigLIP 2's files (default: the usual download cache)")
    ap.add_argument("--whisper-dir", help="folder of faster-whisper-base (default: downloaded once)")
    a = ap.parse_args()
    {"extract": extract, "transcribe": transcribe, "tune": tune, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
