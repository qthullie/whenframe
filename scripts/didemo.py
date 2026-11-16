"""DiDeMo: personal Flickr videos with moments described in plain English.

Each video is cut into 5-second segments and each description was located by
four annotators as a run of segments ("times": [[first, last], ...]). The
reference moment used here is the run most annotators chose. Annotators only
saw the first 30 seconds of each video (six segments, from second 0), so only
those are searched; the mp4 copies are often longer.

    python scripts/didemo.py extract --split test --videos DIR --annotations didemo_test.json
    python scripts/didemo.py tune    --split train
    python scripts/didemo.py evaluate --split test

The second round (docs/PROTOCOL.md) adds a second encoder and French descriptions:

    python scripts/didemo.py extract   --split test --model mobileclip-s2 --list results/didemo_test_videos.txt ...
    python scripts/didemo.py translate --split test --translator-dir DIR
    python scripts/didemo.py tune      --split train --round 2
    python scripts/didemo.py evaluate  --split test --round 2
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import (  # noqa: E402
    CORPUS_KS,
    ENSEMBLE,
    MAIN,
    METHODS,
    SECOND,
    Frozen,
    cost_curve,
    evaluate_all,
    grid_search,
    run,
)
from translate import Translator  # noqa: E402

from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames, thumbnails  # noqa: E402

CACHE = ROOT / "results" / "cache"
DATA = ROOT / "data" / "didemo"
SEGMENT = 5.0
SECONDS = 30.0  # what the annotators saw
BANK_SIZE = 4096


def reference(times: list[list[int]], duration: float) -> tuple[float, float]:
    (a, b), _ = Counter(map(tuple, times)).most_common(1)[0]
    return a * SEGMENT, min((b + 1) * SEGMENT, max(duration, a * SEGMENT + 1.0))


def scene_cuts(path: Path, threshold: float = 0.3) -> list[float]:
    """Shot boundaries in the first SECONDS, as ffmpeg's scene detector sees them at full frame rate."""
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-t", f"{SECONDS:g}", "-i", str(path), "-an", "-vf",
           f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]  # fmt: skip
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    return [float(tok.split(":")[1]) for line in err.splitlines() if "pts_time:" in line
            for tok in line.split() if tok.startswith("pts_time:")]  # fmt: skip


def stem(name: str) -> str:
    """Annotations keep the original file names (.mov, .avi...); the copies used here are all .mp4."""
    return Path(name).stem


def children_cpu() -> float:
    """CPU seconds used so far by finished child processes (ffmpeg)."""
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def video_names(args, ann) -> list[str]:
    """The videos of the split, or exactly those listed in --list, so a run can be repeated on the same set."""
    if args.list:
        return Path(args.list).read_text().split()
    on_disk = {f.stem: f.name for f in Path(args.videos).glob("*.mp4")}
    names = sorted({on_disk[stem(a["video"])] for a in ann if stem(a["video"]) in on_disk})
    return names[: args.limit] if args.limit else names


def suffix(model: str, backend: str = "onnxruntime") -> str:
    """How cache files are told apart: by model, and by runtime when it is not ONNX Runtime."""
    return ("" if model == MAIN else f"_{model}") + ("" if backend == "onnxruntime" else f"_{backend}")


def extract(args) -> None:
    ann = json.loads(Path(args.annotations).read_text())
    folder = Path(args.videos)
    names = video_names(args, ann)
    enc = make_encoder(args.model_dir, args.model, args.backend)
    out = CACHE / f"didemo_{args.split}{suffix(args.model, args.backend)}.npz"
    main = args.model == MAIN and args.backend == "onnxruntime"  # the run the other caches are compared with
    cols = {k: [] for k in ("emb", "times", "keep", "thumbs", "starts", "counts", "durations", "cuts")}
    cost = {k: [] for k in ("decode_cpu", "decode_wall", "encode_cpu", "encode_wall", "scene_cpu", "scene_wall")}
    done, row = [], 0
    t_all = time.time()
    for name in names:
        p = folder / name
        try:
            info = probe(p)
            c0, p0, w0 = children_cpu(), time.process_time(), time.time()
            t, frames = sample_frames(p, 1.0, enc.size, seconds=SECONDS)
            c1, p1, w1 = children_cpu(), time.process_time(), time.time()
            e = enc.encode_images(frames)
            p2, w2 = time.process_time(), time.time()
            cuts = scene_cuts(p) if main else []  # shots are only needed once
            c3, w3 = children_cpu(), time.time()
        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
            print(f"skip {name}: {ex}", file=sys.stderr)
            continue
        if len(frames) == 0:
            print(f"skip {name}: no frame", file=sys.stderr)
            continue
        keep = np.zeros(len(frames), dtype=bool)
        keep[drop_repeats(frames)] = True
        for k, v in (("emb", e.astype(np.float16)), ("times", t.astype(np.float32)), ("keep", keep),
                     ("thumbs", thumbnails(frames).astype(np.float16)), ("starts", row), ("counts", len(frames)),
                     ("durations", min(info.duration or len(frames), SECONDS)), ("cuts", cuts)):  # fmt: skip
            cols[k].append(v)
        for k, v in (("decode_cpu", c1 - c0 + p1 - p0), ("decode_wall", w1 - w0), ("encode_cpu", p2 - p1),
                     ("encode_wall", w2 - w1), ("scene_cpu", c3 - c1), ("scene_wall", w3 - w2)):  # fmt: skip
            cost[k].append(v)
        done.append(name)
        row += len(frames)
        if len(done) % 50 == 0:
            print(f"{len(done)}/{len(names)} videos, {row} frames, {time.time() - t_all:.0f} s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, emb=np.concatenate(cols["emb"]), times=np.concatenate(cols["times"]), keep=np.concatenate(cols["keep"]),
        thumbs=np.concatenate(cols["thumbs"]), starts=np.array(cols["starts"]), counts=np.array(cols["counts"]),
        durations=np.array(cols["durations"], dtype=np.float32), names=np.array(done),
        cuts=np.array(json.dumps(cols["cuts"])), model=np.array(enc.name),
        **{k: np.array(v) for k, v in cost.items()},
    )  # fmt: skip
    if main:
        (ROOT / "results" / f"didemo_{args.split}_videos.txt").write_text("\n".join(done) + "\n")
    print(f"wrote {out}: {len(done)} videos, {row} frames, {time.time() - t_all:.0f} s")


def make_encoder(model_dir: str | None, model: str = MAIN, backend: str = "onnxruntime") -> Encoder:
    """--model-dir points at the main model's files; other models come from the usual cache."""
    folder = model_dir if model_dir and model == MAIN else None
    return Encoder(model, folder=folder, backend=backend)


def load(split: str, annotations: str | None = None, extra: tuple[str, ...] = ()):
    z = np.load(CACHE / f"didemo_{split}.npz")
    names = z["names"].tolist()
    decoded = Frozen(
        names=names, emb=z["emb"].astype(np.float32), times=z["times"], starts=z["starts"], counts=z["counts"],
        durations=z["durations"], keep=z["keep"], cuts=json.loads(str(z["cuts"])), thumbs=z["thumbs"],
    )  # fmt: skip
    # caches made before videos were cut at extraction hold whole videos: keep what the annotators saw
    frozen = decoded.select(decoded.times < SECONDS)
    frozen.durations = np.minimum(frozen.durations, SECONDS)
    ann = json.loads(Path(annotations or DATA / f"didemo_{split}.json").read_text())
    index = {stem(n): i for i, n in enumerate(names)}
    queries = []
    for a in ann:
        if stem(a["video"]) in index:
            v = index[stem(a["video"])]
            queries.append({"text": a["description"], "video": v, "moment": reference(a["times"], frozen.durations[v])})
    timing = ("decode_cpu", "decode_wall", "encode_cpu", "encode_wall", "scene_cpu", "scene_wall")
    cost = {  # what was decoded and encoded, which the timings cover
        "videos": len(names),
        "hours": float(decoded.durations.sum() / 3600),
        "frames": int(len(decoded.times)),
        "kept_fraction": float(frozen.keep.mean()),
        **{k: float(z[k].sum()) for k in timing},
    }
    for model in extra:  # the same frames, embedded by another encoder
        x = np.load(CACHE / f"didemo_{split}{suffix(model)}.npz")
        assert x["names"].tolist() == names, f"{model}: not the same videos"
        mask = x["times"] < SECONDS
        assert np.array_equal(x["times"][mask], frozen.times), f"{model}: not the same frames"
        frozen.extra[model] = x["emb"][mask].astype(np.float32)
        cost[model] = {"hours": float(x["durations"].sum() / 3600), "frames": int(len(x["times"])),
                       **{k: float(x[k].sum()) for k in timing}}  # fmt: skip
    return frozen, queries, cost


def encode(texts: list[str], model_dir: str | None, model: str = MAIN) -> np.ndarray:
    enc = make_encoder(model_dir, model)
    return np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)]).astype(np.float32)


def text_embeddings(queries, split: str, model_dir: str | None, model: str = MAIN, lang: str = "") -> np.ndarray:
    """The descriptions' embeddings; lang "fr" uses their French translations (the translate stage)."""
    path = CACHE / f"didemo_{split}_text{'_' + lang if lang else ''}{suffix(model)}.npy"
    if not path.exists():
        texts = [q["text"] for q in queries]
        if lang:
            texts = json.loads((CACHE / f"didemo_{split}_{lang}.json").read_text())
            assert len(texts) == len(queries), "translations do not match the descriptions"
        np.save(path, encode(texts, model_dir, model))
    return np.load(path)


def bank_texts(args, size: int | None = BANK_SIZE) -> list[str]:
    """Descriptions of training videos outside the tuning set: what people search for, in general.

    The same bank serves tuning and test, and shares no video with either. size=None returns all of them.
    """
    train = json.loads(Path(args.train_annotations or DATA / "didemo_train.json").read_text())
    tuning = {stem(n) for n in (ROOT / "results" / "didemo_train_videos.txt").read_text().split()}
    texts = sorted({a["description"].strip() for a in train if stem(a["video"]) not in tuning})
    return random.Random(0).sample(texts, size) if size else texts


def bank(args, model: str = MAIN) -> np.ndarray:
    path = CACHE / f"didemo_bank_text{suffix(model)}.npy"
    if not path.exists():
        np.save(path, encode(bank_texts(args), args.model_dir, model))
    return np.load(path)


def text_latency(queries, model_dir: str | None, n: int = 50, model: str = MAIN) -> float:
    """Milliseconds to embed one query, as the tool does for each search."""
    enc = make_encoder(model_dir, model)
    enc.encode_texts([queries[0]["text"]])  # loads the text tower
    t0 = time.perf_counter()
    for q in queries[:n]:
        enc.encode_texts([q["text"]])
    return 1000 * (time.perf_counter() - t0) / n


def prepare(args):
    """The split, its query embeddings (one set per model in the second round) and the query banks."""
    extra = tuple(SECOND.values()) if args.round == 2 else ()
    frozen, queries, cost = load(args.split, args.annotations, extra)
    frozen.bank = bank(args)
    text = text_embeddings(queries, args.split, args.model_dir)
    if extra:
        text = {MAIN: text, **{m: text_embeddings(queries, args.split, args.model_dir, m) for m in extra}}
        for m in extra:
            frozen.extra_banks[m] = bank(args, m)
    return frozen, queries, cost, text


def tuned_path(args) -> Path:
    return ROOT / "results" / ("didemo_tuned.json" if args.round == 1 else f"didemo_tuned_round{args.round}.json")


def tune(args) -> None:
    frozen, queries, _, text = prepare(args)
    best = grid_search(frozen, queries, text)
    best["queries"] = len(queries)
    best["videos"] = len(frozen.names)
    tuned_path(args).write_text(json.dumps(best, indent=1))
    if args.round == 1:
        curve = cost_curve(frozen, queries, text, best["default"], best[best["default"]]["params"])
        (ROOT / "results" / f"didemo_{args.split}_curve.json").write_text(json.dumps(curve, indent=1))
    print(json.dumps(best, indent=1))


def evaluate(args) -> None:
    frozen, queries, cost, text = prepare(args)
    tuned = json.loads(tuned_path(args).read_text())
    methods = [m for m in METHODS if m in tuned]
    res = evaluate_all(frozen, queries, text, tuned, methods)
    res["curve"] = cost_curve(frozen, queries, text, tuned["default"], tuned[tuned["default"]]["params"])
    cost["text_ms"] = text_latency(queries, args.model_dir)
    if args.round == 2:
        for m in SECOND.values():
            cost[m]["text_ms"] = text_latency(queries, args.model_dir, model=m)
        # the same searches with the descriptions machine-translated into French
        fr = {m: text_embeddings(queries, args.split, args.model_dir, m, lang="fr") for m in (MAIN, *SECOND.values())}
        res["french"] = {m: run(frozen, queries, fr, m, tuned[m]["params"])
                         for m in ("frame_window_bank", *SECOND, *ENSEMBLE)}  # fmt: skip
    res.update(cost=cost, queries=len(queries), corpus_ks=CORPUS_KS, methods=methods, default=tuned["default"])
    name = f"didemo_{args.split}.json" if args.round == 1 else f"didemo_{args.split}_round{args.round}.json"
    (ROOT / "results" / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({m: res[m]["corpus"] for m in methods}, indent=1))


def translate(args) -> None:
    """French translations of the split's descriptions, in the order load() returns them."""
    _, queries, _ = load(args.split, args.annotations)
    out = CACHE / f"didemo_{args.split}_fr.json"
    t0 = time.time()
    french = Translator(args.translator_dir)([q["text"] for q in queries])
    out.write_text(json.dumps(french, ensure_ascii=False, indent=0))
    print(f"wrote {out}: {len(french)} descriptions in {time.time() - t0:.0f} s")
    for q, f in list(zip(queries, french, strict=True))[:5]:
        print(f"  {q['text']!r} -> {f!r}")


def calibrate(args) -> None:
    """Scale and bias of a second model's displayed confidence, sigmoid(a * cos + b).

    Fitted by least squares so that a * cos + b matches the main model's own logit
    (SigLIP's sigmoid) on tuning pairs: for each description, the frames inside its
    annotated moment and as many frames of other videos. The tool then shows
    confidences on one scale whatever the mode; no ranking depends on them.
    """
    model = args.model
    assert model != MAIN, "the main model's probability is its own sigmoid"
    frozen, queries, _ = load(args.split, args.annotations, (model,))
    text = text_embeddings(queries, args.split, args.model_dir)
    text2 = text_embeddings(queries, args.split, args.model_dir, model)
    rng = np.random.default_rng(0)
    video_of = np.repeat(np.arange(len(frozen.starts)), frozen.counts)
    spec = make_encoder(args.model_dir).spec
    x, target = [], []
    for qi, q in enumerate(queries):
        s, c = frozen.starts[q["video"]], frozen.counts[q["video"]]
        t = frozen.times[s : s + c]
        inside = s + np.flatnonzero((t >= q["moment"][0]) & (t < q["moment"][1]))
        rows = np.concatenate([inside, rng.choice(np.flatnonzero(video_of != q["video"]), len(inside), replace=False)])
        x.append(frozen.extra[model][rows] @ text2[qi])
        target.append(spec.logit_scale * (frozen.emb[rows] @ text[qi]) + spec.logit_bias)
    x, target = np.concatenate(x).astype(np.float64), np.concatenate(target)
    a, b = np.polyfit(x, target, 1)
    r = float(np.corrcoef(a * x + b, target)[0, 1])
    out = ROOT / "results" / "didemo_calibration.json"
    out.write_text(json.dumps({model: {"scale": float(a), "bias": float(b), "pairs": int(len(x)), "fitted_to": MAIN,
                                       "correlation": r}}, indent=1))  # fmt: skip
    print(model, json.loads(out.read_text())[model])


def main_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "tune", "evaluate", "translate", "calibrate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--videos")
    ap.add_argument("--annotations", help="the split's annotations (default data/didemo/didemo_SPLIT.json)")
    ap.add_argument("--train-annotations", help="training annotations, for the query bank")
    ap.add_argument("--model-dir", help="folder of the main model's files (default: the usual download cache)")
    ap.add_argument("--model", default=MAIN, help="encoder to extract with, or to calibrate")
    ap.add_argument("--backend", default="onnxruntime", choices=["onnxruntime", "openvino"],
                    help="runtime of the image encoder at extraction (the first two rounds: onnxruntime)")  # fmt: skip
    ap.add_argument("--round", type=int, default=1, choices=[1, 2], help="2: the second round of the protocol")
    ap.add_argument("--translator-dir", help="folder of the Opus-MT English-French export, for translate")
    ap.add_argument("--limit", type=int, default=0, help="keep the first N videos, in name order")
    ap.add_argument("--list", help="a file naming the videos to use, one per line (overrides --limit)")
    return ap.parse_args(argv)


def main() -> None:
    a = main_args()
    {"extract": extract, "tune": tune, "evaluate": evaluate, "translate": translate, "calibrate": calibrate}[a.stage](a)


if __name__ == "__main__":
    main()
