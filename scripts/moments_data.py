"""More described moments to train whenframe's model v3, and So400m's view of them (docs/PROTOCOL.md, seventh round).

Two collections whose training descriptions give the moment's start and end in seconds:

- Charades-STA (Gao et al., ICCV 2017): 12,408 sentences locating moments in
  5,338 of Charades' training videos (Sigurdsson et al., ECCV 2016), from the
  copy of the annotations kept with 2D-TAN (microsoft/VideoX). The videos are
  read from the full Charades archive on the Hugging Face Hub
  (lmms-eval/charades_sta, four zip files) by byte ranges, one at a time, and
  not kept. 500 of the training videos, drawn once (seed 0), are held out to
  choose the model's settings, as DiDeMo's 500 tuning videos are; none of
  Charades-STA's 1,334 test videos is a training video.
- YouCook2 (Zhou et al., AAAI 2018): the 8,815 steps of the 1,139 training
  videos of the copy nguyenminh04/youcook2-videos (1,106 of them there),
  downloaded one at a time and not kept; none is one of the benchmark's 404
  validation videos, nor a UnAV-100 video.

Their frames are decoded at one frame a second and embedded by SigLIP 2 base
with OpenVINO, as the tool indexes (Charades: every frame, repeats marked, as
DiDeMo's training videos; YouCook2: repeats skipped, as the benchmark's).

The teacher: SigLIP 2 So400m (the large mode) embeds every second frame of the
training videos, at the instants of the base model's frames, so that whenframe's
model can learn to score frames as So400m does (docs/PROTOCOL.md). Charades-STA's
are embedded with the base model, in the same pass; DiDeMo's and MSR-VTT's by
reading their archives again (`teacher`), if the teacher helps on the tuning sets.

    python scripts/moments_data.py charades --annotations charades_sta_train.txt
    python scripts/moments_data.py youcook2 --annotations youcookii_train.jsonl
    python scripts/moments_data.py texts    --charades charades_sta_train.txt --youcook2 youcookii_train.jsonl \
        --tempo tempoHL+didemo_train.json
    python scripts/moments_data.py teacher  --dataset didemo --annotations didemo_train.json
    python scripts/moments_data.py teacher  --dataset msrvtt --annotations msrvtt_train_9k.json

Each stage resumes where it stopped. Writes results/cache/charades_train_openvino.npz (the training videos),
charades_tuning_openvino.npz (the 500 held out), youcook2_train_openvino.npz, *_teacher.npz, the descriptions'
embeddings and results/charades_tuning_videos.txt.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import queue
import random
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
HUB = "https://huggingface.co/datasets"
CHARADES_ZIPS = [f"{HUB}/lmms-eval/charades_sta/resolve/main/Charades_v1_480_part_{i}.zip" for i in range(1, 5)]
YOUCOOK2 = f"{HUB}/nguyenminh04/youcook2-videos/resolve/main/videos/{{}}"
TUNING = ROOT / "results" / "charades_tuning_videos.txt"
HELD_OUT = 500  # Charades-STA training videos kept to choose the settings
EVERY = 2  # the teacher embeds one frame in EVERY
TEACHER = "siglip2-so400m"
DONE = object()


# ---------------------------------------------------------------- the descriptions
def charades_annotations(path: str) -> list[dict]:
    """Charades-STA's lines, `video start end##sentence`."""
    out = []
    for line in Path(path).read_text().splitlines():
        if "##" not in line:
            continue
        head, text = line.split("##", 1)
        video, start, end = head.split()
        out.append({"video": video, "text": text.strip(), "seconds": [float(start), float(end)]})
    return out


def charades_tuning(names: list[str]) -> list[str]:
    """The training videos held out to choose the settings: 500, drawn once."""
    if not TUNING.exists():
        TUNING.write_text("\n".join(sorted(random.Random(0).sample(sorted(set(names)), HELD_OUT))) + "\n")
    return TUNING.read_text().split()


def youcook2_annotations(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            out.append({"video": Path(s["video"]).stem, "text": s["text"].strip(), "seconds": list(s["timestamps"])})
    return out


def tempo_annotations(path: str, didemo: list[dict]) -> list[dict]:
    """TEMPO-HL's descriptions of DiDeMo's training videos (Hendricks et al., EMNLP 2018) that are not DiDeMo's own:
    moments told by what happens before, after or while another (DiDeMo's 5-second segments)."""
    own = {(q["video"], q["text"]) for q in didemo}
    out = []
    for a in json.loads(Path(path).read_text()):
        video = Path(a["video"]).stem
        if not a["annotation_id"].isdigit() and (video, a["description"]) not in own:
            out.append({"video": video, "text": a["description"].strip(), "times": a["times"]})
    return out


# ---------------------------------------------------------------- reading the videos
def charades_videos(names: list[str], work: Path):
    """Each wanted video of the Charades archive, written to `work`, in the archive's order."""
    from msrvtt import Remote

    wanted, index = {f"{n}.mp4" for n in names}, []
    for url in CHARADES_ZIPS:
        z = zipfile.ZipFile(io.BufferedReader(Remote(url), buffer_size=1 << 22))
        index += [(z, info) for info in z.infolist() if Path(info.filename).name in wanted]
    for z, info in index:
        name = Path(info.filename).stem
        path = work / f"{name}.mp4"  # named after the annotations, never the archive's own path
        for attempt in range(5):
            try:
                with z.open(info) as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
            except (OSError, zipfile.BadZipFile) as ex:
                if attempt == 4:
                    print(f"skip {name}: {ex}", file=sys.stderr, flush=True)
                    path = None
                else:
                    time.sleep(10 * (attempt + 1))
        if path is not None:
            yield name, path


def youcook2_videos(names: list[str], work: Path):
    for name in names:
        path = work / f"{name}.mp4"
        for attempt in range(5):
            try:
                url = YOUCOOK2.format(f"{name}.mp4")
                with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as dst:  # noqa: S310 (fixed host)
                    shutil.copyfileobj(r, dst)
                break
            except (OSError, http.client.HTTPException) as ex:
                if attempt == 4 or getattr(ex, "code", None) == 404:
                    print(f"skip {name}: {ex}", file=sys.stderr, flush=True)
                    path = None
                    break
                time.sleep(10 * (attempt + 1))
        if path is not None:
            yield name, path


def pipeline(source, handle, decoders: int = 2, seconds: float | None = None) -> int:
    """Videos read by one thread, decoded by `decoders` threads, handled (embedded, written) by this one. If the
    reading fails, so does this, once the videos read so far are handled: nothing is assembled from a part."""
    todo, ready = queue.Queue(maxsize=6), queue.Queue(maxsize=6)
    failed = []

    def read():
        try:
            for item in source:
                todo.put(item)
        except Exception as ex:  # noqa: BLE001 - raised again in the caller's thread, below
            failed.append(ex)
        finally:
            todo.put(DONE)

    def decode():
        while (item := todo.get()) is not DONE:
            name, path = item
            try:
                duration = probe(path).duration or 0.0
                t, frames = sample_frames(path, 1.0, 224, seconds=seconds)
                ready.put((name, t, frames, duration, None))
            except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
                ready.put((name, None, None, 0.0, ex))
            finally:
                path.unlink(missing_ok=True)
        todo.put(DONE)  # for the other decoders
        ready.put(DONE)

    threading.Thread(target=read, daemon=True).start()
    for _ in range(decoders):
        threading.Thread(target=decode, daemon=True).start()
    finished, n, t0 = 0, 0, time.time()
    while finished < decoders:
        item = ready.get()
        if item is DONE:
            finished += 1
            continue
        name, t, frames, duration, err = item
        if err is not None or not len(frames):
            print(f"skip {name}: {err or 'no frame'}", file=sys.stderr, flush=True)
            continue
        handle(name, t, frames, duration)
        n += 1
        if n % 100 == 0:
            print(f"{n} videos this run, {time.time() - t0:.0f} s", flush=True)
    if failed:  # the videos handled are kept: run the stage again to go on
        raise RuntimeError(f"reading the videos failed after {n} this run") from failed[0]
    return n


def save(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)  # a video is either done or not started


def assemble(parts: Path, out: Path, names: list[str] | None = None) -> None:
    """The videos embedded so far, as the other caches lay them out; teacher rows indexed into the whole."""
    names = names or sorted(p.stem for p in parts.glob("*.npz") if not p.stem.endswith(".tmp"))
    names = [n for n in names if (parts / f"{n}.npz").exists()]
    z = [np.load(parts / f"{n}.npz") for n in names]
    counts = np.array([len(x["times"]) for x in z])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    arrays = {"names": np.array(names), "emb": np.concatenate([x["emb"] for x in z]),
              "times": np.concatenate([x["times"] for x in z]), "keep": np.concatenate([x["keep"] for x in z]),
              "counts": counts, "starts": starts,
              "durations": np.array([float(x["duration"]) for x in z], dtype=np.float32)}  # fmt: skip
    taught = [i for i, x in enumerate(z) if "teacher" in x.files]
    if taught:
        arrays["teacher"] = np.concatenate([z[i]["teacher"] for i in taught])
        arrays["teacher_rows"] = np.concatenate([starts[i] + z[i]["teacher_rows"] for i in taught])
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)
    extra = f", {len(arrays.get('teacher', []))} frames seen by the teacher" if taught else ""
    print(f"wrote {out.name}: {len(names)} videos, {counts.sum()} frames{extra}", flush=True)


# ---------------------------------------------------------------- stages
def charades(args) -> None:
    ann = charades_annotations(args.annotations)
    names = sorted({a["video"] for a in ann})
    tuning = set(charades_tuning(names))
    parts = CACHE / "charades_train_parts"
    parts.mkdir(parents=True, exist_ok=True)
    base = Encoder("siglip2-base", folder=args.model_dir, backend="openvino")
    teacher = Encoder(TEACHER, backend="openvino")
    base.images, teacher.images  # noqa: B018 - compiled before the first video
    todo = [n for n in names if not (parts / f"{n}.npz").exists()][: args.limit or None]
    work = Path(tempfile.mkdtemp(prefix="charades-", dir=args.work))

    def handle(name, t, frames, duration):
        keep = np.zeros(len(frames), dtype=bool)
        keep[drop_repeats(frames)] = True
        out = {"emb": base.encode_images(frames).astype(np.float16), "times": t.astype(np.float32), "keep": keep,
               "duration": duration or float(len(frames))}  # fmt: skip
        if name not in tuning:  # the held-out videos are searched, never taught
            rows = np.arange(0, len(frames), EVERY)
            out |= {"teacher": teacher.encode_images(frames[rows]).astype(np.float16), "teacher_rows": rows}
        save(parts / f"{name}.npz", **out)

    try:
        pipeline(charades_videos(todo, work), handle)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not args.limit:
        assemble(parts, CACHE / "charades_train_openvino.npz", sorted(set(names) - tuning))
        assemble(parts, CACHE / "charades_tuning_openvino.npz", sorted(tuning))


def youcook2(args) -> None:
    names = sorted({s["video"] for s in youcook2_annotations(args.annotations)})
    parts = CACHE / "youcook2_train_parts"
    parts.mkdir(parents=True, exist_ok=True)
    base = Encoder("siglip2-base", folder=args.model_dir, backend="openvino")
    base.images  # noqa: B018
    todo = [n for n in names if not (parts / f"{n}.npz").exists()][: args.limit or None]
    work = Path(tempfile.mkdtemp(prefix="youcook2-", dir=args.work))

    def handle(name, t, frames, duration):
        kept = drop_repeats(frames)  # as the tool indexes: repeated frames are not encoded
        save(parts / f"{name}.npz", emb=base.encode_images(frames[kept]).astype(np.float16),
             times=t[kept].astype(np.float32), keep=np.ones(len(kept), bool),
             duration=duration or float(len(frames)))  # fmt: skip

    try:
        pipeline(youcook2_videos(todo, work), handle)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not args.limit:
        assemble(parts, CACHE / "youcook2_train_openvino.npz")


def teacher(args) -> None:
    """So400m's embeddings of every second frame of DiDeMo's or MSR-VTT's training videos, at the instants of the
    base model's frames (the caches of the fifth and sixth rounds), their archives read again."""
    cache = {"didemo": "didemo_all_openvino", "msrvtt": "msrvtt_openvino"}[args.dataset]
    z = np.load(CACHE / f"{cache}.npz")
    names, times = z["names"].tolist(), z["times"]  # read once: each z["times"] reads a copy, which a slice keeps
    want = {n: times[s : s + c] for n, s, c in zip(names, z["starts"], z["counts"], strict=True)}
    parts = CACHE / f"{args.dataset}_teacher_parts"
    parts.mkdir(parents=True, exist_ok=True)
    enc = Encoder(TEACHER, backend="openvino")
    enc.images  # noqa: B018
    work = Path(tempfile.mkdtemp(prefix="teacher-", dir=args.work))
    left = {n for n in names if not (parts / f"{n}.npz").exists()}

    def handle(name, t, frames, duration):
        at = np.flatnonzero(np.isin(np.round(t, 3), np.round(want[name], 3)))
        if len(at) != len(want[name]):
            print(f"skip {name}: not the same instants ({len(at)} of {len(want[name])})", file=sys.stderr)
            return
        rows = np.arange(0, len(at), EVERY)
        save(parts / f"{name}.npz", teacher=enc.encode_images(frames[at[rows]]).astype(np.float16), teacher_rows=rows)

    if args.dataset == "didemo":
        source, seconds = didemo_stream(left, work), 30.0
    else:
        source, seconds = msrvtt_stream(left, work), None
    try:
        pipeline(source, handle, seconds=seconds)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    done = [n for n in names if (parts / f"{n}.npz").exists()]
    starts = dict(zip(names, z["starts"], strict=True))
    rows = [np.load(parts / f"{n}.npz") for n in done]
    out = CACHE / f"{args.dataset}_teacher.npz"
    np.savez(
        out.with_name(out.stem + ".tmp.npz"),
        teacher=np.concatenate([r["teacher"] for r in rows]),
        teacher_rows=np.concatenate([starts[n] + r["teacher_rows"] for n, r in zip(done, rows, strict=True)]),
    )
    out.with_name(out.stem + ".tmp.npz").replace(out)
    print(f"wrote {out.name}: {len(done)} of {len(names)} videos", flush=True)


def didemo_stream(wanted: set[str], work: Path):
    """DiDeMo's training archive read from the start, the wanted videos written to `work` one at a time."""
    import tarfile

    from train_data import Archive

    stream = io.BufferedReader(Archive(0), buffer_size=1 << 22)
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            name = Path(member.name).stem
            if member.isfile() and name in wanted:
                path = work / f"{name}.mp4"
                with tar.extractfile(member) as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                yield name, path


def msrvtt_stream(wanted: set[str], work: Path):
    from msrvtt import URL, Remote

    with zipfile.ZipFile(io.BufferedReader(Remote(URL), buffer_size=1 << 22)) as z:
        for info in z.infolist():
            name = Path(info.filename).stem
            if not info.is_dir() and name in wanted:
                path = work / f"{name}.mp4"
                with z.open(info) as src, open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                yield name, path


def texts(args) -> None:
    """The descriptions, embedded as the tool embeds a query (SigLIP 2 base) and as the teacher reads them (So400m,
    lower case as the fourth round chose), in blocks so that a stopped run resumes. Each collection's descriptions
    are kept in results/cache/NAME_queries.json, their embeddings in NAME_text.npy and NAME_text_so400m.npy."""

    def write(name: str, queries: list[dict], teacher: bool) -> None:
        (CACHE / f"{name}_queries.json").write_text(json.dumps(queries))
        encode_blocks("siglip2-base", [q["text"] for q in queries], CACHE / f"{name}_text.npy", args.model_dir)
        if teacher:
            encode_blocks(TEACHER, [q["text"] for q in queries], CACHE / f"{name}_text_so400m.npy")

    if args.charades:
        ann = charades_annotations(args.charades)
        tuning = set(charades_tuning([a["video"] for a in ann]))
        write("charades_train", [a for a in ann if a["video"] not in tuning], teacher=True)
        write("charades_tuning", [a for a in ann if a["video"] in tuning], teacher=False)
    if args.youcook2:
        write("youcook2_train", youcook2_annotations(args.youcook2), teacher=False)
    didemo = json.loads((CACHE / "didemo_all_queries.json").read_text())
    if args.tempo:
        embedded = set(np.load(CACHE / "didemo_all_openvino.npz")["names"].tolist())  # the tuning videos are not
        write("tempo", [q for q in tempo_annotations(args.tempo, didemo) if q["video"] in embedded], teacher=True)
    for name in (args.teacher_texts or "").split(","):
        if name:  # the descriptions of the fifth and sixth rounds' training clips, as the teacher reads them
            queries = json.loads((CACHE / f"{name}_queries.json").read_text())
            encode_blocks(TEACHER, [q["text"] for q in queries], CACHE / f"{name}_text_so400m.npy")


def encode_blocks(model: str, texts_: list[str], out: Path, folder: str | None = None, block: int = 2048) -> None:
    if out.exists():
        return
    parts = out.with_name(out.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    backend = "openvino" if model == TEACHER else None
    enc, t0 = Encoder(model, folder=folder, backend=backend), time.time()
    for start in range(0, len(texts_), block):
        part = parts / f"{start:06d}.npy"
        if part.exists():
            continue
        chunk = texts_[start : start + block]
        emb = np.concatenate([enc.encode_texts(chunk[i : i + 64]) for i in range(0, len(chunk), 64)])
        np.save(parts / f"{start:06d}.tmp.npy", emb.astype(np.float16))
        (parts / f"{start:06d}.tmp.npy").replace(part)
        print(f"{out.name}: {start + len(chunk)}/{len(texts_)} descriptions, {time.time() - t0:.0f} s", flush=True)
    emb = np.concatenate([np.load(parts / f"{s:06d}.npy") for s in range(0, len(texts_), block)])
    np.save(out, emb)
    shutil.rmtree(parts)
    print(f"wrote {out.name}: {len(emb)} descriptions", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["charades", "youcook2", "teacher", "texts"])
    ap.add_argument("--annotations", help="charades_sta_train.txt, youcookii_train.jsonl, didemo_train.json, ...")
    ap.add_argument("--dataset", choices=["didemo", "msrvtt"], help="teacher: whose training videos")
    ap.add_argument("--charades", help="texts: charades_sta_train.txt")
    ap.add_argument("--youcook2", help="texts: youcookii_train.jsonl")
    ap.add_argument("--tempo", help="texts: TEMPO-HL's tempoHL+didemo_train.json")
    ap.add_argument("--teacher-texts", help="texts: didemo_all,msrvtt: those descriptions as the teacher reads them")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--work", help="where each video is written while it is decoded (default: the system's)")
    ap.add_argument("--limit", type=int, help="charades, youcook2: only this many videos, not assembled (a check)")
    args = ap.parse_args()
    {"charades": charades, "youcook2": youcook2, "teacher": teacher, "texts": texts}[args.stage](args)


if __name__ == "__main__":
    main()
