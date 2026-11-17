"""MSR-VTT's training clips, embedded to train whenframe's model v2 (docs/PROTOCOL.md, sixth round).

MSR-VTT (Xu et al., CVPR 2016): 10,000 YouTube clips of 10 to 30 seconds in 20
categories, each described by 20 sentences. The 9,000 training clips are read
from the archive of friedrichor/MSR-VTT on the Hugging Face Hub without
downloading it: the zip file's directory, then each clip, are fetched by byte
ranges, written to a temporary file, decoded at one frame a second, embedded by
SigLIP 2 base with OpenVINO (repeats marked, as the tool skips them) and
deleted. Clips cut from a YouTube video that is also one of YouCook2's or
UnAV-100's test or tuning videos are left out.

    python scripts/msrvtt.py extract --annotations msrvtt_train_9k.json --model-dir ...
    python scripts/msrvtt.py texts   --annotations msrvtt_train_9k.json --model-dir ...

Writes results/cache/msrvtt_openvino.npz, msrvtt_text.npy and msrvtt_queries.json.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import queue
import random
import re
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

from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

URL = "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip"
CACHE = ROOT / "results" / "cache"
PARTS = CACHE / "msrvtt_parts"
OUT = CACHE / "msrvtt_openvino.npz"
TEXTS = CACHE / "msrvtt_text.npy"
QUERIES = CACHE / "msrvtt_queries.json"
PER_CLIP = 5  # descriptions kept of each clip's twenty
DONE = object()


class Remote(io.RawIOBase):
    """A file on the web, read by byte ranges: enough for zipfile to find and read members."""

    def __init__(self, url: str):
        self.url, self.position = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed host)
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (fixed host)
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def youtube_id(url: str) -> str | None:
    m = re.search(r"v=([\w-]{11})", url)
    return m.group(1) if m else None


def excluded() -> set[str]:
    """YouTube identifiers of YouCook2's and UnAV-100's videos, from their caches' names."""
    out = set()
    for name in ("youcook2.npz", "unav.npz"):
        f = CACHE / name
        if f.exists():
            out |= {str(n)[:11] for n in np.load(f)["names"]}
    return out


def clips(path: str) -> list[dict]:
    """The training clips kept, each with its YouTube identifier and its descriptions."""
    skip = excluded()
    out = []
    for e in json.loads(Path(path).read_text()):
        if youtube_id(e.get("url", "")) in skip:
            continue
        captions = e["caption"]
        if isinstance(captions, str):
            captions = ast.literal_eval(captions)  # a list written as text in this copy of the annotations
        out.append({"name": Path(e["video"]).stem, "captions": [c.strip() for c in captions if c.strip()]})
    return out


def read(names: set[str], work: Path, todo: queue.Queue) -> None:
    with zipfile.ZipFile(io.BufferedReader(Remote(URL), buffer_size=1 << 22)) as z:
        for info in z.infolist():
            name = Path(info.filename).stem
            if info.is_dir() or name not in names or (PARTS / f"{name}.npz").exists():
                continue
            path = work / f"{name}.mp4"  # named after the annotations, never the archive's own path
            with z.open(info) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            todo.put((name, path))
    todo.put(DONE)


def decode(todo: queue.Queue, ready: queue.Queue) -> None:
    while (item := todo.get()) is not DONE:
        name, path = item
        try:
            duration = probe(path).duration or 0.0
            t, frames = sample_frames(path, 1.0, 224)
            ready.put((name, t, frames, duration, None))
        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
            ready.put((name, None, None, 0.0, ex))
        finally:
            path.unlink(missing_ok=True)
    todo.put(DONE)  # for the other decoder
    ready.put(DONE)


def extract(args) -> None:
    kept = clips(args.annotations)
    PARTS.mkdir(parents=True, exist_ok=True)
    enc = Encoder("siglip2-base", folder=args.model_dir, backend="openvino")
    enc.images  # noqa: B018 - compiled before the first clip
    work = Path(tempfile.mkdtemp(prefix="msrvtt-", dir=args.work))
    todo, ready = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
    threading.Thread(target=read, args=({c["name"] for c in kept}, work, todo), daemon=True).start()
    decoders = [threading.Thread(target=decode, args=(todo, ready), daemon=True) for _ in range(2)]
    for d in decoders:
        d.start()
    finished, n, t0 = 0, 0, time.time()
    while finished < len(decoders):
        item = ready.get()
        if item is DONE:
            finished += 1
            continue
        name, t, frames, duration, err = item
        if err is None and len(frames):
            keep = np.zeros(len(frames), dtype=bool)
            keep[drop_repeats(frames)] = True
            emb = enc.encode_images(frames)
            tmp = PARTS / f"{name}.tmp.npz"
            np.savez(tmp, emb=emb.astype(np.float16), times=t.astype(np.float32), keep=keep,
                     duration=duration or float(len(frames)))  # fmt: skip
            tmp.replace(PARTS / f"{name}.npz")
        else:
            print(f"skip {name}: {err or 'no frame'}", file=sys.stderr, flush=True)
        n += 1
        if n % 250 == 0:
            print(f"{n} clips this run, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    assemble()


def assemble() -> None:
    names = sorted(p.stem for p in PARTS.glob("*.npz") if not p.stem.endswith(".tmp"))
    z = [np.load(PARTS / f"{n}.npz") for n in names]
    counts = np.array([len(x["times"]) for x in z])
    tmp = OUT.with_name(OUT.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, names=np.array(names), emb=np.concatenate([x["emb"] for x in z]),
        times=np.concatenate([x["times"] for x in z]), keep=np.concatenate([x["keep"] for x in z]),
        counts=counts, starts=np.concatenate([[0], np.cumsum(counts)[:-1]]),
        durations=np.array([float(x["duration"]) for x in z], dtype=np.float32),
    )  # fmt: skip
    tmp.replace(OUT)
    print(f"wrote {OUT}: {len(names)} clips, {counts.sum()} frames")


def texts(args) -> None:
    """PER_CLIP descriptions of each embedded clip (drawn once, seed 0), embedded as the tool embeds a query."""
    have = set(np.load(OUT)["names"].tolist())
    rng = random.Random(0)
    queries = []
    for c in clips(args.annotations):
        if c["name"] in have:
            for text in rng.sample(c["captions"], min(PER_CLIP, len(c["captions"]))):
                queries.append({"video": c["name"], "text": text, "times": None})
    enc = Encoder("siglip2-base", folder=args.model_dir)
    t0 = time.time()
    out = np.concatenate([enc.encode_texts([q["text"] for q in queries[i : i + 64]])
                          for i in range(0, len(queries), 64)]).astype(np.float16)  # fmt: skip
    np.save(TEXTS, out)
    QUERIES.write_text(json.dumps(queries))
    print(f"{len(queries)} descriptions of {len(have)} clips embedded in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "assemble", "texts"])
    ap.add_argument("--annotations", required=True, help="msrvtt_train_9k.json of friedrichor/MSR-VTT")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--work", help="where each clip is written while it is decoded (default: the system's)")
    args = ap.parse_args()
    {"extract": extract, "assemble": lambda a: assemble(), "texts": texts}[args.stage](args)


if __name__ == "__main__":
    main()
