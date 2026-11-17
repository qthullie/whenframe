"""The DiDeMo training videos, embedded to train whenframe's temporal model (fifth round).

The 8,511 training videos are read from the archive of friedrichor/DiDeMo on the
Hugging Face Hub (one tar file cut into eight parts of 5 GiB), one video at a
time, and none is kept: the first 30 seconds of each are decoded at one frame a
second and embedded by SigLIP 2 base with OpenVINO, repeats marked as the tool
skips them, as scripts/didemo.py extract does for the benchmark. The 500 tuning
videos are left out: they keep choosing the settings, the model's included.

    python scripts/train_data.py --annotations didemo_train.json --model-dir ...

Resumes where it stopped: the position of the next video in the archive is
noted as it goes, and videos already embedded are skipped. Writes
results/cache/didemo_all_openvino.npz.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

REPO = "friedrichor/DiDeMo"
PARTS = [(f"DiDeMo_Videos_mp4_train.tar.part-{i:03d}", 5368709120) for i in range(7)]
PARTS.append(("DiDeMo_Videos_mp4_train.tar.part-007", 5081006080))
CACHE = ROOT / "results" / "cache"
OUT = CACHE / "didemo_all_openvino.npz"
SECONDS = 30.0  # what DiDeMo's annotators saw
DONE = object()


class Archive(io.RawIOBase):
    """The parts, read one after the other as one file, from byte `start` of the whole."""

    def __init__(self, start: int = 0, url: str = f"https://huggingface.co/datasets/{REPO}/resolve/main/"):
        self.url, self.position, self.response = url, start, None

    def readable(self) -> bool:
        return True

    def _open(self):
        first = 0
        for name, size in PARTS:
            if self.position < first + size:
                req = urllib.request.Request(self.url + name, headers={"Range": f"bytes={self.position - first}-"})
                return urllib.request.urlopen(req, timeout=120)  # noqa: S310 (fixed host)
            first += size
        return None

    def readinto(self, buffer) -> int:
        while True:
            if self.response is None:
                self.response = self._open()
                if self.response is None:
                    return 0  # past the last part
            n = self.response.readinto(buffer)
            if n:
                self.position += n
                return n
            self.response.close()  # the end of a part: the next one goes on
            self.response = None


def stem(name: str) -> str:
    return Path(name).stem


def read(start: int, wanted: set[str], parts: Path, work: Path, todo: queue.Queue) -> None:
    """Reads the archive from `start`, writes each wanted video to `work` and queues it with its offset."""
    stream = io.BufferedReader(Archive(start), buffer_size=1 << 22)
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            name = stem(member.name)
            if not member.isfile() or name not in wanted or (parts / f"{name}.npz").exists():
                continue
            path = work / f"{name}.mp4"  # named after the annotations, never the archive's own path
            with tar.extractfile(member) as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            todo.put((name, path, start + member.offset))
    todo.put(DONE)


def decode(todo: queue.Queue, ready: queue.Queue) -> None:
    while (item := todo.get()) is not DONE:
        name, path, offset = item
        try:
            duration = probe(path).duration or 0.0
            t, frames = sample_frames(path, 1.0, 224, seconds=SECONDS)
            ready.put((name, offset, t, frames, duration, None))
        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
            ready.put((name, offset, None, None, 0.0, ex))
        finally:
            path.unlink(missing_ok=True)
    todo.put(DONE)  # for the other decoder
    ready.put(DONE)


def extract(args) -> None:
    ann = json.loads(Path(args.annotations).read_text())
    tuning = {stem(n) for n in (ROOT / "results" / "didemo_train_videos.txt").read_text().split()}
    wanted = {stem(a["video"]) for a in ann} - tuning
    parts = CACHE / "didemo_all_parts"
    parts.mkdir(parents=True, exist_ok=True)
    note = parts / "position.json"
    start = json.loads(note.read_text())["offset"] if note.exists() else 0
    enc = Encoder("siglip2-base", folder=args.model_dir, backend="openvino")
    enc.images  # noqa: B018 - compiled before the first video
    work = Path(tempfile.mkdtemp(prefix="didemo-", dir=args.work))
    todo, ready = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
    threading.Thread(target=read, args=(start, wanted, parts, work, todo), daemon=True).start()
    decoders = [threading.Thread(target=decode, args=(todo, ready), daemon=True) for _ in range(2)]
    for d in decoders:
        d.start()
    pending: dict[str, int] = {}  # offsets of the videos read and not yet written, for resuming
    finished, t0, n = 0, time.time(), 0
    while finished < len(decoders):
        item = ready.get()
        if item is DONE:
            finished += 1
            continue
        name, offset, t, frames, duration, err = item
        pending[name] = offset
        if err is None and len(frames):
            keep = np.zeros(len(frames), dtype=bool)
            keep[drop_repeats(frames)] = True
            emb = enc.encode_images(frames)
            tmp = parts / f"{name}.tmp.npz"
            np.savez(tmp, emb=emb.astype(np.float16), times=t.astype(np.float32), keep=keep,
                     duration=min(duration or len(frames), SECONDS))  # fmt: skip
            tmp.replace(parts / f"{name}.npz")
        else:
            print(f"skip {name}: {err or 'no frame'}", file=sys.stderr, flush=True)
        del pending[name]
        n += 1
        # resume from the earliest video still in the pipeline, or from this one's successor's neighbourhood
        note.write_text(json.dumps({"offset": min(pending.values(), default=offset)}))
        if n % 100 == 0:
            done = len(list(parts.glob("*.npz")))
            print(f"{done} videos embedded, {n} this run, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    assemble(ann, parts)


def assemble(ann: list, parts: Path) -> None:
    names = sorted(p.stem for p in parts.glob("*.npz") if not p.stem.endswith(".tmp"))
    z = [np.load(parts / f"{n}.npz") for n in names]
    counts = np.array([len(x["times"]) for x in z])
    np.savez_compressed(
        OUT, names=np.array(names), emb=np.concatenate([x["emb"] for x in z]),
        times=np.concatenate([x["times"] for x in z]), keep=np.concatenate([x["keep"] for x in z]),
        counts=counts, starts=np.concatenate([[0], np.cumsum(counts)[:-1]]),
        durations=np.array([float(x["duration"]) for x in z], dtype=np.float32),
    )  # fmt: skip
    print(f"wrote {OUT}: {len(names)} videos, {counts.sum()} frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="DiDeMo's didemo_train.json")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--work", help="where each video is written while it is decoded (default: the system's)")
    ap.add_argument("--assemble", action="store_true", help="only gather the videos embedded so far")
    args = ap.parse_args()
    if args.assemble:
        assemble(json.loads(Path(args.annotations).read_text()), CACHE / "didemo_all_parts")
    else:
        extract(args)


if __name__ == "__main__":
    main()
