"""The sixth round (docs/PROTOCOL.md): five ways of embedding frames, measured the same way on four benchmarks.

    python scripts/compare.py extract --dataset charades --models siglip2-so400m,siglip2-b32   # the same frames
    python scripts/compare.py tune     ...   # each candidate's speech and sound weights, on the tuning sets
    python scripts/compare.py evaluate ...   # once, after the choices are recorded in docs/PROTOCOL.md

The candidates: SigLIP 2 B/32, B/16 (the standard mode) and So400m, and B/16
with whenframe's model v1 and v2. Every encoder sees the frames at the instants
of the standard mode's caches: each video is read again (from disk for DiDeMo,
from the Hugging Face Hub for the others, one at a time and not kept), decoded
at each encoder's size, and only the frames at the cached instants are
embedded, with OpenVINO.
"""

from __future__ import annotations

import argparse
import http.client
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
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from whenframe import temporal  # noqa: E402
from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
HUB = "https://huggingface.co/datasets"
CHARADES_ZIPS = [f"{HUB}/lmms-eval/charades_sta/resolve/main/Charades_v1_480_part_{i}.zip" for i in range(1, 5)]
YOUCOOK2 = f"{HUB}/nguyenminh04/youcook2-videos/resolve/main/videos/{{}}.mp4"
UNAV = f"{HUB}/nguyenminh04/unav100-videos/resolve/main/unav100_{{:03d}}.tar"
# the standard mode's caches, whose frames every encoder sees
BASE = {"didemo-test": "didemo_test_openvino", "didemo-train": "didemo_train_openvino",
        "charades": "charades_test", "youcook2": "youcook2", "unav": "unav"}  # fmt: skip
DONE = object()


def short(model: str) -> str:
    return model.replace("siglip2-", "")


def cache_name(dataset: str, model: str) -> str:
    """Where `model`'s embeddings of the dataset's frames are kept (the base model's: the standard caches)."""
    if model == "siglip2-base":
        return BASE[dataset]
    if dataset.startswith("didemo"):
        return f"didemo_{dataset.split('-')[1]}_{model}_openvino"  # as scripts/didemo.py names them
    return f"{BASE[dataset]}_{short(model)}_openvino"


# ---------------------------------------------------------------- the videos, read again
def didemo_source(dataset: str, names: list[str], folder: str, work: Path):
    for name in names:
        path = Path(folder) / name
        yield name, (path if path.exists() else Path(folder) / f"{name}.mp4"), False


def charades_source(names: list[str], work: Path):
    from msrvtt import Remote

    wanted, readers = set(names), []
    index = {}
    for url in CHARADES_ZIPS:
        z = zipfile.ZipFile(io.BufferedReader(Remote(url), buffer_size=1 << 22))
        readers.append(z)
        for info in z.infolist():
            if Path(info.filename).name in wanted:
                index[Path(info.filename).name] = (z, info)
    for name in names:
        if name not in index:
            print(f"skip {name}: not in the archives", file=sys.stderr, flush=True)
            continue
        z, info = index[name]
        path = work / f"{Path(name).stem}.mp4"  # named after the annotations, never the archive's own path
        with z.open(info) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        yield name, path, True


def youcook2_source(names: list[str], work: Path):
    for name in names:
        path = work / f"{name}.mp4"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(YOUCOOK2.format(name), timeout=300) as r, open(path, "wb") as dst:  # noqa: S310
                    shutil.copyfileobj(r, dst)
                break
            except (OSError, http.client.HTTPException) as ex:
                if attempt == 4:
                    print(f"skip {name}: {ex}", file=sys.stderr, flush=True)
                    path = None
                else:
                    time.sleep(10 * (attempt + 1))
        if path is not None:
            yield name, path, True


def unav_source(names: list[str], work: Path, shards: int = 22):
    wanted, seen = set(names), set()
    for shard in range(shards):
        for attempt in range(4):
            try:
                with (
                    urllib.request.urlopen(UNAV.format(shard), timeout=120) as r,
                    tarfile.open(fileobj=r, mode="r|") as tar,
                ):  # noqa: S310
                    for member in tar:
                        name = Path(member.name).stem
                        if not member.isfile() or name not in wanted or name in seen:
                            continue
                        path = work / f"{name}.mp4"
                        with tar.extractfile(member) as src, open(path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        seen.add(name)
                        yield name, path, True
                break
            except (OSError, tarfile.TarError, http.client.HTTPException) as ex:
                print(f"archive {shard}: {ex}; again", file=sys.stderr, flush=True)
                time.sleep(30 * (attempt + 1))


def extract(args) -> None:
    """Each video of the dataset decoded at each encoder's size; the frames at the cached instants embedded."""
    models = args.models.split(",")
    base = np.load(CACHE / f"{BASE[args.dataset]}.npz")
    names = [str(n) for n in base["names"]]
    starts, counts, times = base["starts"], base["counts"], base["times"]
    want = {n: times[s : s + c] for n, s, c in zip(names, starts, counts, strict=True)}
    parts = {m: CACHE / f"compare_parts_{args.dataset}_{short(m)}" for m in models}
    for p in parts.values():
        p.mkdir(parents=True, exist_ok=True)
    todo_names = [n for n in names if not all((parts[m] / f"{Path(n).stem}.npz").exists() for m in models)]
    if args.limit:
        todo_names = todo_names[: args.limit]
    encoders = {m: Encoder(m, backend="openvino") for m in models}
    for enc in encoders.values():
        enc.images  # noqa: B018 - compiled before the first video
    work = Path(tempfile.mkdtemp(prefix="compare-", dir=args.work))
    source = {"didemo-test": lambda: didemo_source(args.dataset, todo_names, args.videos, work),
              "didemo-train": lambda: didemo_source(args.dataset, todo_names, args.videos, work),
              "charades": lambda: charades_source(todo_names, work),
              "youcook2": lambda: youcook2_source(todo_names, work),
              "unav": lambda: unav_source(todo_names, work)}[args.dataset]  # fmt: skip
    todo, ready = queue.Queue(maxsize=4), queue.Queue(maxsize=4)

    def read():
        try:
            for item in source():
                todo.put(item)
        finally:
            todo.put(DONE)

    def decode():
        while (item := todo.get()) is not DONE:
            name, path, temporary = item
            try:
                out = {}
                for m, enc in encoders.items():
                    t0 = time.process_time()
                    t, frames = sample_frames(path, 1.0, enc.size)
                    at = np.isin(np.round(t, 3), np.round(want[name], 3))
                    out[m] = (t[at], frames[at], time.process_time() - t0)
                ready.put((name, out, None))
            except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
                ready.put((name, None, ex))
            finally:
                if temporary:
                    path.unlink(missing_ok=True)
        todo.put(DONE)
        ready.put(DONE)

    threading.Thread(target=read, daemon=True).start()
    decoders = [threading.Thread(target=decode, daemon=True) for _ in range(2)]
    for d in decoders:
        d.start()
    finished, n, t0 = 0, 0, time.time()
    while finished < len(decoders):
        item = ready.get()
        if item is DONE:
            finished += 1
            continue
        name, out, err = item
        n += 1
        if err is not None:
            print(f"skip {name}: {err}", file=sys.stderr, flush=True)
            continue
        for m, (t, frames, decode_cpu) in out.items():
            if not np.array_equal(np.round(t, 3), np.round(want[name], 3)):
                print(f"skip {name} ({m}): not the same instants ({len(t)} of {len(want[name])})", file=sys.stderr)
                continue
            c0 = time.process_time()
            emb = encoders[m].encode_images(frames)
            tmp = parts[m] / f"{Path(name).stem}.tmp.npz"
            np.savez(tmp, emb=emb.astype(np.float16), times=t.astype(np.float32), decode_cpu=decode_cpu,
                     encode_cpu=time.process_time() - c0)  # fmt: skip
            tmp.replace(parts[m] / f"{Path(name).stem}.npz")
        if n % 50 == 0:
            print(f"{args.dataset}: {n} videos this run, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    if not args.limit:
        for m in models:
            assemble(args.dataset, m)


def assemble(dataset: str, model: str) -> None:
    """The embedded frames in the order of the standard mode's cache, with its layout."""
    base = np.load(CACHE / f"{BASE[dataset]}.npz")
    folder = CACHE / f"compare_parts_{dataset}_{short(model)}"
    names = [str(n) for n in base["names"]]
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        print(f"{dataset}, {model}: {len(missing)} videos missing, e.g. {missing[:3]}; not assembled")
        return
    z = [np.load(folder / f"{Path(n).stem}.npz") for n in names]
    emb = np.concatenate([x["emb"] for x in z])
    assert np.array_equal(np.concatenate([x["times"] for x in z]), base["times"]), "not the same frames"
    extra = {k: base[k] for k in base.files if k not in ("emb", "precision", "decode_cpu", "encode_cpu")}
    out = CACHE / f"{cache_name(dataset, model)}.npz"
    np.savez_compressed(out, **extra, emb=emb, decode_cpu=np.array([float(x["decode_cpu"]) for x in z]),
                        encode_cpu=np.array([float(x["encode_cpu"]) for x in z]))  # fmt: skip
    print(f"wrote {out.name}: {len(names)} videos, {len(emb)} frames")


# ---------------------------------------------------------------- the candidates, searched the same way
@dataclass(frozen=True)
class Candidate:
    name: str
    model: str = "siglip2-base"
    weights: str | None = None  # whenframe's model (a .npz of results/temporal or whenframe/data)
    alpha: float = 1.0  # how much of its correction is added (WiSE-FT's mixing)

    def adapter(self) -> dict | None:
        if self.weights is None:
            return None
        p = temporal.load(Path(self.weights))
        p["out_w"], p["out_b"] = self.alpha * p["out_w"], self.alpha * p["out_b"]
        return p


def bank_params(model: str) -> tuple[float, float]:
    """(beta, strength) of the query bank chosen for the encoder on DiDeMo's tuning videos."""
    if model == "siglip2-so400m":
        best = json.loads((ROOT / "results" / "didemo_tuned_so400m.json").read_text())["best"]
        return best["beta"], best["lam"]
    if model == "siglip2-b32":
        p = json.loads((ROOT / "results" / "didemo_tuned_round2.json").read_text())["siglip32_bank"]["params"]
        return p["beta"], p["lam"]
    p = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())["frame_window_bank"]["params"]
    return p["beta"], p["lam"]


def text_embeddings(model: str, key: str, texts: list[str]) -> np.ndarray:
    """Descriptions embedded by the encoder's text tower, as the tool embeds a query; kept in the cache."""
    path = CACHE / f"{key}_text_compare_{short(model)}.npy"
    if not path.exists():
        enc = Encoder(model, backend="openvino")
        np.save(path, np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)]))
    return np.load(path).astype(np.float32)


def adapted(c: Candidate, emb, times, starts, counts) -> np.ndarray:
    p = c.adapter()
    if p is None:
        return np.asarray(emb, dtype=np.float32)
    out = np.empty(emb.shape, np.float32)
    for s, n in zip(starts, counts, strict=True):
        out[s : s + n] = temporal.apply(p, emb[s : s + n], times[s : s + n])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "assemble"])
    ap.add_argument("--dataset", choices=list(BASE), help="extract: whose frames")
    ap.add_argument("--models", default="siglip2-so400m,siglip2-b32", help="extract: encoders, comma-separated")
    ap.add_argument("--videos", help="extract: the folder of DiDeMo's videos (the others come from the Hub)")
    ap.add_argument("--work", help="extract: where each video is written while it is decoded")
    ap.add_argument("--limit", type=int, help="extract: only the first videos (a trial; nothing assembled)")
    args = ap.parse_args()
    if args.stage == "assemble":
        for m in args.models.split(","):
            assemble(args.dataset, m)
        return
    extract(args)


if __name__ == "__main__":
    main()
