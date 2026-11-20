"""The eighth round (docs/PROTOCOL.md): whenframe learns a library from automatic descriptions of its frames.

Florence-2-base (Microsoft, MIT licence; onnx-community/Florence-2-base-ft, its image encoder run by OpenVINO and
the rest by ONNX Runtime) describes, in each 10 seconds of every video of a library, the frame the index holds
nearest to the block's middle, decoded again from the video at the describer's size. Nothing else about the
library is used: no description, label or annotation of it.

    python scripts/learn.py describe --libraries didemo-tuning,didemo-test --describer-dir DIR (compare.py's paths)
    python scripts/learn.py prepare  --libraries didemo-tuning (compare.py's paths)

The libraries are the benchmarks' tuning and test sets, each as its search sees it: didemo-tuning, didemo-test,
charades-tuning, charades-test, youcook2-tuning, youcook2-test, unav-tuning and unav-test. Their videos are read
again as in their rounds (scripts/compare.py's readers: DiDeMo's from disk, the others' from the Hugging Face
Hub), one at a time, and only the descriptions are kept: results/cache/learn_parts_LIBRARY/, one file a video, so
a stopped run resumes. `prepare` writes what scripts/temporal.py train --library reads: learn_LIBRARY_frames.npz,
learn_LIBRARY_queries.json and learn_LIBRARY_text.npy (the descriptions embedded by SigLIP 2's text tower, as the
tool embeds a query), and learn_LIBRARY_cost.json (the CPU time of decoding and describing).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compare  # noqa: E402

CACHE = ROOT / "results" / "cache"
BLOCK = 10.0  # seconds a description stands for
SIZE = 768  # Florence-2's input, squashed as its processor resizes
LIBRARIES = ("didemo-tuning", "didemo-test", "charades-tuning", "charades-test", "youcook2-tuning", "youcook2-test",
             "unav-tuning", "unav-test")  # fmt: skip
DONE = object()


# ---------------------------------------------------------------- the describer
class Describer:
    """Florence-2-base's caption of an image (task <CAPTION>): greedy decoding, no repeated 3-gram, 60 tokens at
    most, as BART generates (the decoder starts with </s>, then <s> is forced)."""

    PROMPT = "What does the image describe?"  # the processor's text for <CAPTION>
    MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    LAYERS, HEADS, HEAD_DIM, MAX_TOKENS = 6, 12, 64, 60

    def __init__(self, folder: str, threads: int = 4):
        import onnxruntime as ort
        import openvino as ov
        from tokenizers import Tokenizer

        d = Path(folder)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = {n: ort.InferenceSession(str(d / "onnx" / f"{n}.onnx"), opts, providers=["CPUExecutionProvider"])
                     for n in ("embed_tokens", "encoder_model", "decoder_model_merged")}  # fmt: skip
        core = ov.Core()
        self.vision = core.compile_model(str(d / "onnx" / "vision_encoder.onnx"), "CPU",
                                         {"INFERENCE_NUM_THREADS": threads})  # fmt: skip
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.prompt = np.array([self.tok.encode(self.PROMPT).ids], np.int64)

    def run(self, name: str, feeds: dict) -> dict:
        s = self.sess[name]
        return dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds), strict=True))

    def features(self, frames: np.ndarray) -> np.ndarray:
        """(n, 768, 768, 3) uint8 frames -> the image encoder's features, a few frames at a time."""
        out = []
        for i in range(0, len(frames), 4):
            x = (frames[i : i + 4].astype(np.float32) / 255.0 - self.MEAN) / self.STD
            r = self.vision({"pixel_values": np.ascontiguousarray(x.transpose(0, 3, 1, 2))})
            out.append(next(iter(r.values())))
        return np.concatenate(out)

    def caption(self, feats: np.ndarray) -> str:
        emb = self.run("embed_tokens", {"input_ids": self.prompt})["inputs_embeds"]
        x = np.concatenate([feats[None], emb], axis=1)
        mask = np.ones(x.shape[:2], np.int64)
        enc = self.run("encoder_model", {"attention_mask": mask, "inputs_embeds": x})["last_hidden_state"]
        past = {f"past_key_values.{i}.{side}.{kv}": np.zeros((1, self.HEADS, 0, self.HEAD_DIM), np.float32)
                for i in range(self.LAYERS) for side in ("decoder", "encoder") for kv in ("key", "value")}  # fmt: skip
        ids, out, forced = [2], [], [0]
        for step in range(self.MAX_TOKENS + 1):
            e = self.run("embed_tokens", {"input_ids": np.array([[ids[-1]]], np.int64)})["inputs_embeds"]
            r = self.run("decoder_model_merged", {"encoder_attention_mask": mask, "encoder_hidden_states": enc,
                                                  "inputs_embeds": e, "use_cache_branch": np.array([step > 0]),
                                                  **past})  # fmt: skip
            for k, v in r.items():
                if k.startswith("present.") and (step == 0 or ".decoder." in k):
                    past["past_key_values." + k[len("present.") :]] = v  # the encoder's stay those of step 0
            if forced:
                nxt = forced.pop(0)
            else:
                logits = r["logits"][0, -1].copy()
                for j in range(len(ids) - 2):  # no repeated 3-gram
                    if ids[j : j + 2] == ids[-2:]:
                        logits[ids[j + 2]] = -np.inf
                nxt = int(np.argmax(logits))
            ids.append(nxt)
            if nxt == 2:
                break
            out.append(nxt)
        return self.tok.decode(out, skip_special_tokens=True).strip()


# ---------------------------------------------------------------- the libraries, as their searches see them
@dataclass
class Library:
    name: str
    names: list[str]  # the videos, as their dataset's reader names them
    emb: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    durations: np.ndarray  # searched seconds: DiDeMo's first 30


def subset(f, keep: set[str], name: str) -> Library:
    idx = [i for i, n in enumerate(f.names) if n in keep]
    rows = np.concatenate([np.arange(f.starts[i], f.starts[i] + f.counts[i]) for i in idx])
    counts = np.array([f.counts[i] for i in idx])
    return Library(name, [f.names[i] for i in idx], f.emb[rows], f.times[rows],
                   np.concatenate([[0], np.cumsum(counts)[:-1]]), counts,
                   np.asarray(f.durations, np.float64)[idx])  # fmt: skip


def library(name: str, b: compare.Benchmarks) -> Library:
    base = compare.Candidate("base")
    dataset, part = name.rsplit("-", 1)
    if dataset == "didemo":
        import didemo

        f, _, _ = b.didemo(base, "train" if part == "tuning" else "test")
        lib = subset(f, set(f.names), name)
        lib.durations = np.minimum(lib.durations, didemo.SECONDS)
        return lib
    if dataset == "charades":
        if part == "tuning":
            from temporal import Charades

            f = Charades(compare.didemo_bank("siglip2-base"), "frame_window_bank", {}).f
        else:
            import charades

            f, _, _ = b.once("charades", lambda: charades.load(b.ns(annotations=b.args.charades_annotations,
                                                                    **b.shared)))  # fmt: skip
        return subset(f, set(f.names), name)
    if dataset == "youcook2":
        import youcook2

        g, _, _, _, _ = b.youcook2(base)
        tuning = youcook2.split(g.names)
        return subset(g, tuning if part == "tuning" else set(g.names) - tuning, name)
    if dataset == "unav":
        import unav

        z = np.load(CACHE / f"{compare.cache_name('unav', 'siglip2-base')}.npz")
        names = z["names"].tolist()
        tuning = unav.tuning_videos(names)
        lib = unav.Library(z, tuning if part == "tuning" else set(names) - tuning)
        counts = np.array([c for _, c in lib.ranges])
        return Library(name, lib.names, lib.emb, lib.times, np.array([s for s, _ in lib.ranges]), counts,
                       np.asarray(lib.durations, np.float64))  # fmt: skip
    raise ValueError(name)


def blocks(duration: float, times: np.ndarray) -> list[tuple[float, float, float]]:
    """(start, end, instant): the video cut into 10-second blocks from its start, a last block shorter than 5
    seconds joined to the one before; in each, the indexed frame nearest to its middle (a frame stands for the
    second after its instant)."""
    n = max(1, int((duration + BLOCK / 2) // BLOCK))
    out = []
    for k in range(n):
        a, b = k * BLOCK, duration if k == n - 1 else (k + 1) * BLOCK
        t = float(times[np.argmin(np.abs(times + 0.5 - (a + b) / 2))])
        out.append((a, float(b), t))
    return out


def decode(path: Path, instants: list[float], seconds: float | None = None) -> tuple[np.ndarray, float]:
    """The frames at these instants (of the 1-frame-a-second clock the index reads), squashed to SIZE x SIZE;
    and ffmpeg's CPU time."""
    idx = sorted({int(round(t)) for t in instants})
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    limit = ["-t", f"{seconds:g}"] if seconds else []
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *limit, "-i", str(path), "-an", "-sn", "-vf",
           f"fps=1,select='{expr}',scale={SIZE}:{SIZE}:flags=bicubic", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime
    n = len(data) // (SIZE * SIZE * 3)
    frames = np.frombuffer(data[: n * SIZE * SIZE * 3], np.uint8).reshape(n, SIZE, SIZE, 3)
    got = dict(zip(idx[:n], frames, strict=True))  # a short video ends early: its last instants are missing
    return np.stack([got[int(round(t))] for t in instants if int(round(t)) in got]) if got else frames, cpu


# ---------------------------------------------------------------- describing
def sources(dataset: str, names: list[str], work: Path, args):
    if dataset == "didemo":
        for name in names:
            folder = Path(args.didemo_test_videos if name in args.didemo_test_names else args.didemo_train_videos)
            yield name, (folder / name if (folder / name).exists() else folder / f"{name}.mp4"), False
    elif dataset == "charades":
        yield from compare.charades_source([n if n.endswith(".mp4") else f"{n}.mp4" for n in names], work)
    elif dataset == "youcook2":
        yield from compare.youcook2_source(names, work)
    else:
        yield from compare.unav_source(names, work)


def describe(args) -> None:
    b = compare.Benchmarks(args)
    libs = [library(n, b) for n in args.libraries.split(",")]
    datasets = {lib.name.rsplit("-", 1)[0] for lib in libs}
    assert len(datasets) == 1, "one dataset at a time: its videos are read once for all its libraries"
    dataset = datasets.pop()
    plans, where = {}, {}
    for lib in libs:
        folder = CACHE / f"learn_parts_{lib.name}"
        folder.mkdir(parents=True, exist_ok=True)
        for v, n in enumerate(lib.names):
            if not (folder / f"{Path(n).stem}.json").exists():
                t = lib.times[lib.starts[v] : lib.starts[v] + lib.counts[v]]
                plans[n] = blocks(float(lib.durations[v]), np.asarray(t, np.float64))
                where[n] = folder
    todo = list(plans)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{dataset}: {len(todo)} videos to describe", flush=True)
    if not todo:
        return
    args.didemo_test_names = {n for lib in libs if lib.name == "didemo-test" for n in lib.names}
    describer = Describer(args.describer_dir, args.threads)
    work = Path(tempfile.mkdtemp(prefix="learn-", dir=args.work))
    seconds = 30.0 if dataset == "didemo" else None
    found, ready = queue.Queue(maxsize=3), queue.Queue(maxsize=3)

    def read():
        try:
            key = {(n if n.endswith(".mp4") else f"{n}.mp4") if dataset == "charades" else n: n for n in todo}
            for name, path, temporary in sources(dataset, todo, work, args):
                found.put((key.get(name, name), path, temporary))
        finally:
            found.put(DONE)

    def decoding():
        while (item := found.get()) is not DONE:
            name, path, temporary = item
            try:
                frames, cpu = decode(path, [t for _, _, t in plans[name]], seconds)
                ready.put((name, frames, cpu, None))
            except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
                ready.put((name, None, 0.0, ex))
            finally:
                if temporary:
                    path.unlink(missing_ok=True)
        ready.put(DONE)

    threading.Thread(target=read, daemon=True).start()
    threading.Thread(target=decoding, daemon=True).start()
    n, t0 = 0, time.time()
    while (item := ready.get()) is not DONE:
        name, frames, decode_cpu, err = item
        n += 1
        if err is not None or frames is None or len(frames) == 0:
            print(f"skip {name}: {err or 'no frame decoded'}", file=sys.stderr, flush=True)
            continue
        c0 = time.process_time()
        feats = describer.features(frames)
        texts = [describer.caption(x) for x in feats]
        cpu = time.process_time() - c0
        plan = plans[name][: len(texts)]
        part = {"video": name, "blocks": [[a, b] for a, b, _ in plan], "at": [t for _, _, t in plan],
                "texts": texts, "decode_cpu": decode_cpu, "describe_cpu": cpu}  # fmt: skip
        tmp = where[name] / f"{Path(name).stem}.tmp"
        tmp.write_text(json.dumps(part))
        tmp.replace(where[name] / f"{Path(name).stem}.json")
        if n % 20 == 0 or n == len(todo):
            print(f"{dataset}: {n}/{len(todo)} videos, {time.time() - t0:.0f} s; {name}: {texts[0]!r}", flush=True)
    shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- what training reads
def prepare(args) -> None:
    from whenframe.encoder import Encoder

    b = compare.Benchmarks(args)
    enc = None
    for name in args.libraries.split(","):
        lib = library(name, b)
        folder = CACHE / f"learn_parts_{name}"
        parts = {n: folder / f"{Path(n).stem}.json" for n in lib.names}
        missing = [n for n, p in parts.items() if not p.exists()]
        if missing:
            print(f"{name}: {len(missing)} of {len(lib.names)} videos not described (e.g. {missing[:3]})")
        queries, decode_cpu, describe_cpu = [], 0.0, 0.0
        for n, p in parts.items():
            if p.exists():
                part = json.loads(p.read_text())
                decode_cpu, describe_cpu = decode_cpu + part["decode_cpu"], describe_cpu + part["describe_cpu"]
                queries += [{"video": n, "text": t, "seconds": [a, b_]} for (a, b_), t in
                            zip(part["blocks"], part["texts"], strict=True)]  # fmt: skip
        np.savez(CACHE / f"learn_{name}_frames.npz", names=np.array(lib.names), emb=lib.emb.astype(np.float16),
                 times=lib.times.astype(np.float32), keep=np.ones(len(lib.times), bool), counts=lib.counts,
                 durations=lib.durations.astype(np.float32))  # fmt: skip
        (CACHE / f"learn_{name}_queries.json").write_text(json.dumps(queries))
        if enc is None:
            enc = Encoder("siglip2-base", folder=args.model_dir, backend="openvino")
            enc.encode_texts(["a warm-up"])  # compiled before the clock starts, as the indexing costs are measured
        texts = [q["text"] for q in queries]
        c0 = time.process_time()
        emb = np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)])
        embed_cpu = time.process_time() - c0  # embedding the descriptions is part of learning the library
        np.save(CACHE / f"learn_{name}_text.npy", emb.astype(np.float16))
        described = [v for v, n in enumerate(lib.names) if parts[n].exists()]
        hours = float(lib.durations[described].sum() / 3600)  # of the videos described
        cost = {"videos": len(lib.names), "described": len(lib.names) - len(missing), "descriptions": len(queries),
                "hours": hours, "decode_cpu_min_per_hour": decode_cpu / 60 / hours,
                "describe_cpu_min_per_hour": describe_cpu / 60 / hours,
                "embed_cpu_min_per_hour": embed_cpu / 60 / hours}  # fmt: skip
        (CACHE / f"learn_{name}_cost.json").write_text(json.dumps(cost, indent=1))
        print(f"{name}: {len(queries)} descriptions of {cost['described']} videos; decoding "
              f"{cost['decode_cpu_min_per_hour']:.1f}, describing {cost['describe_cpu_min_per_hour']:.1f} and "
              f"embedding the descriptions {cost['embed_cpu_min_per_hour']:.2f} CPU min per hour of video")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["describe", "prepare"])
    ap.add_argument("--libraries", required=True, help=f"among {', '.join(LIBRARIES)}")
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", required=True, help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--describer-dir", help="describe: Florence-2-base-ft's ONNX export (onnx/ and tokenizer.json)")
    ap.add_argument("--didemo-train-videos", help="describe: the folder of DiDeMo's tuning videos")
    ap.add_argument("--didemo-test-videos", help="describe: the folder of DiDeMo's test videos")
    ap.add_argument("--threads", type=int, default=4, help="describe: the describer's threads")
    ap.add_argument("--work", help="describe: where a video is kept while it is read (default: the system's)")
    ap.add_argument("--limit", type=int, help="describe: at most this many videos (a trial)")
    args = ap.parse_args()
    {"describe": describe, "prepare": prepare}[args.stage](args)


if __name__ == "__main__":
    main()
