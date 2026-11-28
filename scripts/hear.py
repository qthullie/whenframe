"""The fourteenth round's part B (docs/PROTOCOL.md): what is heard, by Google DeepMind's EmbeddingGemma 2.

Runs where PyTorch, transformers 5.24.1 and sentence-transformers 6.1.0 are installed (not whenframe's
dependencies), its reference implementation in float32, as its model card says to embed:

    EG2_PYTHON -I scripts/hear.py audio --set unav --config DIR --work DIR       # UnAV-100's videos
    EG2_PYTHON -I scripts/hear.py audio --set youcook2 --config DIR --work DIR   # YouCook2's validation videos
    EG2_PYTHON -I scripts/hear.py texts --config DIR                              # their descriptions

DIR (`--config`) holds the model's small files at the revision below (its configuration, its tokenizer, its
processor's and sentence-transformers' settings); its weights are read into memory from the Hub, never to disk.

`audio` reads each video again as scripts/compare.py reads them (UnAV-100's archives, YouCook2's files), decodes
its sound, mono at 16 kHz, and embeds each of CLAP's windows (5 seconds, overlapping by half) without a prefix:
for UnAV-100 the windows CLAP's scores were computed on (results/cache/unav.npz), for YouCook2 the same cut
(whenframe.sounds.windows) of its sound. A video's windows are kept as they are done (results/cache/eg2_parts_SET)
and gathered, in the order of the set's frames, into results/cache/SET_eg2.npz: their starts, how many each video
has, and their embeddings in float16. Every video of both sets is embedded, so that the test needs no second
reading; only the tuning sets' numbers are computed before the test.

`texts` embeds the descriptions scripts/round14.py hear-texts wrote, each after `task: search result | query: `
(results/cache/round14_eg2_texts.json, to results/cache/round14_eg2_texts.npz).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REPO, REVISION = "google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e"
RATE = 16_000  # its audio: mono at 16 kHz
LENGTH, HOP = 5.0, 2.5  # CLAP's windows: 5 seconds, overlapping by half
PROMPT = "SearchQuery"  # "task: search result | query: "
CACHE = ROOT / "results" / "cache"
TEXTS = CACHE / "round14_eg2_texts.json"


def load_model(config: Path):
    """The model as its model card loads it (sentence-transformers: the transformer, mean pooling, normalised), in
    float32, its weights read into memory from the Hub at the revision above and handed to transformers."""
    import torch
    from safetensors.torch import load
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel

    torch.set_num_threads(4)
    t0 = time.time()
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors") as r:  # noqa: S310
        data = r.read()
    held = {"weights": load(data)}  # handed over once, then let go
    del data
    print(f"{REPO}: weights read into memory in {time.time() - t0:.0f} s", flush=True)
    original = AutoModel.from_pretrained

    def from_memory(name, *args, config=None, **kwargs):  # the files' folder for everything but the weights
        kwargs.pop("revision", None)
        model_class = AutoModel._model_mapping[type(config)]  # the class AutoModel would load for this config
        return model_class.from_pretrained(None, *args, config=config, state_dict=held["weights"], **kwargs)

    AutoModel.from_pretrained = from_memory
    try:
        model = SentenceTransformer(str(config), device="cpu", model_kwargs={"dtype": torch.float32})
    finally:
        AutoModel.from_pretrained = original
        held.clear()
    return model.eval()


def sound(path: Path) -> np.ndarray:
    """The sound track, mono at 16 kHz, float32; empty when there is none."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le",
           "-"]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(run.stdout, dtype=np.float32) if run.returncode == 0 else np.zeros(0, np.float32)


def windows(wave: np.ndarray) -> np.ndarray:
    """Start times (seconds) of whenframe.sounds.windows' cut, at 16 kHz: every 2.5 seconds, and a last one that
    ends with the sound."""
    if len(wave) == 0:
        return np.zeros(0)
    n, step = int(LENGTH * RATE), int(HOP * RATE)
    last = max(0, len(wave) - n)
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return np.array(starts) / RATE


def clips(wave: np.ndarray, starts: np.ndarray) -> list[np.ndarray]:
    n = int(LENGTH * RATE)
    return [wave[int(round(s * RATE)) : int(round(s * RATE)) + n] for s in starts]


def prefetched(source, bound: int):
    """The source's videos decoded to their sound in a thread of their own, ahead of the embedding, so that an
    archive's stream is read on while the model embeds (a stream left waiting is cut by its server); at most `bound`
    sounds wait (0: no bound, for UnAV-100's short clips, under 1 GB in all)."""
    import queue
    import threading

    waiting = queue.Queue(maxsize=bound)

    def read():
        try:
            for name, path, temporary in source:
                wave = sound(path)
                if temporary:
                    path.unlink(missing_ok=True)
                waiting.put((name, wave))
        finally:
            waiting.put(None)

    threading.Thread(target=read, daemon=True).start()
    while (item := waiting.get()) is not None:
        yield item


def audio(args) -> None:
    import compare

    names, given = set_videos(args.set)
    parts = CACHE / f"eg2_parts_{args.set}"
    parts.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (parts / f"{n}.npz").exists()]
    if todo:
        model = load_model(Path(args.config))
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
        if args.set == "unav":
            source = prefetched(compare.unav_source(todo, work), 0)
        else:
            source = prefetched(compare.youcook2_source(todo, work), 3)
        t0 = time.time()
        for k, (name, wave) in enumerate(source, 1):
            w0 = time.time()
            starts = given[name] if given is not None else windows(wave)
            emb = (model.encode([{"audio": {"array": c, "sampling_rate": RATE}} for c in clips(wave, starts)],
                                batch_size=8, convert_to_numpy=True)
                   if len(starts) and len(wave) else np.zeros((0, 768), np.float32))  # fmt: skip
            tmp = parts / f"{name}.tmp.npz"
            np.savez(tmp, starts=np.asarray(starts, np.float32), audio=emb.astype(np.float16),
                     seconds=len(wave) / RATE, wall=time.time() - w0)  # fmt: skip
            tmp.replace(parts / f"{name}.npz")
            if k % 10 == 0:
                print(f"{args.set}: {k} of {len(todo)} videos, {time.time() - t0:.0f} s", flush=True)
    gather(args.set, names)


def set_videos(name: str) -> tuple[list[str], dict[str, np.ndarray] | None]:
    """The set's videos in the order of its frames, and for UnAV-100 the windows CLAP's scores were computed on."""
    if name == "youcook2":
        return [str(n) for n in np.load(CACHE / "youcook2.npz")["names"]], None
    z = np.load(CACHE / "unav.npz")
    names = [str(n) for n in z["names"]]
    first = np.concatenate([[0], np.cumsum(z["wcounts_5"])[:-1]])
    starts = {n: z["wstarts_5"][first[i] : first[i] + z["wcounts_5"][i]] for i, n in enumerate(names)}
    return names, starts


def gather(name: str, names: list[str]) -> None:
    parts = CACHE / f"eg2_parts_{name}"
    missing = [n for n in names if not (parts / f"{n}.npz").exists()]
    if missing:
        print(f"{name}: {len(missing)} videos not read (e.g. {missing[:3]}): not gathered", flush=True)
        return
    got = [np.load(parts / f"{n}.npz") for n in names]
    out = CACHE / f"{name}_eg2.npz"
    np.savez(out, names=np.array(names), wstarts=np.concatenate([g["starts"] for g in got]),
             wcounts=np.array([len(g["starts"]) for g in got]),
             audio=np.concatenate([g["audio"] for g in got]).astype(np.float16),
             seconds=np.array([float(g["seconds"]) for g in got]))  # fmt: skip
    hours = sum(float(g["seconds"]) for g in got) / 3600
    print(f"wrote {out.name}: {len(names)} videos, {hours:.2f} hours of sound, "
          f"{sum(len(g['starts']) for g in got)} windows", flush=True)  # fmt: skip


def texts(args) -> None:
    said = json.loads(TEXTS.read_text())
    model = load_model(Path(args.config))
    out = {k: model.encode(v, prompt_name=PROMPT, batch_size=16, convert_to_numpy=True).astype(np.float32)
           for k, v in said.items()}  # fmt: skip
    np.savez(TEXTS.with_suffix(".npz"), **out)
    print("embedded: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audio", "texts"])
    ap.add_argument("--set", choices=["unav", "youcook2"])
    ap.add_argument("--config", required=True, help="the model's small files, at the revision above")
    ap.add_argument("--work", help="audio: where each video is written while its sound is decoded")
    args = ap.parse_args()
    {"audio": audio, "texts": texts}[args.stage](args)


if __name__ == "__main__":
    main()
