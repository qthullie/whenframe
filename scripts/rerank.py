"""Re-ranking the first moments with a small vision-language model, tried on the tuning videos only.

The default search ranks moments across the tuning videos for a fixed sample of
training descriptions. SmolVLM-256M-Instruct (Hugging Face, Apache 2.0), run by
ONNX Runtime on the CPU, then looks at the best frame of each of the first ten
moments and is asked whether it shows the description. Its answer, the log-odds
of "yes" against "no", re-orders those ten: alone, or added to the search's own
score with a weight chosen on the other half of the sample.

    python scripts/rerank.py --vlm-dir DIR --videos DIR     # DIR: SmolVLM-256M-Instruct with its onnx/ folder

Needs the caches of `make didemo-extract` and `make didemo-tune`. Writes
results/didemo_train_rerank.json; the model's answers are kept in
results/cache/didemo_train_vlm.npz, so a second run only recounts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import SHORTLIST, iou  # noqa: E402
from explore import sign_test  # noqa: E402

from whenframe.bank import offsets  # noqa: E402
from whenframe.moments import windows  # noqa: E402
from whenframe.video import sample_frames  # noqa: E402

TOP = 10  # moments looked at again, per description
QUESTION = 'Does this image match the description "{}"? Answer yes or no.'
WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]


class SmolVLM:
    """Yes-or-no questions about single pictures, for the ONNX export of SmolVLM-256M-Instruct.

    The picture is squashed to 512 x 512 and not split into tiles (64 image
    tokens), which is the processor's behaviour with do_image_splitting=False.
    """

    def __init__(self, folder: str | Path, threads: int = 0) -> None:
        folder = Path(folder)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads

        def session(name: str) -> ort.InferenceSession:
            return ort.InferenceSession(str(folder / "onnx" / name), opts, providers=["CPUExecutionProvider"])

        self.vision = session("vision_encoder.onnx")
        self.embed = session("embed_tokens.onnx")
        self.decoder = session("decoder_model_merged.onnx")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        config = json.loads((folder / "config.json").read_text())
        text = config["text_config"]
        self.layers, self.head_dim = text["num_hidden_layers"], text["head_dim"]
        self.kv_heads = text["num_key_value_heads"]
        self.image_token = config["image_token_id"]
        self.size = config["vision_config"]["image_size"]
        self.image_tokens = json.loads((folder / "processor_config.json").read_text())["image_seq_len"]
        self.yes = self._ids(["Yes", " Yes", "yes", " yes"])
        self.no = self._ids(["No", " No", "no", " no"])

    def _ids(self, words: list[str]) -> list[int]:
        ids = [self.tokenizer.encode(w, add_special_tokens=False).ids for w in words]
        return [i[0] for i in ids if len(i) == 1]

    def prompt(self, question: str) -> np.ndarray:
        image = "<fake_token_around_image><global-img>" + "<image>" * self.image_tokens + "<fake_token_around_image>"
        text = f"<|im_start|>User:{image}{question}<end_of_utterance>\nAssistant:"
        return np.array(self.tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int64)

    def see(self, frames: np.ndarray) -> np.ndarray:
        """Image features of uint8 frames (n, size, size, 3): (n, image_tokens, hidden)."""
        x = (frames.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)[None]
        mask = np.ones((1, len(frames), self.size, self.size), dtype=bool)
        return self.vision.run(None, {"pixel_values": x, "pixel_attention_mask": mask})[0]

    def logits(self, ids: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Next-token logits after the prompt `ids`, once for each picture's features."""
        n = len(features)
        embeds = np.repeat(self.embed.run(None, {"input_ids": ids[None]})[0], n, axis=0)
        embeds[:, ids == self.image_token] = features
        feeds = {
            "inputs_embeds": embeds,
            "attention_mask": np.ones((n, len(ids)), dtype=np.int64),
            "position_ids": np.repeat(np.arange(len(ids), dtype=np.int64)[None], n, axis=0),
        }
        empty = np.zeros((n, self.kv_heads, 0, self.head_dim), dtype=np.float32)
        for i in range(self.layers):
            feeds[f"past_key_values.{i}.key"] = feeds[f"past_key_values.{i}.value"] = empty
        return self.decoder.run(["logits"], feeds)[0][:, -1]

    def yes_or_no(self, question: str, features: np.ndarray) -> np.ndarray:
        """log P(yes) - log P(no), summed over the spellings of each, for each picture."""
        z = self.logits(self.prompt(question), features)
        lse = lambda a: np.log(np.exp(a - a.max(axis=1, keepdims=True)).sum(axis=1)) + a.max(axis=1)  # noqa: E731
        return lse(z[:, self.yes]) - lse(z[:, self.no])


def ranked_moments(frozen, scores: np.ndarray, p: dict) -> list[tuple[int, float, float, float, float]]:
    """The search's first TOP moments for one description: (video, start, end, score, time of the best frame)."""
    best = np.maximum.reduceat(scores, frozen.starts)
    out = []
    for v in np.argsort(-best)[:SHORTLIST]:
        s, c = frozen.starts[v], frozen.counts[v]
        t, g = frozen.times[s : s + c].astype(np.float64), scores[s : s + c].astype(np.float64)
        for w in windows(t, g, float(frozen.durations[v]), p["window"], p["per_video"]):
            inside = np.flatnonzero((t >= w.start) & (t < w.end))
            out.append((int(v), w.start, w.end, w.peak, float(t[inside[np.argmax(g[inside])]])))
    out.sort(key=lambda x: -x[3])
    return out[:TOP]


def latency(call, repeat: int = 5) -> float:
    """Median wall-clock seconds of `call`, after one call to warm up."""
    call()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def found(order: np.ndarray, correct: np.ndarray, k: int) -> bool:
    return bool(correct[order[:k]].any())


def standardise(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(float(x.std()), 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-dir", required=True, help="SmolVLM-256M-Instruct, with its onnx/ folder")
    ap.add_argument("--videos", required=True, help="the tuning videos")
    ap.add_argument("--sample", type=int, default=600, help="descriptions, drawn at random once (seed 0)")
    ap.add_argument("--threads", type=int, default=0)
    a, rest = ap.parse_known_args()
    args = didemo.main_args(["tune", "--split", "train", *rest])
    frozen, queries, _ = didemo.load("train", args.annotations)
    text = didemo.text_embeddings(queries, "train", args.model_dir)
    scores = frozen.emb @ text.T - offsets(frozen.emb, didemo.bank(args), 50.0)[:, None]
    p = {"window": 10.0, "per_video": 3}

    sample = sorted(random.Random(0).sample(range(len(queries)), min(a.sample, len(queries))))
    moments = {qi: ranked_moments(frozen, scores[:, qi], p) for qi in sample}
    cache = didemo.CACHE / "didemo_train_vlm.npz"
    answers = {}
    if cache.exists():
        z = np.load(cache)
        answers = dict(zip(z["queries"].tolist(), z["answers"], strict=True))
    todo = [qi for qi in sample if qi not in answers]
    vlm = SmolVLM(a.vlm_dir, a.threads)
    files = {didemo.stem(f.name): f for f in Path(a.videos).glob("*.mp4")}

    def frame(v: int, t: float) -> np.ndarray:
        _, frames = sample_frames(files[didemo.stem(frozen.names[v])], 1.0, vlm.size, seconds=didemo.SECONDS)
        return frames[min(int(round(t)), len(frames) - 1)]

    if todo:
        needed = {}  # video -> the frames to look at
        for qi in todo:
            for v, *_, at in moments[qi]:
                needed.setdefault(v, set()).add(int(round(at)))
        features = {}
        t0 = time.time()
        for n, (v, ticks) in enumerate(sorted(needed.items())):
            _, frames = sample_frames(files[didemo.stem(frozen.names[v])], 1.0, vlm.size, seconds=didemo.SECONDS)
            ticks = sorted(ticks)
            f = vlm.see(frames[[min(t, len(frames) - 1) for t in ticks]])
            features.update({(v, t): f[i] for i, t in enumerate(ticks)})
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(needed)} videos seen, {time.time() - t0:.0f} s", flush=True)
        for n, qi in enumerate(todo):
            feats = np.stack([features[(v, int(round(at)))] for v, *_, at in moments[qi]])
            answers[qi] = vlm.yes_or_no(QUESTION.format(queries[qi]["text"].strip()), feats)
            if (n + 1) % 100 == 0:
                print(f"{n + 1}/{len(todo)} descriptions asked, {time.time() - t0:.0f} s", flush=True)
                np.savez(cache, queries=np.array(list(answers)), answers=np.stack(list(answers.values())))
        np.savez(cache, queries=np.array(list(answers)), answers=np.stack(list(answers.values())))

    correct = {}
    for qi in sample:
        q = queries[qi]
        correct[qi] = np.array([v == q["video"] and iou((s, e), q["moment"]) >= 0.5 for v, s, e, *_ in moments[qi]])

    def order(qi: int, w: float | None) -> np.ndarray:
        base = np.array([m[3] for m in moments[qi]])
        if w is None:
            return np.arange(len(base))
        if w == np.inf:
            return np.argsort(-answers[qi], kind="stable")
        return np.argsort(-(standardise(base) + w * standardise(answers[qi])), kind="stable")

    def share(qs: list[int], w: float | None, k: int = 5) -> np.ndarray:
        return np.array([found(order(qi, w), correct[qi], k) for qi in qs])

    rng = random.Random(1)
    halves = [sorted(h) for h in np.array_split(rng.sample(sample, len(sample)), 2)]
    chosen, held_out = [], []
    for i in (0, 1):  # choose the weight on one half, count on the other
        tune, held = halves[i], halves[1 - i]
        w = max(WEIGHTS, key=lambda w: share(tune, w).mean())
        chosen.append(w)
        held_out += [(qi, w) for qi in held]

    def fused(k: int) -> np.ndarray:
        return np.array([found(order(qi, w), correct[qi], k) for qi, w in held_out])

    every = [qi for qi, _ in held_out]
    base, cross, oracle = share(every, None), fused(5), float(share(every, None, TOP).mean())
    rows = {
        "model": "HuggingFaceTB/SmolVLM-256M-Instruct (ONNX, float32)",
        "question": QUESTION,
        "descriptions": len(sample),
        "moments_per_description": TOP,
        "orders": {
            "search": {"R@1": float(share(every, None, 1).mean()), "R@5": float(base.mean())},
            "model alone": {"R@1": float(share(every, np.inf, 1).mean()), "R@5": float(share(every, np.inf).mean())},
            "search + model": {"R@1": float(fused(1).mean()), "R@5": float(cross.mean())},
            "best possible": {"R@1": oracle, "R@5": oracle},
        },
        "weights chosen": chosen,
        "in-sample R@5 by weight": {str(w): float(share(sample, w).mean()) for w in WEIGHTS},
        "wins": int((cross & ~base).sum()),
        "losses": int((~cross & base).sum()),
        "p": sign_test(int((cross & ~base).sum()), int((~cross & base).sum())),
    }
    first = moments[sample[0]]  # what one more search costs: ten frames seen, one question each
    frames = np.stack([frame(v, at) for v, *_, at in first])
    question = QUESTION.format(queries[sample[0]]["text"].strip())
    rows["seconds per search"] = latency(lambda: vlm.yes_or_no(question, vlm.see(frames)))
    rows["threads"] = a.threads or os.cpu_count()
    out = ROOT / "results" / "didemo_train_rerank.json"
    out.write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
