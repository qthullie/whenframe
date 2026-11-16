"""Other ways of scoring frames, tried on the tuning videos only after the first test run.

Every variant changes how frame scores are computed or ranked, and is compared
with the default search (fixed 10-second windows, query bank beta 50, lambda 1)
on the same descriptions, one by one: the table gives the share of descriptions
whose moment is in the first five results, the change, how many descriptions
it wins and loses, and a two-sided sign test on those.

    python scripts/explore.py          # needs the caches of `make didemo-extract` and `make didemo-tune`
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import PRIMARY, run  # noqa: E402

from whenframe.bank import offsets  # noqa: E402

WINDOWS = {"window": 10.0, "per_video": 3}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sign_test(win: int, loss: int) -> float:
    n = win + loss
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(win, loss) + 1)) / 2**n)


def main() -> None:
    args = didemo.main_args(["tune", "--split", "train", *sys.argv[1:]])
    frozen, queries, _ = didemo.load("train", args.annotations)
    text = didemo.text_embeddings(queries, "train", args.model_dir)
    bank = didemo.bank(args)
    emb = frozen.emb
    raw = emb @ text.T
    off = offsets(emb, bank, 50.0)
    rows = []

    def measure(name: str, source: str, scores: np.ndarray, rank=None) -> None:
        r = run(frozen, queries, text, "frame_window", WINDOWS, scores, per_query=True)
        found = np.array(r["found"])
        if not rows:
            measure.base = found
        base = measure.base
        win, loss = int((found & ~base).sum()), int((~found & base).sum())
        rows.append({"variant": name, "source": source, "score": r[PRIMARY[0]][PRIMARY[1]],
                     "change": float(found.mean() - base.mean()), "wins": win, "losses": loss,
                     "p": sign_test(win, loss)})  # fmt: skip
        change = 100 * (found.mean() - base.mean())
        print(f"{name:52s} {100 * found.mean():5.1f}  {change:+5.1f}  +{win}/-{loss}", flush=True)

    measure("default search", "", raw - off[:, None])
    measure("no query bank", "", raw)

    sims = emb @ bank.T
    top = -np.sort(-sims, axis=1)
    for k in (8, 32, 128):  # Nearest Neighbor Normalization: the mean of the k best bank scores
        measure(f"bias = mean of the {k} best bank scores", "NNN, arXiv 2410.24114",
                raw - top[:, :k].mean(axis=1)[:, None])  # fmt: skip
    del top

    hubs = np.zeros(len(emb), dtype=bool)  # dynamic inverted softmax: only frames some bank text ranks first
    hubs[np.argmax(sims, axis=0)] = True
    measure("correct only frames ranked first by a bank text", "QB-Norm, arXiv 2112.12777",
            raw - (off * hubs)[:, None])  # fmt: skip
    del sims

    mu_img, mu_txt = emb.mean(axis=0), bank.mean(axis=0)  # the modality gap as a mean shift
    emb_c, text_c, bank_c = unit(emb - mu_img), unit(text - mu_txt), unit(bank - mu_txt)
    measure("centre both modalities, then the bank", "GR-CLIP, arXiv 2507.19054",
            emb_c @ text_c.T - offsets(emb_c, bank_c, 20.0)[:, None])  # fmt: skip

    cache = didemo.CACHE / "didemo_bank_full_text.npy"
    if not cache.exists():
        np.save(cache, didemo.encode(didemo.bank_texts(args, size=None), args.model_dir))
    full = np.load(cache)
    measure(f"bank of {len(full):,} descriptions instead of 4,096", "", raw - offsets(emb, full, 50.0)[:, None])

    video_of = np.repeat(np.arange(len(frozen.starts)), frozen.counts)
    same_prev = np.r_[False, video_of[1:] == video_of[:-1]]
    same_next = np.r_[video_of[:-1] == video_of[1:], False]
    smooth = emb.copy()
    smooth[same_prev] += emb[np.flatnonzero(same_prev) - 1]
    smooth[same_next] += emb[np.flatnonzero(same_next) + 1]
    smooth = unit(smooth)
    measure("each frame averaged with its neighbours", "", smooth @ text.T - offsets(smooth, bank, 50.0)[:, None])

    per_video = np.add.reduceat(off, frozen.starts) / frozen.counts
    measure("also subtract the video's mean offset (x 0.5)", "",
            raw - off[:, None] - 0.5 * per_video[video_of][:, None])  # fmt: skip

    corr = raw - off[:, None]
    best = np.argpartition(-corr, 10, axis=0)[:10]  # alpha query expansion with the 10 best frames
    centroid = unit(emb[best].mean(axis=0))
    measure("add 0.05 x similarity to the 10 best frames", "alpha-QE", corr + 0.05 * (emb @ centroid.T))

    out = ROOT / "results" / "didemo_train_explore.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
