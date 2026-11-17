"""Motion: four frames of each second in one image, a 2x2 grid, scored with the frame (fourth round).

SigLIP sees still images: "jumps into the pool" or "the camera zooms in" leave
little trace in a single frame. Here each second of a video is also shown as a
grid of four frames a quarter of a second apart, which SigLIP 2 reads as one
224-pixel image, as grids of frames are given to vision-language models
(IG-VLM, Kim et al. 2024); the grid's score, corrected by the query bank as the
frame's is, is added to the frame's with a weight chosen on the tuning videos.

    python scripts/motion.py extract --split train --videos ... --annotations ...
    python scripts/motion.py tune --split train --annotations ... --train-annotations ...
    python scripts/motion.py extract --split test ...
    python scripts/motion.py evaluate --split test ...   # once, with the weight chosen on the tuning videos

Writes results/didemo_train_motion.json: the weights tried, on all the tuning
descriptions and on those that say how something moves (MOTION below, fixed
before any grid was scored); then results/didemo_test_motion.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import PRIMARY, frame_scores, run  # noqa: E402
from explore import sign_test  # noqa: E402

from whenframe.video import sample_frames  # noqa: E402

SUB = 4  # frames per grid, a quarter of a second apart
WEIGHTS = (0.25, 0.5, 0.75, 1.0)
# descriptions of movement: of a person, an animal, a thing or the camera (whole words; "left", "waves" or
# "lower" are left out, as they more often say where or what)
MOTION = re.compile(
    r"\b(jump(s|ed|ing)?|run(s|ning)?|ran|walk(s|ed|ing)?|turn(s|ed|ing)?|pan(s|ned|ning)?|zoom(s|ed|ing)?|"
    r"fall(s|ing)?|fell|throw(s|n|ing)?|threw|spin(s|ning)?|spun|danc(e|es|ed|ing)|climb(s|ed|ing)?|"
    r"mov(e|es|ed|ing)|enter(s|ed|ing)?|exit(s|ed|ing)?|swim(s|ming)?|swam|rid(e|es|ing)|rode|"
    r"driv(e|es|ing)|drove|fl(y|ies|ew|ying)|kick(s|ed|ing)?|roll(s|ed|ing)?|slid(e|es|ing)?|wav(e|ed|ing)|"
    r"shak(e|es|ing)|shook|sit(s|ting)?|sat|stand(s|ing)?|stood|lift(s|ed|ing)?|push(es|ed|ing)?|"
    r"pull(s|ed|ing)?|catch(es|ing)?|caught|hit(s|ting)?|swing(s|ing)?|swung|tilt(s|ed|ing)?|"
    r"rotat(e|es|ed|ing)|bounc(e|es|ed|ing)|chas(e|es|ed|ing)|skat(e|es|ed|ing)|ski(s|ing|er|ers)?|"
    r"surf(s|ed|ing)?|hop(s|ped|ping)?|crawl(s|ed|ing)?|step(s|ped|ping)?|lean(s|ed|ing)?|bend(s|ing)?|bent|"
    r"nod(s|ded|ding)?|rais(e|es|ed|ing))\b",
    re.IGNORECASE,
)


def grids(frames: np.ndarray) -> np.ndarray:
    """(n, 2h, 2w, 3) grids of SUB consecutive (h, w) frames: first two on top, the next two below."""
    n = len(frames) // SUB
    f = frames[: n * SUB].reshape(n, 2, 2, *frames.shape[1:])
    return np.concatenate([np.concatenate([f[:, r, 0], f[:, r, 1]], axis=2) for r in range(2)], axis=1)


def extract(args) -> None:
    """For every frame of the split's cache, the grid of the second it starts."""
    frozen, _, _ = didemo.load(args.split, args.annotations)
    enc = didemo.make_encoder(args.model_dir, didemo.MAIN, "openvino")
    enc.images  # noqa: B018 - compiled before timing
    half = enc.size // 2
    out = np.zeros((len(frozen.times), enc.dim), dtype=np.float16)
    cost = {"decode_cpu": 0.0, "encode_cpu": 0.0}
    t0 = time.time()
    for v, name in enumerate(frozen.names):
        c0 = didemo.children_cpu() + time.process_time()
        t, frames = sample_frames(Path(args.videos) / name, float(SUB), half, seconds=didemo.SECONDS)
        c1 = didemo.children_cpu() + time.process_time()
        g = grids(frames)
        e = enc.encode_images(g) if len(g) else np.zeros((0, enc.dim), np.float32)
        cost["decode_cpu"] += c1 - c0
        cost["encode_cpu"] += didemo.children_cpu() + time.process_time() - c1
        s, c = frozen.starts[v], frozen.counts[v]
        starts = t[: len(g) * SUB : SUB]
        for row in range(s, s + c):  # the grid starting closest to the frame
            if len(g):
                out[row] = e[np.abs(starts - frozen.times[row]).argmin()]
        if (v + 1) % 50 == 0:
            print(f"{v + 1}/{len(frozen.names)} videos, {time.time() - t0:.0f} s", flush=True)
    path = didemo.CACHE / f"didemo_{args.split}_motion.npz"
    np.savez_compressed(path, emb=out, names=np.array(frozen.names), **{k: np.array(x) for k, x in cost.items()})
    print(f"wrote {path}: {len(out)} grids, {cost['encode_cpu'] / 60:.1f} CPU min to embed")


def scores(args):
    """The split's descriptions, and the frames' and the grids' corrected scores, as the tool would rank them."""
    frozen, queries, _ = didemo.load(args.split, args.annotations)
    frozen.bank = didemo.bank(args)
    text = didemo.text_embeddings(queries, args.split, args.model_dir)
    z = np.load(didemo.CACHE / f"didemo_{args.split}_motion.npz")
    assert z["names"].tolist() == frozen.names
    # the frames as the tool embeds them, with OpenVINO, and their grids
    ov = np.load(didemo.CACHE / f"didemo_{args.split}{didemo.suffix(didemo.MAIN, 'openvino')}.npz")
    base = replace(frozen, emb=ov["emb"][ov["times"] < didemo.SECONDS].astype(np.float32), _offsets={})
    kept = base.only_kept()
    grid = replace(kept, emb=z["emb"][base.keep].astype(np.float32), _offsets={})
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    method = tuned["default"]
    p = tuned[method]["params"]
    still, moving = frame_scores(kept, text, method, p), frame_scores(grid, text, method, p)
    motion = np.array([bool(MOTION.search(q["text"])) for q in queries])
    cost = {k: float(z[k]) for k in ("decode_cpu", "encode_cpu")}
    cost["hours"] = float(frozen.durations.sum() / 3600)
    return kept, queries, text, method, p, still, moving, motion, cost


def measure(kept, queries, text, method, p, variants, motion) -> list[dict]:
    rows = []
    for name, w, s in variants:
        r = run(kept, queries, text, method, p, scores=s, per_query=True)
        found = np.array(r.pop("found"))
        rows.append({"variant": name, "weight": w, "score": r[PRIMARY[0]][PRIMARY[1]], "found": found,
                     "motion": float(found[motion].mean()), "other": float(found[~motion].mean()),
                     "corpus": r["corpus"], "video": r["video"]})  # fmt: skip
        print(f"{name:24s} R@5 {100 * rows[-1]['score']:.1f}   motion {100 * rows[-1]['motion']:.1f}   "
              f"other {100 * rows[-1]['other']:.1f}", flush=True)  # fmt: skip
    first = rows[0]["found"]
    for r in rows:
        f = r.pop("found")
        win, loss = int((f & ~first).sum()), int((~f & first).sum())
        mw, ml = int((f & ~first & motion).sum()), int((~f & first & motion).sum())
        r.update(wins=win, losses=loss, p=sign_test(win, loss), motion_wins=mw, motion_losses=ml,
                 motion_p=sign_test(mw, ml))  # fmt: skip
    return rows


def tune(args) -> None:
    kept, queries, text, method, p, still, moving, motion, cost = scores(args)
    variants = [("frames", 0.0, still), *((f"frames + {w:g} x grids", w, still + w * moving) for w in WEIGHTS),
                ("grids alone", None, moving)]  # fmt: skip
    rows = measure(kept, queries, text, method, p, variants, motion)
    out = {"split": args.split, "queries": len(queries), "motion_queries": int(motion.sum()), "method": method,
           "params": p, "cost": cost, "rows": rows}  # fmt: skip
    path = ROOT / "results" / f"didemo_{args.split}_motion.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"{int(motion.sum())} of {len(queries)} descriptions say how something moves; wrote {path.relative_to(ROOT)}")


def evaluate(args) -> None:
    tuned = json.loads((ROOT / "results" / "didemo_train_motion.json").read_text())
    w = max((r for r in tuned["rows"] if r["weight"]), key=lambda r: r["score"])["weight"]
    kept, queries, text, method, p, still, moving, motion, cost = scores(args)
    rows = measure(kept, queries, text, method, p, [("frames", 0.0, still), (f"frames + {w:g} x grids", w,
                                                                             still + w * moving)], motion)  # fmt: skip
    out = {"split": args.split, "queries": len(queries), "motion_queries": int(motion.sum()), "weight": w,
           "method": method, "params": p, "cost": cost, "rows": rows}  # fmt: skip
    path = ROOT / "results" / f"didemo_{args.split}_motion.json"
    path.write_text(json.dumps(out, indent=1))
    r = rows[1]
    print(f"grids +{r['wins']} / -{r['losses']} (p = {r['p']:.2g}); wrote {path.relative_to(ROOT)}")


def main() -> None:
    stage = sys.argv[1]
    args = didemo.main_args(["evaluate", *sys.argv[2:]])
    {"extract": extract, "tune": tune, "evaluate": evaluate}[stage](args)


if __name__ == "__main__":
    main()
