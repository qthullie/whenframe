"""The thirteenth round (docs/PROTOCOL.md): a small model, for phones. Stage 1, on the tuning sets.

    python scripts/compare.py extract --dataset D --models pe-core-s16-224,pe-core-s16-384,pe-core-t16-224,...
                                                   # D: didemo-train, charades-tuning, youcook2, unav
    python scripts/round13.py bank ARGS            # each candidate's query bank and spans' strength
    python scripts/round13.py tune ARGS            # the four candidates on the four tuning sets
    python scripts/round13.py cost --videos DIDEMO_TEST_VIDEOS   # indexing costs, the current model's too
    python scripts/round13.py stage1               # sizes, text times, and the rule: results/round13_stage1.json

ARGS are scripts/compare.py's (--annotations, --youcook2-annotations, --unav-annotations, --clap-dir, ...).

`bank` embeds DiDeMo's query bank with each candidate's text tower and chooses, as the ninth round chose PE core
B/16's (scripts/pe_core.py bank): the bank's beta and strength on DiDeMo's tuning videos over the fourth round's
grid, then the bank's strength for adaptive spans on YouCook2's tuning steps (results/round13_bank.json). `tune`
measures each candidate with those settings, its speech and sound weights chosen there (scripts/compare.py tune,
results/round13_tuned.json). `cost` measures the indexing cost of each candidate and of the current model, one
after the other, as the eleventh round measured it (results/round13_cost.json). `stage1` adds each candidate's
files and its text tower's median time per description, and applies the rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CANDIDATES = ("pe-core-s16-224", "pe-core-s16-384", "pe-core-t16-224", "pe-core-t16-384")
CURRENT = "siglip2-base+pe-core-b16"  # the current model: both encoders, each frame decoded once
BANK = ROOT / "results" / "round13_bank.json"
STAGE1 = ROOT / "results" / "round13_stage1.json"
MAX_MB = 150.0  # the protocol's limits for a candidate to be kept
MAX_COST_SHARE = 0.5


def chosen_bank() -> dict:
    return json.loads(BANK.read_text()) if BANK.exists() else {}


def with_banks() -> None:
    """Each candidate's spec with the bank settings chosen for it (results/round13_bank.json)."""
    from whenframe import encoder

    for model, b in chosen_bank().items():
        encoder.MODELS[model] = replace(encoder.MODELS[model], bank_beta=b["bank"]["beta"],
                                        bank_lambda=b["bank"]["lam"], spans_lambda=b["spans"]["lam"])  # fmt: skip


def bank(args) -> None:
    import itertools

    import compare
    import didemo
    from benchmark import SPACE, run

    from whenframe import encoder
    from whenframe.encoder import Encoder

    out = chosen_bank()
    b = compare.Benchmarks(args)
    method = "frame_window_bank"
    params = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())[method]["params"]
    for model in CANDIDATES:
        if model in out:
            continue
        path = compare.CACHE / f"didemo_bank_text_{model}.npy"
        if not path.exists():  # DiDeMo's bank, embedded by the candidate's text tower as the tool embeds a query
            texts = didemo.bank_texts(Namespace(train_annotations=args.annotations))
            enc = Encoder(model, backend="openvino")
            emb = np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)])
            np.save(path, emb.astype(np.float32))
        c = compare.Candidate(model, model)
        f, queries, text = b.didemo(c, "train")
        grid = []
        for beta, lam in itertools.product(SPACE["bank"]["beta"], SPACE["bank"]["lam"]):
            r = run(f, queries, text, method, {**params, "beta": beta, "lam": lam})
            grid.append({"beta": beta, "lam": lam, "score": r["corpus"]["R@5 IoU>=0.5"]})
        best = max(grid, key=lambda x: x["score"])
        print(f"{model}: DiDeMo tuning, beta {best['beta']:g}, strength {best['lam']:g}: "
              f"R@5 {100 * best['score']:.1f}", flush=True)  # fmt: skip
        spec, spans = encoder.MODELS[model], []
        for lam in SPACE["bank"]["lam"]:
            encoder.MODELS[model] = replace(spec, bank_beta=best["beta"], bank_lambda=best["lam"], spans_lambda=lam)
            b._cache = {k: v for k, v in b._cache.items() if model not in map(str, k)}  # nothing kept from before
            y = b.youcook2_score(c, "tuning")
            spans.append({"lam": lam, "score": y["value"], "speech": y["speech"]})
        encoder.MODELS[model] = spec
        pick = max(spans, key=lambda x: x["score"])
        print(f"{model}: YouCook2 tuning, spans' strength {pick['lam']:g}: R@5 {100 * pick['score']:.1f}", flush=True)
        out[model] = {"bank": {"beta": best["beta"], "lam": best["lam"], "score": best["score"], "grid": grid},
                      "spans": {"lam": pick["lam"], "score": pick["score"], "grid": spans}}  # fmt: skip
        BANK.write_text(json.dumps(out, indent=1))


def tune(args) -> None:
    import compare

    with_banks()
    args.round = 13
    args.candidates = ",".join(f"{m}={m}:" for m in CANDIDATES)
    compare.tune(args)


def cost(args) -> None:
    """The candidates' and the current model's indexing costs, one process each after the other, on an idle
    machine (scripts/compare.py cost --round 13)."""
    for model in (*CANDIDATES, CURRENT):
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "compare.py"), "cost", "--round", "13",
                        "--videos", args.videos, "--models", model], check=True)  # fmt: skip


def text_time(model: str, texts: list[str]) -> float:
    """The text tower's median time to embed one description, one at a time, after a first one (compiled)."""
    from whenframe.encoder import Encoder

    enc = Encoder(model, backend="openvino")
    enc.encode_texts(["a warm-up"])
    times = []
    for t in texts:
        t0 = time.perf_counter()
        enc.encode_texts([t])
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def files_mb(model: str) -> float:
    from whenframe.encoder import MODELS, cache_dir

    folder = cache_dir() / "models" / MODELS[model].repo.replace("/", "--")
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6


def stage1(args) -> None:
    import didemo

    tuned = json.loads((ROOT / "results" / "round13_tuned.json").read_text())
    costs = json.loads((ROOT / "results" / "round13_cost.json").read_text())
    _, queries, _ = didemo.load("train", args.annotations)
    texts = [q["text"] for q in queries[:200]]
    limit = MAX_COST_SHARE * costs[CURRENT]["cpu_min_per_hour"]
    rows = {}
    for model in CANDIDATES:
        rows[model] = {"mean": tuned[model]["mean"], "mb": files_mb(model),
                       "cost": costs[model]["cpu_min_per_hour"], "text_ms": 1000 * text_time(model, texts)}  # fmt: skip
        rows[model]["within"] = rows[model]["mb"] <= MAX_MB and rows[model]["cost"] <= limit
    within = [m for m in CANDIDATES if rows[m]["within"]]
    kept = max(within, key=lambda m: rows[m]["mean"]) if within else None
    out = {"current_cost": costs[CURRENT]["cpu_min_per_hour"], "cost_limit": limit, "mb_limit": MAX_MB,
           "candidates": rows, "kept": kept}  # fmt: skip
    STAGE1.write_text(json.dumps(out, indent=1))
    for m, r in rows.items():
        print(
            f"{m}: tuning mean {100 * r['mean']:.2f}, {r['mb']:.1f} MB, {r['cost']:.2f} CPU min per hour "
            f"(limit {limit:.2f}), text {r['text_ms']:.1f} ms{'' if r['within'] else ', outside the limits'}"
        )
    print(f"kept: {kept}" if kept else "none within the limits: the round ends here")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "tune", "cost", "stage1"])
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--test-annotations")
    ap.add_argument("--charades-annotations")
    ap.add_argument("--youcook2-annotations")
    ap.add_argument("--unav-annotations")
    ap.add_argument("--clap-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    args = ap.parse_args()
    {"bank": bank, "tune": tune, "cost": cost, "stage1": stage1}[args.stage](args)


if __name__ == "__main__":
    main()
