"""The ninth round's stage 2 (docs/PROTOCOL.md): whenframe's model v4, v3's recipe unchanged on PE core B/16's frames.

v4 and its two left-out runs (without Charades-STA's videos, without YouCook2's) are trained by scripts/temporal.py
--encoder pe-core-b16 with v3's flags, as scripts/round7.py trained v3-leash30: no teacher, the leash's weight 30,
width 256 and two layers, data seed 0, 6,000 steps and a check every 500, the best check kept. v4 is exported as the
package would ship it (alpha 1), and each is measured on the four tuning sets (scripts/compare.py tune --round 9),
its speech and sound weights chosen there. results/round9_stage2.json records the runs.

    python scripts/round9.py --annotations didemo_train.json --model-dir ... --test-annotations ... (compare.py's)

A stopped run resumes: a run whose weights exist is not trained again (its logged recipe is checked), a candidate
already measured with the same weights is not measured again.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "temporal"
CACHE = ROOT / "results" / "cache"
TUNED9 = ROOT / "results" / "round9_tuned.json"
STAGE2 = ROOT / "results" / "round9_stage2.json"
ENCODER = "pe-core-b16"
V4 = OUT / f"temporal-{ENCODER}-v4.npz"  # v4, exported as the package would ship it
COLLECTIONS = ("msrvtt", "charades", "youcook2", "tempo")
# scripts/round7.py's flags, the seventh round's training
COMMON = ["--steps", "6000", "--every", "500", "--batch", "128", "--lr", "1e-4", "--warmup", "200",
          "--bank-sample", "1024", "--head", "0", "--within", "2", "--between-loss", "infonce", "--init-seed", "0",
          "--mix", "sqrt"]  # fmt: skip
# v3's recipe (results/round7_choices.json: v3-leash30, alpha 1, no soup), and its two left-out runs
RUNS = {"v4": [], "v4-no-charades": ["charades"], "v4-no-youcook2": ["youcook2"]}
# PE's frames of the four collections, and its text tower's embeddings of their descriptions
DATA = [f"{n}_{ENCODER}_openvino.npz" for n in ("didemo_all", "msrvtt", "charades_train", "youcook2_train")] + [
    f"{n}_text_{ENCODER}.npy" for n in ("didemo_all", "tempo", "msrvtt", "charades_train", "youcook2_train")
]


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


def config(without: list[str]) -> tuple[list[str], dict]:
    """A run's flags for scripts/temporal.py train, and the configuration its log must show."""
    extra = [c for c in COLLECTIONS if c not in without]
    flags = ["--extra", ",".join(extra), "--leash", "30", "--width", "256", "--layers", "2", "--heads", "4",
             "--seed", "0", "--encoder", ENCODER]  # fmt: skip
    if "charades" in extra:  # checked on Charades-STA's held-out videos only when it trains on Charades-STA
        flags.append("--check-charades")
    logged = {"extra": ",".join(extra), "leash": 30.0, "width": 256, "layers": 2, "heads": 4, "distil": 0.0,
              "check_charades": "charades" in extra, "seed": 0, "steps": 6000, "encoder": ENCODER}  # fmt: skip
    return flags, logged


class Stage2:
    def __init__(self, args):
        self.args = args
        self.paths = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
        self.compare = self.paths + [x for k in ("test_annotations", "charades_annotations", "youcook2_annotations",
                                                 "unav_annotations", "clap_dir") if getattr(args, k)
                                     for x in (f"--{k.replace('_', '-')}", getattr(args, k))]  # fmt: skip
        self.logs = OUT / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    def train(self, name: str, without: list[str]) -> Path:
        path = OUT / f"{name}.npz"
        flags, expected = config(without)
        if path.exists():  # trained before: with the same recipe, or a mistake to stop at
            c = json.loads((OUT / f"{name}.json").read_text())["config"]
            got = {k: c.get(k) for k in expected}
            assert got == expected, f"{path} was trained otherwise: {got} != {expected}"
            return path
        cmd = [sys.executable, "scripts/temporal.py", "train", *self.paths, "--name", name, *COMMON, *flags]
        log(f"train {name}: {' '.join(flags)}")
        with open(self.logs / f"{name}.log", "w") as out:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=out, stderr=subprocess.STDOUT)
        log(f"trained {name}")
        return path

    def tune(self, specs: list[str]) -> dict:
        t = json.loads(TUNED9.read_text()) if TUNED9.exists() else {}
        todo = [s for s in specs if t.get(s.split("=")[0], {}).get("spec") != s]
        if todo:
            log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
            env = dict(os.environ, OPENBLAS_NUM_THREADS="2")
            subprocess.run([sys.executable, "scripts/compare.py", "tune", "--round", "9", *self.compare,
                            "--candidates", ",".join(todo)], cwd=ROOT, check=True, env=env)  # fmt: skip
            t = json.loads(TUNED9.read_text())
        return t

    @staticmethod
    def spec(name: str, weights: Path) -> str:
        return f"{name}={ENCODER}:{weights.relative_to(ROOT)}@1"

    def run(self) -> None:
        while not all((CACHE / f).exists() for f in DATA):
            log("waiting for PE's frames and descriptions (compare.py extract, pe_core.py texts)")
            time.sleep(600)
        with ThreadPoolExecutor(self.args.parallel) as pool:
            weights = dict(zip(RUNS, pool.map(lambda kv: self.train(*kv), RUNS.items()), strict=True))
        before = V4.read_bytes() if V4.exists() else None
        subprocess.run([sys.executable, "scripts/temporal.py", "export", "--annotations", self.args.annotations,
                        "--encoder", ENCODER, "--model", str(weights["v4"]), "--out", str(V4)], cwd=ROOT,
                       check=True)  # fmt: skip
        if before is not None and before != V4.read_bytes() and TUNED9.exists():  # other weights: measured again
            t = json.loads(TUNED9.read_text())
            t.pop("v4", None)
            TUNED9.write_text(json.dumps(t, indent=1))
        specs = {"v4": self.spec("v4", V4), **{n: self.spec(n, weights[n]) for n in RUNS if n != "v4"}}
        t = self.tune(list(specs.values()))
        runs = {}
        for n, w in weights.items():
            log_ = json.loads(w.with_suffix(".json").read_text())
            runs[n] = {"weights": str(w.relative_to(ROOT)), "spec": specs[n], "best_step": log_["best_step"],
                       "best": log_["best"], "tuning": {k: t[n][k] for k in ("didemo", "charades", "youcook2",
                                                                            "unav", "mean")}}  # fmt: skip
        STAGE2.write_text(json.dumps({"encoder": ENCODER, "v4": str(V4.relative_to(ROOT)), "runs": runs}, indent=1))
        for n, r in runs.items():
            log(f"{n}: best check at step {r['best_step']}; tuning sets' mean {100 * r['tuning']['mean']:.2f}")
        log("stage 2 is trained: record it in docs/PROTOCOL.md, then run the test once")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", required=True, help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--parallel", type=int, default=2, help="runs trained at once")
    Stage2(ap.parse_args()).run()


if __name__ == "__main__":
    main()
