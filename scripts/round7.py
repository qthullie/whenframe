"""The seventh round's choices on the tuning sets (docs/PROTOCOL.md), made one after the other by its rule.

Each choice trains runs of scripts/temporal.py, measures them on the four tuning sets (scripts/compare.py tune
--round 7) and keeps a run only if it raises the mean of the four primary metrics; results/round7_choices.json
records every step. Then v3's recipe without Charades-STA's videos and without YouCook2's, reported beside.

    python scripts/round7.py --annotations didemo_train.json --model-dir ... --test-annotations ... (compare.py's)

A stopped run resumes: a run whose weights exist is not trained again, a candidate already measured with the
same weights is not measured again.
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
TUNED7 = ROOT / "results" / "round7_tuned.json"
CHOICES = ROOT / "results" / "round7_choices.json"
V2 = "v2=results/temporal/temporal-siglip2-base-v2.npz@1"
V3 = OUT / "temporal-siglip2-base-v3.npz"  # the kept run, exported as the package ships it
COLLECTIONS = ("msrvtt", "charades", "youcook2", "tempo")
# the sixth round's training, as v2's, over 6,000 steps with a check every 500 (the seventh round's)
COMMON = ["--steps", "6000", "--every", "500", "--batch", "128", "--lr", "1e-4", "--warmup", "200",
          "--bank-sample", "1024", "--head", "0", "--within", "2", "--between-loss", "infonce", "--init-seed", "0",
          "--mix", "sqrt"]  # fmt: skip
START = {"without": [], "leash": 100.0, "width": 256, "layers": 2, "heads": 4, "distil": 0.0, "teacher": []}
# So400m's DiDeMo and MSR-VTT frames and descriptions (scripts/moments_data.py teacher, texts --teacher-texts)
TEACHER_FILES = ["didemo_teacher.npz", "msrvtt_teacher.npz", "didemo_all_text_so400m.npy", "msrvtt_text_so400m.npy",
                 "tempo_text_so400m.npy"]  # fmt: skip


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


class Round:
    def __init__(self, args):
        self.args = args
        self.paths = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
        self.compare = self.paths + [x for k in ("test_annotations", "charades_annotations", "youcook2_annotations",
                                                 "unav_annotations", "clap_dir") if getattr(args, k)
                                     for x in (f"--{k.replace('_', '-')}", getattr(args, k))]  # fmt: skip
        self.logs = OUT / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.steps = json.loads(CHOICES.read_text())["steps"] if CHOICES.exists() else []

    # ------------------------------------------------------------ training
    @staticmethod
    def config(r: dict, seed: int) -> tuple[list[str], dict]:
        """A recipe's flags for scripts/temporal.py train, and the configuration its run logs."""
        extra = [c for c in COLLECTIONS if c not in r["without"]]
        teacher = [c for c in r["teacher"] if c == "didemo" or c in extra] if r["distil"] else []
        f = ["--extra", ",".join(extra), "--leash", f"{r['leash']:g}", "--width", str(r["width"]),
             "--layers", str(r["layers"]), "--heads", str(r["heads"])]  # fmt: skip
        if teacher:
            f += ["--distil", f"{r['distil']:g}", "--teacher", ",".join(teacher)]
        if "charades" in extra:  # checked on Charades-STA's held-out videos only when it trains on Charades-STA
            f.append("--check-charades")
        logged = {"extra": ",".join(extra), "leash": float(r["leash"]), "width": r["width"], "layers": r["layers"],
                  "heads": r["heads"], "distil": float(r["distil"]) if teacher else 0.0, "teacher": ",".join(teacher),
                  "check_charades": "charades" in extra, "seed": seed, "steps": 6000}  # fmt: skip
        return f, logged

    def train(self, name: str, recipe: dict, seed: int = 0) -> Path:
        path = OUT / f"{name}.npz"
        flags, expected = self.config(recipe, seed)
        if path.exists():  # trained before: with the same recipe, or a mistake to stop at
            c = json.loads((OUT / f"{name}.json").read_text())["config"]
            got = {k: c[k] for k in expected}
            got["teacher"] = c["teacher"] if c["distil"] else ""
            assert got == expected, f"{path} was trained otherwise: {got} != {expected}"
            return path
        cmd = [sys.executable, "scripts/temporal.py", "train", *self.paths, "--name", name, "--seed", str(seed),
               *COMMON, *flags]  # fmt: skip
        log(f"train {name}: {' '.join(flags)} --seed {seed}")
        with open(self.logs / f"{name}.log", "w") as out:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=out, stderr=subprocess.STDOUT)
        log(f"trained {name}")
        return path

    def train_all(self, runs: list[tuple[str, dict, int]]) -> list[Path]:
        with ThreadPoolExecutor(self.args.parallel) as pool:
            return list(pool.map(lambda r: self.train(*r), runs))

    def seeds(self, name: str, recipe: dict) -> list[Path]:
        """The recipe's run (data seed 0) and, for a soup, its runs with data seeds 1 and 2."""
        return self.train_all([(name if s == 0 else f"{name}-s{s}", recipe, s) for s in (0, 1, 2)])

    def soup(self, name: str, runs: list[Path]) -> Path:
        path = OUT / f"{name}.npz"
        if not path.exists():
            subprocess.run([sys.executable, "scripts/temporal.py", "soup", "--annotations", self.args.annotations,
                            "--name", name, "--models", ",".join(map(str, runs))], cwd=ROOT, check=True)  # fmt: skip
        return path

    # ------------------------------------------------------------ measuring
    def tune(self, specs: list[str]) -> dict:
        t = json.loads(TUNED7.read_text()) if TUNED7.exists() else {}
        todo = [s for s in specs if t.get(s.split("=")[0], {}).get("spec") != s]
        if todo:
            log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
            env = dict(os.environ, OPENBLAS_NUM_THREADS="2")
            subprocess.run([sys.executable, "scripts/compare.py", "tune", "--round", "7", *self.compare,
                            "--candidates", ",".join(todo)], cwd=ROOT, check=True, env=env)  # fmt: skip
            t = json.loads(TUNED7.read_text())
        return t

    @staticmethod
    def spec(name: str, weights: Path, alpha: float = 1.0) -> str:
        return f"{name}={weights.relative_to(ROOT)}@{alpha:g}"

    def record(self, choice: int, name: str, recipe: dict, spec: str, mean: float, against: str, against_mean: float,
               kept: bool) -> None:  # fmt: skip
        step = {"choice": choice, "run": name, "recipe": recipe, "spec": spec, "mean": mean, "against": against,
                "against_mean": against_mean, "kept": kept}  # fmt: skip
        self.steps = [s for s in self.steps if s["run"] != name] + [step]
        CHOICES.write_text(json.dumps({"steps": self.steps}, indent=1))
        log(f"choice {choice}: {name} {100 * mean:.2f} against {against} {100 * against_mean:.2f}: "
            f"{'kept' if kept else 'not kept'}")  # fmt: skip

    # ------------------------------------------------------------ the round
    def run(self) -> None:
        data = ["charades_train_openvino.npz", "charades_tuning_openvino.npz", "youcook2_train_openvino.npz",
                "charades_train_text_so400m.npy", "youcook2_train_text.npy", "tempo_text.npy"]  # fmt: skip
        while not all((CACHE / f).exists() for f in data):
            log("waiting for the training data (scripts/moments_data.py)")
            time.sleep(600)
        teach = dict(START, distil=1.0, teacher=["charades"])
        a, b = self.train_all([("v3-noteacher", START, 0), ("v3-teacher", teach, 0)])
        # with SigLIP 2 B/16 and v2 as they are, the references
        t = self.tune(["base", V2, self.spec("v3-noteacher", a), self.spec("v3-teacher", b)])
        current, recipe, weights = "v3-noteacher", START, a

        def compare(choice: int, name: str, new_recipe: dict, new_weights: Path) -> None:
            nonlocal current, recipe, weights, t
            spec = self.spec(name, new_weights)
            t = self.tune([spec])
            kept = t[name]["mean"] > t[current]["mean"]
            self.record(choice, name, new_recipe, spec, t[name]["mean"], current, t[current]["mean"], kept)
            if kept:
                current, recipe, weights = name, new_recipe, new_weights

        self.record(1, "v3-noteacher", START, self.spec("v3-noteacher", a), t["v3-noteacher"]["mean"],
                    "v3-noteacher", t["v3-noteacher"]["mean"], True)  # fmt: skip
        compare(1, "v3-teacher", teach, b)
        if current == "v3-teacher":  # then So400m's frames of DiDeMo and MSR-VTT too
            while not all((CACHE / f).exists() for f in TEACHER_FILES):
                log("waiting for the teacher's DiDeMo and MSR-VTT frames")
                time.sleep(600)
            every = dict(teach, teacher=["charades", "didemo", "msrvtt"])
            compare(1, "v3-teacher-all", every, self.train("v3-teacher-all", every))
        leash = dict(recipe, leash=30.0)
        compare(2, "v3-leash30", leash, self.train("v3-leash30", leash))
        wide = dict(recipe, width=512, layers=4, heads=8)  # 64 dimensions a head, as at width 256
        compare(3, "v3-wide", wide, self.train("v3-wide", wide))
        single, single_recipe = current, recipe
        runs = self.seeds(single, single_recipe)
        compare(4, "v3-soup", single_recipe, self.soup("v3-soup", runs))
        souped = current == "v3-soup"
        # alpha: the best of 0.25, 0.5, 0.75 and 1 (the run kept so far)
        specs = [self.spec(f"v3-a{x:g}", weights, x) for x in (0.25, 0.5, 0.75)]
        t = self.tune(specs)
        alpha, best = 1.0, current
        for x in (0.25, 0.5, 0.75):
            if t[f"v3-a{x:g}"]["mean"] > t[best]["mean"]:
                alpha, best = x, f"v3-a{x:g}"
        for x, s in zip((0.25, 0.5, 0.75), specs, strict=True):
            self.record(5, f"v3-a{x:g}", recipe, s, t[f"v3-a{x:g}"]["mean"], current, t[current]["mean"],
                        best == f"v3-a{x:g}")  # fmt: skip
        log(f"kept: {current} at alpha {alpha:g}")
        before = V3.read_bytes() if V3.exists() else None
        subprocess.run([sys.executable, "scripts/temporal.py", "export", "--annotations", self.args.annotations,
                        "--model", str(weights), "--alpha", f"{alpha:g}", "--out", str(V3)], cwd=ROOT,
                       check=True)  # fmt: skip
        if before is not None and before != V3.read_bytes() and TUNED7.exists():  # other weights: measured again
            t = json.loads(TUNED7.read_text())
            t.pop("v3", None)
            TUNED7.write_text(json.dumps(t, indent=1))
        t = self.tune([self.spec("v3", V3)])
        final = {"run": current, "single": single, "recipe": recipe, "soup": souped, "alpha": alpha,
                 "weights": str(weights.relative_to(ROOT)), "v3": str(V3.relative_to(ROOT)),
                 "mean": t["v3"]["mean"]}  # fmt: skip
        CHOICES.write_text(json.dumps({"steps": self.steps, "final": final}, indent=1))
        # beside, deciding nothing: v3's recipe without one collection, measured on the test it left out
        left = {"v3-no-charades": dict(recipe, without=["charades"]),
                "v3-no-youcook2": dict(recipe, without=["youcook2"])}  # fmt: skip
        runs = self.train_all([(n if s == 0 else f"{n}-s{s}", r, s) for n, r in left.items()
                               for s in ((0, 1, 2) if souped else (0,))])  # fmt: skip
        specs = []
        for i, n in enumerate(left):
            w = self.soup(f"{n}-soup", runs[3 * i : 3 * i + 3]) if souped else runs[i]
            specs.append(self.spec(n, w, alpha))
        self.tune(specs)
        final["left_out"] = specs
        CHOICES.write_text(json.dumps({"steps": self.steps, "final": final}, indent=1))
        log("the seventh round's choices are made: record them in docs/PROTOCOL.md, then run the test once")


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", required=True, help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--parallel", type=int, default=2, help="runs trained at once")
    Round(ap.parse_args()).run()


if __name__ == "__main__":
    main()
