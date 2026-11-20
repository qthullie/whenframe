"""The eighth round's choices on the tuning sets (docs/PROTOCOL.md), made one after the other by its rule, then the
test libraries learnt with the recipe chosen.

Each library is learnt by its starting model from Florence-2's descriptions of its frames (scripts/learn.py), with
scripts/temporal.py train --library; a recipe is measured on the four tuning sets, each searched through the model
its library learnt (scripts/compare.py tune --round 8), and kept only if it raises the mean of the four primary
metrics. results/round8_choices.json records every step.

    python scripts/round8.py choose     --annotations didemo_train.json ... (compare.py's)   # once described
    python scripts/round8.py learn-test --annotations didemo_train.json ... (compare.py's)   # then, before the test

A stopped run resumes: a library learnt with the same recipe is not learnt again, a candidate already measured with
the same weights is not measured again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "temporal"
TUNED8 = ROOT / "results" / "round8_tuned.json"
CHOICES = ROOT / "results" / "round8_choices.json"
BENCHES = ("didemo", "charades", "youcook2", "unav")
# each library's starting model (the seventh round's), and the collections it was trained on, for choice 3's mix
START = {"didemo": ("v3-leash30", "msrvtt,charades,youcook2,tempo"),
         "charades": ("v3-no-charades", "msrvtt,youcook2,tempo"),
         "youcook2": ("v3-no-youcook2", "msrvtt,charades,tempo"),
         "unav": ("v3-leash30", "msrvtt,charades,youcook2,tempo")}  # fmt: skip
# the seventh round's settings; the steps, the warm-up and the checks come from --library and --passes
COMMON = ["--batch", "128", "--lr", "1e-4", "--bank-sample", "1024", "--head", "0", "--within", "2",
          "--between-loss", "infonce", "--leash", "30", "--mix", "sqrt", "--seed", "0"]  # fmt: skip


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
        self.steps = json.loads(CHOICES.read_text()).get("steps", []) if CHOICES.exists() else []

    def run(self, cmd: list[str], name: str | None = None) -> None:
        out = open(self.logs / f"{name}.log", "w") if name else None  # noqa: SIM115
        try:
            subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True, stdout=out, stderr=subprocess.STDOUT if out
                           else None)  # fmt: skip
        finally:
            if out:
                out.close()

    def prepare(self, part: str) -> None:
        libs = ",".join(f"{b}-{part}" for b in BENCHES)
        log(f"prepare {libs}")
        self.run(["scripts/learn.py", "prepare", "--libraries", libs, *self.compare])

    def learn(self, bench: str, part: str, recipe: dict) -> Path:
        """The library bench-part learnt by its starting model with the recipe: {"passes", "share"}."""
        tag = f"p{recipe['passes']:g}" + ("" if recipe["share"] >= 1.0 else f"-mix{recipe['share']:g}")
        name = f"learn-{bench}-{part}-{tag}"
        path = OUT / f"{name}.npz"
        if path.exists():
            return path
        start, extra = START[bench]
        flags = ["--init", str(OUT / f"{start}.npz"), "--library", f"{bench}-{part}", "--passes",
                 f"{recipe['passes']:g}", "--library-share", f"{recipe['share']:g}", "--extra", extra]  # fmt: skip
        log(f"learn {name}: {' '.join(flags[2:])}")
        self.run(["scripts/temporal.py", "train", *self.paths, "--name", name, *COMMON, *flags], name)
        return path

    def spec(self, label: str, paths: dict) -> str:
        return f"{label}=" + "+".join(f"{b}:{Path(p).relative_to(ROOT)}" for b, p in paths.items())

    def tune(self, specs: list[str]) -> dict:
        t = json.loads(TUNED8.read_text()) if TUNED8.exists() else {}
        todo = [s for s in specs if t.get(s.split("=")[0], {}).get("spec") != s]
        if todo:
            log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
            self.run(["scripts/compare.py", "tune", "--round", "8", *self.compare, "--candidates", ",".join(todo)])
            t = json.loads(TUNED8.read_text())
        return t

    def record(self, choice: int, name: str, recipe: dict, t: dict, against: str, kept: bool) -> None:
        step = {"choice": choice, "run": name, "recipe": recipe, "mean": t[name]["mean"], "against": against,
                "against_mean": t[against]["mean"], "kept": kept}  # fmt: skip
        self.steps = [s for s in self.steps if s["run"] != name] + [step]
        CHOICES.write_text(json.dumps({"steps": self.steps}, indent=1))
        log(f"choice {choice}: {name} {100 * t[name]['mean']:.2f} against {against} {100 * t[against]['mean']:.2f}: "
            f"{'kept' if kept else 'not kept'}")  # fmt: skip

    def candidate(self, recipe: dict) -> tuple[str, str]:
        label = f"learn-p{recipe['passes']:g}" + ("" if recipe["share"] >= 1.0 else f"-mix{recipe['share']:g}")
        return label, self.spec(label, {b: self.learn(b, "tuning", recipe) for b in BENCHES})

    def choose(self) -> None:
        self.prepare("tuning")
        start = self.spec("start", {b: OUT / f"{START[b][0]}.npz" for b in BENCHES})
        t = self.tune([start])
        current, recipe = "start", None
        # 1. learning each library, 10 passes over its descriptions
        r1 = {"passes": 10.0, "share": 1.0}
        name, spec = self.candidate(r1)
        t = self.tune([spec])
        kept = t[name]["mean"] > t[current]["mean"]
        self.record(1, name, r1, t, current, kept)
        if kept:
            current, recipe = name, r1
        if recipe is not None:
            # 2. 3 or 30 passes instead: the better of the two, if better
            tried = [(dict(r1, passes=p), *self.candidate(dict(r1, passes=p))) for p in (3.0, 30.0)]
            t = self.tune([s for _, _, s in tried])
            best = max(tried, key=lambda x: t[x[1]]["mean"])
            for r, n, _ in tried:
                self.record(2, n, r, t, current, n == best[1] and t[n]["mean"] > t[current]["mean"])
            if t[best[1]]["mean"] > t[current]["mean"]:
                current, recipe = best[1], best[0]
            # 3. half of each batch from the starting model's own training collections
            r3 = dict(recipe, share=0.5)
            name, spec = self.candidate(r3)
            t = self.tune([spec])
            kept = t[name]["mean"] > t[current]["mean"]
            self.record(3, name, r3, t, current, kept)
            if kept:
                current, recipe = name, r3
        final = {"run": current, "recipe": recipe, "mean": t[current]["mean"], "start_mean": t["start"]["mean"]}
        CHOICES.write_text(json.dumps({"steps": self.steps, "final": final}, indent=1))
        log(f"kept: {current}; the eighth round's choices are made: record them in docs/PROTOCOL.md, then learn the "
            f"test libraries and run the test once")  # fmt: skip

    def learn_test(self) -> None:
        final = json.loads(CHOICES.read_text())["final"]
        if final["recipe"] is None:
            log("no recipe kept: the test compares nothing; the round ends on the tuning sets")
            return
        self.prepare("test")
        paths = {b: self.learn(b, "test", final["recipe"]) for b in BENCHES}
        start = self.spec("start", {b: OUT / f"{START[b][0]}.npz" for b in BENCHES})
        learnt = self.spec("learnt", paths)
        log("the test libraries are learnt; the test, once: python scripts/compare.py evaluate --round 8 ... "
            f"--candidates '{start}~start,{learnt}~{final['run']}'")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["choose", "learn-test"])
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", required=True, help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    args = ap.parse_args()
    r = Round(args)
    r.choose() if args.stage == "choose" else r.learn_test()


if __name__ == "__main__":
    main()
