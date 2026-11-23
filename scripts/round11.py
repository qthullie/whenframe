"""The eleventh round (docs/PROTOCOL.md): the tenth round's fusion of v3 and v4 tested, and v5, whenframe's model
taught by the fusion.

    python scripts/round11.py cost --videos DIDEMO_VIDEOS          # both encoders' cost, each frame decoded once
    python scripts/round11.py test1 ARGS                           # part 1: the fusion, once
    python scripts/round11.py v5 ARGS                              # part 2's training and its choices
    python scripts/round11.py test2 ARGS                           # part 2: v5, once, after its choices are recorded

ARGS are scripts/compare.py's (--annotations, --test-annotations, ...).

teacher: each training frame through v3 and through v4 (results/cache/NAME_teacher_TAG_{a,b,off}.npy), as
scripts/temporal.py --fusion-teacher reads them; the left-out runs' teachers only when v5 goes to the test.

v5: two runs of v3's recipe (scripts/round9.py's flags) with the teacher's losses at weights 1 and 3, trained two
at a time and measured on the tuning sets (compare.py tune --round 11); the choices as the protocol makes them, in
order; then, if the run kept raises v3's mean by at least 1 point, v5 exported (alpha 1) and its two left-out runs
trained and measured. results/round11_v5.json records it.

A stopped run resumes: a run whose weights exist is not trained again (its logged recipe is checked), a candidate
already measured with the same weights is not measured again, teacher files written are kept.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "temporal"
CACHE = ROOT / "results" / "cache"
TUNED7 = ROOT / "results" / "round7_tuned.json"
SCREEN10 = ROOT / "results" / "round10_screen.json"
TUNED11 = ROOT / "results" / "round11_tuned.json"
V5LOG = ROOT / "results" / "round11_v5.json"
V3 = OUT / "temporal-siglip2-base-v3.npz"
V4 = OUT / "temporal-pe-core-b16-v4.npz"
V5 = OUT / "temporal-siglip2-base-v5.npz"  # v5, exported as the package would ship it
# where the teacher's arrays are written (results/cache by default; ROUND11_TEACHER_DIR, such as /dev/shm, when the
# disk is too small for them), each linked into results/cache, where scripts/temporal.py reads it
TEACHER_DIR = Path(os.environ.get("ROUND11_TEACHER_DIR") or CACHE)
PE = "pe-core-b16"
WEIGHT = 0.75  # PE's weight in the fusion, chosen in the tenth round
# the fusion as the tenth round measured it, and its left-out runs fused the same way (reported beside)
FUSIONS = {"fusion": (V3, V4), "fusion-no-charades": (OUT / "v3-no-charades.npz", OUT / "v4-no-charades.npz"),
           "fusion-no-youcook2": (OUT / "v3-no-youcook2.npz", OUT / "v4-no-youcook2.npz")}  # fmt: skip
WITHOUT = {"fusion": "", "fusion-no-charades": "charades", "fusion-no-youcook2": "youcook2"}
FRAMES = {"didemo": "didemo_all_openvino", "msrvtt": "msrvtt_openvino", "charades": "charades_train_openvino",
          "youcook2": "youcook2_train_openvino"}  # fmt: skip
COLLECTIONS = ("msrvtt", "charades", "youcook2", "tempo")
# scripts/round7.py's flags, as scripts/round9.py trained v4 with them
COMMON = ["--steps", "6000", "--every", "500", "--batch", "128", "--lr", "1e-4", "--warmup", "200",
          "--bank-sample", "1024", "--head", "0", "--within", "2", "--between-loss", "infonce", "--init-seed", "0",
          "--mix", "sqrt"]  # fmt: skip
CHOICES = {"v5-d1": 1.0, "v5-d3": 3.0}  # the teacher's weight, in the protocol's order
TEST1 = ["v3", "fusion", "so400m", "v3-no-charades", "fusion-no-charades", "v3-no-youcook2", "fusion-no-youcook2"]
TEST2 = ["v3", "fusion", "v5", "v3-no-charades", "v5-no-charades", "v3-no-youcook2", "v5-no-youcook2"]


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


def fusion_spec(name: str, a: Path, b: Path) -> str:
    return f"{name}={a.relative_to(ROOT)}@1|fuse:{WEIGHT:g}:{PE}:{b.relative_to(ROOT)}@1"


def teacher(tag: str) -> None:
    """Each frame of the collections a run with this teacher trains on, through the fusion's two models, a video at
    a time over all its frames, and each one's query-bank offset (DiDeMo's training descriptions, beta 50)."""
    from whenframe import bank as qbank
    from whenframe import temporal

    a_weights, b_weights = FUSIONS[tag]
    pa, pb = temporal.load(a_weights), temporal.load(b_weights)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    banks = [np.load(CACHE / f"didemo_bank_text{k}.npy").astype(np.float32) for k in ("", f"_{PE}")]
    for c, name in FRAMES.items():
        if c == WITHOUT[tag]:
            continue
        dest = {x: CACHE / f"{name}_teacher_{tag}_{x}.npy" for x in ("a", "b", "off")}
        if all(p.exists() for p in dest.values()):
            continue
        log(f"teacher {tag}: {name}")
        z, zb = np.load(CACHE / f"{name}.npz"), np.load(CACHE / f"{name.replace('_openvino', f'_{PE}_openvino')}.npz")
        times, counts = z["times"], z["counts"]
        assert np.array_equal(times, zb["times"]) and np.array_equal(counts, zb["counts"]), "the same instants"
        emb_a, emb_b = z["emb"], zb["emb"]
        a, b = np.empty(emb_a.shape, np.float16), np.empty(emb_b.shape, np.float16)
        for s, n in zip(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts, strict=True):
            a[s : s + n] = temporal.apply(pa, emb_a[s : s + n].astype(np.float32), times[s : s + n])
            b[s : s + n] = temporal.apply(pb, emb_b[s : s + n].astype(np.float32), times[s : s + n])
        off = np.stack([qbank.offsets(a, banks[0], 50.0), qbank.offsets(b, banks[1], 50.0)], axis=1)
        for x, arr in (("a", a), ("b", b), ("off", off.astype(np.float32))):
            target = TEACHER_DIR / dest[x].name
            tmp = target.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            tmp.replace(target)
            if target != dest[x]:  # written elsewhere (a disk too small: /dev/shm), read through a link
                dest[x].unlink(missing_ok=True)
                dest[x].symlink_to(target)
    log(f"teacher {tag}: ready")


class Round11:
    def __init__(self, args):
        self.args = args
        self.paths = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
        self.compare = self.paths + [x for k in ("test_annotations", "charades_annotations", "youcook2_annotations",
                                                 "unav_annotations", "clap_dir") if getattr(args, k, None)
                                     for x in (f"--{k.replace('_', '-')}", getattr(args, k))]  # fmt: skip
        self.logs = OUT / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ the tuning sets
    def tuned(self) -> dict:
        return json.loads(TUNED11.read_text()) if TUNED11.exists() else {}

    def tune(self, specs: list[str]) -> dict:
        t = self.tuned()
        todo = [s for s in specs if t.get(s.split("=")[0], {}).get("spec") != s]
        if todo:
            log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
            env = dict(os.environ, OPENBLAS_NUM_THREADS="2")
            subprocess.run([sys.executable, "-I", "scripts/compare.py", "tune", "--round", "11", *self.compare,
                            "--candidates", ",".join(todo)], cwd=ROOT, check=True, env=env)  # fmt: skip
            t = self.tuned()
        return t

    def fusion_entry(self) -> None:
        """The fusion's entry, as the tenth round measured and tuned it (PE's weight 0.75): nothing chosen again."""
        t = self.tuned()
        if "fusion" in t:
            return
        s = json.loads(SCREEN10.read_text())[f"fuse-{WEIGHT:g}"]
        t["fusion"] = s | {"spec": fusion_spec("fusion", *FUSIONS["fusion"]), "from": f"round10_screen.json "
                           f"fuse-{WEIGHT:g}"}  # fmt: skip
        TUNED11.write_text(json.dumps(t, indent=1))

    # ------------------------------------------------------------ the tests
    def evaluate(self, part: int, names: list[str]) -> None:
        log(f"test, part {part}: {', '.join(names)}")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4")
        subprocess.run([sys.executable, "-I", "scripts/compare.py", "evaluate", "--round", "11", "--part", str(part),
                        *self.compare, "--candidates", ",".join(names)], cwd=ROOT, check=True, env=env)  # fmt: skip

    def test1(self) -> None:
        self.fusion_entry()
        self.tune([fusion_spec(n, *FUSIONS[n]) for n in ("fusion-no-charades", "fusion-no-youcook2")])
        self.evaluate(1, TEST1)

    def test2(self) -> None:
        record = json.loads(V5LOG.read_text())
        assert record["to_test"], "v5 did not go to the test (docs/PROTOCOL.md)"
        self.evaluate(2, TEST2)

    # ------------------------------------------------------------ v5
    @staticmethod
    def config(distil: float, without: str, tag: str) -> tuple[list[str], dict]:
        """A run's flags for scripts/temporal.py train, and the configuration its log must show."""
        extra = [c for c in COLLECTIONS if c != without]
        flags = ["--extra", ",".join(extra), "--leash", "30", "--width", "256", "--layers", "2", "--heads", "4",
                 "--seed", "0", "--distil", f"{distil:g}", "--fusion-teacher", tag,
                 "--fusion-weight", f"{WEIGHT:g}"]  # fmt: skip
        if "charades" in extra:  # checked on Charades-STA's held-out videos only when it trains on Charades-STA
            flags.append("--check-charades")
        logged = {"extra": ",".join(extra), "leash": 30.0, "width": 256, "layers": 2, "heads": 4, "distil": distil,
                  "check_charades": "charades" in extra, "seed": 0, "steps": 6000, "encoder": "siglip2-base",
                  "fusion_teacher": tag, "fusion_weight": WEIGHT}  # fmt: skip
        return flags, logged

    def train(self, name: str, distil: float, tag: str) -> Path:
        path = OUT / f"{name}.npz"
        flags, expected = self.config(distil, WITHOUT[tag], tag)
        if path.exists():  # trained before: with the same recipe, or a mistake to stop at
            c = json.loads((OUT / f"{name}.json").read_text())["config"]
            got = {k: c.get(k) for k in expected}
            assert got == expected, f"{path} was trained otherwise: {got} != {expected}"
            return path
        teacher(tag)
        cmd = [sys.executable, "-I", "scripts/temporal.py", "train", *self.paths, "--name", name, *COMMON, *flags]
        log(f"train {name}: {' '.join(flags)}")
        with open(self.logs / f"{name}.log", "w") as out:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=out, stderr=subprocess.STDOUT)
        log(f"trained {name}")
        return path

    @staticmethod
    def spec(name: str, weights: Path) -> str:
        return f"{name}={weights.relative_to(ROOT)}@1"

    def v5(self) -> None:
        teacher("fusion")
        with ThreadPoolExecutor(2) as pool:
            weights = dict(zip(CHOICES, pool.map(lambda n: self.train(n, CHOICES[n], "fusion"), CHOICES),
                               strict=True))  # fmt: skip
        t = self.tune([self.spec(n, w) for n, w in weights.items()])
        v3 = json.loads(TUNED7.read_text())["v3"]
        kept, kept_mean, steps = None, v3["mean"], []
        for n in CHOICES:  # each kept only if it raises the mean of the run kept so far (v3 at first)
            m = t[n]["mean"]
            steps.append({"run": n, "distil": CHOICES[n], "mean": m, "against": kept or "v3",
                          "against_mean": kept_mean, "kept": m > kept_mean})  # fmt: skip
            if m > kept_mean:
                kept, kept_mean = n, m
        to_test = kept is not None and kept_mean - v3["mean"] >= 0.01
        record = {"v3_mean": v3["mean"], "steps": steps, "kept": kept, "gain": kept_mean - v3["mean"],
                  "to_test": to_test, "runs": {}}  # fmt: skip
        for n, w in weights.items():
            lg = json.loads(w.with_suffix(".json").read_text())
            record["runs"][n] = {
                "weights": str(w.relative_to(ROOT)),
                "best_step": lg["best_step"],
                "best": lg["best"],
                "tuning": {k: t[n][k] for k in ("didemo", "charades", "youcook2", "unav", "mean")},
            }
        for s in steps:
            log(f"{s['run']}: mean {100 * s['mean']:.2f} against {s['against']} {100 * s['against_mean']:.2f}: "
                f"{'kept' if s['kept'] else 'not kept'}")  # fmt: skip
        if not to_test:
            V5LOG.write_text(json.dumps(record, indent=1))
            log(f"v5 does not go to the test ({100 * record['gain']:+.2f} points against v3, 1 asked for)")
            return
        subprocess.run([sys.executable, "-I", "scripts/temporal.py", "export", "--annotations", self.args.annotations,
                        "--model", str(weights[kept]), "--out", str(V5)], cwd=ROOT, check=True)  # fmt: skip
        left = {"v5-no-charades": "fusion-no-charades", "v5-no-youcook2": "fusion-no-youcook2"}
        with ThreadPoolExecutor(2) as pool:
            lw = dict(zip(left, pool.map(lambda n: self.train(n, CHOICES[kept], left[n]), left), strict=True))
        t = self.tune([self.spec("v5", V5), *(self.spec(n, w) for n, w in lw.items())])
        for n, w in lw.items():
            lg = json.loads(w.with_suffix(".json").read_text())
            record["runs"][n] = {
                "weights": str(w.relative_to(ROOT)),
                "best_step": lg["best_step"],
                "best": lg["best"],
                "tuning": {k: t[n][k] for k in ("didemo", "charades", "youcook2", "unav", "mean")},
            }
        record["v5"] = {
            "weights": str(V5.relative_to(ROOT)),
            "from": kept,
            "tuning": {k: t["v5"][k] for k in ("didemo", "charades", "youcook2", "unav", "mean")},
        }
        V5LOG.write_text(json.dumps(record, indent=1))
        log("v5 goes to the test: record its choices in docs/PROTOCOL.md, then run test2 once")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "cost", "test1", "v5", "test2"])
    ap.add_argument("--tag", default="fusion", choices=sorted(FUSIONS), help="teacher: which fusion")
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    args = ap.parse_args()
    if args.stage == "teacher":
        teacher(args.tag)
        return
    if args.stage == "cost":
        subprocess.run([sys.executable, "-I", "scripts/compare.py", "cost", "--round", "11", "--videos", args.videos,
                        "--models", f"siglip2-base,siglip2-base+{PE}"], cwd=ROOT, check=True)  # fmt: skip
        return
    assert args.annotations, "--annotations, as scripts/compare.py's"
    getattr(Round11(args), args.stage)()


if __name__ == "__main__":
    main()
