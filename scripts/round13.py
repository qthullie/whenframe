"""The thirteenth round (docs/PROTOCOL.md): a small model, for phones.

Stage 1, on the tuning sets:

    python scripts/compare.py extract --dataset D --models pe-core-s16-224,pe-core-s16-384,pe-core-t16-224,...
                                                   # D: didemo-train, charades-tuning, youcook2, unav
    python scripts/round13.py bank ARGS            # each candidate's query bank and spans' strength
    python scripts/round13.py tune ARGS            # the four candidates on the four tuning sets
    python scripts/round13.py cost --videos DIDEMO_TEST_VIDEOS   # indexing costs, the current model's too
    python scripts/round13.py stage1               # sizes, text times, and the rule: results/round13_stage1.json

Stage 2, whenframe's model on the kept candidate's frames:

    python scripts/compare.py extract --dataset D --models KEPT   # D: the four tests, and the four collections
    python scripts/round13.py texts                # the collections' descriptions, embedded by its text tower
    ROUND11_TEACHER_DIR=/dev/shm/... python scripts/round13.py stage2 ARGS   # results/round13_stage2.json

ARGS are scripts/compare.py's (--annotations, --youcook2-annotations, --unav-annotations, --clap-dir, ...).

`bank` embeds DiDeMo's query bank with each candidate's text tower and chooses, as the ninth round chose PE core
B/16's (scripts/pe_core.py bank): the bank's beta and strength on DiDeMo's tuning videos over the fourth round's
grid, then the bank's strength for adaptive spans on YouCook2's tuning steps (results/round13_bank.json). `tune`
measures each candidate with those settings, its speech and sound weights chosen there (scripts/compare.py tune,
results/round13_tuned.json). `cost` measures the indexing cost of each candidate and of the current model, one
after the other, as the eleventh round measured it (results/round13_cost.json). `stage1` adds each candidate's
files and its text tower's median time per description, and applies the rule.

`stage2` trains three runs of v4's recipe on the kept candidate's frames (scripts/temporal.py, as scripts/round9.py
trained v4): without a teacher, and taught by the current model, v3 and v4 fused as scripts/round11.py prepared
them for v5 (`teacher`: every training frame through both, kept where ROUND11_TEACHER_DIR says), at weights 1 and
3. It measures them on the tuning sets (scripts/compare.py tune --round 13), keeps the highest mean, exports it
(alpha 1) as the package would ship it, and trains its two left-out runs, taught, if it is, by the fusion of v3's and
v4's own left-out runs. A stopped stage resumes: a run whose weights exist is not trained again (its logged recipe
is checked), a candidate measured with the same weights is not measured again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
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
CACHE = ROOT / "results" / "cache"
OUT = ROOT / "results" / "temporal"
TUNED = ROOT / "results" / "round13_tuned.json"
STAGE2 = ROOT / "results" / "round13_stage2.json"
SETS = ("didemo_all", "tempo", "msrvtt", "charades_train", "youcook2_train")  # the collections' descriptions
COLLECTIONS = ("msrvtt", "charades", "youcook2", "tempo")  # besides DiDeMo's, as v3, v4 and v5 trained on them
# scripts/round7.py's flags, as scripts/round9.py trained v4 and scripts/round11.py v5 with them
COMMON = ["--steps", "6000", "--every", "500", "--batch", "128", "--lr", "1e-4", "--warmup", "200",
          "--bank-sample", "1024", "--head", "0", "--within", "2", "--between-loss", "infonce", "--init-seed", "0",
          "--mix", "sqrt"]  # fmt: skip
RUNS = {"small-noteacher": 0.0, "small-d1": 1.0, "small-d3": 3.0}  # the teacher's weight, in the protocol's order
# the teacher of a run without a collection: v3's and v4's runs without it, fused (scripts/round11.py's tags)
TEACHERS = {"": "fusion", "charades": "fusion-no-charades", "youcook2": "fusion-no-youcook2"}
WEIGHT = 0.75  # PE's weight in the teacher's fusion, chosen in the tenth round


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


# ---------------------------------------------------------------- stage 2
def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


def kept_model() -> str:
    """Stage 1's candidate, its bank settings as whenframe.encoder.MODELS carries them (the files scripts/temporal.py
    and the tool read), which must be those chosen for it."""
    from whenframe.encoder import MODELS

    model = json.loads(STAGE1.read_text())["kept"]
    assert model, "no candidate was kept at stage 1: the round ended there"
    b, spec = chosen_bank()[model], MODELS[model]
    chosen = (b["bank"]["beta"], b["bank"]["lam"], b["spans"]["lam"])
    assert (spec.bank_beta, spec.bank_lambda, spec.spans_lambda) == chosen, (
        f"whenframe/encoder.py's {model} must carry the settings results/round13_bank.json chose: {chosen}"
    )
    return model


def texts(args) -> None:
    """The descriptions of the collections v3 and v4 trained on, embedded by the kept candidate's text tower as the
    tool embeds a query, in the order of their queries (results/cache/NAME_text_MODEL.npy, float16), as
    scripts/pe_core.py texts embedded them for PE core B/16."""
    from whenframe.encoder import Encoder

    model, enc = kept_model(), None
    for name in SETS:
        out = CACHE / f"{name}_text_{model}.npy"
        if out.exists():
            continue
        enc = enc or Encoder(model, backend="openvino")
        said = [q["text"] for q in json.loads((CACHE / f"{name}_queries.json").read_text())]
        t0 = time.time()
        emb = np.concatenate([enc.encode_texts(said[i : i + 64]) for i in range(0, len(said), 64)])
        tmp = out.with_name(f"{out.stem}.tmp.npy")
        np.save(tmp, emb.astype(np.float16))
        tmp.replace(out)
        log(f"{name}: {len(said)} descriptions in {time.time() - t0:.0f} s")


def config(model: str, distil: float, without: str = "") -> tuple[list[str], dict]:
    """A run's flags for scripts/temporal.py train (v4's, on the candidate's frames), and the configuration its log
    must show."""
    extra = [c for c in COLLECTIONS if c != without]
    flags = ["--extra", ",".join(extra), "--leash", "30", "--width", "256", "--layers", "2", "--heads", "4",
             "--seed", "0", "--encoder", model]  # fmt: skip
    logged = {"extra": ",".join(extra), "leash": 30.0, "width": 256, "layers": 2, "heads": 4, "distil": distil,
              "check_charades": "charades" in extra, "seed": 0, "steps": 6000, "encoder": model}  # fmt: skip
    if distil:
        flags += ["--distil", f"{distil:g}", "--fusion-teacher", TEACHERS[without], "--fusion-weight", f"{WEIGHT:g}"]
        logged |= {"fusion_teacher": TEACHERS[without], "fusion_weight": WEIGHT}
    if "charades" in extra:  # checked on Charades-STA's held-out videos only when it trains on Charades-STA
        flags.append("--check-charades")
    return flags, logged


def teacher(without: str = "") -> None:
    import round11

    round11.teacher(TEACHERS[without])


def train(args, name: str, model: str, distil: float, without: str = "") -> Path:
    path = OUT / f"{name}.npz"
    flags, expected = config(model, distil, without)
    if path.exists():  # trained before: with the same recipe, or a mistake to stop at
        c = json.loads(path.with_suffix(".json").read_text())["config"]
        got = {k: c.get(k) for k in expected}
        assert got == expected, f"{path} was trained otherwise: {got} != {expected}"
        return path
    paths = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
    cmd = [sys.executable, "-I", "scripts/temporal.py", "train", *paths, "--name", name, *COMMON, *flags]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log(f"train {name}: {' '.join(flags)}")
    with open(OUT / "logs" / f"{name}.log", "w") as out:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=out, stderr=subprocess.STDOUT)
    log(f"trained {name}")
    return path


def measured(args, specs: list[str]) -> dict:
    """The runs on the four tuning sets, each measured once for the same spec (scripts/compare.py tune --round 13)."""
    import compare

    done = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    todo = [s for s in specs if done.get(s.split("=")[0], {}).get("spec") != s]
    if todo:
        log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
        with_banks()
        compare.tune(Namespace(**{**vars(args), "round": 13, "candidates": ",".join(todo)}))
    return json.loads(TUNED.read_text())


def spec(name: str, model: str, weights: Path) -> str:
    return f"{name}={model}:{weights.relative_to(ROOT)}@1"


def summary(name: str, weights: Path, tuned: dict) -> dict:
    lg = json.loads(weights.with_suffix(".json").read_text())
    return {"weights": str(weights.relative_to(ROOT)), "best_step": lg["best_step"], "best": lg["best"],
            "seconds": lg["seconds"], "tuning": {k: tuned[name][k] for k in ("didemo", "charades", "youcook2",
                                                                           "unav", "mean")}}  # fmt: skip


def stage2(args) -> None:
    model = kept_model()
    missing = [n for n in SETS if not (CACHE / f"{n}_text_{model}.npy").exists()]
    assert not missing, f"embed the descriptions first (scripts/round13.py texts): {missing}"
    teacher()  # before the runs, which read it together
    with ThreadPoolExecutor(args.parallel) as pool:
        weights = dict(zip(RUNS, pool.map(lambda n: train(args, n, model, RUNS[n]), RUNS), strict=True))
    t = measured(args, [spec(n, model, w) for n, w in weights.items()])
    kept = max(RUNS, key=lambda n: t[n]["mean"])  # the highest tuning mean, in the protocol's order on a tie
    small = OUT / f"temporal-{model}-small.npz"  # the kept run, as the package would ship it (alpha 1)
    subprocess.run([sys.executable, "-I", "scripts/temporal.py", "export", "--annotations", args.annotations,
                    "--encoder", model, "--model", str(weights[kept]), "--out", str(small)], cwd=ROOT,
                   check=True)  # fmt: skip
    left = {f"small-no-{c}": c for c in ("charades", "youcook2")}
    if RUNS[kept]:
        for c in left.values():
            teacher(c)
    with ThreadPoolExecutor(args.parallel) as pool:
        lw = dict(zip(left, pool.map(lambda n: train(args, n, model, RUNS[kept], left[n]), left), strict=True))
    t = measured(args, [spec("small", model, small), *(spec(n, model, w) for n, w in lw.items())])
    record = {"encoder": model, "kept": kept, "small": str(small.relative_to(ROOT)),
              "small_tuning": {k: t["small"][k] for k in ("didemo", "charades", "youcook2", "unav", "mean")},
              "runs": {n: summary(n, w, t) for n, w in {**weights, **lw}.items()}}  # fmt: skip
    STAGE2.write_text(json.dumps(record, indent=1))
    for n, r in record["runs"].items():
        log(f"{n}: best check at step {r['best_step']}; tuning sets' mean {100 * r['tuning']['mean']:.2f}")
    log(f"kept {kept}, exported as {small.name}: record stage 2 in docs/PROTOCOL.md, then run the test once")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "tune", "cost", "stage1", "texts", "stage2"])
    ap.add_argument("--parallel", type=int, default=3, help="stage2: runs trained at once")
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--test-annotations")
    ap.add_argument("--charades-annotations")
    ap.add_argument("--youcook2-annotations")
    ap.add_argument("--unav-annotations")
    ap.add_argument("--clap-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    args = ap.parse_args()
    {"bank": bank, "tune": tune, "cost": cost, "stage1": stage1, "texts": texts, "stage2": stage2}[args.stage](args)


if __name__ == "__main__":
    main()
