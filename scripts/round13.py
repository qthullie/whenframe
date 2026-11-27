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

The test, once, after stage 2 is recorded in docs/PROTOCOL.md:

    python scripts/round13.py test ARGS --media DEMO_MEDIA --web WEB_FOLDER --ort ORT_DIST   # results/round13_test.json

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

`test` measures, once, the small model, its encoder alone and the current model with its text towers in 8 bits (as
the twelfth round tested it: the same numbers again) on the four tests, and the protocol's criteria 43 to 47: the
files a small library downloads and the package's share of them; the mean; DiDeMo's right video among the first
five; the indexing cost, its encoder's as stage 1 measured it and its model's over DiDeMo's test frames, against
the current model's both; and the peak memory of a search of the demo's library, indexed with the small model as
the package would ship it (`as-shipped`: its query bank and weights read from results/cache/round13_package, since
whenframe/data holds neither until the test decides), each search a process of its own. Reported beside: the text
tower's time per description in a browser (scripts/web_text.mjs, on the web tower scripts/small.py web built), and
each left-out run on the test it left out.
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
    """Each candidate measured once (a stopped stage resumes with the candidates not measured yet)."""
    import compare

    with_banks()
    done = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    todo = [f"{m}={m}:" for m in CANDIDATES if done.get(m, {}).get("spec") != f"{m}={m}:"]
    if todo:
        args.round, args.candidates = 13, ",".join(todo)
        compare.tune(args)


def cost(args) -> None:
    """The candidates' and the current model's indexing costs, one process each after the other, on an idle
    machine (scripts/compare.py cost --round 13); a model measured before is not measured again."""
    costs = ROOT / "results" / "round13_cost.json"
    for model in (*CANDIDATES, CURRENT):
        if model in (json.loads(costs.read_text()) if costs.exists() else {}):
            continue
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


# ---------------------------------------------------------------- the test, once
TEST = ROOT / "results" / "round13_test.json"
PARTIAL = TEST.with_name("round13_test.partial.json")  # the four tests' numbers, kept if a later step stops
MIN_MEAN = 0.3715  # criterion 44, as written: 85 % of the twelfth round's 43.70
EVENTFORMER = 0.254  # criterion 45: the best published system's right video among its first five (DiDeMo's test)
MAX_SEARCH_GB = 0.5  # criterion 47
QUERIES = ("a geyser shooting steam into the sky", "a pipe band marching down the street", "fireworks")  # round 12's
BENCHES = ("didemo", "charades", "youcook2", "unav")
LEFT_OUT = {"charades": "small-no-charades", "youcook2": "small-no-youcook2"}
PACKAGE = CACHE / "round13_package"  # the small model's query bank and weights as the package would ship them
SMALL_LIBRARY = CACHE / "site-library-small"  # the demo's library, indexed with the small model
# a command in a process of its own, whose peak (KB) its parent reads once it has ended (as the twelfth round)
PEAK = ("import resource, subprocess, sys; subprocess.run(sys.argv[1:], check=True, stdout=subprocess.DEVNULL); "
        "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)")  # fmt: skip


def as_shipped(package: Path, argv: list[str]) -> int:
    """`whenframe ARGV`, the small model's query bank and weights read from PACKAGE, as whenframe/data would hold
    them; the encoder's files are those in whenframe's cache."""
    from whenframe import bank, cli, temporal

    model = json.loads(STAGE1.read_text())["kept"]
    load_bank, load_weights = bank.load, temporal.load

    def small_bank(m: str):
        return np.load(package / f"bank-{m}.npy").astype(np.float32) if m == model else load_bank(m)

    def small_weights(path=None, name: str = "temporal-siglip2-base.npz"):
        if path is None and name == f"temporal-{model}.npz":
            return load_weights(package / name)
        return load_weights(path, name)

    bank.load, temporal.load = small_bank, small_weights
    return cli.main(argv)


def package(args, model: str, tuned: dict) -> dict:
    """The small model's query bank (DiDeMo's 4,096 descriptions embedded by its text tower, in float16, as the
    package ships each model's) and its weights with the speech and sound settings chosen for it on the tuning sets
    (scripts/temporal.py export --settings): the same weights as the file tested."""
    PACKAGE.mkdir(parents=True, exist_ok=True)
    np.save(PACKAGE / f"bank-{model}.npy", np.load(CACHE / f"didemo_bank_text_{model}.npy").astype(np.float16))
    record = json.loads(STAGE2.read_text())
    out = PACKAGE / f"temporal-{model}.npz"
    subprocess.run([sys.executable, "-I", "scripts/temporal.py", "export", "--annotations", args.annotations,
                    "--encoder", model, "--model", record["runs"][record["kept"]]["weights"], "--settings",
                    f"{TUNED.relative_to(ROOT)}:small", "--out", str(out)], cwd=ROOT, check=True)  # fmt: skip
    shipped, tested = np.load(out), np.load(ROOT / record["small"])
    assert set(shipped.files) - {"search"} == set(tested.files), "not the same weights"
    assert all(np.array_equal(shipped[k], tested[k]) for k in tested.files), "not the same weights"
    assert json.loads(str(shipped["search"])) == {"speech": tuned["small"]["speech"],
                                                  "sound_weight": tuned["small"]["sound_weight"]}  # fmt: skip
    return {p.name: p.stat().st_size for p in (PACKAGE / f"bank-{model}.npy", out)}


def small_library(args, model: str) -> None:
    """The demo's library indexed with the small model, as `whenframe index` indexes a new library (its settings
    the defaults, its frames through the small model); an interrupted build resumes."""
    from dataclasses import asdict

    from whenframe.library import Settings

    SMALL_LIBRARY.mkdir(parents=True, exist_ok=True)
    cfg = SMALL_LIBRARY / "config.json"
    settings = asdict(Settings(model=model, temporal=f"temporal-{model}"))
    if not cfg.exists():
        cfg.write_text(json.dumps(settings, indent=1))
    assert json.loads(cfg.read_text()) == settings, f"{SMALL_LIBRARY} was built otherwise"
    subprocess.run([sys.executable, "-I", "scripts/round13.py", "as-shipped", str(PACKAGE), "--library",
                    str(SMALL_LIBRARY), "index", args.media], cwd=ROOT, check=True)  # fmt: skip


def peak_gb(cmd: list[str]) -> float:
    r = subprocess.run([sys.executable, "-c", PEAK, *cmd], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return int(r.stdout.strip().splitlines()[-1]) * 1024 / 1e9


def four(b, c, speech, sound) -> dict:
    """A candidate's scores on the four tests."""
    return {"didemo": b.didemo_score(c, "test"), "charades": b.charades_score(c),
            "youcook2": b.youcook2_score(c, "test", speech), "unav": b.unav_score(c, "test", sound)}  # fmt: skip


def chosen_for(t: dict):
    from whenframe.speech import SpeechConfig

    sp = t["speech"]
    return SpeechConfig(sp["weight"], sp["pad"], sp["normalise"]), t["sound_weight"]


def test(args) -> None:
    import compare
    import round12
    from compare import Benchmarks, Words, adapter_cost, candidate, paired

    from whenframe.encoder import MODELS, cache_dir

    assert not TEST.exists(), f"{TEST.relative_to(ROOT)} exists: the test is run once"
    model = kept_model()
    with_banks()
    tuned = json.loads(TUNED.read_text())
    missing = [n for n in ("small", model, *LEFT_OUT.values()) if n not in tuned]
    assert not missing, f"not measured on the tuning sets: {missing}"
    for need in (args.media, args.web, args.ort):
        assert need and Path(need).exists(), "the test needs --media, --web and --ort"
    # what the measures need, before any number: the package's files, the demo's library indexed with them
    shipped = package(args, model, tuned)
    small_library(args, model)
    b = Benchmarks(args)

    if PARTIAL.exists():  # the four tests' numbers, measured once already
        res = json.loads(PARTIAL.read_text())
    else:
        items = {}
        # the reference: the current model, its text towers in 8 bits, as the twelfth round tested it
        round12.register_eight()
        compare.WORDS = Words("text8", towers=dict(round12.EIGHT))
        current, speech, sound = round12.fusion()
        items["current"] = four(b, current, speech, sound)
        compare.WORDS = Words()
        for name in ("small", model):
            items[name] = four(b, candidate(tuned[name]["spec"]), *chosen_for(tuned[name]))
        res = {n: {k: {"value": v["value"], "full": v["full"]} for k, v in s.items()} for n, s in items.items()}
        for n in items:
            res[n]["mean"] = float(np.mean([res[n][k]["value"] for k in BENCHES]))
            log(f"{n}: " + ", ".join(f"{k} {100 * res[n][k]['value']:.1f}" for k in BENCHES)
                + f"; mean {100 * res[n]['mean']:.2f}")  # fmt: skip
        res["pairs"] = {"small vs current": paired(items["small"], items["current"]),
                        f"small vs {model}": paired(items["small"], items[model])}  # fmt: skip
        twelfth = json.loads(round12.TEST2.read_text())["runs"]["8bit"]
        res["twelfth_round_unchanged"] = all(res["current"][k]["value"] == twelfth[k] for k in BENCHES)
        res["left_out"] = {}
        for bench, n in LEFT_OUT.items():  # each left-out run on the test it left out
            c, (speech, sound) = candidate(tuned[n]["spec"]), chosen_for(tuned[n])
            s = b.charades_score(c) if bench == "charades" else b.youcook2_score(c, "test", speech)
            res["left_out"][n] = {"test": bench, "value": s["value"], "small": res["small"][bench]["value"]}
        PARTIAL.write_text(json.dumps(res, indent=1))

    # 43: the files a new small library downloads (the encoder's, as whenframe's cache holds them), and the package's
    spec = MODELS[model]
    folder = cache_dir() / "models" / spec.repo.replace("/", "--")
    names = [spec.tokenizer_file]
    for f in (spec.vision_file, spec.text_file):
        names += [f, f.removesuffix(".xml") + ".bin"]
    files = {n: (folder / n).stat().st_size for n in names} | {f"whenframe/data/{n}": s for n, s in shipped.items()}
    web = Path(args.web)
    web_files = {n: (web / n).stat().st_size for n in ("onnx/text_model_web.onnx", "merges.txt")}
    # 46: indexing, the encoder's cost as stage 1 measured it (decoding included) and its model's over DiDeMo's test
    # frames, against the current model's two encoders and v3's and v4's
    costs = json.loads((ROOT / "results" / "round13_cost.json").read_text())
    current = round12.fusion()[0]
    adapters = {"small": adapter_cost(b, candidate(tuned["small"]["spec"])), "v3": adapter_cost(b, current),
                "v4": adapter_cost(b, current.fuse[0])}  # fmt: skip
    small_cost = costs[model]["cpu_min_per_hour"] + adapters["small"]["cpu_min_per_hour"]
    current_cost = costs[CURRENT]["cpu_min_per_hour"] + sum(adapters[k]["cpu_min_per_hour"] for k in ("v3", "v4"))
    # 47: each search a process of its own; the current model's on the demo's library as the site was built
    small_peaks = [peak_gb([sys.executable, "-I", "scripts/round13.py", "as-shipped", str(PACKAGE), "--library",
                            str(SMALL_LIBRARY), "search", q]) for q in QUERIES]  # fmt: skip
    current_peaks = [peak_gb([sys.executable, "-I", "-m", "whenframe", "--library", str(CACHE / "site-library"),
                              "search", q]) for q in QUERIES]  # fmt: skip
    # beside: the web tower in a browser, one description at a time
    subprocess.run(["node", "scripts/web_text.mjs", str(web), args.ort], cwd=ROOT, check=True)
    browser = json.loads((web / "web_browser.json").read_text())
    small = res["small"]
    res |= {
        "encoder": model,
        "files": {"bytes": files, "mb": sum(files.values()) / 1e6, "web": web_files,
                  "web_mb": sum(web_files.values()) / 1e6},
        "cost": {"small": small_cost, "current": current_cost, "encoders": {model: costs[model],
                 CURRENT: costs[CURRENT]}, "adapters": adapters,
                 "share": small_cost / current_cost},
        "memory": {"small_gb": small_peaks, "current_gb": current_peaks, "queries": list(QUERIES)},
        "browser": {k: browser[k] for k in ("median_ms", "max_ms", "load_ms", "threads", "passed", "texts")},
        "share_of_current": small["mean"] / res["current"]["mean"],
    }  # fmt: skip
    res["criteria"] = {
        "43": res["files"]["mb"] <= MAX_MB,
        "44": small["mean"] >= MIN_MEAN,
        "45": small["didemo"]["full"]["video"]["R@5"] > EVENTFORMER,
        "46": small_cost <= MAX_COST_SHARE * current_cost,
        "47": max(small_peaks) <= MAX_SEARCH_GB,
    }
    TEST.write_text(json.dumps(res, indent=1))
    PARTIAL.unlink()
    log(f"files {res['files']['mb']:.1f} MB; mean {100 * small['mean']:.2f} (current {100 * res['current']['mean']:.2f}"
        f", the twelfth round's again: {res['twelfth_round_unchanged']}); DiDeMo's video among the first five "
        f"{100 * small['didemo']['full']['video']['R@5']:.1f} %; cost {small_cost:.2f} against {current_cost:.2f} CPU "
        f"min per hour; search peak {max(small_peaks):.2f} GB (current {max(current_peaks):.2f}); browser "
        f"{browser['median_ms']:.0f} ms per description")  # fmt: skip
    log(f"criteria {res['criteria']}: wrote {TEST.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "tune", "cost", "stage1", "texts", "stage2", "test"])
    ap.add_argument("--parallel", type=int, default=3, help="stage2: runs trained at once")
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--test-annotations")
    ap.add_argument("--charades-annotations")
    ap.add_argument("--youcook2-annotations")
    ap.add_argument("--unav-annotations")
    ap.add_argument("--clap-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    ap.add_argument("--media", help="test: the folder of the demo's videos and photos (site/library.json's)")
    ap.add_argument("--web", help="test: the folder of the web tower (scripts/small.py web)")
    ap.add_argument("--ort", help="test: onnxruntime-web's dist folder (scripts/web_text.mjs)")
    args = ap.parse_args()
    stages = {"bank": bank, "tune": tune, "cost": cost, "stage1": stage1, "texts": texts, "stage2": stage2,
              "test": test}  # fmt: skip
    stages[args.stage](args)


if __name__ == "__main__":
    if sys.argv[1:2] == ["as-shipped"]:  # python scripts/round13.py as-shipped PACKAGE WHENFRAME_ARGS...
        sys.exit(as_shipped(Path(sys.argv[2]), sys.argv[3:]))
    main()
