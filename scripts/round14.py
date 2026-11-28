"""The fourteenth round (docs/PROTOCOL.md): a large model, at most three times the cost.

Part A, stage 1, on the tuning sets (PE core L/14 exported first: scripts/small.py export, check, compress
--text-only, one and install, for pe-core-l14-224):

    python scripts/compare.py extract --dataset D --models pe-core-l14-224   # D: didemo-train, charades-tuning,
                                                                             #    youcook2, unav
    python scripts/round14.py bank ARGS     # its query bank and spans' strength
    python scripts/round14.py tune ARGS     # PE core L/14 alone, and PE core B/16 alone with its text tower in 8 bits
    python scripts/round14.py cost --videos DIDEMO_TEST_VIDEOS   # SigLIP 2 B/16 and PE core L/14 together, then
                                                                 # the current model, one after the other
    python scripts/round14.py stage1 ARGS   # its files and text time beside, and the rule: results/round14_stage1.json

ARGS are scripts/compare.py's (--annotations, --youcook2-annotations, --unav-annotations, --clap-dir, ...).

`bank` embeds DiDeMo's query bank with PE core L/14's text tower and chooses, as the thirteenth round chose its
candidates' (scripts/round13.py bank): the bank's beta and strength on DiDeMo's tuning videos over the fourth round's
grid, then the bank's strength for adaptive spans on YouCook2's tuning steps (results/round14_bank.json). `tune`
measures PE core L/14 with those settings, and PE core B/16 alone with its text tower in 8 bits as the twelfth round
measured it (its descriptions and query bank embedded by that tower), each with its speech and sound weights chosen
there (scripts/compare.py tune --round 14, results/round14_tuned.json). `cost` measures the indexing cost of SigLIP 2
B/16 and PE core L/14 together, each frame decoded once and embedded by both, then the current model's, each in a
process of its own on an idle machine, as the eleventh round measured them (results/round14_cost.json). `stage1`
applies the rule: stage 2 only if PE core L/14 alone has a higher tuning mean than PE core B/16 alone, and the two
encoders together cost at most three times the current model.

Part A, stage 2, if stage 1's rule holds (PE core L/14's bank settings set in whenframe/encoder.py first):

    python scripts/compare.py extract --dataset D --models pe-core-l14-224   # D: didemo-test, charades, and the
                                                                             #    collections didemo-all, msrvtt,
                                                                             #    charades-train, youcook2-train
    python scripts/round14.py texts ARGS    # the collections' descriptions, embedded by its text tower
    python scripts/round14.py stage2 ARGS   # v6, its left-out runs and the large model: results/round14_stage2.json

`stage2` trains v6, v4's recipe unchanged on PE core L/14's frames (scripts/round13.py's flags without a teacher, as
scripts/round9.py trained v4), and its two left-out runs (without Charades-STA's videos, without YouCook2's), two at
a time; it exports v6 as the package would ship it (alpha 1) and measures the three on the four tuning sets. The
large model is v3 and v6 fused as the tenth round fused v3 and v4 (scripts/screen.py), SigLIP 2 B/16's text tower in
8 bits as the tool runs it: PE's weight is the one of the tenth round's grid (0.5, 0.75, 1, 1.5, 2) with the highest
tuning mean (the grid's first on a tie), its speech and sound weights chosen with it. Its left-out runs, v3's and
v6's runs without the same collection fused at that weight, are measured there too. A stopped stage resumes: a run
whose weights exist is not trained again (its logged recipe is checked), a candidate measured with the same spec is
not measured again.

Part B, on the tuning sets, over the current model's frames (its text towers in 8 bits, as the tool runs it):

    python scripts/round14.py hear-texts ARGS   # the descriptions EmbeddingGemma 2 embeds
    EG2_PYTHON -I scripts/hear.py audio --set unav ...; audio --set youcook2 ...; texts ...   # its embeddings
    python scripts/round14.py hear-tune ARGS    # the rule for each set: results/round14_heard.json

`hear-tune` measures UnAV-100's tuning videos with CLAP's scores and with EmbeddingGemma 2's in their place, each
with its weight chosen over the fourth round's grid, and YouCook2's tuning steps with the transcript's full-text
scores (the third round's grid) and with EmbeddingGemma 2's in their place (the third round's speech weights). A
frame takes the best cosine of the windows around its middle, standardised over the library's frames as CLAP's are
(whenframe.sounds). EmbeddingGemma 2 is kept for a set only if that set's tuning metric is higher with it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CANDIDATE = "pe-core-l14-224"
REFERENCE = "pe-core-b16"  # PE core B/16 alone, its text tower in 8 bits (the twelfth round's)
LARGE = f"siglip2-base+{CANDIDATE}"  # both encoders, each frame decoded once
CURRENT = "siglip2-base+pe-core-b16"
MAX_COST = 3.0  # the rule's limit, times the current model's cost
BANK = ROOT / "results" / "round14_bank.json"
TUNED = ROOT / "results" / "round14_tuned.json"
COST = ROOT / "results" / "round14_cost.json"
STAGE1 = ROOT / "results" / "round14_stage1.json"
CACHE = ROOT / "results" / "cache"
TEXTS = CACHE / "round14_eg2_texts.json"  # what scripts/hear.py texts embeds
HEARD = ROOT / "results" / "round14_heard.json"


def chosen_bank() -> dict:
    return json.loads(BANK.read_text()) if BANK.exists() else {}


def with_banks() -> None:
    """PE core L/14's spec with the bank settings chosen for it (results/round14_bank.json)."""
    from whenframe import encoder

    for model, b in chosen_bank().items():
        encoder.MODELS[model] = replace(encoder.MODELS[model], bank_beta=b["bank"]["beta"],
                                        bank_lambda=b["bank"]["lam"], spans_lambda=b["spans"]["lam"])  # fmt: skip


def bank(args) -> None:
    """As scripts/round13.py's bank, for PE core L/14."""
    import round13

    round13.BANK, round13.CANDIDATES = BANK, (CANDIDATE,)
    round13.bank(args)


def tune(args) -> None:
    """PE core L/14 and PE core B/16, each alone, measured once (a stopped stage resumes)."""
    import compare
    import round12
    from compare import Words

    with_banks()
    done = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    args.round = 14
    spec = f"{CANDIDATE}={CANDIDATE}:"
    if done.get(CANDIDATE, {}).get("spec") != spec:
        args.candidates = spec
        compare.tune(args)
    spec = f"{REFERENCE}={REFERENCE}:"
    if done.get(REFERENCE, {}).get("spec") != spec:  # its descriptions and bank embedded by its 8-bit text tower
        round12.register_eight()
        compare.WORDS = Words("text8", towers={REFERENCE: round12.EIGHT[REFERENCE]})
        args.candidates = spec
        compare.tune(args)
        compare.WORDS = Words()


def cost(args) -> None:
    """Both encoders together, then the current model, each in a process of its own (scripts/compare.py cost
    --round 14); a model measured before is not measured again."""
    for model in (LARGE, CURRENT):
        if model in (json.loads(COST.read_text()) if COST.exists() else {}):
            continue
        subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "compare.py"), "cost", "--round", "14",
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

    tuned, costs = json.loads(TUNED.read_text()), json.loads(COST.read_text())
    _, queries, _ = didemo.load("train", args.annotations)
    texts = [q["text"] for q in queries[:200]]
    ratio = costs[LARGE]["cpu_min_per_hour"] / costs[CURRENT]["cpu_min_per_hour"]
    better = tuned[CANDIDATE]["mean"] > tuned[REFERENCE]["mean"]
    out = {
        "means": {m: tuned[m]["mean"] for m in (CANDIDATE, REFERENCE)},
        "cost": {"large": costs[LARGE]["cpu_min_per_hour"], "current": costs[CURRENT]["cpu_min_per_hour"],
                 "ratio": ratio, "limit": MAX_COST},
        "beside": {"files_mb": files_mb(CANDIDATE), "text_ms": 1000 * text_time(CANDIDATE, texts)},
        "higher_mean": better, "within_cost": ratio <= MAX_COST, "stage2": better and ratio <= MAX_COST,
    }  # fmt: skip
    STAGE1.write_text(json.dumps(out, indent=1))
    print(f"tuning means: PE core L/14 {100 * tuned[CANDIDATE]['mean']:.2f}, PE core B/16 "
          f"{100 * tuned[REFERENCE]['mean']:.2f}; both encoders {costs[LARGE]['cpu_min_per_hour']:.2f} CPU min per "
          f"hour against {costs[CURRENT]['cpu_min_per_hour']:.2f} ({ratio:.2f} times, at most {MAX_COST:g}); "
          f"{out['beside']['files_mb']:.0f} MB, text {out['beside']['text_ms']:.1f} ms")  # fmt: skip
    print("stage 2" if out["stage2"] else "the rule is not met: part A ends here")


# ---------------------------------------------------------------- part A, stage 2: v6, and the large model
OUT = ROOT / "results" / "temporal"
STAGE2 = ROOT / "results" / "round14_stage2.json"
SETS = ("didemo_all", "tempo", "msrvtt", "charades_train", "youcook2_train")  # the collections' descriptions
FRAMES = ("didemo_all", "msrvtt", "charades_train", "youcook2_train")  # and their frames
V3 = OUT / "temporal-siglip2-base-v3.npz"
V6 = OUT / f"temporal-{CANDIDATE}-v6.npz"  # v6, exported as the package would ship it (alpha 1)
RUNS = {"v6": "", "v6-no-charades": "charades", "v6-no-youcook2": "youcook2"}  # the collection each run leaves out
V3_WITHOUT = {"charades": OUT / "v3-no-charades.npz", "youcook2": OUT / "v3-no-youcook2.npz"}
FUSES = (0.5, 0.75, 1.0, 1.5, 2.0)  # PE's weight in the fusion: the tenth round's grid
KEYS = ("didemo", "charades", "youcook2", "unav", "mean", "speech", "sound_weight")


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


def kept() -> None:
    """Stage 2 only if stage 1's rule holds, with the bank settings chosen for PE core L/14 carried by
    whenframe.encoder.MODELS (the files scripts/temporal.py and the tool read)."""
    from whenframe.encoder import MODELS

    assert json.loads(STAGE1.read_text())["stage2"], "stage 1's rule is not met: part A ended there"
    b, spec = chosen_bank()[CANDIDATE], MODELS[CANDIDATE]
    chosen = (b["bank"]["beta"], b["bank"]["lam"], b["spans"]["lam"])
    assert (spec.bank_beta, spec.bank_lambda, spec.spans_lambda) == chosen, (
        f"whenframe/encoder.py's {CANDIDATE} must carry the settings results/round14_bank.json chose: {chosen}"
    )


def texts(args) -> None:
    """The collections' descriptions embedded by PE core L/14's text tower, as scripts/round13.py texts embedded
    them for its candidate (results/cache/NAME_text_pe-core-l14-224.npy, float16)."""
    import round13

    kept()
    round13.kept_model = lambda: CANDIDATE
    round13.texts(args)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def large_spec(name: str, v3: Path, v6: Path, weight: float) -> str:
    """v3 and a run of v6 fused as the tenth round fused v3 and v4 (scripts/screen.py), PE's weight `weight`."""
    return f"{name}={rel(v3)}@1|fuse:{weight:g}:{CANDIDATE}:{rel(v6)}@1"


def measured(args, specs: list[str], eight: bool) -> dict:
    """Candidates on the four tuning sets (scripts/compare.py tune --round 14), each measured once for the same spec;
    with `eight`, SigLIP 2 B/16's descriptions and query bank embedded by its text tower in 8 bits, as the tool runs
    it (PE core L/14's tower is in 8 bits already)."""
    import compare
    import round12
    from compare import Words

    done = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    todo = [s for s in specs if done.get(s.split("=")[0], {}).get("spec") != s]
    if todo:
        log(f"tune {', '.join(s.split('=')[0] for s in todo)}")
        with_banks()
        if eight:
            round12.register_eight()
            compare.WORDS = Words("text8", towers={"siglip2-base": round12.EIGHT["siglip2-base"]})
        compare.tune(argparse.Namespace(**{**vars(args), "round": 14, "candidates": ",".join(todo)}))
        compare.WORDS = Words()
    return json.loads(TUNED.read_text())


def stage2(args) -> None:
    import round13

    kept()
    needed = [f"{n}_text_{CANDIDATE}.npy" for n in SETS] + [f"{n}_{CANDIDATE}_openvino.npz" for n in FRAMES]
    missing = [f for f in needed if not (CACHE / f).exists()]
    assert not missing, f"embed these first (scripts/compare.py extract, scripts/round14.py texts): {missing}"
    train = lambda n: round13.train(args, n, CANDIDATE, 0.0, RUNS[n])  # noqa: E731 - v4's recipe, no teacher
    with ThreadPoolExecutor(args.parallel) as pool:
        weights = dict(zip(RUNS, pool.map(train, RUNS), strict=True))
    before = V6.read_bytes() if V6.exists() else None
    subprocess.run([sys.executable, "-I", "scripts/temporal.py", "export", "--annotations", args.annotations,
                    "--encoder", CANDIDATE, "--model", str(weights["v6"]), "--out", str(V6)], cwd=ROOT,
                   check=True)  # fmt: skip
    if before is not None and before != V6.read_bytes() and TUNED.exists():  # other weights: measured again
        t = json.loads(TUNED.read_text())
        TUNED.write_text(json.dumps({k: v for k, v in t.items() if k != "v6" and not k.startswith("large-")}, indent=1))
    alone = [f"v6={CANDIDATE}:{rel(V6)}@1", *(f"{n}={CANDIDATE}:{rel(weights[n])}@1" for n in RUNS if n != "v6")]
    measured(args, alone, eight=False)
    fused = {w: f"large-{w:g}" for w in FUSES}
    t = measured(args, [large_spec(n, V3, V6, w) for w, n in fused.items()], eight=True)
    weight = max(FUSES, key=lambda w: t[fused[w]]["mean"])  # the highest tuning mean, the grid's first on a tie
    left = {f"large-no-{c}": c for c in V3_WITHOUT}
    t = measured(args, [large_spec(n, V3_WITHOUT[c], weights[f"v6-no-{c}"], weight) for n, c in left.items()], True)
    record = {"encoder": CANDIDATE, "v6": rel(V6), "runs": {n: round13.summary(n, weights[n], t) for n in RUNS},
              "fusion": {f"{w:g}": {k: t[n][k] for k in KEYS} for w, n in fused.items()}, "weight": weight,
              "large": fused[weight], "left_out": {n: {k: t[n][k] for k in KEYS} for n in left}}  # fmt: skip
    STAGE2.write_text(json.dumps(record, indent=1))
    for n, r in record["runs"].items():
        log(f"{n}: best check at step {r['best_step']}; tuning sets' mean {100 * r['tuning']['mean']:.2f}")
    for n in fused.values():
        log(f"{n}: tuning sets' mean {100 * t[n]['mean']:.2f}")
    s = t[fused[weight]]
    log(f"the large model: PE's weight {weight:g}, speech {s['speech']}, sound weight {s['sound_weight']:g}; record "
        f"stage 2 in docs/PROTOCOL.md, then run the test once")  # fmt: skip


# ---------------------------------------------------------------- part B: what is heard
def hear_texts(args) -> None:
    """UnAV-100's labels as unav.prepare orders them, and YouCook2's steps in the order of its queries."""
    import compare

    ann = json.loads(Path(args.unav_annotations).read_text())["database"]
    labels = sorted({a["label"] for v in ann.values() for a in v["annotations"]})
    _, queries, _, _, _ = compare.Benchmarks(args).youcook2(compare.Candidate("base"))
    TEXTS.write_text(json.dumps({"unav_labels": labels, "youcook2": [q["text"] for q in queries]}, indent=1))
    print(f"{len(labels)} labels and {len(queries)} steps in {TEXTS.name}")


def heard_frames(times, ranges, video_of_window, starts, audio, text) -> np.ndarray:
    """(frames, descriptions): for each description, the best cosine of the windows around each frame's middle,
    standardised over the library's frames as CLAP's are."""
    from whenframe import sounds

    cos = audio.astype(np.float32) @ text.T
    return np.stack([sounds.standardised(sounds.frame_scores(times, ranges, video_of_window, starts, cos[:, j], 5.0))
                     for j in range(text.shape[0])], axis=1)  # fmt: skip


def hear_tune(args) -> None:
    import compare
    import round12
    import unav
    from compare import Benchmarks, Words

    said = json.loads(TEXTS.read_text())
    texts = np.load(TEXTS.with_suffix(".npz"))
    round12.register_eight()
    compare.WORDS = Words("text8", towers=dict(round12.EIGHT))
    current = round12.fusion()[0]
    b = Benchmarks(args)
    zu = np.load(CACHE / "unav_eg2.npz")

    def heard_unav(z, keep, labels):  # the windows CLAP's scores were computed on, embedded by EmbeddingGemma 2
        assert [str(n) for n in zu["names"]] == z["names"].tolist() and list(labels) == said["unav_labels"]
        assert np.array_equal(zu["wcounts"], z["wcounts_5"]) and np.allclose(zu["wstarts"], z["wstarts_5"])
        zz = {k: z[k] for k in z.files} | {"audio_5": zu["audio"]}
        return unav.standardise(unav.Library(zz, keep).sound(texts["unav_labels"], 5.0))

    zy = np.load(CACHE / "youcook2_eg2.npz")

    def heard_youcook2(g, queries, which):
        assert [q["text"] for q in queries] == said["youcook2"]
        index = {str(n): i for i, n in enumerate(zy["names"])}
        first = np.concatenate([[0], np.cumsum(zy["wcounts"])[:-1]])
        rows, ids = [], []
        for v, name in enumerate(g.names):
            i = index[str(name)]
            rows.append(np.arange(first[i], first[i] + zy["wcounts"][i]))
            ids.append(np.full(zy["wcounts"][i], v))
        rows, ids = np.concatenate(rows), np.concatenate(ids)
        ranges = {v: (int(g.starts[v]), int(g.counts[v])) for v in range(len(g.names))}
        out = np.zeros((len(g.times), len(queries)), np.float32)
        out[:, which] = heard_frames(g.times, ranges, ids, zy["wstarts"][rows].astype(np.float64), zy["audio"][rows],
                                     texts["youcook2"][which])  # fmt: skip
        return out

    res = {"unav": {"clap": b.unav_score(current, "tuning"), "eg2": b.unav_score(current, "tuning", heard=heard_unav)},
           "youcook2": {"transcript": b.youcook2_score(current, "tuning"),
                        "eg2": b.youcook2_score(current, "tuning", heard=heard_youcook2)}}  # fmt: skip
    for k in res:
        for v in res[k].values():
            v.pop("items", None)
    res["kept"] = {"unav": res["unav"]["eg2"]["value"] > res["unav"]["clap"]["value"],
                   "youcook2": res["youcook2"]["eg2"]["value"] > res["youcook2"]["transcript"]["value"]}  # fmt: skip
    res["hours_of_sound"] = {k: float(z["seconds"].sum() / 3600) for k, z in (("unav", zu), ("youcook2", zy))}
    HEARD.write_text(json.dumps(res, indent=1))
    u, y = res["unav"], res["youcook2"]
    print(f"UnAV-100 tuning P@10: CLAP {100 * u['clap']['value']:.1f} (weight {u['clap']['sound_weight']:g}), "
          f"EmbeddingGemma 2 {100 * u['eg2']['value']:.1f} (weight {u['eg2']['sound_weight']:g}); YouCook2 tuning "
          f"R@5: transcript {100 * y['transcript']['value']:.1f}, EmbeddingGemma 2 {100 * y['eg2']['value']:.1f} "
          f"(weight {y['eg2']['heard_weight']:g}); kept: {res['kept']}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "tune", "cost", "stage1", "texts", "stage2", "hear-texts", "hear-tune"])
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--test-annotations")
    ap.add_argument("--charades-annotations")
    ap.add_argument("--youcook2-annotations")
    ap.add_argument("--unav-annotations")
    ap.add_argument("--clap-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    ap.add_argument("--parallel", type=int, default=2, help="stage2: runs trained at once")
    args = ap.parse_args()
    stages = {"bank": bank, "tune": tune, "cost": cost, "stage1": stage1, "texts": texts, "stage2": stage2,
              "hear-texts": hear_texts, "hear-tune": hear_tune}  # fmt: skip
    stages[args.stage](args)


if __name__ == "__main__":
    main()
