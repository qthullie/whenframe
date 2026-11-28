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
    res["sound_cpu_min_per_hour"] = {k: float(z["cpu"].sum() / (z["seconds"].sum() / 3600) / 60)
                                     for k, z in (("unav", zu), ("youcook2", zy))}  # fmt: skip
    HEARD.write_text(json.dumps(res, indent=1))
    u, y = res["unav"], res["youcook2"]
    print(f"UnAV-100 tuning P@10: CLAP {100 * u['clap']['value']:.1f} (weight {u['clap']['sound_weight']:g}), "
          f"EmbeddingGemma 2 {100 * u['eg2']['value']:.1f} (weight {u['eg2']['sound_weight']:g}); YouCook2 tuning "
          f"R@5: transcript {100 * y['transcript']['value']:.1f}, EmbeddingGemma 2 {100 * y['eg2']['value']:.1f} "
          f"(weight {y['eg2']['heard_weight']:g}); kept: {res['kept']}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "tune", "cost", "stage1", "hear-texts", "hear-tune"])
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json")
    ap.add_argument("--test-annotations")
    ap.add_argument("--charades-annotations")
    ap.add_argument("--youcook2-annotations")
    ap.add_argument("--unav-annotations")
    ap.add_argument("--clap-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--videos", help="cost: the folder of DiDeMo's test videos")
    args = ap.parse_args()
    stages = {"bank": bank, "tune": tune, "cost": cost, "stage1": stage1, "hear-texts": hear_texts,
              "hear-tune": hear_tune}  # fmt: skip
    stages[args.stage](args)


if __name__ == "__main__":
    main()
