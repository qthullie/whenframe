"""The tenth round's screening (docs/PROTOCOL.md): three ideas measured on the four tuning sets only, no test.

1. Guidance. Image-text models see what a frame shows more than what happens in it. Each query is pushed away from
   itself without its action, as classifier-free guidance pushes a generator away from its unconditional
   prediction: q_g = (1 + w) q - w q_stripped, at unit length (compare.py guided). q_stripped is the description
   without the words of a verb list mined from the training descriptions (`verbs`).
2. Fusion. v3 (SigLIP 2 B/16's frames) and v4 (PE core B/16's, ninth round) err on different descriptions: their
   corrected scores, standardised and added with PE's weight w (compare.py fused).
3. Round trip. The eighth round's libraries learnt from every automatic description of their frames, and lost.
   Here a library learns only from the descriptions that find their own block first in it, searched by its
   starting model (`filter`), with the eighth round's first recipe (10 passes, `learn`).

    python scripts/screen.py verbs
    python scripts/screen.py filter --annotations ... (compare.py's paths)
    python scripts/screen.py learn  --annotations ...
    python scripts/screen.py tune   --annotations ...      # results/round10_screen.json
    python scripts/screen.py report

A stopped run resumes: a library learnt is not learnt again, a candidate measured with the same spec is not measured
again.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "temporal"
CACHE = ROOT / "results" / "cache"
VERBS = ROOT / "results" / "round10_verbs.json"
KEPT = ROOT / "results" / "round10_round_trip.json"
SCREEN = ROOT / "results" / "round10_screen.json"
BENCHES = ("didemo", "charades", "youcook2", "unav")
V3 = "results/temporal/temporal-siglip2-base-v3.npz@1"
V4 = "pe-core-b16:results/temporal/temporal-pe-core-b16-v4.npz@1"
GUIDES = (0.25, 0.5, 1.0, 2.0)
FUSES = (0.5, 0.75, 1.0, 1.5, 2.0)
BOND = 0.01  # an idea is kept for a tenth round if it raises its reference's tuning mean by at least 1 point

# ---------------------------------------------------------------- 1. the verb list
# the training descriptions of whenframe's model (seventh round): no tuning or test description
SOURCES = ("didemo_all", "msrvtt", "charades_train", "youcook2_train", "tempo")
SUBJECTS = set(
    "person man woman boy girl he she they people someone somebody dog cat camera child kid baby guy lady we i you "
    "men women kids children player players toddler puppy bird horse chef cook".split()
)
AUX = set("is are was were be been being am".split())
FUNCTION = set(
    "a an the and or but in on at to of for with from by into onto over under up down out off his her their its "
    "this that then there here who which while as again also first next back away around through after before when "
    "where not no very just still now some all one two three another other more most what them him it so than too "
    "towards toward across along behind between inside outside near past without against upon within beside below "
    "above beyond during until since via about among throughout underneath"
    .split()
)  # fmt: skip
# words a noun follows: a word found after one of them in more than NOUNISH of its uses is a noun, kept
DETERMINERS = set("a an the his her their its my your our this that these those some".split())
NOUNISH = 0.2


def base_of(word: str) -> str:
    """The base form of a third person singular: watches -> watch, carries -> carry, opens -> open."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if re.search(r"(s|x|z|ch|sh)es$", word):
        return word[:-2]
    return word[:-1]


def verbs(args) -> None:
    """The words a description's action is made of, by a rule fixed before any measure: the word after a subject
    (a person, a dog, the camera...), past an auxiliary, when it has a verb's ending (-ing or -ed after a stem of at
    least 3 letters, -s after one of at least 2) and comes at least 5 times; the first word of YouCook2's steps,
    imperatives, when it comes at least 20 times; the base form of each -s form found among the descriptions' words;
    none of them found after a determiner in more than a fifth of its uses (a noun); and the auxiliaries."""
    after, imperative, uses, nounish = (collections.Counter() for _ in range(4))
    for name in SOURCES:
        for q in json.loads((CACHE / f"{name}_queries.json").read_text()):
            w = re.findall(r"[a-z']+", q["text"].lower())
            uses.update(w)
            nounish.update(b for a, b in zip(w, w[1:], strict=False) if a in DETERMINERS)
            if name == "youcook2_train" and w and w[0] not in FUNCTION:
                imperative[w[0]] += 1
            for i in range(len(w) - 1):
                if w[i] in SUBJECTS:
                    j = i + 1 + (w[i + 1] in AUX and i + 2 < len(w))
                    if w[j] not in FUNCTION | SUBJECTS | AUX and len(w[j]) >= 3:
                        after[w[j]] += 1
    verbal = re.compile(r"[a-z]{3,}(ing|ed)|[a-z]{2,}[^s]s")
    keep = {v for v, c in after.items() if c >= 5 and verbal.fullmatch(v)}
    keep |= {v for v, c in imperative.items() if c >= 20}
    keep |= {base_of(v) for v in list(keep) if v.endswith("s") and uses[base_of(v)]}
    keep = {v for v in keep if nounish[v] <= NOUNISH * uses[v] and v not in FUNCTION}
    strip = sorted(keep | AUX)
    VERBS.write_text(json.dumps({"rule": verbs.__doc__.split("\n\n")[0].replace("\n    ", " "), "sources": SOURCES,
                                 "strip": strip}, indent=1))  # fmt: skip
    print(f"{len(strip)} words: {', '.join(strip[:40])}...")


# ---------------------------------------------------------------- 3. the round trip
START = {"didemo": ("v3-leash30", "msrvtt,charades,youcook2,tempo"),
         "charades": ("v3-no-charades", "msrvtt,youcook2,tempo"),
         "youcook2": ("v3-no-youcook2", "msrvtt,charades,tempo"),
         "unav": ("v3-leash30", "msrvtt,charades,youcook2,tempo")}  # fmt: skip
COMMON8 = ["--batch", "128", "--lr", "1e-4", "--bank-sample", "1024", "--head", "0", "--within", "2",
           "--between-loss", "infonce", "--leash", "30", "--mix", "sqrt", "--seed", "0"]  # fmt: skip


def round_trip(bench: str) -> dict:
    """Which of the library's automatic descriptions find their own block first: the frame that matches the
    description best in the whole library, searched by the library's starting model (query-bank corrected, as the
    benchmarks search), falls inside the block the description was written for."""
    import compare
    from benchmark import bank_offsets

    from whenframe import temporal

    lib = f"{bench}-tuning"
    z = np.load(CACHE / f"learn_{lib}_frames.npz")
    emb, times, counts = z["emb"].astype(np.float32), z["times"].astype(np.float64), z["counts"]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    names = [str(n) for n in z["names"]]
    p = temporal.load(OUT / f"{START[bench][0]}.npz")
    for s, n in zip(starts, counts, strict=True):
        emb[s : s + n] = temporal.apply(p, emb[s : s + n], times[s : s + n])
    beta, lam = compare.bank_params("siglip2-base")
    offsets = bank_offsets(emb, compare.didemo_bank("siglip2-base"), beta)
    queries = json.loads((CACHE / f"learn_{lib}_queries.json").read_text())
    text = np.load(CACHE / f"learn_{lib}_text.npy").astype(np.float32)
    video = {n: i for i, n in enumerate(names)}
    kept = []
    for i in range(0, len(queries), 256):
        best = np.argmax(emb @ text[i : i + 256].T - lam * offsets[:, None], axis=0)
        for k, row in enumerate(best):
            q = queries[i + k]
            v = video[q["video"]]
            a, b = q["seconds"]
            kept.append(bool(starts[v] <= row < starts[v] + counts[v] and a <= times[row] < b))
    return {"descriptions": len(queries), "kept": int(sum(kept)), "mask": kept}


def filter_(args) -> None:
    """The kept descriptions of each tuning library, as scripts/temporal.py train --library reads them:
    learn_LIB-rt_frames.npz (the library's frames, linked), learn_LIB-rt_queries.json and learn_LIB-rt_text.npy."""
    out = json.loads(KEPT.read_text()) if KEPT.exists() else {}
    for bench in BENCHES:
        lib, rt = f"{bench}-tuning", f"{bench}-tuning-rt"
        r = round_trip(bench)
        mask = np.array(r.pop("mask"))
        queries = json.loads((CACHE / f"learn_{lib}_queries.json").read_text())
        text = np.load(CACHE / f"learn_{lib}_text.npy")
        frames = CACHE / f"learn_{rt}_frames.npz"
        if not frames.exists():
            os.link(CACHE / f"learn_{lib}_frames.npz", frames)
        kept = [q for q, k in zip(queries, mask, strict=True) if k]
        (CACHE / f"learn_{rt}_queries.json").write_text(json.dumps(kept))
        np.save(CACHE / f"learn_{rt}_text.npy", text[mask])
        out[bench] = r
        print(f"{lib}: {r['kept']} of {r['descriptions']} descriptions find their own block first "
              f"({100 * r['kept'] / r['descriptions']:.0f} %)", flush=True)  # fmt: skip
    KEPT.write_text(json.dumps(out, indent=1))


def paths(args) -> list[str]:
    out = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
    return out + [
        x
        for k in ("test_annotations", "charades_annotations", "youcook2_annotations", "unav_annotations", "clap_dir")
        if getattr(args, k)
        for x in (f"--{k.replace('_', '-')}", getattr(args, k))
    ]


def learn(args) -> None:
    """Each tuning library learnt by its starting model from its kept descriptions: the eighth round's first
    recipe, 10 passes, the library alone."""
    base = ["--annotations", args.annotations] + (["--model-dir", args.model_dir] if args.model_dir else [])
    for bench in BENCHES:
        name = f"learn-{bench}-tuning-rt-p10"
        if (OUT / f"{name}.npz").exists():
            continue
        start, extra = START[bench]
        flags = ["--init", str(OUT / f"{start}.npz"), "--library", f"{bench}-tuning-rt", "--passes", "10",
                 "--library-share", "1", "--extra", extra]  # fmt: skip
        log = OUT / "logs" / f"{name}.log"
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} learn {name}", flush=True)
        with open(log, "w") as f:
            subprocess.run([sys.executable, "-I", "scripts/temporal.py", "train", *base, "--name", name, *COMMON8,
                            *flags], cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)  # fmt: skip


def specs() -> list[str]:
    """The screening's candidates: v3 and the eighth round's starting models as they are (each idea's reference),
    then each idea over its grid."""
    per = lambda w: "+".join(f"{b}:results/temporal/{w(b)}.npz" for b in BENCHES)  # noqa: E731
    return [
        f"v3={V3}",
        *(f"guide-{w:g}={V3}|guide:{w:g}" for w in GUIDES),
        *(f"fuse-{w:g}={V3}|fuse:{w:g}:{V4}" for w in FUSES),
        f"start={per(lambda b: START[b][0])}",
        f"round-trip={per(lambda b: f'learn-{b}-tuning-rt-p10')}",
    ]


def tune(args) -> None:
    done = json.loads(SCREEN.read_text()) if SCREEN.exists() else {}
    todo = [s for s in specs() if done.get(s.split("=")[0], {}).get("spec") != s]
    for s in todo:  # one at a time: a stopped run keeps what was measured
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} tune {s.split('=')[0]}", flush=True)
        subprocess.run([sys.executable, "-I", "scripts/compare.py", "tune", "--round", "10", *paths(args),
                        "--candidates", s], cwd=ROOT, check=True)  # fmt: skip
    report(args)


def report(args) -> None:
    t = json.loads(SCREEN.read_text())
    cols = ("didemo", "charades", "youcook2", "unav")

    def row(n: str, ref: str | None = None) -> str:
        x = t[n]
        gain = "" if ref is None else f"  {100 * (x['mean'] - t[ref]['mean']):+.2f}"
        return f"{n:>12}  " + "  ".join(f"{100 * x[c]:5.1f}" for c in cols) + f"  {100 * x['mean']:6.2f}{gain}"

    print(f"{'':>12}  DiDeMo  Chara  YouC2  UnAV    mean")
    for n in ["v3", *(f"guide-{w:g}" for w in GUIDES), *(f"fuse-{w:g}" for w in FUSES), "start", "round-trip"]:
        if n in t:
            print(row(n, None if n in ("v3", "start") else ("start" if n == "round-trip" else "v3")))
    for idea, names, ref in (("1. guidance", [f"guide-{w:g}" for w in GUIDES], "v3"),
                             ("2. fusion", [f"fuse-{w:g}" for w in FUSES], "v3"),
                             ("3. round trip", ["round-trip"], "start")):  # fmt: skip
        got = [n for n in names if n in t]
        if got and ref in t:
            best = max(got, key=lambda n: t[n]["mean"])
            gain = t[best]["mean"] - t[ref]["mean"]
            print(f"{idea}: best {best}, {100 * gain:+.2f} points against {ref}: "
                  f"{'kept' if gain >= BOND else 'not kept'}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["verbs", "filter", "learn", "tune", "report"])
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    args = ap.parse_args()
    {"verbs": verbs, "filter": filter_, "learn": learn, "tune": tune, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
