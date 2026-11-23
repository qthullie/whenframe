"""The twelfth round (docs/PROTOCOL.md): descriptions written in French (part 1), and the text towers in 8 bits
(part 2), with whenframe's model as the eleventh round tested it, every setting unchanged.

    python scripts/round12.py french ARGS     # every description of the four tuning and test sets in French
    python scripts/round12.py back ARGS       # the tuning sets' French translated back, by both translators
    python scripts/round12.py detect ARGS     # the detector's rates on the tuning sets (its word lists fixed then)
    python scripts/round12.py tune ARGS       # part 1's choice on the tuning sets: the translator
    python scripts/round12.py test1 ARGS      # part 1's test, once: English, French as it is, translated
    python scripts/round12.py towers ARGS     # part 2: both text towers in 8 bits (NNCF), and the banks they embed
    python scripts/round12.py test2 ARGS      # part 2's test, once
    python scripts/round12.py memory --library DIR    # criterion 42: a search's peak memory, the models' size

ARGS are scripts/compare.py's (--annotations, --test-annotations, ...). The French descriptions, the translations
back and the embeddings are kept in results/cache; results/round12_*.json record what each stage measured.

The French descriptions stand in for French typed by people: Opus-MT en-fr (Xenova's ONNX export, full precision),
greedy, a batch at a time; DiDeMo's are the second round's (results/cache/didemo_{train,test}_fr.json), and a
description of another set written as one of DiDeMo's is given its French. A French description is translated back
by Opus-MT fr-en as the tool would translate a description typed: alone, greedily, timed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import compare  # noqa: E402
from compare import CACHE, Benchmarks, Words, candidate, paired  # noqa: E402

from whenframe import translate  # noqa: E402
from whenframe.encoder import MODELS, cache_dir, download  # noqa: E402

RESULTS = ROOT / "results"
FRENCH = CACHE / "round12_french.json"  # {"sets": {set: [description, ...]}, "french": {description: French}}
BACK = CACHE / "round12_back_{}_{}.json"  # {French: [English, ms]}: the translator's precision, the sets' part
DETECTOR = RESULTS / "round12_detector.json"
TUNED = RESULTS / "round12_tuned.json"
TEST1 = RESULTS / "round12_test.json"
TEST2 = RESULTS / "round12_test_8bit.json"
MEMORY = RESULTS / "round12_memory.json"
TUNED11, TEST11 = RESULTS / "round11_tuned.json", RESULTS / "round11_test.json"
TUNING = ("didemo_tuning", "charades_tuning", "youcook2_tuning", "unav_labels")
TESTS = ("didemo_test", "charades_test", "youcook2_test", "unav_labels")
BENCHES = ("didemo", "charades", "youcook2", "unav")

# Xenova's ONNX exports of Helsinki-NLP's Opus-MT (Apache 2.0), at the revisions measured, with each file's SHA-256
HUB = "https://huggingface.co/Xenova/{}/resolve/{}/{}"
TRANSLATORS = {
    "en-fr": ("opus-mt-en-fr", "28726206f80896b90035bd99cccd5cc1e151f916", {
        "config.json": "b522b73fcc86f77981349c1df2a2e041eed0dbcbea4acf2635019cf21d51ffc0",
        "tokenizer.json": "8391785c1a2139e7af4678571ccd8dc654ecbb72e4be186940f65d7c604f0246",
        "onnx/encoder_model.onnx": "ffa8429c73bea51341e3ebb7af889a9b37294448c371f926dfb3a9a97895b0d9",
        "onnx/decoder_model.onnx": "991e217b5f162b2578726b483d77f06c70ef03fa0caba5d3d5afa699fa5347c0"}),
    "fr-en": ("opus-mt-fr-en", "6b166a182780e118c997879d0ad5be4b53671644", {
        "config.json": "6c2851f154d7b88c7767e4fdcf8a5694a36a456bfa6cbcde1b13fdbade3b56f6",
        "tokenizer.json": "8391785c1a2139e7af4678571ccd8dc654ecbb72e4be186940f65d7c604f0246",
        "onnx/encoder_model.onnx": "6e18e05c0a458a99296c526b6b2a989158ecf890a86072f9b3960df52bfc99a7",
        "onnx/decoder_model.onnx": "1844d3c6f595df70cd0745084afa8fd9e98227bce7ac8fb9a3f1ddd41bcd171b",
        "onnx/encoder_model_quantized.onnx": "e727cb26ac6bf816394c49671af69c9ae5798868fed1e40849872415dffc1772",
        "onnx/decoder_model_quantized.onnx": "8c5a0c5d113a1c4c5ae46e014aacf770f3bff2109fe8b74ead6aae5c5f408ffc"}),
}  # fmt: skip
PRECISIONS = {"full": ("encoder_model.onnx", "decoder_model.onnx"),
              "8bit": ("encoder_model_quantized.onnx", "decoder_model_quantized.onnx")}  # fmt: skip
MARGIN = 0.003  # the 8-bit translator is kept unless the tuning mean is more than 0.3 points lower with it

# part 2: each text tower with its weights in 8 bits, beside the model's own files (scripts/round12.py towers)
EIGHT = {"siglip2-base": "siglip2-base-text8", "pe-core-b16": "pe-core-b16-text8"}
EIGHT_FILE = "onnx/text_model_int8.xml"


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {text}", flush=True)


def translator(direction: str, precision: str = "full") -> translate.Translator:
    """Opus-MT, downloaded once into whenframe's cache, each file checked against its SHA-256."""
    repo, revision, digests = TRANSLATORS[direction]
    folder = cache_dir() / "models" / f"Xenova--{repo}"
    encoder, decoder = PRECISIONS[precision]
    for name in ("config.json", "tokenizer.json", f"onnx/{encoder}", f"onnx/{decoder}"):
        download(HUB.format(repo, revision, name), folder / name, digests[name])
    return translate.Translator(folder, encoder, decoder)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def register_eight() -> None:
    """The 8-bit text towers as encoders of whenframe.encoder.MODELS: the same models, their text tower from the
    8-bit file, run with OpenVINO."""
    for model, name in EIGHT.items():
        MODELS[name] = replace(MODELS[model], text_file=EIGHT_FILE, text_openvino=True)


# ---------------------------------------------------------------- the descriptions
def descriptions(args) -> dict[str, list[str]]:
    """The descriptions of each tuning and test set, in the order each benchmark searches them."""
    import charades
    import didemo
    import youcook2

    ns = argparse.Namespace
    shared = {"model_dir": args.model_dir, "train_annotations": args.annotations}
    out = {}
    for split, name in (("train", "didemo_tuning"), ("test", "didemo_test")):
        out[name] = [q["text"] for q in didemo.load(split, args.annotations if split == "train"
                                                    else args.test_annotations)[1]]  # fmt: skip
    out["charades_test"] = [q["text"] for q in charades.load(ns(annotations=args.charades_annotations, **shared))[1]]
    held = set(np.load(CACHE / "charades_tuning_openvino.npz")["names"].tolist())  # as scripts/temporal.py reads it
    queries = json.loads((CACHE / "charades_tuning_queries.json").read_text())
    out["charades_tuning"] = [q["text"] for q in queries if q["video"] in held]
    f, queries, *_ = youcook2.load(ns(annotations=args.youcook2_annotations, **shared))
    tuning = youcook2.split(f.names)
    out["youcook2_tuning"] = [q["text"] for q in queries if q["name"] in tuning]
    out["youcook2_test"] = [q["text"] for q in queries if q["name"] not in tuning]
    ann = json.loads(Path(args.unav_annotations).read_text())["database"]  # as scripts/unav.py's prepare
    out["unav_labels"] = sorted({a["label"] for v in ann.values() for a in v["annotations"]})
    return out


def french(args) -> None:
    """Every description in French: DiDeMo's as the second round translated them, the others by Opus-MT en-fr."""
    sets = descriptions(args)
    mapping, conflicts = {}, 0
    for split, name in (("train", "didemo_tuning"), ("test", "didemo_test")):
        fr = json.loads((CACHE / f"didemo_{split}_fr.json").read_text())
        assert len(fr) == len(sets[name]), "the second round's translations do not match the descriptions"
        for en, f in zip(sets[name], fr, strict=True):
            conflicts += en in mapping and mapping[en] != f
            mapping.setdefault(en, f)
    todo = sorted({t for v in sets.values() for t in v} - set(mapping), key=lambda t: (len(t), t))
    log(f"{len(mapping)} descriptions from the second round ({conflicts} written twice, translated otherwise: "
        f"the first kept); {len(todo)} to translate")  # fmt: skip
    t0 = time.time()
    tr = translator("en-fr")
    for i in range(0, len(todo), 512):
        mapping.update(zip(todo[i : i + 512], tr(todo[i : i + 512]), strict=True))
        log(f"  {min(i + 512, len(todo))}/{len(todo)} ({time.time() - t0:.0f} s)")
    FRENCH.write_text(json.dumps({"sets": sets, "french": mapping, "second_round_conflicts": conflicts},
                                 ensure_ascii=False, indent=0))  # fmt: skip
    for en in [sets[s][0] for s in sets]:
        log(f"  {en!r} -> {mapping[en]!r}")


def load_french() -> tuple[dict[str, list[str]], dict[str, str]]:
    d = json.loads(FRENCH.read_text())
    return d["sets"], d["french"]


def back(args, names: tuple[str, ...], precision: str, part: str) -> dict[str, list]:
    """The French of these sets translated back into English, a description at a time, timed (results/cache): the
    tuning sets' with both translators, the test sets' (all of them, timed again) with the one chosen."""
    sets, fr = load_french()
    path = Path(str(BACK).format(precision, part))
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted({fr[t] for n in names for t in sets[n]} - set(done))
    if todo:
        log(f"translating back {len(todo)} descriptions, {precision} precision, one at a time")
        tr = translator("fr-en", precision)
        tr(["une personne ouvre la porte"])  # the first run allocates; the tool's first query pays it once
        t0 = time.time()
        for i, text in enumerate(todo, 1):
            s = time.perf_counter()
            (en,) = tr([text], batch=1)
            done[text] = [en, 1000 * (time.perf_counter() - s)]
            if i % 500 == 0 or i == len(todo):
                path.write_text(json.dumps(done, ensure_ascii=False, indent=0))
                log(f"  {i}/{len(todo)} ({time.time() - t0:.0f} s)")
    return done


def words(name: str, precision: str | None = None, part: str = "tuning") -> Words:
    """English (""), French as it is ("fr"), or translated back by the translator in this precision."""
    if not name:
        return Words()
    _, fr = load_french()
    if precision is None:
        return Words(name, say=fr)
    done = json.loads(Path(str(BACK).format(precision, part)).read_text())
    return Words(name, say={en: done[f][0] for en, f in fr.items() if f in done})


# ---------------------------------------------------------------- the detector
def detector_rates(names: tuple[str, ...]) -> dict:
    sets, fr = load_french()
    english = sorted({t for n in names for t in sets[n]})
    french_ = sorted({fr[t] for t in english})
    wrong_en = [t for t in english if translate.is_french(t)]
    wrong_fr = [t for t in french_ if not translate.is_french(t)]
    return {"english": len(english), "english_taken_for_french": len(wrong_en),
            "english_rate": len(wrong_en) / len(english), "french": len(french_),
            "french_taken_for_french": len(french_) - len(wrong_fr),
            "french_rate": 1 - len(wrong_fr) / len(french_),
            "examples": {"english_taken_for_french": wrong_en[:40], "french_missed": wrong_fr[:60]}}  # fmt: skip


def lists() -> dict:
    return {"french": sorted(translate.FRENCH), "english": sorted(translate.ENGLISH),
            "accents": "".join(sorted(translate.ACCENTS)), "elisions": translate.ELISION.pattern}  # fmt: skip


def detect(args) -> None:
    r = detector_rates(TUNING)
    DETECTOR.write_text(json.dumps({"tuning": r, "lists": lists()}, ensure_ascii=False, indent=1))
    log(f"tuning sets: {100 * r['english_rate']:.2f} % of {r['english']} English descriptions taken for French, "
        f"{100 * r['french_rate']:.2f} % of {r['french']} French ones")  # fmt: skip


# ---------------------------------------------------------------- the searches
def fusion() -> tuple:
    """whenframe's model, its speech and sound settings, as the eleventh round tested it."""
    from whenframe.speech import SpeechConfig

    t = json.loads(TUNED11.read_text())["fusion"]
    sp = t["speech"]
    return candidate(t["spec"]), SpeechConfig(sp["weight"], sp["pad"], sp["normalise"]), t["sound_weight"]


def measure(b: Benchmarks, w: Words, part: str) -> dict:
    """The four primary metrics of the fusion on the tuning or test sets, searched with these words."""
    if w.say:  # every description searched has its other version: none is searched in English by mistake
        sets, _ = load_french()
        missing = {t for n in (TUNING if part == "tuning" else TESTS) for t in sets[n]} - set(w.say)
        assert not missing, f"{len(missing)} descriptions without their {w.name} version"
    compare.WORDS = w
    c, speech, sound = fusion()
    if part == "tuning":
        scores = {
            "didemo": b.didemo_score(c, "train"),
            "charades": b.charades_tuning_score(c),
            "youcook2": b.youcook2_score(c, "tuning", speech),
            "unav": b.unav_score(c, "tuning", sound),
        }
    else:
        scores = {
            "didemo": b.didemo_score(c, "test"),
            "charades": b.charades_score(c),
            "youcook2": b.youcook2_score(c, "test", speech),
            "unav": b.unav_score(c, "test", sound),
        }
    compare.WORDS = Words()
    out = {k: {"value": v["value"], "items": v["items"]} for k, v in scores.items()}
    out["mean"] = float(np.mean([v["value"] for v in scores.values()]))
    log(f"{w.name or 'English'} ({part}): " + ", ".join(f"{k} {100 * out[k]['value']:.2f}" for k in BENCHES)
        + f"; mean {100 * out['mean']:.2f}")  # fmt: skip
    return out


def summary(r: dict) -> dict:
    return {k: r[k]["value"] for k in BENCHES} | {"mean": r["mean"]}


def tune(args) -> None:
    """The translator, chosen on the tuning sets; the other ways of searching reported beside."""
    for p in PRECISIONS:
        back(args, TUNING, p, "tuning")
    b = Benchmarks(args)
    runs = {"english": measure(b, words(""), "tuning"), "french": measure(b, words("fr"), "tuning")}
    for p in PRECISIONS:
        runs[f"translated-{p}"] = measure(b, words(f"fr-back-{p}", p, "tuning"), "tuning")
    full, eight = runs["translated-full"]["mean"], runs["translated-8bit"]["mean"]
    chosen = "8bit" if eight >= full - MARGIN else "full"
    times = {p: statistics.median(v[1] for v in json.loads(Path(str(BACK).format(p, "tuning")).read_text()).values())
             for p in PRECISIONS}  # fmt: skip
    TUNED.write_text(json.dumps({"runs": {k: summary(v) for k, v in runs.items()}, "translator": chosen,
                                 "rule": "8bit unless its tuning mean is more than 0.3 points lower",
                                 "median_ms_tuning": times}, indent=1))  # fmt: skip
    log(f"translator: {chosen} (tuning mean, translated: full {100 * full:.2f}, 8-bit {100 * eight:.2f})")


def test1(args) -> None:
    """Once: English, French as it is, and translated back by the translator chosen; criteria 38 to 40."""
    tuned = json.loads(TUNED.read_text())
    chosen = tuned["translator"]
    done = back(args, TESTS, chosen, "test")  # every test description, timed on an idle machine
    sets, fr = load_french()
    ms = [done[f][1] for f in sorted({fr[t] for n in TESTS for t in sets[n]})]
    earlier = json.loads(Path(str(BACK).format(chosen, "tuning")).read_text())
    same = [done[f][0] == earlier[f][0] for f in done if f in earlier]  # the translator gives the same English
    det = detector_rates(TESTS)
    b = Benchmarks(args)
    runs = {"english": measure(b, words(""), "test"), "french": measure(b, words("fr"), "test"),
            "translated": measure(b, words(f"fr-back-{chosen}", chosen, "test"), "test")}  # fmt: skip
    before = json.loads(TEST11.read_text())["fusion"]
    unchanged = all(runs["english"][k]["value"] == before[k]["value"] for k in BENCHES)
    pairs = {"translated vs french": paired(runs["translated"], runs["french"]),
             "translated vs english": paired(runs["translated"], runs["english"]),
             "french vs english": paired(runs["french"], runs["english"])}  # fmt: skip
    gain = runs["translated"]["mean"] - runs["french"]["mean"]
    median = statistics.median(ms)
    res = {"translator": chosen, "runs": {k: summary(v) for k, v in runs.items()}, "pairs": pairs,
           "eleventh_round_unchanged": unchanged, "detector": det, "same_as_tuning": [sum(same), len(same)],
           "translation_ms": {"median": median, "p90": float(np.percentile(ms, 90)), "max": max(ms), "n": len(ms)},
           "criteria": {"38": gain >= 0.01 and pairs["translated vs french"]["p"] < 0.05,
                        "39": det["english_rate"] <= 0.005 and det["french_rate"] >= 0.95,
                        "40": median <= 500.0}, "gain": gain}  # fmt: skip
    TEST1.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(f"the eleventh round's numbers again: {unchanged}")
    log(f"translated against as it is: {100 * gain:+.2f} points, p = {pairs['translated vs french']['p']:.2g}; "
        f"detector {100 * det['english_rate']:.2f} % / {100 * det['french_rate']:.2f} %; median {median:.0f} ms; "
        f"criteria {res['criteria']}")  # fmt: skip


# ---------------------------------------------------------------- part 2
def towers(args) -> None:
    """Both text towers with their weights in 8 bits (NNCF, INT8_ASYM, data-free, one scale per output channel),
    saved beside each model's files, and DiDeMo's bank embedded by them (results/cache)."""
    import didemo

    from whenframe.encoder import Encoder, _import_openvino, fetch

    ov = _import_openvino()  # first: NNCF then finds no usage-report package, and reports nothing
    import nncf  # not a dependency of the benchmarks: installed beside them (it pins numpy below 2.5)

    register_eight()
    out = {}
    for model, name in EIGHT.items():
        folder = fetch(MODELS[model])
        dest = folder / EIGHT_FILE
        if not dest.exists():
            log(f"{model}: compressing {MODELS[model].text_file}")
            m = ov.Core().read_model(str(folder / MODELS[model].text_file))
            m = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM)
            tmp = dest.with_name(dest.stem + ".tmp.xml")
            ov.save_model(m, str(tmp), compress_to_fp16=False)
            tmp.with_suffix(".bin").replace(dest.with_suffix(".bin"))
            tmp.replace(dest)
        source = folder / MODELS[model].text_file
        out[model] = {"file": EIGHT_FILE, "bytes": dest.stat().st_size + dest.with_suffix(".bin").stat().st_size,
                      "full_bytes": source.stat().st_size, "source": MODELS[model].text_file,
                      "source_sha256": sha256(source),
                      "sha256": {f.name: sha256(f) for f in (dest, dest.with_suffix(".bin"))}}  # fmt: skip
        bank = CACHE / f"didemo_bank_text_{name}.npy"
        if not bank.exists():
            texts = didemo.bank_texts(argparse.Namespace(train_annotations=args.annotations))
            enc = Encoder(name, backend="openvino")
            np.save(bank, np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)]))
        log(f"{model}: {out[model]['full_bytes'] / 1e6:.0f} MB -> {out[model]['bytes'] / 1e6:.0f} MB")
    (RESULTS / "round12_towers.json").write_text(json.dumps(out, indent=1))


def test2(args) -> None:
    """Once: the text towers in full precision (the eleventh round's numbers) and in 8 bits; criterion 41."""
    register_eight()
    b = Benchmarks(args)
    runs = {"full": measure(b, words(""), "test"), "8bit": measure(b, Words("text8", towers=dict(EIGHT)), "test")}
    changes = {k: runs["8bit"][k]["value"] - runs["full"][k]["value"] for k in BENCHES}
    loss = runs["full"]["mean"] - runs["8bit"]["mean"]
    res = {"runs": {k: summary(v) for k, v in runs.items()}, "changes": changes, "mean_lost": loss,
           "pair": paired(runs["8bit"], runs["full"]),
           "criteria": {"41": loss <= MARGIN and min(changes.values()) >= -0.005}}  # fmt: skip
    TEST2.write_text(json.dumps(res, indent=1))
    log(
        f"8 bits: {100 * loss:.2f} points lost on the mean; changes "
        + ", ".join(f"{k} {100 * v:+.2f}" for k, v in changes.items())
        + f"; criteria {res['criteria']}"
    )


def memory(args) -> None:
    """Criterion 42: the peak memory of a search of the demo's library, the text towers in 8 bits (each child
    process measured alone), and the size of the models a new library downloads."""
    # `whenframe search`, its text towers those of the files named (the 8-bit ones registered in their place)
    search = ("import sys; from dataclasses import replace; from whenframe import encoder; "
              "[encoder.MODELS.__setitem__(m, replace(encoder.MODELS[m], text_file=sys.argv[1], text_openvino=True))"
              " for m in ('siglip2-base', 'pe-core-b16')] if sys.argv[1] != 'full' else None; "
              "from whenframe import cli; sys.argv = ['whenframe'] + sys.argv[2:]; cli.main()")  # fmt: skip
    # each search in a process of its own, whose peak (KB) its parent reads once it has ended
    peak = ("import resource, subprocess, sys; subprocess.run(sys.argv[1:], check=True, stdout=subprocess.DEVNULL); "
            "print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)")  # fmt: skip
    out = {}
    for which in ("full", EIGHT_FILE):
        gb = []
        for q in args.queries:
            r = subprocess.run([sys.executable, "-c", peak, sys.executable, "-c", search, which, "search",
                                "--library", args.library, q], cwd=ROOT, capture_output=True, text=True,
                               check=True)  # fmt: skip
            gb.append(int(r.stdout.strip().splitlines()[-1]) * 1024 / 1e9)
        out[which] = {"peak_gb": max(gb), "each_gb": gb}
    from whenframe.encoder import fetch

    files = {}
    for model in ("siglip2-base", "pe-core-b16"):
        spec, folder = MODELS[model], fetch(MODELS[model])
        names = [spec.tokenizer_file, spec.vision_file, EIGHT_FILE]
        if spec.vision_file.endswith(".xml"):
            names.append(spec.vision_file.removesuffix(".xml") + ".bin")
        names.append(EIGHT_FILE.removesuffix(".xml") + ".bin")
        files[model] = {n: (folder / n).stat().st_size for n in names}
    download_gb = sum(sum(v.values()) for v in files.values()) / 1e9
    res = {"search": out, "download_gb": download_gb, "files": files,
           "criteria": {"42": out[EIGHT_FILE]["peak_gb"] <= 2.5 and download_gb <= 1.5}}  # fmt: skip
    MEMORY.write_text(json.dumps(res, indent=1))
    log(f"peak: full {out['full']['peak_gb']:.2f} GB, 8 bits {out[EIGHT_FILE]['peak_gb']:.2f} GB; download "
        f"{download_gb:.2f} GB; criteria {res['criteria']}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["french", "back", "detect", "tune", "test1", "towers", "test2", "memory"])
    for k in ("annotations", "test-annotations", "charades-annotations", "youcook2-annotations", "unav-annotations"):
        ap.add_argument(f"--{k}", help="as scripts/compare.py's")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--library", help="memory: the demo's library")
    ap.add_argument("--queries", nargs="+", default=["fireworks over the city"], help="memory: what is searched")
    args = ap.parse_args()
    if args.stage == "back":
        for p in PRECISIONS:
            back(args, TUNING, p, "tuning")
        return
    {"french": french, "detect": detect, "tune": tune, "test1": test1, "towers": towers, "test2": test2,
     "memory": memory}[args.stage](args)  # fmt: skip


if __name__ == "__main__":
    main()
