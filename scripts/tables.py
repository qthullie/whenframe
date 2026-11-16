"""Write the numbers of README.md and docs/RESULTS.md from results/*.json, so none is copied by hand.

Tables live between <!-- table NAME --> and <!-- /table --> markers; everything
outside the markers is left alone. `--check` exits with an error when a file
is out of date (used in CI).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = [ROOT / "README.md", ROOT / "docs" / "RESULTS.md"]
LABEL = {
    "scene": "shots (scene detection)",
    "frame_window": "fixed windows",
    "whenframe": "adaptive spans",
    "whenframe_bank": "adaptive spans + query bank",
    "frame_window_bank": "fixed windows + query bank",
    "whenframe_dedup": "default search, repeats skipped",
}
PRIMARY = "R@5 IoU>=0.5"
YC = PRIMARY  # the same metric on YouCook2


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def per_hour(seconds: float, hours: float) -> str:
    return f"{seconds / hours / 60:.1f}"


def label(r: dict, m: str) -> str:
    return f"**{LABEL[m]}** (default)" if m == r.get("default") else LABEL[m]


def criteria(r: dict) -> list[list[str]]:
    score = {m: r[m]["corpus"][PRIMARY] for m in r["methods"]}
    c = r["cost"]

    def gain(a: str, b: str) -> float:
        return score[a] - score[b]

    margin = score["whenframe"] - max(score["frame_window"], score["scene"])
    loss = gain(r["default"], "whenframe_dedup")
    kept = c["kept_fraction"]
    minutes = (c["decode_cpu"] + c["encode_cpu"]) / c["hours"] / 60
    rows = [
        ["1. adaptive spans beat both baselines by at least 2 points", f"{100 * margin:+.1f} points", margin >= 0.02],
        ["2. skipping repeats encodes at most 70 % of frames and loses at most 1 point",
         f"{100 * kept:.0f} % of frames, {-100 * loss:+.1f} points", kept <= 0.70 and loss <= 0.01],
        ["3. indexing takes at most 10 min of CPU time per hour of video", f"{minutes:.1f} min", minutes <= 10],
        ["4. the query bank adds at least 1 point to adaptive spans",
         f"{100 * gain('whenframe_bank', 'whenframe'):+.1f} points", gain("whenframe_bank", "whenframe") >= 0.01],
        ["5. the query bank adds at least 1 point to fixed windows",
         f"{100 * gain('frame_window_bank', 'frame_window'):+.1f} points",
         gain("frame_window_bank", "frame_window") >= 0.01],
    ]  # fmt: skip
    return [[name, measured, "yes" if ok else "no"] for name, measured, ok in rows]


def didemo_tables(r: dict) -> dict[str, str]:
    methods = r["methods"]
    corpus = table(
        ["method", "R@1", "R@5", "R@10", "R@100", "R@1 (0.7)", "R@5 (0.7)", "right video in top 5"],
        [[label(r, m), *(pct(r[m]["corpus"][f"R@{k} IoU>=0.5"]) for k in (1, 5, 10, 100)),
          pct(r[m]["corpus"]["R@1 IoU>=0.7"]), pct(r[m]["corpus"]["R@5 IoU>=0.7"]), pct(r[m]["video"]["R@5"])]
         for m in methods],
    )  # fmt: skip
    single = table(
        ["method", "R1 IoU>=0.5", "R1 IoU>=0.7", "mean IoU"],
        [[label(r, m), pct(r[m]["single"]["R1 IoU>=0.5"]), pct(r[m]["single"]["R1 IoU>=0.7"]),
          pct(r[m]["single"]["mIoU"])] for m in methods],
    )  # fmt: skip
    c = r["cost"]
    h, kept = c["hours"], c["kept_fraction"]
    cost = table(
        ["per hour of video", "CPU time (min)", "wall clock, 4 vCPU (min)"],
        [
            ["decoding (ffmpeg, 1 frame per second)", per_hour(c["decode_cpu"], h), per_hour(c["decode_wall"], h)],
            ["encoding every frame", per_hour(c["encode_cpu"], h), per_hour(c["encode_wall"], h)],
            [f"encoding, repeats skipped ({100 * kept:.0f} % of frames, estimated)",
             per_hour(c["encode_cpu"] * kept, h), per_hour(c["encode_wall"] * kept, h)],
            ["scene detection (needed by the scene baseline only)", per_hour(c["scene_cpu"], h),
             per_hour(c["scene_wall"], h)],
        ],
    )  # fmt: skip
    search = table(
        ["", "value"],
        [
            ["videos / queries / hours of video", f"{c['videos']} / {r['queries']} / {h:.2f}"],
            ["index size, every frame (MB per hour of video)", f"{c['frames'] * (768 * 2 + 4) / 1e6 / h:.1f}"],
            ["embedding the query (ms)", f"{c['text_ms']:.0f}"],
            *[[f"search, {LABEL[m]} (ms per query)", f"{r[m]['query_ms']:.1f}"] for m in methods],
        ],
    )  # fmt: skip
    curve = r["curve"]
    rows = [[f"skip repeats, threshold {p['threshold']:g}", f"{100 * p['kept']:.0f} %", pct(p["score"])]
            for p in curve["repeats"]]  # fmt: skip
    rows += [[f"one frame in {p['every']}" if p["every"] > 1 else "every frame", f"{100 * p['kept']:.0f} %",
              pct(p["score"])] for p in curve["rate"]]  # fmt: skip
    d, best_baseline = r["default"], max(("scene", "frame_window"), key=lambda m: r[m]["corpus"][PRIMARY])
    headline = (
        f"Searching {c['videos']:,} personal videos at once (DiDeMo test) for {r['queries']:,} descriptions "
        f"written by people, whenframe puts the right moment among its first five results "
        f"**{pct(r[d]['corpus'][PRIMARY])} %** of the time. Indexing one keyframe per shot, as some home video tools "
        f"do, finds it {pct(r['scene']['corpus'][PRIMARY])} % of the time, and the best of the two baselines "
        f"({LABEL[best_baseline]}) {pct(r[best_baseline]['corpus'][PRIMARY])} %."
    )
    return {
        "readme-headline": headline,
        "didemo-criteria": table(["criterion (fixed in advance)", "measured on test", "met"], criteria(r)),
        "didemo-corpus": corpus,
        "didemo-single": single,
        "didemo-cost": cost,
        "didemo-search": search,
        "didemo-curve": table(["frames encoded", "share", PRIMARY], rows),
    }


MODES = [("standard", "frame_window_bank", None), ("economy", "siglip32_bank", "siglip2-b32"),
         ("precision", "ensemble32_bank", None),
         ("MobileCLIP-S2 alone, research only", "mobileclip_bank", "mobileclip-s2"),
         ("SigLIP 2 + MobileCLIP-S2, research only", "ensemble_bank", None)]  # fmt: skip
ENSEMBLE_OF = {"ensemble32_bank": "siglip2-b32", "ensemble_bank": "mobileclip-s2"}


def round2_tables(r: dict, first: dict | None) -> dict[str, str]:
    score = {m: r[m]["corpus"][PRIMARY] for m in r["methods"]}
    c = r["cost"]

    def minutes(model: str | None, kind: str) -> float:
        t = c if model is None else c[model]
        return (t[f"decode_{kind}"] + t[f"encode_{kind}"]) / t["hours"] / 60

    def mode_cost(method: str, model: str | None) -> tuple[float, float, float]:
        if method in ENSEMBLE_OF:
            other = ENSEMBLE_OF[method]
            return (minutes(None, "cpu") + minutes(other, "cpu"), minutes(None, "wall") + minutes(other, "wall"),
                    c["text_ms"] + c[other]["text_ms"])  # fmt: skip
        if model is None:
            return minutes(None, "cpu"), minutes(None, "wall"), c["text_ms"]
        return minutes(model, "cpu"), minutes(model, "wall"), c[model]["text_ms"]

    loss = score["frame_window_bank"] - score["siglip32_bank"]
    gain = score["ensemble32_bank"] - score["frame_window_bank"]
    economy = minutes("siglip2-b32", "cpu")
    rows = [
        ["6. the economy mode loses at most 2 points", f"{-100 * loss:+.1f} points", loss <= 0.02],
        ["7. economy indexing takes at most 10 min of CPU time per hour of video", f"{economy:.1f} min",
         economy <= 10],
        ["8. the precision mode gains at least 1 point", f"{100 * gain:+.1f} points", gain >= 0.01],
    ]  # fmt: skip
    criteria = table(["criterion (fixed before the second run)", "measured on test", "met"],
                     [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    modes = table(
        ["mode", "R@1", "R@5", "R@10", "right video in top 5", "indexing, CPU (min/h)", "indexing, 4 vCPU (min/h)",
         "per search (ms)"],
        [[f"{mode} (`{m}`)", pct(r[m]["corpus"]["R@1 IoU>=0.5"]), pct(r[m]["corpus"][PRIMARY]),
          pct(r[m]["corpus"]["R@10 IoU>=0.5"]), pct(r[m]["video"]["R@5"]), f"{mode_cost(m, model)[0]:.1f}",
          f"{mode_cost(m, model)[1]:.1f}", f"{mode_cost(m, model)[2] + r[m]['query_ms']:.0f}"]
         for mode, m, model in MODES],
    )  # fmt: skip
    french = table(
        ["mode", "English R@5", "French R@5", "change", "right video in top 5, English / French"],
        [[mode, pct(r[m]["corpus"][PRIMARY]), pct(r["french"][m]["corpus"][PRIMARY]),
          f"{100 * (r['french'][m]['corpus'][PRIMARY] - r[m]['corpus'][PRIMARY]):+.1f}",
          f"{pct(r[m]['video']['R@5'])} / {pct(r['french'][m]['video']['R@5'])}"] for mode, m, _ in MODES],
    )  # fmt: skip
    compact = table(
        ["`--mode`", "right moment in the first five (DiDeMo)", "indexing, CPU min per hour of video",
         "search (ms)"],
        [[f"`{mode}`", f"{pct(r[m]['corpus'][PRIMARY])} %", f"{mode_cost(m, model)[0]:.1f}",
          f"{mode_cost(m, model)[2] + r[m]['query_ms']:.0f}"] for mode, m, model in MODES[:3]],
    )  # fmt: skip
    out = {"round2-criteria": criteria, "round2-modes": modes, "round2-french": french, "readme-modes": compact}
    if first:

        def numbers(res: dict, m: str) -> tuple:
            return res[m]["corpus"], res[m]["single"], res[m]["video"]

        same = [m for m in first["methods"] if numbers(r, m) == numbers(first, m)]
        out["round2-check"] = (f"The first round's {len(first['methods'])} methods, evaluated again: "
                               f"{len(same)} give exactly the same numbers as in the first run.")  # fmt: skip
    return out


def round3_tables() -> dict[str, str]:
    """OpenVINO on DiDeMo, YouCook2 and speech: whatever of the third round has been run."""
    out, rows = {}, []
    test, tune = RESULTS / "didemo_test_backends.json", RESULTS / "didemo_train_backends.json"
    if test.exists():
        r = json.loads(test.read_text())
        ov, ort = r["openvino"], r["onnxruntime"]
        cost = sum(ov["cpu_min_per_hour"].values())
        change = ov["corpus"][PRIMARY] - ort["corpus"][PRIMARY]
        rows += [["9. indexing with OpenVINO takes at most 10 min of CPU time per hour of video", f"{cost:.1f} min",
                  cost <= 10],
                 ["10. OpenVINO loses at most 1 point on the primary metric", f"{100 * change:+.1f} points",
                  change >= -0.01]]  # fmt: skip
        both = [json.loads(f.read_text()) for f in (tune, test) if f.exists()]
        out["round3-runtimes"] = table(
            ["image encoder", "videos", "R@5", "encoding, CPU min per hour of video", "CPU ms per frame",
             "won / lost"],
            [[f"{name}, {'tuning' if x['split'] == 'train' else 'test'} videos", f"{x['videos']:,}",
              pct(x[k]["corpus"][PRIMARY]),
              f"{x[k]['cpu_min_per_hour']['encode_cpu']:.1f}", f"{x[k]['encode_ms_per_frame']:.0f}",
              "" if k == "onnxruntime" else f"{x['wins']} / {x['losses']} (p = {x['p']:.2f})"]
             for x in both for k, name in (("onnxruntime", "ONNX Runtime, float32"),
                                           ("openvino", "OpenVINO, bfloat16"))],
        )  # fmt: skip
        out["readme-openvino"] = (
            f"With OpenVINO (`pip install 'whenframe[fast]'`) on a processor that computes in bfloat16, as this "
            f"one does, the standard mode indexes for {cost:.1f} CPU minutes per hour of video instead of "
            f"{sum(ort['cpu_min_per_hour'].values()):.1f}, and finds {pct(ov['corpus'][PRIMARY])} % of the moments "
            f"instead of {pct(ort['corpus'][PRIMARY])} % (DiDeMo test)."
        )
    yc, tuned = RESULTS / "youcook2_test.json", RESULTS / "youcook2_tuned.json"
    if yc.exists():
        r = json.loads(yc.read_text())
        gain = r["speech_bank"]["corpus"]["R@5 IoU>=0.5"] - r["frame_window_bank"]["corpus"]["R@5 IoU>=0.5"]
        asr = r["cost"]["transcribe_cpu_min_per_hour"]
        rows += [["11. speech adds at least 2 points on YouCook2", f"{100 * gain:+.1f} points", gain >= 0.02],
                 ["12. transcription takes at most 15 min of CPU time per hour of video", f"{asr:.1f} min",
                  asr <= 15]]  # fmt: skip
        labels = {
            "frame_window_bank": "fixed windows + query bank (the default search)",
            "whenframe_bank": "adaptive spans + query bank",
            "speech_bank": "the default search + speech",
        }
        out["youcook2"] = table(
            ["method (DiDeMo parameters)", "R1 IoU>=0.3", "R1 IoU>=0.5", "R1 IoU>=0.7", "mean IoU",
             f"all {r['videos']} videos: R@5 IoU>=0.3", "R@5 IoU>=0.5", "R@5, best frame inside the step"],
            [[label, pct(r[m]["single"]["R1 IoU>=0.3"]), pct(r[m]["single"]["R1 IoU>=0.5"]),
              pct(r[m]["single"]["R1 IoU>=0.7"]), pct(r[m]["single"]["mIoU"]), pct(r[m]["corpus"]["R@5 IoU>=0.3"]),
              pct(r[m]["corpus"]["R@5 IoU>=0.5"]), pct(r[m]["corpus_inside"]["R@5"])] for m, label in labels.items()],
        )  # fmt: skip
        c = r["cost"]
        out["youcook2-cost"] = table(
            ["per hour of video", "CPU time (min)"],
            [["decoding (ffmpeg, 1 frame per second)", f"{c['decode_cpu_min_per_hour']:.1f}"],
             [f"encoding the frames kept ({100 * c['frames_kept'] / c['frames_seen']:.0f} %), OpenVINO in "
              f"{'bfloat16' if 'bf' in c['precision'] else c['precision']}", f"{c['encode_cpu_min_per_hour']:.1f}"],
             ["transcribing (Whisper base, int8)", f"{c['transcribe_cpu_min_per_hour']:.1f}"]],
        )  # fmt: skip
        default, said = r["frame_window_bank"]["corpus"], r["speech_bank"]["corpus"]
        out["readme-youcook2"] = (
            f"On {r['videos']} YouCook2 cooking videos (5 minutes on average, searched all at once), the right "
            f"step is among the first five moments {pct(default['R@5 IoU>=0.5'])} % of the time, and "
            f"{pct(said['R@5 IoU>=0.5'])} % when what is said is searched too (`--speech`)."
        )
        w = r["speech_vs_default"]
        out["youcook2-speech"] = (
            f"Speech moves {w['wins']} test steps into the first five moments and {w['losses']} out of them "
            f"(sign test p = {w['p']:.2g}), with weight {r['speech']['weight']:g}, pad {r['speech']['pad']:g} s and "
            f"{r['speech']['normalise']} scores, chosen on the {json.loads(tuned.read_text())['tuning_steps']} "
            f"tuning steps."
        )  # fmt: skip
    if rows:
        out["round3-criteria"] = table(["criterion (fixed before the third run)", "measured on test", "met"],
                                       [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


def round4_tables() -> dict[str, str]:
    """Whatever of the fourth round has been run: speech with adaptive spans first."""
    out, rows = {}, []
    f = RESULTS / "youcook2_test_round4.json"
    if f.exists():
        r = json.loads(f.read_text())
        windows, spans = r["speech_bank"]["corpus"][YC], r["speech_spans"]["corpus"][YC]
        rows.append(["13. speech with adaptive spans beats speech with fixed windows by at least 2 points (YouCook2)",
                     f"{100 * (spans - windows):+.1f} points", spans - windows >= 0.02])  # fmt: skip
        labels = {
            "whenframe_bank": "adaptive spans + query bank",
            "speech_bank": "fixed windows + query bank + speech (third round)",
            "speech_spans": "adaptive spans + query bank + speech",
        }
        out["youcook2-round4"] = table(
            ["method", "R1 IoU>=0.3", "R1 IoU>=0.5", "R1 IoU>=0.7", "mean IoU",
             "all 252 videos: R@1 IoU>=0.5", "R@5 IoU>=0.5", "R@5, best frame inside the step"],
            [[label, pct(r[m]["single"]["R1 IoU>=0.3"]), pct(r[m]["single"]["R1 IoU>=0.5"]),
              pct(r[m]["single"]["R1 IoU>=0.7"]), pct(r[m]["single"]["mIoU"]), pct(r[m]["corpus"]["R@1 IoU>=0.5"]),
              pct(r[m]["corpus"][YC]), pct(r[m]["corpus_inside"]["R@5"])] for m, label in labels.items()],
        )  # fmt: skip
        w, a = r["vs_speech_bank"], r["vs_whenframe_bank"]
        out["youcook2-round4-wins"] = (
            f"Against speech with fixed windows, speech with adaptive spans moves {w['wins']} test steps into the "
            f"first five moments and {w['losses']} out of them (sign test p = {w['p']:.2g}); against the adaptive "
            f"spans without speech, {a['wins']} in and {a['losses']} out (p = {a['p']:.2g})."
        )
        out["readme-youcook2"] = (
            f"On 252 YouCook2 cooking videos (5 minutes on average, searched all at once), the right step is among "
            f"the first five moments {pct(r['whenframe_bank']['corpus'][YC])} % of the time with adaptive spans "
            f"(`--moments spans`), and {pct(spans)} % when what is said is searched too (indexed with `--speech`)."
        )
    if rows:
        out["round4-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                       [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


def charades_table() -> str | None:
    f = RESULTS / "charades_test.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    rows = [[LABEL[m], pct(r[m]["single"]["R1 IoU>=0.5"]), pct(r[m]["single"]["R1 IoU>=0.7"]),
             pct(r[m]["single"]["mIoU"]), pct(r[m]["corpus"][PRIMARY])] for m in r["methods"]]  # fmt: skip
    if "middle_window" in r:  # added after the run, as a reference
        x = r["middle_window"]["single"]
        rows.append(["*10 seconds in the middle, ignoring the description (reference added after the run)*",
                     pct(x["R1 IoU>=0.5"]), pct(x["R1 IoU>=0.7"]), pct(x["mIoU"]), ""])  # fmt: skip
    return table(["method (parameters from DiDeMo, unchanged)", "R1 IoU>=0.5", "R1 IoU>=0.7", "mean IoU",
                  f"all {r['videos']:,} videos at once: R@5 IoU>=0.5"], rows)  # fmt: skip


def explore_table() -> str | None:
    f = RESULTS / "didemo_train_explore.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text())
    return table(["variant (tuning videos)", "source", "R@5", "change", "won / lost", "sign test p"],
                 [[x["variant"], x["source"], pct(x["score"]), f"{100 * x['change']:+.1f}",
                   f"{x['wins']} / {x['losses']}", "" if not x["wins"] + x["losses"] else f"{x['p']:.2f}"]
                  for x in rows])  # fmt: skip


def rerank_table() -> str | None:
    f = RESULTS / "didemo_train_rerank.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    names = {"search": "the search's own order", "model alone": "SmolVLM-256M's answers alone",
             "search + model": "the search's score + SmolVLM-256M's, weight chosen on the other half",
             "best possible": "the best possible order, for a perfect judge"}  # fmt: skip
    head = f"the first {r['moments_per_description']} moments of {r['descriptions']} tuning descriptions"
    return table([head, "R@1", "R@5"], [[names[k], pct(v["R@1"]), pct(v["R@5"])] for k, v in r["orders"].items()])


def tuning_table() -> str | None:
    runs = [("first", "didemo_tuned_first_run.json"), ("second", "didemo_tuned_second_run.json"),
            ("third", "didemo_tuned.json")]  # fmt: skip
    found = [(name, json.loads((RESULTS / f).read_text())) for name, f in runs if (RESULTS / f).exists()]
    if not found:
        return None
    methods = [m for m in LABEL if m != "whenframe_dedup" and any(m in t for _, t in found)]
    rows = [[LABEL[m], *(pct(t[m]["train_score"]) if m in t else "" for _, t in found)] for m in methods]
    return table(["method", *(f"{name} run" for name, _ in found)], rows)


def main() -> int:
    tables = {}
    f = RESULTS / "didemo_test.json"
    if f.exists():
        tables.update(didemo_tables(json.loads(f.read_text())))
    tuning = tuning_table()
    if tuning:
        tables["didemo-tuning"] = tuning
    explore = explore_table()
    if explore:
        tables["didemo-explore"] = explore
    charades = charades_table()
    if charades:
        tables["charades"] = charades
    rerank = rerank_table()
    if rerank:
        tables["didemo-rerank"] = rerank
    tables.update(round3_tables())
    tables.update(round4_tables())  # after the third round's: replaces its README sentence on YouCook2
    f2 = RESULTS / "didemo_test_round2.json"
    if f2.exists():
        tables.update(round2_tables(json.loads(f2.read_text()), json.loads(f.read_text()) if f.exists() else None))

    def fill(m):
        name = m.group(1)
        return f"<!-- table {name} -->\n{tables[name]}\n<!-- /table -->" if name in tables else m.group(0)

    stale = []
    for doc in DOCS:
        text = doc.read_text()
        new = re.sub(r"<!-- table ([\w-]+) -->.*?<!-- /table -->", fill, text, flags=re.S)
        if new == text:
            continue
        if "--check" in sys.argv:
            stale.append(doc.relative_to(ROOT))
        else:
            doc.write_text(new)
            print(f"updated {doc.relative_to(ROOT)}")
    if stale:
        print(f"out of date: {', '.join(map(str, stale))}; run python scripts/tables.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
