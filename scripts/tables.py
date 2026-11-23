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
    compact_rows = [[f"`{mode}`", f"{pct(r[m]['corpus'][PRIMARY])} %", f"{mode_cost(m, model)[0]:.1f}",
                     f"{mode_cost(m, model)[2] + r[m]['query_ms']:.0f}"] for mode, m, model in MODES[:3]]  # fmt: skip
    big = RESULTS / "didemo_test_so400m.json"
    if big.exists():  # the fourth round's larger model, measured with OpenVINO only
        b = json.loads(big.read_text())
        c = b["cost"]
        compact_rows.append(["`large`", f"{pct(b['siglip2-so400m']['corpus'][PRIMARY])} %",
                             f"{sum(c['cpu_min_per_hour'].values()):.1f} (OpenVINO)",
                             f"{c['query_ms'] + c.get('search_ms', 0.0):.0f}"])  # fmt: skip
    compact = table(["`--mode`", "right moment in the first five (DiDeMo)", "indexing, CPU min per hour of video",
                     "search (ms)"], compact_rows)  # fmt: skip
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
            f"(placed by default in a library of long videos), and {pct(spans)} % when what is said is searched "
            f"too (indexed with `--speech`)."
        )
    f = RESULTS / "didemo_test_so400m.json"
    if f.exists():
        r = json.loads(f.read_text())
        big, base, c = r["siglip2-so400m"], r["base"], r["cost"]
        gain = big["corpus"][PRIMARY] - base["corpus"][PRIMARY]
        cost = sum(c["cpu_min_per_hour"].values())
        rows += [["14. So400m beats the base model by at least 2 points (DiDeMo, both with OpenVINO)",
                  f"{100 * gain:+.1f} points", gain >= 0.02],
                 ["15. indexing with So400m takes at most 30 min of CPU time per hour of video", f"{cost:.1f} min",
                  cost <= 30]]  # fmt: skip
        ov = json.loads((RESULTS / "didemo_test_backends.json").read_text())["openvino"]
        base_cost = sum(ov["cpu_min_per_hour"].values())
        text_ms = json.loads((RESULTS / "didemo_test.json").read_text())["cost"]["text_ms"]  # ONNX Runtime
        out["round4-so400m"] = table(
            ["image encoder (OpenVINO, bfloat16)", "R@1", "R@5", "R@10", "right video in top 5",
             "CPU ms per frame", "indexing, CPU min per hour of video", "embedding a description (ms)"],
            [["SigLIP 2 base (the default)", pct(base["corpus"]["R@1 IoU>=0.5"]), pct(base["corpus"][PRIMARY]),
              pct(base["corpus"]["R@10 IoU>=0.5"]), pct(base["video"]["R@5"]), f"{ov['encode_ms_per_frame']:.0f}",
              f"{base_cost:.1f}", f"{text_ms:.0f}"],
             ["SigLIP 2 So400m", pct(big["corpus"]["R@1 IoU>=0.5"]), pct(big["corpus"][PRIMARY]),
              pct(big["corpus"]["R@10 IoU>=0.5"]), pct(big["video"]["R@5"]), f"{c['encode_ms_per_frame']:.0f}",
              f"{cost:.1f}", f"{c['query_ms']:.0f}"]],
        )  # fmt: skip
        out["round4-so400m-wins"] = (
            f"So400m moves {r['wins']} test descriptions into the first five moments and {r['losses']} out of them "
            f"(sign test p = {r['p']:.2g}), with lower-case descriptions and the base model's query bank values "
            f"(beta {r['params']['beta']:g}, strength {r['params']['lam']:g}), chosen on the tuning videos."
        )
    f = RESULTS / "didemo_test_motion.json"
    if f.exists():
        r = json.loads(f.read_text())
        still, both = r["rows"]
        gain = both["score"] - still["score"]
        rows.append([f"16. the motion grids add at least 1 point (DiDeMo, weight {r['weight']:g})",
                     f"{100 * gain:+.1f} points", gain >= 0.01])  # fmt: skip
        out["round4-motion"] = table(
            ["DiDeMo test (OpenVINO, repeats skipped)", "R@1", "R@5", "R@10", "right video in top 5",
             f"R@5, the {r['motion_queries']:,} descriptions of movement", "R@5, the others"],
            [[label, pct(x["corpus"]["R@1 IoU>=0.5"]), pct(x["score"]), pct(x["corpus"]["R@10 IoU>=0.5"]),
              pct(x["video"]["R@5"]), pct(x["motion"]), pct(x["other"])]
             for label, x in (("frames (the default)", still), (f"frames + {r['weight']:g} x grids", both))],
        )  # fmt: skip
        c = r["cost"]
        out["round4-motion-wins"] = (
            f"The grids move {both['wins']} test descriptions into the first five moments and {both['losses']} out "
            f"of them (sign test p = {both['p']:.2g}); among the descriptions of movement, {both['motion_wins']} in "
            f"and {both['motion_losses']} out (p = {both['motion_p']:.2g}). Decoding the frames for the grids and "
            f"embedding them took {(c['decode_cpu'] + c['encode_cpu']) / 60 / c['hours']:.1f} CPU minutes per hour "
            f"of video."
        )
    f = RESULTS / "unav_test.json"
    if f.exists():
        r = json.loads(f.read_text())
        gain = r["both"]["p_at_k"] - r["frames"]["p_at_k"]
        rows.append(["17. frames and sound together beat the frames alone by at least 4 points (UnAV-100, P@10)",
                     f"{100 * gain:+.1f} points", gain >= 0.04])  # fmt: skip
        names = {"frames": "frames (SigLIP 2, the default search)",
                 "sound": f"sound alone (CLAP, {r['length']:g}-second windows)",
                 "both": f"frames + {r['weight']:g} x sound (`--sounds`)"}  # fmt: skip
        out["round4-sounds"] = table(
            [f"UnAV-100, {r['videos']} test videos, {r['labels']} kinds of events", "right among the first ten (P@10)",
             "average precision, first 100 (mAP)"],
            [[label, pct(r[m]["p_at_k"]), pct(r[m]["map"])] for m, label in names.items()],
        )  # fmt: skip
        per = {k: (r["frames"]["labels"][k]["p_at_k"], r["sound"]["labels"][k]["p_at_k"],
                   r["both"]["labels"][k]["p_at_k"], r["frames"]["labels"][k]["events"])
               for k in r["frames"]["labels"]}  # fmt: skip
        order = sorted(per, key=lambda k: per[k][2] - per[k][0])
        out["round4-sounds-labels"] = table(
            ["kind of event (test events)", "frames", "sound", "frames + sound"],
            [[f"{k} ({per[k][3]})", *(pct(v) for v in per[k][:3])] for k in [*order[-8:][::-1], *order[:4]]],
        )  # fmt: skip
        c, w = r["cost"], r["both_vs_frames"]
        heard = c["sound_cpu_min_per_hour"] + c[f"clap_cpu_{r['length']:g}_min_per_hour"]
        out["round4-sounds-wins"] = (
            f"With the sound, {w['up']} kinds of events have more right moments among the first ten and {w['down']} "
            f"fewer (sign test p = {w['p']:.2g}). Decoding the sound and embedding its {r['length']:g}-second "
            f"windows took {heard:.1f} CPU minutes per hour of video."
        )
    if rows:
        out["round4-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                       [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


def round5_tables() -> dict[str, str]:
    """The fifth round: whenframe's own model, the runs on the tuning videos and, once run, the test."""
    out = {}
    # the fifth and sixth rounds' runs; the seventh's (v3-*) and the ninth's (v4-*, on PE core B/16's frames) keep
    # the best check of two sets, in round7-tuning and round9-tuning
    runs = sorted(f for f in (RESULTS / "temporal").glob("*.json") if not f.name.startswith(("v3", "v4")))
    if runs:
        rows = []
        for f in runs:
            r = json.loads(f.read_text())
            c = r["config"]
            if not r["checks"]:
                continue
            best = max(r["checks"], key=lambda k: k["R@5"])
            what = ("adapter and head" if c["head"] else "adapter") if c["lr"] else "head alone"
            rows.append([f"{c['name']}: {what}", f"{c['steps']} x {c['batch']}", f"{c.get('between', 1.0):g}",
                         f"{c['within']:g}", f"{c['leash']:g}", f"{best['step']}", pct(best["R@5"]),
                         pct(best["video_R@5"]), pct(best["single_R1@0.5"]), f"{1 - best['leash']:.3f}"])  # fmt: skip
        without = json.loads(runs[0].read_text())["tuning_without"]
        rows.insert(0, ["SigLIP 2 as it is", "", "", "", "", "", pct(without["R@5"]), pct(without["video_R@5"]),
                        pct(without["single_R1@0.5"]), "1"])  # fmt: skip
        out["round5-tuning"] = table(
            ["500 tuning videos", "steps x descriptions", "between", "within", "leash", "best check", "R@5",
             "right video in top 5", "single R1", "cosine to SigLIP 2"], rows)  # fmt: skip
    f = RESULTS / "temporal_test.json"
    if not f.exists():
        return out
    r = json.loads(f.read_text())
    d, ch, yc, ua = r["didemo"], r["charades"], r["youcook2"], r["unav"]
    gain = d["adapter"]["corpus"][PRIMARY] - d["without"]["corpus"][PRIMARY]
    single, spans = "R1 IoU>=0.5", "speech_spans"
    losses = {
        "Charades-STA, R1 at IoU >= 0.5": ch["adapter"]["single"][single] - ch["without"]["single"][single],
        "YouCook2, R@5, speech + spans": yc[f"adapter_{spans}"]["corpus"][YC] - yc[f"without_{spans}"]["corpus"][YC],
        "UnAV-100, P@10, frames + sound": ua["adapter_both"]["p_at_k"] - ua["without_both"]["p_at_k"],
    }
    cost = d["adapter_cost"]["cpu_min_per_hour"]
    rows = [["18. the adapted frames add at least 1 point on the DiDeMo test", f"{100 * gain:+.1f} points",
             gain >= 0.01],
            ["19. they lose at most 1 point on each other benchmark",
             "; ".join(f"{k} {100 * v:+.1f}" for k, v in losses.items()), min(losses.values()) >= -0.01],
            ["20. the adapter adds at most 1 CPU minute per hour of video", f"{cost:.2f} min",
             cost <= 1.0]]  # fmt: skip
    out["round5-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                   [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    out["round5-didemo"] = table(
        ["DiDeMo test (OpenVINO, repeats skipped)", "R@1", "R@5", "R@10", "R@100", "R@5 (0.7)", "right video in top 5",
         "single R1", "single R1 (0.7)", "single mIoU"],
        [[label, pct(x["corpus"]["R@1 IoU>=0.5"]), pct(x["corpus"][PRIMARY]), pct(x["corpus"]["R@10 IoU>=0.5"]),
          pct(x["corpus"]["R@100 IoU>=0.5"]), pct(x["corpus"]["R@5 IoU>=0.7"]), pct(x["video"]["R@5"]),
          pct(x["single"]["R1 IoU>=0.5"]), pct(x["single"]["R1 IoU>=0.7"]), pct(x["single"]["mIoU"])]
         for label, x in (("SigLIP 2's frames", d["without"]), ("**whenframe's model**", d["adapter"]))],
    )  # fmt: skip
    w = d["adapter_vs_without"]

    def wins(v, unit="descriptions"):
        up, down = (v["wins"], v["losses"]) if "wins" in v else (v["up"], v["down"])
        words = ("in", "out") if unit == "descriptions" else ("labels up", "down")
        return f"{up} {words[0]}, {down} {words[1]} (p = {v['p']:.2g})"

    out["round5-transfer"] = table(
        ["benchmark, search", "metric", "SigLIP 2's frames", "whenframe's model", "descriptions moved"],
        [["Charades-STA, the given video", "R1 IoU>=0.5", pct(ch["without"]["single"]["R1 IoU>=0.5"]),
          pct(ch["adapter"]["single"]["R1 IoU>=0.5"]), ""],
         ["Charades-STA, all 1,334 videos", "R@5", pct(ch["without"]["corpus"][PRIMARY]),
          pct(ch["adapter"]["corpus"][PRIMARY]), wins(ch["adapter_vs_without"])],
         ["YouCook2, fixed windows", "R@5", pct(yc["without_windows"]["corpus"][YC]),
          pct(yc["adapter_windows"]["corpus"][YC]), wins(yc["adapter_vs_without_windows"])],
         ["YouCook2, speech + adaptive spans", "R@5", pct(yc["without_speech_spans"]["corpus"][YC]),
          pct(yc["adapter_speech_spans"]["corpus"][YC]), wins(yc["adapter_vs_without_speech_spans"])],
         ["UnAV-100, frames", "P@10", pct(ua["without_frames"]["p_at_k"]), pct(ua["adapter_frames"]["p_at_k"]),
          wins(ua["adapter_vs_without_frames"], "labels")],
         ["UnAV-100, frames + sound", "P@10", pct(ua["without_both"]["p_at_k"]), pct(ua["adapter_both"]["p_at_k"]),
          wins(ua["adapter_vs_without_both"], "labels")]],
    )  # fmt: skip
    out["round5-wins"] = (
        f"On the DiDeMo test the model moves {w['wins']} descriptions into the first five moments and "
        f"{w['losses']} out of them (sign test p = {w['p']:.2g}). It took "
        f"{d['adapter_cost']['cpu_ms_per_frame']:.2f} ms of CPU per frame, {cost:.2f} CPU minutes per hour of video."
    )
    return out


BENCH = {"didemo": "DiDeMo", "charades": "Charades-STA", "youcook2": "YouCook2", "unav": "UnAV-100"}


def round6_tables() -> dict[str, str]:
    """The sixth round: five ways of embedding frames, compared the same way; whenframe's model v2."""
    out = {}
    f = RESULTS / "compare_tuned.json"
    if f.exists():
        t = json.loads(f.read_text())
        names = {"v2-infonce": "v2, InfoNCE, one run", "v2-sigmoid": "v2, SigLIP's own loss (FLYP)",
                 "v2-soup": "v2, soup of three runs", "v2-soup-a0.25": "v2 soup, alpha 0.25",
                 "v2-soup-a0.5": "v2 soup, alpha 0.5", "v2-soup-a0.75": "v2 soup, alpha 0.75"}  # fmt: skip
        rows = [[label, pct(t[k]["didemo"]), pct(t[k]["youcook2"]), pct(t[k]["unav"]), f"{100 * t[k]['mean']:.2f}"]
                for k, label in (("base", "SigLIP 2 B/16 (standard)"), ("v1", "B/16 + model v1"), *names.items())
                if k in t]  # fmt: skip
        out["round6-tuning"] = table(["tuning sets", "DiDeMo R@5", "YouCook2 R@5", "UnAV-100 P@10", "mean"], rows)
    f = RESULTS / "compare_test.json"
    if not f.exists():
        return out
    r = json.loads(f.read_text())
    label = {"b32": "SigLIP 2 B/32 (`--mode economy`)", "base": "SigLIP 2 B/16 (standard)",
             "so400m": "SigLIP 2 So400m (`--mode large`)", "v1": "B/16 + whenframe's model v1",
             "v2": "B/16 + whenframe's model v2"}  # fmt: skip
    cost = json.loads((RESULTS / "compare_cost.json").read_text()) if (RESULTS / "compare_cost.json").exists() else {}
    adapter = 0.0
    t = RESULTS / "temporal_test.json"
    if t.exists():
        adapter = json.loads(t.read_text())["didemo"]["adapter_cost"]["cpu_min_per_hour"]
    encoder = {"b32": "siglip2-b32", "base": "siglip2-base", "so400m": "siglip2-so400m", "v1": "siglip2-base",
               "v2": "siglip2-base"}  # fmt: skip

    def minutes(n: str) -> str:
        c = cost.get(encoder.get(n, ""), {}).get("cpu_min_per_hour")
        return "" if c is None else f"{c + (adapter if n in ('v1', 'v2') else 0.0):.1f}"

    rows = []
    for n in r["ranking"]:
        x = r[n]
        rows.append([label.get(n, n), pct(x["didemo"]["value"]), pct(x["charades"]["value"]),
                     pct(x["youcook2"]["value"]), pct(x["unav"]["value"]), f"**{100 * x['mean']:.2f}**",
                     minutes(n)])  # fmt: skip
    out["round6-test"] = table(["test sets, ranked by the mean", "DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5",
                                "UnAV-100 P@10", "mean", "indexing, CPU min per hour"], rows)  # fmt: skip
    pairs = [(a, b) for a in r["ranking"] for b in r["ranking"] if r["ranking"].index(a) < r["ranking"].index(b)]
    out["round6-pairs"] = table(
        ["the first finds more than the second?", "items better", "items worse", "sign test p", "said"],
        [[f"{label.get(a, a)} / {label.get(b, b)}", str(r["pairs"][f"{a} vs {b}"]["wins"]),
          str(r["pairs"][f"{a} vs {b}"]["losses"]), f"{r['pairs'][f'{a} vs {b}']['p']:.2g}",
          "yes" if r["pairs"][f"{a} vs {b}"]["p"] < 0.05 and r[a]["mean"] > r[b]["mean"] else "no"]
         for a, b in pairs],
    )  # fmt: skip
    first = RESULTS / "didemo_test.json"
    if first.exists() and {"v2", "base", "so400m"} <= set(r):
        shots = json.loads(first.read_text())["scene"]["corpus"][PRIMARY]
        times = (cost.get("siglip2-so400m", {}).get("cpu_min_per_hour", 0) /
                 max(cost.get("siglip2-base", {}).get("cpu_min_per_hour", 1), 1e-9))  # fmt: skip
        out["readme-headline"] = (
            f"Searching 979 personal videos at once (DiDeMo test) for 3,902 descriptions written by people, "
            f"whenframe puts the right moment among its first five results **{pct(r['v2']['didemo']['value'])} %** "
            f"of the time with its own model ({pct(r['base']['didemo']['value'])} % without it; "
            f"{pct(r['so400m']['didemo']['value'])} % with SigLIP 2 So400m, for {times:.1f} times the CPU time). "
            f"Indexing one keyframe per shot, as some home video tools do, finds it {pct(shots)} % of the time."
        )
        modes = [("`standard`, with whenframe's model (the default)", "v2"), ("`standard --no-temporal`", "base"),
                 ("`economy`", "b32"), ("`large`", "so400m")]  # fmt: skip
        out["readme-models"] = table(
            ["`--mode`", "right moment in the first five (DiDeMo)", "mean of the four benchmarks",
             "indexing, CPU min per hour of video (OpenVINO)"],
            [[m, f"{pct(r[n]['didemo']['value'])} %", f"{100 * r[n]['mean']:.2f}", minutes(n)] for m, n in modes
             if n in r],
        )  # fmt: skip
    if "v2" in r and "base" in r:
        v2, base = r["v2"], r["base"]
        p = r["pairs"]["v2 vs base"]
        losses = {k: v2[k]["value"] - base[k]["value"] for k in ("didemo", "charades", "youcook2", "unav")}
        rows = [
            [
                "21. v2's mean is higher than SigLIP 2 B/16's, sign test p < 0.05",
                f"{100 * (v2['mean'] - base['mean']):+.2f} points, p = {p['p']:.2g}",
                v2["mean"] > base["mean"] and p["p"] < 0.05,
            ],
            [
                "22. v2 loses at most 1 point on each test against B/16",
                "; ".join(f"{BENCH[k]} {100 * v:+.1f}" for k, v in losses.items()),
                min(losses.values()) >= -0.01,
            ],
        ]
        if "v1" in r:
            rows.append(["23. v2's mean is higher than v1's", f"{100 * (v2['mean'] - r['v1']['mean']):+.2f} points",
                         v2["mean"] > r["v1"]["mean"]])  # fmt: skip
        out["round6-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                       [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


# the seventh round's runs on the tuning sets, in the order of its choices (docs/PROTOCOL.md): name, choice, label
SEVENTH = [
    ("v3-noteacher", 1, "v3: all five sets of descriptions, no teacher"),
    ("v3-teacher", 1, "+ So400m's losses, on Charades-STA's frames"),
    ("v3-teacher-all", 1, "+ So400m's losses on DiDeMo's and MSR-VTT's frames too"),
    ("v3-leash30", 2, "the leash's weight 30, not 100"),
    ("v3-wide", 3, "width 512 and four layers, not 256 and two"),
    ("v3-soup", 4, "a soup of three runs (data seeds 0, 1, 2)"),
    ("v3-a0.25", 5, "alpha 0.25"),
    ("v3-a0.5", 5, "alpha 0.5"),
    ("v3-a0.75", 5, "alpha 0.75"),
]
MAIN7 = ["b32", "base", "so400m", "v1", "v2", "v3"]
LEFT_OUT = {"v3-no-charades": "charades", "v3-no-youcook2": "youcook2"}  # reported beside, deciding nothing
LABEL7 = {"b32": "SigLIP 2 B/32 (`--mode economy`)", "base": "SigLIP 2 B/16 (standard)",
          "so400m": "SigLIP 2 So400m (`--mode large`)", "v1": "B/16 + whenframe's model v1",
          "v2": "B/16 + whenframe's model v2", "v3": "B/16 + whenframe's model v3"}  # fmt: skip


def seventh_choices(t: dict) -> tuple[dict[str, bool], str | None]:
    """Each run of SEVENTH found on the tuning sets: kept or not, by the rule (a higher mean than the run kept
    so far), and the run kept at the end; alpha: the best of the four, 1 being the run kept before it."""
    kept, current = {}, None
    for name, choice, _ in SEVENTH:
        if name not in t:
            continue
        if current is None:
            kept[name], current = True, name
        elif choice == 5:
            best = max([current] + [n for n, c, _ in SEVENTH if c == 5 and n in t], key=lambda n: t[n]["mean"])
            kept[name] = name == best
            if kept[name]:
                current = name
        else:
            kept[name] = t[name]["mean"] > t[current]["mean"]
            if kept[name]:
                current = name
    return kept, current


def round7_tables() -> dict[str, str]:
    """The seventh round: whenframe's model v3, taught by So400m, trained on more moments."""
    out = {}
    f = RESULTS / "round7_tuned.json"
    if f.exists():
        t = json.loads(f.read_text())
        kept, _ = seventh_choices(t)
        cols = ("didemo", "charades", "youcook2", "unav")
        rows = [[LABEL7[k], *(pct(t[k][c]) for c in cols), f"{100 * t[k]['mean']:.2f}", ""] for k in ("base", "v2")
                if k in t]  # fmt: skip
        rows += [[label, *(pct(t[k][c]) for c in cols), f"{100 * t[k]['mean']:.2f}", "yes" if kept[k] else "no"]
                 for k, _, label in SEVENTH if k in t]  # fmt: skip
        rows += [[f"*v3 without {BENCH[c]}'s videos (beside)*", *(pct(t[k][x]) for x in cols),
                  f"{100 * t[k]['mean']:.2f}", ""] for k, c in LEFT_OUT.items() if k in t]  # fmt: skip
        out["round7-tuning"] = table(["tuning sets", "DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5", "UnAV-100 P@10",
                                      "mean", "kept"], rows)  # fmt: skip
    f = RESULTS / "round7_test.json"
    if not f.exists():
        return out
    r = json.loads(f.read_text())
    cost = json.loads((RESULTS / "compare_cost.json").read_text()) if (RESULTS / "compare_cost.json").exists() else {}
    encoder = {"b32": "siglip2-b32", "so400m": "siglip2-so400m"}

    def minutes(n: str) -> float | None:
        c = cost.get(encoder.get(n, "siglip2-base"), {}).get("cpu_min_per_hour")
        return None if c is None else c + r[n].get("adapter_cost", {}).get("cpu_min_per_hour", 0.0)

    main = [n for n in r["ranking"] if n in MAIN7]
    rows = [[LABEL7[n], pct(r[n]["didemo"]["value"]), pct(r[n]["charades"]["value"]), pct(r[n]["youcook2"]["value"]),
             pct(r[n]["unav"]["value"]), f"**{100 * r[n]['mean']:.2f}**",
             "" if minutes(n) is None else f"{minutes(n):.1f}"] for n in main]  # fmt: skip
    out["round7-test"] = table(["test sets, ranked by the mean", "DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5",
                                "UnAV-100 P@10", "mean", "indexing, CPU min per hour"], rows)  # fmt: skip
    pairs = [(a, b) for i, a in enumerate(main) for b in main[i + 1 :]]

    def said(a: str, b: str) -> bool:
        return r["pairs"][f"{a} vs {b}"]["p"] < 0.05 and r[a]["mean"] > r[b]["mean"]

    out["round7-pairs"] = table(
        ["the first finds more than the second?", "items better", "items worse", "sign test p", "said"],
        [[f"{LABEL7[a]} / {LABEL7[b]}", str(r["pairs"][f"{a} vs {b}"]["wins"]),
          str(r["pairs"][f"{a} vs {b}"]["losses"]), f"{r['pairs'][f'{a} vs {b}']['p']:.2g}",
          "yes" if said(a, b) else "no"] for a, b in pairs],
    )  # fmt: skip
    left = [(k, c) for k, c in LEFT_OUT.items() if k in r]
    if left and "v3" in r and "base" in r:
        out["round7-left-out"] = table(
            ["the test of a collection left out of training", "B/16 (never trained on it)", "v3 trained without it",
             "v3"],
            [[f"{BENCH[c]}, {('R1 at IoU >= 0.5' if c == 'charades' else 'R@5, speech + spans')}",
              pct(r["base"][c]["value"]), pct(r[k][c]["value"]), pct(r["v3"][c]["value"])] for k, c in left],
        )  # fmt: skip
    if not {"v3", "v2", "base", "so400m"} <= set(r):
        return out
    v3 = r["v3"]
    losses = {k: v3[k]["value"] - r["base"][k]["value"] for k in ("didemo", "charades", "youcook2", "unav")}
    p_large, p_v2 = r["pairs"]["v3 vs so400m"], r["pairs"]["v3 vs v2"]
    met = {
        24: v3["mean"] > r["so400m"]["mean"] and p_large["p"] < 0.05,
        25: v3["mean"] > r["v2"]["mean"] and p_v2["p"] < 0.05,
        26: min(losses.values()) >= -0.01,
    }
    rows = [
        ["24. v3's mean is higher than So400m's, sign test p < 0.05",
         f"{100 * (v3['mean'] - r['so400m']['mean']):+.2f} points, p = {p_large['p']:.2g}", met[24]],
        ["25. v3's mean is higher than v2's, p < 0.05",
         f"{100 * (v3['mean'] - r['v2']['mean']):+.2f} points, p = {p_v2['p']:.2g}", met[25]],
        ["26. v3 loses at most 1 point on each test against B/16",
         "; ".join(f"{BENCH[k]} {100 * v:+.1f}" for k, v in losses.items()), met[26]],
    ]  # fmt: skip
    out["round7-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                   [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    if not (met[25] and met[26]):  # v2 stays the default: the README's tables stay the sixth round's
        return out
    first = RESULTS / "didemo_test.json"
    shots = json.loads(first.read_text())["scene"]["corpus"][PRIMARY] if first.exists() else None
    times = minutes("so400m") / minutes("v3") if minutes("so400m") and minutes("v3") else None
    large = (
        f"; {pct(r['so400m']['didemo']['value'])} % with SigLIP 2 So400m, for {times:.1f} times the CPU time"
        if times
        else ""
    )
    headline = (
        f"Searching 979 personal videos at once (DiDeMo test) for 3,902 descriptions written by people, "
        f"whenframe puts the right moment among its first five results **{pct(v3['didemo']['value'])} %** of the "
        f"time with its own model ({pct(r['base']['didemo']['value'])} % without it{large})."
    )
    pub = RESULTS / "published_didemo.json"
    if pub.exists():  # the right video too, beside the best system published for this search
        best = max(x["R@5"] for x in json.loads(pub.read_text())["video_retrieval"].values())
        video = 100 * v3["didemo"]["full"]["video"]["R@5"]
        headline = headline[:-1] + (
            f", and the right video among its first five **{video:.1f} %** of the time, {video / best:.1f} times as "
            f"often as the best published system trained for this search on a GPU ({best:.1f} %)."
        )
    if met[24]:  # the protocol's sentence, when criterion 24 is met
        headline += (
            f" At the standard mode's cost, it finds more moments than its large mode over four benchmarks (mean "
            f"{100 * v3['mean']:.2f} against {100 * r['so400m']['mean']:.2f}, sign test p = {p_large['p']:.2g})."
        )
        # where that lead is: on the benchmarks whose training videos v3 learnt from, and without them
        so, left = r["so400m"], {c: r[k][c]["value"] for k, c in LEFT_OUT.items() if k in r}
        lead = [k for k in ("didemo", "charades", "youcook2", "unav") if v3[k]["value"] > so[k]["value"]]
        if lead == ["charades", "youcook2"] and set(left) == set(lead):
            headline += (
                f" Its lead is on Charades-STA and YouCook2, whose training videos it learnt from: trained without "
                f"each one's videos, it finds {pct(left['charades'])} and {pct(left['youcook2'])} % of their moments, "
                f"against {pct(so['charades']['value'])} and {pct(so['youcook2']['value'])} % for So400m."
            )
    if shots is not None:
        headline += f" Indexing one keyframe per shot, as some home video tools do, finds it {pct(shots)} % of the "
        headline += "time."
    out["readme-headline"] = headline
    if met[24] and times:  # the README's opening: what the model does, in one figure, with its limit
        # on collections it never learnt from: the runs trained without one's videos, and UnAV-100 for v3 itself
        unseen = [r[k][c]["value"] - r["base"][c]["value"] for k, c in LEFT_OUT.items() if k in r]
        unseen.append(v3["unav"]["value"] - r["base"]["unav"]["value"])
        out["readme-pitch"] = (
            f"**Its own model lets a small encoder find more than one that takes {times:.1f} times the CPU "
            "time.** Trained on more than 100,000 descriptions written by people, it reads each frame with the "
            "frames around it: an action, not only a picture. Over four public benchmarks, it lifts SigLIP 2 B/16 "
            f"above SigLIP 2 So400m (mean {100 * v3['mean']:.2f} against {100 * r['so400m']['mean']:.2f}, sign test "
            f"p = {p_large['p']:.2g}). On collections it never learnt from, it changes SigLIP 2 B/16's results by "
            f"{100 * min(unseen):+.1f} to {100 * max(unseen):+.1f} points."
        )
    modes = [("`standard`, with whenframe's model (the default)", "v3"), ("`standard --no-temporal`", "base"),
             ("`economy`", "b32"), ("`large`", "so400m")]  # fmt: skip
    out["readme-models"] = table(
        ["`--mode`", "right moment in the first five (DiDeMo)", "mean of the four benchmarks",
         "indexing, CPU min per hour of video (OpenVINO)"],
        [[m, f"{pct(r[n]['didemo']['value'])} %", f"{100 * r[n]['mean']:.2f}",
          "" if minutes(n) is None else f"{minutes(n):.1f}"] for m, n in modes if n in r],
    )  # fmt: skip
    return out


def round8_tables() -> dict[str, str]:
    """The eighth round: each library learnt from Florence-2's descriptions of its own frames."""
    out, cols = {}, ("didemo", "charades", "youcook2", "unav")
    head = ["DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5", "UnAV-100 P@10", "mean"]
    f, fc = RESULTS / "round8_tuned.json", RESULTS / "round8_choices.json"
    if f.exists() and fc.exists():
        t, steps = json.loads(f.read_text()), json.loads(fc.read_text())["steps"]
        rows = [["the starting models, as they are", *(pct(t["start"][c]) for c in cols),
                 f"{100 * t['start']['mean']:.2f}", ""]] if "start" in t else []  # fmt: skip
        what = {1: lambda r: "1. each library learnt, 10 passes over its descriptions",
                2: lambda r: f"2. {r['passes']:g} passes instead",
                3: lambda r: f"3. half of each batch from the starting model's own collections, "
                             f"{r['passes']:g} passes"}  # fmt: skip
        rows += [[what[x["choice"]](x["recipe"]), *(pct(t[x["run"]][c]) for c in cols), f"{100 * x['mean']:.2f}",
                  "yes" if x["kept"] else "no"] for x in steps]  # fmt: skip
        out["round8-tuning"] = table(["tuning sets", *head, "kept"], rows)
    f = RESULTS / "round8_cost.json"
    if f.exists():
        c = json.loads(f.read_text())["libraries"]
        which = {"tuning": "tuning videos", "test": "test videos (described, never searched)"}
        rows = [[f"{BENCH[k.split('-')[0]]}, {x['videos']:,} "
                 f"{'held-out videos' if k == 'charades-tuning' else which[k.split('-')[1]]}", f"{x['hours']:.1f}",
                 f"{x['descriptions']:,}", f"{x['decode']:.1f}", f"{x['describe']:.1f}", f"{x['embed']:.2f}",
                 f"{x['learn']['learn-p10']:.2f}" if "learn" in x else ""] for k, x in c.items()]  # fmt: skip
        out["round8-cost"] = table(["library", "hours of video", "descriptions", "decoding again",
                                    "describing (Florence-2-base)", "embedding the descriptions",
                                    "learning, 10 passes"], rows)  # fmt: skip
    f = RESULTS / "round8_test.json"
    if not f.exists():
        return out
    r = json.loads(f.read_text())
    ref, new = r["candidates"][:2]
    names = {ref: "the starting models, as they are", new: "each library learnt, with the recipe chosen"}
    rows = [[names[n], *(pct(r[n][c]["value"]) for c in cols), f"**{100 * r[n]['mean']:.2f}**"] for n in (ref, new)]
    out["round8-test"] = table(["test libraries", *head], rows)
    pair = r["pairs"][f"{new} vs {ref}"]
    out["round8-pairs"] = table(["over the four tests", "items better", "items worse", "sign test p"],
                                [["learnt / as they are", str(pair["wins"]), str(pair["losses"]),
                                  f"{pair['p']:.2g}"]])  # fmt: skip
    u, ch, cost, met = r["unseen"], r["changes"], r["cost"], r["criteria"]
    rows = [
        ["27. on Charades-STA and YouCook2, the mean rises by at least 2 points, sign test p < 0.05",
         f"{100 * u['gain']:+.2f} points (Charades-STA {100 * u['charades']:+.1f}, YouCook2 "
         f"{100 * u['youcook2']:+.1f}); {u['wins']} items better, {u['losses']} worse, p = {u['p']:.2g}", met["27"]],
        ["28. at most 1 point lost on each test", "; ".join(f"{BENCH[k]} {100 * v:+.1f}" for k, v in ch.items()),
         met["28"]],
        ["29. describing and learning cost at most 30 CPU min per hour of video",
         f"{cost['total']:.1f} (decoding {cost['decode']:.1f}, describing {cost['describe']:.1f}, embedding the "
         f"descriptions {cost.get('embed', 0.0):.2f}, learning {cost['learn']:.2f})", met["29"]],
    ]  # fmt: skip
    out["round8-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                   [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


MAIN9 = ["base", "pe", "v3", "v4"]
LABEL9 = {"base": "SigLIP 2 B/16 alone", "pe": "PE core B/16 alone", "v3": "SigLIP 2 B/16 + whenframe's model v3",
          "v4": "PE core B/16 + whenframe's model v4"}  # fmt: skip
LEFT_OUT9 = {"charades": ("v3-no-charades", "v4-no-charades"), "youcook2": ("v3-no-youcook2", "v4-no-youcook2")}


def round9_tables() -> dict[str, str]:
    """The ninth round: Perception Encoder's PE core B/16 in place of SigLIP 2 B/16, and whenframe's model on its
    frames (v4)."""
    out, cols = {}, ("didemo", "charades", "youcook2", "unav")
    head = ["DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5", "UnAV-100 P@10", "mean"]
    f, f7 = RESULTS / "round9_tuned.json", RESULTS / "round7_tuned.json"
    if f.exists() and f7.exists():
        t = json.loads(f7.read_text()) | json.loads(f.read_text())  # v3 and its runs: the seventh round's
        rows = [[LABEL9[k], *(pct(t[k][c]) for c in cols), f"{100 * t[k]['mean']:.2f}"] for k in MAIN9 if k in t]
        rows += [[f"*{k.split('-')[0]} without {BENCH[c]}'s videos (beside)*", *(pct(t[k][x]) for x in cols),
                  f"{100 * t[k]['mean']:.2f}"] for c, runs in LEFT_OUT9.items() for k in runs if k in t]  # fmt: skip
        out["round9-tuning"] = table(["tuning sets", *head], rows)
    f, fc = RESULTS / "round9_test.json", RESULTS / "round9_cost.json"
    if not (f.exists() and fc.exists()):
        return out
    r, cost = json.loads(f.read_text()), json.loads(fc.read_text())
    encoder = {"base": "siglip2-base", "v3": "siglip2-base", "pe": "pe-core-b16", "v4": "pe-core-b16"}

    def minutes(n: str) -> float:
        return cost[encoder[n]]["cpu_min_per_hour"] + r[n].get("adapter_cost", {}).get("cpu_min_per_hour", 0.0)

    main = [n for n in r["ranking"] if n in MAIN9]
    out["round9-test"] = table(
        ["test sets, ranked by the mean", *head, "indexing, CPU min per hour"],
        [[LABEL9[n], *(pct(r[n][c]["value"]) for c in cols), f"**{100 * r[n]['mean']:.2f}**", f"{minutes(n):.1f}"]
         for n in main],
    )  # fmt: skip
    pairs = [(a, b) for i, a in enumerate(main) for b in main[i + 1 :]]

    def said(a: str, b: str) -> bool:
        return r["pairs"][f"{a} vs {b}"]["p"] < 0.05 and r[a]["mean"] > r[b]["mean"]

    out["round9-pairs"] = table(
        ["the first finds more than the second?", "items better", "items worse", "sign test p", "said"],
        [[f"{LABEL9[a]} / {LABEL9[b]}", str(r["pairs"][f"{a} vs {b}"]["wins"]),
          str(r["pairs"][f"{a} vs {b}"]["losses"]), f"{r['pairs'][f'{a} vs {b}']['p']:.2g}",
          "yes" if said(a, b) else "no"] for a, b in pairs],
    )  # fmt: skip
    left = [(c, old, new) for c, (old, new) in LEFT_OUT9.items() if old in r and new in r]
    if left:
        out["round9-left-out"] = table(
            ["the test of a collection left out of training", "SigLIP 2 B/16 alone", "v3 trained without it",
             "PE core B/16 alone", "v4 trained without it", "v4 against v3, both without it"],
            [[f"{BENCH[c]}, {('R1 at IoU >= 0.5' if c == 'charades' else 'R@5, speech + spans')}",
              pct(r["base"][c]["value"]), pct(r[old][c]["value"]), pct(r["pe"][c]["value"]),
              pct(r[new][c]["value"]), f"{100 * (r[new][c]['value'] - r[old][c]['value']):+.1f}"]
             for c, old, new in left],
        )  # fmt: skip
    g, met = r["v4_vs_v3"], r["criteria"]
    rows = [
        ["30. v4's mean is at least 1 point higher than v3's, sign test p < 0.05",
         f"{100 * g['gain']:+.2f} points; {g['wins']} items better, {g['losses']} worse, p = {g['p']:.2g}",
         met["30"]],
        ["31. v4 loses at most 1 point on each test against v3",
         "; ".join(f"{BENCH[k]} {100 * v:+.1f}" for k, v in g["changes"].items()), met["31"]],
        ["32. indexing with PE core B/16 costs at most 1.2 times what SigLIP 2 B/16 costs",
         f"{r['cost_ratio']:.2f} times ({cost['pe-core-b16']['cpu_min_per_hour']:.1f} against "
         f"{cost['siglip2-base']['cpu_min_per_hour']:.1f} CPU min per hour, decoding included)", met["32"]],
    ]  # fmt: skip
    out["round9-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                   [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


REFERENCE10 = {"v3": "v3, the reference of 1 and 2", "start": "the eighth round's starting models, the reference of 3"}


def round10_tables() -> dict[str, str]:
    """The tenth round: guidance, v3 fused with v4, and libraries learnt from the descriptions that find their own
    block, screened on the tuning sets."""
    f = RESULTS / "round10_screen.json"
    if not f.exists():
        return {}
    t, cols = json.loads(f.read_text()), ("didemo", "charades", "youcook2", "unav")

    def what(k: str) -> str:
        if k in REFERENCE10:
            return REFERENCE10[k]
        if k.startswith("guide-"):
            return f"1. guidance, w {k.removeprefix('guide-')}"
        if k.startswith("fuse-"):
            return f"2. fusion, PE's weight {k.removeprefix('fuse-')}"
        return "3. round trip: each library learnt from the descriptions kept"

    rows = []
    for k, x in t.items():
        ref = None if k in REFERENCE10 else t["start" if k == "round-trip" else "v3"]
        rows.append([what(k), *(pct(x[c]) for c in cols), f"{100 * x['mean']:.2f}",
                     "" if ref is None else f"{100 * (x['mean'] - ref['mean']):+.2f}"])  # fmt: skip
    head = ["tuning sets", "DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5", "UnAV-100 P@10", "mean",
            "against its reference"]  # fmt: skip
    return {"round10-tuning": table(head, rows)}


def published_tables() -> dict[str, str]:
    """whenframe beside published systems trained to search a whole library of DiDeMo's test videos (added after
    the test runs: results/published_didemo.json, as their authors report them)."""
    f, f7, th = RESULTS / "published_didemo.json", RESULTS / "round7_test.json", RESULTS / "didemo_test_thumbnail.json"
    if not (f.exists() and f7.exists()):
        return {}
    pub, r = json.loads(f.read_text()), json.loads(f7.read_text())
    ks = ("R@1", "R@5", "R@10", "R@100")
    rows = [[f"**whenframe, {m}**", *(pct(r[n]["didemo"]["full"]["video"][k]) for k in ks), seen, "CPU"]
            for m, n, seen in (("with its own model (the default)", "v3", "yes"),
                               ("`--no-temporal` (SigLIP 2 B/16)", "base", "no"))]  # fmt: skip
    if th.exists():
        t = json.loads(th.read_text())
        rows.append(["one image per video, its middle (SigLIP 2 B/16)", pct(t["middle"]["R@1"]),
                     pct(t["middle"]["R@5"]), pct(t["middle"]["R@10"]), "", "no", "CPU"])  # fmt: skip
    rows += [[name, *(f"{x[k]:.1f}" for k in ks), "yes", "GPU"]
             for name, x in sorted(pub["video_retrieval"].items(), key=lambda kv: -kv[1]["R@5"])]  # fmt: skip
    out = {"didemo-published": table(["the right video among the first k (DiDeMo test)", "R@1", "R@5", "R@10",
                                      "R@100", "trained on DiDeMo", "runs on"], rows)}  # fmt: skip
    c = r["v3"]["didemo"]["full"]["corpus"]
    moments = [["**whenframe, with its own model**", *(pct(c[f"R@{k} IoU>=0.5"]) for k in (1, 10, 100)),
                *(pct(c[f"R@{k} IoU>=0.7"]) for k in (1, 10, 100))]]  # fmt: skip
    for name, x in pub["moment_retrieval_original"].items():
        if name != "about":
            moments.append([name, *(f"{x[i][f'R@{k}']:.1f}" if i in x else "" for i in ("IoU 0.5", "IoU 0.7")
                                    for k in (1, 10, 100))])  # fmt: skip
    out["didemo-published-moments"] = table(["the right moment among the first k of the whole test set",
                                             "R@1, IoU 0.5", "R@10", "R@100", "R@1, IoU 0.7", "R@10", "R@100"],
                                            moments)  # fmt: skip
    return out


LABEL11 = {"v3": "SigLIP 2 B/16 + whenframe's model v3 (`standard`, the default)",
           "fusion": "v3 and v4 together (SigLIP 2 B/16 and PE core B/16; `large`)",
           "so400m": "SigLIP 2 So400m (`so400m`; `large` until this round)"}  # fmt: skip
LEFT_OUT11 = {"charades": ("v3-no-charades", "fusion-no-charades"),
              "youcook2": ("v3-no-youcook2", "fusion-no-youcook2")}  # fmt: skip


def round11_tables() -> dict[str, str]:
    """The eleventh round. Part 1: the tenth round's fusion of v3 and v4, tested once against v3 (and So400m,
    beside); both encoders' indexing cost against SigLIP 2 B/16's, measured again the same way. Part 2: v5, taught
    by the fusion, on the tuning sets (it did not reach the test)."""
    out, cols = {}, ("didemo", "charades", "youcook2", "unav")
    head = ["DiDeMo R@5", "Charades-STA R1", "YouCook2 R@5", "UnAV-100 P@10", "mean"]
    f5, f7, ft = RESULTS / "round11_v5.json", RESULTS / "round7_tuned.json", RESULTS / "round11_tuned.json"
    if f5.exists() and f7.exists() and ft.exists():  # part 2: v5's choices, on the tuning sets only
        v5, v3 = json.loads(f5.read_text()), json.loads(f7.read_text())["v3"]
        runs = [("v3, the reference", v3), ("the fusion, v5's teacher", json.loads(ft.read_text())["fusion"])]
        runs += [(f"v5, the teacher's losses with weight {s['distil']:g}"
                  + (" (kept)" if s["run"] == v5["kept"] else ""), v5["runs"][s["run"]]["tuning"])
                 for s in v5["steps"]]  # fmt: skip
        out["round11-v5"] = table(
            ["tuning sets", *head, "against v3"],
            [[name, *(pct(x[c]) for c in cols), f"{100 * x['mean']:.2f}",
              "" if x is v3 else f"{100 * (x['mean'] - v3['mean']):+.2f}"] for name, x in runs],
        )  # fmt: skip
    f, fc, f6 = RESULTS / "round11_test.json", RESULTS / "round11_cost.json", RESULTS / "compare_cost.json"
    if not (f.exists() and fc.exists()):
        return out
    r, cost = json.loads(f.read_text()), json.loads(fc.read_text())
    one = cost["siglip2-base"]["cpu_min_per_hour"]
    times = {"v3": 1.0, "fusion": cost["siglip2-base+pe-core-b16"]["cpu_min_per_hour"] / one}
    if f6.exists():  # So400m's against B/16's, as the sixth round measured both
        c6 = json.loads(f6.read_text())
        times["so400m"] = c6["siglip2-so400m"]["cpu_min_per_hour"] / c6["siglip2-base"]["cpu_min_per_hour"]
    main = [n for n in r["ranking"] if n in LABEL11]
    out["round11-test"] = table(
        ["test sets, ranked by the mean", *head, "indexing cost, times SigLIP 2 B/16's"],
        [[LABEL11[n], *(pct(r[n][c]["value"]) for c in cols), f"**{100 * r[n]['mean']:.2f}**",
          f"{times[n]:.2f}" if n in times else ""] for n in main],
    )  # fmt: skip
    pairs = [("fusion", "v3"), ("fusion", "so400m"), ("v3", "so400m")]
    out["round11-pairs"] = table(
        ["the first finds more than the second?", "items better", "items worse", "sign test p"],
        [[f"{LABEL11[a]} / {LABEL11[b]}", str(r["pairs"][f"{a} vs {b}"]["wins"]),
          str(r["pairs"][f"{a} vs {b}"]["losses"]), f"{r['pairs'][f'{a} vs {b}']['p']:.2g}"] for a, b in pairs],
    )  # fmt: skip
    out["round11-left-out"] = table(
        ["the test of a collection left out of training", "v3 trained without it", "v3 and v4 trained without it, "
         "together", "change"],
        [[f"{BENCH[c]}, {('R1 at IoU >= 0.5' if c == 'charades' else 'R@5, speech + spans')}",
          pct(r[old][c]["value"]), pct(r[new][c]["value"]), f"{100 * (r[new][c]['value'] - r[old][c]['value']):+.1f}"]
         for c, (old, new) in LEFT_OUT11.items() if old in r and new in r],
    )  # fmt: skip
    g, met = r["fusion_vs_v3"], r["criteria"]
    both = cost["siglip2-base+pe-core-b16"]["cpu_min_per_hour"]
    rows = [
        ["33. the fusion's mean is at least 1 point higher than v3's, sign test p < 0.05",
         f"{100 * g['gain']:+.2f} points; {g['wins']} items better, {g['losses']} worse, p = {g['p']:.2g}", met["33"]],
        ["34. the fusion loses at most 1 point on each test against v3",
         "; ".join(f"{BENCH[k]} {100 * v:+.1f}" for k, v in g["changes"].items()), met["34"]],
        ["35. indexing with both encoders costs at most 2 times what SigLIP 2 B/16 alone costs",
         f"{r['cost_ratio']:.2f} times ({both:.1f} against {one:.1f} CPU min per hour, decoding included)", met["35"]],
    ]  # fmt: skip
    out["round11-criteria"] = table(["criterion (fixed before its test run)", "measured on test", "met"],
                                    [[n, m, "yes" if ok else "no"] for n, m, ok in rows])  # fmt: skip
    return out


def readme_compare() -> dict[str, str]:
    """The README's comparison: whenframe, on a CPU, beside what others do on the same test sets: published systems
    as their authors report them (results/published_*.json), the home tools' ways run here on the same videos."""
    names = ("round7_test", "published_didemo", "published_charades", "didemo_test_thumbnail", "didemo_test")
    files = [RESULTS / f"{n}.json" for n in names]
    if not all(f.exists() for f in files):
        return {}
    r, pub, zero, th, first = (json.loads(f.read_text()) for f in files)
    best, top = max(pub["video_retrieval"].items(), key=lambda kv: kv[1]["R@5"])
    # the three best published without training on Charades-STA, the lowest first
    others = sorted(sorted(zero["zero_shot"], key=lambda x: -x["R1@0.5"])[:3], key=lambda x: x["R1@0.5"])
    video, moment = r["v3"]["didemo"]["full"]["video"]["R@5"], r["v3"]["didemo"]["value"]
    rows = [
        ["**Find the video** among 979 (DiDeMo, in the first 5)", f"**{pct(video)} %**",
         f"{top['R@5']:.1f} % {best}, trained on DiDeMo, on a GPU · {pct(th['middle']['R@5'])} % one image per video"],
        ["**Find the moment** among 979 videos (DiDeMo, in the first 5)", f"**{pct(moment)} %**",
         f"{pct(first['scene']['corpus'][PRIMARY])} % one keyframe per shot, as home video tools index"],
        ["**Place the moment** in a given video, never trained on it (Charades-STA, first answer)",
         f"**{pct(r['v3-no-charades']['charades']['value'])} %**",
         " · ".join(f"{x['R1@0.5']:.1f} % {x['short']}" for x in others)],
    ]  # fmt: skip
    return {"readme-compare": table(["", "whenframe, on a CPU", "others"], rows)}


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
    tables.update(round5_tables())
    tables.update(round6_tables())
    tables.update(round7_tables())  # after the sixth round's: replaces the README's tables if v3 is the default
    tables.update(round8_tables())
    tables.update(round9_tables())
    tables.update(round10_tables())
    tables.update(round11_tables())
    tables.update(published_tables())
    tables.update(readme_compare())
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
