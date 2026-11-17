"""docs/figures/benchmark.png: whenframe against what exists, in a light and a dark version.

    python scripts/benchmark_figure.py

Every whenframe number is read from results/*.json. Published numbers are as
their authors report them, on the same benchmark and metric (Charades-STA test,
R1 at IoU >= 0.5, no training on Charades-STA's annotations); the home-tool
approaches (one image per video, one keyframe per shot, plain frame search) are
run here, on the same videos and descriptions as whenframe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from figures import PX, THEMES, axes, figure, rounded_bar, save  # noqa: E402

RESULTS = ROOT / "results"
PRIMARY = "R@5 IoU>=0.5"

# Charades-STA, zero-shot: (name, what it runs, R1 at IoU 0.5 in %), as published
PUBLISHED = [
    ("Your VLM Already Knows When (2026)", "a 7-billion-parameter vision-language model", 56.8),  # arXiv 2608.08315
    ("TFVTG (ECCV 2024)", "a language model and vision-language models", 49.97),  # arXiv 2408.16219
    ("VTG-GPT (2024)", "language models and an image captioner", 43.68),  # arXiv 2403.02076, table 1
    ("PSVL (ICCV 2021)", "trained on unannotated videos", 31.29),  # arXiv 2110.00428
]


def load(name: str) -> dict | None:
    f = RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


def panels() -> list[dict]:
    """Each panel: a title, a subtitle, and rows (label, note, value in %, is whenframe)."""
    out = []
    ch = load("charades_test.json")
    if ch:
        rows = [(name, note, value, False) for name, note, value in PUBLISHED]
        rows += [("whenframe, adaptive spans", "SigLIP 2 on the CPU, parameters from DiDeMo",
                  100 * ch["whenframe_bank"]["single"]["R1 IoU>=0.5"], True),
                 ("whenframe, default", "SigLIP 2 on the CPU, parameters from DiDeMo",
                  100 * ch["frame_window_bank"]["single"]["R1 IoU>=0.5"], True)]  # fmt: skip
        if "middle_window" in ch:
            rows.append(("10 seconds in the middle", "no model: where moments usually are",
                         100 * ch["middle_window"]["single"]["R1 IoU>=0.5"], False))  # fmt: skip
        out.append({"title": "Placing a described moment in a given video",
                    "subtitle": f"Charades-STA test, {ch['queries']:,} sentences: right moment first (R1, IoU ≥ 0.5),"
                                " no training on Charades-STA",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    th = load("didemo_test_thumbnail.json")
    if th:
        rows = [("whenframe", "every second of every video", 100 * th["default_search"]["R@5"], True),
                ("one image per video, its middle", "as photo libraries index videos", 100 * th["middle"]["R@5"],
                 False),
                ("one image per video, its first", "", 100 * th["first"]["R@5"], False)]  # fmt: skip
        out.append({"title": "Finding the video among a whole library",
                    "subtitle": f"DiDeMo test, {th['videos']:,} personal videos, {th['queries']:,} descriptions: "
                                "right video among the first five",
                    "rows": rows, "unit": "%"})  # fmt: skip
    d = load("didemo_test.json")
    if d:
        rows = []
        big = load("didemo_test_so400m.json")
        if big:
            rows.append(("whenframe, SigLIP 2 So400m", "the larger model, with OpenVINO",
                         100 * big["siglip2-so400m"]["corpus"][PRIMARY], True))  # fmt: skip
        rows += [("whenframe, default", "SigLIP 2 base", 100 * d[d["default"]]["corpus"][PRIMARY], True),
                 ("frames searched as they are", "the same model, 10-second windows, no query bank",
                  100 * d["frame_window"]["corpus"][PRIMARY], False),
                 ("one keyframe per shot", "as home video tools cut videos", 100 * d["scene"]["corpus"][PRIMARY],
                  False)]  # fmt: skip
        out.append({"title": "Finding the moment among a whole library",
                    "subtitle": f"DiDeMo test, {d['cost']['videos']:,} videos at once: right moment among the first"
                                " five (R@5, IoU ≥ 0.5)",
                    "rows": rows, "unit": "%"})  # fmt: skip
    yc, yc4 = load("youcook2_test.json"), load("youcook2_test_round4.json")
    if yc and yc4:
        spans_speech = 100 * yc4["speech_spans"]["corpus"][PRIMARY]
        rows = [("adaptive spans + what is said", "Whisper base, on the CPU", spans_speech, True),
                ("adaptive spans", "", 100 * yc4["whenframe_bank"]["corpus"][PRIMARY], True),
                ("fixed windows + what is said", "", 100 * yc["speech_bank"]["corpus"][PRIMARY], True),
                ("fixed windows (the default)", "frames only", 100 * yc["frame_window_bank"]["corpus"][PRIMARY],
                 True)]  # fmt: skip
        out.append({"title": "Long videos, and what is said in them",
                    "subtitle": f"YouCook2, {yc['videos']} cooking videos of 5 minutes on average, at once: right step "
                                "among the first five (R@5, IoU ≥ 0.5)",
                    "rows": rows, "unit": "%"})  # fmt: skip
    un = load("unav_test.json")
    if un:
        heard = f"CLAP, {un['length']:g}-second windows"
        rows = [("frames and sound", f"SigLIP 2 + CLAP, weight {un['weight']:g}", 100 * un["both"]["p_at_k"], True),
                ("frames only", "SigLIP 2, as before", 100 * un["frames"]["p_at_k"], True),
                ("sound only", heard, 100 * un["sound"]["p_at_k"], True)]  # fmt: skip
        out.append({"title": "Hearing: laughter, applause, a dog barking, music",
                    "subtitle": f"UnAV-100 test, {un['videos']:,} videos at once, {un['labels']} kinds of events: "
                                "share of the first ten moments that are right",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    return out


def cost_points() -> list[tuple[str, float, float]]:
    """(label, CPU minutes per hour of video, R@5 on DiDeMo test) of each way of indexing."""
    r2 = load("didemo_test_round2.json")
    if not r2:
        return []
    c = r2["cost"]

    def minutes(t: dict) -> float:
        return (t["decode_cpu"] + t["encode_cpu"]) / t["hours"] / 60

    pts = [("standard", minutes(c), 100 * r2["frame_window_bank"]["corpus"][PRIMARY]),
           ("economy", minutes(c["siglip2-b32"]), 100 * r2["siglip32_bank"]["corpus"][PRIMARY]),
           ("precision", minutes(c) + minutes(c["siglip2-b32"]),
            100 * r2["ensemble32_bank"]["corpus"][PRIMARY])]  # fmt: skip
    ov = load("didemo_test_backends.json")
    if ov:
        pts.append(("standard, OpenVINO", sum(ov["openvino"]["cpu_min_per_hour"].values()),
                    100 * ov["openvino"]["corpus"][PRIMARY]))  # fmt: skip
    big = load("didemo_test_so400m.json")
    if big:
        pts.append(("So400m, OpenVINO", sum(big["cost"]["cpu_min_per_hour"].values()),
                    100 * big["siglip2-so400m"]["corpus"][PRIMARY]))  # fmt: skip
    return pts


def draw(theme: str):
    t = THEMES[theme]
    ps, pts = panels(), cost_points()
    heights = [0.62 + 0.4 * len(p["rows"]) for p in ps] + ([2.5] if pts else [])
    top = 1.2
    total = top + sum(heights) + 0.25 * len(heights) + 0.45
    fig = figure(t, 8.0, total)
    y = total - 0.2
    fig.text(0.02, y / total, "whenframe against what exists", color=t["ink"], fontsize=17 * PX, fontweight="bold",
             va="top")  # fmt: skip
    fig.text(0.02, (y - 0.32) / total, "Zero-shot, on a 4-vCPU machine without a GPU. Published numbers are as their "
             "authors report them;\nthe other references are run here, on the same videos and descriptions.",
             color=t["ink2"], fontsize=12 * PX, va="top", linespacing=1.5)  # fmt: skip
    key = 10 * PX / 72  # inches
    for x, color, name in ((0.02, t["s1"], "whenframe"), (0.16, t["muted"], "other methods and references")):
        mid = (y - 0.88) / total
        fig.patches.append(plt_rect(fig, x, mid - key / 2 / total, key / 8.0, key / total, color))
        fig.text(x + key / 8.0 + 0.008, mid, name, color=t["ink2"], fontsize=11.5 * PX, va="center")
    y -= top
    for p, h in zip(ps, heights, strict=False):
        y -= 0.25
        fig.text(0.02, y / total, p["title"], color=t["ink"], fontsize=13.5 * PX, fontweight="bold", va="top")
        fig.text(0.02, (y - 0.24) / total, p["subtitle"], color=t["ink2"], fontsize=11.5 * PX, va="top")
        n = len(p["rows"])
        body = 0.4 * n
        ax = axes(fig, [0.43, (y - 0.56 - body) / total, 0.47, body / total], t)
        ax.spines["bottom"].set_visible(False)
        hi = max(r[2] for r in p["rows"])
        ax.set_xlim(0, hi * 1.16)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for i, (label, note, value, ours) in enumerate(p["rows"]):
            rounded_bar(ax, i, value, 16, t["s1"] if ours else t["muted"])
            ax.text(value + 0.012 * ax.get_xlim()[1], i, f"{value:.1f} %", va="center", color=t["ink"],
                    fontsize=12 * PX, fontweight="bold" if ours else "normal")  # fmt: skip
            yy = (y - 0.56 - (i + 0.5) * 0.4) / total
            fig.text(0.415, yy + (0.055 if note else 0) / total, label, ha="right", va="center", color=t["ink"],
                     fontsize=12 * PX)  # fmt: skip
            if note:
                fig.text(0.415, yy - 0.085 / total, note, ha="right", va="center", color=t["muted"],
                         fontsize=10.5 * PX)  # fmt: skip
        y -= h
    if pts:
        y -= 0.25
        fig.text(0.02, y / total, "What indexing costs, and what it finds", color=t["ink"], fontsize=13.5 * PX,
                 fontweight="bold", va="top")  # fmt: skip
        fig.text(0.02, (y - 0.24) / total, "whenframe's ways of indexing: CPU minutes per hour of video, decoding "
                 "included, against R@5 on DiDeMo test", color=t["ink2"], fontsize=11.5 * PX, va="top")  # fmt: skip
        ax = axes(fig, [0.09, (y - 2.35) / total, 0.84, 1.6 / total], t)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.scatter(xs, ys, s=(10 * PX) ** 2 * 4, color=t["s1"], edgecolors=t["surface"], linewidths=2 * PX, zorder=3)
        for label, x, v in pts:
            ax.annotate(f"{label}  {v:.1f} %", (x, v), xytext=(8, -3), textcoords="offset points", color=t["ink2"],
                        fontsize=11 * PX)  # fmt: skip
        ax.set_xlim(0, max(xs) * 1.3)
        lo, hi = min(ys), max(ys)
        ax.set_ylim(lo - 1.2, hi + 1.2)
        ax.grid(axis="y", color=t["grid"], lw=PX)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} %")
        ax.set_xlabel("CPU minutes to index an hour of video", color=t["muted"], fontsize=11 * PX)
    return fig


def plt_rect(fig, x: float, y: float, w: float, h: float, color: str):
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.002", transform=fig.transFigure,
                          color=color, lw=0)  # fmt: skip


def main() -> None:
    for theme in THEMES:
        save(draw(theme), "benchmark", theme)


if __name__ == "__main__":
    main()
