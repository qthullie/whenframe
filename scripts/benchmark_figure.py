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
EXAMPLES = ("dog barking", "baby laughter")  # shown under the sound panel's first bar

# Charades-STA, zero-shot: (name, what it runs, R1 at IoU 0.5 in %), as published (results/published_charades.json)
PUBLISHED = [
    (x["name"], x["runs"], x["R1@0.5"])
    for x in json.loads((RESULTS / "published_charades.json").read_text())["zero_shot"]
]


def load(name: str) -> dict | None:
    f = RESULTS / name
    return json.loads(f.read_text()) if f.exists() else None


def panels() -> list[dict]:
    """Each panel: a title, a subtitle, and rows (label, note, value in %, is whenframe)."""
    out = []
    seven = load("round7_test.json")  # the seventh round: the encoders and whenframe's models, measured the same way

    def ours(key: str) -> list[tuple]:
        """The default (with whenframe's model v3), --no-temporal and --mode large on one benchmark; where v3
        learnt from the benchmark's training videos, also its recipe trained without them (seventh round)."""
        if not seven:
            return []
        default = {"didemo": "its own model, trained on DiDeMo's training videos",
                   "youcook2": "its own model, trained on YouCook2's training videos",
                   "unav": "its own model, never trained on UnAV-100"}  # fmt: skip
        rows = [
            ("whenframe, --no-temporal", "SigLIP 2 B/16 alone", seven["base"][key]["value"]),
            ("whenframe, --mode large", "SigLIP 2 So400m, 3.4 times the CPU", seven["so400m"][key]["value"]),
        ]
        if key in default:  # on Charades-STA, beside zero-shot methods, the model trained without its videos only
            rows.insert(0, ("whenframe, default", default[key], seven["v3"][key]["value"]))
        left = {"charades": ("v3-no-charades", "whenframe, its model without Charades-STA"),
                "youcook2": ("v3-no-youcook2", "its model without YouCook2")}.get(key)  # fmt: skip
        if left and left[0] in seven:
            rows.append((left[1], "the default's recipe, never trained on these videos", seven[left[0]][key]["value"]))
        return [(a, b, 100 * c, True) for a, b, c in rows]

    ch = load("charades_test.json")
    if ch:
        rows = [(name, note, value, False) for name, note, value in PUBLISHED] + ours("charades")
        rows.append(("whenframe, adaptive spans", "SigLIP 2 B/16 alone, parameters from DiDeMo",
                     100 * ch["whenframe_bank"]["single"]["R1 IoU>=0.5"], True))  # fmt: skip
        if "middle_window" in ch:
            rows.append(("10 seconds in the middle", "no model: where moments usually are",
                         100 * ch["middle_window"]["single"]["R1 IoU>=0.5"], False))  # fmt: skip
        out.append({"title": "Placing a described moment in a given video",
                    "subtitle": f"Charades-STA test, {ch['queries']:,} sentences: right moment first (R1, IoU ≥ 0.5),"
                                " no training on Charades-STA",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    th, pub = load("didemo_test_thumbnail.json"), load("published_didemo.json")
    if th:
        rows = [("whenframe", "every second of every video", 100 * th["default_search"]["R@5"], True)]
        if seven:  # the seventh round's test: with whenframe's own model (the default) and without it
            video = {n: 100 * seven[n]["didemo"]["full"]["video"]["R@5"] for n in ("v3", "base")}
            rows = [("whenframe, default", "its own model, trained on DiDeMo's training videos", video["v3"], True),
                    ("whenframe, --no-temporal", "SigLIP 2 B/16 alone", video["base"], True)]  # fmt: skip
        rows += [("one image per video, its middle", "as photo libraries index videos", 100 * th["middle"]["R@5"],
                  False),
                 ("one image per video, its first", "", 100 * th["first"]["R@5"], False)]  # fmt: skip
        subtitle = f"DiDeMo test, {th['videos']:,} personal videos"
        if pub:  # the three best systems published for this search, trained on DiDeMo, as their authors report them
            best = sorted(pub["video_retrieval"].items(), key=lambda kv: -kv[1]["R@5"])[:3]
            rows += [(name, "trained on DiDeMo, ResNet and SlowFast, on a GPU", x["R@5"], False) for name, x in best]
            subtitle += f" ({pub['videos']:,} for the published systems)"
        out.append({"title": "Finding the video among a whole library",
                    "subtitle": f"{subtitle}, {th['queries']:,} descriptions: right video among the first five",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    d = load("didemo_test.json")
    if d:
        rows = ours("didemo") or [
            ("whenframe, default", "SigLIP 2 base", 100 * d[d["default"]]["corpus"][PRIMARY], True)
        ]
        rows += [("frames searched as they are", "the same model, 10-second windows, no query bank",
                  100 * d["frame_window"]["corpus"][PRIMARY], False),
                 ("one keyframe per shot", "as home video tools cut videos", 100 * d["scene"]["corpus"][PRIMARY],
                  False)]  # fmt: skip
        out.append({"title": "Finding the moment among a whole library",
                    "subtitle": f"DiDeMo test, {d['cost']['videos']:,} videos at once: right moment among the first"
                                " five (R@5, IoU ≥ 0.5)",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    yc, yc4 = load("youcook2_test.json"), load("youcook2_test_round4.json")
    if yc and yc4:
        rows = [(f"{n}, speech + spans", note, v, w) for n, note, v, w in ours("youcook2")] or [
            ("adaptive spans + what is said", "Whisper base, on the CPU", 100 * yc4["speech_spans"]["corpus"][PRIMARY],
             True)]  # fmt: skip
        rows += [("adaptive spans", "frames only", 100 * yc4["whenframe_bank"]["corpus"][PRIMARY], True),
                 ("fixed windows + what is said", "", 100 * yc["speech_bank"]["corpus"][PRIMARY], True),
                 ("fixed windows", "frames only", 100 * yc["frame_window_bank"]["corpus"][PRIMARY], True)]  # fmt: skip
        out.append({"title": "Long videos, and what is said in them",
                    "subtitle": f"YouCook2, {yc['videos']} cooking videos of 5 minutes on average, at once: right step "
                                "among the first five (R@5, IoU ≥ 0.5)",
                    "rows": sorted(rows, key=lambda r: -r[2]), "unit": "%"})  # fmt: skip
    un = load("unav_test.json")
    if un:
        per = {k: (un["frames"]["labels"][k]["p_at_k"], un["both"]["labels"][k]["p_at_k"]) for k in EXAMPLES
               if k in un["frames"]["labels"]}  # fmt: skip
        examples = ", ".join(f"{k} {100 * a:.0f} → {100 * b:.0f} %" for k, (a, b) in per.items())
        rows = [(f"{n}, frames + sound", note, v, w) for n, note, v, w in ours("unav")] or [
            ("frames and sound (--sounds)", examples, 100 * un["both"]["p_at_k"], True)]  # fmt: skip
        clap = f"CLAP, {un['length']:g}-second windows"
        rows += [("frames only", "SigLIP 2 B/16 alone, without the sound", 100 * un["frames"]["p_at_k"], True),
                 ("sound only", clap, 100 * un["sound"]["p_at_k"], True)]  # fmt: skip
        rows = sorted(rows, key=lambda r: -r[2])
        out.append({"title": "Hearing: laughter, a dog barking, a siren",
                    "subtitle": f"UnAV-100 test, {un['videos']} videos at once, {un['labels']} kinds of events: "
                                "share of the first ten moments that are right",
                    "rows": rows, "unit": "%"})  # fmt: skip
    return out


def cost_points() -> list[tuple[str, float, float]]:
    """(label, CPU minutes per hour of video, R@5 on DiDeMo test) of each way of indexing."""
    pts = []
    seven, cost = load("round7_test.json"), load("compare_cost.json")
    if seven and cost:  # the seventh round, OpenVINO, every encoder's cost measured the same way
        adapter = seven["v3"]["adapter_cost"]["cpu_min_per_hour"]
        for label, name, enc, extra in (("standard (with its own model)", "v3", "siglip2-base", adapter),
                                        ("--no-temporal", "base", "siglip2-base", 0.0),
                                        ("economy", "b32", "siglip2-b32", 0.0),
                                        ("large", "so400m", "siglip2-so400m", 0.0)):  # fmt: skip
            pts.append((label, cost[enc]["cpu_min_per_hour"] + extra, 100 * seven[name]["didemo"]["value"]))
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
    intro = (
        "On a 4-vCPU machine without a GPU. whenframe's own model learnt from the training videos of DiDeMo, "
        "MSR-VTT,\nCharades-STA and YouCook2. Published numbers are as their authors report them; the other "
        "references are run here."
    )
    fig.text(0.02, (y - 0.32) / total, intro, color=t["ink2"], fontsize=12 * PX, va="top", linespacing=1.5)
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
