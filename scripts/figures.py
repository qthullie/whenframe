"""Figures for the README and docs/RESULTS.md, each in a light and a dark version.

    python scripts/figures.py            # from results/didemo_test.json and the test cache

The example figure reads the cached embeddings (results/cache, see `make didemo-extract`)
and shows no video frame: only the scores and the annotated moment.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import PREDICT, SHORTLIST, frame_scores, iou  # noqa: E402

from whenframe.encoder import MODELS  # noqa: E402
from whenframe.moments import on_grid  # noqa: E402

OUT = ROOT / "docs" / "figures"
DPI = 200
PX = 144 / DPI  # points per pixel of an image shown at half its size, as on a 2x screen
PRIMARY = "R@5 IoU>=0.5"
LABEL = {
    "scene": "shots (scene detection)",
    "frame_window": "fixed windows",
    "whenframe": "adaptive spans",
    "whenframe_bank": "adaptive spans + query bank",
    "frame_window_bank": "fixed windows + query bank",
    "whenframe_dedup": "default, repeats skipped",
}
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "s1": "#3987e5", "s2": "#d95926"},
}  # fmt: skip


def figure(t: dict, width: float, height: float):
    fig = plt.figure(figsize=(width, height), dpi=DPI, facecolor=t["surface"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12 * PX, "text.color": t["ink"]})
    return fig


def axes(fig, rect, t: dict):
    ax = fig.add_axes(rect, facecolor=t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(PX)
    ax.tick_params(colors=t["muted"], labelsize=11.5 * PX, length=0, pad=6 * PX)
    return ax


def titles(fig, t: dict, title: str, subtitle: str, top: float) -> None:
    fig.text(0.02, top, title, color=t["ink"], fontsize=15 * PX, fontweight="bold", va="top")
    fig.text(0.02, top - 0.26 / fig.get_figheight(), subtitle, color=t["ink2"], fontsize=12 * PX, va="top")


def rounded_bar(ax, y: float, length: float, thickness_px: float, color: str) -> None:
    """A horizontal bar with a 4 px rounded end and a square end on the baseline."""
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    per_px_x = (x1 - x0) / box.width * (DPI / 100)  # data units per displayed pixel
    per_px_y = abs(y1 - y0) / box.height * (DPI / 100)
    h = thickness_px * per_px_y
    r = min(4 * per_px_x, length / 2)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), length, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=per_px_y / per_px_x, color=color, lw=0))  # fmt: skip
    ax.add_patch(Rectangle((0, y - h / 2), max(length - r, 0), h, color=color, lw=0))


def segment(ax, y: float, a: float, b: float, thickness_px: float, color: str) -> None:
    """A thin bar from a to b with fully rounded ends."""
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    per_px_x = (x1 - x0) / box.width * (DPI / 100)
    per_px_y = abs(y1 - y0) / box.height * (DPI / 100)
    h = thickness_px * per_px_y
    r = min(thickness_px / 2 * per_px_x, (b - a) / 2)
    ax.add_patch(FancyBboxPatch((a, y - h / 2), b - a, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=per_px_y / per_px_x, color=color, lw=0))  # fmt: skip


def results_bars(r: dict, theme: str) -> plt.Figure:
    t = THEMES[theme]
    highlight = r["default"]
    methods = r["methods"]
    vals = [100 * r[m]["corpus"][PRIMARY] for m in methods]
    height = 1.25 + 0.44 * len(methods)
    fig = figure(t, 8.0, height)
    titles(fig, t, "Right moment among the first five results",
           f"DiDeMo test: {r['cost']['videos']:,} personal videos searched at once, {r['queries']:,} descriptions "
           "(corpus R@5, IoU ≥ 0.5)", 1 - 0.18 / height)  # fmt: skip
    ax = axes(fig, [0.33, 0.06, 0.6, 1 - 1.05 / height], t)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks(range(len(methods)), [LABEL[m] for m in methods], color=t["ink2"], fontsize=12.5 * PX)
    for i, (m, v) in enumerate(zip(methods, vals, strict=True)):
        rounded_bar(ax, i, v, 20, t["s1"] if m == highlight else t["muted"])
        weight = "bold" if m == highlight else "normal"
        ax.text(v + 0.012 * ax.get_xlim()[1], i, f"{v:.1f} %", va="center", color=t["ink"], fontsize=12.5 * PX,
                fontweight=weight)  # fmt: skip
    return fig


def cost_curve(r: dict, theme: str) -> plt.Figure:
    t = THEMES[theme]
    fig = figure(t, 8.0, 4.2)
    titles(fig, t, "Encoding fewer frames: skipping repeats, or lowering the frame rate",
           "Corpus R@5 (IoU ≥ 0.5) on DiDeMo test against the share of frames encoded (1 per second = 100 %)",
           1 - 0.18 / 4.2)  # fmt: skip
    ax = axes(fig, [0.08, 0.13, 0.78, 0.62], t)
    series = [("skip repeated frames", r["curve"]["repeats"], t["s1"]),
              ("lower frame rate", r["curve"]["rate"], t["s2"])]  # fmt: skip
    lo = min(100 * p["score"] for _, pts, _ in series for p in pts)
    hi = max(100 * p["score"] for _, pts, _ in series for p in pts)
    for name, pts, color in series:
        x = [100 * p["kept"] for p in pts]
        y = [100 * p["score"] for p in pts]
        ax.plot(x, y, color=color, lw=2 * PX, solid_capstyle="round", solid_joinstyle="round", marker="o",
                ms=10 * PX, mfc=color, mec=t["surface"], mew=2 * PX, label=name, zorder=3)  # fmt: skip
    default = next(p for p in r["curve"]["repeats"] if p["threshold"] == 4.0)
    ax.annotate("default threshold", (100 * default["kept"], 100 * default["score"]), xytext=(0, 10),
                textcoords="offset points", ha="center", color=t["ink2"], fontsize=11.5 * PX)  # fmt: skip
    ax.set_xlim(0, 104)
    pad = max(0.6, (hi - lo) * 0.25)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", color=t["grid"], lw=PX)
    ax.set_axisbelow(True)
    ax.set_xlabel("frames encoded (%)", color=t["muted"], fontsize=11.5 * PX)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} %")
    ax.legend(loc="lower right", frameon=False, fontsize=11.5 * PX, labelcolor=t["ink2"], handlelength=1.6)
    return fig


def example(theme: str, split: str, annotations: str | None) -> plt.Figure | None:
    """One description found first in the whole library, one not found in the first 10, with their curves."""
    cache = ROOT / "results" / "cache"
    if not (cache / f"didemo_{split}.npz").exists():
        return None
    import didemo

    t = THEMES[theme]
    frozen, queries, _ = didemo.load(split, annotations)
    text = np.load(cache / f"didemo_{split}_text.npy")
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    method = tuned["default"]
    p = tuned[method]["params"]
    frozen.bank = np.load(cache / "didemo_bank_text.npy")
    spec = MODELS["siglip2-base"]
    found, missed = None, None
    for qi, q in enumerate(queries):
        if len(q["text"].split()) < 8:
            continue
        col = frame_scores(frozen, text[qi : qi + 1], method, p)[:, 0]
        raw = frozen.emb @ text[qi]
        best = np.maximum.reduceat(col, frozen.starts)
        ranked = []
        for v in np.argsort(-best)[:SHORTLIST]:
            ranked += [(int(v), *x) for x in PREDICT[method](frozen, col, int(v), p)]
        ranked.sort(key=lambda x: -x[3])
        top = ranked[0]
        if found is None and top[0] == q["video"] and iou(top[1:3], q["moment"]) >= 0.7:
            found = (q, raw, top)
        if missed is None and q["video"] not in [x[0] for x in ranked[:10]]:
            own = PREDICT[method](frozen, col, q["video"], p)[0]
            missed = (q, raw, (q["video"], *own))
        if found and missed:
            break
    height = 6.4
    fig = figure(t, 8.0, height)
    titles(fig, t, "Two descriptions from DiDeMo test, and how the frames scored",
           "SigLIP 2's probability that the frame at each second matches the description",
           1 - 0.18 / height)  # fmt: skip
    panels = [(found, "found first among all videos"), (missed, "right video not in the first 10 results")]
    for k, (case, verdict) in enumerate(panels):
        q, col, (v, a, b, _) = case
        s, c = frozen.starts[v], frozen.counts[v]
        duration = float(frozen.durations[v])
        grid, values = on_grid(frozen.times[s : s + c].astype(np.float64), col[s : s + c].astype(np.float64),
                               duration, 0.5)  # fmt: skip
        prob = 100 / (1 + np.exp(-(spec.logit_scale * values + spec.logit_bias)))
        top = 0.80 - 0.44 * k
        quote = textwrap.shorten(q["text"], 95, placeholder="...")
        fig.text(0.02, top + 0.035, f"\u201c{quote}\u201d", color=t["ink"], fontsize=12.5 * PX)
        fig.text(0.02, top + 0.005, verdict, color=t["ink2"], fontsize=11.5 * PX)
        strip = axes(fig, [0.15, top - 0.075, 0.81, 0.06], t)
        strip.spines["bottom"].set_visible(False)
        strip.set_xlim(0, duration)
        strip.set_ylim(-0.5, 1.5)
        strip.set_xticks([])
        strip.set_yticks([1, 0], ["annotators", "whenframe"], color=t["ink2"], fontsize=11.5 * PX)
        segment(strip, 1, *q["moment"], 8, t["muted"])
        segment(strip, 0, a, b, 8, t["s1"])
        ax = axes(fig, [0.15, top - 0.33, 0.81, 0.22], t)
        ax.step(grid, prob, where="post", color=t["s1"], lw=2 * PX, zorder=3)
        ax.set_xlim(0, duration)
        ax.set_ylim(0, max(5.0, prob.max() * 1.15))
        ax.grid(axis="y", color=t["grid"], lw=PX)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f} %")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f} s")
    return fig


def save(fig: plt.Figure, name: str, theme: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"{name}.png" if theme == "light" else f"{name}-dark.png")
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--annotations", help="the split's annotations (default data/didemo/didemo_SPLIT.json)")
    args = ap.parse_args()
    r = json.loads((ROOT / "results" / f"didemo_{args.split}.json").read_text())
    for theme in THEMES:
        save(results_bars(r, theme), "results", theme)
        save(cost_curve(r, theme), "cost", theme)
        fig = example(theme, args.split, args.annotations)
        if fig is not None:
            save(fig, "example", theme)


if __name__ == "__main__":
    main()
