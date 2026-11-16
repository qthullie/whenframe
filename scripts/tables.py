"""Write docs/RESULTS.md tables from results/*.json, so no number is copied by hand.

Tables live between <!-- table NAME --> and <!-- /table --> markers in
docs/RESULTS.md; everything outside the markers is left alone.
`--check` exits with an error when the file is out of date (used in CI).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOC = ROOT / "docs" / "RESULTS.md"
LABEL = {
    "scene": "shots (scene detection)",
    "frame_window": "fixed windows",
    "whenframe": "adaptive spans",
    "whenframe_bank": "adaptive spans + query bank",
    "frame_window_bank": "fixed windows + query bank",
    "whenframe_dedup": "default search, repeats skipped",
}
PRIMARY = "R@5 IoU>=0.5"


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
    return {
        "didemo-criteria": table(["criterion (fixed in advance)", "measured on test", "met"], criteria(r)),
        "didemo-corpus": corpus,
        "didemo-single": single,
        "didemo-cost": cost,
        "didemo-search": search,
        "didemo-curve": table(["frames encoded", "share", PRIMARY], rows),
    }


def main() -> int:
    tables = {}
    f = RESULTS / "didemo_test.json"
    if f.exists():
        tables.update(didemo_tables(json.loads(f.read_text())))
    text = DOC.read_text()

    def fill(m):
        name = m.group(1)
        return f"<!-- table {name} -->\n{tables[name]}\n<!-- /table -->" if name in tables else m.group(0)

    new = re.sub(r"<!-- table ([\w-]+) -->.*?<!-- /table -->", fill, text, flags=re.S)
    if "--check" in sys.argv:
        if new != text:
            print("docs/RESULTS.md is out of date: run python scripts/tables.py")
            return 1
        return 0
    DOC.write_text(new)
    print(f"updated {DOC.relative_to(ROOT)} ({len(tables)} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
