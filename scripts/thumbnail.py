"""A reference added after the test run: one image per video, as photo libraries index videos.

Each video is represented by a single frame, its first or its middle one, and
each description ranks the videos by that frame's SigLIP 2 score, as a library
that searches a video through its thumbnail does (Immich, for one). It finds
videos, not moments, so it is compared with the default search on the right
video among the first k, the moments of a video counting once.

    python scripts/thumbnail.py --split test --annotations ...

Writes results/didemo_{split}_thumbnail.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402

from whenframe.bank import offsets  # noqa: E402

KS = (1, 5, 10)


def main() -> None:
    args = didemo.main_args(["evaluate", *sys.argv[1:]])
    frozen, queries, _ = didemo.load(args.split, args.annotations)
    text = didemo.text_embeddings(queries, args.split, args.model_dir)
    bank = didemo.bank(args)
    truth = np.array([q["video"] for q in queries])
    out = {"split": args.split, "videos": len(frozen.names), "queries": len(queries)}
    for which in ("first", "middle"):
        rows = frozen.starts if which == "first" else frozen.starts + frozen.counts // 2
        emb = frozen.emb[rows]
        for bank_on in (False, True):
            scores = emb @ text.T - (offsets(emb, bank, 50.0)[:, None] if bank_on else 0.0)
            rank = (scores > scores[truth, np.arange(len(truth))][None, :]).sum(axis=0)
            name = which + ("_bank" if bank_on else "")
            out[name] = {f"R@{k}": float(np.mean(rank < k)) for k in KS}
            print(name, {k: round(100 * v, 1) for k, v in out[name].items()})
    res = json.loads((ROOT / "results" / f"didemo_{args.split}.json").read_text())
    default = res[res["default"]]["video"]
    out["default_search"] = {f"R@{k}": default[f"R@{k}"] for k in KS}
    path = ROOT / "results" / f"didemo_{args.split}_thumbnail.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"default search {out['default_search']}; wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
