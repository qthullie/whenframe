"""The same frames embedded by ONNX Runtime (float32) and by OpenVINO (bfloat16 on this CPU).

How close the two sets of embeddings are, what each costs to compute, and what the
default search finds with each, description by description:

    python scripts/didemo.py extract --split train --backend openvino ...   # then:
    python scripts/backends.py --split train
    python scripts/backends.py --split test     # the third round: run once, after docs/PROTOCOL.md says so

Writes results/didemo_{split}_backends.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import PRIMARY, run  # noqa: E402
from explore import sign_test  # noqa: E402

TIMING = ("decode_cpu", "decode_wall", "encode_cpu", "encode_wall")


def main() -> None:
    args = didemo.main_args(["evaluate", *sys.argv[1:]])
    frozen, queries, cost = didemo.load(args.split, args.annotations)
    frozen.bank = didemo.bank(args)
    text = didemo.text_embeddings(queries, args.split, args.model_dir)
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    method = tuned["default"]
    params = tuned[method]["params"]

    z = np.load(didemo.CACHE / f"didemo_{args.split}{didemo.suffix(didemo.MAIN, 'openvino')}.npz")
    assert z["names"].tolist() == frozen.names, "not the same videos"
    mask = z["times"] < didemo.SECONDS
    assert np.array_equal(z["times"][mask], frozen.times), "not the same frames"
    other = replace(frozen, emb=z["emb"][mask].astype(np.float32), _offsets={})

    cos = (frozen.emb * other.emb).sum(axis=1) / (
        np.linalg.norm(frozen.emb, axis=1) * np.linalg.norm(other.emb, axis=1)
    )
    # each run's own hours: older ONNX Runtime caches hold whole videos, the OpenVINO ones their first 30 seconds
    hours = {"onnxruntime": cost["hours"], "openvino": float(z["durations"].sum() / 3600)}
    result = {"split": args.split, "videos": len(frozen.names), "queries": len(queries), "hours": hours,
              "method": method, "cosine": {"mean": float(cos.mean()), "min": float(cos.min()),
                                           "p01": float(np.percentile(cos, 1))}}  # fmt: skip
    found = {}
    for name, f, timing in (("onnxruntime", frozen, cost), ("openvino", other, {k: float(z[k].sum()) for k in TIMING})):
        r = run(f, queries, text, method, params, per_query=True)
        found[name] = np.array(r.pop("found"))
        frames = int(len(z["times"])) if name == "openvino" else cost["frames"]
        result[name] = {
            **r,
            "cpu_min_per_hour": {k: timing[k] / 60 / hours[name] for k in ("decode_cpu", "encode_cpu")},
            "wall_min_per_hour": {k: timing[k] / 60 / hours[name] for k in ("decode_wall", "encode_wall")},
            "encode_ms_per_frame": 1000 * timing["encode_cpu"] / frames,
        }
        per_hour = result[name]["cpu_min_per_hour"]["encode_cpu"]
        print(f"{name:12s} R@5 {100 * r[PRIMARY[0]][PRIMARY[1]]:.1f}  encoding {per_hour:.1f} CPU min per hour")
    win = int((found["openvino"] & ~found["onnxruntime"]).sum())
    loss = int((~found["openvino"] & found["onnxruntime"]).sum())
    result.update(wins=win, losses=loss, p=sign_test(win, loss))
    print(f"cosine mean {cos.mean():.5f} min {cos.min():.5f}; openvino +{win} / -{loss}")
    out = ROOT / "results" / f"didemo_{args.split}_backends.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
