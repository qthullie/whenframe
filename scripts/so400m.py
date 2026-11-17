"""SigLIP 2 So400m in place of the base model, its frames embedded by OpenVINO (fourth round).

The same frames as the base model's, embedded again (scripts/didemo.py extract
--model siglip2-so400m --backend openvino), searched with the default search:
fixed windows around the best frames, repeats skipped, the query bank. How the
descriptions are cleaned and the bank's beta and strength are chosen on the
tuning videos; the test videos are evaluated once, with them.

    python scripts/so400m.py tune --split train --annotations ... --train-annotations ...
    python scripts/so400m.py evaluate --split test --annotations ... --train-annotations ...
    python scripts/so400m.py latency --split test ...    # milliseconds per search, for the README

Writes results/didemo_tuned_so400m.json, then results/didemo_test_so400m.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import didemo  # noqa: E402
from benchmark import PRIMARY, SPACE, run  # noqa: E402
from explore import sign_test  # noqa: E402

from whenframe.encoder import Encoder  # noqa: E402

MODEL = "siglip2-so400m"
BACKEND = "openvino"
CLEANING = ("canonical", "lower")  # open_clip's cleaning, as the model was trained; or lower case only, as the base
TUNED = ROOT / "results" / "didemo_tuned_so400m.json"


def encoder(clean: str) -> Encoder:
    enc = Encoder(MODEL, backend=BACKEND)
    enc.spec = replace(enc.spec, clean=clean)
    return enc


def embed(texts: list[str], name: str, clean: str) -> np.ndarray:
    """So400m's embeddings of the texts, cleaned one way, cached."""
    path = didemo.CACHE / f"{name}_{MODEL}_{clean}.npy"
    if not path.exists():
        enc = encoder(clean)
        out = np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)])
        np.save(path, out.astype(np.float32))
    return np.load(path)


def frames(args, frozen):
    """The split's frames embedded by So400m, in the order of the base model's cache."""
    z = np.load(didemo.CACHE / f"didemo_{args.split}{didemo.suffix(MODEL, BACKEND)}.npz")
    assert z["names"].tolist() == frozen.names, "not the same videos"
    mask = z["times"] < didemo.SECONDS
    assert np.array_equal(z["times"][mask], frozen.times), "not the same frames"
    timing = {k: float(z[k].sum()) for k in ("decode_cpu", "decode_wall", "encode_cpu", "encode_wall")}
    hours = float(z["durations"].sum() / 3600)
    cost = {"hours": hours, "frames": int(len(z["times"])),
            "cpu_min_per_hour": {k: timing[k] / 60 / hours for k in ("decode_cpu", "encode_cpu")},
            "encode_ms_per_frame": 1000 * timing["encode_cpu"] / len(z["times"])}  # fmt: skip
    return replace(frozen, emb=z["emb"][mask].astype(np.float32), _offsets={}), cost


def prepare(args):
    """The base model's frames embedded by OpenVINO, as the tool runs it, and So400m's; the queries."""
    frozen, queries, _ = didemo.load(args.split, args.annotations)
    frozen.bank = didemo.bank(args)
    base_text = didemo.text_embeddings(queries, args.split, args.model_dir)
    z = np.load(didemo.CACHE / f"didemo_{args.split}{didemo.suffix(didemo.MAIN, BACKEND)}.npz")
    assert z["names"].tolist() == frozen.names, "not the same videos"
    base = replace(frozen, emb=z["emb"][z["times"] < didemo.SECONDS].astype(np.float32), _offsets={})
    big, cost = frames(args, frozen)
    return base, queries, base_text, big, cost


def texts(args, queries, clean: str):
    """So400m's query embeddings and bank, both cleaned the same way."""
    q = embed([x["text"] for x in queries], f"didemo_{args.split}_text", clean)
    bank = embed(didemo.bank_texts(args), "didemo_bank_text", clean)
    return q, bank


def default_method() -> tuple[str, dict]:
    tuned = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())
    return tuned["default"], tuned[tuned["default"]]["params"]


def tune(args) -> None:
    frozen, queries, base_text, big, cost = prepare(args)
    method, params = default_method()
    base = run(frozen.only_kept(), queries, base_text, method, params)
    print(f"base model with OpenVINO: R@5 {100 * base[PRIMARY[0]][PRIMARY[1]]:.1f}", flush=True)
    rows = []
    for clean in CLEANING:
        q, bank = texts(args, queries, clean)
        kept = replace(big, bank=bank, _offsets={}).only_kept()
        for beta, lam in itertools.product(SPACE["bank"]["beta"], SPACE["bank"]["lam"]):
            p = {**params, "beta": beta, "lam": lam}
            r = run(kept, queries, q, method, p)
            rows.append({"clean": clean, "beta": beta, "lam": lam, "score": r[PRIMARY[0]][PRIMARY[1]],
                         "corpus": r["corpus"], "single": r["single"], "video": r["video"]})  # fmt: skip
            print(f"{clean:9s} beta {beta:5g} lambda {lam:4g}: R@5 {100 * rows[-1]['score']:.1f}", flush=True)
    best = max(rows, key=lambda x: x["score"])
    TUNED.write_text(json.dumps({
        "split": args.split, "videos": len(frozen.names), "queries": len(queries), "method": method,
        "params": params, "best": {k: best[k] for k in ("clean", "beta", "lam", "score")},
        "base": {"score": base[PRIMARY[0]][PRIMARY[1]], "corpus": base["corpus"]}, "cost": cost, "grid": rows,
    }, indent=1))  # fmt: skip
    print(f"best: {best['clean']}, beta {best['beta']:g}, lambda {best['lam']:g}: R@5 {100 * best['score']:.1f}")


def query_ms(clean: str, queries, n: int = 50) -> float:
    """Milliseconds to embed one description, as the tool does for each search."""
    enc = encoder(clean)
    enc.encode_texts([queries[0]["text"]])  # loads the text tower
    t0 = time.perf_counter()
    for x in queries[:n]:
        enc.encode_texts([x["text"]])
    return 1000 * (time.perf_counter() - t0) / n


def evaluate(args) -> None:
    frozen, queries, base_text, big, cost = prepare(args)
    method, params = default_method()
    best = json.loads(TUNED.read_text())["best"]
    q, bank = texts(args, queries, best["clean"])
    p = {**params, "beta": best["beta"], "lam": best["lam"]}
    big = replace(big, bank=bank, _offsets={})
    res = {"split": args.split, "videos": len(frozen.names), "queries": len(queries), "method": method, "params": p,
           "clean": best["clean"]}  # fmt: skip
    found = {}
    for name, f, text, pp in (("base", frozen, base_text, params), (MODEL, big, q, p)):
        r = run(f.only_kept(), queries, text, method, pp, per_query=True)
        found[name] = np.array(r.pop("found"))
        res[name] = r
        print(f"{name}: R@5 {100 * r[PRIMARY[0]][PRIMARY[1]]:.1f}", flush=True)
    win = int((found[MODEL] & ~found["base"]).sum())
    loss = int((~found[MODEL] & found["base"]).sum())
    cost["query_ms"] = query_ms(best["clean"], queries)
    res.update(wins=win, losses=loss, p=sign_test(win, loss), cost=cost)
    out = ROOT / "results" / f"didemo_{args.split}_so400m.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"So400m +{win} / -{loss} (p = {res['p']:.2g}); {sum(cost['cpu_min_per_hour'].values()):.1f} CPU min per "
          f"hour; {cost['query_ms']:.0f} ms per query; wrote {out.relative_to(ROOT)}")  # fmt: skip


def search_latency(args) -> None:
    """Milliseconds per search over the test frames once the description is embedded (benchmark.latency), added
    to the evaluation's costs; the evaluation's numbers are left as they are."""
    from benchmark import latency

    out = ROOT / "results" / f"didemo_{args.split}_so400m.json"
    res = json.loads(out.read_text())
    _, queries, _, big, _ = prepare(args)
    q, bank = texts(args, queries, res["clean"])
    kept = replace(big, bank=bank, _offsets={}).only_kept()
    res["cost"]["search_ms"] = latency(kept, q, res["method"], res["params"])
    out.write_text(json.dumps(res, indent=1))
    print(f"{res['cost']['search_ms']:.0f} ms per search, besides embedding the description; wrote "
          f"{out.relative_to(ROOT)}")  # fmt: skip


def main() -> None:
    stage = sys.argv[1]
    args = didemo.main_args(["evaluate", *sys.argv[2:]])
    {"tune": tune, "evaluate": evaluate, "latency": search_latency}[stage](args)


if __name__ == "__main__":
    main()
