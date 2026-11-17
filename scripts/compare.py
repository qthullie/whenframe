"""The sixth round (docs/PROTOCOL.md): five ways of embedding frames, measured the same way on four benchmarks.

    python scripts/compare.py extract --dataset charades --models siglip2-so400m,siglip2-b32   # the same frames
    python scripts/compare.py tune     ...   # each candidate's speech and sound weights, on the tuning sets
    python scripts/compare.py evaluate ...   # once, after the choices are recorded in docs/PROTOCOL.md

The candidates: SigLIP 2 B/32, B/16 (the standard mode) and So400m, and B/16
with whenframe's model v1 and v2. Every encoder sees the frames at the instants
of the standard mode's caches: each video is read again (from disk for DiDeMo,
from the Hugging Face Hub for the others, one at a time and not kept), decoded
at each encoder's size, and only the frames at the cached instants are
embedded, with OpenVINO.
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import queue
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from whenframe import bank as qbank  # noqa: E402
from whenframe import temporal  # noqa: E402
from whenframe.encoder import Encoder  # noqa: E402
from whenframe.video import sample_frames  # noqa: E402

CACHE = ROOT / "results" / "cache"
HUB = "https://huggingface.co/datasets"
CHARADES_ZIPS = [f"{HUB}/lmms-eval/charades_sta/resolve/main/Charades_v1_480_part_{i}.zip" for i in range(1, 5)]
YOUCOOK2 = f"{HUB}/nguyenminh04/youcook2-videos/resolve/main/videos/{{}}.mp4"
UNAV = f"{HUB}/nguyenminh04/unav100-videos/resolve/main/unav100_{{:03d}}.tar"
# the standard mode's caches, whose frames every encoder sees
BASE = {"didemo-test": "didemo_test_openvino", "didemo-train": "didemo_train_openvino",
        "charades": "charades_test", "youcook2": "youcook2", "unav": "unav"}  # fmt: skip
DONE = object()


def short(model: str) -> str:
    return model.replace("siglip2-", "")


def cache_name(dataset: str, model: str) -> str:
    """Where `model`'s embeddings of the dataset's frames are kept (the base model's: the standard caches)."""
    if model == "siglip2-base":
        return BASE[dataset]
    if dataset.startswith("didemo"):
        return f"didemo_{dataset.split('-')[1]}_{model}_openvino"  # as scripts/didemo.py names them
    return f"{BASE[dataset]}_{short(model)}_openvino"


# ---------------------------------------------------------------- the videos, read again
def didemo_source(dataset: str, names: list[str], folder: str, work: Path):
    for name in names:
        path = Path(folder) / name
        yield name, (path if path.exists() else Path(folder) / f"{name}.mp4"), False


def charades_source(names: list[str], work: Path):
    from msrvtt import Remote

    wanted, readers = set(names), []
    index = {}
    for url in CHARADES_ZIPS:
        z = zipfile.ZipFile(io.BufferedReader(Remote(url), buffer_size=1 << 22))
        readers.append(z)
        for info in z.infolist():
            if Path(info.filename).name in wanted:
                index[Path(info.filename).name] = (z, info)
    for name in names:
        if name not in index:
            print(f"skip {name}: not in the archives", file=sys.stderr, flush=True)
            continue
        z, info = index[name]
        path = work / f"{Path(name).stem}.mp4"  # named after the annotations, never the archive's own path
        with z.open(info) as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        yield name, path, True


def youcook2_source(names: list[str], work: Path):
    for name in names:
        path = work / f"{name}.mp4"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(YOUCOOK2.format(name), timeout=300) as r, open(path, "wb") as dst:  # noqa: S310
                    shutil.copyfileobj(r, dst)
                break
            except (OSError, http.client.HTTPException) as ex:
                if attempt == 4:
                    print(f"skip {name}: {ex}", file=sys.stderr, flush=True)
                    path = None
                else:
                    time.sleep(10 * (attempt + 1))
        if path is not None:
            yield name, path, True


def unav_source(names: list[str], work: Path, shards: int = 22):
    wanted, seen = set(names), set()
    for shard in range(shards):
        for attempt in range(4):
            try:
                with (
                    urllib.request.urlopen(UNAV.format(shard), timeout=120) as r,
                    tarfile.open(fileobj=r, mode="r|") as tar,
                ):  # noqa: S310
                    for member in tar:
                        name = Path(member.name).stem
                        if not member.isfile() or name not in wanted or name in seen:
                            continue
                        path = work / f"{name}.mp4"
                        with tar.extractfile(member) as src, open(path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        seen.add(name)
                        yield name, path, True
                break
            except (OSError, tarfile.TarError, http.client.HTTPException) as ex:
                print(f"archive {shard}: {ex}; again", file=sys.stderr, flush=True)
                time.sleep(30 * (attempt + 1))


def extract(args) -> None:
    """Each video of the dataset decoded at each encoder's size; the frames at the cached instants embedded."""
    models = args.models.split(",")
    base = np.load(CACHE / f"{BASE[args.dataset]}.npz")
    names = [str(n) for n in base["names"]]
    starts, counts, times = base["starts"], base["counts"], base["times"]
    want = {n: times[s : s + c] for n, s, c in zip(names, starts, counts, strict=True)}
    parts = {m: CACHE / f"compare_parts_{args.dataset}_{short(m)}" for m in models}
    for p in parts.values():
        p.mkdir(parents=True, exist_ok=True)
    todo_names = [n for n in names if not all((parts[m] / f"{Path(n).stem}.npz").exists() for m in models)]
    if args.limit:
        todo_names = todo_names[: args.limit]
    encoders = {m: Encoder(m, backend="openvino") for m in models}
    for enc in encoders.values():
        enc.images  # noqa: B018 - compiled before the first video
    work = Path(tempfile.mkdtemp(prefix="compare-", dir=args.work))
    source = {"didemo-test": lambda: didemo_source(args.dataset, todo_names, args.videos, work),
              "didemo-train": lambda: didemo_source(args.dataset, todo_names, args.videos, work),
              "charades": lambda: charades_source(todo_names, work),
              "youcook2": lambda: youcook2_source(todo_names, work),
              "unav": lambda: unav_source(todo_names, work)}[args.dataset]  # fmt: skip
    todo, ready = queue.Queue(maxsize=4), queue.Queue(maxsize=4)

    def read():
        try:
            for item in source():
                todo.put(item)
        finally:
            todo.put(DONE)

    def decode():
        while (item := todo.get()) is not DONE:
            name, path, temporary = item
            try:
                out = {}
                for m, enc in encoders.items():
                    t0 = time.process_time()
                    t, frames = sample_frames(path, 1.0, enc.size)
                    at = np.isin(np.round(t, 3), np.round(want[name], 3))
                    out[m] = (t[at], frames[at], time.process_time() - t0)
                ready.put((name, out, None))
            except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
                ready.put((name, None, ex))
            finally:
                if temporary:
                    path.unlink(missing_ok=True)
        todo.put(DONE)
        ready.put(DONE)

    threading.Thread(target=read, daemon=True).start()
    decoders = [threading.Thread(target=decode, daemon=True) for _ in range(2)]
    for d in decoders:
        d.start()
    finished, n, t0 = 0, 0, time.time()
    while finished < len(decoders):
        item = ready.get()
        if item is DONE:
            finished += 1
            continue
        name, out, err = item
        n += 1
        if err is not None:
            print(f"skip {name}: {err}", file=sys.stderr, flush=True)
            continue
        for m, (t, frames, decode_cpu) in out.items():
            if not np.array_equal(np.round(t, 3), np.round(want[name], 3)):
                print(f"skip {name} ({m}): not the same instants ({len(t)} of {len(want[name])})", file=sys.stderr)
                continue
            c0 = time.process_time()
            emb = encoders[m].encode_images(frames)
            tmp = parts[m] / f"{Path(name).stem}.tmp.npz"
            np.savez(tmp, emb=emb.astype(np.float16), times=t.astype(np.float32), decode_cpu=decode_cpu,
                     encode_cpu=time.process_time() - c0)  # fmt: skip
            tmp.replace(parts[m] / f"{Path(name).stem}.npz")
        if n % 50 == 0:
            print(f"{args.dataset}: {n} videos this run, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    if not args.limit:
        for m in models:
            assemble(args.dataset, m)


def assemble(dataset: str, model: str) -> None:
    """The embedded frames in the order of the standard mode's cache, with its layout."""
    base = np.load(CACHE / f"{BASE[dataset]}.npz")
    folder = CACHE / f"compare_parts_{dataset}_{short(model)}"
    names = [str(n) for n in base["names"]]
    missing = [n for n in names if not (folder / f"{Path(n).stem}.npz").exists()]
    if missing:
        print(f"{dataset}, {model}: {len(missing)} videos missing, e.g. {missing[:3]}; not assembled")
        return
    z = [np.load(folder / f"{Path(n).stem}.npz") for n in names]
    emb = np.concatenate([x["emb"] for x in z])
    assert np.array_equal(np.concatenate([x["times"] for x in z]), base["times"]), "not the same frames"
    extra = {k: base[k] for k in base.files if k not in ("emb", "precision", "decode_cpu", "encode_cpu")}
    out = CACHE / f"{cache_name(dataset, model)}.npz"
    np.savez_compressed(out, **extra, emb=emb, decode_cpu=np.array([float(x["decode_cpu"]) for x in z]),
                        encode_cpu=np.array([float(x["encode_cpu"]) for x in z]))  # fmt: skip
    print(f"wrote {out.name}: {len(names)} videos, {len(emb)} frames")


# ---------------------------------------------------------------- the candidates, searched the same way
@dataclass(frozen=True)
class Candidate:
    name: str
    model: str = "siglip2-base"
    weights: str | None = None  # whenframe's model (a .npz of results/temporal or whenframe/data)
    alpha: float = 1.0  # how much of its correction is added (WiSE-FT's mixing)

    def adapter(self) -> dict | None:
        if self.weights is None:
            return None
        p = temporal.load(Path(self.weights))
        p["out_w"], p["out_b"] = self.alpha * p["out_w"], self.alpha * p["out_b"]
        return p


def bank_params(model: str, spans: bool = False) -> tuple[float, float]:
    """(beta, strength) of the query bank as the tool sets them for the encoder and the kind of moments (the values
    chosen on DiDeMo's tuning videos, kept in whenframe.encoder.MODELS)."""
    from whenframe.encoder import MODELS

    spec = MODELS[model]
    lam = spec.spans_lambda if spans and spec.spans_lambda is not None else spec.bank_lambda
    return spec.bank_beta, lam


def text_embeddings(model: str, key: str, texts: list[str]) -> np.ndarray:
    """Descriptions embedded by the encoder's text tower, as the tool embeds a query; kept in the cache."""
    path = CACHE / f"{key}_text_compare_{short(model)}.npy"
    if not path.exists():
        enc = Encoder(model, backend="openvino")
        np.save(path, np.concatenate([enc.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)]))
    return np.load(path).astype(np.float32)


def adapted(c: Candidate, emb, times, starts, counts) -> np.ndarray:
    p = c.adapter()
    if p is None:
        return np.asarray(emb, dtype=np.float32)
    out = np.empty(emb.shape, np.float32)
    for s, n in zip(starts, counts, strict=True):
        out[s : s + n] = temporal.apply(p, emb[s : s + n], times[s : s + n])
    return out


# ---------------------------------------------------------------- the four benchmarks, for one candidate
def didemo_bank(model: str) -> np.ndarray:
    """DiDeMo's bank as each round searched with it (the descriptions embedded by the encoder, in float32)."""
    key = {"siglip2-base": "", "siglip2-b32": "_siglip2-b32", "siglip2-so400m": "_siglip2-so400m_lower"}[model]
    return np.load(CACHE / f"didemo_bank_text{key}.npy").astype(np.float32)


class Benchmarks:
    """The four benchmarks, loaded once; each candidate's frames and descriptions on demand."""

    def __init__(self, args):
        from argparse import Namespace

        self.args = args
        self.shared = {"model_dir": args.model_dir, "train_annotations": args.annotations}
        self.ns = Namespace
        self._cache = {}

    def once(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # DiDeMo: the tuning ("train") or test videos, OpenVINO, repeats skipped
    def didemo(self, c: Candidate, split: str):
        import didemo

        def base():
            ann = self.args.annotations if split == "train" else self.args.test_annotations
            frozen, queries, _ = didemo.load(split, ann)
            return frozen, queries

        frozen, queries = self.once(("didemo", split), base)
        z = self.once(
            ("didemo", split, c.model), lambda: np.load(CACHE / f"{cache_name('didemo-' + split, c.model)}.npz")
        )
        assert z["names"].tolist() == frozen.names, "not the same videos"
        mask = z["times"] < didemo.SECONDS
        assert np.array_equal(z["times"][mask], frozen.times), "not the same frames"
        f = replace(frozen, emb=z["emb"][mask].astype(np.float32), _offsets={}, bank=didemo_bank(c.model)).only_kept()
        f = replace(f, emb=adapted(c, f.emb, f.times, f.starts, f.counts), _offsets={})
        if c.model == "siglip2-so400m":  # lower case, as chosen for it in the fourth round
            text = np.load(CACHE / f"didemo_{split}_text_siglip2-so400m_lower.npy").astype(np.float32)
        else:
            text = didemo.text_embeddings(queries, split, self.args.model_dir, c.model)
        return f, queries, text

    def search_params(self, model: str, method: str) -> dict:
        beta, lam = bank_params(model, spans=method == "whenframe_bank")
        params = json.loads((ROOT / "results" / "didemo_tuned.json").read_text())[method]["params"]
        return {**params, "beta": beta, "lam": lam}

    def didemo_score(self, c: Candidate, split: str) -> dict:
        from benchmark import run

        f, queries, text = self.didemo(c, split)
        r = run(f, queries, text, "frame_window_bank", self.search_params(c.model, "frame_window_bank"),
                per_query=True)  # fmt: skip
        return {"value": r["corpus"]["R@5 IoU>=0.5"], "items": [bool(x) for x in r["found"]], "full": strip(r)}

    # Charades-STA: every frame, DiDeMo's parameters, the moment placed in the given video
    def charades_score(self, c: Candidate) -> dict:
        import charades
        from benchmark import run

        frozen, queries, base_text = self.once("charades", lambda: charades.load(self.ns(
            annotations=self.args.charades_annotations, **self.shared)))  # fmt: skip
        z = self.once(("charades", c.model), lambda: np.load(CACHE / f"{cache_name('charades', c.model)}.npz"))
        assert np.array_equal(z["times"], frozen.times), "not the same frames"
        f = replace(frozen, emb=z["emb"].astype(np.float32), _offsets={}, bank=didemo_bank(c.model))
        f = replace(f, emb=adapted(c, f.emb, f.times, f.starts, f.counts), _offsets={})
        text = base_text if c.model == "siglip2-base" else text_embeddings(
            c.model, "charades_test", [q["text"] for q in queries])  # fmt: skip
        r = run(f, queries, text, "frame_window_bank", self.search_params(c.model, "frame_window_bank"),
                per_query=True)  # fmt: skip
        hits = [x >= 0.5 for x in r["single_iou"]]
        return {"value": r["single"]["R1 IoU>=0.5"], "items": hits, "full": strip(r)}

    # YouCook2: repeats skipped, speech added to the adaptive spans; a third of the videos for tuning
    def youcook2(self, c: Candidate):
        import youcook2

        f, queries, base_text, db, _, _ = self.once("youcook2", lambda: youcook2.load(self.ns(
            annotations=self.args.youcook2_annotations, **self.shared)))  # fmt: skip
        z = self.once(("youcook2", c.model), lambda: np.load(CACHE / f"{cache_name('youcook2', c.model)}.npz"))
        assert np.array_equal(z["times"], f.times), "not the same frames"
        g = replace(f, emb=z["emb"].astype(np.float32), _offsets={}, bank=didemo_bank(c.model))
        g = replace(g, emb=adapted(c, g.emb, g.times, g.starts, g.counts), _offsets={})
        text = base_text if c.model == "siglip2-base" else text_embeddings(
            c.model, "youcook2", [q["text"] for q in queries])  # fmt: skip
        tuning = youcook2.split(f.names)
        parts = {"tuning": [i for i, q in enumerate(queries) if q["name"] in tuning],
                 "test": [i for i, q in enumerate(queries) if q["name"] not in tuning]}  # fmt: skip
        return g, queries, text, db, parts

    def youcook2_score(self, c: Candidate, part: str, cfg=None) -> dict:
        """R@5 with speech and spans; cfg None: the best of the third round's grid on these steps."""
        import youcook2
        from benchmark import frame_scores

        g, queries, text, db, parts = self.youcook2(c)
        params = self.search_params(c.model, "whenframe_bank")
        visual = frame_scores(g, text, "whenframe_bank", params)
        which = parts[part]
        best = None
        for config in [cfg] if cfg is not None else youcook2.GRID:
            r = youcook2.measure(g, queries, youcook2.with_speech(g, queries, visual, db, config, which),
                                 "whenframe_bank", params, which)  # fmt: skip
            if best is None or r["corpus"]["R@5 IoU>=0.5"] > best[1]["corpus"]["R@5 IoU>=0.5"]:
                best = (config, r)
        config, r = best
        return {"value": r["corpus"]["R@5 IoU>=0.5"], "items": [bool(x) for x in r.pop("found")], "full": r,
                "speech": {"weight": config.weight, "pad": config.pad, "normalise": config.normalise}}  # fmt: skip

    # UnAV-100: every frame, frames and sound; a third of the videos for tuning
    def unav_score(self, c: Candidate, part: str, weight: float | None = None) -> dict:
        import unav

        a = self.args
        z, labels, events, base_siglip, clap, _ = self.once("unav", lambda: unav.prepare(self.ns(
            annotations=a.unav_annotations, model_dir=a.model_dir, clap_dir=a.clap_dir)))  # fmt: skip
        zm = self.once(("unav", c.model), lambda: np.load(CACHE / f"{cache_name('unav', c.model)}.npz"))
        names = z["names"].tolist()
        tuning = unav.tuning_videos(names)
        keep = tuning if part == "tuning" else set(names) - tuning
        lib = unav.Library(zm, keep)
        starts = np.array([r for r, _ in lib.ranges])
        counts = np.array([n for _, n in lib.ranges])
        lib.emb = adapted(c, lib.emb, lib.times, starts, counts)
        siglip = base_siglip if c.model == "siglip2-base" else text_embeddings(c.model, "unav_labels", labels)
        beta, lam = bank_params(c.model)
        frames = unav.standardise(lib.frames(siglip, qbank.load(c.model).astype(np.float32), beta, lam))
        heard = self.once(("unav-heard", part), lambda: unav.standardise(
            unav.Library(z, keep).sound(clap, 5.0)))  # fmt: skip
        best = None
        for w in [weight] if weight is not None else unav.WEIGHTS:
            r = unav.evaluate_scores(lib, events, labels, frames + w * heard)
            if best is None or r["p_at_k"] > best[1]["p_at_k"]:
                best = (w, r)
        w, r = best
        return {"value": r["p_at_k"], "items": {k: v["p_at_k"] for k, v in r["labels"].items()},
                "full": {"p_at_k": r["p_at_k"], "map": r["map"]}, "sound_weight": w}  # fmt: skip


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("found", "single_iou", "params")}


NAMED = {"b32": Candidate("b32", "siglip2-b32"), "base": Candidate("base"),
         "so400m": Candidate("so400m", "siglip2-so400m"),
         "v1": Candidate("v1", weights=str(ROOT / "whenframe" / "data" / "temporal-siglip2-base.npz"))}  # fmt: skip


def candidate(spec: str) -> Candidate:
    """b32, base, so400m, v1, or NAME=WEIGHTS@ALPHA (whenframe's model on the base encoder)."""
    if spec in NAMED:
        return NAMED[spec]
    name, rest = spec.split("=", 1)
    weights, _, alpha = rest.partition("@")
    return Candidate(name, weights=weights, alpha=float(alpha or 1.0))


TUNED = ROOT / "results" / "compare_tuned.json"
TEST = ROOT / "results" / "compare_test.json"


def tune(args) -> None:
    """Each candidate on the three tuning sets: its speech and sound weights chosen on the third and fourth rounds'
    grids, and the mean of the three primary metrics (DiDeMo, YouCook2, UnAV-100; Charades-STA has none)."""
    b = Benchmarks(args)
    out = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    for spec in args.candidates.split(","):
        c = candidate(spec)
        d, y, u = b.didemo_score(c, "train"), b.youcook2_score(c, "tuning"), b.unav_score(c, "tuning")
        out[c.name] = {"spec": spec, "model": c.model, "weights": c.weights, "alpha": c.alpha,
                       "didemo": d["value"], "youcook2": y["value"], "unav": u["value"],
                       "mean": float(np.mean([d["value"], y["value"], u["value"]])),
                       "speech": y["speech"], "sound_weight": u["sound_weight"]}  # fmt: skip
        r = out[c.name]
        print(f"{c.name}: DiDeMo {100 * r['didemo']:.1f}, YouCook2 {100 * r['youcook2']:.1f} (speech "
              f"{r['speech']['weight']:g}), UnAV-100 {100 * r['unav']:.1f} (sound {r['sound_weight']:g}); "
              f"mean {100 * r['mean']:.2f}", flush=True)  # fmt: skip
        TUNED.write_text(json.dumps(out, indent=1))


def paired(a: dict, b: dict) -> dict:
    """Items of the four tests where a does better than b, and worse; the sign test's p."""
    from explore import sign_test

    win = loss = 0
    for k in ("didemo", "charades", "youcook2"):
        x, y = np.asarray(a[k]["items"], bool), np.asarray(b[k]["items"], bool)
        win, loss = win + int((x & ~y).sum()), loss + int((~x & y).sum())
    for label, v in a["unav"]["items"].items():
        w = b["unav"]["items"][label]
        win, loss = win + int(v > w), loss + int(v < w)
    return {"wins": win, "losses": loss, "p": sign_test(win, loss)}


def evaluate(args) -> None:
    """Once: every candidate on the four test sets, with the weights chosen for it on the tuning sets."""
    from whenframe.speech import SpeechConfig

    tuned = json.loads(TUNED.read_text())
    b = Benchmarks(args)
    res, items = {"candidates": args.candidates.split(",")}, {}
    for name in res["candidates"]:
        t = tuned[name]
        c = candidate(t["spec"])
        sp = t["speech"]
        scores = {"didemo": b.didemo_score(c, "test"), "charades": b.charades_score(c),
                  "youcook2": b.youcook2_score(c, "test", SpeechConfig(sp["weight"], sp["pad"], sp["normalise"])),
                  "unav": b.unav_score(c, "test", t["sound_weight"])}  # fmt: skip
        items[name] = scores
        res[name] = {k: {"value": v["value"], "full": v["full"]} for k, v in scores.items()}
        res[name]["mean"] = float(np.mean([v["value"] for v in scores.values()]))
        res[name]["tuned"] = t
        print(f"{name}: " + ", ".join(f"{k} {100 * v['value']:.1f}" for k, v in scores.items())
              + f"; mean {100 * res[name]['mean']:.2f}", flush=True)  # fmt: skip
    names = res["candidates"]
    res["ranking"] = sorted(names, key=lambda n: -res[n]["mean"])
    res["pairs"] = {f"{a} vs {b_}": paired(items[a], items[b_]) for a in names for b_ in names if a != b_}
    TEST.write_text(json.dumps(res, indent=1))
    print("ranking: " + " > ".join(f"{n} ({100 * res[n]['mean']:.2f})" for n in res["ranking"]))
    print(f"wrote {TEST.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "assemble", "tune", "evaluate"])
    ap.add_argument(
        "--candidates",
        default="b32,base,so400m,v1",
        help="tune, evaluate: b32, base, so400m, v1, or NAME=WEIGHTS@ALPHA, comma-separated",
    )
    ap.add_argument("--annotations", help="DiDeMo's didemo_train.json (the tuning videos, and the query bank)")
    ap.add_argument("--test-annotations", help="DiDeMo's didemo_test.json")
    ap.add_argument("--charades-annotations", help="Charades-STA's test parquet")
    ap.add_argument("--youcook2-annotations", help="YouCook2's validation parquet")
    ap.add_argument("--unav-annotations", help="UnAV-100's unav100_all.json")
    ap.add_argument("--clap-dir", help="folder of CLAP's files")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files")
    ap.add_argument("--dataset", choices=list(BASE), help="extract: whose frames")
    ap.add_argument("--models", default="siglip2-so400m,siglip2-b32", help="extract: encoders, comma-separated")
    ap.add_argument("--videos", help="extract: the folder of DiDeMo's videos (the others come from the Hub)")
    ap.add_argument("--work", help="extract: where each video is written while it is decoded")
    ap.add_argument("--limit", type=int, help="extract: only the first videos (a trial; nothing assembled)")
    args = ap.parse_args()
    if args.stage == "assemble":
        for m in args.models.split(","):
            assemble(args.dataset, m)
        return
    {"extract": extract, "tune": tune, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
