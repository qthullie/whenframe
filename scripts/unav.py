"""UnAV-100: what is heard, found with CLAP among 2,167 videos with timed events (fourth round).

UnAV-100 (Geng et al., CVPR 2023) marks where 100 kinds of events that can be
both seen and heard happen in untrimmed videos of about 40 seconds: people
laughing, clapping, a dog barking, a guitar playing... Only its 2,167 test
videos are used, from the copy at huggingface.co/datasets/nguyenminh04/unav100-videos,
and nothing is trained on them. Each label is a description, searched for among
the videos all at once, with the default search of the tool (10-second windows
around the best frames); the frames are scored by SigLIP 2 as the tool does,
the sound by CLAP, or both, standardised and added.

A moment counts as right when its best instant, where it is played from, falls
inside an event of the label in that video: half of the events last less than 7
seconds, which a 10-second moment cannot match at IoU 0.5.

Many files of the copy last longer than UnAV-100 says their video does (440
seconds for one of 20.7): the whole YouTube video, it seems, from which
UnAV-100 cut its clip at an offset the annotations do not give. Only the files
that last as long as the annotations say, within ALIGNED seconds, are used;
the others are not embedded.

    python scripts/unav.py extract --annotations unav100_all.json   # streams the 22 archives, keeps no video
    python scripts/unav.py tune --annotations unav100_all.json      # a third of the videos: CLAP's window, weight
    python scripts/unav.py evaluate --annotations unav100_all.json  # the other two thirds, once

Writes results/unav_tuned.json, then results/unav_test.json.
"""

from __future__ import annotations

import argparse
import http.client
import json
import random
import resource
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from explore import sign_test  # noqa: E402

from whenframe import sounds  # noqa: E402
from whenframe.bank import offsets as bank_offsets  # noqa: E402
from whenframe.encoder import Encoder  # noqa: E402
from whenframe.moments import windows  # noqa: E402
from whenframe.video import drop_repeats, probe, sample_frames  # noqa: E402

REPO = "nguyenminh04/unav100-videos"
SHARDS = 22
CACHE = ROOT / "results" / "cache"
PARTS = CACHE / "unav_parts"
FRAMES = CACHE / "unav.npz"
MAIN = "siglip2-base"
LENGTHS = (5.0, 10.0)  # seconds of sound per CLAP window; consecutive windows overlap by half
WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # of CLAP's standardised scores, added to the frames'
SHORTLIST, WINDOW, PER_VIDEO = 64, 10.0, 3  # the tool's default search
K = 10  # moments looked at per label: the primary metric is the share of them that are right
ALIGNED = 1.0  # seconds: a file whose length differs more from the annotations' is not UnAV-100's clip
MIN_EVENTS = 3  # a label is searched for in a part of the videos only if it happens there this often
LIST = 100  # moments ranked per label for the average precision


def cpu() -> float:
    """CPU time of this process and of the ffmpeg runs it waited for."""
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + r.ru_utime + r.ru_stime


def process(video: Path, part: Path, enc: Encoder, clap: sounds.Clap) -> None:
    c0 = cpu()
    info = probe(video)
    t, frames = sample_frames(video, 1.0, enc.size)
    c1 = cpu()
    emb = enc.encode_images(frames)
    c2 = cpu()
    wave = sounds.sound(video)
    c3 = cpu()
    keep = np.zeros(len(frames), dtype=bool)
    keep[drop_repeats(frames)] = True
    out = {"emb": emb.astype(np.float16), "times": t.astype(np.float32), "keep": keep,
           "duration": info.duration or len(frames), "decode_cpu": c1 - c0, "encode_cpu": c2 - c1,
           "sound_cpu": c3 - c2, "sound_seconds": len(wave) / sounds.RATE}  # fmt: skip
    for length in LENGTHS:
        c = cpu()
        starts, clips = sounds.windows(wave, length, length / 2)
        out[f"starts_{length:g}"] = starts.astype(np.float32)
        out[f"audio_{length:g}"] = clap.encode_audio(clips).astype(np.float16)
        out[f"clap_cpu_{length:g}"] = cpu() - c
    tmp = part.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(part)  # a video is either done or not started


def extract(args) -> None:
    """Each video of each archive, read from the stream into an empty folder, embedded, then deleted."""
    ann = json.loads(Path(args.annotations).read_text())["database"]
    enc = Encoder(MAIN, folder=args.model_dir, backend="openvino")
    clap = sounds.Clap(args.clap_dir, backend="openvino")
    enc.images  # noqa: B018 - both towers loaded before the first video is timed
    clap.encode_audio([np.zeros(sounds.RATE, np.float32)])
    PARTS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="unav-", dir=args.work))
    t0 = time.time()
    for shard in range(SHARDS):
        marker = PARTS / f"shard_{shard:03d}.done"
        for attempt in range(4):
            if marker.exists():
                break
            url = f"https://huggingface.co/datasets/{REPO}/resolve/main/unav100_{shard:03d}.tar"
            try:
                with urllib.request.urlopen(url, timeout=120) as r, tarfile.open(fileobj=r, mode="r|") as tar:
                    for member in tar:
                        name = Path(member.name).stem
                        if not member.isfile() or not member.name.endswith(".mp4") or name not in ann:
                            continue
                        part = PARTS / f"{name}.npz"
                        if part.exists() or (PARTS / f"{name}.other").exists():
                            continue
                        video = work / "video.mp4"  # never the archive's own path
                        with tar.extractfile(member) as src, open(video, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        try:
                            seconds = probe(video).duration or 0.0
                            if abs(seconds - ann[name]["duration"]) > ALIGNED:
                                (PARTS / f"{name}.other").write_text(f"{seconds:.3f}\n")  # not the annotated clip
                                continue
                            process(video, part, enc, clap)
                        except Exception as ex:  # noqa: BLE001 - a broken file is skipped, not fatal
                            print(f"skip {name}: {ex}", file=sys.stderr)
                        finally:
                            video.unlink(missing_ok=True)
                marker.touch()
            # a cut stream: the archive is read again, and the videos already embedded are skipped
            except (OSError, tarfile.TarError, http.client.HTTPException) as ex:
                print(f"archive {shard}: {ex}; again in {30 * (attempt + 1)} s", file=sys.stderr, flush=True)
                time.sleep(30 * (attempt + 1))
        done = len(list(PARTS.glob("*.npz")))
        print(f"archive {shard + 1}/{SHARDS}: {done} videos, {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    assemble(ann)


def assemble(ann: dict) -> None:
    names, z = [], []
    for path in sorted(PARTS.glob("*.npz")):
        if path.stem in ann and not path.stem.endswith(".tmp"):
            x = np.load(path)
            if abs(float(x["duration"]) - ann[path.stem]["duration"]) <= ALIGNED:  # embedded before the check
                names.append(path.stem)
                z.append(x)
    other = len(list(PARTS.glob("*.other"))) + sum(1 for p in PARTS.glob("*.npz") if p.stem in ann) - len(names)
    print(f"{len(names)} files last as long as their annotations say; {other} others left out")
    counts = np.array([len(x["times"]) for x in z])
    out = {"names": np.array(names), "emb": np.concatenate([x["emb"] for x in z]),
           "times": np.concatenate([x["times"] for x in z]), "keep": np.concatenate([x["keep"] for x in z]),
           "counts": counts, "starts": np.concatenate([[0], np.cumsum(counts)[:-1]]),
           "left_out": np.array(other)}  # fmt: skip
    for k in ("duration", "decode_cpu", "encode_cpu", "sound_cpu", "sound_seconds",
              *(f"clap_cpu_{length:g}" for length in LENGTHS)):  # fmt: skip
        out[k] = np.array([float(x[k]) for x in z])
    for length in LENGTHS:
        n = np.array([len(x[f"starts_{length:g}"]) for x in z])
        out[f"audio_{length:g}"] = np.concatenate([x[f"audio_{length:g}"] for x in z])
        out[f"wstarts_{length:g}"] = np.concatenate([x[f"starts_{length:g}"] for x in z])
        out[f"wcounts_{length:g}"] = n
    np.savez_compressed(FRAMES, **out)
    print(f"wrote {FRAMES}: {len(names)} videos, {counts.sum()} frames")


# ---------------------------------------------------------------- search and evaluation
def tuning_videos(names: list[str]) -> set[str]:
    """A third of the videos, drawn once, for choosing CLAP's window and weight."""
    return set(random.Random(0).sample(sorted(names), len(names) // 3))


class Library:
    """The embedded videos of one part of the split, as the search sees them."""

    def __init__(self, z, keep_names: set[str]):
        names = z["names"].tolist()
        self.videos = [i for i, n in enumerate(names) if n in keep_names]
        self.names = [names[i] for i in self.videos]
        self.durations = z["duration"][self.videos]
        rows, self.ranges = [], []
        for i in self.videos:
            s, c = int(z["starts"][i]), int(z["counts"][i])
            self.ranges.append((sum(len(r) for r in rows), c))
            rows.append(np.arange(s, s + c))
        self.rows = np.concatenate(rows)
        self.times = z["times"][self.rows].astype(np.float64)
        self.emb = z["emb"][self.rows].astype(np.float32)
        self.z = z

    def frames(self, text: np.ndarray, bank: np.ndarray, beta: float = 50.0, lam: float = 1.0) -> np.ndarray:
        """(frames, labels): SigLIP's cosine minus the query bank's offset, as the tool scores frames."""
        return self.emb @ text.T - lam * bank_offsets(self.emb, bank, beta)[:, None]

    def sound(self, text: np.ndarray, length: float) -> np.ndarray:
        """(frames, labels): CLAP's best cosine of the windows around each frame's middle, as the tool scores
        them (sounds.frame_scores); NaN for frames of videos without sound."""
        z, key = self.z, f"{length:g}"
        counts = z[f"wcounts_{key}"]
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        rows = np.concatenate([np.arange(first[v], first[v] + counts[v]) for v in self.videos])
        ids = np.repeat(np.arange(len(self.videos)), counts[self.videos])
        starts = z[f"wstarts_{key}"][rows].astype(np.float64)
        cos = z[f"audio_{key}"][rows].astype(np.float32) @ text.T
        ranges = dict(enumerate(self.ranges))
        return np.stack([sounds.frame_scores(self.times, ranges, ids, starts, cos[:, j], length)
                         for j in range(len(text))], axis=1)  # fmt: skip


def standardise(s: np.ndarray) -> np.ndarray:
    """Per label, as the tool does for one query: centred and scaled over the library's frames, 0 where nothing
    was heard (sounds.standardised)."""
    return np.stack([sounds.standardised(s[:, j]) for j in range(s.shape[1])], axis=1)


def moments(lib: Library, col: np.ndarray, count: int = LIST) -> list[tuple[int, float, float, float, float]]:
    """(video, start, end, instant, score) of the first moments for one label, as the tool ranks them."""
    best = np.array([col[r : r + c].max() for r, c in lib.ranges])
    out = []
    for i in np.argsort(-best)[:SHORTLIST]:
        r, c = lib.ranges[i]
        t, v = lib.times[r : r + c], col[r : r + c].astype(np.float64)
        for s in windows(t, v, float(lib.durations[i]), WINDOW, PER_VIDEO, 1.0):
            inside = np.flatnonzero((t >= s.start) & (t < s.end))
            inside = inside if len(inside) else np.arange(len(t))
            out.append((int(i), s.start, s.end, float(t[inside[np.argmax(v[inside])]]), s.peak))
    out.sort(key=lambda m: -m[4])
    return out[:count]


def judge(lib: Library, events: dict, label: str, found: list) -> dict:
    """Precision of the first K moments and average precision of the first LIST, each event found once."""
    hit, used = [], set()
    for v, _, _, at, _ in found:
        mid = at + 0.5
        match = next((j for j, (a, b) in enumerate(events.get((lib.names[v], label), ())) if a <= mid <= b
                      and (v, j) not in used), None)  # fmt: skip
        if match is None:
            # an instant inside an event already found is still right, but finds nothing new
            again = any(a <= mid <= b for a, b in events.get((lib.names[v], label), ()))
            hit.append((again, False))
        else:
            used.add((v, match))
            hit.append((True, True))
    total = sum(len(events.get((n, label), ())) for n in lib.names)
    right = np.array([h[0] for h in hit[:K]], dtype=float)
    new = np.array([h[1] for h in hit], dtype=float)
    ap = float((new * np.cumsum(new) / np.arange(1, len(new) + 1)).sum() / total) if total else 0.0
    return {"p_at_k": float(right.sum() / K), "ap": ap, "events": total}


def evaluate_scores(lib: Library, events: dict, labels: list[str], scores: np.ndarray) -> dict:
    """Over the labels that happen at least MIN_EVENTS times in these videos."""
    count = {lab: sum(len(events.get((n, lab), ())) for n in lib.names) for lab in labels}
    per = {lab: judge(lib, events, lab, moments(lib, scores[:, j])) for j, lab in enumerate(labels)
           if count[lab] >= MIN_EVENTS}  # fmt: skip
    return {"p_at_k": float(np.mean([x["p_at_k"] for x in per.values()])),
            "map": float(np.mean([x["ap"] for x in per.values()])), "labels": per}  # fmt: skip


def prepare(args):
    ann = json.loads(Path(args.annotations).read_text())["database"]
    z = np.load(FRAMES)
    labels = sorted({a["label"] for v in ann.values() for a in v["annotations"]})
    events = {}
    for name, v in ann.items():
        for a in v["annotations"]:
            events.setdefault((name, a["label"]), []).append(tuple(a["segment"]))
    siglip = text_embeddings(labels, "siglip", lambda t: Encoder(MAIN, folder=args.model_dir).encode_texts(t))
    clap = text_embeddings(labels, "clap", lambda t: sounds.Clap(args.clap_dir).encode_texts(t))
    bank = np.load(ROOT / "whenframe" / "data" / "bank-siglip2-base.npy").astype(np.float32)
    return z, labels, events, siglip, clap, bank


def text_embeddings(labels: list[str], name: str, encode) -> np.ndarray:
    path = CACHE / f"unav_labels_{name}.npy"
    if not path.exists():
        np.save(path, encode(labels).astype(np.float32))
    return np.load(path)


def compare(a: dict, b: dict) -> dict:
    """Labels whose precision of the first K moments went up, and down, from b to a."""
    up = sum(a["labels"][k]["p_at_k"] > b["labels"][k]["p_at_k"] for k in a["labels"])
    down = sum(a["labels"][k]["p_at_k"] < b["labels"][k]["p_at_k"] for k in a["labels"])
    return {"up": int(up), "down": int(down), "p": sign_test(int(up), int(down))}


def tune(args) -> None:
    z, labels, events, siglip, clap, bank = prepare(args)
    names = z["names"].tolist()
    lib = Library(z, tuning_videos(names))
    frames = standardise(lib.frames(siglip, bank))
    res = {"videos": len(lib.names), "frames": evaluate_scores(lib, events, labels, frames)}
    res["labels"] = len(res["frames"]["labels"])
    print(f"frames: P@{K} {100 * res['frames']['p_at_k']:.1f}  mAP {100 * res['frames']['map']:.1f}", flush=True)
    grid = []
    for length in LENGTHS:
        heard = standardise(lib.sound(clap, length))
        r = evaluate_scores(lib, events, labels, heard)
        grid.append({"length": length, "weight": None, "p_at_k": r["p_at_k"], "map": r["map"]})
        print(f"sound {length:g} s: P@{K} {100 * r['p_at_k']:.1f}  mAP {100 * r['map']:.1f}", flush=True)
        for w in WEIGHTS:
            r = evaluate_scores(lib, events, labels, frames + w * heard)
            grid.append({"length": length, "weight": w, "p_at_k": r["p_at_k"], "map": r["map"]})
            print(f"frames + {w:g} x sound {length:g} s: P@{K} {100 * r['p_at_k']:.1f}  mAP {100 * r['map']:.1f}",
                  flush=True)  # fmt: skip
    best = max((g for g in grid if g["weight"] is not None), key=lambda g: g["p_at_k"])
    res.update(grid=grid, best=best)
    res["frames"].pop("labels")
    (ROOT / "results" / "unav_tuned.json").write_text(json.dumps(res, indent=1))
    print(f"best: frames + {best['weight']:g} x sound in {best['length']:g}-second windows: "
          f"P@{K} {100 * best['p_at_k']:.1f}")  # fmt: skip


def peak_scaled(s: np.ndarray) -> np.ndarray:
    """Per label: 0 at the median frame, 1 at the 99th percentile; 0 where nothing was heard."""
    med, top = np.nanmedian(s, axis=0), np.nanpercentile(s, 99, axis=0)
    return np.nan_to_num((s - med) / np.maximum(top - med, 1e-9), nan=0.0)


def explore(args) -> None:
    """On the tuning videos only: a weight of the sound that depends on the description, two ways, against the
    fixed weight. (a) The fixed weight times how far the sound's 99th percentile stands above the frames' (both
    standardised), within 1/4 and 4. (b) Each modality scaled by its own 99th percentile instead of its spread."""
    z, labels, events, siglip, clap, bank = prepare(args)
    names = z["names"].tolist()
    lib = Library(z, tuning_videos(names))
    best = json.loads((ROOT / "results" / "unav_tuned.json").read_text())["best"]
    raw_f, raw_s = lib.frames(siglip, bank), lib.sound(clap, best["length"])
    frames, heard = standardise(raw_f), standardise(raw_s)
    rows = [
        {"variant": f"fixed weight {best['weight']:g}", **score(lib, events, labels, frames + best["weight"] * heard)}
    ]
    lift = np.clip(np.percentile(heard, 99, axis=0) / np.maximum(np.percentile(frames, 99, axis=0), 1e-9), 0.25, 4)
    pf, ps = peak_scaled(raw_f), peak_scaled(raw_s)
    for w in (0.5, 1.0, 2.0, 3.0):
        rows.append(
            {"variant": f"(a) {w:g} x the sound's lift", **score(lib, events, labels, frames + w * lift * heard)}
        )
        rows.append({"variant": f"(b) peaks, sound weight {w:g}", **score(lib, events, labels, pf + w * ps)})
    for r in rows:
        print(f"{r['variant']:32s} P@{K} {100 * r['p_at_k']:.1f}  mAP {100 * r['map']:.1f}", flush=True)
    (ROOT / "results" / "unav_tuned_explore.json").write_text(json.dumps({"videos": len(lib.names), "rows": rows},
                                                                          indent=1))  # fmt: skip


def score(lib, events, labels, s) -> dict:
    r = evaluate_scores(lib, events, labels, s)
    return {"p_at_k": r["p_at_k"], "map": r["map"]}


def evaluate(args) -> None:
    z, labels, events, siglip, clap, bank = prepare(args)
    names = z["names"].tolist()
    lib = Library(z, set(names) - tuning_videos(names))
    best = json.loads((ROOT / "results" / "unav_tuned.json").read_text())["best"]
    frames = standardise(lib.frames(siglip, bank))
    heard = standardise(lib.sound(clap, best["length"]))
    res = {"videos": len(lib.names), "length": best["length"], "weight": best["weight"]}
    for name, s in (("frames", frames), ("sound", heard), ("both", frames + best["weight"] * heard)):
        res[name] = evaluate_scores(lib, events, labels, s)
        print(f"{name}: P@{K} {100 * res[name]['p_at_k']:.1f}  mAP {100 * res[name]['map']:.1f}", flush=True)
    res["labels"] = len(res["frames"]["labels"])
    res["both_vs_frames"] = compare(res["both"], res["frames"])
    hours = float(z["duration"].sum() / 3600)
    res["cost"] = {"hours": hours, **{f"{k}_min_per_hour": float(z[k].sum()) / 60 / hours
                                      for k in ("decode_cpu", "encode_cpu", "sound_cpu",
                                                *(f"clap_cpu_{length:g}" for length in LENGTHS))}}  # fmt: skip
    out = ROOT / "results" / "unav_test.json"
    out.write_text(json.dumps(res, indent=1))
    w = res["both_vs_frames"]
    print(f"both vs frames: {w['up']} labels up, {w['down']} down (p = {w['p']:.2g}); wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "tune", "explore", "evaluate"])
    ap.add_argument("--annotations", required=True, help="unav100_all.json, from the same dataset repository")
    ap.add_argument("--model-dir", help="folder of SigLIP 2 base's files (default: the usual download cache)")
    ap.add_argument("--clap-dir", help="folder of CLAP's files (default: the usual download cache)")
    ap.add_argument("--work", help="where each video is written while it is embedded (default: the system's)")
    args = ap.parse_args()
    {"extract": extract, "tune": tune, "explore": explore, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
