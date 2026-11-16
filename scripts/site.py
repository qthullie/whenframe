"""Build the data of the demo site (site/): three Blender films, embedded once, searched in the browser.

    python scripts/site.py --films DIR      # Big Buck Bunny, Sintel, Tears of Steel (Blender Foundation, CC BY 3.0)
    python scripts/site.py --preview        # then the README's animation of the site, docs/figures/site.webp

Writes site/data/:
    frames.bin       the frame embeddings of the default model, float16, as the tool stores them
    offsets.bin      each frame's query-bank offset times the tuned strength, float32
    frames.json      when each frame is, which film it belongs to, the films, the model's calibration
    sheets/N.webp    the frames' thumbnails, 10 x 10 per sheet, in frame order
    queries.bin      the suggested searches, embedded here (float32), and queries.json their text

The page embeds any other search in the browser with the same text model (transformers.js).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whenframe import Library, Searcher  # noqa: E402
from whenframe.encoder import MODELS  # noqa: E402
from whenframe.video import find_videos  # noqa: E402

TILE = (192, 108)
COLS, ROWS = 10, 10
FILMS = {
    "BigBuckBunny_640x360.m4v": ("Big Buck Bunny", "https://peach.blender.org/"),
    "Sintel.2010.720p.mkv": ("Sintel", "https://durian.blender.org/"),
    "tears_of_steel_720p.mov": ("Tears of Steel", "https://mango.blender.org/"),
}
SUGGESTIONS = [
    "two people arguing on a bridge", "a girl hugging a baby dragon", "a giant rabbit wakes up and stretches",
    "a sword fight in the snow", "a robot with glowing eyes", "a butterfly flying in a meadow",
    "a dragon flying over the mountains", "an old man in a dark room", "people running away from an explosion",
    "a close-up of a woman crying", "a squirrel holding a nut", "a city at night",
]  # fmt: skip


def frames_at_one_fps(path: Path) -> np.ndarray:
    w, h = TILE
    vf = f"fps=1,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt",
           "rgb24", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def preview(out: Path) -> None:
    """Serve site/, drive it with scripts/site_preview.mjs (Playwright), and make an animated WebP of it."""
    import functools
    import tempfile
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    serve = functools.partial(SimpleHTTPRequestHandler, directory=str(ROOT / "site"))
    serve.log_message = lambda *a: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run = subprocess.run(["node", str(ROOT / "scripts" / "site_preview.mjs"), f"http://127.0.0.1:{httpd.server_port}",
                                  tmp], capture_output=True, text=True, check=True)  # fmt: skip
        finally:
            httpd.shutdown()
        frames = [Image.open(line).convert("RGB") for line in run.stdout.split()]
        frames[0].save(out, "WEBP", save_all=True, append_images=frames[1:], duration=140, loop=0, quality=60,
                       method=6)  # fmt: skip
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="only make the animation of the site")
    ap.add_argument("--films")
    ap.add_argument("--library", default=str(ROOT / "results" / "cache" / "demo-library"))
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    a = ap.parse_args()
    if a.preview:
        return preview(ROOT / "docs" / "figures" / "site.webp")
    if not a.films:
        ap.error("--films is required to build the data")
    out = Path(a.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    lib = Library(a.library)
    for path in find_videos([a.films]):
        lib.add(path)
    model = lib.settings.models[0]
    videos, embs, times = lib.snapshot()
    searcher = Searcher(lib)
    offsets = searcher.offsets(model) * searcher.config.bank * MODELS[model].bank_lambda

    rows = np.concatenate([np.arange(v.start, v.start + v.count) for v in videos])
    thumbs, films, film_of = [], [], []
    for i, v in enumerate(videos):
        name = Path(v.path).name
        title, url = FILMS.get(name, (Path(v.path).stem, ""))
        films.append({"title": title, "file": name, "duration": round(v.duration, 2), "url": url})
        frames = frames_at_one_fps(Path(v.path))
        for t in times[v.start : v.start + v.count]:
            thumbs.append(frames[min(int(round(float(t))), len(frames) - 1)])
            film_of.append(i)
    per_sheet = COLS * ROWS
    for s in range(0, len(thumbs), per_sheet):
        sheet = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]))
        for k, img in enumerate(thumbs[s : s + per_sheet]):
            sheet.paste(Image.fromarray(img), ((k % COLS) * TILE[0], (k // COLS) * TILE[1]))
        sheet.save(out / "sheets" / f"{s // per_sheet}.webp", "WEBP", quality=72, method=6)

    embs[model][rows].astype("<f2").tofile(out / "frames.bin")
    offsets[rows].astype("<f4").tofile(out / "offsets.bin")
    spec = MODELS[model]
    meta = {
        "model": spec.repo, "dim": int(embs[model].shape[1]), "count": int(len(rows)),
        "scale": spec.logit_scale, "bias": spec.logit_bias, "text_length": spec.text_length,
        "window": searcher.config.window, "per_video": searcher.config.per_video,
        "tile": list(TILE), "cols": COLS, "per_sheet": per_sheet,
        "times": [round(float(t), 2) for t in times[rows]], "film": film_of, "films": films,
    }  # fmt: skip
    (out / "frames.json").write_text(json.dumps(meta, separators=(",", ":")))
    results = ROOT / "results" / "didemo_test.json"
    if results.exists():  # the page's numbers come from the benchmark's results, like the docs'
        r = json.loads(results.read_text())
        summary = {"default": r[r["default"]]["corpus"]["R@5 IoU>=0.5"], "shots": r["scene"]["corpus"]["R@5 IoU>=0.5"],
                   "videos": r["cost"]["videos"], "queries": r["queries"]}  # fmt: skip
        (out / "summary.json").write_text(json.dumps(summary))
    q = lib.encoder.encode_texts(SUGGESTIONS).astype("<f4")
    q.tofile(out / "queries.bin")
    (out / "queries.json").write_text(json.dumps(SUGGESTIONS, indent=1))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{len(rows)} frames from {len(films)} films, {len(thumbs) // per_sheet + 1} sheets, {size:.1f} MB")


if __name__ == "__main__":
    main()
