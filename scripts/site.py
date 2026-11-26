"""Build the data of the demo site (site/): someone's phone library, videos and photos that their authors shared on
Flickr under CC BY 2.0 (YFCC100M), indexed once as `whenframe index` indexes a new library, searched in the browser.

    python scripts/site_media.py site/media --originals DIR  # site/library.json's files, from YFCC100M's copy
    python scripts/site.py --media DIR                       # site/data/ and site/credits.html, from the originals
    python scripts/site.py --preview                         # the README's animation, docs/figures/site(-dark).webp

site/library.json lists the files: their names as a phone names them (VID_20140425_181116.mp4), their author, title,
page and licence on Flickr, and the hash that finds them in YFCC100M's copy. They were chosen by hand among the media
shared under CC BY 2.0 that are not DiDeMo's (a benchmark): animals, places, events seen from the crowd, sport, food,
nothing centred on a child or private. Writes site/data/:
    frames.MODEL.bin   each model's frame embeddings, float16, as the tool stores them
    offsets.MODEL.bin  each frame's query-bank offset times its strength, float32
    frames.json        when each frame is, which file it belongs to, the files, the models and how they are combined
    sheets/N.webp      the frames' thumbnails, 10 x 10 per sheet, in frame order (a photo is one frame)
    queries.MODEL.bin  the suggested searches and the films' sentences, embedded here by each model (float32), and
                       queries.json their text, and the English searched for those written in French
    film.json          the films of the page's opening (STORIES), and how `whenframe film` cuts a shot
    expected.json      what the tool itself finds for each suggestion and plans for each film, which the page is
                       checked against (scripts/site_check.mjs)
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whenframe import Library, Searcher, film, temporal  # noqa: E402
from whenframe.encoder import DEFAULT_LIBRARY, FUSED, MODELS  # noqa: E402
from whenframe.library import Settings  # noqa: E402

TILE = (160, 120)
COLS, ROWS = 10, 10
# chosen among others for what they find (scripts/site.py prints the tool's first answers to each)
SUGGESTIONS = [
    "a geyser shooting steam into the sky", "a pipe band marching down the street", "snow falling at night",
    "two cats cuddling", "a dog running in a field", "a boat in the fog", "cooking with big flames",
    "a bee on an orchid", "a clownfish", "an elevated subway train", "people dancing in a bar",
    "a vintage car in a parade", "the pyramids", "a striped lighthouse", "a leopard",
    "hot chocolate with marshmallows", "fireworks", "a castle on a hill by the sea",
]  # fmt: skip
# the films of the page's opening, one shot per sentence as `whenframe film` plans them; the third one written in
# French, translated on the computer as the tool translates it (chosen for what they find, by looking at the shots)
STORIES = [
    ("A summer of festivals", ["a pipe band marching down the street", "a vintage car in a parade",
                               "cooking with big flames", "people dancing in a bar", "a dragon boat race",
                               "fireworks"]),
    ("Animals", ["two cats cuddling", "a dog running in a field", "a horse galloping", "a monkey", "a clownfish",
                 "a bee on an orchid"]),
    ("Au grand air", ["un geyser en éruption", "la neige tombe sur la forêt", "un bateau dans le brouillard",
                      "des milliers d'oies sur un lac", "une cascade", "le soleil se couche"]),
]  # fmt: skip


def frames_at_one_fps(path: Path) -> np.ndarray:
    w, h = TILE
    vf = f"fps=1,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt",
           "rgb24", "-"]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def photo_tile(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail(TILE, Image.LANCZOS)
    tile = Image.new("RGB", TILE)
    tile.paste(im, ((TILE[0] - im.width) // 2, (TILE[1] - im.height) // 2))
    return np.asarray(tile)


def credits_page(items: list[dict], out: Path) -> None:
    """site/credits.html: every file, its title and author on Flickr, and its licence."""
    rows = "\n".join(
        f'<tr><td>{html.escape(it["file"])}</td><td><a href="{html.escape(it["page"])}">'
        f"{html.escape(it['title'] or 'untitled')}</a></td><td>{html.escape(it['author'])}</td>"
        f'<td><a href="{html.escape(it["license_url"])}">CC BY 2.0</a></td></tr>'
        for it in items
    )
    out.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>whenframe demo credits</title>
<link rel="icon" href="logo.svg" type="image/svg+xml">
<style>
  :root {{ --page: #f6f6f3; --ink: #0b0b0b; --ink2: #52514e; --line: #e1e0d9; --accent: #2a78d6;
           color-scheme: light; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --page: #121211; --ink: #f0efec; --ink2: #c3c2b7; --line: #2c2c2a; --accent: #3987e5;
             color-scheme: dark; }}
  }}
  body {{ margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 32px 16px 48px; }}
  a {{ color: var(--accent); }}
  p {{ color: var(--ink2); }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  td, th {{ border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }}
  td:first-child {{ font-family: ui-monospace, monospace; font-size: 13px; white-space: nowrap; }}
</style>
</head>
<body>
<div class="wrap">
<h1>The demo's videos and photos</h1>
<p>{len(items)} files that {len({it["author"] for it in items})} people shared on Flickr under the
<a href="https://creativecommons.org/licenses/by/2.0/">Creative Commons Attribution 2.0</a> licence, taken from
YFCC100M's public copy. The demo renames them as a phone names its files, shows the videos in smaller copies without
their sound and the photos at 500 pixels, and embeds them with whenframe; nothing else is changed.
<a href="./">Back to the demo</a>.</p>
<div class="scroll"><table>
<tr><th>file in the demo</th><th>title on Flickr</th><th>author</th><th>licence</th></tr>
{rows}
</table></div>
</div>
</body>
</html>
""")


def preview(out: Path, scheme: str) -> None:
    """Serve site/, drive it with scripts/site_preview.mjs (Playwright), and make an animated WebP of it."""
    import functools
    import os
    import re
    import tempfile
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from PIL import Image

    class Quiet(SimpleHTTPRequestHandler):
        """Quiet, and answering byte ranges as GitHub Pages does, so that the videos can be seeked."""

        def log_message(self, *args) -> None:
            pass

        def send_head(self):
            path = self.translate_path(self.path.split("?")[0])
            ranged = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
            if not ranged or not os.path.isfile(path):
                return super().send_head()
            size = os.path.getsize(path)
            start, end = int(ranged[1]), int(ranged[2] or size - 1)
            f = open(path, "rb")  # noqa: SIM115 (closed by the server once copied)
            f.seek(start)
            self.send_response(206)
            for k, v in (("Content-Type", self.guess_type(path)), ("Content-Range", f"bytes {start}-{end}/{size}"),
                         ("Content-Length", str(end - start + 1)), ("Accept-Ranges", "bytes")):  # fmt: skip
                self.send_header(k, v)
            self.end_headers()
            return f

        def copyfile(self, source, outputfile) -> None:
            try:
                super().copyfile(source, outputfile)
            except (BrokenPipeError, ConnectionResetError):  # the browser seeked elsewhere
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT / "site")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            run = subprocess.run(["node", str(ROOT / "scripts" / "site_preview.mjs"), f"http://127.0.0.1:{httpd.server_port}",
                                  tmp, scheme], capture_output=True, text=True, check=True)  # fmt: skip
        finally:
            httpd.shutdown()
        # captured at twice the size, kept at 1200 pixels wide: sharp at the README's 800; each frame stays as long
        # as it took to take the next one, so that the film plays at its own pace
        lines = [line.rsplit(" ", 1) for line in run.stdout.splitlines() if line.strip()]
        frames = [Image.open(file).convert("RGB") for file, _ in lines]
        size = (1200, round(frames[0].height * 1200 / frames[0].width))
        frames = [f.resize(size, Image.LANCZOS) for f in frames]
        frames[0].save(out, "WEBP", save_all=True, append_images=frames[1:], duration=[int(ms) for _, ms in lines],
                       loop=0, quality=55, method=6)  # fmt: skip
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


def summary() -> dict:
    """The page's numbers, from the benchmark's results like the docs' (scripts/tables.py, readme_compare):
    whenframe's on DiDeMo's test and on Charades-STA, one image per video's, and what others report."""
    get = {
        n: json.loads((ROOT / "results" / f"{n}.json").read_text())
        for n in ("round11_test", "published_didemo", "published_charades", "didemo_test_thumbnail", "didemo_test")
    }
    r, video = get["round11_test"], get["round11_test"]["fusion"]["didemo"]["full"]["video"]
    name, best = max(get["published_didemo"]["video_retrieval"].items(), key=lambda kv: kv[1]["R@5"])
    others = sorted(get["published_charades"]["zero_shot"], key=lambda x: -x["R1@0.5"])[:2]
    return {
        "video": video["R@5"], "video_r1": video["R@1"],
        "moment": r["fusion"]["didemo"]["value"], "one_image": get["didemo_test_thumbnail"]["middle"]["R@5"],
        "published": {"name": name, "R@5": best["R@5"] / 100},
        "keyframe": get["didemo_test"]["scene"]["corpus"]["R@5 IoU>=0.5"],  # one keyframe per shot, the moments
        "charades": r["fusion-no-charades"]["charades"]["value"],
        "charades_others": [{"name": x["short"], "R1": x["R1@0.5"] / 100} for x in others],
    }  # fmt: skip


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="only make the animation of the site")
    ap.add_argument("--media", help="the folder of site/library.json's files (scripts/site_media.sh)")
    ap.add_argument("--library", default=str(ROOT / "results" / "cache" / "site-library"))
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    a = ap.parse_args()
    if a.preview:
        preview(ROOT / "docs" / "figures" / "site.webp", "light")
        preview(ROOT / "docs" / "figures" / "site-dark.webp", "dark")
        return
    if not a.media:
        ap.error("--media is required to build the data")
    out, media = Path(a.out), Path(a.media)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    items = json.loads((ROOT / "site" / "library.json").read_text())
    credits_page(items, ROOT / "site" / "credits.html")

    # a new library, as `whenframe index` makes it: both encoders, each through its own temporal model
    lib = Library(a.library, Settings(temporal=temporal.for_library(DEFAULT_LIBRARY)))
    for it in items:
        lib.add(media / it["file"])
    videos, embs, times = lib.snapshot()
    by_file = {Path(v.path).name: v for v in videos}
    order = [by_file[it["file"]] for it in items]
    rows = np.concatenate([np.arange(v.start, v.start + v.count) for v in order])
    searcher = Searcher(lib)
    models = lib.settings.models
    fused = FUSED[lib.settings.model]

    thumbs, item_of = [], []
    for i, (it, v) in enumerate(zip(items, order, strict=True)):
        if it["kind"] == "photo":
            thumbs.append(photo_tile(media / it["file"]))
        else:
            frames = frames_at_one_fps(media / it["file"])
            thumbs += [frames[min(int(round(float(t))), len(frames) - 1)] for t in times[v.start : v.start + v.count]]
        item_of += [i] * v.count
    per_sheet = COLS * ROWS
    for s in range(0, len(thumbs), per_sheet):
        sheet = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]))
        for k, img in enumerate(thumbs[s : s + per_sheet]):
            sheet.paste(Image.fromarray(img), ((k % COLS) * TILE[0], (k // COLS) * TILE[1]))
        sheet.save(out / "sheets" / f"{s // per_sheet}.webp", "WEBP", quality=72, method=6)

    # the suggestions, then the films' sentences; those written in French are embedded as the tool searches them
    texts = list(dict.fromkeys(SUGGESTIONS + [line for _, lines in STORIES for line in lines]))
    searched = [searcher.searched_as(t) for t in texts]
    spec = []
    for model, weight in zip(models, fused.weights, strict=True):
        embs[model][rows].astype("<f2").tofile(out / f"frames.{model}.bin")
        strength = searcher.config.bank * searcher._tuned(model)[1]
        (searcher.offsets(model) * strength)[rows].astype("<f4").tofile(out / f"offsets.{model}.bin")
        # one at a time, as the tool embeds a search: a batch's padding changes the last digits, enough to turn a tie
        queries = np.concatenate([lib.encoder_for(model).encode_texts([t]) for t in searched]).astype("<f4")
        queries.tofile(out / f"queries.{model}.bin")
        spec.append({"name": model, "dim": int(embs[model].shape[1]), "weight": weight})
    first = MODELS[models[0]]
    meta = {
        "models": spec, "temporal": lib.settings.temporal, "count": int(len(rows)),
        "scale": first.logit_scale, "bias": first.logit_bias,  # the first model's, for how sure a match is
        "window": searcher.config.window, "per_video": searcher.config.per_video, "moments": searcher.moments,
        "tile": list(TILE), "cols": COLS, "per_sheet": per_sheet,
        "times": [round(float(t), 2) for t in times[rows]], "item": item_of,
        "items": [{"file": it["file"], "kind": it["kind"], "duration": round(v.duration, 2), "taken": it["taken"],
                   "title": it["title"], "author": it["author"], "page": it["page"]}
                  for it, v in zip(items, order, strict=True)],
    }  # fmt: skip
    (out / "frames.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
    (out / "queries.json").write_text(json.dumps({"suggestions": len(SUGGESTIONS), "texts": texts,
                                                  "searched_as": searched}, indent=1, ensure_ascii=False))  # fmt: skip
    (out / "film.json").write_text(json.dumps({
        "stories": [{"title": title, "lines": lines} for title, lines in STORIES],
        "seconds": film.SECONDS, "fade": film.FADE, "shortest": film.SHORTEST, "candidates": 30,
    }, indent=1, ensure_ascii=False))  # fmt: skip
    # what the tool finds, for the check that the page searches as it does (scripts/site_check.mjs)
    expected = {"searches": {}, "films": {}}
    for q in SUGGESTIONS:
        hits = searcher.search(q, k=10)
        expected["searches"][q] = [{"file": Path(h.path).name, "start": round(h.start, 2), "end": round(h.end, 2),
                                    "confidence": round(h.confidence, 4)} for h in hits]  # fmt: skip
    for title, lines in STORIES:
        shots, _ = film.plan_film(searcher, lines)
        expected["films"][title] = [
            {"file": Path(s.path).name, "start": round(s.start, 2), "length": round(s.length, 2)} for s in shots
        ]
    (out / "expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False))
    for old in ("frames.bin", "offsets.bin", "queries.bin", "summary.json"):  # the three films' data, before
        (out / old).unlink(missing_ok=True)
    (out / "summary.json").write_text(json.dumps(summary()))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    n_video = sum(it["kind"] == "video" for it in items)
    print(f"{len(rows)} frames from {n_video} videos and {len(items) - n_video} photos, "
          f"{len(thumbs) // per_sheet + 1} sheets, {size:.1f} MB")  # fmt: skip
    for q, hits in expected["searches"].items():
        print(f"  {q}: " + ", ".join(f"{h['file']} {h['start']:.0f}s {h['confidence']:.2f}" for h in hits[:3]))
    for title, shots in expected["films"].items():
        print(f"  {title}: " + ", ".join(f"{s['file']} {s['start']:.0f}s" for s in shots))


if __name__ == "__main__":
    main()
