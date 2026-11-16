"""The pictures of the README: whenframe serve on three Blender open movies, driven by a browser.

    python scripts/demo.py --films DIR      # Big Buck Bunny, Sintel and Tears of Steel, from download.blender.org

The films are (c) Blender Foundation, under CC BY 3.0. The script indexes them
into its own library, serves the search page, runs scripts/demo.mjs (Playwright,
Node) to take a screenshot of each search, and saves them as WebP.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whenframe import Library  # noqa: E402
from whenframe.server import App, handler  # noqa: E402
from whenframe.video import find_videos  # noqa: E402

SEARCHES = ["two people arguing on a bridge", "a girl hugging a baby dragon", "a giant rabbit wakes up and stretches"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", required=True, help="folder with the films")
    ap.add_argument("--library", default=str(ROOT / "results" / "cache" / "demo-library"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    lib = Library(a.library)
    for path in find_videos([a.films]):
        v = lib.add(path)
        print(f"indexed {path.name}" if v else f"already indexed: {path.name}", flush=True)
    app = App(lib)
    app.searcher.search("warm up", k=1)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        subprocess.run(["node", str(ROOT / "scripts" / "demo.mjs"), url, a.out, *SEARCHES], check=True)
    finally:
        httpd.shutdown()
    from PIL import Image  # frames of films compress far better as WebP than as PNG

    for png in sorted(Path(a.out).glob("demo-*.png")):
        Image.open(png).convert("RGB").save(png.with_suffix(".webp"), "WEBP", quality=82, method=6)
        png.unlink()


if __name__ == "__main__":
    main()
