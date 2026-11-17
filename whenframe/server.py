"""`whenframe serve`: a search page in the browser, with the model loaded once.

Endpoints, all read-only:

    /                      the page
    /api/search?q=..&k=..  moments as JSON
    /api/stats             what the library holds
    /thumb?video=ID&t=S    a JPEG of the frame at S seconds (ffmpeg, cached in memory)
    /video?id=ID           the video file itself, with byte ranges so the browser can seek

Videos are looked up by their id in the library, never by a path taken from the
request, and the server listens on localhost unless told otherwise.
"""

from __future__ import annotations

import json
import mimetypes
import subprocess
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .library import Library
from .search import HEARD, SearchConfig, Searcher, worth_showing
from .video import is_photo

CHUNK = 1 << 20


def thumbnail(path: str, at: float, width: int = 480) -> bytes:
    # no seeking in a photo: ffmpeg drops the only frame of some JPEGs when asked to seek, even to 0
    seek = [] if is_photo(path) else ["-ss", f"{max(at, 0.0):.2f}"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *seek, "-i", path, "-frames:v", "1",
           "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "-"]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, timeout=30).stdout


class App:
    def __init__(self, library: Library, config: SearchConfig | None = None, links: list | None = None):
        self.library = library
        self.links = links or []  # objects with a name and link(path) -> url | None, e.g. jellyfin.Jellyfin
        self.searcher = Searcher(library, config)
        self.lock = threading.Lock()
        self.thumbs: OrderedDict[tuple[int, float], bytes] = OrderedDict()
        self.thumbs_lock = threading.Lock()
        self.page = resources.files("whenframe").joinpath("web/index.html").read_bytes()

    def paths(self) -> dict[int, str]:
        videos, _, _ = self.library.snapshot()
        return {v.id: v.path for v in videos}

    def search(self, query: str, k: int) -> dict:
        t0 = time.perf_counter()
        with self.lock:  # one search at a time: the model and the cached arrays are shared
            hits = self.searcher.search(query, k=k)
        return {
            "query": query,
            "ms": round(1000 * (time.perf_counter() - t0)),
            "hits": [{"video": h.video_id, "name": Path(h.path).name, "folder": str(Path(h.path).parent),
                      "start": h.start, "end": h.end, "at": h.at, "confidence": h.confidence, "said": h.said,
                      "photo": h.photo, "heard": h.heard >= HEARD, "shown": worth_showing(h),
                      "links": [{"name": s.name, "url": u} for s in self.links if (u := s.link(h.path))]}
                     for h in hits],
        }  # fmt: skip

    def thumb(self, video: int, at: float) -> bytes | None:
        key = (video, round(at, 1))
        with self.thumbs_lock:
            if key in self.thumbs:
                return self.thumbs[key]
        path = self.paths().get(video)
        jpeg = thumbnail(path, at) if path else b""
        if jpeg:
            with self.thumbs_lock:
                self.thumbs[key] = jpeg
                while len(self.thumbs) > 512:
                    self.thumbs.popitem(last=False)
        return jpeg or None


def handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "whenframe"

        def log_message(self, *args) -> None:  # quiet: searches are private
            pass

        def send(self, status: int, body: bytes, kind: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if kind.startswith("application/json") else "max-age=3600")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def fail(self, status: HTTPStatus) -> None:
            self.send(status, status.phrase.encode(), "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.send(200, app.page, "text/html; charset=utf-8")
                elif url.path == "/api/search":
                    k = min(max(int(q.get("k", 12)), 1), 50)
                    body = json.dumps(app.search(q.get("q", "").strip(), k)).encode()
                    self.send(200, body, "application/json")
                elif url.path == "/api/stats":
                    self.send(200, json.dumps(app.library.stats()).encode(), "application/json")
                elif url.path == "/thumb":
                    jpeg = app.thumb(int(q["video"]), float(q["t"]))
                    if jpeg:
                        self.send(200, jpeg, "image/jpeg")
                    else:
                        self.fail(HTTPStatus.NOT_FOUND)
                elif url.path == "/video":
                    self.video(int(q["id"]))
                else:
                    self.fail(HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError):
                self.fail(HTTPStatus.BAD_REQUEST)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser moved on

        def video(self, video: int) -> None:
            path = app.paths().get(video)
            if path is None or not Path(path).is_file():
                return self.fail(HTTPStatus.NOT_FOUND)
            size = Path(path).stat().st_size
            kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
            start, end = 0, size - 1
            ranged = self.headers.get("Range", "")
            if ranged.startswith("bytes="):
                first, _, last = ranged[6:].split(",")[0].partition("-")
                start = int(first) if first else max(size - int(last), 0)
                end = int(last) if first and last else size - 1
                if start > end or start >= size:
                    return self.send(416, b"", kind, {"Content-Range": f"bytes */{size}"})
                end = min(end, size - 1)
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(CHUNK, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def serve(library: Library, host: str = "127.0.0.1", port: int = 8765, config: SearchConfig | None = None,
          links: list | None = None) -> None:  # fmt: skip
    app = App(library, config, links)
    app.searcher.search("warm up", k=1)  # loads the text model and the bank offsets before the first visitor
    httpd = ThreadingHTTPServer((host, port), handler(app))
    print(f"whenframe: searching {len(app.paths())} videos at http://{host}:{port} (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
