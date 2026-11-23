"""The demo site's text server: a visitor's own description, embedded by the text towers whenframe searches with.

The demo site (site/) searches a phone's videos and photos in the browser, on frames embedded beforehand. A
description typed there is sent here, embedded by SigLIP 2's and PE core B/16's text towers exactly as
`whenframe search` embeds a query, and sent back; the browser searches with it. Only the text comes here, and nothing
is logged or stored.

    python demo-server/app.py [--host 0.0.0.0] [--port 7860] [--origin https://qthullie.github.io ...]

    GET /embed?q=TEXT   {"vectors": {model: base64 of its unit vector, little-endian float32}, "dims": {model: ...},
                         "ms": ...}
    GET /health         ok, once the text towers are loaded

Only the origins given (or WHENFRAME_ORIGINS, comma-separated) may read the answers from a browser. A description
longer than LIMIT characters is refused, each address may ask RATE times a minute, and one description is
embedded at a time.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

LIMIT = 200  # characters
RATE = 30  # descriptions a minute, per address
ORIGINS = ("https://qthullie.github.io",)


class Limiter:
    """At most `rate` requests a minute per key, counted over the last 60 seconds."""

    def __init__(self, rate: int = RATE, clock=time.monotonic):
        self.rate, self.clock = rate, clock
        self.seen: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            q = self.seen[key]
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= self.rate:
                return False
            q.append(now)
            if len(self.seen) > 10_000:  # forget addresses that have gone quiet
                for k in [k for k, v in self.seen.items() if not v or now - v[-1] >= 60.0]:
                    del self.seen[k]
            return True


def handler(embed, origins=ORIGINS, limit: int = LIMIT, limiter: Limiter | None = None):
    """The request handler; embed(text) -> {model: unit vector (float32)}, called one at a time."""
    origins = set(origins)
    limiter = limiter or Limiter()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "whenframe-demo"

        def log_message(self, *args) -> None:  # quiet: descriptions are private
            pass

        def send(self, status: int, body: dict | str, extra: dict | None = None) -> None:
            data = (json.dumps(body) if isinstance(body, dict) else body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json" if isinstance(body, dict) else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            origin = self.headers.get("Origin")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/health":
                return self.send(200, "ok")
            if url.path != "/embed":
                return self.send(404, {"error": "not found"})
            text = parse_qs(url.query).get("q", [""])[0].strip()
            if not text:
                return self.send(400, {"error": "no description"})
            if len(text) > limit:
                return self.send(400, {"error": f"descriptions are limited to {limit} characters"})
            # behind a proxy (a Space, a load balancer), the visitor's address is the first one it forwarded
            who = self.headers.get("X-Forwarded-For", "").split(",")[0].strip() or self.client_address[0]
            if not limiter.allow(who):
                return self.send(429, {"error": "too many descriptions; try again in a minute"}, {"Retry-After": "60"})
            t0 = time.perf_counter()
            with lock:
                vs = {m: np.asarray(v, dtype="<f4").ravel() for m, v in embed(text).items()}
            ms = round(1000 * (time.perf_counter() - t0))
            self.send(
                200,
                {
                    "vectors": {m: base64.b64encode(v.tobytes()).decode() for m, v in vs.items()},
                    "dims": {m: int(v.size) for m, v in vs.items()},
                    "ms": ms,
                },
            )

    return Handler


def text_encoders() -> dict:
    """The text towers of a new library's models, downloaded once (not their image towers)."""
    from whenframe.encoder import DEFAULT_LIBRARY, MODELS, Encoder, cache_dir, download, fetch_files

    out = {}
    for model in DEFAULT_LIBRARY.split("+"):
        spec = MODELS[model]
        if spec.sources:  # files pinned one by one
            folder = cache_dir() / "models" / spec.repo.replace("/", "--")
            for name, url, digest in spec.sources:
                if name in (spec.tokenizer_file, spec.text_file):
                    download(url, folder / name, digest)
        else:
            folder = fetch_files(spec.repo, (spec.tokenizer_file, spec.text_file))
        out[model] = Encoder(model, folder=folder, backend="onnxruntime")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--origin", action="append", help="a site allowed to read the answers (repeatable)")
    a = ap.parse_args()
    origins = a.origin or [o for o in os.environ.get("WHENFRAME_ORIGINS", "").split(",") if o] or list(ORIGINS)

    encoders = text_encoders()
    for e in encoders.values():
        e.encode_texts(["warm up"])  # loaded before the first visitor
    httpd = ThreadingHTTPServer(
        (a.host, a.port), handler(lambda t: {m: e.encode_texts([t])[0] for m, e in encoders.items()}, origins)
    )
    print(f"embedding descriptions for {', '.join(origins)} at http://{a.host}:{a.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
