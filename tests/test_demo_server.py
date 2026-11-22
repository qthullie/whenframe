"""The demo site's text server (demo-server/app.py), with a stand-in for the text tower."""

import base64
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("demo_app", Path(__file__).parents[1] / "demo-server" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

SITE = "https://qthullie.github.io"


def fake_embed(text: str) -> np.ndarray:
    v = np.arange(1, 769, dtype=np.float32) * (len(text) % 7 + 1)
    return v / np.linalg.norm(v)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def server():
    clock = Clock()
    limiter = app.Limiter(rate=3, clock=clock)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.handler(fake_embed, [SITE], limit=50, limiter=limiter))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", clock
    httpd.shutdown()
    httpd.server_close()


def get(url, origin=SITE):
    req = urllib.request.Request(url, headers={"Origin": origin} if origin else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_a_description_comes_back_as_its_unit_vector(server):
    base, _ = server
    status, headers, body = get(f"{base}/embed?q=a%20dragon%20over%20the%20mountains")
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == SITE
    assert headers["Cache-Control"] == "no-store"
    answer = json.loads(body)
    v = np.frombuffer(base64.b64decode(answer["vector"]), dtype="<f4")
    assert answer["dim"] == 768 and len(v) == 768
    np.testing.assert_allclose(v, fake_embed("a dragon over the mountains"), rtol=1e-6)


def test_only_the_site_may_read_the_answers(server):
    base, _ = server
    status, headers, _ = get(f"{base}/embed?q=hello", origin="https://elsewhere.example")
    assert status == 200 and "Access-Control-Allow-Origin" not in headers


def test_what_it_refuses(server):
    base, _ = server
    assert get(f"{base}/embed?q=%20%20")[0] == 400
    assert get(f"{base}/embed?q=" + "a" * 51)[0] == 400
    assert get(f"{base}/files/../../etc/passwd")[0] == 404
    assert get(f"{base}/health")[0] == 200


def test_each_address_is_limited_a_minute(server):
    base, clock = server
    assert [get(f"{base}/embed?q=one")[0] for _ in range(3)] == [200, 200, 200]
    status, headers, _ = get(f"{base}/embed?q=one")
    assert status == 429 and headers["Retry-After"] == "60"
    clock.t = 61.0  # a minute later
    assert get(f"{base}/embed?q=one")[0] == 200
