"""`whenframe serve` on a toy library: search, thumbnails, byte ranges, and what it refuses."""

import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from test_library import ToyEncoder, make_video

from whenframe import Library, Settings
from whenframe.search import SearchConfig
from whenframe.server import App, handler

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.fixture
def server(tmp_path):
    make_video(str(tmp_path / "a.mp4"), [("grey", 6), ("red", 3), ("grey", 6)])
    make_video(str(tmp_path / "b.mp4"), [("green", 4), ("grey", 8)])
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder())
    for name in ("a", "b"):
        lib.add(tmp_path / f"{name}.mp4")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(App(lib, SearchConfig(window=4.0))))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", lib
    httpd.shutdown()
    httpd.server_close()


def get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
        return r.status, dict(r.headers), r.read()


def test_the_page_and_a_search(server):
    base, _ = server
    status, headers, page = get(base + "/")
    assert status == 200 and b"whenframe" in page and headers["Content-Type"].startswith("text/html")
    _, _, body = get(base + "/api/search?q=red&k=3")
    hits = json.loads(body)["hits"]
    assert hits[0]["name"] == "a.mp4" and hits[0]["start"] <= 6.5 <= hits[0]["end"]


def test_thumbnails_and_byte_ranges(server):
    base, lib = server
    video = lib.videos()[0]
    status, headers, jpeg = get(f"{base}/thumb?video={video.id}&t=7")
    assert status == 200 and jpeg[:2] == b"\xff\xd8" and headers["Content-Type"] == "image/jpeg"
    data = open(video.path, "rb").read()
    status, headers, part = get(f"{base}/video?id={video.id}", {"Range": "bytes=10-109"})
    assert status == 206 and part == data[10:110] and headers["Content-Range"] == f"bytes 10-109/{len(data)}"


def test_unknown_videos_and_bad_requests_are_refused(server):
    base, _ = server
    for path, code in (("/video?id=999", 404), ("/video?id=../../etc/passwd", 400), ("/thumb?video=1", 400),
                       ("/files/a.mp4", 404)):  # fmt: skip
        with pytest.raises(urllib.error.HTTPError) as e:
            get(base + path)
        assert e.value.code == code
