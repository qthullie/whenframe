"""Photos in a library of videos: read upright, indexed as one frame, ranked with the moments of videos."""

import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from test_library import COLORS, ToyEncoder, make_video

from whenframe import Library, SearchConfig, Searcher, Settings
from whenframe.server import App, handler
from whenframe.video import find_media, read_photo

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_photo(path, colour):
    r, g, b = COLORS[colour]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{r:02x}{g:02x}{b:02x}:s=64x48",
           "-frames:v", "1", str(path)]  # fmt: skip
    subprocess.run(cmd, check=True)


@pytest.fixture
def media(tmp_path):
    make_video(str(tmp_path / "a.mp4"), [("grey", 6), ("green", 4)])
    make_photo(tmp_path / "red.jpg", "red")
    make_photo(tmp_path / "blue.png", "blue")
    return tmp_path


def test_read_photo(media):
    frame = read_photo(media / "red.jpg", 32)
    assert frame.shape == (1, 32, 32, 3)
    assert frame[0, ..., 0].mean() > 200 and frame[0, ..., 2].mean() < 60


def test_a_photo_turned_by_exif_is_read_upright(tmp_path):
    image = pytest.importorskip("PIL.Image")
    img = image.new("RGB", (80, 40), (255, 0, 0))
    img.paste((0, 0, 255), (0, 0, 40, 40))  # the left half blue
    exif = image.Exif()
    exif[0x0112] = 6  # shown turned 90 degrees clockwise: the blue half on top
    img.save(tmp_path / "turned.jpg", exif=exif.tobytes())
    frame = read_photo(tmp_path / "turned.jpg", 32)[0].astype(int)
    assert frame[:12, :, 2].mean() > 200 and frame[-12:, :, 0].mean() > 200


def test_photos_are_found_unless_turned_off(media):
    assert sorted(p.name for p in find_media([media])) == ["a.mp4", "blue.png", "red.jpg"]
    assert [p.name for p in find_media([media], photos=False)] == ["a.mp4"]
    assert [p.name for p in find_media([media / "red.jpg"], photos=False)] == []


def test_photos_and_videos_are_searched_together(media):
    lib = Library(media / "lib", Settings(model="toy"), encoder=ToyEncoder())
    for p in find_media([media]):
        lib.add(p)
    assert lib.stats()["videos"] == 1 and lib.stats()["photos"] == 2
    search = Searcher(lib, SearchConfig(window=4.0))
    _, embs, _ = lib.snapshot()
    for colour, name in (("red", "red.jpg"), ("green", "a.mp4"), ("blue", "blue.png")):
        q = ToyEncoder().encode_texts([colour])[0]
        top = search.search(colour, k=3, scores={"toy": embs["toy"] @ q})[0]
        assert top.path.endswith(name) and top.photo == (name != "a.mp4")
        if top.photo:
            assert (top.start, top.end, top.at) == (0.0, 0.0, 0.0)


def test_serve_shows_a_photo(media):
    lib = Library(media / "lib", Settings(model="toy"), encoder=ToyEncoder())
    for p in find_media([media]):
        lib.add(p)
    app = App(lib, SearchConfig(window=4.0))
    q = ToyEncoder().encode_texts(["red"])[0]
    lib_scores = {"toy": lib.snapshot()[1]["toy"] @ q}
    app.searcher.frame_scores = lambda query: lib_scores
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/search?q=red&k=3", timeout=30) as r:
            hit = json.load(r)["hits"][0]
        assert hit["photo"] is True and hit["name"] == "red.jpg"
        with urllib.request.urlopen(f"{base}/video?id={hit['video']}", timeout=30) as r:
            assert r.headers["Content-Type"] == "image/jpeg" and r.read()[:2] == b"\xff\xd8"
        with urllib.request.urlopen(f"{base}/thumb?video={hit['video']}&t=0", timeout=30) as r:
            assert r.read()[:2] == b"\xff\xd8"
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert np.isclose(hit["start"], 0.0) and np.isclose(hit["end"], 0.0)
