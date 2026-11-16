"""`whenframe watch`: what a pass indexes, waits for, forgets, and leaves alone."""

import os
import shutil
import time

import pytest
from test_library import ToyEncoder, make_video

from whenframe import Library, Settings
from whenframe.watch import Watcher

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.fixture
def setup(tmp_path):
    media = tmp_path / "media"
    (media / "2024").mkdir(parents=True)
    make_video(str(media / "a.mp4"), [("grey", 4), ("red", 3)])
    make_video(str(media / "2024" / "b.mp4"), [("green", 4), ("grey", 3)])
    old = time.time() - 3600
    for f in media.rglob("*.mp4"):
        os.utime(f, (old, old))
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder())
    return media, lib


def names(paths):
    return sorted(p.name for p in paths)


def test_a_pass_indexes_new_videos_once(setup):
    media, lib = setup
    w = Watcher(lib, [media])
    assert names(w.sync().indexed) == ["a.mp4", "b.mp4"]
    second = w.sync()
    assert second.indexed == [] and second.forgotten == []
    assert len(lib.videos()) == 2


def test_a_video_still_being_written_waits(setup):
    media, lib = setup
    make_video(str(media / "c.mp4"), [("blue", 3)])  # modified just now
    w = Watcher(lib, [media], settle=60)
    result = w.sync()
    assert names(result.waiting) == ["c.mp4"] and "c.mp4" not in names(result.indexed)
    old = time.time() - 120
    os.utime(media / "c.mp4", (old, old))
    assert names(w.sync().indexed) == ["c.mp4"]


def test_removed_videos_are_forgotten_and_the_index_compacted(setup):
    media, lib = setup
    w = Watcher(lib, [media])
    w.sync()
    (media / "a.mp4").unlink()
    assert names(w.sync().forgotten) == ["a.mp4"]
    assert [v.path for v in lib.videos()] == [str((media / "2024" / "b.mp4").resolve())]
    assert lib.dead_rows() == 0  # half the rows were dead: compacted


def test_an_emptied_folder_looks_unmounted_and_nothing_is_forgotten(setup):
    media, lib = setup
    w = Watcher(lib, [media])
    w.sync()
    for f in list(media.rglob("*.mp4")):
        f.unlink()
    result = w.sync()
    assert result.forgotten == [] and result.unavailable == [media.resolve()]
    assert len(lib.videos()) == 2
    shutil.rmtree(media)
    assert w.sync().unavailable == [media.resolve()]


def test_a_broken_file_is_retried_only_once_changed(setup):
    media, lib = setup
    bad = media / "broken.mp4"
    bad.write_bytes(b"not a video")
    old = time.time() - 3600
    os.utime(bad, (old, old))
    w = Watcher(lib, [media])
    assert names(w.sync().failed) == ["broken.mp4"]
    assert w.sync().failed == []  # unchanged: not tried again
    bad.write_bytes(b"still not a video, but changed")
    os.utime(bad, (old + 1, old + 1))
    assert names(w.sync().failed) == ["broken.mp4"]
