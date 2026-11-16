"""The indexing and search path end to end, on tiny generated videos and a toy encoder."""

import shutil
import subprocess

import numpy as np
import pytest

from whenframe import Library, SearchConfig, Searcher, Settings
from whenframe.video import RepeatFilter, drop_repeats, iter_frames, sample_frames, thumbnails

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "grey": (128, 128, 128)}


class ToyEncoder:
    """Embeds a frame as its mean colour and a word as the colour it names."""

    name = "toy"
    size = 32
    dim = 3

    def encode_images(self, frames):
        x = frames.reshape(len(frames), -1, 3).mean(axis=1) - 100.0
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def encode_texts(self, texts):
        x = np.array([COLORS[t] for t in texts], dtype=np.float64) - 100.0
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def probability(self, cos):
        return 1 / (1 + np.exp(-10 * (cos - 0.5)))


def make_video(path, segments):
    """segments: [(colour, seconds), ...] -> a 10 fps mp4 of solid colour blocks."""
    parts, inputs = [], []
    for i, (name, secs) in enumerate(segments):
        r, g, b = COLORS[name]
        inputs += ["-f", "lavfi", "-i", f"color=c=0x{r:02x}{g:02x}{b:02x}:s=64x64:r=10:d={secs}"]
        parts.append(f"[{i}:v]")
    filt = "".join(parts) + f"concat=n={len(segments)}:v=1:a=0[v]"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[v]", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


@pytest.fixture
def videos(tmp_path):
    make_video(str(tmp_path / "a.mp4"), [("grey", 6), ("red", 3), ("grey", 6)])
    make_video(str(tmp_path / "b.mp4"), [("green", 4), ("grey", 8)])
    make_video(str(tmp_path / "c.mp4"), [("grey", 5), ("blue", 5)])
    return tmp_path


def test_sampling_and_repeats(videos):
    times, frames = sample_frames(videos / "a.mp4", fps=1.0, size=32)
    assert len(frames) == 15 and times[3] == 3.0
    keep = drop_repeats(frames, threshold=4.0, max_gap=100)
    assert keep.tolist() == [0, 6, 9]  # grey, then red, then grey again


def test_reading_can_stop_early(videos):
    times, frames = sample_frames(videos / "a.mp4", fps=1.0, size=32, seconds=5)
    assert len(frames) == 5 and times[-1] == 4.0


def test_frames_read_in_chunks_give_the_same_frames_and_choices(videos):
    times, frames = sample_frames(videos / "a.mp4", fps=1.0, size=32)
    repeats = RepeatFilter(threshold=4.0, max_gap=2)
    chunks = list(iter_frames(videos / "a.mp4", fps=1.0, size=32, chunk=4))
    assert [len(f) for _, f in chunks] == [4, 4, 4, 3]
    assert np.array_equal(np.concatenate([f for _, f in chunks]), frames)
    kept = np.concatenate([t[repeats(thumbnails(f))] for t, f in chunks])
    assert kept.tolist() == times[drop_repeats(frames, threshold=4.0, max_gap=2)].tolist()


def test_search_finds_the_right_video_and_moment(videos, tmp_path):
    lib = Library(tmp_path / "lib", Settings(model="toy", dedup=4.0), encoder=ToyEncoder())
    for name in ("a", "b", "c"):
        lib.add(videos / f"{name}.mp4")
    spans = SearchConfig(moments="spans", step=0.5, sigma=0.0, alpha=0.5, min_length=1.0)
    top = Searcher(lib, spans).search("red", k=3)[0]
    assert top.path.endswith("a.mp4")
    assert 5.0 <= top.start <= 6.5 and 8.5 <= top.end <= 10.0
    top = Searcher(lib, SearchConfig(window=4.0)).search("red", k=3)[0]
    assert top.path.endswith("a.mp4") and (top.start, top.end) == (4.5, 8.5)  # 4 s around the first red frame
    assert Searcher(lib).search("blue", k=1)[0].path.endswith("c.mp4")


def test_the_query_bank_marks_down_frames_that_match_everything(videos, tmp_path):
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder())
    for name in ("a", "b", "c"):
        lib.add(videos / f"{name}.mp4")
    bank = ToyEncoder().encode_texts(["grey"] * 4)  # descriptions of a grey scene: grey frames are hubs
    plain = Searcher(lib, SearchConfig(bank=0.0), bank=bank)
    corrected = Searcher(lib, SearchConfig(bank=1.0, beta=10.0), bank=bank)
    assert plain.offsets() is None
    off = corrected.offsets()
    _, _, times = lib.snapshot()
    assert len(off) == len(times)
    a = lib.videos()[0]
    t = times[a.start : a.start + a.count]
    grey, red = off[a.start + np.flatnonzero(t == 0)[0]], off[a.start + np.flatnonzero(t == 6)[0]]
    assert grey > red  # a.mp4 is grey, then red from 6 s
    assert np.array_equal(lib.offsets(bank, 10.0), off)  # kept on disk, not recomputed


def test_unchanged_videos_are_not_indexed_twice_and_compaction_keeps_results(videos, tmp_path):
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder())
    assert lib.add(videos / "a.mp4") is not None
    assert lib.add(videos / "a.mp4") is None
    lib.add(videos / "b.mp4")
    lib.remove(videos / "a.mp4")
    lib.compact()
    assert [v.path.split("/")[-1] for v in lib.videos()] == ["b.mp4"]
    assert Searcher(lib).search("green", k=1)[0].path.endswith("b.mp4")


def test_a_library_refuses_other_settings(tmp_path):
    Library(tmp_path / "lib", Settings(model="toy", fps=1.0), encoder=ToyEncoder())
    with pytest.raises(ValueError):
        Library(tmp_path / "lib", Settings(model="toy", fps=2.0), encoder=ToyEncoder())
