"""The sound track as CLAP reads it: 48 kHz mono, cut into windows, turned into log-mel features."""

import shutil
import subprocess

import numpy as np
import pytest

from whenframe import sounds


def test_windows_cover_the_whole_sound():
    wave = np.zeros(12 * sounds.RATE, dtype=np.float32)
    starts, clips = sounds.windows(wave, 5.0, 2.5)
    assert starts.tolist() == [0.0, 2.5, 5.0, 7.0]  # the last one ends with the sound
    assert {len(c) for c in clips} == {5 * sounds.RATE}
    short = sounds.windows(wave[: 3 * sounds.RATE], 5.0, 2.5)
    assert short[0].tolist() == [0.0] and len(short[1][0]) == 3 * sounds.RATE
    assert sounds.windows(np.zeros(0, np.float32), 5.0, 2.5)[0].size == 0


def test_a_short_clip_is_repeated_to_fill_ten_seconds():
    rng = np.random.default_rng(0)
    clip = rng.normal(0, 0.1, int(2.5 * sounds.RATE)).astype(np.float32)
    f = sounds.features(clip)
    assert f.shape == (1, 1001, 64) and f.dtype == np.float32
    np.testing.assert_allclose(f, sounds.features(np.tile(clip, 4)), atol=1e-4)
    silent = sounds.features(np.zeros(sounds.RATE, np.float32))
    assert np.allclose(silent, -100.0)  # the floor, 1e-10 in power


def test_mel_filters_are_slaney_triangles_from_50_hz_to_14_khz():
    f = sounds.mel_filters()
    assert f.shape == (513, 64) and (f >= 0).all()
    hz = np.linspace(0, sounds.RATE // 2, 513)
    peaks = hz[f.argmax(axis=0)]
    assert (np.diff(peaks) >= 0).all() and 50 <= peaks[0] < 200 and 12000 < peaks[-1] <= 14000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_sound_of_a_video(tmp_path):
    video = tmp_path / "tone.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x48:d=2", "-f", "lavfi", "-i",
           "sine=frequency=440:duration=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)]  # fmt: skip
    subprocess.run(cmd, check=True)
    wave = sounds.sound(video)
    assert abs(len(wave) - 2 * sounds.RATE) < 0.1 * sounds.RATE
    spectrum = np.abs(np.fft.rfft(wave))
    assert abs(np.fft.rfftfreq(len(wave), 1 / sounds.RATE)[spectrum.argmax()] - 440) < 5
    assert len(sounds.sound(video, 1.0, 0.5)) == pytest.approx(0.5 * sounds.RATE, abs=0.02 * sounds.RATE)


class FakeClap:
    """Embeds a clip as its mean value on the first axis, and records what it was given."""

    def __init__(self):
        self.lengths = []

    def encode_audio(self, clips):
        self.lengths += [len(c) for c in clips]
        out = np.zeros((len(clips), 4), dtype=np.float32)
        out[:, 0] = [c.mean() if len(c) else 0.0 for c in clips]
        out[:, 1] = 1.0
        return out


def test_long_sounds_are_read_a_chunk_at_a_time(monkeypatch):
    rate = 100  # samples per second, to keep the arrays small
    total = 47.0
    reads = []

    def fake_sound(path, start=0.0, seconds=None):
        reads.append((start, seconds))
        n = int(round(max(0.0, min(total, start + seconds) - start) * rate))
        return np.full(n, start, dtype=np.float32)

    monkeypatch.setattr(sounds, "RATE", rate)
    monkeypatch.setattr(sounds, "CHUNK", 20.0)
    monkeypatch.setattr(sounds, "sound", fake_sound)
    clap = FakeClap()
    heard = sounds.Listener(clap, sounds.SoundConfig(length=5.0))("video.mp4", total)
    starts = [s for s, _ in heard]
    assert reads == [(0.0, 25.0), (20.0, 25.0), (40.0, 25.0)]
    assert starts == [*np.arange(0.0, 42.5, 2.5).tolist(), 42.0]  # one grid across the reads, the last at the end
    assert set(clap.lengths) == {5 * rate}


def test_frames_get_the_best_window_around_their_middle():
    times = np.array([0.0, 1.0, 2.0, 3.0, 9.0, 0.0, 1.0])
    ranges = {7: (0, 5), 8: (5, 2)}
    ids = np.array([7, 7, 7])
    starts = np.array([0.0, 2.5, 5.0])
    cos = np.array([0.1, 0.5, 0.2], dtype=np.float32)
    s = sounds.frame_scores(times, ranges, ids, starts, cos, 5.0)
    np.testing.assert_allclose(s[:5], [0.1, 0.1, 0.5, 0.5, 0.2])  # 9.5 s is past the windows: the nearest
    assert np.isnan(s[5:]).all()  # video 8 has no sound
    z = sounds.standardised(s)
    assert (z[5:] == 0).all() and abs(z[:5].mean()) < 1e-6


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_library_searches_what_is_heard(tmp_path):
    from test_library import ToyEncoder, make_video

    from whenframe import Library, SearchConfig, Searcher, Settings

    make_video(str(tmp_path / "a.mp4"), [("grey", 20)])
    make_video(str(tmp_path / "b.mp4"), [("grey", 20)])

    class Fake:
        config = sounds.SoundConfig(length=5.0)

        def __init__(self):
            self.calls = []

        def __call__(self, path, duration):
            self.calls.append(path)
            bark = 14.0 if path.endswith("b.mp4") else -1.0  # a dog barks 14 s into b
            return [(s, np.array([1.0 if s <= bark < s + 5 else 0.0, 1.0], np.float32)) for s in np.arange(0, 16, 2.5)]

    fake = Fake()
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder(), listener=fake)
    for name in ("a", "b"):
        lib.add(tmp_path / f"{name}.mp4")
    assert fake.calls == []  # off until asked for
    lib.enable_sounds()
    for name in ("a", "b"):
        lib.add(tmp_path / f"{name}.mp4")
    assert len(fake.calls) == 2 and lib.stats()["sound_windows"] == 14
    ids, starts, emb = lib.sound_windows()
    heard = emb @ np.array([1.0, 0.0], np.float32)  # "a dog barking"
    search = Searcher(lib, SearchConfig(window=4.0))
    _, embs, _ = lib.snapshot()
    grey = ToyEncoder().encode_texts(["grey"])[0]
    # frames that look alike, as in any library: sound is added in units of their scores' spread
    frames = embs["toy"] @ grey + np.random.default_rng(0).uniform(-0.01, 0.01, len(embs["toy"]))
    top = search.search("a dog barking", k=3, scores={"toy": frames}, heard=heard)[0]
    assert top.path.endswith("b.mp4") and top.start <= 14.0 <= top.end
    lib.remove(tmp_path / "b.mp4")
    assert lib.stats()["sound_windows"] == 7


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_changed_video_is_heard_and_transcribed_again(tmp_path):
    import os

    from test_library import ToyEncoder, make_video
    from test_speech import FakeTranscriber

    from whenframe import Library, Settings

    class Fake:
        config = sounds.SoundConfig(length=5.0)

        def __init__(self):
            self.calls = 0

        def __call__(self, path, duration):
            self.calls += 1
            return [(0.0, np.ones(2, np.float32))]

    make_video(str(tmp_path / "a.mp4"), [("grey", 6)])
    heard, said = Fake(), FakeTranscriber({"a.mp4": [(1.0, 2.0, "hello")]})
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder(), listener=heard, transcriber=said)
    lib.enable_sounds()
    lib.enable_speech("base")
    lib.add(tmp_path / "a.mp4")
    lib.add(tmp_path / "a.mp4")  # unchanged: nothing to do
    assert heard.calls == 1 and len(said.calls) == 1
    make_video(str(tmp_path / "a.mp4"), [("green", 8)])
    os.utime(tmp_path / "a.mp4", (1, 1))  # a different file, whatever the clock
    lib.add(tmp_path / "a.mp4")
    assert heard.calls == 2 and len(said.calls) == 2
    assert lib.stats()["sound_windows"] == 1 and lib.stats()["speech_segments"] == 1


def test_the_defaults_are_the_tuned_ones():
    import json
    from pathlib import Path

    tuned = Path(__file__).resolve().parents[1] / "results" / "unav_tuned.json"
    if not tuned.exists():
        pytest.skip("no tuning run")
    best = json.loads(tuned.read_text())["best"]
    assert (sounds.SoundConfig().length, sounds.SoundConfig().weight) == (best["length"], best["weight"])
