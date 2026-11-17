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
