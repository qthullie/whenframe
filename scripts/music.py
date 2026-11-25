"""The demo site's music (site/music.m4a), composed by this program and dedicated to the public domain (CC0 1.0).

    python scripts/music.py site/music.m4a

A loop of 16 bars at 96 beats a minute, 40 seconds: a bar lasts 2.5 seconds, the time a shot of the site's films
stays before the next one fades in, so that each shot comes in on a bar. Four chords (C major 7, A minor 7, F major 7,
G with a sixth) held by a soft pad, an arpeggio on a piano-like tone, a bell on the first beat of each bar from the
third, and from there a bass and a light beat. The reverb convolves with a decaying noise; what rings past the end
is added to the start, so that the loop has no seam. The noise comes from a fixed seed, and the file is the same
each time ffmpeg encodes it the same way.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 44100
BPM = 96
BEAT = 60 / BPM  # 0.625 s
BAR = 4 * BEAT  # 2.5 s, a shot of the site's films
BARS = 16
LENGTH = BARS * BAR
CHORDS = [  # MIDI notes: the bass, then the pad
    (48, [60, 64, 67, 71]),  # C major 7
    (45, [57, 60, 64, 67]),  # A minor 7
    (41, [53, 57, 60, 64]),  # F major 7
    (43, [55, 59, 62, 64]),  # G, with a sixth
]
ARPEGGIO = [0, 1, 2, 3, 4, 3, 2, 1]  # the chord's notes, an octave up, eighth by eighth (4: the first, an octave up)


def hz(note: float) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def seconds(n: float) -> int:
    return int(round(n * RATE))


class Track:
    """A stereo buffer longer than the loop: what rings past its end is folded back onto its start."""

    def __init__(self, tail: float = 4.0):
        self.x = np.zeros((seconds(LENGTH + tail), 2))

    def add(self, at: float, sound: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
        """pan from -1 (left) to 1 (right), equal power; a sound starting before the loop starts at its end."""
        i = seconds(at % LENGTH)
        n = min(len(sound), len(self.x) - i)
        angle = (pan + 1) * np.pi / 4
        self.x[i : i + n, 0] += gain * np.cos(angle) * sound[:n]
        self.x[i : i + n, 1] += gain * np.sin(angle) * sound[:n]

    def looped(self) -> np.ndarray:
        n = seconds(LENGTH)
        out = self.x[:n].copy()
        out[: len(self.x) - n] += self.x[n:]
        return out


def envelope(n: int, attack: float, release: float) -> np.ndarray:
    a, r = min(seconds(attack), n // 2), min(seconds(release), n // 2)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a) ** 2
    env[n - r :] *= np.linspace(1, 0, r) ** 2
    return env


def pad(note: int, length: float, rng: np.random.Generator) -> np.ndarray:
    """Two voices a few cents apart, odd harmonics falling fast: a soft, warm tone, slowly in and out."""
    t = np.arange(seconds(length)) / RATE
    out = np.zeros_like(t)
    for cents in (-6.0, 6.0):
        f = hz(note + cents / 100)
        phase = rng.uniform(0, 2 * np.pi)
        for k in (1, 3, 5, 7):
            out += np.sin(2 * np.pi * k * f * t + k * phase) / k**2
    return out * envelope(len(t), 0.8, 1.0) / 2


def piano(note: float, length: float = 1.6) -> np.ndarray:
    """Partials slightly sharp, as a string's are, the higher ones dying sooner; a soft hammer."""
    t = np.arange(seconds(length)) / RATE
    f = hz(note)
    out = sum(np.sin(2 * np.pi * k * f * np.sqrt(1 + 0.0004 * k * k) * t) * np.exp(-t * (1.2 + 0.9 * k)) / k**1.5
              for k in range(1, 7))  # fmt: skip
    return out * envelope(len(t), 0.004, 0.05)


def bell(note: float, length: float = 2.4) -> np.ndarray:
    """A few inharmonic partials with long, different decays."""
    t = np.arange(seconds(length)) / RATE
    f = hz(note)
    ratios = ((1.0, 1.0, 1.4), (2.0, 0.5, 0.9), (2.76, 0.35, 0.7), (5.4, 0.15, 0.35))
    out = sum(a * np.sin(2 * np.pi * r * f * t) * np.exp(-t / d) for r, a, d in ratios)
    return out * envelope(len(t), 0.002, 0.1)


def bass(note: int, length: float) -> np.ndarray:
    t = np.arange(seconds(length)) / RATE
    f = hz(note)
    out = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return out * envelope(len(t), 0.01, 0.15) * np.exp(-t * 0.6)


def kick() -> np.ndarray:
    t = np.arange(seconds(0.45)) / RATE
    f = 45 + 75 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t / 0.14)


def snap(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(seconds(0.3)) / RATE
    noise = np.diff(rng.standard_normal(len(t) + 1))  # high-passed
    return (0.6 * noise * np.exp(-t / 0.05) + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)) * 0.5


def hat(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(seconds(0.12)) / RATE
    return np.diff(np.diff(rng.standard_normal(len(t) + 2))) * np.exp(-t / 0.025) * 0.25


def reverb(x: np.ndarray, rng: np.random.Generator, length: float = 2.4, decay: float = 0.5) -> np.ndarray:
    """Convolution with two noises (left and right) dying exponentially, darkened, after 20 ms."""
    t = np.arange(seconds(length)) / RATE
    ir = rng.standard_normal((len(t), 2)) * np.exp(-t / decay)[:, None]
    for _ in range(3):  # three passes of a short moving average: a gentle low-pass
        ir = (ir + np.roll(ir, 1, axis=0) + np.roll(ir, 2, axis=0)) / 3
    ir = np.concatenate([np.zeros((seconds(0.02), 2)), ir]) / np.sqrt((ir**2).sum(axis=0))
    n = 1 << int(np.ceil(np.log2(len(x) + len(ir))))
    wet = np.fft.irfft(np.fft.rfft(x, n, axis=0) * np.fft.rfft(ir, n, axis=0), n, axis=0)
    return wet[: len(x)]


def compose(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tonal, beat = Track(), Track()
    for bar in range(BARS):
        start = bar * BAR
        root, notes = CHORDS[bar % 4]
        for i, note in enumerate(notes):
            tonal.add(start - 0.3, pad(note, BAR + 1.0, rng), 0.05, pan=(-0.5, 0.5)[i % 2])
        up = [n + 12 for n in notes] + [notes[0] + 24]
        for k, which in enumerate(ARPEGGIO):
            swing = 0.06 * BEAT if k % 2 else 0.0
            velocity = (0.9 if k % 2 else 1.0) * rng.uniform(0.85, 1.0)
            tonal.add(start + k * BEAT / 2 + swing, piano(up[which]), 0.07 * velocity, pan=0.3 * np.sin(k))
        if bar >= 2:
            tonal.add(start, bell(notes[-1] + 12), 0.05, pan=-0.2)
            for b in (0, 1.5, 2, 3):  # the bass: on 1, before 3, on 3 and on 4
                beat.add(start + b * BEAT, bass(root, 0.9 * BEAT if b == 1.5 else 1.2 * BEAT), 0.16)
            kicks = (0, 2.5) if bar % 4 == 3 else (0, 2)
            for b in kicks:
                beat.add(start + b * BEAT, kick(), 0.35)
            for b in (1, 3):
                beat.add(start + b * BEAT, snap(rng), 0.12, pan=0.1)
            for k in range(8):
                beat.add(start + k * BEAT / 2 + (0.06 * BEAT if k % 2 else 0), hat(rng), 0.10 if k % 2 else 0.06,
                         pan=0.35)  # fmt: skip
    dry = tonal.looped()
    rhythm = beat.looped()
    wet = reverb(np.concatenate([dry, dry]), rng)[len(dry) :]  # the loop played twice: its tail wraps around
    mix = dry + 0.55 * wet + rhythm + 0.08 * reverb(np.concatenate([rhythm, rhythm]), rng)[len(rhythm) :]
    mix -= mix.mean(axis=0)
    mix *= 10 ** (-21 / 20) / np.sqrt(np.mean(mix**2))  # -21 dB RMS: about -16 LUFS once rounded
    return np.tanh(1.5 * mix) / np.tanh(1.5) * 0.9  # rounded peaks, below -0.9 dB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="the music to write (.m4a)")
    a = ap.parse_args()
    audio = compose()
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "music.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "128k",
                        "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", "-movflags",
                        "+faststart", a.out], check=True)  # fmt: skip
    peak = np.abs(audio).max()
    print(f"wrote {a.out}: {LENGTH:.0f} s, peak {20 * np.log10(peak):.1f} dB, "
          f"RMS {20 * np.log10(np.sqrt(np.mean(audio**2))):.1f} dB")  # fmt: skip


if __name__ == "__main__":
    main()
