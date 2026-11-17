"""What is heard besides speech: laughter, applause, a dog barking, music (CLAP, on the CPU).

LAION's CLAP (larger_clap_general, Apache 2.0) embeds a sound and a text in one
space, as SigLIP does an image and a text. The sound track is cut into windows
of a few seconds, each embedded once at indexing; a search embeds the
description with CLAP's text tower and scores every window by its cosine.

The audio tower reads 10 seconds of 48 kHz sound as a log-mel spectrogram (64
bands from 50 Hz to 14 kHz, 1,001 frames); a shorter window is repeated until it
fills them, as CLAP's own feature extractor does ("repeatpad"). The features are
computed here with numpy, as transformers' ClapFeatureExtractor computes them.

Optional: `whenframe index --sounds`. A search then adds CLAP's score of the
window around each frame, standardised over the library, to the frame's own
standardised score, with the weight and window length chosen on UnAV-100 tuning
videos (docs/PROTOCOL.md, fourth round).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder import _normalize, _OnnxRuntime, _OpenVINO, _session, backend_name, fetch_files

REPO = "Xenova/larger_clap_general"
FILES = ("tokenizer.json", "onnx/audio_model.onnx", "onnx/text_model.onnx")
RATE = 48000
SAMPLES = 10 * RATE  # what the audio tower reads at once
N_FFT, HOP, BANDS, LOW, HIGH = 1024, 480, 64, 50.0, 14000.0
CHUNK = 600.0  # seconds of sound read at a time: an hour at 48 kHz would take 700 MB

SCHEMA = """
create table if not exists sound (
    id integer primary key,
    video_id integer not null,
    start real not null,
    emb blob not null
);
create index if not exists sound_video on sound(video_id);
create table if not exists heard (video_id integer primary key, length real not null, at real not null);
"""


@dataclass(frozen=True)
class SoundConfig:
    # chosen on UnAV-100 tuning videos among 14 combinations (results/unav_tuned.json)
    length: float = 5.0  # seconds per window; windows overlap by half
    weight: float = 2.0  # of the standardised sound score, added to the frames' standardised score


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    above = f >= 1000.0
    return np.where(above, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), mel)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    above = m >= 15.0
    return np.where(above, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), 200.0 * m / 3.0)


def mel_filters() -> np.ndarray:
    """(513, 64) triangular filters, Slaney-normalised, as transformers' mel_filter_bank(norm="slaney")."""
    fft_freqs = np.linspace(0, RATE // 2, N_FFT // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(LOW), _hz_to_mel(HIGH), BANDS + 2))
    slopes = edges[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / np.diff(edges)[:-1]
    up = slopes[:, 2:] / np.diff(edges)[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (edges[2:] - edges[:-2]))[None, :]


_FILTERS = mel_filters()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann


def log_mel(wave: np.ndarray) -> np.ndarray:
    """(frames, 64) power spectrogram on the mel bands, in dB; frames centred every 10 ms."""
    x = np.pad(wave.astype(np.float64), N_FFT // 2, mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT)) ** 2
    mel = np.maximum(1e-10, power @ _FILTERS)
    return (10.0 * np.log10(mel)).astype(np.float32)


def features(clip: np.ndarray) -> np.ndarray:
    """(1, 1001, 64): a clip of at most 10 seconds, repeated to fill them, then zero-padded."""
    clip = clip[:SAMPLES]
    if 0 < len(clip) < SAMPLES:
        clip = np.tile(clip, SAMPLES // len(clip))
    clip = np.pad(clip, (0, SAMPLES - len(clip)))
    return log_mel(clip)[None]


def sound(path: str | Path, start: float = 0.0, seconds: float | None = None) -> np.ndarray:
    """The sound track, mono, 48 kHz, float32; empty when there is none."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}"]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    run = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(run.stdout, dtype=np.float32) if run.returncode == 0 else np.zeros(0, np.float32)


def windows(wave: np.ndarray, length: float, hop: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Start times and clips of `length` seconds every `hop` seconds; the last one ends with the sound."""
    if len(wave) == 0:
        return np.zeros(0), []
    n, step = int(length * RATE), int(hop * RATE)
    last = max(0, len(wave) - n)
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    return np.array(starts) / RATE, [wave[s : s + n] for s in starts]


def grid(seconds: float, length: float, offset: float = 0.0, last: bool = True) -> np.ndarray:
    """Window starts every length / 2 seconds from `offset`, each window inside the sound; with `last`, one more
    that ends with the sound when the grid stops short of it."""
    hop = length / 2
    n = int(np.floor(max(0.0, seconds - length) / hop + 1e-9)) + 1
    starts = offset + hop * np.arange(n)
    if last and seconds > length and starts[-1] + length < offset + seconds - 1e-6:
        starts = np.append(starts, offset + seconds - length)
    return starts


class Listener:
    """A video's sound track as (start, embedding) windows, read CHUNK seconds at a time."""

    def __init__(self, clap: Clap | None = None, config: SoundConfig | None = None):
        self.clap = clap or Clap()
        self.config = config or SoundConfig()

    def __call__(self, path: str, duration: float) -> list[tuple[float, np.ndarray]]:
        length, out, t = self.config.length, [], 0.0
        while True:
            wave = sound(path, t, CHUNK + length)
            seconds = len(wave) / RATE
            if seconds < 0.1:
                break
            final = seconds < CHUNK + length - 1e-3  # the sound ends in this read
            starts = grid(seconds if final else CHUNK + length / 2, length, last=final)
            if not final:
                starts = starts[starts < CHUNK - 1e-6]
            clips = [wave[int(round(s * RATE)) : int(round(s * RATE)) + int(length * RATE)] for s in starts]
            out += list(zip((t + starts).tolist(), self.clap.encode_audio(clips), strict=True))
            if final:
                break
            t += CHUNK
        return out


def frame_scores(times: np.ndarray, ranges: dict[int, tuple[int, int]], video_ids: np.ndarray,
                 starts: np.ndarray, cos: np.ndarray, length: float) -> np.ndarray:  # fmt: skip
    """For every row (frame), the best cosine of the windows around its middle; NaN where nothing was heard.

    A frame stands for the second after it; a window counts for it when the frame's
    middle falls inside the window, or, past the last window, the nearest window.
    """
    out = np.full(len(times), np.nan, dtype=np.float32)
    order = np.argsort(video_ids, kind="stable")
    ids, first = np.unique(video_ids[order], return_index=True)
    bounds = dict(zip(ids.tolist(), zip(first, np.r_[first[1:], len(order)], strict=True), strict=True))
    for video, (row, count) in ranges.items():
        if video not in bounds or count == 0:
            continue
        a, b = bounds[video]
        w = order[a:b]
        s, c = starts[w], cos[w]
        mid = times[row : row + count].astype(np.float64) + 0.5
        inside = (s[None, :] <= mid[:, None]) & (mid[:, None] < s[None, :] + length)
        nearest = np.abs(s[None, :] + length / 2 - mid[:, None]).argmin(axis=1)
        inside[np.arange(count), nearest] |= ~inside.any(axis=1)
        out[row : row + count] = np.where(inside, c[None, :], -np.inf).max(axis=1)
    return out


def standardised(scores: np.ndarray) -> np.ndarray:
    """Centred and scaled over the frames that were heard; 0 for the others."""
    heard = ~np.isnan(scores)
    if heard.sum() < 2:
        return np.zeros_like(scores)
    mean, std = scores[heard].mean(), scores[heard].std()
    return np.where(heard, (scores - mean) / max(float(std), 1e-9), 0.0).astype(np.float32)


class Clap:
    """CLAP's two towers. The audio tower runs with OpenVINO when it is installed: in bfloat16 on this
    machine, 121 ms of CPU time per window instead of 566 with ONNX Runtime (features apart, 34 ms)."""

    def __init__(self, folder: str | Path | None = None, threads: int | None = None, backend: str | None = None):
        from tokenizers import Tokenizer

        folder = Path(folder) if folder else fetch_files(REPO, FILES)
        self.backend = backend_name(backend)
        runtime = _OpenVINO if self.backend == "openvino" else _OnnxRuntime
        self._audio = runtime(folder / "onnx" / "audio_model.onnx", "input_features", "audio_embeds", threads)
        self._text_path = folder / "onnx" / "text_model.onnx"
        self._text = None
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(77)
        self.dim = 512

    def encode_audio(self, clips: list[np.ndarray], batch: int = 8) -> np.ndarray:
        """Unit vectors (n, 512), one per clip of at most 10 seconds at 48 kHz."""
        out = []
        for i in range(0, len(clips), batch):
            out.append(self._audio(np.stack([features(c) for c in clips[i : i + batch]])))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.concatenate(out)).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Unit vectors (n, 512). One text at a time: the exported tower takes no attention mask."""
        if self._text is None:
            self._text = _session(self._text_path, None)
        out = [self._text.run(["text_embeds"], {"input_ids": np.array([self._tokenizer.encode(t).ids])})[0]
               for t in texts]  # fmt: skip
        return _normalize(np.concatenate(out)).astype(np.float32)
