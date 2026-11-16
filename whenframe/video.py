"""Reading frames out of video files with ffmpeg.

Frames are decoded, resampled in time and resized by ffmpeg itself and arrive
as raw RGB bytes, so nothing else in the package touches a codec.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv"}


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int


def probe(path: str | Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:format=duration",
        "-of", "json", str(path),
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise VideoError(f"{path}: {out.stderr.strip() or 'ffprobe failed'}")
    meta = json.loads(out.stdout)
    streams = meta.get("streams") or []
    if not streams:
        raise VideoError(f"{path}: no video stream")
    s = streams[0]
    duration = float(s.get("duration") or meta.get("format", {}).get("duration") or 0.0)
    return VideoInfo(duration, int(s.get("width", 0)), int(s.get("height", 0)))


def iter_frames(path: str | Path, fps: float = 1.0, size: int = 224, chunk: int = 256, seconds: float | None = None):
    """Frames at a fixed rate, squashed to size x size as the encoder expects, `chunk` at a time.

    Yields (times, frames): times in seconds, frames as uint8 (n, size, size, 3).
    The fps filter takes the frame nearest to each tick of the new clock, so the
    i-th output frame stands for the instant i / fps. Memory stays bounded by
    the chunk, whatever the length of the video. `seconds` stops reading there.
    """
    limit = ["-t", f"{seconds:g}"] if seconds else []
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", *limit, "-i", str(path), "-an", "-sn",
        "-vf", f"fps={fps},scale={size}:{size}:flags=bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    frame_bytes = size * size * 3
    done = 0
    with tempfile.TemporaryFile() as err, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err) as proc:
        while True:
            buf = proc.stdout.read(frame_bytes * chunk)
            n = len(buf) // frame_bytes
            if n:
                frames = np.frombuffer(buf[: n * frame_bytes], dtype=np.uint8).reshape(n, size, size, 3)
                yield np.arange(done, done + n) / fps, frames
                done += n
            if n < chunk:
                break
        if proc.wait() != 0:
            err.seek(0)
            raise VideoError(f"{path}: {err.read().decode(errors='replace').strip() or 'ffmpeg failed'}")


def sample_frames(path: str | Path, fps: float = 1.0, size: int = 224, seconds: float | None = None):
    """All frames of `iter_frames` at once: (times, frames)."""
    parts = list(iter_frames(path, fps, size, seconds=seconds))
    if not parts:
        return np.zeros(0), np.zeros((0, size, size, 3), dtype=np.uint8)
    return np.concatenate([t for t, _ in parts]), np.concatenate([f for _, f in parts])


def thumbnails(frames: np.ndarray) -> np.ndarray:
    """16x16 grey versions of the frames, float32 on the 0-255 scale."""
    n, h, w, _ = frames.shape
    kh, kw = h // 16, w // 16
    grey = frames[:, : kh * 16, : kw * 16].mean(axis=3, dtype=np.float32)
    return grey.reshape(n, 16, kh, 16, kw).mean(axis=(2, 4))


class RepeatFilter:
    """Decides which frames are worth encoding, chunk after chunk of one video.

    A frame is kept when it differs from the last kept one (mean absolute
    difference of the thumbnails above `threshold`), or when `max_gap` frames
    in a row were skipped. Still shots, which are common in home videos, then
    cost one embedding instead of one per second.
    """

    def __init__(self, threshold: float = 4.0, max_gap: int = 4):
        self.threshold = threshold
        self.max_gap = max_gap
        self._last = None
        self._skipped = 0

    def __call__(self, thumbs: np.ndarray) -> np.ndarray:
        keep = []
        for i, t in enumerate(thumbs):
            if self._last is None or self._skipped >= self.max_gap or np.abs(t - self._last).mean() > self.threshold:
                keep.append(i)
                self._last = t
                self._skipped = 0
            else:
                self._skipped += 1
        return np.asarray(keep, dtype=np.int64)


def drop_repeats(frames: np.ndarray, threshold: float = 4.0, max_gap: int = 4) -> np.ndarray:
    """Indices of the frames of a whole video that `RepeatFilter` keeps."""
    if len(frames) == 0:
        return np.zeros(0, dtype=np.int64)
    return RepeatFilter(threshold, max_gap)(thumbnails(frames))


def find_videos(paths: list[str | Path]) -> list[Path]:
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
        elif p.is_file():
            found.append(p)
    return found
