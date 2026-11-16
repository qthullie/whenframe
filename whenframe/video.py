"""Reading frames out of video files with ffmpeg.

Frames are decoded, resampled in time and resized by ffmpeg itself and arrive
as raw RGB bytes, so nothing else in the package touches a codec.
"""

from __future__ import annotations

import json
import subprocess
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


def sample_frames(path: str | Path, fps: float = 1.0, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    """Frames at a fixed rate, squashed to size x size as the encoder expects.

    Returns (times, frames): times in seconds, frames as uint8 (n, size, size, 3).
    The fps filter takes the frame nearest to each tick of the new clock, so the
    i-th output frame stands for the instant i / fps.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-sn",
        "-vf", f"fps={fps},scale={size}:{size}:flags=bilinear",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise VideoError(f"{path}: {out.stderr.decode(errors='replace').strip() or 'ffmpeg failed'}")
    frame_bytes = size * size * 3
    n = len(out.stdout) // frame_bytes
    frames = np.frombuffer(out.stdout[: n * frame_bytes], dtype=np.uint8).reshape(n, size, size, 3)
    return np.arange(n) / fps, frames


def drop_repeats(frames: np.ndarray, threshold: float = 4.0, max_gap: int = 4) -> np.ndarray:
    """Indices of the frames worth encoding.

    A frame is kept when it differs from the last kept one (mean absolute
    difference of 16x16 grey thumbnails, 0-255 scale, above `threshold`), or
    when `max_gap` frames have been skipped in a row. Still shots, which are
    common in home videos, then cost one embedding instead of one per second.
    """
    if len(frames) == 0:
        return np.zeros(0, dtype=np.int64)
    n, h, w, _ = frames.shape
    grey = frames.mean(axis=3)
    k = h // 16
    thumbs = grey[:, : k * 16, : k * 16].reshape(n, 16, k, 16, k).mean(axis=(2, 4))
    keep = [0]
    for i in range(1, n):
        if np.abs(thumbs[i] - thumbs[keep[-1]]).mean() > threshold or i - keep[-1] > max_gap:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def find_videos(paths: list[str | Path]) -> list[Path]:
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
        elif p.is_file():
            found.append(p)
    return found
