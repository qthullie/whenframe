"""From a per-frame relevance curve to moments (start, end, score).

The curve is sampled irregularly (repeated frames were never encoded), so it is
first put back on a regular grid by holding each value until the next frame:
a frame that was skipped looked like the last one kept.

A moment grows around a peak while the curve stays above a level between the
video's background (its median) and the peak:

    keep t  while  c(t) >= floor + alpha * (peak - floor)

so a sharp peak gives a short moment and a plateau a long one, without a fixed
window. Further moments come from the next peaks once the first is masked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    peak: float  # highest value of the curve inside the span
    mean: float  # mean value inside the span
    floor: float  # the video's background level


def on_grid(times: np.ndarray, values: np.ndarray, duration: float, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Hold each sample until the next one, on a grid of `step` seconds covering [0, duration]."""
    duration = max(float(duration), float(times[-1]) + step if len(times) else step)
    grid = np.arange(0.0, duration, step)
    idx = np.searchsorted(times, grid + 1e-6, side="right") - 1
    return grid, values[np.clip(idx, 0, len(values) - 1)]


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing, sigma in grid steps; edges are padded with their own value."""
    if sigma <= 0 or len(values) < 3:
        return values
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(values, r, mode="edge"), k, mode="valid")


def spans(grid: np.ndarray, curve: np.ndarray, step: float, alpha: float = 0.6, count: int = 3,
          min_length: float = 1.0) -> list[Span]:  # fmt: skip
    """Up to `count` moments around the highest peaks of `curve`."""
    if len(curve) == 0:
        return []
    floor = float(np.median(curve))
    free = np.ones(len(curve), dtype=bool)
    out = []
    for _ in range(count):
        if not free.any():
            break
        i = int(np.argmax(np.where(free, curve, -np.inf)))
        peak = float(curve[i])
        level = floor + alpha * (peak - floor)
        lo = i
        while lo > 0 and free[lo - 1] and curve[lo - 1] >= level:
            lo -= 1
        hi = i
        while hi < len(curve) - 1 and free[hi + 1] and curve[hi + 1] >= level:
            hi += 1
        start, end = float(grid[lo]), float(grid[hi]) + step
        if end - start < min_length:
            pad = (min_length - (end - start)) / 2
            start, end = max(0.0, start - pad), min(float(grid[-1]) + step, end + pad)
        out.append(Span(start, end, peak, float(curve[lo : hi + 1].mean()), floor))
        free[lo : hi + 1] = False
    return out
