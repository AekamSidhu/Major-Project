"""Gaze traces: the one format every gaze source is converted to.

An eye tracker (GazeBase: 1000 Hz, degrees of visual angle) and the webcam
pipeline (MediaPipe iris ratios: ~10 fps, unitless) produce very different
numbers. Everything downstream works on a `GazeTrace`, and the features are
scale-free, so the reading detector never depends on units.

Invalid samples (blinks, lost face, tracker dropouts) are NaN in `x` and `y`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GazeTrace:
    t: np.ndarray
    """Sample times in seconds, strictly increasing."""
    x: np.ndarray
    """Horizontal gaze, any unit. NaN where invalid."""
    y: np.ndarray
    """Vertical gaze, same unit as `x`. NaN where invalid."""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float)
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if t.ndim != 1 or x.shape != t.shape or y.shape != t.shape:
            raise ValueError("t, x and y must be 1-D arrays of the same length")
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            raise ValueError("t must be strictly increasing")
        # Frozen dataclass: normalise the arrays in place of the originals.
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if len(self) else 0.0

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) > 1 else 0.0

    @property
    def rate(self) -> float:
        """Median sampling rate in Hz."""
        if len(self) < 2:
            return 0.0
        return float(1.0 / np.median(np.diff(self.t)))

    def slice(self, start: float, end: float) -> GazeTrace:
        """Samples with `start <= t < end`."""
        mask = (self.t >= start) & (self.t < end)
        return GazeTrace(self.t[mask], self.x[mask], self.y[mask], dict(self.meta))

    def windows(self, length: float, hop: float) -> Iterator[GazeTrace]:
        """Fixed-length windows every `hop` seconds. A trailing partial window is dropped."""
        if length <= 0 or hop <= 0:
            raise ValueError("length and hop must be positive")
        if len(self) < 2:
            return
        # One sample period of slack so a window that ends exactly on the last
        # sample still counts as complete.
        end = self.t[-1] + 1.0 / max(self.rate, 1e-9)
        start = float(self.t[0])
        while start + length <= end + 1e-9:
            window = self.slice(start, start + length)
            window.meta["window_start"] = start - float(self.t[0])
            yield window
            start += hop


def resample(trace: GazeTrace, fps: float, min_valid_fraction: float = 0.5) -> GazeTrace:
    """Resample to `fps` the way a camera would see it.

    Each output frame is the mean of the input samples in its frame interval
    (a camera integrates light over its exposure), timestamped at the
    interval's centre. A frame whose interval is less than
    `min_valid_fraction` valid becomes NaN, like a frame caught mid-blink.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    step = 1.0 / fps
    if trace.duration < step:
        raise ValueError(f"trace is shorter than one frame at {fps} fps")

    edges = np.arange(trace.t[0], trace.t[-1] + step * 0.5, step)
    n_frames = len(edges) - 1
    frame = np.searchsorted(edges, trace.t, side="right") - 1
    inside = (frame >= 0) & (frame < n_frames)
    ok = inside & trace.valid

    total = np.bincount(frame[inside], minlength=n_frames)
    good = np.bincount(frame[ok], minlength=n_frames)
    sum_x = np.bincount(frame[ok], weights=trace.x[ok], minlength=n_frames)
    sum_y = np.bincount(frame[ok], weights=trace.y[ok], minlength=n_frames)

    keep = (total > 0) & (good >= min_valid_fraction * total) & (good > 0)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    x[keep] = sum_x[keep] / good[keep]
    y[keep] = sum_y[keep] / good[keep]
    t = edges[:-1] + step / 2
    return GazeTrace(t, x, y, {**trace.meta, "fps": fps})


def fill_gaps(trace: GazeTrace, max_gap: float) -> GazeTrace:
    """Linearly interpolate invalid runs no longer than `max_gap` seconds.

    Short gaps (a blink, a dropped frame) would otherwise break velocity
    estimates. Longer gaps stay NaN: guessing across them would invent
    eye movements.
    """
    valid = trace.valid
    if valid.all() or not valid.any():
        return trace
    x, y = trace.x.copy(), trace.y.copy()
    idx = np.flatnonzero(~valid)
    # Split invalid indices into contiguous runs.
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        before, after = run[0] - 1, run[-1] + 1
        if before < 0 or after >= len(trace):
            continue  # can't interpolate at the edges
        if trace.t[after] - trace.t[before] > max_gap + 1e-9:
            continue
        span = trace.t[[before, after]]
        x[run] = np.interp(trace.t[run], span, x[[before, after]])
        y[run] = np.interp(trace.t[run], span, y[[before, after]])
    return GazeTrace(trace.t, x, y, dict(trace.meta))
