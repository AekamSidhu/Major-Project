"""A compact, frame-level copy of an eye-tracking dataset.

Reading 12,000 CSVs at 1000 Hz out of nested zips takes most of an hour, and
every experiment (another frame rate, another noise level, another model)
would repeat it. `build_cache.py` does it once and keeps each recording as
30 Hz camera-style frames: the mean gaze in each 1/30 s interval, plus how
many of that interval's samples were valid.

Keeping the counts makes lower frame rates exact. A 10 fps frame is three
30 Hz frames merged, weighted by their valid samples, which is the same
number `trace.resample` gives on the raw recording (apart from samples that
sit exactly on a frame boundary, which rounding can put on either side).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .trace import GazeTrace

CACHE_RATE = 30.0
META_COLUMNS = ("dataset", "participant", "recording", "task", "label")


@dataclass(frozen=True)
class Frames:
    """One recording at the cache rate."""

    x: np.ndarray
    """Mean horizontal gaze per frame; NaN where no sample was valid."""
    y: np.ndarray
    n_valid: np.ndarray
    """Valid samples in the frame interval."""
    n_total: np.ndarray
    """All samples in the frame interval."""


def bin_samples(t: np.ndarray, x: np.ndarray, y: np.ndarray, rate: float = CACHE_RATE) -> Frames:
    """Average samples into frames of 1/`rate` seconds, like a camera exposure.

    `t` in seconds, increasing. Invalid samples are NaN in `x` or `y`.
    """
    step = 1.0 / rate
    if len(t) < 2 or t[-1] - t[0] < step:
        raise ValueError("recording is shorter than one frame")
    n_frames = int(np.floor((t[-1] - t[0]) * rate + 0.5))
    # The small offset keeps a sample that sits exactly on a frame boundary
    # from landing on either side depending on floating-point rounding.
    frame = np.floor((t - t[0]) * rate + 1e-6).astype(np.int64)
    inside = (frame >= 0) & (frame < n_frames)
    ok = inside & np.isfinite(x) & np.isfinite(y)

    total = np.bincount(frame[inside], minlength=n_frames)
    good = np.bincount(frame[ok], minlength=n_frames)
    sum_x = np.bincount(frame[ok], weights=x[ok], minlength=n_frames)
    sum_y = np.bincount(frame[ok], weights=y[ok], minlength=n_frames)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = np.where(good > 0, sum_x / good, np.nan)
        mean_y = np.where(good > 0, sum_y / good, np.nan)
    return Frames(mean_x.astype(np.float32), mean_y.astype(np.float32), good.astype(np.uint16), total.astype(np.uint16))


def merge_frames(frames: Frames, factor: int, min_valid_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Merge every `factor` frames into one. Returns `(x, y)` with NaN where
    less than `min_valid_fraction` of the merged interval was valid."""
    if factor < 1:
        raise ValueError("factor must be at least 1")
    n = (len(frames.x) // factor) * factor
    shape = (-1, factor)
    good = frames.n_valid[:n].astype(np.float64).reshape(shape)
    total = frames.n_total[:n].astype(np.float64).reshape(shape).sum(axis=1)
    sum_x = (np.nan_to_num(frames.x[:n].astype(np.float64)).reshape(shape) * good).sum(axis=1)
    sum_y = (np.nan_to_num(frames.y[:n].astype(np.float64)).reshape(shape) * good).sum(axis=1)
    good = good.sum(axis=1)
    keep = (good > 0) & (good >= min_valid_fraction * total)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(keep, sum_x / good, np.nan)
        y = np.where(keep, sum_y / good, np.nan)
    return x, y


class TraceCache:
    """Every recording of a dataset as 30 Hz frames, plus who and what it was."""

    def __init__(self, meta: pd.DataFrame, offsets: np.ndarray, x: np.ndarray, y: np.ndarray, n_valid: np.ndarray, n_total: np.ndarray, rate: float = CACHE_RATE) -> None:
        self.meta = meta.reset_index(drop=True)
        self.offsets = offsets
        self.x, self.y, self.n_valid, self.n_total = x, y, n_valid, n_total
        self.rate = float(rate)

    def __len__(self) -> int:
        return len(self.meta)

    def frames(self, index: int) -> Frames:
        a, b = self.offsets[index], self.offsets[index + 1]
        return Frames(self.x[a:b], self.y[a:b], self.n_valid[a:b], self.n_total[a:b])

    def trace(self, index: int, fps: float, speed: float = 1.0) -> GazeTrace:
        """Recording `index` as a `GazeTrace` at `fps`.

        `speed` replays the eye movements faster or slower: at 2/3 every
        movement takes 1.5 times as long, a stand-in for reading aloud, which
        is slower than the silent reading in the datasets. `rate * speed /
        fps` must be a whole number of cache frames per output frame.
        """
        factor = self.rate * speed / fps
        if factor < 0.5 or abs(factor - round(factor)) > 1e-6:
            raise ValueError(f"{fps} fps at speed {speed:g} is not a whole number of {self.rate:g} Hz frames")
        x, y = merge_frames(self.frames(index), int(round(factor)))
        t = (np.arange(len(x)) + 0.5) / fps
        row = self.meta.iloc[index]
        return GazeTrace(t, x, y, {"dataset": row["dataset"], "fps": fps})

    @classmethod
    def from_recordings(cls, rows: list[dict], frames: list[Frames], rate: float = CACHE_RATE) -> TraceCache:
        lengths = np.array([len(f.x) for f in frames], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        cat = lambda name, dtype: np.concatenate([getattr(f, name) for f in frames]).astype(dtype)  # noqa: E731
        return cls(pd.DataFrame(rows), offsets, cat("x", np.float32), cat("y", np.float32), cat("n_valid", np.uint16), cat("n_total", np.uint16), rate)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {}
        for name in self.meta.columns:
            values = self.meta[name]
            # Text columns go in as fixed-width strings: no pickling on load.
            columns[f"meta_{name}"] = values.to_numpy() if pd.api.types.is_numeric_dtype(values) else values.to_numpy(dtype=str)
        np.savez_compressed(path, rate=self.rate, offsets=self.offsets, x=self.x, y=self.y, n_valid=self.n_valid, n_total=self.n_total, **columns)

    @classmethod
    def load(cls, path: str | Path) -> TraceCache:
        with np.load(path, allow_pickle=False) as data:
            meta = pd.DataFrame({key[5:]: data[key] for key in data.files if key.startswith("meta_")})
            return cls(meta, data["offsets"], data["x"], data["y"], data["n_valid"], data["n_total"], float(data["rate"]))
