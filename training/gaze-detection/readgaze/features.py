"""Reading features for one window of a gaze trace.

Reading leaves a sawtooth in horizontal gaze: many small steps one way along
a line, then one large, fast return sweep back to the start of the next line.
The features measure exactly that, and none of them depend on units or on
the overall size of the movements:

- velocity_skew: reading's horizontal velocity is lopsided (lots of small
  forward steps, a few big returns). Looking around is roughly symmetric.
- sweep_rate: return sweeps per minute.
- sweep_regularity: lines of text are similar lengths, so sweeps come at a
  steady rhythm.
- sweep_asymmetry: big jumps in reading go one way only (returns). Jumping
  back and forth between two places goes both ways.
- forward_fraction: between sweeps, gaze mostly creeps in one direction.
- horizontal_dominance: reading moves mostly sideways, not up and down.

The reading direction is never assumed. Whichever way the big jumps go is
treated as "return". This keeps the detector working when the video is
mirrored and for right-to-left scripts (Arabic, Hebrew, Urdu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import skew

from .trace import GazeTrace, fill_gaps

FEATURE_NAMES: tuple[str, ...] = (
    "velocity_skew",
    "sweep_rate",
    "sweep_regularity",
    "sweep_asymmetry",
    "forward_fraction",
    "horizontal_dominance",
)

# Gaussian standard deviation from the median absolute deviation.
_MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class FeatureConfig:
    max_gap: float = 0.3
    """Gaps up to this many seconds (a blink) are interpolated."""
    min_valid_fraction: float = 0.6
    """Windows with less valid gaze than this are skipped."""
    min_steps: int = 20
    """Minimum number of usable frame-to-frame steps in a window."""
    median_frames: int = 3
    """Median filter length. A median removes jitter but keeps sharp jumps, so
    sweeps survive while noise shrinks. 1 turns it off."""
    sweep_threshold: float = 2.0
    """A sweep step must exceed this many noise standard deviations. Noise is
    measured before the median filter, which then removes most of it, so a
    low multiple is enough."""
    min_sweep_span: float = 0.35
    """A sweep must cover at least this share of the window's horizontal range."""


@dataclass(frozen=True)
class SweepAnalysis:
    """What the detector saw in a window. Kept as evidence for the interviewer."""

    times: np.ndarray
    """Start time of each return sweep, in seconds."""
    amplitudes: np.ndarray
    """Signed horizontal size of each return sweep."""
    direction: int
    """+1 or -1: the direction of the return sweeps in this window."""
    forward_jumps: int
    """Large jumps in the other direction (rare when reading)."""
    noise: float
    """Estimated frame-to-frame noise (standard deviation of a step)."""


def analyse_sweeps(window: GazeTrace, config: FeatureConfig = FeatureConfig()) -> SweepAnalysis | None:
    """Find the return sweeps in a window, or None if the window is unusable."""
    prepared = _prepare(window, config)
    if prepared is None:
        return None
    return _analyse(prepared, config)


def compute_features(window: GazeTrace, config: FeatureConfig = FeatureConfig()) -> dict[str, float] | None:
    """Features for one window, keyed by `FEATURE_NAMES`, or None if unusable."""
    prepared = _prepare(window, config)
    if prepared is None:
        return None
    x, y = prepared.x, prepared.y
    sweeps = _analyse(prepared, config)
    dx = np.diff(x)
    steps = dx[np.isfinite(dx)]
    duration_min = max(window.duration, 1e-9) / 60.0

    velocity_skew = float(abs(skew(steps))) if steps.std() > 0 else 0.0
    n_sweeps = len(sweeps.times)

    if n_sweeps >= 3:
        gaps = np.diff(sweeps.times)
        regularity = float(1.0 / (1.0 + gaps.std() / gaps.mean()))
    else:
        regularity = 0.0

    big_jumps = n_sweeps + sweeps.forward_jumps
    asymmetry = (n_sweeps - sweeps.forward_jumps) / big_jumps if big_jumps else 0.0

    # Between sweeps: of the steps clearly above noise, how many go forward
    # (opposite to the return direction)?
    small = steps[np.abs(steps) < config.sweep_threshold * sweeps.noise]
    moving = small[np.abs(small) > 0.5 * sweeps.noise]
    forward_fraction = float(np.mean(np.sign(moving) == -sweeps.direction)) if len(moving) else 0.5

    spread_x = _iqr(x)
    spread_y = _iqr(y)
    dominance = float(np.log((spread_x + 1e-9) / (spread_y + 1e-9)))

    return {
        "velocity_skew": min(velocity_skew, 10.0),
        "sweep_rate": n_sweeps / duration_min,
        "sweep_regularity": regularity,
        "sweep_asymmetry": float(asymmetry),
        "forward_fraction": forward_fraction,
        "horizontal_dominance": float(np.clip(dominance, -5.0, 5.0)),
    }


def _analyse(prepared: _Prepared, config: FeatureConfig) -> SweepAnalysis:
    t, x, noise = prepared.t, prepared.x, prepared.noise
    dx = np.diff(x)
    steps = dx[np.isfinite(dx)]

    direction = _return_direction(steps)
    span = np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
    threshold = max(config.sweep_threshold * noise, 1e-12)

    events = _jump_events(dx, threshold)
    sweep_times, sweep_amps, forward = [], [], 0
    for first, _, total in events:
        if abs(total) < config.min_sweep_span * span:
            continue
        if np.sign(total) == direction:
            sweep_times.append(t[first])
            sweep_amps.append(total)
        else:
            forward += 1
    return SweepAnalysis(
        times=np.asarray(sweep_times),
        amplitudes=np.asarray(sweep_amps),
        direction=direction,
        forward_jumps=forward,
        noise=noise,
    )


class _Prepared(NamedTuple):
    t: np.ndarray
    x: np.ndarray
    """Gap-filled and median-filtered."""
    y: np.ndarray
    noise: float
    """Step noise measured *before* the median filter. The filter leaves many
    steps at exactly zero, which would make noise look far smaller than it is
    and let ordinary jitter count as sweeps."""


def _prepare(window: GazeTrace, config: FeatureConfig) -> _Prepared | None:
    if len(window) < config.min_steps + 1 or window.valid_fraction < config.min_valid_fraction:
        return None
    filled = fill_gaps(window, config.max_gap)
    raw_steps = np.diff(filled.x)
    raw_steps = raw_steps[np.isfinite(raw_steps)]
    x = _median_filter(filled.x, config.median_frames)
    y = _median_filter(filled.y, config.median_frames)
    if len(raw_steps) < config.min_steps or np.isfinite(np.diff(x)).sum() < config.min_steps:
        return None
    return _Prepared(filled.t, x, y, _robust_std(raw_steps))


def _median_filter(values: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return values
    smooth = pd.Series(values).rolling(frames, center=True, min_periods=1).median().to_numpy(copy=True)
    smooth[~np.isfinite(values)] = np.nan  # don't invent samples inside long gaps
    return smooth


def _robust_std(values: np.ndarray) -> float:
    mad = np.median(np.abs(values - np.median(values)))
    return float(_MAD_TO_STD * mad) if mad > 0 else float(np.std(values))


def _return_direction(steps: np.ndarray) -> int:
    """Direction of the long tail of the step distribution: where the big jumps go."""
    if steps.std() == 0:
        return 1
    return 1 if skew(steps) > 0 else -1


def _jump_events(dx: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Merge consecutive same-direction steps above `threshold` into single jumps.

    At 10 fps one sweep can straddle two frames. Returns
    `(first_step, last_step, total_displacement)` per jump.
    """
    big = np.isfinite(dx) & (np.abs(dx) > threshold)
    events: list[tuple[int, int, float]] = []
    i = 0
    while i < len(dx):
        if not big[i]:
            i += 1
            continue
        sign = np.sign(dx[i])
        j = i
        while j + 1 < len(dx) and big[j + 1] and np.sign(dx[j + 1]) == sign:
            j += 1
        events.append((i, j, float(dx[i : j + 1].sum())))
        i = j + 1
    return events


def _iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 0.0
    q75, q25 = np.percentile(finite, [75, 25])
    return float(q75 - q25)
