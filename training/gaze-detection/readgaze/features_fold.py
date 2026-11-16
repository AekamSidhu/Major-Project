"""Reading features that still work when single sweeps drown in noise.

The sweep features need to see each return sweep on its own. On a small,
compressed video tile, or with the text close to the camera, a sweep is no
bigger than the frame-to-frame jitter and they go blind. These features use
the whole window instead:

- lagged steps: over half a second the slow creep along a line adds up while
  the jitter does not, so the lopsidedness of `x[t + k] - x[t]` grows with
  the lag `k` when someone reads, and stays near zero otherwise;
- folding: cut the window into pieces of one candidate line duration and
  average them on top of each other. Noise averages away. If the average is
  a ramp with one sharp drop, that is the reading sawtooth at that rhythm;
- local slopes: the slope of gaze over 1.5 s stretches is steady in sign and
  size while reading a line.

Like the other features they ignore units and reading direction. They are
computed for many windows at once from an `(n, frames, 2)` array, because
that is how the experiments hold their data.
"""

from __future__ import annotations

import numpy as np

LAGS = (1, 2, 3, 5, 8)
"""Frame lags at 10 fps; scaled with the frame rate."""

FOLD_FEATURE_NAMES: tuple[str, ...] = (
    *(f"lag_skew_{k}" for k in LAGS),
    "lag_skew_y",
    "fold_strength",
    "fold_seconds",
    "saw_strength",
    "saw_seconds",
    "saw_shape",
    "saw_direction",
    "smooth_forward",
    "slope_sign",
    "slope_steadiness",
    "y_rank_trend",
)


def fill_all_gaps(raw: np.ndarray) -> np.ndarray:
    """Linear interpolation across every gap (nearest value at the ends).
    Windows with no valid frame become zeros."""
    filled = np.array(raw, dtype=np.float64)
    valid = np.isfinite(filled).all(axis=2)
    grid = np.arange(filled.shape[1])
    for i in np.flatnonzero(~valid.all(axis=1)):
        ok = valid[i]
        if not ok.any():
            filled[i] = 0.0
            continue
        for c in range(filled.shape[2]):
            filled[i, :, c] = np.interp(grid, grid[ok], filled[i, ok, c])
    return filled


def fold_features(raw: np.ndarray, fps: float = 10.0) -> dict[str, np.ndarray]:
    """The features above for each window of `raw` (`(n, frames, 2)`, NaN = lost frame)."""
    filled = fill_all_gaps(raw)
    x = filled[:, :, 0] - filled[:, :, 0].mean(axis=1, keepdims=True)
    y = filled[:, :, 1] - filled[:, :, 1].mean(axis=1, keepdims=True)
    n, frames = x.shape
    scale = fps / 10.0
    out: dict[str, np.ndarray] = {}

    # Lagged steps. Sign them by the half-second lag, so that agreement across
    # lags is positive and disagreement negative, whatever the reading direction.
    lags = [max(int(round(k * scale)), 1) for k in LAGS]
    skews = [_skew(x[:, k:] - x[:, :-k]) for k in lags]
    direction = np.where(skews[LAGS.index(5)] >= 0, 1.0, -1.0)  # +1: big jumps go right
    for k, s in zip(LAGS, skews):
        out[f"lag_skew_{k}"] = s * direction
    k3 = lags[LAGS.index(3)]
    out["lag_skew_y"] = np.abs(_skew(y[:, k3:] - y[:, :-k3]))

    # Folding at candidate line durations of 1.2 to 6 seconds.
    energy = np.maximum((x**2).sum(axis=1), 1e-12)
    periods = np.unique(np.round(np.arange(1.2, 6.01, 0.2) * fps).astype(int))
    periods = periods[(periods >= 4) & (periods <= frames // 2)]
    fold = np.zeros((len(periods), n))
    saw = np.zeros((len(periods), n))
    saw_sign = np.zeros((len(periods), n))
    for j, period in enumerate(periods):
        onehot = np.zeros((frames, period))
        onehot[np.arange(frames), np.arange(frames) % period] = 1.0
        counts = onehot.sum(axis=0)
        profile = (x @ onehot) / counts
        profile -= (profile * counts).sum(axis=1, keepdims=True) / frames
        explained = (profile**2 * counts).sum(axis=1) / energy
        chance = (period - 1) / (frames - 1)  # what folding pure noise explains
        fold[j] = (explained - chance) / (1.0 - chance)
        # How ramp-like is the folded profile? Best circular match with a ramp.
        ramp = np.linspace(-1.0, 1.0, period)
        match = np.fft.irfft(np.fft.rfft(profile, axis=1) * np.conj(np.fft.rfft(ramp)), n=period, axis=1)
        norm = np.linalg.norm(profile, axis=1) * np.linalg.norm(ramp) + 1e-12
        up, down = match.max(axis=1) / norm, -match.min(axis=1) / norm
        saw[j] = np.maximum(up, down) * np.sqrt(np.clip(fold[j], 0.0, None))
        saw_sign[j] = np.where(up >= down, 1.0, -1.0)  # +1: creeps right, jumps back left
    rows = np.arange(n)
    best_fold, best_saw = fold.argmax(axis=0), saw.argmax(axis=0)
    out["fold_strength"] = fold[best_fold, rows]
    out["fold_seconds"] = periods[best_fold] / fps
    out["saw_strength"] = saw[best_saw, rows]
    out["saw_seconds"] = periods[best_saw] / fps
    out["saw_shape"] = saw[best_saw, rows] / np.sqrt(np.clip(fold[best_saw, rows], 1e-6, None))
    # A ramp that creeps right has its big jumps going left: opposite signs agree.
    out["saw_direction"] = -saw_sign[best_saw, rows] * direction

    # Smoothed velocity: half a second of averaging removes most jitter.
    width = max(int(round(5 * scale)), 2)
    smooth = _moving_average(x, width)
    velocity = np.diff(smooth, axis=1)
    out["smooth_forward"] = np.maximum((velocity > 0).mean(axis=1), (velocity < 0).mean(axis=1))

    # Slope of gaze over sliding 1.5 s stretches.
    span = max(int(round(15 * scale)), 3)
    lever = np.arange(span) - (span - 1) / 2.0
    kernel = lever / (lever @ lever)
    slopes = np.stack([x[:, i : i + span] @ kernel for i in range(0, frames - span + 1, max(span // 3, 1))], axis=1)
    positive = (slopes > 0).mean(axis=1)
    out["slope_sign"] = np.maximum(positive, 1.0 - positive)
    centre = np.median(slopes, axis=1)
    spread = 1.4826 * np.median(np.abs(slopes - centre[:, None]), axis=1)
    out["slope_steadiness"] = np.clip(np.abs(centre) / (spread + 1e-12), 0.0, 20.0)

    ranks = np.argsort(np.argsort(y, axis=1), axis=1).astype(float)
    time = np.arange(frames) - (frames - 1) / 2.0
    ranks -= ranks.mean(axis=1, keepdims=True)
    out["y_rank_trend"] = np.abs(ranks @ time) / (np.linalg.norm(ranks, axis=1) * np.linalg.norm(time) + 1e-12)

    return {name: np.nan_to_num(out[name].astype(np.float64)) for name in FOLD_FEATURE_NAMES}


def _skew(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean(axis=1, keepdims=True)
    variance = (centred**2).mean(axis=1)
    return np.where(variance > 0, (centred**3).mean(axis=1) / np.maximum(variance, 1e-300) ** 1.5, 0.0)


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    total = np.cumsum(np.pad(values, ((0, 0), (1, 0))), axis=1)
    return (total[:, width:] - total[:, :-width]) / width
