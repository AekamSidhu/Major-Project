"""More reading features, for the gradient-boosted detector.

The six features in `features.py` describe the return sweeps. These add what
happens around them: how straight the creep along a line is, whether the
sweeps come with a small step down to the next line, whether the pattern
repeats, and where the movement energy sits in frequency. Like the first
six they are unit-free, and the reading direction is never assumed: steps
are measured relative to whichever way the big jumps go.

`extended_features` returns only the new values. Use it together with
`features.compute_features`; `ALL_FEATURE_NAMES` is the combined order.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES, FeatureConfig, _jump_events, _prepare, _return_direction, _robust_std
from .trace import GazeTrace, fill_gaps

EXT_FEATURE_NAMES: tuple[str, ...] = (
    "snr_x",
    "snr_y",
    "still_fraction",
    "step_median",
    "step_q25",
    "step_q75",
    "step_min",
    "step_max",
    "step_kurtosis",
    "velocity_autocorr1",
    "velocity_autocorr2",
    "ramp_r2",
    "ramp_slope_cv",
    "ramp_forward",
    "ramp_residual",
    "period_strength",
    "period_seconds",
    "power_low",
    "power_mid",
    "power_high",
    "sweep_step_down",
    "sweep_step_consistency",
    "y_trend",
    "x_trend",
    "sweep_amplitude_cv",
    "sweep_amplitude_share",
    "line_start_spread",
    "line_end_spread",
    "reversal_rate",
    "path_efficiency",
    "horizontal_energy",
    "jump_rate",
    "forward_time",
    "x_entropy",
    "y_entropy",
    "valid_fraction",
)

ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + EXT_FEATURE_NAMES


def extended_features(window: GazeTrace, config: FeatureConfig = FeatureConfig()) -> dict[str, float]:
    """The extra features for one window. All zeros-ish defaults if unusable."""
    prepared = _prepare(window, config)
    if prepared is None:
        return dict.fromkeys(EXT_FEATURE_NAMES, 0.0)
    t, x, y = prepared.t, prepared.x, prepared.y
    noise_x = max(prepared.noise, 1e-9)
    raw_dy = np.diff(fill_gaps(window, config.max_gap).y)
    raw_dy = raw_dy[np.isfinite(raw_dy)]
    noise_y = max(_robust_std(raw_dy), 1e-9) if len(raw_dy) else 1e-9

    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys, ts = x[ok], y[ok], t[ok]
    dx_all = np.diff(x)
    dy_all = np.diff(y)
    step_ok = np.isfinite(dx_all) & np.isfinite(dy_all)
    dx, dy = dx_all[step_ok], dy_all[step_ok]
    seconds = max(window.duration, 1e-9)
    fps = (len(t) - 1) / seconds

    direction = _return_direction(dx)
    forward = -direction * dx / noise_x  # steps along the line are positive
    span_x = np.percentile(xs, 95) - np.percentile(xs, 5)
    span_y = np.percentile(ys, 95) - np.percentile(ys, 5)

    out: dict[str, float] = {}
    out["snr_x"] = _clip(np.log((span_x + 1e-9) / noise_x), -3, 8)
    out["snr_y"] = _clip(np.log((span_y + 1e-9) / noise_y), -3, 8)
    out["still_fraction"] = float(np.mean(np.abs(dx) < noise_x))
    out["step_median"] = _clip(np.median(forward), -20, 20)
    out["step_q25"] = _clip(np.percentile(forward, 25), -20, 20)
    out["step_q75"] = _clip(np.percentile(forward, 75), -20, 20)
    out["step_min"] = _clip(forward.min(), -60, 60)
    out["step_max"] = _clip(forward.max(), -60, 60)
    out["step_kurtosis"] = _clip(_kurtosis(dx), -3, 50)
    out["velocity_autocorr1"] = _autocorr(dx, 1)
    out["velocity_autocorr2"] = _autocorr(dx, 2)

    # Return sweeps, as in features._analyse, but keeping their positions.
    threshold = config.sweep_threshold * noise_x
    events = [e for e in _jump_events(dx_all, threshold) if abs(e[2]) >= config.min_sweep_span * span_x]
    sweeps = [e for e in events if np.sign(e[2]) == direction]

    out.update(_ramps(t, x, sweeps, direction, noise_x))
    out.update(_periodicity(xs, fps))
    out.update(_spectrum(xs, fps))
    out.update(_vertical_steps(y, sweeps, fps, span_y))

    out["y_trend"] = abs(_corr(ys, ts))
    out["x_trend"] = abs(_corr(xs, ts))

    amplitudes = np.abs([e[2] for e in sweeps])
    out["sweep_amplitude_cv"] = float(amplitudes.std() / amplitudes.mean()) if len(amplitudes) >= 2 else 0.0
    out["sweep_amplitude_share"] = _clip(amplitudes.mean() / (span_x + 1e-9), 0, 3) if len(amplitudes) else 0.0
    starts = np.array([x[min(e[1] + 1, len(x) - 1)] for e in sweeps])
    ends = np.array([x[e[0]] for e in sweeps])
    out["line_start_spread"] = _clip(np.nanstd(starts) / (span_x + 1e-9), 0, 3) if len(starts) >= 2 else 0.0
    out["line_end_spread"] = _clip(np.nanstd(ends) / (span_x + 1e-9), 0, 3) if len(ends) >= 2 else 0.0

    moving = dx[np.abs(dx) > noise_x]
    out["reversal_rate"] = float(np.sum(np.diff(np.sign(moving)) != 0) / seconds) if len(moving) > 1 else 0.0
    out["path_efficiency"] = float(abs(dx.sum()) / (np.abs(dx).sum() + 1e-9))
    out["horizontal_energy"] = float(np.sum(dx**2) / (np.sum(dx**2) + np.sum(dy**2) + 1e-12))
    out["jump_rate"] = float(len(events) / seconds)
    smooth = np.convolve(forward, np.ones(3) / 3, mode="valid") if len(forward) >= 3 else forward
    out["forward_time"] = float(np.mean(smooth > 0.25))
    out["x_entropy"] = _entropy(xs, 10)
    out["y_entropy"] = _entropy(ys, 6)
    out["valid_fraction"] = float(window.valid_fraction)
    return out


def _ramps(t: np.ndarray, x: np.ndarray, sweeps: list, direction: int, noise_x: float) -> dict[str, float]:
    """How well a straight line fits the gaze between consecutive return sweeps."""
    cuts = [0] + [e[1] + 1 for e in sweeps] + [len(x)]
    starts = [0] + [e[0] + 1 for e in sweeps]  # a segment ends where the next sweep starts
    ss_res = ss_tot = 0.0
    slopes, weights = [], []
    for k in range(len(cuts) - 1):
        a = cuts[k]
        b = starts[k + 1] if k + 1 < len(starts) else len(x)
        seg_t, seg_x = t[a:b], x[a:b]
        keep = np.isfinite(seg_x)
        if keep.sum() < 4:
            continue
        seg_t, seg_x = seg_t[keep], seg_x[keep]
        tc = seg_t - seg_t.mean()
        denom = float(tc @ tc)
        slope = float(tc @ (seg_x - seg_x.mean()) / denom) if denom > 0 else 0.0
        resid = seg_x - seg_x.mean() - slope * tc
        ss_res += float(resid @ resid)
        ss_tot += float(((seg_x - seg_x.mean()) ** 2).sum())
        slopes.append(slope)
        weights.append(len(seg_x))
    if not slopes:
        return {"ramp_r2": 0.0, "ramp_slope_cv": 0.0, "ramp_forward": 0.5, "ramp_residual": 0.0}
    slopes_a, weights_a = np.asarray(slopes), np.asarray(weights, dtype=float)
    n = weights_a.sum()
    mean_slope = float(slopes_a @ weights_a / n)
    return {
        "ramp_r2": _clip(1.0 - ss_res / ss_tot, -1, 1) if ss_tot > 0 else 0.0,
        "ramp_slope_cv": _clip(slopes_a.std() / (abs(mean_slope) + 1e-9), 0, 10) if len(slopes) >= 2 else 0.0,
        "ramp_forward": float(weights_a[np.sign(slopes_a) == -direction].sum() / n),
        "ramp_residual": _clip(np.log(np.sqrt(ss_res / n) / noise_x + 1e-9), -5, 5),
    }


def _periodicity(xs: np.ndarray, fps: float) -> dict[str, float]:
    """The strongest repeat of the horizontal trace between 1 s and half the window."""
    centred = xs - xs.mean()
    energy = float(centred @ centred)
    lo, hi = int(round(1.0 * fps)), len(centred) // 2
    if energy <= 0 or hi <= lo:
        return {"period_strength": 0.0, "period_seconds": 0.0}
    acf = np.array([float(centred[:-lag] @ centred[lag:]) for lag in range(lo, hi + 1)]) / energy
    best = int(np.argmax(acf))
    return {"period_strength": _clip(acf[best], -1, 1), "period_seconds": (lo + best) / fps}


def _spectrum(xs: np.ndarray, fps: float) -> dict[str, float]:
    """Share of horizontal movement power that is slow, medium and fast."""
    centred = (xs - xs.mean()) * np.hanning(len(xs))
    power = np.abs(np.fft.rfft(centred)) ** 2
    freq = np.fft.rfftfreq(len(centred), d=1.0 / fps)
    total = power[1:].sum()
    if total <= 0:
        return {"power_low": 0.0, "power_mid": 0.0, "power_high": 0.0}
    band = lambda lo, hi: float(power[(freq >= lo) & (freq < hi)].sum() / total)  # noqa: E731
    return {"power_low": band(0.05, 0.6), "power_mid": band(0.6, 1.5), "power_high": band(1.5, fps)}


def _vertical_steps(y: np.ndarray, sweeps: list, fps: float, span_y: float) -> dict[str, float]:
    """Does gaze drop by a line at each return sweep? Mean vertical change
    across the sweeps (half a second before against half a second after)."""
    half = max(int(round(0.5 * fps)), 1)
    changes = []
    for first, last, _ in sweeps:
        before = y[max(first - half + 1, 0) : first + 1]
        after = y[last + 1 : last + 1 + half]
        if np.isfinite(before).any() and np.isfinite(after).any():
            changes.append(np.nanmean(after) - np.nanmean(before))
    if not changes:
        return {"sweep_step_down": 0.0, "sweep_step_consistency": 0.0}
    changes_a = np.asarray(changes)
    return {
        "sweep_step_down": _clip(abs(changes_a.mean()) / (span_y + 1e-9), 0, 3),
        "sweep_step_consistency": float(abs(changes_a.mean()) / (np.abs(changes_a).mean() + 1e-9)),
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi)) if np.isfinite(value) else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _autocorr(values: np.ndarray, lag: int) -> float:
    return _corr(values[:-lag], values[lag:]) if len(values) > lag + 2 else 0.0


def _kurtosis(values: np.ndarray) -> float:
    spread = values.std()
    if spread == 0:
        return 0.0
    return float(np.mean(((values - values.mean()) / spread) ** 4) - 3.0)


def _entropy(values: np.ndarray, bins: int) -> float:
    """Normalised entropy of where gaze sat: 1 = spread evenly, 0 = one spot."""
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return 0.0
    counts, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))
