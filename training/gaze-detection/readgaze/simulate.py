"""Make a lab eye-tracker trace look like what a webcam would give us.

Public reading datasets are recorded at 1000 Hz with a chin rest. In a video
call we get ~10 fps from a compressed video tile, with head movement, blinks
and lost frames. `simulate_webcam` bridges that gap so a detector developed on
GazeBase is tested on realistic input.

The default noise levels are placeholders. Phase 1 (validating MediaPipe
against ground truth) measures the real jitter, and these defaults should be
updated to match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .trace import GazeTrace, resample


@dataclass(frozen=True)
class WebcamProfile:
    fps: float = 10.0
    """Frame rate of the captured candidate tile."""
    noise: float = 1.5
    """Per-frame measurement jitter (standard deviation, in trace units)."""
    drift: float = 1.0
    """Size of slow head-movement drift (standard deviation, in trace units)."""
    drift_seconds: float = 4.0
    """Time scale of the drift."""
    amplitude: float = 1.0
    """Scales eye movements relative to the noise. Below 1 simulates text
    placed close to the camera, which shrinks reading sweeps."""
    blinks_per_minute: float = 15.0
    blink_seconds: float = 0.25
    dropout_probability: float = 0.02
    """Chance that any single frame loses the face (motion blur, compression)."""


def simulate_webcam(trace: GazeTrace, profile: WebcamProfile, rng: np.random.Generator) -> GazeTrace:
    cam = resample(trace, profile.fps)
    n = len(cam)
    x, y = cam.x.copy(), cam.y.copy()

    if profile.amplitude != 1.0:
        x = np.nanmedian(x) + profile.amplitude * (x - np.nanmedian(x))
        y = np.nanmedian(y) + profile.amplitude * (y - np.nanmedian(y))

    if profile.drift > 0:
        x += _smooth_noise(n, profile.drift, profile.drift_seconds * profile.fps, rng)
        y += _smooth_noise(n, profile.drift, profile.drift_seconds * profile.fps, rng)

    if profile.noise > 0:
        x += rng.normal(0.0, profile.noise, n)
        y += rng.normal(0.0, profile.noise, n)

    lost = rng.random(n) < profile.dropout_probability
    duration = n / profile.fps
    for _ in range(rng.poisson(profile.blinks_per_minute * duration / 60.0)):
        start = rng.uniform(0.0, duration)
        lost |= (cam.t - cam.t[0] >= start) & (cam.t - cam.t[0] < start + profile.blink_seconds)
    x[lost] = np.nan
    y[lost] = np.nan

    return GazeTrace(cam.t, x, y, {**trace.meta, "webcam": asdict(profile)})


def _smooth_noise(n: int, std: float, width_frames: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise: white noise through a moving average, rescaled to `std`."""
    width = max(int(round(width_frames)), 1)
    raw = rng.normal(0.0, 1.0, n + 2 * width)
    smooth = np.convolve(raw, np.ones(width) / width, mode="same")[width : width + n]
    spread = smooth.std()
    return smooth * (std / spread) if spread > 0 else smooth
