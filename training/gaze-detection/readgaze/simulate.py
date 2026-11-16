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

from dataclasses import asdict, dataclass, replace

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


@dataclass(frozen=True)
class ProfileRange:
    """Ranges to draw each recording's noise, drift and movement size from.

    A detector trained at one fixed noise level can learn that level instead
    of the reading pattern. Training on a spread of conditions, from a clean
    full-screen face to a small compressed tile with the text near the
    camera, avoids that.
    """

    noise: tuple[float, float] = (0.3, 3.5)
    drift: tuple[float, float] = (0.0, 2.0)
    amplitude: tuple[float, float] = (0.4, 1.3)

    def draw(self, base: WebcamProfile, rng: np.random.Generator) -> WebcamProfile:
        low, high = np.log(self.amplitude[0]), np.log(self.amplitude[1])
        return replace(
            base,
            noise=float(rng.uniform(*self.noise)),
            drift=float(rng.uniform(*self.drift)),
            amplitude=float(np.exp(rng.uniform(low, high))),
        )


def simulate_webcam(trace: GazeTrace, profile: WebcamProfile, rng: np.random.Generator) -> GazeTrace:
    return degrade(resample(trace, profile.fps), profile, rng)


def degrade(cam: GazeTrace, profile: WebcamProfile, rng: np.random.Generator) -> GazeTrace:
    """Add webcam-level noise, drift, blinks and dropouts to a trace that is
    already at the camera's frame rate (`profile.fps` is used for timing only)."""
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

    return GazeTrace(cam.t, x, y, {**cam.meta, "webcam": asdict(profile)})


def _smooth_noise(n: int, std: float, width_frames: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise: white noise through a moving average, rescaled to `std`."""
    width = max(int(round(width_frames)), 1)
    raw = rng.normal(0.0, 1.0, n + 2 * width)
    smooth = np.convolve(raw, np.ones(width) / width, mode="same")[width : width + n]
    spread = smooth.std()
    return smooth * (std / spread) if spread > 0 else smooth
