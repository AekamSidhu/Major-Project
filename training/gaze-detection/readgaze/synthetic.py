"""Synthetic 1000 Hz eye-movement traces, in degrees of visual angle.

These are for tests and smoke runs only, so the pipeline can be exercised
without downloading GazeBase. They follow textbook reading behaviour
(fixations of ~230 ms, small forward saccades, a return sweep per line) but are
no substitute for real data, and nothing here should be reported as a result.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .trace import GazeTrace

RATE = 1000.0


def reading(
    duration: float = 60.0,
    rng: np.random.Generator | None = None,
    *,
    line_width: float = 20.0,
    words_per_line: int = 10,
    line_height: float = 1.2,
    right_to_left: bool = False,
) -> GazeTrace:
    """Reading lines of text: forward steps along a line, then a return sweep."""
    rng = rng or np.random.default_rng()
    fixations: list[tuple[float, float, float]] = []
    elapsed, line = 0.0, 0
    while elapsed < duration:
        y = (line % 15) * line_height
        n_words = max(3, int(round(rng.normal(words_per_line, 2))))
        for i in range(n_words):
            x = -line_width / 2 + (i + rng.uniform(0.2, 0.8)) * line_width / n_words
            if fixations and rng.random() < 0.08 and i > 1:
                # Occasional short regression to re-read a word.
                x -= line_width / n_words
            dwell = max(0.08, rng.normal(0.23, 0.06))
            fixations.append((dwell, x, y))
            elapsed += dwell
        line += 1
    trace = _render(fixations, rng).slice(0.0, duration)
    if right_to_left:
        trace = GazeTrace(trace.t, -trace.x, trace.y)
    return _tag(trace, "reading")


def fixation(duration: float = 60.0, rng: np.random.Generator | None = None) -> GazeTrace:
    """Holding gaze on one spot, with occasional small corrective jumps."""
    rng = rng or np.random.default_rng()
    fixations: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while elapsed < duration:
        dwell = rng.uniform(1.0, 3.0)
        fixations.append((dwell, rng.normal(0, 0.3), rng.normal(0, 0.3)))
        elapsed += dwell
    return _tag(_render(fixations, rng).slice(0.0, duration), "fixation")


def saccades(duration: float = 60.0, rng: np.random.Generator | None = None, amplitude: float = 15.0) -> GazeTrace:
    """Regular left-right jumps (like GazeBase's horizontal saccade task).

    A hard negative: regular horizontal movement, but symmetric, with no slow
    drift in one direction.
    """
    rng = rng or np.random.default_rng()
    fixations: list[tuple[float, float, float]] = []
    elapsed, side = 0.0, 1.0
    while elapsed < duration:
        dwell = rng.uniform(0.9, 1.2)
        fixations.append((dwell, side * amplitude / 2, rng.normal(0, 0.2)))
        side = -side
        elapsed += dwell
    return _tag(_render(fixations, rng).slice(0.0, duration), "saccades")


def free_viewing(duration: float = 60.0, rng: np.random.Generator | None = None, spread: float = 10.0) -> GazeTrace:
    """Looking around a scene (like watching a video): jumps in random directions."""
    rng = rng or np.random.default_rng()
    fixations: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while elapsed < duration:
        dwell = max(0.1, rng.gamma(2.0, 0.18))
        fixations.append((dwell, rng.normal(0, spread), rng.normal(0, spread * 0.6)))
        elapsed += dwell
    return _tag(_render(fixations, rng).slice(0.0, duration), "free_viewing")


TASKS: dict[str, Callable[..., GazeTrace]] = {
    "reading": reading,
    "fixation": fixation,
    "saccades": saccades,
    "free_viewing": free_viewing,
}
"""Synthetic stand-ins for GazeBase tasks; only "reading" is a positive."""


def _render(
    fixations: list[tuple[float, float, float]],
    rng: np.random.Generator,
    jitter: float = 0.05,
    saccade_seconds: float = 0.03,
) -> GazeTrace:
    """Fixations `(dwell, x, y)` joined by straight-line saccades, sampled at 1000 Hz."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    previous: tuple[float, float] | None = None
    for dwell, x, y in fixations:
        if previous is not None:
            steps = np.linspace(0.0, 1.0, max(int(saccade_seconds * RATE), 2) + 1)[1:-1]
            xs.append(previous[0] + (x - previous[0]) * steps)
            ys.append(previous[1] + (y - previous[1]) * steps)
        n = max(int(dwell * RATE), 1)
        xs.append(x + rng.normal(0.0, jitter, n))
        ys.append(y + rng.normal(0.0, jitter, n))
        previous = (x, y)
    x_all = np.concatenate(xs)
    return GazeTrace(np.arange(len(x_all)) / RATE, x_all, np.concatenate(ys))


def _tag(trace: GazeTrace, kind: str) -> GazeTrace:
    return GazeTrace(trace.t, trace.x, trace.y, {"synthetic": kind})
