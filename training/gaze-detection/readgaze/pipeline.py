"""From recordings to a feature table: one row per usable window.

Every row carries who it came from (`participant`, `recording`), what they were
doing (`task`, `label`: 1 = reading) and the window's features, plus the
rule-based score. Splits always group by `participant`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import synthetic
from .datasets import gazebase
from .detector import rule_score
from .features import FeatureConfig, compute_features
from .simulate import WebcamProfile, simulate_webcam
from .trace import GazeTrace


@dataclass(frozen=True)
class WindowConfig:
    length: float = 10.0
    """Seconds per window. Long enough for a few lines of reading."""
    hop: float = 5.0
    """Seconds between window starts (windows overlap by half)."""


def trace_rows(
    trace: GazeTrace,
    info: Mapping[str, Any],
    window: WindowConfig = WindowConfig(),
    features: FeatureConfig = FeatureConfig(),
) -> list[dict[str, Any]]:
    rows = []
    for part in trace.windows(window.length, window.hop):
        values = compute_features(part, features)
        if values is None:
            continue
        rows.append({**info, "window_start": part.meta["window_start"], **values, "rule_score": rule_score(values)})
    return rows


def gazebase_table(
    root: str,
    profile: WebcamProfile = WebcamProfile(),
    window: WindowConfig = WindowConfig(),
    features: FeatureConfig = FeatureConfig(),
    tasks: Iterable[str] | None = None,
    max_participants: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """GazeBase recordings, degraded to webcam quality, as a feature table."""
    task_order = list(gazebase.TASKS)
    rows: list[dict[str, Any]] = []
    for rec, trace in _limit_participants(gazebase.iter_recordings(root, tasks), max_participants):
        # One generator per recording, so results don't depend on loading order.
        rng = np.random.default_rng([seed, rec.round, rec.participant, rec.session, task_order.index(rec.task)])
        try:
            webcam = simulate_webcam(trace, profile, rng)
        except ValueError:
            continue  # shorter than one frame
        info = {
            "dataset": "GazeBase",
            "participant": f"gb{rec.participant:03d}",
            "recording": f"r{rec.round}-p{rec.participant:03d}-s{rec.session}-{rec.task}",
            "task": rec.task,
            "label": int(rec.is_reading),
        }
        rows += trace_rows(webcam, info, window, features)
    return pd.DataFrame(rows)


def synthetic_table(
    n_participants: int = 6,
    duration: float = 60.0,
    profile: WebcamProfile = WebcamProfile(),
    window: WindowConfig = WindowConfig(),
    features: FeatureConfig = FeatureConfig(),
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic stand-in for `gazebase_table`, for tests and smoke runs only."""
    rows: list[dict[str, Any]] = []
    for p in range(n_participants):
        for t_index, (task, make) in enumerate(synthetic.TASKS.items()):
            rng = np.random.default_rng([seed, p, t_index])
            webcam = simulate_webcam(make(duration, rng), profile, rng)
            info = {
                "dataset": "synthetic",
                "participant": f"syn{p:03d}",
                "recording": f"syn{p:03d}-{task}",
                "task": task,
                "label": int(task == "reading"),
            }
            rows += trace_rows(webcam, info, window, features)
    return pd.DataFrame(rows)


def _limit_participants(
    recordings: Iterator[tuple[gazebase.RecordingId, GazeTrace]], limit: int | None
) -> Iterator[tuple[gazebase.RecordingId, GazeTrace]]:
    seen: set[int] = set()
    for rec, trace in recordings:
        if limit is not None and rec.participant not in seen and len(seen) >= limit:
            continue
        seen.add(rec.participant)
        yield rec, trace
