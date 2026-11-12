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
from .simulate import ProfileRange, WebcamProfile, simulate_webcam
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


def cache_windows(
    cache_path: str,
    profile: WebcamProfile = WebcamProfile(),
    window: WindowConfig = WindowConfig(),
    features: FeatureConfig = FeatureConfig(),
    seed: int = 0,
    workers: int = 4,
    extended: bool = False,
    clean: bool = False,
    vary: ProfileRange | None = None,
    speed: float = 1.0,
    participants: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """A cached dataset (`build_cache.py`) as a feature table plus the raw windows.

    Each recording is degraded to webcam quality with `profile` (its own
    random generator, so the result doesn't depend on how the work is split)
    and cut into windows. Row `i` of the table describes `raw[i]`, a
    `(frames, 2)` array of x and y with NaN where the frame was lost.

    `extended` adds the features of `features_ext`. `clean` skips the
    degradation and the features: just the clean windows and who they belong
    to, for models that add their own noise while training. With `vary`,
    every recording draws its own noise, drift and movement size from those
    ranges (recorded in the `sim_*` columns) instead of using `profile`'s.
    `speed` replays the eye movements slower or faster (`TraceCache.trace`).
    `participants` limits the table to those people.
    """
    from concurrent.futures import ProcessPoolExecutor

    from .cache import TraceCache

    meta = TraceCache.load(cache_path).meta
    wanted = np.arange(len(meta))
    if participants is not None:
        wanted = np.flatnonzero(meta["participant"].isin(set(participants)).to_numpy())
    chunks = [wanted[i : i + 150].tolist() for i in range(0, len(wanted), 150)]
    jobs = [(cache_path, chunk, profile, window, features, seed, extended, clean, vary, speed) for chunk in chunks]
    if workers <= 1:
        parts = [_cache_rows(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_cache_rows, jobs))
    rows = [row for part in parts for row in part[0]]
    raw = np.concatenate([part[1] for part in parts if len(part[1])]) if rows else np.zeros((0, 0, 2), dtype=np.float32)
    return pd.DataFrame(rows), raw


def cache_table(cache_path: str, profile: WebcamProfile = WebcamProfile(), **kwargs: Any) -> pd.DataFrame:
    """`cache_windows` without the raw windows."""
    return cache_windows(cache_path, profile, **kwargs)[0]


_LOADED: dict[str, Any] = {}


def _cache_rows(job) -> tuple[list[dict[str, Any]], np.ndarray]:
    from .cache import TraceCache
    from .features_ext import extended_features
    from .simulate import degrade

    cache_path, indices, base_profile, window, features, seed, extended, clean, vary, speed = job
    # Each worker process keeps the cache it has loaded: jobs are small.
    if cache_path not in _LOADED:
        _LOADED.clear()
        _LOADED[cache_path] = TraceCache.load(cache_path)
    cache = _LOADED[cache_path]
    frames = int(round(window.length * base_profile.fps))
    rows: list[dict[str, Any]] = []
    raw: list[np.ndarray] = []
    for index in indices:
        info = cache.meta.iloc[index].to_dict()
        try:
            trace = cache.trace(index, base_profile.fps, speed)
        except ValueError:
            continue
        if len(trace) < frames:
            continue
        if not clean:
            # One generator per recording, so results don't depend on chunking.
            rng = np.random.default_rng([seed, index])
            profile = vary.draw(base_profile, rng) if vary is not None else base_profile
            trace = degrade(trace, profile, rng)
            info.update(sim_noise=profile.noise, sim_drift=profile.drift, sim_amplitude=profile.amplitude)
        for part in trace.windows(window.length, window.hop):
            if len(part) != frames:
                continue
            row = {**info, "window_start": part.meta["window_start"]}
            if clean:
                if part.valid_fraction < features.min_valid_fraction:
                    continue
            else:
                values = compute_features(part, features)
                if values is None:
                    continue
                row.update(values)
                row["rule_score"] = rule_score(values)
                if extended:
                    row.update(extended_features(part, features))
            rows.append(row)
            raw.append(np.stack([part.x, part.y], axis=1).astype(np.float32))
    return rows, (np.stack(raw) if raw else np.zeros((0, frames, 2), dtype=np.float32))


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
