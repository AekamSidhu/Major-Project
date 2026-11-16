"""Cross-validated training with person-level splits.

Nobody appears in both the training and the test part of a fold
(`StratifiedGroupKFold` grouped by participant), so the scores show how the
detector does on people it has never seen.

Two levels are reported:

- windows: each 10 s window on its own;
- recordings: windows combined with `detector.aggregate`, which is how the app
  will score a whole answer. The exported threshold is at this level.

Thresholds are picked on the same out-of-fold scores they are reported on,
which flatters the false-positive rate slightly. `evaluate.py` on held-out
data (our own recordings, OEP) gives the honest number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .detector import aggregate
from .evaluation import flag_rate_by_group, summarize, threshold_at_fpr
from .features import FEATURE_NAMES
from .model import build_model, export_json

SCORES = ("model_score", "rule_score")


def cross_validate(
    table: pd.DataFrame,
    n_splits: int = 5,
    target_fpr: float = 0.05,
    c: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Out-of-fold model scores for every window, and the report."""
    _check(table, n_splits)
    x = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = table["label"].to_numpy(dtype=int)
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(table), np.nan)
    for train, test in folds.split(x, y, groups=table["participant"]):
        model = build_model(c).fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]

    scored = table.assign(model_score=oof)
    report: dict[str, Any] = {
        "folds": n_splits,
        "target_false_positive_rate": target_fpr,
        "participants": int(table["participant"].nunique()),
        "recordings": int(table["recording"].nunique()),
        "windows": int(len(table)),
    }
    for name in SCORES:
        report[name] = score_report(scored, name, target_fpr)
    return scored, report


def score_report(scored: pd.DataFrame, column: str, target_fpr: float) -> dict[str, Any]:
    """Window- and recording-level metrics for one score column."""
    labels = scored["label"].to_numpy()
    scores = scored[column].to_numpy(dtype=float)
    window_threshold = threshold_at_fpr(scores[labels == 0], target_fpr)

    recordings = recording_scores(scored, column)
    rec_labels = recordings["label"].to_numpy()
    rec_scores = recordings[column].to_numpy(dtype=float)
    rec_threshold = threshold_at_fpr(rec_scores[rec_labels == 0], target_fpr)
    return {
        "windows": summarize(labels, scores, window_threshold),
        "recordings": summarize(rec_labels, rec_scores, rec_threshold),
        "recording_flag_rate_by_task": flag_rate_by_group(recordings["task"].to_numpy(), rec_scores, rec_threshold),
    }


def recording_scores(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per recording: its label, task and aggregated score."""
    return (
        scored.groupby("recording")
        .agg(label=("label", "first"), task=("task", "first"), participant=("participant", "first"), **{column: (column, aggregate)})
        .reset_index()
    )


def fit_final(table: pd.DataFrame, threshold: float, c: float = 1.0, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fit on every window and export the model as JSON with the chosen threshold."""
    model = build_model(c).fit(table[list(FEATURE_NAMES)].to_numpy(dtype=float), table["label"].to_numpy(dtype=int))
    return export_json(model, FEATURE_NAMES, threshold, metadata)


def _check(table: pd.DataFrame, n_splits: int) -> None:
    missing = {"participant", "recording", "task", "label", "rule_score", *FEATURE_NAMES} - set(table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {sorted(missing)}")
    if table["label"].nunique() < 2:
        raise ValueError("feature table needs both reading and non-reading windows")
    if table["participant"].nunique() < n_splits:
        raise ValueError(f"need at least {n_splits} participants for {n_splits} folds")
