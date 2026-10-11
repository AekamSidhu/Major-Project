"""Metrics, with the false-positive rate first.

The number that matters most is how often an honest candidate gets flagged.
So the decision threshold is chosen for a target false-positive rate on
honest (non-reading) examples, and everything else is reported at that
threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate is at most `target_fpr`.

    A score counts as a flag when `score >= threshold`.
    """
    scores = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if len(scores) == 0:
        raise ValueError("need at least one negative score")
    allowed = int(np.floor(target_fpr * len(scores)))  # negatives we may flag
    if allowed >= len(scores):
        return float(scores[-1])
    # Just above the first negative we must not flag.
    return float(np.nextafter(scores[allowed], np.inf))


def summarize(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    positives, negatives = labels.sum(), (~labels).sum()
    true_pos = int((flagged & labels).sum())
    false_pos = int((flagged & ~labels).sum())
    has_both = positives > 0 and negatives > 0
    return {
        "n_positive": int(positives),
        "n_negative": int(negatives),
        "roc_auc": float(roc_auc_score(labels, scores)) if has_both else None,
        "threshold": float(threshold),
        "false_positive_rate": false_pos / negatives if negatives else None,
        "recall": true_pos / positives if positives else None,
        "precision": true_pos / (true_pos + false_pos) if (true_pos + false_pos) else None,
    }


def flag_rate_by_group(groups: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Share of examples flagged in each group (e.g. per task, or per subgroup)."""
    groups = np.asarray(groups)
    flagged = np.asarray(scores, dtype=float) >= threshold
    return {str(g): float(flagged[groups == g].mean()) for g in sorted(set(groups.tolist()))}
