"""The evaluation protocol for the reading detector, in one place.

People are split once, by a fixed seed:

- development people (80%): everything that involves a choice happens here.
  Models, features and settings are compared by person-level
  cross-validation, and decision thresholds are picked on the out-of-fold
  scores.
- test people (20%): never used for any choice. The chosen model is trained
  on all development people and scored on them once, at thresholds that
  were fixed beforehand.

Trying many models on the same cross-validation makes the best one look a
little better than it is. The test people are what keeps the reported
number honest.

Accuracy needs care here. Only about one window in seven is reading, so a
detector that never flags anyone is 87% "accurate". The headline is
therefore balanced accuracy: the mean of the share of reading windows
caught and the share of non-reading windows left alone. It is the accuracy
you would see if reading and not-reading were equally common.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .evaluation import threshold_at_fpr

SPLIT_SEED = 2026
TEST_FRACTION = 0.2


def split_people(participants: Iterable[str], test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    """`(development, test)` participant ids. Depends only on the set of ids and the seed."""
    people = sorted(set(participants))
    order = np.random.default_rng(seed).permutation(len(people))
    n_test = int(round(test_fraction * len(people)))
    test = sorted(people[i] for i in order[:n_test])
    dev = sorted(people[i] for i in order[n_test:])
    return dev, test


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The threshold with the highest balanced accuracy (flag when `score >= threshold`)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_pos = np.cumsum(sorted_labels) / max(sorted_labels.sum(), 1)
    false_pos = np.cumsum(~sorted_labels) / max((~sorted_labels).sum(), 1)
    balanced = (true_pos + 1.0 - false_pos) / 2.0
    # Only cut between different scores.
    last_of_run = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    best = int(np.argmax(np.where(last_of_run, balanced, -1.0)))
    return float(sorted_scores[best])


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    tp = int((flagged & labels).sum())
    fp = int((flagged & ~labels).sum())
    recall = tp / n_pos if n_pos else float("nan")
    specificity = 1.0 - fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "n_reading": n_pos,
        "n_not_reading": n_neg,
        "threshold": float(threshold),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "accuracy": (tp + n_neg - fp) / (n_pos + n_neg),
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
        "roc_auc": float(roc_auc_score(labels, scores)) if n_pos and n_neg else float("nan"),
    }


def thresholds(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> dict[str, float]:
    """The two operating points, both picked on development scores only.

    - balanced: best balanced accuracy.
    - low_fpr: at most `target_fpr` of non-reading examples flagged, which is
      how the app should run (a flagged honest candidate is the costly error).
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    return {
        "balanced": best_balanced_threshold(labels, scores),
        "low_fpr": threshold_at_fpr(scores[~labels], target_fpr),
    }


def report(labels: np.ndarray, scores: np.ndarray, cuts: dict[str, float]) -> dict[str, Any]:
    """Metrics at each operating point in `cuts`."""
    return {name: metrics_at(labels, scores, cut) for name, cut in cuts.items()}


def flag_rates(groups: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    groups = np.asarray(groups)
    flagged = np.asarray(scores, dtype=float) >= threshold
    return {str(g): float(flagged[groups == g].mean()) for g in sorted(set(groups.tolist()))}
