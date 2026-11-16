"""Rule-based reading detector: the explainable baseline.

Three conditions, each softened with a logistic curve so the score moves
smoothly instead of flipping at a hard cut-off:

1. enough return sweeps per minute,
2. a lopsided horizontal velocity (small steps one way, big jumps the other),
3. big jumps going one way only.

The score is their product, so all three have to hold. The defaults are
starting points; `train.py` evaluates this detector next to the learned one
and picks the decision threshold on the score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuleConfig:
    min_sweep_rate: float = 6.0
    """Sweeps per minute. Reading a short answer line by line gives roughly
    10–30 per minute; the 6 leaves room for long lines and slow reading."""
    sweep_rate_softness: float = 2.0
    min_velocity_skew: float = 1.0
    velocity_skew_softness: float = 0.4
    min_sweep_asymmetry: float = 0.4
    sweep_asymmetry_softness: float = 0.15


def rule_score(features: Mapping[str, float], config: RuleConfig = RuleConfig()) -> float:
    """Reading score in [0, 1] for one window."""
    rate = _soft_above(features["sweep_rate"], config.min_sweep_rate, config.sweep_rate_softness)
    lopsided = _soft_above(features["velocity_skew"], config.min_velocity_skew, config.velocity_skew_softness)
    one_way = _soft_above(features["sweep_asymmetry"], config.min_sweep_asymmetry, config.sweep_asymmetry_softness)
    return float(rate * lopsided * one_way)


def aggregate(window_scores: Sequence[float]) -> float:
    """One score per answer from its window scores: the mean of the top half.

    A candidate who reads for most of an answer but glances up now and then
    should still score high, so the lowest windows are ignored. A single
    reading-like window in a long answer, on the other hand, isn't enough.
    """
    scores = np.sort(np.asarray([s for s in window_scores if np.isfinite(s)], dtype=float))[::-1]
    if len(scores) == 0:
        return float("nan")
    top = scores[: max(1, int(np.ceil(len(scores) / 2)))]
    return float(top.mean())


def _soft_above(value: float, threshold: float, softness: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(value - threshold) / softness)))
