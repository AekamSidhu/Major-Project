import numpy as np
import pytest

from readgaze import protocol


def test_people_split_is_fixed_and_disjoint():
    people = [f"p{i:03d}" for i in range(100)]
    dev, test = protocol.split_people(people)
    assert len(test) == 20 and len(dev) == 80
    assert not set(dev) & set(test)
    # Same answer whatever order or duplicates the ids arrive in.
    assert protocol.split_people(people[::-1] + people) == (dev, test)


def test_best_balanced_threshold_separates_clean_scores():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.35, 0.6, 0.9])
    cut = protocol.best_balanced_threshold(labels, scores)
    assert 0.35 < cut <= 0.6
    assert protocol.metrics_at(labels, scores, cut)["balanced_accuracy"] == 1.0


def test_metrics_do_not_reward_never_flagging():
    labels = np.r_[np.zeros(90), np.ones(10)]
    scores = np.zeros(100)  # a detector that never flags anyone
    m = protocol.metrics_at(labels, scores, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["recall"] == 0.0 and m["false_positive_rate"] == 0.0


def test_low_fpr_threshold_respects_the_target():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(1000), np.ones(200)]
    scores = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 200)]
    cuts = protocol.thresholds(labels, scores, target_fpr=0.05)
    assert protocol.metrics_at(labels, scores, cuts["low_fpr"])["false_positive_rate"] <= 0.05 + 1e-9
    assert cuts["balanced"] < cuts["low_fpr"]
