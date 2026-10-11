import numpy as np
import pytest

from readgaze.evaluation import flag_rate_by_group, summarize, threshold_at_fpr
from readgaze.model import build_model, export_json, score_from_json


def test_threshold_keeps_false_positive_rate_at_or_below_target():
    rng = np.random.default_rng(0)
    negatives = rng.random(200)
    for target in (0.0, 0.01, 0.05, 0.2):
        threshold = threshold_at_fpr(negatives, target)
        assert np.mean(negatives >= threshold) <= target + 1e-12


def test_summarize_counts_at_the_threshold():
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.0])
    report = summarize(labels, scores, threshold=0.5)
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["false_positive_rate"] == pytest.approx(1 / 4)
    assert report["precision"] == pytest.approx(2 / 3)
    assert 0 < report["roc_auc"] <= 1


def test_flag_rate_by_group():
    rates = flag_rate_by_group(np.array(["a", "a", "b"]), np.array([0.9, 0.1, 0.9]), 0.5)
    assert rates == {"a": 0.5, "b": 1.0}


def test_exported_json_scores_exactly_like_sklearn():
    # The JSON is what the Rust engine will implement, so it must match.
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    x = rng.normal(size=(200, 3))
    y = (x @ [1.5, -1.0, 0.5] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = build_model().fit(x, y)
    exported = export_json(model, names, threshold=0.5)
    expected = model.predict_proba(x[:20])[:, 1]
    got = [score_from_json(exported, dict(zip(names, row))) for row in x[:20]]
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_rejects_unknown_model_versions():
    with pytest.raises(ValueError):
        score_from_json({"version": 99}, {})
