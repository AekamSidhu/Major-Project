"""End-to-end smoke tests on synthetic data. They check the plumbing, not accuracy."""

import json

import pandas as pd
import pytest

import evaluate
import prepare
import train
from readgaze.features import FEATURE_NAMES
from readgaze.pipeline import synthetic_table
from readgaze.training import cross_validate


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synthetic_table(n_participants=6, duration=30.0)


def test_feature_table_has_one_row_per_window(table):
    assert {"participant", "recording", "task", "label", "window_start", "rule_score", *FEATURE_NAMES} <= set(table.columns)
    assert set(table["label"]) == {0, 1}
    assert table["participant"].nunique() == 6


def test_cross_validation_never_puts_a_person_in_train_and_test(table, monkeypatch):
    from sklearn.model_selection import StratifiedGroupKFold

    seen = []
    original = StratifiedGroupKFold.split

    def spy(self, x, y, groups):
        for train_idx, test_idx in original(self, x, y, groups):
            seen.append((set(groups.iloc[train_idx]), set(groups.iloc[test_idx])))
            yield train_idx, test_idx

    monkeypatch.setattr(StratifiedGroupKFold, "split", spy)
    cross_validate(table, n_splits=3)
    assert seen and all(not (tr & te) for tr, te in seen)


def test_report_has_both_detectors_at_both_levels(table):
    scored, report = cross_validate(table, n_splits=3)
    assert scored["model_score"].between(0, 1).all()
    for name in ("model_score", "rule_score"):
        for level in ("windows", "recordings"):
            assert report[name][level]["false_positive_rate"] <= report["target_false_positive_rate"] + 1e-9


def test_needs_enough_people_for_the_folds(table):
    with pytest.raises(ValueError):
        cross_validate(table[table["participant"] == "syn000"], n_splits=3)


def test_command_line_pipeline(tmp_path):
    features = tmp_path / "features.csv"
    prepare.main(["--synthetic", "6", "--out", str(features)])
    assert json.loads(features.with_suffix(".json").read_text())["source"] == "synthetic"

    results = tmp_path / "results"
    train.main(["--features", str(features), "--out", str(results), "--folds", "3"])
    model = json.loads((results / "model.json").read_text())
    assert model["features"] == list(FEATURE_NAMES)
    assert (results / "cv_report.json").exists()

    evaluate.main(["--model", str(results / "model.json"), "--features", str(features), "--out", str(tmp_path / "eval.json")])
    assert "recordings" in json.loads((tmp_path / "eval.json").read_text())
