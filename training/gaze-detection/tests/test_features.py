import numpy as np
import pytest

from readgaze import synthetic
from readgaze.detector import aggregate, rule_score
from readgaze.features import FEATURE_NAMES, analyse_sweeps, compute_features
from readgaze.simulate import WebcamProfile, simulate_webcam
from readgaze.trace import GazeTrace


def webcam(kind: str, seed: int = 0, duration: float = 10.0, **kwargs) -> GazeTrace:
    rng = np.random.default_rng(seed)
    return simulate_webcam(synthetic.TASKS[kind](duration, rng, **kwargs), WebcamProfile(), rng)


def mean_features(kind: str, seeds=range(8), **kwargs) -> dict[str, float]:
    rows = [compute_features(webcam(kind, s, **kwargs)) for s in seeds]
    return {name: float(np.mean([r[name] for r in rows if r])) for name in FEATURE_NAMES}


def test_returns_every_feature():
    features = compute_features(webcam("reading"))
    assert set(features) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())


def test_reading_has_one_way_regular_sweeps():
    reading = mean_features("reading")
    assert reading["velocity_skew"] > 1.0
    assert reading["sweep_asymmetry"] > 0.8
    assert 8 < reading["sweep_rate"] < 40  # a line every few seconds


def test_fixation_has_almost_no_sweeps():
    assert mean_features("fixation")["sweep_rate"] < 3


def test_back_and_forth_jumps_are_not_one_way():
    saccades = mean_features("saccades")
    assert saccades["sweep_rate"] > 8  # big regular jumps...
    assert saccades["sweep_asymmetry"] < 0.5  # ...but in both directions
    assert saccades["velocity_skew"] < 1.0


def test_reading_direction_does_not_matter():
    # Mirrored video and right-to-left scripts flip the sawtooth.
    ltr = mean_features("reading")
    rtl = mean_features("reading", right_to_left=True)
    for name in ("velocity_skew", "sweep_asymmetry", "sweep_rate"):
        assert rtl[name] == pytest.approx(ltr[name], rel=0.25)


def test_sweep_analysis_reports_times_and_direction():
    analysis = analyse_sweeps(webcam("reading", duration=20.0))
    assert len(analysis.times) >= 3
    assert np.all(np.diff(analysis.times) > 0)
    assert np.all(np.sign(analysis.amplitudes) == analysis.direction)


def test_window_with_too_little_gaze_is_skipped():
    t = np.arange(0, 10, 0.1)
    x = np.full(100, np.nan)
    x[:30] = 0.0
    assert compute_features(GazeTrace(t, x, x.copy())) is None


def test_rule_score_separates_reading_from_everything_else():
    scores = {kind: np.mean([rule_score(compute_features(webcam(kind, s))) for s in range(8)]) for kind in synthetic.TASKS}
    assert scores["reading"] > 0.7
    assert max(v for k, v in scores.items() if k != "reading") < 0.2


def test_aggregate_uses_the_top_half_of_windows():
    assert aggregate([0.9, 0.8, 0.1, 0.0]) == pytest.approx(0.85)
    assert aggregate([0.4]) == pytest.approx(0.4)
    assert np.isnan(aggregate([]))
