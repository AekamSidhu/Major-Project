import numpy as np
import pytest

from readgaze import synthetic
from readgaze.features_ext import ALL_FEATURE_NAMES, EXT_FEATURE_NAMES, extended_features
from readgaze.simulate import WebcamProfile, simulate_webcam
from readgaze.trace import GazeTrace

QUIET = WebcamProfile(noise=0.3, drift=0.2, blinks_per_minute=0, dropout_probability=0)


def window(kind: str, seed: int = 0, profile: WebcamProfile = QUIET) -> GazeTrace:
    rng = np.random.default_rng(seed)
    return next(simulate_webcam(synthetic.TASKS[kind](12.0, rng), profile, rng).windows(10.0, 5.0))


def test_every_feature_is_present_and_finite():
    for kind in synthetic.TASKS:
        values = extended_features(window(kind, profile=WebcamProfile()))
        assert tuple(values) == EXT_FEATURE_NAMES
        assert all(np.isfinite(v) for v in values.values())
    assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES)) == 42


def test_reading_shows_straight_forward_ramps_and_a_step_down_per_line():
    reading = extended_features(window("reading"))
    fixation = extended_features(window("fixation"))
    assert reading["ramp_r2"] > 0.8 > fixation["ramp_r2"]
    assert reading["ramp_forward"] > 0.9
    assert reading["sweep_step_consistency"] > fixation["sweep_step_consistency"]
    assert reading["snr_x"] > fixation["snr_x"]


def test_features_ignore_units_and_reading_direction():
    w = window("reading")
    base = extended_features(w)
    scaled = extended_features(GazeTrace(w.t, w.x * 250.0, w.y * 250.0))
    mirrored = extended_features(GazeTrace(w.t, -w.x, w.y))
    for name in EXT_FEATURE_NAMES:
        assert scaled[name] == pytest.approx(base[name], abs=1e-6), name
        assert mirrored[name] == pytest.approx(base[name], abs=1e-6), name


def test_unusable_window_gives_neutral_values():
    t = np.arange(100) / 10.0
    empty = GazeTrace(t, np.full(100, np.nan), np.full(100, np.nan))
    assert extended_features(empty) == dict.fromkeys(EXT_FEATURE_NAMES, 0.0)
