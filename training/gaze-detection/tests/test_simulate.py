import numpy as np
import pytest

from readgaze import synthetic
from readgaze.simulate import WebcamProfile, simulate_webcam


def test_output_runs_at_the_camera_frame_rate():
    trace = synthetic.reading(20.0, np.random.default_rng(0))
    webcam = simulate_webcam(trace, WebcamProfile(fps=10), np.random.default_rng(1))
    assert webcam.rate == pytest.approx(10)
    assert len(webcam) == pytest.approx(200, abs=1)


def test_blinks_and_dropouts_become_invalid_frames():
    trace = synthetic.fixation(60.0, np.random.default_rng(0))
    profile = WebcamProfile(blinks_per_minute=20, dropout_probability=0.05)
    webcam = simulate_webcam(trace, profile, np.random.default_rng(1))
    assert 0.80 < webcam.valid_fraction < 0.99


def test_same_seed_gives_the_same_trace():
    trace = synthetic.reading(10.0, np.random.default_rng(0))
    a = simulate_webcam(trace, WebcamProfile(), np.random.default_rng(5))
    b = simulate_webcam(trace, WebcamProfile(), np.random.default_rng(5))
    np.testing.assert_array_equal(a.x, b.x)


def test_amplitude_shrinks_eye_movements():
    trace = synthetic.reading(30.0, np.random.default_rng(0))
    quiet = WebcamProfile(noise=0, drift=0, blinks_per_minute=0, dropout_probability=0)
    full = simulate_webcam(trace, quiet, np.random.default_rng(1))
    small = simulate_webcam(trace, WebcamProfile(**{**quiet.__dict__, "amplitude": 0.5}), np.random.default_rng(1))
    assert np.nanstd(small.x) == pytest.approx(0.5 * np.nanstd(full.x), rel=1e-6)
