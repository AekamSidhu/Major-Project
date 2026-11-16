import numpy as np
import pytest

from readgaze.trace import GazeTrace, fill_gaps, resample


def make(t, x, y=None):
    return GazeTrace(np.asarray(t, float), np.asarray(x, float), np.zeros(len(t)) if y is None else np.asarray(y, float))


def test_rejects_mismatched_or_unsorted_arrays():
    with pytest.raises(ValueError):
        GazeTrace(np.arange(3.0), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        make([0.0, 0.2, 0.1], [0, 0, 0])


def test_rate_duration_and_valid_fraction():
    trace = make(np.arange(0, 1, 0.01), np.r_[np.full(50, np.nan), np.zeros(50)])
    assert trace.rate == pytest.approx(100)
    assert trace.duration == pytest.approx(0.99)
    assert trace.valid_fraction == pytest.approx(0.5)


def test_windows_cover_the_trace_with_hops():
    trace = make(np.arange(0, 30, 0.1), np.zeros(300))
    windows = list(trace.windows(10.0, 5.0))
    assert [w.meta["window_start"] for w in windows] == pytest.approx([0, 5, 10, 15, 20])
    assert all(len(w) == 100 for w in windows)


def test_resample_averages_each_frame_interval():
    t = np.arange(0, 1, 0.001)
    trace = make(t, t)  # x equals time
    camera = resample(trace, 10)
    assert len(camera) == 10
    assert camera.t[0] == pytest.approx(0.05)
    # Mean of x over [0, 0.1) is ~0.05.
    assert camera.x[0] == pytest.approx(0.0495, abs=1e-3)


def test_resample_drops_frames_that_are_mostly_invalid():
    t = np.arange(0, 1, 0.001)
    x = np.zeros_like(t)
    x[100:180] = np.nan  # 80% of the second frame is a blink
    camera = resample(make(t, x), 10)
    assert np.isnan(camera.x[1])
    assert np.isfinite(camera.x[[0, 2]]).all()


def test_fill_gaps_interpolates_short_gaps_only():
    t = np.arange(0, 2, 0.1)
    x = np.arange(20, dtype=float)
    x[3] = np.nan  # 0.2 s between neighbours: filled
    x[8:14] = np.nan  # 0.7 s between neighbours: kept
    filled = fill_gaps(make(t, x), max_gap=0.3)
    assert filled.x[3] == pytest.approx(3.0)
    assert np.isnan(filled.x[8:14]).all()
