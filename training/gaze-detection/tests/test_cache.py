import numpy as np
import pytest

from readgaze import synthetic
from readgaze.cache import TraceCache, bin_samples, merge_frames
from readgaze.datasets import gazebasevr
from readgaze.trace import GazeTrace, resample


def reading_with_a_blink():
    trace = synthetic.reading(20.0, np.random.default_rng(0))
    x, y = trace.x.copy(), trace.y.copy()
    x[3000:3180] = np.nan
    y[3000:3180] = np.nan
    # 997 Hz: no sample sits exactly on a 1/30 s or 1/10 s frame boundary.
    return np.arange(len(x)) / 997.0, x, y


def test_merged_cache_frames_match_a_direct_resample():
    t, x, y = reading_with_a_blink()
    frames = bin_samples(t, x, y, rate=30.0)
    merged_x, merged_y = merge_frames(frames, 3)
    direct = resample(GazeTrace(t, x, y), 10.0)
    n = min(len(merged_x), len(direct))
    np.testing.assert_allclose(merged_x[:n], direct.x[:n], atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(merged_y[:n], direct.y[:n], atol=1e-4, equal_nan=True)
    assert np.isnan(merged_x).any()  # the blink survives as invalid frames


def test_cache_round_trips_through_a_file(tmp_path):
    t, x, y = reading_with_a_blink()
    frames = bin_samples(t, x, y)
    rows = [
        {"dataset": "T", "participant": "p1", "recording": "a", "task": "TEX", "label": 1},
        {"dataset": "T", "participant": "p2", "recording": "b", "task": "FXS", "label": 0},
    ]
    cache = TraceCache.from_recordings(rows, [frames, frames])
    cache.save(tmp_path / "c.npz")
    loaded = TraceCache.load(tmp_path / "c.npz")
    assert len(loaded) == 2
    assert loaded.meta["participant"].tolist() == ["p1", "p2"]
    assert loaded.meta["label"].tolist() == [1, 0]
    np.testing.assert_array_equal(loaded.frames(1).x, frames.x)
    trace = loaded.trace(1, 10.0)
    assert trace.rate == pytest.approx(10)
    with pytest.raises(ValueError):
        loaded.trace(0, 7.0)  # 7 fps does not divide 30 Hz


def test_gazebasevr_file_names():
    rec = gazebasevr.RecordingId.parse("data/S_1002_S1_4_TEX.csv")
    assert (rec.round, rec.participant, rec.session, rec.task) == (1, 2, 1, "TEX")
    assert rec.is_reading
    assert gazebasevr.RecordingId.parse("data/notes.csv") is None
