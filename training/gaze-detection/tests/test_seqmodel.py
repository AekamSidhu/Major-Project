import numpy as np
import pytest

torch = pytest.importorskip("torch")

from readgaze import seqmodel, synthetic  # noqa: E402
from readgaze.trace import resample  # noqa: E402


def clean_windows(n_each: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Clean 10 fps windows of synthetic reading (1) and other tasks (0)."""
    windows, labels = [], []
    for i in range(n_each):
        for kind, make in synthetic.TASKS.items():
            trace = resample(make(10.5, np.random.default_rng([i, len(kind)])), 10.0)
            windows.append(np.stack([trace.x[:100], trace.y[:100]], axis=1))
            labels.append(int(kind == "reading"))
    return np.asarray(windows, dtype=np.float32), np.asarray(labels)


def test_input_is_unit_free_and_marks_missing_frames():
    raw, _ = clean_windows(1)
    raw[0, 10:14] = np.nan
    x = seqmodel.to_input(raw)
    assert x.shape == (len(raw), 3, 100)
    assert np.isfinite(x).all()
    assert x[0, 2, 10:14].sum() == 0 and x[0, 2].sum() == 96
    np.testing.assert_allclose(seqmodel.to_input(raw * 300.0 + 7.0), x, atol=1e-4)


def test_augmentation_adds_noise_gaps_and_mirrors():
    raw, _ = clean_windows(8)
    out = seqmodel.augment(raw, 10.0, seqmodel.Augment(), np.random.default_rng(0))
    assert out.shape == raw.shape
    assert 0.01 < np.isnan(out).mean() < 0.3
    assert not np.allclose(np.nan_to_num(out), raw)


def test_a_short_training_run_learns_synthetic_reading():
    raw, labels = clean_windows(40)
    config = seqmodel.TrainConfig(epochs=25, batch=16, lr=3e-3, widths=(16, 24), threads=1)
    mild = seqmodel.Augment(noise=(0.1, 0.6), drift=(0.0, 0.3), amplitude=(0.8, 1.2))
    model = seqmodel.train(raw, labels, 10.0, config, mild)
    noisy = seqmodel.augment(raw, 10.0, mild, np.random.default_rng(1))
    scores = seqmodel.predict(model, noisy)
    assert scores.shape == (len(raw),) and (0 <= scores).all() and (scores <= 1).all()
    assert scores[labels == 1].mean() > scores[labels == 0].mean() + 0.3
