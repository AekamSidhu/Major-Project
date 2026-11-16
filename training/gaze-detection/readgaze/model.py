"""The learned detector: logistic regression on the window features.

Small on purpose. It is explainable (one weight per feature), gives calibrated
probabilities, and trains on a laptop in seconds. The trained model is
exported as plain JSON (feature means, scales, weights, threshold) so the
OpenMeet engine can score windows in Rust without Python. `score_from_json`
is the reference implementation the Rust port must match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

JSON_VERSION = 1


def build_model(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=1000)),
        ]
    )


def export_json(
    model: Pipeline,
    feature_names: Sequence[str],
    threshold: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    clf: LogisticRegression = model.named_steps["clf"]
    return {
        "version": JSON_VERSION,
        "kind": "reading-gaze-logistic",
        "features": list(feature_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": float(threshold),
        "metadata": dict(metadata or {}),
    }


def score_from_json(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    """Reading probability for one window, using only the exported JSON."""
    if model.get("version") != JSON_VERSION:
        raise ValueError(f"unsupported model version {model.get('version')}")
    values = np.array([features[name] for name in model["features"]], dtype=float)
    z = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = float(z @ np.asarray(model["coef"]) + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))
