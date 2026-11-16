"""The two trained answer-integrity models, wrapped for the sidecar.

- `EchoModel`: does an answer's transcript read like an LLM wrote it?
  TF-IDF + logistic regression trained on HC3 (human vs ChatGPT answers),
  stored as plain JSON and scored with plain Python. Trained by
  `training/echo-detection/export_content_model.py` (feature/echo-detection).
- `ReadingModel`: do the candidate's eyes move like someone reading? A small
  1-D CNN on 10 s of gaze at 10 fps, exported to ONNX (three seeds averaged).
  Trained by `training/gaze-detection` (feature/gaze-detection).

Both report a probability plus a level: "high" at the conservative
threshold (few honest candidates flagged), "possible" at the balanced one.
A level is a signal for the interviewer, never a verdict.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .text import to_transcript

MODEL_DIR = Path(__file__).resolve().parent / "models"
_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token pattern

WINDOW_FRAMES = 100
FPS = 10.0
HOP_FRAMES = 50


def level(probability: float, possible: float, high: float) -> str:
    if probability >= high:
        return "high"
    if probability >= possible:
        return "possible"
    return "low"


class EchoModel:
    def __init__(self, path: Path = MODEL_DIR / "echo_tfidf.json") -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.name: str = data["model"]
        self.index = {term: j for j, (term, _, _) in enumerate(data["terms"])}
        self.idf = [t[1] for t in data["terms"]]
        self.coef = [t[2] for t in data["terms"]]
        self.intercept = float(data["intercept"])
        cuts = data["thresholds"]
        self.possible = float(min(cuts["balanced"], cuts["low_false_positive"]))
        self.high = float(max(cuts["balanced"], cuts["low_false_positive"]))
        self.test = data.get("test_questions", {})

    def probability(self, text: str) -> float:
        words = _TOKEN.findall(to_transcript(text))
        counts: dict[int, int] = {}
        for gram in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
            j = self.index.get(gram)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        weights = {j: (1.0 + math.log(c)) * self.idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        z = self.intercept + sum(w / norm * self.coef[j] for j, w in weights.items())
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def assess(self, text: str) -> dict[str, Any]:
        words = len(to_transcript(text).split())
        p = self.probability(text)
        return {
            "ai_probability": p,
            "level": level(p, self.possible, self.high) if words >= 15 else "too_short",
            "words": words,
            "thresholds": {"possible": self.possible, "high": self.high},
            "model": self.name,
        }


def to_input(raw: np.ndarray) -> np.ndarray:
    """`(n, frames, 2)` gaze with NaN gaps -> `(n, 3, frames)` network input.

    The same preparation the network was trained with
    (`readgaze.seqmodel.to_input`): fill gaps, centre on the median, divide by
    the window's spread, clip, and add a valid-frame channel.
    """
    raw = np.asarray(raw, dtype=np.float32)
    n, frames, _ = raw.shape
    valid = np.isfinite(raw).all(axis=2)
    filled = raw.copy()
    grid = np.arange(frames)
    for i in np.flatnonzero(~valid.all(axis=1)):
        ok = valid[i]
        if not ok.any():
            filled[i] = 0.0
            continue
        for c in range(2):
            filled[i, :, c] = np.interp(grid, grid[ok], raw[i, ok, c])
    centred = filled - np.median(filled, axis=1, keepdims=True)
    spread = np.sqrt((centred**2).sum(axis=2).mean(axis=1, keepdims=True))[:, :, None]
    scaled = np.clip(centred / np.maximum(spread, 1e-6), -6.0, 6.0)
    out = np.empty((n, 3, frames), dtype=np.float32)
    out[:, 0] = scaled[:, :, 0]
    out[:, 1] = scaled[:, :, 1]
    out[:, 2] = valid
    return out


@dataclass(frozen=True)
class ReadingResult:
    probability: float
    level: str
    windows: list[float]
    valid_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {"reading_probability": self.probability, "level": self.level, "window_probabilities": self.windows,
                "valid_fraction": self.valid_fraction}


class ReadingModel:
    def __init__(self, folder: Path = MODEL_DIR) -> None:
        import onnxruntime as ort

        meta = json.loads((Path(folder) / "reading_network.json").read_text(encoding="utf-8"))
        self.name: str = meta["model"]
        cuts = meta["thresholds"]
        self.window_possible = float(min(cuts["window_balanced"], cuts["window_low_false_positive"]))
        self.window_high = float(max(cuts["window_balanced"], cuts["window_low_false_positive"]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(folder) / "reading_network.onnx"), options, providers=["CPUExecutionProvider"])

    def window_probabilities(self, windows: np.ndarray) -> np.ndarray:
        if len(windows) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.session.run(None, {"window": to_input(windows)})[0].astype(np.float32)

    def assess(self, frames: np.ndarray, min_valid: float = 0.6) -> ReadingResult | None:
        """Score a gaze series already on the 10 fps grid, `(frames, 2)` with NaN gaps.

        Cut into 10 s windows every 5 s (the last window ends at the last
        frame); windows with less than `min_valid` tracked frames are skipped.
        The answer's score is the mean of the top half of its window scores.
        """
        frames = np.asarray(frames, dtype=np.float32)
        if len(frames) < WINDOW_FRAMES:
            return None
        starts = list(range(0, len(frames) - WINDOW_FRAMES + 1, HOP_FRAMES))
        if starts[-1] != len(frames) - WINDOW_FRAMES:
            starts.append(len(frames) - WINDOW_FRAMES)
        windows = np.stack([frames[s : s + WINDOW_FRAMES] for s in starts])
        valid = np.isfinite(windows).all(axis=2).mean(axis=1)
        windows = windows[valid >= min_valid]
        if len(windows) == 0:
            return None
        scores = np.sort(self.window_probabilities(windows))[::-1]
        top = float(scores[: max(1, math.ceil(len(scores) / 2))].mean())
        return ReadingResult(top, level(top, self.window_possible, self.window_high), [float(s) for s in scores],
                             float(np.isfinite(frames).all(axis=1).mean()))
