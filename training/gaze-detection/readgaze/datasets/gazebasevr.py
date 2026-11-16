"""GazeBaseVR: 407 people, 250 Hz eye tracking in a VR headset, five tasks.

Lohr, Aziz, Friedman & Komogortsev (2023), "GazeBaseVR, a large-scale,
longitudinal, binocular eye-tracking dataset collected in virtual reality",
Scientific Data doi:10.1038/s41597-023-02075-5.
Data: https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391
(CC BY 4.0, one 2.5 GB zip).

One CSV per recording, `data/S_rxxx_Sy_k_TSK.csv`: round `r` (1–3),
participant `xxx`, session `y` (1–2), task `TSK`. Columns used here: `n`
(time, ms) and `x`, `y` (binocular gaze direction, degrees of visual angle;
NaN where tracking was lost).

A different tracker, a different text and self-paced reading. We never train
on it: it is an outside test of a detector trained on GazeBase. Many of its
participants were also recorded for GazeBase under unrelated numbering, so it
is a test on another device and stimulus, not on strictly new people.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

ARTICLE_ID = 21308391
ZIP_NAME = "gazebasevr.zip"

TASKS: dict[str, str] = {
    "TEX": "reading (self-paced)",
    "VRG": "vergence (near/far targets)",
    "PUR": "horizontal smooth pursuit",
    "VID": "video viewing",
    "RAN": "random oblique saccades",
}
READING_TASK = "TEX"

_NAME = re.compile(r"S_(?P<round>\d)(?P<participant>\d{3})_S(?P<session>\d)_\d_(?P<task>[A-Z]{3})\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingId:
    round: int
    participant: int
    session: int
    task: str

    @property
    def is_reading(self) -> bool:
        return self.task == READING_TASK

    @classmethod
    def parse(cls, name: str) -> RecordingId | None:
        match = _NAME.search(name.replace("\\", "/").rsplit("/", 1)[-1])
        if not match:
            return None
        return cls(int(match["round"]), int(match["participant"]), int(match["session"]), match["task"].upper())


def load_csv(source: bytes | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One recording as `(t, x, y)`: seconds and degrees, NaN where invalid."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    frame = pd.read_csv(source, usecols=["n", "x", "y"])
    frame = frame.drop_duplicates(subset="n").sort_values("n")
    return (
        frame["n"].to_numpy(dtype=float) / 1000.0,
        frame["x"].to_numpy(dtype=float),
        frame["y"].to_numpy(dtype=float),
    )
