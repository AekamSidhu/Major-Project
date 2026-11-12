"""DesktopActivity: 8 people doing six things at a desktop, 30 Hz, head free.

Lan, Heit, Scargill & Gorlatova (2020), "GazeGraph: graph-based few-shot
cognitive context sensing from human visual behavior", ACM SenSys.
Data: https://github.com/EyeSyn/GazeGraph (`DesktopActivity.zip`, 2 MB).

Each person did each activity for five minutes wearing a Pupil Core tracker,
free to move: READ (a Wikipedia article, a two-column paper or a textbook),
BROWSE, PLAY, SEARCH, WATCH and WRITE. One CSV per person and activity,
`Pn/Pn_ACTIVITY.csv`: 9000 rows of normalised x, y, no header.

Small, but the closest public data to someone reading at a real screen next
to ordinary screen activities. We never train on it: it is an outside test
of a detector trained on GazeBase.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_URL = "https://raw.githubusercontent.com/EyeSyn/GazeGraph/HEAD/DesktopActivity.zip"
RATE = 30.0
READING_TASK = "READ"
TASKS = ("BROWSE", "PLAY", "READ", "SEARCH", "WATCH", "WRITE")

_NAME = re.compile(r"P(?P<participant>\d+)_(?P<task>[A-Z]+)\.csv$", re.IGNORECASE)


def iter_recordings(zip_path: str | Path) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
    """Yields `(info, t, x, y)` per recording. `y` is flipped so up is positive."""
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            match = _NAME.search(name)
            if not match:
                continue
            frame = pd.read_csv(io.BytesIO(archive.read(name)), header=None, names=["x", "y"])
            task = match["task"].upper()
            participant = int(match["participant"])
            info = {
                "dataset": "DesktopActivity",
                "participant": f"da{participant:02d}",
                "recording": f"da{participant:02d}-{task}",
                "task": task,
                "label": int(task == READING_TASK),
            }
            t = np.arange(len(frame)) / RATE
            yield info, t, frame["x"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float)
