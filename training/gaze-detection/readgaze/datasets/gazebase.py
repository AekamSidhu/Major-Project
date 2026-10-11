"""GazeBase: 322 people, 1000 Hz eye tracking, seven tasks including reading.

Griffith, Lohr, Abdulin & Komogortsev (2021), "GazeBase, a large-scale,
multi-stimulus, longitudinal eye movement dataset", Scientific Data
doi:10.1038/s41597-021-00959-y.
Data: https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257
(CC BY 4.0, one 6.7 GB zip).

Layout: round → participant (a zip per participant) → session → task, one CSV
per recording, named `S_rxxx_Sy_tsk.csv`: round `r` (1–9), participant `xxx`,
session `y` (1–2), task `tsk`. CSV columns used here: `n` (time, ms), `x` and
`y` (gaze, degrees of visual angle), `val` (non-zero = invalid; x and y are
NaN then).

The loader reads straight from the downloaded zip, including the nested
per-participant zips, so nothing has to be extracted.

The same people come back across rounds, so person-level splits must group
by `participant`, never by round.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..trace import GazeTrace

ARTICLE_ID = 12912257
ARTICLE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
ZIP_NAME = "GazeBase_v2_0.zip"

TASKS: dict[str, str] = {
    "TEX": "reading (a passage of a poem, 60 s)",
    "FXS": "fixation",
    "HSS": "horizontal saccades",
    "RAN": "random saccades",
    "VD1": "video viewing, part 1",
    "VD2": "video viewing, part 2",
    "BLG": "gaze-driven game",
}
READING_TASK = "TEX"

_NAME = re.compile(r"S_(?P<round>\d)(?P<participant>\d{3})_S(?P<session>\d)_(?P<task>[A-Z0-9]{3})\.csv$", re.IGNORECASE)


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
        return cls(
            round=int(match["round"]),
            participant=int(match["participant"]),
            session=int(match["session"]),
            task=match["task"].upper(),
        )


def load_csv(source: str | Path | bytes | io.IOBase) -> GazeTrace:
    """One GazeBase recording as a `GazeTrace` (seconds, degrees)."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    frame = pd.read_csv(source, usecols=["n", "x", "y", "val"])
    frame = frame.drop_duplicates(subset="n").sort_values("n")
    invalid = frame["val"].to_numpy() != 0
    x = frame["x"].to_numpy(dtype=float, copy=True)
    y = frame["y"].to_numpy(dtype=float, copy=True)
    x[invalid] = np.nan
    y[invalid] = np.nan
    return GazeTrace(frame["n"].to_numpy(dtype=float) / 1000.0, x, y, {"dataset": "GazeBase", "unit": "deg"})


def iter_recordings(root: str | Path, tasks: Iterable[str] | None = None) -> Iterator[tuple[RecordingId, GazeTrace]]:
    """Every recording under `root`, which may be the downloaded zip or a folder.

    Folders are searched recursively for CSVs and zips; zips are read
    recursively in memory. Recordings are yielded in a stable order.
    """
    wanted = {t.upper() for t in tasks} if tasks is not None else None
    root = Path(root)
    if root.is_file():
        with zipfile.ZipFile(root) as archive:
            yield from _iter_zip(archive, wanted)
        return
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".csv":
            rec = RecordingId.parse(path.name)
            if rec and (wanted is None or rec.task in wanted):
                yield rec, load_csv(path)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                yield from _iter_zip(archive, wanted)


def _iter_zip(archive: zipfile.ZipFile, wanted: set[str] | None) -> Iterator[tuple[RecordingId, GazeTrace]]:
    for name in sorted(archive.namelist()):
        lower = name.lower()
        if lower.endswith(".csv"):
            rec = RecordingId.parse(name)
            if rec and (wanted is None or rec.task in wanted):
                yield rec, load_csv(archive.read(name))
        elif lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
                yield from _iter_zip(inner, wanted)
