"""Reading in the Wild (Seattle): 80 people, 100 h of real-world reading and not reading.

Yang et al. (2025), "Reading Recognition in the Wild", NeurIPS,
arXiv:2505.24848. Data: https://www.projectaria.com/datasets/reading-in-the-wild/
(CC BY-NC 4.0; access is given per email, and the download links in the
file you get expire after about two weeks). Labels and the official
train/val/test split: `ritw_annotations.csv` in
https://github.com/facebookresearch/reading_in_the_wild.

Project Aria glasses recorded people reading (books, phones, laptops, signs;
silently, aloud, skimming, while walking...) and doing other things with and
without text in view. Eye gaze comes from the glasses' eye cameras at 60 Hz
as yaw and pitch angles relative to the head, which is close to what a
webcam sees of a candidate's eyes.

Each sequence's gaze sits in a ~2 GB zip of machine-perception outputs, so
the whole dataset is 2.2 TB. The gaze file is about 2 MB of it. Zips keep a
table of contents at the end, so `fetch_gaze` reads that table and then only
the gaze file's bytes with HTTP range requests: about 2 MB per sequence.

People: `pt_id` in the metadata is a slot number (1-5) on a collection
day, so a person is a day and slot. The authors' split is by collection
date, which keeps every person in one split.

Labelling follows the authors' code: a sequence is trimmed to its annotated
activity (`whisper_start` to `whisper_end`), tasks 8-11 are not reading,
task 16 (switching between the two) is left out, every other task is
reading. Personalised gaze is used where it exists, general gaze otherwise.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ANNOTATIONS_URL = "https://raw.githubusercontent.com/facebookresearch/reading_in_the_wild/main/reading_in_the_wild_seattle/ritw_annotations.csv"
NOT_READING_TASKS = frozenset({8, 9, 10, 11})
SKIPPED_TASKS = frozenset({16})
GAZE_FILES = ("mps/eye_gaze/personalized_eye_gaze.csv", "mps/eye_gaze/general_eye_gaze.csv")


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched with HTTP range requests."""

    def __init__(self, url: str, size: int, retries: int = 4) -> None:
        self.url, self.size, self.pos, self.retries = url, size, 0, retries
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._get(self.pos, self.pos + n - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)

    def _get(self, first: int, last: int) -> bytes:
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={first}-{last}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise OSError(f"server ignored the range request (HTTP {response.status})")
                    return response.read()
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")


def load_annotations(source: str = ANNOTATIONS_URL) -> pd.DataFrame:
    """The authors' per-sequence table: task, split, activity interval, reading mode..."""
    table = pd.read_csv(source)
    table["id"] = table["id"].astype(np.int64)
    return table


def label_for_task(task: int) -> int | None:
    """1 reading, 0 not reading, None for sequences the authors leave out."""
    if task in SKIPPED_TASKS:
        return None
    return 0 if task in NOT_READING_TASKS else 1


def fetch_metadata(entry: dict) -> dict:
    """The sequence's metadata.json (participant id, split, labels)."""
    with urllib.request.urlopen(entry["metadata_json"]["download_url"], timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(n for n in archive.namelist() if n.endswith("metadata.json"))
    return json.loads(archive.read(name))


def fetch_gaze(entry: dict) -> tuple[pd.DataFrame, str]:
    """The sequence's eye-gaze table, read straight out of the remote MPS zip."""
    mps = entry["mps"]
    remote = RangeFile(mps["download_url"], int(mps["file_size_bytes"]))
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as archive:
        names = set(archive.namelist())
        name = next(n for n in GAZE_FILES if n in names)
        raw = archive.read(name)
    return pd.read_csv(io.BytesIO(raw)), name.rsplit("/", 1)[-1]


def gaze_trace(gaze: pd.DataFrame, start: float, end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(t, x, y)` for the activity interval: seconds from the interval start,
    horizontal gaze (mean of the two eyes' yaw) and vertical gaze (pitch), in degrees.

    `start` and `end` are seconds after the first gaze sample, as in the annotations.
    """
    t = (gaze["tracking_timestamp_us"].to_numpy(dtype=np.float64) - float(gaze["tracking_timestamp_us"].iloc[0])) / 1e6
    keep = (t >= start) & (t <= end)
    yaw = np.degrees((gaze["left_yaw_rads_cpf"].to_numpy(dtype=float) + gaze["right_yaw_rads_cpf"].to_numpy(dtype=float)) / 2.0)
    pitch = np.degrees(gaze["pitch_rads_cpf"].to_numpy(dtype=float))
    t, x, y = t[keep], yaw[keep], pitch[keep]
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    unique = np.r_[True, np.diff(t) > 0]
    return t[unique] - start, x[unique], y[unique]
