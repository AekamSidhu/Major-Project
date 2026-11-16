"""Turn a video (e.g. a recorded candidate tile) into a gaze trace."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .landmarks import FaceTracker
from .trace import GazeTrace

COLUMNS = ["t", "x", "y", "openness", "yaw", "pitch", "face_found"]


def video_frames(
    path: str,
    tracker: FaceTracker,
    fps: float = 10.0,
    roi: tuple[int, int, int, int] | None = None,
    max_seconds: float | None = None,
) -> pd.DataFrame:
    """Per-frame gaze for a video file, sampled at `fps`.

    `roi` is `(x, y, width, height)` in pixels: the candidate's tile when the
    video is a recording of the whole meeting window.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    rows: list[dict] = []
    index, next_time = 0, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = index / source_fps
            index += 1
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-9 < next_time:
                continue  # downsample to the target rate
            next_time += 1.0 / fps
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y : y + h, x : x + w]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gaze = tracker.track(rgb, int(round(t * 1000)))
            rows.append({"t": t, **asdict(gaze)})
    finally:
        capture.release()
    return pd.DataFrame(rows, columns=COLUMNS)


def frames_to_trace(frames: pd.DataFrame, source: str = "") -> GazeTrace:
    """The gaze columns of `video_frames` output as a `GazeTrace`."""
    return GazeTrace(
        frames["t"].to_numpy(dtype=float),
        frames["x"].to_numpy(dtype=float),
        frames["y"].to_numpy(dtype=float),
        {"source": source, "unit": "eye widths"},
    )


def face_found_fraction(frames: pd.DataFrame) -> float:
    return float(np.mean(frames["face_found"].astype(bool))) if len(frames) else 0.0
