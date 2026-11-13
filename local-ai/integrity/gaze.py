"""Live gaze from video frames: MediaPipe iris landmarks -> a 10 fps gaze series.

Frames come from one of two sources:
- the browser (the integrity page posts webcam or screen-share frames), or
- a screen region the sidecar captures itself (the candidate's tile in the
  meeting window), about 10 times a second.

Each frame becomes one gaze sample: where the irises sit between the eye
corners, averaged over both eyes (NaN for no face or a blink). Samples are
kept in memory for the last few minutes only and never written to disk.
Frames are dropped as soon as their sample is taken.

The geometry is the same as `training/gaze-detection/readgaze/landmarks.py`,
which the reading network's gaze features were developed against.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
KEEP_SECONDS = 600.0
MAX_SIDE = 640  # frames are scaled down to this before landmarking

# MediaPipe face-mesh indices: (left corner, right corner, upper lid, lower lid, iris centre).
_EYES = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))


def iris_offset(points: np.ndarray, blink_threshold: float = 0.12) -> tuple[float, float]:
    """Mean iris offset of both eyes in eye widths, `(nan, nan)` while blinking.

    `points` are landmark positions in pixels, shape (478, 2+).
    """
    xs, ys, openness = [], [], []
    for left_i, right_i, up_i, low_i, iris_i in _EYES:
        left, right = points[left_i, :2], points[right_i, :2]
        axis = right - left
        width = float(np.hypot(*axis))
        if width < 1e-6:
            continue
        along = axis / width
        across = np.array([-along[1], along[0]])
        offset = points[iris_i, :2] - (left + right) / 2
        xs.append(float(offset @ along / width))
        ys.append(float(offset @ across / width))
        openness.append(float(np.hypot(*(points[low_i, :2] - points[up_i, :2])) / width))
    if not xs or float(np.mean(openness)) < blink_threshold:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.mean(ys))


class Tracker(Protocol):
    def track(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[float, float, bool]:
        """`(x, y, face_found)` for one RGB frame."""


class FaceTracker:
    """MediaPipe Face Landmarker in video mode (one face). Needs `mediapipe`."""

    def __init__(self, model_dir: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        path = Path(model_dir) / "face_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(FACE_MODEL_URL, path)  # 3.8 MB, once
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def track(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[float, float, bool]:
        timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)
        self._last_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return float("nan"), float("nan"), False
        h, w = rgb.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
        x, y = iris_offset(points)
        return x, y, True


def shrink(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    step = int(np.ceil(max(h, w) / max_side))
    return rgb[::step, ::step] if step > 1 else rgb


def decode_image(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 array."""
    try:
        import cv2

        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("not an image")
        return bgr[:, :, ::-1]
    except ImportError:
        from PIL import Image
        import io

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int


class GazeMonitor:
    """Turns frames into a rolling gaze series and keeps the last few minutes."""

    def __init__(self, tracker_factory: Callable[[], Tracker], clock: Callable[[], float] = time.time) -> None:
        self._factory = tracker_factory
        self._tracker: Tracker | None = None
        self._clock = clock
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, float, float, bool]] = deque()
        self._source = "none"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._region: Region | None = None
        self.error: str | None = None

    # -- frames ------------------------------------------------------------
    def add_frame(self, rgb: np.ndarray, source: str = "browser", t: float | None = None) -> tuple[float, float, bool]:
        t = self._clock() if t is None else t
        with self._lock:
            if self._tracker is None:
                self._tracker = self._factory()
            x, y, found = self._tracker.track(shrink(rgb), int(t * 1000))
            self._samples.append((t, x, y, found))
            while self._samples and self._samples[0][0] < t - KEEP_SECONDS:
                self._samples.popleft()
            self._source = source
        return x, y, found

    # -- screen-region capture ------------------------------------------------
    def start_screen(self, region: Region, fps: float = 10.0) -> None:
        self.stop()
        self._region, self._stop = region, threading.Event()
        self._source, self.error = "screen", None
        self._thread = threading.Thread(target=self._capture, args=(region, fps, self._stop), daemon=True, name="gaze-screen")
        self._thread.start()

    def _capture(self, region: Region, fps: float, stop: threading.Event) -> None:
        try:
            import mss

            with mss.mss() as screen:
                box = {"left": region.left, "top": region.top, "width": region.width, "height": region.height}
                while not stop.is_set():
                    started = time.perf_counter()
                    shot = np.asarray(screen.grab(box))  # BGRA
                    self.add_frame(shot[:, :, 2::-1], source="screen")
                    stop.wait(max(0.0, 1.0 / fps - (time.perf_counter() - started)))
        except Exception as error:  # reported through status()
            self.error = f"{type(error).__name__}: {error}"[:300]

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._source == "screen":
            self._source = "none"

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    # -- reading out --------------------------------------------------------
    def series(self, start: float, end: float, fps: float = 10.0) -> np.ndarray:
        """Samples between `start` and `end` (epoch seconds) on a `fps` grid:
        `(frames, 2)`, mean per frame interval, NaN where nothing was tracked."""
        n = int(round((end - start) * fps))
        out = np.full((max(n, 0), 2), np.nan, dtype=np.float32)
        if n <= 0:
            return out
        sums = np.zeros((n, 2))
        counts = np.zeros(n)
        with self._lock:
            samples = [s for s in self._samples if start <= s[0] < end and np.isfinite(s[1]) and np.isfinite(s[2])]
        for t, x, y, _ in samples:
            i = min(int((t - start) * fps), n - 1)
            sums[i] += (x, y)
            counts[i] += 1
        seen = counts > 0
        out[seen] = (sums[seen] / counts[seen, None]).astype(np.float32)
        return out

    def status(self, seconds: float = 10.0) -> dict:
        now = self._clock()
        with self._lock:
            recent = [s for s in self._samples if s[0] >= now - seconds]
            last = self._samples[-1][0] if self._samples else None
        return {
            "source": self._source,
            "capturing_screen": self._thread is not None and self._thread.is_alive(),
            "region": self._region.__dict__ if self._region else None,
            "frames_per_second": len(recent) / seconds,
            "face_found_fraction": float(np.mean([s[3] for s in recent])) if recent else 0.0,
            "tracked_fraction": float(np.mean([np.isfinite(s[1]) for s in recent])) if recent else 0.0,
            "seconds_since_last_frame": None if last is None else now - last,
            "now": now,
            "error": self.error,
        }
