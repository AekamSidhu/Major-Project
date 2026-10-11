"""Eye geometry from MediaPipe Face Landmarker, frame by frame.

Face Landmarker returns 478 points per face: 468 for the face mesh plus 5 per
iris. From those we compute, for each eye, where the iris sits between the
eye corners (horizontal and vertical), how open the eye is (to drop blinks),
and the head's rotation (from the facial transformation matrix).

The iris position is measured in image directions: positive horizontal means
toward the right of the image, positive vertical means down. Whether that is
the candidate's left or right depends on whether the meeting app mirrors the
video, and the reading features don't need to know.

MediaPipe is imported lazily, so the geometry functions (and their tests)
work without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass(frozen=True)
class EyeIndices:
    """Landmark indices of one eye. `left_corner` is the corner on the image's left."""

    left_corner: int
    right_corner: int
    upper_lid: int
    lower_lid: int
    iris_center: int


# MediaPipe face mesh indices. "Right" and "left" are the subject's.
SUBJECT_RIGHT_EYE = EyeIndices(left_corner=33, right_corner=133, upper_lid=159, lower_lid=145, iris_center=468)
SUBJECT_LEFT_EYE = EyeIndices(left_corner=362, right_corner=263, upper_lid=386, lower_lid=374, iris_center=473)


@dataclass(frozen=True)
class EyeGeometry:
    horizontal: float
    """Iris offset from the eye's centre along the corner-to-corner line, in eye widths."""
    vertical: float
    """Iris offset perpendicular to that line (positive = down), in eye widths."""
    openness: float
    """Lid gap divided by eye width. Small while blinking."""


@dataclass(frozen=True)
class FrameGaze:
    x: float
    """Mean horizontal iris offset of both eyes; NaN if no face or a blink."""
    y: float
    """Mean vertical iris offset of both eyes; NaN if no face or a blink."""
    openness: float
    yaw: float
    """Head rotation left/right, degrees (approximate)."""
    pitch: float
    """Head rotation up/down, degrees (approximate)."""
    face_found: bool


NO_FACE = FrameGaze(np.nan, np.nan, np.nan, np.nan, np.nan, False)


def eye_geometry(points: np.ndarray, eye: EyeIndices) -> EyeGeometry:
    """Geometry of one eye from landmark points in pixels, shape (478, 2+)."""
    left = points[eye.left_corner, :2]
    right = points[eye.right_corner, :2]
    axis = right - left
    width = float(np.hypot(*axis))
    if width < 1e-6:
        return EyeGeometry(np.nan, np.nan, np.nan)
    along = axis / width
    across = np.array([-along[1], along[0]])  # points down in image coordinates
    offset = points[eye.iris_center, :2] - (left + right) / 2
    openness = float(np.hypot(*(points[eye.lower_lid, :2] - points[eye.upper_lid, :2])) / width)
    return EyeGeometry(
        horizontal=float(offset @ along / width),
        vertical=float(offset @ across / width),
        openness=openness,
    )


def head_angles(matrix: np.ndarray) -> tuple[float, float]:
    """Approximate (yaw, pitch) in degrees from a 4x4 facial transformation matrix."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    yaw = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    return float(yaw), float(pitch)


def frame_gaze(points: np.ndarray, matrix: np.ndarray | None, blink_threshold: float) -> FrameGaze:
    """Combine both eyes into one gaze sample. Blinks give NaN gaze."""
    eyes = [eye_geometry(points, SUBJECT_RIGHT_EYE), eye_geometry(points, SUBJECT_LEFT_EYE)]
    openness = float(np.nanmean([e.openness for e in eyes]))
    yaw, pitch = head_angles(matrix) if matrix is not None else (np.nan, np.nan)
    if not np.isfinite(openness) or openness < blink_threshold:
        return FrameGaze(np.nan, np.nan, openness, yaw, pitch, True)
    return FrameGaze(
        x=float(np.nanmean([e.horizontal for e in eyes])),
        y=float(np.nanmean([e.vertical for e in eyes])),
        openness=openness,
        yaw=yaw,
        pitch=pitch,
        face_found=True,
    )


class FaceTracker:
    """MediaPipe Face Landmarker in video mode, one face.

    Frames must come in time order (MediaPipe tracks between frames).
    """

    def __init__(self, model_path: str | Path, blink_threshold: float = 0.12) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found; run download_model.py first")
        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._blink_threshold = blink_threshold
        self._last_ms = -1

    def track(self, rgb: np.ndarray, timestamp_ms: int) -> FrameGaze:
        """Gaze for one RGB frame (H x W x 3, uint8)."""
        timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)  # must strictly increase
        self._last_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return NO_FACE
        height, width = rgb.shape[:2]
        points = np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]])
        matrices = result.facial_transformation_matrixes
        matrix = np.asarray(matrices[0]) if matrices else None
        return frame_gaze(points, matrix, self._blink_threshold)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
