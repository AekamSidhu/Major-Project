"""Learned gaze: a ResNet-18 that reads gaze from a normalised face crop (opt-in).

The alternative to the MediaPipe iris offsets in `gaze.py`. It is used only when
the sidecar is given a model file (`--gaze-model` / `OPENMEET_GAZE_MODEL`). The
network was trained on MPIIFaceGaze (CC BY-NC-SA 4.0), so its weights are not
in this repository and must come from the user (see integrity/README.md).

Per frame:
1. MediaPipe Face Landmarker gives the face points (the same tracker as the
   iris method). Six of them (nose tip, chin, eye outer corners, mouth corners)
   fit a metric 3D face model with cv2.solvePnP, using an assumed camera.
2. The face is warped into the training framing: a virtual camera 600 mm away
   with focal length 960 px, rotated so the head is upright
   (`normalising_warp`). This is the notebook's `normalise()`, same formulas.
3. The crop (RGB, uint8, 224x224) goes to the ONNX model, which does the
   ImageNet normalisation inside the graph and returns (pitch, yaw) in radians
   in the normalised frame.
4. The gaze is read in the normalised frame (roll removed, like the eye-corner
   frame the iris offsets use) and scaled to the iris-offset scale (see
   `IRIS_UNITS_PER_SINE`), so the 10 fps series feeds the reading detector like
   the iris method does.

What the training code does that we do not reproduce: undistortion (we assume
no lens distortion) and the true camera calibration of the MPIIFaceGaze laptop
(we assume focal length = frame width). Both are approximations, not measured.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .gaze import iris_offset

FACE_SIZE = 224
VIRTUAL_FOCAL_PX = 960.0  # training: Zhang et al. 2018 normalisation
VIRTUAL_DISTANCE_MM = 600.0
CAMERA_N = np.array([[VIRTUAL_FOCAL_PX, 0.0, FACE_SIZE / 2], [0.0, VIRTUAL_FOCAL_PX, FACE_SIZE / 2], [0.0, 0.0, 1.0]])

# MediaPipe face-mesh indices for the six points the head pose is fitted to.
LANDMARK_IDS = (1, 152, 33, 263, 61, 291)  # nose tip, chin, outer eye corners (image left, right), mouth corners (image left, right)
# Generic metric face model in mm (the usual 6-point model of head-pose work),
# in the order of LANDMARK_IDS. Axes: x to the image right, y up, z to the viewer.
# Not the MPIIFaceGaze face model, which is not in this repository.
MODEL_POINTS_MM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ]
)

# Gaze in the iris-offset scale. MediaPipe's horizontal iris offset is in eye
# widths; an eyeball of radius ~12 mm over an eye width of ~30 mm moves the iris
# by about 0.4 eye widths per unit of sine of the gaze angle. A geometric
# estimate, not a fitted value (the notebook's linear map was not kept).
EYEBALL_RADIUS_MM = 12.0
EYE_WIDTH_MM = 30.0
IRIS_UNITS_PER_SINE = EYEBALL_RADIUS_MM / EYE_WIDTH_MM


def camera_matrix(width: int, height: int, focal_px: float | None = None) -> np.ndarray:
    """Assumed pinhole camera for a frame: focal length = frame width, centre = middle of the frame."""
    f = float(focal_px or width)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def fit_head_pose(points: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head rotation `hR` (model -> camera) and translation from the six landmark points (pixels, shape (6, 2)).

    Raises ValueError when the fit fails.
    """
    import cv2

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_MM, np.asarray(points, dtype=np.float64), cam, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("head pose fit failed")
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def face_centre(hR: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The face centre in camera coordinates, mm: the mean of the six model points.

    Taken to be the training `fc`. That definition is not in this repository and
    was not checked against the MPIIFaceGaze annotations (no dataset here).
    """
    return (hR @ MODEL_POINTS_MM.T + t[:, None]).mean(axis=1)


def normalising_warp(hR: np.ndarray, fc: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The notebook's `normalise()` warp. Returns `(W, R)`.

    `W` maps pixels of the frame (camera `cam`) to the 224x224 crop. `R` rotates
    camera coordinates into the normalised frame (rows: right, down, forward).
    Same formulas as `training/gaze-detection/notebooks/gaze_transfer_learning.ipynb`, cell 3.
    """
    d = float(np.linalg.norm(fc))
    forward = fc / d
    down = np.cross(forward, hR[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.stack([right, down, forward])
    W = CAMERA_N @ np.diag([1.0, 1.0, VIRTUAL_DISTANCE_MM / d]) @ R @ np.linalg.inv(cam)
    return W, R


def normalised_crop(rgb: np.ndarray, W: np.ndarray) -> np.ndarray:
    """The 224x224 RGB crop: the frame warped by `W` (bilinear, black outside, as in training)."""
    import cv2

    return cv2.warpPerspective(np.ascontiguousarray(rgb), W, (FACE_SIZE, FACE_SIZE))


def pitchyaw_to_vector(py: np.ndarray) -> np.ndarray:
    """Unit gaze vector in the normalised frame from (pitch, yaw) in radians (inverse of the notebook's `vec_to_pitchyaw`).

    Positive x is the normalised right (image right), positive y is down.
    """
    p, y = float(py[0]), float(py[1])
    return -np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def trace_from_pitchyaw(py: np.ndarray) -> tuple[float, float]:
    """Horizontal and vertical gaze in the iris-offset scale, from the network output (normalised frame).

    The normalised frame has the head's roll removed (its x axis follows the
    eye line), like the iris offsets, which are measured along the eye corners.
    Positive x is image right and positive y is image down, the iris convention.
    The reading detector's features are scale-free, but its early detector (3-10 s)
    uses absolute thresholds in iris units, so the scale is set to match.
    """
    g = pitchyaw_to_vector(py)
    return float(IRIS_UNITS_PER_SINE * g[0]), float(IRIS_UNITS_PER_SINE * g[1])


class GazeNet:
    """The ONNX gaze model. Input `face` (N, 224, 224, 3) uint8 RGB; output `pitch_yaw` (N, 2) radians."""

    def __init__(self, path: Path, threads: int = 2, tta: bool = False) -> None:
        import onnxruntime as ort

        path = Path(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if [i.name for i in inputs] != ["face"] or [o.name for o in self._session.get_outputs()] != ["pitch_yaw"]:
            raise ValueError(f"{path.name} is not the gaze model (expected input 'face' and output 'pitch_yaw')")
        self.tta = tta
        self.name = path.name

    def predict(self, faces: np.ndarray) -> np.ndarray:
        """Pitch and yaw (radians) for a batch of crops. With `tta`, the mirrored crop is averaged in (yaw negated),
        as the 4.54 degree training result was measured. Off by default for live use: it doubles the network's CPU time."""
        faces = np.asarray(faces, dtype=np.uint8)
        n = len(faces)
        batch = np.concatenate([faces, faces[:, :, ::-1]]) if self.tta else faces
        out = self._session.run(None, {"face": np.ascontiguousarray(batch)})[0].astype(np.float32)
        if not self.tta:
            return out
        plain, mirrored = out[:n], out[n:] * np.array([1.0, -1.0], dtype=np.float32)
        return (plain + mirrored) / 2


TTA_ENV = "OPENMEET_GAZE_TTA"


def tta_requested(environ=None) -> bool:
    """True when the environment asks for test-time augmentation (OPENMEET_GAZE_TTA=1). Off otherwise."""
    import os

    return (os.environ if environ is None else environ).get(TTA_ENV) == "1"


class LandmarkSource(Protocol):
    def points(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """(478, 2) landmark pixels of the one face in `rgb`, or None when there is no face."""


class LearnedGazeTracker:
    """A `Tracker` (see gaze.py) that returns the learned gaze for each frame, in the iris-offset scale.

    Frames with no face give NaN and found=False. Blinks (the iris method's
    test) give NaN with found=True.
    """

    def __init__(self, net: GazeNet, landmarks: LandmarkSource) -> None:
        self._net = net
        self._landmarks = landmarks

    def track(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[float, float, bool]:
        points = self._landmarks.points(rgb, timestamp_ms)
        if points is None:
            return float("nan"), float("nan"), False
        if not np.isfinite(iris_offset(points)[0]):  # blink: the crop would be unreliable
            return float("nan"), float("nan"), True
        h, w = rgb.shape[:2]
        cam = camera_matrix(w, h)
        hR, t = fit_head_pose(points[list(LANDMARK_IDS), :2], cam)
        W, _ = normalising_warp(hR, face_centre(hR, t), cam)
        crop = normalised_crop(rgb, W)
        py = self._net.predict(crop[None])[0]
        x, y = trace_from_pitchyaw(py)
        return x, y, True
