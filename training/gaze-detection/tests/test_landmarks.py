from pathlib import Path

import numpy as np
import pytest

from readgaze.landmarks import (
    SUBJECT_LEFT_EYE,
    SUBJECT_RIGHT_EYE,
    EyeIndices,
    eye_geometry,
    frame_gaze,
    head_angles,
)

MODEL = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def face(iris_dx: float = 0.0, iris_dy: float = 0.0, lid_gap: float = 6.0) -> np.ndarray:
    """478 landmark points with two 30 px wide eyes, irises shifted by (dx, dy) px."""
    points = np.zeros((478, 2))

    def place(eye: EyeIndices, cx: float) -> None:
        points[eye.left_corner] = (cx - 15, 100)
        points[eye.right_corner] = (cx + 15, 100)
        points[eye.upper_lid] = (cx, 100 - lid_gap / 2)
        points[eye.lower_lid] = (cx, 100 + lid_gap / 2)
        points[eye.iris_center] = (cx + iris_dx, 100 + iris_dy)

    place(SUBJECT_RIGHT_EYE, 80)  # the subject's right eye is on the image's left
    place(SUBJECT_LEFT_EYE, 160)
    return points


def test_centred_iris_gives_zero_offset():
    geometry = eye_geometry(face(), SUBJECT_RIGHT_EYE)
    assert geometry.horizontal == pytest.approx(0.0)
    assert geometry.vertical == pytest.approx(0.0)
    assert geometry.openness == pytest.approx(0.2)


def test_both_eyes_agree_on_image_direction():
    # Looking toward the image's right moves both irises the same way.
    points = face(iris_dx=3.0, iris_dy=1.5)
    right, left = eye_geometry(points, SUBJECT_RIGHT_EYE), eye_geometry(points, SUBJECT_LEFT_EYE)
    assert right.horizontal == pytest.approx(0.1)
    assert left.horizontal == pytest.approx(0.1)
    assert right.vertical == pytest.approx(0.05)  # positive = down


def test_blink_gives_no_gaze():
    gaze = frame_gaze(face(lid_gap=1.0), None, blink_threshold=0.12)
    assert gaze.face_found
    assert np.isnan(gaze.x) and np.isnan(gaze.y)


def test_open_eyes_give_gaze_and_head_angles():
    gaze = frame_gaze(face(iris_dx=3.0), np.eye(4), blink_threshold=0.12)
    assert gaze.x == pytest.approx(0.1)
    assert (gaze.yaw, gaze.pitch) == pytest.approx((0.0, 0.0))


def test_head_yaw_from_rotation_matrix():
    angle = np.radians(20)
    matrix = np.eye(4)
    matrix[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    yaw, pitch = head_angles(matrix)
    assert yaw == pytest.approx(20.0)
    assert pitch == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(not MODEL.exists(), reason="run download_model.py to test MediaPipe itself")
def test_mediapipe_reports_no_face_on_a_blank_frame():
    from readgaze.landmarks import FaceTracker

    with FaceTracker(MODEL) as tracker:
        gaze = tracker.track(np.zeros((240, 320, 3), dtype=np.uint8), 0)
        again = tracker.track(np.zeros((240, 320, 3), dtype=np.uint8), 0)  # same timestamp is bumped
    assert not gaze.face_found and not again.face_found
