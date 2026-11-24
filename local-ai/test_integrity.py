"""Tests for the answer-integrity endpoints (real models, fake face tracker: no camera, no MediaPipe)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from integrity.gaze import iris_offset
from integrity.learned_gaze import (
    CAMERA_N,
    EYE_WIDTH_MM,
    EYEBALL_RADIUS_MM,
    LANDMARK_IDS,
    MODEL_POINTS_MM,
    GazeNet,
    LearnedGazeTracker,
    camera_matrix,
    face_centre,
    fit_head_pose,
    normalised_crop,
    normalising_warp,
    pitchyaw_to_vector,
    trace_from_pitchyaw,
)
from integrity.models import EchoModel, ReadingModel, to_input
from integrity.text import to_transcript
from test_server import FakeEmbedder, FakeTranscriber

AI_ANSWER = (
    "There are several key factors to consider when scaling a web application. First, it is important to identify "
    "the bottlenecks in your system, such as database queries or network latency. Additionally, implementing caching "
    "strategies can significantly improve performance. Overall, a combination of horizontal scaling, load balancing "
    "and efficient database indexing will help ensure that your application remains responsive as traffic grows."
)
HUMAN_ANSWER = (
    "um so honestly we didn't really plan for it, the site just fell over one friday and we threw a redis cache in front "
    "of the slow queries, which bought us a few months, then later we split the reporting stuff off onto its own box "
    "because that was what kept killing the database"
)


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class SawtoothTracker:
    """Pretends to see a face whose eyes read lines of text (or hold still)."""

    def __init__(self, clock: Clock, reading: bool) -> None:
        self.clock, self.reading = clock, reading
        self.rng = np.random.default_rng(0)

    def track(self, rgb, timestamp_ms):  # type: ignore[no-untyped-def]
        t = timestamp_ms / 1000.0
        if self.reading:
            phase = (t % 3.0) / 3.0  # one line every 3 s: slow drift right, fast return
            x = 0.25 * phase - 0.12 + self.rng.normal(0, 0.004)
            y = 0.02 * ((t // 3.0) % 8) + self.rng.normal(0, 0.004)
        else:
            x, y = self.rng.normal(0, 0.01), self.rng.normal(0, 0.01)
        return float(x), float(y), True


def make_client(reading: bool):
    clock = Clock()
    app = server.create_app(
        transcriber=FakeTranscriber(), embedder=FakeEmbedder(),
        integrity_tracker=lambda: SawtoothTracker(clock, reading), clock=clock,
    )
    return TestClient(app), clock


def feed(client: TestClient, clock: Clock, seconds: float) -> None:
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(buf, format="JPEG")
    frame = buf.getvalue()
    for _ in range(int(seconds * 10)):
        clock.t += 0.1
        r = client.post("/v1/integrity/gaze/frame", content=frame)
        assert r.status_code == 200, r.text


def test_transcript_normalisation_matches_training():
    assert to_transcript("You do n't  know ( URL_0 ) .\n- **Bold** point") == "you dont know bold point"


def test_echo_model_separates_ai_and_human_answers():
    model = EchoModel()
    ai, human = model.probability(AI_ANSWER), model.probability(HUMAN_ANSWER)
    assert ai > 0.8 > 0.2 > human, (ai, human)
    assert model.assess("yes")["level"] == "too_short"


def test_echo_endpoint():
    client, _ = make_client(reading=False)
    r = client.post("/v1/integrity/echo", json={"text": AI_ANSWER})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level"] == "high" and body["words"] > 15
    assert client.post("/v1/integrity/echo", json={"text": "  "}).status_code == 400


def test_reading_network_runs_and_is_unit_free():
    model = ReadingModel()
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(3, 100, 2)).cumsum(axis=1).astype(np.float32)
    p = model.window_probabilities(raw)
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(model.window_probabilities(raw * 40 + 3), p, atol=1e-4)
    assert to_input(raw).shape == (3, 3, 100)


@pytest.mark.parametrize("reading", [True, False])
def test_reading_endpoint_from_frames(reading):
    client, clock = make_client(reading)
    start = clock.t
    feed(client, clock, 30)
    r = client.post("/v1/integrity/reading", json={"start": start, "end": clock.t})
    assert r.status_code == 200, r.text
    body = r.json()
    if reading:
        assert body["reading_probability"] > 0.8, body
    else:
        assert body["reading_probability"] < 0.5, body
    status = client.get("/v1/integrity/status").json()
    assert status["echo_ready"] and status["reading_ready"]
    assert status["gaze"]["source"] == "browser" and status["gaze"]["face_found_fraction"] == 1.0
    assert status["live_reading"] is not None and len(status["trace_x"]) == 300
    assert status["live_reading"]["early"] is False and status["live_reading"]["calibrated"] is True


def test_reading_rules_by_length_of_gaze():
    # Under 3 s: not enough gaze. 3-10 s: early detector, uncalibrated. 10 s or more: the network, calibrated.
    client, clock = make_client(reading=True)
    feed(client, clock, 2)
    body = client.post("/v1/integrity/reading", json={"seconds": 2}).json()
    assert body["level"] == "not_enough_gaze" and body["reading_probability"] is None
    assert client.post("/v1/integrity/reading", json={"start": 10, "end": 5}).status_code == 400

    client, clock = make_client(reading=True)
    feed(client, clock, 5)
    body = client.post("/v1/integrity/reading", json={"seconds": 5}).json()
    assert body["reading_probability"] is not None, body
    assert body["early"] is True and body["calibrated"] is False

    client, clock = make_client(reading=True)
    feed(client, clock, 12)
    body = client.post("/v1/integrity/reading", json={"seconds": 12}).json()
    assert body["reading_probability"] is not None, body
    assert body["early"] is False and body["calibrated"] is True


def test_short_gaze_without_a_sweep_is_not_enough():
    client, clock = make_client(reading=False)
    feed(client, clock, 5)
    body = client.post("/v1/integrity/reading", json={"seconds": 5}).json()
    assert body["level"] == "not_enough_gaze" and body["reading_probability"] is None


def test_frame_validation_and_page():
    client, _ = make_client(reading=True)
    assert client.post("/v1/integrity/gaze/frame", content=b"").status_code == 400
    assert client.post("/v1/integrity/gaze/frame", content=b"not an image").status_code == 400
    page = client.get("/integrity")
    assert page.status_code == 200 and "OpenMeet integrity check" in page.text


def test_token_protects_the_api_but_not_the_page():
    app = server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(), token="s3cret")
    client = TestClient(app)
    assert client.get("/integrity").status_code == 200
    assert client.post("/v1/integrity/echo", json={"text": AI_ANSWER}).status_code == 401
    ok = client.post("/v1/integrity/echo", json={"text": AI_ANSWER}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_iris_offset_geometry_and_blinks():
    points = np.zeros((478, 2))
    for left, right, up, low, iris in ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473)):
        base = 0.0 if left == 33 else 100.0
        points[left], points[right] = (base, 0), (base + 40, 0)
        points[up], points[low] = (base + 20, -8), (base + 20, 8)
        points[iris] = (base + 30, 0)  # a quarter eye-width to the right
    x, y = iris_offset(points)
    assert x == pytest.approx(0.25) and y == pytest.approx(0.0)
    points[[159, 386], 1] = -1
    points[[145, 374], 1] = 1  # lids almost closed
    assert np.isnan(iris_offset(points)[0])


# -- learned gaze (learned_gaze.py): no weights needed; tiny ONNX models are built in tmp_path --------------------------

def _notebook_normalise_warp(hr, fc, cam):
    """The formulas of the training notebook's `normalise()` (cell 3), without the image warp and the labels."""
    import cv2

    size, focal, dist = 224, 960.0, 600.0
    cam_n = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1.0]])
    d = np.linalg.norm(fc)
    hR = cv2.Rodrigues(hr)[0]
    forward = fc / d
    down = np.cross(forward, hR[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.stack([right, down, forward])
    W = cam_n @ np.diag([1.0, 1.0, dist / d]) @ R @ np.linalg.inv(cam)
    return W, R


def _project(rvec, t, cam):
    import cv2

    return cv2.projectPoints(MODEL_POINTS_MM, rvec, t, cam, None)[0].reshape(-1, 2)


# OpenCV comes with MediaPipe (requirements-integrity.txt), which CI does not install.
try:
    import cv2 as _cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
needs_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="OpenCV not installed (comes with requirements-integrity.txt)")


@needs_cv2
@pytest.mark.parametrize("roll_deg", [0.0, 25.0, -40.0])
def test_normalising_warp_matches_training_code(roll_deg):
    import cv2

    assert CAMERA_N[0, 0] == 960.0 and CAMERA_N[0, 2] == 112.0
    cam = camera_matrix(640, 480)
    rvec = np.array([np.pi + 0.15, -0.25, np.deg2rad(roll_deg)])  # a face looking at the camera (the model is flipped)
    t = np.array([20.0, -35.0, 720.0])
    hR_true = cv2.Rodrigues(rvec)[0]
    fc_true = (hR_true @ MODEL_POINTS_MM.T + t[:, None]).mean(axis=1)
    pts = _project(rvec, t, cam)

    hR, t_est = fit_head_pose(pts, cam)
    fc = face_centre(hR, t_est)
    np.testing.assert_allclose(fc, fc_true, atol=1e-3)
    W, R = normalising_warp(hR, fc, cam)
    W_ref, R_ref = _notebook_normalise_warp(rvec, fc_true, cam)
    np.testing.assert_allclose(R, R_ref, atol=1e-6)
    np.testing.assert_allclose(W, W_ref, rtol=1e-6, atol=1e-9)

    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    crop_ref = cv2.warpPerspective(img, W_ref, (224, 224))  # the training crop for the same pose
    assert np.abs(normalised_crop(img, W).astype(int) - crop_ref.astype(int)).max() <= 1


@needs_cv2
@pytest.mark.parametrize("roll_deg", [-40.0, -10.0, 0.0, 15.0, 40.0])
def test_normalised_frame_removes_head_roll(roll_deg):
    cam = camera_matrix(640, 480)
    rvec = np.array([np.pi + 0.1, 0.2, np.deg2rad(roll_deg)])  # a face looking at the camera (the model is flipped)
    t = np.array([-30.0, 25.0, 650.0])
    pts = _project(rvec, t, cam)
    hR, t_est = fit_head_pose(pts, cam)
    W, R = normalising_warp(hR, face_centre(hR, t_est), cam)
    # The eye line is level in the normalised frame and the mouth is below it, whatever the roll.
    eye_line = R @ (hR @ (MODEL_POINTS_MM[3] - MODEL_POINTS_MM[2]))
    assert abs(eye_line[1]) < 1e-6
    mouth_below = R @ (hR @ (MODEL_POINTS_MM[4:].mean(axis=0) - MODEL_POINTS_MM[2:4].mean(axis=0)))
    assert mouth_below[1] > 0
    # In the crop the outer eye corners are level to within perspective (the eyes differ in depth when the face
    # is turned): about 3 px at this yaw on a 224 px crop. The exact check is the 3D one above.
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ W.T
    crop_xy = homog[:, :2] / homog[:, 2:3]
    assert abs(crop_xy[2, 1] - crop_xy[3, 1]) < 5.0


def test_pitchyaw_convention_matches_training_code():
    # Numpy versions of the notebook's to_vec (torch in training) and vec_to_pitchyaw.
    def to_vec(p, y):
        return np.array([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)])

    def vec_to_pitchyaw(g):
        return np.array([np.arcsin(-g[1]), np.arctan2(-g[0], -g[2])])

    for p, y in [(0.0, 0.0), (-0.2, 0.3), (0.4, -0.5)]:
        g = to_vec(p, y)
        np.testing.assert_allclose(pitchyaw_to_vector(np.array([p, y])), g, atol=1e-12)
        np.testing.assert_allclose(vec_to_pitchyaw(g), [p, y], atol=1e-9)


def test_trace_signs_follow_the_iris_convention_and_scale():
    # Looking to the image's right (negative yaw in the training convention) gives positive x; looking down gives positive y.
    x, y = trace_from_pitchyaw(np.array([0.0, -0.5]))
    assert x > 0 and abs(y) < 1e-12
    x, y = trace_from_pitchyaw(np.array([-0.5, 0.0]))
    assert abs(x) < 1e-12 and y > 0
    x, _ = trace_from_pitchyaw(np.array([0.0, -0.5]))
    assert x == pytest.approx(EYEBALL_RADIUS_MM / EYE_WIDTH_MM * np.sin(0.5), rel=1e-9)


def _tiny_gaze_onnx(path, input_name="face", output_name="pitch_yaw"):
    """An ONNX model with the gaze model's signature: the per-channel mean of the crop times a fixed (3, 2) matrix."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = np.array([[0.5, 0.1], [0.0, 0.25], [-0.2, 0.3]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", [input_name], ["as_float"], to=TensorProto.FLOAT),
            helper.make_node("ReduceMean", ["as_float"], ["means"], axes=[1, 2], keepdims=0),
            helper.make_node("MatMul", ["means", "w"], [output_name]),
        ],
        "tiny_gaze",
        [helper.make_tensor_value_info(input_name, TensorProto.UINT8, ["n", 224, 224, 3])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ["n", 2])],
        initializer=[numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return weights


def test_gaze_net_runs_the_onnx_signature(tmp_path):
    weights = _tiny_gaze_onnx(tmp_path / "tiny.onnx")
    faces = np.random.default_rng(3).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    means = faces.reshape(3, -1, 3).mean(axis=1)
    got = GazeNet(tmp_path / "tiny.onnx", tta=False).predict(faces)
    np.testing.assert_allclose(got, means @ weights, rtol=1e-4, atol=1e-3)
    # With test-time augmentation the mirrored crop's yaw is negated and averaged in. The mean colour is
    # mirror-invariant here, so the averaged yaw is zero and the pitch is unchanged.
    tta = GazeNet(tmp_path / "tiny.onnx", tta=True).predict(faces)
    np.testing.assert_allclose(tta[:, 0], got[:, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(tta[:, 1], 0.0, atol=1e-3)


def test_gaze_net_rejects_a_file_with_another_signature(tmp_path):
    _tiny_gaze_onnx(tmp_path / "other.onnx", input_name="image")
    with pytest.raises(ValueError, match="not the gaze model"):
        GazeNet(tmp_path / "other.onnx")


class _FixedFace:
    """Stands in for FaceTracker: one fixed face (or none), with open or closed eyes. Only the six pose points
    and the eye points used by the iris test are filled in."""

    def __init__(self, six, closed=False):
        self.six, self.closed = six, closed

    def points(self, rgb, timestamp_ms):
        if self.six is None:
            return None
        pts = np.zeros((478, 2))
        pts[list(LANDMARK_IDS)] = self.six
        gap = 0.0 if self.closed else 5.0
        right_corner, left_corner = self.six[2], self.six[3]
        pts[33], pts[133] = right_corner, right_corner + (30, 0)
        pts[159], pts[145] = right_corner + (15, -gap), right_corner + (15, gap)
        pts[468] = right_corner + (15, 0)
        pts[263], pts[362] = left_corner, left_corner + (-30, 0)
        pts[386], pts[374] = left_corner + (-15, -gap), left_corner + (-15, gap)
        pts[473] = left_corner + (-15, 0)
        return pts


@needs_cv2
def test_learned_tracker_gives_finite_gaze_and_handles_blinks_and_no_face(tmp_path):
    _tiny_gaze_onnx(tmp_path / "tiny.onnx")
    net = GazeNet(tmp_path / "tiny.onnx")
    rgb = np.random.default_rng(5).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    face = _project(np.array([np.pi + 0.1, -0.15, 0.05]), np.array([10.0, -20.0, 650.0]), camera_matrix(640, 480))

    x, y, found = LearnedGazeTracker(net, _FixedFace(face)).track(rgb, 1000)
    assert found is True and np.isfinite(x) and np.isfinite(y)
    x, y, found = LearnedGazeTracker(net, _FixedFace(face, closed=True)).track(rgb, 1000)
    assert found is True and np.isnan(x) and np.isnan(y)
    x, y, found = LearnedGazeTracker(net, _FixedFace(None)).track(rgb, 1000)
    assert found is False and np.isnan(x) and np.isnan(y)


def test_no_gaze_model_means_the_iris_method():
    client, _ = make_client(reading=False)  # the test app has no gaze model
    status = client.get("/v1/integrity/status").json()
    assert status["gaze_method"] == {"name": "mediapipe_iris", "model": None}


def test_missing_gaze_model_is_reported_and_never_falls_back(tmp_path):
    import io

    from PIL import Image

    app = server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(), gaze_model=tmp_path / "missing.onnx")
    client = TestClient(app)
    method = client.get("/v1/integrity/status").json()["gaze_method"]
    assert method["name"] == "learned_resnet18" and method["ready"] is False and method["error"]
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(buf, format="JPEG")
    r = client.post("/v1/integrity/gaze/frame", content=buf.getvalue())
    assert r.status_code == 503 and "model unavailable" in r.text


def test_learned_gaze_status_reports_a_loaded_model(tmp_path):
    _tiny_gaze_onnx(tmp_path / "tiny.onnx")
    app = server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(), gaze_model=tmp_path / "tiny.onnx")
    method = TestClient(app).get("/v1/integrity/status").json()["gaze_method"]
    assert method == {"name": "learned_resnet18", "model": "tiny.onnx", "tta": False, "ready": True, "error": None}


def test_tta_is_off_unless_the_flag_or_env_is_set(tmp_path, monkeypatch):
    from integrity.learned_gaze import tta_requested

    _tiny_gaze_onnx(tmp_path / "tiny.onnx")
    assert GazeNet(tmp_path / "tiny.onnx").tta is False
    assert tta_requested({}) is False and tta_requested({"OPENMEET_GAZE_TTA": "0"}) is False
    assert tta_requested({"OPENMEET_GAZE_TTA": "1"}) is True
    monkeypatch.delenv("OPENMEET_GAZE_TTA", raising=False)
    assert tta_requested() is False

    off = TestClient(server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(),
                                       gaze_model=tmp_path / "tiny.onnx"))
    assert off.get("/v1/integrity/status").json()["gaze_method"]["tta"] is False
    on = TestClient(server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(),
                                      gaze_model=tmp_path / "tiny.onnx", gaze_tta=True))
    assert on.get("/v1/integrity/status").json()["gaze_method"]["tta"] is True
