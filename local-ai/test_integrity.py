"""Tests for the answer-integrity endpoints (real models, fake face tracker: no camera, no MediaPipe)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from integrity.gaze import iris_offset
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


def test_reading_needs_ten_seconds_of_gaze():
    client, clock = make_client(reading=True)
    feed(client, clock, 4)
    body = client.post("/v1/integrity/reading", json={"seconds": 4}).json()
    assert body["level"] == "not_enough_gaze" and body["reading_probability"] is None
    assert client.post("/v1/integrity/reading", json={"start": 10, "end": 5}).status_code == 400


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
