"""Tests for OpenMeet Room: who may call what, signaling, transcription and answer checks."""

from __future__ import annotations

import urllib.error

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import room_server
import server
from room import is_phantom
from room.followups import FollowupWriter, parse_followups, rule_followups
from test_integrity import AI_ANSWER, Clock, SawtoothTracker, feed
from test_server import wav_bytes

LOCAL = ("127.0.0.1", 50000)
REMOTE = ("192.168.1.50", 50000)


class ScriptedTranscriber:
    def __init__(self, text: str, confidence: float = 0.9) -> None:
        self.text, self.confidence = text, confidence

    def transcribe(self, audio, model, language, prompt):  # type: ignore[no-untyped-def]
        return server.TranscriptionResult(text=self.text, language="en", duration=len(audio) / 16000, confidence=self.confidence)

    def loaded_model(self) -> str | None:
        return None


def no_model(messages):  # type: ignore[no-untyped-def]
    raise urllib.error.URLError("connection refused")


def make(text: str = "I built the cache layer myself", confidence: float = 0.9, reading: bool = True, chat=no_model):  # type: ignore[no-untyped-def]
    clock = Clock()
    app = room_server.create_room_app(8443, "https", "base.en", transcriber=ScriptedTranscriber(text, confidence),
                                      tracker_factory=lambda: SawtoothTracker(clock, reading), clock=clock,
                                      followup_writer=FollowupWriter(chat=chat))
    return app, clock


def test_interviewer_endpoints_answer_only_this_computer():
    app, _ = make()
    remote = TestClient(app, client=REMOTE)
    assert remote.get("/host").status_code == 403
    assert remote.post("/api/rooms").status_code == 403
    assert remote.post("/v1/integrity/echo", json={"text": AI_ANSWER}).status_code == 403
    assert remote.get("/").status_code == 404
    local = TestClient(app, client=LOCAL)
    room = local.post("/api/rooms").json()
    assert room["invites"][-1].startswith("https://localhost:8443/join/" + room["room"])
    # The candidate's side is reachable from the network.
    assert remote.get(f"/join/{room['room']}").status_code == 200
    assert remote.get("/room.js").status_code == 200


def test_signaling_relays_between_host_and_guest():
    app, _ = make()
    local, remote = TestClient(app, client=LOCAL), TestClient(app, client=REMOTE)
    room = local.post("/api/rooms").json()
    guest_key = room["invites"][0].split("k=")[1]
    with local.websocket_connect(f"/ws/{room['room']}?role=host&k={room['host_key']}") as host:
        assert host.receive_json() == {"type": "hello", "role": "host", "peer_present": False}
        with remote.websocket_connect(f"/ws/{room['room']}?role=guest&k={guest_key}&name=Asha") as guest:
            assert guest.receive_json()["peer_present"] is True
            assert host.receive_json() == {"type": "peer-joined", "role": "guest", "name": "Asha"}
            host.send_json({"type": "offer", "sdp": {"type": "offer", "sdp": "v=0"}})
            assert guest.receive_json()["type"] == "offer"
            guest.send_json({"type": "answer", "sdp": {"type": "answer", "sdp": "v=0"}})
            assert host.receive_json()["type"] == "answer"
        assert host.receive_json() == {"type": "peer-left", "role": "guest"}


def test_wrong_keys_and_remote_hosts_are_refused():
    app, _ = make()
    local, remote = TestClient(app, client=LOCAL), TestClient(app, client=REMOTE)
    room = local.post("/api/rooms").json()
    for client, url in ((remote, f"/ws/{room['room']}?role=guest&k=wrong"),
                        (remote, f"/ws/{room['room']}?role=host&k={room['host_key']}"),  # right key, wrong computer
                        (local, f"/ws/nosuchroom?role=guest&k=x")):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(url) as ws:
                ws.receive_json()


def test_transcription_keeps_speech_and_drops_phantoms():
    app, _ = make("Tell me about a system you scaled.")
    local = TestClient(app, client=LOCAL)
    room = local.post("/api/rooms").json()["room"]
    seg = local.post(f"/api/rooms/{room}/transcribe?speaker=interviewer&start=1&end=3", content=wav_bytes(1.0)).json()
    assert seg["text"] == "Tell me about a system you scaled."
    assert local.post(f"/api/rooms/{room}/transcribe?speaker=judge&start=1&end=3", content=wav_bytes(1.0)).status_code == 400
    for text, conf in (("Thank you.", 0.9), ("We'll see you in the next one.", 0.9), ("real words here", 0.1)):
        app2, _ = make(text, conf)
        c = TestClient(app2, client=LOCAL)
        r2 = c.post("/api/rooms").json()["room"]
        assert c.post(f"/api/rooms/{r2}/transcribe?speaker=candidate&start=1&end=2", content=wav_bytes(0.5)).json()["text"] == ""
    assert is_phantom("Thanks for watching!") and not is_phantom("thanks for the question")


def test_answer_check_and_report():
    app, clock = make()
    local = TestClient(app, client=LOCAL)
    room = local.post("/api/rooms").json()["room"]
    start = clock.t
    feed(local, clock, 25)  # the candidate's eyes read for 25 s
    answer = local.post(f"/api/rooms/{room}/answers", json={"start": start, "end": clock.t, "text": AI_ANSWER,
                                                            "question": "How would you scale it?"}).json()
    assert answer["index"] == 1
    assert answer["echo"]["level"] == "high"
    assert answer["reading"]["reading_probability"] > 0.8
    report = local.get(f"/api/rooms/{room}/report").json()
    assert report["answers"][0]["question"] == "How would you scale it?"


ANSWER = ("We moved the checkout service to Kubernetes and cut p99 latency by 40% over three months. "
          "We also added a Redis cache in front of the product catalogue.")


def test_followups_come_from_the_model_and_avoid_repeats():
    prompts: list[str] = []

    def chat(messages):  # type: ignore[no-untyped-def]
        prompts.append(messages[-1]["content"])
        n = len(prompts)
        return (f'{{"question":"What happens to checkout if Redis goes down? ({n})","reason":"No failure handling.",'
                f'"priority":"high","trigger":"added a Redis cache"}}\n'
                f'{{"question":"How did you measure the 40% drop? ({n})","priority":"urgent"}}\nnot json')

    app, _ = make(chat=chat)
    local, remote = TestClient(app, client=LOCAL), TestClient(app, client=REMOTE)
    room = local.post("/api/rooms").json()["room"]
    assert local.post(f"/api/rooms/{room}/followups", json={}).status_code == 409  # no answers yet
    local.post(f"/api/rooms/{room}/answers", json={"start": 0, "end": 20, "text": ANSWER, "question": "Tell me about a project."})
    assert remote.post(f"/api/rooms/{room}/followups", json={}).status_code == 403

    first = local.post(f"/api/rooms/{room}/followups", json={"answer": 1}).json()
    assert first["source"] == "ai" and first["model"] == "llama3.2" and first["answer"] == 1
    assert [f["priority"] for f in first["items"]] == ["high", "normal"]  # unknown priority becomes normal
    assert "<candidate_answer>" in prompts[0] and "Kubernetes" in prompts[0] and "Tell me about a project." in prompts[0]

    local.post(f"/api/rooms/{room}/followups", json={})  # again for the latest answer
    assert "What happens to checkout if Redis goes down? (1)" in prompts[1].split("<previous_suggestions>")[1]
    report = local.get(f"/api/rooms/{room}/report").json()
    assert report["answers"][0]["followups"]["items"][0]["question"].endswith("(2)")
    assert local.get("/api/status").json()["followup_model_ok"] is True


def test_followups_fall_back_to_rules_without_a_model():
    app, _ = make()
    local = TestClient(app, client=LOCAL)
    room = local.post("/api/rooms").json()["room"]
    local.post(f"/api/rooms/{room}/answers", json={"start": 0, "end": 20, "text": ANSWER})
    result = local.post(f"/api/rooms/{room}/followups", json={}).json()
    assert result["source"] == "rules" and result["model"] is None
    questions = " ".join(f["question"] for f in result["items"])
    assert result["items"][0]["priority"] == "high"
    assert "Kubernetes" in questions or "yourself" in questions
    assert all(f["trigger"] for f in result["items"])  # each one quotes the answer
    assert local.get("/api/status").json()["followup_model_ok"] is False


def test_followup_parsing_and_rules_handle_odd_input():
    assert parse_followups('```json\n[{"question": "Why Redis over Memcached?"}, {"question": "Why Redis over Memcached?"}]\n```') == [
        {"question": "Why Redis over Memcached?", "reason": "", "priority": "normal", "trigger": ""}]
    assert parse_followups('{"followups": [{"question": "short"}, "x"]}') == []
    assert parse_followups("I cannot help with that.") == []
    only = rule_followups("", "Yes.", [])
    assert len(only) == 1 and "example" in only[0]["question"]
    asked = rule_followups("", ANSWER, ["Which part of this did you build or decide yourself?"])
    assert all("yourself" not in f["question"] for f in asked)
