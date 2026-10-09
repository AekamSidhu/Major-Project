"""Tests for the OpenMeet local AI sidecar (no models downloaded: backends are faked)."""

from __future__ import annotations

import io
import math
import struct
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str | None, str | None]] = []

    def transcribe(self, audio, model, language, prompt):  # type: ignore[no-untyped-def]
        self.calls.append((len(audio), model, language, prompt))
        return server.TranscriptionResult(text=" How did you scale it? ", language="en", duration=len(audio) / 16000, confidence=0.87)

    def loaded_model(self) -> str | None:
        return None


class FakeEmbedder:
    def embed(self, texts, model):  # type: ignore[no-untyped-def]
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def loaded_model(self) -> str | None:
        return None


def wav_bytes(seconds: float = 0.5, rate: int = 16000, channels: int = 1) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(i / 10))) * channels for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture()
def client_and_fakes():
    t, e = FakeTranscriber(), FakeEmbedder()
    app = server.create_app(transcriber=t, embedder=e)
    return TestClient(app), t, e


def test_health(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["transcription_model"] == server.DEFAULT_WHISPER_MODEL
    assert r.headers["cache-control"] == "no-store"


def test_transcription_openai_compatible(client_and_fakes):
    client, t, _ = client_and_fakes
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "language": "en", "prompt": "Kafka"},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "How did you scale it?", "confidence": 0.87}
    n, model, lang, prompt = t.calls[0]
    assert n == 8000
    assert model == server.DEFAULT_WHISPER_MODEL  # alias mapped to the local default
    assert lang == "en" and prompt == "Kafka"


def test_transcription_resamples_and_downmixes(client_and_fakes):
    client, t, _ = client_and_fakes
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(1.0, 48000, 2), "audio/wav")})
    assert r.status_code == 200
    assert t.calls[0][0] == 16000


def test_rejects_unknown_model_and_bad_language(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"model": "../../evil"})
    assert r.status_code == 400
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data={"language": "en; rm -rf"})
    assert r.status_code == 400


def test_empty_and_undecodable_audio(client_and_fakes):
    client, _, _ = client_and_fakes
    assert client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400


def test_embeddings(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.post("/v1/embeddings", json={"input": ["ab", "abcd"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert data[1]["embedding"][0] == 4.0
    r = client.post("/v1/embeddings", json={"input": "x", "model": "other"})
    assert r.status_code == 400
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 400


def test_token_auth():
    app = server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder(), token="secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 200  # health stays open for status checks
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 401
    ok = client.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_decode_audio_rejects_garbage():
    with pytest.raises(Exception):
        server.decode_audio(b"RIFF\x00\x00\x00\x00WAVEjunk")


def test_loopback_check():
    assert server.is_loopback("127.0.0.1")
    assert server.is_loopback("localhost")
    assert server.is_loopback("::1")
    assert not server.is_loopback("0.0.0.0")
    assert not server.is_loopback("192.168.1.5")


def test_decode_audio_float_range():
    audio = server.decode_audio(wav_bytes(0.1))
    assert audio.dtype == np.float32
    assert float(np.max(np.abs(audio))) <= 1.0


def test_gpu_runtime_errors_are_recognized():
    assert server.is_gpu_runtime_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"))
    assert server.is_gpu_runtime_error(RuntimeError("Could not load library cudnn_ops64_9.dll"))
    assert not server.is_gpu_runtime_error(ValueError("invalid audio"))


def test_whisper_falls_back_to_cpu_on_missing_cuda_runtime(monkeypatch):
    """A machine with an NVIDIA driver but no cuBLAS must still transcribe."""
    loads: list[str] = []

    class FakeSegment:
        text = " hello "
        avg_logprob = -0.1

    class FakeModel:
        def __init__(self, device: str) -> None:
            self.device = device

        def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            def gen():
                if self.device != "cpu":
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                yield FakeSegment()

            class Info:
                language = "en"
                duration = 1.0

            return gen(), Info()

    t = server.WhisperTranscriber("base.en", device="auto")

    def fake_load(name: str, device: str):  # type: ignore[no-untyped-def]
        loads.append(device)
        return FakeModel(device)

    monkeypatch.setattr(t, "_load", fake_load)
    result = t.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None)
    assert result.text == "hello"
    assert loads == ["auto", "cpu"]
    # Subsequent calls stay on the CPU without retrying the GPU.
    t.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None)
    assert loads == ["auto", "cpu"]
