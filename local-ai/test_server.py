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


class TimedTranscriber(FakeTranscriber):
    """Also gives timings when asked (word_timestamps=True), as WhisperTranscriber does."""

    def __init__(self) -> None:
        super().__init__()
        self.timed: list[bool] = []

    def transcribe(self, audio, model, language, prompt, word_timestamps=False):  # type: ignore[no-untyped-def]
        self.timed.append(word_timestamps)
        result = super().transcribe(audio, model, language, prompt)
        if word_timestamps:
            result.segments = [{"id": 0, "start": 0.1, "end": 0.45, "text": "How did you scale it?"}]
            result.words = [{"word": w, "start": 0.1 + 0.07 * i, "end": 0.15 + 0.07 * i}
                            for i, w in enumerate(["How", "did", "you", "scale", "it?"])]
        return result


def post_audio(client, **data):  # type: ignore[no-untyped-def]
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes(), "audio/wav")}, data=data)


def test_word_timestamps_openai_shape():
    t = TimedTranscriber()
    client = TestClient(server.create_app(transcriber=t, embedder=FakeEmbedder()))
    body = post_audio(client, response_format="verbose_json", **{"timestamp_granularities[]": "word"}).json()
    assert body["task"] == "transcribe" and body["language"] == "en" and body["duration"] == 0.5
    assert body["text"] == "How did you scale it?" and "segments" not in body
    assert [w["word"] for w in body["words"]] == ["How", "did", "you", "scale", "it?"]
    assert set(body["words"][0]) == {"word", "start", "end"}
    # Both granularities: the repeated form field, as OpenAI's clients send it.
    both = post_audio(client, response_format="verbose_json", **{"timestamp_granularities[]": ["word", "segment"]}).json()
    assert both["segments"] == [{"id": 0, "start": 0.1, "end": 0.45, "text": "How did you scale it?"}]
    assert len(both["words"]) == 5
    # verbose_json alone gives segments (OpenAI's default granularity).
    plain = post_audio(client, response_format="verbose_json").json()
    assert "segments" in plain and "words" not in plain
    assert t.timed == [True, True, True]


def test_requests_without_timestamps_are_unchanged():
    t = TimedTranscriber()
    client = TestClient(server.create_app(transcriber=t, embedder=FakeEmbedder()))
    assert post_audio(client).json() == {"text": "How did you scale it?", "confidence": 0.87}
    # Granularities only apply to verbose_json, as in OpenAI's API.
    assert post_audio(client, **{"timestamp_granularities[]": "word"}).json() == {"text": "How did you scale it?", "confidence": 0.87}
    assert post_audio(client, response_format="text").json() == "How did you scale it?"
    assert t.timed == [False, False, False]
    # A backend without timestamps still serves plain requests (four-argument transcribe).
    fake = TestClient(server.create_app(transcriber=FakeTranscriber(), embedder=FakeEmbedder()))
    assert post_audio(fake).status_code == 200


def test_bad_granularity_is_rejected():
    client = TestClient(server.create_app(transcriber=TimedTranscriber(), embedder=FakeEmbedder()))
    r = post_audio(client, response_format="verbose_json", **{"timestamp_granularities[]": "phoneme"})
    assert r.status_code == 400


def test_timings_from_faster_whisper_segments():
    from types import SimpleNamespace as NS

    segments = [
        NS(text=" So we cached it.", start=0.0, end=30.0, avg_logprob=-0.1,
           words=[NS(word=" So", start=0.52, end=0.7), NS(word=" we", start=0.7, end=0.81),
                  NS(word=" ", start=0.81, end=0.82), NS(word=" cached", start=0.9, end=1.3), NS(word=" it.", start=1.3, end=1.51)]),
        NS(text=" ", start=30.0, end=30.0, avg_logprob=-0.1, words=[]),
    ]
    segs, words = server.timings(segments)
    assert segs == [{"id": 0, "start": 0.52, "end": 1.51, "text": "So we cached it."}]
    assert [w["word"] for w in words] == ["So", "we", "cached", "it."]
    assert words[2] == {"word": "cached", "start": 0.9, "end": 1.3}


def test_whisper_asks_for_word_timestamps_only_when_needed(monkeypatch):
    from types import SimpleNamespace as NS

    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            words = [NS(word=" hi", start=0.2, end=0.4)] if kwargs.get("word_timestamps") else None
            return iter([NS(text=" hi", start=0.0, end=1.0, avg_logprob=-0.2, words=words)]), NS(language="en", duration=1.0)

    t = server.WhisperTranscriber("base.en", device="cpu")
    monkeypatch.setattr(t, "_load", lambda name, device: FakeModel())
    plain = t.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None)
    timed = t.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None, word_timestamps=True)
    assert "word_timestamps" not in calls[0] and calls[0]["without_timestamps"] is True
    assert calls[1]["word_timestamps"] is True and calls[1]["without_timestamps"] is True  # same decoding
    assert plain.words is None and plain.segments is None
    assert timed.words == [{"word": "hi", "start": 0.2, "end": 0.4}] and timed.segments[0]["start"] == 0.2


def test_word_timestamps_do_not_redecode_the_tail_of_a_short_clip(monkeypatch):
    """faster-whisper re-seeks to the last word when aligning words; on a clip that fits one window
    the re-decoded silence comes back as invented words, which must not reach the transcript."""
    from types import SimpleNamespace as NS

    decoded: list[int] = []

    def gen(duration):  # type: ignore[no-untyped-def]
        for seek, text in ((0, " So we cached it."), (856, " months."), (880, " Thanks for watching.")):
            decoded.append(seek)
            yield NS(text=text, seek=seek, start=seek / 100, end=seek / 100 + 0.5, avg_logprob=-0.2,
                     words=[NS(word=" " + w, start=seek / 100, end=seek / 100 + 0.2) for w in text.split()])

    class FakeModel:
        def __init__(self, duration: float) -> None:
            self.duration = duration

        def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            return gen(self.duration), NS(language="en", duration=self.duration)

    t = server.WhisperTranscriber("base.en", device="cpu")
    monkeypatch.setattr(t, "_load", lambda name, device: FakeModel(9.7))
    short = t.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None, word_timestamps=True)
    assert short.text == "So we cached it." and [w["word"] for w in short.words] == ["So", "we", "cached", "it."]
    assert decoded == [0, 856]  # stopped at the first re-seek, without decoding further
    long = server.WhisperTranscriber("base.en", device="cpu")
    monkeypatch.setattr(long, "_load", lambda name, device: FakeModel(45.0))
    assert "months." in long.transcribe(np.zeros(16000, dtype=np.float32), "base.en", "en", None, word_timestamps=True).text


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
