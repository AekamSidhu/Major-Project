"""OpenMeet local AI sidecar.

A small, local-only HTTP service that gives OpenMeet fully on-device
transcription (faster-whisper) and text embeddings (fastembed) through
OpenAI-compatible endpoints:

    GET  /health
    GET  /v1/models
    POST /v1/audio/transcriptions   (multipart: file, model, language, prompt)
    POST /v1/embeddings             ({"model": ..., "input": str | [str]})

Privacy:
  * Binds to 127.0.0.1 by default and refuses non-loopback binds unless
    --allow-remote is passed explicitly.
  * Never logs transcript text, prompts or embedding inputs.
  * Optional bearer token (OPENMEET_LOCAL_AI_TOKEN) for defence in depth.

Run:  python server.py --port 8765
"""

from __future__ import annotations

import argparse
import io
import ipaddress
import logging
import math
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

__version__ = "0.1.0"

LOG = logging.getLogger("openmeet.local_ai")

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("OPENMEET_MODEL_DIR", HERE / ".models"))

DEFAULT_WHISPER_MODEL = os.environ.get("OPENMEET_WHISPER_MODEL", "base.en")
DEFAULT_EMBED_MODEL = os.environ.get("OPENMEET_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# faster-whisper model names we allow the client to request. Restricting the
# list prevents a client from triggering arbitrary downloads.
ALLOWED_WHISPER_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo", "turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
}
# OpenAI-style aliases mapped to the default local model.
WHISPER_ALIASES = {"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "default"}

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_EMBED_INPUTS = 256
MAX_EMBED_CHARS = 8000


# ---------------------------------------------------------------------------
# Backends (injectable for tests)
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    confidence: float | None


class Transcriber(Protocol):
    def transcribe(
        self, audio: np.ndarray, model: str, language: str | None, prompt: str | None
    ) -> TranscriptionResult: ...

    def loaded_model(self) -> str | None: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]: ...

    def loaded_model(self) -> str | None: ...


class WhisperTranscriber:
    """faster-whisper (CTranslate2) speech recognition, loaded lazily."""

    def __init__(self, default_model: str, device: str = "auto", compute_type: str = "int8") -> None:
        self.default_model = default_model
        self.device = device
        self.compute_type = compute_type
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last: str | None = None

    def _get(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                from faster_whisper import WhisperModel  # heavy import, lazy

                started = time.perf_counter()
                self._models[name] = WhisperModel(
                    name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(MODEL_DIR / "whisper"),
                )
                LOG.info("loaded whisper model %s in %.1fs", name, time.perf_counter() - started)
            self._last = name
            return self._models[name]

    def transcribe(
        self, audio: np.ndarray, model: str, language: str | None, prompt: str | None
    ) -> TranscriptionResult:
        whisper = self._get(model)
        segments, info = whisper.transcribe(
            audio,
            language=language,
            initial_prompt=prompt or None,
            beam_size=int(os.environ.get("OPENMEET_WHISPER_BEAM", "1")),
            vad_filter=False,  # OpenMeet already segments speech with its own VAD
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            texts.append(seg.text)
            logprobs.append(seg.avg_logprob)
        confidence = None
        if logprobs:
            confidence = float(min(1.0, max(0.0, math.exp(sum(logprobs) / len(logprobs)))))
        return TranscriptionResult(
            text="".join(texts).strip(),
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", len(audio) / 16000)),
            confidence=confidence,
        )

    def loaded_model(self) -> str | None:
        return self._last


class FastEmbedder:
    """fastembed (ONNX) sentence embeddings, loaded lazily."""

    def __init__(self, default_model: str) -> None:
        self.default_model = default_model
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last: str | None = None

    def _get(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                from fastembed import TextEmbedding  # heavy import, lazy

                self._models[name] = TextEmbedding(
                    model_name=name, cache_dir=str(MODEL_DIR / "embeddings")
                )
            self._last = name
            return self._models[name]

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        engine = self._get(model)
        return [vec.astype(float).tolist() for vec in engine.embed(list(texts))]

    def loaded_model(self) -> str | None:
        return self._last


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def decode_audio(data: bytes) -> np.ndarray:
    """Decodes WAV (PCM 8/16/32-bit) into 16 kHz mono float32.

    Other containers are decoded with faster-whisper's PyAV-based decoder.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width {width}")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if rate != 16000 and len(audio) > 0:
            duration = len(audio) / rate
            target = np.linspace(0.0, duration, int(round(duration * 16000)), endpoint=False)
            source = np.arange(len(audio)) / rate
            audio = np.interp(target, source, audio).astype(np.float32)
        return audio.astype(np.float32)
    from faster_whisper.audio import decode_audio as fw_decode  # PyAV

    return fw_decode(io.BytesIO(data), sampling_rate=16000)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str] = Field(...)


def resolve_whisper_model(requested: str | None, default: str) -> str:
    name = (requested or "").strip()
    if not name or name in WHISPER_ALIASES:
        return default
    if name not in ALLOWED_WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"unsupported model '{name}'")
    return name


def create_app(
    transcriber: Transcriber | None = None,
    embedder: Embedder | None = None,
    token: str | None = None,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> FastAPI:
    app = FastAPI(title="OpenMeet local AI", version=__version__, docs_url=None, redoc_url=None)
    transcriber = transcriber or WhisperTranscriber(whisper_model)
    embedder = embedder or FastEmbedder(embed_model)
    # Model inference is CPU-bound; serialize it so concurrent requests do not
    # thrash the CPU (OpenMeet sends at most a couple at a time).
    infer_lock = threading.Lock()

    def check_auth(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.middleware("http")
    async def no_store(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "transcription_model": transcriber.loaded_model() or whisper_model,
            "embedding_model": embedder.loaded_model() or embed_model,
        }

    @app.get("/v1/models", dependencies=[Depends(check_auth)])
    def models() -> dict[str, Any]:
        ids = sorted(ALLOWED_WHISPER_MODELS) + [embed_model]
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"} for m in ids]}

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(check_auth)])
    async def transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(default=None),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str | None = Form(default="json"),
    ) -> JSONResponse:
        data = await file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="audio too large")
        if not data:
            raise HTTPException(status_code=400, detail="empty audio")
        name = resolve_whisper_model(model, whisper_model)
        lang = (language or "").strip().lower() or None
        if lang is not None and (len(lang) > 8 or not lang.replace("-", "").isalpha()):
            raise HTTPException(status_code=400, detail="invalid language")
        try:
            audio = decode_audio(data)
        except Exception as exc:  # noqa: BLE001 - report decoding problems as 400
            raise HTTPException(status_code=400, detail=f"could not decode audio: {type(exc).__name__}") from exc
        started = time.perf_counter()
        with infer_lock:
            result = transcriber.transcribe(audio, name, lang, (prompt or "")[:1000] or None)
        LOG.info(
            "transcribed %.1fs audio in %.0f ms (%d chars)",
            len(audio) / 16000,
            (time.perf_counter() - started) * 1000,
            len(result.text),
        )
        if response_format == "text":
            return JSONResponse(content=result.text)
        body: dict[str, Any] = {"text": result.text}
        if response_format == "verbose_json":
            body.update({"language": result.language, "duration": result.duration})
        if result.confidence is not None:
            body["confidence"] = result.confidence
        return JSONResponse(content=body)

    @app.post("/v1/embeddings", dependencies=[Depends(check_auth)])
    def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")
        if len(texts) > MAX_EMBED_INPUTS:
            raise HTTPException(status_code=400, detail=f"at most {MAX_EMBED_INPUTS} inputs")
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        name = (req.model or "").strip() or embed_model
        if name != embed_model:
            raise HTTPException(status_code=400, detail=f"unsupported model '{name}'")
        with infer_lock:
            vectors = embedder.embed(texts, name)
        return {
            "object": "list",
            "model": name,
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
        }

    return app


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMeet local AI sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("OPENMEET_LOCAL_AI_PORT", "8765")))
    parser.add_argument("--allow-remote", action="store_true", help="allow binding to a non-loopback address")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--download-models", action="store_true", help="download the default models and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not is_loopback(args.host) and not args.allow_remote:
        parser.error("refusing to bind to a non-loopback address without --allow-remote")

    if args.download_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        WhisperTranscriber(args.whisper_model)._get(args.whisper_model)
        FastEmbedder(args.embed_model)._get(args.embed_model)
        print(f"models ready in {MODEL_DIR}")
        return

    import uvicorn

    app = create_app(
        token=os.environ.get("OPENMEET_LOCAL_AI_TOKEN") or None,
        whisper_model=args.whisper_model,
        embed_model=args.embed_model,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
