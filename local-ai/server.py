"""OpenMeet local AI sidecar.

A small, local-only HTTP service that gives OpenMeet fully on-device
transcription (faster-whisper) and text embeddings (fastembed) through
OpenAI-compatible endpoints:

    GET  /health
    GET  /v1/models
    POST /v1/audio/transcriptions   (multipart: file, model, language, prompt, response_format,
                                     timestamp_granularities[]=word|segment with verbose_json)
    POST /v1/embeddings             ({"model": ..., "input": str | [str]})
    GET  /integrity, /v1/integrity/...  answer-integrity checks (integrity/__init__.py)

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

WHISPER_WINDOW_S = 30.0  # Whisper decodes audio in 30 s windows
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
    # Only when timestamps were asked for: [{"id", "start", "end", "text"}] and
    # [{"word", "start", "end"}], in seconds from the start of the audio.
    segments: list[dict[str, Any]] | None = None
    words: list[dict[str, Any]] | None = None


class Transcriber(Protocol):
    # Callers add word_timestamps=True only when they need timings, so a
    # backend without timestamps can keep this four-argument form.
    def transcribe(
        self, audio: np.ndarray, model: str, language: str | None, prompt: str | None
    ) -> TranscriptionResult: ...

    def loaded_model(self) -> str | None: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]: ...

    def loaded_model(self) -> str | None: ...


def is_gpu_runtime_error(exc: BaseException) -> bool:
    """True for errors caused by a missing / unusable CUDA runtime (cuBLAS,
    cuDNN, driver), which warrant falling back to the CPU."""
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "libcudart", "nvrtc"))


class WhisperTranscriber:
    """faster-whisper (CTranslate2) speech recognition, loaded lazily.

    `device` is "auto" (GPU when CUDA is usable, otherwise CPU), "cpu" or
    "cuda" (OPENMEET_WHISPER_DEVICE). With "auto", a machine that has an NVIDIA
    driver but no CUDA runtime libraries (cuBLAS/cuDNN) transparently falls
    back to the CPU instead of failing every request.
    """

    def __init__(self, default_model: str, device: str | None = None, compute_type: str | None = None) -> None:
        self.default_model = default_model
        self.device = (device or os.environ.get("OPENMEET_WHISPER_DEVICE", "auto")).lower()
        self.compute_type = compute_type or os.environ.get("OPENMEET_WHISPER_COMPUTE", "int8")
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last: str | None = None
        self._force_cpu = self.device == "cpu"

    def _load(self, name: str, device: str) -> Any:
        from faster_whisper import WhisperModel  # heavy import, lazy

        started = time.perf_counter()
        model = WhisperModel(
            name,
            device=device,
            compute_type=self.compute_type,
            download_root=str(MODEL_DIR / "whisper"),
        )
        LOG.info("loaded whisper model %s on %s in %.1fs", name, device, time.perf_counter() - started)
        return model

    def _get(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                device = "cpu" if self._force_cpu else self.device
                try:
                    self._models[name] = self._load(name, device)
                except Exception as exc:  # noqa: BLE001 - fall back only for GPU runtime problems
                    if device == "cpu" or self.device == "cuda" or not is_gpu_runtime_error(exc):
                        raise
                    LOG.warning("GPU unavailable for whisper (%s); using the CPU", type(exc).__name__)
                    self._force_cpu = True
                    self._models[name] = self._load(name, "cpu")
            self._last = name
            return self._models[name]

    def _fall_back_to_cpu(self) -> None:
        with self._lock:
            self._force_cpu = True
            self._models.clear()

    def _run(self, whisper: Any, audio: np.ndarray, language: str | None, prompt: str | None,
             word_timestamps: bool = False) -> tuple[list[Any], Any]:
        # Word timings come from aligning the decoded words afterwards
        # (cross-attention), so the decoding settings stay the same.
        timing: dict[str, Any] = {"word_timestamps": True} if word_timestamps else {}
        segments, info = whisper.transcribe(
            audio,
            language=language,
            initial_prompt=prompt or None,
            beam_size=int(os.environ.get("OPENMEET_WHISPER_BEAM", "1")),
            vad_filter=False,  # OpenMeet already segments speech with its own VAD
            condition_on_previous_text=False,
            without_timestamps=True,
            **timing,
        )
        if word_timestamps and float(getattr(info, "duration", 0.0)) <= WHISPER_WINDOW_S:
            # With word timestamps faster-whisper seeks back to the last word
            # and decodes the leftover tail again, which on short clips invents
            # words ("Thanks for watching."). A clip that fits one window keeps
            # the first decode only: the same text as without timestamps.
            first: list[Any] = []
            for seg in segments:
                if first and seg.seek != first[0].seek:
                    break
                first.append(seg)
            return first, info
        # The segment generator decodes lazily: consume it here so runtime
        # errors (e.g. missing cuBLAS) surface inside the fallback guard.
        return list(segments), info

    def transcribe(
        self, audio: np.ndarray, model: str, language: str | None, prompt: str | None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        whisper = self._get(model)
        try:
            segments, info = self._run(whisper, audio, language, prompt, word_timestamps)
        except Exception as exc:  # noqa: BLE001
            if self._force_cpu or self.device == "cuda" or not is_gpu_runtime_error(exc):
                raise
            LOG.warning("GPU transcription failed (%s); switching to the CPU", type(exc).__name__)
            self._fall_back_to_cpu()
            segments, info = self._run(self._get(model), audio, language, prompt, word_timestamps)
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            texts.append(seg.text)
            logprobs.append(seg.avg_logprob)
        confidence = None
        if logprobs:
            confidence = float(min(1.0, max(0.0, math.exp(sum(logprobs) / len(logprobs)))))
        timed, words = timings(segments) if word_timestamps else (None, None)
        return TranscriptionResult(
            text="".join(texts).strip(),
            language=getattr(info, "language", None),
            duration=float(getattr(info, "duration", len(audio) / 16000)),
            confidence=confidence,
            segments=timed,
            words=words,
        )

    def loaded_model(self) -> str | None:
        return self._last


def timings(segments: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """faster-whisper segments decoded with word timestamps, in OpenAI's verbose_json shape.

    A segment's times are its first word's start and last word's end (the
    decoder runs without timestamp tokens, so its own segment times only
    mark the 30 s window).
    """
    out_segments: list[dict[str, Any]] = []
    out_words: list[dict[str, Any]] = []
    for seg in segments:
        text = " ".join(str(seg.text).split())
        words = [w for w in (getattr(seg, "words", None) or []) if str(w.word).strip()]
        out_words += [{"word": str(w.word).strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
                      for w in words]
        if text:
            start, end = (words[0].start, words[-1].end) if words else (seg.start, seg.end)
            out_segments.append({"id": len(out_segments), "start": round(float(start), 3), "end": round(float(end), 3),
                                 "text": text})
    return out_segments, out_words


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
        # fastembed's default batch of 256 long texts needs several GB of RAM; same vectors in smaller batches.
        return [vec.astype(float).tolist() for vec in engine.embed(list(texts), batch_size=32)]

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


GRANULARITIES = ("word", "segment")


def resolve_granularities(values: list[str]) -> set[str]:
    found = {v.strip().lower() for v in values if v.strip()}
    if found - set(GRANULARITIES):
        raise HTTPException(status_code=400, detail="timestamp_granularities[] must be word or segment")
    return found


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
    integrity_tracker: Any = None,
    clock: Any = time.time,
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
        granularities: list[str] | None = Form(default=None, alias="timestamp_granularities[]"),
        granularities_plain: list[str] | None = Form(default=None, alias="timestamp_granularities"),
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
        # As in OpenAI's API: timestamps only with verbose_json; segments unless
        # only words are asked for, words only when asked for.
        granularity = resolve_granularities((granularities or []) + (granularities_plain or []))
        verbose = response_format == "verbose_json"
        want_words = verbose and "word" in granularity
        want_segments = verbose and ("segment" in granularity or not granularity)
        try:
            audio = decode_audio(data)
        except Exception as exc:  # noqa: BLE001 - report decoding problems as 400
            raise HTTPException(status_code=400, detail=f"could not decode audio: {type(exc).__name__}") from exc
        started = time.perf_counter()
        prompt = (prompt or "")[:1000] or None
        with infer_lock:
            if want_words or want_segments:
                result = transcriber.transcribe(audio, name, lang, prompt, word_timestamps=True)
            else:
                result = transcriber.transcribe(audio, name, lang, prompt)
        LOG.info(
            "transcribed %.1fs audio in %.0f ms (%d chars)",
            len(audio) / 16000,
            (time.perf_counter() - started) * 1000,
            len(result.text.strip()),
        )
        text = " ".join(result.text.split())
        if response_format == "text":
            return JSONResponse(content=text)
        body: dict[str, Any] = {"text": text}
        if verbose:
            body = {"task": "transcribe", "language": result.language, "duration": result.duration, "text": text}
            if want_segments:
                fallback = [{"id": 0, "start": 0.0, "end": round(result.duration, 3), "text": text}] if text else []
                body["segments"] = result.segments if result.segments is not None else fallback
            if want_words:
                body["words"] = result.words if result.words is not None else []
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

    # Answer-integrity checks: AI-written answers and reading (optional extras).
    from integrity import add_routes as add_integrity_routes

    add_integrity_routes(app, check_auth, MODEL_DIR, tracker_factory=integrity_tracker, clock=clock)
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
    # HTTP client libraries log full request URLs at INFO (model downloads); keep them quiet.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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
