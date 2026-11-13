"""Answer-integrity checks for interviewers: AI-written answers and reading.

Adds to the sidecar:

    GET  /integrity                       the live check page (HTML)
    GET  /v1/integrity/status             which models are ready, live gaze state
    POST /v1/integrity/echo               {"text"} -> does the answer read like an LLM wrote it?
    POST /v1/integrity/gaze/frame         one JPEG/PNG frame (raw body) from the browser
    POST /v1/integrity/gaze/screen        {"left","top","width","height"} capture that region at 10 fps
    POST /v1/integrity/gaze/stop          stop screen capture
    POST /v1/integrity/gaze/reset         forget the gaze kept so far
    POST /v1/integrity/reading            {"start","end"} (epoch s) or {"seconds"} -> reading score

Everything runs on this machine. Frames are turned into gaze samples and
dropped; samples live in memory for ten minutes. Nothing is logged except
timings. Optional packages: onnxruntime (reading), mediapipe (gaze),
mss (screen capture); without them the matching endpoints answer 503.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .gaze import FaceTracker, GazeMonitor, Region, Tracker, decode_image
from .models import FPS, WINDOW_FRAMES, EchoModel, ReadingModel

LOG = logging.getLogger("openmeet.local_ai.integrity")
PAGE = Path(__file__).resolve().parent / "page.html"
# The shared look of the sidecar's pages (same design system as the app).
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
UI_FILES = {"openmeet.css": "text/css", "inter.woff2": "font/woff2"}
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_TEXT_CHARS = 20000


class EchoRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS)


class ScreenRequest(BaseModel):
    left: int = 0
    top: int = 0
    width: int = Field(..., gt=16, le=8192)
    height: int = Field(..., gt=16, le=8192)


class ReadingRequest(BaseModel):
    start: float | None = None
    end: float | None = None
    seconds: float | None = Field(default=None, gt=0, le=600)


class _Lazy:
    """Loads a model on first use; remembers why it couldn't."""

    def __init__(self, make: Callable[[], Any]) -> None:
        self._make, self._value, self.error = make, None, None
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            if self._value is None and self.error is None:
                try:
                    self._value = self._make()
                except Exception as error:  # missing package or file
                    self.error = f"{type(error).__name__}: {error}"[:300]
            if self._value is None:
                raise HTTPException(status_code=503, detail=f"model unavailable: {self.error}")
            return self._value

    def ready(self) -> bool:
        try:
            self.get()
            return True
        except HTTPException:
            return False


def add_routes(
    app: FastAPI,
    check_auth: Callable[..., None],
    model_dir: Path,
    tracker_factory: Callable[[], Tracker] | None = None,
    clock: Callable[[], float] = time.time,
    echo: EchoModel | None = None,
    reading: ReadingModel | None = None,
) -> GazeMonitor:
    echo_model = _Lazy(lambda: echo or EchoModel())
    reading_model = _Lazy(lambda: reading or ReadingModel())
    monitor = GazeMonitor(tracker_factory or (lambda: FaceTracker(model_dir)), clock)
    auth = [Depends(check_auth)]

    @app.get("/ui/{name}", include_in_schema=False)
    def ui_asset(name: str) -> FileResponse:
        if name not in UI_FILES:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(UI_DIR / name, media_type=UI_FILES[name], headers={"Cache-Control": "max-age=86400"})

    @app.get("/integrity", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/v1/integrity/status", dependencies=auth)
    def status() -> dict[str, Any]:
        out: dict[str, Any] = {"echo_ready": echo_model.ready(), "reading_ready": reading_model.ready(),
                               "echo_error": echo_model.error, "reading_error": reading_model.error, "gaze": monitor.status()}
        if out["reading_ready"]:
            live = reading_model.get().assess(monitor.series(out["gaze"]["now"] - WINDOW_FRAMES / FPS, out["gaze"]["now"]))
            out["live_reading"] = live.as_dict() if live else None
            trace = monitor.series(out["gaze"]["now"] - 30.0, out["gaze"]["now"])
            out["trace_x"] = [None if v != v else round(float(v), 4) for v in trace[:, 0]]
        return out

    @app.post("/v1/integrity/echo", dependencies=auth)
    def echo_check(req: EchoRequest) -> dict[str, Any]:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")
        return echo_model.get().assess(req.text)

    @app.post("/v1/integrity/gaze/frame", dependencies=auth)
    async def frame(request: Request) -> dict[str, Any]:
        data = await request.body()
        if not data or len(data) > MAX_FRAME_BYTES:
            raise HTTPException(status_code=400, detail="send one JPEG or PNG frame under 4 MB as the request body")
        try:
            rgb = decode_image(data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode frame: {type(exc).__name__}") from exc
        try:
            x, y, found = monitor.add_frame(rgb, source="browser")
        except Exception as exc:  # mediapipe missing or failing
            raise HTTPException(status_code=503, detail=f"gaze tracking unavailable: {type(exc).__name__}: {exc}"[:300]) from exc
        return {"face_found": found, "tracked": x == x}

    @app.post("/v1/integrity/gaze/screen", dependencies=auth)
    def screen(req: ScreenRequest) -> dict[str, Any]:
        monitor.start_screen(Region(req.left, req.top, req.width, req.height))
        return monitor.status()

    @app.post("/v1/integrity/gaze/stop", dependencies=auth)
    def stop() -> dict[str, Any]:
        monitor.stop()
        return monitor.status()

    @app.post("/v1/integrity/gaze/reset", dependencies=auth)
    def reset() -> dict[str, Any]:
        monitor.reset()
        return monitor.status()

    @app.post("/v1/integrity/reading", dependencies=auth)
    def reading_check(req: ReadingRequest | None = None) -> dict[str, Any]:
        req = req or ReadingRequest()
        now = clock()
        end = req.end if req.end is not None else now
        start = req.start if req.start is not None else end - (req.seconds or 60.0)
        if not 0 < end - start <= 600:
            raise HTTPException(status_code=400, detail="the span must be between 0 and 600 seconds")
        result = reading_model.get().assess(monitor.series(start, end))
        if result is None:
            return {"reading_probability": None, "level": "not_enough_gaze", "seconds": end - start,
                    "detail": "need at least 10 s with the face tracked in most frames"}
        return {**result.as_dict(), "seconds": end - start}

    return monitor
