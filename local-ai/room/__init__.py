"""OpenMeet Room: a self-hosted 1:1 interview room in the browser.

The interviewer runs `room_server.py` on their own computer and opens
https://localhost:8443. The candidate opens an invite link in any browser
(laptop or phone, same Wi-Fi, or through a tunnel). Video and audio go
directly between the two browsers (WebRTC); this server only introduces
them (signaling) and serves the pages.

The interviewer's browser then sends, to this computer only:
- each speaker's speech, cut at pauses, for local transcription (Whisper);
- frames of the candidate's video, for the reading-gaze check;
- each finished answer's transcript, for the AI-written-answer check.

Who may call what:
- interviewer endpoints (/host, /api/*, /v1/integrity/*) answer only to
  this computer (loopback), so nobody on the network can use them;
- the candidate gets the join page and the signaling socket, nothing else,
  and needs the room's guest key from the invite link.

Nothing is recorded: audio segments are transcribed and dropped, frames
become gaze samples and are dropped. Transcripts and scores live in memory
until the server stops, and the interviewer can download them as a report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

LOG = logging.getLogger("openmeet.room")
HERE = Path(__file__).resolve().parent
MAX_SEGMENT_BYTES = 8 * 1024 * 1024
SPEAKERS = ("interviewer", "candidate")
MIN_CONFIDENCE = 0.35
PHANTOMS = {"thank you", "thanks for watching", "thank you for watching", "well see you in the next one",
            "see you in the next one", "bye", "you", "so", "okay", "subtitles by the amaraorg community"}


def is_phantom(text: str) -> bool:
    """Whisper's stock phrases for non-speech audio."""
    words = "".join(c for c in text.lower() if c.isalnum() or c == " ").split()
    return " ".join(words) in PHANTOMS


def is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or (host or "").startswith("127.")


def lan_addresses() -> list[str]:
    """This computer's addresses other devices on the network can reach."""
    found: list[str] = []
    try:  # the address the default route would use (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass

    # Home and office Wi-Fi is usually 192.168.x or 172.16-31.x; 10.x is often a
    # VPN adapter that a phone on the same Wi-Fi cannot reach.
    def rank(ip: str) -> int:
        parts = [int(p) for p in ip.split(".")]
        if parts[0] == 192 and parts[1] == 168:
            return 0
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 1
        return 2 if parts[0] == 10 else 3

    return sorted(found, key=rank)


@dataclass
class Room:
    id: str
    host_key: str
    guest_key: str
    created: float = field(default_factory=time.time)
    sockets: dict[str, WebSocket] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)
    candidate_name: str = ""


class AnswerRequest(BaseModel):
    start: float
    end: float
    text: str = Field(default="", max_length=20000)
    question: str = Field(default="", max_length=5000)


def add_room_routes(app: FastAPI, transcriber: Any, monitor: Any, echo_factory: Any, reading_factory: Any,
                    port: int, scheme: str, whisper_model: str) -> dict[str, Room]:
    rooms: dict[str, Room] = {}
    infer_lock = threading.Lock()
    models: dict[str, Any] = {}

    def host_only(request: Request) -> None:
        if not is_loopback(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="only the interviewer's computer can use this")

    def model(name: str) -> Any:
        if name not in models:
            try:
                models[name] = echo_factory() if name == "echo" else reading_factory()
            except Exception as error:
                raise HTTPException(status_code=503, detail=f"{name} model unavailable: {error}") from error
        return models[name]

    def room_or_404(room_id: str) -> Room:
        room = rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="no such room")
        return room

    def page(name: str) -> str:
        return (HERE / name).read_text(encoding="utf-8")

    @app.get("/", include_in_schema=False)
    def root(request: Request):  # type: ignore[no-untyped-def]
        if is_loopback(request.client.host if request.client else None):
            return RedirectResponse("/host")
        return HTMLResponse("<p style='font-family:sans-serif'>Ask the interviewer for an invite link.</p>", status_code=404)

    @app.get("/host", response_class=HTMLResponse, dependencies=[Depends(host_only)])
    def host_page() -> str:
        return page("host.html")

    @app.get("/room.js", include_in_schema=False)
    def room_js() -> Response:
        return Response(page("room.js"), media_type="text/javascript")

    @app.get("/join/{room_id}", response_class=HTMLResponse)
    def join_page(room_id: str) -> str:
        return page("guest.html")

    @app.post("/api/rooms", dependencies=[Depends(host_only)])
    def create_room() -> dict[str, Any]:
        room = Room(id=secrets.token_hex(3), host_key=secrets.token_urlsafe(16), guest_key=secrets.token_urlsafe(12))
        rooms[room.id] = room
        path = f"/join/{room.id}?k={room.guest_key}"
        invites = [f"{scheme}://{ip}:{port}{path}" for ip in lan_addresses()] + [f"{scheme}://localhost:{port}{path}"]
        monitor.reset()
        return {"room": room.id, "host_key": room.host_key, "invites": invites}

    @app.get("/api/status", dependencies=[Depends(host_only)])
    def status() -> dict[str, Any]:
        return {"transcription_model": whisper_model, "rooms": len(rooms)}

    @app.post("/api/rooms/{room_id}/transcribe", dependencies=[Depends(host_only)])
    async def transcribe(room_id: str, request: Request, speaker: str, start: float, end: float) -> dict[str, Any]:
        room = room_or_404(room_id)
        if speaker not in SPEAKERS:
            raise HTTPException(status_code=400, detail="speaker must be interviewer or candidate")
        data = await request.body()
        if not data or len(data) > MAX_SEGMENT_BYTES:
            raise HTTPException(status_code=400, detail="send one WAV segment under 8 MB")
        from server import decode_audio  # the sidecar's WAV decoder

        try:
            audio = decode_audio(data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {type(exc).__name__}") from exc

        def run() -> str:
            with infer_lock:
                result = transcriber.transcribe(audio, whisper_model, "en", None)
            text = " ".join(result.text.split())
            # The browser cuts speech by loudness, so noise and tones get through;
            # Whisper then invents stock phrases. Drop those and unsure results.
            if (result.confidence is not None and result.confidence < MIN_CONFIDENCE) or is_phantom(text):
                return ""
            return text

        try:
            text = await asyncio.get_running_loop().run_in_executor(None, run)
        except Exception as exc:  # faster-whisper missing or failing
            raise HTTPException(status_code=503, detail=f"transcription unavailable: {type(exc).__name__}: {exc}"[:300]) from exc
        segment = {"speaker": speaker, "start": start, "end": end, "text": text}
        if text:
            room.transcript.append(segment)
        return segment

    @app.post("/api/rooms/{room_id}/answers", dependencies=[Depends(host_only)])
    def check_answer(room_id: str, req: AnswerRequest) -> dict[str, Any]:
        room = room_or_404(room_id)
        echo = model("echo").assess(req.text) if req.text.strip() else None
        reading = None
        if 0 < req.end - req.start <= 600:
            result = model("reading").assess(monitor.series(req.start, req.end))
            reading = result.as_dict() if result else {"reading_probability": None, "level": "not_enough_gaze"}
        answer = {"index": len(room.answers) + 1, "start": req.start, "end": req.end, "question": req.question,
                  "text": req.text, "echo": echo, "reading": reading}
        room.answers.append(answer)
        return answer

    @app.get("/api/rooms/{room_id}/report", dependencies=[Depends(host_only)])
    def report(room_id: str) -> dict[str, Any]:
        room = room_or_404(room_id)
        return {"room": room.id, "created": room.created, "candidate": room.candidate_name,
                "transcript": room.transcript, "answers": room.answers}

    @app.websocket("/ws/{room_id}")
    async def signaling(ws: WebSocket, room_id: str, role: str, k: str, name: str = "") -> None:
        room = rooms.get(room_id)
        client = ws.client.host if ws.client else None
        allowed = room is not None and (
            (role == "host" and secrets.compare_digest(k, room.host_key) and is_loopback(client))
            or (role == "guest" and secrets.compare_digest(k, room.guest_key))
        )
        if not allowed:
            await ws.close(code=4403)
            return
        assert room is not None
        await ws.accept()
        old = room.sockets.get(role)
        if old is not None:  # a reload replaces the previous connection
            try:
                await old.close(code=4000)
            except Exception:
                pass
        room.sockets[role] = ws
        if role == "guest":
            room.candidate_name = name[:60]
        other = "guest" if role == "host" else "host"
        await ws.send_text(json.dumps({"type": "hello", "role": role, "peer_present": other in room.sockets}))
        if other in room.sockets:
            await _send(room.sockets[other], {"type": "peer-joined", "role": role, "name": room.candidate_name})
        try:
            while True:
                message = await ws.receive_text()
                if len(message) > 64_000:
                    continue
                peer = room.sockets.get(other)
                if peer is not None:
                    await _send(peer, json.loads(message))
        except WebSocketDisconnect:
            pass
        finally:
            if room.sockets.get(role) is ws:
                del room.sockets[role]
                peer = room.sockets.get(other)
                if peer is not None:
                    await _send(peer, {"type": "peer-left", "role": role})

    return rooms


async def _send(ws: WebSocket, message: dict[str, Any]) -> None:
    try:
        await ws.send_text(json.dumps(message))
    except Exception:
        pass
