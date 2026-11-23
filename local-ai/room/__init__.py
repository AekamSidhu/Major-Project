"""OpenMeet Room: a self-hosted 1:1 interview room in the browser.

The interviewer runs `room_server.py` on their own computer and opens
https://localhost:8443. The candidate opens an invite link in any browser
(laptop or phone on the same Wi-Fi, or through a Cloudflare tunnel when the
interviewer starts the room with --tunnel; the tunnel is off by default). Video and audio go
directly between the two browsers (WebRTC); this server only introduces
them (signaling) and serves the pages.

The interviewer's browser then sends, to this computer only:
- each speaker's speech, cut at pauses, for local transcription (Whisper,
  with word timings);
- frames of the candidate's video, for the reading-gaze check;
- each finished answer, for the integrity signals (room/checks.py: AI-written
  text, similarity to an LLM's answer to the same question, delivery timing
  against the candidate's first answers, reading gaze, fused into one flag)
  and for follow-up questions (room/followups.py: Ollama on this computer by
  default; Google Gemini in online mode, which the candidate's join page
  then states; simple rules when no model answers).

Who may call what:
- interviewer endpoints (/host, /api/*, /v1/integrity/*) answer only to
  this computer (loopback, Host naming this computer, no tunnel headers), so
  nobody on the network or through the tunnel can use them;
- the candidate gets the join page and the signaling socket, nothing else,
  and needs the room's guest key from the invite link. No score or flag
  ever reaches the candidate's side.

Nothing is recorded: audio segments are transcribed and dropped, frames
become gaze samples and are dropped. Transcripts and scores live in memory
until the server stops, and the interviewer can download them as a report.
The one exception is study mode (room_server.py --study DIR, study.py):
after the candidate ticks a separate study consent, each answer's
transcript, word timings, gaze series and scores (never audio or video) are
saved under an anonymous id.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from integrity.signals import DeliveryBaseline

from . import checks
from .report import markdown

LOG = logging.getLogger("openmeet.room")
HERE = Path(__file__).resolve().parent
MAX_SEGMENT_BYTES = 8 * 1024 * 1024
SPEAKERS = ("interviewer", "candidate")
MIN_CONFIDENCE = 0.45
OFFLINE_TERMS = "Nothing goes to an online service and nothing is recorded;"
STUDY_TERMS_OFFLINE = "Nothing goes to an online service, and nothing is recorded unless you join the study below;"
GROQ_TERMS = "Your speech (audio) is sent to Groq, an online service, to be transcribed."


def consent_terms(*, groq: bool, online: bool, study: bool) -> str:
    """The sentence on the join page that says what leaves the interviewer's computer.

    `groq`: the candidate's audio goes to Groq for transcription; `online`: the text of the
    questions and answers goes to Google Gemini for follow-ups. Without either, the offline
    sentence; without groq, the sentences used before cloud transcription existed."""
    if not groq and not online:
        return STUDY_TERMS_OFFLINE if study else OFFLINE_TERMS
    record = "Nothing is recorded unless you join the study below." if study else "Nothing is recorded."
    sentences = [GROQ_TERMS] if groq else []
    sentences.append(record)
    if online:
        sentences.append("To suggest follow-up questions and to write the comparison answers, the "
                         "interviewer's questions and the text of your answers (not audio or video) are sent to "
                         "Google Gemini, with emails, phone numbers and your name removed;")
    else:
        sentences[-1] = sentences[-1].rstrip(".") + ";"
    return " ".join(sentences)


STUDY_SLOT = "<!--study-consent-->"
# Whisper's stock lines for non-speech audio: dropped when the whole utterance is one of them.
PHANTOMS = {"thanks for watching", "thank you for watching", "well see you in the next one",
            "see you in the next one", "im not clear", "subtitles by the amaraorg community",
            "for reference", "now cut to the rest", "hooray alright", "hooray", "alright"}
# Short filler words Whisper invents on silence: dropped only when the audio is quiet.
QUIET_PHRASES = {"thank you", "thanks", "you", "bye", "so", "okay", "oh", "hello", "uh", "um", "uh um"}


def is_quiet(samples: Any) -> bool:
    """Near-silent audio: RMS below 0.008, the same threshold as the desktop engine."""
    audio = np.asarray(samples, dtype=np.float32)
    return audio.size == 0 or float(np.sqrt(np.mean(np.square(audio)))) < 0.008


def is_phantom(text: str, quiet: bool = False) -> bool:
    """True for Whisper's stock phrases; short filler words only when `quiet`."""
    joined = " ".join("".join(c for c in text.lower() if c.isalnum() or c == " ").split())
    return joined in PHANTOMS or (quiet and joined in QUIET_PHRASES)


def is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or (host or "").startswith("127.")


# Headers that a reverse proxy or tunnel adds. cloudflared runs on this computer and
# forwards public traffic to localhost, so a tunnelled request arrives from 127.0.0.1
# too; these headers (and the public Host name) are what tell it apart.
PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "forwarded")


def _host_name(value: str) -> str:
    """The name part of a Host header: `localhost:8443` -> `localhost`, `[::1]:8443` -> `::1`."""
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end > 0 else ""
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


def interviewer_names() -> frozenset[str]:
    """Host names that mean this computer: localhost and this computer's network addresses."""
    return frozenset({"localhost", "127.0.0.1", "::1", *lan_addresses()})


def is_interviewer(client: str | None, headers: Mapping[str, str], names: frozenset[str]) -> bool:
    """True only for a request made on this computer: loopback client, no proxy headers,
    and a Host header naming this computer (never the tunnel's hostname)."""
    if not is_loopback(client):
        return False
    if any(name in headers for name in PROXY_HEADERS):
        return False
    return _host_name(headers.get("host", "")) in names


def interviewer_only(names: frozenset[str]) -> Callable[[Request], None]:
    """A FastAPI dependency: refuse anything that is not the interviewer's own computer."""
    def check(request: Request) -> None:
        if not is_interviewer(request.client.host if request.client else None, request.headers, names):
            raise HTTPException(status_code=403, detail="only the interviewer's computer can use this")

    return check


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
    delivery: dict[str, DeliveryBaseline] = field(default_factory=dict)  # per candidate
    study_consent: bool = False  # the candidate ticked the study consent (study mode only)
    study_session: str = ""  # anonymous id of the saved session
    participant: str = ""  # the study's participant id, set by the interviewer

    def speaker_baseline(self) -> DeliveryBaseline:
        return self.delivery.setdefault(self.candidate_name, DeliveryBaseline())


class FollowupRequest(BaseModel):
    answer: int | None = None  # the answer's index; the latest when omitted


class AnswerRequest(BaseModel):
    start: float
    end: float
    text: str = Field(default="", max_length=20000)
    question: str = Field(default="", max_length=5000)
    condition: str | None = Field(default=None, max_length=40)  # study mode: the condition label


class ReferenceRequest(BaseModel):
    question: str = Field(default="", max_length=5000)


class StudyRequest(BaseModel):
    participant: str | None = Field(default=None, max_length=40)


class LabelRequest(BaseModel):
    condition: str = Field(..., max_length=40)


def add_room_routes(app: FastAPI, transcriber: Any, monitor: Any, echo_factory: Any, reading_factory: Any,
                    port: int, scheme: str, whisper_model: str, followup_writer: Any = None,
                    similarity: Any = None, study: Any = None,
                    tunnel_url: str | None = None, cloud_transcription: bool = False) -> dict[str, Room]:
    """`similarity`: a room.similarity.SimilarityChecker (None: echo similarity unavailable);
    `study`: a room.study.StudyRecorder in study mode; `cloud_transcription`: the candidate's
    audio goes to a cloud service (Groq), which the join page states."""
    from .followups import FollowupWriter

    rooms: dict[str, Room] = {}
    infer_lock = threading.Lock()
    followup_lock = threading.Lock()
    writer = followup_writer or FollowupWriter()
    models: dict[str, Any] = {}
    similarity_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="similarity")
    similarity_jobs: dict[tuple[str, int], Future] = {}
    params = inspect.signature(transcriber.transcribe).parameters
    timed = "word_timestamps" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())

    names = interviewer_names()
    host_only = interviewer_only(names)

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
        if is_interviewer(request.client.host if request.client else None, request.headers, names):
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
        html = page("guest.html")
        # The candidate agrees to what actually happens: audio to Groq, text to Gemini.
        terms = consent_terms(groq=cloud_transcription, online=writer.online, study=study is not None)
        html = html.replace(OFFLINE_TERMS, terms)
        if study is not None:
            html = html.replace(STUDY_SLOT, study.consent_html())
        return html

    @app.post("/api/rooms", dependencies=[Depends(host_only)])
    def create_room() -> dict[str, Any]:
        room = Room(id=secrets.token_hex(3), host_key=secrets.token_urlsafe(16), guest_key=secrets.token_urlsafe(12))
        rooms[room.id] = room
        path = f"/join/{room.id}?k={room.guest_key}"
        lan_invites = [f"{scheme}://{ip}:{port}{path}" for ip in lan_addresses()] + [f"{scheme}://localhost:{port}{path}"]
        tunnel_invite = f"{tunnel_url}{path}" if tunnel_url else None
        invites = ([tunnel_invite] if tunnel_invite else []) + lan_invites
        monitor.reset()
        return {"room": room.id, "host_key": room.host_key, "invites": invites,
                "tunnel_url": tunnel_invite, "lan_invites": lan_invites}

    @app.get("/api/status", dependencies=[Depends(host_only)])
    def status() -> dict[str, Any]:
        return {"transcription_model": whisper_model, "rooms": len(rooms), "followup_mode": writer.mode,
                "followup_model": writer.model_name, "followup_model_ok": writer.model_ok,
                "followup_error": writer.last_error,
                "reference_model": similarity.writer.model_name if similarity and similarity.writer.available else None,
                "tunnel_url": tunnel_url,
                "study": {"enabled": True, "conditions": list(study.conditions)} if study else {"enabled": False}}

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

        def run() -> tuple[str, list[dict[str, Any]]]:
            with infer_lock:
                if timed:  # word timings for the delivery signal
                    result = transcriber.transcribe(audio, whisper_model, "en", None, word_timestamps=True)
                else:
                    result = transcriber.transcribe(audio, whisper_model, "en", None)
            text = " ".join(result.text.split())
            # The browser cuts speech by loudness, so noise and tones get through;
            # Whisper then invents stock phrases. Drop those and unsure results.
            if (result.confidence is not None and result.confidence < MIN_CONFIDENCE) or is_phantom(text, is_quiet(audio)):
                return "", []
            # Words on the meeting clock: the segment's audio starts at `start`.
            words = [{"word": w["word"], "start": round(start + w["start"], 3), "end": round(start + w["end"], 3)}
                     for w in getattr(result, "words", None) or []]
            return text, words

        try:
            text, words = await asyncio.get_running_loop().run_in_executor(None, run)
        except Exception as exc:  # faster-whisper missing or failing
            raise HTTPException(status_code=503, detail=f"transcription unavailable: {type(exc).__name__}: {exc}"[:300]) from exc
        segment = {"speaker": speaker, "start": start, "end": end, "text": text}
        if text:
            room.transcript.append({**segment, "words": words})
        return segment

    def save_study(room: Room, answer: dict[str, Any]) -> None:
        # Both: the candidate consented when the answer was given, and still does now.
        # An answer from before the consent is never saved, even if relabelled later.
        if study is not None and room.study_consent and answer.get("study_consent"):
            try:
                study.save_answer(room, answer, monitor)
            except Exception as error:  # a full disk must not break the interview
                LOG.warning("could not save the study answer (%s)", type(error).__name__)

    @app.post("/api/rooms/{room_id}/references", dependencies=[Depends(host_only)])
    def references(room_id: str, req: ReferenceRequest) -> dict[str, Any]:
        """Start writing reference answers to a question (the host calls this when the candidate starts answering)."""
        room = room_or_404(room_id)
        started = similarity is not None and similarity.writer.available and bool(req.question.strip())
        if started:
            similarity.references(req.question, room.candidate_name)
        return {"started": started}

    def condition_label(value: str | None) -> str | None:
        try:
            return study.label(value) if study is not None else None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/rooms/{room_id}/answers", dependencies=[Depends(host_only)])
    def check_answer(room_id: str, req: AnswerRequest) -> dict[str, Any]:
        room = room_or_404(room_id)
        condition = condition_label(req.condition)
        echo = model("echo").assess(req.text) if req.text.strip() else None
        reading = None
        if 0 < req.end - req.start <= 600:
            result = model("reading").assess(monitor.series(req.start, req.end))
            reading = result.as_dict() if result else {"reading_probability": None, "level": "not_enough_gaze"}
        # Delivery: this answer's timing against the candidate's first answers.
        utterances = checks.answer_utterances(room.transcript, req.start, req.end)
        begins = min((u["start"] for u in utterances), default=req.start)
        q_end = checks.question_end(room.transcript, begins, room.answers[-1]["end"] if room.answers else None)
        speaker = room.speaker_baseline()
        delivery = speaker.add(q_end, utterances)
        answer = {"index": len(room.answers) + 1, "start": req.start, "end": req.end, "question": req.question,
                  "text": req.text, "echo": echo, "reading": reading, "delivery": delivery, "question_end": q_end,
                  "similarity": None, "flag": None, "followups": None, "condition": condition,
                  "study_consent": room.study_consent}
        if similarity is not None:
            answer["similarity"] = {"level": "pending"}
            if similarity.writer.available and req.question.strip():
                similarity.references(req.question, room.candidate_name)  # usually started already
        answer["flag"] = checks.flag(answer, speaker.baseline)
        room.answers.append(answer)
        save_study(room, answer)
        if similarity is not None:
            def run() -> None:
                answer["similarity"] = similarity.check(req.question, req.text, room.candidate_name)
                answer["flag"] = checks.flag(answer, speaker.baseline)
                save_study(room, answer)

            similarity_jobs[(room.id, answer["index"])] = similarity_pool.submit(run)
        return answer

    def answer_or_404(room: Room, index: int) -> dict[str, Any]:
        found = next((a for a in room.answers if a["index"] == index), None)
        if found is None:
            raise HTTPException(status_code=404, detail="no such answer")
        return found

    @app.post("/api/rooms/{room_id}/answers/{index}/similarity", dependencies=[Depends(host_only)])
    async def answer_similarity(room_id: str, index: int) -> dict[str, Any]:
        """Wait for the answer's echo similarity (written in the background) and return the updated answer."""
        room = room_or_404(room_id)
        answer = answer_or_404(room, index)
        job = similarity_jobs.get((room.id, index))
        if job is not None:
            await asyncio.get_running_loop().run_in_executor(None, job.result)
        return answer

    @app.post("/api/rooms/{room_id}/answers/{index}/label", dependencies=[Depends(host_only)])
    def label_answer(room_id: str, index: int, req: LabelRequest) -> dict[str, Any]:
        """Study mode: the interviewer sets or corrects an answer's condition label."""
        room = room_or_404(room_id)
        if study is None:
            raise HTTPException(status_code=409, detail="the room is not in study mode")
        answer = answer_or_404(room, index)
        answer["condition"] = condition_label(req.condition)
        save_study(room, answer)
        return {"index": index, "condition": answer["condition"], "saved": room.study_consent and bool(answer["study_consent"])}

    @app.get("/api/rooms/{room_id}/study", dependencies=[Depends(host_only)])
    def study_state(room_id: str) -> dict[str, Any]:
        room = room_or_404(room_id)
        return {"enabled": study is not None, "consent": room.study_consent, "session": room.study_session or None,
                "participant": room.participant or None}

    @app.post("/api/rooms/{room_id}/study", dependencies=[Depends(host_only)])
    def study_participant(room_id: str, req: StudyRequest) -> dict[str, Any]:
        room = room_or_404(room_id)
        if study is None:
            raise HTTPException(status_code=409, detail="the room is not in study mode")
        try:
            room.participant = study.participant_id(req.participant or "")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if room.study_consent:
            study.save_session(room)
        return study_state(room_id)

    @app.post("/api/rooms/{room_id}/followups", dependencies=[Depends(host_only)])
    async def followups(room_id: str, req: FollowupRequest) -> dict[str, Any]:
        room = room_or_404(room_id)
        if not room.answers:
            raise HTTPException(status_code=409, detail="no answers yet")
        answer = room.answers[-1] if req.answer is None else next((a for a in room.answers if a["index"] == req.answer), None)
        if answer is None:
            raise HTTPException(status_code=404, detail="no such answer")
        previous = [i["question"] for a in room.answers for i in (a.get("followups") or {}).get("items", [])]

        def run() -> dict[str, Any]:
            with followup_lock:
                return writer.write(answer["question"], answer["text"], list(room.transcript), previous,
                                    candidate_name=room.candidate_name)

        result = await asyncio.get_running_loop().run_in_executor(None, run)
        answer["followups"] = result
        return {"answer": answer["index"], **result}

    @app.get("/api/rooms/{room_id}/report", dependencies=[Depends(host_only)])
    def report(room_id: str) -> dict[str, Any]:
        room = room_or_404(room_id)
        return {"room": room.id, "created": room.created, "candidate": room.candidate_name,
                "transcript": [{k: v for k, v in s.items() if k != "words"} for s in room.transcript],
                "answers": room.answers}

    @app.get("/api/rooms/{room_id}/report.md", dependencies=[Depends(host_only)])
    def report_markdown(room_id: str) -> PlainTextResponse:
        return PlainTextResponse(markdown(report(room_id)), media_type="text/markdown; charset=utf-8")

    @app.websocket("/ws/{room_id}")
    async def signaling(ws: WebSocket, room_id: str, role: str, k: str, name: str = "", study_consent: str = "") -> None:
        room = rooms.get(room_id)
        client = ws.client.host if ws.client else None
        LOG.info("ws connect: room=%s role=%s client=%s room_exists=%s", room_id, role, client, room is not None)
        allowed = room is not None and (
            (role == "host" and secrets.compare_digest(k, room.host_key)
             and is_interviewer(client, ws.headers, names))
            or (role == "guest" and secrets.compare_digest(k, room.guest_key))
        )
        if not allowed:
            LOG.warning("ws rejected: room=%s role=%s client=%s", room_id, role, client)
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
            # Study mode saves answers only after the candidate's own, separate consent;
            # joining again without it stops the saving.
            room.study_consent = study is not None and study_consent == "1"
            if room.study_consent:
                study.save_session(room)
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
