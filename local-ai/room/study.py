"""Study mode: save labelled answers for calibrating the integrity signals on our own data.

    python room_server.py --study D:/openmeet-study      # a folder outside the repository

Only when the room runs with --study, and only after the candidate ticks the
separate study consent on the join page (the server checks it; the host page
cannot turn it on), each answer is saved as

    <folder>/session-<id>/session.json      anonymous id, participant id, spec thresholds, models
    <folder>/session-<id>/answer-NNN.json   condition label, question, transcript with word
                                            timings, gaze series (10 per second), all signal scores

Times are seconds from the start of the room. Never saved: audio, video
frames, the candidate's name. The participant id (P01...) links a session to
the label sheet and the consent form (training/data-collection), which are
kept apart from this folder. To withdraw a participant, delete their session
folder.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from integrity import signals

from . import checks

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("honest", "read_script", "read_ai", "paraphrase_ai")
PARTICIPANT = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
GAZE_FPS = 10.0

CONSENT_HTML = """<label class="consent study"><input type="checkbox" id="study"> <span><b>Optional study.</b> I also agree
  to take part in the OpenMeet study. For each answer, this computer saves the transcript with word timings, a series of
  eye positions (no video, no audio), the check scores and the interviewer's label for the answer, under an anonymous id,
  as described in the consent form. I can ask for it to be deleted at any time. Joining works without this.</span></label>"""


def finite(value: Any) -> Any:
    """JSON-safe copy: NaN and infinities become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def inside(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


class StudyRecorder:
    conditions = CONDITIONS

    def __init__(self, folder: Path, info: dict[str, Any] | None = None, repo: Path = REPO) -> None:
        folder = Path(folder).resolve()
        if inside(folder, repo):
            raise ValueError(f"the study folder must be outside the repository ({repo}): participant data is never committed")
        self.folder = folder
        self.info = info or {}

    def consent_html(self) -> str:
        return CONSENT_HTML

    @staticmethod
    def label(condition: str | None) -> str | None:
        """A condition label, or None for unlabelled."""
        condition = (condition or "").strip()
        if not condition:
            return None
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
        return condition

    @staticmethod
    def participant_id(value: str) -> str:
        value = value.strip()
        if value and not PARTICIPANT.match(value):
            raise ValueError("participant id: letters, digits, - and _ only, at most 32 (an id like P07, never a name)")
        return value

    def _dir(self, room: Any) -> Path:
        if not room.study_session:
            room.study_session = secrets.token_hex(4)
        folder = self.folder / f"session-{room.study_session}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def _write(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(finite(data), indent=1, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def save_session(self, room: Any) -> Path:
        folder = self._dir(room)
        try:
            spec = signals.thresholds()
        except Exception:  # the spec is missing: still save what we have
            spec = None
        self._write(folder / "session.json", {
            "session": room.study_session, "participant": room.participant or None,
            "date": time.strftime("%Y-%m-%d", time.localtime(room.created)),
            "consent": {"study": bool(room.study_consent)}, "spec_version": signals.SPEC_VERSION, "thresholds": spec,
            "conditions": list(CONDITIONS), "models": self.info,
        })
        return folder

    def save_answer(self, room: Any, answer: dict[str, Any], monitor: Any) -> Path:
        if not room.study_consent:
            raise PermissionError("no study consent")
        folder = self._dir(room)
        if not (folder / "session.json").exists():
            self.save_session(room)
        t0 = room.created

        def rel(t: float | None) -> float | None:
            return None if t is None else round(t - t0, 3)

        def num(v: float) -> float | None:
            return None if not math.isfinite(v) else round(float(v), 5)

        start, end = answer["start"], answer["end"]
        utterances = checks.answer_utterances(room.transcript, start, end)
        gaze = monitor.series(start, end, GAZE_FPS) if 0 < end - start <= 600 else []
        record = {
            "session": room.study_session, "participant": room.participant or None, "answer": answer["index"],
            "condition": answer.get("condition"), "start_s": rel(start), "end_s": rel(end),
            "question": answer.get("question", ""), "question_end_s": rel(answer.get("question_end")),
            "text": answer.get("text", ""),
            "utterances": [{"start_s": rel(u["start"]), "end_s": rel(u["end"]), "text": u["text"],
                            "words": [{"word": w["word"], "start_s": rel(w["start"]), "end_s": rel(w["end"])}
                                      for w in u.get("words") or []]} for u in utterances],
            "gaze": {"fps": GAZE_FPS, "start_s": rel(start), "x": [num(g[0]) for g in gaze], "y": [num(g[1]) for g in gaze]},
            "signals": {"ai_text": answer.get("echo"), "echo_similarity": answer.get("similarity"),
                        "delivery": answer.get("delivery"), "reading": answer.get("reading")},
            "flag": answer.get("flag"),
        }
        path = folder / f"answer-{answer['index']:03d}.json"
        self._write(path, record)
        return path
