"""The interviewer's Markdown report of a room: answers, signals with their evidence, follow-ups, transcript."""

from __future__ import annotations

import datetime as dt
from typing import Any

from .checks import FLAG_LABELS, SIGNAL_NAMES

LEVELS = {"low": "low", "possible": "possible", "high": "high", "too_short": "too short", "baseline": "baseline",
          "no_gaze": "no gaze", "unavailable": "unavailable"}


def _time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def signal_line(s: dict[str, Any]) -> str:
    tag = "" if s["calibrated"] or s["level"] not in ("low", "possible", "high") else ", uncalibrated"
    return f"- {SIGNAL_NAMES[s['name']]}: {LEVELS.get(s['level'], s['level'])}{tag}. {s['evidence'][:1].upper()}{s['evidence'][1:]}."


def markdown(room: dict[str, Any]) -> str:
    created = dt.datetime.fromtimestamp(room["created"]).strftime("%Y-%m-%d %H:%M")
    lines = ["# Interview report", "",
             f"Room {room['room']}, {created}" + (f", candidate {room['candidate']}" if room.get("candidate") else "") + ".", "",
             "Each answer gets one signal for the interviewer, never a verdict: at most *worth a follow-up*, with the "
             "reasons. A signal can be wrong; ask, don't conclude. Signals marked *uncalibrated* have no labelled "
             "interview data behind their thresholds yet, and none of the signals is calibrated on real interviews.", "",
             "## Answers", ""]
    for a in room["answers"]:
        lines += [f"### Answer {a['index']} ({_time(a['start'])}, {round(a['end'] - a['start'])} s)", ""]
        if a.get("question"):
            lines += [f"**Question:** {a['question']}", ""]
        lines += [f"**Answer:** {a.get('text') or '(no transcript)'}", ""]
        f = a.get("flag")
        if f:
            why = f" (signals: {', '.join(SIGNAL_NAMES[n] for n in f['elevated'])})" if f["elevated"] else ""
            lines += [f"**Signal:** {FLAG_LABELS[f['level']]}{why}", ""]
            lines += [signal_line(s) for s in f["signals"]] + [""]
        if a.get("condition"):
            lines += [f"Study condition: {a['condition']}", ""]
        fu = a.get("followups")
        if fu and fu.get("items"):
            lines += [f"**Follow-ups** ({'by ' + str(fu['model']) if fu.get('source') == 'ai' else 'by rules'}):", ""]
            lines += [f"- {i['question']}" + (f" _{i['reason']}_" if i.get("reason") else "") for i in fu["items"]] + [""]
    lines += ["## Transcript", ""]
    lines += [f"- {_time(s['start'])} **{s['speaker']}:** {s['text']}" for s in room["transcript"]]
    return "\n".join(lines) + "\n"
