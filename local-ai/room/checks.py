"""Per-answer integrity signals in the room, fused into one flag with its evidence.

Four signals (spec/integrity/README.md):

- ai_text: does the transcript read like AI-written text? (integrity/models.py)
- echo_similarity: how close is it to the room model's own answer? (similarity.py)
- delivery: does its timing look read, against this candidate's first answers?
- reading: do the eyes move like reading lines of text? (integrity/models.py)

The flag is a signal for the interviewer, never a verdict: at most "worth a
follow-up", always with the reasons. Signals not calibrated on labelled data
are marked so and never make the flag "high" on their own (spec fusion).
"""

from __future__ import annotations

from typing import Any

from integrity import signals

SPEECH_SLACK_S = 0.5  # the browser cuts speech with a little audio before it
SIGNAL_NAMES = {"ai_text": "AI-written text", "echo_similarity": "Similar to an AI answer",
                "delivery": "Delivery", "reading": "Reading"}
FLAG_LABELS = {"low": "Nothing stands out", "possible": "Worth a follow-up",
               "high": "Worth a follow-up: several signals agree", "unavailable": "Not enough to go on"}


def utterance(segment: dict[str, Any]) -> dict[str, Any]:
    """A transcript segment as a spec utterance; with words, its span is the words' span."""
    words = segment.get("words") or []
    if words:
        return {"start": words[0]["start"], "end": words[-1]["end"], "text": segment["text"], "words": words}
    return {"start": segment["start"], "end": segment["end"], "text": segment["text"]}


def answer_utterances(transcript: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """The candidate's transcribed speech between `start` and `end` (meeting clock)."""
    return [utterance(s) for s in transcript
            if s["speaker"] == "candidate" and start - SPEECH_SLACK_S <= s["start"] <= end + SPEECH_SLACK_S]


def question_end(transcript: list[dict[str, Any]], answer_start: float, after: float | None) -> float | None:
    """When the interviewer last stopped talking before the answer began (None if they didn't since `after`)."""
    ends = [utterance(s)["end"] for s in transcript if s["speaker"] == "interviewer"]
    ends = [e for e in ends if e <= answer_start + SPEECH_SLACK_S and (after is None or e > after)]
    return max(ends) if ends else None


def _pct(p: float | None) -> str:
    return "–" if p is None else f"{round(p * 100)}%"


def delivery_evidence(result: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    score, f = result["score"], result["features"]
    if score["level"] == "too_short":
        return "too short to judge the timing"
    if score["level"] == "baseline":
        return "one of the first answers, which set this candidate's usual pace"
    if score["level"] == "unavailable":
        return "no timing to compare"
    parts = []
    usual = lambda name: baseline[name]["median"] if baseline and baseline.get(name) else None  # noqa: E731
    if f.get("latency_s") is not None:
        u = usual("latency_s")
        parts.append(f"started {f['latency_s']:.1f} s after the question" + (f" (usually {u:.1f} s)" if u is not None else ""))
    if f.get("rate_wpm") is not None:
        u = usual("rate_wpm")
        parts.append(f"{f['rate_wpm']:.0f} words a minute" + (f" (usually {u:.0f})" if u is not None else ""))
    if f.get("pause_per_min") is not None:
        u = usual("pause_per_min")
        parts.append(f"{f['pause_per_min']:.1f} pauses a minute" + (f" (usually {u:.1f})" if u is not None else ""))
    if f.get("filler_per_100w") is not None:
        u = usual("filler_per_100w")
        parts.append(f"{f['filler_per_100w']:.1f} fillers per 100 words" + (f" (usually {u:.1f})" if u is not None else ""))
    timing = "word timings" if f.get("granularity") == "word" else "utterance timings only"
    return "; ".join(parts) + f"; index {score['index']:.2f} ({timing})"


def similarity_evidence(result: dict[str, Any] | None) -> str:
    if result is None or result.get("level") == "pending":
        return "waiting for the reference answers"
    if result["level"] == "unavailable":
        return result.get("detail") or "unavailable"
    text = (f"cosine {result['similarity']:.2f} to the closest of {result['references']} answers "
            f"{result['reference_model']} gave to the same question")
    return text + (f" ({result['detail']})" if result.get("detail") else "")


def signal_list(answer: dict[str, Any], baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The four signals of an answer as `{name, level, calibrated, evidence}`, spec level names."""
    th = signals.thresholds()
    echo, reading, delivery, sim = answer.get("echo"), answer.get("reading"), answer.get("delivery"), answer.get("similarity")
    out = []
    if echo is None:
        out.append({"name": "ai_text", "level": "unavailable", "calibrated": th["ai_text"]["calibrated"], "evidence": "no transcript"})
    else:
        evidence = "too short to judge" if echo["level"] == "too_short" else f"{_pct(echo['ai_probability'])} AI-written"
        out.append({"name": "ai_text", "level": echo["level"], "calibrated": th["ai_text"]["calibrated"], "evidence": evidence})
    if sim is None or sim.get("level") == "pending":
        out.append({"name": "echo_similarity", "level": "unavailable", "calibrated": th["echo_similarity"]["calibrated"],
                    "evidence": similarity_evidence(sim), "pending": sim is not None})
    else:
        out.append({"name": "echo_similarity", "level": sim["level"], "calibrated": sim["calibrated"],
                    "evidence": similarity_evidence(sim)})
    if delivery is None:
        out.append({"name": "delivery", "level": "unavailable", "calibrated": th["delivery"]["calibrated"], "evidence": "no timing"})
    else:
        out.append({"name": "delivery", "level": delivery["score"]["level"], "calibrated": delivery["score"]["calibrated"],
                    "evidence": delivery_evidence(delivery, baseline)})
    if reading is None:
        out.append({"name": "reading", "level": "unavailable", "calibrated": th["reading"]["calibrated"], "evidence": "no gaze span"})
    elif reading.get("reading_probability") is None:
        out.append({"name": "reading", "level": "no_gaze", "calibrated": th["reading"]["calibrated"],
                    "evidence": "not enough gaze (needs 10 s with the face in view)"})
    else:
        out.append({"name": "reading", "level": reading["level"], "calibrated": th["reading"]["calibrated"],
                    "evidence": f"{_pct(reading['reading_probability'])} reading-like eye movement"})
    return out


def flag(answer: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """The combined flag: spec fusion over the four signals, with each signal's evidence."""
    items = signal_list(answer, baseline)
    fused = signals.fuse([{k: s[k] for k in ("name", "level", "calibrated")} for s in items])
    return {**fused, "label": FLAG_LABELS[fused["level"]], "signals": items}
