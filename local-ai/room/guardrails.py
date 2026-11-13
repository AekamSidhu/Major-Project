"""Guardrails around the follow-up model.

Before the model sees anything:
- the transcript is data, not instructions: control characters and anything
  that could close the prompt's tags are removed, and lengths are capped;
- in online mode, emails, phone numbers and the candidate's name are replaced
  before the text leaves this computer.

After the model answers, each suggestion must be one short, neutral, open
question about this interview. A suggestion is dropped when it:
- is not a question, or is too short or too long;
- is assistant chatter ("As an AI...") rather than a question;
- asks about a protected characteristic (age, family, religion, health...),
  which interviewers must not ask about;
- judges or accuses the candidate (the integrity scores prompt questions,
  they never become accusations);
- repeats a question already asked or suggested;
- has nothing to do with the question or the answer.
A quoted trigger the candidate never said is removed rather than shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

TAGS = ("question", "candidate_answer", "transcript", "asked_questions", "previous_suggestions")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TAG = re.compile(r"</?\s*(?:%s)\s*>" % "|".join(TAGS), re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?!\w)")
INJECTION = re.compile(
    r"\b(ignore (all |the )?(previous|above|prior|earlier) (instructions|prompts?|rules)|system prompt|"
    r"you are now|disregard (the|your|all) (instructions|rules)|new instructions)\b", re.I)

PROTECTED = re.compile(
    r"\b(how old|your (age|birthday|health|religion|faith|caste|ethnicity|race|nationality|accent|family|spouse|"
    r"husband|wife|partner|kids|children|pregnancy|disability|politics|political views|sexuality)|married|marital|"
    r"pregnan\w*|religio\w*|church|mosque|caste|ethnicity|disabilit\w*|sexual orientation|maternity|paternity|"
    r"plan(s|ning)? to have (kids|children))\b", re.I)
EVALUATIVE = re.compile(
    r"\b((great|good|excellent|impressive|poor|weak|bad|terrible|vague) (answer|response|job|point|explanation)|"
    r"you('re| are) (wrong|right|correct|incorrect|lying)|hiring decision|reject\w*|cheat\w*|"
    r"(are|were) you (reading|using (chatgpt|ai|notes|a script))|read(ing)? (that|this|it) (from|off))\b", re.I)
META = re.compile(r"\b(as an ai|language model|i cannot|i can't help|i'm sorry|here are (some|the|three))\b", re.I)

STOP = set("""a an the and or but if so of to in on at for with from by as is are was were be been being it its this that these
those i we you he she they them our my your their me us do did does done have has had not no yes can could would should will just
then than there here what which who how when where why also very really like um uh okay well actually basically about into over
more most some such only other each every both much many make made used using use would could""".split())
WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#-]{2,}")


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)} - STOP


def similarity(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def clean(text: str, limit: int, redact: bool = False, names: tuple[str, ...] = ()) -> str:
    """Transcript text made safe to put inside the prompt's tags."""
    text = TAG.sub(" ", CONTROL.sub(" ", text))
    if redact:
        text = EMAIL.sub("[email]", text)
        text = PHONE.sub("[phone number]", text)
        for name in sorted({n.strip() for n in names if len(n.strip()) >= 2}, key=len, reverse=True):
            text = re.sub(r"\b%s\b" % re.escape(name), "the candidate", text, flags=re.I)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text[-limit:]


def name_parts(name: str) -> tuple[str, ...]:
    """The full name and each part of it, for redaction."""
    parts = [p for p in name.split() if len(p) >= 2]
    return tuple(dict.fromkeys([name.strip(), *parts]))


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text))


def _said(trigger: str, answer: str) -> bool:
    words = content_words(trigger)
    return bool(words) and len(words & content_words(answer)) / len(words) >= 0.6


@dataclass
class Checked:
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)


def check_followups(items: list[dict[str, Any]], question: str, answer: str, earlier: list[str], limit: int) -> Checked:
    context = content_words(question + " " + answer)
    out = Checked()
    for item in items:
        q = " ".join(str(item.get("question", "")).split())
        words = q.split()
        trigger = str(item.get("trigger", ""))
        if not q.endswith("?"):
            why = "not a question"
        elif not 4 <= len(words) <= 40:
            why = "too short or too long"
        elif META.search(q):
            why = "assistant chatter"
        elif PROTECTED.search(q):
            why = "asks about a protected characteristic"
        elif EVALUATIVE.search(q):
            why = "judges or accuses the candidate"
        elif any(similarity(q, e) >= 0.6 for e in [*earlier, *(k["question"] for k in out.kept)]):
            why = "repeats an earlier question"
        elif context and not (content_words(q) & context) and not _said(trigger, answer):
            why = "not about the answer"
        else:
            why = ""
        if why:
            out.dropped.append((q, why))
            continue
        out.kept.append({**item, "question": q, "trigger": trigger if _said(trigger, answer) else ""})
    out.kept = out.kept[:limit]
    return out
