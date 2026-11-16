"""Follow-up questions by simple rules, for when no AI model is available.

Each rule looks for a gap that interviewers commonly probe (failure handling,
how a number was measured, the candidate's own part, alternatives, scale)
and quotes the sentence it comes from, so the question stays grounded.
"""

from __future__ import annotations

import re

MAX_FOLLOWUPS = 3
PRIORITIES = ("high", "normal", "optional")

STOP = set("""a an the and or but if so of to in on at for with from by as is are was were be been being it its this that these
those i we you he she they them our my your their me us do did does done have has had not no yes can could would should will just
then than there here what which who how when where why also very really like um uh okay well actually basically""".split())
DEPENDENCIES = set("""redis memcached postgres postgresql mysql mongodb dynamodb cassandra kafka rabbitmq sqs s3 elasticsearch
opensearch firebase supabase stripe twilio openai""".split())
WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#/-]*[A-Za-z0-9+#]")
METRIC = re.compile(r"\b(\d[\d,.]*\s?(?:%|percent|ms|milliseconds|seconds|x|times|k|million|users|requests|rps|qps|gb|tb)?)", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def _quote(sentence: str, words: int = 15) -> str:
    parts = sentence.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def _technologies(text: str) -> list[str]:
    """Named things: capitalised words inside a sentence, or words like PostgreSQL and S3 anywhere."""
    seen: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for n, word in enumerate(WORD.findall(sentence)):
            inner = any(c.isupper() or c.isdigit() for c in word[1:])
            if word.lower() in STOP or not ((n > 0 and word[0].isupper()) or inner):
                continue
            if word not in seen:
                seen.append(word)
    return seen


def rule_followups(question: str, answer: str, previous: list[str]) -> list[dict[str, str]]:
    sentences = _sentences(answer)
    if not sentences:
        return [{
            "question": "Could you walk me through that in more detail, with a concrete example?",
            "reason": "The answer was too short to follow up on anything specific.",
            "priority": "high", "trigger": _quote(answer) if answer.strip() else "",
        }]
    lower = answer.lower()
    out: list[dict[str, str]] = []

    def add(q: str, reason: str, priority: str, trigger: str) -> None:
        if q.lower() not in (p.lower() for p in previous) and all(q != o["question"] for o in out):
            out.append({"question": q, "reason": reason, "priority": priority, "trigger": _quote(trigger)})

    def find(pattern: str) -> str:
        return next((s for s in sentences if re.search(pattern, s, re.I)), sentences[0])

    techs = _technologies(answer)
    if techs and not re.search(r"\b(fail|down|outage|unavailable|crash|retry|fallback|timeout)", lower):
        # Prefer a dependency the system calls at run time (a store, a queue) over a platform.
        t = next((w for w in techs if w.lower() in DEPENDENCIES), techs[0])
        add(f"What happens if {t} is slow or unavailable, and how would the rest of the system behave?",
            f"Mentioned {t} but not how the design copes when it fails.", "high", find(re.escape(t)))
    metric = next((m.group(1) for m in METRIC.finditer(answer) if any(c.isalpha() or c == "%" for c in m.group(1))), None)
    if metric:
        add(f"How did you measure the {metric.strip()}, and what was it before your change?",
            "Gave a number without saying how it was measured.", "normal", find(re.escape(metric)))
    we, i = len(re.findall(r"\bwe\b", lower)), len(re.findall(r"\bi\b", lower))
    if we >= 2 and we > i:
        add("Which part of this did you build or decide yourself?",
            "Described the team's work; the candidate's own part is unclear.", "high", find(r"\bwe\b"))
    if not re.search(r"\b(because|instead|trade-?off|alternative|rather than|compared|versus|vs)\b", lower):
        add("Why did you choose this approach, and what alternative did you consider?",
            "Gave an approach without the reasoning or the alternatives.", "normal", sentences[0])
    if not re.search(r"\b(scale|load|traffic|users|million|thousand|growth|10x)\b", lower):
        add("How would this need to change at ten times the load?",
            "Scale limits were not discussed.", "optional", sentences[-1])
    if not out:
        add(f"Can you give a concrete example of {_quote(sentences[0], 8).rstrip('.').lower()}?",
            "A concrete example would show depth.", "normal", sentences[0])
    out.sort(key=lambda f: PRIORITIES.index(f["priority"]))
    return out[:MAX_FOLLOWUPS]
