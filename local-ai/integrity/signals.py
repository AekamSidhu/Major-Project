"""OpenMeet's shared answer-integrity definitions, in Python.

The definitions live in `spec/integrity/` at the repository root (README.md
in prose, reference.py as the reference implementation, thresholds.json for
the levels). This module is the sidecar's and the room's own implementation
of them; `test_signals.py` checks it against the spec's fixtures to 1e-6.

- delivery: does an answer's timing look read rather than spoken freely,
  compared with the same speaker's first answers?
- echo similarity: how close is an answer to what an LLM answers to the same
  question? Also holds the recipe for those reference answers, which the
  HC3 calibration (training/echo-detection/calibrate_similarity.py) uses too:
  thresholds measured with one recipe do not carry over to another.
- fusion: one flag from the per-signal levels.

Standard library only and no imports from this package, so training scripts
can load this file by path without the sidecar's dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

SPEC_VERSION = 1
SPEC_ENV = "OPENMEET_INTEGRITY_SPEC"  # a folder holding thresholds.json, overrides the search

# --- the spec's thresholds ----------------------------------------------------


def find_spec_dir(start: Path | None = None) -> Path | None:
    """`spec/integrity` of the repository this file is in (or OPENMEET_INTEGRITY_SPEC)."""
    env = os.environ.get(SPEC_ENV)
    if env:
        folder = Path(env)
        return folder if (folder / "thresholds.json").is_file() else None
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        folder = parent / "spec" / "integrity"
        if (folder / "thresholds.json").is_file():
            return folder
    return None


_THRESHOLDS: dict[str, Any] | None = None


def thresholds() -> dict[str, Any]:
    """The spec's thresholds.json, read once. Raises FileNotFoundError if the spec is missing."""
    global _THRESHOLDS
    if _THRESHOLDS is None:
        folder = find_spec_dir()
        if folder is None:
            raise FileNotFoundError(f"spec/integrity/thresholds.json not found (set {SPEC_ENV} to its folder)")
        data = json.loads((folder / "thresholds.json").read_text(encoding="utf-8"))
        if data.get("spec_version") != SPEC_VERSION:
            raise ValueError(f"spec version {data.get('spec_version')} != {SPEC_VERSION} implemented here")
        _THRESHOLDS = data
    return _THRESHOLDS


# --- delivery -------------------------------------------------------------------

PAUSE_MIN_S = 0.3
MIN_SPEAKING_S = 1.0
MIN_SPAN_S = 5.0
MIN_WORDS = 10
BASELINE_ANSWERS = 3
MAD_TO_SD = 1.4826
WORD_RE = re.compile(r"[a-z0-9']+")
FILLERS = frozenset({"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm"})
FILLER_BIGRAMS = frozenset({("you", "know"), ("i", "mean")})
WEIGHTS = {"latency_s": 1.0, "rate_wpm": 0.5, "pause_per_min": -0.5, "filler_per_100w": -0.5}
FLOORS = {"latency_s": 0.5, "rate_wpm": 15.0, "pause_per_min": 3.0, "filler_per_100w": 1.0}
SCORED = ("low", "possible", "high")
NOT_SCORED = ("too_short", "baseline", "no_gaze", "unavailable")


def tokens(text: str) -> list[str]:
    """Lower-case runs of a-z, 0-9 and apostrophes."""
    return WORD_RE.findall(text.lower())


def delivery_features(question_end: float | None, utterances: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Timing features of one answer; None when there is no speech.

    `utterances`: the answerer's speech, `{"start", "end", "text", "words"?}`
    in seconds on one clock. Words (`{"word", "start", "end"}`) are used only
    when every utterance has them.
    """
    if not utterances:
        return None
    utts = sorted(utterances, key=lambda u: (u["start"], u["end"]))
    by_word = all(u.get("words") for u in utts)
    if by_word:
        units = sorted((w["start"], w["end"]) for u in utts for w in u["words"])
        toks = [t for u in utts for w in u["words"] for t in tokens(w["word"])]
    else:
        units = [(u["start"], u["end"]) for u in utts]
        toks = [t for u in utts for t in tokens(u["text"])]
    n = len(toks)
    fillers = sum(t in FILLERS for t in toks) + sum((a, b) in FILLER_BIGRAMS for a, b in zip(toks, toks[1:]))
    speaking = sum(max(0.0, u["end"] - u["start"]) for u in utts)
    span = max(u["end"] for u in utts) - utts[0]["start"]
    gaps = [nxt[0] - prev[1] for prev, nxt in zip(units, units[1:])]
    pauses = [g for g in gaps if g >= PAUSE_MIN_S]
    return {
        "granularity": "word" if by_word else "utterance",
        "word_count": n,
        "speaking_s": speaking,
        "span_s": span,
        "latency_s": None if question_end is None else max(0.0, utts[0]["start"] - question_end),
        "rate_wpm": n / speaking * 60.0 if speaking >= MIN_SPEAKING_S else None,
        "pause_per_min": len(pauses) / (span / 60.0) if span >= MIN_SPAN_S else None,
        "mean_pause_s": statistics.fmean(pauses) if pauses else 0.0,
        "filler_per_100w": 100.0 * fillers / n if n else None,
    }


def delivery_baseline(features: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    """Median and robust scale of each weighted feature over the baseline answers."""
    out: dict[str, dict[str, float] | None] = {}
    for name in WEIGHTS:
        values = [f[name] for f in features if f.get(name) is not None]
        if len(values) < 2:
            out[name] = None
            continue
        median = statistics.median(values)
        mad = statistics.median([abs(v - median) for v in values])
        out[name] = {"median": median, "scale": max(MAD_TO_SD * mad, FLOORS[name])}
    return out


def delivery_score(features: dict[str, Any] | None, baseline: dict[str, Any] | None,
                   th: dict[str, Any] | None = None) -> dict[str, Any]:
    """The delivery index and level of one answer against its speaker's baseline."""
    th = th or thresholds()["delivery"]
    calibrated = th["calibrated"]
    if features is None or features["word_count"] < MIN_WORDS:
        return {"level": "too_short", "index": None, "z": {}, "calibrated": calibrated}
    if baseline is None:
        return {"level": "baseline", "index": None, "z": {}, "calibrated": calibrated}
    z: dict[str, float] = {}
    weighted = total = 0.0
    for name, weight in WEIGHTS.items():
        base, value = baseline.get(name), features.get(name)
        if base is None or value is None:
            continue
        z[name] = (value - base["median"]) / base["scale"]
        weighted += weight * z[name]
        total += abs(weight)
    if total == 0.0:
        return {"level": "unavailable", "index": None, "z": {}, "calibrated": calibrated}
    index = weighted / total
    level = "high" if index >= th["high_index"] else "possible" if index >= th["possible_index"] else "low"
    return {"level": level, "index": index, "z": z, "calibrated": calibrated}


class DeliveryBaseline:
    """One speaker's answers scored in order (the spec's `score_session`, one answer at a time).

    The first BASELINE_ANSWERS answers with at least MIN_WORDS words form the
    baseline (level "baseline"); later answers are scored against it.
    """

    def __init__(self, th: dict[str, Any] | None = None) -> None:
        self.th = th
        self.features: list[dict[str, Any]] = []
        self.baseline: dict[str, Any] | None = None

    def add(self, question_end: float | None, utterances: list[dict[str, Any]]) -> dict[str, Any]:
        f = delivery_features(question_end, utterances)
        if f is not None and f["word_count"] >= MIN_WORDS and len(self.features) < BASELINE_ANSWERS:
            self.features.append(f)
            if len(self.features) == BASELINE_ANSWERS:
                self.baseline = delivery_baseline(self.features)
            return {"features": f, "score": delivery_score(f, None, self.th)}
        return {"features": f, "score": delivery_score(f, self.baseline, self.th)}


def score_session(answers: list[dict[str, Any]], th: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    speaker = DeliveryBaseline(th)
    return [speaker.add(a.get("question_end"), a["utterances"]) for a in answers]


# --- echo similarity ---------------------------------------------------------------

# The recipe for reference answers. The HC3 calibration and the room both use
# these, so a threshold measured on HC3 means the same thing in an interview.
# Changing any of them invalidates the echo_similarity thresholds.
REFERENCE_RECIPE = "v1"
REFERENCE_SYSTEM = ("You are a helpful assistant. Answer the user's question directly, in plain prose without "
                    "headings or lists, in about 150 words.")
REFERENCE_TEMPERATURE = 0.7
REFERENCE_MAX_TOKENS = 400


def reference_seed(i: int) -> int:
    """Reference answer i is sampled with seed i (where the model supports seeds)."""
    return i


def reference_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": REFERENCE_SYSTEM}, {"role": "user", "content": question.strip()}]


# Before embedding, the answer and every reference go through
# integrity.text.to_transcript (lower case, words only), as in training.


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def echo_similarity(answer: list[float], references: list[list[float]], th: dict[str, Any] | None = None,
                    embedding_model: str | None = None) -> dict[str, Any]:
    """Highest cosine between the answer's embedding and any reference's.

    `embedding_model`, when given, must be the one the thresholds were
    measured with; otherwise the signal is "unavailable".
    """
    th = th or thresholds()["echo_similarity"]
    calibrated = th["calibrated"]
    if not references or (embedding_model is not None and embedding_model != th["embedding_model"]):
        return {"level": "unavailable", "similarity": None, "calibrated": calibrated}
    sim = max(cosine(answer, r) for r in references)
    level = "high" if sim >= th["high"] else "possible" if sim >= th["possible"] else "low"
    return {"level": level, "similarity": sim, "calibrated": calibrated}


# --- fusion -----------------------------------------------------------------------


def fuse(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """One flag from `{"name", "level", "calibrated"}` per signal.

    Not-scored levels are ignored; an uncalibrated signal counts at most as
    "possible"; "high" needs a calibrated high signal plus at least one other
    elevated signal. `elevated` names the signals behind the flag.
    """
    rank = {"low": 0, "possible": 1, "high": 2}
    scored = [(s["name"], min(rank[s["level"]], 2 if s["calibrated"] else 1)) for s in signals if s["level"] in rank]
    if not scored:
        return {"level": "unavailable", "elevated": []}
    elevated = sorted(name for name, r in scored if r >= 1)
    if any(r == 2 for _, r in scored) and len(elevated) >= 2:
        return {"level": "high", "elevated": elevated}
    return {"level": "possible" if elevated else "low", "elevated": elevated}
