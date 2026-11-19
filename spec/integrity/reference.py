"""Reference implementation of OpenMeet's shared answer-integrity definitions.

Standard library only. The Rust engine (openmeet-intel) and the Python sidecar
(local-ai) each implement these definitions and must reproduce the fixtures in
this folder to within 1e-6:

    delivery_cases.json   delivery features, baseline and delivery score
    fusion_cases.json     the combined flag from per-signal levels
    echo_cases.json       echo similarity (cosine) and its level

Regenerate the fixtures after changing a definition (and bump SPEC_VERSION):

    python -I spec/integrity/reference.py

See README.md in this folder for the definitions in prose.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

SPEC_VERSION = 1
HERE = Path(__file__).resolve().parent
THRESHOLDS = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))

# --- delivery -----------------------------------------------------------------

PAUSE_MIN_S = 0.3  # a gap shorter than this between speech units is not a pause
MIN_SPEAKING_S = 1.0  # below this, rate_wpm is undefined
MIN_SPAN_S = 5.0  # below this, pause_per_min is undefined
MIN_WORDS = 10  # answers with fewer words are "too_short"
BASELINE_ANSWERS = 3  # the first N answers with >= MIN_WORDS words form the baseline
MAD_TO_SD = 1.4826
WORD_RE = re.compile(r"[a-z0-9']+")
FILLERS = frozenset({"um", "umm", "uh", "uhm", "er", "erm", "ah", "hmm", "mm"})
FILLER_BIGRAMS = frozenset({("you", "know"), ("i", "mean")})
# Read-aloud answers tend to start late (waiting for the hidden answer), run
# faster and pause and hesitate less than the same person's own baseline.
WEIGHTS = {"latency_s": 1.0, "rate_wpm": 0.5, "pause_per_min": -0.5, "filler_per_100w": -0.5}
FLOORS = {"latency_s": 0.5, "rate_wpm": 15.0, "pause_per_min": 3.0, "filler_per_100w": 1.0}
SCORED_LEVELS = ("low", "possible", "high")


def tokens(text: str) -> list[str]:
    """Lower-case word tokens: runs of a-z, 0-9 and apostrophes."""
    return WORD_RE.findall(text.lower())


def delivery_features(question_end: float | None, utterances: list[dict]) -> dict | None:
    """Features of one answer.

    `utterances` are the answerer's speech in the answer, each
    {"start", "end", "text", "words"?} in seconds on one clock; `words`, when
    present on every utterance, is [{"word", "start", "end"}] (OpenAI shape).
    `question_end` is when the question finished (None if unknown).
    """
    utts = sorted(utterances, key=lambda u: (u["start"], u["end"]))
    if not utts:
        return None
    use_words = all(u.get("words") for u in utts)
    if use_words:
        units = sorted((w["start"], w["end"]) for u in utts for w in u["words"])
        toks = [t for u in utts for w in u["words"] for t in tokens(w["word"])]
    else:
        units = [(u["start"], u["end"]) for u in utts]
        toks = [t for u in utts for t in tokens(u["text"])]
    n = len(toks)
    fillers = sum(1 for t in toks if t in FILLERS)
    fillers += sum(1 for a, b in zip(toks, toks[1:]) if (a, b) in FILLER_BIGRAMS)
    speaking_s = sum(max(0.0, u["end"] - u["start"]) for u in utts)
    span_s = max(u["end"] for u in utts) - utts[0]["start"]
    pauses = [s1 - e0 for (_, e0), (s1, _) in zip(units, units[1:]) if s1 - e0 >= PAUSE_MIN_S]
    return {
        "granularity": "word" if use_words else "utterance",
        "word_count": n,
        "speaking_s": speaking_s,
        "span_s": span_s,
        "latency_s": None if question_end is None else max(0.0, utts[0]["start"] - question_end),
        "rate_wpm": None if speaking_s < MIN_SPEAKING_S else n / speaking_s * 60.0,
        "pause_per_min": None if span_s < MIN_SPAN_S else len(pauses) / (span_s / 60.0),
        "mean_pause_s": statistics.fmean(pauses) if pauses else 0.0,
        "filler_per_100w": None if n == 0 else 100.0 * fillers / n,
    }


def delivery_baseline(features: list[dict]) -> dict:
    """Median and robust scale per weighted feature over the baseline answers.

    The median of an even count is the mean of the two middle values. A
    feature with fewer than two values has no baseline (None).
    """
    out: dict[str, dict | None] = {}
    for name in WEIGHTS:
        xs = [f[name] for f in features if f.get(name) is not None]
        if len(xs) < 2:
            out[name] = None
            continue
        med = statistics.median(xs)
        mad = statistics.median([abs(x - med) for x in xs])
        out[name] = {"median": med, "scale": max(MAD_TO_SD * mad, FLOORS[name])}
    return out


def delivery_score(features: dict | None, baseline: dict | None) -> dict:
    """Delivery index and level for one answer against the speaker's baseline."""
    th = THRESHOLDS["delivery"]
    if features is None or features["word_count"] < MIN_WORDS:
        return {"level": "too_short", "index": None, "z": {}, "calibrated": th["calibrated"]}
    if baseline is None:
        return {"level": "baseline", "index": None, "z": {}, "calibrated": th["calibrated"]}
    num = den = 0.0
    z: dict[str, float] = {}
    for name, w in WEIGHTS.items():
        b, x = baseline.get(name), features.get(name)
        if b is None or x is None:
            continue
        z[name] = (x - b["median"]) / b["scale"]
        num += w * z[name]
        den += abs(w)
    if den == 0.0:
        return {"level": "unavailable", "index": None, "z": {}, "calibrated": th["calibrated"]}
    index = num / den
    if index >= th["high_index"]:
        level = "high"
    elif index >= th["possible_index"]:
        level = "possible"
    else:
        level = "low"
    return {"level": level, "index": index, "z": z, "calibrated": th["calibrated"]}


def score_session(answers: list[dict]) -> list[dict]:
    """Score a speaker's answers in order.

    The first BASELINE_ANSWERS answers with at least MIN_WORDS words are the
    baseline and get level "baseline"; every later answer is scored against
    them. Answers that are too short never join the baseline.
    """
    base_feats: list[dict] = []
    baseline: dict | None = None
    out = []
    for a in answers:
        f = delivery_features(a.get("question_end"), a["utterances"])
        if f is not None and f["word_count"] >= MIN_WORDS and len(base_feats) < BASELINE_ANSWERS:
            base_feats.append(f)
            if len(base_feats) == BASELINE_ANSWERS:
                baseline = delivery_baseline(base_feats)
            out.append({"features": f, "score": delivery_score(f, None)})
        else:
            out.append({"features": f, "score": delivery_score(f, baseline)})
    return out


# --- echo similarity ---------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def echo_similarity(answer: list[float], references: list[list[float]]) -> dict:
    """Highest cosine between the answer's embedding and any reference answer's."""
    th = THRESHOLDS["echo_similarity"]
    if not references:
        return {"level": "unavailable", "similarity": None, "calibrated": th["calibrated"]}
    sim = max(cosine(answer, r) for r in references)
    level = "high" if sim >= th["high"] else "possible" if sim >= th["possible"] else "low"
    return {"level": level, "similarity": sim, "calibrated": th["calibrated"]}


# --- fusion --------------------------------------------------------------------


def fuse(signals: list[dict]) -> dict:
    """One flag from per-signal levels.

    Each signal is {"name", "level", "calibrated"}. Levels other than low,
    possible and high (too_short, baseline, no_gaze, unavailable) are ignored.
    An uncalibrated signal counts at most as "possible". The flag is "high"
    only when a calibrated signal is high and at least one other signal is
    possible or high; one signal alone never makes it "high".
    """
    rank = {"low": 0, "possible": 1, "high": 2}
    active = [s for s in signals if s["level"] in rank]
    if not active:
        return {"level": "unavailable", "elevated": []}
    eff = [(s["name"], min(rank[s["level"]], 2 if s["calibrated"] else 1)) for s in active]
    elevated = sorted(name for name, lvl in eff if lvl >= 1)
    if any(lvl == 2 for _, lvl in eff) and len(elevated) >= 2:
        level = "high"
    elif elevated:
        level = "possible"
    else:
        level = "low"
    return {"level": level, "elevated": elevated}


# --- fixtures ------------------------------------------------------------------


def _words(start: float, text: str, step: float = 0.35, dur: float = 0.25) -> list[dict]:
    out, t = [], start
    for w in text.split():
        out.append({"word": " " + w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += step
    return out


def _utt(start: float, text: str, step: float = 0.35, words: bool = False) -> dict:
    n = len(text.split())
    u = {"start": start, "end": round(start + (n - 1) * step + 0.25, 3), "text": text}
    if words:
        u["words"] = _words(start, text, step)
    return u


def delivery_fixture() -> dict:
    single = []
    single.append({
        "name": "word timestamps, one filler bigram and two fillers",
        "question_end": 10.0,
        "utterances": [
            _utt(11.2, "um so I mean we cached the results in redis", words=True),
            _utt(16.0, "and uh invalidated them when the source row changed you know", words=True),
        ],
    })
    single.append({
        "name": "utterance timings only",
        "question_end": 3.0,
        "utterances": [
            _utt(5.5, "the service keeps a write ahead log so a crash loses nothing"),
            _utt(10.4, "then a background job compacts it every hour"),
            _utt(15.0, "and we alert if it falls behind"),
        ],
    })
    single.append({
        "name": "question end unknown",
        "question_end": None,
        "utterances": [_utt(0.0, "I would start by measuring where the time actually goes before changing anything")],
    })
    single.append({"name": "too short", "question_end": 1.0, "utterances": [_utt(1.5, "yes exactly")]})
    single.append({"name": "no speech", "question_end": 1.0, "utterances": []})
    for case in single:
        f = delivery_features(case["question_end"], case["utterances"])
        case["expected_features"] = f

    chatty = "well um I think so we tried a few things and uh it mostly worked out fine in the end"
    session_answers = [
        {"question_end": 0.0, "utterances": [_utt(0.9, chatty, step=0.42)]},
        {"question_end": 20.0, "utterances": [_utt(21.1, "hmm let me think " + chatty, step=0.40)]},
        {"question_end": 45.0, "utterances": [_utt(45.7, "ok", step=0.4)]},
        {"question_end": 60.0, "utterances": [_utt(60.8, chatty + " you know", step=0.45), _utt(70.9, "uh yeah that is about it", step=0.45)]},
        {"question_end": 90.0, "utterances": [_utt(91.0, "so um the first project " + chatty, step=0.41)]},
        {
            "question_end": 120.0,
            "utterances": [_utt(124.6, "consistent hashing distributes keys across nodes so that adding or removing a node only remaps a small fraction of the keys which keeps rebalancing cheap", step=0.28)],
        },
        {"question_end": 160.0, "utterances": [_utt(161.0, "uh honestly I do not remember the exact number but um it was around two seconds", step=0.43)]},
    ]
    session = {"name": "session with baseline", "answers": session_answers, "expected": score_session(session_answers)}
    return {
        "spec_version": SPEC_VERSION,
        "tolerance": 1e-6,
        "constants": {
            "pause_min_s": PAUSE_MIN_S,
            "min_speaking_s": MIN_SPEAKING_S,
            "min_span_s": MIN_SPAN_S,
            "min_words": MIN_WORDS,
            "baseline_answers": BASELINE_ANSWERS,
            "mad_to_sd": MAD_TO_SD,
            "fillers": sorted(FILLERS),
            "filler_bigrams": sorted(" ".join(b) for b in FILLER_BIGRAMS),
            "weights": WEIGHTS,
            "floors": FLOORS,
        },
        "single_answers": single,
        "session": session,
    }


def fusion_fixture() -> dict:
    s = lambda name, level, cal: {"name": name, "level": level, "calibrated": cal}  # noqa: E731
    cases = [
        [],
        [s("reading", "no_gaze", True), s("delivery", "baseline", False)],
        [s("ai_text", "low", True), s("reading", "low", True)],
        [s("ai_text", "high", True)],
        [s("ai_text", "high", True), s("reading", "low", True)],
        [s("ai_text", "high", True), s("delivery", "possible", False)],
        [s("ai_text", "high", True), s("reading", "high", True)],
        [s("delivery", "high", False)],
        [s("delivery", "high", False), s("echo_similarity", "high", False)],
        [s("delivery", "high", False), s("ai_text", "possible", True)],
        [s("echo_similarity", "possible", False), s("reading", "too_short", True), s("ai_text", "low", True)],
    ]
    return {"spec_version": SPEC_VERSION, "cases": [{"signals": c, "expected": fuse(c)} for c in cases]}


def echo_fixture() -> dict:
    cases = [
        {"answer": [1.0, 0.0, 0.0], "references": []},
        {"answer": [1.0, 0.0, 0.0], "references": [[0.0, 1.0, 0.0]]},
        {"answer": [1.0, 0.2, 0.0], "references": [[0.0, 1.0, 0.0], [1.0, 0.25, 0.05]]},
        {"answer": [0.6, 0.8, 0.0], "references": [[0.8, 0.6, 0.0]]},
        {"answer": [0.0, 0.0, 0.0], "references": [[1.0, 0.0, 0.0]]},
    ]
    for c in cases:
        c["expected"] = echo_similarity(c["answer"], c["references"])
    return {"spec_version": SPEC_VERSION, "thresholds": THRESHOLDS["echo_similarity"], "cases": cases}


def main() -> int:
    for name, data in (
        ("delivery_cases.json", delivery_fixture()),
        ("fusion_cases.json", fusion_fixture()),
        ("echo_cases.json", echo_fixture()),
    ):
        (HERE / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("wrote", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
