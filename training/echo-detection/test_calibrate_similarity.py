"""Smoke test of the echo-similarity calibration on a synthetic HC3 (no Ollama, no downloads)."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

import calibrate_similarity as cs
from train_content import load_hc3

HERE = Path(__file__).resolve().parent
TOPICS = ["caching", "databases", "interest", "vaccines", "planets", "compilers", "taxes", "sleep"]
FILLER = "really honestly kind of maybe just stuff like that i guess we found out the hard way at work".split()


def fake_hc3(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(0)
    sources = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    with path.open("w", encoding="utf-8") as f:
        for q in range(n):
            topic = TOPICS[q % len(TOPICS)]
            ai = f"{topic} is an important concept {q} . In summary , {topic} works by " + " ".join(f"{topic}{k}" for k in range(20))
            humans = [" ".join(rng.choice(FILLER, 25)) + f" {topic} q{q} answer {h}" for h in range(2)]
            f.write(json.dumps({"question": f"What about {topic} number {q}?", "human_answers": humans,
                                "chatgpt_answers": [ai], "source": sources[q % len(sources)]}) + "\n")


def fake_generate(question: str, i: int) -> str:
    topic = question.split()[2]
    return f"{topic} is a key concept. Overall, {topic} works by " + " ".join(f"{topic}{k}" for k in range(20 - i))


def fake_embed(texts):
    out = np.zeros((len(texts), 64))
    for row, text in enumerate(texts):
        for word in text.split():
            out[row, zlib.crc32(word.encode()) % 64] += 1.0
    return out


def test_calibration_runs_end_to_end_and_respects_the_split(tmp_path):
    data = tmp_path / "all.jsonl"
    fake_hc3(data)
    table = load_hc3(data)
    cache = cs.ReferenceCache(tmp_path / "refs.jsonl", "fake")
    logs: list[str] = []
    result = cs.calibrate(table, 2, 3, 3, cache, fake_generate, fake_embed, logs.append)
    t = result["thresholds"]
    assert t["possible"] <= t["high"]
    assert result["test"]["roc_auc"] > 0.9  # the fake AI answers echo the fake references
    assert result["calibrated"] is True and "ROC-AUC >= 0.70" in result["calibration_rule"]
    assert set(result["test"]["by_source"]) <= {"finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"}
    assert 0 <= result["test"]["within_question_auc"] <= 1
    # References are cached: a second run generates nothing.
    calls: list[str] = []
    again = cs.calibrate(table, 2, 3, 3, cs.ReferenceCache(tmp_path / "refs.jsonl", "fake"),
                         lambda q, i: calls.append(q) or "x", fake_embed, logs.append)
    assert not calls and again["thresholds"] == result["thresholds"]


def test_update_spec_touches_only_the_echo_block(tmp_path):
    spec = json.loads((cs.SPEC_THRESHOLDS).read_text(encoding="utf-8"))
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    result = {"thresholds": {"possible": 0.81, "high": 0.86}, "calibrated": False, "development": {"questions": 10},
              "test": {"questions": 5, "roc_auc": 0.6, "at_possible": {"false_positive_rate": 0.3},
                       "at_high": {"false_positive_rate": 0.05}}}
    cs.update_spec(path, result, "x.json")
    after = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "echo_similarity"} == {k: v for k, v in spec.items() if k != "echo_similarity"}
    block = after["echo_similarity"]
    assert (block["possible"], block["high"], block["calibrated"]) == (0.81, 0.86, False)
    assert "too weak" in block["source"] and block["embedding_model"] == spec["echo_similarity"]["embedding_model"]


def test_recipe_comes_from_the_app():
    assert cs.SIGNALS.reference_messages("Why?")[0]["content"] == cs.SIGNALS.REFERENCE_SYSTEM
    assert cs.up4(0.80001) == 0.8001 and cs.up4(0.8) == 0.8


def test_transcript_normalisation_is_the_apps():
    """The calibration normalises text exactly as the sidecar does (same file)."""
    app = (HERE.parents[1] / "local-ai" / "integrity" / "text.py").read_text(encoding="utf-8")
    assert (HERE / "content" / "text.py").read_text(encoding="utf-8") == app
