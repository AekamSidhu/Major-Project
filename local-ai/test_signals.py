"""The sidecar's integrity definitions reproduce the shared spec's fixtures (spec/integrity/*.json)."""

from __future__ import annotations

import json
import math

import pytest

from integrity import signals

SPEC = signals.find_spec_dir()
TOL = 1e-6


def fixture(name: str) -> dict:
    assert SPEC is not None, "spec/integrity not found"
    return json.loads((SPEC / name).read_text(encoding="utf-8"))


def same(actual, expected, path: str = "") -> None:
    """Equal structure; numbers within the spec's tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), (path, actual, expected)
        for k in expected:
            same(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), (path, actual, expected)
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, (path, actual, expected)
    else:
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (path, actual)
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOL), (path, actual, expected)


def test_spec_is_found_and_matches_this_implementation():
    th = signals.thresholds()
    assert th["spec_version"] == signals.SPEC_VERSION
    consts = fixture("delivery_cases.json")["constants"]
    assert consts["weights"] == signals.WEIGHTS and consts["floors"] == signals.FLOORS
    assert sorted(consts["fillers"]) == sorted(signals.FILLERS)
    assert sorted(consts["filler_bigrams"]) == sorted(" ".join(b) for b in signals.FILLER_BIGRAMS)
    assert (consts["pause_min_s"], consts["min_speaking_s"], consts["min_span_s"], consts["min_words"],
            consts["baseline_answers"], consts["mad_to_sd"]) == (
        signals.PAUSE_MIN_S, signals.MIN_SPEAKING_S, signals.MIN_SPAN_S, signals.MIN_WORDS,
        signals.BASELINE_ANSWERS, signals.MAD_TO_SD)


def test_delivery_single_answers():
    cases = fixture("delivery_cases.json")["single_answers"]
    assert cases
    for case in cases:
        same(signals.delivery_features(case["question_end"], case["utterances"]), case["expected_features"], case["name"])


def test_delivery_session_with_baseline():
    session = fixture("delivery_cases.json")["session"]
    same(signals.score_session(session["answers"]), session["expected"], session["name"])
    # The incremental scorer the room uses gives the same answers.
    speaker = signals.DeliveryBaseline()
    same([speaker.add(a.get("question_end"), a["utterances"]) for a in session["answers"]], session["expected"])


def test_echo_similarity_cases():
    data = fixture("echo_cases.json")
    # The fixtures were generated from the current thresholds.json.
    assert data["thresholds"] == signals.thresholds()["echo_similarity"]
    for case in data["cases"]:
        same(signals.echo_similarity(case["answer"], case["references"]), case["expected"])


def test_echo_similarity_needs_the_calibrated_embedder():
    model = signals.thresholds()["echo_similarity"]["embedding_model"]
    assert signals.echo_similarity([1.0, 0.0], [[1.0, 0.0]], embedding_model=model)["level"] != "unavailable"
    assert signals.echo_similarity([1.0, 0.0], [[1.0, 0.0]], embedding_model="hash-384")["level"] == "unavailable"


def test_fusion_cases():
    cases = fixture("fusion_cases.json")["cases"]
    assert cases
    for case in cases:
        same(signals.fuse(case["signals"]), case["expected"])


def test_spec_location_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv(signals.SPEC_ENV, str(tmp_path))
    assert signals.find_spec_dir() is None  # no thresholds.json there
    (tmp_path / "thresholds.json").write_text("{}", encoding="utf-8")
    assert signals.find_spec_dir() == tmp_path


@pytest.mark.parametrize("text,expected", [("Um, I MEAN it's 2 o'clock", ["um", "i", "mean", "it's", "2", "o'clock"]), ("", [])])
def test_tokens(text, expected):
    assert signals.tokens(text) == expected
