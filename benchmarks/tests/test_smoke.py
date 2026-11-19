"""Smoke tests for the benchmark scripts: synthetic data only, no network, GPU or datasets."""

from __future__ import annotations

import importlib.util
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

import numpy as np
import pytest

import bench_engine
import bench_followups
import bench_fusion
import bench_room
import benchlib
import librispeech


def signals():
    spec = importlib.util.spec_from_file_location("om_signals_test", benchlib.LOCAL_AI / "integrity" / "signals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- benchlib -------------------------------------------------------------------------


def test_percentiles_match_numpy():
    xs = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0]
    for q in (0, 50, 90, 95, 100):
        assert benchlib.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))
    s = benchlib.summary(xs + [None, float("nan")])
    assert s["n"] == 7 and s["median"] == 5.0 and s["min"] == 1.0 and s["max"] == 9.0
    assert benchlib.summary([]) == {"n": 0}


def test_wilson_interval_contains_the_rate():
    lo, hi = benchlib.wilson(20, 100)
    assert lo < 0.2 < hi and 0 <= lo and hi <= 1
    assert benchlib.wilson(0, 0) is None
    assert benchlib.rate(0, 0) is None


def test_wav_round_trip(tmp_path: Path):
    tone = 0.5 * np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000)
    data = benchlib.wav_bytes(tone)
    assert benchlib.wav_duration(data) == pytest.approx(1.0)
    path = tmp_path / "tone.wav"
    path.write_bytes(data)
    back = benchlib.wav_samples(path)
    assert np.abs(back - tone).max() < 1e-3


def test_multipart_body_parses():
    body, ctype = benchlib.multipart([("model", "base.en"), ("language", "en")], [("file", "a.wav", b"RIFFxxxx", "audio/wav")])
    msg = BytesParser(policy=policy.default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
    assert parts["model"].get_content().strip() == "base.en"
    assert parts["file"].get_content() == b"RIFFxxxx"


def test_sidecar_env_puts_cuda_folders_on_path(tmp_path: Path):
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    env = benchlib.sidecar_env(device="cuda", cuda_libs=tmp_path, model_dir=tmp_path / "models")
    assert env["OPENMEET_WHISPER_DEVICE"] == "cuda"
    assert str(tmp_path / "nvidia" / "cublas" / "bin") in env["PATH"]
    with pytest.raises(FileNotFoundError):
        benchlib.sidecar_env(cuda_libs=tmp_path / "empty")


# --- transcription ---------------------------------------------------------------------


def test_librispeech_listing_and_seeded_subset(tmp_path: Path):
    for spk, chapter in (("11", "100"), ("22", "200")):
        folder = tmp_path / "LibriSpeech" / "test-clean" / spk / chapter
        folder.mkdir(parents=True)
        lines = [f"{spk}-{chapter}-{i:04d} WORDS OF UTTERANCE {i}" for i in range(10)]
        (folder / f"{spk}-{chapter}.trans.txt").write_text("\n".join(lines) + "\n")
    items = librispeech.utterances(tmp_path, "test-clean")
    assert len(items) == 20 and items[0].id == "11-100-0000" and items[0].text == "WORDS OF UTTERANCE 0"
    a, b = librispeech.subset(items, 5, 2026), librispeech.subset(list(reversed(items)), 5, 2026)
    assert [u.id for u in a] == [u.id for u in b] == sorted(u.id for u in a)
    assert librispeech.subset(items, 50, 1) == items


def test_word_error_rate_with_whisper_normaliser():
    pytest.importorskip("jiwer")
    pytest.importorskip("whisper_normalizer")
    import bench_asr

    norm = bench_asr.normaliser()
    same = bench_asr.word_errors(norm("MISTER SMITH PAID TWENTY DOLLARS"), norm("Mr. Smith paid $20."))
    assert same["sub"] + same["del"] + same["ins"] == 0
    rows = [bench_asr.word_errors("the cat sat", "the cat sat down"), bench_asr.word_errors("a b c d", ""),
            bench_asr.word_errors("one two", "one three")]
    w = bench_asr.wer_of(rows)
    assert (w["insertions"], w["deletions"], w["substitutions"], w["ref_words"]) == (1, 4, 1, 9)
    assert w["wer"] == pytest.approx(6 / 9)


def test_asr_log_patterns():
    import bench_asr

    log = "x loaded whisper model base.en on cuda in 1.0s\ny transcribed 3.0s audio in 31 ms (36 chars)\n"
    assert bench_asr.LOADED.findall(log) == [("base.en", "cuda", "1.0")]
    assert bench_asr.TRANSCRIBED.findall(log) == [("3.0", "31")]


# --- detection: the fused flag ------------------------------------------------------------


def test_fusion_rates_on_synthetic_answers():
    pytest.importorskip("sklearn")
    rows = [
        {"source": "a", "label": 1, "ai_level": "high", "echo_level": "high", "echo_level_1ref": "possible", "ai_probability": 0.9, "similarity": 0.95, "similarity_first": 0.9},
        {"source": "a", "label": 1, "ai_level": "high", "echo_level": "low", "echo_level_1ref": "low", "ai_probability": 0.8, "similarity": 0.5, "similarity_first": 0.5},
        {"source": "b", "label": 0, "ai_level": "low", "echo_level": "high", "echo_level_1ref": "high", "ai_probability": 0.1, "similarity": 0.93, "similarity_first": 0.93},
        {"source": "b", "label": 0, "ai_level": "low", "echo_level": "low", "echo_level_1ref": "low", "ai_probability": 0.2, "similarity": 0.4, "similarity_first": 0.4},
    ]
    out = bench_fusion.evaluate(rows, signals().fuse, {"ai_text": True, "echo_similarity": True})
    fused = out["configs"]["fused (ai_text + echo)"]
    # AI answer 1: high + high -> high; AI answer 2: high alone -> possible; human 1: echo high alone -> possible.
    assert fused["at_high"]["detected"] == 1 and fused["at_possible"]["detected"] == 2
    assert fused["at_possible"]["false_flags"] == 1 and fused["at_high"]["false_flags"] == 0
    alone = out["configs"]["fused, ai_text only available"]
    assert alone["at_high"]["detected"] == 0  # one signal never makes the flag high
    assert out["configs"]["ai_text alone"]["at_high"]["detection_rate"] == 1.0
    assert out["false_flags_by_cause"] == {"possible: ai_text low, echo high": 1}
    assert set(out["by_source"]) == {"a", "b"}


def test_fusion_level_cuts():
    assert bench_fusion.level(0.9201, 0.8712, 0.9201) == "high"
    assert bench_fusion.level(0.88, 0.8712, 0.9201) == "possible"
    assert bench_fusion.level(0.5, 0.8712, 0.9201) == "low"


# --- room ---------------------------------------------------------------------------------


def tone(seconds: float) -> np.ndarray:
    return (0.3 * np.sin(np.arange(int(seconds * 16000)) * 2 * np.pi * 220 / 16000)).astype(np.float32)


def test_host_page_segmenter_cuts_at_long_pauses_and_at_20_s():
    audio = np.concatenate([tone(3.0), np.zeros(24000, np.float32), tone(2.0)])
    pieces = bench_room.split_speech(audio)
    assert len(pieces) == 2
    (s1, e1, sent1), (s2, e2, sent2) = pieces
    assert s1 == 0.0 and 2.8 < e1 < 3.4 and sent1 > e1 + 0.8  # sent after more than 0.8 s of silence
    assert 3.5 < s2 < 4.6 and 6.2 < e2 < 6.9  # two blocks of pre-roll before the speech
    # 1.0 s of silence leaves only three fully silent 0.256 s blocks (0.77 s): not a pause for the host page.
    one_second = np.concatenate([tone(3.0), np.zeros(16000, np.float32), tone(2.0)])
    assert len(bench_room.split_speech(one_second)) == 1
    long = bench_room.split_speech(tone(25.0))
    assert len(long) == 2 and long[0][2] == pytest.approx(long[0][1]) and 19.9 < long[0][1] < 20.6


def test_room_timeline_order():
    q, a = tone(2.0), tone(5.0)
    events = bench_room.timeline([("x", "own", 2.5, 1.5), ("y", "llm", 0.8, 0.6)], {"x": q, "y": q}, {"x": a, "y": a})
    kinds = [k for _, k, _ in events]
    assert kinds.count("close") == 2 and kinds.count("answer_start") == 2
    times = [t for t, _, _ in events]
    assert times == sorted(times)
    close_1 = next(t for t, k, e in events if k == "close" and e["turn"] == 1)
    second_question = min(t for t, k, e in events if k == "piece" and e["turn"] == 2 and e["speaker"] == "interviewer")
    assert close_1 < second_question  # the interviewer starting to talk closes the answer


# --- follow-ups and engine summaries ----------------------------------------------------------


def test_followup_quality_summary():
    rows = [{"raw": 3, "kept": 2, "dropped": ["not a question"], "grounded": 1},
            {"raw": 0, "kept": 0, "dropped": [], "grounded": 0}]
    out = bench_followups.summarise_quality(rows)
    assert out["suggestions_before_guardrails"] == 3 and out["kept_total"] == 2
    assert out["calls_with_at_least_one_kept"] == 1 and out["dropped_by_reason"] == {"not a question": 1}


def test_engine_live_summary():
    raw = json.loads(json.dumps({
        "questions": [{"question_detected": True, "transcripts": 1, "speech_end_to_transcript_ms": 900.0,
                       "speech_end_to_question_ms": 901.0, "question_to_first_token_ms": 300.0, "answer_chars": 500},
                      {"question_detected": False, "transcripts": 2}],
        "answers": [{"followups": 3, "speech_end_to_followups_done_ms": 4000.0}],
        "summary_ms": 5000.0, "summary_ok": True, "engine_latency_stats": []}))
    out = bench_engine.summarise_live(raw)
    assert out["candidate_mode"]["detected"] == 1 and out["candidate_mode"]["single_transcript"] == 1
    assert out["candidate_mode"]["speech_end_to_transcript_ms"]["median"] == 900.0
    assert out["interviewer_mode"]["with_followups"] == 1


# --- the doc's tables ------------------------------------------------------------------------


def test_tables_fill_only_replaces_marked_blocks(monkeypatch):
    import tables

    monkeypatch.setattr(tables, "SECTIONS", {"demo": lambda: print("| a | b |\n|---|---|\n| 1 | 2 |")})
    doc = "# Title\n\nprose\n<!-- tables:demo -->\nold numbers\n<!-- /tables -->\nmore prose\n"
    new = tables.fill(doc)
    assert new == "# Title\n\nprose\n<!-- tables:demo -->\n| a | b |\n|---|---|\n| 1 | 2 |\n<!-- /tables -->\nmore prose\n"
    assert tables.fill(new) == new  # idempotent: --check passes right after a fill


def test_table_helpers():
    import tables

    assert tables.pct(0.12345) == "12.3%" and tables.pct(None) == "–"
    assert tables.lat({"n": 3, "median": 1.0, "p90": 2.0, "p95": 3.0}) == "1 / 2 / 3"
    assert tables.lat({"n": 0}) == "–"
