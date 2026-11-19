"""Smoke tests of the delivery evaluation on synthetic data (no downloads, no Whisper)."""

from __future__ import annotations

import io
import json
import math
import struct
import wave

import numpy as np

import evaluate_delivery as ed


def words(read: bool, start: float, seed: int, n: int = 30) -> list[dict]:
    """Read: steady and quick, no fillers. Spontaneous: slower, pauses every few words, fillers."""
    rng = np.random.default_rng(seed)
    out, t = [], start
    for i in range(n):
        if not read and i % 6 == 3:
            out.append({"word": "um", "start": t, "end": t + 0.3})
            t += 0.3 + 0.1
            continue
        out.append({"word": f"w{i}", "start": t, "end": t + 0.22})
        t += 0.25 if read else 0.32 + (0.6 if i % 5 == 4 else 0.0) + rng.uniform(0, 0.05)
    return out


def test_folder_of_labelled_answers(tmp_path):
    for s in range(6):
        for label, k in (("spontaneous", 7), ("read", 4)):
            folder = tmp_path / f"S{s}" / label
            folder.mkdir(parents=True)
            for i in range(k):
                name = f"{'a' if label == 'spontaneous' else 'b'}{i:02d}.json"  # spontaneous first: the baseline
                (folder / name).write_text(json.dumps({"words": words(label == "read", 0.0, s * 100 + i)}), encoding="utf-8")
    answers = ed.baseline_first(ed.folder_answers(tmp_path))
    assert sum(a.get("role") == "baseline" for a in answers) == 18
    result = ed.evaluate(answers)
    s = result["summary"]
    assert s["speakers"] == 6 and s["answers"] == {"read": 24, "spontaneous": 24}
    assert s["roc_auc"] > 0.9 and s["roc_auc_95ci_by_speaker"][0] > 0.5
    assert s["feature_auc"]["filler_per_100w"] > 0.9 and s["feature_auc"]["latency_s"] is None
    assert result["fitted_weights_for_information"]["test_speakers"] == 2
    assert result["thresholds"]["calibrated"] is False or result["thresholds"]["calibrated"] is True


def test_utterances_split_at_the_rooms_silence():
    ws = [{"word": "a", "start": 0.0, "end": 0.2}, {"word": "b", "start": 0.4, "end": 0.6}, {"word": "c", "start": 1.5, "end": 1.7}]
    utts = ed.utterances_from_words(ws)
    assert [(u["start"], u["end"], u["text"]) for u in utts] == [(0.0, 0.6, "a b"), (1.5, 1.7, "c")]


def textgrid(tiers: dict[str, list[tuple[float, float, str]]], xmax: float) -> str:
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {xmax}", "tiers? <exists>",
             f"size = {len(tiers)}", "item []:"]
    for k, (name, ivs) in enumerate(tiers.items(), 1):
        lines += [f"    item [{k}]:", '        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                  f"        xmax = {xmax}", f"        intervals: size = {len(ivs)}"]
        for j, (a, b, t) in enumerate(ivs, 1):
            lines += [f"        intervals [{j}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{t}"']
    return "\n".join(lines) + "\n"


def test_spice_textgrid_to_answers(tmp_path):
    read_words = [(1.0 + 0.3 * i, 1.0 + 0.3 * i + 0.25, f"r{i}") for i in range(60)]  # 60 words: two answers
    talk_words = [(40.0 + 0.35 * i, 40.0 + 0.35 * i + 0.3, "um" if i % 7 == 0 else f"t{i}") for i in range(45)]
    talk_words += [(60.0 + 0.35 * i, 60.0 + 0.35 * i + 0.3, f"u{i}") for i in range(30)]  # after a 2 s+ gap
    tiers = {"task": [(0, 30, "sentences"), (30, 35, "storyboard"), (35, 80, "interview")],
             "utterance": [(1.0, 5.0, "x"), (5.0, 19.0, "x"), (40.0, 56.0, "x"), (60.0, 71.0, "x")],
             "word": [(a, b, t) for a, b, t in read_words + talk_words]}
    path = tmp_path / "VX01A_English_I1_20200101.TextGrid"
    path.write_bytes(textgrid(tiers, 80.0).encode("utf-16"))
    answers = ed.spice_answers(path)
    assert [a["label"] for a in answers] == ["read", "read", "spontaneous", "spontaneous"]  # the 15-word rest is dropped
    assert all(sum(len(u["words"]) for u in a["utterances"]) == ed.CHUNK_WORDS for a in answers)
    assert answers[0]["speaker"] == "VX01A" and len(answers[0]["utterances"]) == 2  # split by the transcript's utterances


def test_remote_wav_reads_one_channel_by_byte_range():
    rate, seconds = 44100, 2.0
    n = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<hh", int(10000 * math.sin(2 * math.pi * 220 * i / rate)), 0) for i in range(n)))
    data = buf.getvalue()

    class LocalWav(ed.RemoteWav):
        def _get(self, start: int, end: int) -> bytes:
            return data[start : end + 1]

    wav = LocalWav("local")
    assert (wav.channels, wav.rate) == (2, rate)
    left, right = wav.read(0.5, 1.5, channel=0), wav.read(0.5, 1.5, channel=1)
    assert len(left) == 16000 and np.abs(left).max() > 0.25 and np.abs(right).max() == 0.0
