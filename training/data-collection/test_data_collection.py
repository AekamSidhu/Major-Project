"""Tests of the collection kit: randomisation, and calibration on synthetic study sessions."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import calibrate
import randomise

HERE = Path(__file__).resolve().parent
STUDY_PY = HERE.parents[1] / "local-ai" / "room" / "study.py"


# --- randomise ------------------------------------------------------------------------------


def test_each_participant_gets_a_balanced_plan():
    bank = randomise.load_bank()
    rows = randomise.plan("P07", bank)
    assert len(rows) == 15 and [r["answer_no"] for r in rows] == [str(i) for i in range(1, 16)]
    assert [r["question_type"] for r in rows[:3]] == ["warmup"] * 3 and {r["condition"] for r in rows[:3]} == {"honest"}
    study = rows[3:]
    cells = Counter((r["condition"], r["question_type"]) for r in study)
    assert set(cells.values()) == {1} and len(cells) == 12  # each condition: one question of each type
    assert randomise.no_triples([r["condition"] for r in study])
    assert len({r["question_id"] for r in rows}) == 15
    material = [r for r in study if r["condition"] != "honest"]
    assert all(r["material_id"] == f"P07-{r['question_id']}" for r in material)
    assert Counter(r["screen_position"] for r in material) in (Counter(side=5, below_camera=4), Counter(side=4, below_camera=5))
    assert all(not r["screen_position"] for r in study if r["condition"] == "honest")


def test_plans_are_reproducible_and_differ_between_people(tmp_path):
    bank = randomise.load_bank()
    assert randomise.plan("P01", bank) == randomise.plan("P01", bank)
    assert [r["question_id"] for r in randomise.plan("P01", bank)] != [r["question_id"] for r in randomise.plan("P02", bank)]
    rows = randomise.plan("P01", bank)
    randomise.write(rows, tmp_path / "P01.csv")
    with (tmp_path / "P01.csv").open(encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f))[0]["question_id"] == "W01"
    with (HERE / "label_sheet_template.csv").open(encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == randomise.COLUMNS


def test_conditions_match_the_room():
    found = re.search(r"CONDITIONS = \(([^)]*)\)", STUDY_PY.read_text(encoding="utf-8"))
    assert found and tuple(re.findall(r'"(\w+)"', found.group(1))) == randomise.CONDITIONS


# --- calibrate ------------------------------------------------------------------------------

MEANS = {  # per condition: ai_text, similarity, delivery index, reading
    "honest": (0.2, 0.70, 0.0, 0.2), "read_script": (0.25, 0.72, 2.0, 0.8),
    "read_ai": (0.85, 0.88, 2.2, 0.85), "paraphrase_ai": (0.6, 0.82, 0.8, 0.5),
}


def write_sessions(folder: Path, people: int = 12, seed: int = 0, shift_test: float = 0.0) -> None:
    """Sessions in the room's study format (session.json + answer-NNN.json)."""
    rng = np.random.default_rng(seed)
    for p in range(people):
        sid = f"{p:08x}"
        d = folder / f"session-{sid}"
        d.mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({"session": sid, "participant": f"P{p:02d}",
                                                    "models": {"reference_model": "llama3.2"}}), encoding="utf-8")
        plan = ["honest"] * 3 + ["honest", "read_script", "read_ai", "paraphrase_ai"] * 3
        for n, cond in enumerate(plan, 1):
            ai, sim, idx, rd = (m + rng.normal(0, 0.08) + shift_test for m in MEANS[cond])
            baseline = n <= 3
            (d / f"answer-{n:03d}.json").write_text(json.dumps({
                "session": sid, "participant": f"P{p:02d}", "answer": n, "condition": cond,
                "signals": {"ai_text": {"ai_probability": ai, "level": "low"},
                            "echo_similarity": {"similarity": sim, "level": "low"},
                            "delivery": {"score": {"index": None if baseline else idx, "level": "baseline" if baseline else "low"}},
                            "reading": {"reading_probability": rd, "level": "low"}},
                "flag": {"level": "possible" if cond != "honest" else "low"}}), encoding="utf-8")


def test_calibration_splits_people_and_proposes_thresholds(tmp_path):
    write_sessions(tmp_path)
    rows = calibrate.load_sessions(tmp_path)
    assert len(rows) == 180 and {r["participant"] for r in rows} == {f"P{p:02d}" for p in range(12)}
    result = calibrate.calibrate(rows)
    assert result["participants"] == {"development": 8, "test": 4}
    for name in ("ai_text", "echo_similarity", "delivery", "reading"):
        sig = result["signals"][name]
        assert sig["status"] == "ok", (name, sig)
        assert sig["proposed"]["possible"] <= sig["proposed"]["high"]
        assert sig["test"]["roc_auc"] > 0.8
        assert sig["development"]["at_high"]["false_positive_rate"] <= 0.05 + 1e-9
    # delivery leaves the baseline answers and paraphrases out; content keeps read_script as a negative
    d = result["signals"]["delivery"]["answers"]
    assert d["development"]["negative"] == 8 * 3 and d["development"]["positive"] == 8 * 6
    assert result["signals"]["ai_text"]["negatives"] == ["honest", "read_script"]
    assert result["flag_as_run_test_people"]["honest"]["possible_or_high"] == 0.0


def test_thresholds_come_from_development_people_only(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_sessions(a)
    write_sessions(b)
    rows_a, rows_b = calibrate.load_sessions(a), calibrate.load_sessions(b)
    _, test = calibrate.split_people([r["participant"] for r in rows_a], 0.3)
    for r in rows_b:  # change only the test people's scores
        if r["participant"] in test:
            r["signals"]["ai_text"]["ai_probability"] += 0.3
    ra, rb = calibrate.calibrate(rows_a), calibrate.calibrate(rows_b)
    assert ra["signals"]["ai_text"]["proposed"] == rb["signals"]["ai_text"]["proposed"]
    assert ra["signals"]["ai_text"]["test"] != rb["signals"]["ai_text"]["test"]


def test_label_sheet_corrects_and_excludes(tmp_path):
    write_sessions(tmp_path, people=2)
    rows = calibrate.load_sessions(tmp_path)
    sheet = tmp_path / "labels.csv"
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=randomise.COLUMNS)
        w.writeheader()
        w.writerow({**{c: "" for c in randomise.COLUMNS}, "session": "00000000", "answer_no": "4", "condition": "read_ai", "exclude": "no"})
        w.writerow({**{c: "" for c in randomise.COLUMNS}, "session": "00000000", "answer_no": "5", "exclude": "yes"})
    fixed = calibrate.apply_labels(rows, sheet)
    assert len(fixed) == len(rows) - 1
    assert next(r for r in fixed if r["session"] == "00000000" and r["answer"] == 4)["condition"] == "read_ai"


def test_too_little_data_is_reported_not_guessed(tmp_path):
    write_sessions(tmp_path, people=3)
    result = calibrate.calibrate(calibrate.load_sessions(tmp_path))
    assert all(s["status"] == "not enough data" for s in result["signals"].values())
    assert calibrate.calibrate([])["status"] == "not enough people"


def test_metrics():
    y = np.array([True, True, False, False])
    s = np.array([0.9, 0.6, 0.6, 0.1])
    assert calibrate.auc(y, s) == pytest.approx(0.875)
    assert calibrate.at(y, s, 0.6) == {"recall": 1.0, "false_positive_rate": 0.5, "balanced_accuracy": 0.75}
    assert calibrate.threshold_at_fpr(y, s, 0.0) > 0.6
