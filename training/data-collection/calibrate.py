"""Thresholds for each integrity signal from study sessions, split by person.

    python calibrate.py D:/openmeet-study --labels sheets/all.csv --out results/own_data_thresholds.json

Reads the sessions the room saved in study mode (local-ai/room/study.py:
session-<id>/answer-NNN.json) and, optionally, the label sheets
(randomise.py), which can correct an answer's condition or exclude it
(matched on the room's session id and answer number).

For each signal, the conditions that should raise it are the positives:

| Signal | Positives | Negatives | Left out |
|---|---|---|---|
| ai_text, echo_similarity | read_ai, paraphrase_ai | honest, read_script | |
| delivery, reading | read_script, read_ai | honest | paraphrase_ai (partly read) |

Protocol: people are split once (seed 2026) into development (70%) and test
(30%); no person is on both sides. On development people, `possible` = the
lower and `high` = the higher of the balanced-accuracy cut and the cut with
a 5% false-positive rate; test people are scored once. A signal with fewer
than 10 positive or negative answers on either side is reported as "not
enough data". The output is a proposal: review it, then edit
spec/integrity/thresholds.json by hand and rerun spec/integrity/reference.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

SEED = 2026
FPR_TARGET = 0.05
MIN_PER_CLASS = 10
SIGNALS: dict[str, dict[str, Any]] = {
    "ai_text": {"score": ("ai_text", "ai_probability"), "positive": {"read_ai", "paraphrase_ai"}, "negative": {"honest", "read_script"}},
    "echo_similarity": {"score": ("echo_similarity", "similarity"), "positive": {"read_ai", "paraphrase_ai"},
                        "negative": {"honest", "read_script"}},
    "delivery": {"score": ("delivery", "score", "index"), "positive": {"read_script", "read_ai"}, "negative": {"honest"}},
    "reading": {"score": ("reading", "reading_probability"), "positive": {"read_script", "read_ai"}, "negative": {"honest"}},
}


# --- loading ------------------------------------------------------------------------------


def load_sessions(folder: Path) -> list[dict[str, Any]]:
    """One row per saved answer: session, participant, answer number, condition, scores, flag."""
    rows = []
    for session_dir in sorted(p for p in Path(folder).glob("session-*") if p.is_dir()):
        meta_path = session_dir / "session.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        for path in sorted(session_dir.glob("answer-*.json")):
            a = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"session": a["session"], "participant": a.get("participant") or meta.get("participant") or a["session"],
                         "answer": int(a["answer"]), "condition": a.get("condition"), "signals": a.get("signals") or {},
                         "flag": (a.get("flag") or {}).get("level"),
                         "reference_model": (meta.get("models") or {}).get("reference_model")})
    return rows


def apply_labels(rows: list[dict[str, Any]], sheet: Path) -> list[dict[str, Any]]:
    """The label sheet corrects conditions and excludes answers (matched on session and answer number)."""
    with sheet.open(encoding="utf-8", newline="") as f:
        labels = {(r["session"].strip(), int(r["answer_no"])): r for r in csv.DictReader(f) if r.get("session", "").strip()}
    out = []
    for row in rows:
        label = labels.get((row["session"], row["answer"]))
        if label is not None:
            if label.get("exclude", "").strip().lower() in ("yes", "y", "true", "1"):
                continue
            if label.get("condition", "").strip():
                row = {**row, "condition": label["condition"].strip()}
        out.append(row)
    return out


def score_of(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row["signals"]
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


# --- thresholds and metrics -----------------------------------------------------------------


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """The cut (flag when score >= cut) with the best balanced accuracy."""
    cuts = np.unique(s)
    scores = [((s[y] >= c).mean() + (s[~y] < c).mean()) / 2 for c in cuts]
    return float(cuts[int(np.argmax(scores))])


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target: float = FPR_TARGET) -> float:
    """The lowest cut that flags at most `target` of the negatives."""
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(target * len(neg)))
    return float(neg[-1]) if k >= len(neg) else float(np.nextafter(neg[k], np.inf))


def at(y: np.ndarray, s: np.ndarray, cut: float) -> dict[str, float]:
    flag = s >= cut
    recall, fpr = float(flag[y].mean()), float(flag[~y].mean())
    return {"recall": recall, "false_positive_rate": fpr, "balanced_accuracy": (recall + 1 - fpr) / 2}


def split_people(people: list[str], test_fraction: float, seed: int = SEED) -> tuple[set[str], set[str]]:
    people = sorted(set(people))
    order = np.random.default_rng(seed).permutation(len(people))
    n_test = max(1, int(round(test_fraction * len(people))))
    test = {people[i] for i in order[:n_test]}
    return set(people) - test, test


def calibrate_signal(rows: list[dict[str, Any]], spec: dict[str, Any], dev: set[str], test: set[str]) -> dict[str, Any]:
    usable = [(r["participant"], r["condition"] in spec["positive"], score_of(r, spec["score"])) for r in rows
              if r["condition"] in spec["positive"] | spec["negative"]]
    usable = [u for u in usable if u[2] is not None]

    def part(people: set[str]) -> tuple[np.ndarray, np.ndarray]:
        sel = [u for u in usable if u[0] in people]
        return np.array([u[1] for u in sel], dtype=bool), np.array([u[2] for u in sel], dtype=float)

    yd, sd = part(dev)
    yt, st = part(test)
    sizes = {"development": {"positive": int(yd.sum()), "negative": int((~yd).sum())},
             "test": {"positive": int(yt.sum()), "negative": int((~yt).sum())}}
    if min(sizes["development"].values()) < MIN_PER_CLASS or min(sizes["test"].values()) < MIN_PER_CLASS:
        return {"status": "not enough data", "answers": sizes}
    cuts = {"balanced": best_threshold(yd, sd), "fpr_5": threshold_at_fpr(yd, sd)}
    possible, high = min(cuts.values()), max(cuts.values())
    return {"status": "ok", "answers": sizes, "positives": sorted(spec["positive"]), "negatives": sorted(spec["negative"]),
            "proposed": {"possible": possible, "high": high},
            "development": {"roc_auc": auc(yd, sd), "at_possible": at(yd, sd, possible), "at_high": at(yd, sd, high)},
            "test": {"roc_auc": auc(yt, st), "at_possible": at(yt, st, possible), "at_high": at(yt, st, high)}}


def flag_rates(rows: list[dict[str, Any]], people: set[str]) -> dict[str, dict[str, float]]:
    """The combined flag as it ran in the sessions, per condition."""
    out = {}
    for cond in sorted({r["condition"] for r in rows if r["condition"]}):
        mine = [r for r in rows if r["condition"] == cond and r["participant"] in people and r["flag"] in ("low", "possible", "high")]
        if mine:
            out[cond] = {"answers": len(mine), "possible_or_high": float(np.mean([r["flag"] in ("possible", "high") for r in mine])),
                         "high": float(np.mean([r["flag"] == "high" for r in mine]))}
    return out


def calibrate(rows: list[dict[str, Any]], test_fraction: float = 0.3, seed: int = SEED) -> dict[str, Any]:
    rows = [r for r in rows if r["condition"]]
    people = sorted({r["participant"] for r in rows})
    if len(people) < 2:
        return {"status": "not enough people", "participants": len(people)}
    dev, test = split_people(people, test_fraction, seed)
    return {"participants": {"development": len(dev), "test": len(test)}, "answers": len(rows),
            "reference_models": sorted({str(r["reference_model"]) for r in rows}),
            "signals": {name: calibrate_signal(rows, spec, dev, test) for name, spec in SIGNALS.items()},
            "flag_as_run_test_people": flag_rates(rows, test)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", type=Path, help="the room's --study folder")
    ap.add_argument("--labels", type=Path, help="label sheet CSV (randomise.py's all.csv, filled in)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    rows = load_sessions(a.study)
    if a.labels:
        rows = apply_labels(rows, a.labels)
    result = {"what": "proposed integrity thresholds from own study sessions (split by person); a proposal, not applied",
              "protocol": {"seed": a.seed, "test_fraction": a.test_fraction, "fpr_target": FPR_TARGET, "min_per_class": MIN_PER_CLASS},
              **calibrate(rows, a.test_fraction, a.seed)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name, r in result.get("signals", {}).items():
        if r["status"] != "ok":
            print(f"{name:16s} {r['status']} {r['answers']}")
        else:
            t = r["test"]
            print(f"{name:16s} possible {r['proposed']['possible']:.4g} high {r['proposed']['high']:.4g} | test AUC {t['roc_auc']:.3f}, "
                  f"FPR {t['at_possible']['false_positive_rate']:.2f} / {t['at_high']['false_positive_rate']:.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
