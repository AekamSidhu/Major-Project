"""Retrain the echo detector on HC3 + interview data, export for the sidecar.

Run on Kaggle (CPU is fine — TF-IDF trains in seconds):

    python retrain_with_interviews.py

Or locally:

    python retrain_with_interviews.py --data data/hc3_all.jsonl --interview data/interview_qa.jsonl --out ../../local-ai/integrity/models

This script:
  1. Downloads HC3 (if not cached) and loads the interview JSONL
  2. Combines them — interview questions get ids offset by 100 000
  3. Trains TF-IDF + logistic regression on the combined data
  4. Evaluates on HC3 test set (unseen questions) AND held-out interview data
  5. Exports echo_tfidf.json ready for the sidecar

The interview split holds back 30% of interview questions for calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content.text import to_transcript  # noqa: E402

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
SEED = 2026
QID_OFFSET = 100_000
TOKEN = re.compile(r"(?u)\b\w\w+\b")


# --- data -----------------------------------------------------------------------
def load_hc3(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading HC3 to {path} ...", flush=True)
        urllib.request.urlretrieve(HC3_URL, path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for qid, line in enumerate(f):
            item = json.loads(line)
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for answer in item[key] or []:
                    if isinstance(answer, str) and answer.strip():
                        rows.append({"qid": qid, "source": item["source"],
                                     "question": item["question"], "label": label, "raw": answer})
    table = pd.DataFrame(rows)
    table["text"] = table["raw"].map(to_transcript)
    table = table[table["text"].str.split().str.len() >= 15]
    return table.drop_duplicates("text").reset_index(drop=True)


def load_interview(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for qid, line in enumerate(f):
            item = json.loads(line)
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for answer in item[key] or []:
                    if isinstance(answer, str) and answer.strip():
                        rows.append({"qid": QID_OFFSET + qid, "source": item.get("source", "interview"),
                                     "question": item["question"], "label": label, "raw": answer})
    table = pd.DataFrame(rows)
    table["text"] = table["raw"].map(to_transcript)
    table = table[table["text"].str.split().str.len() >= 15]
    return table.drop_duplicates("text").reset_index(drop=True)


def split_questions(qids, fraction: float, seed: int) -> set:
    unique = np.array(sorted(set(qids)))
    order = np.random.default_rng(seed).permutation(len(unique))
    return set(unique[order[: int(round(fraction * len(unique)))]].tolist())


# --- metrics --------------------------------------------------------------------
def best_threshold(y, s) -> float:
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    tpr = np.cumsum(yy) / max(yy.sum(), 1)
    fpr = np.cumsum(~yy) / max((~yy).sum(), 1)
    last = np.r_[ss[1:] != ss[:-1], True]
    return float(ss[int(np.argmax(np.where(last, (tpr + 1 - fpr) / 2, -1)))])


def threshold_at_fpr(y, s, target=0.05) -> float:
    neg = np.sort(np.asarray(s, dtype=float)[~np.asarray(y).astype(bool)])[::-1]
    k = int(np.floor(target * len(neg)))
    return float(neg[-1]) if k >= len(neg) else float(np.nextafter(neg[k], np.inf))


def metrics(y, s, cut) -> dict:
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    flag = s >= cut
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    rec = tp / max(y.sum(), 1)
    spec = 1 - fp / max((~y).sum(), 1)
    return {"n_ai": int(y.sum()), "n_human": int((~y).sum()), "threshold": float(cut),
            "balanced_accuracy": (rec + spec) / 2, "recall": rec,
            "false_positive_rate": 1 - spec,
            "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")}


def report(name, y, s, cuts) -> dict:
    out = {k: metrics(y, s, c) for k, c in cuts.items()}
    b = out["balanced"]
    print(f"  {name:45s} AUC {b['roc_auc']:.4f}  BA {100*b['balanced_accuracy']:.1f}%  "
          f"recall {100*b['recall']:.1f}%  FPR {100*b['false_positive_rate']:.1f}%", flush=True)
    return out


# --- plain-python scorer (parity check) ----------------------------------------
def score_like_sidecar(model: dict, text: str) -> float:
    words = TOKEN.findall(to_transcript(text))
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    counts: dict[int, int] = {}
    for g in grams:
        j = model["index"].get(g)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    weights = {j: (1.0 + math.log(c)) * model["idf"][j] for j, c in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    z = model["intercept"] + sum(w / norm * model["coef"][j] for j, w in weights.items())
    return 1.0 / (1.0 + math.exp(-z))


# --- main -----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "hc3_all.jsonl")
    ap.add_argument("--interview", type=Path, default=Path(__file__).parent / "data" / "interview_qa.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument("--max-features", type=int, default=60_000)
    ap.add_argument("--interview-holdout", type=float, default=0.30, help="fraction of interview Qs held out for calibration")
    a = ap.parse_args()

    # --- Generate interview JSONL if missing ---
    if not a.interview.exists():
        print("Interview JSONL not found — generating from interview_data.py ...", flush=True)
        from interview_data import QUESTIONS, to_hc3_jsonl
        a.interview.parent.mkdir(parents=True, exist_ok=True)
        lines = to_hc3_jsonl(QUESTIONS)
        a.interview.write_text("\n".join(lines), encoding="utf-8")
        print(f"  wrote {len(QUESTIONS)} questions to {a.interview}")

    # --- Load data ---
    hc3 = load_hc3(a.data)
    interview = load_interview(a.interview)
    n_hc3_q = hc3["qid"].nunique()
    n_int_q = interview["qid"].nunique()
    print(f"\nHC3: {len(hc3)} answers from {n_hc3_q} questions")
    print(f"Interview: {len(interview)} answers from {n_int_q} questions")

    # --- Split ---
    # HC3: 80% dev / 20% test (same as original training)
    hc3_test_q = split_questions(hc3["qid"], 0.2, SEED)
    hc3_dev = hc3[~hc3["qid"].isin(hc3_test_q)]
    hc3_val_q = split_questions(hc3_dev["qid"], 0.15, SEED + 1)
    hc3_train = hc3_dev[~hc3_dev["qid"].isin(hc3_val_q)]
    hc3_val = hc3_dev[hc3_dev["qid"].isin(hc3_val_q)]
    hc3_test = hc3[hc3["qid"].isin(hc3_test_q)]

    # Interview: holdout% for calibration, rest goes to training
    int_holdout_q = split_questions(interview["qid"], a.interview_holdout, SEED + 2)
    int_train = interview[~interview["qid"].isin(int_holdout_q)]
    int_cal = interview[interview["qid"].isin(int_holdout_q)]
    print(f"Interview train: {len(int_train)} answers ({int_train['qid'].nunique()} Qs)")
    print(f"Interview calibration: {len(int_cal)} answers ({int_cal['qid'].nunique()} Qs)")

    # Combined training set
    train = pd.concat([hc3_train, int_train], ignore_index=True)
    print(f"\nCombined training: {len(train)} answers ({train['qid'].nunique()} Qs)")
    print(f"  HC3 share: {len(hc3_train)/len(train):.1%}, interview share: {len(int_train)/len(train):.1%}\n")

    # --- Train TF-IDF ---
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=a.max_features, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    clf.fit(vec.fit_transform(train["text"]), train["label"])

    # --- Evaluate ---
    # Thresholds from HC3 validation (same as original — keeps comparison fair)
    val_s = clf.predict_proba(vec.transform(hc3_val["text"]))[:, 1]
    yv = hc3_val["label"].to_numpy()
    cuts = {"balanced": best_threshold(yv, val_s), "low_false_positive": threshold_at_fpr(yv, val_s)}

    print("=" * 70)
    print(f"Thresholds from HC3 validation: balanced={cuts['balanced']:.4f}, low_fp={cuts['low_false_positive']:.4f}")
    print("=" * 70)

    result = {"thresholds": cuts}

    # HC3 validation
    print("\n--- HC3 validation ---")
    result["hc3_val"] = report("HC3 validation (all)", yv, val_s, cuts)

    # HC3 test (unseen questions)
    print("\n--- HC3 test (unseen questions) ---")
    test_s = clf.predict_proba(vec.transform(hc3_test["text"]))[:, 1]
    yt = hc3_test["label"].to_numpy()
    result["hc3_test"] = report("HC3 test", yt, test_s, cuts)

    # Interview calibration (held-out interview questions)
    print("\n--- Interview calibration (held-out questions) ---")
    if len(int_cal) > 0:
        cal_s = clf.predict_proba(vec.transform(int_cal["text"]))[:, 1]
        yc = int_cal["label"].to_numpy()
        result["interview_cal"] = report("Interview calibration", yc, cal_s, cuts)

        # Per-question detail
        print("\n  Per-question detail (calibration set):")
        for qid in sorted(int_cal["qid"].unique()):
            part = int_cal[int_cal["qid"] == qid]
            q = part["question"].iloc[0][:60]
            scores = clf.predict_proba(vec.transform(part["text"]))[:, 1]
            human_s = scores[part["label"].to_numpy() == 0]
            ai_s = scores[part["label"].to_numpy() == 1]
            h_mean = f"{human_s.mean():.3f}" if len(human_s) else "n/a"
            a_mean = f"{ai_s.mean():.3f}" if len(ai_s) else "n/a"
            print(f"    {q:60s}  human={h_mean}  AI={a_mean}")

        # Interview-optimised thresholds
        int_cuts = {"balanced": best_threshold(yc, cal_s), "low_false_positive": threshold_at_fpr(yc, cal_s, target=0.0)}
        print(f"\n  Interview-optimised thresholds: balanced={int_cuts['balanced']:.4f}, low_fp={int_cuts['low_false_positive']:.4f}")
        result["interview_thresholds"] = int_cuts
        result["interview_cal_at_int_thresh"] = report("Interview (own thresholds)", yc, cal_s, int_cuts)
        # Also check HC3 test at interview thresholds
        result["hc3_test_at_int_thresh"] = report("HC3 test (interview thresholds)", yt, test_s, int_cuts)

    # --- Export ---
    a.out.mkdir(parents=True, exist_ok=True)
    terms = vec.get_feature_names_out()
    model = {
        "model": f"OpenMeet AI-answer detector v2: TF-IDF (1-2 word grams, {len(terms)} terms) + logistic regression",
        "trained_on": "HC3 (CC BY-SA 4.0) + interview Q&A (GPT-5/GPT-4.1/Claude 4.0 Sonnet style); as transcripts; train questions only",
        "normalisation": "lower case, words only (content/text.py to_transcript), then scikit-learn's default token pattern",
        "thresholds": cuts,
        "interview_calibration": {
            "thresholds": result.get("interview_thresholds", {}),
            "note": "Interview-optimised thresholds for use alongside HC3-derived ones",
        },
        "validation": result.get("hc3_val", {}),
        "test_questions": result.get("hc3_test", {}),
        "intercept": float(clf.intercept_[0]),
        "terms": [[str(t), round(float(i), 5), round(float(c), 6)] for t, i, c in zip(terms, vec.idf_, clf.coef_[0])],
    }

    # Parity check
    check = {"index": {t: j for j, (t, _, _) in enumerate(model["terms"])},
             "idf": [t[1] for t in model["terms"]],
             "coef": [t[2] for t in model["terms"]], "intercept": model["intercept"]}
    sample = hc3_test.sample(min(300, len(hc3_test)), random_state=0)
    ours = np.array([score_like_sidecar(check, t) for t in sample["raw"]])
    theirs = clf.predict_proba(vec.transform(sample["text"]))[:, 1]
    gap = float(np.abs(ours - theirs).max())
    assert gap < 1e-3, f"Parity check failed: max gap = {gap}"
    model["parity_max_abs_diff"] = gap

    path = a.out / "echo_tfidf.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.1f} MB), parity gap {gap:.1e}")

    # Also save full results
    rpath = a.out / "retrain_results.json"
    result_clean = {k: v for k, v in result.items() if k != "terms"}
    rpath.write_text(json.dumps(result_clean, indent=1, default=float))
    print(f"Full results: {rpath}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    hc3_ba = result["hc3_test"]["balanced"]["balanced_accuracy"]
    hc3_auc = result["hc3_test"]["balanced"]["roc_auc"]
    print(f"HC3 test:        BA {100*hc3_ba:.1f}%  AUC {hc3_auc:.4f}")
    if "interview_cal" in result:
        int_ba = result["interview_cal"]["balanced"]["balanced_accuracy"]
        int_rec = result["interview_cal"]["balanced"]["recall"]
        int_fpr = result["interview_cal"]["balanced"]["false_positive_rate"]
        print(f"Interview cal:   BA {100*int_ba:.1f}%  recall {100*int_rec:.1f}%  FPR {100*int_fpr:.1f}%")
        if "interview_cal_at_int_thresh" in result:
            ir = result["interview_cal_at_int_thresh"]["balanced"]
            print(f"Interview (own): BA {100*ir['balanced_accuracy']:.1f}%  recall {100*ir['recall']:.1f}%  FPR {100*ir['false_positive_rate']:.1f}%")
    print(f"\nTo deploy: copy {path} to local-ai/integrity/models/echo_tfidf.json")


if __name__ == "__main__":
    main()
