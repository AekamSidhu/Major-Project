"""Export the TF-IDF + logistic-regression AI-answer detector for the OpenMeet sidecar.

    python export_content_model.py --data data/hc3_all.jsonl --out ../../local-ai/integrity/models

Same data, transcript normalisation and question split as train_content.py:
fitted on the train questions, threshold from the validation questions,
scored once on the test questions. Writes `echo_tfidf.json`: every term with
its idf and weight, so the sidecar scores answers with plain Python and no
scikit-learn. A parity check against scikit-learn runs before writing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content.text import to_transcript  # noqa: E402
from train_content import SEED, best_threshold, load_hc3, metrics, split_questions, threshold_at_fpr  # noqa: E402

TOKEN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token pattern


def score_like_sidecar(model: dict, text: str) -> float:
    """Reference implementation the sidecar copies: transcript -> tf-idf -> logistic."""
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


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "hc3_all.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-features", type=int, default=60_000)
    a = ap.parse_args()

    table = load_hc3(a.data)
    test_q = split_questions(table["qid"], 0.2, SEED)
    dev = table[~table["qid"].isin(test_q)]
    val_q = split_questions(dev["qid"], 0.15, SEED + 1)
    train, val, test = dev[~dev["qid"].isin(val_q)], dev[dev["qid"].isin(val_q)], table[table["qid"].isin(test_q)]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=a.max_features, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    clf.fit(vec.fit_transform(train["text"]), train["label"])
    val_s = clf.predict_proba(vec.transform(val["text"]))[:, 1]
    yv = val["label"].to_numpy()
    cuts = {"balanced": best_threshold(yv, val_s), "low_false_positive": threshold_at_fpr(yv, val_s)}
    test_s = clf.predict_proba(vec.transform(test["text"]))[:, 1]
    yt = test["label"].to_numpy()
    val_m, test_m = metrics(yv, val_s, cuts["balanced"]), metrics(yt, test_s, cuts["balanced"])
    print(f"{a.max_features} features: validation balanced acc {100 * val_m['balanced_accuracy']:.2f}% (AUC {val_m['roc_auc']:.4f}); "
          f"test {100 * test_m['balanced_accuracy']:.2f}% (AUC {test_m['roc_auc']:.4f})", flush=True)

    terms = vec.get_feature_names_out()
    model = {
        "model": f"OpenMeet AI-answer detector v1: TF-IDF (1-2 word grams, {len(terms)} terms) + logistic regression",
        "trained_on": "HC3 (CC BY-SA 4.0), human vs ChatGPT answers, as transcripts; train questions only",
        "normalisation": "lower case, words only (content/text.py to_transcript), then scikit-learn's default token pattern",
        "thresholds": cuts,
        "validation": val_m,
        "test_questions": test_m,
        "intercept": float(clf.intercept_[0]),
        "terms": [[str(t), round(float(i), 5), round(float(c), 6)] for t, i, c in zip(terms, vec.idf_, clf.coef_[0])],
    }
    # Parity: the plain-Python scorer must match scikit-learn.
    check = {"index": {t: j for j, (t, _, _) in enumerate(model["terms"])}, "idf": [t[1] for t in model["terms"]],
             "coef": [t[2] for t in model["terms"]], "intercept": model["intercept"]}
    sample = test.sample(300, random_state=0)
    ours = np.array([score_like_sidecar(check, t) for t in sample["raw"]])
    theirs = clf.predict_proba(vec.transform(sample["text"]))[:, 1]
    gap = float(np.abs(ours - theirs).max())
    assert gap < 1e-3, gap
    model["parity_max_abs_diff"] = gap
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "echo_tfidf.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), parity gap {gap:.1e}")


if __name__ == "__main__":
    main()
