"""The AI-written-text model, v1 against v2, on the same HC3 test questions.

    python benchmarks/bench_ai_text.py --data M:/openmeet-data/hc3/all.jsonl \
        --v1 M:/openmeet-work/tmp-t04/echo_tfidf_v1.json

v1 is `local-ai/integrity/models/echo_tfidf.json` at commit 0836f1e (the model
the first HC3 benchmark used); v2 is the file in the working tree (commit
0ca0fad and later, retrained on HC3 plus interview Q&A). Both are scored by the
sidecar's own `EchoModel` (local-ai/integrity/models.py), each at its own
stored thresholds: `possible` is the lower of its two thresholds and `high`
the higher, as the sidecar uses them.

The test questions are exactly those of training/echo-detection/train_content.py
(its `load_hc3`, `split_questions` and seed 2026; 20% of questions are test).
Neither model's training data was scored here. The stored metrics each model
file carries (its `test_questions`, computed by its training script) are read
back and compared with the recomputed ones, to show that the split is the
same one. Balanced accuracy, ROC-AUC, detection rate (recall) and false-flag
rate are reported overall, per HC3 source, and for the interview-style source
wiki_csai.

Writes results/ai_text_v1_vs_v2.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

ECHO_DIR = benchlib.REPO / "training" / "echo-detection"
sys.path.insert(0, str(ECHO_DIR))
import train_content  # noqa: E402  (the split and loader used for the models' training)


def load_echo_model(path: Path):
    """The sidecar's EchoModel (local-ai/integrity/models.py) on any model file."""
    import importlib
    import types

    if "om_integrity" not in sys.modules:
        package = types.ModuleType("om_integrity")
        package.__path__ = [str(benchlib.LOCAL_AI / "integrity")]
        sys.modules["om_integrity"] = package
    return importlib.import_module("om_integrity.models").EchoModel(path)


def scores_at(labels: np.ndarray, flags: np.ndarray) -> dict:
    tp = int((flags & labels).sum())
    fp = int((flags & ~labels).sum())
    n_ai, n_h = int(labels.sum()), int((~labels).sum())
    recall = tp / n_ai if n_ai else float("nan")
    fpr = fp / n_h if n_h else float("nan")
    return {"detection_rate": recall, "false_flag_rate": fpr, "balanced_accuracy": (recall + (1 - fpr)) / 2,
            "detected": tp, "of_ai": n_ai, "false_flags": fp, "of_human": n_h,
            "ci_false_flag_95": benchlib.wilson(fp, n_h), "ci_detection_95": benchlib.wilson(tp, n_ai)}


def evaluate(labels: np.ndarray, probs: np.ndarray, possible: float, high: float) -> dict:
    out = {"roc_auc": float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else None,
           "possible": possible, "high": high,
           "at_possible": scores_at(labels, probs >= possible),
           "at_high": scores_at(labels, probs >= high)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="HC3 all.jsonl")
    ap.add_argument("--v1", type=Path, required=True, help="v1 model file (git show 0836f1e:local-ai/integrity/models/echo_tfidf.json)")
    ap.add_argument("--v2", type=Path, default=benchlib.LOCAL_AI / "integrity" / "models" / "echo_tfidf.json")
    ap.add_argument("--out", type=Path, default=benchlib.RESULTS / "ai_text_v1_vs_v2.json")
    a = ap.parse_args()

    started = time.time()
    table = train_content.load_hc3(a.data)
    test_q = train_content.split_questions(table["qid"], 0.2, train_content.SEED)
    test = table[table["qid"].isin(test_q)].reset_index(drop=True)
    labels = test["label"].to_numpy().astype(bool)
    sources = test["source"].to_numpy()
    print(f"HC3 test: {len(test)} answers ({int(labels.sum())} ChatGPT, {int((~labels).sum())} human) "
          f"from {len(test_q)} questions", flush=True)

    result = {
        "what": "the AI-written-text model (sidecar EchoModel), v1 (commit 0836f1e) against v2 (working tree), "
                "scored on the HC3 test questions of train_content.py",
        "environment": benchlib.environment(["scikit-learn", "numpy"]),
        "data": {"file": "HC3 all.jsonl (CC BY-SA 4.0)", "split": "train_content.py: split_questions(qid, 0.2, seed 2026), "
                 "answers as transcripts, at least 15 words, duplicates dropped",
                 "test_answers": len(test), "test_questions": len(test_q),
                 "chatgpt_answers": int(labels.sum()), "human_answers": int((~labels).sum())},
        "models": {},
        "by_source": {},
        "stored_test_metrics": {},
    }
    per_model = {}
    for name, path in (("v1", a.v1), ("v2", a.v2)):
        model = load_echo_model(path)
        probs = np.array([model.probability(t) for t in test["text"]], dtype=float)
        per_model[name] = probs
        result["models"][name] = {"file": str(path.name), "name": model.name,
                                  "source": "0836f1e" if name == "v1" else "working tree (0ca0fad or later)",
                                  **evaluate(labels, probs, model.possible, model.high)}
        stored = model.test
        result["stored_test_metrics"][name] = {k: v for k, v in stored.items()} if stored else None
        print(f"{name}: AUC {result['models'][name]['roc_auc']:.4f}", flush=True)

    for src in sorted(set(sources)):
        m = sources == src
        row = {"chatgpt_answers": int(labels[m].sum()), "human_answers": int((~labels[m]).sum())}
        for name in ("v1", "v2"):
            model = result["models"][name]
            p = per_model[name][m]
            row[name] = {"roc_auc": float(roc_auc_score(labels[m], p)) if 0 < labels[m].sum() < m.sum() else None,
                         "at_possible": scores_at(labels[m], p >= model["possible"]),
                         "at_high": scores_at(labels[m], p >= model["high"])}
        result["by_source"][src] = row

    benchlib.write_json(a.out, result)
    for name in ("v1", "v2"):
        m = result["models"][name]
        print(f"{name}: balanced acc at high {m['at_high']['balanced_accuracy']:.4f}; false flags possible "
              f"{m['at_possible']['false_flag_rate']:.4f}, high {m['at_high']['false_flag_rate']:.4f}; "
              f"detected possible {m['at_possible']['detection_rate']:.4f}, high {m['at_high']['detection_rate']:.4f}")
    print(f"-> {a.out} [{round(time.time() - started)} s]")


if __name__ == "__main__":
    main()
