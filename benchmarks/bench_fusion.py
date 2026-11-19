"""The combined integrity flag on HC3 test questions: detection rate and false flags, with an ablation.

    python benchmarks/bench_fusion.py --data M:/openmeet-data/hc3/all.jsonl --cache M:/openmeet-data/hc3/cache

HC3 has two of the four signals: the transcript text (`ai_text`, the
sidecar's TF-IDF model, local-ai/integrity/models/echo_tfidf.json) and echo
similarity (the answer against llama3.2's reference answers, bge-small
embeddings, thresholds in spec/integrity/thresholds.json). Reading and
delivery need video and audio, which HC3 does not have.

Protocol (no choice is made on the data scored here):
- HC3 loaded and split exactly as training/echo-detection/train_content.py
  does (answers as transcripts, at least 15 words, seed 2026, 20% of
  questions are test). The AI-text model was fitted on train questions and
  its thresholds chosen on validation questions; the echo thresholds were
  chosen on 500 development questions. Only test questions are scored here.
- The same 500 test questions as calibrate_similarity.py (100 per source,
  questions with both a human and a ChatGPT answer), with its cached
  reference answers (2 per question; nothing is generated: a missing
  reference is an error).
- Each answer gets the two signals' levels and the spec's fused flag
  (local-ai/integrity/signals.py `fuse`). A ChatGPT answer flagged is a
  detection; a human answer flagged is a false flag.
- Ablation: each signal alone, the fused flag, the fused flag when only one
  signal is available (as in rules mode, where echo similarity is
  unavailable), and echo similarity with one reference answer instead of two
  (at the thresholds measured for two).

Writes results/fusion_hc3.json and results/fusion_hc3_scores.csv (scores only, no text).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

ECHO_DIR = benchlib.REPO / "training" / "echo-detection"
LEVELS = ("possible", "high")


def load_calibration():
    """training/echo-detection/calibrate_similarity.py as a module (its split, sampling and cache)."""
    spec = importlib.util.spec_from_file_location("calibrate_similarity", ECHO_DIR / "calibrate_similarity.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_echo_model():
    """The sidecar's EchoModel (local-ai/integrity/models.py) without importing the sidecar's web packages."""
    if "om_integrity" not in sys.modules:
        package = types.ModuleType("om_integrity")
        package.__path__ = [str(benchlib.LOCAL_AI / "integrity")]
        sys.modules["om_integrity"] = package
    return importlib.import_module("om_integrity.models").EchoModel()


def level(value: float, possible: float, high: float) -> str:
    return "high" if value >= high else "possible" if value >= possible else "low"


# --- evaluation (pure functions, covered by the smoke tests) -------------------------


def flags(rows: list[dict[str, Any]], fuse: Callable[[list[dict]], dict], calibrated: dict[str, bool]) -> dict[str, list[str]]:
    """Per configuration, the flag level of every row."""
    def fused(names: list[str], echo_key: str = "echo_level") -> list[str]:
        out = []
        for r in rows:
            sig = []
            if "ai_text" in names:
                sig.append({"name": "ai_text", "level": r["ai_level"], "calibrated": calibrated["ai_text"]})
            if "echo_similarity" in names:
                sig.append({"name": "echo_similarity", "level": r[echo_key], "calibrated": calibrated["echo_similarity"]})
            out.append(fuse(sig)["level"])
        return out

    return {
        "ai_text alone": [r["ai_level"] for r in rows],
        "echo similarity alone": [r["echo_level"] for r in rows],
        "fused (ai_text + echo)": fused(["ai_text", "echo_similarity"]),
        "fused, ai_text only available": fused(["ai_text"]),
        "fused, echo only available": fused(["echo_similarity"]),
        "echo similarity alone, 1 reference": [r["echo_level_1ref"] for r in rows],
        "fused (ai_text + echo, 1 reference)": fused(["ai_text", "echo_similarity"], "echo_level_1ref"),
    }


def rates(labels: list[int], levels: list[str]) -> dict[str, Any]:
    """Detection rate (label 1 flagged) and false-flag rate (label 0 flagged) at >= possible and at high."""
    y = np.asarray(labels)
    out: dict[str, Any] = {"chatgpt_answers": int((y == 1).sum()), "human_answers": int((y == 0).sum())}
    for cut in LEVELS:
        hit = np.array([lv == "high" or (cut == "possible" and lv == "possible") for lv in levels])
        tp, fp = int((hit & (y == 1)).sum()), int((hit & (y == 0)).sum())
        out[f"at_{cut}"] = {
            "detection_rate": benchlib.rate(tp, int((y == 1).sum())), "detection_95ci": benchlib.wilson(tp, int((y == 1).sum())),
            "false_flag_rate": benchlib.rate(fp, int((y == 0).sum())), "false_flag_95ci": benchlib.wilson(fp, int((y == 0).sum())),
            "detected": tp, "false_flags": fp,
        }
    return out


def evaluate(rows: list[dict[str, Any]], fuse: Callable[[list[dict]], dict], calibrated: dict[str, bool]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    per_config = flags(rows, fuse, calibrated)
    labels = [r["label"] for r in rows]
    sources = sorted({r["source"] for r in rows})
    result: dict[str, Any] = {"configs": {}, "by_source": {}, "roc_auc": {}}
    for name, levels in per_config.items():
        result["configs"][name] = rates(labels, levels)
    for src in sources:
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        result["by_source"][src] = {name: rates([labels[i] for i in idx], [lv[i] for i in idx]) for name, lv in per_config.items()}
    y = np.asarray(labels)
    for key, name in (("ai_probability", "ai_text"), ("similarity", "echo similarity (2 references)"),
                      ("similarity_first", "echo similarity (1 reference)")):
        s = np.asarray([r[key] for r in rows])
        result["roc_auc"][name] = float(roc_auc_score(y, s))
    # Why honest answers were flagged by the fused rule.
    fused = per_config["fused (ai_text + echo)"]
    why: dict[str, int] = {}
    for r, lv in zip(rows, fused):
        if r["label"] == 0 and lv in LEVELS:
            key = f"{lv}: ai_text {r['ai_level']}, echo {r['echo_level']}"
            why[key] = why.get(key, 0) + 1
    result["false_flags_by_cause"] = dict(sorted(why.items(), key=lambda kv: -kv[1]))
    return result


# --- the run ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="HC3 all.jsonl")
    ap.add_argument("--cache", type=Path, required=True, help="calibrate_similarity.py's cache folder (reference answers, fastembed)")
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--out", type=Path, default=benchlib.RESULTS / "fusion_hc3.json")
    a = ap.parse_args()

    started = time.time()
    cal = load_calibration()
    signals = cal.SIGNALS
    th = signals.thresholds()
    echo_th = th["echo_similarity"]
    model, references = echo_th["reference_model"], int(echo_th["references"])
    cache = cal.ReferenceCache(a.cache / f"references_{model.replace(':', '_')}_{signals.REFERENCE_RECIPE}.jsonl", model)

    def no_generation(question: str, i: int) -> str:
        raise RuntimeError("a reference answer is missing from the cache; run calibrate_similarity.py first")

    table = cal.load_hc3(a.data)
    test_q = cal.split_questions(table["qid"], 0.2, cal.SEED)
    picked = cal.sample_questions(table, test_q, a.per_source, cal.SEED + 12)
    assert set(picked) <= test_q
    print(f"HC3 test questions: {len(picked)} of {len(test_q)}; embedding answers and cached references...", flush=True)
    embed = cal.fastembed_embedder(echo_th["embedding_model"], a.cache / "fastembed")
    scored, _ = cal.score_answers(table, picked, references, cache, no_generation, embed, log=lambda m: print(m, flush=True))

    echo_model = load_echo_model()
    text_by_index = table[table["qid"].isin(picked)]["text"].tolist()
    assert len(text_by_index) == len(scored)
    rows = []
    for row, text in zip(scored, text_by_index):
        assessed = echo_model.assess(text)
        rows.append({**row, "ai_probability": assessed["ai_probability"], "ai_level": assessed["level"],
                     "echo_level": level(row["similarity"], echo_th["possible"], echo_th["high"]),
                     "echo_level_1ref": level(row["similarity_first"], echo_th["possible"], echo_th["high"])})
    assert all(r["ai_level"] in ("low", "possible", "high") for r in rows), "HC3 answers have at least 15 words"
    calibrated = {"ai_text": bool(th["ai_text"]["calibrated"]), "echo_similarity": bool(echo_th["calibrated"])}
    result = {
        "what": "the spec's fused integrity flag (ai_text + echo similarity) on HC3 test questions",
        "environment": benchlib.environment(["fastembed", "onnxruntime", "numpy", "pandas", "scikit-learn"]),
        "data": "HC3 all.jsonl (CC BY-SA 4.0); train_content.py's question split; calibrate_similarity.py's test sample",
        "protocol": {"questions": len(picked), "per_source": a.per_source, "seed": cal.SEED + 12, "references": references,
                     "reference_model": model, "embedding_model": echo_th["embedding_model"],
                     "thresholds": {"ai_text": {"possible": echo_model.possible, "high": echo_model.high},
                                    "echo_similarity": {"possible": echo_th["possible"], "high": echo_th["high"]}},
                     "calibrated": calibrated,
                     "fusion": "spec/integrity/README.md: high needs a calibrated high signal plus another elevated one"},
        "answers": len(rows),
        **evaluate(rows, signals.fuse, calibrated),
        "seconds": round(time.time() - started),
    }
    benchlib.write_json(a.out, result)
    with a.out.with_name(a.out.stem + "_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "source", "label", "ai_probability", "similarity_2ref", "similarity_1ref"])
        for r in rows:
            w.writerow([r["qid"], r["source"], r["label"], f"{r['ai_probability']:.6f}", f"{r['similarity']:.6f}",
                        f"{r['similarity_first']:.6f}"])
    for name, res in result["configs"].items():
        p, h = res["at_possible"], res["at_high"]
        print(f"{name:40s} possible: detect {p['detection_rate']:.3f} false {p['false_flag_rate']:.3f} | "
              f"high: detect {h['detection_rate']:.3f} false {h['false_flag_rate']:.3f}")
    print(f"-> {a.out} [{result['seconds']} s]")


if __name__ == "__main__":
    main()
