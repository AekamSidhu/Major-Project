"""Calibrate the echo-similarity signal on HC3 (Phase 1 of the echo plan).

    python calibrate_similarity.py --data M:/openmeet-data/hc3/all.jsonl --cache M:/openmeet-data/hc3/cache \
        --out results/echo_similarity.json [--update-spec]
    python calibrate_similarity.py ... --time 10       # measure generation speed only

Echo similarity (spec/integrity/README.md): a local LLM answers the
interviewer's question `references` times; the answer and the references are
embedded; similarity = the highest cosine. Here the "answers" are HC3's human
and ChatGPT answers, and we ask whether ChatGPT's answers sit closer to
llama3.2's own answers than people's answers do.

Protocol, fixed before any test question is scored:
- HC3 all.jsonl with train_content.py's loading (answers as transcripts, at
  least 15 words, duplicates dropped) and its question split (seed 2026, 20%
  of questions are test). Nothing from the test questions is used to choose
  anything.
- The same number of questions is sampled from each HC3 source in each split
  (only questions with both a human and a ChatGPT answer), so the small,
  interview-like sources (wiki_csai, open_qa) are not drowned by reddit_eli5.
- Reference answers follow the app's recipe exactly
  (local-ai/integrity/signals.py: system prompt, temperature, token cap,
  seed i for reference i), from the model and count in thresholds.json. Every
  answer and reference goes through to_transcript before embedding, as in
  the room.
- Thresholds on development questions only: `possible` = the lower and
  `high` = the higher of the balanced-accuracy cut and the cut with a 5%
  false-positive rate (human answers flagged), as the AI-text model does.
  Rounded up to 4 decimals; every number is computed with the rounded values.
- Test questions are scored once with those thresholds.
- The signal counts as calibrated (CALIBRATION_RULE) only if, on test
  questions, ROC-AUC >= 0.70 and the false-positive rate at `high` <= 10%.

Writes aggregate numbers only; the reference answers are cached under
--cache (outside the repository).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from content.text import to_transcript  # noqa: E402
from train_content import SEED, best_threshold, load_hc3, metrics, split_questions, threshold_at_fpr  # noqa: E402

SPEC_THRESHOLDS = REPO / "spec" / "integrity" / "thresholds.json"
FPR_TARGET = 0.05
MIN_TEST_AUC = 0.70
MAX_TEST_FPR_AT_HIGH = 0.10
CALIBRATION_RULE = (f"calibrated only if, on test questions, ROC-AUC >= {MIN_TEST_AUC:.2f} and the false-positive "
                    f"rate at `high` <= {MAX_TEST_FPR_AT_HIGH:.0%}")


def load_signals():
    """local-ai/integrity/signals.py by path (stdlib only; no sidecar packages needed)."""
    spec = importlib.util.spec_from_file_location("openmeet_signals", REPO / "local-ai" / "integrity" / "signals.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SIGNALS = load_signals()


# --- reference answers --------------------------------------------------------------


def ollama_generator(model: str, url: str = "http://127.0.0.1:11434") -> Callable[[str, int], str]:
    """Reference answer i to a question, from Ollama, with the app's recipe."""

    def generate(question: str, i: int) -> str:
        body = {"model": model, "messages": SIGNALS.reference_messages(question), "stream": False,
                "options": {"temperature": SIGNALS.REFERENCE_TEMPERATURE, "num_predict": SIGNALS.REFERENCE_MAX_TOKENS,
                            "seed": SIGNALS.reference_seed(i)}}
        request = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["message"]["content"]

    return generate


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ReferenceCache:
    """Append-only JSONL of generated references, so a stopped run resumes."""

    def __init__(self, path: Path, model: str) -> None:
        self.path, self.model = path, model
        self.items: dict[tuple[str, int], str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row["model"] == model and row["recipe"] == SIGNALS.REFERENCE_RECIPE:
                        self.items[(row["q"], row["i"])] = row["text"]

    def get(self, question: str, i: int, generate: Callable[[str, int], str]) -> tuple[str, float | None]:
        key = (sha(question), i)
        if key in self.items:
            return self.items[key], None
        started = time.perf_counter()
        text = generate(question, i)
        took = time.perf_counter() - started
        self.items[key] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"q": key[0], "i": i, "model": self.model, "recipe": SIGNALS.REFERENCE_RECIPE,
                                "text": text, "seconds": round(took, 2)}) + "\n")
        return text, took


# --- embeddings -----------------------------------------------------------------------


def fastembed_embedder(model: str, cache_dir: Path) -> Callable[[Sequence[str]], np.ndarray]:
    """The sidecar's embedder: fastembed TextEmbedding(model).embed, as in local-ai/server.py."""
    from fastembed import TextEmbedding

    engine = TextEmbedding(model_name=model, cache_dir=str(cache_dir))
    # fastembed's default batch (256 texts of up to 512 tokens) needs several GB of RAM;
    # the vectors are the same with smaller batches.
    return lambda texts: np.asarray(list(engine.embed(list(texts), batch_size=16)), dtype=np.float64)


def embed_all(texts: list[str], embed: Callable[[Sequence[str]], np.ndarray], batch: int = 256) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), batch):
        chunk = unique[start : start + batch]
        for text, vec in zip(chunk, embed(chunk)):
            out[text] = np.asarray(vec, dtype=np.float64)
    return out


# --- the run --------------------------------------------------------------------------------


def sample_questions(table, qids: set, per_source: int, seed: int) -> list[int]:
    """`per_source` questions from each HC3 source that have both a human and a ChatGPT answer."""
    part = table[table["qid"].isin(qids)]
    both = part.groupby("qid")["label"].nunique()
    eligible = part[part["qid"].isin(both[both == 2].index)]
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for source in sorted(eligible["source"].unique()):
        ids = np.array(sorted(eligible.loc[eligible["source"] == source, "qid"].unique()))
        picked += sorted(rng.choice(ids, size=min(per_source, len(ids)), replace=False).tolist())
    return picked


def score_answers(table, picked: list[int], references: int, cache: ReferenceCache, generate, embed,
                  log: Callable[[str], None] = print) -> tuple[list[dict], dict]:
    """Similarity of every human and ChatGPT answer to its question's references."""
    questions = table.drop_duplicates("qid").set_index("qid")["question"]
    refs: dict[int, list[str]] = {}
    took: list[float] = []
    started = time.perf_counter()
    for n, qid in enumerate(picked, 1):
        refs[qid] = []
        for i in range(references):
            text, seconds = cache.get(questions[qid], i, generate)
            refs[qid].append(to_transcript(text))
            if seconds is not None:
                took.append(seconds)
        if n % 50 == 0 or n == len(picked):
            log(f"  references for {n}/{len(picked)} questions [{time.perf_counter() - started:.0f} s]")
    rows = table[table["qid"].isin(picked)]
    vectors = embed_all(rows["text"].tolist() + [r for rs in refs.values() for r in rs], embed)
    out = []
    for row in rows.itertuples():
        answer = vectors[row.text]
        sims = [SIGNALS.cosine(answer.tolist(), vectors[r].tolist()) for r in refs[row.qid]]
        out.append({"qid": int(row.qid), "source": row.source, "label": int(row.label), "similarity": max(sims),
                    "similarity_first": sims[0]})
    speed = {"generated": len(took), "seconds_per_reference": float(np.mean(took)) if took else None}
    return out, speed


def within_question_auc(rows: list[dict]) -> float | None:
    """Share of (ChatGPT, human) answer pairs to the same question where ChatGPT's is more similar."""
    wins = total = 0.0
    by_q: dict[int, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    for answers in by_q.values():
        ai = [a["similarity"] for a in answers if a["label"] == 1]
        human = [a["similarity"] for a in answers if a["label"] == 0]
        for a in ai:
            for h in human:
                wins += 1.0 if a > h else 0.5 if a == h else 0.0
                total += 1
    return wins / total if total else None


def summarise(rows: list[dict], possible: float, high: float, key: str = "similarity") -> dict:
    y = np.array([r["label"] for r in rows])
    s = np.array([r[key] for r in rows])
    at_possible, at_high = metrics(y, s, possible), metrics(y, s, high)
    return {"answers": len(rows), "questions": len({r["qid"] for r in rows}), "chatgpt_answers": int(y.sum()),
            "human_answers": int((1 - y).sum()), "roc_auc": at_possible["roc_auc"],
            "mean_similarity": {"chatgpt": float(s[y == 1].mean()), "human": float(s[y == 0].mean())},
            "at_possible": {k: at_possible[k] for k in ("balanced_accuracy", "recall", "false_positive_rate")},
            "at_high": {k: at_high[k] for k in ("balanced_accuracy", "recall", "false_positive_rate")}}


def up4(x: float) -> float:
    """Round a cut up to 4 decimals (never flags more than the exact cut)."""
    return math.ceil(float(x) * 1e4 - 1e-9) / 1e4


def calibrate(table, references: int, per_source_dev: int, per_source_test: int, cache: ReferenceCache, generate, embed,
              log: Callable[[str], None] = print) -> dict:
    test_q = split_questions(table["qid"], 0.2, SEED)
    dev_q = set(table["qid"]) - test_q
    dev_pick = sample_questions(table, dev_q, per_source_dev, SEED + 11)
    test_pick = sample_questions(table, test_q, per_source_test, SEED + 12)
    assert not set(dev_pick) & set(test_pick)

    log(f"development: {len(dev_pick)} questions")
    dev, dev_speed = score_answers(table, dev_pick, references, cache, generate, embed, log)
    yd = np.array([r["label"] for r in dev])
    sd = np.array([r["similarity"] for r in dev])
    cuts = {"balanced": best_threshold(yd, sd), "fpr_5": threshold_at_fpr(yd, sd, FPR_TARGET)}
    possible = up4(min(cuts.values()))
    high = up4(max(cuts.values()))
    result = {"thresholds": {"possible": possible, "high": high, "balanced_cut": cuts["balanced"], "fpr_5_cut": cuts["fpr_5"]},
              "development": summarise(dev, possible, high)}
    log(f"thresholds from development questions: possible {possible}, high {high}")

    # Test questions, scored once with the development thresholds.
    log(f"test: {len(test_pick)} questions")
    test, test_speed = score_answers(table, test_pick, references, cache, generate, embed, log)
    result["test"] = summarise(test, possible, high)
    result["test"]["within_question_auc"] = within_question_auc(test)
    result["test"]["by_source"] = {src: summarise([r for r in test if r["source"] == src], possible, high)
                                   for src in sorted({r["source"] for r in test})}
    first = summarise(test, possible, high, key="similarity_first")
    result["test"]["one_reference_roc_auc"] = first["roc_auc"]
    t = result["test"]
    result["calibration_rule"] = CALIBRATION_RULE
    result["calibrated"] = bool(t["roc_auc"] >= MIN_TEST_AUC and t["at_high"]["false_positive_rate"] <= MAX_TEST_FPR_AT_HIGH)
    result["speed"] = {"development": dev_speed, "test": test_speed}
    return result


def update_spec(path: Path, result: dict, out_name: str) -> None:
    """Write the measured thresholds into the echo_similarity block of thresholds.json (nothing else)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data["echo_similarity"]
    t = result["test"]
    block["possible"] = result["thresholds"]["possible"]
    block["high"] = result["thresholds"]["high"]
    block["calibrated"] = result["calibrated"]
    worst = ""
    by_source = t.get("by_source") or {}
    if by_source:  # textbook-style sources flag honest answers far more often: say so where the threshold lives
        src, s = max(by_source.items(), key=lambda kv: kv[1]["at_high"]["false_positive_rate"])
        worst = (f"; worst source {src}: false-positive rate {s['at_possible']['false_positive_rate']:.0%} at possible, "
                 f"{s['at_high']['false_positive_rate']:.0%} at high (textbook answers look alike)")
    block["source"] = (f"HC3 (training/echo-detection/calibrate_similarity.py, {out_name}): thresholds from "
                       f"{result['development']['questions']} development questions; on {t['questions']} test questions "
                       f"ROC-AUC {t['roc_auc']:.3f}, false-positive rate {t['at_possible']['false_positive_rate']:.1%} at possible "
                       f"and {t['at_high']['false_positive_rate']:.1%} at high" + worst
                       + ("" if result["calibrated"] else "; too weak to count as calibrated"))
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="HC3 all.jsonl")
    ap.add_argument("--cache", type=Path, required=True, help="folder for reference answers and models (outside the repo)")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "echo_similarity.json")
    ap.add_argument("--per-source-dev", type=int, default=100)
    ap.add_argument("--per-source-test", type=int, default=100)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--time", type=int, default=0, help="only time this many questions' references, then stop")
    ap.add_argument("--update-spec", action="store_true", help="write the thresholds into spec/integrity/thresholds.json")
    a = ap.parse_args()

    spec = json.loads(SPEC_THRESHOLDS.read_text(encoding="utf-8"))["echo_similarity"]
    model, embedder, references = spec["reference_model"], spec["embedding_model"], int(spec["references"])
    cache = ReferenceCache(a.cache / f"references_{model.replace(':', '_')}_{SIGNALS.REFERENCE_RECIPE}.jsonl", model)
    generate = ollama_generator(model, a.ollama)
    table = load_hc3(a.data)
    print(f"HC3: {table['qid'].nunique()} questions, {len(table)} answers; references: {references} x {model}; embeddings: {embedder}",
          flush=True)

    if a.time:
        test_q = split_questions(table["qid"], 0.2, SEED)
        picked = sample_questions(table, set(table["qid"]) - test_q, 2, SEED + 11)[: a.time]
        questions = table.drop_duplicates("qid").set_index("qid")["question"]
        generate(questions[picked[0]], 99)  # load the model
        started = time.perf_counter()
        for qid in picked:
            for i in range(references):
                cache.get(questions[qid], i, generate)
        per_q = (time.perf_counter() - started) / len(picked)
        print(f"{per_q:.1f} s per question ({references} references): about {3600 / per_q:.0f} questions an hour")
        return

    started = time.time()
    result = {"what": "echo similarity on HC3: human vs ChatGPT answers against llama3.2 reference answers",
              "data": "HC3 all.jsonl (CC BY-SA 4.0), answers as transcripts (content/text.py), train_content.py's question split",
              "embedding_model": embedder, "reference_model": model, "references": references,
              "recipe": {"version": SIGNALS.REFERENCE_RECIPE, "system": SIGNALS.REFERENCE_SYSTEM,
                         "temperature": SIGNALS.REFERENCE_TEMPERATURE, "max_tokens": SIGNALS.REFERENCE_MAX_TOKENS,
                         "seed": "reference i uses seed i"},
              "sampling": {"per_source_dev": a.per_source_dev, "per_source_test": a.per_source_test, "seed": SEED},
              "fpr_target": FPR_TARGET}
    log = lambda m: print(m, flush=True)  # noqa: E731
    result.update(calibrate(table, references, a.per_source_dev, a.per_source_test, cache, generate,
                            fastembed_embedder(embedder, a.cache / "fastembed"), log))
    result["seconds"] = round(time.time() - started)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    t = result["test"]
    print(f"\nTEST ({t['questions']} questions, {t['answers']} answers): ROC-AUC {t['roc_auc']:.3f}; "
          f"at possible {result['thresholds']['possible']}: balanced acc {t['at_possible']['balanced_accuracy']:.3f}, "
          f"FPR {t['at_possible']['false_positive_rate']:.3f}; at high {result['thresholds']['high']}: "
          f"balanced acc {t['at_high']['balanced_accuracy']:.3f}, FPR {t['at_high']['false_positive_rate']:.3f}; "
          f"calibrated: {result['calibrated']} -> {a.out}")
    if a.update_spec:
        update_spec(SPEC_THRESHOLDS, result, a.out.name)
        print(f"updated {SPEC_THRESHOLDS}; now run: python -I spec/integrity/reference.py")


if __name__ == "__main__":
    main()
