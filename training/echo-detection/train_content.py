"""Train the content half of the echo signal: was this answer written by an LLM?

    python train_content.py --out results/content          # full run (Kaggle GPU, ~1 h)
    python train_content.py --out /tmp/smoke --quick        # plumbing check, CPU, minutes

Data: HC3 (Guo et al. 2023), ~24k questions from five sources, each with human
answers and ChatGPT answers (CC BY-SA 4.0). Every answer is first turned into
what a transcript of it would look like (`content.text.to_transcript`), so
the detector sees what OpenMeet will see after Whisper.

Protocol, fixed before results are read:
- questions are split once (seed 2026): 80% development, 20% test. No
  question has answers on both sides. Development is split again, 85/15,
  into train and validation questions.
- candidates: TF-IDF + logistic regression, a fine-tuned DistilRoBERTa, and
  their average. The one with the best validation balanced accuracy is
  chosen, with its threshold from validation. Test questions are scored once.
- other LLMs: the candidate in an interview probably used a newer model than
  2022 ChatGPT. On a GPU, two small open models (Qwen2.5-1.5B, SmolLM2-1.7B)
  answer 400 test questions each; their answers against the human answers to
  the same questions give an "unseen model" check.
- leave one source out (TF-IDF only, cheap): train on four sources, test on
  the fifth, for an unseen-topic check.
- answer length: test answers cut to their first 50, 100 and 200 words.

Outputs in --out: content_results.json, the fine-tuned model
(`model/`, Hugging Face format) and `model.onnx` + `tokenizer.json` for the
OpenMeet sidecar (onnxruntime, no PyTorch needed there).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content.text import first_words, to_transcript  # noqa: E402

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
SEED = 2026
GENERATORS = {"qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}


# --- data -----------------------------------------------------------------------
def load_hc3(path: Path) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HC3_URL, path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for qid, line in enumerate(f):
            item = json.loads(line)
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for answer in item[key] or []:
                    if isinstance(answer, str) and answer.strip():
                        rows.append({"qid": qid, "source": item["source"], "question": item["question"], "label": label, "raw": answer})
    table = pd.DataFrame(rows)
    table["text"] = table["raw"].map(to_transcript)
    table = table[table["text"].str.split().str.len() >= 15]
    return table.drop_duplicates("text").reset_index(drop=True)


def split_questions(qids, fraction: float, seed: int) -> set:
    """A fixed random `fraction` of the question ids."""
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
    return {"n_ai": int(y.sum()), "n_human": int((~y).sum()), "threshold": float(cut), "balanced_accuracy": (rec + spec) / 2,
            "accuracy": float((flag == y).mean()), "recall": rec, "false_positive_rate": 1 - spec,
            "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")}


def report(name, y, s, cuts) -> dict:
    out = {k: metrics(y, s, c) for k, c in cuts.items()}
    b = out["balanced"]
    print(f"  {name:38s} AUC {b['roc_auc']:.4f}  balanced acc {100 * b['balanced_accuracy']:.2f}%  "
          f"(AI caught {100 * b['recall']:.1f}%, humans flagged {100 * b['false_positive_rate']:.1f}%)  "
          f"| at 5% FPR: AI caught {100 * out['low_fpr']['recall']:.1f}%", flush=True)
    return out


# --- models ---------------------------------------------------------------------
def tfidf_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True),
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000),
    )


class Transformer:
    """A small pretrained encoder fine-tuned to tell AI answers from human ones."""

    def __init__(self, name: str, max_len: int, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.max_len, self.device = torch, max_len, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

    def fit(self, texts, labels, epochs: int, batch: int, lr: float, seed: int) -> None:
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup

        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        labels = np.asarray(labels)
        weight = torch.tensor([1.0, (labels == 0).sum() / max((labels == 1).sum(), 1)], device=self.device, dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(texts) / batch))
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.device == "cuda")
        words = [t.split() for t in texts]
        started = time.time()
        for epoch in range(epochs):
            self.model.train()
            total = 0.0
            order = rng.permutation(len(texts))
            for i, start in enumerate(range(0, len(order), batch)):
                idx = order[start : start + batch]
                # Random crops of 40-220 words: interview answers vary in length.
                crops = []
                for j in idx:
                    w = words[j]
                    n = int(rng.integers(40, 221))
                    s = int(rng.integers(0, max(len(w) - n, 0) + 1)) if rng.random() < 0.5 else 0
                    crops.append(" ".join(w[s : s + n]))
                enc = self.tok(crops, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
                y = torch.as_tensor(labels[idx], device=self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    loss = loss_fn(self.model(**enc).logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += float(loss.detach())
                if i % 500 == 0:
                    print(f"    epoch {epoch + 1} step {i} loss {total / (i + 1):.4f} [{time.time() - started:.0f} s]", flush=True)

    def predict(self, texts, batch: int = 128):
        torch = self.torch
        self.model.eval()
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    logits = self.model(**enc).logits.float()
                out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out

    def export(self, folder: Path) -> dict:
        """Hugging Face folder, plus ONNX + tokenizer.json for onnxruntime."""
        torch = self.torch
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(folder / "model")
        self.tok.save_pretrained(folder / "model")
        info = {}
        try:
            model = self.model.float().cpu().eval()
            enc = self.tok(["this is a test answer"], return_tensors="pt")
            torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(folder / "model.onnx"),
                              input_names=["input_ids", "attention_mask"], output_names=["logits"],
                              dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "logits": {0: "batch"}},
                              opset_version=17, dynamo=False)
            import onnxruntime as ort

            session = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
            onnx_logits = session.run(None, {k: v.numpy() for k, v in enc.items() if k in ("input_ids", "attention_mask")})[0]
            with torch.no_grad():
                torch_logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.numpy()
            info["onnx_max_abs_diff"] = float(np.abs(onnx_logits - torch_logits).max())
            info["onnx_mb"] = round((folder / "model.onnx").stat().st_size / 1e6, 1)
            self.tok.backend_tokenizer.save(str(folder / "tokenizer.json"))
        except Exception as error:  # the HF folder is enough to rebuild it
            info["onnx_error"] = f"{type(error).__name__}: {error}"[:300]
        model = self.model.to(self.device)
        return info


def generate_answers(model_id: str, questions: list[str], device: str, max_new_tokens: int, seed: int) -> list[str]:
    """Answers from another LLM, the way a chat assistant would give them."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    torch.manual_seed(seed)
    answers = []
    for start in range(0, len(questions), 32):
        chunk = questions[start : start + 32]
        if tok.chat_template:
            prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in chunk]
        else:  # base models (smoke tests only)
            prompts = [f"Question: {q}\nAnswer:" for q in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
        answers += tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return answers


# --- the run --------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "hc3_all.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--generated", type=int, default=400, help="test questions answered by each other LLM")
    ap.add_argument("--quick", action="store_true", help="tiny sample and tiny model: plumbing check only")
    a = ap.parse_args()
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "none"

    table = load_hc3(a.data)
    if a.quick:
        keep = split_questions(table["qid"], 0.04, 1)
        table = table[table["qid"].isin(keep)].reset_index(drop=True)
        a.model, a.epochs, a.generated = "sshleifer/tiny-distilroberta-base", 1, 8
    test_q = split_questions(table["qid"], 0.2, SEED)
    dev = table[~table["qid"].isin(test_q)]
    val_q = split_questions(dev["qid"], 0.15, SEED + 1)
    train, val, test = dev[~dev["qid"].isin(val_q)], dev[dev["qid"].isin(val_q)], table[table["qid"].isin(test_q)]
    result = {"data": "HC3 all.jsonl, answers as transcripts", "device": device,
              "sizes": {n: {"answers": len(t), "questions": int(t["qid"].nunique()), "ai_share": float(t["label"].mean())} for n, t in (("train", train), ("val", val), ("test", test))}}
    print(f"HC3: train {len(train)}, val {len(val)}, test {len(test)} answers; AI share {table['label'].mean():.2f}; device {device}", flush=True)

    # Candidates on validation questions.
    tfidf = tfidf_model().fit(train["text"], train["label"])
    val_scores = {"tfidf": tfidf.predict_proba(val["text"])[:, 1]}
    transformer = None
    if device != "none":
        transformer = Transformer(a.model, a.max_len, device)
        transformer.fit(train["text"].tolist(), train["label"].to_numpy(), a.epochs, 32, 3e-5, SEED)
        val_scores["transformer"] = transformer.predict(val["text"].tolist())
        val_scores["average"] = (val_scores["tfidf"] + val_scores["transformer"]) / 2
    print(f"validation questions [{time.time() - t0:.0f} s]", flush=True)
    yv = val["label"].to_numpy()
    cuts = {c: {"balanced": best_threshold(yv, s), "low_fpr": threshold_at_fpr(yv, s)} for c, s in val_scores.items()}
    result["validation"] = {c: report(f"VAL {c}", yv, s, cuts[c]) for c, s in val_scores.items()}
    chosen = max(val_scores, key=lambda c: result["validation"][c]["balanced"]["balanced_accuracy"])
    result["chosen"] = chosen
    print(f"chosen on validation: {chosen}", flush=True)

    def score(cand, texts):
        texts = list(texts)
        if cand == "tfidf":
            return tfidf.predict_proba(texts)[:, 1]
        if cand == "transformer":
            return transformer.predict(texts)
        return (tfidf.predict_proba(texts)[:, 1] + transformer.predict(texts)) / 2

    # Test questions, scored once.
    yt = test["label"].to_numpy()
    result["test"] = {}
    for cand in val_scores:
        result["test"][cand] = {"full": report(f"TEST {cand}", yt, score(cand, test["text"]), cuts[cand])}
        for n in (50, 100, 200):
            short = test["text"].map(lambda t: first_words(t, n))
            result["test"][cand][f"first_{n}_words"] = report(f"TEST {cand} first {n} words", yt, score(cand, short), cuts[cand])
    result["test_by_source"] = {}
    for src, part in test.groupby("source"):
        result["test_by_source"][src] = report(f"TEST {chosen} {src}", part["label"].to_numpy(), score(chosen, part["text"]), cuts[chosen])

    # Answers from LLMs the detector never saw.
    if device == "cuda" or a.quick:
        rng = np.random.default_rng(SEED)
        questions = test.drop_duplicates("qid")
        picked = questions.iloc[rng.permutation(len(questions))[: a.generated]]
        humans = test[(test["label"] == 0) & test["qid"].isin(picked["qid"])].drop_duplicates("qid")
        result["other_llms"] = {}
        for short_name, model_id in (GENERATORS.items() if not a.quick else [("tiny", "sshleifer/tiny-gpt2")]):
            try:
                made = generate_answers(model_id, picked["question"].tolist(), device if device != "none" else "cpu", 220 if not a.quick else 20, SEED)
            except Exception as error:
                result["other_llms"][short_name] = {"error": f"{type(error).__name__}: {error}"[:300]}
                print(f"  {short_name}: generation failed: {error}", flush=True)
                continue
            ai = [to_transcript(t) for t in made]
            ai = [t for t in ai if len(t.split()) >= 15] or ai
            texts = ai + humans["text"].tolist()
            y = np.r_[np.ones(len(ai)), np.zeros(len(humans))]
            result["other_llms"][short_name] = {cand: report(f"OTHER LLM {short_name} {cand}", y, score(cand, texts), cuts[cand]) for cand in val_scores}
            (a.out / f"generated_{short_name}.jsonl").write_text("\n".join(json.dumps({"q": q, "a": t}) for q, t in zip(picked["question"], made)), encoding="utf-8")

    # Unseen topic: leave one HC3 source out (TF-IDF only).
    result["leave_one_source_out_tfidf"] = {}
    for src in sorted(dev["source"].unique()):
        m = tfidf_model().fit(dev.loc[dev["source"] != src, "text"], dev.loc[dev["source"] != src, "label"])
        part = test[test["source"] == src]
        if part["label"].nunique() == 2:
            result["leave_one_source_out_tfidf"][src] = report(f"UNSEEN SOURCE {src} (tfidf)", part["label"].to_numpy(), m.predict_proba(part["text"])[:, 1], cuts["tfidf"])

    if transformer is not None:
        result["export"] = transformer.export(a.out)
    result["seconds"] = round(time.time() - t0)
    (a.out / "content_results.json").write_text(json.dumps(result, indent=1, default=float))
    h = result["test"][chosen]["full"]["balanced"]
    print(f"\nHEADLINE content ({chosen}, test questions): balanced accuracy {100 * h['balanced_accuracy']:.2f}%, AUC {h['roc_auc']:.4f} "
          f"[{time.time() - t0:.0f} s] -> {a.out / 'content_results.json'}", flush=True)


if __name__ == "__main__":
    main()
