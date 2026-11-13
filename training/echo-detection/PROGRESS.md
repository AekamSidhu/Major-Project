# Echo detection: work in progress (started 2026-11-13)

Temporary hand-off notes; fold into README.md when the results are in.

**Goal (Karan, 2026-11-13):** a working app first, not the most accurate
model. The reading detector (gaze branch) is trained, so start the AI-answer
model here.

## Content half (Phase 1): is this answer written by an LLM?

- `train_content.py`, run on Kaggle by `kaggle_content.sh` (notebook
  "OpenMeet echo detection content", account KaranChadha24, private, GPU T4 x2).
  Log: https://www.kaggle.com/code/karanchadha24/openmeet-echo-detection-content/log
  Output: `/kaggle/working/content/` with `content_results.json`, `model/`
  (Hugging Face), `model.onnx` + `tokenizer.json` (for the sidecar), and
  the other-LLM answers it generated.
- Data: HC3 `all.jsonl` (CC BY-SA 4.0), every answer turned into a
  transcript (`content/text.py`: lower case, words only), because HC3's
  Reddit answers are pre-tokenised ("do n't", " , ") and ChatGPT's have
  markdown: both would be shortcuts.
- Protocol: questions split once, 80% development / 20% test (seed 2026),
  development split 85/15 into train / validation. Candidates TF-IDF + LR,
  fine-tuned DistilRoBERTa, average; chosen on validation balanced
  accuracy; test questions scored once. Extra: answers from Qwen2.5-1.5B and
  SmolLM2-1.7B (unseen LLMs), answers cut to 50/100/200 words,
  leave-one-source-out (TF-IDF).
- Smoke run (4% of HC3, tiny model, CPU): TF-IDF 95% balanced accuracy on
  test questions; numbers mean nothing beyond "the plumbing works".

## Not started

- Delivery half (Phase 2: read vs spontaneous speech from timing and
  fillers; needs word timestamps from the sidecar and ALLSSTAR audio).
- App integration (Phase 5).
