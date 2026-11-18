# AI-answer detector (content half of the echo signal)

The fine-tuned DistilRoBERTa from the Kaggle run "OpenMeet echo detection
content" (`train_content.py`, run by `kaggle_content.sh`). On validation it
beat TF-IDF + logistic regression, so it is the `chosen` model in
`../../results/content/content_results.json`, which also has its test numbers.
The sidecar ships the 1.9 MB TF-IDF model (`local-ai/integrity/models/echo_tfidf.json`)
instead, to keep the app small and free of a 330 MB download.

| File | What |
|---|---|
| `model.onnx` + `tokenizer.json` | For onnxruntime (no PyTorch): inputs `input_ids`, `attention_mask` (int64, up to 512 tokens); output `logits`, softmax index 1 = AI-written. Run `content.text.to_transcript` on the answer first. |
| `model/` | Hugging Face folder (`AutoModelForSequenceClassification.from_pretrained`), for further fine-tuning. |

`model.onnx` and `model/model.safetensors` are stored with **Git LFS**: run
`git lfs install` once, then `git lfs pull`. Checked after download: the ONNX
and PyTorch logits agree to 7e-6.

Trained on HC3 (CC BY-SA 4.0); the weights are shared under the same terms.
