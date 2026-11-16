#!/usr/bin/env bash
# Content half of the echo signal on a Kaggle notebook (GPU T4, internet on). About 1 hour.
#
#   !git clone -q -b feature/echo-detection https://github.com/AekamSidhu/Major-Project.git /kaggle/temp/Major-Project
#   !bash /kaggle/temp/Major-Project/training/echo-detection/kaggle_content.sh
#
# Results, the fine-tuned model and its ONNX export go to /kaggle/working/content.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-/kaggle/working/content}"
mkdir -p "$OUT" /kaggle/temp/echo-data
pip install -q onnx onnxruntime 2>&1 | tail -1 || true
python -c "import torch, transformers, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'sklearn', sklearn.__version__)"
python -u train_content.py --data /kaggle/temp/echo-data/hc3_all.jsonl --out "$OUT" 2>&1 | grep -v -i "warning\|warn(\|it/s\]" | tee "$OUT/content_run.log"
