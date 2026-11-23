# Echo detection: what is left (updated 2026-11-23)

Temporary hand-off notes; the results are written up in README.md.

## Done

- **AI-written text.** `train_content.py` trained the TF-IDF model on HC3
  (version 1). `retrain_with_interviews.py` retrained it with 54 interview
  question and answer pairs (version 2, 2026-11-22). `export_content_model.py`
  writes the shipped model, `local-ai/integrity/models/echo_tfidf.json`. The
  numbers are in README.md and docs/BENCHMARKS.md. The retraining does not
  reproduce the shipped file bit for bit (see BENCHMARKS.md).
- **Echo similarity.** `calibrate_similarity.py` on 500 development and 500
  test HC3 questions, with llama3.2 reference answers. Results in
  `results/echo_similarity.json`; thresholds in `spec/integrity/thresholds.json`;
  fixtures regenerated.
- **Word timestamps.** `POST /v1/audio/transcriptions` with
  `timestamp_granularities[]=word` (local-ai).
- **Delivery on SpiCE, hand-aligned words.** `evaluate_delivery.py` on the 34
  English sessions. ROC-AUC 0.55, see README.md and `results/delivery_spice.json`.
  ALLSSTAR was not used: its Hugging Face copy has no task labels or licence.
- **Room and desktop app.** Delivery, echo similarity and the fused flag are
  computed per answer in `local-ai/room` (with study mode) and in the desktop
  app's integrity analyzer.
- **Collection kit.** `training/data-collection`.

## Still to do

1. Delivery with Whisper timings on SpiCE. The result file
   `results/delivery_spice_whisper.json` is not in the repository. The first run
   was stopped at 400 of 839 files when the machine ran low on memory. Audio
   snippets are cached for 419 of the 839 files. The command is
   `python evaluate_delivery.py spice --textgrids <english TextGrids> --whisper
   --files <english_files.json> --audio-cache <audio cache> --out
   results/delivery_spice_whisper.json` (about 75 minutes on a CPU).
2. Filler recall of Whisper (PodcastFillers), and whether a filler-rich prompt
   keeps them.
3. RAID: echo similarity and the AI-text model on other LLMs and paraphrases.
4. Own labelled interviews (Phase 3), then thresholds from them with
   `training/data-collection/calibrate.py`.
