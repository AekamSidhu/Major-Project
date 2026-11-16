# Echo detection: work in progress (started 2026-11-13, updated 2026-11-17)

Temporary hand-off notes; the results are written up in README.md.

**Goal (Karan, 2026-11-13):** a working app first, not the most accurate
model.

## Done

- **AI-written text (Phase 1).** `train_content.py` on Kaggle (notebook
  "OpenMeet echo detection content", account KaranChadha24, private;
  output in `/kaggle/working/content/`), shipped as the TF-IDF model by
  `export_content_model.py` (`local-ai/integrity/models/echo_tfidf.json`):
  97.9% balanced accuracy, AUC 0.998 on HC3 test questions.
- **Echo similarity (Phase 1).** `calibrate_similarity.py`, 500 development
  and 500 test HC3 questions, llama3.2 references (cache:
  `M:\openmeet-data\hc3\cache`, outside the repo). Results in
  `results/echo_similarity.json`; thresholds in `spec/integrity/thresholds.json`,
  fixtures regenerated.
- **Word timestamps.** `POST /v1/audio/transcriptions` with
  `timestamp_granularities[]=word` (local-ai).
- **Delivery on SpiCE (Phase 2).** `evaluate_delivery.py` on the 34 English
  sessions' hand-aligned words (TextGrids only, 119 MB, in
  `M:\openmeet-data\spice\english`): AUC 0.55, see README. ALLSSTAR needs
  SpeechBox registration and was not used.
- **Room integration (Phase 5/6).** Delivery, echo similarity and the fused
  flag per answer in `local-ai/room`, plus study mode.
- **Collection kit (Phase 3 prep).** `training/data-collection`.

## Still to do

1. Delivery with Whisper timings on SpiCE:
   `python evaluate_delivery.py spice --textgrids M:/openmeet-data/spice/english --whisper --files M:/openmeet-data/spice/english_files.json --audio-cache M:/openmeet-data/spice/audio --out results/delivery_spice_whisper.json`
   (about 75 min on a CPU; audio snippets are cached in `audio/`, 419 of
   839 fetched). The first run was stopped at 400/839 when the machine ran
   low on memory.
2. Filler recall of Whisper (PodcastFillers), and whether a filler-rich
   prompt keeps them.
3. RAID: echo similarity and the AI-text model on other LLMs and
   paraphrases.
4. Own data (Phase 3), then thresholds from it (`calibrate.py`).
