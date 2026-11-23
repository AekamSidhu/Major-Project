# Reading detector: what is left (updated 2026-11-23)

The results are written up in README.md, section "Reading detector", and the
result JSONs are in `results/` (`reading_detector_results.json`,
`cv_cnn_gpu.json`, `robust_hgb_full.json`, `ritw_10s.json`, `ritw_5s.json`).
The shipped network is `local-ai/integrity/models/reading_network.json` and
`reading_network.onnx`. Delete this file when the items below are done.

## Kaggle runs (private notebooks, both finished)

1. **OpenMeet reading detector**: GazeBase end to end on GPU T4 x2, 85 min.
   Output also has `model_cnn_gpu_seed{0,1,2}.pt` and the logs.
2. **OpenMeet reading detector RitW**: 100 min, on CPU (Accelerator None).
   Output also has `ritw_{10,5}s_network_seed{0,1}.pt`.

The notebook outputs are not in the repository. The RitW download-links file
is personal and signed, and it must never be committed.

## Still to do

1. Zero-shot check (not run): run the RitW notebook with the GazeBase-only
   network as an extra input (`GAZEBASE_WEIGHTS` pointing at the first run's
   output), so the shipped network is scored on RitW test people untouched.
   Add the number to the README's RitW table.
2. One chart for the README (test balanced accuracy by condition).
3. DesktopActivity shows the GazeBase threshold is too strict for real
   desktop reading. Set the threshold from our own recordings (Phase 3), not
   from test data. The collection kit is ready in `training/data-collection`
   (protocol, consent, randomisation, the room's `--study` mode, and
   `calibrate.py`, which proposes the reading thresholds split by person).

The full test suite last ran on 2026-11-16 (Python 3.11): 55 passed with the
MediaPipe face model downloaded, 52 passed and 3 skipped without it.
