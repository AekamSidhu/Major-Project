# Reading detector: what is left (updated 2026-11-13)

The results are written up in README.md, section "Reading detector", and the
result JSONs are in `results/` (`reading_detector_results.json`,
`cv_cnn_gpu.json`, `robust_hgb_full.json`, `ritw_10s.json`, `ritw_5s.json`).
Delete this file when the items below are done.

## Kaggle runs (account KaranChadha24, private notebooks), both finished

1. **OpenMeet reading detector**, GazeBase end to end on GPU T4 x2, 85 min:
   https://www.kaggle.com/code/karanchadha24/openmeet-reading-detector
   Output also has `model_cnn_gpu_seed{0,1,2}.pt` and the logs.
2. **OpenMeet reading detector RitW**, 100 min. It ran with Accelerator
   **None** (CPU), which was fine but slower:
   https://www.kaggle.com/code/karanchadha24/openmeet-reading-detector-ritw
   Output also has `ritw_{10,5}s_network_seed{0,1}.pt`.

To fetch output files without the Kaggle CLI: on the notebook's Output tab,
the page posts `{"kernelSessionId": N, "fileName": "results/<file>"}` to
`/api/i/datasets.DatasetService/GetKernelOutputDataViewExternal` (with the
`x-xsrf-token` header from the `XSRF-TOKEN` cookie); the file text is in
`dataView.dataRaw.data`. Session ids: run 1 368940191, run 2 368940027.

## Still to do

1. Zero-shot check (not run): on the RitW notebook, Add Input > the output
   of "OpenMeet reading detector", add
   `GAZEBASE_WEIGHTS=/kaggle/input/<that input's folder>/results` to the
   second command of the cell, set Accelerator to GPU T4 x2, Save & Run All.
   This scores the GazeBase-only network on RitW test people, untouched.
   Add the number to the README's RitW table.
2. One chart for the README (test balanced accuracy by condition).
3. Run the full pytest suite once (`python -m pytest -q`); not run since the
   Kaggle changes.
4. DesktopActivity shows the GazeBase threshold is too strict for real
   desktop reading. Set the threshold from our own recordings (Phase 3),
   not from test data.

## Data on the laptop (outside OneDrive)

`C:\Users\karaa\openmeet-data\`: GazeBase, GazeBaseVR and DesktopActivity
zips, 30 Hz caches, feature tables and the earlier local CPU results. The
Reading in the Wild download-links file (personal, signed, expires about
2026-12-12) is in `openmeet-data/ritw/` and in Downloads. Never commit it.
