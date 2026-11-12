# Reading detector: work in progress (started 2026-11-12)

Temporary hand-off notes. Fold the results into README.md and working.md when
the work is finished, then delete this file.

**Goal (from Karan):** find the best public dataset for the "is this person
reading?" model, train on it, and get a defensible accuracy number (target
was "something like 85%"), trying different models and validations.

## Where everything is

- **Code (uncommitted, branch `feature/gaze-detection`)** in `training/gaze-detection/`:
  - new: `build_cache.py`, `readgaze/cache.py`, `readgaze/protocol.py`,
    `readgaze/features_ext.py` (36 features), `readgaze/features_fold.py`
    (16 low-SNR features), `readgaze/seqmodel.py` (1-D CNN, needs torch),
    `readgaze/datasets/gazebasevr.py`, `readgaze/datasets/desktopactivity.py`,
    `tests/test_{cache,protocol,features_ext,seqmodel}.py`,
    `experiments/*.py` (the exploration scripts) and `experiments/queue.sh`
  - changed: `readgaze/simulate.py` (`degrade`, `ProfileRange`),
    `readgaze/pipeline.py` (`cache_windows`), `readgaze/features.py` (faster
    median and skew, same results)
  - torch (CPU) was installed into `.venv`; not yet in `requirements.txt`
- **Data (outside OneDrive)** in `C:\Users\karaa\openmeet-data\`:
  `gazebase/GazeBase_v2_0.zip` (6.7 GB), `gazebasevr/gazebasevr.zip` (2.5 GB),
  `desktopactivity/DesktopActivity.zip` (2 MB), `cache/*.npz` (30 Hz frame
  caches), `tables/` (feature tables `.pkl`, raw windows `_raw.npy`, folding
  features `_fold.npy`, out-of-fold scores `_oof_*.npy`, result JSONs),
  `logs/` (queue logs)
- The first CNN run logs to the old session scratchpad:
  `C:\Users\karaa\AppData\Local\Temp\claude\18678cee3ca54b37\99ba60a2-4653-4177-b77a-048e784c00d3\scratchpad\exp\cnn_v1.log`;
  its results are `tables/cv_cnn_v1.json` and `tables/*_oof_cnn_v1.npy`,
  `tables/*_final_cnn_v1.npy`, `tables/model_cnn_v1.pt`

## How to run things

From `training/gaze-detection/experiments/`, always with the venv Python in
isolated mode: `../.venv/Scripts/python.exe -I -u <script>.py ...`.
**One heavy job at a time**: the laptop has 7.7 GB RAM and 4 cores; two jobs
at once swap and crawl. A table build takes 5–8 minutes, a CNN 5-fold run
about an hour.

## Datasets (decided)

- **GazeBase** (train + test): 322 people, 12,334 recordings, 211 h, reading
  (TEX) vs six other tasks, 1000 Hz eye tracker, CC BY 4.0. Degraded to
  10 fps webcam quality by `simulate.degrade`.
- **GazeBaseVR** (outside test only): 407 people, VR headset, 5 tasks.
- **DesktopActivity** (outside test only): 8 people at a real desktop, head
  free, READ vs BROWSE/PLAY/SEARCH/WATCH/WRITE, 30 Hz.
- **Reading in the Wild** (Meta, 100 h, 111 people) is the best-matched
  benchmark (published gaze-only accuracy 82.3% on 2 s windows, 85.8% on
  5 s) but the download needs Karan's email and a licence click. Not used.
  Ask him before submitting anything.

## Protocol (fixed before looking at test people)

- People split once (`protocol.split_people`, seed 2026): 258 development,
  64 test. All choices by 5-fold person-level CV on development people.
  Test people are scored once, at the end, with thresholds from development.
- Unit: 10 s window every 5 s at 10 fps; a recording ("answer") score is the
  mean of its top half of window scores.
- Headline metric: **balanced accuracy** (plain accuracy is 86% for a
  detector that never flags, because only 14% of windows are reading). Also
  ROC-AUC and recall at 5% false positives.
- Conditions: nominal (noise 1.5°, drift 1°), hard (noise 3°, drift 2°),
  small (movements x0.5), small_hard (both). "mixed" = each recording draws
  its own noise 0.3–3.5°, drift 0–2°, size 0.4–1.3 (for training).

## Results so far (development people, CV)

Trained and tested at nominal:

| Detector | Window bal. acc. | Window AUC | Recording bal. acc. |
|---|---|---|---|
| Rule-based | 80.5% | 0.865 | 92.5% |
| Logistic, 6 features (the old v1) | 82.4% | 0.896 | 93.4% |
| Boosting, 6 features | 86.7% | 0.943 | 96.9% |
| Logistic, 42 features | 96.4% | 0.993 | 99.4% |
| Boosting, 42 features | 97.8% | 0.998 | 99.9% |
| CNN probe (2 epochs, 1 fold) | 98.3% | 0.998 | 99.8% |

Boosting, 42 features, window balanced accuracy at the nominal threshold
(AUC in brackets):

| Trained on | nominal | hard | small | small_hard |
|---|---|---|---|---|
| nominal only | 97.8 (0.998) | 79.4 (0.915) | 79.4 (0.915) | 39.0 (0.395) |
| mixed | 96.7 (0.995) | 90.8 (0.972) | 91.1 (0.973) | 74.0 (0.831) |
| nominal + mixed | 97.5 (0.997) | 91.2 (0.973) | 91.3 (0.973) | 72.7 (0.812) |

So training on mixed conditions is essential: a model trained at one noise
level learns that level and collapses under stress.

Zero-shot on other datasets (boosting 42, trained on GazeBase development
people, nominal + mixed, thresholds from GazeBase):

- GazeBaseVR: window 94.4% (AUC 0.982), recording 99.3%. Vergence windows
  are flagged 21% of the time, but no vergence recording is.
- DesktopActivity: window 82.8% (AUC 0.918). Recording AUC 1.00: every READ
  recording outranks every other one. At the GazeBase threshold 5 of 8 READ
  recordings are flagged and no other; at the 5%-false-positive threshold
  8 of 8. Window flag rates: READ 73%, WRITE 13%, SEARCH 10%, BROWSE 6%,
  PLAY 5%, WATCH 3%.

## Overnight results (2026-11-12/13, local CPU, development people)

Window balanced accuracy at the nominal threshold:

| Detector | nominal | hard / small | small_hard |
|---|---|---|---|
| Boosting, 58 features (+16 folding), nominal+mixed | 97.9 | 92.3 / 92.5 | 72.5 |
| Network (1-D CNN, 10 epochs), seed 0 | 98.6 | 97.1 | 85.5 |
| Network, seed 1 | 98.7 | 97.1 | 83.9 |

- hard and small give identical network scores: the network divides each
  window by its own spread, so double noise and half-size movements are the
  same input. Expected, not a bug.
- Leave-one-behaviour-out (boosting): a behaviour never seen in training is
  still separated from reading (AUC 0.97–0.999), but fixation is then
  flagged 34% of the time (15% when seen) and video 21% (6%).
- Mean over the four conditions, development CV: boosting 88.8%, network
  94.6%, average of both 94.1% (dry run of `final.py`).

## Now: Kaggle (Karan, 2026-11-13: push freely, run on Kaggle GPUs)

`experiments/kaggle_run.sh` runs everything on a Kaggle GPU notebook:
downloads, caches, tables, boosting CV, network CV (3 seeds, 12 epochs, GPU),
then `final.py` picks the detector on development people and scores the
locked test people once. Output: `/kaggle/working/results/`
(`reading_detector_results.json`, logs, network weights).

Reading in the Wild: Karan said to request access with his email.

## Earlier: running unattended (`experiments/queue.sh`, log in `openmeet-data/logs/queue.log`), all finished

1. CNN 5-fold CV + final fit (`cnn_v1`), all four conditions and both
   outside datasets
2. `robust.py full` → `logs/robust_full.log`: does adding the 16 folding
   features (58 total) help under hard conditions?
3. `loto.py` → `logs/loto.log`: leave one non-reading behaviour out of training
4. tables for development people: slow reader (speed 2/3), window 5/20/30 s,
   5 and 15 fps, a second mixed draw; then their folding features
5. second CNN run, seed 1 (`cnn_v1_s1`) → `logs/cnn_v1_s1.log`

## Still to do

1. Read the logs above. Compare on development CV only: boosting 42,
   boosting 58, CNN, and the average of CNN and boosting scores.
2. Short script for the secondary tables: 5-fold CV on each `*_mixed` table
   (window length, frame rate); for `gazebase_slow_mixed`, score it with
   fold models trained on normal-speed data.
3. Pick the final model and thresholds from development results. Then the
   locked test: fit on all development people, score the 64 test people in
   all four conditions (tables contain everyone; CNN test scores are already
   in `*_final_cnn_v1.npy`), plus both outside datasets.
4. Deliverables: `results/reading_detector_results.json`, one chart,
   a README section (datasets, protocol, results, limits), a working.md
   decision-log row, torch in `requirements.txt`, the full pytest suite
   green, and a short memory update. Fold `experiments/` into one
   reproducible script if time allows. **Do not commit or push** (CLAUDE.md).

## Limits to state honestly in the write-up

- GazeBase reading is silent reading of a poem with a chin rest; negatives
  are lab tasks, not a person talking in an interview. Webcam noise is
  simulated, not measured on video.
- small_hard (text half-size and double noise) is near the floor at 10 fps.
- Only our own recordings (Phase 3) give the real number.
