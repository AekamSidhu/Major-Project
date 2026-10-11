# Gaze detection — `feature/gaze-detection`

Work branch for building, training and evaluating the **reading-gaze** ("eye
detection") signal of OpenMeet's answer-integrity check (cheating detection).
The other signal, AI-echo answer analysis, is developed separately on
`feature/echo-detection`. The two are combined only after both are merged.

> Status: the code for Phases 0–2 is in place: the environment, the GazeBase
> loader, webcam simulation, reading features, rule-based and learned
> detectors, and the MediaPipe video pipeline, covered by 41 tests. **No
> training has been run yet.** Nothing from this branch is wired into the app.
> The `IntegrityAnalyzer` and `GazeProvider` exist only in the design diagrams
> (process 11.0 "Assess answer integrity" in the DFDs, steps 11.5–11.6). See
> [Getting started](#getting-started).

## What we're detecting

Eyes move in a recognisable way when someone reads text off a screen. They
step left to right along a line, then snap back to the start of the next
line. Plotted over time, horizontal gaze looks like a sawtooth.

While the candidate answers, OpenMeet watches their video tile in the meeting
window and asks: are the eyes doing that reading pattern, or holding on one
off-camera spot for long stretches?

Looking away on its own is **not** the signal. People naturally look up or
away when they think. We look specifically for *reading*.

## How the signal works (planned)

1. **Find the face.** Capture the candidate's video tile from the meeting
   window at about 10 fps. This is the same picture the interviewer already
   sees, and nothing is installed on the candidate's machine. Crop to the
   face.
2. **Landmarks.** Run MediaPipe Face Landmarker in the local AI sidecar. Its
   478 landmarks include the irises, and its blendshape scores include
   eye-look directions, which give a cheap second gaze estimate.
3. **Gaze direction.** Compute the iris position relative to the eye corners
   and lids, giving horizontal and vertical gaze ratios. Read head pose from
   MediaPipe's facial transformation matrix, so head turns aren't mistaken
   for eye movement. Drop blink frames.
4. **Reading pattern.** Build a gaze time series over each answer and
   extract features:
   - **return sweeps**: large, fast right-to-left jumps after slower
     left-to-right drift
   - **regularity**: whether the sweeps repeat at a similar rhythm, since
     lines of text are similar lengths
   - **off-camera fixation**: share of the answer spent looking at one
     off-screen spot, compared with the same person while listening
5. **Score.** Turn the features into a per-answer reading score.

Start rule-based: a sawtooth detector plus thresholds, which is easy to
explain. Then try a learned model, first logistic regression on windowed
features, and a small 1D-CNN or GRU only if there is enough data.

## Known hard parts

Be ready to explain these in the report and the viva:

- **Frame rate.** At 10 fps we can't see individual saccades, which last only
  tens of milliseconds. Return sweeps happen once per line, every few
  seconds, which 10 fps captures easily (it resolves motion up to 5 Hz).
  The detector must rely on line-level patterns, not saccades.
- **Resolution and compression.** In gallery view the face may be only
  100–200 px wide, which leaves each eye just a few pixels. Measure how
  accuracy drops as the tile shrinks.
- **Glasses, reflections, lighting, dark irises and camera position** (a
  laptop camera below the screen versus an external webcam).
- **Text right next to the camera.** A cheater can place the answer just
  under the webcam, which makes the sweeps small. Record this condition and
  report the result honestly.
- **Finding the right tile.** Meeting apps lay out video differently. Version
  1 uses the screen-region selection OpenMeet already has: the interviewer
  marks the candidate's tile.
- **Capture rate.** The app's screen sampler runs at 1 fps by default with
  change detection. Gaze needs a separate 10 fps sampler for the tile only,
  and only while the candidate is answering, to keep CPU use low.

## Datasets

Public data lets us build the gaze tracker and the reading detector before
our own recordings exist. None of it shows exactly our case (a candidate in a
video call reading an AI answer), so our own collection is still required.

| Dataset | What it is | Access | How we use it |
|---|---|---|---|
| [OEP (Online Exam Proctoring)](https://cvlab.cse.msu.edu/oep-dataset.html) | 24 people acting out several kinds of cheating in online exams, recorded by webcam, wearable camera and microphone (Atoum et al., *Automated Online Exam Proctoring*) | Download from the MSU page; cite the paper | Closest public match to our task: an external test set on real webcam footage of cheating |
| [EVE](https://ait.ethz.ch/eve) | Webcam video synchronised with screen recordings and ground-truth gaze, covering natural eye movements ([code](https://github.com/swook/EVE)) | Request form; CC BY-NC-SA 4.0 with extra terms | Check our MediaPipe gaze ratios against true gaze, then degrade the video (smaller, compressed, 10 fps) to see where the signal breaks |
| [MPIIFaceGaze](https://doi.org/10.18419/darus-3240) | 37,667 face images from laptop webcams in everyday conditions, with gaze labels ([MPI page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)) | Open download; CC BY-NC-SA 4.0 | Validate gaze direction across lighting, glasses and real laptop cameras |
| [GazeBase](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257) | 322 people recorded by a 1000 Hz eye tracker on seven tasks, including **reading**, fixation and video viewing | Open download on figshare; also loadable with pymovements | Real reading scanpaths. Downsample to 10 Hz and add webcam-level noise to create realistic "reading" vs. "not reading" traces for the detector |
| [GECO (Ghent Eye-tracking Corpus)](https://expsy.ugent.be/downloads/geco/) | English monolinguals and Dutch–English bilinguals reading a whole novel, word-level eye-movement measures | Open download (Excel files) | Reading statistics (fixation durations, line timing) and a native vs. non-native reader comparison for the bias check |
| [MIT Interview Dataset](https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances/) | 138 recorded mock job interviews (69 students), with video, audio and transcripts | Request form; needs an academic email | Natural gaze while answering interview questions, including looking away to think (hard negatives) |

Tools: [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
([Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)),
OpenCV (head pose, video re-encoding), and
[pymovements](https://github.com/aeye-lab/pymovements) (loads eye-tracking
datasets such as GazeBase, and detects fixations and saccades).

## Our own data

Mock interviews with volunteers, held over a real call so the video goes
through real compression. Record the candidate's tile **from the
interviewer's screen**, exactly what OpenMeet would see.

| Condition | Label |
|---|---|
| Answers naturally, looking at the screen or camera | not reading |
| Thinks with natural gaze aversion (up, away) | not reading (hard negative) |
| Glances at their own notes now and then | separate label (whether this counts depends on the interview's rules) |
| Reads an AI answer: near the camera, side of the screen, second monitor, phone | reading |

Vary glasses or no glasses, lighting, webcams, and tile size (gallery view
versus speaker view).

**One collection for both branches.** Each session records the candidate's
audio as well, so the same recordings and labels feed the echo branch too.
Plan and run data collection together, once.

**Data rules**

- Face video is sensitive personal data. Get written consent from every
  participant. Anyone can withdraw and have their data deleted.
- **Never commit video, frames, landmark dumps or anything else tied to a
  participant.** Keep them in the team's private storage. Commit only code
  and aggregate results. Add `training/*/data/` to `.gitignore` before the
  first recording lands in this folder.
- We never identify anyone: no face recognition, no stored face templates.
  Use anonymous session ids.

## How we'll measure it

- Split train and test **by person**, so no participant appears in both.
- Report ROC-AUC, precision and recall, and the **false-positive rate at the
  chosen threshold**.
- Robustness curves: accuracy against face size in pixels and against frame
  rate (5, 10, 15 fps).
- Per-subgroup results: glasses, lighting, camera position, native vs.
  non-native readers.
- Per-frame processing time on a laptop CPU. It must keep up with 10 fps.

## Implementation plan

Each phase ends with something we can show. Dates go in once we know the
project deadlines.

### Phase 0: Set up

- Python environment for this folder (mediapipe, which brings OpenCV, plus
  numpy, scipy, pandas, scikit-learn and matplotlib).
- Add `training/*/data/` to `.gitignore`.
- Write down the evaluation protocol (person-level split, metrics, target
  false-positive rate) **before** looking at any results.

*Done when:* the skeleton scripts run end to end on a short test video.

### Phase 1: Gaze from video (MediaPipe; EVE, MPIIFaceGaze)

- Wrap Face Landmarker in video mode. Output per-frame horizontal and
  vertical gaze ratios for each eye, head pose, and a blink flag.
- Compare our ratios with ground-truth gaze on EVE and MPIIFaceGaze. We need
  *relative* movement (left vs. right), not calibrated screen coordinates,
  so measure correlation, not absolute error.
- Degradation study: shrink faces to 200, 150, 100 and 75 px wide, re-encode
  at low bitrate with ffmpeg, and resample to 5, 10 and 15 fps. Find the
  smallest tile where horizontal sweeps are still visible.

*Done when:* a short gaze-quality report exists, stating the minimum usable
tile size.

### Phase 2: Reading detector on eye-movement data (GazeBase, GECO)

- Take GazeBase reading and non-reading tasks, downsample from 1000 Hz to
  10 Hz, and add noise at the level measured in Phase 1. This gives
  webcam-quality "reading" and "not reading" traces.
- Build the rule-based sawtooth detector: find return sweeps, count them per
  minute, and measure the regularity of the gaps between them.
- Train the learned baseline: windowed features into logistic regression.
- Use GECO to sanity-check line timing and compare native and non-native
  readers.

*Done when:* we have a ROC-AUC on the simulated traces and a chosen detector
configuration.

### Phase 3: Collect our own data (shared with the echo branch)

- Write the protocol, consent form and condition schedule. Randomise the
  condition order per participant.
- Pilot with 2–3 teammates, then fix whatever was awkward.
- Record over a real call from the interviewer's side, and keep a label
  sheet per answer.
- Keep OEP aside as an external test set.

*Done when:* the dataset is collected and documented in a data card (in git,
no raw data).

### Phase 4: Evaluation

- Run the full pipeline (video → landmarks → gaze series → reading score) on
  our recordings with the person-level split.
- Choose the threshold for a target false-positive rate (e.g. ≤ 5%).
- Produce ROC curves, the false-positive rate at the threshold, robustness
  curves, per-subgroup results, the OEP external-test result, per-frame CPU
  time, and a look at every false positive.

*Done when:* results and plots in `results/` are ready for the report.

### Phase 5: Integrate into OpenMeet

- `local-ai`: a loopback-only `POST /v1/gaze` endpoint. It takes a face crop
  and returns landmarks, gaze ratios and head pose, and never logs or stores
  frames. Add `mediapipe` to `requirements.txt`, with tests on synthetic
  images.
- `openmeet-core`: a `GazeProvider` trait and gaze types, alongside the
  existing provider traits.
- `openmeet-providers`: a `LocalAiGaze` adapter for the endpoint.
- `openmeet-platform` / `openmeet-engine`: a 10 fps sampler for the selected
  candidate tile, running only while the candidate is answering.
- `openmeet-engine`: the `IntegrityAnalyzer` builds the gaze series and
  computes the reading score in Rust (rules plus logistic weights from a
  small JSON file). The sidecar stays stateless.
- Settings: gaze on/off, **off by default**. The privacy page shows it when
  it's on.

*Done when:* an engine test with a mock gaze provider covers the flow end to
end, and a live test runs it with the real sidecar.

### Phase 6: Fuse with the echo signal (joint, after both branches merge)

Combine the reading and echo scores into one flag, re-tune the threshold on
the shared recordings, and report how much each signal adds.

## Getting started

Python 3.11–3.13. From this folder (`training/gaze-detection`):

```bash
py -3.13 -m venv .venv              # macOS/Linux: python3 -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python download_model.py            # MediaPipe face model, 3.8 MB, into models/
python -m pytest                    # 41 tests, about 20 s
```

`.venv/`, `data/` and `models/` are git-ignored. This folder sits inside
OneDrive, which will try to sync the virtual environment and the 6.7 GB
dataset. Exclude them from OneDrive, or keep the dataset elsewhere and pass
`--data`.

### Running the pipeline

```bash
# Plumbing check on synthetic traces (no download; the numbers mean nothing)
python prepare.py --synthetic 8
python train.py --features data/features/synthetic_10fps.csv --out data/smoke

# The real thing
python download_gazebase.py                # 6.7 GB, resumes if interrupted
python prepare.py --max-participants 10    # quick first look
python prepare.py                          # all of GazeBase → data/features/gazebase_10fps.csv
python train.py                            # → results/cv_report.json, results/model.json
python prepare.py --noise 3 --out data/features/gazebase_noisy.csv   # robustness runs

# Held-out data (our own recordings, OEP), at the saved threshold
python evaluate.py --model results/model.json --features data/features/<held-out>.csv

# Try it on a video, e.g. a clip of yourself reading vs. talking
python extract_video.py clip.mp4                       # rule-based, no training needed
python extract_video.py meeting.mp4 --roi 640 0 640 360 --model results/model.json
```

### Code layout

```
training/gaze-detection/
  readgaze/                 the library
    trace.py                GazeTrace: the common format; resampling, gap filling, windows
    simulate.py             lab eye-tracker trace → webcam-quality trace
    synthetic.py            synthetic traces for tests and smoke runs only
    datasets/gazebase.py    GazeBase loader (reads the zip directly, nested zips included)
    features.py             the six reading features, plus sweep times as evidence
    detector.py             rule-based detector; window scores → one answer score
    model.py                logistic regression; JSON export for the Rust engine
    evaluation.py           threshold at a target false-positive rate; metrics
    pipeline.py             recordings → feature table (one row per window)
    training.py             person-level cross-validation; final fit
    landmarks.py            MediaPipe Face Landmarker → iris offsets, blinks, head pose
    video.py                video file → gaze trace
  download_gazebase.py      dataset download (into data/, git-ignored)
  download_model.py         face model download (into models/, git-ignored)
  prepare.py                build the feature table
  train.py                  cross-validate and export model.json
  evaluate.py               score held-out data at the saved threshold
  extract_video.py          run the detector on a video
  tests/                    pytest suite
  results/                  aggregate metrics and model.json (committed)
```

## Evaluation protocol

Fixed before any results are seen:

- **Unit of decision: one answer.** Gaze is cut into 10 s windows every 5 s,
  and the answer's score is the mean of its top half of window scores. On
  GazeBase, one recording stands in for one answer.
- **Positives and negatives.** GazeBase's reading task (TEX) is positive.
  Every other task is negative, and flag rates are reported per task. The
  horizontal-saccade task (HSS) is the hardest negative.
- **Split by person.** 5-fold `StratifiedGroupKFold`, grouped by participant
  ID. GazeBase participants return across rounds, so grouping by round would
  leak.
- **Operating point.** The threshold allows at most **5% false positives** on
  honest recordings. Recall at that threshold is the headline number;
  ROC-AUC comes second.
- **Both detectors, every time.** The learned model is only worth keeping if
  it beats the rule-based one.
- **Robustness.** Repeat with doubled noise (`--noise 3`) and halved eye
  movements (`--amplitude 0.5`, text near the camera).
- **Known gap.** GazeBase is silent reading of a poem with the head on a chin
  rest. Our case is reading aloud from a chat window with the head free. The
  simulation narrows that gap but doesn't close it, so only our own
  recordings (Phase 3) give the real numbers.

## Training in the cloud

Yes, it can be trained in the cloud, but it doesn't need to be. Training is
logistic regression on a few thousand rows of six numbers, which takes
seconds on any laptop, and there is no GPU work at all. The slow step is
`prepare.py` working through the 6.7 GB GazeBase download, which is CPU and
disk bound. A GPU only matters later, if we try a 1D-CNN or GRU, or run
MediaPipe over many hours of video.

Free options: Google Colab or Kaggle Notebooks, or a university lab or HPC
machine. In a Colab notebook:

```bash
!git clone -b feature/gaze-detection https://github.com/AekamSidhu/Major-Project.git
%cd Major-Project/training/gaze-detection
!pip install -q -r requirements-dev.txt
!python -m pytest -q
!python download_gazebase.py
!python prepare.py
!python train.py
```

Then download `results/cv_report.json` and `results/model.json` and commit
them from your own machine.

**Data rule for the cloud:** public datasets (GazeBase is CC BY 4.0) are
fine. Our own recordings of participants are not, unless the consent form
explicitly allows cloud processing and the university's data policy permits
that service. Otherwise, process them locally: everything here runs on a
laptop.

## Ground rules

- A flag is a **signal for the interviewer, never a verdict**. It always
  comes with evidence (timestamps, the gaze trace), and the human decides.
- Missing a cheater is better than flagging an honest candidate. Choose
  thresholds accordingly.
- In the app, frames are processed in memory and discarded. They never go to
  a cloud provider.

## Checklist

- [x] Phase 0: environment, `.gitignore`, evaluation protocol written
- [ ] Phase 1: gaze-quality report and minimum tile size (MediaPipe pipeline ready)
- [ ] Phase 2: reading detector measured on simulated traces (code ready, not yet run on GazeBase)
- [ ] Phase 3: own dataset collected (with the echo branch)
- [ ] Phase 4: evaluation, threshold and false-positive rate, robustness and bias checks
- [ ] Phase 5: integrated into the app behind a setting
- [ ] Phase 6: fused with the echo signal
