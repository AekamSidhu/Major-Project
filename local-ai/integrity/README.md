# Answer-integrity check

Two signals an interviewer can watch while a candidate answers, served by
the local AI sidecar at `http://127.0.0.1:8765/integrity` (the main window's
**Local AI → Integrity check** button opens it):

1. **Reading.** Do the candidate's eyes move like someone reading lines of
   text? MediaPipe finds the irises in each video frame (the shared meeting
   window cropped to the candidate's tile, a webcam, or a screen region the
   sidecar captures), giving a gaze series at 10 fps. A small 1-D CNN scores
   each 10 s of it; an answer's score is the mean of its top half of window
   scores.
2. **AI-written answer.** Does the answer's transcript read like an LLM
   wrote it? TF-IDF + logistic regression over the transcript (lower case,
   words only, so formatting can't give it away).

A score is a reason to ask a follow-up question, never a verdict. Levels:
`possible` at the balanced threshold, `high` at the conservative one, both
chosen on development data only.

## Models (`models/`)

| File | What | Trained on | Held-out result |
|---|---|---|---|
| `reading_network.onnx` + `.json` | 1-D CNN, 3 seeds averaged, 2.1 MB | GazeBase (CC BY 4.0), 258 people, degraded to webcam quality | 98.8% window balanced accuracy on 64 unseen people; 83.0% on Reading in the Wild test people (real-world reading) |
| `echo_tfidf.json` | TF-IDF (1–2 word grams, 60k terms) + logistic regression, 1.9 MB | HC3 (CC BY-SA 4.0), human vs ChatGPT answers | 97.9% balanced accuracy on unseen questions (AUC 0.998) |

Training code and full results: `training/gaze-detection` (branch
`feature/gaze-detection`, `export_onnx.py`) and `training/echo-detection`
(branch `feature/echo-detection`, `export_content_model.py`).

**Not calibrated on real interviews yet.** GazeBase is lab eye tracking with
simulated webcam noise; HC3 answers are written text from 2022 ChatGPT. Our
own recordings (Phase 3 of both training plans) are needed before the
thresholds mean anything in a real interview.

## API

    GET  /v1/integrity/status        models ready, live gaze, last-10-s reading score
    POST /v1/integrity/echo          {"text": "..."}
    POST /v1/integrity/gaze/frame    one JPEG/PNG frame as the request body
    POST /v1/integrity/gaze/screen   {"left","top","width","height"}
    POST /v1/integrity/gaze/stop | /gaze/reset
    POST /v1/integrity/reading       {"start","end"} epoch seconds, or {"seconds"}

## Privacy

Frames are turned into one gaze sample each and dropped. Gaze samples stay
in memory for ten minutes and are never written to disk; transcripts sent
for scoring are not stored or logged. Everything runs on this computer.

## Packages

`onnxruntime` and `pillow` come with `requirements.txt`. Live gaze needs
`requirements-integrity.txt` (MediaPipe, mss); the setup scripts install it.
The face model (3.8 MB) downloads on first use into `.models/`.
