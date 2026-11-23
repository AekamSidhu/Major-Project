# Answer-integrity check

Signals an interviewer can watch while a candidate answers. They are
served by the local AI sidecar at `http://127.0.0.1:8765/integrity` (the
main window's **Local AI → Integrity check** button opens it) and computed
per answer in the interview room (`../room/README.md`):

1. **Reading.** Do the candidate's eyes move like someone reading lines of
   text? MediaPipe finds the irises in each video frame (the shared meeting
   window cropped to the candidate's tile, a webcam, or a screen region the
   sidecar captures), giving a gaze series at 10 fps. A small 1-D CNN scores
   each 10 s of it; an answer's score is the mean of its top half of window
   scores.
2. **AI-written text.** Does the answer's transcript read like an LLM wrote
   it? TF-IDF + logistic regression over the transcript (lower case, words
   only, so formatting can't give it away).
3. **Echo similarity.** How close is the answer to what an LLM answers to
   the same question? The LLM writes 2 answers to the interviewer's
   question; the answer and those are embedded with
   `BAAI/bge-small-en-v1.5`; the highest cosine counts.
4. **Delivery.** Does the answer's timing look read rather than spoken
   freely, compared with the same candidate's first answers? Delay after the
   question, words a minute, pauses a minute and fillers, from Whisper's
   word timings.

A score is a reason to ask a follow-up question, never a verdict. Levels:
`possible` at the balanced threshold, `high` at the conservative one, both
chosen on development data only.

## One definition, two implementations

The signals' definitions, thresholds and test fixtures live in
`spec/integrity/` at the repository root, shared with the desktop app's Rust
engine. `signals.py` is this side's implementation (delivery features,
baseline and index, echo similarity, the fused flag, and the recipe for the
reference answers), standard library only; `test_signals.py` reproduces all
three fixtures to 1e-6. It finds the spec by walking up from this folder
(or `OPENMEET_INTEGRITY_SPEC`).

The four signals are fused into one flag (spec fusion): `high` only when a
calibrated signal is high and another signal is raised too; an uncalibrated
signal counts at most as `possible`; one signal alone never makes it
`high`. The flag lists the raised signals, so the interviewer sees why.

## Models and thresholds

| Signal | What | Trained or calibrated on | Held-out result |
|---|---|---|---|
| Reading | `models/reading_network.onnx` + `.json`: 1-D CNN, 3 seeds averaged, 2.1 MB | GazeBase (CC BY 4.0), 258 people, degraded to webcam quality | 98.8% window balanced accuracy on 64 unseen people; 83.0% on Reading in the Wild test people, measured with a separate network trained on that dataset (the shipped one has not been scored there) |
| AI-written text | `models/echo_tfidf.json`: TF-IDF (1–2 word grams, 60k terms) + logistic regression, 1.9 MB | HC3 (CC BY-SA 4.0) + interview Q&A (GPT-5 / GPT-4.1 / Claude 4.0 Sonnet style) | 98.0% balanced accuracy on unseen HC3 questions (AUC 0.998); 93.5% interview AI recall at 0% FPR |
| Echo similarity | thresholds in `spec/integrity/thresholds.json` (no trained model: Llama 3.2 + bge-small) | HC3: ChatGPT vs human answers against Llama 3.2's answers; thresholds (0.8712, 0.9201) from 500 development questions | 500 test questions: ROC-AUC 0.876; honest answers flagged 18.6% at `possible`, 6.6% at `high`; but 36% at `high` on computer-science definitions (textbook answers look alike) |
| Delivery | hand-set weights and thresholds in the spec | not calibrated | on SpiCE read sentences vs interview speech (34 speakers, 30-word stretches, no question delay measurable): ROC-AUC 0.55, and 28% of the interview stretches reach `possible` (not an interview false-positive rate, but a warning) |

Training code and full results: `training/gaze-detection`
(`export_onnx.py`) and `training/echo-detection` (`export_content_model.py`,
`calibrate_similarity.py`, `evaluate_delivery.py`).

### Echo detector — v2 retrained with interview data (2026-11-22)

The echo detector was **retrained** on HC3 + 54 interview Q&A pairs written
in the style of GPT-5, GPT-4.1, and Claude 4.0 Sonnet (the models Parakeet
AI uses). 38 interview questions went into training alongside HC3; 16 were
held out for calibration. The result is a single model that handles both
academic Q&A and interview-style answers.

| Dataset | Threshold | Value | FPR | AI recall |
|---------|-----------|-------|-----|-----------|
| HC3 test (unseen Qs) | `possible` | 0.247 | 5.0% | 99.2% |
| HC3 test (unseen Qs) | `high` | 0.462 | 1.8% | 97.8% |
| Interview (31 Qs, old calibration set) | `possible` | 0.247 | 0% | 93.5% |
| Interview (31 Qs, old calibration set) | `high` | 0.462 | 0% | 80.6% |
| Interview (16 held-out Qs) | `possible` | 0.247 | 0% | ~90% |

Mean AI score on interviews jumped from 0.22 (v1) to 0.72 (v2). Behavioural
questions like "Tell me about yourself" that v1 missed now score 0.4–0.9.
The fused flag (above) asks for both: a calibrated `high` plus another raised
signal. Two raised signals are therefore stronger evidence than one, but the
combination has not been measured on real interviews.

Training: `training/echo-detection/retrain_with_interviews.py`.
Calibration: `training/echo-detection/calibrate_interview.py`.

### Reading detector — synthetic interview calibration (2026-11-22)

The reading model was tested against synthetic gaze patterns matching the
Parakeet AI scenario: someone reading AI-generated text from an overlay
(left-to-right sweeps at 150–300 wpm) versus normal interview gaze
(looking at camera, thinking, gesturing, occasional glances).

| Scenario | Score range | At `high` (0.588) |
|----------|-----------|-------------------|
| Reading from overlay (8 scenarios, 15–60 s) | 1.000 | 100% detected |
| Normal interview gaze (8 scenarios, 15–60 s) | 0.007–0.347 | 0% false positives |

On these synthetic patterns the existing thresholds separate the two cases at
the `high` level. At `possible` (0.140) some nervous or fidgety speakers score
above threshold, so `high` is the level to use in interviews. These are
generated patterns, not recordings of candidates, so they say little about
real false-positive rates.

Calibration script: `training/gaze-detection/calibrate_reading.py`.

**Not fully calibrated on real interviews yet.** GazeBase is lab eye tracking
with simulated webcam noise; HC3 answers are written text from 2022 ChatGPT;
SpiCE's read speech is isolated sentences, not a read-out answer. Our own
recordings (`training/data-collection`, the room's study mode) are needed
before the thresholds mean anything in a real interview.

## API

    GET  /v1/integrity/status        models ready, live gaze, last-10-s reading score
    POST /v1/integrity/echo          {"text": "..."}  (the AI-written-text model)
    POST /v1/integrity/gaze/frame    one JPEG/PNG frame as the request body
    POST /v1/integrity/gaze/screen   {"left","top","width","height"}
    POST /v1/integrity/gaze/stop | /gaze/reset
    POST /v1/integrity/reading       {"start","end"} epoch seconds, or {"seconds"}

Delivery needs word timings: the sidecar's `POST /v1/audio/transcriptions`
returns them as OpenAI does (`response_format=verbose_json` with
`timestamp_granularities[]=word` and/or `segment`). The decoding is the
same as for a plain request; a clip that fits one 30 s window keeps its
first decode, because faster-whisper otherwise re-decodes the silent tail
and invents words. The caller (the room, or the desktop app) computes
delivery and echo similarity with the spec's definitions.

## Privacy

Frames are turned into one gaze sample each and dropped. Gaze samples stay
in memory for ten minutes and are never written to disk; transcripts sent
for scoring are not stored or logged. Everything runs on this computer,
except reference answers in the room's online mode (the interviewer's
question only, sent to Gemini).

## Packages

`onnxruntime` and `pillow` come with `requirements.txt`. Live gaze needs
`requirements-integrity.txt` (MediaPipe, mss); the setup scripts install it.
The face model (3.8 MB) downloads on first use into `.models/`.
