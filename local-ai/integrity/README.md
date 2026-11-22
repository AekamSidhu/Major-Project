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
| Reading | `models/reading_network.onnx` + `.json`: 1-D CNN, 3 seeds averaged, 2.1 MB | GazeBase (CC BY 4.0), 258 people, degraded to webcam quality | 98.8% window balanced accuracy on 64 unseen people; 83.0% on Reading in the Wild test people (real-world reading) |
| AI-written text | `models/echo_tfidf.json`: TF-IDF (1–2 word grams, 60k terms) + logistic regression, 1.9 MB | HC3 (CC BY-SA 4.0), human vs ChatGPT answers | 97.9% balanced accuracy on unseen questions (AUC 0.998) |
| Echo similarity | thresholds in `spec/integrity/thresholds.json` (no trained model: Llama 3.2 + bge-small) | HC3: ChatGPT vs human answers against Llama 3.2's answers; thresholds (0.8712, 0.9201) from 500 development questions | 500 test questions: ROC-AUC 0.876; honest answers flagged 18.6% at `possible`, 6.6% at `high`; but 36% at `high` on computer-science definitions (textbook answers look alike) |
| Delivery | hand-set weights and thresholds in the spec | not calibrated | on SpiCE read sentences vs interview speech (34 speakers, 30-word stretches, no question delay measurable): ROC-AUC 0.55, and 28% of the interview stretches reach `possible` (not an interview false-positive rate, but a warning) |

Training code and full results: `training/gaze-detection`
(`export_onnx.py`) and `training/echo-detection` (`export_content_model.py`,
`calibrate_similarity.py`, `evaluate_delivery.py`).

### Echo detector — interview calibration (2026-11-22)

The echo detector's thresholds have been calibrated against the LLM output
that tools like Parakeet AI generate (GPT-5, GPT-4.1, Claude 4.0 Sonnet).
A set of 31 common interview questions was answered in both the AI style
those models produce and in natural human speech; the model's thresholds
were set so that no honest answer is flagged while catching as many
AI-generated answers as possible.

| Threshold | Value | Human false-positive rate | AI recall |
|-----------|-------|--------------------------|-----------|
| `possible` | 0.059 | 0% (0/31) | 68% |
| `high` | 0.108 | 0% (0/31) | 52% |

Technical interview answers (processes vs threads, REST APIs, system design)
are detected reliably (scores 0.3–0.8). Behavioural answers (strengths,
failures, career goals) have a weaker signal because modern LLMs write them
more naturally. The echo detector is strongest when combined with the reading
detector: a candidate whose eyes track like reading **and** whose answer
scores as AI-written is a much stronger signal than either alone.

Calibration script: `training/echo-detection/calibrate_interview.py`.

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
