# Answer integrity (in the app)

An opt-in check for **interviewer mode**, off by default. After each
candidate answer OpenMeet computes four signals in the background, combines
them into one flag and shows it privately in the overlay, with its evidence:

> **Possible AI-assisted answer — worth a follow-up**
> Answer at 23:54–25:00 · "How does consistent hashing help…"
> AI-written text score 0.91 (high) · Close to a local LLM's answer:
> similarity 0.88 (uncalibrated) · Started 3.1 s later than usual ·
> Eye movements look like reading: score 0.87 (high)

A flag is a **signal for the interviewer, never a verdict**: a reason to ask a
follow-up question. The definitions, thresholds and test fixtures are shared
with the local AI sidecar in [`spec/integrity/`](../spec/integrity/README.md).
The sidecar's own browser page (`/integrity`) is described in
[`local-ai/integrity/README.md`](../local-ai/integrity/README.md).

## Signals

| Signal | Question | Computed by | Calibrated |
|---|---|---|---|
| `ai_text` | Does the transcript read like AI-written text? | Local AI sidecar, `POST /v1/integrity/echo` | yes (public data) |
| `echo_similarity` | How close is the answer to what an LLM answers to the same question? | The engine: the configured LLM writes `references` answers when the question is detected; answer and references are normalised like a transcript and embedded with the configured embedder; highest cosine | as `thresholds.json` says, and only with references from its `reference_model` |
| `delivery` | Does the timing look read rather than spoken freely, compared with this candidate's own first answers? | The engine, from transcript timings (word timestamps when the recognizer gives them) | as `thresholds.json` says (not yet) |
| `reading` | Do the eyes move like reading during the answer? | Local AI sidecar, `POST /v1/integrity/reading` for the answer's time window, while it captures the candidate's video region (`/gaze/screen`) | yes (public data) |

Levels are `low`, `possible`, `high`, plus the not-scored `too_short`,
`baseline`, `no_gaze` and `unavailable` (each with a note saying why, e.g.
"Set the candidate's video region", "Needs the BAAI/bge-small-en-v1.5
embedding model"). Echo is `unavailable` unless the embedder is the model
named in `thresholds.json` (Settings → AI providers → Embeddings: Local AI).

**Combined flag** (spec "The combined flag"): not-scored levels are ignored;
an uncalibrated signal counts at most as `possible`; the flag is `high` only
when a calibrated signal is `high` and another signal is raised — one signal
alone never makes it `high`. With no scored signal the flag is `unavailable`.

## When an answer is checked

* An **answer** is the candidate's speech from an interviewer question until
  the next interviewer question (or the end of the meeting).
* Whenever the candidate pauses (the turn-end silence, Settings → assistant),
  the answer so far is checked as a background request. If the candidate
  continues, the same answer is checked again and its flag replaced
  (`revision`); an older revision never overwrites a newer one.
* Delivery is computed at that moment, in order, against the baseline: the
  first three *finished* answers with at least ten words. Those answers show
  "Learning this candidate's usual delivery (answer n of 3)".
* Reference answers are requested from the LLM as soon as the question is
  detected, so a local model is free again before follow-ups are generated.
  They follow the shared **reference recipe v1** the echo thresholds were
  measured with (`local-ai/integrity/signals.py`): system prompt "You are a
  helpful assistant. Answer the user's question directly, in plain prose
  without headings or lists, in about 150 words.", the trimmed question as the
  user message, temperature 0.7, at most 400 tokens, `references` answers
  (two) with seeds 0 and 1 where the provider takes a seed (OpenAI-style APIs,
  Ollama). Before embedding, the answer and every reference go through
  `to_transcript` (a port of `local-ai/integrity/text.py`: no URLs, markdown or
  punctuation, lower case, no apostrophes). Checks never use the UI request
  slots; an emergency stop cancels them.
* Echo is `unavailable` unless the embedder is the model in `thresholds.json`
  (`BAAI/bge-small-en-v1.5`, Local AI embeddings). With references from
  another LLM than its `reference_model` (`llama3.2`) the signal is reported
  as uncalibrated (it then counts at most as `possible`) and its evidence says
  which model wrote the references.
* With word timestamps an utterance spans its first word's start to its last
  word's end, for the candidate's utterances and for the end of the question
  (as the sidecar and the room do).

## Settings

Settings → **Answer integrity**: the switch (off by default), one toggle per
signal, the candidate video region (a display plus a rectangle on it, with
presets and a preview captured the way the sidecar sees it), the Local AI
status, a reminder to tell the candidate, and the limits below. With the check
on, Local AI is started for interviewer meetings even if transcription runs in
the cloud. Word timestamps (`verbose_json` + `timestamp_granularities[]=word`)
are requested from the transcriber only while the delivery signal is on; a
server that rejects them is asked plainly from then on, and OpenAI models
other than `whisper-1` are never asked.

## Where it shows

* **Overlay**: an "Answer check" card for the latest answer — raised flags
  with their evidence, quiet results as one line; can be hidden until the
  answer is re-checked.
* **Live view**: every checked answer of the meeting.
* **History**: an "Answer integrity" tab; **export** has an "Answer
  integrity" part (Markdown, text, JSON).
* **Privacy page**: each part (AI-text and reading check, video region,
  reference answers, answer embeddings) with its Local / Cloud location.
  Details in [PRIVACY.md](../PRIVACY.md#answer-integrity-check-interviewer-mode).
* **Storage**: table `integrity_flags` (migration 2), deleted with its meeting,
  by *Delete all meeting data* and by retention.

## Limits

* **Thresholds are not calibrated on real interviews.** AI-text, reading and
  echo similarity are thresholded on public data (HC3, GazeBase); delivery
  thresholds are hand-set and marked uncalibrated. Which signals count as calibrated is read from
  `thresholds.json`.
* Fluent or prepared speakers, accents, transcription errors and nerves can
  raise signals without any AI help. Whisper often drops filler words.
* Without word timestamps, utterance timings include the voice-activity
  padding (about 0.24 s before and 0.2 s after the speech), which shortens
  measured latency and lowers the speaking rate a little — evenly for all of a
  candidate's answers.
* Reading needs about 10 s with the face tracked; shorter answers are
  `no_gaze`. The sidecar scores at most the last 600 s of an answer.
* Speaker separation comes from the audio source (system audio = candidate).
  Without system audio, all speech is "unknown" and the answer may include
  the interviewer.
* The candidate region is in physical pixels of the virtual desktop
  (Windows); on macOS the sidecar's capture may use points instead.

## Demo and tests

* Demo mode → **Answer integrity check (interviewer)**: three natural answers
  form the baseline, the fourth is read out; runs without devices, models or
  network (scripted sidecar, a stand-in embedder named like the spec model).
* `crates/openmeet-intel/tests/integrity_spec.rs`: parity with the three spec
  fixtures (to 1e-6), the spec constants, and `to_transcript` against outputs
  of the Python function. The fixtures and thresholds are read from the files,
  so the tests pass with the provisional and with the calibrated spec.
* `crates/openmeet-engine/tests/integrity.rs`: question → answer through the
  real VAD → all four signals → fused flag → event → stored.
* Provider mock-server tests for the sidecar client and word timestamps; store
  migration, cascade and revision tests; vitest for the card and reducers.

## Notes on the spec

* The sidecar reports `not_enough_gaze` for too little gaze; the app maps it
  to the spec's `no_gaze`.
* `thresholds.json` names a `reference_model` (`llama3.2`); the app uses the
  configured LLM whatever it is and reports echo as uncalibrated when that LLM
  is a different model (an Ollama `:latest` tag counts as the same model).
* The reference prompt is the bare recipe, without the data-block framing the
  other prompts use: changing it would invalidate the thresholds, and its
  output is only embedded, never shown or acted on.
* The delivery definition constants (weights, floors, minimums, fillers) live
  in `openmeet-intel` and a test checks them against the `constants` block of
  `delivery_cases.json`; thresholds are read from `thresholds.json`.
