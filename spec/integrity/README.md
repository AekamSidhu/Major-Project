# Answer-integrity signals: shared definitions (spec v1)

The desktop app (Rust, `openmeet-intel` / `openmeet-engine`) and the local AI
sidecar and interview room (Python, `local-ai`) both compute these signals.
This folder is the single source of truth:

| File | What |
|---|---|
| `reference.py` | Reference implementation (standard library only). Regenerates the fixtures. |
| `thresholds.json` | Level thresholds and whether each signal is calibrated. Both sides read this file. |
| `delivery_cases.json` | Delivery features, baseline and score: inputs and expected outputs. |
| `fusion_cases.json` | Combined flag from per-signal levels. |
| `echo_cases.json` | Echo similarity from embeddings. |

Both implementations must reproduce every fixture to within `1e-6`. Change a
definition here first, bump `SPEC_VERSION`, run
`python -I spec/integrity/reference.py`, then update both implementations.

A flag is a **signal for the interviewer, never a verdict**. It always comes
with its evidence, and the human decides.

## Signals

| Name | Question | Computed by | Calibrated |
|---|---|---|---|
| `ai_text` | Does the answer's transcript read like AI-written text? | sidecar `POST /v1/integrity/echo` (TF-IDF + LR on HC3) | yes (public data) |
| `echo_similarity` | How close is the answer to what a local LLM answers to the same question? | the caller: reference answers from the LLM, embeddings, cosine | see `thresholds.json` |
| `delivery` | Does the answer's timing look read rather than spoken freely, compared with this speaker's own baseline? | the caller, from transcript timings | no |
| `reading` | Do the eyes move like reading lines of text during the answer? | sidecar `POST /v1/integrity/reading` (1-D CNN on gaze) | yes (public data) |

The `reading` signal is uncalibrated for answers shorter than 10 s: the sidecar scores 3 to 10 s of gaze with a hand-tuned early detector (`early: true`, `calibrated: false`), and it counts at most as `possible`.

Levels: `low`, `possible`, `high`, plus the not-scored levels `too_short`,
`baseline`, `no_gaze` and `unavailable`.

## Word timestamps (sidecar transcription)

`POST /v1/audio/transcriptions` follows OpenAI's API. Word timestamps are
requested with `response_format=verbose_json` and the repeated form field
`timestamp_granularities[]=word` (and/or `segment`). The response then has:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.34,
  "text": "so we cached the results",
  "segments": [{"id": 0, "start": 0.0, "end": 3.2, "text": "so we cached the results"}],
  "words": [{"word": "so", "start": 0.12, "end": 0.30}]
}
```

Times are seconds from the start of the uploaded audio. `words` appears only
when word granularity is requested and `segments` only when segment
granularity is requested (or no granularity is given with `verbose_json`).
Requests without these fields behave exactly as before. A caller places words
on the meeting clock by adding the utterance's start time.

## Delivery

Input per answer: `question_end` (seconds, or null when unknown) and the
answerer's utterances in the answer, `{start, end, text, words?}` in seconds
on one clock. Words are used only when every utterance has them.

| Feature | Definition |
|---|---|
| `granularity` | `word` if every utterance has words, else `utterance` |
| `word_count` | tokens: lower-case runs of `[a-z0-9']`, from the words (or the text) |
| `speaking_s` | sum of utterance durations |
| `span_s` | last utterance end minus first utterance start |
| `latency_s` | first utterance start minus `question_end`, at least 0; null if `question_end` is null |
| `rate_wpm` | `word_count / speaking_s * 60`; null if `speaking_s < 1.0` |
| `pause_per_min` | gaps of at least 0.3 s between consecutive units (words, or utterances), per minute of `span_s`; null if `span_s < 5.0` |
| `mean_pause_s` | mean of those gaps, 0 if none (evidence only) |
| `filler_per_100w` | 100 × fillers / `word_count`; fillers are the tokens `um umm uh uhm er erm ah hmm mm` plus the bigrams `you know`, `i mean`; null if no words |

**Baseline.** The first 3 answers with at least 10 words form the speaker's
baseline and get level `baseline`. For each weighted feature: median (mean of
the two middle values for an even count) and scale = max(1.4826 × MAD,
floor). A feature with fewer than 2 values has no baseline. Answers with fewer
than 10 words are `too_short` and never join the baseline.

**Score.** z = (x − median) / scale for each feature with a value and a
baseline. Index = Σ wᵢzᵢ / Σ|wᵢ| over those features, with weights
`latency_s +1.0`, `rate_wpm +0.5`, `pause_per_min −0.5`, `filler_per_100w
−0.5` and floors `0.5 s`, `15 wpm`, `3 per min`, `1 per 100 words`. Level
`high` at `high_index`, `possible` at `possible_index` (`thresholds.json`).
Whisper often drops fillers, so the filler rate is a weak feature.

## Echo similarity

Reference answers: the LLM answers the interviewer's question
`references` times (`thresholds.json`). The answer and each reference are
embedded with the same embedding model; similarity = the highest cosine
between the answer and any reference (0 when either vector is all zeros).
Levels from `possible` and `high` in `thresholds.json`. The thresholds hold
only for the embedding model named there; with another embedder the signal is
`unavailable`.

## The combined flag

Each signal contributes `{name, level, calibrated}`. Not-scored levels are
ignored. An uncalibrated signal counts at most as `possible`. The flag is:

- `high` when a calibrated signal is `high` **and** at least one other signal
  is `possible` or `high`;
- `possible` when any signal is `possible` or `high`;
- `low` when every scored signal is `low`;
- `unavailable` when no signal was scored.

One signal on its own never makes the flag `high`. The output lists the
signals that were elevated, so the interviewer sees why.
