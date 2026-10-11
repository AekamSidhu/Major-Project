# Echo detection — `feature/echo-detection`

Work branch for building, training and evaluating the **AI-echo** signal of
OpenMeet's answer-integrity check (cheating detection). The other signal, reading
gaze ("eye detection"), is developed separately on `feature/gaze-detection`.
The two are combined only after both are merged.

> Status: research and training. Nothing from this branch is wired into the
> app yet. The `IntegrityAnalyzer` exists only in the design diagrams
> (process 11.0 "Assess answer integrity" in the DFDs, steps 11.1–11.4).

## What we're detecting

In a remote interview, a candidate can paste the question into an AI
assistant, or let a tool transcribe it automatically, and then read the
generated answer aloud. The echo signal looks at each answer in two ways:

1. **Content.** Is this answer close to what an LLM would say to the same
   question?
2. **Delivery.** Does the candidate sound like they're reading, compared with
   how *this person* normally talks?

Neither signal is enough on its own. A well-prepared candidate can give a
textbook answer, and a fluent speaker can sound scripted. That's why we
combine them and compare each person against their own baseline.

## How the signal works (planned)

**Content: echo score (steps 11.3, 11.4)**

- When the interviewer's question ends, ask the LLM for 3–4 reference answers
  to it, sampled with temperature > 0 so they vary.
- When the candidate's answer turn ends, embed the answer and the references.
- Echo score = highest cosine similarity between the answer and any
  reference.

**Delivery: anomaly against the speaker's baseline (steps 11.1, 11.2)**

- Baseline: the candidate's first few minutes of small talk. Measure speech
  rate, pause pattern and filler rate.
- For each answer, compare against the baseline:
  - **Start delay** after the question ends. Reading an AI answer means
    waiting for the answer to be generated first.
  - **Speech-rate evenness.** Reading aloud is unusually steady.
  - **Fillers and self-corrections** ("um", "uh", "I mean", restarts). People
    who are reading use fewer of these.
  - **Where pauses fall.** Thinking pauses come at clause boundaries. Reading
    pauses come mid-sentence, when the reader loses their place or scrolls.
- This needs word-level timestamps from Whisper. The local AI sidecar doesn't
  return them yet. faster-whisper supports `word_timestamps=True`, so exposing
  them is the first code change on this branch.

**Combine**

Echo score plus delivery features go into a small classifier, which outputs
the probability that the answer was read from an AI. Start with logistic
regression: it is explainable, gives calibrated probabilities and works with
little data. Only move to anything bigger if the data justifies it.

## What "training" means here

We are **not** training or fine-tuning a language model. The LLM, the
embedding model and Whisper are used as they are. What we build and tune:

- which features to extract, and how (window sizes, normalisation)
- a small classifier on top of those features
- the decision threshold, chosen for a low false-positive rate

## Data we need

Mock interviews with volunteers, held over a real call (Zoom, Meet or Teams)
so the audio goes through real compression. Each session starts with 2–3
minutes of small talk for the baseline.

| Condition | Label | Why it's in the set |
|---|---|---|
| Answers naturally | genuine | the normal case |
| Reads an AI answer off the screen | echo | the case we want to catch |
| Gives a rehearsed or memorised answer | genuine (hard negative) | fluent, but not reading |
| Gives a textbook answer to a factual question | genuine (hard negative) | content close to an LLM's |
| Paraphrases an AI answer on the fly | echo (hard positive) | the adaptive cheater |

Include native and non-native English speakers, different accents, and both
nervous and relaxed speakers. Target size: to be agreed with the team.

**Data rules**

- Get written consent from every participant. Anyone can withdraw and have
  their data deleted.
- **Never commit recordings, transcripts or any participant data to git.**
  Keep them in the team's private storage. Commit only code, feature
  definitions and aggregate results. Add `training/*/data/` to `.gitignore`
  before the first recording lands in this folder.
- Use anonymous session ids, never names.

## How we'll measure it

- Split train and test **by person**, never by answer, so no speaker appears
  in both.
- Report ROC-AUC, precision and recall, and the **false-positive rate at the
  chosen threshold**: how often an honest candidate gets flagged. That's the
  number that matters most.
- Break results down by subgroup (non-native speakers, hard negatives) to
  check for bias.
- Ablation: content only, delivery only, both.
- Cross-model check: generate references with the local model (e.g.
  `llama3.2`) while the "cheater" used a different assistant (GPT, Claude,
  Gemini). Does the echo score still separate the classes?

## Planned layout

```
training/echo-detection/
  README.md      this file
  data/          local only, never committed
  features/      text and timing feature extraction
  notebooks/     exploration
  train.py       classifier training
  evaluate.py    metrics and plots for the report
  results/       aggregate metrics only
```

## Where it plugs into the app (later)

- `local-ai/server.py`: return word timestamps from transcription.
- `openmeet-intel`: the reference-answer prompt (versioned and tested like
  the other prompts), feature extraction and scoring, all pure and
  unit-tested.
- `openmeet-engine`: an `IntegrityAnalyzer` that runs when an answer turn
  ends, in parallel with follow-up generation. It reuses the configured
  `LlmProvider` and `EmbeddingProvider` and emits an integrity-flag event to
  the overlay.
- The trained weights and threshold ship as a small config file, so this
  signal needs no Python at runtime.

## Ground rules

- A flag is a **signal for the interviewer, never a verdict**. It always
  comes with evidence (scores, timestamps), and the human decides.
- Missing a cheater is better than flagging an honest candidate. Choose
  thresholds accordingly.
- Runs locally by default, like the rest of OpenMeet, and only when the
  interviewer turns it on.

## Done when

- [ ] Sidecar returns word timestamps
- [ ] Dataset collected under the data rules above
- [ ] Features and classifier trained, with results on held-out speakers
- [ ] Threshold chosen and its false-positive rate documented
- [ ] Per-subgroup bias check written up
- [ ] Integrated into the engine behind a setting
- [ ] Fused with the gaze signal (after both branches are merged)
