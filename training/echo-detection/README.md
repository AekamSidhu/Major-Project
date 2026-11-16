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

## Known hard parts

- **Whisper drops fillers.** Whisper tends to write clean text and leave out
  "um" and "uh", which would hide one of our best delivery features. Two
  fixes to try: a transcription prompt that itself contains fillers (this
  nudges Whisper to keep them), or a small separate filler detector
  (PodcastFillers below).
- **Spoken answers aren't formatted.** An AI answer on screen has headings
  and bullet points. Once read aloud and transcribed, it doesn't. Strip
  formatting from reference answers before comparing them.
- **A different model wrote the cheater's answer.** We generate references
  with a local model, but the candidate probably used GPT, Claude or Gemini.
  The echo score only works if answers from different LLMs land close
  together. Test this explicitly.
- **Textbook questions.** For "what is a hash map?", every good answer looks
  alike, human or AI. Expect the content signal to be weak on factual
  questions and stronger on open or behavioural ones. Report results per
  question type.

## Datasets

Public data lets us build and test each half of the signal before our own
recordings exist. None of it contains the exact case we care about (a person
*reading an AI answer aloud in an interview*), so our own collection is still
required.

| Dataset | What it is | Access | How we use it |
|---|---|---|---|
| [HC3 (Human ChatGPT Comparison Corpus)](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | ~24k questions, each with human answers and ChatGPT answers (subsets include computer science, open QA, finance, medicine) | Hugging Face, CC BY-SA 4.0 | Develop the content signal: do ChatGPT answers score higher than human answers against our local references? ([paper and code](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)) |
| [RAID](https://github.com/liamdugan/raid) | Large benchmark of AI-generated text from 11 models, including paraphrasing and other adversarial attacks | GitHub / `pip install raid-bench` | Robustness of the content signal to other models and to paraphrasing |
| [ALLSSTAR](https://huggingface.co/datasets/NathanRoll/ALLSSTAR) (via [SpeechBox](https://speechbox.linguistics.northwestern.edu/)) | Scripted (read) and spontaneous speech from the same talkers, native and non-native English | SpeechBox (Northwestern); the Hugging Face copy is a community upload, so check which tasks it includes | Develop the delivery signal on read vs. spontaneous speech, and check the native vs. non-native gap |
| [PodcastFillers](https://podcastfillers.github.io/) | 145 h of podcast audio, ~35k hand-labelled "uh" and "um" ([Zenodo](https://zenodo.org/records/7121457)) | Annotations non-commercial (fine for research) | Measure whether our transcription keeps fillers; train a filler detector if it doesn't |
| [MIT Interview Dataset](https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances/) | 138 recorded mock job interviews (69 students), with audio, video and transcripts | Request form; needs an academic email | Genuine interview answers: what normal delivery looks like in an interview (negatives) |

Tools: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (word
timestamps), Ollama (reference answers), fastembed `BAAI/bge-small-en-v1.5`
(the same embedding model the sidecar uses, so offline results match the
app).

## Our own data

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

**One collection for both branches.** Each session records audio *and* the
candidate's video, so the same recordings and labels feed the gaze branch
too. Plan and run data collection together, once.

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
- Break results down by subgroup (non-native speakers, hard negatives,
  question type) to check for bias.
- Ablation: content only, delivery only, both.
- Cross-model check: generate references with the local model (e.g.
  `llama3.2`) while the "cheater" used a different assistant (GPT, Claude,
  Gemini). Does the echo score still separate the classes?

## Implementation plan

Each phase ends with something we can show. Dates go in once we know the
project deadlines.

### Phase 0: Set up

- Python environment for this folder (numpy, pandas, scikit-learn,
  faster-whisper, fastembed, matplotlib; Ollama for reference answers).
- Add `training/*/data/` to `.gitignore`.
- Write down the evaluation protocol (person-level split, metrics, target
  false-positive rate) **before** looking at any results.

*Done when:* the skeleton scripts run end to end on a toy example.

### Phase 1: Content signal on public text (HC3, RAID)

- For HC3 questions (start with the computer-science and open-QA subsets),
  generate 3–4 reference answers with `llama3.2` through Ollama and cache
  them.
- Strip formatting from all answers, embed them, and compute the echo score
  for every human and every ChatGPT answer.
- Tune the number of references (1–6), the embedding model (fastembed vs. the
  app's built-in hashing embedder), the reference model (`llama3.2` vs. a 7B
  model), and whole-answer vs. per-sentence scoring.
- Check robustness on RAID's paraphrase and cross-model subsets.

*Done when:* we have a content-only ROC-AUC on HC3 and a chosen
configuration.

### Phase 2: Delivery signal on public speech (ALLSSTAR, PodcastFillers, MIT Interview)

- Add word timestamps to the sidecar, matching OpenAI's API
  (`response_format=verbose_json`, `timestamp_granularities[]=word`) so the
  sidecar stays OpenAI-compatible.
- Write the feature extractor: response latency, speech rate and its
  variation, pause lengths and where they fall, filler rate,
  repetitions and restarts.
- Measure filler recall on PodcastFillers, with and without a filler-rich
  prompt. If it's still poor, add a small filler detector.
- Train a read vs. spontaneous classifier on ALLSSTAR, split by speaker, and
  compare native and non-native speakers.

*Done when:* we have a delivery-only ROC-AUC on ALLSSTAR and a measured
filler recall.

### Phase 3: Collect our own data (shared with the gaze branch)

- Write the protocol, consent form and question bank (a mix of technical,
  open and behavioural questions).
- Pilot with 2–3 teammates, then fix whatever was awkward.
- Run sessions over a real call. Randomise the condition order per
  participant and keep a label sheet per answer.

*Done when:* the dataset is collected and documented in a data card (in git,
no raw data).

### Phase 4: Combined model and evaluation

- Build per-answer features from both content and delivery, normalised
  against each speaker's small-talk baseline (z-scores).
- Train logistic regression with leave-one-speaker-out cross-validation and
  check calibration.
- Choose the threshold for a target false-positive rate (e.g. ≤ 5%).
- Produce ROC curves, the false-positive rate at the threshold, per-subgroup
  results, the ablation table, the cross-model check, and a look at every
  false positive.

*Done when:* results and plots in `results/` are ready for the report.

### Phase 5: Integrate into OpenMeet

- `local-ai`: word timestamps, with tests in `test_server.py`.
- `openmeet-core`: integrity-flag type, an `INTEGRITY_FLAG` event, and
  settings (on/off, threshold).
- `openmeet-intel`: a versioned reference-answer prompt; feature extraction
  and logistic scoring in pure Rust, reading the trained weights from a small
  JSON file; a parity test that reproduces the Python scores on fixtures.
- `openmeet-engine`: an `IntegrityAnalyzer` that builds the baseline from the
  first minutes and scores each finished answer in a background request,
  never blocking follow-ups.
- `openmeet-store`: a migration to persist flags.
- UI: a flag badge with the evidence (scores, timestamps), a settings toggle,
  and an entry on the privacy page.

*Done when:* an engine test with mocks covers the flow end to end, and a live
test runs it with local models.

### Phase 6: Fuse with the gaze signal (joint, after both branches merge)

Combine the echo and reading scores into one flag, re-tune the threshold on
the shared recordings, and report how much each signal adds.

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

## Ground rules

- A flag is a **signal for the interviewer, never a verdict**. It always
  comes with evidence (scores, timestamps), and the human decides.
- Missing a cheater is better than flagging an honest candidate. Choose
  thresholds accordingly.
- Runs locally by default, like the rest of OpenMeet, and only when the
  interviewer turns it on.

## Checklist

- [ ] Phase 0: environment, `.gitignore`, evaluation protocol written
- [ ] Phase 1: content signal measured on HC3 and RAID
- [ ] Phase 2: sidecar word timestamps; delivery signal measured on ALLSSTAR; filler recall measured
- [ ] Phase 3: own dataset collected (with the gaze branch)
- [ ] Phase 4: combined model, threshold and false-positive rate, bias check
- [ ] Phase 5: integrated into the app behind a setting
- [ ] Phase 6: fused with the gaze signal
