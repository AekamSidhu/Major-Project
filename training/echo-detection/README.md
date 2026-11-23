# Echo detection

Building, training and measuring the **AI-echo** signals of OpenMeet's
answer-integrity check: does an answer read like, or sound like, it came from
an AI assistant? The other signal, reading gaze, lives in
`training/gaze-detection`. All signals are defined once in `spec/integrity/`
and combined into one flag per answer.

> Status (2026-11-23): three signals are built and run in the interview
> room (`local-ai/room`): the AI-written-text model (TF-IDF + logistic
> regression), echo similarity (thresholds measured on HC3) and delivery
> (hand-set, measured on SpiCE, where it barely works). None is calibrated
> on real interviews yet; the kit for collecting our own labelled
> interviews is in `training/data-collection`.

## What we're detecting

In a remote interview, a candidate can paste the question into an AI
assistant, or let a tool transcribe it automatically, and then read the
generated answer aloud. Each answer is looked at in three ways:

1. **AI-written text.** Does the transcript read like LLM writing?
2. **Echo similarity.** Is it close to what an LLM answers to the same
   question?
3. **Delivery.** Does it sound read, compared with how *this person*
   answered earlier?

None of them is enough on its own: a well-prepared candidate can give a
textbook answer, and a fluent speaker can sound scripted. A flag is a
**signal for the interviewer, never a verdict**.

## What is built and measured

| Signal | How | Code | Result on held-out data |
|---|---|---|---|
| AI-written text (`ai_text`) | TF-IDF (1-2 word grams, 60k terms) + logistic regression on the transcript (lower case, words only) | `train_content.py` (compares TF-IDF, a fine-tuned DistilRoBERTa and their average; Kaggle), `export_content_model.py` writes `local-ai/integrity/models/echo_tfidf.json` | HC3 test questions: 97.9% balanced accuracy, AUC 0.998 |
| Echo similarity (`echo_similarity`) | Llama 3.2 answers the question twice; highest cosine (bge-small-en-v1.5) between the answer and those | `calibrate_similarity.py` → `results/echo_similarity.json`, thresholds in `spec/integrity/thresholds.json` | HC3 test questions: AUC 0.876; human answers flagged 18.6% at `possible`, 6.6% at `high` (36% at `high` on wiki_csai) |
| Delivery (`delivery`) | Spec index: delay after the question, words a minute, pauses, fillers, against the speaker's first 3 answers | `evaluate_delivery.py` → `results/delivery_spice.json`; weights and thresholds hand-set in the spec | SpiCE, 34 speakers (30-word stretches, no question delay measurable): AUC 0.55 read vs spontaneous; 28% of spontaneous stretches reach `possible` |

The shipped content model is the TF-IDF one (fitted on train questions,
thresholds from validation questions, test questions scored once; the
numbers are stored in `echo_tfidf.json`): it is 1.9 MB of JSON and scores
an answer in plain Python on any laptop. `train_content.py` also fine-tunes
a DistilRoBERTa for comparison; that run's results live in the Kaggle
notebook's output, not in this repository. The word timestamps
delivery needs are in the sidecar (`POST /v1/audio/transcriptions` with
`timestamp_granularities[]=word`, OpenAI's API).

## Echo similarity on HC3 (Phase 1)

`calibrate_similarity.py`, protocol fixed before any test question was
scored:

- HC3 `all.jsonl` (CC BY-SA 4.0) loaded as in `train_content.py` (every
  answer as a transcript, at least 15 words, duplicates dropped) with its
  question split (seed 2026, 20% of questions are test).
- 100 questions from each of HC3's 5 sources per split (only questions with
  both a human and a ChatGPT answer): 500 development and 500 test
  questions.
- Reference answers exactly as the room writes them
  (`local-ai/integrity/signals.py`: system prompt, temperature 0.7, at most
  400 tokens, seed i for reference i), 2 per question from `llama3.2`
  through Ollama, cached outside the repository. Answer and references go
  through `to_transcript` before embedding.
- `possible` = the lower, `high` = the higher of the balanced cut and the
  cut with a 5% false-positive rate on development questions, rounded up to
  4 decimals. The signal counts as calibrated only if, on test questions,
  ROC-AUC >= 0.70 and the false-positive rate at `high` <= 10%.

Results (`results/echo_similarity.json`; about 2 s per reference on an
RTX 4060):

| | Development (500 questions, 1,396 answers) | Test (500 questions, 1,411 answers) |
|---|---|---|
| ROC-AUC | 0.882 | **0.876** |
| Mean similarity, ChatGPT / human | 0.907 / 0.801 | 0.905 / 0.800 |
| At `possible` = 0.8712: ChatGPT caught / human flagged | 84% / 18% | 81% / **18.6%** |
| At `high` = 0.9201: ChatGPT caught / human flagged | 51% / 5.0% | 47% / **6.6%** |

By source, test questions (human answers flagged at `possible` / `high`):

| Source | ROC-AUC | Humans flagged | ChatGPT caught |
|---|---|---|---|
| finance | 0.931 | 5% / 0% | 84% / 43% |
| medicine | 0.937 | 12% / 2% | 83% / 39% |
| open_qa | 0.861 | 33% / 7% | 86% / 51% |
| reddit_eli5 | 0.900 | 3% / 0% | 52% / 15% |
| wiki_csai | 0.792 | **70% / 36%** | 92% / 78% |

Within one question, a ChatGPT answer is more similar to the references
than a human answer to the same question 93% of the time (pairwise AUC
0.932), so most of the overlap comes from questions whose every answer is
alike. One reference instead of two gives AUC 0.859.

By the rule fixed beforehand the signal counts as calibrated (AUC 0.876 >=
0.70, 6.6% <= 10% at `high`), so `thresholds.json` now has
`calibrated: true` and these thresholds. The caveat matters for interviews:
`wiki_csai` (computer-science definitions, the most interview-like source)
flags 36% of human answers at `high`, because a correct textbook answer is
close to any LLM's. HC3's "human" answers there are Wikipedia text, which a
spoken answer is less like, but expect many false alarms on factual
technical questions; open and behavioural questions are where the signal
can help. The same caveat is in the spec's `source` string.

## Delivery on public speech (Phase 2)

The plan named ALLSSTAR. Its Hugging Face copy (`NathanRoll/ALLSSTAR`,
563 clips) has no task labels and no licence, and how SpeechBox itself
grants access could not be checked here, so it was not used. Datasets with
the same speakers reading and speaking freely (access checked 2026-11-17
unless marked):

| Dataset | Read / spontaneous | Access | Used |
|---|---|---|---|
| [SpiCE](https://doi.org/10.5683/SP2/MJOXP3) (Johnson et al., UBC) | Sentence reading, picture-story narration, 25-min interview; 34 bilingual speakers; hand transcripts with forced-aligned words (participant only) | Open download, CC BY 4.0; audio served without login | **Yes** |
| [ALLSSTAR](https://speechbox.linguistics.northwestern.edu/) | Scripted sentences and passage; picture-story narration and open questions; L1 and L2 English | SpeechBox web app; access terms not verified | No |
| [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) | Read prompts plus a spontaneous "suitcase" narrative, 22 of 24 speakers | Download form (name, email, affiliation) and licence acceptance; CC BY-NC 4.0 | No (form) |
| [IViE](https://www.phon.ox.ac.uk/files/apps/IViE/) | Read sentences and story, retold story, Map Task, conversation | Free after registration (name, affiliation, email) and accepting conditions; academic, non-commercial only | No (registration) |
| [Chains](https://chains.ucd.ie/corpus.php) | Read texts and unscripted retelling (Hiberno-English) | Research use; page unreachable when checked, terms not verified | No |

`evaluate_delivery.py` (also runs on any folder
`DIR/<speaker>/<read|spontaneous>/*.wav|json`). Protocol: answers are 30
consecutive words within one task (dropped at a gap over 2 s, so both
classes have the same length); each speaker's baseline is their first 3
interview answers; then up to 15 more interview answers per speaker and
every read answer are scored with the spec's weights and thresholds.
Latency cannot be measured (no question before a read sentence), so the
index uses rate, pauses and fillers only. Timings from the corpus's hand
transcripts (forced alignment), which keep the fillers.

| | Read (125) | Spontaneous (510) |
|---|---|---|
| Delivery index, mean | 0.59 | 0.42 |
| Reaches `possible` | 23% | 28% |
| Reaches `high` | 9% | 12% |
| Words a minute, mean | 194 | 205 |
| Pauses a minute, mean | 19.6 | 16.0 |
| Fillers per 100 words, mean | 0.16 | 3.18 |

ROC-AUC of the index, read against spontaneous: **0.547** (95% interval
from resampling speakers 0.50-0.59). Per feature, in the direction the spec
weights it: fillers 0.70, rate 0.44, pauses 0.38. Logistic regression on
the same z-scores, trained on 24 speakers and tested on 10 (for
information; the spec is unchanged): AUC 0.76 against 0.56 for the spec
index on the same test speakers, with the pause weight's sign reversed.

What it means: on this data the spec's delivery index is close to chance,
and with the hand-set thresholds it flags honest spontaneous answers more
often than read ones. Reading isolated sentences is slower and has more
pauses than conversation, the opposite of what the index assumes for a
read-out answer; only fewer fillers points the right way, and Whisper
drops many fillers. SpiCE's reading is not the case we care about (a
continuous answer read off a screen after a delay), so this does not settle
the weights, but it is a clear warning: keep delivery uncalibrated and
treat it as weak until our own recordings say otherwise.

A second run with Whisper's word timings on the participants' audio (fetched
by HTTP range requests, as the room would time it) was stopped at 400 of 839
answers when the machine ran low on memory; it has no result yet.

## Known hard parts

- **Whisper drops fillers**, one of the few delivery features that worked
  on SpiCE. A filler-rich transcription prompt or a separate filler
  detector (PodcastFillers) are the options.
- **Spoken answers aren't formatted.** Every answer and reference goes
  through `to_transcript` (lower case, words only) before any model sees it.
- **A different model wrote the cheater's answer.** References come from
  Llama 3.2; HC3's AI answers are from 2022 ChatGPT, so the HC3 numbers
  already measure a cross-model case. Newer assistants and paraphrasing are
  still untested (RAID, own data).
- **Textbook questions.** For "what is a hash map?", every good answer looks
  alike. Results are reported per HC3 source.
- **Short answers.** The AI-text model needs 15 words; delivery needs 10
  words and three earlier answers.

## Our own data

The collection kit (protocol, consent form with the cloud choice, question
bank, the four conditions, randomisation, label sheet, data card and
`calibrate.py`) is in `training/data-collection`, shared with `training/gaze-detection`.
Sessions run in the interview room's study mode
(`room_server.py --study DIR`).

## How we measure

- Split by question (HC3) or by speaker/person (SpiCE, own data), never by
  answer.
- Choose on development data; score test data once.
- Report ROC-AUC and the **false-positive rate at the chosen threshold**:
  how often an honest candidate gets flagged.
- Break results down by subgroup (source, native vs non-native, question
  type) to check for bias.

## Layout

```
training/echo-detection/
  README.md, PROGRESS.md
  content/text.py               transcript normalisation (identical to local-ai/integrity/text.py)
  train_content.py              AI-written-text candidates on HC3 (Kaggle: kaggle_content.sh)
  export_content_model.py       the shipped TF-IDF model as JSON for the sidecar
  calibrate_similarity.py       echo-similarity thresholds on HC3
  evaluate_delivery.py          the spec's delivery index on read vs spontaneous speech
  test_*.py, content/test_*.py  synthetic smoke tests (python -m pytest -q)
  results/                      aggregate metrics only
  data/                         local only, never committed
```

Run the tests from this folder with Python 3.11 (`numpy`, `pandas`,
`scikit-learn`; `fastembed` and `faster-whisper` only for the real runs).

## Ground rules

- A flag is a **signal for the interviewer, never a verdict**. It always
  comes with evidence, and the human decides.
- Missing a cheater is better than flagging an honest candidate. Choose
  thresholds accordingly.
- Runs locally by default, like the rest of OpenMeet.

## Checklist

- [x] Phase 0: environment, `.gitignore`, evaluation protocol written
- [x] Phase 1: AI-written-text model trained and shipped (HC3); echo similarity measured on HC3 and its thresholds set
- [ ] Phase 1: robustness to other models and paraphrasing (RAID)
- [x] Phase 2: sidecar word timestamps; delivery measured on SpiCE (hand timings)
- [ ] Phase 2: delivery with Whisper timings on SpiCE (run stopped at 400 of 839 files; see PROGRESS.md); filler recall
- [x] Phase 3: collection kit and room study mode ready
- [ ] Phase 3: own dataset collected (shared with `training/gaze-detection`)
- [ ] Phase 4: thresholds from own data (`training/data-collection/calibrate.py`), bias check
- [x] Phase 5: integrated into the interview room and the desktop app (`docs/INTEGRITY.md`)
- [x] Phase 6: fused with the gaze signal (spec fusion, one flag per answer)
