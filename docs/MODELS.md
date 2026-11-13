# A beginner's guide to the models in OpenMeet

This guide explains, in plain language, which machine-learning models OpenMeet
uses, which ones we trained ourselves, how we trained them and how well they
work. You do not need any background in machine learning to follow it. Words
that may be new are explained in the [glossary](#glossary) at the end.

OpenMeet uses two kinds of models:

- **Models we trained** for the *answer-integrity check*, which helps an
  interviewer notice when a candidate may be reading an answer or giving an
  answer written by an AI tool.
- **Ready-made models** that we use as they are: speech-to-text, face and
  eye landmarks, and a language model that suggests follow-up questions.

> **Important.** The integrity scores are reasons to ask a follow-up
> question. They are never proof and never a decision about a candidate.
> They are not yet calibrated on real interviews.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Model 1: the reading detector](#2-model-1-the-reading-detector)
3. [Model 2: the gaze model](#3-model-2-the-gaze-model)
4. [Model 3: the AI-written answer detector](#4-model-3-the-ai-written-answer-detector)
5. [Ready-made models we use](#5-ready-made-models-we-use)
6. [How the scores are shown](#6-how-the-scores-are-shown)
7. [How we kept the results honest](#7-how-we-kept-the-results-honest)
8. [Where the code is and how to retrain](#8-where-the-code-is-and-how-to-retrain)
9. [Glossary](#glossary)

---

## 1. The big picture

During an interview, OpenMeet looks at two things the candidate produces:
their eye movements and their words.

```
Candidate's video ──> face and iris points ──> gaze trace ──> READING DETECTOR ──> "reading" score
   (camera)            (MediaPipe)              (10 per s)     (trained by us)

Candidate's voice ──> transcript ──────────────────────────> AI-WRITTEN DETECTOR ──> "AI-written" score
   (microphone)        (Whisper)                               (trained by us)

Transcript ─────────> FOLLOW-UP MODEL (Llama 3.2 or Gemini) ──> suggested follow-up questions
```

Everything runs on the interviewer's own computer. The one exception is the
optional *online* mode for follow-up questions, which sends answer text (never
audio or video) to Google Gemini. The candidate is told about this before
joining.

## 2. Model 1: the reading detector

### The question it answers

*From 10 seconds of eye movements, is this person reading?*

### Why eye movements give reading away

When people read a line of text, their eyes do not glide smoothly. They jump
forward in small steps (from one group of words to the next), pause briefly on
each, and at the end of the line sweep back to the start of the next one. Seen
as a graph of horizontal eye position over time, reading looks like a
**sawtooth**: a series of small climbs followed by a sharp drop. Talking,
thinking or looking at the interviewer does not produce this pattern.

### The data

| Dataset | What it contains | How we used it |
|---|---|---|
| **GazeBase** | 322 people recorded with a lab eye tracker (1,000 samples per second). Each person reads a poem and does six other tasks: holding their gaze still, following dots, watching videos and playing a game. | Training, and a locked test set of 64 people |
| **GazeBaseVR** | 407 people in a virtual-reality headset, 5 tasks | Outside test only (never trained on) |
| **DesktopActivity** | 8 people at a normal desk: reading, browsing, playing, searching, watching, writing | Outside test only |
| **Reading in the Wild** (Meta) | 81 people wearing eye-tracking glasses in daily life, reading and not reading in the same places | A separate real-world training and test set |

A lab eye tracker is far more precise than a laptop webcam. To make the
training data look like what OpenMeet will really see, we **degraded** it to
webcam quality: 10 samples per second instead of 1,000, plus random noise,
slow drift, blinks and dropped frames.

### What the model looks at (the features)

The input is a 10-second window of gaze: 100 horizontal and 100 vertical
positions. We tried two kinds of model:

1. **Gradient boosting on 58 hand-made measurements** of the window, which
   describe how the eyes jump, pause and sweep back.
2. **A small 1-D convolutional neural network (CNN)** that reads the 100 raw
   positions directly and learns the sawtooth shape by itself. Before the
   network sees a window, the window is scaled by its own spread, so the
   network cares about the *shape* of the movement more than its size.

The CNN was chosen because it stayed accurate when we added more noise. The
final model averages three copies trained with different random starts. It
is 2.1 MB and runs on an ordinary CPU.

### How it was trained

1. The 322 GazeBase people were split **once**, before any testing: 258 for
   development and 64 kept aside for the final test.
2. Every decision (which features, which model, which threshold) was made
   with 5-fold cross-validation on the development people only.
3. The chosen model was retrained on all 258 development people.
4. The 64 test people were scored **once**.

Training ran on free Kaggle GPUs.

### Results

The main measure is **balanced accuracy**: the average of "how many reading
windows were caught" and "how many non-reading windows were correctly left
alone". We use it because only 14% of the windows are reading, so a model that
never says "reading" would already score 86% plain accuracy.

**GazeBase, 64 unseen people** (each window is 10 seconds):

| Condition | Balanced accuracy per window | Per recording |
|---|---|---|
| Webcam-like noise (normal) | **98.8%** | 99.9% |
| Twice the noise and drift | 97.5% | 99.5% |
| Smaller eye movements (smaller text or farther screen) | 97.5% | 99.5% |
| Smaller movements *and* twice the noise | 85.5% | 73.4% |
| A slow reader | 96.5% | 99.0% |

**Data the model never trained on:**

| Dataset | Balanced accuracy per window |
|---|---|
| GazeBaseVR (VR headset) | 96.6% |
| DesktopActivity (a normal desk) | 74.0% |

**Real-world reading (Reading in the Wild, trained and tested on its own
people):** 83.0% balanced accuracy per 10-second window with eye-tracker
input, 79.1% with webcam-style noise added. Over a whole recording it is
92.9% and 90.4%. The published gaze-only model on this dataset reaches about
82–86% using 60 samples per second; ours uses 10 per second, the rate a webcam
pipeline gives.

### What this means

- In clean, lab-like conditions the detector is very reliable.
- Real life is harder: about 4 out of 5 windows are classified correctly, and
  about 9 out of 10 whole answers.
- The threshold that decides "reading" needs to be set again on real
  interview recordings, which we have not collected yet.

## 3. Model 2: the gaze model

### The question it answers

*From a webcam image of a face, in which direction are the eyes looking?*
The answer is two angles: up/down (pitch) and left/right (yaw).

### Why we trained it

The app currently gets eye direction from **MediaPipe** iris points, which
are quick but rough. A better gaze trace would make the reading detector more
accurate on real webcams.

### Data and method

- **MPIIFaceGaze**: 37,667 laptop-webcam photos of 15 people, each labelled
  with where the person was looking.
- **Transfer learning**: instead of starting from scratch, we started from a
  ResNet-18 network that had already been trained for gaze on another dataset
  (ETH-XGaze, 110 different people) and fine-tuned it on MPIIFaceGaze. We
  also compared a ResNet-18 pre-trained on everyday photos (ImageNet) and a
  larger ResNet-50.
- To prepare it for video calls, training images were randomly shrunk and
  compressed like a low-quality video stream.
- Tested **leave-one-person-out**: train on 14 people, test on the 15th, and
  repeat for all 15.

### Results

Error is measured in **degrees** between the predicted and the true gaze
direction. Lower is better.

| Method | Average error |
|---|---|
| MediaPipe iris points (what the app uses today) | 7.18° |
| ResNet-18 pre-trained on everyday photos, fine-tuned | 5.49° |
| **ResNet-18 pre-trained for gaze, fine-tuned (chosen)** | **4.54°** |

The chosen model keeps working on small, compressed faces (5.76° on a 64-pixel
face with heavy compression) and takes about 33 ms per face on one CPU core.
For comparison, published results on this dataset range from about 3.9° to
4.8°.

### Status

Trained and measured, but **not yet used in the app**: live video frames
first need the same face alignment step that the training images had. Its
training data is licensed for non-commercial research only, so the trained
weights are not published.

## 4. Model 3: the AI-written answer detector

### The question it answers

*Does the transcript of this answer read like text written by an AI
assistant?*

### The data

**HC3** (Human ChatGPT Comparison Corpus): questions from several areas
(everyday explanations, finance, medicine, open questions and others), each
with answers written by people and answers written by ChatGPT.

To keep the comparison fair, every answer was first turned into what a
**speech transcript** of it would look like: lower case, words only, no
punctuation, lists or formatting. Otherwise the model could learn shortcuts,
for example that ChatGPT uses bullet points.

### What the model looks at (the features)

**TF-IDF** turns a text into numbers by counting its words and two-word
phrases, giving more weight to the ones that are distinctive (common in this
answer, rare in answers overall). We keep the 60,000 most useful words and
phrases. AI-written answers tend to use certain phrasings and structures more
often than people do when they speak.

### The model

**Logistic regression**, one of the simplest classifiers: it learns a weight
for each word or phrase and adds them up into a probability. We also tried a
fine-tuned DistilRoBERTa language model. The app uses the TF-IDF model because
it is small (1.9 MB) and scores an answer instantly on any laptop.

### How it was trained

The questions were split once: 80% for development and 20% kept aside, so no
question appears on both sides. The threshold was set on development data,
and the test questions were scored once.

### Results

On questions it had never seen: **97.9% balanced accuracy** and an **AUC of
0.998** (1.0 would be perfect ranking).

### Limits

HC3 compares written answers from 2022-era ChatGPT. A candidate who reads out
an answer from a newer model, or rephrases it in their own words, is harder to
catch. Short answers carry little evidence, so answers below a minimum length
are reported as "too short".

## 5. Ready-made models we use

These were not trained by us. We use them as published.

| Model | What it does in OpenMeet | Where it runs |
|---|---|---|
| **Whisper** (faster-whisper, `base.en`) | Turns speech into text | On this computer |
| **MediaPipe Face Landmarker** | Finds the face and the two irises in each video frame | On this computer |
| **Llama 3.2** through Ollama | Writes follow-up questions in the interview room; answers and summaries in the desktop app | On this computer |
| **Gemini 3.5 Flash** (optional) | Writes follow-up questions in the room's online mode | Google's servers |
| **fastembed** sentence embeddings | Finds related parts of the earlier conversation in the desktop app | On this computer |

### Follow-up questions and their guardrails

After each answer, the interview room asks a language model for up to three
follow-up questions. The request is built with **LangChain** and passes
through guardrails on both sides:

- **Before the model:** the transcript is marked as data, never as
  instructions; text that could break the prompt is removed; and in online
  mode, emails, phone numbers and the candidate's name are replaced before
  anything leaves the computer.
- **After the model:** a suggestion is dropped if it is not a question, judges
  or accuses the candidate, asks about personal characteristics such as age,
  family, religion or health, repeats an earlier question, or has nothing to
  do with the answer.

If no model is available, simple rules write grounded follow-ups instead, and
the page says so.

## 6. How the scores are shown

Each score is turned into a level using two thresholds chosen on development
data:

| Level | Meaning |
|---|---|
| **low** | Nothing unusual |
| **possible** | Above the balanced threshold: worth a follow-up question |
| **high** | Above the stricter threshold, where few honest candidates are flagged |
| **too short / no gaze** | Not enough evidence to score |

## 7. How we kept the results honest

- **Split by person (or by question), never by sample.** The same person's
  data is never on both the training and the test side, so the model cannot
  succeed by recognising people.
- **Test sets were locked and scored once.** All choices were made on
  development data first.
- **Outside datasets** that the model never saw show how it behaves on new
  hardware and new situations.
- **Balanced accuracy** is reported instead of plain accuracy, because reading
  windows are rare.
- **Limits are written down** next to every result.

## 8. Where the code is and how to retrain

| Part | Branch | Folder |
|---|---|---|
| Reading detector and gaze model: training code, notebooks, full results | `feature/gaze-detection` | `training/gaze-detection` |
| AI-written answer detector: training code | `feature/echo-detection` | `training/echo-detection` |
| The trained models inside the app, and the integrity check page | `feature/integrity-check` | `local-ai/integrity` |
| The browser interview room and follow-up questions | `feature/interview-room` | `local-ai/room` |

All training ran on free Kaggle GPUs. The scripts `kaggle_run.sh` (reading
detector), `kaggle_content.sh` (AI-written detector) and the notebook
`gaze_transfer_learning.ipynb` (gaze model) rebuild everything from the public
datasets. The full reading-detector and gaze-model write-up, with every
number, is in `training/gaze-detection/README.md`.

**Dataset licences:** GazeBase (CC BY 4.0), HC3 (CC BY-SA 4.0), MPIIFaceGaze
and ETH-XGaze (CC BY-NC-SA 4.0, non-commercial). Check each dataset's terms
before redistributing anything trained on it.

---

## Glossary

**Model.** A program whose behaviour is learned from examples instead of being
written by hand.

**Training.** Showing a model many labelled examples (for instance "this
window is reading", "this one is not") so it adjusts itself to give the right
answers.

**Feature.** A number that describes the input, such as how often the eyes
jump forward. Some models use hand-made features; neural networks learn their
own.

**Development set and test set.** Data used to build and tune the model, and
separate data kept aside to measure it fairly at the end.

**Cross-validation.** Splitting the development data into parts (folds),
training on all but one and checking on the one left out, then repeating so
every part is checked once.

**Balanced accuracy.** The average of the accuracy on positive examples
(reading) and on negative examples (not reading). It is fair when one kind is
much rarer than the other.

**AUC.** How well a model ranks positive examples above negative ones, from
0.5 (random guessing) to 1.0 (perfect).

**Threshold.** The score above which the app says "possible" or "high".

**CNN (convolutional neural network).** A neural network that slides small
learned patterns along the input to find shapes, such as the sawtooth of
reading.

**Transfer learning.** Starting from a model already trained on a related task
and fine-tuning it on your own data, instead of starting from scratch.

**TF-IDF.** A way to turn text into numbers by counting words and phrases,
weighted by how distinctive they are.

**Logistic regression.** A simple model that adds up a weight for each feature
and turns the total into a probability.

**ONNX.** A standard file format for trained models, so they can run inside
the app without the training software.

**Guardrails.** Checks placed before and after a language model so that what
goes in is safe and what comes out stays on task.
