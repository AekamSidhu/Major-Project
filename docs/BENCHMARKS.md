# Benchmarks

What OpenMeet's local pipeline costs in time on one laptop, how accurate its
transcription is, and how often the answer-integrity check catches AI
answers and flags honest ones. This is objective O4 of the project ("test
latency, transcription accuracy, detection rate and false flags; ablate").

Every number in the tables below is printed from a result file in
[`benchmarks/results/`](../benchmarks/results) by
[`benchmarks/tables.py`](../benchmarks/tables.py)
(`python benchmarks/tables.py --doc docs/BENCHMARKS.md --check` fails if
they differ). The numbers written in the text are checked against the result
files by [`benchmarks/check_doc.py`](../benchmarks/check_doc.py). The scripts,
their inputs and how to run them are in [Reproduce](#reproduce). Read
[Limits](#limits) before quoting a number.

## Summary

- **Transcription.** On 600 LibriSpeech utterances, Whisper `base.en` (the
  sidecar's default) makes 3.9% word errors on test-clean and 10.7% on
  test-other. Its median request takes about 0.41 s on the CPU and 0.09 s on the
  GPU for a clip of about 7 s on average (section 1).
- **Live pipeline.** In the desktop engine's live run (12 spoken questions in
  real time), the first answer token appears a median 1.8 s after the speaker
  stops on the CPU and 0.7 s on the GPU; all 12 questions were detected
  (section 2, with percentiles). In the first room run the question's text was
  not back in time in 3 of its 8 turns; that was a timing race in the room's host
  page, fixed and rerun (section 2). The echo reference answers now start while
  the candidate is still answering, so the echo signal arrives about 1.6 s after
  the card instead of about 8 s (section 2, on a busy machine).
- **AI-written text.** The shipped model (v2) and the first model (v1) each
  catch 97.8% of ChatGPT answers at their `high` level on the HC3 test
  questions, and flag 1.8% (v2) and 2.0% (v1) of human answers (section 3).
- **Fused flag** (two of the four signals, on HC3). At `high` it catches 80.0%
  of ChatGPT answers and flags 4.7% of human ones. At `possible` it catches
  99.6% and flags 25.0%. Neither level is a verdict. The textbook-style source
  (wiki_csai) is the weak spot: 25% of its human answers are flagged at `high`.

## Machine and software

All runs: one laptop, 13th Gen Intel Core i9-13900H (14 cores, 20 logical
CPUs), 16 GB RAM (15.7 GB visible), NVIDIA GeForce RTX 4060 Laptop GPU
(8 GB, driver 621.24), Windows 11 Home (build 26300), Python 3.11.9, Rust
1.99.0, Ollama serving `llama3.2` (3.2B, Q4_K_M) on the GPU. Each table ends
with its own run's date, commit and package versions. Another build (a Rust
compile in another folder) sometimes ran on the same machine at the same time;
CPU timings can be a little slower than on an idle machine.

Defaults unless a table says otherwise: Whisper `base.en` with int8 weights,
greedy decoding (beam 1) and faster-whisper's 4 CPU threads, as the sidecar
ships; embeddings `BAAI/bge-small-en-v1.5` (fastembed, ONNX); follow-ups and
reference answers from `llama3.2` at the room's settings. Latencies are
wall-clock times seen by the caller, reported as median / 90th / 95th
percentile; a model's first (loading) request is reported separately and left
out of the percentiles.

## 1. Transcription accuracy and speed

**How.** A seeded random 300 utterances from each of LibriSpeech test-clean
and test-other (`random.Random(2026).sample` over the sorted utterance ids;
CC BY 4.0, from openslr.org) went one at a time through the sidecar's real
endpoint, `POST /v1/audio/transcriptions`, as 16 kHz 16-bit WAV with
`language=en` and no prompt. A fresh sidecar was started for each model, so
only one Whisper model was in memory. Word error rate (WER) uses OpenAI
Whisper's English text normaliser on both sides (the `whisper_normalizer`
package, a copy of `whisper/normalizers/english.py`), with the errors summed
over the whole subset. Latency is the HTTP request's wall time; the real-time
factor (RTF) is latency divided by the audio's length (0.05 = 20 times faster
than real time). The GPU runs used the same sidecar with CUDA 12 cuBLAS and
cuDNN 9 from NVIDIA's pip wheels on its `PATH` (the sidecar needs them for
`OPENMEET_WHISPER_DEVICE=cuda`; nothing in the product changed). Source:
`benchmarks/results/asr_cpu.json`, `asr_cuda.json`, `asr_cpu_words.json`.

<!-- tables:asr -->
### Transcription: WER and speed

WER with a 95% bootstrap interval over utterances in brackets. First request: a fresh sidecar's first transcription, which loads the model (models already on disk).

| Model | Device | WER test-clean | WER test-other | Latency per utterance, ms (median / p90 / p95) | Real-time factor (median) | Aggregate RTF | First request, s |
|---|---|---|---|---|---|---|---|
| tiny.en | cpu | 5.37% (4.5–6.2) | 14.58% (12.7–16.6) | 217 / 260 / 273 | 0.041 | 0.0319 | 1.2 |
| base.en | cpu | 3.89% (3.2–4.6) | 10.70% (9.3–12.3) | 409 / 494 / 523 | 0.078 | 0.0607 | 11.2 |
| small.en | cpu | 3.27% (2.6–4.1) | 7.00% (5.9–8.2) | 1,232 / 1,497 / 1,648 | 0.234 | 0.1859 | 26.2 |
| tiny.en | cuda | 5.34% (4.5–6.1) | 14.16% (12.4–16.0) | 62 / 116 / 136 | 0.011 | 0.0101 | 1.6 |
| base.en | cuda | 4.01% (3.3–4.7) | 10.72% (9.3–12.3) | 88 / 157 / 190 | 0.016 | 0.0142 | 1.3 |
| small.en | cuda | 3.40% (2.7–4.3) | 7.18% (6.0–8.4) | 156 / 284 / 338 | 0.029 | 0.0254 | 1.9 |

300 + 300 utterances (40 + 32 speakers, 69.6 min of audio), seed 2026.

*Run 2026-11-19 17:47, commit 79dbf40; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0, fastapi 0.165.0, uvicorn 0.54.0, whisper_normalizer 1.0.1, jiwer 4.0.0.*

*Run 2026-11-19 18:09, commit 1e093c0; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0, fastapi 0.165.0, uvicorn 0.54.0, whisper_normalizer 1.0.1, jiwer 4.0.0.*

Word timestamps (verbose_json, as the room asks), CPU:

| Model | Word timings | WER test-clean | WER test-other | Latency, ms (median / p90 / p95) | Aggregate RTF |
|---|---|---|---|---|---|
| base.en | yes | 4.66% (3.4–6.1) | 10.77% (8.3–13.8) | 992 / 1,518 / 1,623 | 0.1621 |

(100 utterances per split, the first of the same seeded order.) *Run 2026-11-19 18:13, commit 1e093c0; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0, fastapi 0.165.0, uvicorn 0.54.0, whisper_normalizer 1.0.1, jiwer 4.0.0.*
<!-- /tables -->

![Word error rate against latency for each Whisper model on CPU and GPU](../benchmarks/results/asr_wer_latency.png)

The word-timestamp run (verbose output, which the room asks for) uses a
different, smaller sample (100 utterances per split), so its latency is not
directly comparable with the plain CPU run; in the tables it is about 2.4
times the plain CPU median.

## 2. Pipeline latency

Three views of the same pipeline:

- **The desktop engine** (Rust). Its own Diagnostics benchmark
  (`Engine::run_benchmark`: synthetic input, a mock LLM, so it times the
  engine's own work), run 20 times headless; then the real pipeline with local
  models, as `crates/openmeet-engine/tests/live.rs` runs it: 12 spoken
  interview questions (Windows text-to-speech, 2.5-4.9 s each) pushed into the
  engine as 20 ms system-audio frames **in real time**, through voice activity
  detection, the sidecar's Whisper (`base.en`, on the CPU and then on the GPU),
  question detection and llama3.2 on the GPU writing the suggested answer; then
  12 spoken candidate answers in interviewer mode (follow-up questions) and the
  meeting summary. Times are measured from the moment the last speech frame was
  pushed, so they include the VAD's 700 ms silence hangover (and, for
  follow-ups, the 1.5 s turn-end wait), which a user waits for too. Both use
  only the engine's public API, from a separate crate
  ([`benchmarks/engine-bench`](../benchmarks/engine-bench)).
- **The sidecar's endpoints** one at a time: embeddings, the AI-written-text
  model, one video frame through the gaze tracker (MediaPipe) at 10 frames a
  second, and the reading network.
- **The interview room** end to end (the question is read when the answer closes, as the host page does now; `--question-at start` reproduces the first run's rule; the reference answers start when the question's transcripts are back during the answer, `--references settled`, and `--references answer` is the rule before that): a scripted 8-answer interview played
  against the room's HTTP API in real time, following the host page's own rules
  (speech cut into pieces by a Python copy of its segmenter, a video frame 10
  times a second, the answer closed when the interviewer speaks again, then echo
  similarity and follow-ups requested together, as `host.html` does). Whisper
  `base.en` on the CPU (the benchmark's setting; the room's own default is
  `small.en`, see `local-ai/room/README.md`), follow-ups and reference answers
  from llama3.2.

<!-- tables:engine -->
### Desktop engine: built-in benchmark (synthetic input, mock LLM)

| Item | Median | p90 | p95 | Unit |
|---|---|---|---|---|
| Question detection | 0.101 | 0.1144 | 0.1309 | ms |
| Voice activity detection | 4.542e+04 | 4.807e+04 | 4.908e+04 | x realtime |
| Local embedding | 0.007 | 0.007 | 0.0071 | ms |
| Context retrieval (1,500 items) | 0.507 | 0.5579 | 0.5846 | ms |
| Prompt assembly | 0.2045 | 0.236 | 0.2467 | ms |
| Pipeline: transcript → question detected | 0.2645 | 0.3279 | 0.3481 | ms |
| Pipeline: question → first token (mock LLM) | 0.0855 | 0.3773 | 0.4059 | ms |
| Pipeline: first token → complete (mock LLM) | 0.0405 | 0.0506 | 0.0665 | ms |

20 runs. *Run 2026-11-23 18:56, commit f10ae37; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; rustc 1.99.0 (b940084d7 2026-09-28).*

### Desktop engine: live pipeline, Whisper base.en on CPU

| Stage (from the end of the speech) | ms (median / p90 / p95) | n |
|---|---|---|
| transcript final (VAD hangover + Whisper) | 1,386 / 2,092 / 2,104 | 12 |
| question detected | 1,386 / 2,092 / 2,104 | 12 |
| question detected -> first answer token | 389 / 496 / 601 | 12 |
| first answer token | 1,840 / 2,531 / 2,647 | 12 |
| first token -> answer complete | 3,807 / 6,413 / 8,230 | 12 |
| answer complete | 6,015 / 7,606 / 9,231 | 12 |
| interviewer mode: first follow-up | 7,394 / 16,091 / 23,697 | 11 |
| interviewer mode: follow-ups done | 18,148 / 31,691 / 34,444 | 11 |

Questions detected: 12 of 12; one transcript per question: 12; answers with follow-ups: 10 of 12; summary after stop: 33.3 s (ok).

Engine's own latency tracker: AudioToTranscript p50 1,389 ms, p95 5,111 ms (n 49); TranscriptToDetection p50 1 ms, p95 11 ms (n 49); DetectionToFirstToken p50 726 ms, p95 1,486 ms (n 24); FirstTokenToComplete p50 7,617 ms, p95 22,030 ms (n 23)

*Run 2026-11-23 21:27, commit 72f9343; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0; Ollama 0.34.0; rustc 1.99.0 (b940084d7 2026-09-28).*

### Desktop engine: live pipeline, Whisper base.en on CUDA

| Stage (from the end of the speech) | ms (median / p90 / p95) | n |
|---|---|---|
| transcript final (VAD hangover + Whisper) | 354 / 465 / 493 | 12 |
| question detected | 354 / 465 / 493 | 12 |
| question detected -> first answer token | 337 / 508 / 2,100 | 12 |
| first answer token | 701 / 876 / 2,436 | 12 |
| first token -> answer complete | 3,802 / 8,875 / 10,010 | 12 |
| answer complete | 4,503 / 11,256 / 12,616 | 12 |
| interviewer mode: first follow-up | 3,595 / 4,548 / 5,316 | 11 |
| interviewer mode: follow-ups done | 6,764 / 10,834 / 15,289 | 11 |

Questions detected: 12 of 12; one transcript per question: 12; answers with follow-ups: 11 of 12; summary after stop: 17.2 s (ok).

Engine's own latency tracker: AudioToTranscript p50 868 ms, p95 1,075 ms (n 49); TranscriptToDetection p50 1 ms, p95 3 ms (n 49); DetectionToFirstToken p50 513 ms, p95 1,329 ms (n 23); FirstTokenToComplete p50 3,951 ms, p95 10,945 ms (n 23)

*Run 2026-11-23 21:45, commit 04a488c; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0; Ollama 0.34.0; rustc 1.99.0 (b940084d7 2026-09-28).*
<!-- /tables -->

<!-- tables:sidecar -->
### Sidecar endpoints

| Endpoint | What is sent | Latency, ms (median / p90 / p95) | First request, ms | n |
|---|---|---|---|---|
| `POST /v1/embeddings` | one answer | 24.4 / 62.6 / 81.7 | 1,206 | 50 |
| `POST /v1/embeddings` | answer + 2 reference answers | 168.8 / 269.9 / 335.0 | – | 50 |
| `POST /v1/integrity/echo` | one answer (AI-text model) | 4.30 / 23.40 / 25.82 | 92 | 50 |
| `POST /v1/integrity/gaze/frame` | one JPEG frame, 10 a second | 25.6 / 32.8 / 35.0 | 2,456 | 636 |
| `POST /v1/integrity/reading` | last 10 s of gaze | 14.48 / 26.07 / 26.86 | – | 50 |
| `POST /v1/integrity/reading` | last 60 s of gaze | 27.92 / 33.68 / 34.98 | – | 50 |

Gaze feed: 636 frames in public-domain NASA portrait (scikit-image astronaut.png), JPEG q80; face found in 636; 9.09 frames a second achieved; 0 errors.

*Run 2026-11-23 18:42, commit 8f7828a; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; fastapi 0.163.1, uvicorn 0.54.0, fastembed 0.9.1, onnxruntime 1.32.0, mediapipe 1.2.3, opencv-contrib-python 5.0.0.93, numpy 2.4.6.*
<!-- /tables -->

<!-- tables:followups -->
### Follow-up questions: local LLM vs rules

| Mode | Latency per answer, ms (median / p90 / p95) | Suggestions before guardrails | Kept (shown) | Answers with at least one kept | Fell back to rules |
|---|---|---|---|---|---|
| local (llama3.2, structured output) | 2,614 / 3,236 / 7,279 | 95 | 95 (2.64 per answer) | 35 / 36 | 1 |
| rules | 0.243 / 0.345 / 0.379 | 72 | 72 (2.0 per answer) | 36 / 36 | – |

Streamed (same prompt and model, plain text): first token 29 / 33 / 35 ms, whole answer 1,253 / 1,603 / 1,699 ms (median / p90 / p95). First local call after loading: 7,985 ms.

Echo reference answers (2 per question, llama3.2): 3,992 / 4,409 / 4,555 ms; 152 words each (median).

Dropped by the guardrails (local): none; rules output under the same guardrails: none dropped. Kept with a trigger the candidate actually said: local 89 of 95, rules 72 of 72.

*Run 2026-11-23 18:44, commit ef94b29; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; langchain-core 1.8.1, langchain-ollama 1.1.0, ollama 0.6.3; Ollama 0.34.0.*
<!-- /tables -->

<!-- tables:room -->
### Interview room, end to end

| Step (turns 1-7) | ms (median / p90 / p95) | n |
|---|---|---|
| answer closed -> its last transcripts back | 4,947 / 13,463 / 18,099 | 7 |
| POST /answers (AI text, delivery, reading, flag) | 54 / 378 / 606 | 7 |
| answer closed -> card shown | 4,974 / 13,993 / 18,400 | 7 |
| card -> echo similarity filled in | 1,599 / 2,466 / 2,791 | 7 |
| card -> follow-ups shown | 3,819 / 4,648 / 4,972 | 7 |
| every transcription request (both speakers, all turns) | 6,587 / 28,522 / 33,545 | 25 |
| every gaze frame | 55 / 89 / 106 | 3277 |

The question's text was back 7.03 s (median; max 11.64 s) after the interviewer stopped.

| Turn | Answer | Candidate starts after | Question known | Pieces (late) | Flag | AI text | Echo similarity | Delivery | Reading | Follow-ups |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | q02 (written) | 2.5 s | yes | 2 (0) | possible | low (0%) | low (0.613) | baseline | possible | 3 (ai) |
| 2 | q03 (written) | 2.5 s | yes | 2 (0) | possible | low (2%) | low (0.755) | baseline | high | 1 (ai) |
| 3 | q04 (written) | 2.5 s | yes | 1 (0) | possible | low (5%) | low (0.686) | baseline | possible | 3 (ai) |
| 4 | q05 (written) | 2.5 s | yes | 2 (0) | possible | low (0%) | low (0.621) | low | possible | 3 (rules) |
| 5 | q01 (llama3.2, read out) | 2.5 s | yes | 3 (0) | high | high (96%) | low (0.859) | low | possible | 3 (ai) |
| 6 | q07 (written) | 2.5 s | yes | 1 (0) | possible | low (12%) | low (0.738) | low | possible | 1 (rules) |
| 7 | q06 (llama3.2, read out) | 2.5 s | yes | 4 (0) | high | high (82%) | high (0.955) | low | low | 1 (rules) |
| 8 | q08 (written) | 0.8 s | yes | 2 (0) | possible | low (15%) | low (0.748) | low | possible | 2 (ai) |

*Run 2026-11-24 18:01, commit 919a79c; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; faster-whisper 1.2.1, ctranslate2 4.9.0, fastembed 0.9.1, onnxruntime 1.32.0, mediapipe 1.2.3, langchain-ollama 1.1.0, fastapi 0.163.1, uvicorn 0.54.0; Ollama 0.34.0.*
<!-- /tables -->

![Median and 90th percentile of each live stage](../benchmarks/results/pipeline_latency.png)

The live engine numbers are the desktop app's own pipeline on this machine;
the room numbers come from the room's HTTP API and include its browser-side
requests through the sidecar and Ollama. The latency to the first answer token
is what the candidate sees first; the complete answer follows after it.

In the first room run (commit 0a177e7) the question's text was not back in
time for 3 of the 8 turns (2, 5 and 8). Those turns still produced answers,
flags and follow-ups, but without the question text. The cause was the room's
host page: it read the question when the candidate began speaking, and a
question took about 1.8 s to transcribe on the CPU, so a slow one arrived after
that point and was dropped. The host page now reads the question when the answer
closes, taking the interviewer's segments that began before the candidate
(`--question-at close`, the default). It also sends that question to the
reference answers as soon as its transcripts are back, during the answer
(`--references settled`, the default). The table is a rerun with those rules
(commit 919a79c): the question's text was transcribed back in 8 of its 8 turns,
and the echo similarity was scored for all 8 answers. The machine was busy during
the run (another agent's Rust build; free RAM 2.9 GB), and the median transcription
request took 6.6 s (25 requests). So the card, which waits for the question's
transcript, came 5.0 s (median) after the answer closed, and the timings are not
a clean comparison with the first run.

The change itself is measured on the same machine state: the same rerun with
`--references answer` (the reference answers start when `/answers` arrives, as
before) ran with free RAM 1.5 GB and the same settings. There, echo similarity was
filled in 8.3 s (median) after the card. With `--references settled` it was 1.6 s
(median) after the card, while the card itself took about the same time (median
5.0 s after close in the settled run, 5.9 s in the baseline). The baseline run is
not kept in the repository. In the settled run turn 8, the quick turn-taking case,
was the one where the question at close differed from the text sent early, so
its references were requested again at close (this run did not record why).
The room run is 8 answers, so these are findings about these runs, not rates.

## 3. Detection rate and false flags

A flag is a reason for the interviewer to ask a follow-up question, never a
verdict. "Detection rate" is the share of AI answers (or reading windows)
flagged, "false-flag rate" the share of honest ones flagged, at the spec's
two levels: `possible` (the lower threshold) and `high` (the higher one). Every
threshold below was chosen on development data before these test sets were
scored.

<!-- tables:held_out -->
### Detection rate and false flags: held-out results already measured

| Signal | Test data (held out) | ROC-AUC | Detection rate | False-flag rate | Source |
|---|---|---|---|---|---|
| Reading (shipped network) | GazeBase, 64 unseen people, 10 s windows, webcam-quality noise | 0.999 | 99.7% at `possible` | 4.9% at `possible` (balanced accuracy 98.8% at `high`) | `training/gaze-detection/results/reading_detector_results.json` |
| Reading, hardest condition | same people, small face + heavy noise | 0.949 | 97.6% at `possible` | 28.5% at `possible` | same file |
| Reading, real life (a model trained on RitW, **not** the shipped one) | Reading in the Wild, 17 unseen people | 0.908 | 78.7% | 12.8% | `training/gaze-detection/results/ritw_10s.json` |
| AI-written text (v2, shipped) | HC3, all test questions (5,244 ChatGPT + 9,959 human answers) | 0.998 | 99.2% / 97.8% | 5.0% / 1.8% | `benchmarks/results/ai_text_v1_vs_v2.json` |
| Echo similarity | HC3, 500 test questions (100 per source) | 0.876 | 81.1% / 46.7% | 18.6% / 6.6% | `training/echo-detection/results/echo_similarity.json` |
| Delivery (uncalibrated) | SpiCE, 34 speakers: read vs spontaneous 30-word stretches | 0.547 | 23.2% / 8.8% of read | 27.6% / 12.2% of spontaneous | `training/echo-detection/results/delivery_spice.json` |

A / B = at `possible` / at `high`.
<!-- /tables -->

These were measured when each model was built; the table only gathers them
(details in [MODELS.md](MODELS.md) and each `training/*/README.md`). Read them
with their caveats:

- **Reading** is near perfect on GazeBase, which is lab eye tracking (a chin
  rest, silent poem reading) made noisy to look like a webcam; in its hardest
  condition (small face, heavy noise) about 28% of non-reading windows are
  flagged at `possible`. The real-life figure (Reading in the Wild) is for a
  model trained on those people's data, not the shipped GazeBase network,
  which has not been scored there yet.
- **AI-written text** is measured on HC3's written answers; the per-source
  table below shows it flags more human answers on textbook-style questions
  than the all-sources figure suggests.
- **Echo similarity** is the same 500 test questions as the fusion table below.
- **Delivery** is close to chance on SpiCE and flags spontaneous speech
  slightly more often than read speech; it stays uncalibrated, so the fused
  flag counts it at most as `possible`.

### The AI-written text model: v1 and v2

The shipped model is v2. It was retrained (commit 0ca0fad) on HC3 plus 54
interview-style Q&A pairs. **Correction to the spec:** v2 does store held-out
test metrics. Its `test_questions` field holds the metrics on the HC3 test
questions, nested under `balanced` and `low_false_positive`, as its training
script computed them. The numbers below use those stored values and check them
against a fresh recomputation (the stored-versus-recomputed table below); the
two agree to two decimals.

**Leakage check.** `training/echo-detection/retrain_with_interviews.py` splits
HC3 with the same function and seed as `train_content.py` (seed 2026: 20% of
questions are test; the 15% validation split uses seed 2027). The interview
questions carry ids offset by 100,000, so they cannot share a question with
HC3. No interview question, and no interview answer (first 200 characters),
appears among the HC3 test questions or answers. So the test questions were
not used for training or for choosing thresholds.

**Reproducibility.** Rerunning `retrain_with_interviews.py` on this machine
does **not** reproduce the committed v2 bit for bit. The rerun's
output is `benchmarks/results/retrain_v2_rerun.json`: its thresholds are
0.4900 / 0.2476 for HC3 against 0.4624 / 0.2466 in the committed file (its
interview thresholds differ too), and its vocabulary differs from the committed
file's by 407 of the 60,000 terms (a one-off comparison of the two model files,
not a script in this folder).
Every number for v2 in this document is for the committed file as shipped
(`local-ai/integrity/models/echo_tfidf.json`), not for a fresh retraining. The
cause of the difference was not found. v1's stored test metrics are reproduced
with the same split code, to two decimals.

<!-- tables:ai_text -->
### AI-written text model: v1 against v2 (HC3 test questions, the same split)

| Model | ROC-AUC | Balanced accuracy at `high` | Detected at `possible` | False flags at `possible` | Detected at `high` | False flags at `high` |
|---|---|---|---|---|---|---|
| v1 (commit 0836f1e, HC3 only) | 0.9980 | 97.9% | 99.1% | 4.9% | 97.8% | 2.0% |
| v2 (shipped; HC3 + interview Q&A) | 0.9980 | 98.0% | 99.2% | 5.0% | 97.8% | 1.8% |

Thresholds (the sidecar's: `possible` is the lower of the two stored thresholds, `high` the higher): v1 0.2494 / 0.4444; v2 0.2466 / 0.4624. 15,203 answers (5,244 ChatGPT, 9,959 human) to 4,761 test questions.

By HC3 source (ROC-AUC within the source; detection and false flags at `high`):

| Source | ChatGPT / human | v1 AUC | v2 AUC | v1 detected / false flags | v2 detected / false flags |
|---|---|---|---|---|---|
| finance | 904 / 781 | 0.999 | 0.999 | 98.9% / 1.3% | 99.1% / 1.3% |
| medicine | 270 / 258 | 1.000 | 1.000 | 100.0% / 0.8% | 100.0% / 0.4% |
| open_qa | 688 / 206 | 0.968 | 0.970 | 94.2% / 13.6% | 94.6% / 14.1% |
| reddit_eli5 | 3,214 / 8,554 | 0.999 | 0.999 | 98.0% / 1.4% | 97.8% / 1.1% |
| wiki_csai | 168 / 160 | 0.986 | 0.983 | 99.4% / 24.4% | 98.8% / 26.2% |

In the fused flag (HC3 sample of 1,411 answers, two references; see section 3):

| Configuration | v1 detected / false flags at `possible` | v2 detected / false flags at `possible` | v1 detected / false flags at `high` | v2 detected / false flags at `high` |
|---|---|---|---|---|
| ai_text alone | 98.5% / 13.0% | 98.8% / 13.2% | 97.4% / 6.4% | 97.8% / 6.8% |
| fused (ai_text + echo) | 99.6% / 25.0% | 99.6% / 25.0% | 79.8% / 4.4% | 80.0% / 4.7% |

Stored test metrics in each model file (`balanced` level, written by its training script), against the recomputed ones:

| File | Stored balanced accuracy | Recomputed | Stored recall | Recomputed | Stored false-positive rate | Recomputed |
|---|---|---|---|---|---|---|
| v1 | 97.90% | 97.90% | 97.79% | 97.79% | 1.98% | 1.98% |
| v2 | 97.99% | 97.99% | 97.77% | 97.77% | 1.79% | 1.79% |

*Run 2026-11-23 18:34, commit f312f96; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; scikit-learn 1.9.1, numpy 2.4.6.*
<!-- /tables -->

![Detection and false flags of each signal and the fused flag on HC3 test questions](../benchmarks/results/fusion_hc3.png)

![False flags at 'high' by HC3 source](../benchmarks/results/fusion_by_source.png)

**How** ([`benchmarks/bench_fusion.py`](../benchmarks/bench_fusion.py)).
HC3 has two of the four signals: the transcript text (`ai_text`, the sidecar's
AI-text model, scored with the sidecar's own code) and echo similarity (the
answer against llama3.2's two reference answers, embedded with bge-small,
exactly as the room computes it). The answers are the 1,411 answers to the
same 500 test questions (100 per source) the echo calibration scored, with its
cached reference answers; both models' test questions come from the same split
(`train_content.py`, seed 2026), so no answer here was used to fit a model or
choose a threshold. Each answer gets both levels and the spec's fused flag
(`local-ai/integrity/signals.py` `fuse`: `high` only when a calibrated signal
is `high` and another is at least `possible`). The table above uses v2; v1 was
run through the same benchmark (`--ai-model`, `fusion_hc3_v1.json`).

**What it shows** (v2).

- **Fusion does what the spec intends at `high`: fewer false flags at a modest
  cost in detection.** The fused flag at `high` flags 4.7% of human answers
  (AI text alone 6.8%, echo alone 6.6%) and still catches 80.0% of ChatGPT
  answers, because a `high` now needs both signals to agree.
- **At `possible` it does the opposite.** Either signal raising `possible`
  raises the flag, so the fused flag catches 99.6% but flags 25.0% of honest
  answers, more than either signal alone.
- **One signal alone can never reach `high`** (the rows "only available"): in
  rules mode, where echo similarity is unavailable, the text model's `high`
  becomes `possible` and the flag never says "several signals agree".
- **The AI-text model is weaker on this 100-per-source sample than on its full
  test set:** 6.8% false flags at `high` here against 1.8% on all HC3 test
  questions. The full test set is mostly reddit_eli5, where it flags about 1%;
  on open_qa it flags 16% of this sample's human answers at `high` and on
  wiki_csai 28%.
- **Textbook questions stay the hard case.** On wiki_csai (computer-science
  definitions, the most interview-like source) the fused flag still flags 25%
  of human answers at `high` and 80% at `possible`. HC3's "human" answers there
  are Wikipedia text, which reads much like an LLM's; a spoken answer would
  differ more, but this is the warning for factual technical questions.
- **Two reference answers beat one** (echo alone ROC-AUC 0.876 against 0.859;
  46.7% against 37.7% caught at `high`). With one reference the thresholds,
  measured for two, flag less: fewer false flags (3.8% fused at `high`) but
  fewer detections (72.5%).

<!-- tables:fusion -->
### The combined flag on HC3 test questions (new)

| Configuration | ROC-AUC | Detected at `possible` | False flags at `possible` | Detected at `high` | False flags at `high` |
|---|---|---|---|---|---|
| ai_text alone | 0.993 | 98.8% | 13.2% (10.9–16.0) | 97.8% | 6.8% (5.2–9.0) |
| echo similarity alone | 0.876 | 81.1% | 18.6% (15.9–21.7) | 46.7% | 6.6% (4.9–8.6) |
| fused (ai_text + echo) | – | 99.6% | 25.0% (21.9–28.4) | 80.0% | 4.7% (3.3–6.5) |
| fused, ai_text only available | – | 98.8% | 13.2% (10.9–16.0) | 0.0% | 0.0% (0.0–0.6) |
| fused, echo only available | – | 81.1% | 18.6% (15.9–21.7) | 0.0% | 0.0% (0.0–0.6) |
| echo similarity alone, 1 reference | 0.859 | 73.6% | 15.9% (13.3–18.8) | 37.7% | 5.5% (4.1–7.5) |
| fused (ai_text + echo, 1 reference) | – | 99.6% | 23.4% (20.4–26.8) | 72.5% | 3.8% (2.6–5.5) |

1411 answers (724 ChatGPT, 687 human) to 500 test questions; 95% Wilson intervals in brackets.

By source (fused flag, 2 references; detected / false flags):

| Source | ChatGPT / human answers | At `possible` | At `high` | AI text alone at `high` | Echo alone at `high` |
|---|---|---|---|---|---|
| finance | 123 / 100 | 100% / 11% | 84% / 0% | 100% / 1% | 43% / 0% |
| medicine | 104 / 100 | 100% / 13% | 83% / 0% | 100% / 1% | 38% / 2% |
| open_qa | 297 / 100 | 100% / 54% | 83% / 7% | 96% / 16% | 51% / 7% |
| reddit_eli5 | 100 / 287 | 98% / 5% | 51% / 0% | 96% / 0% | 15% / 0% |
| wiki_csai | 100 / 100 | 100% / 80% | 92% / 25% | 100% / 28% | 78% / 36% |

Human answers flagged by the fused rule, by cause:

- possible: ai_text low, echo possible: 59
- possible: ai_text possible, echo low: 23
- possible: ai_text low, echo high: 22
- possible: ai_text high, echo low: 21
- high: ai_text high, echo high: 17
- possible: ai_text possible, echo possible: 15
- high: ai_text high, echo possible: 9
- high: ai_text possible, echo high: 6

*Run 2026-11-23 18:38, commit f312f96; 13th Gen Intel(R) Core(TM) i9-13900H, 15.7 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 621.24; Windows-10-10.0.26300-SP0; Python 3.11.9; fastembed 0.9.1, onnxruntime 1.32.0, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1.*
<!-- /tables -->

## 4. Ablations

The AI-text model version (v1 against v2, in section 3), the number of echo
reference answers (one against two, in section 3) and each signal on its own
(in the fusion table) are the ablations. Follow-ups compare the local LLM with
the rules engine; see section 2.

## 5. Problems found

- **Spec premise corrected:** v2 stores held-out test metrics under nested
  keys; the spec said it did not.
- **`tables.py` crashed on v2**: its held-out table read v1's flat layout. It
  now reads `ai_text_v1_vs_v2.json`.
- **The fusion results file was v1** until it was rerun with v2. The v1 rerun
  reproduces the numbers committed with 79dbf40 exactly.
- **The committed v2 model is not reproducible from the repository's script on
  this machine** (section 3).
- **The room's host page read the question too early** (section 2): in the
  first run the question was missing in 3 of 8 turns. It is fixed; the rerun was
  made on a busy machine, so its timings are not a clean comparison.
- **Ollama stopped responding during the live engine run** (the first attempt
  failed with connection refused). The Ollama server was restarted and the run
  repeated from the beginning; the reported numbers come from that complete
  repeat.

## Limits

- **One machine.** Every timing is from one laptop (above); a laptop CPU's
  speed also depends on its power plan and temperature, which were not
  controlled. Treat latencies as this machine's, not as guarantees.
- **Read speech and synthetic voices.** LibriSpeech is audiobook speech, far
  cleaner and more fluent than an interview over a video call; expect higher
  WER in meetings (accents, crosstalk, compression, jargon). The pipeline and
  room runs speak with Windows' built-in voices, which Whisper finds easy.
- **No prompt.** The desktop app passes the previous transcript and the
  vocabulary to Whisper as a prompt; these runs send none, so they show the
  model alone.
- **HC3 is not an interview.** Its "AI" answers are 2022 ChatGPT's, its
  "human" answers are written (Reddit, Wikipedia, medical and finance Q&A
  sites), and the two text signals see the written text, not a speech
  transcript of it. The fused-flag numbers use two of the four signals:
  reading and delivery need video and audio that HC3 does not have, so how the
  full four-signal flag behaves in a real interview is still unmeasured. Our
  own labelled recordings (`training/data-collection`) are what will settle
  the thresholds.
- **The room and engine runs are small.** 8 answers in the room and 12
  questions and 12 answers in the engine are enough for latency medians, not
  for tail percentiles (a p95 of 8 values is close to the maximum) and not for
  accuracy; their levels are a sanity check.
- **Quality of follow-up questions** is measured only by the room's own
  guardrails (how many suggestions survive them), not by people.
- **Not measured:** the cloud providers (no API key was used, commands above),
  the desktop app's overlay render time and screen/OCR processing (they need
  the GUI and a display), and the reading signal on real faces (the gaze
  benchmark sends one still photo, which times the tracker but says nothing
  about accuracy).

## Reproduce

From the repository root, with Python 3.10-3.12 and, for section 2's engine
runs, Rust. Datasets and downloaded models stay outside the repository; the
paths below are the ones used here.

**Environment.**

```bash
python -m venv venv-bench
venv-bench/Scripts/pip install -r local-ai/requirements.txt -r local-ai/requirements-integrity.txt -r benchmarks/requirements.txt
# GPU transcription only: CUDA 12 cuBLAS + cuDNN 9 in a folder of their own (about 2 GB)
venv-bench/Scripts/pip install --target M:/openmeet-work/cuda-libs nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
ollama pull llama3.2                     # Ollama on 127.0.0.1:11434
export OPENMEET_BENCH_TMP=M:/openmeet-work/tmp/bench HF_HOME=M:/cache/hf   # logs, Hugging Face cache
```

**Data** (public, no login).

```bash
python benchmarks/librispeech.py --root M:/openmeet-data/librispeech        # test-clean + test-other, md5-checked (674 MB)
python benchmarks/make_inputs.py speech --out M:/openmeet-data/bench-speech  # Windows text-to-speech of data/*.json
curl -L -o M:/openmeet-data/faces/astronaut.png \
  https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png  # NASA, public domain
```

The sidecar's model folder is `M:/openmeet-data/models`. Whisper downloads
there on first use. The bge-small embedding model is read from
`models/embeddings` (copy it from the echo calibration's fastembed cache, or
let the first run download it). The face landmarker is downloaded there on
first gaze use.

HC3 (`all.jsonl`) and the cached llama3.2 reference answers come from the echo
calibration (`training/echo-detection/calibrate_similarity.py`, which downloads
HC3 and writes the cache); the fusion benchmark never generates a missing
reference. `data/llm_answers.json` was written once with
`python benchmarks/make_inputs.py llm-answers` and is committed as used.

**Runs** (each writes `benchmarks/results/<name>.json`; servers use ports 8795
and 8463 and are stopped at the end).

```bash
M=M:/openmeet-data/models   # where the sidecar keeps Whisper, bge-small and the face model
python benchmarks/bench_asr.py --root M:/openmeet-data/librispeech --device cpu --n 300 --model-dir $M --out benchmarks/results/asr_cpu.json
python benchmarks/bench_asr.py --root M:/openmeet-data/librispeech --device cuda --cuda-libs M:/openmeet-work/cuda-libs --n 300 --model-dir $M --out benchmarks/results/asr_cuda.json
python benchmarks/bench_asr.py --root M:/openmeet-data/librispeech --device cpu --n 100 --models base.en --word-timestamps --model-dir $M --out benchmarks/results/asr_cpu_words.json
python benchmarks/bench_ai_text.py --data M:/openmeet-data/hc3/all.jsonl --v1 M:/openmeet-work/tmp-t04/echo_tfidf_v1.json   # v1 = git show 0836f1e:local-ai/integrity/models/echo_tfidf.json
python benchmarks/bench_fusion.py --data M:/openmeet-data/hc3/all.jsonl --cache M:/openmeet-data/hc3/cache
python benchmarks/bench_fusion.py --data M:/openmeet-data/hc3/all.jsonl --cache M:/openmeet-data/hc3/cache --ai-model M:/openmeet-work/tmp-t04/echo_tfidf_v1.json --out benchmarks/results/fusion_hc3_v1.json
python benchmarks/bench_sidecar.py --face M:/openmeet-data/faces/astronaut.png --model-dir $M
python benchmarks/bench_followups.py
python benchmarks/bench_room.py --speech M:/openmeet-data/bench-speech --face M:/openmeet-data/faces/astronaut.png --model-dir $M
CARGO_TARGET_DIR=M:/openmeet-work/target-bench python benchmarks/bench_engine.py offline --runs 20 --out benchmarks/results/engine_offline.json
CARGO_TARGET_DIR=M:/openmeet-work/target-bench python benchmarks/bench_engine.py live --speech M:/openmeet-data/bench-speech --device cpu --model-dir $M --out benchmarks/results/engine_live_cpu.json
CARGO_TARGET_DIR=M:/openmeet-work/target-bench python benchmarks/bench_engine.py live --speech M:/openmeet-data/bench-speech --device cuda --cuda-libs M:/openmeet-work/cuda-libs --model-dir $M --out benchmarks/results/engine_live_cuda.json
python benchmarks/plot.py                                   # charts
python benchmarks/tables.py --doc docs/BENCHMARKS.md        # this document's tables
python benchmarks/check_doc.py                              # every number in the text against its result file
python -m pytest -q benchmarks                              # smoke tests: synthetic, no network, GPU or data
```

**Cloud comparison (not run: no API key was used).** With a key in the
environment, the same scripts measure the cloud providers on the same inputs:

```bash
# OpenAI transcription on the same 600 LibriSpeech utterances
OPENAI_API_KEY=... python benchmarks/bench_asr.py --root M:/openmeet-data/librispeech --n 300 --device auto \
  --endpoint https://api.openai.com/v1/audio/transcriptions --api-key-env OPENAI_API_KEY \
  --models gpt-4o-mini-transcribe --out benchmarks/results/asr_openai.json
# The room's online mode (Gemini) on the same 12 answers
OPENMEET_GEMINI_API_KEY=... python benchmarks/bench_followups.py --mode online --out benchmarks/results/followups_online.json
```

`--endpoint` sends exactly the requests the local runs send, to any
OpenAI-compatible server; it was checked here against the local sidecar with a
bearer token. Echo similarity's thresholds were measured with llama3.2's
reference answers, so with Gemini's references the signal counts as
uncalibrated (the room says so).
