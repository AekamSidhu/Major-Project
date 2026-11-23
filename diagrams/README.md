# OpenMeet diagrams

Architecture diagrams of OpenMeet **as built**. Every box, arrow, field and
state is taken from the code named in the first lines of its source file
(`' Derived from ...` in PlantUML, `// Derived from ...` in Graphviz). Nothing
is drawn from plans or slides. Each source has a PNG and an SVG render next to
it, made by `render.sh`.

Workplace mode is not drawn yet, because it is still being built (see the list
at the end). Everything else describes what is in `main` today.

Shared look: `_style.iuml` (included by every PlantUML source).

## Data flow diagrams (DFD)

Yourdon–DeMarco notation, Graphviz sources.

- `dfd/dfd-level-0-context.dot`: OpenMeet as one process with its external entities (user, operating system, local AI services, remote providers).
- `dfd/dfd-level-1.dot`: processes 1.0 to 11.0, data stores D1 to D4 and the flows between them (engine, intel, store).
- `dfd/dfd-level-2-audio-transcription.dot`: 1.0 capture audio and 2.0 segment and transcribe (platform audio, `vad.rs`, `transcription.rs`).
- `dfd/dfd-level-2-screen-context.dot`: 4.0 capture and interpret the selected screen (platform screen, `screen.rs`, vision prompts).
- `dfd/dfd-level-2-suggestions.dot`: 3.0 analyse the conversation, 5.0 assemble context, 6.0 generate suggestions (`conversation.rs`, `assist.rs`, prompts).
- `dfd/dfd-level-2-data-management.dot`: 8.0 settings and keys, 9.0 persist and export, 10.0 summary (`store/`, `settings.rs`, keyring).
- `dfd/dfd-level-2-integrity.dot`: 11.0 answer integrity, 11.1 to 11.8 (`engine/integrity.rs`, `intel/integrity/`, sidecar integrity endpoints, `spec/integrity`).

## UML

The 14 UML 2.5 diagram types. PlantUML sources, except the communication
diagrams and one package diagram, which are Graphviz.

### 1. Activity

- `uml/activity/activity-01-meeting-lifecycle.puml`: setup, activate, live, stop, summary, store (`engine.rs`, `state.rs`, `commands/meeting.rs`).
- `uml/activity/activity-02-audio-capture-per-source.puml`: device to 16 kHz frames to the engine, per source (`platform/audio`, `capture.rs`).
- `uml/activity/activity-03-vad-utterance-segmentation.puml`: the segmenter's onset, hangover, discard and cut rules (`intel/vad.rs`).
- `uml/activity/activity-04-integrity-fusion.puml`: the four signal levels to one flag, rule by rule (`intel/integrity/fusion.rs`, `spec/integrity`).

### 2. Class

Core types (`openmeet-core`):

- `uml/class/core/class-01-meeting-aggregate.puml`: the meeting aggregate (`types.rs`).
- `uml/class/core/class-02-transcript-and-questions.puml`: transcript segments, drafts with word timings, speakers, detected questions (`types.rs`).
- `uml/class/core/class-03-suggestions-and-requests.puml`: AI requests, suggestions, follow-ups, RequestKind (incl. integrity kinds) (`types.rs`).
- `uml/class/core/class-04-conversation-intelligence.puml`: claims, topics, action items, screen context (`types.rs`).
- `uml/class/core/class-05-health-latency-privacy-rules.puml`: component health (incl. Integrity), latency, privacy rules (`types.rs`).
- `uml/class/core/class-06-settings-ai-and-assistant.puml`: `Settings` root, AI providers, assistant behaviour, defaults (`settings.rs`).
- `uml/class/core/class-07-settings-capture-privacy-general.puml`: audio, screen, privacy, general, profile, integrity settings (`settings.rs`).

Engine and intelligence (`openmeet-engine`, `openmeet-intel`):

- `uml/class/engine-intel/class-01-engine-handle-and-workers.puml`: `Engine`, `Inner`, the workers and their channels (`engine.rs`, `integrity.rs`).
- `uml/class/engine-intel/class-02-meeting-runtime-and-state-machines.puml`: per-meeting runtime and its state machines (`runtime.rs`, `topics.rs`).
- `uml/class/engine-intel/class-03-persistence-queue-and-forget.puml`: persistence queue and forgetting (`persist.rs`, `store/database.rs`).
- `uml/class/engine-intel/class-04-request-manager-slots.puml`: `RequestManager`, slots, stale-response suppression (`requests.rs`).
- `uml/class/engine-intel/class-05-integrity-run-and-checks.puml`: per-meeting integrity state, check jobs, gaze commands (`engine/integrity.rs`).

Integrity vocabulary:

- `uml/class/integrity/class-01-integrity-flag-and-service.puml`: signals, levels, `IntegrityFlag`, the sidecar `IntegrityService` trait, `fuse` (`core/integrity.rs`, `intel/integrity/fusion.rs`).

Providers and platform (`openmeet-providers`, `openmeet-platform`):

- `uml/class/platform-app/class-01-provider-adapters.puml`: provider adapters and the settings-driven factory (`providers/factory.rs`).
- `uml/class/platform-app/class-02-provider-http-streaming.puml`: HTTP plumbing, stream decoding, sidecar status client (`providers/http.rs`, `local_ai.rs`).
- `uml/class/platform-app/class-03-platform-audio-capture.puml`: audio capture on its own thread, 16 kHz mono frames (`platform/audio`).
- `uml/class/platform-app/class-04-platform-screen-capture.puml`: privacy-aware screen sampler (`platform/screen`).

### 3. Communication

Graphviz. Message order between objects for one turn.

- `uml/communication/communication-01-final-transcript-to-question.dot`: final transcript segment to a detected question, candidate mode (`transcription.rs`, `conversation.rs`).
- `uml/communication/communication-02-question-to-streamed-answer.dot`: detected question to a streamed suggested answer, Primary slot (`assist.rs`, UI).
- `uml/communication/communication-03-candidate-turn-to-followups.dot`: candidate turn end to streamed follow-up questions, interviewer mode (`conversation.rs`, `runtime.rs`).

### 4. Component

- `uml/component/component-01-system-overview.puml`: the desktop app, its crates, the sidecar, the interview room (separate process), Ollama, cloud APIs, OS services (`src-tauri`, `local-ai`).
- `uml/component/component-02-trait-interfaces.puml`: provided and required trait seams, `ProviderSet` and its fields (`openmeet-core`, `openmeet-providers`).

### 5. Composite structure

- `uml/composite-structure/composite-structure-01-engine-internals.puml`: parts of `Engine` (`Arc<Inner>`), their ports, and the workers connected to them (`engine.rs`, `conversation.rs`, `persist.rs`, `integrity.rs`).

### 6. Deployment

- `uml/deployment/deployment-01-single-machine.puml`: the default local-first setup on one user machine (processes, sidecar, Ollama, keyring, cloud APIs).
- `uml/deployment/deployment-02-windows.puml`: the Windows-specific pieces (audio, focus, OCR, capture exclusion, Python from the venv).
- `uml/deployment/deployment-03-interview-room.puml`: the interview room: interviewer's computer with `room_server.py`, candidate's browser, WebRTC, STUN, Ollama, optional Gemini (`local-ai/room*`, `integrity/`).

### 7. Interaction overview

- `uml/interaction-overview/interaction-overview-01-candidate-turn.puml`: one candidate turn, as references to the sequence diagrams (`conversation.rs`, `integrity.rs`, `assist.rs`).

### 8. Object

- `uml/object/object-01-interviewer-meeting-in-progress.puml`: an interviewer-mode meeting with follow-ups streaming into the Followups slot (`types.rs`, `engine.rs`).
- `uml/object/object-02-candidate-answer-streaming.puml`: a candidate-mode meeting with an answer streaming into the Primary slot (`assist.rs`, overlay state).

### 9. Package

- `uml/package/package-01-workspace.puml`: the Cargo workspace packages and their dependencies, plus `src-tauri` and `local-ai`.
- `uml/package/package-02-external-dependencies.dot`: crates and packages with the key external crates, from `Cargo.lock` and `package.json`.

### 10. Profile

- `uml/profile/profile-01-diagram-stereotypes.puml`: the stereotypes the other diagrams use, the UML metaclass each one extends, and tagged values.

### 11. Sequence

Live meeting (`uml/sequence/live-meeting/`):

- `sequence-01-start-meeting.puml`: explicit activation, engine, capture (`engine.rs`, `commands/meeting.rs`).
- `sequence-02-audio-to-transcript.puml`: capture, VAD, transcription (word timestamps when the delivery signal is on), events, overlay (`transcription.rs`, `vad.rs`, `openai.rs`).
- `sequence-03-final-segment-tracking.puml`: a final segment: question detection, conversation tracking, integrity hook (`conversation.rs`, `claims.rs`, `detect/`).
- `sequence-04-semantic-question-check.puml`: the optional semantic check for an ambiguous question (`assist.rs`, `question_detection.rs`).

System (`uml/sequence/system/`):

- `sequence-01-app-startup.puml`: app startup and `setup()` (`lib.rs`, `database.rs`, migrations).
- `sequence-02-provider-set-build.puml`: building the `ProviderSet` from settings (`providers_build.rs`, `factory.rs`, `secrets.rs`).
- `sequence-03-onboarding-first-run.puml`: first-run onboarding until `onboarding_complete` (`App.tsx`, `commands/`).
- `sequence-04-save-settings.puml`: sanitise, validate, persist, apply settings (`settings.rs`, `commands/app.rs`).

Integrity and room (`uml/sequence/integrity/`, `uml/sequence/room/`):

- `integrity/sequence-01-answer-integrity-check.puml`: in the desktop app, one answer: references, the four signals, fusion, event, store (`conversation.rs`, `integrity.rs`, `fusion.rs`, sidecar client).
- `room/sequence-01-room-one-answer.puml`: in the interview room, one answer from the invite to the flag (`room/__init__.py`, `room/checks.py`, `host.html`, `room.js`).

### 12. State machine

- `uml/state-machine/state-01-meeting-lifecycle.puml`: `MeetingState` with `PauseReason`, one runtime per meeting (`types.rs`, `runtime.rs`).
- `uml/state-machine/state-02-ai-request-answer-state.puml`: `AnswerState` of one request ticket (`requests.rs`, `assist.rs`).
- `uml/state-machine/state-03-request-slot-occupancy.puml`: the exclusive request slots, Primary and Followups (`requests.rs`).
- `uml/state-machine/state-04-answer-integrity-flag.puml`: the life of one candidate answer's integrity flag (`integrity.rs`, `conversation.rs`).

### 13. Timing

- `uml/timing/timing-01-candidate-turn-timeline.puml`: VAD onset, hangover, transcription, and the two turn-end timers, with default values (`vad.rs`, `conversation.rs`, `integrity.rs`, `settings.rs`). The times are illustrative; the rules and defaults are the code's.

### 14. Use case

- `uml/use-case/usecase-01-system-overview.puml`: what the user, the candidate, the operating system, AI providers, the sidecar and the interview room can do (`commands/`, `settings.rs`, `room_server.py`).

## Entity-relationship (ER)

- `er/er-01-physical-schema.puml`: the SQLite schema v2 as tables, columns, keys and foreign keys (`migrations/0001_initial.sql`, `0002_integrity_flags.sql`).
- `er/er-02-conceptual-chen-ownership.dot`: Chen conceptual model, meeting ownership by `meeting_id` (same migrations).
- `er/er-03-conceptual-chen-references.dot`: Chen conceptual model, cross-references held in id columns with no foreign key (same migrations).

## Flowcharts

- `flowcharts/flowchart-02-final-segment-pipeline.dot`: `Engine::handle_final()` for every final draft, both modes, including the integrity hooks (`conversation.rs`, `integrity.rs`).
- `flowcharts/flowchart-03-question-sentence-scoring.dot`: heuristic scoring of one sentence, `score_sentence()` (`intel/detect/mod.rs`).
- `flowcharts/flowchart-04-question-detection-context-merge.dot`: `detect()` with the rhetorical check, lead-ins, fragments, continuations and speaker role (`intel/detect/mod.rs`).

## C4 views

Plain PlantUML (no remote includes, so they render offline).

- `c4/c4-01-context.puml`: level 1, the system context: people, OpenMeet, the meeting app, local and remote AI services, Gemini (optional), STUN.
- `c4/c4-02-containers.puml`: level 2, the containers: desktop app, sidecar, interview room server, browser pages, SQLite, OS keyring, model files.
- `c4/c4-03-components.puml`: level 3, the components of the desktop app's Rust side, one box per crate and per module the diagrams name.

## Rendering

Needs Java 8 or newer, PlantUML and Graphviz (`dot` on the PATH). Tested with
PlantUML 1.2024.8 on Java 8 and Graphviz 14. Download the PlantUML jar from
<https://github.com/plantuml/plantuml/releases> and keep it outside the
repository.

Everything (PNG and SVG next to every source; fails on any error):

```bash
PLANTUML_JAR=/path/to/plantuml.jar bash diagrams/render.sh
```

On Windows (Git Bash), with the jar in `M:\openmeet-work\tools\`:

```bash
PLANTUML_JAR=/m/openmeet-work/tools/plantuml.jar bash diagrams/render.sh
```

One PlantUML source by hand (PlantUML takes one output format per run):

```bash
java -jar plantuml.jar -charset UTF-8 -tpng diagrams/er/er-01-physical-schema.puml
java -jar plantuml.jar -charset UTF-8 -tsvg diagrams/er/er-01-physical-schema.puml
dot -Tpng diagrams/dfd/dfd-level-0-context.dot -o diagrams/dfd/dfd-level-0-context.png
dot -Tsvg diagrams/dfd/dfd-level-0-context.dot -o diagrams/dfd/dfd-level-0-context.svg
```

## To add after workplace mode merges

Not drawn here, on purpose. When workplace mode lands, add: the `Mode`
variant and its state in the class and state diagrams, the workplace use
cases and the DFD 0 entity ("workplace team", currently removed), and any
workplace event or settings class.
