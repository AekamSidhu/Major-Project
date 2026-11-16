# Meeting Interaction Sequence

The interviewer asks a question, the candidate answers over the call, and
OpenMeet runs two things in parallel when the answer turn ends: grounded
follow-up suggestions, and the integrity check (AI-echo answer analysis plus
reading-gaze detection). Message names are `MeetingEvent` variants
(`crates/openmeet-core/src/events.rs`).

Notation (UML 2.5): `->>` synchronous call or message, `-)` asynchronous
signal, `-->>` reply / return, `par` fragments run concurrently.

```mermaid
sequenceDiagram
    autonumber
    actor Int as Interviewer (user)
    actor Cand as Candidate
    participant Cap as CaptureManager<br/>(audio + screen)
    participant Trans as VAD transcription<br/>+ Transcriber
    participant Conv as Conversation loop
    participant Req as RequestManager
    participant Integ as Integrity analyzer
    participant AI as Local AI<br/>(LLM · embeddings · gaze)
    participant UI as Overlay (webview)

    Int-)Cap: asks a question (microphone)
    Cap-)Trans: AudioFrame
    Trans-)Conv: final segment (speaker = interviewer)
    Conv->>Conv: track asked question
    Conv-)UI: TRANSCRIPT_FINAL

    Cand-)Cap: answers (system audio) and is on camera (meeting window)
    Cap-)Trans: AudioFrame
    Cap-)Integ: candidate video tile (10 fps)
    Trans-)Conv: final segments with word timestamps (speaker = candidate)
    Conv-)UI: TRANSCRIPT_FINAL
    Conv->>Conv: answer turn ends (silence, enough words)

    par Follow-up suggestions
        Conv->>Req: begin(Followups slot)
        Req-)UI: ANSWER_STARTED (requestId)
        Req->>AI: stream(interviewer-followup prompt)
        loop NDJSON items, while requestId is current
            AI-->>Req: follow-up / claim / gap
            Req-)UI: ANSWER_DELTA
        end
        Req-)UI: FOLLOWUPS_READY
    and Integrity check
        Conv->>Integ: question + answer turn + timing
        Integ->>AI: reference-answer prompt (3–4 samples)
        AI-->>Integ: reference answers
        Integ->>AI: embed answer and references
        AI-->>Integ: vectors
        Integ->>Integ: echo score + delivery features vs. baseline
        Integ->>AI: face crops while answering
        AI-->>Integ: gaze landmarks
        Integ->>Integ: reading score, fuse signals
        opt combined score ≥ threshold
            Integ-)UI: INTEGRITY_FLAG (scores, timestamps)
        end
    end
    UI-)Int: follow-ups and any integrity flag with evidence
```
