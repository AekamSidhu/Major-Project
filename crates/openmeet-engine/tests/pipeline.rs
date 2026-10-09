//! End-to-end engine tests with deterministic mock providers (spec §72).
//! No network, no API keys, no audio hardware.

use std::sync::Arc;
use std::time::Duration;

use openmeet_core::audio::{AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::now_ms;
use openmeet_core::ipc::{GenerateAnswerRequest, SimulationAction, StartMeetingRequest};
use openmeet_core::providers::{OcrEngine, ProviderSet};
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::{MeetingStore, MemoryStore};
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps, VadTranscriptionProvider};
use openmeet_intel::mock::{MockLlmProvider, MockOcr, MockTranscriber};
use parking_lot::Mutex;
use tokio::sync::Notify;

/// Sink that records events and wakes waiters.
#[derive(Default)]
struct TestSink {
    events: Mutex<Vec<MeetingEvent>>,
    notify: Notify,
}

impl EventSink for TestSink {
    fn emit(&self, event: MeetingEvent) {
        self.events.lock().push(event);
        self.notify.notify_waiters();
    }
}

impl TestSink {
    fn events(&self) -> Vec<MeetingEvent> {
        self.events.lock().clone()
    }

    fn count(&self, name: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| e.name() == name)
            .count()
    }

    /// Waits until `pred` matches an event; returns it.
    async fn wait_for(
        &self,
        timeout: Duration,
        pred: impl Fn(&MeetingEvent) -> bool,
    ) -> Option<MeetingEvent> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.notify.notified();
            if let Some(e) = self.events.lock().iter().find(|e| pred(e)).cloned() {
                return Some(e);
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return self.events.lock().iter().find(|e| pred(e)).cloned();
            }
        }
    }
}

struct Harness {
    engine: Engine,
    sink: Arc<TestSink>,
    store: Arc<MemoryStore>,
    llm: MockLlmProvider,
}

fn settings() -> Settings {
    let mut s = Settings::default();
    s.assistant.semantic_detection = false;
    s.assistant.turn_end_silence_ms = 400;
    s.assistant.followup_min_words = 8;
    s.privacy.retention = RetentionPolicy::SevenDays;
    s
}

fn harness_with(
    settings: Settings,
    providers: impl FnOnce(&MockLlmProvider) -> ProviderSet,
) -> Harness {
    let sink = Arc::new(TestSink::default());
    let store = Arc::new(MemoryStore::new());
    let llm = MockLlmProvider::new();
    llm.set_chunk_delay_ms(1);
    let providers = providers(&llm);
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: store.clone(),
            providers,
            settings,
        },
        tokio::runtime::Handle::current(),
    );
    Harness {
        engine,
        sink,
        store,
        llm,
    }
}

fn harness() -> Harness {
    harness_with(settings(), |llm| ProviderSet {
        llm: Some(Arc::new(llm.clone())),
        ..Default::default()
    })
}

fn start_request(mode: Mode) -> StartMeetingRequest {
    StartMeetingRequest {
        mode,
        title: Some("Test meeting".into()),
        microphone: false,
        system_audio: true,
        screen: None,
        focus_app_id: None,
    }
}

const WAIT: Duration = Duration::from_secs(10);

fn is_complete(e: &MeetingEvent) -> bool {
    matches!(e, MeetingEvent::AnswerComplete { .. })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn audio_to_overlay_pipeline_with_mock_providers() {
    // audio -> VAD -> mock transcription -> question detection -> context ->
    // mock LLM -> overlay events.
    let h = harness_with(settings(), |llm| {
        let transcriber = Arc::new(MockTranscriber::new(vec![
            "How would you design a rate limiter for our public API?".into(),
        ]));
        ProviderSet {
            llm: Some(Arc::new(llm.clone())),
            transcription: Some(Arc::new(VadTranscriptionProvider::new(transcriber))),
            ..Default::default()
        }
    });
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();

    // 0.4 s silence, 1.5 s "speech", 1.2 s silence, as 20 ms system-audio frames
    // timestamped in the recent past (as if captured in real time).
    let mut t = now_ms() - 3_200;
    let mut push = |samples: Vec<f32>| {
        for chunk in samples.chunks(320) {
            h.engine.push_audio(AudioFrame {
                source: AudioSourceKind::System,
                samples: chunk.to_vec(),
                captured_at_ms: t,
            });
            t += 20;
        }
    };
    let n = |ms: u32| (PIPELINE_SAMPLE_RATE * ms / 1000) as usize;
    push(
        (0..n(400))
            .map(|i| if i % 2 == 0 { 0.0004 } else { -0.0004 })
            .collect(),
    );
    push(
        (0..n(1500))
            .map(|i| (i as f32 * 0.09).sin() * 0.35)
            .collect(),
    );
    push(
        (0..n(1200))
            .map(|i| if i % 2 == 0 { 0.0004 } else { -0.0004 })
            .collect(),
    );

    let q = h
        .sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::QuestionDetected { .. }))
        .await
        .expect("question detected from audio");
    let MeetingEvent::QuestionDetected { question } = q else {
        unreachable!()
    };
    assert!(question.question.to_lowercase().contains("rate limiter"));
    assert_eq!(question.speaker_role, SpeakerRole::Remote);
    assert!(question.requires_response);

    let done = h
        .sink
        .wait_for(WAIT, is_complete)
        .await
        .expect("answer completed");
    let MeetingEvent::AnswerComplete { suggestion, .. } = done else {
        unreachable!()
    };
    assert_eq!(suggestion.kind, SuggestionKind::Answer);
    assert!(!suggestion.content.trim().is_empty());
    assert!(h.sink.count("ANSWER_DELTA") > 0, "answer must stream");
    assert!(h.sink.count("TRANSCRIPT_FINAL") >= 1);

    // The LLM saw the question with the meeting context.
    let reqs = h.llm.requests();
    let answer_req = reqs
        .iter()
        .find(|r| r.kind == RequestKind::CandidateAnswer)
        .expect("candidate answer request");
    assert!(
        answer_req.messages[0]
            .content
            .to_lowercase()
            .contains("rate limiter")
    );

    // Latency diagnostics were recorded for each stage of the pipeline.
    let stages: Vec<LatencyStage> = h.engine.latency_stats().iter().map(|s| s.stage).collect();
    assert!(stages.contains(&LatencyStage::AudioToTranscript));
    assert!(stages.contains(&LatencyStage::TranscriptToDetection));
    assert!(stages.contains(&LatencyStage::DetectionToFirstToken));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interviewer_mode_generates_grounded_followups() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Interviewer))
        .await
        .unwrap();
    h.engine
        .simulate(SimulationAction::Transcript {
            speaker: SpeakerRole::User,
            text: "How would you design a rate limiter?".into(),
        })
        .unwrap();
    h.engine
        .simulate(SimulationAction::CandidateAnswer {
            text: "I'd probably use Redis with a token bucket per API key and keep the counters in memory for speed.".into(),
        })
        .unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::FollowupsReady { done: true, .. })
        })
        .await
        .expect("follow-ups after the candidate's turn");
    let MeetingEvent::FollowupsReady { questions, .. } = ev else {
        unreachable!()
    };
    assert!(!questions.is_empty(), "at least one follow-up");
    for f in &questions {
        assert!(!f.question.trim().is_empty());
        assert!(!f.reason.trim().is_empty(), "each follow-up explains why");
        assert_ne!(
            f.question.to_lowercase(),
            "how would you design a rate limiter?",
            "must not repeat the asked question"
        );
    }
    let req = h
        .llm
        .requests()
        .into_iter()
        .find(|r| r.kind == RequestKind::InterviewerFollowups)
        .unwrap();
    assert!(
        req.messages[0].content.contains("Redis"),
        "prompt grounded in the candidate's words"
    );
    // The asked question was recorded for duplicate detection.
    let snap = h.engine.snapshot();
    assert!(
        snap.questions
            .iter()
            .any(|q| q.speaker_role == SpeakerRole::User)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_followups_are_filtered() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Interviewer))
        .await
        .unwrap();
    h.llm.push_response(
        [
            r#"{"type":"followup","question":"How did you handle API scaling?","category":"technical","reason":"repeat","priority":"normal","trigger":"we scaled"}"#,
            r#"{"type":"followup","question":"Which component became the scaling bottleneck?","category":"technical","reason":"Candidate mentioned scaling but not the bottleneck","priority":"high","trigger":"we scaled the API"}"#,
        ]
        .join("\n"),
    );
    h.engine
        .simulate(SimulationAction::Transcript {
            speaker: SpeakerRole::User,
            text: "How did you scale the API?".into(),
        })
        .unwrap();
    h.engine
        .simulate(SimulationAction::CandidateAnswer {
            text: "We scaled the API horizontally behind a load balancer and added caching in front of the database.".into(),
        })
        .unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::FollowupsReady { done: true, .. })
        })
        .await
        .expect("follow-ups");
    let MeetingEvent::FollowupsReady { questions, .. } = ev else {
        unreachable!()
    };
    let texts: Vec<&str> = questions.iter().map(|q| q.question.as_str()).collect();
    assert_eq!(
        texts,
        vec!["Which component became the scaling bottleneck?"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn newer_question_cancels_stale_answer() {
    let h = harness();
    h.llm.set_first_token_latency_ms(400);
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine
        .simulate(SimulationAction::Question {
            text: "How did you design the billing system?".into(),
        })
        .unwrap();
    let first = h
        .sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::AnswerStarted { .. }))
        .await
        .unwrap();
    let MeetingEvent::AnswerStarted { request_id: a, .. } = first else {
        unreachable!()
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    h.engine
        .simulate(SimulationAction::Question {
            text: "What database did you choose and why?".into(),
        })
        .unwrap();
    let done = h.sink.wait_for(WAIT, is_complete).await.unwrap();
    let MeetingEvent::AnswerComplete {
        request_id: b,
        suggestion,
    } = done
    else {
        unreachable!()
    };
    assert_ne!(a, b, "the completed answer belongs to the newer question");
    assert!(suggestion.question_text.unwrap().contains("database"));
    tokio::time::sleep(Duration::from_millis(600)).await;
    // No deltas or completion for the stale request were ever emitted.
    for e in h.sink.events() {
        match e {
            MeetingEvent::AnswerDelta { request_id, .. }
            | MeetingEvent::AnswerComplete { request_id, .. } => {
                assert_ne!(request_id, a, "stale answer leaked to the UI");
            }
            _ => {}
        }
    }
    assert!(h.sink.events().iter().any(|e| matches!(e,
        MeetingEvent::AnswerStateChanged { request_id, state: AnswerState::Cancelled, .. } if *request_id == a)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn emergency_stop_halts_everything_immediately() {
    let h = harness();
    h.llm.set_first_token_latency_ms(50);
    h.llm.set_chunk_delay_ms(50);
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine
        .simulate(SimulationAction::Question {
            text: "Can you explain how you handled retries in the payment service?".into(),
        })
        .unwrap();
    h.sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::AnswerDelta { .. }))
        .await
        .expect("streaming started");
    h.engine.emergency_stop();
    let deltas_at_stop = h.sink.count("ANSWER_DELTA");
    assert!(h.sink.count("PROCESSING_STOPPED") == 1);
    assert_eq!(h.engine.meeting_state(), MeetingState::Paused);
    assert!(h.engine.processing_stopped());
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        h.sink.count("ANSWER_DELTA"),
        deltas_at_stop,
        "no output after the stop"
    );
    assert_eq!(h.sink.count("ANSWER_COMPLETE"), 0);
    // New requests are refused until the user resumes.
    assert!(h.engine.ask("anything".into()).is_err());
    h.engine.resume().unwrap();
    assert!(!h.engine.processing_stopped());
    assert!(
        h.engine
            .ask("Now explain idempotency keys briefly.".into())
            .is_ok()
    );
    h.sink
        .wait_for(WAIT, is_complete)
        .await
        .expect("works again after resume");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paused_meeting_does_not_auto_answer() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine.pause().unwrap();
    h.engine
        .simulate(SimulationAction::Question {
            text: "What was the hardest bug you fixed last year?".into(),
        })
        .unwrap();
    h.sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::TranscriptFinal { .. }))
        .await
        .expect("speech before the pause is still transcribed");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(h.sink.count("ANSWER_STARTED"), 0);
    assert_eq!(h.llm.request_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn provider_failure_is_reported_and_transcription_continues() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.llm.push_failure(openmeet_core::ProviderError::Network(
        "connection refused".into(),
    ));
    h.engine
        .simulate(SimulationAction::Question {
            text: "How do you approach code reviews?".into(),
        })
        .unwrap();
    let failed = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(
                e,
                MeetingEvent::AnswerStateChanged {
                    state: AnswerState::Failed,
                    ..
                }
            )
        })
        .await
        .expect("failure is surfaced, never silent");
    let MeetingEvent::AnswerStateChanged { error, .. } = failed else {
        unreachable!()
    };
    assert!(error.unwrap().contains("AI provider unavailable"));
    // The pipeline keeps working.
    h.engine
        .simulate(SimulationAction::Transcript {
            speaker: SpeakerRole::User,
            text: "I usually start with the tests.".into(),
        })
        .unwrap();
    h.sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::TranscriptFinal { segment } if segment.text.contains("tests")))
        .await
        .expect("transcription unaffected by LLM failure");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn manual_ask_works_without_a_meeting_or_transcription() {
    let h = harness();
    let id = h
        .engine
        .ask("Explain eventual consistency simply.".into())
        .unwrap();
    let done = h.sink.wait_for(WAIT, is_complete).await.unwrap();
    let MeetingEvent::AnswerComplete {
        request_id,
        suggestion,
    } = done
    else {
        unreachable!()
    };
    assert_eq!(request_id, id);
    assert_eq!(suggestion.kind, SuggestionKind::Ask);
    // Refinements work on detached suggestions too.
    let refined = h.engine.refine(&id, Refinement::Shorter).unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::AnswerComplete { request_id, .. } if *request_id == refined))
        .await
        .unwrap();
    let MeetingEvent::AnswerComplete { suggestion, .. } = ev else {
        unreachable!()
    };
    assert_eq!(suggestion.parent_id.as_deref(), Some(id.as_str()));
    assert_eq!(suggestion.kind, SuggestionKind::Refinement);
    assert!(h.engine.ask("   ".into()).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stop_meeting_produces_summary_and_persists_everything() {
    let h = harness();
    let meeting = h
        .engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine
        .simulate(SimulationAction::Question {
            text: "Why did you choose Kafka for the event pipeline?".into(),
        })
        .unwrap();
    h.sink.wait_for(WAIT, is_complete).await.unwrap();
    h.engine
        .simulate(SimulationAction::Transcript {
            speaker: SpeakerRole::User,
            text:
                "We needed durable, replayable events and our team already ran Kafka in production."
                    .into(),
        })
        .unwrap();
    h.sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::TranscriptFinal { segment } if segment.text.contains("durable")))
        .await
        .unwrap();
    h.engine.stop_meeting().await.unwrap();
    let ev = h
        .sink
        .wait_for(Duration::from_secs(20), |e| {
            matches!(e, MeetingEvent::SummaryReady { .. })
        })
        .await
        .expect("summary generated");
    let MeetingEvent::SummaryReady { summary } = ev else {
        unreachable!()
    };
    assert_eq!(summary.meeting_id, meeting.id);
    assert!(!summary.markdown.trim().is_empty());
    assert_eq!(h.engine.meeting_state(), MeetingState::Completed);
    h.engine.flush_persistence().await;
    let record = h
        .store
        .load_meeting(&meeting.id)
        .expect("meeting persisted");
    assert_eq!(record.meeting.state, MeetingState::Completed);
    assert!(record.meeting.ended_at.is_some());
    assert_eq!(record.segments.len(), 2);
    assert_eq!(record.questions.len(), 1);
    assert_eq!(
        record.answers.len(),
        1,
        "the user's spoken answer is linked to the question"
    );
    assert!(record.answers[0].text.contains("durable"));
    assert!(
        record
            .suggestions
            .iter()
            .any(|s| s.kind == SuggestionKind::Answer)
    );
    assert!(record.summary.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retention_none_persists_nothing() {
    let mut s = settings();
    s.privacy.retention = RetentionPolicy::None;
    let h = harness_with(s, |llm| ProviderSet {
        llm: Some(Arc::new(llm.clone())),
        ..Default::default()
    });
    let meeting = h
        .engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    assert!(!meeting.persisted);
    h.engine
        .simulate(SimulationAction::Question {
            text: "What motivates you in your work?".into(),
        })
        .unwrap();
    h.sink.wait_for(WAIT, is_complete).await.unwrap();
    h.engine.stop_meeting().await.unwrap();
    h.sink
        .wait_for(Duration::from_secs(20), |e| {
            matches!(e, MeetingEvent::SummaryReady { .. })
        })
        .await
        .unwrap();
    h.engine.flush_persistence().await;
    assert!(
        h.store.load_meeting(&meeting.id).is_err(),
        "nothing written to disk"
    );
    assert_eq!(h.store.embedding_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn screen_context_is_fused_with_the_question() {
    let h = harness_with(settings(), |llm| ProviderSet {
        llm: Some(Arc::new(llm.clone())),
        ocr: Some(Arc::new(MockOcr {
            text: "API -> Queue -> Workers -> PostgreSQL".into(),
        }) as Arc<dyn OcrEngine>),
        ..Default::default()
    });
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine.push_screen_frame(ScreenFrame {
        id: "f1".into(),
        captured_at: now_ms(),
        source_label: "Display 1".into(),
        application: Some("Excalidraw".into()),
        window_title: Some("architecture.excalidraw".into()),
        width: 4,
        height: 4,
        png: vec![0u8; 16],
        hash: 42,
        change_score: 1.0,
        manual: true,
    });
    let ev = h
        .sink
        .wait_for(WAIT, |e| matches!(e, MeetingEvent::ScreenUpdated { .. }))
        .await
        .expect("observation produced from OCR");
    let MeetingEvent::ScreenUpdated { observation } = ev else {
        unreachable!()
    };
    assert!(observation.visible_text.unwrap().contains("Queue"));
    assert_eq!(observation.frame_hash.as_deref(), Some("000000000000002a"));

    h.engine
        .simulate(SimulationAction::Question {
            text: "Can you explain this part of the architecture?".into(),
        })
        .unwrap();
    h.sink.wait_for(WAIT, is_complete).await.unwrap();
    let req = h
        .llm
        .requests()
        .into_iter()
        .find(|r| r.kind == RequestKind::CandidateAnswer)
        .unwrap();
    assert!(
        req.messages[0].content.contains("Workers"),
        "screen observation must be part of the context for deictic questions"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn answer_controls_pin_and_dismiss() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    let id = h
        .engine
        .generate_answer(GenerateAnswerRequest {
            question: Some("What is a good API pagination strategy?".into()),
            style: Some(AnswerStyle::Technical),
        })
        .unwrap();
    h.sink.wait_for(WAIT, is_complete).await.unwrap();
    h.engine.pin(&id, true).unwrap();
    assert_eq!(h.engine.snapshot().pinned.len(), 1);
    h.engine.dismiss(&id);
    assert_eq!(h.sink.count("SUGGESTION_DISMISSED"), 1);
    assert!(h.engine.snapshot().active.is_none());
    assert!(h.engine.refine("nope", Refinement::Shorter).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn meeting_lifecycle_rejects_invalid_transitions() {
    let h = harness();
    assert!(h.engine.pause().is_err());
    assert!(h.engine.stop_meeting().await.is_err());
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    assert!(
        h.engine
            .start_meeting(start_request(Mode::Candidate))
            .await
            .is_err()
    );
    h.engine.pause().unwrap();
    h.engine.pause().unwrap(); // idempotent
    h.engine.resume().unwrap();
    h.engine.stop_meeting().await.unwrap();
    assert!(h.engine.resume().is_err());
    let states: Vec<MeetingState> = h
        .sink
        .events()
        .into_iter()
        .filter_map(|e| match e {
            MeetingEvent::MeetingStateChanged { state, .. } => Some(state),
            _ => None,
        })
        .collect();
    assert!(states.starts_with(&[
        MeetingState::Starting,
        MeetingState::Active,
        MeetingState::Paused,
        MeetingState::Active,
        MeetingState::Stopping
    ]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mode_toggle_and_style_cycle() {
    let h = harness();
    assert_eq!(h.engine.toggle_mode(), Mode::Interviewer);
    assert_eq!(h.engine.mode(), Mode::Interviewer);
    let s0 = h.engine.style();
    assert_ne!(h.engine.cycle_style(), s0);
    assert!(h.sink.count("MODE_CHANGED") >= 1);
    assert!(h.sink.count("STYLE_CHANGED") >= 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn demo_mode_runs_without_external_services() {
    let h = harness();
    let scenarios = h.engine.demo_scenarios();
    assert!(scenarios.iter().any(|s| s.id == "rate-limiter-interviewer"));
    let meeting = h
        .engine
        .start_demo("rate-limiter-interviewer")
        .await
        .unwrap();
    assert!(!meeting.persisted, "demo meetings are never stored");
    assert!(h.engine.demo_running());
    h.sink
        .wait_for(Duration::from_secs(60), |e| {
            matches!(e, MeetingEvent::FollowupsReady { done: true, .. })
        })
        .await
        .expect("demo produces follow-ups");
    h.engine.stop_demo().await.unwrap();
    h.sink
        .wait_for(Duration::from_secs(30), |e| {
            matches!(e, MeetingEvent::SummaryReady { .. })
        })
        .await
        .expect("demo summary");
    assert!(!h.engine.demo_running());
    // The configured providers are restored after the demo.
    assert!(h.engine.ask("Back to normal?".into()).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn developer_simulation_of_ai_failure_and_latency() {
    let h = harness();
    h.engine
        .start_meeting(start_request(Mode::Candidate))
        .await
        .unwrap();
    h.engine
        .simulate(SimulationAction::AiFailure {
            message: "simulated outage".into(),
        })
        .unwrap();
    assert!(h.engine.simulation_active());
    h.engine.ask("Will this fail?".into()).unwrap();
    h.sink
        .wait_for(WAIT, |e| {
            matches!(
                e,
                MeetingEvent::AnswerStateChanged {
                    state: AnswerState::Failed,
                    ..
                }
            )
        })
        .await
        .expect("simulated failure surfaced");
    h.engine
        .simulate(SimulationAction::AiResponse {
            text: "[[ANSWER]]\nSimulated answer.".into(),
        })
        .unwrap();
    h.engine.ask("And now?".into()).unwrap();
    let ev = h.sink.wait_for(WAIT, is_complete).await.unwrap();
    let MeetingEvent::AnswerComplete { suggestion, .. } = ev else {
        unreachable!()
    };
    assert!(suggestion.content.contains("Simulated answer"));
    h.engine.clear_simulation();
    assert!(!h.engine.simulation_active());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn benchmark_runs_offline() {
    let h = harness();
    let items = h.engine.run_benchmark().await;
    let names: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
    assert!(names.contains(&"Question detection"));
    assert!(names.iter().any(|n| n.starts_with("Pipeline: question")));
    for i in &items {
        if i.name == "Question detection" {
            assert!(
                i.value < 5.0,
                "detection should be well under 5 ms, got {}",
                i.value
            );
        }
    }
}
