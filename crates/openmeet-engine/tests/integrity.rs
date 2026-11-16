//! Answer-integrity check end to end with deterministic mock providers:
//! interviewer question → candidate answer (real VAD + mock transcription
//! with word timestamps) → all four signals → fused flag → event → stored.
//! No network, no API keys, no audio hardware, no sidecar.

use std::sync::Arc;
use std::time::Duration;

use openmeet_core::audio::{AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::integrity::{
    IntegrityFlag, IntegrityLevel, IntegrityService, IntegritySignalName,
};
use openmeet_core::ipc::{Rect, StartMeetingRequest};
use openmeet_core::providers::{EmbeddingProvider, ProviderSet};
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::{MeetingStore, MemoryStore};
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps, VadTranscriptionProvider};
use openmeet_intel::demo::{self, DemoStep};
use openmeet_intel::integrity::spec_thresholds;
use openmeet_intel::mock::{MockEmbeddingProvider, MockIntegrityService, MockTranscriber};
use parking_lot::Mutex;
use tokio::sync::Notify;

const WAIT: Duration = Duration::from_secs(15);

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
    fn count(&self, name: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| e.name() == name)
            .count()
    }

    fn flags(&self) -> Vec<IntegrityFlag> {
        self.events
            .lock()
            .iter()
            .filter_map(|e| match e {
                MeetingEvent::IntegrityChecked { flag } => Some(flag.clone()),
                _ => None,
            })
            .collect()
    }

    /// Waits until at least `n` events named `name` were emitted.
    async fn wait_count(&self, name: &str, n: usize) -> bool {
        let deadline = tokio::time::Instant::now() + WAIT;
        loop {
            let notified = self.notify.notified();
            if self.count(name) >= n {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                let names: Vec<String> = self
                    .events
                    .lock()
                    .iter()
                    .filter(|e| e.name() != "AUDIO_LEVEL")
                    .map(|e| match e {
                        MeetingEvent::TranscriptFinal { segment } => {
                            format!("FINAL[{:?}] {}", segment.speaker_role, segment.text)
                        }
                        MeetingEvent::Notice { message, .. } => format!("NOTICE {message}"),
                        other => other.name().to_string(),
                    })
                    .collect();
                eprintln!("waiting for {n} {name}; events: {names:#?}");
                return self.count(name) >= n;
            }
        }
    }
}

/// The integrity demo's dialogue: (interviewer question, candidate answer).
fn rounds() -> Vec<(String, String)> {
    let texts: Vec<String> = demo::script(demo::INTEGRITY)
        .unwrap()
        .into_iter()
        .filter_map(|s| match s {
            DemoStep::Say { text, .. } => Some(text),
            DemoStep::Screen { .. } => None,
        })
        .collect();
    texts
        .chunks(2)
        .map(|c| (c[0].clone(), c[1].clone()))
        .collect()
}

struct Harness {
    engine: Engine,
    sink: Arc<TestSink>,
    store: Arc<MemoryStore>,
    integrity: MockIntegrityService,
    llm: openmeet_intel::mock::MockLlmProvider,
}

fn settings(region: Option<Rect>) -> Settings {
    let mut s = Settings::default();
    s.assistant.semantic_detection = false;
    s.assistant.turn_end_silence_ms = 400;
    s.assistant.followup_min_words = 8;
    s.privacy.retention = RetentionPolicy::SevenDays;
    s.integrity.enabled = true;
    s.integrity.candidate_region = region;
    s
}

fn harness(settings: Settings, bge_embedder: bool) -> Harness {
    let texts: Vec<String> = rounds().into_iter().flat_map(|(q, a)| [q, a]).collect();
    let llm = demo::demo_llm(demo::INTEGRITY);
    llm.set_chunk_delay_ms(0);
    llm.set_first_token_latency_ms(0);
    let integrity = MockIntegrityService::new()
        .with_embedding_model(spec_thresholds().echo_similarity.embedding_model.clone());
    for (p, level) in [
        (0.12, IntegrityLevel::Low),
        (0.2, IntegrityLevel::Low),
        (0.15, IntegrityLevel::Low),
        (0.9, IntegrityLevel::High),
    ] {
        integrity.push_reading(Some(p), level);
    }
    let embeddings: Option<Arc<dyn EmbeddingProvider>> = bge_embedder.then(|| {
        Arc::new(
            MockEmbeddingProvider::new()
                .with_model(spec_thresholds().echo_similarity.embedding_model.clone()),
        ) as Arc<dyn EmbeddingProvider>
    });
    let sink = Arc::new(TestSink::default());
    let store = Arc::new(MemoryStore::new());
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: store.clone(),
            providers: ProviderSet {
                llm: Some(Arc::new(llm.clone())),
                transcription: Some(Arc::new(VadTranscriptionProvider::new(Arc::new(
                    MockTranscriber::new(texts),
                )))),
                embeddings,
                integrity: Some(Arc::new(integrity.clone()) as Arc<dyn IntegrityService>),
                ..Default::default()
            },
            settings,
        },
        tokio::runtime::Handle::current(),
    );
    Harness {
        engine,
        sink,
        store,
        integrity,
        llm,
    }
}

fn interviewer() -> StartMeetingRequest {
    StartMeetingRequest {
        mode: Mode::Interviewer,
        title: Some("Integrity test".into()),
        microphone: true,
        system_audio: true,
        screen: None,
        focus_app_id: None,
    }
}

/// Pushes `speech_ms` of tone at `at_ms` (preceded by quiet and followed by
/// enough silence to end the utterance) in 100 ms frames.
fn speak(engine: &Engine, source: AudioSourceKind, at_ms: i64, speech_ms: u32) {
    let n = |ms: u32| (PIPELINE_SAMPLE_RATE * ms / 1000) as usize;
    let quiet = |ms: u32| -> Vec<f32> {
        (0..n(ms))
            .map(|i| if i % 2 == 0 { 0.0004 } else { -0.0004 })
            .collect()
    };
    let tone: Vec<f32> = (0..n(speech_ms))
        .map(|i| (i as f32 * 0.09).sin() * 0.35)
        .collect();
    let mut samples = quiet(400);
    samples.extend(tone);
    samples.extend(quiet(1300));
    let mut t = at_ms - 400;
    for chunk in samples.chunks(n(100)) {
        engine.push_audio(AudioFrame {
            source,
            samples: chunk.to_vec(),
            captured_at_ms: t,
        });
        t += 100;
    }
}

/// Speech durations (ms) and gaps after the question (ms) per round: three
/// natural answers, then one that starts late and runs fast.
const ROUNDS: [(u32, u32); 4] = [
    (10_500, 1_000),
    (11_000, 1_300),
    (10_000, 900),
    (6_000, 4_800),
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn question_answer_signals_flag_event_and_store() {
    let region = Rect {
        x: 1920,
        y: 120,
        width: 640,
        height: 360,
    };
    let h = harness(settings(Some(region)), true);
    let meeting = h.engine.start_meeting(interviewer()).await.unwrap();
    let base = meeting.started_at + 1_000;

    for (r, (speech_ms, gap_ms)) in ROUNDS.into_iter().enumerate() {
        let t = base + r as i64 * 30_000;
        speak(&h.engine, AudioSourceKind::Microphone, t, 2_000);
        assert!(
            h.sink.wait_count("QUESTION_DETECTED", r + 1).await,
            "question {r} detected"
        );
        speak(
            &h.engine,
            AudioSourceKind::System,
            t + 2_000 + i64::from(gap_ms),
            speech_ms,
        );
        assert!(
            h.sink.wait_count("INTEGRITY_CHECKED", r + 1).await,
            "answer {r} checked"
        );
    }

    let flags = h.sink.flags();
    assert_eq!(flags.len(), 4, "one check per answer");
    let ids: std::collections::HashSet<&str> = flags.iter().map(|f| f.id.as_str()).collect();
    assert_eq!(ids.len(), 4, "each answer has its own flag");

    // Answers 1-3 form the delivery baseline and raise nothing.
    for f in &flags[..3] {
        let d = f.signal(IntegritySignalName::Delivery).unwrap();
        assert_eq!(d.level, IntegrityLevel::Baseline, "{f:#?}");
        assert_eq!(d.delivery.as_ref().unwrap().granularity, "word");
        assert_eq!(
            f.signal(IntegritySignalName::AiText).unwrap().level,
            IntegrityLevel::Low
        );
        assert_eq!(
            f.signal(IntegritySignalName::Reading).unwrap().level,
            IntegrityLevel::Low
        );
        let echo = f.signal(IntegritySignalName::EchoSimilarity).unwrap();
        assert_eq!(echo.level, IntegrityLevel::Low, "{echo:#?}");
        assert_eq!(f.level, IntegrityLevel::Low);
        assert!(f.question_text.is_some());
    }

    // The read-out answer: all four signals elevated, the flag is high.
    let last = &flags[3];
    assert_eq!(last.level, IntegrityLevel::High, "{last:#?}");
    assert_eq!(
        last.elevated,
        vec![
            IntegritySignalName::AiText,
            IntegritySignalName::Delivery,
            IntegritySignalName::EchoSimilarity,
            IntegritySignalName::Reading,
        ]
    );
    assert_eq!(
        last.headline,
        openmeet_intel::integrity::evidence::headline(IntegrityLevel::High)
    );
    let delivery = last.signal(IntegritySignalName::Delivery).unwrap();
    assert!(delivery.level.is_elevated(), "{delivery:#?}");
    assert!(!delivery.calibrated);
    assert!(
        delivery
            .evidence
            .iter()
            .any(|l| l.contains("later than usual")),
        "{:?}",
        delivery.evidence
    );
    let echo = last.signal(IntegritySignalName::EchoSimilarity).unwrap();
    assert_eq!(echo.level, IntegrityLevel::High);
    assert!(echo.evidence[0].contains("(uncalibrated)"));
    assert!(
        last.question_text
            .as_deref()
            .unwrap()
            .contains("consistent hashing")
    );
    assert!(last.answer_end_ms > last.answer_start_ms);

    // Reading was asked for exactly each answer's time window, while the
    // candidate region was being captured.
    let windows = h.integrity.reading_windows();
    assert_eq!(windows.len(), 4);
    let start = (meeting.started_at + last.answer_start_ms) as f64 / 1000.0;
    let end = (meeting.started_at + last.answer_end_ms) as f64 / 1000.0;
    assert!((windows[3].0 - start).abs() < 1e-6 && (windows[3].1 - end).abs() < 1e-6);
    assert_eq!(h.integrity.region(), Some(region));
    let calls = h.integrity.calls();
    assert_eq!(&calls[..2], &["reset_gaze", "start_gaze"], "{calls:?}");

    // Reference answers were requested once per question, in the background,
    // and follow-ups still came.
    let refs = h
        .llm
        .requests()
        .iter()
        .filter(|r| r.kind == RequestKind::IntegrityReference)
        .count();
    assert_eq!(
        refs,
        4 * spec_thresholds().echo_similarity.references as usize
    );
    assert!(h.sink.count("FOLLOWUPS_READY") > 0);

    // The live snapshot carries the flags (latest first).
    let snap = h.engine.snapshot();
    assert_eq!(snap.integrity_flags.len(), 4);
    assert_eq!(snap.integrity_flags[0].id, last.id);

    // Ending the meeting stops gaze capture, forgets the gaze and keeps the
    // flags with the meeting.
    h.engine.stop_meeting().await.unwrap();
    h.engine.flush_persistence().await;
    let record = h.store.load_meeting(&meeting.id).unwrap();
    assert_eq!(record.integrity_flags.len(), 4);
    assert_eq!(
        record.integrity_flags.iter().find(|f| f.id == last.id),
        Some(last)
    );
    assert!(!h.integrity.capturing());
    let deadline = tokio::time::Instant::now() + WAIT;
    while !h.integrity.calls().ends_with(&["reset_gaze".to_string()]) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "gaze reset at the end"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        h.engine
            .current_record(&meeting.id)
            .unwrap()
            .integrity_flags
            .len(),
        4
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unavailable_signals_say_why() {
    // No candidate region and the built-in hashing embedder: reading and
    // echo cannot be scored, the other two still are.
    let h = harness(settings(None), false);
    let meeting = h.engine.start_meeting(interviewer()).await.unwrap();
    let t = meeting.started_at + 1_000;
    speak(&h.engine, AudioSourceKind::Microphone, t, 2_000);
    assert!(h.sink.wait_count("QUESTION_DETECTED", 1).await);
    speak(&h.engine, AudioSourceKind::System, t + 3_000, 10_000);
    assert!(h.sink.wait_count("INTEGRITY_CHECKED", 1).await);
    let flag = h.sink.flags().remove(0);
    let reading = flag.signal(IntegritySignalName::Reading).unwrap();
    assert_eq!(reading.level, IntegrityLevel::Unavailable);
    assert!(reading.note.as_deref().unwrap().contains("video region"));
    let echo = flag.signal(IntegritySignalName::EchoSimilarity).unwrap();
    assert_eq!(echo.level, IntegrityLevel::Unavailable);
    assert!(
        echo.note
            .as_deref()
            .unwrap()
            .contains(&spec_thresholds().echo_similarity.embedding_model)
    );
    assert!(
        flag.signal(IntegritySignalName::AiText)
            .unwrap()
            .level
            .is_scored()
    );
    assert_eq!(
        flag.signal(IntegritySignalName::Delivery).unwrap().level,
        IntegrityLevel::Baseline
    );
    assert_eq!(flag.level, IntegrityLevel::Low);
    // Without a region the gaze capture never starts.
    assert!(!h.integrity.calls().iter().any(|c| c == "start_gaze"));
    h.engine.stop_meeting().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_check_is_off_by_default_and_in_candidate_mode() {
    let mut s = settings(Some(Rect {
        x: 0,
        y: 0,
        width: 320,
        height: 240,
    }));
    s.integrity.enabled = false;
    let h = harness(s, true);
    let meeting = h.engine.start_meeting(interviewer()).await.unwrap();
    let t = meeting.started_at + 1_000;
    speak(&h.engine, AudioSourceKind::Microphone, t, 2_000);
    assert!(h.sink.wait_count("QUESTION_DETECTED", 1).await);
    speak(&h.engine, AudioSourceKind::System, t + 3_000, 10_000);
    assert!(h.sink.wait_count("FOLLOWUPS_READY", 1).await);
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(h.sink.count("INTEGRITY_CHECKED"), 0);
    assert!(h.integrity.calls().is_empty(), "{:?}", h.integrity.calls());
    assert!(
        !h.llm
            .requests()
            .iter()
            .any(|r| r.kind == RequestKind::IntegrityReference)
    );
    h.engine.stop_meeting().await.unwrap();
    assert!(
        h.store
            .load_meeting(&meeting.id)
            .unwrap()
            .integrity_flags
            .is_empty()
    );

    // Turned on, but the meeting is in candidate mode: still nothing.
    let mut s = settings(None);
    s.integrity.enabled = true;
    let h = harness(s, true);
    let mut req = interviewer();
    req.mode = Mode::Candidate;
    h.engine.start_meeting(req).await.unwrap();
    let t = now();
    speak(&h.engine, AudioSourceKind::System, t, 2_000);
    assert!(h.sink.wait_count("QUESTION_DETECTED", 1).await);
    speak(&h.engine, AudioSourceKind::Microphone, t + 3_000, 8_000);
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert_eq!(h.sink.count("INTEGRITY_CHECKED"), 0);
    assert!(h.integrity.calls().is_empty());
    h.engine.stop_meeting().await.unwrap();
}

fn now() -> i64 {
    openmeet_core::ids::now_ms()
}
