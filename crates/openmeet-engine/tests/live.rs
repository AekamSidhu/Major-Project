//! Live end-to-end test against real local services: recorded speech → VAD →
//! local AI sidecar (Whisper) → question detection → Ollama → overlay
//! events, then interviewer follow-ups and the meeting summary.
//!
//! Ignored by default. Needs Ollama serving `llama3.2`, the local AI sidecar
//! on 127.0.0.1:8765 and two 16 kHz mono 16-bit WAV files:
//!
//! ```text
//! OPENMEET_LIVE_QUESTION_WAV=question.wav   # asks how to design a rate limiter
//! OPENMEET_LIVE_ANSWER_WAV=answer.wav       # a candidate answer about Redis
//! cargo test -p openmeet-engine --test live -- --ignored --nocapture
//! ```
//!
//! Output is limited to timings and sizes plus the recognized question.

use std::sync::Arc;
use std::time::{Duration, Instant};

use openmeet_core::audio::{AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::now_ms;
use openmeet_core::ipc::StartMeetingRequest;
use openmeet_core::providers::{ProviderSet, StaticSecrets};
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::{MeetingStore, MemoryStore};
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps, VadTranscriptionProvider};
use openmeet_providers::factory::{self, BuildContext};
use parking_lot::Mutex;
use tokio::sync::Notify;

#[derive(Default)]
struct Sink {
    events: Mutex<Vec<MeetingEvent>>,
    notify: Notify,
}

impl EventSink for Sink {
    fn emit(&self, event: MeetingEvent) {
        self.events.lock().push(event);
        self.notify.notify_waiters();
    }
}

impl Sink {
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
                return None;
            }
        }
    }

    fn failures(&self) -> Vec<String> {
        self.events
            .lock()
            .iter()
            .filter_map(|e| match e {
                MeetingEvent::AnswerStateChanged {
                    state: AnswerState::Failed,
                    error,
                    ..
                } => error.clone(),
                MeetingEvent::ComponentStatusChanged { health }
                    if health.status == ComponentStatus::Error =>
                {
                    Some(format!("{:?}: {:?}", health.component, health.message))
                }
                _ => None,
            })
            .collect()
    }
}

/// Reads a 16 kHz mono 16-bit PCM WAV file.
fn read_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert!(bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut i = 12;
    let mut format_ok = false;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &bytes[i + 8..(i + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            assert_eq!(
                (channels, rate, bits),
                (1, PIPELINE_SAMPLE_RATE, 16),
                "{path}: need 16 kHz mono 16-bit PCM"
            );
            format_ok = true;
        } else if id == b"data" {
            assert!(format_ok, "{path}: data before fmt");
            return body
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("{path}: no data chunk");
}

/// Pushes `speech` followed by trailing quiet as 20 ms system-audio frames,
/// timestamped as if they had just been captured.
fn push_speech(engine: &Engine, speech: &[f32]) {
    let quiet: Vec<f32> = (0..PIPELINE_SAMPLE_RATE as usize * 3 / 2)
        .map(|i| if i % 2 == 0 { 0.0004 } else { -0.0004 })
        .collect();
    let all: Vec<f32> = speech.iter().copied().chain(quiet).collect();
    let total_ms = (all.len() as u64 * 1000 / u64::from(PIPELINE_SAMPLE_RATE)) as i64;
    let mut t = now_ms() - total_ms;
    for chunk in all.chunks(320) {
        engine.push_audio(AudioFrame {
            source: AudioSourceKind::System,
            samples: chunk.to_vec(),
            captured_at_ms: t,
        });
        t += 20;
    }
}

const SLOW: Duration = Duration::from_secs(240);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs Ollama (llama3.2), the local AI sidecar and WAV files"]
async fn live_local_pipeline() {
    let question_wav =
        std::env::var("OPENMEET_LIVE_QUESTION_WAV").expect("OPENMEET_LIVE_QUESTION_WAV");
    let answer_wav = std::env::var("OPENMEET_LIVE_ANSWER_WAV").expect("OPENMEET_LIVE_ANSWER_WAV");

    let mut settings = Settings::default();
    settings.assistant.turn_end_silence_ms = 800;
    // Domain vocabulary is passed to Whisper as a prompt (Settings → Transcription).
    settings.transcription.vocabulary = "rate limiter, API, Redis, token bucket, failover".into();
    settings.privacy.retention = RetentionPolicy::SevenDays;
    let secrets = StaticSecrets::default();
    let ctx = BuildContext {
        settings: &settings,
        secrets: &secrets,
    };
    let llm = factory::build_llm(&ctx).expect("Ollama provider");
    let transcriber = factory::build_transcriber(&ctx).expect("local AI transcriber");
    println!(
        "providers: llm={:?}/{} transcription={:?}/{}",
        settings.ai.llm.provider,
        settings.ai.llm.model,
        settings.transcription.provider,
        settings.transcription.model
    );

    let sink = Arc::new(Sink::default());
    let store = Arc::new(MemoryStore::new());
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: store.clone(),
            providers: ProviderSet {
                llm: Some(llm),
                transcription: Some(Arc::new(VadTranscriptionProvider::new(transcriber))),
                ..Default::default()
            },
            settings: settings.clone(),
        },
        tokio::runtime::Handle::current(),
    );

    // --- Candidate mode: spoken question → suggested answer ----------------
    let meeting = engine
        .start_meeting(StartMeetingRequest {
            mode: Mode::Candidate,
            title: Some("Live test".into()),
            microphone: false,
            system_audio: true,
            screen: None,
            focus_app_id: None,
        })
        .await
        .expect("start meeting");
    let t0 = Instant::now();
    push_speech(&engine, &read_wav(&question_wav));

    let ev = sink
        .wait_for(SLOW, |e| matches!(e, MeetingEvent::TranscriptFinal { .. }))
        .await
        .unwrap_or_else(|| panic!("no transcript; failures: {:?}", sink.failures()));
    let MeetingEvent::TranscriptFinal { segment } = ev else {
        unreachable!()
    };
    println!("transcript after {:?}: {:?}", t0.elapsed(), segment.text);
    let heard = segment.text.to_lowercase();
    assert!(
        heard.contains("rate") && heard.contains("api"),
        "unexpected transcript"
    );

    let ev = sink
        .wait_for(SLOW, |e| matches!(e, MeetingEvent::QuestionDetected { .. }))
        .await
        .expect("question detected");
    let MeetingEvent::QuestionDetected { question } = ev else {
        unreachable!()
    };
    println!(
        "question after {:?}: category={:?} method={:?}",
        t0.elapsed(),
        question.category,
        question.method
    );
    assert!(question.requires_response);

    let first = sink
        .wait_for(SLOW, |e| matches!(e, MeetingEvent::AnswerDelta { .. }))
        .await
        .unwrap_or_else(|| panic!("no answer tokens; failures: {:?}", sink.failures()));
    drop(first);
    println!("first answer token after {:?}", t0.elapsed());
    let ev = sink
        .wait_for(SLOW, |e| matches!(e, MeetingEvent::AnswerComplete { .. }))
        .await
        .unwrap_or_else(|| panic!("answer did not complete; failures: {:?}", sink.failures()));
    let MeetingEvent::AnswerComplete { suggestion, .. } = ev else {
        unreachable!()
    };
    println!(
        "answer complete after {:?}: {} chars, sectioned={}",
        t0.elapsed(),
        suggestion.content.len(),
        suggestion.content.contains("[[ANSWER]]")
    );
    assert!(
        suggestion.content.contains("[[ANSWER]]"),
        "the model followed the answer format"
    );

    // --- Interviewer mode: spoken candidate answer → follow-ups --------------
    engine.set_mode(Mode::Interviewer);
    let t1 = Instant::now();
    push_speech(&engine, &read_wav(&answer_wav));
    let ev = sink
        .wait_for(SLOW, |e| {
            matches!(e, MeetingEvent::FollowupsReady { done: true, .. })
        })
        .await
        .unwrap_or_else(|| panic!("no follow-ups; failures: {:?}", sink.failures()));
    let MeetingEvent::FollowupsReady { questions, .. } = ev else {
        unreachable!()
    };
    println!("{} follow-ups after {:?}", questions.len(), t1.elapsed());
    for q in &questions {
        println!("  - [{:?}/{:?}] {}", q.category, q.priority, q.question);
    }
    assert!(!questions.is_empty());

    // --- Summary ------------------------------------------------------------
    let t2 = Instant::now();
    engine.stop_meeting().await.expect("stop");
    let ev = sink
        .wait_for(SLOW, |e| matches!(e, MeetingEvent::SummaryReady { .. }))
        .await
        .expect("summary");
    let MeetingEvent::SummaryReady { summary } = ev else {
        unreachable!()
    };
    println!(
        "summary after {:?}: {} chars of markdown",
        t2.elapsed(),
        summary.markdown.len()
    );
    engine.flush_persistence().await;
    let record = store.load_meeting(&meeting.id).expect("persisted");
    assert!(record.summary.is_some());
    for s in engine.latency_stats() {
        println!(
            "latency {:?}: last={}ms p50={}ms",
            s.stage, s.last_ms, s.p50_ms
        );
    }
    assert!(
        sink.failures().is_empty(),
        "failures: {:?}",
        sink.failures()
    );
}
