//! Diagnostics benchmark (spec §75).
//!
//! Measures the engine's own processing cost with synthetic input and a
//! deterministic local mock provider, so it runs offline and never sends
//! meeting data anywhere. Real-provider latencies come from the live latency
//! tracker and are reported alongside.

use std::sync::Arc;
use std::time::{Duration, Instant};

use openmeet_core::audio::{AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ipc::{BenchmarkItem, SimulationAction, StartMeetingRequest};
use openmeet_core::providers::ProviderSet;
use openmeet_core::settings::Settings;
use openmeet_core::store::MemoryStore;
use openmeet_core::types::*;
use openmeet_intel::detect::{DetectionInput, detect};
use openmeet_intel::mock::MockLlmProvider;
use openmeet_intel::prompts::{PromptContext, TranscriptLine, candidate_answer};
use openmeet_intel::vad::{Segmenter, VadConfig};
use tokio::sync::mpsc;

use crate::engine::{Engine, EngineDeps};
use crate::memory::{ContextMemory, MemoryItem, MemoryKind, RetrievalQuery};

struct ChannelSink(mpsc::UnboundedSender<MeetingEvent>);

impl EventSink for ChannelSink {
    fn emit(&self, event: MeetingEvent) {
        let _ = self.0.send(event);
    }
}

fn item(name: &str, value: f64, unit: &str, note: Option<&str>) -> BenchmarkItem {
    BenchmarkItem {
        name: name.to_string(),
        value: (value * 1000.0).round() / 1000.0,
        unit: unit.to_string(),
        note: note.map(str::to_string),
    }
}

fn sample_segment(i: usize, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        id: format!("s{i}"),
        meeting_id: "bench".into(),
        utterance_id: format!("u{i}"),
        speaker_id: Some("remote".into()),
        speaker_role: SpeakerRole::Remote,
        source: AudioSourceKind::System,
        start_time: i as i64 * 3000,
        end_time: i as i64 * 3000 + 2500,
        text: text.into(),
        is_final: true,
        confidence: None,
    }
}

const SAMPLES: [&str; 8] = [
    "How would you design a rate limiter for a public API?",
    "We moved the billing service to Kafka last year.",
    "Tell me about a time you disagreed with your manager.",
    "So the latency dropped from 400 to 120 milliseconds.",
    "Can you walk me through how the cache invalidation worked",
    "I think that's everything on my side.",
    "What trade-offs did you consider when choosing Postgres?",
    "The question I was thinking about was the schema design.",
];

impl Engine {
    /// Runs the offline benchmark suite.
    pub async fn run_benchmark(&self) -> Vec<BenchmarkItem> {
        let mut items = Vec::new();

        // 1. Question detection.
        let segments: Vec<TranscriptSegment> = SAMPLES
            .iter()
            .enumerate()
            .map(|(i, t)| sample_segment(i, t))
            .collect();
        let iterations = 400;
        let t = Instant::now();
        for n in 0..iterations {
            let i = n % segments.len();
            let _ = detect(&DetectionInput {
                segment: &segments[i],
                recent: &segments[..i],
                mode: Mode::Candidate,
                last_question: None,
            });
        }
        items.push(item(
            "Question detection",
            t.elapsed().as_secs_f64() * 1000.0 / iterations as f64,
            "ms",
            Some("Heuristic detector per final transcript segment"),
        ));

        // 2. VAD throughput.
        let seconds = 30u32;
        let n = (PIPELINE_SAMPLE_RATE * seconds) as usize;
        let audio: Vec<f32> = (0..n)
            .map(|i| {
                let burst = (i / (PIPELINE_SAMPLE_RATE as usize * 2)) % 2 == 0;
                if burst {
                    (i as f32 * 0.07).sin() * 0.3
                } else {
                    0.0005 * if i % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect();
        let mut seg = Segmenter::new(AudioSourceKind::Microphone, VadConfig::default());
        let t = Instant::now();
        let mut ts = 0i64;
        for chunk in audio.chunks(320) {
            let _ = seg.push(&AudioFrame {
                source: AudioSourceKind::Microphone,
                samples: chunk.to_vec(),
                captured_at_ms: ts,
            });
            ts += 20;
        }
        let elapsed = t.elapsed().as_secs_f64();
        items.push(item(
            "Voice activity detection",
            seconds as f64 / elapsed.max(1e-9),
            "x realtime",
            Some("Higher is better"),
        ));

        // 3. Local embeddings.
        let hashing = self.hashing();
        let t = Instant::now();
        for i in 0..300 {
            let _ = hashing.embed_one(SAMPLES[i % SAMPLES.len()]);
        }
        items.push(item(
            "Local embedding",
            t.elapsed().as_secs_f64() * 1000.0 / 300.0,
            "ms",
            Some("Built-in hashing embedder"),
        ));

        // 4. Retrieval over a long meeting.
        let mut memory = ContextMemory::new();
        for i in 0..1500 {
            let text = SAMPLES[i % SAMPLES.len()];
            let id = format!("m{i}");
            memory.add(MemoryItem {
                id: id.clone(),
                kind: MemoryKind::Segment,
                text: format!("{text} #{i}"),
                at_ms: i as i64 * 2000,
            });
            memory.set_vector(&id, "hash", hashing.embed_one(text));
        }
        let q = hashing.embed_one("why did you pick kafka");
        let t = Instant::now();
        for _ in 0..50 {
            let _ = memory.retrieve(RetrievalQuery {
                text: "why did you pick kafka",
                vector: Some(("hash", &q)),
                k: 6,
                exclude_from_ms: Some(2_800_000),
                now_ms: 3_000_000,
                boost_screen: false,
            });
        }
        items.push(item(
            "Context retrieval (1,500 items)",
            t.elapsed().as_secs_f64() * 1000.0 / 50.0,
            "ms",
            None,
        ));

        // 5. Prompt assembly.
        let ctx = PromptContext {
            mode: Mode::Candidate,
            recent_transcript: (0..120)
                .map(|i| TranscriptLine {
                    speaker: "Interviewer".into(),
                    text: SAMPLES[i % SAMPLES.len()].into(),
                    at_ms: i as i64 * 1500,
                })
                .collect(),
            ..Default::default()
        };
        let t = Instant::now();
        for _ in 0..100 {
            let _ = candidate_answer::build(
                &ctx,
                SAMPLES[0],
                QuestionCategory::SystemDesign,
                AnswerStyle::Concise,
                ResponseLength::Medium,
                3500,
            );
        }
        items.push(item(
            "Prompt assembly",
            t.elapsed().as_secs_f64() * 1000.0 / 100.0,
            "ms",
            None,
        ));

        // 6. End-to-end pipeline with the mock provider (no network).
        match end_to_end(&self.inner.handle).await {
            Some((detect_ms, first_ms, complete_ms)) => {
                items.push(item(
                    "Pipeline: transcript → question detected",
                    detect_ms,
                    "ms",
                    Some("Includes event bus and conversation loop"),
                ));
                items.push(item(
                    "Pipeline: question → first token (mock LLM)",
                    first_ms,
                    "ms",
                    Some("Engine overhead only; provider latency excluded"),
                ));
                items.push(item(
                    "Pipeline: first token → complete (mock LLM)",
                    complete_ms,
                    "ms",
                    None,
                ));
            }
            None => items.push(item(
                "Pipeline end-to-end",
                f64::NAN,
                "ms",
                Some("Did not complete within 10 s"),
            )),
        }

        // 7. Observed real-world latencies.
        for s in self.latency_stats() {
            let name = match s.stage {
                LatencyStage::AudioToTranscript => "Observed: audio → transcript (p50)",
                LatencyStage::TranscriptToDetection => "Observed: transcript → detection (p50)",
                LatencyStage::DetectionToFirstToken => "Observed: request → first token (p50)",
                LatencyStage::FirstTokenToComplete => "Observed: first token → complete (p50)",
                LatencyStage::ScreenProcessing => "Observed: screen processing (p50)",
                LatencyStage::OverlayRender => "Observed: overlay render (p50)",
            };
            items.push(item(name, s.p50_ms, "ms", Some(&format!("{} samples, p95 {:.0} ms", s.count, s.p95_ms))));
        }
        items
    }
}

/// Runs a throwaway engine with a mock LLM and measures detection and
/// streaming overhead for one injected question.
async fn end_to_end(handle: &tokio::runtime::Handle) -> Option<(f64, f64, f64)> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mock = MockLlmProvider::new();
    mock.set_chunk_delay_ms(0);
    let mut settings = Settings::default();
    settings.privacy.retention = openmeet_core::settings::RetentionPolicy::None;
    settings.assistant.semantic_detection = false;
    let engine = Engine::new(
        EngineDeps {
            sink: Arc::new(ChannelSink(tx)),
            store: Arc::new(MemoryStore::new()),
            providers: ProviderSet {
                llm: Some(Arc::new(mock)),
                ..Default::default()
            },
            settings,
        },
        handle.clone(),
    );
    engine
        .start_meeting(StartMeetingRequest {
            mode: Mode::Candidate,
            title: Some("Benchmark".into()),
            microphone: false,
            system_audio: true,
            screen: None,
            focus_app_id: None,
        })
        .await
        .ok()?;
    let start = Instant::now();
    engine
        .simulate(SimulationAction::Question {
            text: "How would you design a URL shortener?".into(),
        })
        .ok()?;
    let mut detected = None;
    let mut first = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let result = loop {
        let ev = match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(ev)) => ev,
            _ => break None,
        };
        match ev {
            MeetingEvent::QuestionDetected { .. } if detected.is_none() => {
                detected = Some(start.elapsed());
            }
            MeetingEvent::AnswerDelta { .. } if first.is_none() => first = Some(start.elapsed()),
            MeetingEvent::AnswerComplete { .. } => {
                let (Some(d), Some(f)) = (detected, first) else { break None };
                let done = start.elapsed();
                break Some((
                    d.as_secs_f64() * 1000.0,
                    (f - d).as_secs_f64() * 1000.0,
                    (done - f).as_secs_f64() * 1000.0,
                ));
            }
            _ => {}
        }
    };
    engine.emergency_stop();
    result
}
