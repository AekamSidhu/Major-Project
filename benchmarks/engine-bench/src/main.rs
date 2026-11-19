//! Headless OpenMeet engine benchmarks, through the engine's public API only.
//!
//! ```text
//! engine-bench offline --runs 20 --out offline.json
//! engine-bench live --speech DIR --local-ai http://127.0.0.1:8795 --out live.json
//! ```
//!
//! `offline` runs `Engine::run_benchmark` (the Diagnostics benchmark: question
//! detection, VAD, hashing embeddings, retrieval, prompt assembly and the
//! pipeline with a mock LLM) `--runs` times and writes every value.
//!
//! `live` runs the real pipeline as `crates/openmeet-engine/tests/live.rs`
//! does (VAD -> sidecar Whisper -> question detection -> Ollama), with the
//! audio pushed in real time as 20 ms frames: every `*_question.wav` in DIR in
//! candidate mode (speech end -> transcript, -> question detected, -> first
//! answer token, -> answer complete), then every `*_answer.wav` in
//! interviewer mode (speech end -> follow-ups), then the summary. It writes
//! timings and counts only, never text.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openmeet_core::audio::{AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::now_ms;
use openmeet_core::ipc::StartMeetingRequest;
use openmeet_core::providers::{ProviderSet, StaticSecrets};
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::MemoryStore;
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps, VadTranscriptionProvider};
use openmeet_providers::factory::{self, BuildContext};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};

#[derive(Default)]
struct Sink {
    events: Mutex<Vec<(Instant, MeetingEvent)>>,
    notify: Notify,
}

impl EventSink for Sink {
    fn emit(&self, event: MeetingEvent) {
        self.events.lock().push((Instant::now(), event));
        self.notify.notify_waiters();
    }
}

impl Sink {
    /// The first event after `since` that matches, waiting up to `timeout`.
    async fn wait_for(
        &self,
        since: Instant,
        timeout: Duration,
        pred: impl Fn(&MeetingEvent) -> bool,
    ) -> Option<(Instant, MeetingEvent)> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.notify.notified();
            if let Some(found) = self
                .events
                .lock()
                .iter()
                .find(|(t, e)| *t >= since && pred(e))
                .cloned()
            {
                return Some(found);
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return None;
            }
        }
    }

    fn count(&self, since: Instant, pred: impl Fn(&MeetingEvent) -> bool) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|(t, e)| *t >= since && pred(e))
            .count()
    }
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").expect("--out FILE"));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let result = match args.get(1).map(String::as_str) {
        Some("offline") => {
            let runs = arg(&args, "--runs").map_or(20, |r| r.parse().expect("--runs N"));
            rt.block_on(offline(runs))
        }
        Some("live") => {
            let speech = PathBuf::from(arg(&args, "--speech").expect("--speech DIR"));
            let local_ai =
                arg(&args, "--local-ai").unwrap_or_else(|| "http://127.0.0.1:8795".into());
            let ollama = arg(&args, "--ollama").unwrap_or_else(|| "http://127.0.0.1:11434".into());
            let model = arg(&args, "--whisper-model").unwrap_or_else(|| "base.en".into());
            rt.block_on(live(&speech, &local_ai, &ollama, &model))
        }
        _ => panic!("usage: engine-bench offline|live ... --out FILE"),
    };
    std::fs::write(&out, serde_json::to_string_pretty(&result).unwrap() + "\n").expect("write");
    println!("-> {}", out.display());
}

// --- offline -----------------------------------------------------------------

async fn offline(runs: usize) -> Value {
    let engine = Engine::new(
        EngineDeps {
            sink: Arc::new(Sink::default()),
            store: Arc::new(MemoryStore::new()),
            providers: ProviderSet::default(),
            settings: Settings::default(),
        },
        tokio::runtime::Handle::current(),
    );
    let mut items: Vec<(String, String, Option<String>, Vec<f64>)> = Vec::new();
    for run in 0..runs {
        for it in engine.run_benchmark().await {
            match items.iter_mut().find(|(n, ..)| *n == it.name) {
                Some((.., values)) => values.push(it.value),
                None => items.push((it.name, it.unit, it.note, vec![it.value])),
            }
        }
        eprintln!("offline run {}/{runs}", run + 1);
    }
    json!({
        "runs": runs,
        "items": items.into_iter().map(|(name, unit, note, values)| json!({
            "name": name, "unit": unit, "note": note, "values": values,
        })).collect::<Vec<_>>(),
    })
}

// --- live --------------------------------------------------------------------

/// Reads a 16 kHz mono 16-bit PCM WAV file.
fn read_wav(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    assert!(bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &bytes[i + 8..(i + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes([body[14], body[15]]);
            assert_eq!((channels, rate, bits), (1, PIPELINE_SAMPLE_RATE, 16), "need 16 kHz mono 16-bit");
        } else if id == b"data" {
            return body
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("{}: no data chunk", path.display());
}

fn wavs(dir: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("speech dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(suffix)))
        .collect();
    out.sort();
    out
}

/// A capture thread stand-in: pushes 20 ms frames in real time, speech from
/// the queue when there is some, otherwise faint noise (a quiet room).
fn start_capture(engine: Engine) -> mpsc::UnboundedSender<(Vec<f32>, tokio::sync::oneshot::Sender<Instant>)> {
    let (tx, mut rx) = mpsc::unbounded_channel::<(Vec<f32>, tokio::sync::oneshot::Sender<Instant>)>();
    tokio::spawn(async move {
        let frame = (PIPELINE_SAMPLE_RATE / 50) as usize;
        let quiet: Vec<f32> = (0..frame).map(|i| if i % 2 == 0 { 0.0004 } else { -0.0004 }).collect();
        let mut tick = tokio::time::interval(Duration::from_millis(20));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
        let mut current: Option<(Vec<f32>, usize, tokio::sync::oneshot::Sender<Instant>)> = None;
        loop {
            tick.tick().await;
            if current.is_none() {
                match rx.try_recv() {
                    Ok((samples, done)) => current = Some((samples, 0, done)),
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                    Err(mpsc::error::TryRecvError::Empty) => {}
                }
            }
            let samples = match current.as_mut() {
                Some((speech, pos, _)) => {
                    let end = (*pos + frame).min(speech.len());
                    let chunk = speech[*pos..end].to_vec();
                    *pos = end;
                    chunk
                }
                None => quiet.clone(),
            };
            engine.push_audio(AudioFrame {
                source: AudioSourceKind::System,
                samples,
                captured_at_ms: now_ms(),
            });
            if current.as_ref().is_some_and(|(s, pos, _)| *pos >= s.len()) {
                let (_, _, done) = current.take().unwrap();
                let _ = done.send(Instant::now());
            }
        }
    });
    tx
}

async fn speak(
    capture: &mpsc::UnboundedSender<(Vec<f32>, tokio::sync::oneshot::Sender<Instant>)>,
    samples: Vec<f32>,
) -> Instant {
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    capture.send((samples, done_tx)).expect("capture task");
    done_rx.await.expect("speech pushed")
}

const SLOW: Duration = Duration::from_secs(180);

async fn live(speech: &Path, local_ai: &str, ollama: &str, whisper_model: &str) -> Value {
    let mut settings = Settings::default();
    settings.transcription.base_url = Some(local_ai.to_string());
    settings.transcription.model = whisper_model.to_string();
    settings.ai.llm.base_url = Some(ollama.to_string());
    settings.privacy.retention = RetentionPolicy::None;
    let secrets = StaticSecrets::default();
    let ctx = BuildContext { settings: &settings, secrets: &secrets };
    let llm = factory::build_llm(&ctx).expect("Ollama provider");
    let transcriber = factory::build_transcriber(&ctx).expect("local AI transcriber");
    let sink = Arc::new(Sink::default());
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: Arc::new(MemoryStore::new()),
            providers: ProviderSet {
                llm: Some(llm),
                transcription: Some(Arc::new(VadTranscriptionProvider::new(transcriber))),
                ..Default::default()
            },
            settings: settings.clone(),
        },
        tokio::runtime::Handle::current(),
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
        .expect("start meeting");
    let capture = start_capture(engine.clone());
    tokio::time::sleep(Duration::from_secs(2)).await; // the VAD learns the noise floor

    // Candidate mode: spoken question -> suggested answer.
    let mut questions = Vec::new();
    for path in wavs(speech, "_question.wav") {
        let audio = read_wav(&path);
        let seconds = audio.len() as f64 / f64::from(PIPELINE_SAMPLE_RATE);
        let started = Instant::now();
        let end = speak(&capture, audio).await;
        let mut row = json!({"file": path.file_name().unwrap().to_string_lossy(), "speech_s": seconds});
        let transcript = sink
            .wait_for(started, SLOW, |e| matches!(e, MeetingEvent::TranscriptFinal { .. }))
            .await;
        let detected = sink
            .wait_for(started, Duration::from_secs(20), |e| matches!(e, MeetingEvent::QuestionDetected { .. }))
            .await;
        if let Some((t, _)) = &transcript {
            row["speech_end_to_transcript_ms"] = json!(ms(t.saturating_duration_since(end)));
        }
        if let Some((t, _)) = &detected {
            row["speech_end_to_question_ms"] = json!(ms(t.saturating_duration_since(end)));
            let first = sink
                .wait_for(*t, SLOW, |e| matches!(e, MeetingEvent::AnswerDelta { .. }))
                .await;
            let done = sink
                .wait_for(*t, SLOW, |e| matches!(e, MeetingEvent::AnswerComplete { .. }))
                .await;
            if let Some((f, _)) = &first {
                row["question_to_first_token_ms"] = json!(ms(f.saturating_duration_since(*t)));
                row["speech_end_to_first_token_ms"] = json!(ms(f.saturating_duration_since(end)));
            }
            if let (Some((f, _)), Some((c, MeetingEvent::AnswerComplete { suggestion, .. }))) = (&first, &done) {
                row["first_token_to_complete_ms"] = json!(ms(c.saturating_duration_since(*f)));
                row["speech_end_to_complete_ms"] = json!(ms(c.saturating_duration_since(end)));
                row["answer_chars"] = json!(suggestion.content.chars().count());
            }
        }
        row["question_detected"] = json!(detected.is_some());
        row["transcripts"] = json!(sink.count(started, |e| matches!(e, MeetingEvent::TranscriptFinal { .. })));
        eprintln!("{row}");
        questions.push(row);
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // Interviewer mode: spoken candidate answer -> follow-up questions.
    engine.set_mode(Mode::Interviewer);
    tokio::time::sleep(Duration::from_secs(1)).await;
    let mut answers = Vec::new();
    for path in wavs(speech, "_answer.wav").into_iter().filter(|p| !p.to_string_lossy().contains("_llm_")) {
        let audio = read_wav(&path);
        let seconds = audio.len() as f64 / f64::from(PIPELINE_SAMPLE_RATE);
        let started = Instant::now();
        let end = speak(&capture, audio).await;
        let mut row = json!({"file": path.file_name().unwrap().to_string_lossy(), "speech_s": seconds});
        let first = sink
            .wait_for(started, SLOW, |e| matches!(e, MeetingEvent::FollowupsReady { .. }))
            .await;
        let done = sink
            .wait_for(started, SLOW, |e| matches!(e, MeetingEvent::FollowupsReady { done: true, .. }))
            .await;
        if let Some((t, _)) = &first {
            row["speech_end_to_first_followup_ms"] = json!(ms(t.saturating_duration_since(end)));
        }
        if let Some((t, MeetingEvent::FollowupsReady { questions, .. })) = &done {
            row["speech_end_to_followups_done_ms"] = json!(ms(t.saturating_duration_since(end)));
            row["followups"] = json!(questions.len());
        }
        eprintln!("{row}");
        answers.push(row);
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    let stats: Vec<Value> = engine
        .latency_stats()
        .into_iter()
        .map(|s| json!({"stage": format!("{:?}", s.stage), "count": s.count, "p50_ms": s.p50_ms,
                        "p95_ms": s.p95_ms, "max_ms": s.max_ms}))
        .collect();

    let t = Instant::now();
    engine.stop_meeting().await.expect("stop");
    let summary = sink
        .wait_for(t, SLOW, |e| matches!(e, MeetingEvent::SummaryReady { .. } | MeetingEvent::SummaryFailed { .. }))
        .await;
    let summary_ms = summary.as_ref().map(|(s, _)| ms(s.saturating_duration_since(t)));
    let summary_ok = matches!(summary, Some((_, MeetingEvent::SummaryReady { .. })));
    drop(capture);
    json!({
        "whisper_model": whisper_model,
        "llm": {"provider": "ollama", "model": settings.ai.llm.model},
        "settings": {"turn_end_silence_ms": settings.assistant.turn_end_silence_ms,
                      "semantic_detection": settings.assistant.semantic_detection,
                      "max_output_tokens": settings.ai.llm.max_output_tokens},
        "questions": questions,
        "answers": answers,
        "engine_latency_stats": stats,
        "summary_ms": summary_ms,
        "summary_ok": summary_ok,
    })
}
