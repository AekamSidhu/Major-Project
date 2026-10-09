//! Streaming transcription built from voice-activity detection plus an
//! utterance-level [`Transcriber`] (spec §15).
//!
//! `VadTranscriptionProvider` turns any batch recognizer (OpenAI
//! `/audio/transcriptions`, the local faster-whisper sidecar, ...) into a
//! streaming [`TranscriptionProvider`]:
//!
//! * audio is segmented into utterances by `openmeet_intel::vad::Segmenter`,
//! * finished utterances are transcribed concurrently (bounded),
//! * final transcripts are emitted **in utterance order** even when requests
//!   complete out of order,
//! * optional interim transcripts are produced for long utterances and never
//!   emitted after the final of the same utterance,
//! * known recognizer hallucinations on near-silent audio are filtered.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openmeet_core::audio::{AudioClip, AudioFrame};
use openmeet_core::error::ProviderError;
use openmeet_core::providers::{
    Transcriber, TranscriptionEvent, TranscriptionProvider, TranscriptionRequest,
    TranscriptionSession, TranscriptionSessionConfig,
};
use openmeet_core::types::{AudioSourceKind, ProviderInfo, TranscriptDraft};
use openmeet_intel::vad::{Segmenter, SegmenterEvent, VadConfig};
use parking_lot::Mutex;
use tokio::runtime::Handle;
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

/// Maximum concurrent final transcriptions per session.
const MAX_CONCURRENT_FINALS: usize = 2;
/// Per-utterance transcription timeout.
const TRANSCRIBE_TIMEOUT: Duration = Duration::from_secs(30);
/// Minimum interval between level events.
const LEVEL_INTERVAL: Duration = Duration::from_millis(125);
/// Characters of previous transcript passed as a biasing prompt.
const PROMPT_TAIL_CHARS: usize = 220;

/// Streaming adapter over a batch [`Transcriber`].
pub struct VadTranscriptionProvider {
    transcriber: Arc<dyn Transcriber>,
}

impl VadTranscriptionProvider {
    pub fn new(transcriber: Arc<dyn Transcriber>) -> Self {
        VadTranscriptionProvider { transcriber }
    }

    pub fn transcriber(&self) -> Arc<dyn Transcriber> {
        self.transcriber.clone()
    }
}

#[async_trait]
impl TranscriptionProvider for VadTranscriptionProvider {
    fn info(&self) -> ProviderInfo {
        self.transcriber.info()
    }

    fn start_session(
        &self,
        config: TranscriptionSessionConfig,
        events: mpsc::UnboundedSender<TranscriptionEvent>,
    ) -> Result<Box<dyn TranscriptionSession>, ProviderError> {
        let handle = Handle::try_current()
            .map_err(|_| ProviderError::Other("transcription requires an async runtime".into()))?;
        Ok(Box::new(VadSession::new(
            self.transcriber.clone(),
            config,
            events,
            handle,
        )))
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        self.transcriber.health_check().await
    }
}

/// State shared between the session and its transcription tasks.
struct Shared {
    source: AudioSourceKind,
    events: mpsc::UnboundedSender<TranscriptionEvent>,
    order: Mutex<OrderState>,
    /// Utterances whose final result has been emitted (late interims dropped).
    finalized: Mutex<HashSet<String>>,
    /// Tail of the last final transcript, used to bias the next request.
    last_text: Mutex<String>,
}

#[derive(Default)]
struct OrderState {
    next_seq: u64,
    ready: BTreeMap<u64, Option<TranscriptionEvent>>,
}

impl Shared {
    /// Stores the result for `seq` and emits every consecutive ready result.
    fn complete(&self, seq: u64, utterance_id: &str, event: Option<TranscriptionEvent>) {
        let mut order = self.order.lock();
        order.ready.insert(seq, event);
        while let Some(ev) = {
            let next = order.next_seq;
            order.ready.remove(&next)
        } {
            order.next_seq += 1;
            if let Some(ev) = ev {
                if let TranscriptionEvent::Draft(d) = &ev {
                    let mut last = self.last_text.lock();
                    *last = tail_chars(&d.text, PROMPT_TAIL_CHARS);
                }
                let _ = self.events.send(ev);
            }
        }
        drop(order);
        let mut fin = self.finalized.lock();
        fin.insert(utterance_id.to_string());
        // Bound memory in very long meetings.
        if fin.len() > 4096 {
            fin.clear();
        }
    }
}

fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        s.to_string()
    } else {
        s.chars().skip(count - n).collect()
    }
}

struct VadSession {
    segmenter: Segmenter,
    transcriber: Arc<dyn Transcriber>,
    shared: Arc<Shared>,
    cancel: CancellationToken,
    finals: Arc<Semaphore>,
    config: TranscriptionSessionConfig,
    next_seq: u64,
    interim_inflight: Arc<AtomicBool>,
    last_level_emit: Option<Instant>,
    handle: Handle,
    stopped: bool,
}

impl VadSession {
    fn new(
        transcriber: Arc<dyn Transcriber>,
        config: TranscriptionSessionConfig,
        events: mpsc::UnboundedSender<TranscriptionEvent>,
        handle: Handle,
    ) -> Self {
        let vad = VadConfig::from_settings(
            config.vad_sensitivity,
            config.silence_ms,
            config.interim_results,
        );
        VadSession {
            segmenter: Segmenter::new(config.source, vad),
            transcriber,
            shared: Arc::new(Shared {
                source: config.source,
                events,
                order: Mutex::new(OrderState::default()),
                finalized: Mutex::new(HashSet::new()),
                last_text: Mutex::new(String::new()),
            }),
            cancel: CancellationToken::new(),
            finals: Arc::new(Semaphore::new(MAX_CONCURRENT_FINALS)),
            config,
            next_seq: 0,
            interim_inflight: Arc::new(AtomicBool::new(false)),
            last_level_emit: None,
            handle,
            stopped: false,
        }
    }

    fn send(&self, ev: TranscriptionEvent) {
        let _ = self.shared.events.send(ev);
    }

    fn handle_events(&mut self, events: Vec<SegmenterEvent>) {
        for ev in events {
            match ev {
                SegmenterEvent::SpeechStarted { at_ms, .. } => {
                    self.send(TranscriptionEvent::SpeechStarted {
                        source: self.config.source,
                        at_ms,
                    });
                }
                SegmenterEvent::Interim(clip) => self.transcribe_interim(clip),
                SegmenterEvent::Utterance(clip) => {
                    self.send(TranscriptionEvent::SpeechEnded {
                        source: self.config.source,
                        at_ms: clip.end_ms,
                        utterance_id: clip.utterance_id.clone(),
                        discarded: false,
                    });
                    self.transcribe_final(clip);
                }
                SegmenterEvent::Discarded { utterance_id } => {
                    self.send(TranscriptionEvent::SpeechEnded {
                        source: self.config.source,
                        at_ms: openmeet_core::ids::now_ms(),
                        utterance_id,
                        discarded: true,
                    });
                }
            }
        }
    }

    fn prompt(&self) -> Option<String> {
        let vocab = self.config.vocabulary.clone().unwrap_or_default();
        let tail = self.shared.last_text.lock().clone();
        let p = format!("{} {}", vocab.trim(), tail.trim())
            .trim()
            .to_string();
        if p.is_empty() { None } else { Some(p) }
    }

    fn transcribe_final(&mut self, clip: AudioClip) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let shared = self.shared.clone();
        let transcriber = self.transcriber.clone();
        let cancel = self.cancel.child_token();
        let finals = self.finals.clone();
        let request = TranscriptionRequest {
            clip,
            language: self.config.language.clone(),
            prompt: self.prompt(),
        };
        self.handle.spawn(async move {
            let utterance_id = request.clip.utterance_id.clone();
            let source = request.clip.source;
            let permit = tokio::select! {
                _ = cancel.cancelled() => None,
                p = finals.acquire_owned() => p.ok(),
            };
            let Some(_permit) = permit else {
                shared.complete(seq, &utterance_id, None);
                return;
            };
            let clip = request.clip.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => Err(ProviderError::Cancelled),
                r = tokio::time::timeout(TRANSCRIBE_TIMEOUT, transcriber.transcribe(request, cancel.clone())) => {
                    r.unwrap_or(Err(ProviderError::Timeout))
                }
            };
            let event = match result {
                Ok(out) => {
                    let text = clean_transcript(&out.text);
                    if text.is_empty() || is_probable_hallucination(&text, &clip) {
                        Some(TranscriptionEvent::Empty {
                            source,
                            utterance_id: utterance_id.clone(),
                        })
                    } else {
                        Some(TranscriptionEvent::Draft(TranscriptDraft {
                            utterance_id: utterance_id.clone(),
                            source,
                            start_ms: clip.start_ms,
                            end_ms: clip.end_ms,
                            text,
                            is_final: true,
                            confidence: out.confidence,
                            speaker_role: None,
                        }))
                    }
                }
                // Cancelled sessions report nothing further.
                Err(ProviderError::Cancelled) => None,
                Err(error) => {
                    tracing::warn!(error_kind = ?std::mem::discriminant(&error), "utterance transcription failed");
                    Some(TranscriptionEvent::Error {
                        source,
                        utterance_id: Some(utterance_id.clone()),
                        error,
                    })
                }
            };
            shared.complete(seq, &utterance_id, event);
        });
    }

    fn transcribe_interim(&mut self, clip: AudioClip) {
        // One interim at a time, and never compete with finals for capacity.
        if self.finals.available_permits() == 0
            || self
                .interim_inflight
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return;
        }
        let shared = self.shared.clone();
        let transcriber = self.transcriber.clone();
        let cancel = self.cancel.child_token();
        let inflight = self.interim_inflight.clone();
        let request = TranscriptionRequest {
            clip,
            language: self.config.language.clone(),
            prompt: self.prompt(),
        };
        self.handle.spawn(async move {
            let utterance_id = request.clip.utterance_id.clone();
            let clip = request.clip.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => Err(ProviderError::Cancelled),
                r = tokio::time::timeout(TRANSCRIBE_TIMEOUT, transcriber.transcribe(request, cancel.clone())) => {
                    r.unwrap_or(Err(ProviderError::Timeout))
                }
            };
            inflight.store(false, Ordering::Release);
            if let Ok(out) = result {
                let text = clean_transcript(&out.text);
                if text.is_empty() || is_probable_hallucination(&text, &clip) {
                    return;
                }
                // Hold the finalized lock while sending so a final cannot slip
                // in between the check and the send.
                let finalized = shared.finalized.lock();
                if finalized.contains(&utterance_id) {
                    return;
                }
                let _ = shared.events.send(TranscriptionEvent::Draft(TranscriptDraft {
                    utterance_id,
                    source: shared.source,
                    start_ms: clip.start_ms,
                    end_ms: clip.end_ms,
                    text,
                    is_final: false,
                    confidence: out.confidence,
                    speaker_role: None,
                }));
            }
        });
    }
}

impl TranscriptionSession for VadSession {
    fn push_audio(&mut self, frame: AudioFrame) {
        if self.stopped {
            return;
        }
        let events = self.segmenter.push(&frame);
        self.handle_events(events);
        let now = Instant::now();
        if self
            .last_level_emit
            .is_none_or(|t| now.duration_since(t) >= LEVEL_INTERVAL)
        {
            self.last_level_emit = Some(now);
            self.send(TranscriptionEvent::Level {
                source: self.config.source,
                level: self.segmenter.level(),
            });
        }
    }

    fn flush(&mut self) {
        if self.stopped {
            return;
        }
        let events = self.segmenter.flush();
        self.handle_events(events);
    }

    fn stop(&mut self) {
        self.stopped = true;
        self.cancel.cancel();
    }
}

impl Drop for VadSession {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Removes recognizer annotations ("[BLANK_AUDIO]", "(music)", "♪") and
/// normalizes whitespace.
pub fn clean_transcript(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth_square = 0usize;
    let mut depth_round = 0usize;
    for c in text.chars() {
        match c {
            '[' => depth_square += 1,
            ']' => depth_square = depth_square.saturating_sub(1),
            '♪' | '♫' => {}
            _ if depth_square > 0 => {}
            _ => out.push(c),
        }
    }
    // Drop parenthesized sound annotations like "(music)" / "(silence)".
    let mut cleaned = String::with_capacity(out.len());
    let mut buf = String::new();
    for c in out.chars() {
        if c == '(' {
            depth_round += 1;
            buf.clear();
            continue;
        }
        if depth_round > 0 {
            if c == ')' {
                depth_round -= 1;
                let lower = buf.trim().to_ascii_lowercase();
                let annotation = [
                    "music",
                    "silence",
                    "applause",
                    "laughter",
                    "laughs",
                    "inaudible",
                    "noise",
                    "blank_audio",
                    "background noise",
                    "coughs",
                    "sighs",
                ]
                .iter()
                .any(|a| lower == *a);
                if !annotation {
                    cleaned.push('(');
                    cleaned.push_str(&buf);
                    cleaned.push(')');
                }
                buf.clear();
            } else {
                buf.push(c);
            }
            continue;
        }
        cleaned.push(c);
    }
    if depth_round > 0 {
        cleaned.push('(');
        cleaned.push_str(&buf);
    }
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Whisper-family models produce stock phrases on near-silent audio. Only
/// filter them when the audio really was quiet, so genuine "Thank you." from a
/// participant survives.
pub fn is_probable_hallucination(text: &str, clip: &AudioClip) -> bool {
    let norm: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if norm.is_empty() {
        return true;
    }
    const ALWAYS: [&str; 6] = [
        "subtitles by the amaraorg community",
        "thanks for watching",
        "thank you for watching",
        "please subscribe",
        "like and subscribe",
        "transcribed by",
    ];
    if ALWAYS.iter().any(|p| norm.starts_with(p)) && clip.duration_ms() < 4000 {
        return true;
    }
    const QUIET_ONLY: [&str; 8] = [
        "thank you",
        "thanks",
        "you",
        "bye",
        "so",
        "okay",
        "uh",
        "um",
    ];
    let quiet = rms(&clip.samples) < 0.008;
    quiet && QUIET_ONLY.contains(&norm.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    use openmeet_core::audio::PIPELINE_SAMPLE_RATE;
    use openmeet_core::providers::TranscriptionOutput;

    /// Returns "utterance N" with a per-call delay schedule so completions can
    /// arrive out of order.
    struct ScriptedTranscriber {
        calls: AtomicUsize,
        delays_ms: Vec<u64>,
        fail_on: Option<usize>,
    }

    #[async_trait]
    impl Transcriber for ScriptedTranscriber {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                id: "scripted".into(),
                display_name: "Scripted".into(),
                model: "test".into(),
                location: openmeet_core::types::ProviderLocation::Mock,
                endpoint: None,
            }
        }

        async fn transcribe(
            &self,
            request: TranscriptionRequest,
            cancel: CancellationToken,
        ) -> Result<openmeet_core::providers::TranscriptionOutput, ProviderError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let delay = self.delays_ms.get(n).copied().unwrap_or(0);
            tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
            }
            if self.fail_on == Some(n) {
                return Err(ProviderError::Network("boom".into()));
            }
            let final_marker = if request.clip.is_final {
                ""
            } else {
                " (interim)"
            };
            Ok(TranscriptionOutput {
                text: format!("utterance {n}{final_marker}"),
                confidence: Some(0.9),
                language: None,
            })
        }

        async fn health_check(&self) -> Result<(), ProviderError> {
            Ok(())
        }
    }

    fn tone(ms: u32, amp: f32) -> Vec<f32> {
        let n = (PIPELINE_SAMPLE_RATE * ms / 1000) as usize;
        (0..n)
            .map(|i| {
                (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / PIPELINE_SAMPLE_RATE as f32).sin()
                    * amp
            })
            .collect()
    }

    fn silence(ms: u32) -> Vec<f32> {
        let n = (PIPELINE_SAMPLE_RATE * ms / 1000) as usize;
        (0..n)
            .map(|i| if i % 2 == 0 { 0.0005 } else { -0.0005 })
            .collect()
    }

    fn feed(session: &mut Box<dyn TranscriptionSession>, samples: &[f32], t0: &mut i64) {
        for chunk in samples.chunks(320) {
            session.push_audio(AudioFrame {
                source: AudioSourceKind::System,
                samples: chunk.to_vec(),
                captured_at_ms: *t0,
            });
            *t0 += (chunk.len() as i64 * 1000) / PIPELINE_SAMPLE_RATE as i64;
        }
    }

    fn config() -> TranscriptionSessionConfig {
        TranscriptionSessionConfig {
            source: AudioSourceKind::System,
            language: Some("en".into()),
            vocabulary: None,
            interim_results: false,
            vad_sensitivity: 0.6,
            silence_ms: 500,
        }
    }

    async fn collect_finals(
        rx: &mut mpsc::UnboundedReceiver<TranscriptionEvent>,
        expected: usize,
    ) -> Vec<TranscriptionEvent> {
        let mut out = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while out.len() < expected {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(ev)) => match ev {
                    TranscriptionEvent::Draft(ref d) if d.is_final => out.push(ev),
                    TranscriptionEvent::Error { .. } | TranscriptionEvent::Empty { .. } => {
                        out.push(ev)
                    }
                    _ => {}
                },
                _ => break,
            }
        }
        out
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finals_are_emitted_in_utterance_order() {
        // First utterance is slow, second fast: output must still be ordered.
        let provider = VadTranscriptionProvider::new(Arc::new(ScriptedTranscriber {
            calls: AtomicUsize::new(0),
            delays_ms: vec![400, 10],
            fail_on: None,
        }));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut session = provider.start_session(config(), tx).unwrap();
        let mut t = 1_000_000;
        feed(&mut session, &silence(500), &mut t);
        feed(&mut session, &tone(1200, 0.3), &mut t);
        feed(&mut session, &silence(900), &mut t);
        feed(&mut session, &tone(1200, 0.3), &mut t);
        feed(&mut session, &silence(900), &mut t);
        let finals = collect_finals(&mut rx, 2).await;
        let texts: Vec<String> = finals
            .iter()
            .filter_map(|e| match e {
                TranscriptionEvent::Draft(d) => Some(d.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["utterance 0", "utterance 1"]);
        if let TranscriptionEvent::Draft(d) = &finals[0] {
            assert!(d.start_ms >= 1_000_000 && d.end_ms > d.start_ms);
            assert_eq!(d.source, AudioSourceKind::System);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failures_do_not_block_later_utterances() {
        let provider = VadTranscriptionProvider::new(Arc::new(ScriptedTranscriber {
            calls: AtomicUsize::new(0),
            delays_ms: vec![50, 10],
            fail_on: Some(0),
        }));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut session = provider.start_session(config(), tx).unwrap();
        let mut t = 0;
        feed(&mut session, &silence(400), &mut t);
        feed(&mut session, &tone(1000, 0.3), &mut t);
        feed(&mut session, &silence(900), &mut t);
        feed(&mut session, &tone(1000, 0.3), &mut t);
        feed(&mut session, &silence(900), &mut t);
        let events = collect_finals(&mut rx, 2).await;
        assert!(matches!(events[0], TranscriptionEvent::Error { .. }));
        assert!(matches!(&events[1], TranscriptionEvent::Draft(d) if d.text == "utterance 1"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_finalizes_current_utterance_and_stop_cancels() {
        let provider = VadTranscriptionProvider::new(Arc::new(ScriptedTranscriber {
            calls: AtomicUsize::new(0),
            delays_ms: vec![0, 5_000],
            fail_on: None,
        }));
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut session = provider.start_session(config(), tx).unwrap();
        let mut t = 0;
        feed(&mut session, &silence(400), &mut t);
        feed(&mut session, &tone(1000, 0.3), &mut t);
        session.flush();
        let events = collect_finals(&mut rx, 1).await;
        assert!(matches!(&events[0], TranscriptionEvent::Draft(d) if d.text == "utterance 0"));

        // Second utterance takes 5 s; stop must cancel it and emit nothing.
        feed(&mut session, &tone(1000, 0.3), &mut t);
        session.flush();
        session.stop();
        let events = collect_finals(&mut rx, 1).await;
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn cleans_annotations() {
        assert_eq!(clean_transcript(" [BLANK_AUDIO] "), "");
        assert_eq!(clean_transcript("Hello (music) world ♪"), "Hello world");
        assert_eq!(
            clean_transcript("We used (mostly) Postgres"),
            "We used (mostly) Postgres"
        );
        assert_eq!(clean_transcript("a  b\n c"), "a b c");
    }

    #[test]
    fn hallucination_filter_only_on_quiet_audio() {
        let quiet = AudioClip {
            utterance_id: "u".into(),
            source: AudioSourceKind::System,
            samples: vec![0.001; 16_000],
            sample_rate: 16_000,
            start_ms: 0,
            end_ms: 1000,
            is_final: true,
        };
        let mut loud = quiet.clone();
        loud.samples = tone(1000, 0.3);
        assert!(is_probable_hallucination("Thank you.", &quiet));
        assert!(!is_probable_hallucination("Thank you.", &loud));
        assert!(is_probable_hallucination("Thanks for watching!", &loud));
        assert!(!is_probable_hallucination("How did you scale it?", &quiet));
    }
}
