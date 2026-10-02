//! Screen context processing (spec §26–§30).
//!
//! The platform layer delivers change-filtered, privacy-filtered frames. Here
//! each frame goes through local OCR (when available) and, rate-limited, a
//! vision model; the result becomes a structured [`ScreenObservation`]. Raw
//! images never leave this function and are never stored.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use openmeet_core::CoreError;
use openmeet_core::events::{MeetingEvent, NoticeLevel};
use openmeet_core::ids::{new_id, now_ms};
use openmeet_core::providers::{VisionContext, VisionRequest};
use openmeet_core::types::*;
use openmeet_intel::text::truncate_to_tokens;
use openmeet_intel::vision::describe_ocr_text;
use tokio::runtime::Handle;
use tokio::sync::watch;

use crate::embedder::EmbedJob;
use crate::engine::{Engine, Inner};
use crate::memory::{MemoryItem, MemoryKind};
use crate::metrics::record_latency;
use crate::persist::PersistOp;

const MAX_OBSERVATIONS: usize = 300;
const MAX_VISIBLE_TEXT_CHARS: usize = 4000;
const VISION_TIMEOUT: Duration = Duration::from_secs(45);

pub(crate) fn spawn_screen_worker(
    weak: Weak<Inner>,
    mut rx: watch::Receiver<Option<ScreenFrame>>,
    handle: &Handle,
) {
    handle.spawn(async move {
        while rx.changed().await.is_ok() {
            let frame = rx.borrow_and_update().clone();
            let Some(frame) = frame else { continue };
            let Some(inner) = weak.upgrade() else { break };
            Engine { inner }.process_screen_frame(frame).await;
        }
    });
}

fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

impl Engine {
    pub(crate) async fn process_screen_frame(&self, frame: ScreenFrame) {
        let started = Instant::now();
        let settings = self.settings();
        let (recent_transcript, question, allow_vision) = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st
                .runtime
                .as_mut()
                .filter(|r| r.state() == MeetingState::Active)
            else {
                return;
            };
            let since = rt.now_rel() - 60_000;
            let recent = rt
                .lines_since(since)
                .iter()
                .map(|l| format!("{}: {}", l.speaker, l.text))
                .collect::<Vec<_>>()
                .join("\n");
            let question = rt.latest_open_question(120_000).map(|q| q.question.clone());
            // Rate-limit vision-model calls (cost control); manual captures bypass it.
            let now = now_ms();
            while rt.vision_calls.front().is_some_and(|t| now - t > 60_000) {
                rt.vision_calls.pop_front();
            }
            let allow = frame.manual
                || rt.vision_calls.len() < settings.screen.max_vision_per_minute as usize;
            (truncate_to_tokens(&recent, 300), question, allow)
        };
        let providers = self.inner.providers.read().clone();
        let png = Arc::new(frame.png.clone());
        let ctx = VisionContext {
            source_label: frame.source_label.clone(),
            application: frame.application.clone(),
            window_title: frame.window_title.clone(),
            recent_transcript,
            question,
        };

        // 1. Local OCR.
        let ocr_text = match providers.ocr.clone() {
            Some(ocr) => {
                let bytes = png.clone();
                match tokio::task::spawn_blocking(move || ocr.recognize_png(&bytes)).await {
                    Ok(Ok(out)) => Some(out.text).filter(|t| !t.trim().is_empty()),
                    Ok(Err(e)) => {
                        self.inner.health.set(
                            Component::Vision,
                            ComponentStatus::Degraded,
                            Some(format!("OCR failed: {}", e.user_message())),
                        );
                        None
                    }
                    Err(_) => None,
                }
            }
            None => None,
        };

        // 2. Vision model (optional, rate limited, cancellable).
        let mut result = None;
        if let (Some(vision), true) = (providers.vision.clone(), allow_vision) {
            {
                let mut st = self.inner.state.lock();
                if let Some(rt) = st.runtime.as_mut() {
                    rt.vision_calls.push_back(now_ms());
                }
            }
            let ticket = self.inner.requests.begin(
                RequestSlot::Background,
                RequestKind::ScreenAnalysis,
                |_| None,
            );
            let request = VisionRequest {
                image_png: png.clone(),
                width: frame.width,
                height: frame.height,
                ocr_text: ocr_text.clone(),
                context: ctx.clone(),
            };
            let outcome = tokio::select! {
                _ = ticket.cancel.cancelled() => None,
                r = tokio::time::timeout(VISION_TIMEOUT, vision.analyze(request, ticket.cancel.clone())) => Some(r),
            };
            match outcome {
                None => {
                    self.inner
                        .requests
                        .set_state(&ticket, AnswerState::Cancelled, None);
                    return;
                }
                Some(Ok(Ok(r))) => {
                    self.inner
                        .requests
                        .set_state(&ticket, AnswerState::Complete, None);
                    self.inner
                        .health
                        .set(Component::Vision, ComponentStatus::Active, None);
                    result = Some(r);
                }
                Some(Ok(Err(e))) => {
                    self.inner
                        .requests
                        .set_state(&ticket, AnswerState::Failed, None);
                    self.inner.health.set(
                        Component::Vision,
                        ComponentStatus::Degraded,
                        Some(e.user_message()),
                    );
                    if frame.manual {
                        self.notice(
                            NoticeLevel::Warning,
                            format!("Screen analysis failed: {}", e.user_message()),
                            Some(Component::Vision),
                        );
                    }
                }
                Some(Err(_)) => {
                    self.inner
                        .requests
                        .set_state(&ticket, AnswerState::Failed, None);
                    self.inner.health.set(
                        Component::Vision,
                        ComponentStatus::Degraded,
                        Some("Screen analysis timed out".into()),
                    );
                }
            }
        }
        // 3. Fall back to an OCR-derived description.
        let result = result.or_else(|| ocr_text.as_ref().map(|t| describe_ocr_text(t, &ctx)));
        let Some(r) = result else {
            if providers.vision.is_none() && providers.ocr.is_none() {
                self.inner.health.set(
                    Component::Vision,
                    ComponentStatus::Unsupported,
                    Some("No OCR or vision provider is available on this system".into()),
                );
                if frame.manual {
                    self.notice(
                        NoticeLevel::Warning,
                        "Screen captured, but no OCR or vision provider is configured.",
                        Some(Component::Vision),
                    );
                }
            }
            return;
        };
        let visible = r
            .visible_text
            .clone()
            .or(ocr_text)
            .map(|t| truncate_chars(t.trim(), MAX_VISIBLE_TEXT_CHARS))
            .filter(|t| !t.is_empty());
        let observation = ScreenObservation {
            id: new_id(),
            meeting_id: None,
            timestamp: frame.captured_at,
            source: frame.source_label.clone(),
            application: r.application.clone().or(frame.application.clone()),
            title: frame.window_title.clone(),
            description: r.description.trim().to_string(),
            visible_text: visible,
            entities: r.entities.clone(),
            content_type: r.content_type.clone(),
            frame_hash: Some(format!("{:016x}", frame.hash)),
        };
        if self.add_observation(observation).is_ok() {
            record_latency(
                &self.inner.latency,
                self.sink(),
                LatencyStage::ScreenProcessing,
                started.elapsed().as_secs_f64() * 1000.0,
                None,
            );
        }
    }

    /// Adds an observation to the meeting context (also used by simulation
    /// and demo mode).
    pub(crate) fn add_observation(&self, mut observation: ScreenObservation) -> Result<(), CoreError> {
        let (store, job) = {
            let mut st = self.inner.state.lock();
            let rt = st
                .runtime
                .as_mut()
                .filter(|r| r.state().is_live())
                .ok_or_else(|| CoreError::InvalidState("Start a meeting first.".into()))?;
            observation.meeting_id = Some(rt.meeting.id.clone());
            let at = rt.rel_ms(observation.timestamp);
            let mut text = String::from("Screen");
            if let Some(app) = &observation.application {
                text.push_str(&format!(" ({app})"));
            }
            text.push_str(&format!(": {}", observation.description));
            if let Some(v) = &observation.visible_text {
                text.push_str(&format!(" Visible text: {}", truncate_chars(v, 400)));
            }
            rt.memory.add(MemoryItem {
                id: observation.id.clone(),
                kind: MemoryKind::Observation,
                text: text.clone(),
                at_ms: at,
            });
            rt.observations.push(observation.clone());
            if rt.observations.len() > MAX_OBSERVATIONS {
                rt.observations.remove(0);
            }
            (
                rt.store.clone(),
                EmbedJob {
                    meeting_id: rt.meeting.id.clone(),
                    item_id: observation.id.clone(),
                    kind: MemoryKind::Observation,
                    text,
                },
            )
        };
        self.persist(&store, PersistOp::Observation(observation.clone()));
        let _ = self.inner.embed_tx.send(job);
        self.inner
            .health
            .set(Component::Screen, ComponentStatus::Active, None);
        self.emit(MeetingEvent::ScreenUpdated { observation });
        Ok(())
    }
}
