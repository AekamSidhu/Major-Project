//! Demo mode (spec §74) and developer simulation tools (spec §73).
//!
//! Demo meetings run the real engine pipeline (detection, context, request
//! orchestration, streaming) with deterministic local providers and scripted
//! dialogue, so the complete UX works without microphones, accounts or
//! network access. Demo meetings are never persisted.

use std::sync::Arc;
use std::time::Duration;

use openmeet_core::CoreError;
use openmeet_core::error::ProviderError;
use openmeet_core::events::NoticeLevel;
use openmeet_core::ids::{Id, new_id, now_ms};
use openmeet_core::ipc::{DemoScenario, SimulationAction, StartMeetingRequest};
use openmeet_core::providers::ProviderSet;
use openmeet_core::types::*;
use openmeet_intel::claims::extract_entities;
use openmeet_intel::demo::{self, DemoStep};
use openmeet_intel::mock::{MockLlmProvider, MockVisionProvider};
use tokio_util::sync::CancellationToken;

use crate::engine::Engine;

/// Delay between partial transcript updates while "speaking".
const WORD_GROUP_DELAY: Duration = Duration::from_millis(220);

impl Engine {
    pub fn demo_scenarios(&self) -> Vec<DemoScenario> {
        demo::scenarios()
    }

    pub fn demo_running(&self) -> bool {
        self.inner.sim.lock().demo_meeting.is_some()
    }

    /// Starts a scripted demo meeting with deterministic local providers.
    pub async fn start_demo(&self, scenario_id: &str) -> Result<Meeting, CoreError> {
        let scenario = demo::scenarios()
            .into_iter()
            .find(|s| s.id == scenario_id)
            .ok_or_else(|| {
                CoreError::NotFound(format!("Unknown demo scenario '{scenario_id}'."))
            })?;
        let script = demo::script(scenario_id).ok_or_else(|| {
            CoreError::NotFound(format!("Unknown demo scenario '{scenario_id}'."))
        })?;
        if self.is_live() {
            return Err(CoreError::InvalidState(
                "End the current meeting before starting a demo.".into(),
            ));
        }
        {
            let mut sim = self.inner.sim.lock();
            if sim.saved_providers.is_none() {
                sim.saved_providers = Some(self.inner.providers.read().clone());
            }
            *self.inner.providers.write() = ProviderSet {
                llm: Some(Arc::new(demo::demo_llm(scenario_id))),
                fast_llm: None,
                transcription: None,
                vision: Some(Arc::new(MockVisionProvider)),
                ocr: None,
                embeddings: None,
                issues: Vec::new(),
            };
        }
        let meeting = match self.start_meeting_inner(
            StartMeetingRequest {
                mode: scenario.mode,
                title: Some(format!("Demo: {}", scenario.title)),
                microphone: false,
                system_audio: true,
                screen: None,
                focus_app_id: None,
            },
            true,
        ) {
            Ok(m) => m,
            Err(e) => {
                self.restore_demo_providers();
                return Err(e);
            }
        };
        let cancel = CancellationToken::new();
        {
            let mut sim = self.inner.sim.lock();
            sim.demo_cancel = Some(cancel.clone());
            sim.demo_meeting = Some(meeting.id.clone());
        }
        // Transcription is simulated: report it as such rather than "missing".
        self.inner.health.set(
            Component::Transcription,
            ComponentStatus::Streaming,
            Some("Demo script (simulated speech)".into()),
        );
        let engine = self.clone();
        let meeting_id = meeting.id.clone();
        self.inner.handle.spawn(async move {
            engine.run_demo_script(meeting_id, script, cancel).await;
        });
        Ok(meeting)
    }

    async fn run_demo_script(
        &self,
        meeting_id: Id,
        script: Vec<DemoStep>,
        cancel: CancellationToken,
    ) {
        for step in script {
            let delay = match &step {
                DemoStep::Say { delay_ms, .. } | DemoStep::Screen { delay_ms, .. } => *delay_ms,
            };
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
            }
            if self.current_meeting_id().as_deref() != Some(meeting_id.as_str())
                || self.meeting_state() != MeetingState::Active
            {
                return;
            }
            match step {
                DemoStep::Say { speaker, text, .. } => {
                    self.simulate_speech(speaker, &text, Some(&cancel)).await;
                }
                DemoStep::Screen {
                    description,
                    visible_text,
                    content_type,
                    application,
                    ..
                } => {
                    let mut entity_source = description.clone();
                    if let Some(v) = &visible_text {
                        entity_source.push(' ');
                        entity_source.push_str(v);
                    }
                    let _ = self.add_observation(ScreenObservation {
                        id: new_id(),
                        meeting_id: None,
                        timestamp: now_ms(),
                        source: "Demo screen".into(),
                        application,
                        title: None,
                        description,
                        visible_text,
                        entities: extract_entities(&entity_source),
                        content_type,
                        frame_hash: None,
                    });
                }
            }
        }
    }

    /// Emits the words of `text` as partial transcripts, then the final one,
    /// exactly like a live transcription session would.
    async fn simulate_speech(
        &self,
        speaker: SpeakerRole,
        text: &str,
        cancel: Option<&CancellationToken>,
    ) {
        let utterance_id = new_id();
        let source = if speaker == SpeakerRole::User {
            AudioSourceKind::Microphone
        } else {
            AudioSourceKind::System
        };
        let start = now_ms();
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut partial = String::new();
        for (i, w) in words.iter().enumerate() {
            if cancel.is_some_and(|c| c.is_cancelled()) {
                return;
            }
            if !partial.is_empty() {
                partial.push(' ');
            }
            partial.push_str(w);
            if i % 3 == 2 && i + 1 < words.len() {
                let _ = self.inject_draft(TranscriptDraft {
                    utterance_id: utterance_id.clone(),
                    source,
                    start_ms: start,
                    end_ms: now_ms(),
                    text: partial.clone(),
                    is_final: false,
                    confidence: None,
                    speaker_role: Some(speaker),
                });
                tokio::time::sleep(WORD_GROUP_DELAY).await;
            }
        }
        let _ = self.inject_draft(TranscriptDraft {
            utterance_id,
            source,
            start_ms: start,
            end_ms: now_ms(),
            text: text.to_string(),
            is_final: true,
            confidence: Some(0.95),
            speaker_role: Some(speaker),
        });
    }

    /// Ends a running demo (summary is generated by the demo provider).
    pub async fn stop_demo(&self) -> Result<(), CoreError> {
        let demo_meeting = {
            let mut sim = self.inner.sim.lock();
            if let Some(c) = sim.demo_cancel.take() {
                c.cancel();
            }
            sim.demo_meeting.clone()
        };
        match demo_meeting {
            Some(id)
                if self.current_meeting_id().as_deref() == Some(id.as_str()) && self.is_live() =>
            {
                self.stop_meeting().await
            }
            _ => {
                self.restore_demo_providers();
                Ok(())
            }
        }
    }

    // ------------------------------------------------------------------
    // Developer simulation (spec §73)
    // ------------------------------------------------------------------

    /// Applies a developer-tools simulation action.
    pub fn simulate(&self, action: SimulationAction) -> Result<(), CoreError> {
        match action {
            SimulationAction::Transcript { speaker, text } => self.inject_final(speaker, text),
            SimulationAction::Question { text } => self.inject_final(SpeakerRole::Remote, text),
            SimulationAction::CandidateAnswer { text } => {
                if self.mode() != Mode::Interviewer {
                    self.notice(
                        NoticeLevel::Info,
                        "Candidate answers trigger follow-ups in interviewer mode.",
                        None,
                    );
                }
                self.inject_final(SpeakerRole::Remote, text)
            }
            SimulationAction::ScreenChange {
                description,
                visible_text,
            } => {
                let mut source = description.clone();
                if let Some(v) = &visible_text {
                    source.push(' ');
                    source.push_str(v);
                }
                self.add_observation(ScreenObservation {
                    id: new_id(),
                    meeting_id: None,
                    timestamp: now_ms(),
                    source: "Simulated screen".into(),
                    application: None,
                    title: None,
                    description,
                    visible_text,
                    entities: extract_entities(&source),
                    content_type: None,
                    frame_hash: None,
                })
            }
            SimulationAction::AiResponse { text } => {
                self.install_mock_llm().push_response(text);
                Ok(())
            }
            SimulationAction::AiFailure { message } => {
                self.install_mock_llm()
                    .push_failure(ProviderError::Network(message));
                Ok(())
            }
            SimulationAction::NetworkLatency { ms } => {
                self.install_mock_llm()
                    .set_first_token_latency_ms(ms as u64);
                Ok(())
            }
        }
    }

    fn inject_final(&self, speaker: SpeakerRole, text: String) -> Result<(), CoreError> {
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err(CoreError::InvalidInput("Text must not be empty.".into()));
        }
        let now = now_ms();
        let words = text.split_whitespace().count() as i64;
        self.inject_draft(TranscriptDraft {
            utterance_id: new_id(),
            source: if speaker == SpeakerRole::User {
                AudioSourceKind::Microphone
            } else {
                AudioSourceKind::System
            },
            start_ms: now - (words * 350).clamp(500, 30_000),
            end_ms: now,
            text,
            is_final: true,
            confidence: Some(1.0),
            speaker_role: Some(speaker),
        })
    }

    /// Replaces the LLM with a deterministic mock (developer tools only).
    fn install_mock_llm(&self) -> MockLlmProvider {
        let mock = {
            let mut sim = self.inner.sim.lock();
            if let Some(m) = &sim.mock {
                return m.clone();
            }
            let mock = MockLlmProvider::new();
            let mut providers = self.inner.providers.write();
            sim.saved_llm = Some(providers.llm.take());
            providers.llm = Some(Arc::new(mock.clone()));
            sim.mock = Some(mock.clone());
            mock
        };
        self.inner.health.set(
            Component::Llm,
            ComponentStatus::Active,
            Some("Developer mock provider".into()),
        );
        self.notice(
            NoticeLevel::Info,
            "Developer tools: AI responses now come from the local mock provider.",
            Some(Component::Llm),
        );
        mock
    }

    /// Removes the developer mock and restores the configured LLM.
    pub fn clear_simulation(&self) {
        let saved = {
            let mut sim = self.inner.sim.lock();
            sim.mock = None;
            sim.saved_llm.take()
        };
        if let Some(llm) = saved {
            self.inner.providers.write().llm = llm;
            let present = self.inner.providers.read().llm.is_some();
            self.inner.health.set(
                Component::Llm,
                if present {
                    ComponentStatus::Active
                } else {
                    ComponentStatus::Disabled
                },
                None,
            );
        }
    }

    pub fn simulation_active(&self) -> bool {
        self.inner.sim.lock().mock.is_some()
    }
}
