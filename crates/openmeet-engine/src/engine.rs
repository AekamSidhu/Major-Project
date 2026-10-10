//! The public engine API: meeting lifecycle, audio/screen ingestion, assistant
//! actions, snapshots and diagnostics.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use openmeet_core::CoreError;
use openmeet_core::audio::AudioFrame;
use openmeet_core::events::{EventSink, MeetingEvent, NoticeLevel};
use openmeet_core::ids::{Id, now_ms};
use openmeet_core::ipc::{EngineSnapshot, GenerateAnswerRequest, StartMeetingRequest};
use openmeet_core::providers::{
    EmbeddingProvider, LlmProvider, ProviderSet, TranscriptionSession, TranscriptionSessionConfig,
};
use openmeet_core::settings::Settings;
use openmeet_core::store::{MeetingStore, NullStore};
use openmeet_core::types::*;
use openmeet_intel::mock::MockLlmProvider;
use openmeet_intel::text::HashingEmbedder;
use parking_lot::{Mutex, RwLock};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::embedder::{EmbedJob, spawn_embedder};
use crate::metrics::{HealthRegistry, LatencyTracker, record_latency};
use crate::persist::{PersistOp, Persister};
use crate::requests::RequestManager;
use crate::runtime::MeetingRuntime;

/// Engine errors cross the IPC boundary unchanged.
pub type EngineError = CoreError;

/// Capacity of the audio frame queue (~10 s of 20 ms frames per source).
const AUDIO_QUEUE: usize = 1024;
/// How long a graceful stop waits for in-flight transcriptions.
const STOP_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
/// Deleted meeting ids remembered so late writes for them are dropped.
const MAX_DELETED: usize = 256;

/// Everything the engine needs from the outside world.
pub struct EngineDeps {
    pub sink: Arc<dyn EventSink>,
    /// Persistent store used when the retention policy keeps meeting data.
    pub store: Arc<dyn MeetingStore>,
    pub providers: ProviderSet,
    pub settings: Settings,
}

pub(crate) enum AudioControl {
    Start(Vec<(AudioSourceKind, Box<dyn TranscriptionSession>)>),
    /// Finalize current utterances; the optional sender is completed once the
    /// sessions have been flushed.
    Flush(Option<tokio::sync::oneshot::Sender<()>>),
    Stop,
}

pub(crate) struct EngineState {
    pub runtime: Option<MeetingRuntime>,
    pub mode: Mode,
    pub style: AnswerStyle,
    pub processing_stopped: bool,
    /// Suggestions produced outside a meeting (Ask OpenMeet with no meeting),
    /// kept so overlay refinements keep working.
    pub detached: std::collections::VecDeque<Suggestion>,
    pub detached_active: Option<openmeet_core::ipc::ActiveRequestView>,
}

/// Developer-tools / demo provider overrides.
#[derive(Default)]
pub(crate) struct SimState {
    /// Mock LLM installed by developer simulation actions.
    pub mock: Option<MockLlmProvider>,
    /// LLM that was active before the mock was installed.
    pub saved_llm: Option<Option<Arc<dyn LlmProvider>>>,
    /// Providers active before a demo started.
    pub saved_providers: Option<ProviderSet>,
    pub demo_cancel: Option<CancellationToken>,
    pub demo_meeting: Option<Id>,
}

pub(crate) struct Inner {
    pub sink: Arc<dyn EventSink>,
    pub db_store: Arc<dyn MeetingStore>,
    pub providers: RwLock<ProviderSet>,
    pub settings: RwLock<Settings>,
    pub hashing: Arc<HashingEmbedder>,
    pub state: Mutex<EngineState>,
    pub requests: RequestManager,
    pub latency: LatencyTracker,
    pub health: Arc<HealthRegistry>,
    pub persist: Persister,
    pub audio_tx: mpsc::Sender<AudioFrame>,
    pub audio_ctl: mpsc::UnboundedSender<AudioControl>,
    pub screen_tx: watch::Sender<Option<ScreenFrame>>,
    pub embed_tx: mpsc::UnboundedSender<EmbedJob>,
    pub accepting_audio: AtomicBool,
    pub dropped_frames: AtomicU64,
    pub sim: Mutex<SimState>,
    /// Meetings the user deleted while work for them could still be queued.
    pub deleted: Mutex<std::collections::VecDeque<Id>>,
    /// Cancels the background summary of a meeting (keyed by meeting id).
    pub summaries: Mutex<std::collections::HashMap<Id, CancellationToken>>,
    pub handle: Handle,
}

/// Cheap, cloneable handle to the engine.
#[derive(Clone)]
pub struct Engine {
    pub(crate) inner: Arc<Inner>,
}

impl Engine {
    /// Creates the engine and spawns its background workers on `handle`.
    pub fn new(deps: EngineDeps, handle: Handle) -> Engine {
        let sink = deps.sink;
        let health = Arc::new(HealthRegistry::new(sink.clone()));
        let persist = Persister::spawn(sink.clone(), health.clone());
        let (audio_tx, audio_rx) = mpsc::channel(AUDIO_QUEUE);
        let (audio_ctl, ctl_rx) = mpsc::unbounded_channel();
        let (screen_tx, screen_rx) = watch::channel(None);
        let (embed_tx, embed_rx) = mpsc::unbounded_channel();
        let mode = deps.settings.mode;
        let style = deps.settings.assistant.answer_style;
        let inner = Arc::new(Inner {
            sink: sink.clone(),
            db_store: deps.store,
            providers: RwLock::new(deps.providers),
            settings: RwLock::new(deps.settings.sanitized()),
            hashing: Arc::new(HashingEmbedder::new(HashingEmbedder::DEFAULT_DIMS)),
            state: Mutex::new(EngineState {
                runtime: None,
                mode,
                style,
                processing_stopped: false,
                detached: std::collections::VecDeque::new(),
                detached_active: None,
            }),
            requests: RequestManager::new(sink.clone()),
            latency: LatencyTracker::new(),
            health,
            persist,
            audio_tx,
            audio_ctl,
            screen_tx,
            embed_tx,
            accepting_audio: AtomicBool::new(false),
            dropped_frames: AtomicU64::new(0),
            sim: Mutex::new(SimState::default()),
            deleted: Mutex::new(std::collections::VecDeque::new()),
            summaries: Mutex::new(std::collections::HashMap::new()),
            handle: handle.clone(),
        });
        handle.spawn(audio_router(audio_rx, ctl_rx));
        crate::screen::spawn_screen_worker(Arc::downgrade(&inner), screen_rx, &handle);
        spawn_embedder(Arc::downgrade(&inner), embed_rx, &handle);
        let engine = Engine { inner };
        engine.refresh_provider_health();
        engine
            .inner
            .health
            .set(Component::Meeting, ComponentStatus::Inactive, None);
        engine
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn settings(&self) -> Settings {
        self.inner.settings.read().clone()
    }

    pub(crate) fn sink(&self) -> &Arc<dyn EventSink> {
        &self.inner.sink
    }

    pub(crate) fn emit(&self, event: MeetingEvent) {
        self.inner.sink.emit(event);
    }

    pub(crate) fn notice(
        &self,
        level: NoticeLevel,
        message: impl Into<String>,
        component: Option<Component>,
    ) {
        self.emit(MeetingEvent::Notice {
            level,
            message: message.into(),
            component,
        });
    }

    pub fn mode(&self) -> Mode {
        self.inner.state.lock().mode
    }

    pub fn style(&self) -> AnswerStyle {
        self.inner.state.lock().style
    }

    pub(crate) fn llm(&self) -> Option<Arc<dyn LlmProvider>> {
        self.inner.providers.read().llm.clone()
    }

    pub(crate) fn background_llm(&self) -> Option<Arc<dyn LlmProvider>> {
        self.inner.providers.read().background_llm()
    }

    /// Active embedding provider (falls back to the built-in hashing embedder).
    pub(crate) fn embedder(&self) -> Arc<dyn EmbeddingProvider> {
        self.inner
            .providers
            .read()
            .embeddings
            .clone()
            .unwrap_or_else(|| self.inner.hashing.clone() as Arc<dyn EmbeddingProvider>)
    }

    pub fn current_meeting_id(&self) -> Option<Id> {
        self.inner
            .state
            .lock()
            .runtime
            .as_ref()
            .map(|r| r.meeting.id.clone())
    }

    pub fn meeting_state(&self) -> MeetingState {
        self.inner
            .state
            .lock()
            .runtime
            .as_ref()
            .map(|r| r.state())
            .unwrap_or(MeetingState::Idle)
    }

    pub fn is_live(&self) -> bool {
        self.meeting_state().is_live()
    }

    pub fn processing_stopped(&self) -> bool {
        self.inner.state.lock().processing_stopped
    }

    /// Store of the current meeting (NullStore when nothing is retained).
    pub(crate) fn current_store(&self) -> Arc<dyn MeetingStore> {
        self.inner
            .state
            .lock()
            .runtime
            .as_ref()
            .map(|r| r.store.clone())
            .unwrap_or_else(|| Arc::new(NullStore))
    }

    pub(crate) fn persist(&self, store: &Arc<dyn MeetingStore>, op: PersistOp) {
        // A write that arrives after the user deleted its meeting must not
        // bring the meeting back.
        if let Some(id) = op.meeting_id() {
            if self.is_deleted(id) {
                tracing::debug!(op = op.name(), "dropped a write for a deleted meeting");
                return;
            }
        }
        self.inner.persist.send(store, op);
    }

    pub(crate) fn is_deleted(&self, meeting_id: &str) -> bool {
        self.inner.deleted.lock().iter().any(|d| d == meeting_id)
    }

    /// Call before deleting a stored meeting: cancels its summary if one is
    /// still being written, drops any later writes for it, waits for writes
    /// already queued, and clears it from memory if it is the last meeting.
    /// Fails while the meeting is still running or ending.
    pub async fn forget_meeting(&self, meeting_id: &str) -> Result<(), CoreError> {
        {
            let st = self.inner.state.lock();
            if let Some(rt) = st.runtime.as_ref().filter(|r| r.meeting.id == meeting_id) {
                let s = rt.state();
                if s.is_live() || matches!(s, MeetingState::Starting | MeetingState::Stopping) {
                    return Err(CoreError::InvalidState(
                        "End the meeting before deleting it.".into(),
                    ));
                }
            }
        }
        {
            let mut deleted = self.inner.deleted.lock();
            if !deleted.iter().any(|d| d == meeting_id) {
                if deleted.len() >= MAX_DELETED {
                    deleted.pop_front();
                }
                deleted.push_back(meeting_id.to_string());
            }
        }
        if let Some(token) = self.inner.summaries.lock().remove(meeting_id) {
            token.cancel();
        }
        self.flush_persistence().await;
        let cleared = {
            let mut st = self.inner.state.lock();
            let hit = st
                .runtime
                .as_ref()
                .is_some_and(|r| r.meeting.id == meeting_id);
            if hit {
                st.runtime = None;
            }
            hit.then_some(st.mode)
        };
        if let Some(mode) = cleared {
            // `Idle` without a meeting id tells windows to drop the meeting.
            self.emit(MeetingEvent::MeetingStateChanged {
                meeting_id: None,
                state: MeetingState::Idle,
                mode,
                paused_reason: None,
            });
        }
        Ok(())
    }

    /// [`Engine::forget_meeting`] for every meeting (before deleting all data).
    pub async fn forget_all_meetings(&self) -> Result<(), CoreError> {
        match self.current_meeting_id() {
            Some(id) => self.forget_meeting(&id).await,
            None => {
                self.flush_persistence().await;
                Ok(())
            }
        }
    }

    /// Waits until every queued write of the current meeting is applied.
    pub async fn flush_persistence(&self) {
        let store = self.current_store();
        self.inner.persist.flush(&store).await;
        let db = self.inner.db_store.clone();
        self.inner.persist.flush(&db).await;
    }

    // ------------------------------------------------------------------
    // Settings & providers
    // ------------------------------------------------------------------

    /// Applies new settings (already validated by the caller). Provider
    /// changes are applied separately through [`Engine::set_providers`].
    pub fn update_settings(&self, settings: Settings) {
        let settings = settings.sanitized();
        let (style_changed, mode_changed) = {
            let mut st = self.inner.state.lock();
            let live = st.runtime.as_ref().is_some_and(|r| r.state().is_live());
            let style_changed = st.style != settings.assistant.answer_style;
            st.style = settings.assistant.answer_style;
            let mode_changed = !live && st.mode != settings.mode;
            if mode_changed {
                st.mode = settings.mode;
            }
            (
                style_changed.then_some(st.style),
                mode_changed.then_some(st.mode),
            )
        };
        *self.inner.settings.write() = settings;
        if let Some(style) = style_changed {
            self.emit(MeetingEvent::StyleChanged { style });
        }
        if let Some(mode) = mode_changed {
            self.emit(MeetingEvent::ModeChanged { mode });
        }
    }

    /// Hot-swaps providers. Running transcription sessions are restarted when
    /// the transcription provider changed.
    pub fn set_providers(&self, providers: ProviderSet) {
        let transcription_changed = {
            let current = self.inner.providers.read();
            !same_arc(&current.transcription, &providers.transcription)
        };
        {
            let mut sim = self.inner.sim.lock();
            // A running demo keeps its deterministic providers; the new set is
            // applied when the demo ends.
            if sim.demo_meeting.is_some() {
                sim.saved_providers = Some(providers);
                return;
            }
            // Keep a developer mock override in place (restored on clear).
            let mut p = providers;
            if let Some(mock) = sim.mock.clone() {
                sim.saved_llm = Some(p.llm.take());
                p.llm = Some(Arc::new(mock));
            }
            *self.inner.providers.write() = p;
        }
        self.refresh_provider_health();
        if transcription_changed {
            let restart = {
                let st = self.inner.state.lock();
                st.runtime
                    .as_ref()
                    .filter(|r| r.state() == MeetingState::Active && r.sessions_running)
                    .map(sources_of)
            };
            if let Some(sources) = restart {
                let _ = self.inner.audio_ctl.send(AudioControl::Stop);
                self.start_sessions(&sources);
            }
        }
    }

    fn refresh_provider_health(&self) {
        let p = self.inner.providers.read().clone();
        let h = &self.inner.health;
        let status_for = |present: bool, component: Component| {
            if present {
                (ComponentStatus::Active, None)
            } else if let Some(issue) = p.issue_for(component) {
                (ComponentStatus::Error, Some(issue.message.clone()))
            } else {
                (ComponentStatus::Disabled, None)
            }
        };
        let (s, m) = status_for(p.llm.is_some(), Component::Llm);
        h.set(Component::Llm, s, m);
        let (s, m) = status_for(p.transcription.is_some(), Component::Transcription);
        h.set(Component::Transcription, s, m);
        let (s, m) = status_for(p.vision.is_some() || p.ocr.is_some(), Component::Vision);
        h.set(Component::Vision, s, m);
        let emb_msg = p
            .embeddings
            .as_ref()
            .map(|e| e.info().display_name)
            .or(Some("Built-in hashing embeddings".to_string()));
        h.set(Component::Embeddings, ComponentStatus::Active, emb_msg);
    }

    /// Provider infos for the data-flow view.
    pub fn provider_set(&self) -> ProviderSet {
        self.inner.providers.read().clone()
    }

    // ------------------------------------------------------------------
    // Meeting lifecycle (spec §39, §86)
    // ------------------------------------------------------------------

    pub async fn start_meeting(&self, req: StartMeetingRequest) -> Result<Meeting, CoreError> {
        self.start_meeting_inner(req, false)
    }

    pub(crate) fn start_meeting_inner(
        &self,
        req: StartMeetingRequest,
        demo: bool,
    ) -> Result<Meeting, CoreError> {
        let settings = self.settings();
        {
            let st = self.inner.state.lock();
            if let Some(rt) = &st.runtime {
                if matches!(
                    rt.state(),
                    MeetingState::Starting
                        | MeetingState::Active
                        | MeetingState::Paused
                        | MeetingState::Stopping
                ) {
                    return Err(CoreError::InvalidState(
                        "A meeting is already running. End it before starting another.".into(),
                    ));
                }
            }
        }
        self.inner.requests.cancel_all(false);

        let now = now_ms();
        let title = req
            .title
            .as_ref()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| default_title(req.mode));
        let persisted = settings.privacy.retention.persists() && !demo;
        let meeting = Meeting {
            id: openmeet_core::new_id(),
            title,
            mode: req.mode,
            state: MeetingState::Starting,
            started_at: now,
            ended_at: None,
            participants: MeetingRuntime::participants_for(req.mode, req.system_audio),
            persisted,
        };
        let store: Arc<dyn MeetingStore> = if persisted {
            self.inner.db_store.clone()
        } else {
            Arc::new(NullStore)
        };
        let (conv_tx, conv_rx) = mpsc::unbounded_channel();
        let (trans_tx, trans_rx) = mpsc::unbounded_channel();
        let mut rt = MeetingRuntime::new(
            meeting.clone(),
            store.clone(),
            &settings.profile.planned_topic_list(),
            req.system_audio,
            req.microphone,
            demo,
            conv_tx,
            trans_tx,
        );
        rt.transition(MeetingState::Starting)
            .map_err(CoreError::InvalidState)?;
        let meeting_id = meeting.id.clone();
        {
            let mut st = self.inner.state.lock();
            st.mode = req.mode;
            st.processing_stopped = false;
            st.runtime = Some(rt);
        }
        self.inner.settings.write().mode = req.mode;
        self.emit_state(&meeting_id);
        self.emit(MeetingEvent::ModeChanged { mode: req.mode });
        crate::conversation::spawn_conversation(
            self.clone(),
            meeting_id.clone(),
            trans_rx,
            conv_rx,
        );

        // Demo meetings are fed by a script, not by audio.
        if !demo {
            let mut sources = Vec::new();
            if req.microphone {
                sources.push(AudioSourceKind::Microphone);
            }
            if req.system_audio {
                sources.push(AudioSourceKind::System);
            }
            self.start_sessions(&sources);
        }

        let started = {
            let mut st = self.inner.state.lock();
            let rt = st.runtime.as_mut().expect("runtime just installed");
            rt.transition(MeetingState::Active)
                .map_err(CoreError::InvalidState)?;
            rt.meeting.clone()
        };
        self.persist(&store, PersistOp::Meeting(started.clone()));
        self.inner.accepting_audio.store(true, Ordering::Release);
        self.emit(MeetingEvent::MeetingStarted {
            meeting: started.clone(),
        });
        self.emit_state(&meeting_id);
        self.inner
            .health
            .set(Component::Meeting, ComponentStatus::Active, None);
        self.inner
            .health
            .set(Component::QuestionDetector, ComponentStatus::Active, None);
        tracing::info!(meeting = %meeting_id, mode = ?req.mode, demo, "meeting started");
        Ok(started)
    }

    /// Starts transcription sessions for `sources` and hands them to the
    /// audio router.
    fn start_sessions(&self, sources: &[AudioSourceKind]) {
        if sources.is_empty() {
            return;
        }
        let settings = self.settings();
        let provider = self.inner.providers.read().transcription.clone();
        let trans_tx = {
            let st = self.inner.state.lock();
            match st.runtime.as_ref() {
                Some(rt) => rt.trans_tx.clone(),
                None => return,
            }
        };
        let Some(provider) = provider else {
            let msg = self
                .inner
                .providers
                .read()
                .issue_for(Component::Transcription)
                .map(|i| i.message.clone())
                .unwrap_or_else(|| "No transcription provider is configured.".into());
            self.inner
                .health
                .set(Component::Transcription, ComponentStatus::Error, Some(msg));
            self.notice(
                NoticeLevel::Warning,
                "Transcription unavailable. Check transcription settings — Ask OpenMeet still works.",
                Some(Component::Transcription),
            );
            return;
        };
        let mut sessions = Vec::new();
        for &source in sources {
            let cfg = TranscriptionSessionConfig {
                source,
                language: settings.transcription.language.clone(),
                vocabulary: Some(settings.transcription.vocabulary.clone())
                    .filter(|v| !v.trim().is_empty()),
                interim_results: settings.transcription.interim_results,
                vad_sensitivity: settings.audio.vad_sensitivity,
                silence_ms: settings.audio.silence_ms,
            };
            match provider.start_session(cfg, trans_tx.clone()) {
                Ok(s) => sessions.push((source, s)),
                Err(e) => {
                    self.inner.health.set(
                        Component::Transcription,
                        ComponentStatus::Error,
                        Some(e.user_message()),
                    );
                }
            }
        }
        if !sessions.is_empty() {
            self.inner.health.set(
                Component::Transcription,
                ComponentStatus::Streaming,
                Some(provider.info().display_name),
            );
            if let Some(rt) = self.inner.state.lock().runtime.as_mut() {
                rt.sessions_running = true;
            }
            let _ = self.inner.audio_ctl.send(AudioControl::Start(sessions));
        }
    }

    pub(crate) fn emit_state(&self, meeting_id: &str) {
        let ev = {
            let st = self.inner.state.lock();
            st.runtime
                .as_ref()
                .filter(|r| r.meeting.id == meeting_id)
                .map(|r| MeetingEvent::MeetingStateChanged {
                    meeting_id: Some(r.meeting.id.clone()),
                    state: r.state(),
                    mode: r.mode(),
                    paused_reason: r.paused_reason,
                })
        };
        if let Some(ev) = ev {
            self.emit(ev);
        }
    }

    /// Pauses all processing (spec §39). Speech captured before the pause is
    /// still transcribed; no new AI requests start until resumed.
    pub fn pause(&self) -> Result<(), CoreError> {
        let meeting_id = {
            let mut st = self.inner.state.lock();
            let rt = st
                .runtime
                .as_mut()
                .ok_or_else(|| CoreError::InvalidState("No meeting is running.".into()))?;
            if rt.state() == MeetingState::Paused {
                return Ok(());
            }
            rt.transition(MeetingState::Paused)
                .map_err(CoreError::InvalidState)?;
            rt.paused_reason = Some(PauseReason::User);
            rt.meeting.id.clone()
        };
        self.inner.accepting_audio.store(false, Ordering::Release);
        let _ = self.inner.audio_ctl.send(AudioControl::Flush(None));
        self.inner.requests.cancel_all(false);
        self.inner
            .health
            .set(Component::Meeting, ComponentStatus::Paused, None);
        self.emit_state(&meeting_id);
        Ok(())
    }

    /// Resumes after a pause or an emergency stop.
    pub fn resume(&self) -> Result<(), CoreError> {
        let (meeting_id, restart) = {
            let mut st = self.inner.state.lock();
            st.processing_stopped = false;
            let rt = st
                .runtime
                .as_mut()
                .ok_or_else(|| CoreError::InvalidState("No meeting is running.".into()))?;
            if rt.state() == MeetingState::Active {
                return Ok(());
            }
            rt.transition(MeetingState::Active)
                .map_err(CoreError::InvalidState)?;
            rt.paused_reason = None;
            let restart = (!rt.sessions_running && !rt.demo).then(|| sources_of(rt));
            (rt.meeting.id.clone(), restart)
        };
        if let Some(sources) = restart {
            self.start_sessions(&sources);
        }
        self.inner.accepting_audio.store(true, Ordering::Release);
        self.inner
            .health
            .set(Component::Meeting, ComponentStatus::Active, None);
        self.emit_state(&meeting_id);
        Ok(())
    }

    /// Emergency privacy stop (spec §40): stops transcription, cancels every
    /// AI request and pending frame immediately. The desktop layer also stops
    /// audio/screen capture and hides the overlay.
    pub fn emergency_stop(&self) {
        self.inner.accepting_audio.store(false, Ordering::Release);
        let _ = self.inner.audio_ctl.send(AudioControl::Stop);
        self.inner.requests.cancel_all(true);
        let _ = self.inner.screen_tx.send(None);
        if let Some(cancel) = self.inner.sim.lock().demo_cancel.take() {
            cancel.cancel();
        }
        let meeting_id = {
            let mut st = self.inner.state.lock();
            st.processing_stopped = true;
            st.runtime.as_mut().and_then(|rt| {
                rt.sessions_running = false;
                rt.pending_utterances.clear();
                rt.speaking.clear();
                if rt.state() == MeetingState::Active {
                    let _ = rt.transition(MeetingState::Paused);
                }
                if rt.state() == MeetingState::Paused {
                    rt.paused_reason = Some(PauseReason::Emergency);
                    Some(rt.meeting.id.clone())
                } else {
                    None
                }
            })
        };
        self.emit(MeetingEvent::ProcessingStopped {
            reason: "Emergency stop: audio, screen and AI processing halted.".into(),
        });
        if let Some(id) = meeting_id {
            self.emit_state(&id);
            self.inner.health.set(
                Component::Meeting,
                ComponentStatus::Paused,
                Some("Emergency stop".into()),
            );
        }
        self.inner.health.set(
            Component::Transcription,
            ComponentStatus::Paused,
            Some("Emergency stop".into()),
        );
        tracing::warn!("emergency stop");
    }

    /// Ends the meeting: drains in-flight transcriptions, then generates the
    /// summary in the background (spec §41).
    pub async fn stop_meeting(&self) -> Result<(), CoreError> {
        let (meeting_id, demo) = {
            let mut st = self.inner.state.lock();
            let rt = st
                .runtime
                .as_mut()
                .ok_or_else(|| CoreError::InvalidState("No meeting is running.".into()))?;
            if !rt.state().is_live() {
                return Err(CoreError::InvalidState(
                    "The meeting is not running.".into(),
                ));
            }
            rt.transition(MeetingState::Stopping)
                .map_err(CoreError::InvalidState)?;
            (rt.meeting.id.clone(), rt.demo)
        };
        self.inner.accepting_audio.store(false, Ordering::Release);
        self.emit_state(&meeting_id);
        if let Some(cancel) = self.inner.sim.lock().demo_cancel.take() {
            cancel.cancel();
        }

        // Finalize the utterance in progress and wait for pending transcripts.
        let deadline = tokio::time::Instant::now() + STOP_DRAIN_TIMEOUT;
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let _ = self.inner.audio_ctl.send(AudioControl::Flush(Some(ack_tx)));
        let _ = tokio::time::timeout_at(deadline, ack_rx).await;
        // Let the conversation loop register the utterances submitted by the
        // flush before checking what is still pending.
        tokio::time::sleep(Duration::from_millis(150)).await;
        loop {
            let pending = {
                let st = self.inner.state.lock();
                st.runtime
                    .as_ref()
                    .map(|r| r.pending_utterances.len())
                    .unwrap_or(0)
            };
            if pending == 0 || tokio::time::Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let _ = self.inner.audio_ctl.send(AudioControl::Stop);
        self.inner.requests.cancel_all(false);

        let (store, meeting, answer, topics) = {
            let mut st = self.inner.state.lock();
            let rt = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id)
                .ok_or_else(|| CoreError::InvalidState("The meeting ended unexpectedly.".into()))?;
            rt.sessions_running = false;
            let answer = rt.close_answer();
            rt.meeting.ended_at = Some(now_ms());
            rt.transition(MeetingState::Summarizing)
                .map_err(CoreError::InvalidState)?;
            (
                rt.store.clone(),
                rt.meeting.clone(),
                answer,
                rt.topics.topics(),
            )
        };
        if let Some(a) = answer {
            self.persist(&store, PersistOp::Answer(a));
        }
        self.persist(
            &store,
            PersistOp::Topics {
                meeting_id: meeting_id.clone(),
                topics,
            },
        );
        self.persist(&store, PersistOp::Meeting(meeting));
        self.emit(MeetingEvent::MeetingStopped {
            meeting_id: meeting_id.clone(),
        });
        self.emit_state(&meeting_id);
        self.inner
            .health
            .set(Component::Meeting, ComponentStatus::Inactive, None);
        self.inner
            .health
            .set(Component::QuestionDetector, ComponentStatus::Inactive, None);

        // The summary task captures the LLM now, so demo providers can be
        // restored immediately afterwards.
        crate::assist::spawn_meeting_summary(self.clone(), meeting_id.clone());
        if demo {
            self.restore_demo_providers();
        }
        tracing::info!(meeting = %meeting_id, "meeting stopped");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Ingestion
    // ------------------------------------------------------------------

    /// Feeds captured audio. Non-blocking; frames are dropped (and counted)
    /// if the pipeline falls behind or the meeting is not active.
    pub fn push_audio(&self, frame: AudioFrame) {
        if !self.inner.accepting_audio.load(Ordering::Acquire) {
            return;
        }
        if self.inner.audio_tx.try_send(frame).is_err() {
            let dropped = self.inner.dropped_frames.fetch_add(1, Ordering::Relaxed) + 1;
            if dropped % 100 == 1 {
                tracing::warn!(dropped, "audio frames dropped: pipeline overloaded");
                self.inner.health.set(
                    Component::Transcription,
                    ComponentStatus::Degraded,
                    Some("Audio pipeline is overloaded; some audio was dropped".into()),
                );
            }
        }
    }

    /// Feeds a change-filtered screen frame. Only the latest pending frame is
    /// kept; frames are ignored unless a meeting is active.
    pub fn push_screen_frame(&self, frame: ScreenFrame) {
        if self.meeting_state() != MeetingState::Active {
            return;
        }
        let _ = self.inner.screen_tx.send(Some(frame));
    }

    // ------------------------------------------------------------------
    // Assistant actions
    // ------------------------------------------------------------------

    /// Generate (or regenerate) a suggested answer (hotkey Ctrl+Shift+Enter).
    pub fn generate_answer(&self, req: GenerateAnswerRequest) -> Result<Id, CoreError> {
        crate::assist::generate_answer(self, req)
    }

    /// Interviewer: follow-ups for the candidate's latest answer.
    /// Candidate: questions the user could ask the interviewer.
    pub fn generate_followups(&self) -> Result<Id, CoreError> {
        crate::assist::generate_followups(self)
    }

    /// Ask OpenMeet anything with the current meeting / screen context.
    pub fn ask(&self, prompt: String) -> Result<Id, CoreError> {
        crate::assist::ask(self, prompt)
    }

    /// Clarifying questions for the latest (ambiguous) question.
    pub fn clarify(&self) -> Result<Id, CoreError> {
        crate::assist::clarify(self)
    }

    /// Overlay answer controls (Shorter, Longer, Regenerate, ...).
    pub fn refine(&self, request_id: &str, refinement: Refinement) -> Result<Id, CoreError> {
        crate::assist::refine(self, request_id, refinement)
    }

    pub fn cancel_request(&self, request_id: &str) -> bool {
        self.inner.requests.cancel(request_id)
    }

    pub fn dismiss(&self, request_id: &str) {
        self.inner.requests.cancel(request_id);
        {
            let mut st = self.inner.state.lock();
            if let Some(rt) = st.runtime.as_mut() {
                if rt
                    .active_primary
                    .as_ref()
                    .is_some_and(|a| a.request_id == request_id)
                {
                    rt.active_primary = None;
                }
                if rt.latest_followups_request.as_deref() == Some(request_id) {
                    rt.latest_followups.clear();
                    rt.latest_followups_request = None;
                }
            }
        }
        self.emit(MeetingEvent::SuggestionDismissed {
            request_id: request_id.to_string(),
        });
    }

    pub fn pin(&self, request_id: &str, pinned: bool) -> Result<(), CoreError> {
        let (store, suggestion) = {
            let mut st = self.inner.state.lock();
            let rt = st
                .runtime
                .as_mut()
                .ok_or_else(|| CoreError::InvalidState("No meeting is running.".into()))?;
            let store = rt.store.clone();
            let s = rt.suggestion_mut(request_id).ok_or_else(|| {
                CoreError::NotFound("That suggestion is no longer available.".into())
            })?;
            s.pinned = pinned;
            (store, s.clone())
        };
        self.persist(
            &store,
            PersistOp::Pinned {
                id: request_id.to_string(),
                pinned,
            },
        );
        self.emit(MeetingEvent::SuggestionPinned {
            request_id: request_id.to_string(),
            pinned,
            suggestion: Some(suggestion),
        });
        Ok(())
    }

    pub fn set_mode(&self, mode: Mode) {
        let meeting_id = {
            let mut st = self.inner.state.lock();
            st.mode = mode;
            st.runtime.as_mut().and_then(|rt| {
                if rt.state().is_live() {
                    rt.meeting.mode = mode;
                    rt.meeting.participants =
                        MeetingRuntime::participants_for(mode, rt.system_audio_active);
                    rt.turn.reset();
                    Some(rt.meeting.id.clone())
                } else {
                    None
                }
            })
        };
        self.inner.settings.write().mode = mode;
        self.emit(MeetingEvent::ModeChanged { mode });
        if let Some(id) = meeting_id {
            self.emit_state(&id);
        }
    }

    pub fn toggle_mode(&self) -> Mode {
        let next = self.mode().toggled();
        self.set_mode(next);
        next
    }

    pub fn set_style(&self, style: AnswerStyle) {
        self.inner.state.lock().style = style;
        self.inner.settings.write().assistant.answer_style = style;
        self.emit(MeetingEvent::StyleChanged { style });
    }

    pub fn cycle_style(&self) -> AnswerStyle {
        let next = self.style().next();
        self.set_style(next);
        next
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    /// Lets the desktop layer report health of OS components (microphone,
    /// overlay, hotkeys, capture privacy...).
    pub fn set_component_status(
        &self,
        component: Component,
        status: ComponentStatus,
        message: Option<String>,
    ) {
        self.inner.health.set(component, status, message);
    }

    pub fn components(&self) -> Vec<ComponentHealth> {
        self.inner.health.all()
    }

    pub fn latency_stats(&self) -> Vec<LatencyStats> {
        self.inner.latency.stats()
    }

    /// Records a latency sample measured outside the engine (overlay render).
    pub fn record_latency(&self, stage: LatencyStage, ms: f64) {
        record_latency(&self.inner.latency, &self.inner.sink, stage, ms, None);
    }

    pub fn active_request_count(&self) -> u32 {
        self.inner.requests.active_count() as u32
    }

    pub fn dropped_audio_frames(&self) -> u64 {
        self.inner.dropped_frames.load(Ordering::Relaxed)
    }

    /// Complete state for windows that (re)load mid-meeting.
    pub fn snapshot(&self) -> EngineSnapshot {
        let st = self.inner.state.lock();
        let components = self.inner.health.all();
        match st.runtime.as_ref() {
            Some(rt) => {
                let tail_start = rt.segments.len().saturating_sub(60);
                EngineSnapshot {
                    meeting: Some(rt.meeting.clone()),
                    state: rt.state(),
                    paused_reason: rt.paused_reason,
                    mode: st.mode,
                    style: st.style,
                    transcript: rt.segments[tail_start..].to_vec(),
                    questions: rt.questions.iter().rev().take(20).rev().cloned().collect(),
                    active: rt.active_primary.clone(),
                    followups: rt.latest_followups.clone(),
                    followups_request_id: rt.latest_followups_request.clone(),
                    pinned: rt
                        .suggestions
                        .iter()
                        .filter(|s| s.pinned)
                        .cloned()
                        .collect(),
                    topics: rt.topics.topics(),
                    suggested_next_topic: rt.topics.suggested_next(),
                    components,
                    last_observation: rt.observations.last().cloned(),
                    last_summary: rt.summary.clone(),
                    demo_running: rt.demo && rt.state().is_live(),
                    processing_stopped: st.processing_stopped,
                }
            }
            None => EngineSnapshot {
                meeting: None,
                state: MeetingState::Idle,
                paused_reason: None,
                mode: st.mode,
                style: st.style,
                transcript: Vec::new(),
                questions: Vec::new(),
                active: st.detached_active.clone(),
                followups: Vec::new(),
                followups_request_id: None,
                pinned: Vec::new(),
                topics: Vec::new(),
                suggested_next_topic: None,
                components,
                last_observation: None,
                last_summary: None,
                demo_running: false,
                processing_stopped: st.processing_stopped,
            },
        }
    }

    /// Full record of the current (or just finished) meeting held in memory.
    /// Used for history/export of meetings that are not persisted (retention
    /// "do not retain", demo meetings).
    pub fn current_record(&self, meeting_id: &str) -> Option<MeetingRecord> {
        let st = self.inner.state.lock();
        st.runtime
            .as_ref()
            .filter(|r| r.meeting.id == meeting_id)
            .map(|r| r.to_record())
    }

    /// The in-memory meeting when it is not persisted to the database.
    pub fn unpersisted_meeting(&self) -> Option<MeetingListItem> {
        let st = self.inner.state.lock();
        st.runtime
            .as_ref()
            .filter(|r| !r.meeting.persisted)
            .map(|r| MeetingListItem {
                id: r.meeting.id.clone(),
                title: r.meeting.title.clone(),
                mode: r.meeting.mode,
                state: r.state(),
                started_at: r.meeting.started_at,
                ended_at: r.meeting.ended_at,
                segment_count: r.segments.len() as u32,
                question_count: r.questions.len() as u32,
                has_summary: r.summary.is_some(),
            })
    }

    /// Generates a summary for a stored meeting (history view) and saves it.
    pub async fn summarize_record(
        &self,
        record: MeetingRecord,
    ) -> Result<MeetingSummary, CoreError> {
        let summary = crate::assist::summarize_record(self, &record).await;
        let store = self.inner.db_store.clone();
        if record.meeting.persisted {
            self.persist(&store, PersistOp::Summary(summary.clone()));
            self.inner.persist.flush(&store).await;
        }
        Ok(summary)
    }

    pub(crate) fn hashing(&self) -> Arc<HashingEmbedder> {
        self.inner.hashing.clone()
    }

    pub(crate) fn restore_demo_providers(&self) {
        let saved = {
            let mut sim = self.inner.sim.lock();
            sim.demo_meeting = None;
            sim.saved_providers.take()
        };
        if let Some(p) = saved {
            self.set_providers(p);
        }
    }
}

fn same_arc<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => Arc::ptr_eq(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn sources_of(rt: &MeetingRuntime) -> Vec<AudioSourceKind> {
    let mut v = Vec::new();
    if rt.microphone_active {
        v.push(AudioSourceKind::Microphone);
    }
    if rt.system_audio_active {
        v.push(AudioSourceKind::System);
    }
    v
}

fn default_title(mode: Mode) -> String {
    match mode {
        Mode::Candidate => "Meeting".to_string(),
        Mode::Interviewer => "Interview".to_string(),
    }
}

/// Routes audio frames to the transcription session of their source.
async fn audio_router(
    mut frames: mpsc::Receiver<AudioFrame>,
    mut control: mpsc::UnboundedReceiver<AudioControl>,
) {
    let mut sessions: HashMap<AudioSourceKind, Box<dyn TranscriptionSession>> = HashMap::new();
    loop {
        tokio::select! {
            biased;
            ctl = control.recv() => match ctl {
                Some(AudioControl::Start(new_sessions)) => {
                    for s in sessions.values_mut() {
                        s.stop();
                    }
                    sessions.clear();
                    for (source, s) in new_sessions {
                        sessions.insert(source, s);
                    }
                    // Drop frames queued for the previous sessions.
                    while frames.try_recv().is_ok() {}
                }
                Some(AudioControl::Flush(ack)) => {
                    while let Ok(frame) = frames.try_recv() {
                        if let Some(s) = sessions.get_mut(&frame.source) {
                            s.push_audio(frame);
                        }
                    }
                    for s in sessions.values_mut() {
                        s.flush();
                    }
                    if let Some(ack) = ack {
                        let _ = ack.send(());
                    }
                }
                Some(AudioControl::Stop) => {
                    for s in sessions.values_mut() {
                        s.stop();
                    }
                    sessions.clear();
                    while frames.try_recv().is_ok() {}
                }
                None => break,
            },
            frame = frames.recv() => match frame {
                Some(frame) => {
                    if let Some(s) = sessions.get_mut(&frame.source) {
                        s.push_audio(frame);
                    }
                }
                None => break,
            },
        }
    }
    for s in sessions.values_mut() {
        s.stop();
    }
}
