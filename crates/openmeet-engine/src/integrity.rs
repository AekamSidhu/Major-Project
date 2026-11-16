//! Answer-integrity analyzer (interviewer mode, opt-in; spec/integrity).
//!
//! An *answer* is the candidate's speech from one interviewer question until
//! the next one (or the end of the meeting). Whenever the candidate pauses
//! (the turn-end silence), the answer so far is checked in the background;
//! if the candidate continues, the same answer is checked again and its flag
//! replaced (`revision` grows). The answer is final when the next question
//! arrives or the meeting ends; only then does it join the candidate's
//! delivery baseline.
//!
//! Signals:
//! * `ai_text` — the sidecar's `/v1/integrity/echo` on the transcript;
//! * `echo_similarity` — the configured LLM answers the question
//!   `references` times (started as soon as the question is detected, so a
//!   local model is idle again before follow-ups are needed), answer and
//!   references are embedded with the configured embedder, highest cosine;
//!   `unavailable` unless the embedder is the model the thresholds hold for;
//! * `delivery` — computed synchronously from transcript timings (word
//!   timestamps when the recognizer provided them) against the baseline, so
//!   the baseline order never depends on network timing;
//! * `reading` — the sidecar's `/v1/integrity/reading` for the answer's time
//!   window, while gaze capture of the candidate's video region runs.
//!
//! Nothing here blocks the conversation loop or the UI request slots: checks
//! and reference answers are `Background` requests (cancelled by an emergency
//! stop), and a stale revision never replaces a newer one.

use std::collections::VecDeque;
use std::sync::{Arc, Weak};
use std::time::Duration;

use openmeet_core::error::ProviderError;
use openmeet_core::events::MeetingEvent;
use openmeet_core::ids::{Id, new_id, now_ms};
use openmeet_core::integrity::{
    IntegrityFlag, IntegrityLevel, IntegrityService, IntegritySignal, IntegritySignalName,
};
use openmeet_core::ipc::Rect;
use openmeet_core::providers::{ChatMessage, EmbeddingProvider, LlmProvider, LlmRequest};
use openmeet_core::settings::Settings;
use openmeet_core::types::*;
use openmeet_intel::integrity::evidence::{
    ai_text_signal, delivery_signal, echo_signal, fusion_inputs, headline, not_scored,
    reading_signal,
};
use openmeet_intel::integrity::{
    DeliveryTracker, SPEC_VERSION, SpeechUnit, SpeechWord, delivery_features, echo,
    echo_similarity, fuse, is_echo_embedding_model, spec_thresholds,
};
use openmeet_intel::prompts::integrity_reference;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::engine::{Engine, Inner};
use crate::persist::PersistOp;
use crate::requests::RequestTicket;
use crate::runtime::MeetingRuntime;

/// Timeout of one reference-answer request.
const REFERENCE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a check waits for the question's reference answers.
const REFERENCE_WAIT: Duration = Duration::from_secs(90);
/// Timeout of the sidecar scoring calls and of embedding.
const SCORING_TIMEOUT: Duration = Duration::from_secs(40);
/// Reference answers kept for this many recent questions.
const MAX_REFERENCE_QUESTIONS: usize = 8;
/// Flags in the engine snapshot (latest first).
const MAX_SNAPSHOT_FLAGS: usize = 20;
/// How long the end-of-meeting gaze reset waits for running checks.
const GAZE_RESET_WAIT: Duration = Duration::from_secs(60);
/// Region handed to the scripted demo sidecar (which captures nothing).
pub(crate) const DEMO_REGION: Rect = Rect {
    x: 0,
    y: 0,
    width: 640,
    height: 360,
};

/// Reference answers of one question.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ReferenceState {
    Pending,
    Ready(Vec<String>),
    Failed(String),
}

/// The interviewer question an answer responds to.
#[derive(Debug, Clone)]
pub(crate) struct IntegrityQuestion {
    pub id: Id,
    pub text: String,
    /// When the question finished, meeting-relative ms.
    pub end_ms: i64,
}

/// One candidate utterance (meeting-relative ms).
#[derive(Debug, Clone)]
pub(crate) struct AnswerUnit {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub words: Option<Vec<WordTiming>>,
}

/// The candidate's answer in progress.
#[derive(Debug, Clone)]
pub(crate) struct OpenAnswer {
    pub id: Id,
    pub question: Option<IntegrityQuestion>,
    pub units: Vec<AnswerUnit>,
    /// Units covered by the latest check.
    pub checked_units: usize,
    pub revision: u32,
    pub created_at: i64,
    /// Request id of the check in flight.
    pub running: Option<Id>,
}

/// Per-meeting integrity state (part of [`MeetingRuntime`]).
#[derive(Default)]
pub(crate) struct IntegrityRun {
    /// On for this meeting regardless of settings (the integrity demo).
    pub forced: bool,
    /// Region used instead of the settings one (demo).
    pub demo_region: Option<Rect>,
    pub question: Option<IntegrityQuestion>,
    pub answer: Option<OpenAnswer>,
    /// Bumped whenever the candidate starts or finishes an utterance; a
    /// check timer only fires for the latest version.
    pub version: u64,
    pub delivery: DeliveryTracker,
    pub references: VecDeque<(Id, watch::Receiver<ReferenceState>)>,
    /// Latest revision of every flag, in answer order.
    pub flags: Vec<IntegrityFlag>,
    /// Gaze capture of the candidate region was requested.
    pub gaze_running: bool,
    /// The first gaze start of the meeting resets the sidecar's samples.
    pub gaze_reset_done: bool,
    /// The sidecar's embedding model (resolves "default").
    pub sidecar_embedding_model: Option<String>,
}

impl IntegrityRun {
    /// The check runs for this meeting right now.
    pub fn active(&self, settings: &Settings, mode: Mode) -> bool {
        mode == Mode::Interviewer && (self.forced || settings.integrity.enabled)
    }

    /// `name` is computed (check active and signal enabled).
    pub fn signal_on(&self, settings: &Settings, mode: Mode, name: IntegritySignalName) -> bool {
        self.active(settings, mode) && (self.forced || settings.integrity.signal_enabled(name))
    }

    /// Candidate video region for gaze.
    pub fn region(&self, settings: &Settings) -> Option<Rect> {
        self.demo_region.or(settings.integrity.candidate_region)
    }

    /// Flags for the snapshot, latest first.
    pub fn snapshot_flags(&self) -> Vec<IntegrityFlag> {
        let mut flags = self.flags.clone();
        flags.sort_by_key(|f| std::cmp::Reverse((f.answer_start_ms, f.created_at)));
        flags.truncate(MAX_SNAPSHOT_FLAGS);
        flags
    }
}

/// Work decided under the state lock, run after releasing it.
pub(crate) enum IntegrityAction {
    Check(Box<CheckJob>),
    References {
        question: String,
        tx: watch::Sender<ReferenceState>,
    },
    Timer {
        version: u64,
        delay_ms: u64,
    },
}

/// Everything a background check needs, captured at trigger time.
pub(crate) struct CheckJob {
    meeting_id: Id,
    flag_id: Id,
    revision: u32,
    question: Option<IntegrityQuestion>,
    start_ms: i64,
    end_ms: i64,
    meeting_started_at: i64,
    text: String,
    words: u32,
    delivery: Option<IntegritySignal>,
    ai_text: bool,
    echo: bool,
    reading: bool,
    region_set: bool,
    gaze_running: bool,
    references: Option<watch::Receiver<ReferenceState>>,
    sidecar_embedding_model: Option<String>,
    created_at: i64,
    previous: Option<Id>,
}

// ---------------------------------------------------------------------------
// Conversation hooks (called with the state lock held)
// ---------------------------------------------------------------------------

/// End of the question, from its transcript segments.
fn question_end_ms(rt: &MeetingRuntime, q: &DetectedQuestion) -> i64 {
    rt.segments
        .iter()
        .filter(|s| q.segment_ids.contains(&s.id))
        .map(|s| s.end_time)
        .max()
        .unwrap_or_else(|| rt.now_rel())
}

/// The interviewer asked `question`: the previous answer is final, the
/// reference answers for the new question start.
pub(crate) fn on_interviewer_question(
    rt: &mut MeetingRuntime,
    settings: &Settings,
    question: &DetectedQuestion,
) -> Vec<IntegrityAction> {
    let mode = rt.mode();
    if !rt.integrity.active(settings, mode) {
        return Vec::new();
    }
    let mut actions: Vec<IntegrityAction> = finalize_answer(rt, settings)
        .map(|job| IntegrityAction::Check(Box::new(job)))
        .into_iter()
        .collect();
    let end_ms = question_end_ms(rt, question);
    rt.integrity.question = Some(IntegrityQuestion {
        id: question.id.clone(),
        text: question.question.clone(),
        end_ms,
    });
    if rt
        .integrity
        .signal_on(settings, mode, IntegritySignalName::EchoSimilarity)
    {
        let (tx, rx) = watch::channel(ReferenceState::Pending);
        let refs = &mut rt.integrity.references;
        refs.push_back((question.id.clone(), rx));
        while refs.len() > MAX_REFERENCE_QUESTIONS {
            refs.pop_front();
        }
        actions.push(IntegrityAction::References {
            question: question.question.clone(),
            tx,
        });
    }
    actions
}

/// A final utterance of the candidate joins the open answer.
pub(crate) fn on_candidate_segment(
    rt: &mut MeetingRuntime,
    settings: &Settings,
    segment: &TranscriptSegment,
    words: Option<Vec<WordTiming>>,
) -> Vec<IntegrityAction> {
    if !rt.integrity.active(settings, rt.mode()) {
        return Vec::new();
    }
    let question = rt.integrity.question.clone();
    let answer = rt.integrity.answer.get_or_insert_with(|| OpenAnswer {
        id: new_id(),
        question,
        units: Vec::new(),
        checked_units: 0,
        revision: 0,
        created_at: now_ms(),
        running: None,
    });
    answer.units.push(AnswerUnit {
        start_ms: segment.start_time,
        end_ms: segment.end_time,
        text: segment.text.clone(),
        words,
    });
    rt.integrity.version += 1;
    vec![IntegrityAction::Timer {
        version: rt.integrity.version,
        delay_ms: u64::from(settings.assistant.turn_end_silence_ms),
    }]
}

/// The candidate started speaking: a pending check timer is outdated.
pub(crate) fn on_candidate_speech_started(rt: &mut MeetingRuntime) {
    rt.integrity.version += 1;
}

/// The check timer fired: check the answer so far if the candidate paused.
pub(crate) fn on_timer(
    rt: &mut MeetingRuntime,
    settings: &Settings,
    version: u64,
) -> Option<CheckJob> {
    if rt.state() != MeetingState::Active
        || version != rt.integrity.version
        || !rt.integrity.active(settings, rt.mode())
    {
        return None;
    }
    let candidate_speaking = rt
        .speaking
        .iter()
        .any(|s| rt.role_for(*s) != SpeakerRole::User);
    if candidate_speaking {
        return None;
    }
    let mut answer = rt.integrity.answer.take()?;
    let job = (answer.units.len() > answer.checked_units)
        .then(|| prepare_check(rt, settings, &mut answer, false));
    rt.integrity.answer = Some(answer);
    job
}

/// Closes the open answer (next question, meeting end): a last check when
/// something was not checked yet, then the answer joins the baseline.
pub(crate) fn finalize_answer(rt: &mut MeetingRuntime, settings: &Settings) -> Option<CheckJob> {
    let mut answer = rt.integrity.answer.take()?;
    let job = (answer.units.len() > answer.checked_units)
        .then(|| prepare_check(rt, settings, &mut answer, true));
    if job.is_none() {
        // Already checked in full: just record it in the baseline.
        let features = answer_features(&answer);
        rt.integrity.delivery.commit(features.as_ref());
    }
    job
}

fn to_speech_units(answer: &OpenAnswer) -> Vec<SpeechUnit> {
    let s = |ms: i64| ms as f64 / 1000.0;
    answer
        .units
        .iter()
        .map(|u| SpeechUnit {
            start: s(u.start_ms),
            end: s(u.end_ms),
            text: u.text.clone(),
            words: u.words.as_ref().map(|ws| {
                ws.iter()
                    .map(|w| SpeechWord {
                        word: w.word.clone(),
                        start: s(w.start_ms),
                        end: s(w.end_ms),
                    })
                    .collect()
            }),
        })
        .collect()
}

fn answer_features(answer: &OpenAnswer) -> Option<openmeet_intel::integrity::DeliveryFeatures> {
    let question_end = answer.question.as_ref().map(|q| q.end_ms as f64 / 1000.0);
    delivery_features(question_end, &to_speech_units(answer))
}

/// Captures a check of `answer` (and, when `last`, records it in the
/// delivery baseline).
fn prepare_check(
    rt: &mut MeetingRuntime,
    settings: &Settings,
    answer: &mut OpenAnswer,
    last: bool,
) -> CheckJob {
    let mode = rt.mode();
    let th = spec_thresholds();
    answer.revision += 1;
    answer.checked_units = answer.units.len();
    let features = answer_features(answer);
    let baseline_answers = rt.integrity.delivery.baseline_answers();
    let delivery = rt
        .integrity
        .signal_on(settings, mode, IntegritySignalName::Delivery)
        .then(|| {
            let score = rt.integrity.delivery.score(features.as_ref(), &th.delivery);
            delivery_signal(
                features.as_ref(),
                &score,
                rt.integrity.delivery.baseline(),
                baseline_answers,
            )
        });
    if last {
        rt.integrity.delivery.commit(features.as_ref());
    }
    let text = answer
        .units
        .iter()
        .map(|u| u.text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let references = answer.question.as_ref().and_then(|q| {
        rt.integrity
            .references
            .iter()
            .find(|(id, _)| *id == q.id)
            .map(|(_, rx)| rx.clone())
    });
    CheckJob {
        meeting_id: rt.meeting.id.clone(),
        flag_id: answer.id.clone(),
        revision: answer.revision,
        question: answer.question.clone(),
        start_ms: answer.units.first().map_or(0, |u| u.start_ms),
        end_ms: answer.units.iter().map(|u| u.end_ms).max().unwrap_or(0),
        meeting_started_at: rt.meeting.started_at,
        words: openmeet_intel::text::word_count(&text) as u32,
        text,
        delivery,
        ai_text: rt
            .integrity
            .signal_on(settings, mode, IntegritySignalName::AiText),
        echo: rt
            .integrity
            .signal_on(settings, mode, IntegritySignalName::EchoSimilarity),
        reading: rt
            .integrity
            .signal_on(settings, mode, IntegritySignalName::Reading),
        region_set: rt.integrity.region(settings).is_some(),
        gaze_running: rt.integrity.gaze_running,
        references,
        sidecar_embedding_model: rt.integrity.sidecar_embedding_model.clone(),
        created_at: answer.created_at,
        previous: answer.running.take(),
    }
}

// ---------------------------------------------------------------------------
// Background work
// ---------------------------------------------------------------------------

impl Engine {
    /// Runs integrity actions decided under the state lock.
    pub(crate) fn run_integrity_actions(&self, meeting_id: &str, actions: Vec<IntegrityAction>) {
        for action in actions {
            match action {
                IntegrityAction::Check(job) => self.spawn_integrity_check(*job),
                IntegrityAction::References { question, tx } => self.spawn_references(question, tx),
                IntegrityAction::Timer { version, delay_ms } => {
                    let tx = {
                        let st = self.inner.state.lock();
                        st.runtime
                            .as_ref()
                            .filter(|r| r.meeting.id == meeting_id)
                            .map(|r| r.conv_tx.clone())
                    };
                    if let Some(tx) = tx {
                        self.inner.handle.spawn(async move {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                            let _ =
                                tx.send(crate::conversation::ConvInput::IntegrityTimer { version });
                        });
                    }
                }
            }
        }
    }

    /// The configured LLM answers the question `references` times.
    fn spawn_references(&self, question: String, tx: watch::Sender<ReferenceState>) {
        let Some(llm) = self.llm() else {
            let _ = tx.send(ReferenceState::Failed(
                "no AI provider is configured".to_string(),
            ));
            return;
        };
        let ticket = self.inner.requests.begin(
            RequestSlot::Background,
            RequestKind::IntegrityReference,
            |_| None,
        );
        let engine = self.clone();
        self.inner.handle.spawn(async move {
            let wanted = spec_thresholds().echo_similarity.references.max(1) as usize;
            engine
                .inner
                .requests
                .set_state(&ticket, AnswerState::Requesting, None);
            let mut refs = Vec::new();
            let mut last_error: Option<ProviderError> = None;
            for variant in 0..wanted {
                if ticket.is_cancelled() {
                    break;
                }
                match reference_answer(&ticket, llm.clone(), &question, variant).await {
                    Ok(text) if !text.trim().is_empty() => refs.push(text),
                    Ok(_) => {}
                    Err(e) => last_error = Some(e),
                }
            }
            let state = if ticket.is_cancelled() {
                ReferenceState::Failed("cancelled".to_string())
            } else if refs.is_empty() {
                ReferenceState::Failed(
                    last_error
                        .map(|e| e.user_message())
                        .unwrap_or_else(|| "the model returned no answer".to_string()),
                )
            } else {
                ReferenceState::Ready(refs)
            };
            let ok = matches!(state, ReferenceState::Ready(_));
            tracing::debug!(request_id = %ticket.id, ok, "integrity reference answers finished");
            let _ = tx.send(state);
            engine.inner.requests.set_state(
                &ticket,
                if ok {
                    AnswerState::Complete
                } else {
                    AnswerState::Failed
                },
                None,
            );
        });
    }

    fn spawn_integrity_check(&self, job: CheckJob) {
        if let Some(previous) = &job.previous {
            // A newer revision of the same answer replaces the running one.
            self.inner.requests.cancel(previous);
        }
        let ticket =
            self.inner
                .requests
                .begin(RequestSlot::Background, RequestKind::IntegrityCheck, |_| {
                    None
                });
        {
            let mut st = self.inner.state.lock();
            if let Some(answer) = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == job.meeting_id)
                .and_then(|r| r.integrity.answer.as_mut())
                .filter(|a| a.id == job.flag_id)
            {
                answer.running = Some(ticket.id.clone());
            }
        }
        let engine = self.clone();
        self.inner.handle.spawn(async move {
            engine.run_integrity_check(job, ticket).await;
        });
    }

    async fn run_integrity_check(&self, job: CheckJob, ticket: RequestTicket) {
        self.inner
            .requests
            .set_state(&ticket, AnswerState::Requesting, None);
        let providers = self.provider_set();
        let service = providers.integrity.clone();
        let llm_local = providers
            .llm
            .as_ref()
            .is_none_or(|l| l.info().location != ProviderLocation::Remote);
        let embedder = self.embedder();
        let th = spec_thresholds();
        let cancel = ticket.cancel.clone();

        let ai = async {
            if !job.ai_text {
                return None;
            }
            Some(ai_text_check(service.as_ref(), &job.text, &cancel, th.ai_text.calibrated).await)
        };
        let reading = async {
            if !job.reading {
                return None;
            }
            Some(reading_check(service.as_ref(), &job, &cancel, th.reading.calibrated).await)
        };
        let echo = async {
            if !job.echo {
                return None;
            }
            Some(
                self.echo_check(service.as_ref(), embedder, &job, llm_local, &cancel)
                    .await,
            )
        };
        let (ai, echo, reading) = tokio::join!(ai, echo, reading);
        if ticket.is_cancelled() {
            self.inner
                .requests
                .set_state(&ticket, AnswerState::Cancelled, None);
            return;
        }
        let signals: Vec<IntegritySignal> = [ai, echo, job.delivery.clone(), reading]
            .into_iter()
            .flatten()
            .collect();
        let fused = fuse(&fusion_inputs(&signals));
        let flag = IntegrityFlag {
            id: job.flag_id.clone(),
            meeting_id: job.meeting_id.clone(),
            question_id: job.question.as_ref().map(|q| q.id.clone()),
            question_text: job.question.as_ref().map(|q| q.text.clone()),
            answer_start_ms: job.start_ms,
            answer_end_ms: job.end_ms,
            answer_words: job.words,
            level: fused.level,
            elevated: fused.elevated,
            headline: headline(fused.level).to_string(),
            signals,
            revision: job.revision,
            spec_version: SPEC_VERSION,
            created_at: job.created_at,
            updated_at: now_ms(),
        };
        self.store_integrity_flag(flag);
        self.inner
            .requests
            .set_state(&ticket, AnswerState::Complete, None);
    }

    /// Keeps, persists and emits a flag unless a newer revision of the same
    /// answer is already known.
    fn store_integrity_flag(&self, flag: IntegrityFlag) {
        let store = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == flag.meeting_id)
            else {
                tracing::debug!(flag_id = %flag.id, "integrity check finished after its meeting closed");
                return;
            };
            let flags = &mut rt.integrity.flags;
            match flags.iter_mut().find(|f| f.id == flag.id) {
                Some(existing) if existing.revision > flag.revision => {
                    tracing::debug!(flag_id = %flag.id, "stale integrity revision dropped");
                    return;
                }
                Some(existing) => *existing = flag.clone(),
                None => flags.push(flag.clone()),
            }
            if let Some(a) = rt
                .integrity
                .answer
                .as_mut()
                .filter(|a| a.id == flag.id && a.revision == flag.revision)
            {
                a.running = None;
            }
            rt.store.clone()
        };
        tracing::info!(
            flag_id = %flag.id,
            revision = flag.revision,
            level = flag.level.as_str(),
            signals = flag.signals.len(),
            "answer integrity checked"
        );
        self.persist(&store, PersistOp::IntegrityFlag(flag.clone()));
        self.emit(MeetingEvent::IntegrityChecked { flag });
    }

    /// The echo-similarity signal of one answer.
    async fn echo_check(
        &self,
        service: Option<&Arc<dyn IntegrityService>>,
        embedder: Arc<dyn EmbeddingProvider>,
        job: &CheckJob,
        llm_local: bool,
        cancel: &CancellationToken,
    ) -> IntegritySignal {
        let th = spec_thresholds();
        let cal = th.echo_similarity.calibrated;
        let unavailable = |note: String| {
            not_scored(
                IntegritySignalName::EchoSimilarity,
                IntegrityLevel::Unavailable,
                cal,
                note,
            )
        };
        if job.question.is_none() {
            return unavailable("No interviewer question was detected for this answer".into());
        }
        let info = embedder.info();
        let mut model = info.model.clone();
        if info.id == "local_ai" && (model.is_empty() || model == "default") {
            // The sidecar's default embedding model: ask it once per meeting.
            let known = job.sidecar_embedding_model.clone();
            model = match known {
                Some(m) => m,
                None => match self
                    .sidecar_embedding_model(service, &job.meeting_id, cancel)
                    .await
                {
                    Some(m) => m,
                    None => model,
                },
            };
        }
        if !is_echo_embedding_model(&model, th) {
            return unavailable(format!(
                "Needs the {} embedding model (Settings → AI providers → Embeddings: Local AI); the current embedder is {}",
                th.echo_similarity.embedding_model,
                if model.is_empty() {
                    info.display_name
                } else {
                    model
                }
            ));
        }
        let Some(mut rx) = job.references.clone() else {
            return unavailable("No reference answers were requested for this question".into());
        };
        let settled = async {
            rx.wait_for(|s| *s != ReferenceState::Pending)
                .await
                .map(|state| state.clone())
        };
        let waited = tokio::select! {
            _ = cancel.cancelled() => return unavailable("Cancelled".into()),
            r = tokio::time::timeout(REFERENCE_WAIT, settled) => r,
        };
        let refs = match waited {
            Ok(Ok(state)) => match state {
                ReferenceState::Ready(refs) => refs,
                ReferenceState::Failed(e) => {
                    return unavailable(format!("Could not get reference answers: {e}"));
                }
                ReferenceState::Pending => Vec::new(),
            },
            Ok(Err(_)) => return unavailable("Reference answers were not produced".into()),
            Err(_) => return unavailable("Reference answers took too long".into()),
        };
        if refs.is_empty() {
            return unavailable("No reference answers to compare with".into());
        }
        let mut texts = vec![job.text.clone()];
        texts.extend(refs);
        let vectors = tokio::select! {
            _ = cancel.cancelled() => return unavailable("Cancelled".into()),
            r = tokio::time::timeout(SCORING_TIMEOUT, embedder.embed(texts.clone(), cancel.clone())) => r,
        };
        let vectors = match vectors {
            Ok(Ok(v)) if v.len() == texts.len() => v,
            Ok(Ok(_)) => {
                return unavailable("The embedder returned the wrong number of vectors".into());
            }
            Ok(Err(e)) => return unavailable(format!("Embedding failed: {}", e.user_message())),
            Err(_) => return unavailable("Embedding took too long".into()),
        };
        let answer = echo::widen(&vectors[0]);
        let references: Vec<Vec<f64>> = vectors[1..].iter().map(|v| echo::widen(v)).collect();
        echo_signal(
            &echo_similarity(&answer, &references, &th.echo_similarity),
            llm_local,
        )
    }

    /// The sidecar's embedding model, cached in the meeting runtime.
    async fn sidecar_embedding_model(
        &self,
        service: Option<&Arc<dyn IntegrityService>>,
        meeting_id: &str,
        cancel: &CancellationToken,
    ) -> Option<String> {
        let service = service?;
        let status = tokio::time::timeout(SCORING_TIMEOUT, service.status(cancel.clone()))
            .await
            .ok()?
            .ok()?;
        let model = status.embedding_model?;
        let mut st = self.inner.state.lock();
        if let Some(rt) = st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id) {
            rt.integrity.sidecar_embedding_model = Some(model.clone());
        }
        Some(model)
    }

    // ------------------------------------------------------------------
    // Gaze capture lifecycle
    // ------------------------------------------------------------------

    /// Starts or stops gaze capture of the candidate region so it runs
    /// exactly while an interviewer meeting with the reading signal is
    /// active (stopped on pause, emergency stop, meeting end, mode change and
    /// when the setting is turned off).
    pub(crate) fn sync_integrity_gaze(&self) {
        let settings = self.settings();
        let service = self.inner.providers.read().integrity.clone();
        let cmd = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st.runtime.as_mut() else {
                return;
            };
            let region = rt.integrity.region(&settings);
            let want = rt.state() == MeetingState::Active
                && rt
                    .integrity
                    .signal_on(&settings, rt.mode(), IntegritySignalName::Reading)
                && region.is_some()
                && service.is_some();
            match (want, rt.integrity.gaze_running, service, region) {
                (true, false, Some(service), Some(region)) => {
                    rt.integrity.gaze_running = true;
                    let reset = !rt.integrity.gaze_reset_done;
                    rt.integrity.gaze_reset_done = true;
                    Some(GazeCmd::Start {
                        service,
                        region,
                        reset,
                        meeting_id: rt.meeting.id.clone(),
                    })
                }
                (false, true, service, _) => {
                    rt.integrity.gaze_running = false;
                    service.map(|service| GazeCmd::Stop { service })
                }
                _ => None,
            }
        };
        if let Some(cmd) = cmd {
            let _ = self.inner.gaze_tx.send(cmd);
        }
    }

    /// After the meeting ended: forget the gaze kept by the sidecar once the
    /// last checks have read it.
    pub(crate) fn reset_integrity_gaze_after_checks(&self) {
        if let Some(service) = self.inner.providers.read().integrity.clone() {
            let _ = self
                .inner
                .gaze_tx
                .send(GazeCmd::ResetAfterChecks { service });
        }
    }

    /// Health of the integrity check at meeting start.
    pub(crate) fn integrity_meeting_started(&self) {
        let settings = self.settings();
        let has_service = self.inner.providers.read().integrity.is_some();
        let active = {
            let st = self.inner.state.lock();
            st.runtime
                .as_ref()
                .is_some_and(|rt| rt.integrity.active(&settings, rt.mode()))
        };
        if !active {
            self.inner
                .health
                .set(Component::Integrity, ComponentStatus::Disabled, None);
            return;
        }
        let (status, message) = if has_service {
            (
                ComponentStatus::Active,
                "Checking each answer in the background".to_string(),
            )
        } else {
            (
                ComponentStatus::Degraded,
                "Local AI is not available: only delivery and echo can be checked".to_string(),
            )
        };
        self.inner
            .health
            .set(Component::Integrity, status, Some(message));
        self.sync_integrity_gaze();
    }
}

/// One reference answer from the LLM.
async fn reference_answer(
    ticket: &RequestTicket,
    llm: Arc<dyn LlmProvider>,
    question: &str,
    variant: usize,
) -> Result<String, ProviderError> {
    let parts = integrity_reference::build(question, variant);
    let request = LlmRequest {
        request_id: ticket.id.clone(),
        kind: RequestKind::IntegrityReference,
        system: parts.system,
        messages: vec![ChatMessage::user(parts.user)],
        max_output_tokens: integrity_reference::MAX_OUTPUT_TOKENS,
        temperature: Some(0.7),
        json_output: false,
        latency_sensitive: false,
    };
    crate::assist::complete(ticket, llm, request, REFERENCE_TIMEOUT)
        .await
        .map(|t| integrity_reference::parse(&t))
}

fn service_error(what: &str, e: &ProviderError) -> String {
    match e {
        ProviderError::Network(_) | ProviderError::Timeout => {
            format!("{what} unavailable: Local AI did not answer")
        }
        ProviderError::Http(m) if m.starts_with("503") => {
            format!("{what} unavailable: the model is not installed in Local AI")
        }
        other => format!("{what} unavailable: {}", other.user_message()),
    }
}

async fn ai_text_check(
    service: Option<&Arc<dyn IntegrityService>>,
    text: &str,
    cancel: &CancellationToken,
    calibrated: bool,
) -> IntegritySignal {
    let unavailable = |note: String| {
        not_scored(
            IntegritySignalName::AiText,
            IntegrityLevel::Unavailable,
            calibrated,
            note,
        )
    };
    let Some(service) = service else {
        return unavailable("AI-written text check needs Local AI".into());
    };
    let result = tokio::select! {
        _ = cancel.cancelled() => return unavailable("Cancelled".into()),
        r = tokio::time::timeout(SCORING_TIMEOUT, service.ai_text(text.to_string(), cancel.clone())) => r,
    };
    match result {
        Ok(Ok(r)) => ai_text_signal(r.probability, r.level, calibrated),
        Ok(Err(e)) => unavailable(service_error("AI-written text check", &e)),
        Err(_) => unavailable("AI-written text check took too long".into()),
    }
}

async fn reading_check(
    service: Option<&Arc<dyn IntegrityService>>,
    job: &CheckJob,
    cancel: &CancellationToken,
    calibrated: bool,
) -> IntegritySignal {
    let unavailable = |note: String| {
        not_scored(
            IntegritySignalName::Reading,
            IntegrityLevel::Unavailable,
            calibrated,
            note,
        )
    };
    let Some(service) = service else {
        return unavailable("Reading check needs Local AI".into());
    };
    if !job.region_set {
        return unavailable(
            "Set the candidate's video region in Settings → Answer integrity".into(),
        );
    }
    if !job.gaze_running {
        return unavailable("Gaze capture was not running during this answer".into());
    }
    let epoch = |ms: i64| (job.meeting_started_at + ms) as f64 / 1000.0;
    let result = tokio::select! {
        _ = cancel.cancelled() => return unavailable("Cancelled".into()),
        r = tokio::time::timeout(
            SCORING_TIMEOUT,
            service.reading(epoch(job.start_ms), epoch(job.end_ms), cancel.clone()),
        ) => r,
    };
    match result {
        Ok(Ok(r)) => reading_signal(r.probability, r.level, calibrated),
        Ok(Err(e)) => unavailable(service_error("Reading check", &e)),
        Err(_) => unavailable("Reading check took too long".into()),
    }
}

// ---------------------------------------------------------------------------
// Gaze worker
// ---------------------------------------------------------------------------

/// Gaze commands, applied in order by one worker so a stop never overtakes
/// the start it belongs to.
pub(crate) enum GazeCmd {
    Start {
        service: Arc<dyn IntegrityService>,
        region: Rect,
        reset: bool,
        meeting_id: Id,
    },
    Stop {
        service: Arc<dyn IntegrityService>,
    },
    ResetAfterChecks {
        service: Arc<dyn IntegrityService>,
    },
}

/// Applies gaze commands sequentially (spawned with the engine).
pub(crate) fn spawn_gaze_worker(
    inner: Weak<Inner>,
    mut rx: mpsc::UnboundedReceiver<GazeCmd>,
    handle: &tokio::runtime::Handle,
) {
    handle.spawn(async move {
        while let Some(cmd) = rx.recv().await {
            let cancel = CancellationToken::new();
            match cmd {
                GazeCmd::Start {
                    service,
                    region,
                    reset,
                    meeting_id,
                } => {
                    if reset {
                        let _ = service.reset_gaze(cancel.clone()).await;
                    }
                    let result = service.start_gaze(region, cancel).await;
                    let Some(inner) = inner.upgrade() else { break };
                    let engine = Engine { inner };
                    match result {
                        Ok(status) => {
                            let message = match status.error {
                                Some(_) => "Gaze capture reported an error; reading may be unavailable",
                                None => "Watching the candidate's video region for reading",
                            };
                            engine.inner.health.set(
                                Component::Integrity,
                                ComponentStatus::Active,
                                Some(message.to_string()),
                            );
                        }
                        Err(e) => {
                            tracing::warn!(error_kind = ?std::mem::discriminant(&e), "gaze capture could not start");
                            {
                                let mut st = engine.inner.state.lock();
                                if let Some(rt) =
                                    st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id)
                                {
                                    rt.integrity.gaze_running = false;
                                }
                            }
                            engine.inner.health.set(
                                Component::Integrity,
                                ComponentStatus::Degraded,
                                Some(service_error("Gaze capture", &e)),
                            );
                        }
                    }
                }
                GazeCmd::Stop { service } => {
                    let _ = service.stop_gaze(cancel).await;
                }
                GazeCmd::ResetAfterChecks { service } => {
                    let deadline = tokio::time::Instant::now() + GAZE_RESET_WAIT;
                    loop {
                        let busy = match inner.upgrade() {
                            Some(i) => i.requests.background_running(RequestKind::IntegrityCheck),
                            None => false,
                        };
                        if !busy || tokio::time::Instant::now() >= deadline {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                    let _ = service.reset_gaze(cancel).await;
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_follows_mode_settings_and_demo() {
        let mut s = Settings::default();
        let mut run = IntegrityRun::default();
        assert!(!run.active(&s, Mode::Interviewer), "off by default");
        s.integrity.enabled = true;
        assert!(run.active(&s, Mode::Interviewer));
        assert!(!run.active(&s, Mode::Candidate), "interviewer mode only");
        s.integrity.reading = false;
        assert!(!run.signal_on(&s, Mode::Interviewer, IntegritySignalName::Reading));
        assert!(run.signal_on(&s, Mode::Interviewer, IntegritySignalName::AiText));
        s.integrity.enabled = false;
        run.forced = true;
        assert!(run.signal_on(&s, Mode::Interviewer, IntegritySignalName::Reading));
        assert_eq!(run.region(&s), None);
        run.demo_region = Some(DEMO_REGION);
        assert_eq!(run.region(&s), Some(DEMO_REGION));
    }
}
