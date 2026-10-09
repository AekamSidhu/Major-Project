//! Per-meeting conversation loop: turns transcription events into transcript
//! segments, runs question detection, tracks answers and interviewer turns,
//! and triggers assistant actions (spec §17, §20, §22).

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use openmeet_core::events::{MeetingEvent, NoticeLevel};
use openmeet_core::ids::{Id, new_id, now_ms};
use openmeet_core::providers::TranscriptionEvent;
use openmeet_core::types::*;
use openmeet_intel::claims::{extract_action_items, extract_claims};
use openmeet_intel::detect::voice::parse_voice_command;
use openmeet_intel::detect::{DetectionInput, DetectionResult, detect};
use openmeet_intel::prompts::TranscriptLine;
use openmeet_intel::text::{estimate_tokens, word_count};
use tokio::sync::mpsc;

use crate::embedder::EmbedJob;
use crate::engine::{Engine, Inner};
use crate::memory::{MemoryItem, MemoryKind};
use crate::metrics::record_latency;
use crate::persist::PersistOp;
use crate::runtime::MeetingRuntime;

/// Tokens of old (outside the raw window) transcript that trigger compression.
const COMPRESSION_TRIGGER_TOKENS: usize = 900;
/// Minimum spacing of LLM topic analysis.
const TOPIC_ANALYSIS_INTERVAL_MS: i64 = 180_000;
const TOPIC_ANALYSIS_MIN_FINALS: u32 = 12;
/// Words of recent context passed to the detector.
const DETECTION_CONTEXT_SEGMENTS: usize = 8;

/// Inputs into a meeting's conversation loop besides transcription events.
#[derive(Debug, Clone)]
pub enum ConvInput {
    /// Injected transcript (developer simulation, demo mode).
    Draft(TranscriptDraft),
    /// Interviewer mode: the turn-end timer for `version` fired.
    TurnTimer { version: u64 },
}

/// Work decided while holding the state lock, executed after releasing it.
enum Action {
    Answer(DetectedQuestion),
    Semantic {
        segment: TranscriptSegment,
        heuristic: DetectionResult,
        recent: Vec<TranscriptLine>,
    },
    Voice(VoiceCommand),
    Compress {
        lines: Vec<TranscriptLine>,
        until_ms: i64,
        previous: String,
    },
    TopicAnalysis {
        lines: Vec<TranscriptLine>,
        topics: Vec<Topic>,
    },
    TurnTimer {
        version: u64,
        delay_ms: u64,
    },
}

pub(crate) fn spawn_conversation(
    engine: Engine,
    meeting_id: Id,
    mut trans_rx: mpsc::UnboundedReceiver<TranscriptionEvent>,
    mut conv_rx: mpsc::UnboundedReceiver<ConvInput>,
) {
    let handle = engine.inner.handle.clone();
    let weak: Weak<Inner> = Arc::downgrade(&engine.inner);
    drop(engine);
    handle.spawn(async move {
        let mut trans_open = true;
        let mut conv_open = true;
        while trans_open || conv_open {
            tokio::select! {
                ev = trans_rx.recv(), if trans_open => match ev {
                    Some(ev) => {
                        let Some(inner) = weak.upgrade() else { break };
                        Engine { inner }.on_transcription_event(&meeting_id, ev);
                    }
                    None => trans_open = false,
                },
                input = conv_rx.recv(), if conv_open => match input {
                    Some(input) => {
                        let Some(inner) = weak.upgrade() else { break };
                        Engine { inner }.on_conv_input(&meeting_id, input);
                    }
                    None => conv_open = false,
                },
            }
        }
        tracing::debug!(meeting = %meeting_id, "conversation loop finished");
    });
}

fn speaker_id(role: SpeakerRole) -> &'static str {
    match role {
        SpeakerRole::User => "user",
        SpeakerRole::Remote => "remote",
        SpeakerRole::Unknown => "speaker",
    }
}

fn segment_from_draft(rt: &MeetingRuntime, d: &TranscriptDraft) -> TranscriptSegment {
    let role = d.speaker_role.unwrap_or_else(|| rt.role_for(d.source));
    let start = rt.rel_ms(d.start_ms);
    TranscriptSegment {
        id: if d.is_final {
            new_id()
        } else {
            format!("partial-{}", d.utterance_id)
        },
        meeting_id: rt.meeting.id.clone(),
        utterance_id: d.utterance_id.clone(),
        speaker_id: Some(speaker_id(role).to_string()),
        speaker_role: role,
        source: d.source,
        start_time: start,
        end_time: rt.rel_ms(d.end_ms).max(start),
        text: d.text.trim().to_string(),
        is_final: d.is_final,
        confidence: d.confidence,
    }
}

fn question_from(
    rt: &MeetingRuntime,
    segment: &TranscriptSegment,
    r: &DetectionResult,
    method: DetectionMethod,
) -> DetectedQuestion {
    DetectedQuestion {
        id: new_id(),
        meeting_id: rt.meeting.id.clone(),
        segment_ids: if r.segment_ids.is_empty() {
            vec![segment.id.clone()]
        } else {
            r.segment_ids.clone()
        },
        is_question: true,
        confidence: r.confidence,
        question: if r.question.trim().is_empty() {
            segment.text.clone()
        } else {
            r.question.clone()
        },
        parts: r.parts.clone(),
        category: r.category,
        speaker_id: segment.speaker_id.clone(),
        speaker_role: segment.speaker_role,
        requires_response: r.requires_response,
        method,
        supersedes: r.supersedes.clone(),
        detected_at: now_ms(),
    }
}

impl Engine {
    pub(crate) fn on_transcription_event(&self, meeting_id: &str, ev: TranscriptionEvent) {
        match ev {
            TranscriptionEvent::Level { source, level } => {
                let emit = {
                    let mut st = self.inner.state.lock();
                    match st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id) {
                        Some(rt) => {
                            let now = now_ms();
                            let last = rt.last_level_emit.get(&source).copied().unwrap_or(0);
                            if now - last >= 100 {
                                rt.last_level_emit.insert(source, now);
                                true
                            } else {
                                false
                            }
                        }
                        None => false,
                    }
                };
                if emit {
                    self.emit(MeetingEvent::AudioLevel { source, level });
                }
            }
            TranscriptionEvent::SpeechStarted { source, .. } => {
                let mut st = self.inner.state.lock();
                if let Some(rt) = st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id) {
                    rt.speaking.insert(source);
                    if rt.mode() == Mode::Interviewer && rt.role_for(source) != SpeakerRole::User {
                        // The candidate keeps talking: invalidate the turn-end timer.
                        rt.turn.version += 1;
                    }
                }
            }
            TranscriptionEvent::SpeechEnded {
                source,
                utterance_id,
                discarded,
                ..
            } => {
                let action = {
                    let mut st = self.inner.state.lock();
                    let settings = self.inner.settings.read();
                    st.runtime
                        .as_mut()
                        .filter(|r| r.meeting.id == meeting_id)
                        .and_then(|rt| {
                            rt.speaking.remove(&source);
                            if !discarded {
                                rt.pending_utterances.insert(utterance_id);
                                None
                            } else {
                                reschedule_turn(rt, source, settings.assistant.turn_end_silence_ms)
                            }
                        })
                };
                if let Some(a) = action {
                    self.run_actions(meeting_id, vec![a]);
                }
            }
            TranscriptionEvent::Draft(d) => {
                if d.is_final {
                    self.handle_final(meeting_id, d, true);
                } else {
                    self.handle_partial(meeting_id, d);
                }
            }
            TranscriptionEvent::Error {
                source,
                utterance_id,
                error,
            } => {
                let (notify, action) = {
                    let mut st = self.inner.state.lock();
                    let settings = self.inner.settings.read();
                    match st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id) {
                        Some(rt) => {
                            if let Some(id) = &utterance_id {
                                rt.pending_utterances.remove(id);
                            }
                            let now = now_ms();
                            let notify = now - rt.last_transcription_notice > 30_000;
                            if notify {
                                rt.last_transcription_notice = now;
                            }
                            (
                                notify,
                                reschedule_turn(rt, source, settings.assistant.turn_end_silence_ms),
                            )
                        }
                        None => (false, None),
                    }
                };
                self.inner.health.set(
                    Component::Transcription,
                    ComponentStatus::Degraded,
                    Some(error.user_message()),
                );
                if notify {
                    self.notice(
                        NoticeLevel::Warning,
                        format!("Transcription problem: {}", error.user_message()),
                        Some(Component::Transcription),
                    );
                }
                if let Some(a) = action {
                    self.run_actions(meeting_id, vec![a]);
                }
            }
            TranscriptionEvent::Empty {
                source,
                utterance_id,
            } => {
                let action = {
                    let mut st = self.inner.state.lock();
                    let settings = self.inner.settings.read();
                    st.runtime
                        .as_mut()
                        .filter(|r| r.meeting.id == meeting_id)
                        .and_then(|rt| {
                            rt.pending_utterances.remove(&utterance_id);
                            reschedule_turn(rt, source, settings.assistant.turn_end_silence_ms)
                        })
                };
                if let Some(a) = action {
                    self.run_actions(meeting_id, vec![a]);
                }
            }
        }
    }

    pub(crate) fn on_conv_input(&self, meeting_id: &str, input: ConvInput) {
        match input {
            ConvInput::Draft(d) => {
                if d.is_final {
                    self.handle_final(meeting_id, d, false);
                } else {
                    self.handle_partial(meeting_id, d);
                }
            }
            ConvInput::TurnTimer { version } => self.on_turn_timer(meeting_id, version),
        }
    }

    fn handle_partial(&self, meeting_id: &str, d: TranscriptDraft) {
        let segment = {
            let st = self.inner.state.lock();
            st.runtime
                .as_ref()
                .filter(|r| r.meeting.id == meeting_id && r.accepts_transcript())
                .map(|rt| segment_from_draft(rt, &d))
        };
        if let Some(segment) = segment.filter(|s| !s.text.is_empty()) {
            self.emit(MeetingEvent::TranscriptPartial { segment });
        }
    }

    /// Core of the real-time pipeline for one finalized utterance.
    fn handle_final(&self, meeting_id: &str, d: TranscriptDraft, measured: bool) {
        let received = Instant::now();
        let settings = self.settings();
        let mut events: Vec<MeetingEvent> = Vec::new();
        let mut actions: Vec<Action> = Vec::new();
        let mut embed_jobs: Vec<EmbedJob> = Vec::new();
        let mut persist_ops: Vec<PersistOp> = Vec::new();
        let mut detection_ms: Option<f64> = None;

        let store = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id && r.accepts_transcript())
            else {
                return;
            };
            rt.pending_utterances.remove(&d.utterance_id);
            let segment = segment_from_draft(rt, &d);
            if segment.text.is_empty() {
                return;
            }
            let active = rt.state() == MeetingState::Active;
            let mode = rt.mode();
            let role = segment.speaker_role;

            // Detection context: the finals preceding this one.
            let ctx_start = rt.segments.len().saturating_sub(DETECTION_CONTEXT_SEGMENTS);
            let recent: Vec<TranscriptSegment> = rt.segments[ctx_start..].to_vec();
            rt.add_final(segment.clone());
            events.push(MeetingEvent::TranscriptFinal {
                segment: segment.clone(),
            });
            persist_ops.push(PersistOp::Segment(segment.clone()));

            // Memory for retrieval.
            let label = rt.speaker_label(role);
            let item_text = format!("{label}: {}", segment.text);
            rt.memory.add(MemoryItem {
                id: segment.id.clone(),
                kind: MemoryKind::Segment,
                text: item_text.clone(),
                at_ms: segment.start_time,
            });
            embed_jobs.push(EmbedJob {
                meeting_id: meeting_id.to_string(),
                item_id: segment.id.clone(),
                kind: MemoryKind::Segment,
                text: item_text,
            });

            // Optional voice commands from the user's own microphone.
            let voice = (settings.general.voice_commands && role != SpeakerRole::Remote)
                .then(|| parse_voice_command(&segment.text))
                .flatten();
            if let Some(cmd) = voice {
                events.push(MeetingEvent::VoiceCommandRecognized {
                    command: cmd.clone(),
                });
                if active {
                    actions.push(Action::Voice(cmd));
                }
            } else {
                // Claims, action items, topics.
                let claims = extract_claims(&segment);
                if !claims.is_empty() {
                    rt.claims.extend(claims.iter().cloned());
                    persist_ops.push(PersistOp::Claims(claims.clone()));
                    for c in &claims {
                        rt.memory.add(MemoryItem {
                            id: c.id.clone(),
                            kind: MemoryKind::Claim,
                            text: c.text.clone(),
                            at_ms: segment.start_time,
                        });
                    }
                    events.push(MeetingEvent::ClaimsExtracted { claims });
                }
                let items = extract_action_items(&segment);
                if !items.is_empty() {
                    rt.action_items.extend(items.iter().cloned());
                    persist_ops.push(PersistOp::ActionItems(items));
                    events.push(MeetingEvent::ActionItemsUpdated {
                        items: rt.action_items.clone(),
                    });
                }
                if rt.topics.observe(&segment) {
                    let topics = rt.topics.topics();
                    persist_ops.push(PersistOp::Topics {
                        meeting_id: meeting_id.to_string(),
                        topics: topics.clone(),
                    });
                    events.push(MeetingEvent::TopicsUpdated {
                        topics,
                        suggested_next: rt.topics.suggested_next(),
                    });
                }

                // Question detection (spec §17).
                let result = detect(&DetectionInput {
                    segment: &segment,
                    recent: &recent,
                    mode,
                    last_question: rt.questions.last(),
                });
                // Final transcript received -> detection decided.
                detection_ms = Some(received.elapsed().as_secs_f64() * 1000.0);
                let threshold = settings.assistant.detection_threshold;
                let mut is_accepted_question = false;

                if result.is_question {
                    let asker_is_user = role == SpeakerRole::User;
                    let accept = match mode {
                        Mode::Candidate => result.confidence >= threshold,
                        Mode::Interviewer if asker_is_user => result.confidence >= threshold * 0.85,
                        Mode::Interviewer => {
                            result.requires_response && result.confidence >= threshold
                        }
                    };
                    if accept {
                        is_accepted_question = true;
                        let q = question_from(rt, &segment, &result, DetectionMethod::Heuristic);
                        rt.questions.push(q.clone());
                        persist_ops.push(PersistOp::Question(q.clone()));
                        rt.memory.add(MemoryItem {
                            id: q.id.clone(),
                            kind: MemoryKind::Question,
                            text: format!("{label} asked: {}", q.question),
                            at_ms: segment.start_time,
                        });
                        embed_jobs.push(EmbedJob {
                            meeting_id: meeting_id.to_string(),
                            item_id: q.id.clone(),
                            kind: MemoryKind::Question,
                            text: q.question.clone(),
                        });
                        events.push(MeetingEvent::QuestionDetected {
                            question: q.clone(),
                        });

                        // Answer tracking: the next speech of the other side
                        // answers this question.
                        let answers_question = match mode {
                            Mode::Candidate => role != SpeakerRole::User,
                            Mode::Interviewer => asker_is_user || role == SpeakerRole::Unknown,
                        };
                        if answers_question {
                            if let Some(prev) = rt.open_answer_for(&q) {
                                remember_answer(rt, &prev, &mut persist_ops, &mut embed_jobs);
                            }
                        }
                        if mode == Mode::Interviewer && asker_is_user {
                            // A new interviewer question starts a new candidate turn.
                            rt.turn.reset();
                        }
                        if active
                            && mode == Mode::Candidate
                            && q.requires_response
                            && settings.assistant.auto_answer
                        {
                            actions.push(Action::Answer(q));
                        }
                    } else if active
                        && mode == Mode::Candidate
                        && result.requires_response
                        && result.ambiguous
                        && settings.assistant.semantic_detection
                    {
                        let lines = recent.iter().map(|s| rt.line_for(s)).collect();
                        actions.push(Action::Semantic {
                            segment: segment.clone(),
                            heuristic: result.clone(),
                            recent: lines,
                        });
                    }
                } else if active
                    && mode == Mode::Candidate
                    && role != SpeakerRole::User
                    && result.ambiguous
                    && settings.assistant.semantic_detection
                {
                    let lines = recent.iter().map(|s| rt.line_for(s)).collect();
                    actions.push(Action::Semantic {
                        segment: segment.clone(),
                        heuristic: result.clone(),
                        recent: lines,
                    });
                }

                if !is_accepted_question {
                    rt.extend_answer(&segment);
                }

                // Interviewer mode: accumulate the candidate's turn.
                if mode == Mode::Interviewer
                    && role != SpeakerRole::User
                    && !(is_accepted_question && role == SpeakerRole::Unknown)
                {
                    rt.turn.push(&segment);
                    if active {
                        actions.push(Action::TurnTimer {
                            version: rt.turn.version,
                            delay_ms: settings.assistant.turn_end_silence_ms as u64,
                        });
                    }
                }
            }

            // Background context compression (spec §52).
            if active {
                if let Some(a) = compression_due(rt, &settings) {
                    actions.push(a);
                }
                rt.finals_since_topic_analysis += 1;
                let now = now_ms();
                if rt.finals_since_topic_analysis >= TOPIC_ANALYSIS_MIN_FINALS
                    && now - rt.last_topic_analysis_ms >= TOPIC_ANALYSIS_INTERVAL_MS
                {
                    rt.finals_since_topic_analysis = 0;
                    rt.last_topic_analysis_ms = now;
                    let since = rt.now_rel() - 300_000;
                    actions.push(Action::TopicAnalysis {
                        lines: rt.lines_since(since),
                        topics: rt.topics.topics(),
                    });
                }
            }
            rt.store.clone()
        };

        if measured {
            let audio_to_text = (now_ms() - d.end_ms) as f64;
            if (0.0..120_000.0).contains(&audio_to_text) {
                record_latency(
                    &self.inner.latency,
                    &self.inner.sink,
                    LatencyStage::AudioToTranscript,
                    audio_to_text,
                    None,
                );
            }
        }
        if let Some(ms) = detection_ms {
            record_latency(
                &self.inner.latency,
                &self.inner.sink,
                LatencyStage::TranscriptToDetection,
                ms,
                None,
            );
        }
        for op in persist_ops {
            self.persist(&store, op);
        }
        for ev in events {
            self.emit(ev);
        }
        for job in embed_jobs {
            let _ = self.inner.embed_tx.send(job);
        }
        self.run_actions(meeting_id, actions);
    }

    fn on_turn_timer(&self, meeting_id: &str, version: u64) {
        let settings = self.settings();
        let turn_text = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id && r.state() == MeetingState::Active)
            else {
                return;
            };
            if rt.turn.version != version || rt.mode() != Mode::Interviewer {
                return;
            }
            let remote_speaking = rt
                .speaking
                .iter()
                .any(|s| rt.role_for(*s) != SpeakerRole::User);
            if remote_speaking || rt.turn.text.trim().is_empty() {
                return;
            }
            if word_count(&rt.turn.text) < settings.assistant.followup_min_words as usize {
                return;
            }
            let text = rt.turn.text.clone();
            rt.last_turn_text = text.clone();
            rt.turn.reset();
            text
        };
        if settings.assistant.auto_followups {
            if let Err(e) = crate::assist::start_followups(self, turn_text) {
                tracing::debug!(error = %e, "automatic follow-ups not started");
            }
        }
    }

    fn run_actions(&self, meeting_id: &str, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Answer(q) => {
                    let style = self.style();
                    if let Err(e) = crate::assist::answer_question(self, &q, style) {
                        tracing::debug!(error = %e, "automatic answer not started");
                    }
                }
                Action::Semantic {
                    segment,
                    heuristic,
                    recent,
                } => crate::assist::spawn_semantic_check(
                    self.clone(),
                    meeting_id.to_string(),
                    segment,
                    heuristic,
                    recent,
                ),
                Action::Voice(cmd) => self.execute_voice_command(cmd),
                Action::Compress {
                    lines,
                    until_ms,
                    previous,
                } => crate::assist::spawn_compression(
                    self.clone(),
                    meeting_id.to_string(),
                    previous,
                    lines,
                    until_ms,
                ),
                Action::TopicAnalysis { lines, topics } => crate::assist::spawn_topic_analysis(
                    self.clone(),
                    meeting_id.to_string(),
                    lines,
                    topics,
                ),
                Action::TurnTimer { version, delay_ms } => {
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
                            let _ = tx.send(ConvInput::TurnTimer { version });
                        });
                    }
                }
            }
        }
    }

    fn execute_voice_command(&self, cmd: VoiceCommand) {
        let latest_primary = {
            let st = self.inner.state.lock();
            st.runtime
                .as_ref()
                .and_then(|r| r.active_primary.as_ref().map(|a| a.request_id.clone()))
        };
        let result = match cmd {
            VoiceCommand::Explain => self.ask(
                "Explain the latest question or what is currently on screen, simply and briefly."
                    .to_string(),
            ),
            VoiceCommand::FollowUp => self.generate_followups(),
            VoiceCommand::Answer => self.generate_answer(Default::default()),
            VoiceCommand::Concise => match latest_primary {
                Some(id) => self.refine(&id, Refinement::Shorter),
                None => self.generate_answer(openmeet_core::ipc::GenerateAnswerRequest {
                    question: None,
                    style: Some(AnswerStyle::Concise),
                }),
            },
            VoiceCommand::Detailed => match latest_primary {
                Some(id) => self.refine(&id, Refinement::Longer),
                None => self.generate_answer(openmeet_core::ipc::GenerateAnswerRequest {
                    question: None,
                    style: Some(AnswerStyle::Detailed),
                }),
            },
            VoiceCommand::Ask(text) => self.ask(text),
        };
        if let Err(e) = result {
            self.notice(NoticeLevel::Warning, e.to_string(), None);
        }
    }

    /// Injects a transcript line (simulation / demo).
    pub(crate) fn inject_draft(
        &self,
        draft: TranscriptDraft,
    ) -> Result<(), openmeet_core::CoreError> {
        let tx = {
            let st = self.inner.state.lock();
            st.runtime
                .as_ref()
                .filter(|r| r.accepts_transcript())
                .map(|r| r.conv_tx.clone())
        };
        let tx = tx.ok_or_else(|| {
            openmeet_core::CoreError::InvalidState("Start a meeting first.".into())
        })?;
        tx.send(ConvInput::Draft(draft))
            .map_err(|_| openmeet_core::CoreError::Internal("conversation loop stopped".into()))
    }
}

/// When remote speech ends without producing text, re-arm the turn timer so
/// follow-ups still fire for the text collected so far.
fn reschedule_turn(
    rt: &mut MeetingRuntime,
    source: AudioSourceKind,
    delay_ms: u32,
) -> Option<Action> {
    (rt.mode() == Mode::Interviewer
        && rt.state() == MeetingState::Active
        && rt.role_for(source) != SpeakerRole::User
        && !rt.turn.text.is_empty())
    .then(|| {
        rt.turn.version += 1;
        Action::TurnTimer {
            version: rt.turn.version,
            delay_ms: delay_ms as u64,
        }
    })
}

pub(crate) fn remember_answer(
    rt: &mut MeetingRuntime,
    answer: &AnswerRecord,
    persist_ops: &mut Vec<PersistOp>,
    embed_jobs: &mut Vec<EmbedJob>,
) {
    persist_ops.push(PersistOp::Answer(answer.clone()));
    let label = rt.speaker_label(answer.speaker_role);
    rt.memory.add(MemoryItem {
        id: answer.id.clone(),
        kind: MemoryKind::Answer,
        text: format!("{label} answered: {}", answer.text),
        at_ms: rt.rel_ms(answer.created_at),
    });
    embed_jobs.push(EmbedJob {
        meeting_id: rt.meeting.id.clone(),
        item_id: answer.id.clone(),
        kind: MemoryKind::Answer,
        text: answer.text.clone(),
    });
}

fn compression_due(
    rt: &MeetingRuntime,
    settings: &openmeet_core::settings::Settings,
) -> Option<Action> {
    let window_start = rt.now_rel() - settings.assistant.context_window_secs as i64 * 1000;
    let old: Vec<&TranscriptSegment> = rt
        .segments
        .iter()
        .filter(|s| s.end_time < window_start && s.start_time >= rt.memory.summarized_until_ms)
        .collect();
    let tokens: usize = old.iter().map(|s| estimate_tokens(&s.text)).sum();
    if tokens < COMPRESSION_TRIGGER_TOKENS {
        return None;
    }
    let until_ms = old.iter().map(|s| s.end_time).max()? + 1;
    Some(Action::Compress {
        lines: old.iter().map(|s| rt.line_for(s)).collect(),
        until_ms,
        previous: rt.memory.long_term_summary.clone(),
    })
}
