//! AI flows: candidate answers, interviewer follow-ups, Ask OpenMeet,
//! refinements, semantic question detection, context compression, topic
//! analysis and post-meeting summaries (spec §20–§24, §31, §34, §41, §52).

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openmeet_core::CoreError;
use openmeet_core::error::ProviderError;
use openmeet_core::events::{MeetingEvent, NoticeLevel};
use openmeet_core::ids::{Id, new_id, now_ms};
use openmeet_core::ipc::{ActiveRequestView, GenerateAnswerRequest};
use openmeet_core::providers::{ChatMessage, LlmChunk, LlmProvider, LlmRequest};
use openmeet_core::types::*;
use openmeet_intel::claims::extract_claims;
use openmeet_intel::detect::{DetectionResult, classify_category, has_deictic_reference};
use openmeet_intel::export::extractive_summary;
use openmeet_intel::prompts::interviewer_followup::{FollowupItem, FollowupStreamParser};
use openmeet_intel::prompts::{
    PromptContext, PromptParts, TranscriptLine, ask as ask_prompt, candidate_answer,
    candidate_clarification, context_compression, interviewer_followup, meeting_summary,
    question_detection, refine as refine_prompt, topic_analysis,
};
use openmeet_intel::text::{cosine, lexical_similarity, tail_to_tokens, truncate_to_tokens};
use tokio_util::sync::CancellationToken;

use crate::conversation::remember_answer;
use crate::embedder::EmbedJob;
use crate::engine::Engine;
use crate::memory::{MemoryItem, MemoryKind, RetrievalQuery};
use crate::metrics::record_latency;
use crate::persist::PersistOp;
use crate::requests::RequestTicket;

/// Time allowed for the provider to accept a request.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum silence between streamed chunks (first token / later tokens).
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(45);
const CHUNK_TIMEOUT: Duration = Duration::from_secs(30);
/// Budget for embedding the retrieval query with a remote embedding model.
const QUERY_EMBED_TIMEOUT: Duration = Duration::from_millis(300);
const MAX_FOLLOWUPS: usize = 4;
/// Lexical similarity at or above which a follow-up repeats an earlier question.
const DUPLICATE_LEXICAL: f32 = 0.6;
/// Hashing-embedding cosine at or above which a follow-up is a duplicate.
const DUPLICATE_COSINE: f32 = 0.82;
const MAX_ASK_CHARS: usize = 4000;

const CANDIDATE_QUESTIONS_PROMPT: &str = "Suggest three thoughtful, specific questions I could ask the interviewer next, grounded in what has been discussed so far. Avoid anything already answered.";

// ---------------------------------------------------------------------------
// Public entry points (called through Engine methods)
// ---------------------------------------------------------------------------

enum Builder {
    Answer {
        question: String,
        category: QuestionCategory,
        style: AnswerStyle,
        length: ResponseLength,
    },
    Clarify {
        question: String,
    },
    Ask {
        prompt: String,
    },
    Refine {
        question: Option<String>,
        previous: String,
        refinement: Refinement,
        style: AnswerStyle,
    },
}

impl Builder {
    fn build(&self, ctx: &PromptContext, budget: usize) -> PromptParts {
        match self {
            Builder::Answer {
                question,
                category,
                style,
                length,
            } => candidate_answer::build(ctx, question, *category, *style, *length, budget),
            Builder::Clarify { question } => candidate_clarification::build(ctx, question, budget),
            Builder::Ask { prompt } => ask_prompt::build(ctx, prompt, budget),
            Builder::Refine {
                question,
                previous,
                refinement,
                style,
            } => refine_prompt::build(
                ctx,
                question.as_deref(),
                previous,
                *refinement,
                *style,
                budget,
            ),
        }
    }
}

struct PrimarySpec {
    kind: RequestKind,
    suggestion_kind: SuggestionKind,
    question_id: Option<Id>,
    question_text: Option<String>,
    category: Option<QuestionCategory>,
    style: Option<AnswerStyle>,
    parent_id: Option<Id>,
    /// Text used for retrieval.
    query: String,
    /// Unix ms the triggering question was detected (latency anchor).
    detected_at: Option<i64>,
    builder: Builder,
}

pub(crate) fn generate_answer(
    engine: &Engine,
    req: GenerateAnswerRequest,
) -> Result<Id, CoreError> {
    let style = req.style.unwrap_or_else(|| engine.style());
    if let Some(q) = req
        .question
        .map(|q| q.trim().to_string())
        .filter(|q| !q.is_empty())
    {
        let category = classify_category(&q);
        return start_primary(
            engine,
            answer_spec(
                None,
                q.clone(),
                category,
                style,
                engine.settings().assistant.response_length,
                None,
            ),
        );
    }
    let (latest, recent_remote) = {
        let st = engine.inner.state.lock();
        match st.runtime.as_ref() {
            Some(rt) => (
                rt.latest_open_question(5 * 60_000).cloned(),
                rt.recent_remote_text(60_000),
            ),
            None => (None, String::new()),
        }
    };
    if let Some(q) = latest {
        return answer_question(engine, &q, style);
    }
    if !recent_remote.trim().is_empty() {
        let text = truncate_to_tokens(&recent_remote, 300);
        let category = classify_category(&text);
        return start_primary(
            engine,
            answer_spec(
                None,
                text,
                category,
                style,
                engine.settings().assistant.response_length,
                None,
            ),
        );
    }
    Err(CoreError::InvalidState(
        "No question detected yet. Use Ask OpenMeet to type a question.".into(),
    ))
}

fn answer_spec(
    question_id: Option<Id>,
    question: String,
    category: QuestionCategory,
    style: AnswerStyle,
    length: ResponseLength,
    detected_at: Option<i64>,
) -> PrimarySpec {
    PrimarySpec {
        kind: RequestKind::CandidateAnswer,
        suggestion_kind: SuggestionKind::Answer,
        question_id,
        question_text: Some(question.clone()),
        category: Some(category),
        style: Some(style),
        parent_id: None,
        query: question.clone(),
        detected_at,
        builder: Builder::Answer {
            question,
            category,
            style,
            length,
        },
    }
}

pub(crate) fn answer_question(
    engine: &Engine,
    q: &DetectedQuestion,
    style: AnswerStyle,
) -> Result<Id, CoreError> {
    let length = engine.settings().assistant.response_length;
    start_primary(
        engine,
        answer_spec(
            Some(q.id.clone()),
            q.question.clone(),
            q.category,
            style,
            length,
            Some(q.detected_at),
        ),
    )
}

pub(crate) fn ask(engine: &Engine, prompt: String) -> Result<Id, CoreError> {
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return Err(CoreError::InvalidInput(
            "Type a question for OpenMeet.".into(),
        ));
    }
    if prompt.chars().count() > MAX_ASK_CHARS {
        return Err(CoreError::InvalidInput(format!(
            "Questions are limited to {MAX_ASK_CHARS} characters."
        )));
    }
    start_primary(
        engine,
        PrimarySpec {
            kind: RequestKind::AskAi,
            suggestion_kind: SuggestionKind::Ask,
            question_id: None,
            question_text: Some(prompt.clone()),
            category: None,
            style: Some(engine.style()),
            parent_id: None,
            query: prompt.clone(),
            detected_at: None,
            builder: Builder::Ask { prompt },
        },
    )
}

pub(crate) fn clarify(engine: &Engine) -> Result<Id, CoreError> {
    let question = {
        let st = engine.inner.state.lock();
        st.runtime
            .as_ref()
            .and_then(|rt| rt.questions.last().cloned())
    }
    .ok_or_else(|| CoreError::InvalidState("No question to clarify yet.".into()))?;
    start_primary(
        engine,
        PrimarySpec {
            kind: RequestKind::Clarification,
            suggestion_kind: SuggestionKind::Clarification,
            question_id: Some(question.id.clone()),
            question_text: Some(question.question.clone()),
            category: Some(question.category),
            style: None,
            parent_id: None,
            query: question.question.clone(),
            detected_at: None,
            builder: Builder::Clarify {
                question: question.question,
            },
        },
    )
}

pub(crate) fn refine(
    engine: &Engine,
    request_id: &str,
    refinement: Refinement,
) -> Result<Id, CoreError> {
    // Completed suggestion, or the text streamed so far.
    let source = {
        let st = engine.inner.state.lock();
        let from_rt = st
            .runtime
            .as_ref()
            .and_then(|rt| rt.suggestion(request_id).cloned());
        let from_detached = || st.detached.iter().find(|s| s.id == request_id).cloned();
        let from_active = || {
            st.runtime
                .as_ref()
                .and_then(|rt| rt.active_primary.clone())
                .or_else(|| st.detached_active.clone())
                .filter(|a| a.request_id == request_id && !a.text.trim().is_empty())
        };
        match from_rt.or_else(from_detached) {
            Some(s) => Some((
                s.kind,
                s.question_id.clone(),
                s.question_text.clone(),
                s.category,
                s.style,
                s.content.clone(),
            )),
            None => from_active().map(|a| {
                (
                    match a.kind {
                        RequestKind::AskAi => SuggestionKind::Ask,
                        RequestKind::Clarification => SuggestionKind::Clarification,
                        RequestKind::Refinement => SuggestionKind::Refinement,
                        _ => SuggestionKind::Answer,
                    },
                    None,
                    a.question_text.clone(),
                    a.category,
                    a.style,
                    a.text.clone(),
                )
            }),
        }
    };
    let (kind, question_id, question_text, category, style, previous) = source
        .ok_or_else(|| CoreError::NotFound("That suggestion is no longer available.".into()))?;
    let style = style.unwrap_or_else(|| engine.style());

    if refinement == Refinement::Regenerate {
        let text = question_text.clone().unwrap_or_default();
        let mut spec = match kind {
            SuggestionKind::Ask => PrimarySpec {
                kind: RequestKind::AskAi,
                suggestion_kind: SuggestionKind::Ask,
                question_id: None,
                question_text: Some(text.clone()),
                category: None,
                style: Some(style),
                parent_id: None,
                query: text.clone(),
                detected_at: None,
                builder: Builder::Ask { prompt: text },
            },
            SuggestionKind::Clarification => PrimarySpec {
                kind: RequestKind::Clarification,
                suggestion_kind: SuggestionKind::Clarification,
                question_id,
                question_text: Some(text.clone()),
                category,
                style: None,
                parent_id: None,
                query: text.clone(),
                detected_at: None,
                builder: Builder::Clarify { question: text },
            },
            _ => {
                if text.trim().is_empty() {
                    return Err(CoreError::InvalidState("Nothing to regenerate.".into()));
                }
                let category = category.unwrap_or_else(|| classify_category(&text));
                answer_spec(
                    question_id,
                    text,
                    category,
                    style,
                    engine.settings().assistant.response_length,
                    None,
                )
            }
        };
        spec.parent_id = Some(request_id.to_string());
        return start_primary(engine, spec);
    }

    let query = question_text
        .clone()
        .unwrap_or_else(|| truncate_to_tokens(&previous, 120));
    start_primary(
        engine,
        PrimarySpec {
            kind: RequestKind::Refinement,
            suggestion_kind: SuggestionKind::Refinement,
            question_id,
            question_text: question_text.clone(),
            category,
            style: Some(style),
            parent_id: Some(request_id.to_string()),
            query,
            detected_at: None,
            builder: Builder::Refine {
                question: question_text,
                previous,
                refinement,
                style,
            },
        },
    )
}

pub(crate) fn generate_followups(engine: &Engine) -> Result<Id, CoreError> {
    if engine.mode() == Mode::Candidate {
        return ask(engine, CANDIDATE_QUESTIONS_PROMPT.to_string());
    }
    let turn = {
        let mut st = engine.inner.state.lock();
        let rt = st
            .runtime
            .as_mut()
            .ok_or_else(|| CoreError::InvalidState("Start a meeting first.".into()))?;
        if !rt.turn.text.trim().is_empty() {
            let t = rt.turn.text.clone();
            rt.last_turn_text = t.clone();
            rt.turn.reset();
            t
        } else if !rt.last_turn_text.trim().is_empty() {
            rt.last_turn_text.clone()
        } else {
            rt.recent_remote_text(120_000)
        }
    };
    if turn.trim().is_empty() {
        return Err(CoreError::InvalidState(
            "No candidate answer to analyze yet.".into(),
        ));
    }
    start_followups(engine, turn)
}

// ---------------------------------------------------------------------------
// Context assembly (spec §18, §30, §52)
// ---------------------------------------------------------------------------

pub(crate) async fn build_prompt_context(engine: &Engine, query: &str) -> PromptContext {
    let settings = engine.settings();
    let provider = engine.embedder();
    let info = provider.info();
    let query_vec: Option<(String, Vec<f32>)> = if info.id == "local_hash" {
        Some((info.model.clone(), engine.hashing().embed_one(query)))
    } else {
        match tokio::time::timeout(
            QUERY_EMBED_TIMEOUT,
            provider.embed(vec![query.to_string()], CancellationToken::new()),
        )
        .await
        {
            Ok(Ok(mut v)) if !v.is_empty() => Some((info.model.clone(), v.remove(0))),
            _ => None,
        }
    };

    let st = engine.inner.state.lock();
    let mut ctx = PromptContext {
        mode: st.mode,
        profile: settings.profile.clone(),
        ..Default::default()
    };
    let Some(rt) = st.runtime.as_ref() else {
        return ctx;
    };
    let now_rel = rt.now_rel();
    let window_start = now_rel - settings.assistant.context_window_secs as i64 * 1000;
    let deictic = has_deictic_reference(query);
    let now = now_ms();
    let recent_obs: Vec<&ScreenObservation> = rt
        .observations
        .iter()
        .rev()
        .filter(|o| now - o.timestamp <= if deictic { 180_000 } else { 120_000 })
        .take(if deictic { 2 } else { 1 })
        .collect();

    ctx.long_term_summary = rt.memory.long_term_summary.clone();
    ctx.recent_transcript = rt.lines_since(window_start);
    ctx.references_screen = deictic && !recent_obs.is_empty();
    ctx.screen = recent_obs.into_iter().rev().cloned().collect();
    ctx.retrieved = rt.memory.retrieve(RetrievalQuery {
        text: query,
        vector: query_vec.as_ref().map(|(m, v)| (m.as_str(), v.as_slice())),
        k: 6,
        exclude_from_ms: Some(window_start),
        now_ms: now_rel,
        boost_screen: deictic,
    });
    ctx.topics = rt.topics.topics();
    let asked = rt.asked_questions();
    ctx.asked_questions = asked[asked.len().saturating_sub(25)..].to_vec();
    ctx.previous_suggestions = rt
        .all_followups
        .iter()
        .rev()
        .take(25)
        .map(|f| f.question.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    ctx.claims = rt.claims[rt.claims.len().saturating_sub(15)..].to_vec();
    ctx
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/// Streams a completion, invoking `on_delta(delta, is_first)` for every visible
/// text chunk. Honors the ticket's cancellation token and timeouts.
async fn stream_text(
    engine: &Engine,
    ticket: &RequestTicket,
    llm: Arc<dyn LlmProvider>,
    request: LlmRequest,
    latency_anchor_ms: Option<i64>,
    mut on_delta: impl FnMut(&str, bool),
) -> Result<String, ProviderError> {
    let cancel = ticket.cancel.clone();
    let started_ms = now_ms();
    let mut stream = tokio::select! {
        _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
        r = tokio::time::timeout(CONNECT_TIMEOUT, llm.stream(request, cancel.clone())) => {
            r.map_err(|_| ProviderError::Timeout)??
        }
    };
    let mut text = String::new();
    let mut first_token_at: Option<Instant> = None;
    loop {
        let wait = if first_token_at.is_none() {
            FIRST_TOKEN_TIMEOUT
        } else {
            CHUNK_TIMEOUT
        };
        let next = tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            n = tokio::time::timeout(wait, stream.next()) => n.map_err(|_| ProviderError::Timeout)?,
        };
        match next {
            Some(Ok(LlmChunk::Delta(d))) => {
                if d.is_empty() {
                    continue;
                }
                let first = first_token_at.is_none();
                if first {
                    first_token_at = Some(Instant::now());
                    let anchor = latency_anchor_ms.unwrap_or(started_ms);
                    record_latency(
                        &engine.inner.latency,
                        engine.sink(),
                        LatencyStage::DetectionToFirstToken,
                        (now_ms() - anchor).max(0) as f64,
                        Some(ticket.id.clone()),
                    );
                }
                text.push_str(&d);
                on_delta(&d, first);
            }
            Some(Ok(LlmChunk::Done { stop_reason, .. })) => {
                if stop_reason.as_deref() == Some("refusal") {
                    return Err(ProviderError::Refused("refusal".into()));
                }
                break;
            }
            Some(Err(e)) => return Err(e),
            None => break,
        }
    }
    if let Some(t) = first_token_at {
        record_latency(
            &engine.inner.latency,
            engine.sink(),
            LatencyStage::FirstTokenToComplete,
            t.elapsed().as_secs_f64() * 1000.0,
            Some(ticket.id.clone()),
        );
    }
    Ok(text)
}

/// Collects a full response (background work).
async fn complete(
    ticket: &RequestTicket,
    llm: Arc<dyn LlmProvider>,
    request: LlmRequest,
    timeout: Duration,
) -> Result<String, ProviderError> {
    let cancel = ticket.cancel.clone();
    let fut = openmeet_core::providers::complete_text(llm.as_ref(), request, cancel.clone());
    tokio::select! {
        _ = cancel.cancelled() => Err(ProviderError::Cancelled),
        r = tokio::time::timeout(timeout, fut) => r.unwrap_or(Err(ProviderError::Timeout)),
    }
}

fn llm_health(engine: &Engine, result: &Result<String, ProviderError>) {
    match result {
        Ok(_) => engine
            .inner
            .health
            .set(Component::Llm, ComponentStatus::Active, None),
        Err(ProviderError::Cancelled) | Err(ProviderError::Refused(_)) => {}
        Err(
            e @ (ProviderError::Unauthorized(_)
            | ProviderError::NotConfigured(_)
            | ProviderError::BlockedByLocalOnly(_)),
        ) => engine.inner.health.set(
            Component::Llm,
            ComponentStatus::Error,
            Some(e.user_message()),
        ),
        Err(e) => engine.inner.health.set(
            Component::Llm,
            ComponentStatus::Degraded,
            Some(e.user_message()),
        ),
    }
}

fn update_view(engine: &Engine, request_id: &str, f: impl FnOnce(&mut ActiveRequestView)) {
    let mut st = engine.inner.state.lock();
    let view = match st.runtime.as_mut() {
        Some(rt) => rt.active_primary.as_mut(),
        None => st.detached_active.as_mut(),
    };
    if let Some(v) = view.filter(|v| v.request_id == request_id) {
        f(v);
    }
}

// ---------------------------------------------------------------------------
// Primary slot (answers, ask, refinements, clarifications)
// ---------------------------------------------------------------------------

fn start_primary(engine: &Engine, spec: PrimarySpec) -> Result<Id, CoreError> {
    if engine.processing_stopped() {
        return Err(CoreError::InvalidState(
            "Processing is stopped. Resume to use OpenMeet.".into(),
        ));
    }
    let llm = engine.llm();
    let ticket = engine
        .inner
        .requests
        .begin(RequestSlot::Primary, spec.kind, |id| {
            Some(MeetingEvent::AnswerStarted {
                request_id: id.clone(),
                kind: spec.kind,
                slot: RequestSlot::Primary,
                question_id: spec.question_id.clone(),
                question_text: spec.question_text.clone(),
                category: spec.category,
                style: spec.style,
                parent_id: spec.parent_id.clone(),
            })
        });
    let view = ActiveRequestView {
        request_id: ticket.id.clone(),
        kind: spec.kind,
        slot: RequestSlot::Primary,
        state: AnswerState::Created,
        question_text: spec.question_text.clone(),
        category: spec.category,
        style: spec.style,
        text: String::new(),
        error: None,
    };
    {
        let mut st = engine.inner.state.lock();
        match st.runtime.as_mut() {
            Some(rt) => rt.active_primary = Some(view),
            None => st.detached_active = Some(view),
        }
    }
    let Some(llm) = llm else {
        let msg = "AI provider unavailable. Check provider settings.".to_string();
        engine
            .inner
            .requests
            .set_state(&ticket, AnswerState::Failed, Some(msg.clone()));
        update_view(engine, &ticket.id, |v| {
            v.state = AnswerState::Failed;
            v.error = Some(msg);
        });
        return Ok(ticket.id);
    };
    let id = ticket.id.clone();
    let engine2 = engine.clone();
    engine
        .inner
        .handle
        .spawn(async move { run_primary(engine2, ticket, llm, spec).await });
    Ok(id)
}

async fn run_primary(
    engine: Engine,
    ticket: RequestTicket,
    llm: Arc<dyn LlmProvider>,
    spec: PrimarySpec,
) {
    let rm = &engine.inner.requests;
    let set = |state: AnswerState, error: Option<String>| {
        rm.set_state(&ticket, state, error.clone());
        update_view(&engine, &ticket.id, |v| {
            if !v.state.is_terminal() {
                v.state = state;
                v.error = error;
            }
        });
    };
    set(AnswerState::ContextBuilding, None);
    let settings = engine.settings();
    let ctx = build_prompt_context(&engine, &spec.query).await;
    if ticket.is_cancelled() {
        set(AnswerState::Cancelled, None);
        return;
    }
    let parts = spec
        .builder
        .build(&ctx, settings.assistant.context_token_budget as usize);
    let request = LlmRequest {
        request_id: ticket.id.clone(),
        kind: spec.kind,
        system: parts.system,
        messages: vec![ChatMessage::user(parts.user)],
        max_output_tokens: settings.ai.llm.max_output_tokens,
        temperature: Some(settings.ai.llm.temperature),
        json_output: false,
        latency_sensitive: true,
    };
    set(AnswerState::Requesting, None);
    let info = llm.info();
    let result = stream_text(
        &engine,
        &ticket,
        llm,
        request,
        spec.detected_at,
        |delta, first| {
            if first {
                set(AnswerState::Streaming, None);
            }
            rm.emit_if_current(
                &ticket,
                MeetingEvent::AnswerDelta {
                    request_id: ticket.id.clone(),
                    text: delta.to_string(),
                },
            );
            update_view(&engine, &ticket.id, |v| v.text.push_str(delta));
        },
    )
    .await;
    llm_health(&engine, &result);
    match result {
        Ok(text) => {
            let suggestion = Suggestion {
                id: ticket.id.clone(),
                meeting_id: engine.current_meeting_id(),
                kind: spec.suggestion_kind,
                question_id: spec.question_id.clone(),
                question_text: spec.question_text.clone(),
                category: spec.category,
                style: spec.style,
                parent_id: spec.parent_id.clone(),
                prompt_id: parts.id.to_string(),
                prompt_version: parts.version,
                content: text,
                followups: Vec::new(),
                provider: info.id.clone(),
                model: info.model.clone(),
                created_at: now_ms() - ticket.started.elapsed().as_millis() as i64,
                completed_at: Some(now_ms()),
                pinned: false,
            };
            store_suggestion(&engine, &suggestion);
            rm.emit_if_current(
                &ticket,
                MeetingEvent::AnswerComplete {
                    request_id: ticket.id.clone(),
                    suggestion,
                },
            );
            set(AnswerState::Complete, None);
        }
        Err(ProviderError::Cancelled) => set(AnswerState::Cancelled, None),
        Err(e) => {
            tracing::warn!(request = %ticket.id, error = ?std::mem::discriminant(&e), "assistant request failed");
            set(AnswerState::Failed, Some(e.user_message()));
        }
    }
}

fn store_suggestion(engine: &Engine, suggestion: &Suggestion) {
    let store = {
        let mut st = engine.inner.state.lock();
        match st.runtime.as_mut() {
            Some(rt) if suggestion.meeting_id.as_deref() == Some(rt.meeting.id.as_str()) => {
                rt.push_suggestion(suggestion.clone());
                Some(rt.store.clone())
            }
            _ => {
                if st.detached.len() >= 20 {
                    st.detached.pop_front();
                }
                st.detached.push_back(suggestion.clone());
                None
            }
        }
    };
    if let Some(store) = store {
        engine.persist(&store, PersistOp::Suggestion(suggestion.clone()));
    }
}

// ---------------------------------------------------------------------------
// Follow-ups slot (interviewer mode, spec §22–§24)
// ---------------------------------------------------------------------------

pub(crate) fn start_followups(engine: &Engine, turn_text: String) -> Result<Id, CoreError> {
    if engine.processing_stopped() {
        return Err(CoreError::InvalidState(
            "Processing is stopped. Resume to use OpenMeet.".into(),
        ));
    }
    let llm = engine.llm();
    let preview = truncate_to_tokens(&turn_text, 40);
    let ticket = engine.inner.requests.begin(
        RequestSlot::Followups,
        RequestKind::InterviewerFollowups,
        |id| {
            Some(MeetingEvent::AnswerStarted {
                request_id: id.clone(),
                kind: RequestKind::InterviewerFollowups,
                slot: RequestSlot::Followups,
                question_id: None,
                question_text: Some(preview.clone()),
                category: None,
                style: None,
                parent_id: None,
            })
        },
    );
    {
        let mut st = engine.inner.state.lock();
        if let Some(rt) = st.runtime.as_mut() {
            rt.latest_followups.clear();
            rt.latest_followups_request = Some(ticket.id.clone());
        }
    }
    let Some(llm) = llm else {
        engine.inner.requests.set_state(
            &ticket,
            AnswerState::Failed,
            Some("AI provider unavailable. Check provider settings.".into()),
        );
        return Ok(ticket.id);
    };
    let id = ticket.id.clone();
    let engine2 = engine.clone();
    engine
        .inner
        .handle
        .spawn(async move { run_followups(engine2, ticket, llm, turn_text).await });
    Ok(id)
}

fn is_duplicate(
    engine: &Engine,
    candidate: &str,
    existing: &[String],
    accepted: &[FollowUpQuestion],
) -> bool {
    let hashing = engine.hashing();
    let cv = hashing.embed_one(candidate);
    existing
        .iter()
        .map(String::as_str)
        .chain(accepted.iter().map(|f| f.question.as_str()))
        .any(|e| {
            lexical_similarity(candidate, e) >= DUPLICATE_LEXICAL
                || cosine(&cv, &hashing.embed_one(e)) >= DUPLICATE_COSINE
        })
}

async fn run_followups(
    engine: Engine,
    ticket: RequestTicket,
    llm: Arc<dyn LlmProvider>,
    turn_text: String,
) {
    let rm = &engine.inner.requests;
    rm.set_state(&ticket, AnswerState::ContextBuilding, None);
    let settings = engine.settings();
    let ctx = build_prompt_context(&engine, &turn_text).await;
    if ticket.is_cancelled() {
        rm.set_state(&ticket, AnswerState::Cancelled, None);
        return;
    }
    // Heuristic claims from the turn ground the prompt.
    let pseudo = TranscriptSegment {
        id: new_id(),
        meeting_id: engine.current_meeting_id().unwrap_or_default(),
        utterance_id: String::new(),
        speaker_id: Some("remote".into()),
        speaker_role: SpeakerRole::Remote,
        source: AudioSourceKind::System,
        start_time: 0,
        end_time: 0,
        text: turn_text.clone(),
        is_final: true,
        confidence: None,
    };
    let heuristic_claims = extract_claims(&pseudo);
    let parts = interviewer_followup::build(
        &ctx,
        &turn_text,
        &heuristic_claims,
        MAX_FOLLOWUPS,
        settings.assistant.context_token_budget as usize,
    );
    let mut existing = ctx.asked_questions.clone();
    existing.extend(ctx.previous_suggestions.iter().cloned());
    let request = LlmRequest {
        request_id: ticket.id.clone(),
        kind: RequestKind::InterviewerFollowups,
        system: parts.system,
        messages: vec![ChatMessage::user(parts.user)],
        max_output_tokens: settings.ai.llm.max_output_tokens.min(1500),
        temperature: Some(settings.ai.llm.temperature),
        json_output: true,
        latency_sensitive: true,
    };
    rm.set_state(&ticket, AnswerState::Requesting, None);
    let info = llm.info();
    let mut parser = FollowupStreamParser::new();
    let mut accepted: Vec<FollowUpQuestion> = Vec::new();
    let handle_items = |items: Vec<FollowupItem>, accepted: &mut Vec<FollowUpQuestion>| {
        for item in items {
            match item {
                FollowupItem::Followup {
                    question,
                    category,
                    reason,
                    priority,
                    trigger,
                } => {
                    let question = question.trim().to_string();
                    if question.is_empty()
                        || accepted.len() >= MAX_FOLLOWUPS
                        || is_duplicate(&engine, &question, &existing, accepted)
                    {
                        continue;
                    }
                    accepted.push(FollowUpQuestion {
                        id: new_id(),
                        request_id: ticket.id.clone(),
                        question,
                        category,
                        reason,
                        priority,
                        trigger,
                        created_at: now_ms(),
                    });
                    {
                        let mut st = engine.inner.state.lock();
                        if let Some(rt) = st.runtime.as_mut() {
                            if rt.latest_followups_request.as_deref() == Some(ticket.id.as_str()) {
                                rt.latest_followups = accepted.clone();
                            }
                        }
                    }
                    rm.emit_if_current(
                        &ticket,
                        MeetingEvent::FollowupsReady {
                            request_id: ticket.id.clone(),
                            questions: accepted.clone(),
                            done: false,
                        },
                    );
                }
                FollowupItem::Claim { text, kind } => {
                    let text = text.trim().to_string();
                    if text.is_empty() {
                        continue;
                    }
                    let (claim, store) = {
                        let mut st = engine.inner.state.lock();
                        let Some(rt) = st.runtime.as_mut() else {
                            continue;
                        };
                        if rt
                            .claims
                            .iter()
                            .any(|c| lexical_similarity(&c.text, &text) >= 0.8)
                        {
                            continue;
                        }
                        let claim = Claim {
                            id: new_id(),
                            meeting_id: rt.meeting.id.clone(),
                            text,
                            kind,
                            segment_id: None,
                            speaker_role: SpeakerRole::Remote,
                            source: ExtractionSource::Llm,
                            created_at: now_ms(),
                        };
                        rt.claims.push(claim.clone());
                        (claim, rt.store.clone())
                    };
                    engine.persist(&store, PersistOp::Claims(vec![claim.clone()]));
                    engine.emit(MeetingEvent::ClaimsExtracted {
                        claims: vec![claim],
                    });
                }
                FollowupItem::Topic { name, status } => {
                    let update = {
                        let mut st = engine.inner.state.lock();
                        st.runtime.as_mut().and_then(|rt| {
                            rt.topics.apply_analysis(&[(name, status)]).then(|| {
                                (
                                    rt.meeting.id.clone(),
                                    rt.topics.topics(),
                                    rt.topics.suggested_next(),
                                    rt.store.clone(),
                                )
                            })
                        })
                    };
                    if let Some((mid, topics, next, store)) = update {
                        engine.persist(
                            &store,
                            PersistOp::Topics {
                                meeting_id: mid,
                                topics: topics.clone(),
                            },
                        );
                        engine.emit(MeetingEvent::TopicsUpdated {
                            topics,
                            suggested_next: next,
                        });
                    }
                }
                FollowupItem::Gap { text } => {
                    let mut st = engine.inner.state.lock();
                    if let Some(rt) = st.runtime.as_mut() {
                        let text = text.trim().to_string();
                        if !text.is_empty()
                            && !rt.gaps.iter().any(|g| lexical_similarity(g, &text) >= 0.8)
                        {
                            rt.gaps.push(text);
                        }
                    }
                }
            }
        }
    };
    let mut first_seen = false;
    let result = stream_text(&engine, &ticket, llm, request, None, |delta, first| {
        if first && !first_seen {
            first_seen = true;
            rm.set_state(&ticket, AnswerState::Streaming, None);
        }
        let items = parser.push(delta);
        handle_items(items, &mut accepted);
    })
    .await;
    llm_health(&engine, &result);
    match result {
        Ok(text) => {
            let tail = parser.finish();
            handle_items(tail, &mut accepted);
            rm.emit_if_current(
                &ticket,
                MeetingEvent::FollowupsReady {
                    request_id: ticket.id.clone(),
                    questions: accepted.clone(),
                    done: true,
                },
            );
            let suggestion = Suggestion {
                id: ticket.id.clone(),
                meeting_id: engine.current_meeting_id(),
                kind: SuggestionKind::Followups,
                question_id: None,
                question_text: Some(truncate_to_tokens(&turn_text, 60)),
                category: None,
                style: None,
                parent_id: None,
                prompt_id: parts.id.to_string(),
                prompt_version: parts.version,
                content: text,
                followups: accepted.clone(),
                provider: info.id.clone(),
                model: info.model.clone(),
                created_at: now_ms() - ticket.started.elapsed().as_millis() as i64,
                completed_at: Some(now_ms()),
                pinned: false,
            };
            {
                let mut st = engine.inner.state.lock();
                if let Some(rt) = st.runtime.as_mut() {
                    rt.all_followups.extend(accepted.iter().cloned());
                }
            }
            store_suggestion(&engine, &suggestion);
            rm.emit_if_current(
                &ticket,
                MeetingEvent::AnswerComplete {
                    request_id: ticket.id.clone(),
                    suggestion,
                },
            );
            rm.set_state(&ticket, AnswerState::Complete, None);
        }
        Err(ProviderError::Cancelled) => rm.set_state(&ticket, AnswerState::Cancelled, None),
        Err(e) => rm.set_state(&ticket, AnswerState::Failed, Some(e.user_message())),
    }
}

// ---------------------------------------------------------------------------
// Background work
// ---------------------------------------------------------------------------

/// LLM confirmation of an ambiguous heuristic detection (spec §17).
pub(crate) fn spawn_semantic_check(
    engine: Engine,
    meeting_id: String,
    segment: TranscriptSegment,
    heuristic: DetectionResult,
    recent: Vec<TranscriptLine>,
) {
    let Some(llm) = engine.background_llm() else {
        return;
    };
    let ticket = engine.inner.requests.begin(
        RequestSlot::Background,
        RequestKind::QuestionClassification,
        |_| None,
    );
    let handle = engine.inner.handle.clone();
    handle.spawn(async move {
        let mode = engine.mode();
        let parts = question_detection::build(&segment.text, &recent, mode);
        let request = LlmRequest {
            request_id: ticket.id.clone(),
            kind: RequestKind::QuestionClassification,
            system: parts.system,
            messages: vec![ChatMessage::user(parts.user)],
            max_output_tokens: 300,
            temperature: Some(0.0),
            json_output: true,
            latency_sensitive: true,
        };
        engine
            .inner
            .requests
            .set_state(&ticket, AnswerState::Requesting, None);
        let result = complete(&ticket, llm, request, Duration::from_secs(5)).await;
        let classification = result
            .as_ref()
            .ok()
            .and_then(|t| question_detection::parse(t));
        engine.inner.requests.set_state(
            &ticket,
            if result.is_ok() {
                AnswerState::Complete
            } else {
                AnswerState::Failed
            },
            None,
        );
        let Some(c) = classification else { return };
        if !(c.is_question && c.requires_response && c.confidence >= 0.5) {
            return;
        }
        let settings = engine.settings();
        let mut persist_ops = Vec::new();
        let mut embed_jobs: Vec<EmbedJob> = Vec::new();
        let (question, store) = {
            let mut st = engine.inner.state.lock();
            let Some(rt) = st
                .runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id && r.state() == MeetingState::Active)
            else {
                return;
            };
            let text = [
                c.question.trim(),
                heuristic.question.trim(),
                segment.text.trim(),
            ]
            .into_iter()
            .find(|t| !t.is_empty())
            .unwrap_or_default()
            .to_string();
            let q = DetectedQuestion {
                id: new_id(),
                meeting_id: meeting_id.clone(),
                segment_ids: if heuristic.segment_ids.is_empty() {
                    vec![segment.id.clone()]
                } else {
                    heuristic.segment_ids.clone()
                },
                is_question: true,
                confidence: c.confidence.max(heuristic.confidence),
                question: text,
                parts: if heuristic.parts.is_empty() {
                    vec![c.question.clone()]
                } else {
                    heuristic.parts.clone()
                },
                category: c.category,
                speaker_id: segment.speaker_id.clone(),
                speaker_role: segment.speaker_role,
                requires_response: true,
                method: DetectionMethod::Semantic,
                supersedes: heuristic.supersedes.clone(),
                detected_at: now_ms(),
            };
            rt.questions.push(q.clone());
            persist_ops.push(PersistOp::Question(q.clone()));
            if let Some(prev) = rt.open_answer_for(&q) {
                remember_answer(rt, &prev, &mut persist_ops, &mut embed_jobs);
            }
            rt.memory.add(MemoryItem {
                id: q.id.clone(),
                kind: MemoryKind::Question,
                text: format!("{} asked: {}", rt.speaker_label(q.speaker_role), q.question),
                at_ms: segment.start_time,
            });
            (q, rt.store.clone())
        };
        for op in persist_ops {
            engine.persist(&store, op);
        }
        for job in embed_jobs {
            let _ = engine.inner.embed_tx.send(job);
        }
        engine.emit(MeetingEvent::QuestionDetected {
            question: question.clone(),
        });
        if settings.assistant.auto_answer && engine.mode() == Mode::Candidate {
            let _ = answer_question(&engine, &question, engine.style());
        }
    });
}

/// Rolling compression of conversation older than the raw window.
pub(crate) fn spawn_compression(
    engine: Engine,
    meeting_id: String,
    previous: String,
    lines: Vec<TranscriptLine>,
    until_ms: i64,
) {
    if engine
        .inner
        .requests
        .background_running(RequestKind::Compression)
    {
        return;
    }
    let ticket =
        engine
            .inner
            .requests
            .begin(RequestSlot::Background, RequestKind::Compression, |_| None);
    let handle = engine.inner.handle.clone();
    handle.spawn(async move {
        let mode = engine.mode();
        let llm_summary = match engine.background_llm() {
            Some(llm) => {
                let parts = context_compression::build(&previous, &lines, mode);
                let request = LlmRequest {
                    request_id: ticket.id.clone(),
                    kind: RequestKind::Compression,
                    system: parts.system,
                    messages: vec![ChatMessage::user(parts.user)],
                    max_output_tokens: 800,
                    temperature: Some(0.1),
                    json_output: false,
                    latency_sensitive: false,
                };
                engine
                    .inner
                    .requests
                    .set_state(&ticket, AnswerState::Requesting, None);
                complete(&ticket, llm, request, Duration::from_secs(90))
                    .await
                    .ok()
                    .map(|t| context_compression::parse(&t))
                    .filter(|s| !s.trim().is_empty())
            }
            None => None,
        };
        if ticket.is_cancelled() {
            engine
                .inner
                .requests
                .set_state(&ticket, AnswerState::Cancelled, None);
            return;
        }
        let summary = llm_summary.unwrap_or_else(|| extractive_compress(&previous, &lines));
        {
            let mut st = engine.inner.state.lock();
            if let Some(rt) = st.runtime.as_mut().filter(|r| r.meeting.id == meeting_id) {
                if until_ms > rt.memory.summarized_until_ms {
                    rt.memory.long_term_summary = summary;
                    rt.memory.summarized_until_ms = until_ms;
                }
            }
        }
        engine
            .inner
            .requests
            .set_state(&ticket, AnswerState::Complete, None);
    });
}

/// Compression fallback without an LLM: keeps questions, numbers and named
/// technologies, bounded in size.
pub(crate) fn extractive_compress(previous: &str, lines: &[TranscriptLine]) -> String {
    let mut out = String::new();
    if !previous.trim().is_empty() {
        out.push_str(previous.trim());
        out.push('\n');
    }
    for line in lines {
        let t = line.text.trim();
        let keep = t.contains('?')
            || t.chars().any(|c| c.is_ascii_digit())
            || !openmeet_intel::claims::extract_entities(t).is_empty();
        if keep {
            out.push_str(&format!(
                "- {}: {}\n",
                line.speaker,
                truncate_to_tokens(t, 60)
            ));
        }
    }
    tail_to_tokens(out.trim(), 700)
}

pub(crate) fn spawn_topic_analysis(
    engine: Engine,
    meeting_id: String,
    lines: Vec<TranscriptLine>,
    topics: Vec<Topic>,
) {
    let Some(llm) = engine.background_llm() else {
        return;
    };
    if lines.is_empty()
        || engine
            .inner
            .requests
            .background_running(RequestKind::TopicAnalysis)
    {
        return;
    }
    let ticket =
        engine
            .inner
            .requests
            .begin(RequestSlot::Background, RequestKind::TopicAnalysis, |_| {
                None
            });
    let handle = engine.inner.handle.clone();
    handle.spawn(async move {
        let mode = engine.mode();
        let parts = topic_analysis::build(&lines, &topics, mode);
        let request = LlmRequest {
            request_id: ticket.id.clone(),
            kind: RequestKind::TopicAnalysis,
            system: parts.system,
            messages: vec![ChatMessage::user(parts.user)],
            max_output_tokens: 500,
            temperature: Some(0.1),
            json_output: true,
            latency_sensitive: false,
        };
        let result = complete(&ticket, llm, request, Duration::from_secs(60)).await;
        engine.inner.requests.set_state(
            &ticket,
            if result.is_ok() {
                AnswerState::Complete
            } else {
                AnswerState::Failed
            },
            None,
        );
        let Ok(text) = result else { return };
        let items = topic_analysis::parse(&text);
        if items.is_empty() {
            return;
        }
        let update = {
            let mut st = engine.inner.state.lock();
            st.runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id)
                .and_then(|rt| {
                    rt.topics.apply_analysis(&items).then(|| {
                        (
                            rt.topics.topics(),
                            rt.topics.suggested_next(),
                            rt.store.clone(),
                        )
                    })
                })
        };
        if let Some((topics, next, store)) = update {
            engine.persist(
                &store,
                PersistOp::Topics {
                    meeting_id: meeting_id.clone(),
                    topics: topics.clone(),
                },
            );
            engine.emit(MeetingEvent::TopicsUpdated {
                topics,
                suggested_next: next,
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Summaries (spec §41)
// ---------------------------------------------------------------------------

pub(crate) fn spawn_meeting_summary(engine: Engine, meeting_id: String) {
    let llm = engine.llm();
    let snapshot = {
        let st = engine.inner.state.lock();
        st.runtime
            .as_ref()
            .filter(|r| r.meeting.id == meeting_id)
            .map(|rt| {
                (
                    rt.to_record(),
                    rt.memory.long_term_summary.clone(),
                    rt.gaps.clone(),
                    rt.store.clone(),
                )
            })
    };
    let Some((record, long_term, gaps, store)) = snapshot else {
        return;
    };
    let handle = engine.inner.handle.clone();
    handle.spawn(async move {
        let summary = summarize_with(&engine, &record, llm, &long_term, &gaps).await;
        let meeting = {
            let mut st = engine.inner.state.lock();
            st.runtime
                .as_mut()
                .filter(|r| r.meeting.id == meeting_id)
                .map(|rt| {
                    rt.summary = Some(summary.clone());
                    let _ = rt.transition(MeetingState::Completed);
                    rt.meeting.clone()
                })
        };
        engine.persist(&store, PersistOp::Summary(summary.clone()));
        if let Some(m) = meeting {
            engine.persist(&store, PersistOp::Meeting(m));
        } else {
            // A new meeting already replaced this one: still mark it completed.
            let mut m = record.meeting.clone();
            m.state = MeetingState::Completed;
            engine.persist(&store, PersistOp::Meeting(m));
        }
        engine.emit(MeetingEvent::SummaryReady { summary });
        engine.emit_state(&meeting_id);
        engine.notice(NoticeLevel::Info, "Meeting summary is ready.", None);
    });
}

pub(crate) async fn summarize_record(engine: &Engine, record: &MeetingRecord) -> MeetingSummary {
    summarize_with(engine, record, engine.llm(), "", &[]).await
}

fn fallback_summary(record: &MeetingRecord, gaps: &[String]) -> MeetingSummary {
    let mut s = extractive_summary(record);
    merge_gaps(&mut s, gaps);
    if s.markdown.trim().is_empty() {
        s.markdown =
            meeting_summary::render_markdown(&s, &record.meeting.title, record.meeting.mode);
    }
    s
}

fn merge_gaps(s: &mut MeetingSummary, gaps: &[String]) {
    if gaps.is_empty() {
        return;
    }
    if let Some(iv) = s.interviewer.as_mut() {
        for g in gaps {
            if !iv
                .clarification_areas
                .iter()
                .any(|c| lexical_similarity(c, g) >= 0.8)
            {
                iv.clarification_areas.push(g.clone());
            }
        }
    }
}

async fn summarize_with(
    engine: &Engine,
    record: &MeetingRecord,
    llm: Option<Arc<dyn LlmProvider>>,
    long_term: &str,
    gaps: &[String],
) -> MeetingSummary {
    let mode = record.meeting.mode;
    let Some(llm) = llm else {
        return fallback_summary(record, gaps);
    };
    if record.segments.is_empty() && record.observations.is_empty() && record.questions.is_empty() {
        return fallback_summary(record, gaps);
    }
    let lines: Vec<TranscriptLine> = record
        .segments
        .iter()
        .map(|s| TranscriptLine {
            speaker: s.speaker_role.label(mode).to_string(),
            text: s.text.clone(),
            at_ms: s.start_time,
        })
        .collect();
    let questions: Vec<String> = record
        .questions
        .iter()
        .map(|q| q.question.clone())
        .collect();
    let answers: Vec<String> = record.answers.iter().map(|a| a.text.clone()).collect();
    let followups: Vec<String> = record
        .suggestions
        .iter()
        .flat_map(|s| s.followups.iter().map(|f| f.question.clone()))
        .collect();
    let observations: Vec<String> = record
        .observations
        .iter()
        .map(|o| o.description.clone())
        .collect();
    let action_items: Vec<String> = record.action_items.iter().map(|a| a.text.clone()).collect();
    let input = meeting_summary::SummaryInput {
        mode,
        title: &record.meeting.title,
        long_term_summary: long_term,
        transcript: &lines,
        questions: &questions,
        answers: &answers,
        followups: &followups,
        topics: &record.topics,
        claims: &record.claims,
        observations: &observations,
        action_items: &action_items,
    };
    let parts = meeting_summary::build(&input, 9000);
    let ticket = engine
        .inner
        .requests
        .begin(RequestSlot::Background, RequestKind::Summary, |_| None);
    let info = llm.info();
    let request = LlmRequest {
        request_id: ticket.id.clone(),
        kind: RequestKind::Summary,
        system: parts.system,
        messages: vec![ChatMessage::user(parts.user)],
        max_output_tokens: 3000,
        temperature: Some(0.2),
        json_output: true,
        latency_sensitive: false,
    };
    engine
        .inner
        .requests
        .set_state(&ticket, AnswerState::Requesting, None);
    let result = complete(&ticket, llm, request, Duration::from_secs(180)).await;
    llm_health(engine, &result);
    let generated_by = format!("{}/{}", info.id, info.model);
    let parsed = result
        .as_ref()
        .ok()
        .and_then(|t| meeting_summary::parse(t, &record.meeting.id, mode, &generated_by));
    engine.inner.requests.set_state(
        &ticket,
        if parsed.is_some() {
            AnswerState::Complete
        } else {
            AnswerState::Failed
        },
        None,
    );
    match parsed {
        Some(mut s) => {
            merge_gaps(&mut s, gaps);
            if s.markdown.trim().is_empty() {
                s.markdown = meeting_summary::render_markdown(&s, &record.meeting.title, mode);
            }
            s
        }
        None => {
            engine.notice(
                NoticeLevel::Warning,
                "The AI summary could not be generated; created a basic summary instead.",
                Some(Component::Llm),
            );
            fallback_summary(record, gaps)
        }
    }
}
