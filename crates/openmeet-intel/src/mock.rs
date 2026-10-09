//! Deterministic providers for tests, developer tools and demo mode
//! (spec §72). They never touch the network.
//!
//! [`MockLlmProvider`] reads the prompt it receives (the same prompts real
//! models get) and produces realistic output in exactly the formats the
//! parsers expect, streamed in small chunks — so the overlay, follow-up
//! parsing, summaries and the demo behave like the real product.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use openmeet_core::error::ProviderError;
use openmeet_core::providers::{
    EmbeddingProvider, LlmChunk, LlmProvider, LlmRequest, LlmStream, OcrEngine, OcrOutput,
    TokenUsage, Transcriber, TranscriptionOutput, TranscriptionRequest, VisionContext,
    VisionProvider, VisionRequest, VisionResult,
};
use openmeet_core::types::{
    AnswerSections, ProviderInfo, ProviderLocation, QuestionCategory, RequestKind,
};
use parking_lot::Mutex;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::claims::{extract_entities, extract_metrics};
use crate::detect::{classify_category, split_parts};
use crate::prompts::candidate_answer;
use crate::prompts::shared::{extract_block, extract_block_items, extract_field, extract_question};
use crate::text::{
    HashingEmbedder, estimate_tokens, lexical_similarity, split_sentences, truncate_to_tokens,
};
use crate::vision::describe_ocr_text;

/// Custom response function (demo scenarios).
pub(crate) type Responder = Arc<dyn Fn(&LlmRequest) -> Option<String> + Send + Sync>;

struct State {
    script: VecDeque<Result<String, ProviderError>>,
    first_token_ms: u64,
    chunk_ms: u64,
    requests: Vec<LlmRequest>,
}

/// Deterministic, streaming LLM. Cheap to clone (shared state).
#[derive(Clone)]
pub struct MockLlmProvider {
    state: Arc<Mutex<State>>,
    responder: Option<Responder>,
}

impl Default for MockLlmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockLlmProvider {
    /// Default behaviour: realistic output derived from the request.
    pub fn new() -> Self {
        MockLlmProvider {
            state: Arc::new(Mutex::new(State {
                script: VecDeque::new(),
                first_token_ms: 0,
                chunk_ms: 15,
                requests: Vec::new(),
            })),
            responder: None,
        }
    }

    /// Returns `responses` in order (for any request kind), then the default.
    pub fn with_script(responses: Vec<String>) -> Self {
        let m = Self::new();
        m.state.lock().script.extend(responses.into_iter().map(Ok));
        m
    }

    /// Custom responder consulted before the default behaviour.
    pub(crate) fn with_responder(responder: Responder) -> Self {
        let mut m = Self::new();
        m.responder = Some(responder);
        m
    }

    /// The next request returns `text`.
    pub fn push_response(&self, text: String) {
        self.state.lock().script.push_back(Ok(text));
    }

    /// The next request fails with `error`.
    pub fn push_failure(&self, error: ProviderError) {
        self.state.lock().script.push_back(Err(error));
    }

    /// Delay before the first chunk.
    pub fn set_first_token_latency_ms(&self, ms: u64) {
        self.state.lock().first_token_ms = ms;
    }

    /// Delay between chunks.
    pub fn set_chunk_delay_ms(&self, ms: u64) {
        self.state.lock().chunk_ms = ms;
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<LlmRequest> {
        self.state.lock().requests.clone()
    }

    /// Number of requests received.
    pub fn request_count(&self) -> usize {
        self.state.lock().requests.len()
    }

    fn response_for(&self, request: &LlmRequest) -> Result<String, ProviderError> {
        if let Some(scripted) = self.state.lock().script.pop_front() {
            return scripted;
        }
        if let Some(r) = &self.responder {
            if let Some(text) = r(request) {
                return Ok(text);
            }
        }
        Ok(default_response(request))
    }
}

/// Splits text into small streaming chunks (≈ 2 words), keeping whitespace.
fn chunks(text: &str) -> VecDeque<String> {
    let mut out = VecDeque::new();
    let mut cur = String::new();
    let mut words = 0;
    for ch in text.chars() {
        cur.push(ch);
        if ch.is_whitespace() {
            words += 1;
            if words >= 2 || ch == '\n' {
                out.push_back(std::mem::take(&mut cur));
                words = 0;
            }
        }
    }
    if !cur.is_empty() {
        out.push_back(cur);
    }
    out
}

struct StreamState {
    chunks: VecDeque<String>,
    first: bool,
    first_ms: u64,
    chunk_ms: u64,
    cancel: CancellationToken,
    done: bool,
    usage: TokenUsage,
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "mock".into(),
            display_name: "Mock (deterministic, local)".into(),
            model: "mock-1".into(),
            location: ProviderLocation::Mock,
            endpoint: None,
        }
    }

    fn supports_images(&self) -> bool {
        true
    }

    async fn stream(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> Result<LlmStream, ProviderError> {
        let response = self.response_for(&request);
        let (first_ms, chunk_ms) = {
            let mut st = self.state.lock();
            st.requests.push(request.clone());
            (st.first_token_ms, st.chunk_ms)
        };
        let text = response?;
        let input_tokens = estimate_tokens(&request.system)
            + request
                .messages
                .iter()
                .map(|m| estimate_tokens(&m.content))
                .sum::<usize>();
        let state = StreamState {
            usage: TokenUsage {
                input_tokens: input_tokens as u32,
                output_tokens: estimate_tokens(&text) as u32,
            },
            chunks: chunks(&text),
            first: true,
            first_ms,
            chunk_ms,
            cancel,
            done: false,
        };
        let stream = futures::stream::unfold(state, |mut st| async move {
            if st.done {
                return None;
            }
            let delay = if st.first { st.first_ms } else { st.chunk_ms };
            st.first = false;
            if delay > 0 {
                tokio::select! {
                    _ = st.cancel.cancelled() => {
                        st.done = true;
                        return Some((Err(ProviderError::Cancelled), st));
                    }
                    _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
                }
            }
            if st.cancel.is_cancelled() {
                st.done = true;
                return Some((Err(ProviderError::Cancelled), st));
            }
            match st.chunks.pop_front() {
                Some(c) => Some((Ok(LlmChunk::Delta(c)), st)),
                None => {
                    st.done = true;
                    let usage = Some(st.usage);
                    Some((
                        Ok(LlmChunk::Done {
                            stop_reason: Some("end_turn".to_string()),
                            usage,
                        }),
                        st,
                    ))
                }
            }
        });
        Ok(stream.boxed())
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(vec!["mock-1".to_string()])
    }
}

// ---------------------------------------------------------------------------
// Default responses
// ---------------------------------------------------------------------------

fn user_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == openmeet_core::providers::ChatRole::User)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Renders answer sections in the streamed [[SECTION]] format.
pub(crate) fn render_sections(s: &AnswerSections) -> String {
    let mut out = String::new();
    if !s.structure.is_empty() {
        out.push_str("[[STRUCTURE]]\n");
        for (i, item) in s.structure.iter().enumerate() {
            out.push_str(&format!("{}. {}\n", i + 1, item));
        }
    }
    out.push_str("[[ANSWER]]\n");
    out.push_str(s.answer.trim());
    out.push('\n');
    if !s.key_points.is_empty() {
        out.push_str("[[KEY_POINTS]]\n");
        for p in &s.key_points {
            out.push_str(&format!("- {p}\n"));
        }
    }
    if let Some(e) = s.example.as_ref().filter(|e| !e.trim().is_empty()) {
        out.push_str(&format!("[[EXAMPLE]]\n{}\n", e.trim()));
    }
    if let Some(f) = s.follow_up.as_ref().filter(|f| !f.trim().is_empty()) {
        out.push_str(&format!("[[FOLLOW_UP]]\n{}\n", f.trim()));
    }
    if let Some(b) = s.basis.as_ref().filter(|b| !b.trim().is_empty()) {
        out.push_str(&format!("[[BASIS]]\n{}\n", b.trim()));
    }
    if let Some(c) = s.caveat.as_ref().filter(|c| !c.trim().is_empty()) {
        out.push_str(&format!("[[CAVEAT]]\n{}\n", c.trim()));
    }
    out
}

fn short(q: &str) -> String {
    truncate_to_tokens(q.trim(), 30)
}

const INSUFFICIENT: &str = "Context insufficient — verify before using. I don't have enough context about your own experience to answer this reliably; add your background in Settings → Profile.";

fn candidate_answer_text(question: &str, category: QuestionCategory) -> String {
    let entities = extract_entities(question);
    let tech = entities.first().cloned();
    let q = short(question);
    let mut s = AnswerSections::default();
    match category {
        QuestionCategory::SystemDesign => {
            s.structure = vec![
                "Clarify requirements and expected scale".into(),
                "Define the API and data model".into(),
                "Sketch the high-level components".into(),
                "Choose storage and caching".into(),
                "Plan for scaling and failure handling".into(),
            ];
            s.answer = format!(
                "I'd start by clarifying the expected scale and the read/write ratio, because that drives every later choice. Then I'd define a small API, sketch the main components{}, and walk through how data flows between them. Finally I'd cover how it scales, what fails first, and how I'd monitor it.",
                tech.as_ref()
                    .map(|t| format!(" (including where {t} fits)"))
                    .unwrap_or_default()
            );
            s.key_points = vec![
                "State assumptions out loud before designing".into(),
                "Keep the first version simple, then scale the bottleneck".into(),
                "Name the trade-offs (consistency, cost, latency) explicitly".into(),
            ];
            s.follow_up = Some("What scale and latency targets should I design for?".into());
            s.basis = Some("Suggested framing: a standard system-design structure, not taken from your own experience.".into());
        }
        QuestionCategory::Behavioral
        | QuestionCategory::Experience
        | QuestionCategory::Leadership => {
            s.answer = "A good structure here: briefly set the situation and your responsibility, describe the specific actions you took, and finish with a concrete result and what you learned.".into();
            s.key_points = vec![
                "Situation and your role, in one or two sentences".into(),
                "Two or three specific actions you personally took".into(),
                "A measurable result and the lesson you took away".into(),
            ];
            s.basis = Some("Suggested framing: STAR structure. The details must come from your own experience.".into());
            s.caveat = Some(INSUFFICIENT.into());
        }
        QuestionCategory::Tradeoff => {
            s.answer = format!(
                "I'd frame it as a trade-off: what problem we needed to solve, the options we compared{}, what each one costs, and why the chosen option fit our constraints best — plus what would make me revisit the decision.",
                tech.as_ref()
                    .map(|t| format!(" (for example {t} versus a simpler alternative)"))
                    .unwrap_or_default()
            );
            s.key_points = vec![
                "Start from the requirement, not the technology".into(),
                "Compare at least two realistic alternatives".into(),
                "Be explicit about what you gave up".into(),
            ];
            s.basis = Some("Inference: answer structure only; check the specifics against what actually happened.".into());
        }
        _ => {
            s.answer = format!(
                "Here's how I'd approach \"{q}\": give the short answer first, then one concrete detail{}, and close with the main trade-off or risk.",
                tech.as_ref()
                    .map(|t| format!(" about {t}"))
                    .unwrap_or_default()
            );
            s.key_points = vec![
                "Lead with the direct answer".into(),
                "Support it with one specific detail or example".into(),
                "Mention the main trade-off".into(),
            ];
            s.basis = Some(
                "Suggested framing: generic structure; fill in specifics you know to be true."
                    .into(),
            );
        }
    }
    render_sections(&s)
}

fn ask_text(question: &str) -> String {
    let entities = extract_entities(question);
    let mut s = AnswerSections {
        answer: format!(
            "Short answer: {}. Start from the goal behind the question, then add the one or two details that matter most.",
            if entities.is_empty() {
                format!("it depends on the context of \"{}\"", short(question))
            } else {
                format!("this is mainly about {}", entities.join(", "))
            }
        ),
        key_points: vec![
            "Answer the core question first".into(),
            "Add one concrete detail".into(),
        ],
        ..Default::default()
    };
    s.basis = Some("Inference: based only on the meeting context available to OpenMeet.".into());
    render_sections(&s)
}

fn refine_text(user: &str) -> String {
    let previous = extract_block(user, "previous_answer").unwrap_or_default();
    let change = extract_field(user, "Requested change").unwrap_or_else(|| "regenerate".into());
    let mut s = candidate_answer::parse(&previous);
    if s.answer.trim().is_empty() {
        s.answer = previous.trim().to_string();
    }
    match change.as_str() {
        "shorter" => {
            s.answer = split_sentences(&s.answer)
                .into_iter()
                .next()
                .unwrap_or_default();
            s.key_points.truncate(2);
            s.structure.clear();
            s.example = None;
            s.basis = None;
        }
        "longer" => {
            s.answer.push_str(" To make it concrete, walk through one real example step by step and close with what you would improve next time.");
            s.key_points
                .push("Close with what you would do differently".into());
        }
        "more_technical" => {
            s.answer.push_str(" Technically, call out the data model, the consistency guarantees, and the failure modes you designed for.");
            s.key_points
                .push("Name the consistency model and failure modes".into());
        }
        "simplify" => {
            s.answer = format!(
                "In simple terms: {}",
                split_sentences(&s.answer)
                    .into_iter()
                    .next()
                    .unwrap_or_default()
            );
            s.key_points.truncate(2);
            s.structure.clear();
        }
        "give_example" => {
            s.example = Some("For example: when traffic doubled before a launch, the team first measured where time was spent, fixed the single slowest dependency, and only then added capacity.".into());
        }
        "show_rationale" => {
            s.basis = Some("Key considerations: answer the question that was asked, stay grounded in facts you can back up, and make trade-offs explicit.".into());
        }
        _ => {}
    }
    render_sections(&s)
}

fn clarification_text(question: &str) -> String {
    format!(
        "[[CLARIFY]]\n- What scale or constraints should I assume?\n- Is the focus on design, implementation or past experience?\n- Is there a specific part of \"{}\" you want me to go deeper on?\n[[ANSWER]]\nThe question can be read in more than one way; asking one short clarifying question first will make the answer more relevant.\n",
        short(question)
    )
}

/// A follow-up that fits the kind of technology mentioned.
fn kind_followup(e: &str, trigger: String) -> (String, &'static str, String, &'static str, String) {
    use crate::lexicon::{TechKind, tech_kind};
    let (q, cat, reason) = match tech_kind(e) {
        Some(TechKind::Queue) => (
            format!(
                "What delivery guarantees did you need from {e}, and how did consumers handle duplicates?"
            ),
            "technical",
            format!("{e} was mentioned without its delivery semantics or duplicate handling."),
        ),
        Some(TechKind::Database) => (
            format!(
                "How did you keep {e} consistent under concurrent writes, and how did schema changes roll out?"
            ),
            "technical",
            format!("{e} was mentioned but not its consistency model or migrations."),
        ),
        Some(TechKind::Cache) => (
            format!(
                "How did you invalidate data in {e}, and what happens when it is cold or unavailable?"
            ),
            "technical",
            format!("Caching with {e} was mentioned without invalidation or failure behaviour."),
        ),
        Some(TechKind::Cloud) | Some(TechKind::Infra) => (
            format!("What did running on {e} cost, and what operational burden did it add?"),
            "tradeoff",
            format!("{e} was mentioned without its operational cost."),
        ),
        Some(TechKind::Observability) => (
            format!("Which signals in {e} would have caught the last incident earliest?"),
            "technical",
            format!("{e} was mentioned, but not which signals mattered."),
        ),
        _ => (
            format!("What happens to the system when {e} is slow or unavailable?"),
            "technical",
            format!("No failure handling was described for {e}."),
        ),
    };
    (q, cat, reason, "normal", trigger)
}

/// Follow-up candidates grounded in the candidate's turn.
fn followup_text(user: &str) -> String {
    let turn = extract_block(user, "candidate_answer").unwrap_or_default();
    let mut avoid = extract_block_items(user, "asked_questions");
    avoid.extend(extract_block_items(user, "previous_suggestions"));
    let entities = extract_entities(&turn);
    let metrics = extract_metrics(&turn);
    let sentences = split_sentences(&turn);
    let trigger_for = |needle: &str| -> String {
        sentences
            .iter()
            .find(|s| s.to_lowercase().contains(&needle.to_lowercase()))
            .cloned()
            .or_else(|| sentences.first().cloned())
            .map(|s| truncate_to_tokens(&s, 30))
            .unwrap_or_default()
    };
    let mut candidates: Vec<(String, &str, String, &str, String)> = Vec::new();
    if let Some(e) = entities.first() {
        candidates.push((
            format!(
                "What made {e} the right choice here, and which alternatives did you rule out?"
            ),
            "tradeoff",
            format!("The candidate chose {e} but did not explain why it beat the alternatives."),
            "high",
            trigger_for(e),
        ));
        candidates.push(kind_followup(e, trigger_for(e)));
    }
    if let Some(m) = metrics.first() {
        candidates.push((
            format!("How did you measure {m}, and what was the baseline before the change?"),
            "experience",
            format!("The candidate quoted {m} without saying how it was measured."),
            "normal",
            trigger_for(m),
        ));
    }
    if let Some(e) = entities.get(1) {
        candidates.push((
            format!(
                "How do {} and {e} interact, and where could that integration break?",
                entities[0]
            ),
            "architecture",
            format!(
                "Both {} and {e} were mentioned, but not how they work together.",
                entities[0]
            ),
            "optional",
            trigger_for(e),
        ));
    }
    candidates.push((
        "What trade-offs did you accept with that approach, and would you make the same call today?".into(),
        "tradeoff",
        "The answer described what was done but not what it cost.".into(),
        "normal",
        sentences.first().map(|s| truncate_to_tokens(s, 30)).unwrap_or_default(),
    ));
    candidates.push((
        "Can you walk me through a concrete example of that end to end?".into(),
        "follow_up",
        "A specific example would show the depth of hands-on involvement.".into(),
        "optional",
        sentences
            .last()
            .map(|s| truncate_to_tokens(s, 30))
            .unwrap_or_default(),
    ));

    let mut lines: Vec<String> = Vec::new();
    for e in entities.iter().take(3) {
        lines.push(
            json!({"type": "claim", "text": format!("Uses {e}"), "kind": "technology"}).to_string(),
        );
    }
    for m in metrics.iter().take(2) {
        lines.push(
            json!({"type": "claim", "text": format!("Reported {m}"), "kind": "metric"}).to_string(),
        );
    }
    let mut emitted = 0;
    for (q, cat, reason, prio, trigger) in candidates {
        if emitted >= 3 {
            break;
        }
        if avoid.iter().any(|a| lexical_similarity(a, &q) >= 0.6) {
            continue;
        }
        lines.push(
            json!({"type": "followup", "question": q, "category": cat, "reason": reason, "priority": prio, "trigger": trigger})
                .to_string(),
        );
        emitted += 1;
    }
    if !entities.is_empty() {
        lines.push(json!({"type": "gap", "text": "Failure handling and operational cost were not discussed"}).to_string());
        lines.push(
            json!({"type": "topic", "name": "Reliability & failure handling", "status": "pending"})
                .to_string(),
        );
    }
    lines.join("\n") + "\n"
}

fn summary_text(user: &str) -> String {
    let meeting = extract_block(user, "meeting").unwrap_or_default();
    let title = meeting
        .lines()
        .find_map(|l| l.trim().strip_prefix("Title:"))
        .map(|t| t.trim().to_string())
        .unwrap_or_else(|| "the meeting".into());
    let interviewer = meeting.to_lowercase().contains("interviewer");
    let questions = extract_block_items(user, "questions");
    let answers = extract_block_items(user, "answers");
    let followups = extract_block_items(user, "followups");
    let actions = extract_block_items(user, "action_items");
    let topic_lines = extract_block_items(user, "topics");
    let topic_name = |l: &str| l.split(" (").next().unwrap_or(l).trim().to_string();
    let discussed: Vec<String> = topic_lines
        .iter()
        .filter(|l| !l.contains("not discussed"))
        .map(|l| topic_name(l))
        .collect();
    let not_discussed: Vec<String> = topic_lines
        .iter()
        .filter(|l| l.contains("not discussed"))
        .map(|l| topic_name(l))
        .collect();
    let transcript = extract_block(user, "transcript").unwrap_or_default();
    let tech = extract_entities(&transcript);
    let overview = format!(
        "{} covered {} question{}{}.",
        title,
        questions.len(),
        if questions.len() == 1 { "" } else { "s" },
        if discussed.is_empty() {
            String::new()
        } else {
            format!(", mainly about {}", discussed.join(", "))
        }
    );
    let mut obj = json!({
        "overview": overview,
        "keyTopics": discussed,
        "questionsAsked": questions,
        "answers": answers.iter().take(10).collect::<Vec<_>>(),
        "decisions": [],
        "actionItems": actions,
        "openQuestions": [],
        "technicalTopics": tech,
        "followUpSuggestions": followups.iter().take(6).collect::<Vec<_>>(),
    });
    if interviewer {
        obj["interviewer"] = json!({
            "questionCoverage": topic_lines,
            "clarificationAreas": [],
            "claimsToVerify": extract_block_items(user, "claims"),
            "unexploredTopics": not_discussed,
        });
    }
    obj.to_string()
}

fn classification_text(user: &str) -> String {
    let utterance = extract_block(user, "latest_utterance").unwrap_or_default();
    let is_q = !split_parts(&utterance).is_empty() || utterance.trim_end().ends_with('?');
    json!({
        "is_question": is_q,
        "confidence": if is_q { 0.8 } else { 0.2 },
        "question": utterance.trim(),
        "category": classify_category(&utterance).as_str(),
        "requires_response": is_q,
    })
    .to_string()
}

fn compression_text(user: &str) -> String {
    let previous = extract_block(user, "summary").unwrap_or_default();
    let transcript = extract_block(user, "transcript").unwrap_or_default();
    let mut out = String::new();
    if !previous.trim().is_empty() {
        out.push_str(previous.trim());
        out.push('\n');
    }
    for line in transcript.lines().filter(|l| !l.trim().is_empty()).take(12) {
        out.push_str(&format!("- {}\n", truncate_to_tokens(line.trim(), 25)));
    }
    out
}

fn screen_text(user: &str) -> String {
    let screen = extract_block(user, "screen").unwrap_or_default();
    let app = screen
        .lines()
        .find_map(|l| l.trim().strip_prefix("Application:"))
        .map(|a| a.trim().to_string());
    let ctx = VisionContext {
        source_label: "Screen".into(),
        application: app.clone(),
        window_title: None,
        recent_transcript: String::new(),
        question: None,
    };
    let r = match extract_block(user, "ocr").filter(|t| !t.trim().is_empty()) {
        Some(ocr) => describe_ocr_text(&ocr, &ctx),
        None => VisionResult {
            description: format!(
                "Screen of {}",
                app.clone().unwrap_or_else(|| "an application".into())
            ),
            visible_text: None,
            entities: Vec::new(),
            application: app,
            content_type: Some("other".into()),
        },
    };
    json!({
        "description": r.description,
        "visible_text": r.visible_text,
        "entities": r.entities,
        "application": r.application,
        "content_type": r.content_type,
    })
    .to_string()
}

fn topic_analysis_text(user: &str) -> String {
    let items: Vec<serde_json::Value> = extract_block_items(user, "topics")
        .iter()
        .map(|l| {
            let name = l
                .split(" (")
                .next()
                .unwrap_or(l)
                .split(':')
                .next()
                .unwrap_or(l)
                .trim()
                .to_string();
            let lower = l.to_lowercase();
            let status = if lower.contains("current") {
                "current"
            } else if lower.contains("covered") {
                "covered"
            } else {
                "pending"
            };
            json!({"name": name, "status": status})
        })
        .collect();
    json!({"topics": items}).to_string()
}

/// Default deterministic output for a request.
pub(crate) fn default_response(request: &LlmRequest) -> String {
    let user = user_text(request);
    let question = extract_question(&user).unwrap_or_default();
    match request.kind {
        RequestKind::CandidateAnswer => {
            let category = extract_field(&user, "Question category")
                .and_then(|c| QuestionCategory::parse_lenient(&c))
                .unwrap_or_else(|| classify_category(&question));
            candidate_answer_text(&question, category)
        }
        RequestKind::AskAi => ask_text(&question),
        RequestKind::Refinement => refine_text(&user),
        RequestKind::Clarification => clarification_text(&question),
        RequestKind::InterviewerFollowups => followup_text(&user),
        RequestKind::Summary => summary_text(&user),
        RequestKind::QuestionClassification => classification_text(&user),
        RequestKind::Compression => compression_text(&user),
        RequestKind::ScreenAnalysis => screen_text(&user),
        RequestKind::TopicAnalysis => topic_analysis_text(&user),
    }
}

// ---------------------------------------------------------------------------
// Other mocks
// ---------------------------------------------------------------------------

/// Returns scripted transcripts in order, then empty text.
pub struct MockTranscriber {
    texts: Mutex<VecDeque<String>>,
    latency_ms: u64,
    calls: AtomicUsize,
}

impl MockTranscriber {
    /// Returns `texts` in order (one per final utterance).
    pub fn new(texts: Vec<String>) -> Self {
        MockTranscriber {
            texts: Mutex::new(texts.into()),
            latency_ms: 0,
            calls: AtomicUsize::new(0),
        }
    }

    /// Simulated recognition latency.
    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Number of transcription requests received.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Transcriber for MockTranscriber {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "mock_transcriber".into(),
            display_name: "Mock transcription".into(),
            model: "mock".into(),
            location: ProviderLocation::Mock,
            endpoint: None,
        }
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        cancel: CancellationToken,
    ) -> Result<TranscriptionOutput, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.latency_ms > 0 {
            tokio::select! {
                _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                _ = tokio::time::sleep(Duration::from_millis(self.latency_ms)) => {}
            }
        }
        // Interim requests peek; finals consume.
        let text = if request.clip.is_final {
            self.texts.lock().pop_front().unwrap_or_default()
        } else {
            self.texts.lock().front().cloned().unwrap_or_default()
        };
        Ok(TranscriptionOutput {
            text,
            confidence: Some(0.95),
            language: request.language,
        })
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        Ok(())
    }
}

/// Deterministic vision: derives the observation from OCR text / context.
pub struct MockVisionProvider;

#[async_trait]
impl VisionProvider for MockVisionProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "mock_vision".into(),
            display_name: "Mock vision".into(),
            model: "mock".into(),
            location: ProviderLocation::Mock,
            endpoint: None,
        }
    }

    async fn analyze(
        &self,
        request: VisionRequest,
        _cancel: CancellationToken,
    ) -> Result<VisionResult, ProviderError> {
        Ok(
            match request.ocr_text.as_deref().filter(|t| !t.trim().is_empty()) {
                Some(text) => describe_ocr_text(text, &request.context),
                None => VisionResult {
                    description: format!(
                        "{} ({}x{})",
                        if request.context.source_label.is_empty() {
                            "Screen"
                        } else {
                            &request.context.source_label
                        },
                        request.width,
                        request.height
                    ),
                    visible_text: None,
                    entities: Vec::new(),
                    application: request.context.application.clone(),
                    content_type: Some("other".into()),
                },
            },
        )
    }
}

/// Hashing embeddings labelled as a mock provider.
pub struct MockEmbeddingProvider {
    inner: HashingEmbedder,
}

impl Default for MockEmbeddingProvider {
    fn default() -> Self {
        MockEmbeddingProvider {
            inner: HashingEmbedder::new(HashingEmbedder::DEFAULT_DIMS),
        }
    }
}

impl MockEmbeddingProvider {
    /// New provider (384 dimensions).
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "mock_embeddings".into(),
            display_name: "Mock embeddings".into(),
            model: "mock-hash".into(),
            location: ProviderLocation::Mock,
            endpoint: None,
        }
    }

    async fn embed(
        &self,
        texts: Vec<String>,
        cancel: CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        if cancel.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }
        Ok(texts.iter().map(|t| self.inner.embed_one(t)).collect())
    }
}

/// OCR that always "reads" `text`.
pub struct MockOcr {
    /// Text returned for every image.
    pub text: String,
}

impl OcrEngine for MockOcr {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "mock_ocr".into(),
            display_name: "Mock OCR".into(),
            model: String::new(),
            location: ProviderLocation::Mock,
            endpoint: None,
        }
    }

    fn recognize_png(&self, _png: &[u8]) -> Result<OcrOutput, ProviderError> {
        Ok(OcrOutput {
            text: self.text.clone(),
            lines: self.text.lines().map(str::to_string).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::interviewer_followup::{FollowupItem, FollowupStreamParser};
    use crate::prompts::{
        PromptContext, context_compression, interviewer_followup, meeting_summary,
        question_detection, refine, screen_analysis, topic_analysis,
    };
    use openmeet_core::providers::ChatMessage;
    use openmeet_core::types::{AnswerStyle, Mode, Refinement, ResponseLength, Topic, TopicStatus};

    fn req(kind: RequestKind, system: String, user: String) -> LlmRequest {
        LlmRequest {
            request_id: "r".into(),
            kind,
            system,
            messages: vec![ChatMessage::user(user)],
            max_output_tokens: 1000,
            temperature: None,
            json_output: false,
            latency_sensitive: true,
        }
    }

    async fn collect(p: &MockLlmProvider, r: LlmRequest) -> Result<String, ProviderError> {
        openmeet_core::providers::complete_text(p, r, CancellationToken::new()).await
    }

    #[tokio::test]
    async fn candidate_answers_parse_into_sections() {
        let ctx = PromptContext::default();
        for (q, cat) in [
            (
                "How would you design a URL shortener?",
                QuestionCategory::SystemDesign,
            ),
            (
                "Tell me about a time you disagreed with your manager.",
                QuestionCategory::Behavioral,
            ),
            ("Why did you choose Kafka?", QuestionCategory::Tradeoff),
            ("How does a B-tree index work?", QuestionCategory::Technical),
        ] {
            let p = candidate_answer::build(
                &ctx,
                q,
                cat,
                AnswerStyle::Concise,
                ResponseLength::Medium,
                2000,
            );
            let text = collect(
                &MockLlmProvider::new(),
                req(RequestKind::CandidateAnswer, p.system, p.user),
            )
            .await
            .unwrap();
            let s = candidate_answer::parse(&text);
            assert!(!s.answer.is_empty(), "{q}");
            assert!(!s.key_points.is_empty(), "{q}");
            if cat == QuestionCategory::SystemDesign {
                assert!(!s.structure.is_empty());
            }
            if cat == QuestionCategory::Behavioral {
                assert!(
                    s.caveat.unwrap().contains("Context insufficient"),
                    "never invents personal history"
                );
            }
        }
    }

    #[tokio::test]
    async fn followups_are_grounded_and_parseable() {
        let ctx = PromptContext {
            mode: Mode::Interviewer,
            asked_questions: vec!["How would you design a rate limiter?".into()],
            ..Default::default()
        };
        let turn =
            "I'd probably use Redis with a token bucket per API key. It handles 5k QPS at peak.";
        let p = interviewer_followup::build(&ctx, turn, &[], 4, 3000);
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::InterviewerFollowups, p.system, p.user),
        )
        .await
        .unwrap();
        let mut parser = FollowupStreamParser::new();
        let mut items = parser.push(&text);
        items.extend(parser.finish());
        let followups: Vec<_> = items
            .iter()
            .filter_map(|i| match i {
                FollowupItem::Followup {
                    question,
                    reason,
                    trigger,
                    ..
                } => Some((question.clone(), reason.clone(), trigger.clone())),
                _ => None,
            })
            .collect();
        assert!((2..=4).contains(&followups.len()), "{followups:?}");
        assert!(followups.iter().any(|(q, _, _)| q.contains("Redis")));
        assert!(
            followups
                .iter()
                .all(|(_, r, t)| !r.is_empty() && t.is_some())
        );
        assert!(
            items
                .iter()
                .any(|i| matches!(i, FollowupItem::Claim { .. }))
        );
    }

    #[tokio::test]
    async fn every_structured_output_parses() {
        let topics = vec![Topic {
            id: "architecture".into(),
            meeting_id: "m".into(),
            name: "Architecture".into(),
            status: TopicStatus::Covered,
            mentions: 3,
            first_mentioned_at: Some(0),
            last_mentioned_at: Some(0),
            planned: false,
            summary: None,
        }];
        let input = meeting_summary::SummaryInput {
            mode: Mode::Interviewer,
            title: "Backend interview",
            long_term_summary: "",
            transcript: &[],
            questions: &["How would you design a rate limiter?".to_string()],
            answers: &["Redis token bucket".to_string()],
            followups: &[],
            topics: &topics,
            claims: &[],
            observations: &[],
            action_items: &[],
        };
        let p = meeting_summary::build(&input, 3000);
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::Summary, p.system, p.user),
        )
        .await
        .unwrap();
        let s =
            meeting_summary::parse(&text, "m", Mode::Interviewer, "mock").expect("summary parses");
        assert!(s.overview.contains("Backend interview"));
        assert!(s.interviewer.is_some());

        let p = question_detection::build("so how did you scale it", &[], Mode::Candidate);
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::QuestionClassification, p.system, p.user),
        )
        .await
        .unwrap();
        assert!(question_detection::parse(&text).unwrap().is_question);

        let p = context_compression::build("", &[], Mode::Candidate);
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::Compression, p.system, p.user),
        )
        .await
        .unwrap();
        let _ = context_compression::parse(&text);

        let vctx = VisionContext {
            source_label: "Display".into(),
            application: Some("Excalidraw".into()),
            ..Default::default()
        };
        let p = screen_analysis::build(&vctx, Some("API → Queue\nQueue → Workers"));
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::ScreenAnalysis, p.system, p.user),
        )
        .await
        .unwrap();
        let v = screen_analysis::parse(&text);
        assert_eq!(v.content_type.as_deref(), Some("diagram"));

        let p = topic_analysis::build(&[], &topics, Mode::Interviewer);
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::TopicAnalysis, p.system, p.user),
        )
        .await
        .unwrap();
        assert!(!topic_analysis::parse(&text).is_empty());
    }

    #[tokio::test]
    async fn refinements_transform_previous_answer() {
        let previous =
            "[[ANSWER]]\nFirst sentence. Second sentence.\n[[KEY_POINTS]]\n- a\n- b\n- c\n";
        let ctx = PromptContext::default();
        let p = refine::build(
            &ctx,
            Some("Q?"),
            previous,
            Refinement::Shorter,
            AnswerStyle::Concise,
            2000,
        );
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::Refinement, p.system, p.user),
        )
        .await
        .unwrap();
        let s = candidate_answer::parse(&text);
        assert_eq!(s.answer, "First sentence.");
        assert_eq!(s.key_points.len(), 2);
        let p = refine::build(
            &ctx,
            Some("Q?"),
            previous,
            Refinement::GiveExample,
            AnswerStyle::Concise,
            2000,
        );
        let text = collect(
            &MockLlmProvider::new(),
            req(RequestKind::Refinement, p.system, p.user),
        )
        .await
        .unwrap();
        assert!(candidate_answer::parse(&text).example.is_some());
    }

    #[tokio::test]
    async fn scripted_responses_failures_and_cancellation() {
        let m = MockLlmProvider::with_script(vec!["one".into()]);
        m.push_failure(ProviderError::Network("down".into()));
        let r = req(RequestKind::AskAi, String::new(), "Question: hi".into());
        assert_eq!(collect(&m, r.clone()).await.unwrap(), "one");
        assert!(matches!(
            collect(&m, r.clone()).await,
            Err(ProviderError::Network(_))
        ));
        assert!(
            !collect(&m, r.clone()).await.unwrap().is_empty(),
            "then the default"
        );
        assert_eq!(m.request_count(), 3);

        m.set_first_token_latency_ms(5_000);
        let cancel = CancellationToken::new();
        let mut stream = m.stream(r, cancel.clone()).await.unwrap();
        let started = std::time::Instant::now();
        cancel.cancel();
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(ProviderError::Cancelled)));
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn streams_in_small_chunks() {
        let m = MockLlmProvider::with_script(vec!["a b c d e f g h".into()]);
        m.set_chunk_delay_ms(0);
        let mut s = m
            .stream(
                req(RequestKind::AskAi, String::new(), String::new()),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let mut deltas = 0;
        while let Some(Ok(chunk)) = s.next().await {
            if matches!(chunk, LlmChunk::Delta(_)) {
                deltas += 1;
            }
        }
        assert!(deltas >= 3);
    }

    #[tokio::test]
    async fn transcriber_and_ocr() {
        use openmeet_core::audio::AudioClip;
        use openmeet_core::types::AudioSourceKind;
        let t = MockTranscriber::new(vec!["hello there".into()]);
        let clip = AudioClip {
            utterance_id: "u".into(),
            source: AudioSourceKind::System,
            samples: vec![0.0; 160],
            sample_rate: 16_000,
            start_ms: 0,
            end_ms: 10,
            is_final: true,
        };
        let out = t
            .transcribe(
                TranscriptionRequest {
                    clip: clip.clone(),
                    language: None,
                    prompt: None,
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.text, "hello there");
        let out = t
            .transcribe(
                TranscriptionRequest {
                    clip,
                    language: None,
                    prompt: None,
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(out.text, "");
        assert_eq!(t.calls(), 2);
        let ocr = MockOcr {
            text: "line one\nline two".into(),
        };
        assert_eq!(ocr.recognize_png(&[]).unwrap().lines.len(), 2);
    }
}
