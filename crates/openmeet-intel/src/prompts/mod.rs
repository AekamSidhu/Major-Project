//! Versioned, provider-independent prompt builders and tolerant output
//! parsers.
//!
//! Every prompt module exposes `ID`, `VERSION`, a `build(...)` returning
//! [`PromptParts`] and, where the output is structured, a parser that tolerates
//! code fences, leading prose, CRLF line endings and partially malformed
//! output.
//!
//! Prompt-injection hardening: all untrusted content (transcript, screen text,
//! OCR, retrieved snippets, profile, candidate answers) is placed inside
//! clearly delimited blocks such as `<transcript>…</transcript>`, and every
//! system prompt states that content inside those blocks is data, never
//! instructions. Closing / opening tags that appear inside untrusted text are
//! neutralized so data cannot break out of its block.
//!
//! Every builder ends its user prompt with the question or request on its own
//! line: `Question: <text>`.

use openmeet_core::types::{Claim, Mode, ScreenObservation, Topic, UserProfile};

pub mod ask;
pub mod candidate_answer;
pub mod candidate_clarification;
pub mod context_compression;
pub mod interviewer_followup;
pub mod meeting_summary;
pub mod question_detection;
pub mod refine;
pub mod screen_analysis;
pub mod topic_analysis;

mod context;
pub(crate) mod shared;

pub use context::render_context;
pub(crate) use context::{ContextOptions, render_context_with};

/// A built prompt: system prompt plus the user message.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptParts {
    /// Stable prompt identifier (stored with every suggestion).
    pub id: &'static str,
    /// Prompt template version (bumped whenever the wording changes).
    pub version: u32,
    /// System prompt.
    pub system: String,
    /// User message (context blocks followed by the `Question:` line).
    pub user: String,
}

/// One transcript line as rendered into prompts.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptLine {
    /// Display label of the speaker ("You", "Interviewer", "Candidate", …).
    pub speaker: String,
    /// What was said.
    pub text: String,
    /// Milliseconds since the meeting started.
    pub at_ms: i64,
}

/// A retrieved piece of earlier context.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextSnippet {
    /// "earlier_discussion" | "previous_question" | "previous_answer" |
    /// "screen" | "claim" | "fact"
    pub kind: String,
    /// Snippet text.
    pub text: String,
    /// Meeting-relative milliseconds, when known.
    pub at_ms: Option<i64>,
}

/// Everything a prompt builder may use as context.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PromptContext {
    /// Which side of the conversation the user is on.
    pub mode: Mode,
    /// User-provided background, role context, planned topics, preferences.
    pub profile: UserProfile,
    /// Rolling compressed summary of the conversation before the raw window.
    pub long_term_summary: String,
    /// Retrieved earlier context, best match first.
    pub retrieved: Vec<ContextSnippet>,
    /// Raw recent transcript, oldest first.
    pub recent_transcript: Vec<TranscriptLine>,
    /// Recent screen observations, oldest first.
    pub screen: Vec<ScreenObservation>,
    /// The question uses deictic references ("this", "the diagram") → the
    /// screen is primary context and is rendered first.
    pub references_screen: bool,
    /// Conversation topics (covered, current, pending).
    pub topics: Vec<Topic>,
    /// Questions already asked in the meeting (interviewer mode: by the user).
    pub asked_questions: Vec<String>,
    /// Follow-up questions OpenMeet already suggested.
    pub previous_suggestions: Vec<String>,
    /// Notable statements extracted from the conversation.
    pub claims: Vec<Claim>,
}
