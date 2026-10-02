//! Domain model shared by every subsystem and exported to TypeScript.
//!
//! Conventions:
//! * JSON field names are camelCase, enum values snake_case.
//! * Timestamps are Unix milliseconds (`i64`) and exported to TS as `number`.
//! * Transcript `start_time` / `end_time` are milliseconds relative to the
//!   meeting start.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::ids::Id;

// ---------------------------------------------------------------------------
// Modes, speakers and sources
// ---------------------------------------------------------------------------

/// Which side of the conversation OpenMeet is assisting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Mode {
    /// Assists the person answering questions.
    #[default]
    Candidate,
    /// Assists the person asking questions.
    Interviewer,
}

impl Mode {
    pub fn toggled(self) -> Mode {
        match self {
            Mode::Candidate => Mode::Interviewer,
            Mode::Interviewer => Mode::Candidate,
        }
    }
}

/// Where a piece of audio came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum AudioSourceKind {
    /// The local user's microphone.
    Microphone,
    /// System / meeting output audio (other participants), where supported.
    System,
}

/// Coarse speaker attribution derived from the audio source.
///
/// * `User` – the person running OpenMeet (microphone).
/// * `Remote` – other meeting participants (system audio).
/// * `Unknown` – mixed source (e.g. only a microphone picking up speakers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum SpeakerRole {
    User,
    Remote,
    Unknown,
}

impl SpeakerRole {
    /// Human label for the speaker given the assistance mode.
    pub fn label(self, mode: Mode) -> &'static str {
        match (self, mode) {
            (SpeakerRole::User, _) => "You",
            (SpeakerRole::Remote, Mode::Candidate) => "Interviewer",
            (SpeakerRole::Remote, Mode::Interviewer) => "Candidate",
            (SpeakerRole::Unknown, _) => "Speaker",
        }
    }
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

/// One transcribed utterance (or an interim version of it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TranscriptSegment {
    pub id: Id,
    pub meeting_id: Id,
    /// Partial and final versions of the same utterance share this id, so the
    /// UI can replace a partial with its final text.
    pub utterance_id: Id,
    pub speaker_id: Option<String>,
    pub speaker_role: SpeakerRole,
    pub source: AudioSourceKind,
    /// Milliseconds since meeting start.
    #[ts(type = "number")]
    pub start_time: i64,
    /// Milliseconds since meeting start.
    #[ts(type = "number")]
    pub end_time: i64,
    pub text: String,
    pub is_final: bool,
    pub confidence: Option<f32>,
}

/// Provider-neutral transcription output before it is attached to a meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct TranscriptDraft {
    pub utterance_id: Id,
    pub source: AudioSourceKind,
    /// Unix milliseconds.
    #[ts(type = "number")]
    pub start_ms: i64,
    /// Unix milliseconds.
    #[ts(type = "number")]
    pub end_ms: i64,
    pub text: String,
    pub is_final: bool,
    pub confidence: Option<f32>,
    /// Optional explicit speaker override (simulation / diarizing providers).
    pub speaker_role: Option<SpeakerRole>,
}

// ---------------------------------------------------------------------------
// Questions and follow-ups
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum QuestionCategory {
    Technical,
    Coding,
    SystemDesign,
    Behavioral,
    Experience,
    Product,
    Leadership,
    Clarification,
    Architecture,
    Tradeoff,
    Culture,
    #[default]
    General,
    FollowUp,
}

impl QuestionCategory {
    pub const ALL: [QuestionCategory; 13] = [
        QuestionCategory::Technical,
        QuestionCategory::Coding,
        QuestionCategory::SystemDesign,
        QuestionCategory::Behavioral,
        QuestionCategory::Experience,
        QuestionCategory::Product,
        QuestionCategory::Leadership,
        QuestionCategory::Clarification,
        QuestionCategory::Architecture,
        QuestionCategory::Tradeoff,
        QuestionCategory::Culture,
        QuestionCategory::General,
        QuestionCategory::FollowUp,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            QuestionCategory::Technical => "technical",
            QuestionCategory::Coding => "coding",
            QuestionCategory::SystemDesign => "system_design",
            QuestionCategory::Behavioral => "behavioral",
            QuestionCategory::Experience => "experience",
            QuestionCategory::Product => "product",
            QuestionCategory::Leadership => "leadership",
            QuestionCategory::Clarification => "clarification",
            QuestionCategory::Architecture => "architecture",
            QuestionCategory::Tradeoff => "tradeoff",
            QuestionCategory::Culture => "culture",
            QuestionCategory::General => "general",
            QuestionCategory::FollowUp => "follow_up",
        }
    }

    /// Lenient parser used for LLM output ("System Design", "system-design", ...).
    pub fn parse_lenient(s: &str) -> Option<QuestionCategory> {
        let norm: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c == '-' || c == ' ' { '_' } else { c })
            .collect();
        let norm = match norm.as_str() {
            "followup" => "follow_up",
            "systemdesign" => "system_design",
            "trade_off" | "trade_offs" | "tradeoffs" => "tradeoff",
            other => other,
        };
        QuestionCategory::ALL
            .into_iter()
            .find(|c| c.as_str() == norm)
    }

    /// Display label, e.g. "System Design".
    pub fn label(self) -> &'static str {
        match self {
            QuestionCategory::Technical => "Technical",
            QuestionCategory::Coding => "Coding",
            QuestionCategory::SystemDesign => "System Design",
            QuestionCategory::Behavioral => "Behavioral",
            QuestionCategory::Experience => "Experience",
            QuestionCategory::Product => "Product",
            QuestionCategory::Leadership => "Leadership",
            QuestionCategory::Clarification => "Clarification",
            QuestionCategory::Architecture => "Architecture",
            QuestionCategory::Tradeoff => "Trade-off",
            QuestionCategory::Culture => "Culture",
            QuestionCategory::General => "General",
            QuestionCategory::FollowUp => "Follow-up",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DetectionMethod {
    /// Rule/feature based detector (fast path).
    Heuristic,
    /// Confirmed or produced by an LLM classifier.
    Semantic,
    /// The user explicitly asked for help (hotkey / Ask OpenMeet).
    Manual,
    /// Triggered by a voice command.
    Voice,
    /// Injected by developer tools / demo mode.
    Simulated,
}

/// Output of question detection, attached to a meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DetectedQuestion {
    pub id: Id,
    pub meeting_id: Id,
    /// Transcript segments that make up the question (lead-ins are merged).
    pub segment_ids: Vec<Id>,
    pub is_question: bool,
    pub confidence: f32,
    /// Cleaned question text.
    pub question: String,
    /// Individual sub-questions when the utterance was multi-part.
    pub parts: Vec<String>,
    pub category: QuestionCategory,
    pub speaker_id: Option<String>,
    pub speaker_role: SpeakerRole,
    /// True when the OpenMeet user is expected to answer it.
    pub requires_response: bool,
    pub method: DetectionMethod,
    /// Earlier question this one extends (e.g. continued after a pause).
    pub supersedes: Option<Id>,
    #[ts(type = "number")]
    pub detected_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Priority {
    High,
    #[default]
    Normal,
    Optional,
}

/// A follow-up question suggested to the interviewer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct FollowUpQuestion {
    pub id: Id,
    pub request_id: Id,
    pub question: String,
    pub category: QuestionCategory,
    /// Concise user-facing rationale ("Candidate mentioned X but not Y").
    pub reason: String,
    pub priority: Priority,
    /// The candidate statement that triggered this follow-up.
    pub trigger: Option<String>,
    #[ts(type = "number")]
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Answers / AI requests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum AnswerStyle {
    #[default]
    Concise,
    Detailed,
    Technical,
    Simple,
    Star,
    ExampleDriven,
    StepByStep,
}

impl AnswerStyle {
    pub const ALL: [AnswerStyle; 7] = [
        AnswerStyle::Concise,
        AnswerStyle::Detailed,
        AnswerStyle::Technical,
        AnswerStyle::Simple,
        AnswerStyle::Star,
        AnswerStyle::ExampleDriven,
        AnswerStyle::StepByStep,
    ];

    /// Next style in the cycle (hotkey "cycle answer style").
    pub fn next(self) -> AnswerStyle {
        let idx = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    pub fn label(self) -> &'static str {
        match self {
            AnswerStyle::Concise => "Concise",
            AnswerStyle::Detailed => "Detailed",
            AnswerStyle::Technical => "Technical",
            AnswerStyle::Simple => "Simple",
            AnswerStyle::Star => "STAR",
            AnswerStyle::ExampleDriven => "Example-driven",
            AnswerStyle::StepByStep => "Step-by-step",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ResponseLength {
    Short,
    #[default]
    Medium,
    Long,
}

/// Overlay answer controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Refinement {
    Regenerate,
    Shorter,
    Longer,
    MoreTechnical,
    Simplify,
    GiveExample,
    /// Concise user-facing rationale / key considerations. Never hidden
    /// chain-of-thought.
    ShowRationale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum RequestKind {
    CandidateAnswer,
    InterviewerFollowups,
    AskAi,
    Refinement,
    Clarification,
    Summary,
    Compression,
    QuestionClassification,
    TopicAnalysis,
    ScreenAnalysis,
}

/// Logical UI slot a request renders into. Starting a request in a slot
/// cancels whatever was running in that slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum RequestSlot {
    /// The main overlay card: answers, Ask OpenMeet, refinements.
    Primary,
    /// Interviewer follow-up suggestions.
    Followups,
    /// Not rendered: compression, classification, summaries, screen analysis.
    Background,
}

/// Per-request lifecycle (spec §87).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum AnswerState {
    Created,
    ContextBuilding,
    Requesting,
    Streaming,
    Complete,
    Failed,
    Cancelled,
}

impl AnswerState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            AnswerState::Complete | AnswerState::Failed | AnswerState::Cancelled
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum SuggestionKind {
    Answer,
    Followups,
    Ask,
    Refinement,
    Clarification,
}

/// A completed AI suggestion (persisted in `ai_suggestions`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Suggestion {
    /// Equal to the request id that produced it.
    pub id: Id,
    pub meeting_id: Option<Id>,
    pub kind: SuggestionKind,
    pub question_id: Option<Id>,
    pub question_text: Option<String>,
    pub category: Option<QuestionCategory>,
    pub style: Option<AnswerStyle>,
    /// For refinements: the suggestion that was refined.
    pub parent_id: Option<Id>,
    pub prompt_id: String,
    pub prompt_version: u32,
    /// Full streamed text (sectioned format for answers).
    pub content: String,
    pub followups: Vec<FollowUpQuestion>,
    pub provider: String,
    pub model: String,
    #[ts(type = "number")]
    pub created_at: i64,
    #[ts(type = "number | null")]
    pub completed_at: Option<i64>,
    pub pinned: bool,
}

/// Parsed sections of a candidate answer (see `prompts::candidate_answer`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AnswerSections {
    pub structure: Vec<String>,
    pub answer: String,
    pub key_points: Vec<String>,
    pub example: Option<String>,
    pub follow_up: Option<String>,
    pub basis: Option<String>,
    pub caveat: Option<String>,
}

/// What someone said in response to a detected question (the user's own
/// answer in candidate mode, the candidate's answer in interviewer mode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AnswerRecord {
    pub id: Id,
    pub meeting_id: Id,
    pub question_id: Option<Id>,
    pub speaker_role: SpeakerRole,
    pub text: String,
    pub segment_ids: Vec<Id>,
    #[ts(type = "number")]
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Claims, topics, action items
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ClaimKind {
    Claim,
    Technology,
    Decision,
    Tradeoff,
    Metric,
    Ownership,
    Implementation,
    Risk,
    Unknown,
}

impl ClaimKind {
    pub fn parse_lenient(s: &str) -> ClaimKind {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "technology" | "tech" => ClaimKind::Technology,
            "decision" => ClaimKind::Decision,
            "tradeoff" | "trade_off" => ClaimKind::Tradeoff,
            "metric" | "metrics" => ClaimKind::Metric,
            "ownership" => ClaimKind::Ownership,
            "implementation" | "implementation_detail" => ClaimKind::Implementation,
            "risk" => ClaimKind::Risk,
            "unknown" | "gap" => ClaimKind::Unknown,
            _ => ClaimKind::Claim,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExtractionSource {
    Heuristic,
    Llm,
}

/// A notable statement (mostly from the candidate in interviewer mode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Claim {
    pub id: Id,
    pub meeting_id: Id,
    pub text: String,
    pub kind: ClaimKind,
    pub segment_id: Option<Id>,
    pub speaker_role: SpeakerRole,
    pub source: ExtractionSource,
    #[ts(type = "number")]
    pub created_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TopicStatus {
    /// Not discussed yet (planned or suggested).
    Pending,
    /// Being discussed right now.
    Current,
    /// Discussed earlier.
    Covered,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Topic {
    /// Stable slug, e.g. "observability".
    pub id: String,
    pub meeting_id: Id,
    pub name: String,
    pub status: TopicStatus,
    pub mentions: u32,
    #[ts(type = "number | null")]
    pub first_mentioned_at: Option<i64>,
    #[ts(type = "number | null")]
    pub last_mentioned_at: Option<i64>,
    /// True when the user listed it as a planned topic.
    pub planned: bool,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ActionItem {
    pub id: Id,
    pub meeting_id: Id,
    pub text: String,
    pub owner: Option<String>,
    pub source_segment_id: Option<Id>,
    #[ts(type = "number")]
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Screen context
// ---------------------------------------------------------------------------

/// Structured screen observation. Raw screenshots are never part of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ScreenObservation {
    pub id: Id,
    pub meeting_id: Option<Id>,
    #[ts(type = "number")]
    pub timestamp: i64,
    /// Human description of the capture source ("Display 1", "Window: VS Code").
    pub source: String,
    pub application: Option<String>,
    pub title: Option<String>,
    pub description: String,
    pub visible_text: Option<String>,
    pub entities: Vec<String>,
    /// "code" | "diagram" | "document" | "slides" | "browser" | "terminal" | "other"
    pub content_type: Option<String>,
    /// Perceptual hash of the frame (hex) used for change detection.
    pub frame_hash: Option<String>,
}

/// A captured, privacy-filtered, change-detected frame ready for OCR/vision.
/// Lives only in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenFrame {
    pub id: Id,
    pub captured_at: i64,
    pub source_label: String,
    pub application: Option<String>,
    pub window_title: Option<String>,
    pub width: u32,
    pub height: u32,
    /// PNG-encoded, downscaled frame.
    pub png: Vec<u8>,
    /// 64-bit perceptual hash.
    pub hash: u64,
    /// 0..1, how different from the previously accepted frame.
    pub change_score: f32,
    /// True for an explicit manual capture (bypasses change detection).
    pub manual: bool,
}

// ---------------------------------------------------------------------------
// Meeting lifecycle
// ---------------------------------------------------------------------------

/// Explicit meeting state machine (spec §39 / §86).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum MeetingState {
    #[default]
    Idle,
    Starting,
    Active,
    Paused,
    Stopping,
    Summarizing,
    /// Meeting finished; summary available (or summary failed / skipped).
    Completed,
}

impl MeetingState {
    /// Whether `self -> next` is a legal transition.
    pub fn can_transition_to(self, next: MeetingState) -> bool {
        use MeetingState::*;
        matches!(
            (self, next),
            (Idle, Starting)
                | (Completed, Starting)
                | (Starting, Active)
                | (Starting, Idle)
                | (Active, Paused)
                | (Paused, Active)
                | (Active, Stopping)
                | (Paused, Stopping)
                | (Stopping, Summarizing)
                | (Stopping, Completed)
                | (Summarizing, Completed)
                | (Completed, Idle)
        )
    }

    /// Meeting is running (possibly paused).
    pub fn is_live(self) -> bool {
        matches!(self, MeetingState::Active | MeetingState::Paused)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum PauseReason {
    User,
    /// Emergency privacy stop.
    Emergency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Participant {
    pub id: String,
    pub role: SpeakerRole,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Meeting {
    pub id: Id,
    pub title: String,
    pub mode: Mode,
    pub state: MeetingState,
    #[ts(type = "number")]
    pub started_at: i64,
    #[ts(type = "number | null")]
    pub ended_at: Option<i64>,
    pub participants: Vec<Participant>,
    /// False when the retention policy keeps this meeting in memory only.
    pub persisted: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct InterviewerSummary {
    pub question_coverage: Vec<String>,
    pub clarification_areas: Vec<String>,
    pub claims_to_verify: Vec<String>,
    pub unexplored_topics: Vec<String>,
}

/// Post-meeting summary. Never contains hiring recommendations or scores.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MeetingSummary {
    pub meeting_id: Id,
    pub overview: String,
    pub key_topics: Vec<String>,
    pub questions_asked: Vec<String>,
    pub answers: Vec<String>,
    pub decisions: Vec<String>,
    pub action_items: Vec<String>,
    pub open_questions: Vec<String>,
    pub technical_topics: Vec<String>,
    pub follow_up_suggestions: Vec<String>,
    pub interviewer: Option<InterviewerSummary>,
    /// Rendered markdown of the whole summary.
    pub markdown: String,
    #[ts(type = "number")]
    pub generated_at: i64,
    /// "provider/model" or "extractive" when produced without an LLM.
    pub generated_by: String,
}

/// Everything stored about one meeting (history view, export).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MeetingRecord {
    pub meeting: Meeting,
    pub segments: Vec<TranscriptSegment>,
    pub questions: Vec<DetectedQuestion>,
    pub answers: Vec<AnswerRecord>,
    pub suggestions: Vec<Suggestion>,
    pub topics: Vec<Topic>,
    pub claims: Vec<Claim>,
    pub observations: Vec<ScreenObservation>,
    pub action_items: Vec<ActionItem>,
    pub summary: Option<MeetingSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MeetingListItem {
    pub id: Id,
    pub title: String,
    pub mode: Mode,
    pub state: MeetingState,
    #[ts(type = "number")]
    pub started_at: i64,
    #[ts(type = "number | null")]
    pub ended_at: Option<i64>,
    pub segment_count: u32,
    pub question_count: u32,
    pub has_summary: bool,
}

// ---------------------------------------------------------------------------
// Health, diagnostics, latency
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Component {
    Meeting,
    Microphone,
    SystemAudio,
    Transcription,
    QuestionDetector,
    Llm,
    Vision,
    Screen,
    Embeddings,
    Database,
    Overlay,
    CapturePrivacy,
    Hotkeys,
}

impl Component {
    pub fn label(self) -> &'static str {
        match self {
            Component::Meeting => "Meeting",
            Component::Microphone => "Microphone",
            Component::SystemAudio => "System audio",
            Component::Transcription => "Transcription",
            Component::QuestionDetector => "Question detector",
            Component::Llm => "LLM",
            Component::Vision => "Vision",
            Component::Screen => "Screen capture",
            Component::Embeddings => "Embeddings",
            Component::Database => "Database",
            Component::Overlay => "Overlay",
            Component::CapturePrivacy => "Capture privacy",
            Component::Hotkeys => "Hotkeys",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ComponentStatus {
    Inactive,
    Starting,
    Active,
    Streaming,
    Paused,
    Degraded,
    Error,
    Unsupported,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ComponentHealth {
    pub component: Component,
    pub status: ComponentStatus,
    pub message: Option<String>,
    #[ts(type = "number")]
    pub updated_at: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum LatencyStage {
    /// End of speech -> final transcript available.
    AudioToTranscript,
    /// Final transcript -> question detected.
    TranscriptToDetection,
    /// Request start -> first streamed token.
    DetectionToFirstToken,
    /// First token -> response complete.
    FirstTokenToComplete,
    /// Frame captured -> observation produced.
    ScreenProcessing,
    /// Event received -> rendered in overlay (reported by the UI).
    OverlayRender,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct LatencySample {
    pub stage: LatencyStage,
    pub ms: f64,
    pub request_id: Option<Id>,
    #[ts(type = "number")]
    pub at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct LatencyStats {
    pub stage: LatencyStage,
    pub count: u32,
    pub last_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

// ---------------------------------------------------------------------------
// Privacy rules
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum PrivacyRuleType {
    /// Process / application name, e.g. "1Password".
    Application,
    /// Window title substring.
    Window,
    /// Domain matched against browser window titles, e.g. "bank.example.com".
    Domain,
    /// Rectangle "x,y,width,height" in source pixels that is blacked out.
    Region,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum PrivacyAction {
    #[default]
    Exclude,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct PrivacyRule {
    pub id: Id,
    #[serde(rename = "type")]
    pub rule_type: PrivacyRuleType,
    pub value: String,
    pub action: PrivacyAction,
    #[ts(type = "number")]
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// Where a provider processes data. Drives the privacy / data-flow UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ProviderLocation {
    /// Runs on this device (in-process or a localhost service).
    Local,
    /// Sends data to a remote endpoint.
    Remote,
    /// Deterministic mock used by tests / demo mode. Never sends data anywhere.
    Mock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ProviderInfo {
    /// Stable provider id, e.g. "openai", "anthropic", "ollama", "local_ai".
    pub id: String,
    pub display_name: String,
    pub model: String,
    pub location: ProviderLocation,
    /// Host (never full URLs with credentials) for display purposes.
    pub endpoint: Option<String>,
}

// ---------------------------------------------------------------------------
// Voice commands
// ---------------------------------------------------------------------------

/// Optional spoken commands ("OpenMeet, explain that").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
#[ts(export)]
pub enum VoiceCommand {
    /// "explain that" – explain the latest question / screen content.
    Explain,
    /// "give me a follow-up" – generate follow-ups.
    FollowUp,
    /// "answer that" – generate an answer for the latest question.
    Answer,
    /// "make that concise/shorter".
    Concise,
    /// "more detail / longer".
    Detailed,
    /// Free-form request ("OpenMeet, what does this code do?").
    Ask(String),
}

// ---------------------------------------------------------------------------
// User-provided context
// ---------------------------------------------------------------------------

/// Context the user provides so suggestions never need to invent facts.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct UserProfile {
    /// Candidate mode: resume / background notes in the user's own words.
    pub background: String,
    /// Role / job description / meeting agenda.
    pub role_context: String,
    /// Interviewer mode: planned topics, one per line.
    pub planned_topics: String,
    /// Extra standing instructions for the assistant.
    pub custom_instructions: String,
}

impl UserProfile {
    pub fn planned_topic_list(&self) -> Vec<String> {
        self.planned_topics
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches(['-', '*', '•'])
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meeting_state_machine_transitions() {
        use MeetingState::*;
        assert!(Idle.can_transition_to(Starting));
        assert!(Starting.can_transition_to(Active));
        assert!(Active.can_transition_to(Paused));
        assert!(Paused.can_transition_to(Active));
        assert!(Active.can_transition_to(Stopping));
        assert!(Stopping.can_transition_to(Summarizing));
        assert!(Summarizing.can_transition_to(Completed));
        assert!(Completed.can_transition_to(Starting));
        assert!(!Idle.can_transition_to(Active));
        assert!(!Completed.can_transition_to(Active));
        assert!(!Summarizing.can_transition_to(Active));
        assert!(!Idle.can_transition_to(Paused));
    }

    #[test]
    fn category_lenient_parse() {
        assert_eq!(
            QuestionCategory::parse_lenient("System Design"),
            Some(QuestionCategory::SystemDesign)
        );
        assert_eq!(
            QuestionCategory::parse_lenient("follow-up"),
            Some(QuestionCategory::FollowUp)
        );
        assert_eq!(
            QuestionCategory::parse_lenient("trade-off"),
            Some(QuestionCategory::Tradeoff)
        );
        assert_eq!(QuestionCategory::parse_lenient("nonsense"), None);
    }

    #[test]
    fn style_cycles_through_all() {
        let mut s = AnswerStyle::Concise;
        for _ in 0..AnswerStyle::ALL.len() {
            s = s.next();
        }
        assert_eq!(s, AnswerStyle::Concise);
    }

    #[test]
    fn serde_shapes_match_ts_contract() {
        let rule = PrivacyRule {
            id: "r1".into(),
            rule_type: PrivacyRuleType::Application,
            value: "1Password".into(),
            action: PrivacyAction::Exclude,
            created_at: 1,
        };
        let v = serde_json::to_value(&rule).unwrap();
        assert_eq!(v["type"], "application");
        assert_eq!(v["action"], "exclude");
        assert_eq!(v["createdAt"], 1);

        let cmd = VoiceCommand::Ask("what is this".into());
        let v = serde_json::to_value(&cmd).unwrap();
        assert_eq!(v["kind"], "ask");
        assert_eq!(v["text"], "what is this");
        let v = serde_json::to_value(VoiceCommand::Explain).unwrap();
        assert_eq!(v["kind"], "explain");
    }

    #[test]
    fn planned_topics_parse() {
        let p = UserProfile {
            planned_topics: "- Architecture\n\n* Testing\nObservability  ".into(),
            ..Default::default()
        };
        assert_eq!(
            p.planned_topic_list(),
            vec!["Architecture", "Testing", "Observability"]
        );
    }
}
