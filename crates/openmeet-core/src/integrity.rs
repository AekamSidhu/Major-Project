//! Answer-integrity check (interviewer mode, opt-in).
//!
//! After each candidate answer the engine scores up to four signals — does
//! the transcript read like AI-written text, how close is it to what an LLM
//! answers to the same question, does the timing look read rather than
//! spoken freely, do the eyes move like reading — and fuses them into one
//! flag with its evidence. A flag is a **signal for the interviewer, never a
//! verdict**. Definitions, thresholds and fixtures: `spec/integrity/`.
//!
//! This module holds the shared vocabulary: signal names and levels, the flag
//! the UI receives (exported to TypeScript), and the contract of the local AI
//! sidecar's integrity endpoints ([`IntegrityService`]).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use ts_rs::TS;

use crate::error::ProviderError;
use crate::ids::Id;
use crate::ipc::Rect;
use crate::types::ProviderInfo;

/// One of the four answer-integrity signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum IntegritySignalName {
    /// Does the answer's transcript read like AI-written text? (sidecar)
    AiText,
    /// How close is the answer to what an LLM answers to the same question?
    EchoSimilarity,
    /// Does the answer's timing look read rather than spoken freely,
    /// compared with this speaker's own baseline?
    Delivery,
    /// Do the eyes move like reading lines of text during the answer? (sidecar)
    Reading,
}

impl IntegritySignalName {
    pub const ALL: [IntegritySignalName; 4] = [
        IntegritySignalName::AiText,
        IntegritySignalName::EchoSimilarity,
        IntegritySignalName::Delivery,
        IntegritySignalName::Reading,
    ];

    /// The spec's name (`ai_text`, `echo_similarity`, `delivery`, `reading`).
    pub fn as_str(self) -> &'static str {
        match self {
            IntegritySignalName::AiText => "ai_text",
            IntegritySignalName::EchoSimilarity => "echo_similarity",
            IntegritySignalName::Delivery => "delivery",
            IntegritySignalName::Reading => "reading",
        }
    }

    /// Parses the spec's name.
    pub fn parse(s: &str) -> Option<IntegritySignalName> {
        IntegritySignalName::ALL
            .into_iter()
            .find(|n| n.as_str() == s.trim())
    }

    /// Display label, e.g. "AI-written text".
    pub fn label(self) -> &'static str {
        match self {
            IntegritySignalName::AiText => "AI-written text",
            IntegritySignalName::EchoSimilarity => "Echo of an LLM answer",
            IntegritySignalName::Delivery => "Delivery",
            IntegritySignalName::Reading => "Reading",
        }
    }
}

/// Level of one signal or of the combined flag.
///
/// `low`, `possible` and `high` are scored levels; the others say why a
/// signal was not scored and are ignored when signals are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum IntegrityLevel {
    Low,
    Possible,
    High,
    /// Too little speech to judge.
    TooShort,
    /// The answer is one of the first answers that form the speaker's
    /// delivery baseline.
    Baseline,
    /// Not enough gaze (face not tracked long enough during the answer).
    NoGaze,
    /// The signal could not be computed (service down, signal off, wrong
    /// embedding model, no question, ...).
    Unavailable,
}

impl IntegrityLevel {
    /// The spec's name (`low`, `possible`, `high`, `too_short`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            IntegrityLevel::Low => "low",
            IntegrityLevel::Possible => "possible",
            IntegrityLevel::High => "high",
            IntegrityLevel::TooShort => "too_short",
            IntegrityLevel::Baseline => "baseline",
            IntegrityLevel::NoGaze => "no_gaze",
            IntegrityLevel::Unavailable => "unavailable",
        }
    }

    /// Parses the spec's name.
    pub fn parse(s: &str) -> Option<IntegrityLevel> {
        [
            IntegrityLevel::Low,
            IntegrityLevel::Possible,
            IntegrityLevel::High,
            IntegrityLevel::TooShort,
            IntegrityLevel::Baseline,
            IntegrityLevel::NoGaze,
            IntegrityLevel::Unavailable,
        ]
        .into_iter()
        .find(|l| l.as_str() == s.trim())
    }

    /// `low`, `possible` or `high`.
    pub fn is_scored(self) -> bool {
        matches!(
            self,
            IntegrityLevel::Low | IntegrityLevel::Possible | IntegrityLevel::High
        )
    }

    /// `possible` or `high`.
    pub fn is_elevated(self) -> bool {
        matches!(self, IntegrityLevel::Possible | IntegrityLevel::High)
    }
}

/// Delivery features of an answer next to the speaker's baseline medians
/// (evidence for the delivery signal).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DeliveryEvidence {
    /// "word" when word timestamps were available, else "utterance".
    pub granularity: String,
    pub word_count: u32,
    /// Seconds from the end of the question to the first word.
    pub latency_s: Option<f64>,
    pub baseline_latency_s: Option<f64>,
    pub rate_wpm: Option<f64>,
    pub baseline_rate_wpm: Option<f64>,
    pub pause_per_min: Option<f64>,
    pub baseline_pause_per_min: Option<f64>,
    pub filler_per_100w: Option<f64>,
    pub baseline_filler_per_100w: Option<f64>,
    /// Baseline answers collected so far (the baseline needs three).
    pub baseline_answers: u32,
}

/// One scored (or not scored) signal with its evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct IntegritySignal {
    pub name: IntegritySignalName,
    pub level: IntegrityLevel,
    /// Thresholds were chosen on labelled data (`spec/integrity/thresholds.json`).
    /// An uncalibrated signal counts at most as `possible`.
    pub calibrated: bool,
    /// The signal's main number: AI-text probability, echo similarity,
    /// delivery index or reading probability.
    pub score: Option<f64>,
    /// Plain-language evidence ("AI-written text score 0.91 (high)").
    pub evidence: Vec<String>,
    /// Why the signal was not scored, for not-scored levels.
    pub note: Option<String>,
    /// Delivery only: the features behind the score.
    pub delivery: Option<DeliveryEvidence>,
}

/// The combined answer-integrity flag of one candidate answer.
///
/// `id` identifies the answer: a re-check of the same answer (the candidate
/// kept talking) replaces the earlier flag (`revision` grows).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct IntegrityFlag {
    pub id: Id,
    pub meeting_id: Id,
    /// The interviewer question the answer responds to, when one was detected.
    pub question_id: Option<Id>,
    pub question_text: Option<String>,
    /// Answer time range, milliseconds since meeting start.
    #[ts(type = "number")]
    pub answer_start_ms: i64,
    #[ts(type = "number")]
    pub answer_end_ms: i64,
    pub answer_words: u32,
    /// Combined level: `low`, `possible`, `high` or `unavailable`.
    pub level: IntegrityLevel,
    /// Signals that were `possible` or `high` after calibration capping.
    pub elevated: Vec<IntegritySignalName>,
    /// One line for the interviewer ("Possible AI-assisted answer — worth a
    /// follow-up"). Never a verdict.
    pub headline: String,
    pub signals: Vec<IntegritySignal>,
    pub revision: u32,
    /// Spec version the levels were computed with.
    pub spec_version: u32,
    #[ts(type = "number")]
    pub created_at: i64,
    #[ts(type = "number")]
    pub updated_at: i64,
}

impl IntegrityFlag {
    pub fn signal(&self, name: IntegritySignalName) -> Option<&IntegritySignal> {
        self.signals.iter().find(|s| s.name == name)
    }
}

/// Whether the local AI sidecar's integrity check can be used (settings page).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct IntegrityServiceView {
    /// The sidecar answered.
    pub reachable: bool,
    /// The AI-written-text model is loaded.
    pub ai_text_ready: bool,
    /// The reading model is loaded.
    pub reading_ready: bool,
    /// Embedding model the sidecar serves (the echo check needs the one named
    /// in the spec thresholds).
    pub embedding_model: Option<String>,
    /// Plain-language status ("Ready", "Local AI is not running", ...).
    pub message: String,
}

// ---------------------------------------------------------------------------
// Sidecar contract
// ---------------------------------------------------------------------------

/// Live gaze state reported by the sidecar.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GazeStatus {
    /// "screen", "browser" or "none".
    pub source: String,
    pub capturing_screen: bool,
    pub frames_per_second: f64,
    pub face_found_fraction: f64,
    pub tracked_fraction: f64,
    pub error: Option<String>,
}

/// `GET /v1/integrity/status` (+ the sidecar's embedding model from `/health`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IntegrityServiceStatus {
    pub ai_text_ready: bool,
    pub reading_ready: bool,
    pub ai_text_error: Option<String>,
    pub reading_error: Option<String>,
    pub gaze: GazeStatus,
    pub embedding_model: Option<String>,
}

/// `POST /v1/integrity/echo`: does the transcript read like AI-written text?
#[derive(Debug, Clone, PartialEq)]
pub struct AiTextResult {
    pub probability: f64,
    /// `low`, `possible`, `high` or `too_short`.
    pub level: IntegrityLevel,
    pub words: u32,
    pub model: String,
}

/// `POST /v1/integrity/reading`: do the eyes move like reading?
#[derive(Debug, Clone, PartialEq)]
pub struct ReadingResult {
    pub probability: Option<f64>,
    /// `low`, `possible`, `high`, or `no_gaze` when there was not enough
    /// tracked gaze in the window.
    pub level: IntegrityLevel,
    /// Fraction of frames with a tracked face.
    pub valid_fraction: Option<f64>,
    /// Number of 10-second windows scored.
    pub windows: u32,
    pub seconds: f64,
}

/// The local AI sidecar's integrity endpoints (`/v1/integrity/*`).
#[async_trait]
pub trait IntegrityService: Send + Sync {
    fn info(&self) -> ProviderInfo;

    /// Which models are ready, live gaze state and the sidecar's embedding model.
    async fn status(
        &self,
        cancel: CancellationToken,
    ) -> Result<IntegrityServiceStatus, ProviderError>;

    /// Scores an answer's transcript (`/echo`, the `ai_text` signal).
    async fn ai_text(
        &self,
        text: String,
        cancel: CancellationToken,
    ) -> Result<AiTextResult, ProviderError>;

    /// Starts capturing `region` (absolute screen coordinates) for gaze at 10 fps.
    async fn start_gaze(
        &self,
        region: Rect,
        cancel: CancellationToken,
    ) -> Result<GazeStatus, ProviderError>;

    /// Stops the screen capture for gaze.
    async fn stop_gaze(&self, cancel: CancellationToken) -> Result<GazeStatus, ProviderError>;

    /// Forgets the gaze samples kept so far.
    async fn reset_gaze(&self, cancel: CancellationToken) -> Result<GazeStatus, ProviderError>;

    /// Reading score for a time window (Unix epoch seconds, span at most 600 s).
    async fn reading(
        &self,
        start_epoch_s: f64,
        end_epoch_s: f64,
        cancel: CancellationToken,
    ) -> Result<ReadingResult, ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_levels_round_trip() {
        for n in IntegritySignalName::ALL {
            assert_eq!(IntegritySignalName::parse(n.as_str()), Some(n));
            assert_eq!(serde_json::to_value(n).unwrap(), n.as_str());
        }
        for s in [
            "low",
            "possible",
            "high",
            "too_short",
            "baseline",
            "no_gaze",
            "unavailable",
        ] {
            let l = IntegrityLevel::parse(s).unwrap();
            assert_eq!(l.as_str(), s);
            assert_eq!(serde_json::to_value(l).unwrap(), s);
        }
        assert!(IntegrityLevel::Low.is_scored());
        assert!(!IntegrityLevel::Baseline.is_scored());
        assert!(IntegrityLevel::High.is_elevated());
        assert!(!IntegrityLevel::Low.is_elevated());
        assert_eq!(IntegrityLevel::parse("not_enough_gaze"), None);
    }

    #[test]
    fn flag_json_shape() {
        let flag = IntegrityFlag {
            id: "a1".into(),
            meeting_id: "m1".into(),
            question_id: None,
            question_text: Some("Why Redis?".into()),
            answer_start_ms: 1000,
            answer_end_ms: 9000,
            answer_words: 30,
            level: IntegrityLevel::Possible,
            elevated: vec![IntegritySignalName::AiText],
            headline: "Possible AI-assisted answer — worth a follow-up".into(),
            signals: vec![IntegritySignal {
                name: IntegritySignalName::AiText,
                level: IntegrityLevel::High,
                calibrated: true,
                score: Some(0.91),
                evidence: vec!["AI-written text score 0.91 (high)".into()],
                note: None,
                delivery: None,
            }],
            revision: 1,
            spec_version: 1,
            created_at: 5,
            updated_at: 6,
        };
        let v = serde_json::to_value(&flag).unwrap();
        assert_eq!(v["answerStartMs"], 1000);
        assert_eq!(v["elevated"][0], "ai_text");
        assert_eq!(v["signals"][0]["level"], "high");
        let back: IntegrityFlag = serde_json::from_value(v).unwrap();
        assert_eq!(back, flag);
        assert!(back.signal(IntegritySignalName::AiText).is_some());
        assert!(back.signal(IntegritySignalName::Reading).is_none());
    }
}
