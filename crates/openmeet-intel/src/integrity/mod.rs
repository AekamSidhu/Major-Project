//! Answer-integrity signals computed in Rust: delivery (timing against the
//! speaker's own baseline), echo similarity (cosine between the answer and
//! LLM reference answers) and the fusion of all signals into one flag.
//!
//! Every definition follows `spec/integrity/README.md` and `reference.py`;
//! the parity tests in `tests/integrity_spec.rs` reproduce the spec fixtures
//! to within 1e-6. Level thresholds and calibration flags are read from
//! `spec/integrity/thresholds.json` (embedded at compile time), never
//! hard-coded here. The AI-text and reading signals come from the local AI
//! sidecar; [`evidence`] turns all four into plain-language evidence.

pub mod delivery;
pub mod echo;
pub mod evidence;
pub mod fusion;
pub mod text;

use std::sync::OnceLock;

use serde::Deserialize;

pub use delivery::{
    DeliveryBaseline, DeliveryFeatures, DeliveryScore, DeliveryTracker, FeatureBaseline,
    Granularity, SpeechUnit, SpeechWord, delivery_baseline, delivery_features, delivery_score,
};
pub use echo::{EchoResult, cosine64, echo_similarity};
pub use fusion::{FusedFlag, FusionInput, fuse};
pub use text::to_transcript;

/// Version of the shared definitions this implementation follows.
pub const SPEC_VERSION: u32 = 1;

/// The spec's `thresholds.json`, embedded at compile time so both sides of
/// the spec always read the same numbers.
pub const THRESHOLDS_JSON: &str = include_str!("../../../../spec/integrity/thresholds.json");

/// Echo-similarity thresholds (`thresholds.json` → `echo_similarity`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EchoThresholds {
    /// The only embedding model the thresholds hold for.
    pub embedding_model: String,
    /// LLM the thresholds were set with.
    pub reference_model: String,
    /// Reference answers to request from the LLM per question.
    pub references: u32,
    /// Similarity at which the level is `possible`.
    pub possible: f64,
    /// Similarity at which the level is `high`.
    pub high: f64,
    /// Thresholds were measured on labelled data.
    pub calibrated: bool,
    /// Where the numbers come from.
    #[serde(default)]
    pub source: String,
}

/// Delivery-index thresholds (`thresholds.json` → `delivery`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DeliveryThresholds {
    /// Delivery index at which the level is `possible`.
    pub possible_index: f64,
    /// Delivery index at which the level is `high`.
    pub high_index: f64,
    /// Thresholds were measured on labelled data.
    pub calibrated: bool,
    /// Where the numbers come from.
    #[serde(default)]
    pub source: String,
}

/// Calibration of a signal whose levels come from the sidecar.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SignalCalibration {
    /// Thresholds were measured on labelled data.
    pub calibrated: bool,
    /// Where the numbers come from.
    #[serde(default)]
    pub source: String,
}

/// Parsed `thresholds.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct IntegrityThresholds {
    /// Spec version the file belongs to.
    pub spec_version: u32,
    /// Echo similarity.
    pub echo_similarity: EchoThresholds,
    /// Delivery.
    pub delivery: DeliveryThresholds,
    /// AI-written text (levels from the sidecar).
    pub ai_text: SignalCalibration,
    /// Reading (levels from the sidecar).
    pub reading: SignalCalibration,
}

impl IntegrityThresholds {
    /// Parses a `thresholds.json` document.
    pub fn from_json(json: &str) -> Result<IntegrityThresholds, String> {
        serde_json::from_str(json).map_err(|e| format!("invalid integrity thresholds: {e}"))
    }

    /// Whether the thresholds of `name` were calibrated on labelled data.
    pub fn calibrated(&self, name: openmeet_core::integrity::IntegritySignalName) -> bool {
        use openmeet_core::integrity::IntegritySignalName as N;
        match name {
            N::AiText => self.ai_text.calibrated,
            N::EchoSimilarity => self.echo_similarity.calibrated,
            N::Delivery => self.delivery.calibrated,
            N::Reading => self.reading.calibrated,
        }
    }
}

/// The thresholds embedded from `spec/integrity/thresholds.json`.
///
/// The file is part of the source tree and covered by tests, so a parse
/// failure is a build defect, not a runtime condition.
pub fn spec_thresholds() -> &'static IntegrityThresholds {
    static PARSED: OnceLock<IntegrityThresholds> = OnceLock::new();
    PARSED.get_or_init(|| {
        IntegrityThresholds::from_json(THRESHOLDS_JSON)
            .expect("spec/integrity/thresholds.json is embedded and tested")
    })
}

/// Whether an embedding model is the one the echo thresholds hold for
/// (case-insensitive, surrounding whitespace ignored).
pub fn is_echo_embedding_model(model: &str, thresholds: &IntegrityThresholds) -> bool {
    let want = thresholds.echo_similarity.embedding_model.trim();
    !want.is_empty() && model.trim().eq_ignore_ascii_case(want)
}

/// Whether an LLM model is the one the echo thresholds were measured with
/// (case-insensitive; an Ollama `:latest` tag is the same model). With any
/// other model the echo signal is reported as uncalibrated.
pub fn is_reference_model(model: &str, thresholds: &IntegrityThresholds) -> bool {
    let norm = |m: &str| {
        let m = m.trim().to_ascii_lowercase();
        m.strip_suffix(":latest").map(str::to_string).unwrap_or(m)
    };
    let want = norm(&thresholds.echo_similarity.reference_model);
    !want.is_empty() && norm(model) == want
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::integrity::IntegritySignalName;

    #[test]
    fn embedded_thresholds_parse_and_match_this_spec_version() {
        let t = spec_thresholds();
        assert_eq!(t.spec_version, SPEC_VERSION);
        assert!(t.echo_similarity.references >= 1);
        assert!(t.echo_similarity.possible <= t.echo_similarity.high);
        assert!(t.delivery.possible_index <= t.delivery.high_index);
        assert_eq!(
            t.calibrated(IntegritySignalName::Delivery),
            t.delivery.calibrated
        );
    }

    #[test]
    fn echo_embedding_model_match() {
        let t = spec_thresholds();
        let model = t.echo_similarity.embedding_model.clone();
        assert!(is_echo_embedding_model(&model, t));
        assert!(is_echo_embedding_model(
            &format!("  {}", model.to_uppercase()),
            t
        ));
        assert!(!is_echo_embedding_model("hash-384", t));
        assert!(!is_echo_embedding_model("", t));
    }

    #[test]
    fn reference_model_match() {
        let t = spec_thresholds();
        let model = t.echo_similarity.reference_model.clone();
        assert!(is_reference_model(&model, t));
        assert!(is_reference_model(&format!("{model}:latest"), t));
        assert!(is_reference_model(&model.to_uppercase(), t));
        assert!(!is_reference_model("gpt-5-mini", t));
        assert!(!is_reference_model(&format!("{model}:70b"), t));
    }
}
