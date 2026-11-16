//! Plain-language evidence for answer-integrity signals and flags.
//!
//! The wording follows the repository rule: a flag is a signal for the
//! interviewer, never a verdict ("Possible AI-assisted answer — worth a
//! follow-up", never "cheating detected"). Uncalibrated signals say so.

use openmeet_core::integrity::{
    DeliveryEvidence, IntegrityLevel, IntegritySignal, IntegritySignalName,
};

use super::delivery::{BASELINE_ANSWERS, DeliveryBaseline, DeliveryFeatures, DeliveryScore};
use super::echo::EchoResult;
use super::fusion::FusionInput;
use crate::timefmt::clock;

/// Headline of a combined flag.
pub fn headline(level: IntegrityLevel) -> &'static str {
    match level {
        IntegrityLevel::High => "Several signals point to AI assistance — worth a follow-up",
        IntegrityLevel::Possible => "Possible AI-assisted answer — worth a follow-up",
        IntegrityLevel::Low => "Nothing unusual in this answer",
        _ => "Not enough to check this answer",
    }
}

/// "Answer at 23:54–25:00" (meeting-relative times).
pub fn answer_range(start_ms: i64, end_ms: i64) -> String {
    format!("Answer at {}–{}", clock(start_ms), clock(end_ms))
}

/// Level and calibration in brackets: "(high)", "(possible, uncalibrated)".
fn bracket(level: IntegrityLevel, calibrated: bool) -> String {
    if calibrated {
        format!("({})", level.as_str())
    } else {
        format!("({}, uncalibrated)", level.as_str())
    }
}

/// A signal that was not scored, with the reason.
pub fn not_scored(
    name: IntegritySignalName,
    level: IntegrityLevel,
    calibrated: bool,
    note: impl Into<String>,
) -> IntegritySignal {
    let note = note.into();
    IntegritySignal {
        name,
        level,
        calibrated,
        score: None,
        evidence: vec![note.clone()],
        note: Some(note),
        delivery: None,
    }
}

/// The AI-written-text signal from the sidecar's probability and level.
pub fn ai_text_signal(
    probability: f64,
    level: IntegrityLevel,
    calibrated: bool,
) -> IntegritySignal {
    if !level.is_scored() {
        return not_scored(
            IntegritySignalName::AiText,
            level,
            calibrated,
            match level {
                IntegrityLevel::TooShort => "Too short for the AI-written text check",
                _ => "AI-written text check unavailable",
            },
        );
    }
    IntegritySignal {
        name: IntegritySignalName::AiText,
        level,
        calibrated,
        score: Some(probability),
        evidence: vec![format!(
            "AI-written text score {probability:.2} {}",
            bracket(level, calibrated)
        )],
        note: None,
        delivery: None,
    }
}

/// The echo-similarity signal. `llm_local` decides "a local LLM" vs "a
/// cloud LLM" in the wording.
pub fn echo_signal(result: &EchoResult, llm_local: bool) -> IntegritySignal {
    let who = if llm_local {
        "a local LLM's answer"
    } else {
        "a cloud LLM's answer"
    };
    let Some(sim) = result.similarity.filter(|_| result.level.is_scored()) else {
        return not_scored(
            IntegritySignalName::EchoSimilarity,
            result.level,
            result.calibrated,
            "No reference answer to compare with",
        );
    };
    let cal = if result.calibrated {
        String::new()
    } else {
        " (uncalibrated)".to_string()
    };
    let line = if result.level.is_elevated() {
        format!("Close to {who}: similarity {sim:.2}{cal}")
    } else {
        format!("Not close to {who}: similarity {sim:.2}{cal}")
    };
    IntegritySignal {
        name: IntegritySignalName::EchoSimilarity,
        level: result.level,
        calibrated: result.calibrated,
        score: Some(sim),
        evidence: vec![line],
        note: None,
        delivery: None,
    }
}

/// The delivery signal with its feature evidence. `baseline_answers` is the
/// number of baseline answers recorded before this one.
pub fn delivery_signal(
    features: Option<&DeliveryFeatures>,
    score: &DeliveryScore,
    baseline: Option<&DeliveryBaseline>,
    baseline_answers: usize,
) -> IntegritySignal {
    let med = |name: &str| baseline.and_then(|b| b.get(name)).map(|b| b.median);
    let evidence_numbers = features.map(|f| DeliveryEvidence {
        granularity: f.granularity.as_str().to_string(),
        word_count: f.word_count as u32,
        latency_s: f.latency_s,
        baseline_latency_s: med("latency_s"),
        rate_wpm: f.rate_wpm,
        baseline_rate_wpm: med("rate_wpm"),
        pause_per_min: f.pause_per_min,
        baseline_pause_per_min: med("pause_per_min"),
        filler_per_100w: f.filler_per_100w,
        baseline_filler_per_100w: med("filler_per_100w"),
        baseline_answers: baseline_answers as u32,
    });
    let not_scored_note = match score.level {
        IntegrityLevel::TooShort => Some("Too short to judge delivery".to_string()),
        IntegrityLevel::Baseline => Some(format!(
            "Learning this candidate's usual delivery (answer {} of {BASELINE_ANSWERS})",
            (baseline_answers + 1).min(BASELINE_ANSWERS)
        )),
        IntegrityLevel::Unavailable => Some("No timing to compare with the baseline".to_string()),
        _ => None,
    };
    if let Some(note) = not_scored_note {
        let mut s = not_scored(
            IntegritySignalName::Delivery,
            score.level,
            score.calibrated,
            note,
        );
        s.delivery = evidence_numbers;
        return s;
    }
    let mut lines = vec![format!(
        "Delivery index {:.1} {}",
        score.index.unwrap_or(0.0),
        bracket(score.level, score.calibrated)
    )];
    if let Some(f) = features {
        let z = |name: &str| score.z.get(name).copied().unwrap_or(0.0);
        if let (Some(x), Some(m)) = (f.latency_s, med("latency_s"))
            && z("latency_s") >= 1.0
            && x > m
        {
            lines.push(format!("Started {:.1} s later than usual", x - m));
        }
        if let (Some(x), Some(m)) = (f.rate_wpm, med("rate_wpm"))
            && z("rate_wpm") >= 1.0
        {
            lines.push(format!(
                "Spoke faster than usual ({x:.0} vs {m:.0} words/min)"
            ));
        }
        if let (Some(x), Some(m)) = (f.pause_per_min, med("pause_per_min"))
            && z("pause_per_min") <= -1.0
        {
            lines.push(format!(
                "Paused less than usual ({x:.1} vs {m:.1} pauses/min)"
            ));
        }
        if let (Some(x), Some(m)) = (f.filler_per_100w, med("filler_per_100w"))
            && z("filler_per_100w") <= -1.0
        {
            lines.push(format!(
                "Fewer filler words than usual ({x:.1} vs {m:.1} per 100 words)"
            ));
        }
        if lines.len() == 1 && score.level == IntegrityLevel::Low {
            lines.push("Timing close to this candidate's usual".to_string());
        }
    }
    IntegritySignal {
        name: IntegritySignalName::Delivery,
        level: score.level,
        calibrated: score.calibrated,
        score: score.index,
        evidence: lines,
        note: None,
        delivery: evidence_numbers,
    }
}

/// The reading signal from the sidecar's probability and level.
pub fn reading_signal(
    probability: Option<f64>,
    level: IntegrityLevel,
    calibrated: bool,
) -> IntegritySignal {
    let Some(p) = probability.filter(|_| level.is_scored()) else {
        return not_scored(
            IntegritySignalName::Reading,
            level,
            calibrated,
            match level {
                IntegrityLevel::NoGaze => {
                    "Not enough gaze: the face was not tracked long enough during the answer"
                }
                _ => "Reading check unavailable",
            },
        );
    };
    let line = if level.is_elevated() {
        format!(
            "Eye movements look like reading: score {p:.2} {}",
            bracket(level, calibrated)
        )
    } else {
        format!("Eye movements do not look like reading: score {p:.2}")
    };
    IntegritySignal {
        name: IntegritySignalName::Reading,
        level,
        calibrated,
        score: Some(p),
        evidence: vec![line],
        note: None,
        delivery: None,
    }
}

/// Fusion inputs of a set of signals.
pub fn fusion_inputs(signals: &[IntegritySignal]) -> Vec<FusionInput> {
    signals
        .iter()
        .map(|s| FusionInput {
            name: s.name,
            level: s.level,
            calibrated: s.calibrated,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrity::delivery::{SpeechUnit, delivery_features};
    use crate::integrity::{DeliveryTracker, spec_thresholds};

    #[test]
    fn headlines_are_signals_not_verdicts() {
        for level in [
            IntegrityLevel::High,
            IntegrityLevel::Possible,
            IntegrityLevel::Low,
            IntegrityLevel::Unavailable,
        ] {
            let h = headline(level).to_lowercase();
            assert!(!h.contains("cheat"), "{h}");
            assert!(!h.contains("detected"), "{h}");
        }
        assert_eq!(
            headline(IntegrityLevel::Possible),
            "Possible AI-assisted answer — worth a follow-up"
        );
        assert_eq!(answer_range(1_434_000, 1_500_000), "Answer at 23:54–25:00");
    }

    #[test]
    fn evidence_lines() {
        let s = ai_text_signal(0.912, IntegrityLevel::High, true);
        assert_eq!(s.evidence, vec!["AI-written text score 0.91 (high)"]);
        assert_eq!(s.score, Some(0.912));
        let s = ai_text_signal(0.2, IntegrityLevel::TooShort, true);
        assert_eq!(s.score, None);
        assert!(s.note.is_some());

        let e = EchoResult {
            level: IntegrityLevel::Possible,
            similarity: Some(0.881),
            calibrated: false,
        };
        assert_eq!(
            echo_signal(&e, true).evidence,
            vec!["Close to a local LLM's answer: similarity 0.88 (uncalibrated)"]
        );
        assert!(echo_signal(&e, false).evidence[0].contains("cloud LLM"));

        let r = reading_signal(Some(0.87), IntegrityLevel::High, true);
        assert!(r.evidence[0].starts_with("Eye movements look like reading"));
        let r = reading_signal(None, IntegrityLevel::NoGaze, true);
        assert_eq!(r.level, IntegrityLevel::NoGaze);
        assert!(r.note.unwrap().contains("gaze"));
    }

    #[test]
    fn delivery_evidence_names_what_changed() {
        let th = spec_thresholds().delivery.clone();
        let mut tr = DeliveryTracker::new();
        let natural = "well um I think so we tried a few things and uh it mostly worked out fine";
        for i in 0..3 {
            let s = i as f64 * 30.0;
            let u = SpeechUnit {
                start: s + 1.0,
                end: s + 7.0,
                text: natural.into(),
                words: None,
            };
            let f = delivery_features(Some(s), &[u]);
            let before = tr.baseline_answers();
            let score = tr.observe(f.as_ref(), &th);
            let sig = delivery_signal(f.as_ref(), &score, tr.baseline(), before);
            assert_eq!(sig.level, IntegrityLevel::Baseline);
            assert!(sig.evidence[0].contains(&format!("answer {} of 3", i + 1)));
        }
        let read = SpeechUnit {
            start: 104.6,
            end: 110.0,
            text: "consistent hashing distributes keys across nodes so that adding a node only remaps a small fraction of keys".into(),
            words: None,
        };
        let f = delivery_features(Some(100.0), &[read]);
        let score = tr.score(f.as_ref(), &th);
        let sig = delivery_signal(f.as_ref(), &score, tr.baseline(), 3);
        assert!(sig.level.is_elevated(), "{sig:?}");
        assert!(
            sig.evidence
                .iter()
                .any(|l| l.starts_with("Started 3.6 s later than usual")),
            "{:?}",
            sig.evidence
        );
        assert!(sig.evidence.iter().any(|l| l.starts_with("Spoke faster")));
        let d = sig.delivery.unwrap();
        assert_eq!(d.granularity, "utterance");
        assert_eq!(d.baseline_answers, 3);
        assert!(d.baseline_latency_s.is_some());
    }
}
