//! Echo similarity: how close is the candidate's answer to what an LLM
//! answers to the same question? The highest cosine between the answer's
//! embedding and any reference answer's (spec `echo_similarity`).

use super::EchoThresholds;
use openmeet_core::integrity::IntegrityLevel;

/// Echo similarity of one answer.
#[derive(Debug, Clone, PartialEq)]
pub struct EchoResult {
    /// `low`, `possible`, `high` or `unavailable`.
    pub level: IntegrityLevel,
    /// Highest cosine, `None` without references.
    pub similarity: Option<f64>,
    /// The echo thresholds are calibrated.
    pub calibrated: bool,
}

/// Cosine similarity in f64; 0 when either vector is all zeros. Extra
/// components of the longer vector are ignored (as Python's `zip`).
pub fn cosine64(a: &[f64], b: &[f64]) -> f64 {
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / (na * nb)
}

/// Highest cosine between `answer` and any of `references`, with its level.
/// No references → `unavailable`.
pub fn echo_similarity(
    answer: &[f64],
    references: &[Vec<f64>],
    thresholds: &EchoThresholds,
) -> EchoResult {
    let Some(sim) = references
        .iter()
        .map(|r| cosine64(answer, r))
        .reduce(f64::max)
    else {
        return EchoResult {
            level: IntegrityLevel::Unavailable,
            similarity: None,
            calibrated: thresholds.calibrated,
        };
    };
    let level = if sim >= thresholds.high {
        IntegrityLevel::High
    } else if sim >= thresholds.possible {
        IntegrityLevel::Possible
    } else {
        IntegrityLevel::Low
    };
    EchoResult {
        level,
        similarity: Some(sim),
        calibrated: thresholds.calibrated,
    }
}

/// Widens f32 embeddings (as providers return them) for [`echo_similarity`].
pub fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_edge_cases() {
        assert_eq!(cosine64(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert!((cosine64(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((cosine64(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-12);
        assert_eq!(widen(&[0.5_f32, -1.0]), vec![0.5, -1.0]);
    }

    #[test]
    fn levels_follow_the_spec_thresholds() {
        let t = super::super::spec_thresholds().echo_similarity.clone();
        let r = echo_similarity(&[1.0, 0.0], &[vec![1.0, 0.0]], &t);
        assert_eq!(r.level, IntegrityLevel::High);
        assert_eq!(r.calibrated, t.calibrated);
        let r = echo_similarity(&[1.0, 0.0], &[vec![0.0, 1.0]], &t);
        assert_eq!(r.level, IntegrityLevel::Low);
        let r = echo_similarity(&[1.0, 0.0], &[], &t);
        assert_eq!(r.level, IntegrityLevel::Unavailable);
        assert_eq!(r.similarity, None);
    }
}
