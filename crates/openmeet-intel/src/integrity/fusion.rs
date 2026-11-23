//! The combined flag (spec "The combined flag").
//!
//! Not-scored levels are ignored; an uncalibrated signal counts at most as
//! `possible`. The flag is `high` only when a calibrated signal is `high`
//! and at least one other signal is `possible` or `high` — one signal on its
//! own never makes it `high`.

use openmeet_core::integrity::{IntegrityLevel, IntegritySignalName};

/// One signal's contribution to the flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionInput {
    /// Which signal.
    pub name: IntegritySignalName,
    /// Its level (not-scored levels are ignored).
    pub level: IntegrityLevel,
    /// Its thresholds are calibrated.
    pub calibrated: bool,
}

/// The combined level and the signals that raised it (sorted by spec name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedFlag {
    /// `low`, `possible`, `high` or `unavailable`.
    pub level: IntegrityLevel,
    /// Signals counted as `possible` or `high`.
    pub elevated: Vec<IntegritySignalName>,
}

fn rank(level: IntegrityLevel) -> Option<u8> {
    match level {
        IntegrityLevel::Low => Some(0),
        IntegrityLevel::Possible => Some(1),
        IntegrityLevel::High => Some(2),
        _ => None,
    }
}

/// One flag from per-signal levels.
pub fn fuse(signals: &[FusionInput]) -> FusedFlag {
    let effective: Vec<(IntegritySignalName, u8)> = signals
        .iter()
        .filter_map(|s| rank(s.level).map(|r| (s.name, r.min(if s.calibrated { 2 } else { 1 }))))
        .collect();
    if effective.is_empty() {
        return FusedFlag {
            level: IntegrityLevel::Unavailable,
            elevated: Vec::new(),
        };
    }
    let mut elevated: Vec<IntegritySignalName> = effective
        .iter()
        .filter(|(_, r)| *r >= 1)
        .map(|(n, _)| *n)
        .collect();
    elevated.sort_by_key(|n| n.as_str());
    let level = if effective.iter().any(|(_, r)| *r == 2) && elevated.len() >= 2 {
        IntegrityLevel::High
    } else if !elevated.is_empty() {
        IntegrityLevel::Possible
    } else {
        IntegrityLevel::Low
    };
    FusedFlag { level, elevated }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(name: IntegritySignalName, level: IntegrityLevel, calibrated: bool) -> FusionInput {
        FusionInput {
            name,
            level,
            calibrated,
        }
    }

    #[test]
    fn one_signal_never_makes_the_flag_high() {
        let f = fuse(&[s(IntegritySignalName::AiText, IntegrityLevel::High, true)]);
        assert_eq!(f.level, IntegrityLevel::Possible);
        let f = fuse(&[
            s(IntegritySignalName::Reading, IntegrityLevel::High, true),
            s(IntegritySignalName::AiText, IntegrityLevel::Possible, true),
        ]);
        assert_eq!(f.level, IntegrityLevel::High);
        assert_eq!(
            f.elevated,
            vec![IntegritySignalName::AiText, IntegritySignalName::Reading]
        );
    }

    #[test]
    fn uncalibrated_highs_cap_at_possible() {
        let f = fuse(&[
            s(IntegritySignalName::Delivery, IntegrityLevel::High, false),
            s(
                IntegritySignalName::EchoSimilarity,
                IntegrityLevel::High,
                false,
            ),
        ]);
        assert_eq!(f.level, IntegrityLevel::Possible);
        assert_eq!(fuse(&[]).level, IntegrityLevel::Unavailable);
    }
}
