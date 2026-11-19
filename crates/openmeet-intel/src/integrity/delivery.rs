//! Delivery: does an answer's timing look read rather than spoken freely,
//! compared with the same speaker's earlier answers?
//!
//! Mirrors `spec/integrity/reference.py` (`delivery_features`,
//! `delivery_baseline`, `delivery_score`, `score_session`). The definition
//! constants below are part of the spec and are checked against the
//! `constants` block of `delivery_cases.json` by the parity tests; the level
//! thresholds come from `thresholds.json`.

use std::collections::BTreeMap;

use super::DeliveryThresholds;
use openmeet_core::integrity::IntegrityLevel;

/// A gap shorter than this between speech units is not a pause.
pub const PAUSE_MIN_S: f64 = 0.3;
/// Below this much speech, `rate_wpm` is undefined.
pub const MIN_SPEAKING_S: f64 = 1.0;
/// Below this span, `pause_per_min` is undefined.
pub const MIN_SPAN_S: f64 = 5.0;
/// Answers with fewer words are `too_short`.
pub const MIN_WORDS: usize = 10;
/// The first N answers with at least [`MIN_WORDS`] words form the baseline.
pub const BASELINE_ANSWERS: usize = 3;
/// Scales a median absolute deviation to a standard deviation.
pub const MAD_TO_SD: f64 = 1.4826;
/// Single-token fillers.
pub const FILLERS: [&str; 9] = ["ah", "er", "erm", "hmm", "mm", "uh", "uhm", "um", "umm"];
/// Two-token fillers.
pub const FILLER_BIGRAMS: [(&str, &str); 2] = [("i", "mean"), ("you", "know")];

/// Weighted features in the spec's order: (name, weight, floor of the scale).
/// Read-aloud answers tend to start late, run faster and pause and hesitate
/// less than the same person's baseline.
pub const WEIGHTS: [(&str, f64, f64); 4] = [
    ("latency_s", 1.0, 0.5),
    ("rate_wpm", 0.5, 15.0),
    ("pause_per_min", -0.5, 3.0),
    ("filler_per_100w", -0.5, 1.0),
];

/// One recognized word with times in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechWord {
    /// The word as recognized (may carry a leading space).
    pub word: String,
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
}

/// One utterance of the answerer, times in seconds on one clock.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechUnit {
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
    /// Transcript of the utterance.
    pub text: String,
    /// Word timestamps; used only when every utterance has (non-empty) words.
    pub words: Option<Vec<SpeechWord>>,
}

/// Whether features came from word or utterance timings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Every utterance had word timestamps.
    Word,
    /// Utterance start and end times only.
    Utterance,
}

impl Granularity {
    /// The spec's name (`word` / `utterance`).
    pub fn as_str(self) -> &'static str {
        match self {
            Granularity::Word => "word",
            Granularity::Utterance => "utterance",
        }
    }
}

/// Delivery features of one answer.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveryFeatures {
    /// Word or utterance timings.
    pub granularity: Granularity,
    /// Tokens in the answer.
    pub word_count: usize,
    /// Sum of utterance durations.
    pub speaking_s: f64,
    /// Last utterance end minus first utterance start.
    pub span_s: f64,
    /// First utterance start minus the question end (at least 0).
    pub latency_s: Option<f64>,
    /// Words per minute of speech.
    pub rate_wpm: Option<f64>,
    /// Pauses (gaps of at least [`PAUSE_MIN_S`]) per minute of span.
    pub pause_per_min: Option<f64>,
    /// Mean pause length (evidence only).
    pub mean_pause_s: f64,
    /// Fillers per 100 words.
    pub filler_per_100w: Option<f64>,
}

impl DeliveryFeatures {
    /// A weighted feature by its spec name.
    pub fn get(&self, name: &str) -> Option<f64> {
        match name {
            "latency_s" => self.latency_s,
            "rate_wpm" => self.rate_wpm,
            "pause_per_min" => self.pause_per_min,
            "filler_per_100w" => self.filler_per_100w,
            _ => None,
        }
    }
}

/// Median and robust scale of one feature over the baseline answers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureBaseline {
    /// Median over the baseline answers.
    pub median: f64,
    /// max(1.4826 × MAD, floor).
    pub scale: f64,
}

/// Per-feature baseline; `None` for a feature with fewer than two values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeliveryBaseline {
    /// Keyed by the spec's feature name.
    pub features: BTreeMap<&'static str, Option<FeatureBaseline>>,
}

impl DeliveryBaseline {
    /// Baseline of one feature, `None` when it has none.
    pub fn get(&self, name: &str) -> Option<FeatureBaseline> {
        self.features.get(name).copied().flatten()
    }
}

/// Delivery level of one answer.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliveryScore {
    /// `low`, `possible`, `high` or a not-scored level.
    pub level: IntegrityLevel,
    /// Weighted mean z-score (scored levels only).
    pub index: Option<f64>,
    /// z-score of every feature that had a value and a baseline.
    pub z: BTreeMap<&'static str, f64>,
    /// The delivery thresholds are calibrated.
    pub calibrated: bool,
}

/// Lower-case word tokens: runs of `a-z`, `0-9` and apostrophes.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn by_time(a: &(f64, f64), b: &(f64, f64)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
}

/// Features of one answer; `None` when there is no speech.
///
/// `question_end` is when the question finished (seconds on the same clock
/// as the utterances), `None` when unknown.
pub fn delivery_features(
    question_end: Option<f64>,
    utterances: &[SpeechUnit],
) -> Option<DeliveryFeatures> {
    let mut utts: Vec<&SpeechUnit> = utterances.iter().collect();
    utts.sort_by(|a, b| by_time(&(a.start, a.end), &(b.start, b.end)));
    let first = *utts.first()?;
    let use_words = utts
        .iter()
        .all(|u| u.words.as_ref().is_some_and(|w| !w.is_empty()));
    let (units, toks): (Vec<(f64, f64)>, Vec<String>) = if use_words {
        let mut units: Vec<(f64, f64)> = utts
            .iter()
            .flat_map(|u| u.words.iter().flatten().map(|w| (w.start, w.end)))
            .collect();
        units.sort_by(by_time);
        let toks = utts
            .iter()
            .flat_map(|u| u.words.iter().flatten().flat_map(|w| tokens(&w.word)))
            .collect();
        (units, toks)
    } else {
        let units = utts.iter().map(|u| (u.start, u.end)).collect();
        let toks = utts.iter().flat_map(|u| tokens(&u.text)).collect();
        (units, toks)
    };
    let n = toks.len();
    let mut fillers = toks
        .iter()
        .filter(|t| FILLERS.contains(&t.as_str()))
        .count();
    fillers += toks
        .windows(2)
        .filter(|w| FILLER_BIGRAMS.contains(&(w[0].as_str(), w[1].as_str())))
        .count();
    let speaking_s: f64 = utts.iter().map(|u| (u.end - u.start).max(0.0)).sum();
    let last_end = utts.iter().map(|u| u.end).fold(f64::NEG_INFINITY, f64::max);
    let span_s = last_end - first.start;
    let pauses: Vec<f64> = units
        .windows(2)
        .map(|w| w[1].0 - w[0].1)
        .filter(|gap| *gap >= PAUSE_MIN_S)
        .collect();
    Some(DeliveryFeatures {
        granularity: if use_words {
            Granularity::Word
        } else {
            Granularity::Utterance
        },
        word_count: n,
        speaking_s,
        span_s,
        latency_s: question_end.map(|q| (first.start - q).max(0.0)),
        rate_wpm: (speaking_s >= MIN_SPEAKING_S).then(|| n as f64 / speaking_s * 60.0),
        pause_per_min: (span_s >= MIN_SPAN_S).then(|| pauses.len() as f64 / (span_s / 60.0)),
        mean_pause_s: if pauses.is_empty() {
            0.0
        } else {
            pauses.iter().sum::<f64>() / pauses.len() as f64
        },
        filler_per_100w: (n > 0).then(|| 100.0 * fillers as f64 / n as f64),
    })
}

/// Median as Python's `statistics.median`: the mean of the two middle values
/// for an even count.
fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Median and robust scale per weighted feature over the baseline answers.
pub fn delivery_baseline(features: &[DeliveryFeatures]) -> DeliveryBaseline {
    let mut out = DeliveryBaseline::default();
    for (name, _, floor) in WEIGHTS {
        let xs: Vec<f64> = features.iter().filter_map(|f| f.get(name)).collect();
        if xs.len() < 2 {
            out.features.insert(name, None);
            continue;
        }
        let med = median(&xs);
        let deviations: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
        let mad = median(&deviations);
        out.features.insert(
            name,
            Some(FeatureBaseline {
                median: med,
                scale: (MAD_TO_SD * mad).max(floor),
            }),
        );
    }
    out
}

/// Delivery index and level of one answer against the speaker's baseline.
pub fn delivery_score(
    features: Option<&DeliveryFeatures>,
    baseline: Option<&DeliveryBaseline>,
    thresholds: &DeliveryThresholds,
) -> DeliveryScore {
    let not_scored = |level| DeliveryScore {
        level,
        index: None,
        z: BTreeMap::new(),
        calibrated: thresholds.calibrated,
    };
    let Some(f) = features.filter(|f| f.word_count >= MIN_WORDS) else {
        return not_scored(IntegrityLevel::TooShort);
    };
    let Some(baseline) = baseline else {
        return not_scored(IntegrityLevel::Baseline);
    };
    let mut num = 0.0;
    let mut den = 0.0;
    let mut z = BTreeMap::new();
    for (name, w, _) in WEIGHTS {
        let (Some(b), Some(x)) = (baseline.get(name), f.get(name)) else {
            continue;
        };
        let zi = (x - b.median) / b.scale;
        z.insert(name, zi);
        num += w * zi;
        den += w.abs();
    }
    if den == 0.0 {
        return not_scored(IntegrityLevel::Unavailable);
    }
    let index = num / den;
    let level = if index >= thresholds.high_index {
        IntegrityLevel::High
    } else if index >= thresholds.possible_index {
        IntegrityLevel::Possible
    } else {
        IntegrityLevel::Low
    };
    DeliveryScore {
        level,
        index: Some(index),
        z,
        calibrated: thresholds.calibrated,
    }
}

/// One speaker's answers in order (the spec's `score_session`), with a
/// provisional score for an answer that is still growing.
///
/// The first [`BASELINE_ANSWERS`] answers with at least [`MIN_WORDS`] words
/// are the baseline (level `baseline`); every later answer is scored against
/// them. Too-short answers never join the baseline.
#[derive(Debug, Clone, Default)]
pub struct DeliveryTracker {
    base: Vec<DeliveryFeatures>,
    baseline: Option<DeliveryBaseline>,
}

impl DeliveryTracker {
    /// A tracker with no answers yet.
    pub fn new() -> Self {
        Self::default()
    }

    fn joins_baseline(&self, f: Option<&DeliveryFeatures>) -> bool {
        f.is_some_and(|f| f.word_count >= MIN_WORDS) && self.base.len() < BASELINE_ANSWERS
    }

    /// Score of `features` if the answer ended now, without recording it.
    pub fn score(
        &self,
        features: Option<&DeliveryFeatures>,
        thresholds: &DeliveryThresholds,
    ) -> DeliveryScore {
        if self.joins_baseline(features) {
            delivery_score(features, None, thresholds)
        } else {
            delivery_score(features, self.baseline.as_ref(), thresholds)
        }
    }

    /// Records a finished answer (it joins the baseline when it qualifies).
    pub fn commit(&mut self, features: Option<&DeliveryFeatures>) {
        if let Some(f) = features.filter(|_| self.joins_baseline(features)) {
            self.base.push(f.clone());
            if self.base.len() == BASELINE_ANSWERS {
                self.baseline = Some(delivery_baseline(&self.base));
            }
        }
    }

    /// Scores a finished answer and records it (one step of `score_session`).
    pub fn observe(
        &mut self,
        features: Option<&DeliveryFeatures>,
        thresholds: &DeliveryThresholds,
    ) -> DeliveryScore {
        let score = self.score(features, thresholds);
        self.commit(features);
        score
    }

    /// The baseline, once [`BASELINE_ANSWERS`] answers have been recorded.
    pub fn baseline(&self) -> Option<&DeliveryBaseline> {
        self.baseline.as_ref()
    }

    /// Baseline answers recorded so far.
    pub fn baseline_answers(&self) -> usize {
        self.base.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn th() -> DeliveryThresholds {
        super::super::spec_thresholds().delivery.clone()
    }

    fn unit(start: f64, end: f64, text: &str) -> SpeechUnit {
        SpeechUnit {
            start,
            end,
            text: text.into(),
            words: None,
        }
    }

    #[test]
    fn tokens_follow_the_spec_pattern() {
        assert_eq!(
            tokens("Um, I'd say 2 Things—RIGHT?"),
            vec!["um", "i'd", "say", "2", "things", "right"]
        );
        assert!(tokens("…").is_empty());
    }

    #[test]
    fn no_speech_has_no_features_and_is_too_short() {
        assert_eq!(delivery_features(Some(1.0), &[]), None);
        assert_eq!(
            delivery_score(None, None, &th()).level,
            IntegrityLevel::TooShort
        );
    }

    #[test]
    fn empty_word_lists_fall_back_to_utterances() {
        let mut u = unit(0.0, 2.0, "one two three");
        u.words = Some(Vec::new());
        let f = delivery_features(None, &[u]).unwrap();
        assert_eq!(f.granularity, Granularity::Utterance);
        assert_eq!(f.word_count, 3);
    }

    #[test]
    fn median_matches_python() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn tracker_scores_against_the_first_three_long_answers() {
        let t = th();
        let mut tr = DeliveryTracker::new();
        let long = |s: f64| {
            vec![unit(
                s,
                s + 6.0,
                "well um I think we tried a few things and it mostly worked",
            )]
        };
        for i in 0..3 {
            let f = delivery_features(Some(i as f64 * 20.0), &long(i as f64 * 20.0 + 1.0));
            assert_eq!(tr.observe(f.as_ref(), &t).level, IntegrityLevel::Baseline);
        }
        assert!(tr.baseline().is_some());
        // A short answer never joins the baseline.
        let f = delivery_features(Some(100.0), &[unit(101.0, 101.5, "yes")]);
        assert_eq!(tr.observe(f.as_ref(), &t).level, IntegrityLevel::TooShort);
        assert_eq!(tr.baseline_answers(), 3);
        // Provisional scoring does not record anything.
        let f = delivery_features(Some(120.0), &long(121.0));
        let s = tr.score(f.as_ref(), &t);
        assert_eq!(s.level, IntegrityLevel::Low);
        assert_eq!(tr.baseline_answers(), 3);
    }
}
