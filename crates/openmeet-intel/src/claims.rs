//! Claims, entities, metrics and action items extracted from transcript
//! segments with deterministic heuristics (no model needed).
//!
//! Claims feed interviewer follow-ups ("claims to verify"), retrieval and the
//! meeting summary; action items feed the summary.

use std::sync::LazyLock;

use openmeet_core::ids::{new_id, now_ms};
use openmeet_core::types::{
    ActionItem, Claim, ClaimKind, ExtractionSource, SpeakerRole, TranscriptSegment,
};
use regex::Regex;

use crate::lexicon::find_technologies;
use crate::text::{normalize, split_sentences, word_count};

/// Maximum characters of a claim / action item.
const MAX_CLAIM_CHARS: usize = 240;

/// Technologies mentioned in `text`, canonical casing, first occurrence first.
pub fn extract_entities(text: &str) -> Vec<String> {
    find_technologies(text)
        .iter()
        .map(|t| t.name.to_string())
        .collect()
}

static METRIC_RES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let num = r"(?:\d{1,3}(?:[,.]\d{3})+|\d+(?:\.\d+)?)";
    let words = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty|fifty|a hundred|hundreds of|several|a few)";
    let scale = r"(?:\s*(?:k|m|b|thousand|million|billion|trillion))?";
    let rate =
        r"(?:\s*(?:per|/|a|an|every)\s*(?:second|sec|s|minute|min|hour|hr|day|week|month|year))?";
    let units_count = r"(?:requests?|reqs?|rps|qps|tps|users?|customers?|clients?|events?|messages?|transactions?|queries|calls|orders?|records?|rows?|jobs?|tasks?|tickets?|services?|microservices?|servers?|nodes?|instances?|pods?|clusters?|regions?|engineers?|developers?|people|teams?|lines of code|deploys?|deployments?|releases?)";
    [
        // 40%, 40 percent
        format!(r"(?i)\b{num}\s*(?:%|percent\b)"),
        // 10 million requests per day, 5k QPS
        format!(r"(?i)\b(?:{num}|{words}){scale}\s*{units_count}{rate}\b"),
        // 200 ms, 3 seconds, 2 weeks
        format!(r"(?i)\b{num}\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b"),
        // 500 GB, 2 TB
        format!(r"(?i)\b{num}\s*(?:kb|mb|gb|tb|pb|kib|mib|gib|tib)\b"),
        // 5x faster, 10 times
        format!(r"(?i)\b{num}\s*(?:x|times)\b"),
        // p99 of 120ms handled above; p99 / 99.9% availability
        r"(?i)\bp(?:50|90|95|99|999)\b(?:\s+(?:latency\s+)?(?:of|was|is|at|around)?\s*\d+(?:\.\d+)?\s*(?:ms|s))?".to_string(),
        // $2 million, $500k
        format!(r"(?i)\$\s?{num}{scale}"),
    ]
    .iter()
    .map(|p| Regex::new(p).expect("metric pattern is valid"))
    .collect()
});

/// Numbers with units ("10 million requests per day", "40%", "200 ms",
/// "5k QPS"), in order of appearance, unique.
pub fn extract_metrics(text: &str) -> Vec<String> {
    let mut found: Vec<(usize, usize)> = Vec::new();
    for re in METRIC_RES.iter() {
        for m in re.find_iter(text) {
            found.push((m.start(), m.end()));
        }
    }
    found.sort();
    // Merge overlapping matches, keeping the longest span.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in found {
        match merged.last_mut() {
            Some(last) if s < last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    let mut out: Vec<String> = Vec::new();
    for (s, e) in merged {
        let m = text[s..e].trim().trim_end_matches([',', '.']).to_string();
        if !m.is_empty() && !out.iter().any(|x| x.eq_ignore_ascii_case(&m)) {
            out.push(m);
        }
    }
    out
}

const DECISION: &[&str] = &[
    "we decided",
    "i decided",
    "we chose",
    "i chose",
    "we went with",
    "i went with",
    "we picked",
    "we opted",
    "we settled on",
    "the decision was",
    "we ended up going with",
    "we switched to",
    "we moved to",
    "we migrated to",
    "we replaced",
];
const TRADEOFF: &[&str] = &[
    "trade-off",
    "tradeoff",
    "trade off",
    "the downside",
    "downside was",
    "at the cost of",
    "in exchange for",
    "the cost was",
    "but it meant",
    "the drawback",
    "we gave up",
    "we sacrificed",
    "the catch was",
    "on the flip side",
];
const RISK: &[&str] = &[
    "single point of failure",
    "the risk",
    "risky",
    "bottleneck",
    "outage",
    "failure mode",
    "data loss",
    "could fail",
    "might fail",
    "race condition",
    "security hole",
    "vulnerab",
    "incident",
    "went down",
    "took down",
    "blast radius",
];
const OWNERSHIP: &[&str] = &[
    "i led",
    "i owned",
    "i was responsible",
    "i was in charge",
    "my team owned",
    "i managed",
    "i drove",
    "i headed",
    "i was the tech lead",
    "i was the lead",
    "i ran the",
    "i mentored",
    "i coordinated",
    "i was accountable",
];
const FIRST_PERSON_ACTION: &[&str] = &[
    "i built",
    "i designed",
    "i implemented",
    "i wrote",
    "i migrated",
    "i architected",
    "i created",
    "i launched",
    "i shipped",
    "i reduced",
    "i improved",
    "i optimized",
    "i scaled",
    "i introduced",
    "i automated",
    "i rewrote",
    "i refactored",
    "i set up",
    "i added",
    "i fixed",
    "i developed",
    "we built",
    "we designed",
    "we implemented",
    "we wrote",
    "we migrated",
    "we architected",
    "we created",
    "we launched",
    "we shipped",
    "we reduced",
    "we improved",
    "we optimized",
    "we scaled",
    "we introduced",
    "we automated",
    "we rewrote",
    "we refactored",
    "we set up",
    "we added",
    "we fixed",
    "we developed",
    "we sharded",
    "we cached",
    "we partitioned",
    "we split",
    "we moved",
    "we used",
    "i used",
    "we rolled out",
    "i rolled out",
];
const IMPLEMENTATION: &[&str] = &[
    " by adding",
    " by using",
    "using a ",
    "using an ",
    " on top of",
    " behind a ",
    " with a ",
    " so that ",
    "we put ",
    "we store",
    "we stored",
    "we kept",
    "we run",
    "we ran",
    "is stored",
    "implemented with",
    "built on",
    "backed by",
];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn clip(text: &str) -> String {
    let t = text.trim();
    if t.chars().count() <= MAX_CLAIM_CHARS {
        t.to_string()
    } else {
        let mut s: String = t.chars().take(MAX_CLAIM_CHARS - 1).collect();
        s.push('…');
        s
    }
}

/// Classifies one sentence as a claim, or `None` when it is not a notable
/// statement.
fn classify_sentence(sentence: &str) -> Option<ClaimKind> {
    let lower = format!(" {} ", normalize(sentence));
    let raw_lower = sentence.to_lowercase();
    if sentence.trim_end().ends_with('?') || word_count(sentence) < 4 {
        return None;
    }
    let has_metric = !extract_metrics(sentence).is_empty();
    let has_tech = !find_technologies(sentence).is_empty();
    let kind = if contains_any(&raw_lower, DECISION) {
        ClaimKind::Decision
    } else if contains_any(&raw_lower, TRADEOFF) {
        ClaimKind::Tradeoff
    } else if contains_any(&raw_lower, RISK) {
        ClaimKind::Risk
    } else if contains_any(&raw_lower, OWNERSHIP) {
        ClaimKind::Ownership
    } else if has_metric {
        ClaimKind::Metric
    } else if contains_any(&lower, FIRST_PERSON_ACTION) {
        if contains_any(&raw_lower, IMPLEMENTATION) || has_tech {
            ClaimKind::Implementation
        } else {
            ClaimKind::Claim
        }
    } else if has_tech
        && (lower.contains(" we ") || lower.contains(" i ") || lower.contains(" our "))
    {
        ClaimKind::Technology
    } else {
        return None;
    };
    Some(kind)
}

/// Notable statements in a FINAL segment: decisions, trade-offs, risks,
/// metrics, ownership, implementation details and technologies used.
/// Deduplicated within the segment.
pub fn extract_claims(segment: &TranscriptSegment) -> Vec<Claim> {
    let mut out: Vec<Claim> = Vec::new();
    let now = now_ms();
    for sentence in split_sentences(&segment.text) {
        let Some(kind) = classify_sentence(&sentence) else {
            continue;
        };
        let text = clip(&sentence);
        if out.iter().any(|c| c.text.eq_ignore_ascii_case(&text)) {
            continue;
        }
        out.push(Claim {
            id: new_id(),
            meeting_id: segment.meeting_id.clone(),
            text,
            kind,
            segment_id: Some(segment.id.clone()),
            speaker_role: segment.speaker_role,
            source: ExtractionSource::Heuristic,
            created_at: now,
        });
    }
    out
}

const SELF_COMMIT: &[&str] = &[
    "i'll send",
    "i will send",
    "i'll share",
    "i will share",
    "i'll follow up",
    "i will follow up",
    "i'll get back",
    "i will get back",
    "i'll schedule",
    "i will schedule",
    "i'll set up",
    "i will set up",
    "i'll email",
    "i will email",
    "i'll write up",
    "i'll put together",
    "i'll look into",
    "i will look into",
    "i'll check",
    "i will check",
    "i'll circle back",
    "i'll loop in",
    "i'll forward",
    "i'll reach out",
    "i will reach out",
    "let me send",
    "let me follow up",
    "let me get back",
    "let me check",
    "let me look into",
];
const TEAM_COMMIT: &[&str] = &[
    "we'll send",
    "we will send",
    "we'll follow up",
    "we will follow up",
    "we'll get back",
    "we will get back",
    "we'll schedule",
    "we will schedule",
    "we'll share",
    "we will share",
    "we'll reach out",
    "we will reach out",
    "we'll be in touch",
    "we will be in touch",
    "let's schedule",
    "let's set up",
    "let's follow up",
    "let's sync",
    "let's meet",
    "action item",
    "next step is",
    "next steps are",
    "the next step",
    "to do:",
    "todo:",
];
const REQUEST: &[&str] = &[
    "can you send",
    "could you send",
    "please send",
    "can you share",
    "could you share",
    "please share",
    "can you email",
    "could you email",
    "send me",
    "share with me",
    "can you follow up",
    "could you follow up",
    "can you schedule",
    "could you schedule",
];

fn owner_label(role: SpeakerRole, self_owned: bool) -> Option<String> {
    match (role, self_owned) {
        (SpeakerRole::User, true) | (SpeakerRole::Remote, false) => Some("You".to_string()),
        (SpeakerRole::Remote, true) | (SpeakerRole::User, false) => {
            Some("Other participant".to_string())
        }
        (SpeakerRole::Unknown, _) => None,
    }
}

/// Commitments and requests ("I'll send the design doc", "let's schedule a
/// follow-up", "can you share the slides?") in a FINAL segment.
pub fn extract_action_items(segment: &TranscriptSegment) -> Vec<ActionItem> {
    let mut out: Vec<ActionItem> = Vec::new();
    let now = now_ms();
    for sentence in split_sentences(&segment.text) {
        let lower = sentence.to_lowercase().replace('’', "'");
        let owner = if contains_any(&lower, SELF_COMMIT) {
            Some(owner_label(segment.speaker_role, true))
        } else if contains_any(&lower, REQUEST) {
            Some(owner_label(segment.speaker_role, false))
        } else if contains_any(&lower, TEAM_COMMIT) {
            Some(None)
        } else {
            None
        };
        let Some(owner) = owner else { continue };
        if word_count(&sentence) < 3 {
            continue;
        }
        let text = clip(&sentence);
        if out.iter().any(|a| a.text.eq_ignore_ascii_case(&text)) {
            continue;
        }
        out.push(ActionItem {
            id: new_id(),
            meeting_id: segment.meeting_id.clone(),
            text,
            owner,
            source_segment_id: Some(segment.id.clone()),
            created_at: now,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::types::AudioSourceKind;

    fn seg(text: &str, role: SpeakerRole) -> TranscriptSegment {
        TranscriptSegment {
            id: "s1".into(),
            meeting_id: "m1".into(),
            utterance_id: "u1".into(),
            speaker_id: None,
            speaker_role: role,
            source: AudioSourceKind::System,
            start_time: 0,
            end_time: 1000,
            text: text.into(),
            is_final: true,
            confidence: None,
        }
    }

    #[test]
    fn entities_use_canonical_casing() {
        assert_eq!(
            extract_entities("we run postgres on k8s and stream through kafka"),
            vec!["PostgreSQL", "Kubernetes", "Kafka"]
        );
        assert!(extract_entities("nothing technical").is_empty());
    }

    #[test]
    fn metrics() {
        let m = extract_metrics(
            "The system handles 10 million requests per day with a p99 of 120 ms and 40% less cost.",
        );
        assert!(
            m.iter().any(|x| x.contains("10 million requests per day")),
            "{m:?}"
        );
        assert!(m.iter().any(|x| x.contains("120 ms")), "{m:?}");
        assert!(m.iter().any(|x| x == "40%"), "{m:?}");
        assert_eq!(extract_metrics("about 5k QPS at peak"), vec!["5k QPS"]);
        assert_eq!(extract_metrics("it got 3x faster"), vec!["3x"]);
        assert!(extract_metrics("no numbers here").is_empty());
    }

    #[test]
    fn claim_kinds() {
        let s = seg(
            "We decided to move billing to Kafka. The downside was more operational work. \
             Latency dropped from 400 ms to 120 ms. I led the migration across four teams. \
             We sharded Postgres by customer id using a consistent hash. Do you have questions?",
            SpeakerRole::Remote,
        );
        let claims = extract_claims(&s);
        let kinds: Vec<ClaimKind> = claims.iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            vec![
                ClaimKind::Decision,
                ClaimKind::Tradeoff,
                ClaimKind::Metric,
                ClaimKind::Ownership,
                ClaimKind::Implementation
            ]
        );
        assert!(
            claims
                .iter()
                .all(|c| c.source == ExtractionSource::Heuristic)
        );
        assert!(claims.iter().all(|c| c.speaker_role == SpeakerRole::Remote));
    }

    #[test]
    fn ordinary_chatter_is_not_a_claim() {
        assert!(
            extract_claims(&seg(
                "Yeah, that makes sense. Sounds good to me.",
                SpeakerRole::Remote
            ))
            .is_empty()
        );
        assert!(extract_claims(&seg("How did you scale it?", SpeakerRole::Remote)).is_empty());
    }

    #[test]
    fn technology_mentions() {
        let c = extract_claims(&seg(
            "Our backend runs on Go with Redis in front.",
            SpeakerRole::Remote,
        ));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].kind, ClaimKind::Technology);
    }

    #[test]
    fn action_items_with_owners() {
        let items = extract_action_items(&seg(
            "Great talking to you. I'll send over the design doc tomorrow. Can you share the slides with me? Let's schedule a follow-up next week.",
            SpeakerRole::User,
        ));
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].owner.as_deref(), Some("You"));
        assert_eq!(items[1].owner.as_deref(), Some("Other participant"));
        assert_eq!(items[2].owner, None);
        assert!(
            extract_action_items(&seg("We talked about caching.", SpeakerRole::User)).is_empty()
        );
    }

    #[test]
    fn duplicates_are_removed() {
        let c = extract_claims(&seg(
            "We migrated to Kafka. We migrated to Kafka.",
            SpeakerRole::Remote,
        ));
        assert_eq!(c.len(), 1);
    }
}
