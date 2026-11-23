//! Claims, entities, metrics and action items extracted from transcript
//! segments with deterministic heuristics (no model needed).
//!
//! Claims feed interviewer follow-ups ("claims to verify"), retrieval and the
//! meeting summary; action items feed the summary.

use std::sync::LazyLock;

use openmeet_core::ids::{new_id, now_ms};
use openmeet_core::types::{
    ActionItem, Claim, ClaimKind, ExtractionSource, Mode, SpeakerRole, TranscriptSegment,
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
    // Spoken agreements in team meetings.
    "let's go with",
    "let us go with",
    "we'll go with",
    "we will go with",
    "we're going with",
    "we are going with",
    "we agreed",
    "we've agreed",
    "we have agreed",
    "it's decided",
    "it is decided",
    "decision is",
    "the plan is to",
    "let's stick with",
    "we'll stick with",
    "we're sticking with",
    "final decision",
    "we're dropping",
    "we'll drop",
    "let's drop",
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
    let raw_lower = sentence.to_lowercase().replace('’', "'");
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

/// Owner label of a commitment (`self_owned`) or a request made by `role`.
pub(crate) fn owner_label(role: SpeakerRole, self_owned: bool) -> Option<String> {
    match (role, self_owned) {
        (SpeakerRole::User, true) | (SpeakerRole::Remote, false) => Some("You".to_string()),
        (SpeakerRole::Remote, true) | (SpeakerRole::User, false) => {
            Some(OTHER_PARTICIPANT.to_string())
        }
        (SpeakerRole::Unknown, _) => None,
    }
}

/// Owner label for a commitment made by another participant whose name is
/// unknown.
pub(crate) const OTHER_PARTICIPANT: &str = "Other participant";

/// Verbs that make "Priya will …", "I'll …" or "we need to …" a task.
const TASK_VERBS: &str = r"(?:take|handle|own|update|fix|write|draft|prepare|send|share|review|ping|file|open|create|set up|schedule|follow up|look into|look at|check|investigate|book|organi[sz]e|run|add|move|merge|deploy|ship|finish|test|document|talk to|sync|reach out|loop in|circle back|put together|pair|help|start|coordinate|summari[sz]e|email|call|contact|research|compare|benchmark|migrate|clean up|remove|rename|refactor|reply|respond|confirm|ask|collect|gather|publish|post|present|demo|close|triage|estimate|plan|design|build|implement|configure|upgrade|roll back|roll out|monitor|sort out|figure out|dig into|get|make|do|chase|nudge|decide|agree on|choose|pick|order|tell|invite|go through|walk through|write up|kick off|unblock|escalate|flag|track|measure|verify|validate|double-check|approve|sign off|finali[sz]e|drop|draft)";

/// Capitalized words that are never a person's (or team's) name.
const NOT_NAMES: &[&str] = &[
    "I", "We", "You", "They", "He", "She", "It", "Let", "Let's", "Someone", "Somebody", "Everyone",
    "Everybody", "Anyone", "Anybody", "Nobody", "This", "That", "These", "Those", "Then", "So",
    "And", "But", "Also", "Okay", "OK", "Ok", "Yes", "No", "Yeah", "Sure", "Well", "Now", "Next",
    "The", "Our", "My", "Your", "Their", "His", "Her", "There", "Here", "What", "Who", "When",
    "Where", "Why", "How", "Which", "If", "Maybe", "Probably", "Today", "Tomorrow", "Tonight",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January",
    "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "Great", "Cool", "Right", "Alright", "First", "Finally", "Please",
    "Thanks", "Hi", "Hey", "Hello", "Actually", "Honestly", "Basically", "Just", "Again", "Once",
    "After", "Before", "Until", "Since", "Because", "Or", "Nothing", "Everything", "Something",
    "Team", "Folks", "Guys", "All", "One", "Each", "Every", "Both", "Either", "Neither", "Who's",
    "Action", "Todo", "Step", "Item",
];

fn is_name(word: &str) -> bool {
    let mut chars = word.chars();
    chars.next().is_some_and(char::is_uppercase)
        && word.chars().all(|c| c.is_alphabetic() || c == '-' || c == '\'')
        && !NOT_NAMES.contains(&word)
}

static NAMED_COMMIT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"\b([A-Z][A-Za-z'-]+)(?:\s+and\s+([A-Z][A-Za-z'-]+))?\s+(?:will|(?i:is going to|is gonna|are going to|can|could|should|to|needs to|has to|agreed to|offered to|volunteered to))\s+(?i:(?:also|probably|quickly|then)\s+)?(?i:{TASK_VERBS})\b"
    ))
    .expect("named commitment pattern is valid")
});
static NAMED_REQUEST_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"^\s*(?:(?i:okay|ok|so|and|hey)\s*,?\s+)?([A-Z][A-Za-z'-]+)\s*,\s*(?i:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:also\s+)?{TASK_VERBS})\b"
    ))
    .expect("named request pattern is valid")
});
static ASSIGN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:(?i:assign(?:ed|ing)?\s+(?:[a-z'-]+\s+){0,5}?to)|(?i:owner(?:\s+is|:)))\s+([A-Z][A-Za-z'-]+)",
    )
    .expect("assignment pattern is valid")
});
static OWNS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b([A-Z][A-Za-z'-]+)\s+(?:owns|has|takes|(?i:is taking|is on|can take|will take))\s+(?i:it|this|that)\b")
        .expect("ownership pattern is valid")
});

/// A named owner in a commitment or assignment: "Priya will update the
/// docs", "Priya and Sam will pair on it" ("Priya and Sam"), "assign it to
/// Sam", "Alex, can you take this?", "Dana is taking it". `None` when no name
/// is stated (OpenMeet never guesses one).
pub fn extract_named_owner(sentence: &str) -> Option<String> {
    let s = sentence.replace('’', "'");
    if let Some(c) = NAMED_REQUEST_RE.captures(&s)
        && is_name(&c[1])
    {
        return Some(c[1].to_string());
    }
    for c in NAMED_COMMIT_RE.captures_iter(&s) {
        if !is_name(&c[1]) {
            continue;
        }
        return Some(match c.get(2).filter(|m| is_name(m.as_str())) {
            Some(second) => format!("{} and {}", &c[1], second.as_str()),
            None => c[1].to_string(),
        });
    }
    for re in [&*ASSIGN_RE, &*OWNS_RE] {
        if let Some(c) = re.captures(&s)
            && is_name(&c[1])
        {
            return Some(c[1].to_string());
        }
    }
    None
}

static DUE_RES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let day = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri)";
    let month = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?";
    let ord = r"\d{1,2}(?:st|nd|rd|th)?";
    let period = r"(?:day|week|month|sprint|quarter|year|the day|the week|the month|the sprint|the quarter|the year)";
    let event = r"(?:release|demo|launch|deadline|standup|stand-up|retro|review|planning|offsite|cutover|freeze|code freeze|next meeting|next sync|next standup|next stand-up|next sprint|weekend|holidays)";
    [
        format!(
            r"(?i)\b(?:by|before|until|no later than)\s+(?:end of {period}|eod|eow|cob|tomorrow(?: morning| afternoon| evening)?|tonight|today|(?:this|next)\s+(?:week|sprint|month|quarter)|(?:this\s+|next\s+)?{day}(?:\s+(?:morning|afternoon|evening))?|{month}\s+{ord}|(?:the\s+)?{ord}(?:\s+of\s+{month})?|q[1-4]|(?:the\s+)?(?:next\s+|this\s+)?{event})\b"
        ),
        format!(r"(?i)\b(?:end of {period}|eod|eow)\b"),
        format!(r"(?i)\b(?:on|this|next)\s+{day}\b"),
        r"(?i)\b(?:this|next)\s+(?:week|sprint|month|quarter)\b".to_string(),
        r"(?i)\b(?:within|in)\s+(?:a|an|one|two|three|four|five|a couple of|\d+)\s+(?:hours?|days?|weeks?|months?|sprints?)\b".to_string(),
        r"(?i)\b(?:tomorrow(?: morning| afternoon)?|tonight|later today|today)\b".to_string(),
        r"(?i)\b(?:asap|as soon as possible)\b".to_string(),
    ]
    .iter()
    .map(|p| Regex::new(p).expect("due-date pattern is valid"))
    .collect()
});

/// When something is due, as said ("by Friday", "next sprint", "end of
/// day", "by March 3rd", "in two weeks"); the earliest-starting, longest
/// phrase. Never resolved to a calendar date.
pub fn extract_due(text: &str) -> Option<String> {
    let mut best: Option<(usize, usize)> = None;
    for re in DUE_RES.iter() {
        if let Some(m) = re.find(text) {
            let better = match best {
                None => true,
                Some((s, e)) => m.start() < s || (m.start() == s && m.end() > e),
            };
            if better {
                best = Some((m.start(), m.end()));
            }
        }
    }
    best.map(|(s, e)| text[s..e].trim().to_string())
}

static WORKPLACE_SELF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(?:i'll|i will|i'm going to|i am going to|i'm gonna|i can|let me|i've got to|i need to)\s+(?:also\s+|just\s+|quickly\s+|go\s+ahead\s+and\s+|try\s+to\s+)?{TASK_VERBS}\b"
    ))
    .expect("commitment pattern is valid")
});
static WORKPLACE_TEAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?i)\b(?:we need to|we should|we have to|we've got to|we must|someone needs to|someone should|somebody needs to|somebody should|someone has to|need someone to|let's make sure (?:we |to )?|make sure (?:we|to)|can someone|could someone|can somebody|could somebody|let's)\s+(?:also\s+|probably\s+|still\s+)?{TASK_VERBS}\b"
    ))
    .expect("team task pattern is valid")
});

/// Commitments and requests ("I'll send the design doc", "let's schedule a
/// follow-up", "can you share the slides?") in a FINAL segment, with the
/// owner and due date when stated.
///
/// In workplace mode the phrasing of team meetings counts too: any "I'll /
/// I can + task" commitment, "Priya will update the docs", "Alex, can you
/// take this?", "we need to update the runbook", "someone should check the
/// alerts". Named owners ("Priya") take precedence over the speaker.
pub fn extract_action_items(segment: &TranscriptSegment, mode: Mode) -> Vec<ActionItem> {
    let workplace = mode == Mode::Workplace;
    let mut out: Vec<ActionItem> = Vec::new();
    let now = now_ms();
    for sentence in split_sentences(&segment.text) {
        let lower = sentence.to_lowercase().replace('’', "'");
        let question = sentence.trim_end().ends_with('?');
        let named = extract_named_owner(&sentence);
        let request_owner = || {
            if workplace
                && segment.speaker_role != SpeakerRole::User
                && !crate::detect::addressed_to_listener(&sentence)
            {
                None
            } else {
                owner_label(segment.speaker_role, false)
            }
        };
        let owner = if contains_any(&lower, SELF_COMMIT) {
            Some(owner_label(segment.speaker_role, true))
        } else if contains_any(&lower, REQUEST) {
            Some(request_owner())
        } else if contains_any(&lower, TEAM_COMMIT) {
            Some(None)
        } else if workplace && named.is_some() {
            Some(None)
        } else if workplace && !question && WORKPLACE_SELF_RE.is_match(&lower) {
            Some(owner_label(segment.speaker_role, true))
        } else if workplace && WORKPLACE_TEAM_RE.is_match(&lower) {
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
            owner: named.or(owner),
            due: extract_due(&sentence),
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
        let items = extract_action_items(
            &seg(
                "Great talking to you. I'll send over the design doc tomorrow. Can you share the slides with me? Let's schedule a follow-up next week.",
                SpeakerRole::User,
            ),
            Mode::Candidate,
        );
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].owner.as_deref(), Some("You"));
        assert_eq!(items[0].due.as_deref(), Some("tomorrow"));
        assert_eq!(items[1].owner.as_deref(), Some("Other participant"));
        assert_eq!(items[1].due, None);
        assert_eq!(items[2].owner, None);
        assert_eq!(items[2].due.as_deref(), Some("next week"));
        assert!(
            extract_action_items(
                &seg("We talked about caching.", SpeakerRole::User),
                Mode::Candidate
            )
            .is_empty()
        );
    }

    #[test]
    fn interview_answers_do_not_become_team_tasks() {
        // Workplace phrasing is only widened in workplace mode.
        for mode in [Mode::Candidate, Mode::Interviewer] {
            let items = extract_action_items(
                &seg(
                    "We need to handle retries there. I can add a cache in front.",
                    SpeakerRole::Remote,
                ),
                mode,
            );
            assert!(items.is_empty(), "{mode:?}: {items:?}");
        }
    }

    #[test]
    fn workplace_action_items_table() {
        // (speaker, sentence, expected owner, expected due); None = not an action item.
        type Case = (SpeakerRole, &'static str, Option<(Option<&'static str>, Option<&'static str>)>);
        let cases: &[Case] = &[
            (SpeakerRole::Remote, "Priya will pair with Sam on it and have a fix ready by Thursday.", Some((Some("Priya"), Some("by Thursday")))),
            (SpeakerRole::Remote, "Priya and Sam will review the migration plan next week.", Some((Some("Priya and Sam"), Some("next week")))),
            (SpeakerRole::Remote, "So we need to update the runbook.", Some((None, None))),
            (SpeakerRole::Remote, "Someone should check the alert thresholds before the release.", Some((None, Some("before the release")))),
            (SpeakerRole::User, "I'll draft the announcement by end of day.", Some((Some("You"), Some("by end of day")))),
            (SpeakerRole::User, "I can take the runbook update.", Some((Some("You"), None))),
            (SpeakerRole::Remote, "I'll look into the flaky test.", Some((Some("Other participant"), None))),
            (SpeakerRole::Remote, "Alex, can you file a ticket for the timeout?", Some((Some("Alex"), None))),
            (SpeakerRole::Remote, "Can you share the dashboard link after the call?", Some((Some("You"), None))),
            (SpeakerRole::Remote, "Could someone send me the dashboard link?", Some((None, None))),
            (SpeakerRole::Remote, "Let's assign the cleanup to Dana.", Some((Some("Dana"), None))),
            (SpeakerRole::Remote, "Action item: Sam to benchmark the new cache in two weeks.", Some((Some("Sam"), Some("in two weeks")))),
            (SpeakerRole::Remote, "Let's go with the token refresh fix this sprint.", None),
            (SpeakerRole::Remote, "We shipped the export last week.", None),
            (SpeakerRole::Remote, "Monday will be busy for everyone.", None),
            (SpeakerRole::Remote, "Who can take the runbook update?", None),
            (SpeakerRole::User, "I think it went well overall.", None),
        ];
        for (role, text, expected) in cases {
            let items = extract_action_items(&seg(text, *role), Mode::Workplace);
            match expected {
                None => assert!(items.is_empty(), "{text}: {items:?}"),
                Some((owner, due)) => {
                    assert_eq!(items.len(), 1, "{text}: {items:?}");
                    assert_eq!(items[0].owner.as_deref(), *owner, "{text}");
                    assert_eq!(items[0].due.as_deref(), *due, "{text}");
                }
            }
        }
    }

    #[test]
    fn due_dates_as_spoken() {
        for (text, due) in [
            ("I'll send it by Friday.", Some("by Friday")),
            ("Ship it by the 15th please", Some("by the 15th")),
            ("Let's have it done by March 3rd.", Some("by March 3rd")),
            ("before next standup", Some("before next standup")),
            ("we can do it next sprint", Some("next sprint")),
            ("I'll get to it on Monday", Some("on Monday")),
            ("by EOD", Some("by EOD")),
            ("let's aim for end of the week", Some("end of the week")),
            ("within two days", Some("within two days")),
            ("I'll do it ASAP", Some("ASAP")),
            ("We talked about the roadmap.", None),
            ("It took two weeks last time.", None),
        ] {
            assert_eq!(extract_due(text).as_deref(), due, "{text}");
        }
    }

    #[test]
    fn named_owners_are_never_guessed() {
        for (text, owner) in [
            ("Priya will update the docs.", Some("Priya")),
            ("Hey Sam, could you review the PR?", Some("Sam")),
            ("The owner is Dana.", Some("Dana")),
            ("Dana is taking it.", Some("Dana")),
            ("We will update the docs.", None),
            ("Then we'll update the docs.", None),
            ("Tomorrow will be quiet.", None),
            ("Let's update the docs.", None),
        ] {
            assert_eq!(extract_named_owner(text).as_deref(), owner, "{text}");
        }
    }

    #[test]
    fn meeting_decisions_are_claims() {
        for text in [
            "Okay, let's go with the token refresh fix this sprint.",
            "We agreed to freeze the schema until the launch.",
            "So the plan is to migrate billing first.",
            "We're dropping the legacy CSV export.",
        ] {
            let c = extract_claims(&seg(text, SpeakerRole::Remote));
            assert_eq!(c.len(), 1, "{text}");
            assert_eq!(c[0].kind, ClaimKind::Decision, "{text}");
        }
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
