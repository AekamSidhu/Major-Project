//! Workplace mode (team meetings: stand-ups, reviews, planning, 1:1s).
//!
//! * [`WorkplaceTracker`] keeps the live board: decisions, action items with
//!   owner and due date, questions and whether they were answered, and
//!   clarifying questions for vague items.
//! * [`action_item_issues`] / [`decision_issues`] detect vagueness (no owner,
//!   no due date, unclear scope, tentative) and [`clarifying_question`]
//!   words the question the user could ask.
//! * [`extractive_minutes`] builds the meeting minutes and a follow-up
//!   message draft without an LLM; [`render_minutes_markdown`] renders
//!   minutes (LLM or extractive) as Markdown.
//!
//! Everything builds on the shared extractors: decisions are claims of kind
//! `decision` ([`crate::claims::extract_claims`]), action items come from
//! [`crate::claims::extract_action_items`], questions from
//! [`crate::detect`]. Deterministic, local, no logging of content.

use std::collections::HashSet;
use std::sync::LazyLock;

use openmeet_core::ids::{Id, now_ms};
use openmeet_core::types::*;
use regex::Regex;

use crate::claims::{OTHER_PARTICIPANT, extract_due, extract_named_owner, owner_label};
use crate::prompts::TranscriptLine;
use crate::text::{content_tokens, lexical_similarity, normalize, truncate_to_tokens, word_count};
use crate::timefmt::{clock, duration, utc_datetime};

/// A reply this soon after a question may answer it without repeating its
/// words.
const NEXT_REPLY_WINDOW_MS: i64 = 5 * 60_000;
/// Related replies answer a question for this long.
const RELATED_REPLY_WINDOW_MS: i64 = 15 * 60_000;
/// "I'll take it" / "by Friday" refer to an action item at most this old.
const ASSIGN_WINDOW_MS: i64 = 3 * 60_000;
/// A restated action item is merged into an earlier one at most this old.
const MERGE_WINDOW_MS: i64 = 10 * 60_000;
/// Words of a reply kept as the answer of an open question.
const ANSWER_WORDS: usize = 30;
/// Characters of an item quoted in a clarifying question.
const QUOTE_CHARS: usize = 70;

// ---------------------------------------------------------------------------
// Task phrases and vagueness
// ---------------------------------------------------------------------------

/// Who-says-it leads of a task ("I'll", "we need to", "Priya will", "Alex,
/// can you"), followed by filler adverbs.
const TASK_LEADS: &str = r"(?:(?i:i'll|i will|i'm going to|i am going to|i'm gonna|i can|let me|i need to|i've got to|we'll|we will|we need to|we should|we have to|we've got to|we must|someone needs to|someone should|somebody needs to|somebody should|someone has to|need someone to|let's make sure(?: we| to)?|make sure (?:we|to)|let's|can someone|could someone|can somebody|could somebody|can you|could you|would you|please|action items?:?|next steps? (?:is|are)(?: to)?|todo:?|to do:?)|[A-Z][A-Za-z'-]+(?:\s+and\s+[A-Z][A-Za-z'-]+)?\s+(?i:will|is going to|is gonna|are going to|can|could|should|to|needs to|has to|agreed to|offered to|volunteered to)|[A-Z][A-Za-z'-]+\s*,\s*(?i:can|could|would|will)\s+you)\s+(?i:(?:also|just|please|probably|quickly|then|go ahead and|try to)\s+)*";

/// The first lead anywhere in the sentence ("Next, the docs are out of date,
/// so we need to …").
static TASK_LEAD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"^(?:.*?(?:^|[\s,;:—-]))??{TASK_LEADS}"))
        .expect("task lead pattern is valid")
});
/// A further lead right at the start ("Action item: Sam to …").
static TASK_LEAD_START_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"^{TASK_LEADS}")).expect("task lead pattern is valid")
});

/// The task of an action item without who says it or when it is due:
/// "Priya will pair with Sam on it by Thursday." → "Pair with Sam on it".
pub fn task_phrase(text: &str) -> String {
    let t = text.trim().replace('’', "'");
    let mut rest = match TASK_LEAD_RE.find(&t) {
        Some(m) if m.end() < t.len() => t[m.end()..].to_string(),
        _ => t.clone(),
    };
    for _ in 0..2 {
        match TASK_LEAD_START_RE.find(&rest) {
            Some(m) if m.end() < rest.len() => rest = rest[m.end()..].to_string(),
            _ => break,
        }
    }
    let mut rest = rest.trim().to_string();
    if let Some(due) = extract_due(&rest) {
        rest = rest.replacen(&due, "", 1);
    }
    let rest = rest
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
        .trim_start_matches("to ")
        .trim_end_matches(" please")
        .trim_end_matches(',')
        .trim()
        .to_string();
    let rest = if rest.is_empty() {
        t.trim_end_matches(['.', '!', '?']).to_string()
    } else {
        rest
    };
    let mut chars = rest.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => rest,
    }
}

/// Words that say nothing about what is to be done.
const SCOPE_FILLER: &[&str] = &[
    "a", "an", "the", "it", "it's", "its", "that", "this", "those", "these", "them", "thing",
    "things", "stuff", "something", "one", "some", "any", "look", "looking", "take", "taking",
    "handle", "handling", "do", "doing", "done", "deal", "dealing", "sort", "sorting", "figure",
    "figuring", "check", "checking", "follow", "following", "think", "thinking", "work",
    "working", "care", "get", "getting", "make", "making", "have", "having", "go", "going",
    "need", "want", "sure", "up", "out", "into", "onto", "on", "about", "with", "at", "to", "for",
    "of", "in", "over", "by", "from", "around", "and", "or", "but", "i", "we", "you", "they",
    "he", "she", "me", "us", "also", "just", "quickly", "please", "probably", "bit", "again",
    "more", "later", "soon", "now", "then", "there", "here", "what", "so", "ok", "okay", "all",
    "everything", "whatever", "same", "other", "rest", "can", "will", "should", "let's", "lets",
    "see", "try", "start", "first", "next", "look", "a", "little", "yeah", "yes", "sure",
];

fn has_scope(text: &str) -> bool {
    normalize(text)
        .split_whitespace()
        .any(|w| !SCOPE_FILLER.contains(&w) && w.chars().any(char::is_alphanumeric))
}

static DECISION_LEAD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^.*?\b(?:let's go with|let us go with|we'll go with|we will go with|we're going with|we are going with|we agreed(?: on| to| that)?|we've agreed(?: on| to| that)?|we have agreed(?: on| to| that)?|it's decided|it is decided|decision is(?: to| that)?|the plan is to|let's stick with|we'll stick with|we're sticking with|final decision(?: is)?:?|we're dropping|we'll drop|let's drop|we decided(?: to| on| that)?|i decided(?: to)?|we chose|we went with|we picked|we opted(?: for| to)?|we settled on|the decision was(?: to)?|we ended up going with|we switched to|we moved to|we migrated to|we replaced)\b")
        .expect("decision lead pattern is valid")
});

static TENTATIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:probably|maybe|might|i think|i guess|for now|tentatively|kind of|sort of|unless|assuming|for the time being|let's say|let's try|for the moment|in principle|if possible)\b")
        .expect("tentative pattern is valid")
});

/// Why an action item is vague, in a fixed order (scope, owner, due date).
pub fn action_item_issues(item: &ActionItem) -> Vec<VagueIssue> {
    let mut out = Vec::new();
    if !has_scope(&task_phrase(&item.text)) {
        out.push(VagueIssue::UnclearScope);
    }
    if item.owner.as_deref().is_none_or(|o| o.trim().is_empty()) {
        out.push(VagueIssue::NoOwner);
    }
    if item.due.as_deref().is_none_or(|d| d.trim().is_empty()) {
        out.push(VagueIssue::NoDueDate);
    }
    out
}

/// Why a decision is vague: what was decided is unclear ("let's go with
/// that"), or it was stated tentatively ("probably", "for now").
pub fn decision_issues(decision: &Claim) -> Vec<VagueIssue> {
    let text = decision.text.replace('’', "'");
    let object = match DECISION_LEAD_RE.find(&text) {
        Some(m) => text[m.end()..].to_string(),
        None => text.clone(),
    };
    let mut out = Vec::new();
    if !has_scope(&object) {
        out.push(VagueIssue::UnclearScope);
    }
    if TENTATIVE_RE.is_match(&text) {
        out.push(VagueIssue::Tentative);
    }
    out
}

fn quote(text: &str) -> String {
    let t = text.trim().trim_end_matches(['.', '!', '?', ',']);
    let clipped: String = if t.chars().count() > QUOTE_CHARS {
        let mut s: String = t.chars().take(QUOTE_CHARS - 1).collect();
        s.push('…');
        s
    } else {
        t.to_string()
    };
    format!("“{clipped}”")
}

/// The clarifying question for a vague item (`None` when nothing is vague).
/// `text` is the action item's task phrase or the decision as said; `owner`
/// the action item's owner.
pub fn clarifying_question(
    kind: WorkItemKind,
    text: &str,
    owner: Option<&str>,
    issues: &[VagueIssue],
) -> Option<String> {
    let has = |i: VagueIssue| issues.contains(&i);
    let q = quote(text);
    let question = match kind {
        WorkItemKind::Decision => {
            if has(VagueIssue::UnclearScope) {
                format!(
                    "What exactly did we decide in {q}? Worth saying it in one line for the notes."
                )
            } else if has(VagueIssue::Tentative) {
                format!("Is {q} decided, or still open?")
            } else {
                return None;
            }
        }
        WorkItemKind::ActionItem => {
            let scope = has(VagueIssue::UnclearScope);
            let no_owner = has(VagueIssue::NoOwner);
            let no_due = has(VagueIssue::NoDueDate);
            match (scope, no_owner, no_due) {
                (true, true, _) => format!("What exactly does {q} involve, and who will take it?"),
                (true, false, true) => format!("What exactly does {q} involve, and by when?"),
                (true, false, false) => {
                    format!("What exactly does {q} involve — what does done look like?")
                }
                (false, true, true) => format!("Who will take {q}, and by when?"),
                (false, true, false) => format!("Who will own {q}?"),
                (false, false, true) => match owner.filter(|o| is_person(o)) {
                    Some(name) => format!("{name}, by when can you have {q} done?"),
                    None => format!("By when should {q} be done?"),
                },
                (false, false, false) => return None,
            }
        }
    };
    Some(question)
}

/// A named owner (not "You" or the unnamed "Other participant").
fn is_person(owner: &str) -> bool {
    owner != "You" && owner != OTHER_PARTICIPANT && !owner.trim().is_empty()
}

// ---------------------------------------------------------------------------
// Replies to questions and to action items
// ---------------------------------------------------------------------------

static DEFERRAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:not sure|don't know|do not know|dunno|no idea|not yet|no date yet|take (?:that|this|it) offline|offline|park (?:that|this|it)|let's park|parking lot|get back to (?:you|everyone|the team)|circle back|i'll check|let me check|need to check|have to check|i'll find out|let me find out|we'll see|tbd|to be decided|to be determined|table (?:that|this|it)|hold off|next (?:meeting|sync|planning|week's meeting)|later this week|revisit)\b")
        .expect("deferral pattern is valid")
});

/// A reply that points back at the question ("let's park that").
static DEICTIC_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:that|this|it|those|these)\b").expect("deictic pattern is valid")
});

static ANSWER_CUE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\W*(?:yes|yeah|yep|yup|no|nope|sure|absolutely|definitely|correct|right|it's|it is|it was|it should|that's|that is|they're|we're|we are|we'll|we will|we did|we have|i think|i'd|i would|i'm|i am|i have|i did|probably|maybe|hopefully|currently|right now|so far|about|around|roughly|on|by|in|at|after|before|once|soon|next|this|not|we can|you can|should be|there's|there are|the plan is|good question|monday|tuesday|wednesday|thursday|friday|today|tomorrow|tonight)\b|^\W*\d")
        .expect("answer cue pattern is valid")
});

/// "I'll take it", "I can do that", "on it", "leave it with me".
static TAKE_IT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:(?:i'll|i will|i can|i've got|i got|i'm happy to|happy to|i could)\s+(?:take|do|handle|own|grab|pick up|look after|take care of|cover)?\s*(?:it|that|this|that one|this one)\b|i'm on it|i am on it|leave it with me|sign me up|count me in|that's mine|i'll take care of (?:it|that|this))")
        .expect("take-it pattern is valid")
});

/// A named owner taking an earlier item: "Priya can take it", "Dana is
/// taking that", "assign it to Sam".
static NAMED_TAKE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:take|takes|taking|do|does|doing|handle|handles|handling|own|owns|has|on|grab|grabs)\s+(?:it|that|this|that one|this one)\b|\bassign(?:ed)?\s+(?:it|that|this)\s+to\b")
        .expect("named take pattern is valid")
});

fn related(a: &str, b: &str) -> bool {
    let ta: HashSet<String> = content_tokens(a).into_iter().collect();
    content_tokens(b).iter().any(|t| ta.contains(t))
}

fn first_words(text: &str, n: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= n {
        words.join(" ")
    } else {
        format!("{} …", words[..n].join(" "))
    }
}

// ---------------------------------------------------------------------------
// Live tracker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TrackedQuestion {
    q: OpenQuestion,
    /// Final segments seen after the question.
    seen_any: u32,
    /// Final segments seen after the question from the other side (the user
    /// for a colleague's question, colleagues for the user's).
    seen_other: u32,
}

#[derive(Debug, Clone)]
struct TrackedItem {
    item: ActionItem,
    /// Meeting-relative start of the segment that created it.
    at_ms: i64,
}

/// What one segment changed on the board.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkplaceUpdate {
    /// The board changed: emit it.
    pub changed: bool,
    /// Action items that are new or were updated (owner / due date filled
    /// in): persist these (upserts by id).
    pub items: Vec<ActionItem>,
    /// Ids of the questions this segment answered.
    pub answered: Vec<Id>,
}

/// The live board of a workplace meeting.
#[derive(Debug, Clone)]
pub struct WorkplaceTracker {
    decisions: Vec<Claim>,
    items: Vec<TrackedItem>,
    questions: Vec<TrackedQuestion>,
    clarifications: Vec<ClarifyingQuestion>,
}

impl Default for WorkplaceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkplaceTracker {
    /// An empty board.
    pub fn new() -> Self {
        WorkplaceTracker {
            decisions: Vec::new(),
            items: Vec::new(),
            questions: Vec::new(),
            clarifications: Vec::new(),
        }
    }

    /// Updates the board from one FINAL segment.
    ///
    /// * `claims` — claims extracted from the segment (decisions are used);
    /// * `items` — action items extracted from it (workplace phrasing);
    /// * `question` — the question it asked, when the detector accepted one.
    pub fn observe(
        &mut self,
        segment: &TranscriptSegment,
        claims: &[Claim],
        items: Vec<ActionItem>,
        question: Option<&DetectedQuestion>,
    ) -> WorkplaceUpdate {
        let mut update = WorkplaceUpdate::default();
        let at = segment.start_time;
        let had_items = !items.is_empty();

        match question {
            Some(q) => {
                if let Some(old) = &q.supersedes {
                    self.questions.retain(|t| &t.q.id != old);
                }
                self.questions.push(TrackedQuestion {
                    q: OpenQuestion {
                        id: q.id.clone(),
                        question: q.question.trim().to_string(),
                        speaker_role: q.speaker_role,
                        addressed_to_you: q.requires_response,
                        asked_at_ms: at,
                        answered: false,
                        answer: None,
                        deferred: false,
                    },
                    seen_any: 0,
                    seen_other: 0,
                });
                update.changed = true;
            }
            None => {
                if self.resolve_questions(segment, &mut update) {
                    update.changed = true;
                }
            }
        }
        let asked_now = question.map(|q| q.id.as_str()).unwrap_or_default();
        for t in self.questions.iter_mut() {
            if t.q.id != asked_now && t.q.asked_at_ms <= at {
                t.seen_any += 1;
                if (segment.speaker_role == SpeakerRole::User)
                    != (t.q.speaker_role == SpeakerRole::User)
                {
                    t.seen_other += 1;
                }
            }
        }

        for c in claims.iter().filter(|c| c.kind == ClaimKind::Decision) {
            if self
                .decisions
                .iter()
                .any(|d| lexical_similarity(&d.text, &c.text) >= 0.85)
            {
                continue;
            }
            self.decisions.push(c.clone());
            update.changed = true;
        }

        for item in items {
            // "I'll take it" / "Dana can take that by Friday": who (or when)
            // for an earlier item, not a new task.
            if !has_scope(&task_phrase(&item.text))
                && let Some(t) = self.assignment_target(at, item.owner.is_some(), item.due.is_some())
            {
                let existing = &mut self.items[t].item;
                if existing.owner.is_none() {
                    existing.owner = item.owner.clone();
                }
                if existing.due.is_none() {
                    existing.due = item.due.clone();
                }
                update.items.push(existing.clone());
                update.changed = true;
                continue;
            }
            match self.merge_target(&item, at) {
                Some(idx) => {
                    let existing = &mut self.items[idx].item;
                    let mut touched = false;
                    if existing.owner.is_none() && item.owner.is_some() {
                        existing.owner = item.owner.clone();
                        touched = true;
                    }
                    if existing.due.is_none() && item.due.is_some() {
                        existing.due = item.due.clone();
                        touched = true;
                    }
                    if touched {
                        update.items.push(existing.clone());
                        update.changed = true;
                    }
                }
                None => {
                    update.items.push(item.clone());
                    self.items.push(TrackedItem { item, at_ms: at });
                    update.changed = true;
                }
            }
        }

        if !had_items
            && question.is_none()
            && let Some(item) = self.apply_assignment(segment)
        {
            update.items.push(item);
            update.changed = true;
        }

        if self.refresh_clarifications() {
            update.changed = true;
        }
        update
    }

    /// Resolves (or defers) the most recent open question this segment
    /// replies to. True when the board changed.
    fn resolve_questions(&mut self, segment: &TranscriptSegment, update: &mut WorkplaceUpdate) -> bool {
        let text = segment.text.trim();
        if word_count(text) < 2 {
            return false;
        }
        let at = segment.start_time;
        let role = segment.speaker_role;
        for t in self.questions.iter_mut().rev() {
            let q = &mut t.q;
            if q.answered || q.asked_at_ms > at {
                continue;
            }
            let age = at - q.asked_at_ms;
            if age > RELATED_REPLY_WINDOW_MS {
                continue;
            }
            // The user does not answer their own question. Between colleagues
            // (both "Others") the asker cannot be told apart from whoever
            // answers, so such a reply must refer to the question.
            if role == SpeakerRole::User && q.speaker_role == SpeakerRole::User {
                continue;
            }
            let cross = (role == SpeakerRole::User) != (q.speaker_role == SpeakerRole::User);
            let is_related = related(&q.question, text) && word_count(text) >= 3;
            let cue = ANSWER_CUE_RE.is_match(text);
            let recent = age <= NEXT_REPLY_WINDOW_MS;
            let (defers, answers) = if cross {
                let next = t.seen_other == 0 && recent;
                (next || is_related, (next && cue) || is_related)
            } else {
                let next = t.seen_any == 0 && recent;
                let refers_back = DEICTIC_RE.is_match(text);
                (
                    next && (refers_back || is_related),
                    is_related && (next || cue),
                )
            };
            if DEFERRAL_RE.is_match(text) {
                if !defers {
                    continue;
                }
                if q.deferred {
                    return false;
                }
                q.deferred = true;
                return true;
            }
            if !answers {
                continue;
            }
            q.answered = true;
            q.deferred = false;
            q.answer = Some(first_words(text, ANSWER_WORDS));
            update.answered.push(q.id.clone());
            return true;
        }
        false
    }

    /// The earlier item `item` restates ("I can take the runbook update"
    /// after "we need to update the runbook"), if any.
    fn merge_target(&self, item: &ActionItem, at: i64) -> Option<usize> {
        let task = task_phrase(&item.text);
        let new_tokens: HashSet<String> = content_tokens(&task).into_iter().collect();
        self.items
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, t)| at - t.at_ms <= MERGE_WINDOW_MS)
            .find(|(_, t)| {
                let old = task_phrase(&t.item.text);
                let sim = lexical_similarity(&old, &task);
                let shared = content_tokens(&old)
                    .iter()
                    .filter(|w| new_tokens.contains(*w))
                    .count();
                let fills_gap = (t.item.owner.is_none() && item.owner.is_some())
                    || (t.item.due.is_none() && item.due.is_some());
                let compatible_owner = match (&t.item.owner, &item.owner) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                };
                sim >= 0.8 || (fills_gap && compatible_owner && (sim >= 0.5 || shared >= 2))
            })
            .map(|(i, _)| i)
    }

    /// The latest recent item that lacks what a reply provides (an owner
    /// first, else a due date).
    fn assignment_target(&self, at: i64, gives_owner: bool, gives_due: bool) -> Option<usize> {
        let recent = |t: &&TrackedItem| at >= t.at_ms && at - t.at_ms <= ASSIGN_WINDOW_MS;
        let find = |pred: &dyn Fn(&TrackedItem) -> bool| {
            self.items
                .iter()
                .enumerate()
                .rev()
                .filter(|(_, t)| recent(t))
                .find(|(_, t)| pred(t))
                .map(|(i, _)| i)
        };
        if gives_owner && let Some(i) = find(&|t| t.item.owner.is_none()) {
            return Some(i);
        }
        if gives_due {
            return find(&|t| t.item.due.is_none());
        }
        None
    }

    /// "I'll take it" / "Priya can do that" / "by Friday" after an action
    /// item that lacked an owner or a due date.
    fn apply_assignment(&mut self, segment: &TranscriptSegment) -> Option<ActionItem> {
        let text = segment.text.trim().replace('’', "'");
        let at = segment.start_time;
        let owner = if TAKE_IT_RE.is_match(&text) {
            owner_label(segment.speaker_role, true)
        } else if NAMED_TAKE_RE.is_match(&text) {
            extract_named_owner(&text)
        } else {
            None
        };
        let due = extract_due(&text).filter(|_| owner.is_some() || word_count(&text) <= 12);
        let idx = self.assignment_target(at, owner.is_some(), due.is_some())?;
        let item = &mut self.items[idx].item;
        if item.owner.is_none() {
            item.owner = owner;
        }
        if item.due.is_none() {
            item.due = due;
        }
        Some(item.clone())
    }

    /// Recomputes the clarifying questions; true when they changed.
    fn refresh_clarifications(&mut self) -> bool {
        let now = now_ms();
        let mut next: Vec<ClarifyingQuestion> = Vec::new();
        let mut add = |target_id: &str, kind: WorkItemKind, text: String, owner: Option<&str>, issues: Vec<VagueIssue>| {
            let Some(question) = clarifying_question(kind, &text, owner, &issues) else {
                return;
            };
            let id = format!("clarify-{target_id}");
            let created_at = self
                .clarifications
                .iter()
                .find(|c| c.id == id && c.question == question)
                .map(|c| c.created_at)
                .unwrap_or(now);
            next.push(ClarifyingQuestion {
                id,
                target_id: target_id.to_string(),
                target_kind: kind,
                target_text: text,
                issues,
                question,
                created_at,
            });
        };
        for d in &self.decisions {
            add(
                &d.id,
                WorkItemKind::Decision,
                d.text.clone(),
                None,
                decision_issues(d),
            );
        }
        for t in &self.items {
            add(
                &t.item.id,
                WorkItemKind::ActionItem,
                task_phrase(&t.item.text),
                t.item.owner.as_deref(),
                action_item_issues(&t.item),
            );
        }
        next.sort_by_key(|c| c.created_at);
        let changed = next != self.clarifications;
        self.clarifications = next;
        changed
    }

    /// The board as shown in the UI.
    pub fn board(&self) -> WorkplaceBoard {
        WorkplaceBoard {
            decisions: self.decisions.clone(),
            action_items: self.action_items(),
            open_questions: self.questions.iter().map(|t| t.q.clone()).collect(),
            clarifications: self.clarifications.clone(),
        }
    }

    /// Action items, oldest first.
    pub fn action_items(&self) -> Vec<ActionItem> {
        self.items.iter().map(|t| t.item.clone()).collect()
    }
}

// ---------------------------------------------------------------------------
// Catch me up (without an LLM)
// ---------------------------------------------------------------------------

/// "Catch me up" without an LLM: what the board recorded in the last
/// `minutes` (meeting-relative `since_ms`), in the `[[SECTION]]` answer
/// format. `since_unix_ms` is the same instant as a Unix timestamp.
pub fn extractive_catch_up(
    lines: &[TranscriptLine],
    board: &WorkplaceBoard,
    minutes: u32,
    since_ms: i64,
    since_unix_ms: i64,
) -> String {
    let recent: Vec<&TranscriptLine> = lines.iter().filter(|l| l.at_ms >= since_ms).collect();
    let mut speakers: Vec<&str> = Vec::new();
    for l in &recent {
        if !speakers.contains(&l.speaker.as_str()) {
            speakers.push(&l.speaker);
        }
    }
    let span = if minutes == 1 {
        "minute".to_string()
    } else {
        format!("{minutes} minutes")
    };
    let mut points: Vec<String> = Vec::new();
    for d in board.decisions.iter().filter(|d| d.created_at >= since_unix_ms) {
        points.push(format!("Decided: {}", d.text.trim()));
    }
    for a in board.action_items.iter().filter(|a| a.created_at >= since_unix_ms) {
        points.push(format!("Action item: {}", action_item_line(a)));
    }
    for q in board.open_questions.iter().filter(|q| !q.answered) {
        if q.addressed_to_you {
            points.push(format!("Asked you: {}", q.question));
        } else if q.asked_at_ms >= since_ms {
            points.push(format!("Open question: {}", q.question));
        }
    }
    let answer = if recent.is_empty() {
        format!("Nothing was said in the last {span}.")
    } else {
        format!(
            "In the last {span}: {} line{} of discussion ({}).",
            recent.len(),
            if recent.len() == 1 { "" } else { "s" },
            speakers.join(", ")
        )
    };
    let mut out = format!("[[ANSWER]]\n{answer}\n");
    if !points.is_empty() {
        out.push_str("\n[[KEY_POINTS]]\n");
        for p in points.iter().take(8) {
            out.push_str(&format!("- {p}\n"));
        }
    }
    out.push_str(
        "\n[[CAVEAT]]\nBuilt without an AI model from the decisions, action items and questions OpenMeet tracked.\n",
    );
    out
}

// ---------------------------------------------------------------------------
// Minutes
// ---------------------------------------------------------------------------

/// One line for an action item: "Task — Owner, by Friday".
pub fn action_item_line(item: &ActionItem) -> String {
    minutes_item_line(&MinutesActionItem {
        task: task_phrase(&item.text),
        owner: item.owner.clone(),
        due: item.due.clone(),
    })
}

/// One line for a minutes action item: "Update the runbook — You, by
/// Friday" ("owner to be confirmed" when nobody was named).
pub fn minutes_item_line(item: &MinutesActionItem) -> String {
    let owner = item
        .owner
        .as_deref()
        .filter(|o| !o.trim().is_empty())
        .unwrap_or("owner to be confirmed");
    match item.due.as_deref().filter(|d| !d.trim().is_empty()) {
        Some(due) => format!("{} — {owner}, {due}", item.task.trim()),
        None => format!("{} — {owner}", item.task.trim()),
    }
}

fn dedup_push(list: &mut Vec<String>, item: String, max: usize) {
    let item = item.trim().to_string();
    if item.is_empty() || list.len() >= max {
        return;
    }
    if list
        .iter()
        .any(|x| x.eq_ignore_ascii_case(&item) || lexical_similarity(x, &item) >= 0.85)
    {
        return;
    }
    list.push(item);
}

/// A plain-text follow-up message the user can copy and send to the
/// attendees: decisions, action items with owners and due dates, open
/// questions. Never signed with a name.
pub fn follow_up_message(title: &str, summary: &MeetingSummary, items: &[MinutesActionItem]) -> String {
    let title = title.trim();
    let mut out = String::from("Hi all,\n\n");
    out.push_str(&if title.is_empty() {
        "Thanks for today's meeting. Here's a quick recap.\n".to_string()
    } else {
        format!("Thanks for joining {title}. Here's a quick recap.\n")
    });
    let overview = summary.overview.trim();
    if !overview.is_empty() && summary.generated_by != "extractive" {
        out.push_str(&format!("\n{overview}\n"));
    }
    if !summary.decisions.is_empty() {
        out.push_str("\nDecisions:\n");
        for d in &summary.decisions {
            out.push_str(&format!("- {}\n", d.trim()));
        }
    }
    if !items.is_empty() {
        out.push_str("\nAction items:\n");
        for i in items {
            out.push_str(&format!("- {}\n", minutes_item_line(i)));
        }
    }
    if !summary.open_questions.is_empty() {
        out.push_str("\nOpen questions:\n");
        for q in &summary.open_questions {
            out.push_str(&format!("- {}\n", q.trim()));
        }
    }
    if summary.decisions.is_empty() && items.is_empty() && summary.open_questions.is_empty() {
        out.push_str("\nNo decisions or action items were recorded.\n");
    }
    out.push_str("\nLet me know if I missed anything.\n");
    out
}

/// Minutes without an LLM: summary line, decisions (claims of kind
/// `decision`), action items with owner and due date, questions that got no
/// answer, clarifying questions for what is still vague, and a follow-up
/// message draft. `generated_by` is "extractive".
pub fn extractive_minutes(record: &MeetingRecord) -> MeetingSummary {
    let m = &record.meeting;
    let mut decisions = Vec::new();
    for c in record.claims.iter().filter(|c| c.kind == ClaimKind::Decision) {
        dedup_push(&mut decisions, c.text.clone(), 15);
    }
    let mut items: Vec<MinutesActionItem> = Vec::new();
    for a in &record.action_items {
        let task = task_phrase(&a.text);
        if items
            .iter()
            .any(|i| lexical_similarity(&i.task, &task) >= 0.85)
        {
            continue;
        }
        items.push(MinutesActionItem {
            task,
            owner: a.owner.clone(),
            due: a.due.clone(),
        });
    }
    let answered: HashSet<&str> = record
        .answers
        .iter()
        .filter_map(|a| a.question_id.as_deref())
        .collect();
    let superseded: HashSet<&str> = record
        .questions
        .iter()
        .filter_map(|q| q.supersedes.as_deref())
        .collect();
    let mut questions_asked = Vec::new();
    let mut open_questions = Vec::new();
    for q in record.questions.iter().filter(|q| !superseded.contains(q.id.as_str())) {
        dedup_push(&mut questions_asked, q.question.clone(), 25);
        if !answered.contains(q.id.as_str()) {
            dedup_push(&mut open_questions, q.question.clone(), 15);
        }
    }
    let mut key_topics = Vec::new();
    let mut topics: Vec<&Topic> = record
        .topics
        .iter()
        .filter(|t| t.status != TopicStatus::Pending)
        .collect();
    topics.sort_by_key(|t| std::cmp::Reverse(t.mentions));
    for t in topics {
        dedup_push(&mut key_topics, t.name.clone(), 8);
    }
    let mut to_clarify = Vec::new();
    for c in record.claims.iter().filter(|c| c.kind == ClaimKind::Decision) {
        if let Some(q) = clarifying_question(WorkItemKind::Decision, &c.text, None, &decision_issues(c)) {
            dedup_push(&mut to_clarify, q, 10);
        }
    }
    for a in &record.action_items {
        if let Some(q) = clarifying_question(
            WorkItemKind::ActionItem,
            &task_phrase(&a.text),
            a.owner.as_deref(),
            &action_item_issues(a),
        ) {
            dedup_push(&mut to_clarify, q, 10);
        }
    }
    let plural = |n: usize, one: &str, many: &str| {
        if n == 1 {
            format!("1 {one}")
        } else {
            format!("{n} {many}")
        }
    };
    let length = m
        .ended_at
        .map(|end| format!(" ({})", duration(end - m.started_at)))
        .unwrap_or_default();
    let mut overview = format!(
        "Team meeting “{}”{length}: {}, {} and {}.",
        m.title.trim(),
        plural(decisions.len(), "decision", "decisions"),
        plural(items.len(), "action item", "action items"),
        plural(open_questions.len(), "open question", "open questions"),
    );
    if !key_topics.is_empty() {
        overview.push_str(&format!(
            " Main topics: {}.",
            key_topics.iter().take(4).cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    let mut summary = MeetingSummary {
        meeting_id: m.id.clone(),
        overview,
        key_topics,
        questions_asked,
        answers: Vec::new(),
        decisions,
        action_items: items.iter().map(minutes_item_line).collect(),
        open_questions,
        technical_topics: Vec::new(),
        follow_up_suggestions: to_clarify,
        interviewer: None,
        workplace: None,
        markdown: String::new(),
        generated_at: now_ms(),
        generated_by: "extractive".to_string(),
    };
    let message = follow_up_message(&m.title, &summary, &items);
    summary.workplace = Some(WorkplaceMinutes {
        action_items: items,
        follow_up_message: message,
    });
    summary.markdown = render_minutes_markdown(&summary, &m.title);
    summary
}

fn push_section(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("\n## {heading}\n\n"));
    for i in items {
        out.push_str(&format!("- {}\n", crate::prompts::shared::one_line(i)));
    }
}

/// Meeting minutes as Markdown: summary, decisions, action items (owner, due
/// date), open questions, key topics, what to clarify, and the follow-up
/// message draft.
pub fn render_minutes_markdown(summary: &MeetingSummary, title: &str) -> String {
    let title = crate::prompts::shared::one_line(title);
    let mut out = format!(
        "# {}\n",
        if title.is_empty() {
            "Meeting minutes"
        } else {
            &title
        }
    );
    out.push_str("\n## Summary\n\n");
    if summary.overview.trim().is_empty() {
        out.push_str("No summary is available for this meeting.\n");
    } else {
        out.push_str(summary.overview.trim());
        out.push('\n');
    }
    push_section(&mut out, "Decisions", &summary.decisions);
    let items: Vec<String> = match &summary.workplace {
        Some(w) if !w.action_items.is_empty() => {
            w.action_items.iter().map(minutes_item_line).collect()
        }
        _ => summary.action_items.clone(),
    };
    push_section(&mut out, "Action items", &items);
    push_section(&mut out, "Open questions", &summary.open_questions);
    push_section(&mut out, "Key topics", &summary.key_topics);
    push_section(&mut out, "To clarify", &summary.follow_up_suggestions);
    if let Some(w) = summary
        .workplace
        .as_ref()
        .filter(|w| !w.follow_up_message.trim().is_empty())
    {
        out.push_str("\n## Follow-up message\n\n```text\n");
        out.push_str(w.follow_up_message.trim().replace("```", "'''").as_str());
        out.push_str("\n```\n");
    }
    let source = if summary.generated_by.trim().is_empty() {
        String::new()
    } else if summary.generated_by == "extractive" {
        " without an AI model (extractive minutes)".to_string()
    } else {
        format!(" by {}", crate::prompts::shared::one_line(&summary.generated_by))
    };
    if summary.generated_at > 0 {
        out.push_str(&format!(
            "\n---\n\n_Generated{source} on {}._\n",
            utc_datetime(summary.generated_at)
        ));
    } else if !source.is_empty() {
        out.push_str(&format!("\n---\n\n_Generated{source}._\n"));
    }
    out
}

/// Board lines for prompts: "[mm:ss] text" for decisions.
pub fn decision_lines(decisions: &[Claim], since_unix_ms: Option<i64>) -> Vec<String> {
    decisions
        .iter()
        .filter(|d| since_unix_ms.is_none_or(|s| d.created_at >= s))
        .map(|d| d.text.trim().to_string())
        .collect()
}

/// Board lines for prompts: action items with owner and due date.
pub(crate) fn action_item_lines(items: &[ActionItem]) -> Vec<String> {
    items
        .iter()
        .map(|a| {
            format!(
                "{} | owner: {} | due: {}",
                a.text.trim(),
                a.owner.as_deref().unwrap_or("not stated"),
                a.due.as_deref().unwrap_or("not stated")
            )
        })
        .collect()
}

/// Board lines for prompts: questions with their status.
pub(crate) fn question_lines(questions: &[OpenQuestion], label: impl Fn(SpeakerRole) -> &'static str) -> Vec<String> {
    questions
        .iter()
        .map(|q| {
            let status = if q.answered {
                "answered"
            } else if q.deferred {
                "deferred, still open"
            } else {
                "open"
            };
            let to = if q.addressed_to_you { ", asked of You" } else { "" };
            format!(
                "[{}] {} ({}{to}): {}",
                clock(q.asked_at_ms),
                label(q.speaker_role),
                status,
                truncate_to_tokens(q.question.trim(), 80)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;
