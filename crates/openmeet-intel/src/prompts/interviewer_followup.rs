//! Interviewer mode: grounded follow-up questions for the candidate's latest
//! answer, plus claims, gaps and topic updates, streamed as NDJSON.

use openmeet_core::types::{Claim, ClaimKind, Mode, Priority, QuestionCategory, TopicStatus};
use serde_json::{Map, Value};

use super::shared::{
    DATA_RULE, NON_EVALUATIVE_RULE, TRANSCRIPT_NOTE, assemble, block, category_list,
    claim_kind_slug, mode_line, one_line, parse_topic_status, question_line, role_intro, sanitize,
    sanitize_line,
};
use super::{ContextOptions, PromptContext, PromptParts, render_context_with};
use crate::json::{get_str, parse_lenient};
use crate::text::{estimate_tokens, tail_to_tokens, truncate_to_tokens};

/// Stable prompt identifier.
pub const ID: &str = "interviewer-followup";
/// Prompt template version.
pub const VERSION: u32 = 1;

/// Cap for the candidate turn included in the prompt (the end is kept).
const MAX_TURN_TOKENS: usize = 1500;
/// Cap for the heuristic claims block.
const MAX_DETECTED: usize = 12;

fn task(max_followups: usize) -> String {
    format!(
        "Task: suggest follow-up questions the user (the interviewer) could ask next about the \
candidate's latest answer in <candidate_answer>, and note what that answer claimed and left open.

Follow-up rules:
1. Grounded — every follow-up builds on something the candidate actually said in \
<candidate_answer> (or earlier in <transcript>). \"trigger\" quotes that statement in the \
candidate's own words (at most about 15 words). \"reason\" names the missing information in one \
short sentence, e.g. \"Mentioned Redis counters but not what happens when Redis is unavailable.\"
2. Deeper, not wider — probe mechanisms, trade-offs and alternatives considered, failure modes, \
scale limits, how results were measured, and what the candidate personally did versus the team. \
Never ask about something the candidate already explained.
3. No repeats — never repeat or paraphrase a question in <asked_questions> or \
<previous_suggestions>. Two questions that would get the same answer count as the same question. \
If an area was already asked about, go one level deeper or pick another gap.
4. Neutral — {NON_EVALUATIVE_RULE} No trick or leading questions.
5. Speakable — one open question per follow-up: a single sentence of at most 30 words that the \
interviewer can read aloud, asking one thing.
6. Priority — \"high\" for the most important open gap in a central claim, \"normal\" for useful \
depth, \"optional\" for nice-to-have.
7. If the answer is too short or vague to ground anything, suggest one follow-up asking the \
candidate to expand on the most specific thing they said.
Return at most {max_followups} follow-ups, most valuable first. Fewer, sharper questions beat many \
shallow ones.

Also report:
- claim: up to 4 notable, checkable statements from the latest answer (technologies, metrics, \
decisions, trade-offs, ownership, implementation details, risks) that are not already listed in \
<claims>. Paraphrase faithfully; never add details the candidate did not say.
- gap: up to 3 important things the answer left unaddressed, as short phrases (\"No consistency \
model discussed\"). Describe what is missing, never how good the answer was.
- topic: the topic of this answer with status \"current\"; use the exact name from <topics> when \
one matches. Mark a listed topic \"covered\" only when the conversation clearly moved on from it."
    )
}

fn format_rules() -> String {
    format!(
        "Output format — NDJSON: one JSON object per line and nothing else (no prose, no code \
fences, no array brackets, no blank lines). Write the followup lines first, then claim, gap and \
topic lines:
{{\"type\":\"followup\",\"question\":\"…\",\"category\":\"<category>\",\"reason\":\"…\",\"priority\":\"high|normal|optional\",\"trigger\":\"…\"}}
{{\"type\":\"claim\",\"text\":\"…\",\"kind\":\"claim|technology|decision|tradeoff|metric|ownership|implementation|risk\"}}
{{\"type\":\"gap\",\"text\":\"…\"}}
{{\"type\":\"topic\",\"name\":\"…\",\"status\":\"pending|current|covered\"}}
<category> is one of: {}.",
        category_list()
    )
}

/// Builds the follow-up prompt for the candidate's latest turn.
pub fn build(
    ctx: &PromptContext,
    candidate_turn: &str,
    heuristic_claims: &[Claim],
    max_followups: usize,
    budget_tokens: usize,
) -> PromptParts {
    let max_followups = max_followups.clamp(1, 10);
    let system = assemble(&[
        role_intro(Mode::Interviewer).to_string(),
        DATA_RULE.to_string(),
        TRANSCRIPT_NOTE.to_string(),
        task(max_followups),
        format_rules(),
    ]);

    let header = format!(
        "{}\nMaximum follow-ups: {max_followups}",
        mode_line(Mode::Interviewer)
    );
    let turn = tail_to_tokens(sanitize(candidate_turn.trim()).trim(), MAX_TURN_TOKENS);
    let turn_block = format!(
        "The candidate's latest answer (the turn to follow up on):\n{}",
        block("candidate_answer", &turn)
    );
    let detected: Vec<String> = heuristic_claims
        .iter()
        .filter(|c| !c.text.trim().is_empty())
        .take(MAX_DETECTED)
        .map(|c| {
            format!(
                "- [{}] {}",
                claim_kind_slug(c.kind),
                truncate_to_tokens(&sanitize_line(&c.text), 60)
            )
        })
        .collect();
    let detected_block = if detected.is_empty() {
        String::new()
    } else {
        format!(
            "Statements detected automatically in the latest answer (may be incomplete):\n{}",
            block("detected_in_answer", &detected.join("\n"))
        )
    };
    let q_line = question_line(&format!(
        "Suggest up to {max_followups} follow-up questions about the candidate's latest answer."
    ));
    let reserved = estimate_tokens(&header)
        + estimate_tokens(&turn_block)
        + estimate_tokens(&detected_block)
        + estimate_tokens(&q_line)
        + 12;
    let options = ContextOptions {
        include_asked: true,
        include_claims: true,
        include_topics: true,
    };
    let mut context_ctx = ctx.clone();
    context_ctx.mode = Mode::Interviewer;
    let context = render_context_with(
        &context_ctx,
        budget_tokens.saturating_sub(reserved),
        options,
    );
    PromptParts {
        id: ID,
        version: VERSION,
        system,
        user: assemble(&[header, context, detected_block, turn_block, q_line]),
    }
}

/// One parsed line of follow-up output.
#[derive(Debug, Clone, PartialEq)]
pub enum FollowupItem {
    /// A suggested follow-up question.
    Followup {
        /// The question the interviewer could ask.
        question: String,
        /// Question category.
        category: QuestionCategory,
        /// Concise rationale (what is missing).
        reason: String,
        /// Importance.
        priority: Priority,
        /// The candidate statement that triggered it.
        trigger: Option<String>,
    },
    /// A notable statement from the candidate.
    Claim {
        /// The statement.
        text: String,
        /// Kind of statement.
        kind: ClaimKind,
    },
    /// A topic status update.
    Topic {
        /// Topic name.
        name: String,
        /// New status.
        status: TopicStatus,
    },
    /// Something the answer left unaddressed.
    Gap {
        /// Description of the gap.
        text: String,
    },
}

/// Longest multi-line object buffered while waiting for its closing brace.
const MAX_PENDING_BYTES: usize = 16 * 1024;

/// Incremental NDJSON parser: tolerant of code fences, partial lines, blank
/// lines, CRLF, JSON arrays, pretty-printed objects and malformed lines
/// (skipped).
#[derive(Debug, Default, Clone)]
pub struct FollowupStreamParser {
    /// Text of the current (incomplete) line.
    line: String,
    /// A multi-line object being accumulated.
    pending: String,
}

impl FollowupStreamParser {
    /// Creates an empty parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a streamed delta; returns every item completed by it.
    pub fn push(&mut self, delta: &str) -> Vec<FollowupItem> {
        let mut out = Vec::new();
        for ch in delta.chars() {
            if ch == '\n' {
                let line = std::mem::take(&mut self.line);
                self.process_line(&line, &mut out);
            } else {
                self.line.push(ch);
            }
        }
        // A complete object without its newline yet: emit it right away so
        // the overlay can show the first follow-up as early as possible.
        if self.pending.is_empty() {
            let t = self.line.trim();
            if t.starts_with('{') && t.ends_with('}') {
                if let Some(Value::Object(obj)) = parse_lenient(t) {
                    self.line.clear();
                    out.extend(item_from_object(&obj));
                }
            }
        }
        out
    }

    /// Flushes the final (unterminated) line and any buffered object.
    pub fn finish(&mut self) -> Vec<FollowupItem> {
        let mut out = Vec::new();
        let line = std::mem::take(&mut self.line);
        self.process_line(&line, &mut out);
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            out.extend(items_from_text(&pending));
        }
        out
    }

    fn process_line(&mut self, raw: &str, out: &mut Vec<FollowupItem>) {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            return;
        }
        if !self.pending.is_empty() {
            // A self-contained object on its own line means the buffered
            // text was a malformed line, not a pretty-printed object.
            let standalone = trimmed.starts_with('{') && balance(trimmed) == Some(0);
            if standalone {
                self.pending.clear();
            } else {
                self.pending.push('\n');
                self.pending.push_str(line);
                if balance(&self.pending) == Some(0) {
                    let pending = std::mem::take(&mut self.pending);
                    out.extend(items_from_text(&pending));
                } else if self.pending.len() > MAX_PENDING_BYTES {
                    self.pending.clear();
                }
                return;
            }
        }
        if trimmed.is_empty() || !trimmed.contains('{') {
            return;
        }
        match balance(trimmed) {
            Some(0) => out.extend(items_from_text(trimmed)),
            Some(depth) if depth > 0 && looks_like_object_start(trimmed) => {
                let start = trimmed.find('{').unwrap_or(0);
                self.pending = trimmed[start..].to_string();
            }
            // Malformed (unbalanced) line: skipped.
            _ => {}
        }
    }
}

/// Brace depth at the end of `text` (string-aware), `None` when it goes
/// negative.
fn balance(text: &str) -> Option<i64> {
    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut escaped = false;
    for c in text.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    Some(depth)
}

/// An unbalanced line that plausibly opens a pretty-printed object (as
/// opposed to a truncated single-line object).
fn looks_like_object_start(line: &str) -> bool {
    let t = line.trim_end();
    t.ends_with('{') || t.ends_with(',') || t.ends_with(':') || t.ends_with('[')
}

/// Every item found in a chunk of text (one or more objects, arrays).
fn items_from_text(text: &str) -> Vec<FollowupItem> {
    crate::json::all_json_objects(text)
        .iter()
        .filter_map(item_from_object)
        .collect()
}

fn clean_text(s: &str) -> String {
    one_line(s.trim().trim_matches(['"', '\u{201c}', '\u{201d}']).trim())
}

fn parse_priority(raw: Option<String>) -> Priority {
    match raw
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default()
        .as_str()
    {
        "high" | "critical" | "important" | "must" | "1" => Priority::High,
        "optional" | "low" | "nice_to_have" | "nice-to-have" | "3" => Priority::Optional,
        _ => Priority::Normal,
    }
}

/// Maps one JSON object to an item. Unknown or incomplete objects yield
/// `None`.
pub(crate) fn item_from_object(obj: &Map<String, Value>) -> Option<FollowupItem> {
    let ty = get_str(obj, &["type", "item_type", "itemType"])
        .map(|t| {
            t.to_ascii_lowercase()
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
        })
        .unwrap_or_default();
    let has_question = get_str(obj, &["question"]).is_some();
    let kind = match ty.as_str() {
        "followup" | "followupquestion" | "question" | "suggestion" => "followup",
        "claim" | "statement" | "fact" => "claim",
        "gap" | "missing" | "unknown" | "openquestion" => "gap",
        "topic" | "topicupdate" => "topic",
        "" if has_question => "followup",
        "" if get_str(obj, &["status"]).is_some() && get_str(obj, &["name"]).is_some() => "topic",
        "" if get_str(obj, &["kind"]).is_some() && get_str(obj, &["text"]).is_some() => "claim",
        _ => return None,
    };
    match kind {
        "followup" => {
            let question = clean_text(&get_str(obj, &["question", "text", "q"])?);
            if question.is_empty() {
                return None;
            }
            let trigger = get_str(obj, &["trigger", "quote", "source", "because_of"])
                .map(|t| clean_text(&t))
                .filter(|t| !t.is_empty());
            let reason = get_str(obj, &["reason", "rationale", "why", "missing"])
                .map(|r| clean_text(&r))
                .filter(|r| !r.is_empty())
                .or_else(|| trigger.as_ref().map(|t| format!("Builds on: \"{t}\"")))
                .unwrap_or_default();
            let category = get_str(obj, &["category"])
                .and_then(|c| QuestionCategory::parse_lenient(&c))
                .unwrap_or(QuestionCategory::FollowUp);
            Some(FollowupItem::Followup {
                question,
                category,
                reason,
                priority: parse_priority(get_str(obj, &["priority", "importance"])),
                trigger,
            })
        }
        "claim" => {
            let text = clean_text(&get_str(obj, &["text", "claim", "statement"])?);
            (!text.is_empty()).then(|| FollowupItem::Claim {
                text,
                kind: ClaimKind::parse_lenient(
                    &get_str(obj, &["kind", "category"]).unwrap_or_else(|| "claim".into()),
                ),
            })
        }
        "gap" => {
            let text = clean_text(&get_str(obj, &["text", "gap", "description"])?);
            (!text.is_empty()).then_some(FollowupItem::Gap { text })
        }
        _ => {
            let name = clean_text(&get_str(obj, &["name", "topic", "text"])?);
            let status = get_str(obj, &["status", "state"])
                .and_then(|s| parse_topic_status(&s))
                .unwrap_or(TopicStatus::Current);
            (!name.is_empty()).then_some(FollowupItem::Topic { name, status })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::shared::{extract_block, extract_block_items, extract_question};
    use openmeet_core::types::{ExtractionSource, SpeakerRole};

    const SAMPLE: &str = concat!(
        "{\"type\":\"followup\",\"question\":\"What happens if Redis is unavailable?\",\"category\":\"tradeoff\",\"reason\":\"Mentioned Redis but not its failure mode.\",\"priority\":\"high\",\"trigger\":\"I'd probably use Redis\"}\n",
        "{\"type\":\"claim\",\"text\":\"Uses Redis for counters\",\"kind\":\"technology\"}\n",
        "{\"type\":\"gap\",\"text\":\"No consistency model discussed\"}\n",
        "{\"type\":\"topic\",\"name\":\"Rate limiting\",\"status\":\"current\"}\n",
    );

    fn claim(text: &str) -> Claim {
        Claim {
            id: "c".into(),
            meeting_id: "m".into(),
            text: text.into(),
            kind: ClaimKind::Technology,
            segment_id: None,
            speaker_role: SpeakerRole::Remote,
            source: ExtractionSource::Heuristic,
            created_at: 0,
        }
    }

    #[test]
    fn build_grounds_prompt_in_the_turn() {
        let ctx = PromptContext {
            mode: Mode::Interviewer,
            asked_questions: vec!["How would you design a rate limiter?".into()],
            previous_suggestions: vec!["What is your eviction policy?".into()],
            ..Default::default()
        };
        let p = build(
            &ctx,
            "I'd probably use Redis with a token bucket per API key.",
            &[claim("Redis")],
            4,
            3500,
        );
        assert_eq!(p.id, ID);
        assert!(p.system.contains("NDJSON"));
        assert!(p.system.contains("trigger"));
        assert!(p.system.contains("hiring recommendation"));
        assert!(p.system.contains("<previous_suggestions>"));
        assert!(
            extract_block(&p.user, "candidate_answer")
                .unwrap()
                .contains("token bucket")
        );
        assert_eq!(
            extract_block_items(&p.user, "asked_questions"),
            vec!["How would you design a rate limiter?"]
        );
        assert_eq!(
            extract_block_items(&p.user, "previous_suggestions"),
            vec!["What is your eviction policy?"]
        );
        assert_eq!(
            extract_block_items(&p.user, "detected_in_answer"),
            vec!["[technology] Redis"]
        );
        assert!(p.user.contains("Maximum follow-ups: 4"));
        assert!(
            extract_question(&p.user)
                .unwrap()
                .starts_with("Suggest up to 4")
        );
    }

    #[test]
    fn parses_all_item_types() {
        let mut p = FollowupStreamParser::new();
        let mut items = p.push(SAMPLE);
        items.extend(p.finish());
        assert_eq!(items.len(), 4);
        match &items[0] {
            FollowupItem::Followup {
                question,
                category,
                priority,
                trigger,
                reason,
            } => {
                assert_eq!(question, "What happens if Redis is unavailable?");
                assert_eq!(*category, QuestionCategory::Tradeoff);
                assert_eq!(*priority, Priority::High);
                assert_eq!(trigger.as_deref(), Some("I'd probably use Redis"));
                assert!(reason.contains("failure mode"));
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(
            items[1],
            FollowupItem::Claim {
                text: "Uses Redis for counters".into(),
                kind: ClaimKind::Technology
            }
        );
        assert_eq!(
            items[2],
            FollowupItem::Gap {
                text: "No consistency model discussed".into()
            }
        );
        assert_eq!(
            items[3],
            FollowupItem::Topic {
                name: "Rate limiting".into(),
                status: TopicStatus::Current
            }
        );
    }

    /// Deterministic xorshift for chunk sizes.
    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn random_chunking_yields_identical_items() {
        let expected = {
            let mut p = FollowupStreamParser::new();
            let mut v = p.push(SAMPLE);
            v.extend(p.finish());
            v
        };
        let text = format!("```json\r\n{}```", SAMPLE.replace('\n', "\r\n"));
        let chars: Vec<char> = text.chars().collect();
        for round in 0..200u64 {
            let mut seed = 0x9E37_79B9_7F4A_7C15 ^ (round + 1);
            let mut p = FollowupStreamParser::new();
            let mut items = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                let n = (next(&mut seed) % 17 + 1) as usize;
                let end = (i + n).min(chars.len());
                let chunk: String = chars[i..end].iter().collect();
                items.extend(p.push(&chunk));
                i = end;
            }
            items.extend(p.finish());
            assert_eq!(items, expected, "round {round}");
        }
    }

    #[test]
    fn early_emission_without_newline() {
        let mut p = FollowupStreamParser::new();
        assert!(p.push("{\"type\":\"gap\",\"text\":\"No ").is_empty());
        let items = p.push("metrics\"}");
        assert_eq!(items.len(), 1);
        assert!(p.push("\n").is_empty(), "not emitted twice");
        assert!(p.finish().is_empty());
    }

    #[test]
    fn tolerates_prose_arrays_pretty_json_and_malformed_lines() {
        let text = "Sure, here are the follow-ups:\n\n[\n  {\"type\":\"followup\",\"question\":\"Why Redis?\",\"reason\":\"No alternatives\"},\n  {\n    \"type\": \"gap\",\n    \"text\": \"No failure handling\"\n  },\n{\"type\":\"followup\",\"question\":\"broken\n{\"type\":\"claim\",\"text\":\"Uses Lua scripts\",\"kind\":\"implementation\"}\n{\"type\":\"unknown_thing\",\"x\":1}\n{\"type\":\"followup\",\"question\":\"   \"}\n]\nHope this helps!";
        let mut p = FollowupStreamParser::new();
        let mut items = p.push(text);
        items.extend(p.finish());
        assert_eq!(
            items,
            vec![
                FollowupItem::Followup {
                    question: "Why Redis?".into(),
                    category: QuestionCategory::FollowUp,
                    reason: "No alternatives".into(),
                    priority: Priority::Normal,
                    trigger: None,
                },
                FollowupItem::Gap {
                    text: "No failure handling".into()
                },
                FollowupItem::Claim {
                    text: "Uses Lua scripts".into(),
                    kind: ClaimKind::Implementation
                },
            ]
        );
    }

    #[test]
    fn missing_reason_falls_back_to_trigger_and_type_is_inferred() {
        let mut p = FollowupStreamParser::new();
        let items = p.push("{\"question\":\"How did you measure 40%?\",\"trigger\":\"cut latency by 40%\",\"priority\":\"low\"}\n{\"name\":\"Observability\",\"status\":\"not discussed\"}\n");
        assert_eq!(
            items,
            vec![
                FollowupItem::Followup {
                    question: "How did you measure 40%?".into(),
                    category: QuestionCategory::FollowUp,
                    reason: "Builds on: \"cut latency by 40%\"".into(),
                    priority: Priority::Optional,
                    trigger: Some("cut latency by 40%".into()),
                },
                FollowupItem::Topic {
                    name: "Observability".into(),
                    status: TopicStatus::Pending
                }
            ]
        );
    }
}
