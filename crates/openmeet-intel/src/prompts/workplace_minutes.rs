//! Workplace mode: meeting minutes after a team meeting — summary,
//! decisions, action items with owners and due dates, open questions, what
//! to clarify, and a follow-up message draft. Tolerant JSON parser into a
//! [`MeetingSummary`] with its `workplace` part.
//!
//! The prompt is grounded in what OpenMeet tracked live (the board) and the
//! transcript; owners and dates are only kept when they were said. No
//! hiring language, no judgments of colleagues' performance — the parser
//! drops evaluative lines that slip through.

use std::sync::LazyLock;

use openmeet_core::ids::now_ms;
use openmeet_core::types::{
    ActionItem, MeetingSummary, MinutesActionItem, Mode, OpenQuestion, SpeakerRole, Topic,
    TopicStatus, WorkplaceMinutes,
};
use regex::Regex;
use serde_json::{Map, Value};

use super::meeting_summary::is_evaluative;
use super::shared::{
    DATA_RULE, TRANSCRIPT_NOTE, assemble, block, mode_line, one_line, question_line, sanitize,
    sanitize_line,
};
use super::{PromptParts, TranscriptLine};
use crate::json::{first_json_object, get, value_to_str_list};
use crate::text::{estimate_tokens, truncate_to_tokens};
use crate::timefmt::clock;
use crate::workplace::{action_item_lines, follow_up_message, minutes_item_line, question_lines};

/// Stable prompt identifier.
pub const ID: &str = "workplace-minutes";
/// Prompt template version.
pub const VERSION: u32 = 1;

/// Everything the minutes prompt may use.
pub struct MinutesInput<'a> {
    /// Meeting title.
    pub title: &'a str,
    /// Rolling summary of the early part of the meeting (may be empty).
    pub long_term_summary: &'a str,
    /// Transcript, oldest first.
    pub transcript: &'a [TranscriptLine],
    /// Decisions tracked live.
    pub decisions: &'a [String],
    /// Action items tracked live (owner / due date when said).
    pub action_items: &'a [ActionItem],
    /// Questions asked, with whether they were answered.
    pub questions: &'a [OpenQuestion],
    /// Conversation topics.
    pub topics: &'a [Topic],
    /// Screen observation descriptions.
    pub observations: &'a [String],
}

const TASK: &str = "Task: write the minutes of the team meeting that just ended, for the user who \
ran OpenMeet (\"You\" in the transcript), as one JSON object with these keys (use [] or null when \
nothing applies):
- \"overview\": 2–4 sentences: what the meeting covered and where it landed.
- \"keyTopics\": the main topics, most important first.
- \"decisions\": decisions that were made, one short line each.
- \"actionItems\": array of {\"task\": \"…\", \"owner\": \"…\" or null, \"due\": \"…\" or null} — \
commitments and next steps. \"owner\" is the person or team that was named, \"You\" for the user, \
or null; \"due\" is the time frame as said (\"by Friday\", \"next sprint\") or null.
- \"openQuestions\": questions that were asked but not answered in the meeting (including deferred \
ones).
- \"toClarify\": short questions the user could send to clarify vague items (no owner, no date, \
unclear scope, tentative decisions).
- \"followUpMessage\": a short, friendly plain-text message the user can send to the attendees: a \
greeting, one sentence of summary, the decisions, the action items with owners and dates, the open \
questions, and a sign-off without a name. Use \"- \" bullets, no Markdown headings.
Rules:
1. Use only facts from the input blocks. The board blocks (<decisions>, <action_items>, \
<open_questions>) were tracked live and are a starting point: correct them from <transcript> when \
it clearly says otherwise, but never add names, owners, dates, numbers or outcomes that are not in \
the input. Never guess an owner or a date.
2. Stay neutral about people: describe what was said and decided, never judge anyone's \
performance, never rate, rank or score anyone.
3. Be concise and factual; keep exact names and figures.
Output exactly one JSON object and nothing else (no prose, no code fences).";

fn list_block(tag: &str, header: &str, items: &[String], cap: usize) -> String {
    if items.is_empty() {
        return String::new();
    }
    let body: Vec<String> = items
        .iter()
        .map(|i| format!("- {}", truncate_to_tokens(&sanitize_line(i), 120)))
        .collect();
    format!(
        "{header}\n{}",
        block(tag, &truncate_to_tokens(&body.join("\n"), cap))
    )
}

fn topics_block(topics: &[Topic]) -> String {
    let lines: Vec<String> = topics
        .iter()
        .filter(|t| t.status != TopicStatus::Pending || t.planned)
        .map(|t| {
            let status = match t.status {
                TopicStatus::Covered => "covered",
                TopicStatus::Current => "discussed at the end",
                TopicStatus::Pending => "on the agenda, not discussed",
            };
            format!("- {} ({status})", sanitize_line(&t.name))
        })
        .collect();
    if lines.is_empty() {
        String::new()
    } else {
        format!("Topics:\n{}", block("topics", &lines.join("\n")))
    }
}

/// Transcript within `cap` tokens; keeps the opening and the end when it
/// does not fit.
fn transcript_block(lines: &[TranscriptLine], cap: usize) -> String {
    let rendered: Vec<String> = lines
        .iter()
        .filter(|l| !l.text.trim().is_empty())
        .map(|l| {
            format!(
                "[{}] {}: {}",
                clock(l.at_ms),
                sanitize_line(&l.speaker),
                truncate_to_tokens(&sanitize_line(&l.text), 400)
            )
        })
        .collect();
    if rendered.is_empty() || cap < 40 {
        return String::new();
    }
    let total: usize = rendered.iter().map(|l| estimate_tokens(l) + 1).sum();
    let body = if total <= cap {
        rendered.join("\n")
    } else {
        let head_cap = cap * 3 / 10;
        let tail_cap = cap - head_cap - 8;
        let mut head: Vec<&String> = Vec::new();
        let mut used = 0;
        for l in &rendered {
            let c = estimate_tokens(l) + 1;
            if used + c > head_cap {
                break;
            }
            used += c;
            head.push(l);
        }
        let mut tail: Vec<&String> = Vec::new();
        used = 0;
        for l in rendered.iter().rev().take(rendered.len() - head.len()) {
            let c = estimate_tokens(l) + 1;
            if used + c > tail_cap {
                break;
            }
            used += c;
            tail.push(l);
        }
        tail.reverse();
        let omitted = rendered.len() - head.len() - tail.len();
        let mut out: Vec<String> = head.into_iter().cloned().collect();
        out.push(format!("[… {omitted} lines omitted …]"));
        out.extend(tail.into_iter().cloned());
        out.join("\n")
    };
    format!(
        "Transcript ([mm:ss] since the meeting started):\n{}",
        block("transcript", &body)
    )
}

fn label(role: SpeakerRole) -> &'static str {
    role.label(Mode::Workplace)
}

/// Builds the minutes prompt within roughly `budget_tokens` of user content.
pub fn build(input: &MinutesInput, budget_tokens: usize) -> PromptParts {
    let system = assemble(&[
        "You are OpenMeet's meeting minute-taker. You write the minutes of a team meeting that has \
         just ended — a stand-up, review, planning session or 1:1 between colleagues. Nobody was \
         interviewed or evaluated."
            .to_string(),
        DATA_RULE.to_string(),
        TRANSCRIPT_NOTE.to_string(),
        TASK.to_string(),
    ]);
    let budget = budget_tokens.max(800);
    let small = budget / 10;
    let title = sanitize_line(input.title);
    let meeting = format!(
        "Meeting details:\n{}",
        block(
            "meeting",
            &format!(
                "Title: {}\n{}",
                if title.is_empty() {
                    "Untitled meeting"
                } else {
                    &title
                },
                mode_line(Mode::Workplace)
            )
        )
    );
    let earlier = sanitize(input.long_term_summary.trim());
    let earlier = if earlier.trim().is_empty() {
        String::new()
    } else {
        format!(
            "Summary of the early part of the meeting:\n{}",
            block("summary", &truncate_to_tokens(earlier.trim(), budget / 6))
        )
    };
    let observations = input.observations.to_vec();
    let before = vec![
        meeting,
        earlier,
        list_block(
            "decisions",
            "Decisions tracked during the meeting:",
            input.decisions,
            small,
        ),
        list_block(
            "action_items",
            "Action items tracked during the meeting:",
            &action_item_lines(input.action_items),
            small,
        ),
        list_block(
            "open_questions",
            "Questions asked during the meeting:",
            &question_lines(input.questions, label),
            small,
        ),
        topics_block(input.topics),
        list_block(
            "observations",
            "What was shown on screen:",
            &observations,
            small / 2,
        ),
    ];
    let q_line = question_line("Write the meeting minutes JSON.");
    let used: usize = before.iter().map(|p| estimate_tokens(p)).sum();
    let cap = budget.saturating_sub(used + estimate_tokens(&q_line) + 16);
    let mut parts = before;
    parts.push(transcript_block(input.transcript, cap));
    parts.push(q_line);
    PromptParts {
        id: ID,
        version: VERSION,
        system,
        user: assemble(&parts),
    }
}

static PERFORMANCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:performed|performing|under-?perform\w*|did (?:a |an )?(?:poor|bad|great|good|terrible|excellent|amazing) job|(?:poor|weak|strong|bad|low|top) performer|performance (?:issue|problem|review|rating))\b",
    )
    .expect("performance pattern is valid")
});

/// Scores, ratings, hiring language or judgments of a colleague's
/// performance: never part of the minutes.
fn is_judgment(text: &str) -> bool {
    is_evaluative(text) || PERFORMANCE_RE.is_match(text)
}

fn clean_list(values: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for v in values {
        let v = one_line(&v);
        if v.is_empty() || is_judgment(&v) {
            continue;
        }
        if !out.iter().any(|x| x.eq_ignore_ascii_case(&v)) {
            out.push(v);
        }
    }
    out
}

fn list(obj: &Map<String, Value>, keys: &[&str]) -> Vec<String> {
    clean_list(get(obj, keys).map(value_to_str_list).unwrap_or_default())
}

fn opt_text(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => {
            let s = one_line(s);
            let lower = s.to_lowercase();
            (!s.is_empty()
                && !matches!(
                    lower.as_str(),
                    "null" | "none" | "n/a" | "na" | "unknown" | "tbd" | "not stated" | "-"
                ))
            .then_some(s)
        }
        _ => None,
    }
}

fn action_items(obj: &Map<String, Value>) -> Vec<MinutesActionItem> {
    let Some(v) = get(obj, &["actionItems", "action_items", "actions", "nextSteps"]) else {
        return Vec::new();
    };
    let mut out: Vec<MinutesActionItem> = Vec::new();
    let mut push = |item: MinutesActionItem| {
        if item.task.is_empty() || is_judgment(&item.task) {
            return;
        }
        if !out.iter().any(|x| x.task.eq_ignore_ascii_case(&item.task)) {
            out.push(item);
        }
    };
    match v {
        Value::Array(values) => {
            for value in values {
                match value {
                    Value::Object(m) => push(MinutesActionItem {
                        task: opt_text(get(m, &["task", "text", "item", "action"]))
                            .unwrap_or_default(),
                        owner: opt_text(get(m, &["owner", "assignee", "who"])),
                        due: opt_text(get(m, &["due", "dueDate", "deadline", "when"])),
                    }),
                    Value::String(s) => push(MinutesActionItem {
                        task: one_line(s),
                        owner: None,
                        due: None,
                    }),
                    _ => {}
                }
            }
        }
        other => {
            for s in value_to_str_list(other) {
                push(MinutesActionItem {
                    task: one_line(&s),
                    owner: None,
                    due: None,
                });
            }
        }
    }
    out
}

/// Parses the model's JSON; `None` if unusable. `markdown` is left empty
/// (render it with [`crate::workplace::render_minutes_markdown`]). A missing
/// follow-up message is drafted from the parsed minutes.
pub fn parse(text: &str, meeting_id: &str, title: &str, generated_by: &str) -> Option<MeetingSummary> {
    let mut obj = first_json_object(text)?;
    if get(&obj, &["overview"]).is_none()
        && let Some(Value::Object(inner)) = get(&obj, &["minutes", "summary", "result"])
    {
        obj = inner.clone();
    }
    let overview = match get(&obj, &["overview", "summary"]) {
        Some(Value::String(s)) => one_line(s),
        Some(v @ Value::Array(_)) => value_to_str_list(v).join(" "),
        _ => String::new(),
    };
    let overview = crate::text::split_sentences(&overview)
        .into_iter()
        .filter(|s| !is_judgment(s))
        .collect::<Vec<_>>()
        .join(" ");
    let items = action_items(&obj);
    let mut summary = MeetingSummary {
        meeting_id: meeting_id.to_string(),
        overview,
        key_topics: list(&obj, &["keyTopics", "topics"]),
        questions_asked: Vec::new(),
        answers: Vec::new(),
        decisions: list(&obj, &["decisions"]),
        action_items: items.iter().map(minutes_item_line).collect(),
        open_questions: list(&obj, &["openQuestions", "open_questions", "unresolved"]),
        technical_topics: Vec::new(),
        follow_up_suggestions: list(&obj, &["toClarify", "clarify", "followUps"]),
        interviewer: None,
        workplace: None,
        markdown: String::new(),
        generated_at: now_ms(),
        generated_by: generated_by.to_string(),
    };
    let has_content = !summary.overview.is_empty()
        || !summary.decisions.is_empty()
        || !items.is_empty()
        || !summary.open_questions.is_empty();
    if !has_content {
        return None;
    }
    let message = match get(&obj, &["followUpMessage", "follow_up_message", "message", "email"]) {
        Some(Value::String(s)) if !s.trim().is_empty() => {
            let kept: Vec<&str> = s
                .lines()
                .filter(|l| !is_judgment(l))
                .collect();
            kept.join("\n").trim().to_string()
        }
        _ => String::new(),
    };
    let message = if message.is_empty() {
        follow_up_message(title, &summary, &items)
    } else {
        message
    };
    summary.workplace = Some(WorkplaceMinutes {
        action_items: items,
        follow_up_message: message,
    });
    Some(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::shared::{extract_block, extract_block_items, extract_question};

    #[test]
    fn build_uses_the_board_and_forbids_guessing() {
        let t = vec![TranscriptLine {
            speaker: "Others".into(),
            text: "Priya will fix the login timeout by Thursday.".into(),
            at_ms: 12_000,
        }];
        let items = vec![ActionItem {
            id: "a".into(),
            meeting_id: "m".into(),
            text: "We need to update the runbook.".into(),
            owner: None,
            due: None,
            source_segment_id: None,
            created_at: 0,
        }];
        let qs = vec![OpenQuestion {
            id: "q".into(),
            question: "Do we have a date for the cutover?".into(),
            speaker_role: SpeakerRole::Remote,
            addressed_to_you: false,
            asked_at_ms: 60_000,
            answered: false,
            answer: None,
            deferred: true,
        }];
        let decisions = vec!["Let's go with the token refresh fix.".to_string()];
        let p = build(
            &MinutesInput {
                title: "Sprint planning",
                long_term_summary: "",
                transcript: &t,
                decisions: &decisions,
                action_items: &items,
                questions: &qs,
                topics: &[],
                observations: &[],
            },
            6000,
        );
        assert_eq!((p.id, p.version), (ID, VERSION));
        assert!(p.system.contains("Never guess an owner or a date"));
        assert!(p.system.contains("followUpMessage"));
        assert!(p.system.contains("data captured from the"));
        assert!(p.system.contains("never judge anyone's performance"));
        for word in ["hire", "hiring", "Candidate", "Interviewer"] {
            assert!(!p.system.contains(word) && !p.user.contains(word), "{word}");
        }
        assert_eq!(
            extract_block_items(&p.user, "decisions"),
            vec!["Let's go with the token refresh fix."]
        );
        assert_eq!(
            extract_block_items(&p.user, "action_items"),
            vec!["We need to update the runbook. | owner: not stated | due: not stated"]
        );
        assert!(
            extract_block(&p.user, "open_questions")
                .unwrap()
                .contains("[01:00] Others (deferred, still open): Do we have a date")
        );
        assert!(
            extract_block(&p.user, "meeting")
                .unwrap()
                .contains("Mode: workplace")
        );
        assert!(
            extract_block(&p.user, "transcript")
                .unwrap()
                .contains("[00:12] Others: Priya will fix")
        );
        assert!(extract_question(&p.user).is_some());
    }

    #[test]
    fn parse_structured_minutes_and_drop_evaluations() {
        let json = r#"Sure:
```json
{
  "overview": "Sprint planning for the login fix and the docs. Priya performed poorly last sprint.",
  "keyTopics": ["Login timeout", "Runbook"],
  "decisions": ["Ship the token refresh fix this sprint", "Score: 7/10"],
  "actionItems": [
    {"task": "Fix the login timeout", "owner": "Priya", "due": "by Thursday"},
    {"task": "Update the runbook", "owner": "You", "due": null},
    {"task": "Check with the data team", "owner": "null", "due": "N/A"},
    "Book the retro room"
  ],
  "openQuestions": ["Date for the production cutover"],
  "toClarify": ["When will the runbook update be done?"],
  "followUpMessage": "Hi all,\n\nQuick recap of sprint planning.\n- Priya: fix the login timeout by Thursday\n\nThanks!"
}
```"#;
        let s = parse(json, "m1", "Sprint planning", "ollama/llama3.2").unwrap();
        assert_eq!(s.meeting_id, "m1");
        assert_eq!(s.overview, "Sprint planning for the login fix and the docs.");
        assert_eq!(s.decisions, vec!["Ship the token refresh fix this sprint"]);
        assert_eq!(s.open_questions, vec!["Date for the production cutover"]);
        assert_eq!(s.follow_up_suggestions, vec!["When will the runbook update be done?"]);
        let w = s.workplace.as_ref().unwrap();
        assert_eq!(w.action_items.len(), 4);
        assert_eq!(w.action_items[0].owner.as_deref(), Some("Priya"));
        assert_eq!(w.action_items[0].due.as_deref(), Some("by Thursday"));
        assert_eq!(w.action_items[1].due, None);
        assert_eq!(w.action_items[2].owner, None, "\"null\" strings are no owner");
        assert_eq!(w.action_items[2].due, None);
        assert_eq!(w.action_items[3].task, "Book the retro room");
        assert_eq!(
            s.action_items[0],
            "Fix the login timeout — Priya, by Thursday"
        );
        assert!(w.follow_up_message.starts_with("Hi all,\n\nQuick recap"));
        assert!(s.interviewer.is_none());
    }

    #[test]
    fn parse_drafts_a_message_and_rejects_garbage() {
        let s = parse(
            r#"{"minutes": {"overview": "Short sync.", "decisions": ["Keep the release date"]}}"#,
            "m",
            "Weekly sync",
            "x",
        )
        .unwrap();
        let w = s.workplace.unwrap();
        assert!(w.follow_up_message.contains("Thanks for joining Weekly sync."));
        assert!(w.follow_up_message.contains("- Keep the release date"));
        assert!(parse("no json here", "m", "t", "x").is_none());
        assert!(parse(r#"{"overview": ""}"#, "m", "t", "x").is_none());
    }
}
