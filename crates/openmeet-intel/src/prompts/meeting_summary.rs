//! Post-meeting summary: prompt, tolerant JSON parser and Markdown renderer.
//!
//! Summaries describe what was discussed. They never contain hiring
//! recommendations, scores or evaluative judgments of the candidate — the
//! prompt forbids them and the parser drops any that slip through.

use std::sync::LazyLock;

use openmeet_core::ids::now_ms;
use openmeet_core::types::{
    Claim, InterviewerSummary, MeetingSummary, Mode, Topic, TopicStatus,
};
use regex::Regex;
use serde_json::{Map, Value};

use super::shared::{
    DATA_RULE, NON_EVALUATIVE_RULE, TRANSCRIPT_NOTE, assemble, block, claim_kind_slug, mode_line,
    one_line, question_line, sanitize, sanitize_line,
};
use super::{PromptParts, TranscriptLine};
use crate::json::{first_json_object, get, value_to_str_list};
use crate::text::{estimate_tokens, truncate_to_tokens};
use crate::timefmt::{clock, utc_datetime};

/// Stable prompt identifier.
pub const ID: &str = "meeting-summary";
/// Prompt template version.
pub const VERSION: u32 = 1;

/// Everything the summary prompt may use.
pub struct SummaryInput<'a> {
    /// Assistance mode of the meeting.
    pub mode: Mode,
    /// Meeting title.
    pub title: &'a str,
    /// Rolling summary of the early part of the meeting (may be empty).
    pub long_term_summary: &'a str,
    /// Transcript, oldest first.
    pub transcript: &'a [TranscriptLine],
    /// Detected questions.
    pub questions: &'a [String],
    /// Answers given (by the user in candidate mode, the candidate in
    /// interviewer mode).
    pub answers: &'a [String],
    /// Follow-up questions OpenMeet suggested.
    pub followups: &'a [String],
    /// Conversation topics.
    pub topics: &'a [Topic],
    /// Extracted claims.
    pub claims: &'a [Claim],
    /// Screen observation descriptions.
    pub observations: &'a [String],
    /// Action items.
    pub action_items: &'a [String],
}

const ITEM_TOKENS: usize = 120;

fn task(mode: Mode) -> String {
    let interviewer = match mode {
        Mode::Interviewer => {
            "\n- \"interviewer\": an object with \"questionCoverage\" (planned topics and the \
questions asked, each with whether it was covered, e.g. \"Rate limiting — covered\"), \
\"clarificationAreas\" (points the candidate's answers left unspecified — say what is missing, not \
how good the answer was), \"claimsToVerify\" (specific, checkable statements such as metrics, \
ownership or production experience, quoted or closely paraphrased) and \"unexploredTopics\" (planned \
or relevant topics that were not discussed)."
        }
        Mode::Candidate => "",
    };
    format!(
        "Task: write the post-meeting summary for the user as one JSON object with these keys \
(every list is an array of short strings; use [] when nothing applies):
- \"overview\": 2–4 sentences on what the meeting covered.
- \"keyTopics\": the main topics, most important first.
- \"questionsAsked\": the questions that were asked, close to their wording, in order.
- \"answers\": one line per answered question: the question in a few words, then the key points of \
the answer that was given.
- \"decisions\": decisions that were made or stated.
- \"actionItems\": commitments and next steps, with the owner when stated.
- \"openQuestions\": questions or topics left unresolved.
- \"technicalTopics\": technologies, systems and techniques discussed.
- \"followUpSuggestions\": useful next steps for the user (things to look into, prepare or ask \
next), grounded in the conversation.{interviewer}
Rules:
1. Use only facts from the input blocks. Never add names, numbers, technologies or outcomes that \
are not in them. Keep exact names and figures.
2. {NON_EVALUATIVE_RULE} The summary describes what was said, never how well anyone performed: no \
scores, ratings, strengths/weaknesses lists, verdicts or hire / no-hire language.
3. Be concise and factual; attribute statements to speakers by their labels where it matters.
Output exactly one JSON object and nothing else (no prose, no code fences)."
    )
}

fn list_block(tag: &str, header: &str, items: &[String], cap_tokens: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut used = 0usize;
    for item in items {
        let line = format!("- {}", truncate_to_tokens(&sanitize_line(item), ITEM_TOKENS));
        if line.trim() == "-" {
            continue;
        }
        let cost = estimate_tokens(&line) + 1;
        if used + cost > cap_tokens {
            lines.push("- …".to_string());
            break;
        }
        used += cost;
        lines.push(line);
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("{header}\n{}", block(tag, &lines.join("\n")))
    }
}

fn topics_block(topics: &[Topic]) -> String {
    if topics.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = topics
        .iter()
        .map(|t| {
            let status = match t.status {
                TopicStatus::Covered => "covered",
                TopicStatus::Current => "discussed at the end",
                TopicStatus::Pending => "not discussed",
            };
            let planned = if t.planned { ", planned" } else { "" };
            format!("- {} ({status}{planned})", sanitize_line(&t.name))
        })
        .collect();
    format!("Topics:\n{}", block("topics", &lines.join("\n")))
}

fn claims_block(claims: &[Claim], cap_tokens: usize) -> String {
    let items: Vec<String> = claims
        .iter()
        .filter(|c| !c.text.trim().is_empty())
        .map(|c| format!("[{}] {}", claim_kind_slug(c.kind), c.text.trim()))
        .collect();
    list_block(
        "claims",
        "Notable statements extracted during the meeting:",
        &items,
        cap_tokens,
    )
}

/// Renders the transcript within `cap` tokens; when it does not fit, keeps
/// the opening and the end and marks the omission.
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

/// Builds the summary prompt within roughly `budget_tokens` of user content.
pub fn build(input: &SummaryInput, budget_tokens: usize) -> PromptParts {
    let mode = input.mode;
    let system = assemble(&[
        "You are OpenMeet's meeting summarizer. You write a factual summary of a meeting that has \
         just ended, for the user who ran OpenMeet (\"You\" in the transcript)."
            .to_string(),
        DATA_RULE.to_string(),
        TRANSCRIPT_NOTE.to_string(),
        task(mode),
    ]);

    let budget = budget_tokens.max(800);
    let title = sanitize_line(input.title);
    let meeting = format!(
        "Meeting details:\n{}",
        block(
            "meeting",
            &format!(
                "Title: {}\n{}",
                if title.is_empty() { "Untitled meeting" } else { &title },
                mode_line(mode)
            )
        )
    );
    let summary_text = sanitize(input.long_term_summary.trim());
    let summary = if summary_text.trim().is_empty() {
        String::new()
    } else {
        format!(
            "Summary of the early part of the meeting:\n{}",
            block("summary", &truncate_to_tokens(summary_text.trim(), budget / 6))
        )
    };
    let small = budget / 10;
    let questions_header = match mode {
        Mode::Candidate => "Questions detected (asked by the interviewer / other participants):",
        Mode::Interviewer => "Questions detected (asked by the user and by the candidate):",
    };
    let answers_header = match mode {
        Mode::Candidate => "Answers the user gave:",
        Mode::Interviewer => "Answers the candidate gave:",
    };
    let parts_before_transcript = vec![
        meeting,
        summary,
        list_block("questions", questions_header, input.questions, small),
        list_block("answers", answers_header, input.answers, budget / 6),
        list_block(
            "followups",
            "Follow-up questions OpenMeet suggested:",
            input.followups,
            small / 2,
        ),
        topics_block(input.topics),
        claims_block(input.claims, small),
        list_block(
            "observations",
            "What was shown on screen:",
            input.observations,
            small / 2,
        ),
        list_block(
            "action_items",
            "Action items detected:",
            input.action_items,
            small / 2,
        ),
    ];
    let used: usize = parts_before_transcript
        .iter()
        .map(|p| estimate_tokens(p))
        .sum();
    let q_line = question_line("Write the meeting summary JSON.");
    let transcript_cap = budget.saturating_sub(used + estimate_tokens(&q_line) + 16);
    let mut parts = parts_before_transcript;
    parts.push(transcript_block(input.transcript, transcript_cap));
    parts.push(q_line);
    PromptParts {
        id: ID,
        version: VERSION,
        system,
        user: assemble(&parts),
    }
}

static EVALUATIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        \b(?:strong|weak|lean|leaning)\s+(?:no[\s-]+)?hire\b
        | \bno[\s-]hire\b
        | \bhire\s+recommendation
        | \bhiring\s+(?:recommendation|decision|verdict)
        | \brecommend(?:ed|s|ing)?\s+(?:to\s+)?(?:hire|hiring|not\s+hiring|rejecting|reject|advancing|advance|moving\s+(?:forward|ahead)|proceeding|passing)
        | \b(?:should|would)\s+(?:not\s+)?(?:be\s+)?(?:hired|hire\s+(?:him|her|them|the\s+candidate))
        | \b(?:reject|advance|pass)\s+(?:the\s+)?candidate\b
        | \bcandidate\s+(?:score|rating|grade)
        | \b(?:score|rating|grade)\s*(?:of|:)?\s*\d+(?:\.\d+)?\s*(?:/|out\s+of)\s*\d+
        | \b\d+(?:\.\d+)?\s*/\s*(?:5|10|100)\b
        | \b(?:scored|rated)\s+(?:the\s+)?candidate
        | \b(?:not\s+)?a\s+(?:good|strong|great|poor|bad)\s+(?:fit|hire|candidate)\b
        | \b(?:candidate|they|he|she)\s+(?:performed|did)\s+(?:well|poorly|badly|great|excellent)
        ",
    )
    .expect("valid evaluative regex")
});

/// True for hiring recommendations, scores and similar evaluative content
/// that a summary must never contain.
pub(crate) fn is_evaluative(text: &str) -> bool {
    EVALUATIVE_RE.is_match(text)
}

/// Removes evaluative sentences from free text.
fn strip_evaluative_sentences(text: &str) -> String {
    if !is_evaluative(text) {
        return text.to_string();
    }
    crate::text::split_sentences(text)
        .into_iter()
        .filter(|s| !is_evaluative(s))
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_list(values: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for v in values {
        let v = one_line(&v);
        if v.is_empty() || is_evaluative(&v) {
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

/// Parses the model's JSON; `None` if unusable. `markdown` is left empty
/// (render it with [`render_markdown`], which needs the meeting title).
pub fn parse(
    text: &str,
    meeting_id: &str,
    mode: Mode,
    generated_by: &str,
) -> Option<MeetingSummary> {
    let mut obj = first_json_object(text)?;
    // Some models wrap the object: {"summary": {...}}.
    if get(&obj, &["overview"]).is_none() {
        if let Some(Value::Object(inner)) = get(&obj, &["summary", "meeting_summary", "result"]) {
            obj = inner.clone();
        }
    }
    let overview = match get(&obj, &["overview", "summary"]) {
        Some(Value::String(s)) => strip_evaluative_sentences(&one_line(s)),
        Some(v @ Value::Array(_)) => {
            strip_evaluative_sentences(&value_to_str_list(v).join(" "))
        }
        _ => String::new(),
    };
    let interviewer = match mode {
        Mode::Candidate => None,
        Mode::Interviewer => {
            let source = match get(&obj, &["interviewer", "interviewer_notes"]) {
                Some(Value::Object(m)) => m.clone(),
                _ => obj.clone(),
            };
            Some(InterviewerSummary {
                question_coverage: list(&source, &["questionCoverage", "coverage"]),
                clarification_areas: list(
                    &source,
                    &["clarificationAreas", "areasToClarify", "clarifications"],
                ),
                claims_to_verify: list(&source, &["claimsToVerify", "claims"]),
                unexplored_topics: list(&source, &["unexploredTopics", "notDiscussed"]),
            })
        }
    };
    let summary = MeetingSummary {
        meeting_id: meeting_id.to_string(),
        overview,
        key_topics: list(&obj, &["keyTopics", "topics"]),
        questions_asked: list(&obj, &["questionsAsked", "questions"]),
        answers: list(&obj, &["answers", "answerSummaries"]),
        decisions: list(&obj, &["decisions"]),
        action_items: list(&obj, &["actionItems", "actions", "nextSteps"]),
        open_questions: list(&obj, &["openQuestions", "unresolved"]),
        technical_topics: list(&obj, &["technicalTopics", "technologies"]),
        follow_up_suggestions: list(
            &obj,
            &["followUpSuggestions", "followUps", "followups", "suggestions"],
        ),
        interviewer,
        markdown: String::new(),
        generated_at: now_ms(),
        generated_by: generated_by.to_string(),
    };
    let has_content = !summary.overview.is_empty()
        || !summary.key_topics.is_empty()
        || !summary.questions_asked.is_empty()
        || !summary.answers.is_empty()
        || !summary.decisions.is_empty()
        || !summary.action_items.is_empty()
        || !summary.technical_topics.is_empty();
    has_content.then_some(summary)
}

fn md_escape_line(s: &str) -> String {
    one_line(s)
}

fn push_section(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("\n## {heading}\n\n"));
    for i in items {
        out.push_str(&format!("- {}\n", md_escape_line(i)));
    }
}

fn push_subsection(out: &mut String, heading: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("\n### {heading}\n\n"));
    for i in items {
        out.push_str(&format!("- {}\n", md_escape_line(i)));
    }
}

/// Renders the summary as Markdown (never empty).
pub fn render_markdown(summary: &MeetingSummary, title: &str, mode: Mode) -> String {
    let title = one_line(title);
    let mut out = format!(
        "# {}\n",
        if title.is_empty() {
            "Meeting summary"
        } else {
            &title
        }
    );
    out.push_str("\n## Overview\n\n");
    if summary.overview.trim().is_empty() {
        out.push_str("No overview is available for this meeting.\n");
    } else {
        out.push_str(summary.overview.trim());
        out.push('\n');
    }
    push_section(&mut out, "Key topics", &summary.key_topics);
    push_section(&mut out, "Questions asked", &summary.questions_asked);
    push_section(&mut out, "Answers", &summary.answers);
    push_section(&mut out, "Decisions", &summary.decisions);
    push_section(&mut out, "Action items", &summary.action_items);
    push_section(&mut out, "Open questions", &summary.open_questions);
    push_section(&mut out, "Technical topics", &summary.technical_topics);
    push_section(
        &mut out,
        match mode {
            Mode::Candidate => "Suggested follow-ups",
            Mode::Interviewer => "Areas to explore next",
        },
        &summary.follow_up_suggestions,
    );
    if let Some(iv) = &summary.interviewer {
        let any = !iv.question_coverage.is_empty()
            || !iv.clarification_areas.is_empty()
            || !iv.claims_to_verify.is_empty()
            || !iv.unexplored_topics.is_empty();
        if any {
            out.push_str("\n## Interviewer notes\n");
            push_subsection(&mut out, "Question coverage", &iv.question_coverage);
            push_subsection(&mut out, "Areas to clarify", &iv.clarification_areas);
            push_subsection(&mut out, "Claims to verify", &iv.claims_to_verify);
            push_subsection(&mut out, "Unexplored topics", &iv.unexplored_topics);
        }
    }
    let source = if summary.generated_by.trim().is_empty() {
        String::new()
    } else if summary.generated_by == "extractive" {
        " without an AI model (extractive summary)".to_string()
    } else {
        format!(" by {}", one_line(&summary.generated_by))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::shared::{extract_block, extract_block_items, extract_question};
    use openmeet_core::types::{ClaimKind, ExtractionSource, SpeakerRole};

    fn line(speaker: &str, text: &str, at: i64) -> TranscriptLine {
        TranscriptLine {
            speaker: speaker.into(),
            text: text.into(),
            at_ms: at,
        }
    }

    fn input<'a>(
        mode: Mode,
        transcript: &'a [TranscriptLine],
        questions: &'a [String],
        claims: &'a [Claim],
    ) -> SummaryInput<'a> {
        SummaryInput {
            mode,
            title: "Backend interview",
            long_term_summary: "",
            transcript,
            questions,
            answers: &[],
            followups: &[],
            topics: &[],
            claims,
            observations: &[],
            action_items: &[],
        }
    }

    #[test]
    fn build_forbids_evaluation_and_includes_data() {
        let t = vec![
            line("You", "How would you design a rate limiter?", 0),
            line("Candidate", "I'd use Redis with a token bucket.", 5_000),
        ];
        let q = vec!["How would you design a rate limiter?".to_string()];
        let c = vec![Claim {
            id: "c".into(),
            meeting_id: "m".into(),
            text: "Uses Redis".into(),
            kind: ClaimKind::Technology,
            segment_id: None,
            speaker_role: SpeakerRole::Remote,
            source: ExtractionSource::Heuristic,
            created_at: 0,
        }];
        let p = build(&input(Mode::Interviewer, &t, &q, &c), 9000);
        assert_eq!(p.id, ID);
        assert!(p.system.contains("hiring recommendation"));
        assert!(p.system.contains("claimsToVerify"));
        assert!(p.system.contains("followUpSuggestions"));
        assert_eq!(
            extract_block_items(&p.user, "questions"),
            vec!["How would you design a rate limiter?"]
        );
        assert!(
            extract_block(&p.user, "transcript")
                .unwrap()
                .contains("[00:05] Candidate: I'd use Redis")
        );
        assert!(extract_block(&p.user, "meeting").unwrap().contains("Backend interview"));
        assert!(extract_question(&p.user).is_some());
        let cand = build(&input(Mode::Candidate, &t, &q, &c), 9000);
        assert!(!cand.system.contains("claimsToVerify"));
    }

    #[test]
    fn long_transcripts_keep_head_and_tail_within_budget() {
        let t: Vec<TranscriptLine> = (0..3000)
            .map(|i| line("Speaker", &format!("statement number {i} about the system"), i * 1000))
            .collect();
        let p = build(&input(Mode::Candidate, &t, &[], &[]), 4000);
        assert!(estimate_tokens(&p.user) <= 4200, "{}", estimate_tokens(&p.user));
        assert!(p.user.contains("statement number 0 "));
        assert!(p.user.contains("statement number 2999"));
        assert!(p.user.contains("lines omitted"));
    }

    #[test]
    fn parse_full_interviewer_summary_and_drop_evaluations() {
        let json = r#"Here it is:
```json
{
  "overview": "The interview covered rate limiter design. Overall a strong hire.",
  "keyTopics": ["Rate limiting", "Redis"],
  "questionsAsked": ["How would you design a rate limiter?"],
  "answers": ["Rate limiter: Redis token bucket per API key"],
  "decisions": [],
  "actionItems": ["Send the take-home exercise (You)"],
  "openQuestions": ["Multi-region behaviour"],
  "technicalTopics": ["Redis", "token bucket"],
  "followUpSuggestions": ["Ask about failure handling", "Candidate score: 8/10"],
  "interviewer": {
    "questionCoverage": ["Rate limiting — covered"],
    "clarificationAreas": ["Consistency model not specified"],
    "claimsToVerify": ["Ran Redis at 50k QPS"],
    "unexploredTopics": ["Observability"]
  }
}
```"#;
        let s = parse(json, "m1", Mode::Interviewer, "openai/gpt").unwrap();
        assert_eq!(s.meeting_id, "m1");
        assert_eq!(s.generated_by, "openai/gpt");
        assert_eq!(s.overview, "The interview covered rate limiter design.");
        assert_eq!(s.key_topics, vec!["Rate limiting", "Redis"]);
        assert_eq!(s.follow_up_suggestions, vec!["Ask about failure handling"]);
        let iv = s.interviewer.unwrap();
        assert_eq!(iv.claims_to_verify, vec!["Ran Redis at 50k QPS"]);
        assert_eq!(iv.unexplored_topics, vec!["Observability"]);
        assert!(s.markdown.is_empty());
    }

    #[test]
    fn parse_tolerates_wrappers_and_rejects_garbage() {
        let s = parse(
            "{\"summary\": {\"overview\": \"Talked about Kafka.\", \"keyTopics\": \"- Kafka\\n- Events\"}}",
            "m",
            Mode::Candidate,
            "x",
        )
        .unwrap();
        assert_eq!(s.overview, "Talked about Kafka.");
        assert_eq!(s.key_topics, vec!["Kafka", "Events"]);
        assert!(s.interviewer.is_none());
        assert!(parse("no json", "m", Mode::Candidate, "x").is_none());
        assert!(parse("{\"overview\": \"\"}", "m", Mode::Candidate, "x").is_none());
    }

    #[test]
    fn evaluative_detection() {
        for bad in [
            "Strong hire.",
            "I recommend hiring this candidate.",
            "Recommend moving forward to the next round",
            "Score: 7/10",
            "The candidate should be hired",
            "Not a good fit for the team",
            "Candidate rating: 4 out of 5",
            "Lean no hire",
        ] {
            assert!(is_evaluative(bad), "{bad}");
        }
        for ok in [
            "Discussed the credit score service and its PostgreSQL schema.",
            "Hiring manager will send the schedule.",
            "Latency dropped by 40%.",
            "Used a 24/7 on-call rotation.",
        ] {
            assert!(!is_evaluative(ok), "{ok}");
        }
    }

    #[test]
    fn markdown_rendering() {
        let s = MeetingSummary {
            meeting_id: "m".into(),
            overview: "We discussed caching.".into(),
            key_topics: vec!["Caching".into()],
            action_items: vec!["Send notes".into()],
            interviewer: Some(InterviewerSummary {
                claims_to_verify: vec!["Cut p99 by 40%".into()],
                ..Default::default()
            }),
            generated_by: "extractive".into(),
            generated_at: 1_790_953_200_000,
            ..Default::default()
        };
        let md = render_markdown(&s, "Design sync", Mode::Interviewer);
        assert!(md.starts_with("# Design sync\n"));
        assert!(md.contains("## Overview\n\nWe discussed caching."));
        assert!(md.contains("## Key topics\n\n- Caching"));
        assert!(md.contains("### Claims to verify\n\n- Cut p99 by 40%"));
        assert!(md.contains("without an AI model"));
        assert!(md.contains("2026-10-02 15:00 UTC"));
        assert!(!md.contains("## Decisions"));
        let empty = render_markdown(&MeetingSummary::default(), "", Mode::Candidate);
        assert!(empty.starts_with("# Meeting summary"));
        assert!(empty.contains("No overview"));
    }
}
