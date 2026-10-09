//! Meeting export (Markdown, plain text, JSON) and the extractive summary
//! used when no LLM is available (spec §41, §42).
//!
//! Nothing here evaluates people: no scores, no hiring recommendations.

use std::collections::HashSet;

use openmeet_core::ids::now_ms;
use openmeet_core::ipc::{ExportFormat, ExportParts};
use openmeet_core::types::*;
use serde_json::{Value, json};

use crate::claims::extract_entities;
use crate::prompts::meeting_summary;
use crate::text::{lexical_similarity, truncate_to_tokens, word_count};
use crate::timefmt::{clock, duration, utc_datetime};

/// Renders `record` in `format`, including only the selected parts.
pub fn render(record: &MeetingRecord, format: ExportFormat, parts: &ExportParts) -> String {
    match format {
        ExportFormat::Markdown => render_markdown(record, parts),
        ExportFormat::Text => render_text(record, parts),
        ExportFormat::Json => render_json(record, parts),
    }
}

fn mode_label(mode: Mode) -> &'static str {
    match mode {
        Mode::Candidate => "Candidate",
        Mode::Interviewer => "Interviewer",
    }
}

fn header_line(record: &MeetingRecord) -> String {
    let m = &record.meeting;
    let length = m
        .ended_at
        .map(|end| duration(end - m.started_at))
        .unwrap_or_else(|| "in progress".to_string());
    format!(
        "{} · {} · {} mode",
        utc_datetime(m.started_at),
        length,
        mode_label(m.mode)
    )
}

fn summary_of(record: &MeetingRecord) -> MeetingSummary {
    record
        .summary
        .clone()
        .unwrap_or_else(|| extractive_summary(record))
}

/// Answers given to a question (spoken answers recorded in the meeting).
fn answers_for<'a>(record: &'a MeetingRecord, q: &DetectedQuestion) -> Vec<&'a AnswerRecord> {
    record
        .answers
        .iter()
        .filter(|a| a.question_id.as_deref() == Some(q.id.as_str()))
        .collect()
}

/// AI suggestions produced for a question.
fn suggestions_for<'a>(record: &'a MeetingRecord, q: &DetectedQuestion) -> Vec<&'a Suggestion> {
    record
        .suggestions
        .iter()
        .filter(|s| {
            s.question_id.as_deref() == Some(q.id.as_str()) && s.kind != SuggestionKind::Followups
        })
        .collect()
}

/// Visible answer text of a suggestion (the [[ANSWER]] section, or the whole
/// text for unformatted output).
fn suggestion_answer(s: &Suggestion) -> String {
    let sections = crate::prompts::candidate_answer::parse(&s.content);
    if !sections.answer.trim().is_empty() {
        sections.answer.trim().to_string()
    } else {
        s.content.trim().to_string()
    }
}

fn all_followups(record: &MeetingRecord) -> Vec<&FollowUpQuestion> {
    record
        .suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::Followups)
        .flat_map(|s| s.followups.iter())
        .collect()
}

fn speaker(record: &MeetingRecord, role: SpeakerRole) -> &'static str {
    role.label(record.meeting.mode)
}

/// Markdown export.
pub fn render_markdown(record: &MeetingRecord, parts: &ExportParts) -> String {
    let m = &record.meeting;
    let mut out = format!("# {}\n\n_{}_\n", m.title.trim(), header_line(record));
    if parts.summary {
        let summary = summary_of(record);
        let md = if summary.markdown.trim().is_empty() {
            meeting_summary::render_markdown(&summary, &m.title, m.mode)
        } else {
            summary.markdown.clone()
        };
        // Drop the summary's own title; demote its headings one level.
        let body: Vec<String> = md
            .lines()
            .skip_while(|l| l.starts_with("# ") || l.trim().is_empty())
            .map(|l| {
                if l.starts_with('#') {
                    format!("#{l}")
                } else {
                    l.to_string()
                }
            })
            .collect();
        out.push_str("\n## Summary\n\n");
        out.push_str(body.join("\n").trim());
        out.push('\n');
        if summary.generated_by == "extractive" {
            out.push_str("\n_Summary generated without an AI model._\n");
        }
    }
    if parts.questions {
        out.push_str("\n## Questions and answers\n");
        if record.questions.is_empty() {
            out.push_str("\nNo questions were detected.\n");
        }
        for q in &record.questions {
            out.push_str(&format!(
                "\n### {} — {}\n\n**{}** ({}, {})\n",
                clock(
                    record
                        .segments
                        .iter()
                        .find(|s| q.segment_ids.contains(&s.id))
                        .map(|s| s.start_time)
                        .unwrap_or(0)
                ),
                q.category.label(),
                q.question.trim(),
                speaker(record, q.speaker_role),
                if q.requires_response {
                    "needs an answer"
                } else {
                    "asked"
                }
            ));
            for a in answers_for(record, q) {
                out.push_str(&format!(
                    "\n> **{} answered:** {}\n",
                    speaker(record, a.speaker_role),
                    a.text.trim()
                ));
            }
            for s in suggestions_for(record, q) {
                out.push_str(&format!(
                    "\n*OpenMeet suggestion:* {}\n",
                    suggestion_answer(s).replace('\n', " ")
                ));
            }
        }
    }
    if parts.followups {
        let followups = all_followups(record);
        if !followups.is_empty() {
            out.push_str("\n## Follow-up questions suggested\n\n");
            for f in followups {
                out.push_str(&format!(
                    "- **{}** _({}, {})_ — {}\n",
                    f.question.trim(),
                    f.category.label(),
                    match f.priority {
                        Priority::High => "high priority",
                        Priority::Normal => "normal",
                        Priority::Optional => "optional",
                    },
                    f.reason.trim()
                ));
            }
        }
    }
    if parts.transcript {
        out.push_str("\n## Transcript\n\n");
        if record.segments.is_empty() {
            out.push_str("No transcript.\n");
        }
        for s in record.segments.iter().filter(|s| s.is_final) {
            out.push_str(&format!(
                "**[{}] {}:** {}\n\n",
                clock(s.start_time),
                speaker(record, s.speaker_role),
                s.text.trim()
            ));
        }
    }
    out.trim_end().to_string() + "\n"
}

fn push_list(out: &mut String, title: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(&format!("\n{title}\n"));
    for i in items {
        out.push_str(&format!("  - {}\n", i.trim()));
    }
}

/// Plain-text export.
pub fn render_text(record: &MeetingRecord, parts: &ExportParts) -> String {
    let m = &record.meeting;
    let title = m.title.trim();
    let mut out = format!(
        "{title}\n{}\n{}\n",
        "=".repeat(title.chars().count().max(3)),
        header_line(record)
    );
    if parts.summary {
        let s = summary_of(record);
        out.push_str("\nSUMMARY\n\n");
        out.push_str(if s.overview.trim().is_empty() {
            "No overview available."
        } else {
            s.overview.trim()
        });
        out.push('\n');
        push_list(&mut out, "Key topics:", &s.key_topics);
        push_list(&mut out, "Questions asked:", &s.questions_asked);
        push_list(&mut out, "Answers:", &s.answers);
        push_list(&mut out, "Decisions:", &s.decisions);
        push_list(&mut out, "Action items:", &s.action_items);
        push_list(&mut out, "Open questions:", &s.open_questions);
        push_list(&mut out, "Technical topics:", &s.technical_topics);
        push_list(&mut out, "Follow-up suggestions:", &s.follow_up_suggestions);
        if let Some(iv) = &s.interviewer {
            push_list(
                &mut out,
                "Interview question coverage:",
                &iv.question_coverage,
            );
            push_list(
                &mut out,
                "Areas requiring clarification:",
                &iv.clarification_areas,
            );
            push_list(
                &mut out,
                "Candidate claims to verify:",
                &iv.claims_to_verify,
            );
            push_list(&mut out, "Topics not yet explored:", &iv.unexplored_topics);
        }
    }
    if parts.questions {
        out.push_str("\nQUESTIONS AND ANSWERS\n");
        for q in &record.questions {
            out.push_str(&format!(
                "\nQ ({}): {}\n",
                speaker(record, q.speaker_role),
                q.question.trim()
            ));
            for a in answers_for(record, q) {
                out.push_str(&format!(
                    "A ({}): {}\n",
                    speaker(record, a.speaker_role),
                    a.text.trim()
                ));
            }
            for s in suggestions_for(record, q) {
                out.push_str(&format!(
                    "OpenMeet suggestion: {}\n",
                    suggestion_answer(s).replace('\n', " ")
                ));
            }
        }
    }
    if parts.followups {
        let f = all_followups(record);
        if !f.is_empty() {
            out.push_str("\nFOLLOW-UP QUESTIONS SUGGESTED\n\n");
            for f in f {
                out.push_str(&format!(
                    "- {} [{}] {}\n",
                    f.question.trim(),
                    f.category.label(),
                    f.reason.trim()
                ));
            }
        }
    }
    if parts.transcript {
        out.push_str("\nTRANSCRIPT\n\n");
        for s in record.segments.iter().filter(|s| s.is_final) {
            out.push_str(&format!(
                "[{}] {}: {}\n",
                clock(s.start_time),
                speaker(record, s.speaker_role),
                s.text.trim()
            ));
        }
    }
    out.trim_end().to_string() + "\n"
}

/// JSON export of the selected parts (pretty-printed).
pub fn render_json(record: &MeetingRecord, parts: &ExportParts) -> String {
    let mut root = serde_json::Map::new();
    root.insert("format".into(), json!("openmeet.meeting.v1"));
    root.insert("exportedAt".into(), json!(now_ms()));
    root.insert(
        "meeting".into(),
        serde_json::to_value(&record.meeting).unwrap_or(Value::Null),
    );
    if parts.summary {
        root.insert(
            "summary".into(),
            serde_json::to_value(summary_of(record)).unwrap_or(Value::Null),
        );
        root.insert(
            "topics".into(),
            serde_json::to_value(&record.topics).unwrap_or(Value::Null),
        );
        root.insert(
            "claims".into(),
            serde_json::to_value(&record.claims).unwrap_or(Value::Null),
        );
        root.insert(
            "actionItems".into(),
            serde_json::to_value(&record.action_items).unwrap_or(Value::Null),
        );
        root.insert(
            "screenObservations".into(),
            serde_json::to_value(&record.observations).unwrap_or(Value::Null),
        );
    }
    if parts.questions {
        let qs: Vec<Value> = record
            .questions
            .iter()
            .map(|q| {
                json!({
                    "question": q,
                    "answers": answers_for(record, q),
                    "suggestions": suggestions_for(record, q),
                })
            })
            .collect();
        root.insert("questions".into(), Value::Array(qs));
    }
    if parts.followups {
        let f: Vec<&FollowUpQuestion> = all_followups(record);
        root.insert(
            "followups".into(),
            serde_json::to_value(f).unwrap_or(Value::Null),
        );
    }
    if parts.transcript {
        let segs: Vec<&TranscriptSegment> = record.segments.iter().filter(|s| s.is_final).collect();
        root.insert(
            "transcript".into(),
            serde_json::to_value(segs).unwrap_or(Value::Null),
        );
    }
    serde_json::to_string_pretty(&Value::Object(root)).unwrap_or_else(|_| "{}".to_string()) + "\n"
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

/// A useful summary without any LLM: overview, questions, answers, topics,
/// decisions, action items, technical topics; interviewer coverage notes.
/// `generated_by` is "extractive".
pub fn extractive_summary(record: &MeetingRecord) -> MeetingSummary {
    let m = &record.meeting;
    let mode = m.mode;
    let finals: Vec<&TranscriptSegment> = record.segments.iter().filter(|s| s.is_final).collect();

    // Topics: covered/current first by mentions.
    let mut topics: Vec<&Topic> = record
        .topics
        .iter()
        .filter(|t| t.status != TopicStatus::Pending || t.mentions >= 2)
        .collect();
    topics.sort_by_key(|t| {
        (
            t.status == TopicStatus::Pending,
            std::cmp::Reverse(t.mentions),
        )
    });
    let key_topics: Vec<String> = topics.iter().take(8).map(|t| t.name.clone()).collect();

    let mut questions_asked = Vec::new();
    for q in &record.questions {
        dedup_push(&mut questions_asked, q.question.clone(), 25);
    }
    let mut answers = Vec::new();
    for a in &record.answers {
        let q = a
            .question_id
            .as_ref()
            .and_then(|id| record.questions.iter().find(|q| &q.id == id))
            .map(|q| q.question.clone());
        let text = truncate_to_tokens(a.text.trim(), 70);
        let item = match q {
            Some(q) => format!("{} → {}", truncate_to_tokens(&q, 30), text),
            None => text,
        };
        dedup_push(&mut answers, item, 20);
    }
    let mut decisions = Vec::new();
    for c in record
        .claims
        .iter()
        .filter(|c| c.kind == ClaimKind::Decision)
    {
        dedup_push(&mut decisions, c.text.clone(), 10);
    }
    let mut action_items = Vec::new();
    for a in &record.action_items {
        let item = match &a.owner {
            Some(o) => format!("{} ({o})", a.text.trim()),
            None => a.text.trim().to_string(),
        };
        dedup_push(&mut action_items, item, 15);
    }
    let answered: HashSet<&str> = record
        .answers
        .iter()
        .filter_map(|a| a.question_id.as_deref())
        .collect();
    let mut open_questions = Vec::new();
    for q in record
        .questions
        .iter()
        .filter(|q| q.requires_response && !answered.contains(q.id.as_str()))
    {
        dedup_push(&mut open_questions, q.question.clone(), 10);
    }
    let mut technical_topics = Vec::new();
    let all_text: String = finals
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    for e in extract_entities(&all_text) {
        dedup_push(&mut technical_topics, e, 15);
    }
    for o in &record.observations {
        for e in &o.entities {
            dedup_push(&mut technical_topics, e.clone(), 15);
        }
    }
    let mut follow_up_suggestions = Vec::new();
    for f in record
        .suggestions
        .iter()
        .filter(|s| s.kind == SuggestionKind::Followups)
        .flat_map(|s| s.followups.iter())
    {
        dedup_push(&mut follow_up_suggestions, f.question.clone(), 10);
    }
    if mode == Mode::Interviewer {
        for t in record
            .topics
            .iter()
            .filter(|t| t.status == TopicStatus::Pending && (t.planned || t.mentions > 0))
        {
            dedup_push(
                &mut follow_up_suggestions,
                format!("Explore {}", t.name),
                10,
            );
        }
    }

    let mut overview = match m.ended_at {
        Some(end) => format!(
            "{} meeting \"{}\" ({}): {} transcript segments and {} detected questions.",
            mode_label(mode),
            m.title.trim(),
            duration(end - m.started_at),
            finals.len(),
            record.questions.len()
        ),
        None => format!(
            "Ongoing {} meeting \"{}\": {} transcript segments and {} detected questions so far.",
            mode_label(mode).to_lowercase(),
            m.title.trim(),
            finals.len(),
            record.questions.len()
        ),
    };
    if !key_topics.is_empty() {
        overview.push_str(&format!(
            " Main topics: {}.",
            key_topics
                .iter()
                .take(4)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !technical_topics.is_empty() {
        overview.push_str(&format!(
            " Technologies mentioned: {}.",
            technical_topics
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let interviewer = (mode == Mode::Interviewer).then(|| {
        let mut coverage = Vec::new();
        for t in &record.topics {
            let state = match t.status {
                TopicStatus::Covered | TopicStatus::Current => "discussed",
                TopicStatus::Pending if t.mentions > 0 => "mentioned briefly",
                TopicStatus::Pending => continue,
            };
            dedup_push(&mut coverage, format!("{} — {state}", t.name), 15);
        }
        let mut clarification = Vec::new();
        for a in &record.answers {
            if word_count(&a.text) < 12 {
                if let Some(q) = a
                    .question_id
                    .as_ref()
                    .and_then(|id| record.questions.iter().find(|q| &q.id == id))
                {
                    dedup_push(
                        &mut clarification,
                        format!("Short answer to: {}", truncate_to_tokens(&q.question, 30)),
                        10,
                    );
                }
            }
        }
        for c in record
            .claims
            .iter()
            .filter(|c| c.kind == ClaimKind::Unknown)
        {
            dedup_push(&mut clarification, c.text.clone(), 10);
        }
        let mut verify = Vec::new();
        for c in record.claims.iter().filter(|c| {
            c.speaker_role != SpeakerRole::User
                && matches!(
                    c.kind,
                    ClaimKind::Metric
                        | ClaimKind::Ownership
                        | ClaimKind::Claim
                        | ClaimKind::Implementation
                )
        }) {
            dedup_push(&mut verify, c.text.clone(), 10);
        }
        let mut unexplored = Vec::new();
        for t in record
            .topics
            .iter()
            .filter(|t| t.status == TopicStatus::Pending && (t.planned || t.mentions > 0))
        {
            dedup_push(&mut unexplored, t.name.clone(), 10);
        }
        InterviewerSummary {
            question_coverage: coverage,
            clarification_areas: clarification,
            claims_to_verify: verify,
            unexplored_topics: unexplored,
        }
    });

    let mut summary = MeetingSummary {
        meeting_id: m.id.clone(),
        overview,
        key_topics,
        questions_asked,
        answers,
        decisions,
        action_items,
        open_questions,
        technical_topics,
        follow_up_suggestions,
        interviewer,
        markdown: String::new(),
        generated_at: now_ms(),
        generated_by: "extractive".to_string(),
    };
    summary.markdown = meeting_summary::render_markdown(&summary, &m.title, mode);
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(mode: Mode) -> MeetingRecord {
        let meeting = Meeting {
            id: "m1".into(),
            title: "Backend interview".into(),
            mode,
            state: MeetingState::Completed,
            started_at: 1_760_000_000_000,
            ended_at: Some(1_760_000_000_000 + 45 * 60_000),
            participants: vec![],
            persisted: true,
        };
        let seg = |id: &str, role: SpeakerRole, at: i64, text: &str| TranscriptSegment {
            id: id.into(),
            meeting_id: "m1".into(),
            utterance_id: id.into(),
            speaker_id: None,
            speaker_role: role,
            source: AudioSourceKind::System,
            start_time: at,
            end_time: at + 3000,
            text: text.into(),
            is_final: true,
            confidence: None,
        };
        let (asker, answerer) = match mode {
            Mode::Interviewer => (SpeakerRole::User, SpeakerRole::Remote),
            Mode::Candidate => (SpeakerRole::Remote, SpeakerRole::User),
        };
        let q = DetectedQuestion {
            id: "q1".into(),
            meeting_id: "m1".into(),
            segment_ids: vec!["s1".into()],
            is_question: true,
            confidence: 0.9,
            question: "How would you design a rate limiter?".into(),
            parts: vec![],
            category: QuestionCategory::SystemDesign,
            speaker_id: None,
            speaker_role: asker,
            requires_response: mode == Mode::Candidate,
            method: DetectionMethod::Heuristic,
            supersedes: None,
            detected_at: 0,
        };
        MeetingRecord {
            meeting,
            segments: vec![
                seg("s1", asker, 1000, "How would you design a rate limiter?"),
                seg(
                    "s2",
                    answerer,
                    5000,
                    "I'd use Redis with a token bucket per API key.",
                ),
            ],
            questions: vec![q],
            answers: vec![AnswerRecord {
                id: "a1".into(),
                meeting_id: "m1".into(),
                question_id: Some("q1".into()),
                speaker_role: answerer,
                text: "I'd use Redis with a token bucket per API key.".into(),
                segment_ids: vec!["s2".into()],
                created_at: 0,
            }],
            suggestions: vec![Suggestion {
                id: "f1".into(),
                meeting_id: Some("m1".into()),
                kind: SuggestionKind::Followups,
                question_id: None,
                question_text: None,
                category: None,
                style: None,
                parent_id: None,
                prompt_id: "interviewer-followup".into(),
                prompt_version: 1,
                content: String::new(),
                followups: vec![FollowUpQuestion {
                    id: "fq".into(),
                    request_id: "f1".into(),
                    question: "What happens when Redis is unavailable?".into(),
                    category: QuestionCategory::Tradeoff,
                    reason: "Candidate relies on Redis but did not mention failure handling."
                        .into(),
                    priority: Priority::High,
                    trigger: Some("I'd use Redis".into()),
                    created_at: 0,
                }],
                provider: "mock".into(),
                model: "m".into(),
                created_at: 0,
                completed_at: Some(0),
                pinned: false,
            }],
            topics: vec![Topic {
                id: "system-design".into(),
                meeting_id: "m1".into(),
                name: "System design".into(),
                status: TopicStatus::Covered,
                mentions: 4,
                first_mentioned_at: Some(1000),
                last_mentioned_at: Some(5000),
                planned: false,
                summary: None,
            }],
            claims: vec![Claim {
                id: "c1".into(),
                meeting_id: "m1".into(),
                text: "We handle 10 million requests per day.".into(),
                kind: ClaimKind::Metric,
                segment_id: None,
                speaker_role: SpeakerRole::Remote,
                source: ExtractionSource::Heuristic,
                created_at: 0,
            }],
            observations: vec![],
            action_items: vec![],
            summary: None,
        }
    }

    #[test]
    fn markdown_has_selected_parts() {
        let r = record(Mode::Interviewer);
        let all = render_markdown(&r, &ExportParts::default());
        assert!(all.starts_with("# Backend interview"));
        assert!(all.contains("## Summary"));
        assert!(all.contains("## Questions and answers"));
        assert!(all.contains("How would you design a rate limiter?"));
        assert!(all.contains("Candidate answered:"), "{all}");
        assert!(all.contains("## Follow-up questions suggested"));
        assert!(all.contains("What happens when Redis is unavailable?"));
        assert!(all.contains("## Transcript"));
        assert!(all.contains("[00:01] You:"), "{all}");
        let only = render_markdown(
            &r,
            &ExportParts {
                summary: false,
                transcript: true,
                questions: false,
                followups: false,
            },
        );
        assert!(!only.contains("## Summary"));
        assert!(only.contains("## Transcript"));
    }

    #[test]
    fn text_and_json() {
        let r = record(Mode::Candidate);
        let t = render_text(&r, &ExportParts::default());
        assert!(t.contains("SUMMARY") && t.contains("TRANSCRIPT"));
        assert!(t.contains("Q (Interviewer): How would you design a rate limiter?"));
        let j: Value = serde_json::from_str(&render_json(&r, &ExportParts::default())).unwrap();
        assert_eq!(j["format"], "openmeet.meeting.v1");
        assert_eq!(
            j["questions"][0]["answers"][0]["text"],
            "I'd use Redis with a token bucket per API key."
        );
        assert!(j["transcript"].as_array().unwrap().len() == 2);
        let j: Value = serde_json::from_str(&render_json(
            &r,
            &ExportParts {
                summary: false,
                transcript: false,
                questions: false,
                followups: true,
            },
        ))
        .unwrap();
        assert!(j.get("transcript").is_none());
        assert!(j.get("followups").is_some());
    }

    #[test]
    fn extractive_summary_is_useful_and_non_evaluative() {
        let r = record(Mode::Interviewer);
        let s = extractive_summary(&r);
        assert_eq!(s.generated_by, "extractive");
        assert!(s.overview.contains("Backend interview"));
        assert!(
            s.questions_asked
                .contains(&"How would you design a rate limiter?".to_string())
        );
        assert!(s.answers[0].contains("Redis"));
        assert!(s.technical_topics.contains(&"Redis".to_string()));
        let iv = s.interviewer.as_ref().unwrap();
        assert!(
            iv.question_coverage
                .iter()
                .any(|c| c.starts_with("System design"))
        );
        assert!(iv.claims_to_verify.iter().any(|c| c.contains("10 million")));
        assert!(!s.markdown.trim().is_empty());
        let lower = s.markdown.to_lowercase();
        for word in ["hire", "score", "rating", "recommend"] {
            assert!(
                !lower.contains(word),
                "summary must not be evaluative: {word}"
            );
        }
        let c = extractive_summary(&record(Mode::Candidate));
        assert!(c.interviewer.is_none());
        assert!(c.open_questions.is_empty(), "the question was answered");
    }
}
