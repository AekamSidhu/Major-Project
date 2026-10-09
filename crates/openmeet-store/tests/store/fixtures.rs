//! Builders for domain values used across the integration tests.

use std::path::Path;

use openmeet_core::store::{EmbeddingRecord, MeetingStore};
use openmeet_core::types::*;
use openmeet_store::Database;

/// A fixed "now" (2025-10-09) so tests never depend on the wall clock.
pub const NOW: i64 = 1_760_000_000_000;
pub const HOUR: i64 = 60 * 60 * 1000;
pub const DAY: i64 = 24 * HOUR;

/// Marker embedded in every piece of meeting content, used to prove that
/// content never reaches logs or error messages.
pub const SECRET: &str = "SECRET-CONTENT-4242";

pub fn secret(text: &str) -> String {
    format!("{text} {SECRET}")
}

/// A file database in `dir`, plus a raw connection helper for inspection.
pub fn open_file_db(dir: &Path) -> Database {
    Database::open(&dir.join("openmeet.db")).expect("open database")
}

pub fn raw_connection(dir: &Path) -> rusqlite::Connection {
    let conn = rusqlite::Connection::open(dir.join("openmeet.db")).expect("raw connection");
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    conn
}

pub fn meeting(id: &str, started_at: i64) -> Meeting {
    Meeting {
        id: id.to_string(),
        title: secret("Platform interview"),
        mode: Mode::Interviewer,
        state: MeetingState::Active,
        started_at,
        ended_at: None,
        participants: vec![
            Participant {
                id: "user".into(),
                role: SpeakerRole::User,
                label: "You".into(),
            },
            Participant {
                id: "remote-1".into(),
                role: SpeakerRole::Remote,
                label: "Candidate".into(),
            },
        ],
        persisted: true,
    }
}

pub fn segment(meeting_id: &str, id: &str, start_time: i64, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        utterance_id: format!("utt-{id}"),
        speaker_id: Some("remote-1".into()),
        speaker_role: SpeakerRole::Remote,
        source: AudioSourceKind::System,
        start_time,
        end_time: start_time + 2_500,
        text: secret(text),
        is_final: true,
        confidence: Some(0.875),
    }
}

pub fn question(meeting_id: &str, id: &str, detected_at: i64) -> DetectedQuestion {
    DetectedQuestion {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        segment_ids: vec!["s1".into(), "s2".into()],
        is_question: true,
        confidence: 0.75,
        question: secret("How would you design a rate limiter?"),
        parts: vec![
            secret("How would you design it?"),
            secret("How does it scale?"),
        ],
        category: QuestionCategory::SystemDesign,
        speaker_id: Some("remote-1".into()),
        speaker_role: SpeakerRole::Remote,
        requires_response: true,
        method: DetectionMethod::Heuristic,
        supersedes: None,
        detected_at,
    }
}

pub fn answer(meeting_id: &str, id: &str, created_at: i64) -> AnswerRecord {
    AnswerRecord {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        question_id: Some("q1".into()),
        speaker_role: SpeakerRole::Remote,
        text: secret("I'd probably use Redis with a token bucket"),
        segment_ids: vec!["s3".into()],
        created_at,
    }
}

pub fn suggestion(meeting_id: Option<&str>, id: &str, created_at: i64) -> Suggestion {
    Suggestion {
        id: id.to_string(),
        meeting_id: meeting_id.map(str::to_string),
        kind: SuggestionKind::Followups,
        question_id: Some("q1".into()),
        question_text: Some(secret("How would you design a rate limiter?")),
        category: Some(QuestionCategory::Tradeoff),
        style: Some(AnswerStyle::Star),
        parent_id: Some("parent-1".into()),
        prompt_id: "interviewer-followup".into(),
        prompt_version: 3,
        content: secret("{\"type\":\"followup\"}"),
        followups: vec![FollowUpQuestion {
            id: format!("{id}-f1"),
            request_id: id.to_string(),
            question: secret("What happens when Redis is unavailable?"),
            category: QuestionCategory::Tradeoff,
            reason: secret("Candidate mentioned Redis but not failure modes"),
            priority: Priority::High,
            trigger: Some(secret("I'd probably use Redis")),
            created_at,
        }],
        provider: "anthropic".into(),
        model: "claude-opus-5-5".into(),
        created_at,
        completed_at: Some(created_at + 1_500),
        pinned: false,
    }
}

pub fn topic(meeting_id: &str, id: &str, status: TopicStatus, last_mentioned_at: i64) -> Topic {
    Topic {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        name: format!("Topic {id}"),
        status,
        mentions: 4,
        first_mentioned_at: Some(last_mentioned_at - 60_000),
        last_mentioned_at: Some(last_mentioned_at),
        planned: status == TopicStatus::Pending,
        summary: Some(secret("Discussed consistency")),
    }
}

pub fn claim(meeting_id: &str, id: &str, created_at: i64) -> Claim {
    Claim {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        text: secret("Uses Redis for counters"),
        kind: ClaimKind::Technology,
        segment_id: Some("s3".into()),
        speaker_role: SpeakerRole::Remote,
        source: ExtractionSource::Llm,
        created_at,
    }
}

pub fn observation(meeting_id: Option<&str>, id: &str, timestamp: i64) -> ScreenObservation {
    ScreenObservation {
        id: id.to_string(),
        meeting_id: meeting_id.map(str::to_string),
        timestamp,
        source: "Window: VS Code".into(),
        application: Some("Code".into()),
        title: Some(secret("limiter.rs")),
        description: secret("Rust code implementing a token bucket"),
        visible_text: Some(secret("fn take(&mut self) -> bool")),
        entities: vec!["Rust".into(), "Redis".into()],
        content_type: Some("code".into()),
        frame_hash: Some("a1b2c3d4e5f60718".into()),
    }
}

pub fn action_item(meeting_id: &str, id: &str, created_at: i64) -> ActionItem {
    ActionItem {
        id: id.to_string(),
        meeting_id: meeting_id.to_string(),
        text: secret("Send the take-home assignment"),
        owner: Some("You".into()),
        source_segment_id: Some("s4".into()),
        created_at,
    }
}

pub fn summary(meeting_id: &str, generated_at: i64) -> MeetingSummary {
    MeetingSummary {
        meeting_id: meeting_id.to_string(),
        overview: secret("Rate limiter design discussion"),
        key_topics: vec!["System design".into(), "Redis".into()],
        questions_asked: vec![secret("How would you design a rate limiter?")],
        answers: vec![secret("Token bucket in Redis")],
        decisions: vec![],
        action_items: vec![secret("Send the take-home assignment")],
        open_questions: vec![secret("Multi-region behaviour")],
        technical_topics: vec!["Redis".into(), "Token bucket".into()],
        follow_up_suggestions: vec![secret("Ask about consistency guarantees")],
        interviewer: Some(InterviewerSummary {
            question_coverage: vec!["Design".into()],
            clarification_areas: vec![secret("Failure handling")],
            claims_to_verify: vec![secret("Scaled to 10k QPS")],
            unexplored_topics: vec!["Observability".into()],
        }),
        markdown: secret("# Summary"),
        generated_at,
        generated_by: "anthropic/claude-opus-5-5".into(),
    }
}

pub fn embedding(
    meeting_id: &str,
    item_id: &str,
    model: &str,
    vector: Vec<f32>,
    created_at: i64,
) -> EmbeddingRecord {
    EmbeddingRecord {
        item_id: item_id.to_string(),
        meeting_id: meeting_id.to_string(),
        kind: "segment".into(),
        model: model.to_string(),
        vector,
        created_at,
    }
}

pub fn privacy_rule(
    id: &str,
    rule_type: PrivacyRuleType,
    value: &str,
    created_at: i64,
) -> PrivacyRule {
    PrivacyRule {
        id: id.to_string(),
        rule_type,
        value: value.to_string(),
        action: PrivacyAction::Exclude,
        created_at,
    }
}

/// Saves one meeting with at least one row in every meeting-owned table and
/// returns the record `load_meeting` must produce.
pub fn populate(db: &Database, meeting_id: &str, started_at: i64) -> MeetingRecord {
    let p = |suffix: &str| format!("{meeting_id}-{suffix}");
    let m = meeting(meeting_id, started_at);
    db.save_meeting(&m).unwrap();

    let segments = vec![
        segment(
            meeting_id,
            &p("s1"),
            1_000,
            "How would you design a rate limiter?",
        ),
        segment(meeting_id, &p("s2"), 5_000, "I'd probably use Redis."),
    ];
    for s in &segments {
        db.save_segment(s).unwrap();
    }
    let questions = vec![question(meeting_id, &p("q1"), started_at + 3_000)];
    db.save_question(&questions[0]).unwrap();
    let answers = vec![answer(meeting_id, &p("a1"), started_at + 8_000)];
    db.save_answer(&answers[0]).unwrap();
    let suggestions = vec![suggestion(Some(meeting_id), &p("r1"), started_at + 9_000)];
    db.save_suggestion(&suggestions[0]).unwrap();
    let topics = vec![
        topic(
            meeting_id,
            "system-design",
            TopicStatus::Covered,
            started_at + 4_000,
        ),
        topic(
            meeting_id,
            "observability",
            TopicStatus::Pending,
            started_at + 4_500,
        ),
    ];
    db.save_topics(meeting_id, &topics).unwrap();
    let claims = vec![claim(meeting_id, &p("c1"), started_at + 8_500)];
    db.save_claims(&claims).unwrap();
    let observations = vec![observation(Some(meeting_id), &p("o1"), started_at + 6_000)];
    db.save_observation(&observations[0]).unwrap();
    let action_items = vec![action_item(meeting_id, &p("ai1"), started_at + 9_500)];
    db.save_action_items(&action_items).unwrap();
    let summary = summary(meeting_id, started_at + 10_000);
    db.save_summary(&summary).unwrap();
    db.save_embedding(&embedding(
        meeting_id,
        &p("s1"),
        "hash-384",
        vec![0.25, -0.5, 1.0],
        started_at + 2_000,
    ))
    .unwrap();

    MeetingRecord {
        meeting: m,
        segments,
        questions,
        answers,
        suggestions,
        topics,
        claims,
        observations,
        action_items,
        summary: Some(summary),
    }
}

/// Rows per meeting-owned table for `meeting_id`, via a raw connection.
pub fn child_row_counts(conn: &rusqlite::Connection, meeting_id: &str) -> Vec<(&'static str, i64)> {
    const TABLES: [&str; 11] = [
        "participants",
        "transcript_segments",
        "questions",
        "answers",
        "ai_suggestions",
        "topics",
        "claims",
        "screen_observations",
        "action_items",
        "meeting_summaries",
        "embeddings",
    ];
    TABLES
        .iter()
        .map(|table| {
            let count: i64 = conn
                .query_row(
                    &format!("SELECT COUNT(*) FROM {table} WHERE meeting_id = ?1"),
                    [meeting_id],
                    |r| r.get(0),
                )
                .unwrap();
            (*table, count)
        })
        .collect()
}
