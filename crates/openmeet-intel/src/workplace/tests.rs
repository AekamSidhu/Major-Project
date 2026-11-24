#![allow(clippy::type_complexity)] // table-driven cases

use super::*;
use crate::claims::{extract_action_items, extract_claims};
use crate::detect::{DetectionInput, detect};
use openmeet_core::types::AudioSourceKind;

fn seg(id: &str, role: SpeakerRole, at_s: i64, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        id: id.into(),
        meeting_id: "m".into(),
        utterance_id: id.into(),
        speaker_id: None,
        speaker_role: role,
        source: if role == SpeakerRole::User {
            AudioSourceKind::Microphone
        } else {
            AudioSourceKind::System
        },
        start_time: at_s * 1000,
        end_time: at_s * 1000 + 3000,
        text: text.into(),
        is_final: true,
        confidence: None,
    }
}

fn item(text: &str, owner: Option<&str>, due: Option<&str>) -> ActionItem {
    ActionItem {
        id: format!("a-{}", text.len()),
        meeting_id: "m".into(),
        text: text.into(),
        owner: owner.map(str::to_string),
        due: due.map(str::to_string),
        source_segment_id: None,
        created_at: 0,
    }
}

fn decision(text: &str) -> Claim {
    Claim {
        id: format!("d-{}", text.len()),
        meeting_id: "m".into(),
        text: text.into(),
        kind: ClaimKind::Decision,
        segment_id: None,
        speaker_role: SpeakerRole::Remote,
        source: ExtractionSource::Heuristic,
        created_at: 0,
    }
}

/// Feeds a meeting through detection, extraction and the tracker the way the
/// engine does, returning the tracker and every update.
fn run(lines: &[(SpeakerRole, i64, &str)]) -> (WorkplaceTracker, Vec<WorkplaceUpdate>) {
    let mut tracker = WorkplaceTracker::new();
    let mut updates = Vec::new();
    let mut recent: Vec<TranscriptSegment> = Vec::new();
    let mut last_q: Option<DetectedQuestion> = None;
    for (i, (role, at, text)) in lines.iter().enumerate() {
        let s = seg(&format!("s{i}"), *role, *at, text);
        let r = detect(&DetectionInput {
            segment: &s,
            recent: &recent,
            mode: Mode::Workplace,
            last_question: last_q.as_ref(),
        });
        let q = (r.is_question && r.confidence >= 0.6 * 0.85).then(|| DetectedQuestion {
            id: format!("q{i}"),
            meeting_id: "m".into(),
            segment_ids: r.segment_ids.clone(),
            is_question: true,
            confidence: r.confidence,
            question: r.question.clone(),
            parts: r.parts.clone(),
            category: r.category,
            speaker_id: None,
            speaker_role: *role,
            requires_response: r.requires_response,
            method: DetectionMethod::Heuristic,
            supersedes: r.supersedes.clone(),
            detected_at: 0,
        });
        let claims = extract_claims(&s);
        let items = extract_action_items(&s, Mode::Workplace);
        updates.push(tracker.observe(&s, &claims, items, q.as_ref()));
        if q.is_some() {
            last_q = q;
        }
        recent.push(s);
    }
    (tracker, updates)
}

#[test]
fn task_phrases_drop_who_and_when() {
    for (text, task) in [
        (
            "Priya will pair with Sam on it and have a fix ready by Thursday.",
            "Pair with Sam on it and have a fix ready",
        ),
        ("So we need to update the runbook.", "Update the runbook"),
        (
            "I'll draft the announcement by end of day.",
            "Draft the announcement",
        ),
        (
            "Alex, can you file a ticket for the timeout?",
            "File a ticket for the timeout",
        ),
        (
            "Action item: Sam to benchmark the new cache in two weeks.",
            "Benchmark the new cache",
        ),
        (
            "Someone should check the alert thresholds.",
            "Check the alert thresholds",
        ),
        ("Please send the deck.", "Send the deck"),
        ("Update the roadmap", "Update the roadmap"),
    ] {
        assert_eq!(task_phrase(text), task, "{text}");
    }
}

#[test]
fn action_item_vagueness_table() {
    use VagueIssue::*;
    let cases: &[(&str, Option<&str>, Option<&str>, &[VagueIssue])] = &[
        ("Update the runbook", Some("Priya"), Some("by Friday"), &[]),
        (
            "We need to update the runbook.",
            None,
            None,
            &[NoOwner, NoDueDate],
        ),
        ("I'll update the runbook.", Some("You"), None, &[NoDueDate]),
        (
            "Someone should look into it.",
            None,
            Some("by Friday"),
            &[UnclearScope, NoOwner],
        ),
        (
            "I'll take care of that.",
            Some("You"),
            Some("tomorrow"),
            &[UnclearScope],
        ),
        (
            "Priya will fix the login timeout by Thursday.",
            Some("Priya"),
            Some("by Thursday"),
            &[],
        ),
        (
            "Let's follow up.",
            None,
            None,
            &[UnclearScope, NoOwner, NoDueDate],
        ),
    ];
    for (text, owner, due, expected) in cases {
        assert_eq!(
            action_item_issues(&item(text, *owner, *due)),
            expected.to_vec(),
            "{text}"
        );
    }
}

#[test]
fn decision_vagueness_table() {
    use VagueIssue::*;
    let cases: &[(&str, &[VagueIssue])] = &[
        ("Let's go with the token refresh fix this sprint.", &[]),
        ("Okay, let's go with that.", &[UnclearScope]),
        ("We agreed on it.", &[UnclearScope]),
        (
            "Let's go with dropping the legacy CSV export for now.",
            &[Tentative],
        ),
        ("I think we agreed to freeze the schema.", &[Tentative]),
        ("We decided to move billing to Kafka.", &[]),
    ];
    for (text, expected) in cases {
        assert_eq!(
            decision_issues(&decision(text)),
            expected.to_vec(),
            "{text}"
        );
    }
}

#[test]
fn clarifying_questions_table() {
    use VagueIssue::*;
    use WorkItemKind::*;
    let cases: &[(
        WorkItemKind,
        &str,
        Option<&str>,
        &[VagueIssue],
        Option<&str>,
    )] = &[
        (
            ActionItem,
            "Update the runbook",
            None,
            &[NoOwner, NoDueDate],
            Some("Who will take “Update the runbook”, and by when?"),
        ),
        (
            ActionItem,
            "Update the runbook",
            None,
            &[NoOwner],
            Some("Who will own “Update the runbook”?"),
        ),
        (
            ActionItem,
            "Update the runbook",
            Some("Priya"),
            &[NoDueDate],
            Some("Priya, by when can you have “Update the runbook” done?"),
        ),
        (
            ActionItem,
            "Update the runbook",
            Some("You"),
            &[NoDueDate],
            Some("By when should “Update the runbook” be done?"),
        ),
        (
            ActionItem,
            "Look into it",
            None,
            &[UnclearScope, NoOwner, NoDueDate],
            Some("What exactly does “Look into it” involve, and who will take it?"),
        ),
        (
            ActionItem,
            "Take care of that",
            Some("You"),
            &[UnclearScope],
            Some("What exactly does “Take care of that” involve — what does done look like?"),
        ),
        (ActionItem, "Update the runbook", Some("Priya"), &[], None),
        (
            Decision,
            "Okay, let's go with that.",
            None,
            &[UnclearScope],
            Some(
                "What exactly did we decide in “Okay, let's go with that”? Worth saying it in one line for the notes.",
            ),
        ),
        (
            Decision,
            "Let's go with dropping the CSV export for now.",
            None,
            &[Tentative],
            Some("Is “Let's go with dropping the CSV export for now” decided, or still open?"),
        ),
        (
            Decision,
            "We decided to move billing to Kafka.",
            None,
            &[],
            None,
        ),
    ];
    for (kind, text, owner, issues, expected) in cases {
        assert_eq!(
            clarifying_question(*kind, text, *owner, issues).as_deref(),
            *expected,
            "{text} {issues:?}"
        );
    }
}

#[test]
fn decisions_and_action_items_are_tracked_with_owners_and_dates() {
    use SpeakerRole::{Remote, User};
    let (t, _) = run(&[
        (
            Remote,
            5,
            "Let's go with the token refresh fix this sprint.",
        ),
        (
            Remote,
            12,
            "Priya will pair with Sam on it and have a fix ready by Thursday.",
        ),
        (
            Remote,
            20,
            "Next, the onboarding docs are out of date, so we need to update the runbook.",
        ),
    ]);
    let b = t.board();
    assert_eq!(b.decisions.len(), 1);
    assert!(b.decisions[0].text.contains("token refresh fix"));
    assert_eq!(b.action_items.len(), 2, "{:?}", b.action_items);
    assert_eq!(b.action_items[0].owner.as_deref(), Some("Priya"));
    assert_eq!(b.action_items[0].due.as_deref(), Some("by Thursday"));
    assert_eq!(b.action_items[1].owner, None);
    // Only the runbook item is vague (no owner, no date).
    assert_eq!(b.clarifications.len(), 1, "{:?}", b.clarifications);
    let c = &b.clarifications[0];
    assert_eq!(c.target_kind, WorkItemKind::ActionItem);
    assert_eq!(c.target_id, b.action_items[1].id);
    assert_eq!(
        c.question,
        "Who will take “Update the runbook”, and by when?"
    );
    let _ = User;
}

#[test]
fn owners_and_dates_fill_in_from_later_replies() {
    use SpeakerRole::{Remote, User};
    // An explicit restatement merges into the earlier item.
    let (t, updates) = run(&[
        (Remote, 20, "So we need to update the runbook."),
        (
            User,
            30,
            "I can take the runbook update, I'll have it done by Friday.",
        ),
    ]);
    let b = t.board();
    assert_eq!(b.action_items.len(), 1, "{:?}", b.action_items);
    assert_eq!(b.action_items[0].owner.as_deref(), Some("You"));
    assert_eq!(b.action_items[0].due.as_deref(), Some("by Friday"));
    assert!(b.clarifications.is_empty(), "{:?}", b.clarifications);
    // The update carries the changed item (persisted as an upsert).
    assert_eq!(updates[1].items.len(), 1);
    assert_eq!(updates[1].items[0].id, b.action_items[0].id);

    // "I'll take it" and a bare date refer to the latest open item.
    let (t, _) = run(&[
        (Remote, 20, "Someone should check the alert thresholds."),
        (Remote, 25, "I'll take it."),
        (Remote, 28, "Let's say by Wednesday."),
    ]);
    let a = &t.board().action_items[0];
    assert_eq!(a.owner.as_deref(), Some("Other participant"));
    assert_eq!(a.due.as_deref(), Some("by Wednesday"));

    // A named colleague taking it.
    let (t, _) = run(&[
        (Remote, 20, "We should review the vendor contract."),
        (Remote, 22, "Dana can take that by next week."),
    ]);
    let a = &t.board().action_items[0];
    assert_eq!(a.owner.as_deref(), Some("Dana"));
    assert_eq!(a.due.as_deref(), Some("by next week"));

    // Too late to refer back: a new item instead of an assignment.
    let (t, _) = run(&[
        (Remote, 20, "We should review the vendor contract."),
        (User, 20 + 600, "I'll take it."),
    ]);
    assert_eq!(t.board().action_items[0].owner, None);
}

#[test]
fn questions_are_answered_deferred_or_left_open() {
    use SpeakerRole::{Remote, User};
    let (t, updates) = run(&[
        (
            Remote,
            40,
            "Can you give us a quick update on the billing export?",
        ),
        (
            User,
            48,
            "Sure. The batch rewrite is done and staging tests passed yesterday.",
        ),
        (Remote, 60, "Do we have a date for the production cutover?"),
        (User, 66, "Not yet, I'll check with the data team."),
        (
            Remote,
            80,
            "Should we also move the reporting service to the new cluster this sprint?",
        ),
        (Remote, 84, "Let's park that until next planning."),
        (Remote, 100, "Is the staging environment back up?"),
        (Remote, 103, "Yes, staging came back up this morning."),
    ]);
    let qs = t.board().open_questions;
    assert_eq!(qs.len(), 4, "{qs:?}");
    // Answered by the user's reply; it was asked of the user.
    assert!(qs[0].addressed_to_you);
    assert!(qs[0].answered);
    assert!(
        qs[0]
            .answer
            .as_deref()
            .unwrap()
            .starts_with("Sure. The batch rewrite")
    );
    assert_eq!(updates[1].answered, vec![qs[0].id.clone()]);
    // "Not yet" defers it: still open.
    assert!(!qs[1].answered && qs[1].deferred);
    assert!(!qs[1].addressed_to_you);
    // Parked by a colleague.
    assert!(!qs[2].answered && qs[2].deferred);
    // Answered by another colleague (related reply that sounds like an answer).
    assert!(qs[3].answered, "{:?}", qs[3]);
    // The deferral also produced an action item for the user.
    let items = t.board().action_items;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].owner.as_deref(), Some("You"));
}

#[test]
fn unrelated_chatter_does_not_answer_a_question() {
    use SpeakerRole::{Remote, User};
    // (lines, answered?)
    let cases: &[(&[(SpeakerRole, i64, &str)], bool)] = &[
        // The asker's own follow-up and an unrelated next line do not answer it.
        (
            &[
                (User, 10, "When is the design review?"),
                (User, 12, "I want to make sure the mocks are ready."),
                (Remote, 40, "Moving on to the budget for next quarter."),
            ],
            false,
        ),
        // A short reply that sounds like an answer does.
        (
            &[
                (User, 10, "When is the design review?"),
                (Remote, 14, "Friday afternoon, I think."),
            ],
            true,
        ),
        // Between colleagues the reply must refer to the question.
        (
            &[
                (Remote, 10, "Who owns the flaky deploy job?"),
                (Remote, 13, "Also the coffee machine is broken again."),
            ],
            false,
        ),
        (
            &[
                (Remote, 10, "Who owns the flaky deploy job?"),
                (Remote, 13, "I think Dana owns the deploy job now."),
            ],
            true,
        ),
        // Too late for an unrelated reply.
        (
            &[
                (Remote, 10, "Can you send the slides?"),
                (User, 10 + 400, "Yes, sounds good."),
            ],
            false,
        ),
    ];
    for (lines, answered) in cases {
        let (t, _) = run(lines);
        let qs = t.board().open_questions;
        assert_eq!(qs.len(), 1, "{lines:?}");
        assert_eq!(qs[0].answered, *answered, "{lines:?}");
    }
}

fn record() -> MeetingRecord {
    let meeting = Meeting {
        id: "m".into(),
        title: "Sprint planning".into(),
        mode: Mode::Workplace,
        state: MeetingState::Completed,
        started_at: 1_790_000_000_000,
        ended_at: Some(1_790_000_000_000 + 30 * 60_000),
        participants: vec![],
        persisted: true,
    };
    let q = |id: &str, text: &str| DetectedQuestion {
        id: id.into(),
        meeting_id: "m".into(),
        segment_ids: vec![],
        is_question: true,
        confidence: 0.9,
        question: text.into(),
        parts: vec![],
        category: QuestionCategory::General,
        speaker_id: None,
        speaker_role: SpeakerRole::Remote,
        requires_response: false,
        method: DetectionMethod::Heuristic,
        supersedes: None,
        detected_at: 0,
    };
    MeetingRecord {
        meeting,
        segments: vec![],
        questions: vec![
            q("q1", "Can you give us an update on the export?"),
            q("q2", "Do we have a date for the cutover?"),
        ],
        answers: vec![AnswerRecord {
            id: "a".into(),
            meeting_id: "m".into(),
            question_id: Some("q1".into()),
            speaker_role: SpeakerRole::User,
            text: "The rewrite is done.".into(),
            segment_ids: vec![],
            created_at: 0,
        }],
        suggestions: vec![],
        topics: vec![],
        claims: vec![
            decision("Let's go with the token refresh fix this sprint."),
            Claim {
                kind: ClaimKind::Metric,
                ..decision("Latency dropped by 40% last week.")
            },
        ],
        observations: vec![],
        action_items: vec![
            item(
                "Priya will pair with Sam on it and have a fix ready by Thursday.",
                Some("Priya"),
                Some("by Thursday"),
            ),
            item("So we need to update the runbook.", None, None),
        ],
        summary: None,
        integrity_flags: vec![],
    }
}

#[test]
fn extractive_minutes_have_decisions_owners_dates_and_open_questions() {
    let s = extractive_minutes(&record());
    assert_eq!(s.generated_by, "extractive");
    assert_eq!(
        s.decisions,
        vec!["Let's go with the token refresh fix this sprint."]
    );
    assert_eq!(s.open_questions, vec!["Do we have a date for the cutover?"]);
    assert_eq!(s.questions_asked.len(), 2);
    assert!(s.interviewer.is_none());
    let w = s.workplace.as_ref().expect("minutes");
    assert_eq!(w.action_items.len(), 2);
    assert_eq!(
        w.action_items[0].task,
        "Pair with Sam on it and have a fix ready"
    );
    assert_eq!(w.action_items[0].owner.as_deref(), Some("Priya"));
    assert_eq!(w.action_items[0].due.as_deref(), Some("by Thursday"));
    assert_eq!(
        s.action_items,
        vec![
            "Pair with Sam on it and have a fix ready — Priya, by Thursday",
            "Update the runbook — owner to be confirmed"
        ]
    );
    assert_eq!(
        s.follow_up_suggestions,
        vec!["Who will take “Update the runbook”, and by when?"]
    );
    assert!(
        s.overview
            .contains("1 decision, 2 action items and 1 open question")
    );
    assert!(s.overview.contains("(30 min)"));
    // Follow-up message draft.
    let m = &w.follow_up_message;
    assert!(m.starts_with("Hi all,"));
    assert!(m.contains("Thanks for joining Sprint planning."));
    assert!(m.contains("- Pair with Sam on it and have a fix ready — Priya, by Thursday"));
    assert!(m.contains("Open questions:\n- Do we have a date for the cutover?"));
    // Markdown minutes.
    let md = &s.markdown;
    assert!(md.starts_with("# Sprint planning\n"));
    for heading in [
        "## Summary",
        "## Decisions",
        "## Action items",
        "## Open questions",
        "## To clarify",
        "## Follow-up message",
    ] {
        assert!(md.contains(heading), "{heading}\n{md}");
    }
    assert!(md.contains("```text\nHi all,"));
    for word in ["hire", "candidate", "interview", "score"] {
        assert!(!md.to_lowercase().contains(word), "{word}");
    }
}

#[test]
fn minutes_without_anything_recorded_say_so() {
    let mut r = record();
    r.claims.clear();
    r.action_items.clear();
    r.questions.clear();
    let s = extractive_minutes(&r);
    let w = s.workplace.unwrap();
    assert!(
        w.follow_up_message
            .contains("No decisions or action items were recorded.")
    );
    assert!(
        s.overview
            .contains("0 decisions, 0 action items and 0 open questions")
    );
}

#[test]
fn minutes_from_the_board_keep_deferred_and_answered_questions() {
    let mut r = record();
    r.claims.clear();
    r.action_items.clear();
    r.questions.clear();
    r.answers.clear();
    let question = |id: &str, text: &str, answered: bool, deferred: bool| OpenQuestion {
        id: id.into(),
        question: text.into(),
        speaker_role: SpeakerRole::Remote,
        addressed_to_you: true,
        asked_at_ms: 1_000,
        answered,
        answer: answered.then(|| "Dana".to_string()),
        deferred,
    };
    let board = WorkplaceBoard {
        decisions: vec![decision("We agreed to freeze the schema.")],
        action_items: vec![item("Update the runbook.", None, None)],
        open_questions: vec![
            question("q1", "Can you send the deck?", false, true),
            question("q2", "Who owns the cutover?", true, false),
            question("q3", "Do we have a date for the cutover?", false, false),
        ],
        clarifications: vec![],
    };
    let s = extractive_minutes_from_board(&r, &board);
    assert_eq!(s.generated_by, "extractive");
    assert_eq!(s.decisions, vec!["We agreed to freeze the schema."]);
    assert_eq!(
        s.questions_asked,
        vec![
            "Can you send the deck?",
            "Who owns the cutover?",
            "Do we have a date for the cutover?"
        ]
    );
    assert_eq!(
        s.open_questions,
        vec![
            "Can you send the deck? (deferred)",
            "Do we have a date for the cutover?"
        ]
    );
    assert_eq!(
        s.action_items,
        vec!["Update the runbook — owner to be confirmed"]
    );
    assert!(
        s.overview
            .contains("1 decision, 1 action item and 2 open questions")
    );
    let md = &s.markdown;
    assert!(md.contains("- Can you send the deck? (deferred)"), "{md}");
    assert!(!md.contains("Who owns the cutover? (deferred)"), "{md}");
    // Without the board the recorded meeting is used as before.
    let from_record = extractive_minutes(&record());
    assert_eq!(
        from_record.open_questions,
        vec!["Do we have a date for the cutover?"]
    );
}

#[test]
fn catch_up_without_a_model_lists_what_happened_recently() {
    let lines = vec![
        TranscriptLine {
            speaker: "Others".into(),
            text: "Earlier stuff".into(),
            at_ms: 10_000,
        },
        TranscriptLine {
            speaker: "Others".into(),
            text: "Let's go with the token refresh fix.".into(),
            at_ms: 400_000,
        },
        TranscriptLine {
            speaker: "You".into(),
            text: "Sounds good.".into(),
            at_ms: 410_000,
        },
    ];
    let mut d = decision("Let's go with the token refresh fix.");
    d.created_at = 2_000;
    let mut old = decision("We agreed to freeze the schema.");
    old.created_at = 500;
    let board = WorkplaceBoard {
        decisions: vec![old, d],
        action_items: vec![],
        open_questions: vec![OpenQuestion {
            id: "q".into(),
            question: "Can you send the deck?".into(),
            speaker_role: SpeakerRole::Remote,
            addressed_to_you: true,
            asked_at_ms: 20_000,
            answered: false,
            answer: None,
            deferred: false,
        }],
        clarifications: vec![],
    };
    let text = extractive_catch_up(&lines, &board, 5, 300_000, 1_000);
    let s = crate::prompts::candidate_answer::parse(&text);
    assert_eq!(
        s.answer,
        "In the last 5 minutes: 2 lines of discussion (Others, You)."
    );
    assert_eq!(
        s.key_points,
        vec![
            "Decided: Let's go with the token refresh fix.",
            "Asked you: Can you send the deck?"
        ]
    );
    assert!(s.caveat.unwrap().contains("without an AI model"));
    let empty = extractive_catch_up(&[], &WorkplaceBoard::default(), 1, 0, 0);
    assert!(empty.contains("Nothing was said in the last minute."));
}
