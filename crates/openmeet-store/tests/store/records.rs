//! `MeetingStore` behaviour: every upsert path, `load_meeting` completeness
//! and ordering, and errors for unknown meetings.

use openmeet_core::StoreError;
use openmeet_core::store::MeetingStore;
use openmeet_core::types::*;
use openmeet_store::Database;

use crate::fixtures::*;

fn db_with_meeting(id: &str) -> Database {
    let db = Database::open_in_memory().unwrap();
    db.save_meeting(&meeting(id, NOW)).unwrap();
    db
}

#[test]
fn load_meeting_returns_everything_that_was_saved() {
    let db = Database::open_in_memory().unwrap();
    let expected = populate(&db, "m1", NOW);
    let other = populate(&db, "m2", NOW + HOUR);
    assert_eq!(db.load_meeting("m1").unwrap(), expected);
    assert_eq!(db.load_meeting("m2").unwrap(), other);
}

#[test]
fn load_unknown_meeting_is_not_found() {
    let db = Database::open_in_memory().unwrap();
    assert_eq!(
        db.load_meeting("missing"),
        Err(StoreError::NotFound("missing".into()))
    );
}

#[test]
fn save_meeting_inserts_then_updates_including_participants() {
    let db = Database::open_in_memory().unwrap();
    let mut m = meeting("m1", NOW);
    db.save_meeting(&m).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().meeting, m);

    m.title = "Renamed".into();
    m.mode = Mode::Candidate;
    m.state = MeetingState::Completed;
    m.ended_at = Some(NOW + HOUR);
    m.persisted = false;
    m.participants = vec![
        Participant {
            id: "z-last".into(),
            role: SpeakerRole::Unknown,
            label: "Speaker".into(),
        },
        Participant {
            id: "a-first".into(),
            role: SpeakerRole::Remote,
            label: "Interviewer".into(),
        },
    ];
    db.save_meeting(&m).unwrap();
    let loaded = db.load_meeting("m1").unwrap().meeting;
    assert_eq!(loaded, m, "participant order is preserved");
    assert_eq!(db.list_meetings(10, 0).unwrap().len(), 1);

    m.participants.clear();
    db.save_meeting(&m).unwrap();
    assert!(
        db.load_meeting("m1")
            .unwrap()
            .meeting
            .participants
            .is_empty()
    );
}

#[test]
fn resaving_a_meeting_keeps_everything_recorded_for_it() {
    // Guards against `INSERT OR REPLACE`, which would delete the meeting row
    // and cascade-delete all of its children.
    let db = Database::open_in_memory().unwrap();
    let mut expected = populate(&db, "m1", NOW);
    expected.meeting.state = MeetingState::Stopping;
    expected.meeting.ended_at = Some(NOW + HOUR);
    db.save_meeting(&expected.meeting).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap(), expected);
}

#[test]
fn segments_upsert_by_id() {
    let db = db_with_meeting("m1");
    let mut s = segment("m1", "s1", 1_000, "How would you");
    s.is_final = false;
    s.confidence = None;
    db.save_segment(&s).unwrap();
    s.text = secret("How would you design a rate limiter?");
    s.is_final = true;
    s.confidence = Some(0.5);
    s.end_time = 4_000;
    s.speaker_id = None;
    s.speaker_role = SpeakerRole::Unknown;
    s.source = AudioSourceKind::Microphone;
    db.save_segment(&s).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().segments, vec![s]);
}

#[test]
fn questions_upsert_by_id() {
    let db = db_with_meeting("m1");
    let mut q = question("m1", "q1", NOW + 1);
    db.save_question(&q).unwrap();
    q.category = QuestionCategory::FollowUp;
    q.method = DetectionMethod::Semantic;
    q.parts = vec![];
    q.segment_ids = vec!["s9".into()];
    q.supersedes = Some("q0".into());
    q.requires_response = false;
    q.is_question = false;
    q.speaker_id = None;
    q.speaker_role = SpeakerRole::User;
    q.confidence = 0.25;
    db.save_question(&q).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().questions, vec![q]);
}

#[test]
fn nan_scores_do_not_break_writes() {
    let db = db_with_meeting("m1");
    let mut q = question("m1", "q1", NOW);
    q.confidence = f32::NAN;
    db.save_question(&q).unwrap();
    let mut s = segment("m1", "s1", 0, "hello");
    s.confidence = Some(f32::NAN);
    db.save_segment(&s).unwrap();
    let record = db.load_meeting("m1").unwrap();
    assert_eq!(record.questions[0].confidence, 0.0);
    assert_eq!(record.segments[0].confidence, None);
}

#[test]
fn answers_upsert_by_id() {
    let db = db_with_meeting("m1");
    let mut a = answer("m1", "a1", NOW);
    db.save_answer(&a).unwrap();
    a.question_id = None;
    a.text = secret("Actually, a sliding window log");
    a.segment_ids = vec!["s4".into(), "s5".into()];
    a.created_at = NOW + 5;
    db.save_answer(&a).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().answers, vec![a]);
}

#[test]
fn suggestions_upsert_and_pin() {
    let db = db_with_meeting("m1");
    let mut s = suggestion(Some("m1"), "r1", NOW);
    db.save_suggestion(&s).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().suggestions, vec![s.clone()]);

    s.content = secret("[[ANSWER]]\nA refined answer");
    s.kind = SuggestionKind::Refinement;
    s.category = None;
    s.style = None;
    s.followups.clear();
    s.completed_at = None;
    s.prompt_version = 4;
    db.save_suggestion(&s).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().suggestions, vec![s.clone()]);

    db.set_suggestion_pinned("r1", true).unwrap();
    s.pinned = true;
    assert_eq!(db.load_meeting("m1").unwrap().suggestions, vec![s.clone()]);
    db.set_suggestion_pinned("r1", false).unwrap();
    assert!(!db.load_meeting("m1").unwrap().suggestions[0].pinned);

    assert_eq!(
        db.set_suggestion_pinned("unknown", true),
        Err(StoreError::NotFound("unknown".into()))
    );
}

#[test]
fn suggestions_and_observations_without_a_meeting_are_not_persisted() {
    let db = db_with_meeting("m1");
    db.save_suggestion(&suggestion(None, "r1", NOW)).unwrap();
    db.save_observation(&observation(None, "o1", NOW)).unwrap();
    let record = db.load_meeting("m1").unwrap();
    assert!(record.suggestions.is_empty());
    assert!(record.observations.is_empty());
    assert_eq!(
        db.set_suggestion_pinned("r1", true),
        Err(StoreError::NotFound("r1".into()))
    );
}

#[test]
fn topics_are_replaced_as_an_ordered_list() {
    let db = db_with_meeting("m1");
    let a = topic("m1", "architecture", TopicStatus::Covered, NOW + 10);
    let b = topic("m1", "security", TopicStatus::Current, NOW + 20);
    let c = topic("m1", "testing", TopicStatus::Pending, NOW + 30);
    db.save_topics("m1", &[a.clone(), b.clone(), c.clone()])
        .unwrap();
    assert_eq!(
        db.load_meeting("m1").unwrap().topics,
        vec![a.clone(), b, c.clone()]
    );

    let mut c2 = c.clone();
    c2.status = TopicStatus::Current;
    c2.mentions = 9;
    c2.summary = None;
    c2.first_mentioned_at = None;
    c2.last_mentioned_at = None;
    db.save_topics("m1", &[c2.clone(), a.clone()]).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().topics, vec![c2, a]);

    db.save_topics("m1", &[]).unwrap();
    assert!(db.load_meeting("m1").unwrap().topics.is_empty());
}

#[test]
fn topics_are_attributed_to_the_meeting_they_are_saved_for() {
    let db = db_with_meeting("m1");
    let foreign = topic("other-meeting", "architecture", TopicStatus::Current, NOW);
    db.save_topics("m1", std::slice::from_ref(&foreign))
        .unwrap();
    let stored = db.load_meeting("m1").unwrap().topics;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].meeting_id, "m1");
    assert_eq!(stored[0].name, foreign.name);
}

#[test]
fn claims_upsert_atomically() {
    let db = db_with_meeting("m1");
    let mut c1 = claim("m1", "c1", NOW);
    let c2 = claim("m1", "c2", NOW + 1);
    db.save_claims(&[c1.clone(), c2.clone()]).unwrap();
    c1.kind = ClaimKind::Metric;
    c1.text = secret("Handles 10k QPS");
    c1.source = ExtractionSource::Heuristic;
    c1.segment_id = None;
    db.save_claims(std::slice::from_ref(&c1)).unwrap();
    assert_eq!(
        db.load_meeting("m1").unwrap().claims,
        vec![c1.clone(), c2.clone()]
    );

    // One claim for an unknown meeting: nothing from the batch is written.
    let c3 = claim("m1", "c3", NOW + 2);
    let orphan = claim("ghost", "c4", NOW + 3);
    assert_eq!(
        db.save_claims(&[c3, orphan]),
        Err(StoreError::NotFound("ghost".into()))
    );
    assert_eq!(db.load_meeting("m1").unwrap().claims, vec![c1, c2]);
    db.save_claims(&[]).unwrap();
}

#[test]
fn observations_upsert_by_id() {
    let db = db_with_meeting("m1");
    let mut o = observation(Some("m1"), "o1", NOW);
    db.save_observation(&o).unwrap();
    o.description = secret("A sequence diagram");
    o.content_type = Some("diagram".into());
    o.entities.clear();
    o.visible_text = None;
    o.application = None;
    o.title = None;
    o.frame_hash = None;
    o.timestamp = NOW + 7;
    db.save_observation(&o).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().observations, vec![o]);
}

#[test]
fn action_items_upsert_atomically() {
    let db = db_with_meeting("m1");
    let mut item = action_item("m1", "ai1", NOW);
    db.save_action_items(std::slice::from_ref(&item)).unwrap();
    item.owner = None;
    item.source_segment_id = None;
    item.text = secret("Schedule the second round");
    db.save_action_items(std::slice::from_ref(&item)).unwrap();
    assert_eq!(
        db.save_action_items(&[
            action_item("m1", "ai2", NOW),
            action_item("ghost", "ai3", NOW)
        ]),
        Err(StoreError::NotFound("ghost".into()))
    );
    assert_eq!(db.load_meeting("m1").unwrap().action_items, vec![item]);
    db.save_action_items(&[]).unwrap();
}

#[test]
fn summary_is_replaced_on_save() {
    let db = db_with_meeting("m1");
    assert_eq!(db.load_meeting("m1").unwrap().summary, None);
    let mut s = summary("m1", NOW);
    db.save_summary(&s).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().summary, Some(s.clone()));
    s.interviewer = None;
    s.overview = secret("Shorter overview");
    s.generated_by = "extractive".into();
    s.decisions = vec![secret("Use Redis")];
    db.save_summary(&s).unwrap();
    assert_eq!(db.load_meeting("m1").unwrap().summary, Some(s));
}

#[test]
fn writes_for_an_unknown_meeting_are_not_found() {
    let db = db_with_meeting("m1");
    let not_found = Err(StoreError::NotFound("ghost".into()));
    assert_eq!(db.save_segment(&segment("ghost", "s1", 0, "x")), not_found);
    assert_eq!(db.save_question(&question("ghost", "q1", NOW)), not_found);
    assert_eq!(db.save_answer(&answer("ghost", "a1", NOW)), not_found);
    assert_eq!(
        db.save_suggestion(&suggestion(Some("ghost"), "r1", NOW)),
        not_found
    );
    assert_eq!(
        db.save_topics("ghost", &[topic("ghost", "t", TopicStatus::Current, NOW)]),
        not_found
    );
    assert_eq!(db.save_topics("ghost", &[]), not_found);
    assert_eq!(db.save_claims(&[claim("ghost", "c1", NOW)]), not_found);
    assert_eq!(
        db.save_observation(&observation(Some("ghost"), "o1", NOW)),
        not_found
    );
    assert_eq!(
        db.save_action_items(&[action_item("ghost", "ai1", NOW)]),
        not_found
    );
    assert_eq!(db.save_summary(&summary("ghost", NOW)), not_found);
    assert_eq!(
        db.save_embedding(&embedding("ghost", "s1", "m", vec![1.0], NOW)),
        not_found
    );
    // Nothing leaked into the existing meeting.
    let record = db.load_meeting("m1").unwrap();
    assert!(record.segments.is_empty() && record.questions.is_empty());
    assert!(db.load_embeddings("ghost").unwrap().is_empty());
}

#[test]
fn lists_are_returned_in_chronological_order() {
    let db = db_with_meeting("m1");
    // Saved deliberately out of order.
    for (id, start) in [("s3", 9_000), ("s1", 1_000), ("s2", 5_000)] {
        db.save_segment(&segment("m1", id, start, id)).unwrap();
    }
    // Same start time: the shorter segment first, then first-written.
    let mut tie_long = segment("m1", "s0-long", 1_000, "long");
    tie_long.end_time = 8_000;
    db.save_segment(&tie_long).unwrap();
    for (id, at) in [("q2", 20), ("q1", 10), ("q3", 30)] {
        db.save_question(&question("m1", id, NOW + at)).unwrap();
    }
    for (id, at) in [("a2", 20), ("a3", 30), ("a1", 10)] {
        db.save_answer(&answer("m1", id, NOW + at)).unwrap();
    }
    for (id, at) in [("r3", 30), ("r1", 10), ("r2", 20)] {
        db.save_suggestion(&suggestion(Some("m1"), id, NOW + at))
            .unwrap();
    }
    db.save_claims(&[
        claim("m1", "c2", NOW + 20),
        claim("m1", "c3", NOW + 30),
        claim("m1", "c1", NOW + 10),
    ])
    .unwrap();
    for (id, at) in [("o2", 20), ("o1", 10), ("o3", 30)] {
        db.save_observation(&observation(Some("m1"), id, NOW + at))
            .unwrap();
    }
    db.save_action_items(&[
        action_item("m1", "ai3", NOW + 30),
        action_item("m1", "ai1", NOW + 10),
        action_item("m1", "ai2", NOW + 20),
    ])
    .unwrap();

    let record = db.load_meeting("m1").unwrap();
    let ids = |v: Vec<&str>| v.into_iter().map(String::from).collect::<Vec<_>>();
    assert_eq!(
        record
            .segments
            .iter()
            .map(|s| s.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["s1", "s0-long", "s2", "s3"])
    );
    assert_eq!(
        record
            .questions
            .iter()
            .map(|q| q.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["q1", "q2", "q3"])
    );
    assert_eq!(
        record
            .answers
            .iter()
            .map(|a| a.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["a1", "a2", "a3"])
    );
    assert_eq!(
        record
            .suggestions
            .iter()
            .map(|s| s.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["r1", "r2", "r3"])
    );
    assert_eq!(
        record
            .claims
            .iter()
            .map(|c| c.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["c1", "c2", "c3"])
    );
    assert_eq!(
        record
            .observations
            .iter()
            .map(|o| o.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["o1", "o2", "o3"])
    );
    assert_eq!(
        record
            .action_items
            .iter()
            .map(|a| a.id.clone())
            .collect::<Vec<_>>(),
        ids(vec!["ai1", "ai2", "ai3"])
    );
}

#[test]
fn text_is_stored_verbatim() {
    let db = db_with_meeting("m1");
    let tricky = [
        "'; DROP TABLE meetings; --",
        "Unicode: Größe, 日本語, emoji 🚀, RTL שלום",
        "embedded\0nul and \"quotes\" and \\backslashes\\",
        "",
        "  leading and trailing whitespace \n\t",
    ];
    for (i, text) in tricky.iter().enumerate() {
        let mut s = segment("m1", &format!("s{i}"), i as i64 * 1_000, "");
        s.text = (*text).to_string();
        db.save_segment(&s).unwrap();
    }
    let long = "word ".repeat(200_000);
    let mut s = segment("m1", "long", 99_000, "");
    s.text = long.clone();
    db.save_segment(&s).unwrap();

    let segments = db.load_meeting("m1").unwrap().segments;
    for (i, text) in tricky.iter().enumerate() {
        assert_eq!(segments[i].text, *text);
    }
    assert_eq!(segments.last().unwrap().text, long);
    assert_eq!(db.list_meetings(10, 0).unwrap().len(), 1);
}
