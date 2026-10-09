//! History operations: listing, deletion, retention, embeddings and crash
//! recovery.

use openmeet_core::StoreError;
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::MeetingStore;
use openmeet_core::types::*;
use openmeet_store::Database;

use crate::fixtures::*;

#[test]
fn list_meetings_is_newest_first_with_counts() {
    let db = Database::open_in_memory().unwrap();
    populate(&db, "full", NOW);
    let mut empty = meeting("empty", NOW + HOUR);
    empty.state = MeetingState::Completed;
    empty.ended_at = Some(NOW + 2 * HOUR);
    empty.mode = Mode::Candidate;
    db.save_meeting(&empty).unwrap();
    db.save_meeting(&meeting("oldest", NOW - DAY)).unwrap();
    for i in 0..3 {
        db.save_segment(&segment("oldest", &format!("o-s{i}"), i * 1_000, "x"))
            .unwrap();
    }

    let items = db.list_meetings(10, 0).unwrap();
    assert_eq!(
        items,
        vec![
            MeetingListItem {
                id: "empty".into(),
                title: empty.title.clone(),
                mode: Mode::Candidate,
                state: MeetingState::Completed,
                started_at: NOW + HOUR,
                ended_at: Some(NOW + 2 * HOUR),
                segment_count: 0,
                question_count: 0,
                has_summary: false,
            },
            MeetingListItem {
                id: "full".into(),
                title: meeting("full", NOW).title,
                mode: Mode::Interviewer,
                state: MeetingState::Active,
                started_at: NOW,
                ended_at: None,
                segment_count: 2,
                question_count: 1,
                has_summary: true,
            },
            MeetingListItem {
                id: "oldest".into(),
                title: meeting("oldest", NOW).title,
                mode: Mode::Interviewer,
                state: MeetingState::Active,
                started_at: NOW - DAY,
                ended_at: None,
                segment_count: 3,
                question_count: 0,
                has_summary: false,
            },
        ]
    );

    let page = |limit, offset| {
        db.list_meetings(limit, offset)
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(page(2, 0), vec!["empty", "full"]);
    assert_eq!(page(2, 2), vec!["oldest"]);
    assert!(page(2, 3).is_empty());
    assert!(page(0, 0).is_empty());
}

#[test]
fn list_meetings_breaks_start_time_ties_deterministically() {
    let db = Database::open_in_memory().unwrap();
    for id in ["b", "a", "c"] {
        db.save_meeting(&meeting(id, NOW)).unwrap();
    }
    let ids: Vec<String> = db
        .list_meetings(10, 0)
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(ids, vec!["c", "b", "a"]);
}

#[test]
fn delete_meeting_cascades_to_every_child_table() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    populate(&db, "doomed", NOW);
    let survivor = populate(&db, "survivor", NOW + HOUR);
    let raw = raw_connection(dir.path());
    assert!(
        child_row_counts(&raw, "doomed").iter().all(|(_, n)| *n > 0),
        "fixture must populate every table: {:?}",
        child_row_counts(&raw, "doomed")
    );

    db.delete_meeting("doomed").unwrap();

    assert_eq!(
        db.load_meeting("doomed"),
        Err(StoreError::NotFound("doomed".into()))
    );
    for (table, count) in child_row_counts(&raw, "doomed") {
        assert_eq!(count, 0, "{table} still has rows of the deleted meeting");
    }
    assert!(db.load_embeddings("doomed").unwrap().is_empty());
    assert_eq!(db.load_meeting("survivor").unwrap(), survivor);
    assert_eq!(db.load_embeddings("survivor").unwrap().len(), 1);
    assert_eq!(
        db.delete_meeting("doomed"),
        Err(StoreError::NotFound("doomed".into()))
    );
}

#[test]
fn deleted_content_does_not_remain_in_database_files() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    db.save_meeting(&meeting("m1", NOW)).unwrap();
    let marker = "UNIQUE-DELETED-PHRASE-93177";
    let mut s = segment("m1", "s1", 0, "");
    s.text = format!("{marker} {}", "padding ".repeat(64));
    db.save_segment(&s).unwrap();
    db.delete_meeting("m1").unwrap();

    // secure_delete zeroes the freed content and the WAL is truncated, so
    // the phrase is in neither the main file nor the -wal file.
    for name in ["openmeet.db", "openmeet.db-wal"] {
        let path = dir.path().join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            let found = bytes.windows(marker.len()).any(|w| w == marker.as_bytes());
            assert!(!found, "deleted text still present in {name}");
        }
    }
}

#[test]
fn delete_all_meeting_data_keeps_settings_and_privacy_rules() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    populate(&db, "m1", NOW);
    populate(&db, "m2", NOW + HOUR);
    let mut settings = Settings::default();
    settings.profile.background = secret("Ten years of backend work");
    db.save_settings(&settings).unwrap();
    let rule = privacy_rule("r1", PrivacyRuleType::Application, "1Password", NOW);
    db.add_privacy_rule(&rule).unwrap();

    db.delete_all_meeting_data().unwrap();

    assert!(db.list_meetings(100, 0).unwrap().is_empty());
    let raw = raw_connection(dir.path());
    for id in ["m1", "m2"] {
        for (table, count) in child_row_counts(&raw, id) {
            assert_eq!(count, 0, "{table}");
        }
    }
    assert_eq!(db.load_settings().unwrap(), Some(settings));
    assert_eq!(db.list_privacy_rules().unwrap(), vec![rule]);
    db.health_check().unwrap();

    // The database keeps working afterwards.
    let again = populate(&db, "m3", NOW);
    assert_eq!(db.load_meeting("m3").unwrap(), again);
    db.delete_all_meeting_data().unwrap();
    db.delete_all_meeting_data().unwrap();
}

/// Meetings whose start times sit on both sides of every retention cutoff.
fn retention_fixture() -> (Database, Vec<(&'static str, i64)>) {
    let db = Database::open_in_memory().unwrap();
    let meetings = vec![
        ("31d", NOW - 31 * DAY),
        ("30d-1ms", NOW - 30 * DAY - 1),
        ("30d", NOW - 30 * DAY),
        ("8d", NOW - 8 * DAY),
        ("7d-1ms", NOW - 7 * DAY - 1),
        ("7d", NOW - 7 * DAY),
        ("2d", NOW - 2 * DAY),
        ("1d-1ms", NOW - DAY - 1),
        ("1d", NOW - DAY),
        ("1h", NOW - HOUR),
        ("now", NOW),
        ("future", NOW + HOUR),
    ];
    for (id, started_at) in &meetings {
        populate(&db, id, *started_at);
    }
    (db, meetings)
}

fn remaining(db: &Database) -> Vec<String> {
    let mut ids: Vec<String> = db
        .list_meetings(1_000, 0)
        .unwrap()
        .into_iter()
        .map(|m| m.id)
        .collect();
    ids.sort();
    ids
}

fn sorted(ids: &[&str]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
    ids.sort();
    ids
}

#[test]
fn retention_forever_keeps_everything() {
    let (db, meetings) = retention_fixture();
    assert_eq!(
        db.apply_retention(RetentionPolicy::Forever, NOW).unwrap(),
        0
    );
    assert_eq!(remaining(&db).len(), meetings.len());
}

#[test]
fn retention_thirty_days_cutoff() {
    let (db, _) = retention_fixture();
    assert_eq!(
        db.apply_retention(RetentionPolicy::ThirtyDays, NOW)
            .unwrap(),
        2
    );
    assert_eq!(
        remaining(&db),
        sorted(&[
            "30d", "8d", "7d-1ms", "7d", "2d", "1d-1ms", "1d", "1h", "now", "future"
        ])
    );
}

#[test]
fn retention_seven_days_cutoff() {
    let (db, _) = retention_fixture();
    assert_eq!(
        db.apply_retention(RetentionPolicy::SevenDays, NOW).unwrap(),
        5
    );
    assert_eq!(
        remaining(&db),
        sorted(&["7d", "2d", "1d-1ms", "1d", "1h", "now", "future"])
    );
}

#[test]
fn retention_one_day_cutoff() {
    let (db, _) = retention_fixture();
    assert_eq!(db.apply_retention(RetentionPolicy::OneDay, NOW).unwrap(), 8);
    assert_eq!(remaining(&db), sorted(&["1d", "1h", "now", "future"]));
    // Idempotent.
    assert_eq!(db.apply_retention(RetentionPolicy::OneDay, NOW).unwrap(), 0);
}

#[test]
fn retention_none_deletes_everything_but_settings_and_rules() {
    let (db, meetings) = retention_fixture();
    db.save_settings(&Settings::default()).unwrap();
    let rule = privacy_rule("r1", PrivacyRuleType::Domain, "bank.example.com", NOW);
    db.add_privacy_rule(&rule).unwrap();
    assert_eq!(
        db.apply_retention(RetentionPolicy::None, NOW).unwrap() as usize,
        meetings.len()
    );
    assert!(remaining(&db).is_empty());
    for (id, _) in &meetings {
        assert!(db.load_embeddings(id).unwrap().is_empty());
    }
    assert_eq!(db.load_settings().unwrap(), Some(Settings::default()));
    assert_eq!(db.list_privacy_rules().unwrap(), vec![rule]);
    assert_eq!(db.apply_retention(RetentionPolicy::None, NOW).unwrap(), 0);
}

#[test]
fn retention_handles_extreme_clocks() {
    let (db, meetings) = retention_fixture();
    assert_eq!(
        db.apply_retention(RetentionPolicy::SevenDays, i64::MIN)
            .unwrap(),
        0
    );
    assert_eq!(
        db.apply_retention(RetentionPolicy::SevenDays, i64::MAX)
            .unwrap() as usize,
        meetings.len()
    );
}

#[test]
fn embeddings_round_trip_bit_exactly_in_creation_order() {
    let db = Database::open_in_memory().unwrap();
    db.save_meeting(&meeting("m1", NOW)).unwrap();
    db.save_meeting(&meeting("m2", NOW)).unwrap();
    let special = vec![
        0.0,
        -0.0,
        1.0e-38,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let wide: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.001).sin()).collect();
    let records = vec![
        embedding("m1", "s2", "hash-384", wide, NOW + 20),
        embedding("m1", "s1", "hash-384", special, NOW + 10),
        embedding("m1", "s3", "hash-384", vec![], NOW + 30),
    ];
    for r in &records {
        db.save_embedding(r).unwrap();
    }
    db.save_embedding(&embedding("m2", "x", "hash-384", vec![9.0], NOW))
        .unwrap();

    let loaded = db.load_embeddings("m1").unwrap();
    let expected = [&records[1], &records[0], &records[2]];
    assert_eq!(loaded.len(), 3);
    for (got, want) in loaded.iter().zip(expected) {
        assert_eq!(got.item_id, want.item_id);
        assert_eq!(got.meeting_id, want.meeting_id);
        assert_eq!(got.kind, want.kind);
        assert_eq!(got.model, want.model);
        assert_eq!(got.created_at, want.created_at);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.vector), bits(&want.vector));
    }
    assert!(db.load_embeddings("unknown").unwrap().is_empty());
}

#[test]
fn embeddings_upsert_per_item_kind_and_model() {
    let db = Database::open_in_memory().unwrap();
    db.save_meeting(&meeting("m1", NOW)).unwrap();
    db.save_embedding(&embedding("m1", "s1", "hash-384", vec![1.0, 2.0], NOW))
        .unwrap();
    // Same item + kind + model: replaced.
    db.save_embedding(&embedding("m1", "s1", "hash-384", vec![3.0], NOW + 5))
        .unwrap();
    // Another model or kind: kept separately.
    db.save_embedding(&embedding(
        "m1",
        "s1",
        "text-embedding-3-small",
        vec![4.0],
        NOW + 6,
    ))
    .unwrap();
    let mut as_question = embedding("m1", "s1", "hash-384", vec![5.0], NOW + 7);
    as_question.kind = "question".into();
    db.save_embedding(&as_question).unwrap();

    let loaded = db.load_embeddings("m1").unwrap();
    let summary: Vec<(String, String, Vec<f32>, i64)> = loaded
        .into_iter()
        .map(|e| (e.kind, e.model, e.vector, e.created_at))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("segment".into(), "hash-384".into(), vec![3.0], NOW + 5),
            (
                "segment".into(),
                "text-embedding-3-small".into(),
                vec![4.0],
                NOW + 6
            ),
            ("question".into(), "hash-384".into(), vec![5.0], NOW + 7),
        ]
    );
}

#[test]
fn crash_recovery_completes_meetings_left_in_progress() {
    let db = Database::open_in_memory().unwrap();
    let states = [
        ("idle", MeetingState::Idle),
        ("starting", MeetingState::Starting),
        ("active", MeetingState::Active),
        ("paused", MeetingState::Paused),
        ("stopping", MeetingState::Stopping),
        ("summarizing", MeetingState::Summarizing),
        ("completed", MeetingState::Completed),
    ];
    for (id, state) in states {
        let mut m = meeting(id, NOW - HOUR);
        m.state = state;
        db.save_meeting(&m).unwrap();
    }
    // active: last activity is the end of its last segment (relative time).
    db.save_segment(&segment("active", "a-s1", 60_000, "x"))
        .unwrap(); // ends at +62.5 s
    db.save_question(&question("active", "a-q1", NOW - HOUR + 30_000))
        .unwrap();
    // paused: last activity is an observation captured after the segments.
    db.save_segment(&segment("paused", "p-s1", 1_000, "x"))
        .unwrap();
    db.save_observation(&observation(Some("paused"), "p-o1", NOW - HOUR + 600_000))
        .unwrap();
    // stopping: the engine already recorded when the meeting ended.
    let mut stopping = meeting("stopping", NOW - HOUR);
    stopping.state = MeetingState::Stopping;
    stopping.ended_at = Some(NOW - HOUR + 42);
    db.save_meeting(&stopping).unwrap();
    // summarizing: activity stamped in the future (clock skew) is clamped to now.
    db.save_claims(&[claim("summarizing", "sz-c1", NOW + DAY)])
        .unwrap();
    // completed: untouched even though it has later activity.
    let mut completed = meeting("completed", NOW - HOUR);
    completed.state = MeetingState::Completed;
    completed.ended_at = Some(NOW - HOUR + 1);
    db.save_meeting(&completed).unwrap();
    db.save_answer(&answer("completed", "c-a1", NOW - 5))
        .unwrap();

    assert_eq!(db.recover_interrupted_meetings(NOW).unwrap(), 5);

    let get = |id: &str| db.load_meeting(id).unwrap().meeting;
    let expect = |id: &str, ended_at: i64| {
        let m = get(id);
        assert_eq!(m.state, MeetingState::Completed, "{id}");
        assert_eq!(m.ended_at, Some(ended_at), "{id}");
    };
    expect("starting", NOW - HOUR); // no activity: ends when it started
    expect("active", NOW - HOUR + 62_500);
    expect("paused", NOW - HOUR + 600_000);
    expect("stopping", NOW - HOUR + 42);
    expect("summarizing", NOW);
    assert_eq!(get("idle").state, MeetingState::Idle);
    assert_eq!(get("idle").ended_at, None);
    assert_eq!(get("completed"), completed);

    // Everything recorded is kept, and a second pass has nothing to do.
    assert_eq!(db.load_meeting("active").unwrap().segments.len(), 1);
    assert_eq!(db.recover_interrupted_meetings(NOW).unwrap(), 0);
}

#[test]
fn crash_recovery_never_ends_a_meeting_before_it_started() {
    let db = Database::open_in_memory().unwrap();
    // The wall clock is behind the meeting start (clock changed after a crash).
    db.save_meeting(&meeting("m1", NOW)).unwrap();
    db.save_segment(&segment("m1", "s1", 5_000, "x")).unwrap();
    assert_eq!(db.recover_interrupted_meetings(NOW - DAY).unwrap(), 1);
    assert_eq!(db.load_meeting("m1").unwrap().meeting.ended_at, Some(NOW));
}
