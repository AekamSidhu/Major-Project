//! Concurrent access: one shared handle used from many threads, and several
//! handles (connections) on the same file, including concurrent opening and
//! migration of a fresh file.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use openmeet_core::settings::Settings;
use openmeet_core::store::MeetingStore;
use openmeet_core::types::TopicStatus;
use openmeet_store::{Database, SCHEMA_VERSION};

use crate::fixtures::*;

const WRITERS: usize = 6;
const SEGMENTS_PER_MEETING: usize = 30;

/// Writes one meeting with `SEGMENTS_PER_MEETING` segments plus periodic
/// questions, topics and claims (multi-statement transactions).
fn write_meeting(db: &Database, meeting_id: &str, started_at: i64) {
    db.save_meeting(&meeting(meeting_id, started_at)).unwrap();
    for i in 0..SEGMENTS_PER_MEETING {
        let start = i as i64 * 1_000;
        db.save_segment(&segment(
            meeting_id,
            &format!("{meeting_id}-s{i:03}"),
            start,
            "x",
        ))
        .unwrap();
        if i % 5 == 0 {
            db.save_question(&question(
                meeting_id,
                &format!("{meeting_id}-q{i:03}"),
                started_at + start,
            ))
            .unwrap();
            db.save_topics(
                meeting_id,
                &[
                    topic(meeting_id, "architecture", TopicStatus::Covered, started_at),
                    topic(
                        meeting_id,
                        "testing",
                        TopicStatus::Current,
                        started_at + start,
                    ),
                ],
            )
            .unwrap();
            let claims: Vec<_> = (0..5)
                .map(|c| {
                    claim(
                        meeting_id,
                        &format!("{meeting_id}-c{i:03}-{c}"),
                        started_at + start,
                    )
                })
                .collect();
            db.save_claims(&claims).unwrap();
        }
    }
}

fn assert_meeting_complete(db: &Database, meeting_id: &str) {
    let record = db.load_meeting(meeting_id).unwrap();
    assert_eq!(record.segments.len(), SEGMENTS_PER_MEETING, "{meeting_id}");
    let ids: Vec<String> = record.segments.iter().map(|s| s.id.clone()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "segments of {meeting_id} are chronological");
    assert_eq!(record.questions.len(), SEGMENTS_PER_MEETING / 5);
    assert_eq!(record.topics.len(), 2);
    assert_eq!(record.claims.len(), SEGMENTS_PER_MEETING);
}

#[test]
fn one_shared_handle_serves_concurrent_writers_and_readers() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open_file_db(dir.path()));
    let done = Arc::new(AtomicBool::new(false));

    let readers: Vec<_> = (0..2)
        .map(|_| {
            let db = Arc::clone(&db);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut passes = 0usize;
                loop {
                    // Read `done` before the pass so one full pass always runs
                    // after the writers have finished.
                    let finished = done.load(Ordering::Acquire);
                    for item in db.list_meetings(100, 0).unwrap() {
                        let record = db.load_meeting(&item.id).unwrap();
                        // Rows are only ever added, so a later read sees at
                        // least what the listing counted.
                        assert!(record.segments.len() >= item.segment_count as usize);
                        assert!(record.questions.len() >= item.question_count as usize);
                    }
                    db.health_check().unwrap();
                    passes += 1;
                    if finished {
                        return passes;
                    }
                }
            })
        })
        .collect();

    let writers: Vec<_> = (0..WRITERS)
        .map(|w| {
            let db = Arc::clone(&db);
            thread::spawn(move || write_meeting(&db, &format!("m{w}"), NOW + w as i64))
        })
        .collect();
    for writer in writers {
        writer.join().expect("writer thread panicked");
    }
    done.store(true, Ordering::Release);
    for reader in readers {
        assert!(reader.join().expect("reader thread panicked") >= 1);
    }

    let items = db.list_meetings(100, 0).unwrap();
    assert_eq!(items.len(), WRITERS);
    for item in &items {
        assert_eq!(item.segment_count as usize, SEGMENTS_PER_MEETING);
        assert_eq!(item.question_count as usize, SEGMENTS_PER_MEETING / 5);
        assert_meeting_complete(&db, &item.id);
    }
}

#[test]
fn separate_handles_open_migrate_and_write_one_file_concurrently() {
    let dir = tempfile::tempdir().unwrap();
    let path = Arc::new(dir.path().join("openmeet.db"));
    let start = Arc::new(Barrier::new(WRITERS));

    // Every thread opens (and so migrates) the fresh file at the same time,
    // then writes through its own connection while the others do too.
    let threads: Vec<_> = (0..WRITERS)
        .map(|w| {
            let path = Arc::clone(&path);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                start.wait();
                let db = Database::open(&path).expect("concurrent open");
                assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
                write_meeting(&db, &format!("m{w}"), NOW + w as i64);
                // Cross-connection reads and writes while other handles keep
                // writing.
                assert!(!db.list_meetings(100, 0).unwrap().is_empty());
                db.save_settings(&Settings::default()).unwrap();
                assert!(db.load_settings().unwrap().is_some());
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("handle thread panicked");
    }

    let db = Database::open(&path).unwrap();
    db.health_check().unwrap();
    let items = db.list_meetings(100, 0).unwrap();
    assert_eq!(items.len(), WRITERS);
    for item in &items {
        assert_meeting_complete(&db, &item.id);
    }
}
