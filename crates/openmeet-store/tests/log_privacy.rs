//! Privacy of diagnostics: meeting content, settings content and corrupted
//! stored values never appear in log output or in returned errors.
//!
//! This is its own test binary with a single test: it installs a global
//! `tracing` subscriber, which reliably sees every event. (A thread-scoped
//! subscriber in a binary whose other tests log concurrently can miss events
//! because `tracing` caches per-callsite interest process-wide.)

use std::io;
use std::sync::{Arc, Mutex};

use openmeet_core::StoreError;
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::MeetingStore;
use openmeet_core::types::PrivacyRuleType;

#[allow(dead_code)]
#[path = "store/fixtures.rs"]
mod fixtures;

use fixtures::*;

/// Collects everything a `tracing` subscriber writes.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

impl io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn assert_clean(err: &StoreError) {
    assert!(!err.to_string().contains(SECRET), "{err}");
    assert!(!format!("{err:?}").contains(SECRET), "{err:?}");
}

#[test]
fn logs_and_errors_never_contain_meeting_content() {
    let capture = Capture::default();
    let writer = capture.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || writer.clone())
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("this binary installs the only global subscriber");

    let mut errors: Vec<StoreError> = Vec::new();
    {
        let dir = tempfile::tempdir().unwrap();
        let db = open_file_db(dir.path());

        // Normal operation over every kind of stored content.
        populate(&db, "m1", NOW);
        populate(&db, "m-old", NOW - 40 * DAY);
        db.load_meeting("m1").unwrap();
        db.load_embeddings("m1").unwrap();
        db.list_meetings(10, 0).unwrap();
        let mut settings = Settings::default();
        settings.profile.background = secret("profile background");
        settings.profile.custom_instructions = secret("instructions");
        db.save_settings(&settings).unwrap();
        db.load_settings().unwrap();
        db.add_privacy_rule(&privacy_rule(
            "rule-1",
            PrivacyRuleType::Window,
            &secret("Private window"),
            NOW,
        ))
        .unwrap();
        db.list_privacy_rules().unwrap();
        db.save_suggestion(&suggestion(None, "r-none", NOW))
            .unwrap();
        db.save_observation(&observation(None, "o-none", NOW))
            .unwrap();
        db.recover_interrupted_meetings(NOW).unwrap();
        assert_eq!(
            db.apply_retention(RetentionPolicy::ThirtyDays, NOW)
                .unwrap(),
            1
        );
        db.delete_meeting("m1").unwrap();

        // Failing writes.
        errors.push(
            db.save_segment(&segment("ghost", "s-ghost", 0, "lost"))
                .unwrap_err(),
        );
        errors.push(
            db.save_claims(&[claim("ghost", "c-ghost", NOW)])
                .unwrap_err(),
        );

        // Stored values damaged outside the store (another tool, a bug, disk
        // corruption) are reported without echoing the value.
        populate(&db, "m3", NOW);
        let raw = raw_connection(dir.path());
        raw.execute(
            "UPDATE transcript_segments SET speaker_role = ?1 WHERE meeting_id = 'm3'",
            [SECRET],
        )
        .unwrap();
        errors.push(db.load_meeting("m3").unwrap_err());
        raw.execute(
            "UPDATE transcript_segments SET speaker_role = 'remote' WHERE meeting_id = 'm3'",
            [],
        )
        .unwrap();
        raw.execute(
            "UPDATE questions SET parts = ?1 WHERE meeting_id = 'm3'",
            [format!("[\"{SECRET}\", ")],
        )
        .unwrap();
        errors.push(db.load_meeting("m3").unwrap_err());
        raw.execute(
            "UPDATE embeddings SET dims = dims + 1 WHERE meeting_id = 'm3'",
            [],
        )
        .unwrap();
        errors.push(db.load_embeddings("m3").unwrap_err());

        // Unusable and partially incompatible settings documents.
        raw.execute(
            "UPDATE settings SET value = ?1",
            [format!("{{\"profile\": {{\"background\": \"{SECRET}\"")],
        )
        .unwrap();
        assert_eq!(db.load_settings().unwrap(), None);
        raw.execute(
            "UPDATE settings SET value = ?1",
            [format!(
                "{{\"mode\": \"{SECRET}\", \"profile\": {{\"background\": \"{SECRET}\"}}}}"
            )],
        )
        .unwrap();
        let salvaged = db.load_settings().unwrap().unwrap();
        assert!(salvaged.profile.background.contains(SECRET));
        drop(raw);

        db.delete_all_meeting_data().unwrap();
    }

    assert_eq!(errors.len(), 5);
    assert!(matches!(&errors[0], StoreError::NotFound(id) if id == "ghost"));
    assert!(matches!(&errors[1], StoreError::NotFound(id) if id == "ghost"));
    for err in &errors[2..] {
        assert!(matches!(err, StoreError::Serialization(_)), "{err:?}");
    }
    for err in &errors {
        assert_clean(err);
    }

    let logs = capture.contents();
    // The capture works and the operations above were logged...
    for expected in [
        "database opened",
        "applied database migration",
        "deleted meeting",
        "retention deleted meetings",
        "recovered interrupted meetings",
        "stored settings are not valid JSON",
        "some stored settings did not match",
        "deleted all meeting data",
    ] {
        assert!(
            logs.contains(expected),
            "missing log {expected:?} in:\n{logs}"
        );
    }
    // ...without any of the content.
    assert!(
        !logs.contains(SECRET),
        "meeting content leaked into logs:\n{logs}"
    );
}
