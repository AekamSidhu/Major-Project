//! Opening file databases: creation, migrations, re-open idempotence and
//! refusal of files this build cannot safely use.

use std::path::Path;

use openmeet_core::StoreError;
use openmeet_core::settings::Settings;
use openmeet_core::store::MeetingStore;
use openmeet_core::types::{Mode, PrivacyRuleType};
use openmeet_store::{Database, SCHEMA_VERSION};

use crate::fixtures::*;

const TABLES: [&str; 14] = [
    "action_items",
    "ai_suggestions",
    "answers",
    "claims",
    "embeddings",
    "meeting_summaries",
    "meetings",
    "participants",
    "privacy_rules",
    "questions",
    "screen_observations",
    "settings",
    "topics",
    "transcript_segments",
];

fn user_version(dir: &Path) -> u32 {
    raw_connection(dir)
        .pragma_query_value(None, "user_version", |r| r.get(0))
        .unwrap()
}

/// Every schema object (type, name, SQL), to prove re-opening changes nothing.
fn schema(dir: &Path) -> Vec<(String, String, Option<String>)> {
    let conn = raw_connection(dir);
    let mut stmt = conn
        .prepare("SELECT type, name, sql FROM sqlite_schema ORDER BY type, name")
        .unwrap();
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn table_names(dir: &Path) -> Vec<String> {
    schema(dir)
        .into_iter()
        .filter(|(ty, name, _)| ty == "table" && !name.starts_with("sqlite_"))
        .map(|(_, name, _)| name)
        .collect()
}

#[test]
fn fresh_file_is_created_with_parent_directories_and_migrated() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("app data").join("OpenMeet");
    let path = nested.join("openmeet.db");
    let db = Database::open(&path).unwrap();
    assert!(path.is_file());

    assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
    db.health_check().unwrap();
    assert!(db.list_meetings(10, 0).unwrap().is_empty());
    assert!(db.list_privacy_rules().unwrap().is_empty());
    assert_eq!(db.load_settings().unwrap(), None);

    assert_eq!(user_version(&nested), SCHEMA_VERSION);
    assert_eq!(table_names(&nested), TABLES);
    let journal_mode: String = raw_connection(&nested)
        .pragma_query_value(None, "journal_mode", |r| r.get(0))
        .unwrap();
    assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
}

#[test]
fn in_memory_databases_are_private_and_fully_migrated() {
    let a = Database::open_in_memory().unwrap();
    let b = Database::open_in_memory().unwrap();
    assert_eq!(a.schema_version().unwrap(), SCHEMA_VERSION);
    a.health_check().unwrap();
    populate(&a, "m1", NOW);
    assert_eq!(a.list_meetings(10, 0).unwrap().len(), 1);
    assert!(b.list_meetings(10, 0).unwrap().is_empty());
    assert_eq!(b.load_meeting("m1"), Err(StoreError::NotFound("m1".into())));
}

#[test]
fn an_existing_empty_file_is_initialized() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("openmeet.db"), b"").unwrap();
    let db = open_file_db(dir.path());
    assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
    assert_eq!(table_names(dir.path()), TABLES);
}

#[test]
fn reopening_is_idempotent_and_keeps_all_data() {
    let dir = tempfile::tempdir().unwrap();
    let mut settings = Settings {
        mode: Mode::Interviewer,
        ..Settings::default()
    };
    settings.profile.background = secret("Ten years of distributed systems");
    let rule = privacy_rule("r1", PrivacyRuleType::Application, "KeePassXC", NOW);
    let expected = {
        let db = open_file_db(dir.path());
        let expected = populate(&db, "m1", NOW);
        db.save_settings(&settings).unwrap();
        db.add_privacy_rule(&rule).unwrap();
        expected
    };
    let schema_before = schema(dir.path());

    for _ in 0..3 {
        let db = open_file_db(dir.path());
        assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
        db.health_check().unwrap();
        assert_eq!(db.load_meeting("m1").unwrap(), expected);
        assert_eq!(db.load_embeddings("m1").unwrap().len(), 1);
        assert_eq!(db.load_settings().unwrap(), Some(settings.clone()));
        assert_eq!(db.list_privacy_rules().unwrap(), vec![rule.clone()]);
        assert_eq!(db.list_meetings(10, 0).unwrap().len(), 1);
    }
    assert_eq!(schema(dir.path()), schema_before);
    assert_eq!(user_version(dir.path()), SCHEMA_VERSION);
}

#[test]
fn several_handles_on_one_file_share_data() {
    let dir = tempfile::tempdir().unwrap();
    let first = open_file_db(dir.path());
    let second = open_file_db(dir.path());
    let expected = populate(&first, "m1", NOW);
    assert_eq!(second.load_meeting("m1").unwrap(), expected);
    second.delete_meeting("m1").unwrap();
    assert!(first.list_meetings(10, 0).unwrap().is_empty());
}

#[test]
fn a_database_from_a_newer_openmeet_is_refused_untouched() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open_file_db(dir.path());
        populate(&db, "m1", NOW);
    }
    raw_connection(dir.path())
        .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
        .unwrap();
    let before = schema(dir.path());

    let err = Database::open(&dir.path().join("openmeet.db")).unwrap_err();
    assert!(matches!(err, StoreError::Migration(_)), "{err:?}");
    assert!(err.to_string().contains("update OpenMeet"), "{err}");

    assert_eq!(user_version(dir.path()), SCHEMA_VERSION + 1);
    assert_eq!(schema(dir.path()), before);
    let meetings: i64 = raw_connection(dir.path())
        .query_row("SELECT COUNT(*) FROM meetings", [], |r| r.get(0))
        .unwrap();
    assert_eq!(meetings, 1);
}

#[test]
fn a_file_that_is_not_a_database_is_an_error_and_left_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("openmeet.db");
    let garbage = secret("this is not a SQLite database").repeat(32);
    std::fs::write(&path, &garbage).unwrap();

    let err = Database::open(&path).unwrap_err();
    assert!(matches!(err, StoreError::Database(_)), "{err:?}");
    assert!(!err.to_string().contains(SECRET), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), garbage.as_bytes());
}

#[test]
fn database_is_usable_through_the_engine_contract() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<Database>();

    let db = std::sync::Arc::new(Database::open_in_memory().unwrap());
    let store: std::sync::Arc<dyn MeetingStore> = db.clone();
    store.save_meeting(&meeting("m1", NOW)).unwrap();
    store
        .save_segment(&segment("m1", "s1", 0, "hello"))
        .unwrap();
    assert_eq!(db.list_meetings(1, 0).unwrap()[0].segment_count, 1);
}
