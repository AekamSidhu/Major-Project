//! Settings persistence: round trips, replacement and tolerance of stored
//! documents written by other versions (or damaged on disk).

use std::path::Path;

use openmeet_core::settings::{
    LlmProviderKind, OverlayPosition, RetentionPolicy, Settings, ShortcutAction,
};
use openmeet_core::types::{AnswerStyle, Mode};
use openmeet_store::Database;

use crate::fixtures::*;

/// Settings with a non-default value in every section.
fn customized() -> Settings {
    let mut s = Settings {
        mode: Mode::Interviewer,
        ..Settings::default()
    };
    s.ai.llm.provider = LlmProviderKind::Anthropic;
    s.ai.llm.model = "claude-opus-5-5".into();
    s.ai.llm.base_url = None;
    s.ai.llm.fast_model = Some("claude-haiku-4-5".into());
    s.transcription.language = None;
    s.transcription.vocabulary = "Kafka, PostgreSQL".into();
    s.assistant.answer_style = AnswerStyle::Star;
    s.audio.microphone_device = Some("USB Microphone (2- Rode)".into());
    s.screen.enabled = true;
    s.privacy.local_only = true;
    s.privacy.retention = RetentionPolicy::ThirtyDays;
    s.overlay.position = OverlayPosition::Custom;
    s.overlay.opacity = 0.75;
    s.overlay.custom_positions.insert(
        "display-1|zoom".into(),
        openmeet_core::ipc::Point { x: 120, y: -40 },
    );
    s.shortcuts
        .bindings
        .insert(ShortcutAction::AskAi, "CommandOrControl+Alt+K".into());
    s.general.onboarding_complete = true;
    s.profile.background = secret("Staff engineer, Größe 日本語 🚀");
    s.profile.planned_topics = "- Architecture\n- Testing".into();
    s
}

/// Overwrites the stored settings document with `value`, as a different
/// OpenMeet version (or disk damage) could have left it.
fn overwrite_stored(dir: &Path, value: &str) {
    let changed = raw_connection(dir)
        .execute("UPDATE settings SET value = ?1", [value])
        .unwrap();
    assert_eq!(changed, 1, "a settings document must already be stored");
}

#[test]
fn settings_are_absent_until_saved() {
    let db = Database::open_in_memory().unwrap();
    assert_eq!(db.load_settings().unwrap(), None);
}

#[test]
fn settings_round_trip_exactly_and_are_replaced_on_save() {
    let db = Database::open_in_memory().unwrap();
    db.save_settings(&Settings::default()).unwrap();
    assert_eq!(db.load_settings().unwrap(), Some(Settings::default()));

    let custom = customized();
    db.save_settings(&custom).unwrap();
    assert_eq!(db.load_settings().unwrap(), Some(custom.clone()));

    let mut changed = custom;
    changed.mode = Mode::Candidate;
    changed.profile.background.clear();
    db.save_settings(&changed).unwrap();
    assert_eq!(db.load_settings().unwrap(), Some(changed));
}

#[test]
fn settings_survive_reopening_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let custom = customized();
    open_file_db(dir.path()).save_settings(&custom).unwrap();
    assert_eq!(
        open_file_db(dir.path()).load_settings().unwrap(),
        Some(custom)
    );
}

#[test]
fn unusable_documents_load_as_none_and_can_be_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    db.save_settings(&customized()).unwrap();
    let unusable = [
        "",
        "not json at all",
        "{\"mode\": \"interviewer\"",
        "[1, 2, 3]",
        "\"just a string\"",
        "42",
        "null",
    ];
    for value in unusable {
        overwrite_stored(dir.path(), value);
        assert_eq!(db.load_settings().unwrap(), None, "{value:?}");
    }

    // Bytes that are not even UTF-8.
    raw_connection(dir.path())
        .execute("UPDATE settings SET value = CAST(X'FFFE7B00' AS TEXT)", [])
        .unwrap();
    assert_eq!(db.load_settings().unwrap(), None);

    // Saving again repairs the stored document.
    db.save_settings(&Settings::default()).unwrap();
    assert_eq!(db.load_settings().unwrap(), Some(Settings::default()));
}

#[test]
fn documents_from_older_versions_get_defaults_for_missing_fields() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    db.save_settings(&Settings::default()).unwrap();
    overwrite_stored(
        dir.path(),
        r#"{"version":0,"mode":"interviewer","overlay":{"opacity":0.5},
            "privacy":{"retention":"one_day"}}"#,
    );
    let loaded = db.load_settings().unwrap().unwrap();
    let defaults = Settings::default();
    assert_eq!(loaded.version, 0);
    assert_eq!(loaded.mode, Mode::Interviewer);
    assert_eq!(loaded.overlay.opacity, 0.5);
    assert_eq!(loaded.overlay.width, defaults.overlay.width);
    assert_eq!(loaded.privacy.retention, RetentionPolicy::OneDay);
    assert_eq!(
        loaded.privacy.capture_exclusion,
        defaults.privacy.capture_exclusion
    );
    assert_eq!(loaded.ai, defaults.ai);
    assert_eq!(loaded.shortcuts, defaults.shortcuts);

    overwrite_stored(dir.path(), "{}");
    assert_eq!(db.load_settings().unwrap(), Some(defaults));
}

#[test]
fn incompatible_fields_are_reset_without_losing_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_file_db(dir.path());
    db.save_settings(&Settings::default()).unwrap();
    overwrite_stored(
        dir.path(),
        r#"{
            "mode": "interviewer",
            "privacy": { "retention": "until_the_heat_death", "localOnly": true },
            "overlay": { "width": "wide", "fontSize": 18 },
            "audio": 7,
            "profile": { "background": "kept", "roleContext": ["not", "a", "string"] },
            "someFutureSection": { "enabled": true }
        }"#,
    );
    let loaded = db.load_settings().unwrap().unwrap();
    let defaults = Settings::default();
    assert_eq!(loaded.mode, Mode::Interviewer);
    assert_eq!(loaded.privacy.retention, defaults.privacy.retention);
    assert!(loaded.privacy.local_only);
    assert_eq!(loaded.overlay.width, defaults.overlay.width);
    assert_eq!(loaded.overlay.font_size, 18);
    assert_eq!(loaded.audio, defaults.audio);
    assert_eq!(loaded.profile.background, "kept");
    assert_eq!(loaded.profile.role_context, defaults.profile.role_context);
}
