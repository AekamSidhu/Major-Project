//! Persisted application settings.
//!
//! The [`Settings`] document is stored as JSON in the `settings` table.
//! Loading is tolerant, so an older or partially incompatible document never
//! prevents the app from starting:
//!
//! * missing fields take their defaults (every settings struct has serde
//!   defaults);
//! * a field whose stored value no longer fits its type (e.g. written by a
//!   different version) is reset to its default on its own, keeping every
//!   other stored field;
//! * a document that is not valid JSON, or not a JSON object, is ignored
//!   (`None`).
//!
//! Problems are reported with `tracing::warn!` using sizes, positions and
//! counts only; the settings contain the user's profile and are never
//! logged.

use openmeet_core::StoreError;
use openmeet_core::now_ms;
use openmeet_core::settings::Settings;
use rusqlite::{OptionalExtension, params};
use serde::Deserialize;
use serde_json::{Map, Value};
use tracing::warn;

use crate::database::Database;
use crate::error::SqlResultExt;

/// Row of the `settings` table holding the application settings document.
const APP_SETTINGS: &str = "app";

const SELECT_SETTINGS: &str = "SELECT CAST(value AS BLOB) FROM settings WHERE name = ?1";

const UPSERT_SETTINGS: &str = "
    INSERT INTO settings (name, value, updated_at) VALUES (?1, ?2, ?3)
    ON CONFLICT (name) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at";

impl Database {
    /// The stored settings, or `None` when none were saved yet or the stored
    /// document is unusable (see the module documentation for how old or
    /// partially incompatible documents are handled).
    pub fn load_settings(&self) -> Result<Option<Settings>, StoreError> {
        let raw: Option<Vec<u8>> = self
            .lock()
            .prepare_cached(SELECT_SETTINGS)
            .db()?
            .query_row([APP_SETTINGS], |row| row.get(0))
            .optional()
            .db()?;
        Ok(raw.and_then(|raw| decode_settings(&raw)))
    }

    /// Stores `settings`, replacing the previous document. The value is
    /// stored as given; callers sanitize/validate before saving.
    pub fn save_settings(&self, settings: &Settings) -> Result<(), StoreError> {
        let json = serde_json::to_string(settings).map_err(|e| {
            StoreError::Serialization(format!("cannot encode settings ({:?})", e.classify()))
        })?;
        self.lock()
            .prepare_cached(UPSERT_SETTINGS)
            .db()?
            .execute(params![APP_SETTINGS, json, now_ms()])
            .db()?;
        Ok(())
    }
}

/// Decodes a stored settings document (see the module documentation).
pub(crate) fn decode_settings(raw: &[u8]) -> Option<Settings> {
    let value: Value = match serde_json::from_slice(raw) {
        Ok(value) => value,
        Err(e) => {
            warn!(
                bytes = raw.len(),
                category = ?e.classify(),
                line = e.line(),
                column = e.column(),
                "stored settings are not valid JSON; ignoring them"
            );
            return None;
        }
    };
    // Checked before deserializing: serde would also accept a JSON array as
    // a sequence of struct fields.
    let Value::Object(stored) = &value else {
        warn!(
            bytes = raw.len(),
            "stored settings are not a JSON object; ignoring them"
        );
        return None;
    };
    if let Ok(settings) = Settings::deserialize(&value) {
        return Some(settings);
    }
    let (settings, reset) = salvage_settings(stored);
    warn!(
        reset_fields = reset,
        "some stored settings did not match the current format and were reset to defaults"
    );
    Some(settings)
}

/// Builds settings from the default document, taking every stored field
/// (recursively, per nested object) that keeps the whole document valid.
/// Returns the settings and the number of stored fields that were dropped.
fn salvage_settings(stored: &Map<String, Value>) -> (Settings, usize) {
    let mut document = match serde_json::to_value(Settings::default()) {
        Ok(document @ Value::Object(_)) => document,
        _ => return (Settings::default(), stored.len()),
    };
    let mut path = Vec::new();
    let mut reset = 0;
    merge_compatible(&mut document, &mut path, stored, &mut reset);
    match Settings::deserialize(&document) {
        Ok(settings) => (settings, reset),
        Err(_) => (Settings::default(), stored.len()),
    }
}

/// Merges `incoming` into the object at `path` of `document` one field at a
/// time. Invariant: `document` deserializes as [`Settings`] before and after
/// every step, so a failed check is caused by the field just merged.
fn merge_compatible(
    document: &mut Value,
    path: &mut Vec<String>,
    incoming: &Map<String, Value>,
    reset: &mut usize,
) {
    for (key, value) in incoming {
        path.push(key.clone());
        let previous = replace_at(document, path, Some(value.clone()));
        if Settings::deserialize(&*document).is_err() {
            replace_at(document, path, previous.clone());
            match (value, previous) {
                // Keep whatever part of a nested object is still valid.
                (Value::Object(nested), Some(Value::Object(_))) => {
                    merge_compatible(document, path, nested, reset);
                }
                _ => *reset += 1,
            }
        }
        path.pop();
    }
}

/// Sets (`Some`) or removes (`None`) the member at `path`, whose parents are
/// objects; returns the previous value.
fn replace_at(document: &mut Value, path: &[String], value: Option<Value>) -> Option<Value> {
    let (last, parents) = path.split_last()?;
    let mut node = document;
    for key in parents {
        node = node.as_object_mut()?.get_mut(key)?;
    }
    let object = node.as_object_mut()?;
    match value {
        Some(value) => object.insert(last.clone(), value),
        None => object.remove(last),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::settings::{LlmProviderKind, RetentionPolicy, ShortcutAction};
    use openmeet_core::types::Mode;

    #[test]
    fn valid_documents_decode_unchanged() {
        let mut settings = Settings {
            mode: Mode::Interviewer,
            ..Settings::default()
        };
        settings.privacy.retention = RetentionPolicy::ThirtyDays;
        settings.profile.background = "Ten years of Rust".into();
        let raw = serde_json::to_vec(&settings).unwrap();
        assert_eq!(decode_settings(&raw), Some(settings));
    }

    #[test]
    fn old_documents_get_defaults_for_missing_fields() {
        let decoded =
            decode_settings(br#"{"mode":"interviewer","overlay":{"opacity":0.5}}"#).unwrap();
        assert_eq!(decoded.mode, Mode::Interviewer);
        assert_eq!(decoded.overlay.opacity, 0.5);
        assert_eq!(decoded.overlay.width, Settings::default().overlay.width);
        assert_eq!(decoded.ai, Settings::default().ai);
        assert_eq!(decode_settings(b"{}"), Some(Settings::default()));
    }

    #[test]
    fn invalid_json_is_ignored() {
        assert_eq!(decode_settings(b""), None);
        assert_eq!(decode_settings(b"{\"mode\": "), None);
        assert_eq!(decode_settings(b"not json at all"), None);
        assert_eq!(decode_settings(&[0xff, 0xfe, 0x00]), None);
        assert_eq!(decode_settings(b"[1, 2, 3]"), None);
        assert_eq!(decode_settings(b"\"text\""), None);
        assert_eq!(decode_settings(b"null"), None);
    }

    #[test]
    fn incompatible_fields_are_reset_individually() {
        let raw = br#"{
            "mode": "interviewer",
            "overlay": { "opacity": "very opaque", "width": 500 },
            "ai": { "llm": { "provider": "some_future_provider", "model": "my-model" } },
            "general": "not an object",
            "profile": { "background": "kept" },
            "shortcuts": { "bindings": { "ask_ai": "Ctrl+K", "not_an_action": "Ctrl+J" } },
            "unknownFutureSection": { "x": 1 }
        }"#;
        let decoded = decode_settings(raw).unwrap();
        let defaults = Settings::default();
        assert_eq!(decoded.mode, Mode::Interviewer);
        assert_eq!(decoded.overlay.opacity, defaults.overlay.opacity);
        assert_eq!(decoded.overlay.width, 500);
        assert_eq!(decoded.ai.llm.provider, LlmProviderKind::Ollama);
        assert_eq!(decoded.ai.llm.model, "my-model");
        assert_eq!(decoded.general, defaults.general);
        assert_eq!(decoded.profile.background, "kept");
        assert_eq!(decoded.shortcuts.accelerator(ShortcutAction::AskAi), "Ctrl+K");
        assert_eq!(
            decoded.shortcuts.accelerator(ShortcutAction::ToggleOverlay),
            ShortcutAction::ToggleOverlay.default_accelerator()
        );
    }

    #[test]
    fn salvage_counts_reset_fields() {
        let stored = match serde_json::from_str::<Value>(
            r#"{"mode":"nope","overlay":{"width":"wide","height":3},"version":2}"#,
        )
        .unwrap()
        {
            Value::Object(map) => map,
            _ => unreachable!(),
        };
        let (settings, reset) = salvage_settings(&stored);
        assert_eq!(reset, 2, "mode and overlay.width are invalid");
        assert_eq!(settings.mode, Mode::Candidate);
        assert_eq!(settings.version, 2);
    }
}
