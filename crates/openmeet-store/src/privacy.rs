//! Screen-capture privacy rules (excluded applications, windows, domains and
//! masked regions). They are user configuration, not meeting data, so they
//! survive `delete_all_meeting_data` and retention.

use openmeet_core::StoreError;
use openmeet_core::now_ms;
use openmeet_core::types::PrivacyRule;
use rusqlite::{Row, params};

use crate::codec::SqlEnum;
use crate::database::Database;
use crate::error::SqlResultExt;

const SELECT_RULES: &str = "
    SELECT id, rule_type, value, rule_action, created_at
    FROM privacy_rules ORDER BY created_at, id";

const UPSERT_RULE: &str = "
    INSERT INTO privacy_rules (id, rule_type, value, rule_action, created_at, updated_at)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6)
    ON CONFLICT (id) DO UPDATE SET
        rule_type = excluded.rule_type,
        value = excluded.value,
        rule_action = excluded.rule_action,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at";

const DELETE_RULE: &str = "DELETE FROM privacy_rules WHERE id = ?1";

impl Database {
    /// All privacy rules, oldest first.
    pub fn list_privacy_rules(&self) -> Result<Vec<PrivacyRule>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(SELECT_RULES).db()?;
        let rows = stmt.query_map([], rule_from_row).db()?;
        let rules = rows.collect::<rusqlite::Result<Vec<_>>>().db()?;
        Ok(rules)
    }

    /// Adds a rule, or replaces the stored rule with the same id (so editing
    /// a rule is saving it again).
    pub fn add_privacy_rule(&self, rule: &PrivacyRule) -> Result<(), StoreError> {
        self.lock()
            .prepare_cached(UPSERT_RULE)
            .db()?
            .execute(params![
                rule.id,
                SqlEnum(rule.rule_type),
                rule.value,
                SqlEnum(rule.action),
                rule.created_at,
                now_ms(),
            ])
            .db()?;
        Ok(())
    }

    /// Deletes a rule; [`StoreError::NotFound`] when no rule has this id.
    pub fn delete_privacy_rule(&self, id: &str) -> Result<(), StoreError> {
        let deleted = self
            .lock()
            .prepare_cached(DELETE_RULE)
            .db()?
            .execute([id])
            .db()?;
        if deleted == 0 {
            return Err(StoreError::NotFound(id.to_string()));
        }
        Ok(())
    }
}

fn rule_from_row(row: &Row<'_>) -> rusqlite::Result<PrivacyRule> {
    Ok(PrivacyRule {
        id: row.get("id")?,
        rule_type: row.get::<_, SqlEnum<_>>("rule_type")?.0,
        value: row.get("value")?,
        action: row.get::<_, SqlEnum<_>>("rule_action")?.0,
        created_at: row.get("created_at")?,
    })
}
