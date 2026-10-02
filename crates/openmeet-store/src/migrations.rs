//! Versioned schema migrations tracked with `PRAGMA user_version`.
//!
//! Each migration is embedded SQL applied in its own `IMMEDIATE` transaction
//! together with the version bump, so a failed migration leaves the database
//! at the previous version with no partial changes. The version is re-read
//! after the write lock is taken, so several processes opening the same file
//! at once never apply a migration twice.
//!
//! Adding a migration: add `migrations/NNNN_<name>.sql`, append it to
//! [`MIGRATIONS`] and bump [`SCHEMA_VERSION`]. Never edit a released
//! migration. Migrations run with `foreign_keys = ON`; a future migration
//! that rebuilds a table (SQLite's 12-step ALTER procedure) must extend this
//! runner to disable foreign key enforcement around it.

use std::time::Instant;

use openmeet_core::StoreError;
use rusqlite::{Connection, TransactionBehavior};
use tracing::info;

use crate::database::elapsed_ms;
use crate::error::SqlResultExt;

/// Latest schema version understood by this build of the store. Databases
/// with a higher `user_version` are refused by [`crate::Database::open`].
pub const SCHEMA_VERSION: u32 = 1;

/// One schema migration: `sql` upgrades a database from `version - 1` to
/// `version`.
pub(crate) struct Migration {
    pub version: u32,
    pub name: &'static str,
    pub sql: &'static str,
}

/// All migrations in ascending, contiguous version order starting at 1.
pub(crate) const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "initial schema",
    sql: include_str!("../migrations/0001_initial.sql"),
}];

const _: () = assert!(
    MIGRATIONS[MIGRATIONS.len() - 1].version == SCHEMA_VERSION,
    "SCHEMA_VERSION must equal the version of the last migration"
);

/// Current `PRAGMA user_version` of the main database.
pub(crate) fn user_version(conn: &Connection) -> Result<u32, StoreError> {
    let version: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .db()?;
    u32::try_from(version)
        .map_err(|_| StoreError::Migration(format!("invalid schema version {version}")))
}

/// Brings the database up to [`SCHEMA_VERSION`]; returns the final version.
pub(crate) fn migrate(conn: &mut Connection) -> Result<u32, StoreError> {
    run_migrations(conn, MIGRATIONS)
}

/// Applies every migration newer than the database's version, one
/// transaction per migration.
pub(crate) fn run_migrations(
    conn: &mut Connection,
    migrations: &[Migration],
) -> Result<u32, StoreError> {
    let latest = migrations.last().map_or(0, |m| m.version);

    // Fast path: an up-to-date database needs no write lock.
    let current = user_version(conn)?;
    ensure_supported(current, latest)?;
    if current == latest {
        return Ok(current);
    }

    loop {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| {
                StoreError::Migration(format!("cannot start a migration transaction: {e}"))
            })?;
        // Re-read under the write lock: another connection may have migrated
        // the file in the meantime.
        let current = user_version(&tx)?;
        ensure_supported(current, latest)?;
        let Some(next) = migrations.iter().find(|m| m.version > current) else {
            return Ok(current);
        };

        let started = Instant::now();
        tx.execute_batch(next.sql).map_err(|e| {
            StoreError::Migration(format!(
                "migration {} ({}) failed: {e}",
                next.version, next.name
            ))
        })?;
        tx.pragma_update(None, "user_version", next.version)
            .map_err(|e| {
                StoreError::Migration(format!(
                    "cannot record schema version {}: {e}",
                    next.version
                ))
            })?;
        tx.commit().map_err(|e| {
            StoreError::Migration(format!(
                "cannot commit migration {}: {e}",
                next.version
            ))
        })?;
        info!(
            version = next.version,
            migration = next.name,
            elapsed_ms = elapsed_ms(started),
            "applied database migration"
        );
    }
}

fn ensure_supported(current: u32, latest: u32) -> Result<(), StoreError> {
    if current > latest {
        return Err(StoreError::Migration(format!(
            "the database uses schema version {current}, but this version of OpenMeet \
             supports up to version {latest}; update OpenMeet to open it"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
        conn
    }

    fn table_names(conn: &Connection) -> Vec<String> {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_schema WHERE type = 'table' \
                 AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<String>>>()
            .unwrap()
    }

    #[test]
    fn migration_list_is_contiguous_and_ends_at_schema_version() {
        for (index, migration) in MIGRATIONS.iter().enumerate() {
            assert_eq!(migration.version as usize, index + 1, "{}", migration.name);
            assert!(!migration.sql.trim().is_empty());
        }
        assert_eq!(MIGRATIONS.last().unwrap().version, SCHEMA_VERSION);
    }

    #[test]
    fn migrates_an_empty_database() {
        let mut conn = memory();
        assert_eq!(user_version(&conn).unwrap(), 0);
        assert_eq!(migrate(&mut conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(user_version(&conn).unwrap(), SCHEMA_VERSION);
        assert_eq!(
            table_names(&conn),
            vec![
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
            ]
        );
    }

    #[test]
    fn migrating_twice_is_a_no_op() {
        let mut conn = memory();
        migrate(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO settings (name, value, updated_at) VALUES ('k', '{}', 1)",
            [],
        )
        .unwrap();
        assert_eq!(migrate(&mut conn).unwrap(), SCHEMA_VERSION);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM settings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn failing_migration_rolls_back_completely() {
        let migrations = [
            Migration {
                version: 1,
                name: "first",
                sql: "CREATE TABLE first (x INTEGER);",
            },
            Migration {
                version: 2,
                name: "broken",
                sql: "CREATE TABLE second (x INTEGER); THIS IS NOT SQL;",
            },
        ];
        let mut conn = memory();
        let err = run_migrations(&mut conn, &migrations).unwrap_err();
        assert!(
            matches!(&err, StoreError::Migration(m) if m.contains("migration 2 (broken)")),
            "{err:?}"
        );
        assert_eq!(user_version(&conn).unwrap(), 1);
        assert_eq!(table_names(&conn), vec!["first"]);
    }

    #[test]
    fn newer_schema_is_refused_and_left_untouched() {
        let mut conn = memory();
        conn.pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .unwrap();
        let err = migrate(&mut conn).unwrap_err();
        assert!(matches!(err, StoreError::Migration(_)), "{err:?}");
        assert_eq!(user_version(&conn).unwrap(), SCHEMA_VERSION + 1);
        assert!(table_names(&conn).is_empty());
    }

    #[test]
    fn meeting_owned_tables_cascade_and_are_indexed_by_meeting() {
        let mut conn = memory();
        migrate(&mut conn).unwrap();
        let owned: Vec<String> = table_names(&conn)
            .into_iter()
            .filter(|t| !matches!(t.as_str(), "meetings" | "privacy_rules" | "settings"))
            .collect();
        assert_eq!(owned.len(), 11);
        for table in &owned {
            let fks: Vec<(String, String, String)> = conn
                .prepare("SELECT \"table\", \"from\", on_delete FROM pragma_foreign_key_list(?1)")
                .unwrap()
                .query_map([table], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert_eq!(
                fks,
                vec![(
                    "meetings".to_string(),
                    "meeting_id".to_string(),
                    "CASCADE".to_string()
                )],
                "{table}"
            );

            // Some index (or the primary key) must start with meeting_id so
            // cascades and per-meeting reads do not scan the whole table.
            let leading_columns: Vec<String> = conn
                .prepare(
                    "SELECT ii.name FROM pragma_index_list(?1) AS il, \
                     pragma_index_info(il.name) AS ii WHERE ii.seqno = 0",
                )
                .unwrap()
                .query_map([table], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            assert!(
                leading_columns.iter().any(|c| c == "meeting_id"),
                "{table} has no index on meeting_id: {leading_columns:?}"
            );
        }
    }

    #[test]
    fn only_embedding_vectors_are_binary() {
        // Privacy by schema: there is nowhere to put raw audio or screenshots.
        let mut conn = memory();
        migrate(&mut conn).unwrap();
        let mut blobs = Vec::new();
        for table in table_names(&conn) {
            let columns: Vec<(String, String)> = conn
                .prepare("SELECT name, type FROM pragma_table_info(?1)")
                .unwrap()
                .query_map([&table], |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            for (name, ty) in columns {
                if ty.eq_ignore_ascii_case("BLOB") || ty.eq_ignore_ascii_case("ANY") {
                    blobs.push(format!("{table}.{name}"));
                }
            }
        }
        assert_eq!(blobs, vec!["embeddings.vector"]);
    }
}
