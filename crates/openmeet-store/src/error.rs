//! Mapping of SQLite errors onto [`StoreError`].
//!
//! Every statement binds values as parameters (nothing is interpolated into
//! SQL), so SQLite/rusqlite error messages only ever contain static SQL,
//! column names, error codes and numbers — never meeting content.

use openmeet_core::StoreError;

/// Converts a rusqlite error into the store's error type. Column decoding
/// problems become [`StoreError::Serialization`], everything else
/// [`StoreError::Database`].
pub(crate) fn db_error(err: rusqlite::Error) -> StoreError {
    use rusqlite::Error as E;
    match err {
        E::FromSqlConversionFailure(..)
        | E::InvalidColumnType(..)
        | E::IntegralValueOutOfRange(..)
        | E::Utf8Error(..)
        | E::ToSqlConversionFailure(..) => StoreError::Serialization(err.to_string()),
        other => StoreError::Database(other.to_string()),
    }
}

/// `?`-friendly conversion of rusqlite results.
pub(crate) trait SqlResultExt<T> {
    fn db(self) -> Result<T, StoreError>;
}

impl<T> SqlResultExt<T> for rusqlite::Result<T> {
    fn db(self) -> Result<T, StoreError> {
        self.map_err(db_error)
    }
}

/// True when `err` is a foreign key constraint violation.
pub(crate) fn is_foreign_key_violation(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if e.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FOREIGNKEY
    )
}

/// Error for a write of a meeting-owned row. The only foreign key of those
/// tables is `meeting_id`, so a violation means the meeting does not exist
/// (reported like `MemoryStore` does: `NotFound(meeting_id)`).
pub(crate) fn meeting_write_error(err: rusqlite::Error, meeting_id: &str) -> StoreError {
    if is_foreign_key_violation(&err) {
        StoreError::NotFound(meeting_id.to_string())
    } else {
        db_error(err)
    }
}

/// Converts a count reported by SQLite into the `u32` used by the API.
pub(crate) fn count_u32<T: TryInto<u32>>(count: T) -> u32 {
    count.try_into().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn foreign_key_violations_map_to_not_found() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE parent (id TEXT PRIMARY KEY);
             CREATE TABLE child (id TEXT PRIMARY KEY,
                                 parent_id TEXT NOT NULL REFERENCES parent (id));",
        )
        .unwrap();
        let err = conn
            .execute("INSERT INTO child VALUES ('c1', 'missing')", [])
            .unwrap_err();
        assert!(is_foreign_key_violation(&err));
        assert_eq!(
            meeting_write_error(err, "missing"),
            StoreError::NotFound("missing".into())
        );

        let err = conn
            .execute("INSERT INTO nope VALUES (1)", [])
            .unwrap_err();
        assert!(!is_foreign_key_violation(&err));
        assert!(matches!(
            meeting_write_error(err, "m"),
            StoreError::Database(_)
        ));
    }

    #[test]
    fn decoding_errors_map_to_serialization() {
        let conn = Connection::open_in_memory().unwrap();
        let err = conn
            .query_row("SELECT 'text'", [], |r| r.get::<_, i64>(0))
            .unwrap_err();
        assert!(matches!(db_error(err), StoreError::Serialization(_)));
    }

    #[test]
    fn counts_saturate() {
        assert_eq!(count_u32(5usize), 5);
        assert_eq!(count_u32(7i64), 7);
        assert_eq!(count_u32(u64::MAX), u32::MAX);
    }
}
