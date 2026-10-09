//! The [`Database`] handle: opening, connection configuration and health.

use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use openmeet_core::StoreError;
use parking_lot::{Mutex, MutexGuard};
use rusqlite::Connection;
use tracing::{debug, info, warn};

use crate::error::SqlResultExt;
use crate::migrations::{self, SCHEMA_VERSION};

/// How long a statement waits for a lock held by another connection.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
/// Distinct statements prepared by the store, with headroom.
const STATEMENT_CACHE_CAPACITY: usize = 64;
/// Size the `-wal` file is truncated back to after checkpoints.
const JOURNAL_SIZE_LIMIT_BYTES: i64 = 16 * 1024 * 1024;

/// OpenMeet's SQLite database.
///
/// A single connection guarded by a mutex: every method takes the lock for
/// its whole duration, so a `Database` can be shared between threads (e.g. in
/// an `Arc`) and multi-statement operations are never interleaved. Methods
/// block on disk I/O and must not be called from the audio thread or directly
/// on an async executor.
pub struct Database {
    conn: Mutex<Connection>,
    file_backed: bool,
}

impl fmt::Debug for Database {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database")
            .field("file_backed", &self.file_backed)
            .finish_non_exhaustive()
    }
}

impl Database {
    /// Opens (creating if needed) the database file at `path` and migrates
    /// it to [`SCHEMA_VERSION`]. Missing parent directories are created.
    ///
    /// On Unix, directories created here are private to the user (0700) and a
    /// new database file is created with mode 0600 (SQLite gives its `-wal`
    /// and `-shm` files the same mode).
    ///
    /// Fails with [`StoreError::Migration`] when the file was written by a
    /// newer OpenMeet with a schema this build does not know.
    pub fn open(path: &Path) -> Result<Database, StoreError> {
        let started = Instant::now();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            private_fs::create_dir_all(parent).map_err(|e| {
                StoreError::Database(format!("cannot create the database directory: {e}"))
            })?;
        }
        private_fs::create_file(path)
            .map_err(|e| StoreError::Database(format!("cannot create the database file: {e}")))?;
        let conn = Connection::open(path).db()?;
        let db = Self::initialize(conn, true)?;
        info!(
            schema_version = SCHEMA_VERSION,
            elapsed_ms = elapsed_ms(started),
            "database opened"
        );
        Ok(db)
    }

    /// Opens a private, empty in-memory database with the full schema (tests,
    /// demo mode). Its content disappears when the `Database` is dropped.
    pub fn open_in_memory() -> Result<Database, StoreError> {
        let conn = Connection::open_in_memory().db()?;
        Self::initialize(conn, false)
    }

    fn initialize(mut conn: Connection, file_backed: bool) -> Result<Database, StoreError> {
        configure(&conn, file_backed)?;
        migrations::migrate(&mut conn)?;
        Ok(Database {
            conn: Mutex::new(conn),
            file_backed,
        })
    }

    /// The schema version recorded in the database (`PRAGMA user_version`).
    pub fn schema_version(&self) -> Result<u32, StoreError> {
        migrations::user_version(&self.lock())
    }

    /// Cheap liveness check for diagnostics: the database file is readable,
    /// its schema is at [`SCHEMA_VERSION`] and foreign keys are enforced.
    pub fn health_check(&self) -> Result<(), StoreError> {
        let conn = self.lock();
        let _objects: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_schema", [], |row| row.get(0))
            .db()?;
        let version = migrations::user_version(&conn)?;
        if version != SCHEMA_VERSION {
            return Err(StoreError::Migration(format!(
                "database schema version {version} does not match the expected version {SCHEMA_VERSION}"
            )));
        }
        let foreign_keys: bool = conn
            .pragma_query_value(None, "foreign_keys", |row| row.get(0))
            .db()?;
        if !foreign_keys {
            return Err(StoreError::Database(
                "foreign key enforcement is disabled".to_string(),
            ));
        }
        Ok(())
    }

    /// Exclusive access to the connection for one operation.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock()
    }
}

/// Per-connection settings. `foreign_keys` is verified because cascading
/// deletes are what guarantee that deleting a meeting removes all of its
/// data.
fn configure(conn: &Connection, file_backed: bool) -> Result<(), StoreError> {
    conn.busy_timeout(BUSY_TIMEOUT).db()?;
    conn.set_prepared_statement_cache_capacity(STATEMENT_CACHE_CAPACITY);
    // secure_delete: deleted rows are overwritten with zeros instead of
    // lingering in free pages. temp_store: sorter and VACUUM temporaries stay
    // in memory, so meeting content is never written to temp files.
    conn.execute_batch(
        "PRAGMA foreign_keys = ON;
         PRAGMA secure_delete = ON;
         PRAGMA temp_store = MEMORY;",
    )
    .db()?;
    let foreign_keys: bool = conn
        .pragma_query_value(None, "foreign_keys", |row| row.get(0))
        .db()?;
    if !foreign_keys {
        return Err(StoreError::Database(
            "SQLite foreign key enforcement is unavailable".to_string(),
        ));
    }

    if file_backed {
        let mode = enable_wal(conn, BUSY_TIMEOUT)?;
        if mode.eq_ignore_ascii_case("wal") {
            // NORMAL is durable against application crashes in WAL mode and
            // avoids an fsync per transaction.
            conn.pragma_update(None, "synchronous", "NORMAL").db()?;
            conn.pragma_update(None, "journal_size_limit", JOURNAL_SIZE_LIMIT_BYTES)
                .db()?;
        } else {
            // Keep the default (FULL) synchronous mode, which rollback
            // journals need to be safe against power loss.
            warn!(journal_mode = %mode, "WAL journal mode unavailable; using the reported mode");
        }
    }
    Ok(())
}

/// Switches the file to WAL journaling and returns the journal mode SQLite
/// reports.
///
/// Switching a fresh (rollback-journal) file to WAL needs an exclusive lock,
/// which the statement has to upgrade to from the read lock it already
/// holds. SQLite reports `SQLITE_BUSY` immediately in that situation instead
/// of invoking the busy handler, so when several connections open a new file
/// at the same moment the switch is retried here with a short backoff, for
/// at most `max_wait`. On a file already in WAL mode it is a no-op.
fn enable_wal(conn: &Connection, max_wait: Duration) -> Result<String, StoreError> {
    let deadline = Instant::now() + max_wait;
    let mut delay = Duration::from_millis(1);
    let mut retries = 0u32;
    loop {
        match conn
            .pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get::<_, String>(0))
        {
            Ok(mode) => {
                if retries > 0 {
                    debug!(retries, "enabled write-ahead logging after lock contention");
                }
                return Ok(mode);
            }
            Err(e) if is_busy(&e) && Instant::now() + delay < deadline => {
                retries += 1;
                std::thread::sleep(delay);
                delay = (delay * 2).min(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(StoreError::Database(format!(
                    "cannot enable write-ahead logging: {e}"
                )));
            }
        }
    }
}

/// True for `SQLITE_BUSY` / `SQLITE_LOCKED` (and their extended codes).
fn is_busy(err: &rusqlite::Error) -> bool {
    matches!(
        err,
        rusqlite::Error::SqliteFailure(e, _)
            if matches!(
                e.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            )
    )
}

/// Copies the write-ahead log into the database file and truncates it to
/// zero bytes so content deleted from the database does not linger in the
/// `-wal` file. A failure is logged but not fatal: the deletion itself is
/// already committed and SQLite checkpoints again later.
pub(crate) fn truncate_wal(conn: &Connection) {
    let result = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
        ))
    });
    match result {
        Ok((0, _, _)) => debug!("write-ahead log checkpointed and truncated"),
        Ok((_, log_frames, checkpointed_frames)) => warn!(
            log_frames,
            checkpointed_frames, "write-ahead log checkpoint incomplete: database busy"
        ),
        Err(e) => warn!(error = %e, "write-ahead log checkpoint failed"),
    }
}

/// Rebuilds the database file so freed pages are released.
pub(crate) fn vacuum(conn: &Connection) -> Result<(), StoreError> {
    let started = Instant::now();
    conn.execute_batch("VACUUM").map_err(|e| {
        StoreError::Database(format!(
            "meeting data was deleted, but compacting the database failed: {e}"
        ))
    })?;
    debug!(elapsed_ms = elapsed_ms(started), "database compacted");
    Ok(())
}

/// Milliseconds since `started`, for logs.
pub(crate) fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// File-system helpers that keep the database private to the current user
/// on Unix. Windows relies on the per-user ACLs of the app data directory.
mod private_fs {
    use std::io;
    use std::path::Path;

    #[cfg(unix)]
    pub(super) fn create_dir_all(dir: &Path) -> io::Result<()> {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
    }

    #[cfg(not(unix))]
    pub(super) fn create_dir_all(dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)
    }

    /// Creates an empty file (a valid empty SQLite database) readable only
    /// by its owner. An existing file is left untouched.
    #[cfg(unix)]
    pub(super) fn create_file(path: &Path) -> io::Result<()> {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
        {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// SQLite creates the file itself on platforms without Unix modes.
    #[cfg(not(unix))]
    pub(super) fn create_file(_path: &Path) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_database_is_configured_and_migrated() {
        let db = Database::open_in_memory().unwrap();
        assert_eq!(db.schema_version().unwrap(), SCHEMA_VERSION);
        db.health_check().unwrap();
        let conn = db.lock();
        let secure_delete: i64 = conn
            .pragma_query_value(None, "secure_delete", |r| r.get(0))
            .unwrap();
        assert_eq!(secure_delete, 1);
        let temp_store: i64 = conn
            .pragma_query_value(None, "temp_store", |r| r.get(0))
            .unwrap();
        assert_eq!(temp_store, 2, "temp_store must be MEMORY");
    }

    #[test]
    fn file_database_uses_wal_and_busy_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("nested").join("openmeet.db")).unwrap();
        assert!(format!("{db:?}").contains("file_backed: true"));
        let conn = db.lock();
        let mode: String = conn
            .pragma_query_value(None, "journal_mode", |r| r.get(0))
            .unwrap();
        assert_eq!(mode.to_ascii_lowercase(), "wal");
        let timeout: i64 = conn
            .pragma_query_value(None, "busy_timeout", |r| r.get(0))
            .unwrap();
        assert_eq!(timeout, 5000);
        let synchronous: i64 = conn
            .pragma_query_value(None, "synchronous", |r| r.get(0))
            .unwrap();
        assert_eq!(synchronous, 1, "synchronous must be NORMAL in WAL mode");
    }

    #[test]
    fn health_check_detects_disabled_foreign_keys_and_schema_drift() {
        let db = Database::open_in_memory().unwrap();
        db.lock()
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();
        assert!(matches!(db.health_check(), Err(StoreError::Database(_))));
        db.lock()
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        db.health_check().unwrap();
        db.lock()
            .pragma_update(None, "user_version", SCHEMA_VERSION + 5)
            .unwrap();
        assert!(matches!(db.health_check(), Err(StoreError::Migration(_))));
    }

    #[test]
    fn busy_errors_are_recognized_and_wal_switch_reports_failures() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locked.db");
        let holder = Connection::open(&path).unwrap();
        holder
            .execute_batch("CREATE TABLE t (x INTEGER); BEGIN EXCLUSIVE;")
            .unwrap();

        let other = Connection::open(&path).unwrap();
        other.busy_timeout(Duration::ZERO).unwrap();
        let err = other
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get::<_, i64>(0))
            .unwrap_err();
        assert!(is_busy(&err), "{err:?}");
        assert!(!is_busy(&rusqlite::Error::InvalidQuery));

        // While another connection holds the file exclusively the switch
        // keeps retrying until `max_wait`, then fails with a typed,
        // descriptive error.
        let max_wait = Duration::from_millis(150);
        let started = Instant::now();
        let err = enable_wal(&other, max_wait).unwrap_err();
        assert!(started.elapsed() >= max_wait / 2, "{:?}", started.elapsed());
        assert!(
            matches!(&err, StoreError::Database(m) if m.contains("write-ahead logging")),
            "{err:?}"
        );

        holder.execute_batch("COMMIT;").unwrap();
        assert_eq!(
            enable_wal(&other, max_wait).unwrap().to_ascii_lowercase(),
            "wal"
        );
    }

    #[test]
    fn wal_checkpoint_is_harmless_in_memory() {
        let db = Database::open_in_memory().unwrap();
        truncate_wal(&db.lock());
        vacuum(&db.lock()).unwrap();
    }
}
