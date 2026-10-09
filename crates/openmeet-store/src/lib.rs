//! SQLite persistence for OpenMeet.
//!
//! [`Database`] owns a single [`rusqlite::Connection`] behind a mutex and
//! implements [`openmeet_core::store::MeetingStore`] (the engine's
//! persistence contract) plus the history, settings, privacy-rule and
//! retention operations used by the desktop app.
//!
//! * **Schema**: versioned migrations tracked with `PRAGMA user_version`. Each
//!   migration is embedded SQL applied inside its own transaction; a database
//!   written by a newer OpenMeet (higher schema version) is refused instead of
//!   being modified.
//! * **Connection**: WAL journal (file databases), `foreign_keys = ON`,
//!   `secure_delete = ON` (deleted content is overwritten), in-memory temp
//!   storage and a 5 s busy timeout.
//! * **Privacy**: no raw audio and no screenshots are ever stored. Logs and
//!   error messages contain ids, counts, sizes and durations only — never
//!   transcript text, model output, OCR text or settings content. Deleting a
//!   meeting (or applying retention) also checkpoints the write-ahead log so
//!   deleted content does not linger in the `-wal` file.
//!
//! All methods are blocking; call them from a background thread (for example
//! the engine's persistence worker or `tokio::task::spawn_blocking`).
//!
//! ```
//! use std::sync::Arc;
//!
//! use openmeet_core::store::MeetingStore;
//! use openmeet_store::{Database, SCHEMA_VERSION};
//!
//! let db = Arc::new(Database::open_in_memory()?);
//! assert_eq!(db.schema_version()?, SCHEMA_VERSION);
//! // The engine only sees the persistence contract.
//! let store: Arc<dyn MeetingStore> = db.clone();
//! assert!(store.load_meeting("unknown").is_err());
//! assert!(db.list_meetings(50, 0)?.is_empty());
//! # Ok::<(), openmeet_core::StoreError>(())
//! ```

#![warn(missing_docs)]

mod codec;
mod database;
mod error;
mod history;
mod migrations;
mod privacy;
mod records;
mod settings;

pub use database::Database;
pub use migrations::SCHEMA_VERSION;
