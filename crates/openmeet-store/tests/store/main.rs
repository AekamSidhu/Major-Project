//! Integration tests of the SQLite store, built as a single test binary.
//!
//! None of them needs network access, API keys, audio devices or a display.
//! The log-privacy test lives in its own binary (`tests/log_privacy.rs`).

mod concurrency;
mod database;
mod fixtures;
mod history;
mod privacy;
mod records;
mod settings;
