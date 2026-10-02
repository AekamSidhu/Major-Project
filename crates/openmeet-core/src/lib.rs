//! OpenMeet core contract.
//!
//! This crate defines the shared language every other OpenMeet crate speaks:
//! domain types, the meeting event bus, provider traits, persisted settings and
//! the typed IPC payloads exchanged with the React frontend.
//!
//! It deliberately has no OS, network or database dependencies so that every
//! subsystem (and every test) can depend on it cheaply.

pub mod audio;
pub mod error;
pub mod events;
pub mod ids;
pub mod ipc;
pub mod providers;
pub mod settings;
pub mod store;
pub mod types;

pub use error::{CoreError, ProviderError, StoreError};
pub use events::{EventSink, MeetingEvent};
pub use ids::{Id, new_id, now_ms};
