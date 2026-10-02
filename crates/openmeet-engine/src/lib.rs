//! OpenMeet real-time meeting engine.
//!
//! The engine is OS- and network-agnostic: it receives audio frames and screen
//! frames from the platform layer, drives provider traits from
//! `openmeet-core`, and reports everything as [`openmeet_core::MeetingEvent`]s
//! through an [`openmeet_core::EventSink`].
//!
//! Pipeline (spec §13):
//!
//! ```text
//! AudioFrame → VAD session → Transcriber → TranscriptSegment → conversation loop
//!   → question detection / turn tracking → context assembly → LLM stream → events
//! ScreenFrame → change-filtered (platform) → OCR / vision → ScreenObservation → context
//! ```

mod assist;
mod benchmark;
mod conversation;
mod demo;
mod embedder;
mod engine;
mod memory;
mod metrics;
mod persist;
mod requests;
mod runtime;
mod screen;
pub mod state;
pub mod transcription;

pub use engine::{Engine, EngineDeps, EngineError};
pub use metrics::{HealthRegistry, LatencyTracker};
pub use requests::{RequestManager, RequestTicket};
pub use transcription::VadTranscriptionProvider;
