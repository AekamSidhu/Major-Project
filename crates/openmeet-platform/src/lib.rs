//! OpenMeet OS integration.
//!
//! This crate is the only place where OpenMeet talks to the operating system
//! directly. It provides:
//!
//! * [`audio`] – microphone and system-audio capture (cpal on a dedicated
//!   thread; WASAPI loopback on Windows, Core Audio process taps on macOS
//!   14.6+, PulseAudio/PipeWire monitor sources on Linux), converted to the
//!   16 kHz mono pipeline format.
//! * [`screen`] – screen source enumeration and a privacy-aware screen sampler
//!   (privacy rules, region masks, downscaling, change detection, PNG).
//! * [`ocr`] – the local OS OCR engine (Windows.Media.Ocr on Windows).
//! * [`privacy`] – excluding the overlay window from screen capture using the
//!   documented OS mechanism, with honest limitation statements.
//! * [`focus`] – keeping the meeting application focused.
//! * [`apps`] – detecting running meeting applications (UX only).
//! * [`system`] – process resource statistics for diagnostics.
//!
//! Only documented OS APIs are used: no hooks, no injection into other
//! processes, no keylogging and no undocumented capture tricks. Nothing in this
//! crate logs transcript text, OCR text, window contents or screenshots.

#![warn(missing_docs)]

pub mod apps;
pub mod audio;
mod error;
pub mod focus;
pub mod ocr;
pub mod privacy;
pub mod screen;
pub mod system;
#[cfg(windows)]
mod win;

pub use error::PlatformError;
