//! OpenMeet intelligence layer: pure algorithms with no network and no OS
//! access.
//!
//! * [`vad`] – energy voice-activity detection and utterance segmentation.
//! * [`text`] – normalization, stemming, paraphrase-robust similarity, token
//!   budgeting, local hashing embeddings and a BM25 keyword index.
//! * [`detect`] – heuristic question detection (plus voice commands).
//! * [`topics`] – conversation topic tracking.
//! * [`claims`] – claims, entities, metrics and action-item extraction.
//! * [`prompts`] – versioned, provider-independent prompt builders and tolerant
//!   output parsers.
//! * [`vision`] – screen understanding adapters (vision LLM / local OCR).
//! * [`export`] – Markdown / text / JSON export and the extractive summary.
//! * [`mock`] – deterministic providers for tests, developer tools and demos.
//! * [`demo`] – scripted demo scenarios.
//!
//! Privacy: nothing in this crate logs transcript text, prompts, model output,
//! OCR text or screenshots. Logs only carry ids, sizes, counts and durations.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

// pub mod claims;
// pub mod demo;
// pub mod detect;
// pub mod export;
// pub mod mock;
pub mod prompts;
pub mod text;
// pub mod topics;
pub mod vad;
// pub mod vision;

mod json;
mod lexicon;
mod timefmt;
