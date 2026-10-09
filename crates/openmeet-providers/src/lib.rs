//! OpenMeet HTTP provider adapters.
//!
//! Implementations of the `openmeet-core` provider traits for:
//! * OpenAI, any OpenAI-compatible server (LM Studio, vLLM, llama.cpp, …) and
//!   the local AI sidecar ([`openai`]);
//! * Anthropic ([`anthropic`]);
//! * Ollama ([`ollama`]);
//!
//! plus the sidecar status client ([`local_ai`]), the settings-driven
//! [`factory`], an incremental Server-Sent Events parser ([`sse`]) and shared
//! HTTP plumbing ([`http`]).
//!
//! Every network operation takes a `CancellationToken` and stops promptly
//! when it fires. Logs and errors never contain API keys, prompts, transcripts
//! or model output — only ids, sizes, counts, durations and error kinds.

#![warn(missing_docs)]

pub mod anthropic;
pub mod factory;
pub mod http;
pub mod local_ai;
pub mod ollama;
pub mod openai;
pub mod sse;

mod lines;
mod stream;
mod think;

pub use factory::{
    BuildContext, build_embeddings, build_fast_llm, build_llm, build_transcriber, build_vision_llm,
    resolved_base_url,
};
pub use local_ai::{LocalAiClient, LocalAiHealth};
pub use sse::{SseEvent, SseParser};
