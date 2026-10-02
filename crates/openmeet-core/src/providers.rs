//! Provider abstraction (spec §15, §29, §45).
//!
//! The meeting engine only talks to these traits; concrete adapters (OpenAI,
//! Anthropic, Ollama, the local sidecar, OS OCR, mocks) live in other crates.
//! Every long-running operation takes a [`CancellationToken`] and must stop
//! promptly when it fires.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::audio::{AudioClip, AudioFrame};
use crate::error::ProviderError;
use crate::ids::Id;
use crate::ipc::SecretProvider;
use crate::types::{
    AudioSourceKind, Component, ProviderInfo, ProviderLocation, RequestKind, TranscriptDraft,
};

// ---------------------------------------------------------------------------
// LLM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    User,
    Assistant,
}

/// Inline image for vision-capable models.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageInput {
    /// e.g. "image/png"
    pub mime_type: String,
    pub data_base64: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Images attached to this message (placed before the text).
    pub images: Vec<ImageInput>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: ChatRole::User,
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: ChatRole::Assistant,
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn with_image(mut self, image: ImageInput) -> Self {
        self.images.push(image);
        self
    }
}

/// Provider-neutral chat completion request.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmRequest {
    pub request_id: Id,
    pub kind: RequestKind,
    /// System prompt (prompt templates live in `openmeet-intel::prompts`).
    pub system: String,
    /// Conversation; must start with a user message and alternate roles.
    pub messages: Vec<ChatMessage>,
    /// Visible-output budget. Adapters may add headroom for models that spend
    /// tokens on internal reasoning.
    pub max_output_tokens: u32,
    /// `None` = provider default. Adapters omit it for models that reject
    /// sampling parameters.
    pub temperature: Option<f32>,
    /// The caller expects JSON / NDJSON output (adapters may enable a JSON
    /// mode when it is compatible with streaming line-delimited output).
    pub json_output: bool,
    /// Live, latency-critical request: adapters should use the lowest
    /// reasoning effort the model allows.
    pub latency_sensitive: bool,
}

impl LlmRequest {
    pub fn new(kind: RequestKind, system: impl Into<String>, user: impl Into<String>) -> Self {
        LlmRequest {
            request_id: crate::ids::new_id(),
            kind,
            system: system.into(),
            messages: vec![ChatMessage::user(user)],
            max_output_tokens: 1200,
            temperature: None,
            json_output: false,
            latency_sensitive: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LlmChunk {
    /// Visible text delta. Reasoning/thinking content is never surfaced.
    Delta(String),
    /// End of stream.
    Done {
        stop_reason: Option<String>,
        usage: Option<TokenUsage>,
    },
}

pub type LlmStream = BoxStream<'static, Result<LlmChunk, ProviderError>>;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn info(&self) -> ProviderInfo;

    fn supports_images(&self) -> bool;

    /// Starts a streaming completion. Resolves once the provider accepted the
    /// request (non-success HTTP statuses become errors here). The returned
    /// stream ends after a `Done` chunk, an error, or cancellation.
    async fn stream(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> Result<LlmStream, ProviderError>;

    /// Cheap connectivity / credential check.
    async fn health_check(&self) -> Result<(), ProviderError>;

    /// Models offered by the provider (for the settings UI).
    async fn list_models(&self) -> Result<Vec<String>, ProviderError>;
}

/// Runs a request to completion and returns the concatenated text.
pub async fn complete_text(
    provider: &dyn LlmProvider,
    request: LlmRequest,
    cancel: CancellationToken,
) -> Result<String, ProviderError> {
    let mut stream = provider.stream(request, cancel.clone()).await?;
    let mut out = String::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
            next = stream.next() => match next {
                Some(Ok(LlmChunk::Delta(t))) => out.push_str(&t),
                Some(Ok(LlmChunk::Done { stop_reason, .. })) => {
                    if stop_reason.as_deref() == Some("refusal") {
                        return Err(ProviderError::Refused("refusal".to_string()));
                    }
                    break;
                }
                Some(Err(e)) => return Err(e),
                None => break,
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionRequest {
    pub clip: AudioClip,
    /// ISO-639-1 hint, `None` = auto.
    pub language: Option<String>,
    /// Biasing prompt (vocabulary and/or recent transcript tail).
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranscriptionOutput {
    pub text: String,
    pub confidence: Option<f32>,
    pub language: Option<String>,
}

/// Utterance-level recognizer (OpenAI /audio/transcriptions, faster-whisper
/// sidecar, ...). Wrapped by a VAD session adapter to provide streaming.
#[async_trait]
pub trait Transcriber: Send + Sync {
    fn info(&self) -> ProviderInfo;

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        cancel: CancellationToken,
    ) -> Result<TranscriptionOutput, ProviderError>;

    async fn health_check(&self) -> Result<(), ProviderError>;
}

/// Events produced by a streaming transcription session.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionEvent {
    /// Voice activity started (lets the UI show "listening" / the engine track
    /// turns before text is available).
    SpeechStarted { source: AudioSourceKind, at_ms: i64 },
    /// Voice activity ended. When `discarded` is false the utterance is now
    /// being transcribed and exactly one final `Draft` or `Error` with the same
    /// `utterance_id` follows; `discarded` utterances (too short / noise)
    /// produce nothing further.
    SpeechEnded {
        source: AudioSourceKind,
        at_ms: i64,
        utterance_id: String,
        discarded: bool,
    },
    /// Partial or final transcript.
    Draft(TranscriptDraft),
    /// Recoverable error for one utterance; the session keeps running.
    Error {
        source: AudioSourceKind,
        utterance_id: Option<String>,
        error: ProviderError,
    },
    /// A final transcription produced no usable text (silence, noise or a
    /// filtered recognizer hallucination). Closes `utterance_id`.
    Empty {
        source: AudioSourceKind,
        utterance_id: String,
    },
    /// Normalized input level 0..1.
    Level { source: AudioSourceKind, level: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionSessionConfig {
    pub source: AudioSourceKind,
    pub language: Option<String>,
    pub vocabulary: Option<String>,
    pub interim_results: bool,
    /// 0..1, higher = more sensitive.
    pub vad_sensitivity: f32,
    /// Silence that ends an utterance.
    pub silence_ms: u32,
}

/// A running streaming transcription for one audio source (spec §15:
/// start / stop / partial / final).
pub trait TranscriptionSession: Send {
    /// Feed 16 kHz mono audio. Must not block.
    fn push_audio(&mut self, frame: AudioFrame);
    /// Finalize the current utterance now (pause / stop / turn end).
    fn flush(&mut self);
    /// Stop the session; in-flight transcriptions are cancelled.
    fn stop(&mut self);
}

#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    fn info(&self) -> ProviderInfo;

    /// Starts a session. Events are delivered on `events` until `stop`.
    fn start_session(
        &self,
        config: TranscriptionSessionConfig,
        events: mpsc::UnboundedSender<TranscriptionEvent>,
    ) -> Result<Box<dyn TranscriptionSession>, ProviderError>;

    async fn health_check(&self) -> Result<(), ProviderError>;
}

// ---------------------------------------------------------------------------
// Vision & OCR
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VisionContext {
    pub source_label: String,
    pub application: Option<String>,
    pub window_title: Option<String>,
    /// Short recent transcript excerpt for grounding.
    pub recent_transcript: String,
    /// Question currently being answered, if any.
    pub question: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionRequest {
    /// PNG bytes (downscaled frame). Shared to avoid copies.
    pub image_png: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    /// Local OCR text, when available.
    pub ocr_text: Option<String>,
    pub context: VisionContext,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VisionResult {
    pub description: String,
    pub visible_text: Option<String>,
    pub entities: Vec<String>,
    pub application: Option<String>,
    /// "code" | "diagram" | "document" | "slides" | "browser" | "terminal" | "other"
    pub content_type: Option<String>,
}

#[async_trait]
pub trait VisionProvider: Send + Sync {
    fn info(&self) -> ProviderInfo;

    async fn analyze(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<VisionResult, ProviderError>;
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OcrOutput {
    pub text: String,
    pub lines: Vec<String>,
}

/// Local OCR engine. Blocking – call from `spawn_blocking`.
pub trait OcrEngine: Send + Sync {
    fn info(&self) -> ProviderInfo;
    fn recognize_png(&self, png: &[u8]) -> Result<OcrOutput, ProviderError>;
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn info(&self) -> ProviderInfo;

    /// One vector per input text, same order. Vectors need not be normalized.
    async fn embed(
        &self,
        texts: Vec<String>,
        cancel: CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError>;
}

// ---------------------------------------------------------------------------
// Secrets & provider sets
// ---------------------------------------------------------------------------

/// Resolves API keys (OS keychain first, environment variables as a
/// development fallback). Implemented by the desktop app.
pub trait SecretResolver: Send + Sync {
    fn api_key(&self, provider: SecretProvider) -> Option<String>;
}

/// Static resolver for tests.
#[derive(Debug, Default, Clone)]
pub struct StaticSecrets(pub Vec<(SecretProvider, String)>);

impl SecretResolver for StaticSecrets {
    fn api_key(&self, provider: SecretProvider) -> Option<String> {
        self.0
            .iter()
            .find(|(p, _)| *p == provider)
            .map(|(_, k)| k.clone())
    }
}

/// A provider that could not be constructed, with a user-facing reason.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderIssue {
    pub component: Component,
    pub message: String,
}

/// The active set of providers used by the engine. Any member may be absent;
/// the engine degrades gracefully (spec §92).
#[derive(Clone, Default)]
pub struct ProviderSet {
    pub llm: Option<Arc<dyn LlmProvider>>,
    /// Optional cheaper model for classification / compression.
    pub fast_llm: Option<Arc<dyn LlmProvider>>,
    pub transcription: Option<Arc<dyn TranscriptionProvider>>,
    pub vision: Option<Arc<dyn VisionProvider>>,
    pub ocr: Option<Arc<dyn OcrEngine>>,
    pub embeddings: Option<Arc<dyn EmbeddingProvider>>,
    pub issues: Vec<ProviderIssue>,
}

impl ProviderSet {
    /// Model used for background work (classification, compression).
    pub fn background_llm(&self) -> Option<Arc<dyn LlmProvider>> {
        self.fast_llm.clone().or_else(|| self.llm.clone())
    }

    /// True when no configured component sends data off the device.
    pub fn all_local(&self) -> bool {
        let locs = [
            self.llm.as_ref().map(|p| p.info().location),
            self.fast_llm.as_ref().map(|p| p.info().location),
            self.transcription.as_ref().map(|p| p.info().location),
            self.vision.as_ref().map(|p| p.info().location),
            self.ocr.as_ref().map(|p| p.info().location),
            self.embeddings.as_ref().map(|p| p.info().location),
        ];
        locs.into_iter()
            .flatten()
            .all(|l| l != ProviderLocation::Remote)
    }

    pub fn issue_for(&self, component: Component) -> Option<&ProviderIssue> {
        self.issues.iter().find(|i| i.component == component)
    }
}

impl std::fmt::Debug for ProviderSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSet")
            .field("llm", &self.llm.as_ref().map(|p| p.info()))
            .field("fast_llm", &self.fast_llm.as_ref().map(|p| p.info()))
            .field(
                "transcription",
                &self.transcription.as_ref().map(|p| p.info()),
            )
            .field("vision", &self.vision.as_ref().map(|p| p.info()))
            .field("ocr", &self.ocr.as_ref().map(|p| p.info()))
            .field("embeddings", &self.embeddings.as_ref().map(|p| p.info()))
            .field("issues", &self.issues)
            .finish()
    }
}

/// Classifies a base URL as local (loopback) or remote.
pub fn location_for_url(url: &str) -> ProviderLocation {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or("");
    let host = if let Some(stripped) = authority.strip_prefix('[') {
        stripped.split(']').next().unwrap_or("")
    } else {
        authority.split(':').next().unwrap_or("")
    }
    .to_ascii_lowercase();
    let is_loopback = host == "localhost"
        || host == "::1"
        || host.starts_with("127.")
        || host.ends_with(".localhost");
    if is_loopback {
        ProviderLocation::Local
    } else {
        ProviderLocation::Remote
    }
}

/// Host part of a URL for display (no path, no credentials).
pub fn display_host(url: &str) -> String {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or("");
    authority.rsplit('@').next().unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_locations() {
        assert_eq!(
            location_for_url("http://127.0.0.1:11434"),
            ProviderLocation::Local
        );
        assert_eq!(
            location_for_url("http://localhost:8765/v1"),
            ProviderLocation::Local
        );
        assert_eq!(
            location_for_url("http://[::1]:8080"),
            ProviderLocation::Local
        );
        assert_eq!(
            location_for_url("https://api.openai.com"),
            ProviderLocation::Remote
        );
        assert_eq!(
            location_for_url("http://192.168.1.20:11434"),
            ProviderLocation::Remote
        );
        assert_eq!(
            location_for_url("http://localhost.evil.com"),
            ProviderLocation::Remote
        );
    }

    #[test]
    fn display_hosts() {
        assert_eq!(display_host("https://api.openai.com/v1"), "api.openai.com");
        assert_eq!(display_host("http://u:p@host:1/x"), "host:1");
    }

    #[test]
    fn static_secrets() {
        let s = StaticSecrets(vec![(SecretProvider::Openai, "k".into())]);
        assert_eq!(s.api_key(SecretProvider::Openai).as_deref(), Some("k"));
        assert!(s.api_key(SecretProvider::Anthropic).is_none());
    }
}
