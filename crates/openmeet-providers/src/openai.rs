//! OpenAI adapters, also used for any OpenAI-compatible server (LM Studio,
//! vLLM, llama.cpp, …) and for the local AI sidecar.
//!
//! * [`OpenAiLlm`] — streaming `POST /v1/chat/completions`.
//! * [`OpenAiTranscriber`] — multipart `POST /v1/audio/transcriptions`.
//! * [`OpenAiEmbeddings`] — `POST /v1/embeddings`.
//!
//! Model rules (see [`is_reasoning_model`]): reasoning models (`gpt-5*`,
//! `o1*`, `o3*`, `o4*`) get no `temperature`, a `reasoning_effort` and
//! [`REASONING_HEADROOM_TOKENS`] extra output tokens; `api.openai.com` gets
//! `max_completion_tokens`, compatible servers `max_tokens`.
//!
//! Models and servers drift, so when a request is rejected with HTTP 400/422
//! because of one of these optional parameters (unsupported `temperature`,
//! effort value, token field, `stream_options`, or a too-large token limit)
//! the adapter adjusts that parameter, retries, and remembers the adjustment
//! for later requests.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use openmeet_core::ProviderError;
use openmeet_core::audio::encode_wav_pcm16;
use openmeet_core::providers::{
    ChatRole, EmbeddingProvider, LlmChunk, LlmProvider, LlmRequest, LlmStream, TokenUsage,
    Transcriber, TranscriptionOutput, TranscriptionRequest,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::ProviderInfo;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::multipart::{Form, Part};
use reqwest::{Method, RequestBuilder, Response};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{
    Endpoint, HttpFailure, MAX_JSON_BODY_BYTES, Timeouts, build_client, error_kind,
    openai_api_base, parse_json, parse_token_limit, provider_info, read_body, read_json, root_url,
    send, sensitive_header, validate_endpoint_url,
};
use crate::sse::SseParser;
use crate::stream::{ChunkDecoder, ChunkQueue, StreamMeta, from_items, into_llm_stream};
use crate::think::ThinkFilter;

/// Extra completion tokens reserved for hidden reasoning on reasoning models.
pub const REASONING_HEADROOM_TOKENS: u32 = 2000;
/// Clips shorter than this are not sent (providers reject them and they
/// cannot contain a word).
pub const MIN_TRANSCRIPTION_CLIP_MS: u64 = 100;
/// Transcription prompts are cut to their last this-many characters (Whisper
/// only considers the final ~224 tokens anyway).
pub const MAX_TRANSCRIPTION_PROMPT_CHARS: usize = 1000;
/// Texts per embeddings request.
pub const EMBEDDING_BATCH_SIZE: usize = 64;
/// Attempts per chat request, including parameter-adjusting retries.
const MAX_ATTEMPTS: usize = 4;
/// Display name of the sidecar's transcription service.
pub const LOCAL_AI_TRANSCRIPTION_NAME: &str = "Local AI (faster-whisper)";
/// Display name of the sidecar's embedding service.
pub const LOCAL_AI_EMBEDDINGS_NAME: &str = "Local AI (fastembed)";
/// Model reported in [`ProviderInfo`] when the sidecar's default embedding
/// model is used.
const LOCAL_AI_DEFAULT_MODEL_LABEL: &str = "default";

/// Flavour of an OpenAI-style endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenAiFlavor {
    /// The OpenAI API (`api.openai.com` or a configured override of it).
    OpenAi,
    /// Any server implementing the OpenAI REST API.
    Compatible,
    /// The bundled local AI sidecar (faster-whisper, local embeddings).
    LocalAi,
}

impl OpenAiFlavor {
    /// Stable provider id used in [`ProviderInfo::id`].
    pub fn provider_id(self) -> &'static str {
        match self {
            OpenAiFlavor::OpenAi => "openai",
            OpenAiFlavor::Compatible => "openai_compatible",
            OpenAiFlavor::LocalAi => "local_ai",
        }
    }

    /// Human-readable provider name.
    pub fn display_name(self) -> &'static str {
        match self {
            OpenAiFlavor::OpenAi => "OpenAI",
            OpenAiFlavor::Compatible => "OpenAI-compatible",
            OpenAiFlavor::LocalAi => "Local AI",
        }
    }
}

// ---------------------------------------------------------------------------
// Model rules
// ---------------------------------------------------------------------------

/// Lower-cased model name without a routing prefix (`openai/gpt-5` → `gpt-5`).
fn base_model(model: &str) -> String {
    let m = model.trim().to_ascii_lowercase();
    match m.rsplit_once('/') {
        Some((_, name)) => name.to_string(),
        None => m,
    }
}

/// OpenAI reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`): they reject
/// `temperature`, take `reasoning_effort`, and spend output tokens on hidden
/// reasoning.
pub fn is_reasoning_model(model: &str) -> bool {
    let m = base_model(model);
    ["gpt-5", "o1", "o3", "o4"].iter().any(|p| m.starts_with(p))
}

/// `reasoning_effort` for a reasoning model: `"minimal"` for latency-sensitive
/// requests on `gpt-5*` (`"low"` on the o-series, which has no minimal
/// effort), otherwise the configured effort.
pub fn reasoning_effort_for(
    model: &str,
    latency_sensitive: bool,
    configured: ReasoningEffort,
) -> &'static str {
    if latency_sensitive {
        if base_model(model).starts_with("gpt-5") {
            "minimal"
        } else {
            "low"
        }
    } else {
        configured.as_str()
    }
}

/// Whether the endpoint is the OpenAI API itself.
fn is_openai_host(base_url: &str) -> bool {
    openmeet_core::providers::display_host(base_url)
        .split(':')
        .next()
        .is_some_and(|h| h.eq_ignore_ascii_case("api.openai.com"))
}

/// Best-effort image-input support from the model name.
fn model_supports_images(flavor: OpenAiFlavor, model: &str) -> bool {
    let m = base_model(model);
    if flavor == OpenAiFlavor::OpenAi {
        const TEXT_ONLY: [&str; 8] = [
            "gpt-3.5",
            "o1-mini",
            "o1-preview",
            "o3-mini",
            "gpt-oss",
            "davinci",
            "babbage",
            "gpt-4-32k",
        ];
        let text_only =
            TEXT_ONLY.iter().any(|p| m.starts_with(p)) || m == "gpt-4" || m.starts_with("gpt-4-0");
        return !text_only;
    }
    const VISION_MARKERS: [&str; 28] = [
        "vision",
        "-vl",
        "vl-",
        "_vl",
        "llava",
        "pixtral",
        "gemma-3",
        "gemma3",
        "minicpm-v",
        "moondream",
        "internvl",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "claude",
        "gemini",
        "llama-4",
        "llama4",
        "mistral-small-3",
        "multimodal",
        "molmo",
        "idefics",
        "smolvlm",
        "glm-4v",
        "glm-4.5v",
        "qwen2.5-omni",
        "qwen3-omni",
        "kimi-vl",
    ];
    VISION_MARKERS.iter().any(|marker| m.contains(marker))
}

/// Best-effort guess whether a model served by an OpenAI-compatible server
/// accepts images, from its name.
pub fn name_suggests_images(model: &str) -> bool {
    model_supports_images(OpenAiFlavor::Compatible, model)
}

/// Chat-capable model ids from the OpenAI model list (embeddings, speech,
/// image and Responses-API-only models are hidden).
fn is_chat_model_id(id: &str) -> bool {
    const EXCLUDE: [&str; 14] = [
        "embedding",
        "tts",
        "whisper",
        "transcribe",
        "dall-e",
        "gpt-image",
        "moderation",
        "davinci",
        "babbage",
        "sora",
        "realtime",
        "codex",
        "deep-research",
        "computer-use",
    ];
    let id = id.to_ascii_lowercase();
    !EXCLUDE.iter().any(|x| id.contains(x)) && !id.ends_with("-pro") && !id.contains("-pro-")
}

// ---------------------------------------------------------------------------
// Shared endpoint plumbing
// ---------------------------------------------------------------------------

struct Base {
    flavor: OpenAiFlavor,
    display_name: &'static str,
    base_url: String,
    api_base: String,
    root: String,
    host: String,
    api_key: Option<String>,
    timeouts: Timeouts,
    client: reqwest::Client,
}

impl Base {
    fn new(flavor: OpenAiFlavor, endpoint: Endpoint, what: &str) -> Result<Self, ProviderError> {
        validate_endpoint_url(&endpoint.base_url, what)?;
        if let Some(key) = &endpoint.api_key {
            sensitive_header(&format!("Bearer {key}"))?;
        }
        let client = build_client(endpoint.location(), endpoint.timeouts.connect)?;
        Ok(Base {
            flavor,
            display_name: flavor.display_name(),
            api_base: openai_api_base(&endpoint.base_url),
            root: root_url(&endpoint.base_url),
            host: endpoint.host(),
            base_url: endpoint.base_url,
            api_key: endpoint.api_key,
            timeouts: endpoint.timeouts,
            client,
        })
    }

    fn request(&self, method: Method, path: &str) -> Result<RequestBuilder, ProviderError> {
        let mut builder = self
            .client
            .request(method, format!("{}{}", self.api_base, path));
        if let Some(key) = &self.api_key {
            builder = builder.header(AUTHORIZATION, sensitive_header(&format!("Bearer {key}"))?);
        }
        Ok(builder)
    }

    fn info(&self, model: &str) -> ProviderInfo {
        provider_info(
            self.flavor.provider_id(),
            self.display_name,
            model,
            &self.base_url,
        )
    }

    fn secret(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// `GET /v1/models` → sorted ids.
    async fn model_ids(&self, cancel: &CancellationToken) -> Result<Vec<String>, ProviderError> {
        let response = self.get_models_response(cancel).await?;
        if !response.status().is_success() {
            return Err(
                HttpFailure::read(response, cancel, self.secret(), &self.host)
                    .await
                    .into_error(),
            );
        }
        let v: Value = read_json(
            response,
            "model list",
            cancel,
            self.timeouts.health,
            &self.host,
        )
        .await?;
        let mut ids: Vec<String> = v
            .get("data")
            .or_else(|| v.get("models"))
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|m| m.get("id").or_else(|| m.get("name")))
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .ok_or_else(|| {
                ProviderError::InvalidResponse("the model list has no `data` array".to_string())
            })?;
        ids.sort();
        ids.dedup();
        Ok(ids)
    }

    async fn get_models_response(
        &self,
        cancel: &CancellationToken,
    ) -> Result<Response, ProviderError> {
        let builder = self.request(Method::GET, "/models")?;
        send(builder, cancel, self.timeouts.health, &self.host).await
    }

    /// Cheap reachability / credential check. Compatible servers without a
    /// model list (404/405) still count as reachable.
    async fn health(&self) -> Result<(), ProviderError> {
        let cancel = CancellationToken::new();
        let response = self.get_models_response(&cancel).await?;
        let status = response.status();
        if status.is_success()
            || (self.flavor != OpenAiFlavor::OpenAi && matches!(status.as_u16(), 404 | 405))
        {
            return Ok(());
        }
        Err(
            HttpFailure::read(response, &cancel, self.secret(), &self.host)
                .await
                .into_error(),
        )
    }
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

/// Which output-token field the server expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenField {
    MaxCompletionTokens,
    MaxTokens,
}

/// Optional request parameters, decided by model rules and learned quirks.
#[derive(Debug, Clone, PartialEq)]
struct ChatParams {
    token_field: TokenField,
    max_tokens: u32,
    temperature: Option<f32>,
    reasoning_effort: Option<&'static str>,
    stream_options: bool,
}

/// Parameter adjustments learned from rejected requests.
#[derive(Debug, Clone, Copy, Default)]
struct ChatQuirks {
    no_temperature: bool,
    minimal_effort_replacement: Option<&'static str>,
    omit_effort: bool,
    token_field: Option<TokenField>,
    no_stream_options: bool,
    max_tokens_limit: Option<u32>,
}

/// Streaming chat completions against OpenAI or an OpenAI-compatible server.
pub struct OpenAiLlm {
    base: Base,
    model: String,
    reasoning_effort: ReasoningEffort,
    supports_images: bool,
    quirks: Mutex<ChatQuirks>,
}

impl OpenAiLlm {
    /// Creates the adapter. Fails for an invalid base URL, an empty model or
    /// an API key that cannot be sent as a header.
    pub fn new(
        flavor: OpenAiFlavor,
        endpoint: Endpoint,
        model: impl Into<String>,
    ) -> Result<Self, ProviderError> {
        let model = model.into().trim().to_string();
        if model.is_empty() {
            return Err(ProviderError::NotConfigured(
                "LLM model is not set".to_string(),
            ));
        }
        let base = Base::new(flavor, endpoint, "LLM")?;
        Ok(OpenAiLlm {
            supports_images: model_supports_images(flavor, &model),
            base,
            model,
            reasoning_effort: ReasoningEffort::Low,
            quirks: Mutex::new(ChatQuirks::default()),
        })
    }

    /// Effort for requests that are not latency-sensitive (reasoning models).
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Overrides the image-support guess derived from the model name.
    pub fn with_image_support(mut self, supported: bool) -> Self {
        self.supports_images = supported;
        self
    }

    /// Configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Endpoint flavour.
    pub fn flavor(&self) -> OpenAiFlavor {
        self.base.flavor
    }

    fn quirks(&self) -> MutexGuard<'_, ChatQuirks> {
        self.quirks.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn params_for(&self, request: &LlmRequest) -> ChatParams {
        let quirks = *self.quirks();
        let reasoning = is_reasoning_model(&self.model);
        let default_field =
            if self.base.flavor == OpenAiFlavor::OpenAi || is_openai_host(&self.base.base_url) {
                TokenField::MaxCompletionTokens
            } else {
                TokenField::MaxTokens
            };
        let mut max_tokens = request.max_output_tokens.max(1);
        if reasoning {
            max_tokens = max_tokens.saturating_add(REASONING_HEADROOM_TOKENS);
        }
        if let Some(limit) = quirks.max_tokens_limit {
            max_tokens = max_tokens.min(limit);
        }
        let temperature = if reasoning || quirks.no_temperature {
            None
        } else {
            request
                .temperature
                .filter(|t| t.is_finite())
                .map(|t| t.clamp(0.0, 2.0))
        };
        let reasoning_effort = if reasoning && !quirks.omit_effort {
            let effort = reasoning_effort_for(
                &self.model,
                request.latency_sensitive,
                self.reasoning_effort,
            );
            Some(match (effort, quirks.minimal_effort_replacement) {
                ("minimal", Some(replacement)) => replacement,
                _ => effort,
            })
        } else {
            None
        };
        ChatParams {
            token_field: quirks.token_field.unwrap_or(default_field),
            max_tokens,
            temperature,
            reasoning_effort,
            stream_options: !quirks.no_stream_options,
        }
    }

    /// Adjusts the parameter a 400/422 message complains about; `None` when
    /// the message is not about an optional parameter.
    fn downgrade(&self, params: &ChatParams, message: &str) -> Option<ChatParams> {
        let mut next = params.clone();
        let mut quirks = self.quirks();
        let mentions_tokens =
            message.contains("max_tokens") || message.contains("max_completion_tokens");
        if mentions_tokens {
            if let Some(limit) = parse_token_limit(message).filter(|l| *l < params.max_tokens) {
                next.max_tokens = limit.max(1);
                quirks.max_tokens_limit = Some(next.max_tokens);
                return Some(next);
            }
        }
        if params.temperature.is_some() && message.contains("temperature") {
            next.temperature = None;
            quirks.no_temperature = true;
            return Some(next);
        }
        if let Some(effort) = params.reasoning_effort {
            if message.contains("reasoning_effort")
                || message.contains("reasoning effort")
                || message.contains("reasoning.effort")
            {
                if effort == "minimal" {
                    let replacement = if message.contains("'none'") || message.contains("\"none\"")
                    {
                        "none"
                    } else {
                        "low"
                    };
                    next.reasoning_effort = Some(replacement);
                    quirks.minimal_effort_replacement = Some(replacement);
                } else {
                    next.reasoning_effort = None;
                    quirks.omit_effort = true;
                }
                return Some(next);
            }
        }
        if params.token_field == TokenField::MaxTokens
            && message.contains("max_tokens")
            && message.contains("max_completion_tokens")
        {
            next.token_field = TokenField::MaxCompletionTokens;
            quirks.token_field = Some(next.token_field);
            return Some(next);
        }
        let rejected = [
            "unsupported",
            "unrecognized",
            "unknown",
            "not permitted",
            "not allowed",
            "extra",
        ]
        .iter()
        .any(|w| message.contains(w));
        if params.token_field == TokenField::MaxCompletionTokens
            && message.contains("max_completion_tokens")
            && rejected
        {
            next.token_field = TokenField::MaxTokens;
            quirks.token_field = Some(next.token_field);
            return Some(next);
        }
        if params.stream_options && message.contains("stream_options") {
            next.stream_options = false;
            quirks.no_stream_options = true;
            return Some(next);
        }
        None
    }

    async fn open_stream(
        &self,
        response: Response,
        request_id: String,
        cancel: CancellationToken,
        started: Instant,
    ) -> Result<LlmStream, ProviderError> {
        let filter_think = self.base.flavor != OpenAiFlavor::OpenAi;
        if is_json_response(&response) {
            // The server ignored `stream: true` and answered in one piece.
            let mut response = response;
            let body = read_body(
                &mut response,
                MAX_JSON_BODY_BYTES,
                false,
                &cancel,
                self.base.timeouts.stream_idle,
                &self.base.host,
            )
            .await?;
            let value: Value = parse_json(&body, "chat completion")?;
            return Ok(from_items(completion_items(&value, filter_think)));
        }
        let decoder = OpenAiDecoder::new(filter_think);
        let meta = StreamMeta {
            provider: self.base.flavor.provider_id(),
            request_id,
            host: self.base.host.clone(),
            started,
        };
        Ok(into_llm_stream(
            response,
            decoder,
            cancel,
            self.base.timeouts.stream_idle,
            meta,
        ))
    }
}

/// Builds the `/chat/completions` body.
fn chat_body(model: &str, request: &LlmRequest, params: &ChatParams) -> Value {
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if !request.system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    for message in &request.messages {
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        if message.images.is_empty() || message.role == ChatRole::Assistant {
            messages.push(json!({"role": role, "content": message.content}));
            continue;
        }
        let mut parts: Vec<Value> = message
            .images
            .iter()
            .map(|image| {
                json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{};base64,{}", image.mime_type, image.data_base64)}
                })
            })
            .collect();
        if !message.content.is_empty() {
            parts.push(json!({"type": "text", "text": message.content}));
        }
        messages.push(json!({"role": role, "content": parts}));
    }
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), json!(true));
    if params.stream_options {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    let field = match params.token_field {
        TokenField::MaxCompletionTokens => "max_completion_tokens",
        TokenField::MaxTokens => "max_tokens",
    };
    body.insert(field.into(), json!(params.max_tokens));
    if let Some(t) = params.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(effort) = params.reasoning_effort {
        body.insert("reasoning_effort".into(), json!(effort));
    }
    Value::Object(body)
}

fn is_json_response(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("application/json"))
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    if !value.is_object() {
        return None;
    }
    let get = |keys: [&str; 2]| {
        keys.iter()
            .find_map(|k| value.get(*k).and_then(Value::as_u64))
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
    };
    let input = get(["prompt_tokens", "input_tokens"]);
    let output = get(["completion_tokens", "output_tokens"]);
    if input.is_none() && output.is_none() {
        return None;
    }
    Some(TokenUsage {
        input_tokens: input.unwrap_or(0),
        output_tokens: output.unwrap_or(0),
    })
}

/// Maps an error object found inside a 200 response (stream or body).
fn embedded_error(error: &Value) -> ProviderError {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())
        .map(|m| crate::http::sanitize_message(m, None))
        .unwrap_or_else(|| "unknown error".to_string());
    let kind = error
        .get("type")
        .or_else(|| error.get("code"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if kind.contains("rate_limit") {
        ProviderError::RateLimited(message)
    } else {
        ProviderError::Other(format!("the provider reported an error: {message}"))
    }
}

/// Items for a non-streaming chat completion body.
fn completion_items(value: &Value, filter_think: bool) -> Vec<Result<LlmChunk, ProviderError>> {
    if let Some(error) = value.get("error") {
        return vec![Err(embedded_error(error))];
    }
    let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    else {
        return vec![Err(ProviderError::InvalidResponse(
            "the chat completion has no choices".to_string(),
        ))];
    };
    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if filter_think {
        let mut filter = ThinkFilter::new();
        let mut visible = filter.push(&text);
        visible.push_str(&filter.finish());
        text = visible;
    }
    let refused = message
        .get("refusal")
        .and_then(Value::as_str)
        .is_some_and(|r| !r.is_empty());
    let stop_reason = if refused {
        Some("refusal".to_string())
    } else {
        choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let mut items = Vec::new();
    if !text.is_empty() {
        items.push(Ok(LlmChunk::Delta(text)));
    }
    items.push(Ok(LlmChunk::Done {
        stop_reason,
        usage: value.get("usage").and_then(parse_usage),
    }));
    items
}

/// Decodes the chat-completions SSE stream.
struct OpenAiDecoder {
    sse: SseParser,
    think: Option<ThinkFilter>,
    finish_reason: Option<String>,
    usage: Option<TokenUsage>,
    refusal: bool,
    done: bool,
    skipped_events: usize,
}

impl OpenAiDecoder {
    fn new(filter_think: bool) -> Self {
        OpenAiDecoder {
            sse: SseParser::new(),
            think: filter_think.then(ThinkFilter::new),
            finish_reason: None,
            usage: None,
            refusal: false,
            done: false,
            skipped_events: 0,
        }
    }

    fn visible(&mut self, text: &str) -> String {
        match self.think.as_mut() {
            Some(filter) => filter.push(text),
            None => text.to_string(),
        }
    }

    fn finish_ok(&mut self, out: &mut ChunkQueue) {
        if let Some(filter) = self.think.as_mut() {
            let rest = filter.finish();
            if !rest.is_empty() {
                out.push_back(Ok(LlmChunk::Delta(rest)));
            }
        }
        let stop_reason = if self.refusal {
            Some("refusal".to_string())
        } else {
            self.finish_reason.clone()
        };
        out.push_back(Ok(LlmChunk::Done {
            stop_reason,
            usage: self.usage,
        }));
        self.done = true;
    }

    fn handle(&mut self, data: &str, out: &mut ChunkQueue) {
        if self.done {
            return;
        }
        let data = data.trim();
        if data.is_empty() {
            return;
        }
        if data == "[DONE]" {
            self.finish_ok(out);
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            self.skipped_events += 1;
            return;
        };
        if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
            out.push_back(Err(embedded_error(error)));
            self.done = true;
            return;
        }
        let choice = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| {
                choices
                    .iter()
                    .find(|c| c.get("index").and_then(Value::as_u64).unwrap_or(0) == 0)
            });
        if let Some(choice) = choice {
            if let Some(delta) = choice.get("delta") {
                // Only `content` is visible text; `reasoning_content` /
                // `reasoning` (thinking) and tool calls are ignored.
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    let visible = self.visible(text);
                    if !visible.is_empty() {
                        out.push_back(Ok(LlmChunk::Delta(visible)));
                    }
                }
                if delta
                    .get("refusal")
                    .and_then(Value::as_str)
                    .is_some_and(|r| !r.is_empty())
                {
                    self.refusal = true;
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(reason.to_string());
            }
        }
        if let Some(usage) = value.get("usage").and_then(parse_usage) {
            self.usage = Some(usage);
        }
    }
}

impl ChunkDecoder for OpenAiDecoder {
    fn decode(&mut self, bytes: &[u8], out: &mut ChunkQueue) {
        for event in self.sse.push(bytes) {
            self.handle(&event.data, out);
            if self.done {
                return;
            }
        }
    }

    fn finish(&mut self, out: &mut ChunkQueue) {
        for event in self.sse.finish() {
            self.handle(&event.data, out);
        }
        if self.skipped_events > 0 {
            tracing::debug!(
                count = self.skipped_events,
                "skipped non-JSON stream events"
            );
        }
        // Some servers close without `[DONE]`; a finish reason or usage
        // block still proves the reply is complete.
        if !self.done && (self.finish_reason.is_some() || self.usage.is_some()) {
            self.finish_ok(out);
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiLlm {
    fn info(&self) -> ProviderInfo {
        self.base.info(&self.model)
    }

    fn supports_images(&self) -> bool {
        self.supports_images
    }

    async fn stream(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> Result<LlmStream, ProviderError> {
        if request.messages.is_empty() {
            return Err(ProviderError::Other(
                "the request contains no messages".to_string(),
            ));
        }
        let started = Instant::now();
        let provider = self.base.flavor.provider_id();
        let images: usize = request.messages.iter().map(|m| m.images.len()).sum();
        let mut params = self.params_for(&request);
        let mut attempt = 1;
        loop {
            tracing::debug!(
                provider,
                model = %self.model,
                request_id = %request.request_id,
                kind = ?request.kind,
                messages = request.messages.len(),
                images,
                max_tokens = params.max_tokens,
                attempt,
                "sending chat completion request"
            );
            let body = serde_json::to_vec(&chat_body(&self.model, &request, &params))
                .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
            let builder = self
                .base
                .request(Method::POST, "/chat/completions")?
                .header(CONTENT_TYPE, "application/json")
                .header(ACCEPT, "text/event-stream")
                .body(body);
            let response = match send(
                builder,
                &cancel,
                self.base.timeouts.response,
                &self.base.host,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    log_failure(provider, &request.request_id, &e);
                    return Err(e);
                }
            };
            if response.status().is_success() {
                tracing::debug!(
                    provider,
                    request_id = %request.request_id,
                    status = response.status().as_u16(),
                    headers_ms = started.elapsed().as_millis() as u64,
                    "chat completion stream opened"
                );
                return self
                    .open_stream(response, request.request_id.clone(), cancel, started)
                    .await;
            }
            let failure =
                HttpFailure::read(response, &cancel, self.base.secret(), &self.base.host).await;
            if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }
            if matches!(failure.status.as_u16(), 400 | 422) && attempt < MAX_ATTEMPTS {
                if let Some(next) = self.downgrade(&params, &failure.message_lower()) {
                    tracing::debug!(
                        provider,
                        request_id = %request.request_id,
                        status = failure.status.as_u16(),
                        attempt,
                        "request rejected; retrying with adjusted parameters"
                    );
                    params = next;
                    attempt += 1;
                    continue;
                }
            }
            let error = failure.into_error();
            log_failure(provider, &request.request_id, &error);
            return Err(error);
        }
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        self.base.health().await
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let ids = self.base.model_ids(&CancellationToken::new()).await?;
        Ok(if self.base.flavor == OpenAiFlavor::OpenAi {
            ids.into_iter().filter(|id| is_chat_model_id(id)).collect()
        } else {
            ids
        })
    }
}

fn log_failure(provider: &'static str, request_id: &str, error: &ProviderError) {
    if matches!(error, ProviderError::Cancelled) {
        tracing::debug!(provider, request_id, "request cancelled");
    } else {
        tracing::warn!(
            provider,
            request_id,
            error_kind = error_kind(error),
            "request failed"
        );
    }
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

/// Utterance transcription through `POST /v1/audio/transcriptions`.
pub struct OpenAiTranscriber {
    base: Base,
    model: String,
}

impl OpenAiTranscriber {
    /// Creates the adapter. Fails for an invalid base URL or an empty model.
    pub fn new(
        flavor: OpenAiFlavor,
        endpoint: Endpoint,
        model: impl Into<String>,
    ) -> Result<Self, ProviderError> {
        let model = model.into().trim().to_string();
        if model.is_empty() {
            return Err(ProviderError::NotConfigured(
                "transcription model is not set".to_string(),
            ));
        }
        let mut base = Base::new(flavor, endpoint, "transcription")?;
        if flavor == OpenAiFlavor::LocalAi {
            base.display_name = LOCAL_AI_TRANSCRIPTION_NAME;
        }
        Ok(OpenAiTranscriber { base, model })
    }

    /// Configured model.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Last `max` characters of `text`.
fn tail_chars(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    text.chars().skip(count - max).collect()
}

#[async_trait]
impl Transcriber for OpenAiTranscriber {
    fn info(&self) -> ProviderInfo {
        self.base.info(&self.model)
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        cancel: CancellationToken,
    ) -> Result<TranscriptionOutput, ProviderError> {
        let clip = &request.clip;
        let rate = clip.sample_rate.max(1);
        let provider = self.base.flavor.provider_id();
        if (clip.samples.len() as u64) * 1000 < u64::from(rate) * MIN_TRANSCRIPTION_CLIP_MS {
            tracing::debug!(
                provider,
                utterance_id = %clip.utterance_id,
                samples = clip.samples.len(),
                "clip too short to transcribe; skipped"
            );
            return Ok(TranscriptionOutput::default());
        }
        let wav = encode_wav_pcm16(&clip.samples, rate);
        let wav_bytes = wav.len();
        let file = Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|_| ProviderError::Other("could not build the audio upload".to_string()))?;
        let mut form = Form::new()
            .part("file", file)
            .text("model", self.model.clone())
            .text("response_format", "json");
        if let Some(language) = request
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("auto"))
        {
            form = form.text("language", language.to_ascii_lowercase());
        }
        if let Some(prompt) = request
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
        {
            form = form.text("prompt", tail_chars(prompt, MAX_TRANSCRIPTION_PROMPT_CHARS));
        }
        let started = Instant::now();
        tracing::debug!(
            provider,
            model = %self.model,
            utterance_id = %clip.utterance_id,
            audio_ms = clip.duration_ms(),
            wav_bytes,
            "sending transcription request"
        );
        let builder = self
            .base
            .request(Method::POST, "/audio/transcriptions")?
            .header(ACCEPT, "application/json")
            .multipart(form);
        let deadline = self.base.timeouts.request;
        let response = send(builder, &cancel, deadline, &self.base.host)
            .await
            .inspect_err(|e| log_failure(provider, &clip.utterance_id, e))?;
        if !response.status().is_success() {
            let error = HttpFailure::read(response, &cancel, self.base.secret(), &self.base.host)
                .await
                .into_error();
            log_failure(provider, &clip.utterance_id, &error);
            return Err(error);
        }
        let remaining = deadline
            .saturating_sub(started.elapsed())
            .max(Duration::from_secs(1));
        let value: Value = read_json(
            response,
            "transcription",
            &cancel,
            remaining,
            &self.base.host,
        )
        .await?;
        let text = value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProviderError::InvalidResponse("the transcription response has no text".to_string())
            })?
            .trim()
            .to_string();
        let language = value
            .get("language")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string);
        let confidence = value
            .get("confidence")
            .and_then(Value::as_f64)
            .filter(|c| c.is_finite())
            .map(|c| c.clamp(0.0, 1.0) as f32);
        tracing::debug!(
            provider,
            utterance_id = %clip.utterance_id,
            elapsed_ms = started.elapsed().as_millis() as u64,
            chars = text.chars().count(),
            "transcription completed"
        );
        Ok(TranscriptionOutput {
            text,
            confidence,
            language,
        })
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        if self.base.flavor == OpenAiFlavor::LocalAi {
            let health = crate::local_ai::fetch_health(
                &self.base.client,
                &self.base.root,
                self.base.timeouts.health,
                &self.base.host,
                &CancellationToken::new(),
            )
            .await?;
            return if health.ok {
                Ok(())
            } else {
                Err(ProviderError::Other(
                    "the local AI service is not ready".to_string(),
                ))
            };
        }
        self.base.health().await
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// Embeddings through `POST /v1/embeddings`.
pub struct OpenAiEmbeddings {
    base: Base,
    model: String,
}

impl OpenAiEmbeddings {
    /// Creates the adapter. Fails for an invalid base URL or an empty model;
    /// the local AI sidecar accepts an empty model (it then uses the model it
    /// was started with, and `model` is omitted from requests).
    pub fn new(
        flavor: OpenAiFlavor,
        endpoint: Endpoint,
        model: impl Into<String>,
    ) -> Result<Self, ProviderError> {
        let model = model.into().trim().to_string();
        if model.is_empty() && flavor != OpenAiFlavor::LocalAi {
            return Err(ProviderError::NotConfigured(
                "embedding model is not set".to_string(),
            ));
        }
        let mut base = Base::new(flavor, endpoint, "embeddings")?;
        if flavor == OpenAiFlavor::LocalAi {
            base.display_name = LOCAL_AI_EMBEDDINGS_NAME;
        }
        Ok(OpenAiEmbeddings { base, model })
    }

    /// Configured model (empty = the sidecar's default).
    pub fn model(&self) -> &str {
        &self.model
    }

    fn body(&self, input: &[&str]) -> Value {
        if self.model.is_empty() {
            json!({"input": input})
        } else {
            json!({"model": self.model, "input": input})
        }
    }

    async fn embed_batch(
        &self,
        batch: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        // Blank strings are rejected by the OpenAI API.
        let input: Vec<&str> = batch
            .iter()
            .map(|t| if t.trim().is_empty() { " " } else { t.as_str() })
            .collect();
        let body = serde_json::to_vec(&self.body(&input))
            .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
        let builder = self
            .base
            .request(Method::POST, "/embeddings")?
            .header(CONTENT_TYPE, "application/json")
            .body(body);
        let started = Instant::now();
        let deadline = self.base.timeouts.request;
        let response = send(builder, cancel, deadline, &self.base.host).await?;
        if !response.status().is_success() {
            return Err(
                HttpFailure::read(response, cancel, self.base.secret(), &self.base.host)
                    .await
                    .into_error(),
            );
        }
        let remaining = deadline
            .saturating_sub(started.elapsed())
            .max(Duration::from_secs(1));
        let value: Value =
            read_json(response, "embeddings", cancel, remaining, &self.base.host).await?;
        let items = value.get("data").and_then(Value::as_array).ok_or_else(|| {
            ProviderError::InvalidResponse(
                "the embeddings response has no `data` array".to_string(),
            )
        })?;
        let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(items.len());
        for (position, item) in items.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(Value::as_u64)
                .map(|i| i as usize)
                .unwrap_or(position);
            let vector = item
                .get("embedding")
                .and_then(decode_embedding)
                .ok_or_else(|| {
                    ProviderError::InvalidResponse(
                        "an embedding in the response is malformed".to_string(),
                    )
                })?;
            indexed.push((index, vector));
        }
        indexed.sort_by_key(|(i, _)| *i);
        if indexed.len() != batch.len()
            || indexed
                .iter()
                .enumerate()
                .any(|(expected, (i, _))| *i != expected)
        {
            return Err(ProviderError::InvalidResponse(format!(
                "expected {} embeddings, received {}",
                batch.len(),
                indexed.len()
            )));
        }
        Ok(indexed.into_iter().map(|(_, v)| v).collect())
    }
}

/// An embedding as a float array or a base64 string of little-endian f32.
fn decode_embedding(value: &Value) -> Option<Vec<f32>> {
    let vector: Vec<f32> = if let Some(items) = value.as_array() {
        items
            .iter()
            .map(|x| x.as_f64().map(|f| f as f32))
            .collect::<Option<Vec<f32>>>()?
    } else {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(value.as_str()?)
            .ok()?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    (!vector.is_empty() && vector.iter().all(|x| x.is_finite())).then_some(vector)
}

/// Checks that every vector has the same dimension.
pub(crate) fn check_dimensions(vectors: &[Vec<f32>]) -> Result<(), ProviderError> {
    if let Some(first) = vectors.first() {
        if vectors.iter().any(|v| v.len() != first.len()) {
            return Err(ProviderError::InvalidResponse(
                "embeddings have inconsistent dimensions".to_string(),
            ));
        }
    }
    Ok(())
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddings {
    fn info(&self) -> ProviderInfo {
        if self.model.is_empty() {
            self.base.info(LOCAL_AI_DEFAULT_MODEL_LABEL)
        } else {
            self.base.info(&self.model)
        }
    }

    async fn embed(
        &self,
        texts: Vec<String>,
        cancel: CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBEDDING_BATCH_SIZE) {
            let batch_vectors = self.embed_batch(batch, &cancel).await.inspect_err(|e| {
                tracing::warn!(
                    provider = self.base.flavor.provider_id(),
                    texts = texts.len(),
                    error_kind = error_kind(e),
                    "embedding request failed"
                )
            })?;
            vectors.extend(batch_vectors);
        }
        check_dimensions(&vectors)?;
        tracing::debug!(
            provider = self.base.flavor.provider_id(),
            texts = texts.len(),
            dims = vectors.first().map(Vec::len).unwrap_or(0),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "embeddings computed"
        );
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::providers::{ChatMessage, ImageInput};
    use openmeet_core::types::RequestKind;

    fn llm(flavor: OpenAiFlavor, url: &str, model: &str) -> OpenAiLlm {
        OpenAiLlm::new(flavor, Endpoint::new(url), model).unwrap()
    }

    fn request(latency: bool) -> LlmRequest {
        let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "sys", "hello");
        r.temperature = Some(0.3);
        r.max_output_tokens = 1000;
        r.latency_sensitive = latency;
        r
    }

    #[test]
    fn reasoning_model_detection() {
        for m in [
            "gpt-5",
            "gpt-5-mini",
            "GPT-5.1",
            "o1",
            "o3-mini",
            "o4-mini",
            "openai/gpt-5-nano",
        ] {
            assert!(is_reasoning_model(m), "{m}");
        }
        for m in [
            "gpt-4o",
            "gpt-4.1-mini",
            "llama3.2",
            "qwen2.5:7b",
            "gpt-oss-20b",
        ] {
            assert!(!is_reasoning_model(m), "{m}");
        }
        assert_eq!(
            reasoning_effort_for("gpt-5-mini", true, ReasoningEffort::High),
            "minimal"
        );
        assert_eq!(
            reasoning_effort_for("o3", true, ReasoningEffort::High),
            "low"
        );
        assert_eq!(
            reasoning_effort_for("gpt-5", false, ReasoningEffort::Medium),
            "medium"
        );
    }

    #[test]
    fn params_follow_model_rules() {
        let openai = llm(OpenAiFlavor::OpenAi, "https://api.openai.com", "gpt-5-mini");
        let p = openai.params_for(&request(true));
        assert_eq!(p.token_field, TokenField::MaxCompletionTokens);
        assert_eq!(p.max_tokens, 1000 + REASONING_HEADROOM_TOKENS);
        assert_eq!(p.temperature, None);
        assert_eq!(p.reasoning_effort, Some("minimal"));

        let classic = llm(
            OpenAiFlavor::OpenAi,
            "https://api.openai.com/v1",
            "gpt-4o-mini",
        );
        let p = classic.params_for(&request(true));
        assert_eq!(p.max_tokens, 1000);
        assert_eq!(p.temperature, Some(0.3));
        assert_eq!(p.reasoning_effort, None);

        let compat = llm(
            OpenAiFlavor::Compatible,
            "http://localhost:1234/v1",
            "qwen2.5-7b",
        );
        assert_eq!(
            compat.params_for(&request(true)).token_field,
            TokenField::MaxTokens
        );
        // A compatible flavour pointed at the OpenAI host still gets the
        // OpenAI field.
        let compat_openai = llm(OpenAiFlavor::Compatible, "https://api.openai.com", "gpt-4o");
        assert_eq!(
            compat_openai.params_for(&request(true)).token_field,
            TokenField::MaxCompletionTokens
        );
        let effort = llm(OpenAiFlavor::OpenAi, "https://api.openai.com", "o3")
            .with_reasoning_effort(ReasoningEffort::High);
        assert_eq!(
            effort.params_for(&request(false)).reasoning_effort,
            Some("high")
        );
    }

    #[test]
    fn downgrades_are_learned() {
        let l = llm(OpenAiFlavor::OpenAi, "https://api.openai.com", "gpt-5.1");
        let p = l.params_for(&request(true));
        let msg = "unsupported value: 'reasoning_effort' does not support 'minimal' with this model. supported values are: 'none', 'low', 'medium', and 'high'.";
        let next = l.downgrade(&p, msg).unwrap();
        assert_eq!(next.reasoning_effort, Some("none"));
        assert_eq!(l.params_for(&request(true)).reasoning_effort, Some("none"));
        // Non-minimal efforts are unaffected by the learned replacement.
        assert_eq!(l.params_for(&request(false)).reasoning_effort, Some("low"));

        let t = llm(OpenAiFlavor::Compatible, "http://localhost:1/v1", "m");
        let p = t.params_for(&request(true));
        let next = t
            .downgrade(&p, "unsupported parameter: 'temperature'")
            .unwrap();
        assert_eq!(next.temperature, None);
        assert_eq!(t.params_for(&request(true)).temperature, None);

        let s = llm(OpenAiFlavor::Compatible, "http://localhost:1/v1", "m");
        let p = s.params_for(&request(true));
        let next = s
            .downgrade(&p, "body.stream_options: extra inputs are not permitted")
            .unwrap();
        assert!(!next.stream_options);

        let m = llm(OpenAiFlavor::Compatible, "http://localhost:1/v1", "m");
        let p = m.params_for(&request(true));
        let next = m
            .downgrade(&p, "unsupported parameter: 'max_tokens' is not supported with this model. use 'max_completion_tokens' instead.")
            .unwrap();
        assert_eq!(next.token_field, TokenField::MaxCompletionTokens);

        let limit = llm(OpenAiFlavor::OpenAi, "https://api.openai.com", "gpt-4o");
        let mut big = request(true);
        big.max_output_tokens = 20_000;
        let p = limit.params_for(&big);
        let next = limit
            .downgrade(&p, "max_tokens is too large: 20000. this model supports at most 16384 completion tokens, whereas you provided 20000.")
            .unwrap();
        assert_eq!(next.max_tokens, 16_384);

        assert!(limit.downgrade(&p, "messages must not be empty").is_none());
    }

    #[test]
    fn body_shape() {
        let l = llm(OpenAiFlavor::OpenAi, "https://api.openai.com", "gpt-4o");
        let mut r = request(true);
        r.messages = vec![ChatMessage::user("look").with_image(ImageInput {
            mime_type: "image/png".into(),
            data_base64: "AAAA".into(),
        })];
        let body = chat_body("gpt-4o", &r, &l.params_for(&r));
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        let content = &body["messages"][1]["content"];
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(content[0]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(content[1]["text"], "look");
        assert_eq!(body["max_completion_tokens"], 1000);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn image_support_guess() {
        assert!(model_supports_images(OpenAiFlavor::OpenAi, "gpt-4o-mini"));
        assert!(model_supports_images(OpenAiFlavor::OpenAi, "gpt-5"));
        assert!(!model_supports_images(
            OpenAiFlavor::OpenAi,
            "gpt-3.5-turbo"
        ));
        assert!(!model_supports_images(OpenAiFlavor::OpenAi, "o3-mini"));
        assert!(model_supports_images(
            OpenAiFlavor::Compatible,
            "Qwen2.5-VL-7B-Instruct"
        ));
        assert!(model_supports_images(
            OpenAiFlavor::Compatible,
            "llava-v1.6"
        ));
        assert!(!model_supports_images(
            OpenAiFlavor::Compatible,
            "qwen2.5-7b-instruct"
        ));
    }

    #[test]
    fn chat_model_filter() {
        assert!(is_chat_model_id("gpt-4o"));
        assert!(is_chat_model_id("gpt-5-mini"));
        assert!(!is_chat_model_id("text-embedding-3-small"));
        assert!(!is_chat_model_id("whisper-1"));
        assert!(!is_chat_model_id("gpt-4o-mini-transcribe"));
        assert!(!is_chat_model_id("o3-pro"));
        assert!(!is_chat_model_id("dall-e-3"));
    }

    #[test]
    fn decoder_handles_reasoning_refusal_and_usage() {
        let mut d = OpenAiDecoder::new(true);
        let mut q = ChunkQueue::new();
        let stream = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"secret\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"<think>hidden</think>Hi\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"other choice\"}}]}\n\n",
            "data: not json\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        );
        d.decode(stream.as_bytes(), &mut q);
        let items: Vec<_> = q.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            items,
            vec![
                LlmChunk::Delta("Hi".into()),
                LlmChunk::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(TokenUsage {
                        input_tokens: 3,
                        output_tokens: 2
                    }),
                }
            ]
        );

        let mut d = OpenAiDecoder::new(false);
        let mut q = ChunkQueue::new();
        d.decode(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            &mut q,
        );
        d.finish(&mut q);
        assert_eq!(
            q.pop_back().unwrap().unwrap(),
            LlmChunk::Done {
                stop_reason: Some("refusal".into()),
                usage: None
            }
        );
    }

    #[test]
    fn non_streaming_body() {
        let v: Value = serde_json::from_str(
            r#"{"choices":[{"message":{"content":"<think>x</think>Answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#,
        )
        .unwrap();
        let items: Vec<_> = completion_items(&v, true)
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(items[0], LlmChunk::Delta("Answer".into()));
        assert!(matches!(items[1], LlmChunk::Done { .. }));
    }

    #[test]
    fn embedding_decoding() {
        assert_eq!(decode_embedding(&json!([0.5, -1.0])), Some(vec![0.5, -1.0]));
        let b64 = base64::engine::general_purpose::STANDARD
            .encode([1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat());
        assert_eq!(decode_embedding(&json!(b64)), Some(vec![1.0, 2.0]));
        assert_eq!(decode_embedding(&json!([])), None);
        assert_eq!(decode_embedding(&json!(["x"])), None);
        assert!(check_dimensions(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn prompt_tail() {
        assert_eq!(tail_chars("abcdef", 3), "def");
        assert_eq!(tail_chars("äöü", 5), "äöü");
    }
}
