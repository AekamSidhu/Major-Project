//! Anthropic Messages API adapter (raw HTTP — there is no official Rust SDK).
//!
//! `POST {base}/v1/messages` with `stream: true`; the SSE stream is decoded
//! into visible text only (`text_delta`), ignoring `thinking` /
//! `redacted_thinking` / `fallback` and other non-text blocks.
//!
//! Model rules ([`ModelRules::for_model`]):
//! * `temperature` only for models that accept sampling parameters (3.x and
//!   4.0 – 4.6); Opus 4.7+, Sonnet 5+, Fable and Mythos reject it;
//! * `thinking` is never sent (always-on thinking models reject
//!   `{"type":"disabled"}` and budgets); instead `output_config.effort` is sent
//!   to models that support effort (Opus/Sonnet 4.6+, Fable, Mythos — never
//!   Haiku): `"low"` for latency-sensitive requests, the configured effort
//!   otherwise;
//! * models whose thinking may run when `thinking` is omitted (Opus 5+,
//!   Sonnet 5+, Fable, Mythos) get [`THINKING_HEADROOM_TOKENS`] extra
//!   `max_tokens`;
//! * Opus 5.x, Fable 5.x and Sonnet 5.5 opt into server-side refusal
//!   fallbacks (`anthropic-beta: server-side-fallback-2026-07-01` +
//!   `"fallbacks": "default"`).
//!
//! If a request is still rejected with HTTP 400 because of one of these
//! optional parameters, the parameter is dropped (or the token limit lowered),
//! the request retried, and the adjustment remembered.

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use async_trait::async_trait;
use openmeet_core::ProviderError;
use openmeet_core::providers::{
    ChatRole, ImageInput, LlmChunk, LlmProvider, LlmRequest, LlmStream, TokenUsage,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::ProviderInfo;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderName};
use reqwest::{Method, RequestBuilder, Response};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{
    Endpoint, HttpFailure, MAX_JSON_BODY_BYTES, Timeouts, build_client, error_kind, parse_json,
    parse_token_limit, provider_info, read_body, read_json, root_url, sanitize_message, send,
    sensitive_header, validate_endpoint_url,
};
use crate::sse::SseParser;
use crate::stream::{ChunkDecoder, ChunkQueue, StreamMeta, from_items, into_llm_stream};

/// `anthropic-version` header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Beta flag for the `"fallbacks": "default"` refusal-fallback mode.
pub const REFUSAL_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Extra `max_tokens` for models whose thinking may run although `thinking`
/// is omitted.
pub const THINKING_HEADROOM_TOKENS: u32 = 4000;
/// Default model.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
const MAX_ATTEMPTS: usize = 4;
const MAX_MODEL_PAGES: usize = 10;
const PROVIDER_ID: &str = "anthropic";

/// Request rules for a Claude model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRules {
    /// Accepts `temperature` / `top_p` / `top_k`.
    pub sampling: bool,
    /// Accepts `output_config.effort`.
    pub effort: bool,
    /// Tokens added to `max_tokens` for thinking that cannot be turned off.
    pub thinking_headroom: u32,
    /// Opt into server-side refusal fallbacks.
    pub refusal_fallbacks: bool,
}

impl ModelRules {
    /// Rules for unrecognised model ids: only parameters every model accepts.
    pub const CONSERVATIVE: ModelRules = ModelRules {
        sampling: false,
        effort: false,
        thinking_headroom: 0,
        refusal_fallbacks: false,
    };

    /// Derives the rules from a model id such as `claude-opus-5-5`,
    /// `claude-sonnet-4-5-20250929` or `claude-3-7-sonnet-latest`.
    pub fn for_model(model: &str) -> ModelRules {
        let Some(id) = parse_model_id(model) else {
            return ModelRules::CONSERVATIVE;
        };
        let version = id.version;
        let at_least = |min: (u32, u32)| version.is_some_and(|v| v >= min);
        let below = |max: (u32, u32)| version.is_some_and(|v| v < max);
        let major_is = |m: u32| version.is_some_and(|v| v.0 == m);
        match id.family.as_str() {
            "fable" | "mythos" => ModelRules {
                sampling: false,
                effort: true,
                thinking_headroom: THINKING_HEADROOM_TOKENS,
                refusal_fallbacks: id.family == "fable" && major_is(5),
            },
            "opus" => ModelRules {
                sampling: below((4, 7)),
                effort: at_least((4, 6)),
                thinking_headroom: if at_least((5, 0)) { THINKING_HEADROOM_TOKENS } else { 0 },
                refusal_fallbacks: major_is(5),
            },
            "sonnet" => ModelRules {
                sampling: below((5, 0)),
                effort: at_least((4, 6)),
                thinking_headroom: if at_least((5, 0)) { THINKING_HEADROOM_TOKENS } else { 0 },
                refusal_fallbacks: version == Some((5, 5)),
            },
            "haiku" => ModelRules {
                sampling: below((5, 0)),
                effort: false,
                thinking_headroom: 0,
                refusal_fallbacks: false,
            },
            _ => ModelRules {
                sampling: below((4, 7)),
                effort: false,
                thinking_headroom: if at_least((5, 0)) { THINKING_HEADROOM_TOKENS } else { 0 },
                refusal_fallbacks: false,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelId {
    family: String,
    version: Option<(u32, u32)>,
}

/// Parses `claude-<family>-<major>[-<minor>][-<date>]` and the legacy
/// `claude-<major>[-<minor>]-<family>[-<date>]` forms (routing prefixes such as
/// `anthropic/` and suffixes such as `@date` or `[1m]` are tolerated).
fn parse_model_id(model: &str) -> Option<ModelId> {
    let lower = model.trim().to_ascii_lowercase();
    let start = lower.find("claude-")?;
    let name = &lower[start + "claude-".len()..];
    let tokens: Vec<&str> = name
        .split(['-', '@', ':', '[', ']'])
        .filter(|t| !t.is_empty())
        .collect();
    let small_number = |t: &&str| t.len() <= 2 && t.chars().all(|c| c.is_ascii_digit());
    let first = tokens.first()?;
    if small_number(first) {
        let major: u32 = first.parse().ok()?;
        let (minor, family_at) = match tokens.get(1).filter(small_number) {
            Some(m) => (m.parse().ok()?, 2),
            None => (0, 1),
        };
        let family = tokens.get(family_at)?.to_string();
        return Some(ModelId {
            family,
            version: Some((major, minor)),
        });
    }
    let family = first.to_string();
    let version = match tokens.get(1).filter(small_number) {
        Some(major) => {
            let major: u32 = major.parse().ok()?;
            let minor = tokens
                .get(2)
                .filter(small_number)
                .and_then(|m| m.parse().ok())
                .unwrap_or(0);
            Some((major, minor))
        }
        None => None,
    };
    Some(ModelId { family, version })
}

/// Parameters of one request attempt.
#[derive(Debug, Clone, PartialEq)]
struct Params {
    max_tokens: u32,
    temperature: Option<f32>,
    effort: Option<&'static str>,
    fallbacks: bool,
}

/// Adjustments learned from rejected requests.
#[derive(Debug, Clone, Copy, Default)]
struct Quirks {
    no_temperature: bool,
    no_effort: bool,
    no_fallbacks: bool,
    max_tokens_limit: Option<u32>,
}

/// Streaming Claude adapter.
pub struct AnthropicLlm {
    client: reqwest::Client,
    base_url: String,
    root: String,
    host: String,
    api_key: Option<String>,
    timeouts: Timeouts,
    model: String,
    rules: ModelRules,
    reasoning_effort: ReasoningEffort,
    quirks: Mutex<Quirks>,
}

impl AnthropicLlm {
    /// Creates the adapter. An empty model selects [`DEFAULT_MODEL`]. The API
    /// key is optional here (a local proxy may not need one); the factory
    /// requires it for the remote API.
    pub fn new(endpoint: Endpoint, model: impl Into<String>) -> Result<Self, ProviderError> {
        validate_endpoint_url(&endpoint.base_url, "Anthropic")?;
        if let Some(key) = &endpoint.api_key {
            sensitive_header(key)?;
        }
        let mut model = model.into().trim().to_string();
        if model.is_empty() {
            model = DEFAULT_MODEL.to_string();
        }
        let location = endpoint.location();
        Ok(AnthropicLlm {
            client: build_client(location, endpoint.timeouts.connect)?,
            root: root_url(&endpoint.base_url),
            host: endpoint.host(),
            base_url: endpoint.base_url,
            api_key: endpoint.api_key,
            timeouts: endpoint.timeouts,
            rules: ModelRules::for_model(&model),
            model,
            reasoning_effort: ReasoningEffort::Low,
            quirks: Mutex::new(Quirks::default()),
        })
    }

    /// Effort for requests that are not latency-sensitive.
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Rules applied to the configured model.
    pub fn rules(&self) -> ModelRules {
        self.rules
    }

    fn quirks(&self) -> MutexGuard<'_, Quirks> {
        self.quirks.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn request(&self, method: Method, path: &str) -> Result<RequestBuilder, ProviderError> {
        let mut builder = self
            .client
            .request(method, format!("{}{}", self.root, path))
            .header("anthropic-version", ANTHROPIC_VERSION);
        if let Some(key) = &self.api_key {
            builder = builder.header(HeaderName::from_static("x-api-key"), sensitive_header(key)?);
        }
        Ok(builder)
    }

    fn params_for(&self, request: &LlmRequest) -> Params {
        let quirks = *self.quirks();
        let mut max_tokens = request
            .max_output_tokens
            .max(1)
            .saturating_add(self.rules.thinking_headroom);
        if let Some(limit) = quirks.max_tokens_limit {
            max_tokens = max_tokens.min(limit);
        }
        let temperature = if self.rules.sampling && !quirks.no_temperature {
            request
                .temperature
                .filter(|t| t.is_finite())
                .map(|t| t.clamp(0.0, 1.0))
        } else {
            None
        };
        let effort = (self.rules.effort && !quirks.no_effort).then(|| {
            if request.latency_sensitive {
                "low"
            } else {
                self.reasoning_effort.as_str()
            }
        });
        Params {
            max_tokens,
            temperature,
            effort,
            fallbacks: self.rules.refusal_fallbacks && !quirks.no_fallbacks,
        }
    }

    fn downgrade(&self, params: &Params, message: &str) -> Option<Params> {
        let mut next = params.clone();
        let mut quirks = self.quirks();
        if message.contains("max_tokens") {
            if let Some(limit) = parse_token_limit(message).filter(|l| *l < params.max_tokens) {
                next.max_tokens = limit.max(1);
                quirks.max_tokens_limit = Some(next.max_tokens);
                return Some(next);
            }
        }
        if params.temperature.is_some()
            && ["temperature", "top_p", "top_k", "sampling"]
                .iter()
                .any(|w| message.contains(w))
        {
            next.temperature = None;
            quirks.no_temperature = true;
            return Some(next);
        }
        if params.effort.is_some() && (message.contains("output_config") || message.contains("effort"))
        {
            next.effort = None;
            quirks.no_effort = true;
            return Some(next);
        }
        if params.fallbacks && (message.contains("fallback") || message.contains("anthropic-beta")) {
            next.fallbacks = false;
            quirks.no_fallbacks = true;
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
        let is_json = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.to_ascii_lowercase().starts_with("application/json"));
        if is_json {
            let mut response = response;
            let body = read_body(
                &mut response,
                MAX_JSON_BODY_BYTES,
                false,
                &cancel,
                self.timeouts.stream_idle,
                &self.host,
            )
            .await?;
            let value: Value = parse_json(&body, "message")?;
            return Ok(from_items(message_items(&value)));
        }
        let meta = StreamMeta {
            provider: PROVIDER_ID,
            request_id,
            host: self.host.clone(),
            started,
        };
        Ok(into_llm_stream(
            response,
            AnthropicDecoder::new(),
            cancel,
            self.timeouts.stream_idle,
            meta,
        ))
    }
}

/// Merges consecutive messages of the same role (the API requires
/// alternation) and renders content blocks; images precede the text.
fn render_messages(request: &LlmRequest) -> Vec<Value> {
    let mut merged: Vec<(ChatRole, String, Vec<ImageInput>)> = Vec::new();
    for message in &request.messages {
        match merged.last_mut() {
            Some((role, text, images)) if *role == message.role => {
                if !message.content.is_empty() {
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(&message.content);
                }
                images.extend(message.images.iter().cloned());
            }
            _ => merged.push((message.role, message.content.clone(), message.images.clone())),
        }
    }
    merged
        .into_iter()
        .map(|(role, text, images)| {
            let role_name = match role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            };
            // Image blocks are only valid in user turns.
            if images.is_empty() || role == ChatRole::Assistant {
                return json!({"role": role_name, "content": text});
            }
            let mut blocks: Vec<Value> = images
                .iter()
                .map(|image| {
                    json!({
                        "type": "image",
                        "source": {"type": "base64", "media_type": image.mime_type, "data": image.data_base64}
                    })
                })
                .collect();
            if !text.is_empty() {
                blocks.push(json!({"type": "text", "text": text}));
            }
            json!({"role": role_name, "content": blocks})
        })
        .collect()
}

/// Builds the `/v1/messages` body. `thinking` is never sent.
fn request_body(model: &str, request: &LlmRequest, params: &Params) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("max_tokens".into(), json!(params.max_tokens));
    body.insert("stream".into(), json!(true));
    if !request.system.trim().is_empty() {
        body.insert("system".into(), json!(request.system));
    }
    body.insert("messages".into(), Value::Array(render_messages(request)));
    if let Some(t) = params.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(effort) = params.effort {
        body.insert("output_config".into(), json!({"effort": effort}));
    }
    if params.fallbacks {
        body.insert("fallbacks".into(), json!("default"));
    }
    Value::Object(body)
}

/// Maps an `error` object (SSE `error` event or JSON body).
fn api_error(error: &Value) -> ProviderError {
    let kind = error.get("type").and_then(Value::as_str).unwrap_or("");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(|m| sanitize_message(m, None))
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| kind.replace('_', " "));
    match kind {
        "authentication_error" | "permission_error" => ProviderError::Unauthorized(message),
        "rate_limit_error" => ProviderError::RateLimited(message),
        "timeout_error" => ProviderError::Timeout,
        "overloaded_error" => ProviderError::Http(format!("529 {message}")),
        "api_error" => ProviderError::Http(format!("500 {message}")),
        "request_too_large" => ProviderError::Http(format!("413 {message}")),
        "not_found_error" => ProviderError::Http(format!("404 {message}")),
        "billing_error" => ProviderError::Http(format!("402 {message}")),
        "invalid_request_error" => ProviderError::Http(format!("400 {message}")),
        _ => ProviderError::Other(format!("the provider reported an error: {message}")),
    }
}

fn usage_total_input(usage: &Value) -> Option<u32> {
    let get = |k: &str| usage.get(k).and_then(Value::as_u64);
    let input = get("input_tokens")?;
    let total = input
        + get("cache_creation_input_tokens").unwrap_or(0)
        + get("cache_read_input_tokens").unwrap_or(0);
    Some(u32::try_from(total).unwrap_or(u32::MAX))
}

/// Items for a non-streaming message body.
fn message_items(value: &Value) -> Vec<Result<LlmChunk, ProviderError>> {
    if let Some(error) = value.get("error") {
        return vec![Err(api_error(error))];
    }
    let text: String = value
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    let usage = value.get("usage").map(|u| TokenUsage {
        input_tokens: usage_total_input(u).unwrap_or(0),
        output_tokens: u
            .get("output_tokens")
            .and_then(Value::as_u64)
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
            .unwrap_or(0),
    });
    let mut items = Vec::new();
    if !text.is_empty() {
        items.push(Ok(LlmChunk::Delta(text)));
    }
    items.push(Ok(LlmChunk::Done {
        stop_reason: value
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(str::to_string),
        usage,
    }));
    items
}

/// Decodes the Messages SSE stream.
struct AnthropicDecoder {
    sse: SseParser,
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    stop_reason: Option<String>,
    done: bool,
}

impl AnthropicDecoder {
    fn new() -> Self {
        AnthropicDecoder {
            sse: SseParser::new(),
            input_tokens: None,
            output_tokens: None,
            stop_reason: None,
            done: false,
        }
    }

    fn finish_ok(&mut self, out: &mut ChunkQueue) {
        let usage = (self.input_tokens.is_some() || self.output_tokens.is_some()).then(|| {
            TokenUsage {
                input_tokens: self.input_tokens.unwrap_or(0),
                output_tokens: self.output_tokens.unwrap_or(0),
            }
        });
        out.push_back(Ok(LlmChunk::Done {
            stop_reason: self.stop_reason.clone(),
            usage,
        }));
        self.done = true;
    }

    fn handle(&mut self, event: Option<&str>, data: &str, out: &mut ChunkQueue) {
        if self.done || data.trim().is_empty() {
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return;
        };
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .or(event)
            .unwrap_or("");
        match kind {
            "message_start" => {
                let message = value.get("message");
                if let Some(usage) = message.and_then(|m| m.get("usage")) {
                    self.input_tokens = usage_total_input(usage).or(self.input_tokens);
                    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
                        self.output_tokens = Some(u32::try_from(n).unwrap_or(u32::MAX));
                    }
                }
                if let Some(served) = message.and_then(|m| m.get("model")).and_then(Value::as_str) {
                    tracing::debug!(provider = PROVIDER_ID, served_model = served, "message started");
                }
            }
            "content_block_start" => {
                let block = value.get("content_block");
                let block_type = block.and_then(|b| b.get("type")).and_then(Value::as_str);
                if block_type == Some("text") {
                    if let Some(text) = block.and_then(|b| b.get("text")).and_then(Value::as_str) {
                        if !text.is_empty() {
                            out.push_back(Ok(LlmChunk::Delta(text.to_string())));
                        }
                    }
                } else if block_type == Some("fallback") {
                    tracing::debug!(provider = PROVIDER_ID, "refusal fallback switched models");
                }
                // `thinking`, `redacted_thinking`, `fallback`, tool blocks: ignored.
            }
            "content_block_delta" => {
                let delta = value.get("delta");
                if delta.and_then(|d| d.get("type")).and_then(Value::as_str) == Some("text_delta") {
                    if let Some(text) = delta.and_then(|d| d.get("text")).and_then(Value::as_str) {
                        if !text.is_empty() {
                            out.push_back(Ok(LlmChunk::Delta(text.to_string())));
                        }
                    }
                }
                // `thinking_delta`, `signature_delta`, `input_json_delta`,
                // `citations_delta`: never surfaced.
            }
            "message_delta" => {
                if let Some(reason) = value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                {
                    self.stop_reason = Some(reason.to_string());
                    if reason == "refusal" {
                        let category = value
                            .get("delta")
                            .and_then(|d| d.get("stop_details"))
                            .and_then(|s| s.get("category"))
                            .and_then(Value::as_str)
                            .unwrap_or("none");
                        tracing::debug!(provider = PROVIDER_ID, category, "model declined the request");
                    }
                }
                if let Some(usage) = value.get("usage") {
                    if let Some(n) = usage.get("output_tokens").and_then(Value::as_u64) {
                        self.output_tokens = Some(u32::try_from(n).unwrap_or(u32::MAX));
                    }
                    if let Some(total) = usage_total_input(usage) {
                        self.input_tokens = Some(total);
                    }
                }
            }
            "message_stop" => self.finish_ok(out),
            "error" => {
                let error = value.get("error").cloned().unwrap_or(Value::Null);
                out.push_back(Err(api_error(&error)));
                self.done = true;
            }
            // `ping`, `content_block_stop` and unknown events.
            _ => {}
        }
    }
}

impl ChunkDecoder for AnthropicDecoder {
    fn decode(&mut self, bytes: &[u8], out: &mut ChunkQueue) {
        for event in self.sse.push(bytes) {
            self.handle(event.event.as_deref(), &event.data, out);
            if self.done {
                return;
            }
        }
    }

    fn finish(&mut self, out: &mut ChunkQueue) {
        for event in self.sse.finish() {
            self.handle(event.event.as_deref(), &event.data, out);
        }
        // A stop reason means the message completed even without
        // `message_stop`.
        if !self.done && self.stop_reason.is_some() {
            self.finish_ok(out);
        }
    }
}

fn log_failure(request_id: &str, error: &ProviderError) {
    if matches!(error, ProviderError::Cancelled) {
        tracing::debug!(provider = PROVIDER_ID, request_id, "request cancelled");
    } else {
        tracing::warn!(
            provider = PROVIDER_ID,
            request_id,
            error_kind = error_kind(error),
            "request failed"
        );
    }
}

/// Percent-encodes a query value (model ids are plain ASCII already).
fn encode_query(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[async_trait]
impl LlmProvider for AnthropicLlm {
    fn info(&self) -> ProviderInfo {
        provider_info(PROVIDER_ID, "Anthropic", &self.model, &self.base_url)
    }

    fn supports_images(&self) -> bool {
        true
    }

    async fn stream(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> Result<LlmStream, ProviderError> {
        if request.messages.is_empty() {
            return Err(ProviderError::Other("the request contains no messages".to_string()));
        }
        let started = Instant::now();
        let images: usize = request.messages.iter().map(|m| m.images.len()).sum();
        let mut params = self.params_for(&request);
        let mut attempt = 1;
        loop {
            tracing::debug!(
                provider = PROVIDER_ID,
                model = %self.model,
                request_id = %request.request_id,
                kind = ?request.kind,
                messages = request.messages.len(),
                images,
                max_tokens = params.max_tokens,
                effort = params.effort.unwrap_or("none"),
                fallbacks = params.fallbacks,
                attempt,
                "sending messages request"
            );
            let body = serde_json::to_vec(&request_body(&self.model, &request, &params))
                .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
            let mut builder = self
                .request(Method::POST, "/v1/messages")?
                .header(CONTENT_TYPE, "application/json")
                .header(ACCEPT, "text/event-stream")
                .body(body);
            if params.fallbacks {
                builder = builder.header("anthropic-beta", REFUSAL_FALLBACK_BETA);
            }
            let response = match send(builder, &cancel, self.timeouts.response, &self.host).await {
                Ok(r) => r,
                Err(e) => {
                    log_failure(&request.request_id, &e);
                    return Err(e);
                }
            };
            if response.status().is_success() {
                tracing::debug!(
                    provider = PROVIDER_ID,
                    request_id = %request.request_id,
                    headers_ms = started.elapsed().as_millis() as u64,
                    "messages stream opened"
                );
                return self
                    .open_stream(response, request.request_id.clone(), cancel, started)
                    .await;
            }
            let failure =
                HttpFailure::read(response, &cancel, self.api_key.as_deref(), &self.host).await;
            if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }
            if failure.status.as_u16() == 400 && attempt < MAX_ATTEMPTS {
                if let Some(next) = self.downgrade(&params, &failure.message_lower()) {
                    tracing::debug!(
                        provider = PROVIDER_ID,
                        request_id = %request.request_id,
                        attempt,
                        "request rejected; retrying with adjusted parameters"
                    );
                    params = next;
                    attempt += 1;
                    continue;
                }
            }
            let error = failure.into_error();
            log_failure(&request.request_id, &error);
            return Err(error);
        }
    }

    async fn health_check(&self) -> Result<(), ProviderError> {
        let cancel = CancellationToken::new();
        let builder = self.request(Method::GET, "/v1/models?limit=1")?;
        let response = send(builder, &cancel, self.timeouts.health, &self.host).await?;
        if response.status().is_success() {
            return Ok(());
        }
        Err(HttpFailure::read(response, &cancel, self.api_key.as_deref(), &self.host)
            .await
            .into_error())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let cancel = CancellationToken::new();
        let mut ids: Vec<String> = Vec::new();
        let mut after: Option<String> = None;
        for _ in 0..MAX_MODEL_PAGES {
            let mut path = "/v1/models?limit=1000".to_string();
            if let Some(after_id) = &after {
                path.push_str("&after_id=");
                path.push_str(&encode_query(after_id));
            }
            let builder = self.request(Method::GET, &path)?;
            let response = send(builder, &cancel, self.timeouts.health, &self.host).await?;
            if !response.status().is_success() {
                return Err(HttpFailure::read(response, &cancel, self.api_key.as_deref(), &self.host)
                    .await
                    .into_error());
            }
            let page: Value =
                read_json(response, "model list", &cancel, self.timeouts.health, &self.host).await?;
            let data = page.get("data").and_then(Value::as_array).ok_or_else(|| {
                ProviderError::InvalidResponse("the model list has no `data` array".to_string())
            })?;
            for id in data.iter().filter_map(|m| m.get("id").and_then(Value::as_str)) {
                if !ids.iter().any(|existing| existing == id) {
                    ids.push(id.to_string());
                }
            }
            let has_more = page.get("has_more").and_then(Value::as_bool).unwrap_or(false);
            after = page.get("last_id").and_then(Value::as_str).map(str::to_string);
            if !has_more || after.is_none() {
                break;
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::providers::ChatMessage;
    use openmeet_core::types::RequestKind;

    #[test]
    fn model_id_parsing() {
        let p = |m: &str| parse_model_id(m).map(|id| (id.family, id.version));
        assert_eq!(p("claude-opus-5-5"), Some(("opus".into(), Some((5, 5)))));
        assert_eq!(p("claude-opus-5"), Some(("opus".into(), Some((5, 0)))));
        assert_eq!(p("claude-sonnet-4-5-20250929"), Some(("sonnet".into(), Some((4, 5)))));
        assert_eq!(p("claude-opus-4-20250514"), Some(("opus".into(), Some((4, 0)))));
        assert_eq!(p("claude-3-7-sonnet-latest"), Some(("sonnet".into(), Some((3, 7)))));
        assert_eq!(p("claude-3-haiku-20240307"), Some(("haiku".into(), Some((3, 0)))));
        assert_eq!(p("claude-mythos-preview"), Some(("mythos".into(), None)));
        assert_eq!(p("anthropic/claude-haiku-4-5"), Some(("haiku".into(), Some((4, 5)))));
        assert_eq!(p("claude-opus-4-5@20251101"), Some(("opus".into(), Some((4, 5)))));
        assert_eq!(p("gpt-4o"), None);
    }

    #[test]
    fn sampling_rules() {
        for m in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-sonnet-5-5",
            "claude-fable-5",
            "claude-fable-5-1",
            "claude-mythos-5-1",
            "something-else",
        ] {
            assert!(!ModelRules::for_model(m).sampling, "{m}");
        }
        for m in [
            "claude-haiku-4-5",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-3-5-sonnet-20241022",
        ] {
            assert!(ModelRules::for_model(m).sampling, "{m}");
        }
    }

    #[test]
    fn effort_rules() {
        for m in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-5-5",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-sonnet-5-5",
            "claude-fable-5-1",
        ] {
            assert!(ModelRules::for_model(m).effort, "{m}");
        }
        for m in ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-3-7-sonnet-latest"] {
            assert!(!ModelRules::for_model(m).effort, "{m}");
        }
    }

    #[test]
    fn headroom_and_fallback_rules() {
        for m in ["claude-opus-5", "claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-fable-5-1", "claude-mythos-5-1"] {
            assert_eq!(ModelRules::for_model(m).thinking_headroom, THINKING_HEADROOM_TOKENS, "{m}");
        }
        for m in ["claude-opus-4-8", "claude-opus-4-6", "claude-haiku-4-5"] {
            assert_eq!(ModelRules::for_model(m).thinking_headroom, 0, "{m}");
        }
        for m in ["claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1", "claude-sonnet-5-5"] {
            assert!(ModelRules::for_model(m).refusal_fallbacks, "{m}");
        }
        for m in ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5", "claude-mythos-5-1"] {
            assert!(!ModelRules::for_model(m).refusal_fallbacks, "{m}");
        }
    }

    fn llm(model: &str) -> AnthropicLlm {
        AnthropicLlm::new(
            Endpoint::new("https://api.anthropic.com").with_api_key(Some("k".into())),
            model,
        )
        .unwrap()
    }

    fn request(latency: bool) -> LlmRequest {
        let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "sys", "hi");
        r.temperature = Some(1.7);
        r.max_output_tokens = 1000;
        r.latency_sensitive = latency;
        r
    }

    #[test]
    fn params_and_body() {
        let opus = llm("claude-opus-5-5");
        let p = opus.params_for(&request(true));
        assert_eq!(p, Params { max_tokens: 5000, temperature: None, effort: Some("low"), fallbacks: true });
        let body = request_body("claude-opus-5-5", &request(true), &p);
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["output_config"]["effort"], "low");
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
        assert_eq!(body["system"], "sys");

        let haiku = llm("claude-haiku-4-5").with_reasoning_effort(ReasoningEffort::High);
        let p = haiku.params_for(&request(false));
        assert_eq!(p, Params { max_tokens: 1000, temperature: Some(1.0), effort: None, fallbacks: false });

        let sonnet46 = llm("claude-sonnet-4-6").with_reasoning_effort(ReasoningEffort::Medium);
        let p = sonnet46.params_for(&request(false));
        assert_eq!(p.effort, Some("medium"));
        assert!(p.temperature.is_some());
        assert_eq!(llm("").model(), DEFAULT_MODEL);
    }

    #[test]
    fn messages_merge_and_images_first() {
        let mut r = request(true);
        r.messages = vec![
            ChatMessage::user("a"),
            ChatMessage::user("b").with_image(ImageInput {
                mime_type: "image/png".into(),
                data_base64: "QUJD".into(),
            }),
            ChatMessage::assistant("c"),
            ChatMessage::user("d"),
        ];
        let msgs = render_messages(&r);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["content"][0]["type"], "image");
        assert_eq!(msgs[0]["content"][0]["source"]["media_type"], "image/png");
        assert_eq!(msgs[0]["content"][1]["text"], "a\n\nb");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "d");
    }

    #[test]
    fn downgrades() {
        let l = llm("claude-opus-5-5");
        let p = l.params_for(&request(true));
        let next = l.downgrade(&p, "fallbacks: extra inputs are not permitted").unwrap();
        assert!(!next.fallbacks);
        assert!(!l.params_for(&request(true)).fallbacks);
        let next = l.downgrade(&next, "output_config.effort: unexpected value").unwrap();
        assert_eq!(next.effort, None);
        let next = l
            .downgrade(&next, "max_tokens: 5000 > 4096, which is the maximum allowed number of output tokens")
            .unwrap();
        assert_eq!(next.max_tokens, 4096);
        assert!(l.downgrade(&next, "messages: roles must alternate").is_none());
    }

    #[test]
    fn error_events() {
        assert!(matches!(
            api_error(&json!({"type": "overloaded_error", "message": "Overloaded"})),
            ProviderError::Http(m) if m == "529 Overloaded"
        ));
        assert!(matches!(
            api_error(&json!({"type": "rate_limit_error", "message": "slow"})),
            ProviderError::RateLimited(_)
        ));
        assert!(matches!(
            api_error(&json!({"type": "authentication_error", "message": "bad key"})),
            ProviderError::Unauthorized(_)
        ));
    }

    #[test]
    fn query_encoding() {
        assert_eq!(encode_query("claude-opus-5"), "claude-opus-5");
        assert_eq!(encode_query("a b&c"), "a%20b%26c");
    }
}
