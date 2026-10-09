//! Ollama adapters: [`OllamaLlm`] (streaming `POST /api/chat`, NDJSON) and
//! [`OllamaEmbeddings`] (`POST /api/embed`).
//!
//! Ollama usually runs on this device (`http://127.0.0.1:11434`), but a LAN
//! host is reported — and treated by local-only mode — as remote, because
//! meeting content then leaves the device.
//!
//! Reasoning is never surfaced: `message.thinking` is ignored and an inline
//! `<think>…</think>` block at the start of the output is removed. For
//! latency-sensitive requests `think: false` is sent (`"low"` for `gpt-oss`,
//! which cannot disable thinking); if a server rejects the field it is dropped
//! and the adjustment remembered.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use openmeet_core::ProviderError;
use openmeet_core::providers::{
    ChatRole, EmbeddingProvider, LlmChunk, LlmProvider, LlmRequest, LlmStream, TokenUsage,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::ProviderInfo;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Method, RequestBuilder, StatusCode};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{
    Endpoint, HttpFailure, Timeouts, build_client, error_kind, provider_info, read_json, root_url,
    sanitize_message, send, sensitive_header, validate_endpoint_url,
};
use crate::lines::LineSplitter;
use crate::openai::check_dimensions;
use crate::stream::{ChunkDecoder, ChunkQueue, StreamMeta, into_llm_stream};
use crate::think::ThinkFilter;

const PROVIDER_ID: &str = "ollama";
const DISPLAY_NAME: &str = "Ollama";
/// Extra `num_predict` when the model may think (thinking tokens count
/// against the limit).
pub const THINKING_HEADROOM_TOKENS: u32 = 2000;
/// Texts per `/api/embed` request.
pub const EMBEDDING_BATCH_SIZE: usize = 64;
const MAX_ATTEMPTS: usize = 3;

/// Service root for Ollama: trailing `/v1` (OpenAI-compatible path) or `/api`
/// are removed (`http://host:11434/v1` → `http://host:11434`).
pub fn ollama_root(base_url: &str) -> String {
    let root = root_url(base_url);
    if root.to_ascii_lowercase().ends_with("/api") {
        root[..root.len() - 4].to_string()
    } else {
        root
    }
}

/// Best-effort image-input support from an Ollama model name.
pub fn model_supports_images(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    let name = m.rsplit_once('/').map(|(_, n)| n).unwrap_or(&m);
    const MARKERS: [&str; 20] = [
        "vision",
        "llava",
        "bakllava",
        "gemma3",
        "gemma-3",
        "qwen2.5vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "qwen3vl",
        "minicpm-v",
        "moondream",
        "llama4",
        "mistral-small3.1",
        "mistral-small3.2",
        "granite3.2-vision",
        "internvl",
        "-vl",
        "pixtral",
        "smolvlm",
        "glm-4v",
    ];
    MARKERS.iter().any(|marker| name.contains(marker))
}

fn is_gpt_oss(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    let name = m.rsplit_once('/').map(|(_, n)| n).unwrap_or(&m);
    name.starts_with("gpt-oss")
}

/// Value of the `think` field for a request, `None` to omit it.
fn think_value(model: &str, latency_sensitive: bool, effort: ReasoningEffort) -> Option<Value> {
    if is_gpt_oss(model) {
        // gpt-oss cannot turn thinking off; it takes an effort level.
        let level = if latency_sensitive {
            "low"
        } else {
            effort.as_str()
        };
        return Some(json!(level));
    }
    if latency_sensitive || effort == ReasoningEffort::Low {
        Some(json!(false))
    } else {
        None
    }
}

struct Base {
    client: reqwest::Client,
    base_url: String,
    root: String,
    host: String,
    api_key: Option<String>,
    timeouts: Timeouts,
}

impl Base {
    fn new(endpoint: Endpoint) -> Result<Self, ProviderError> {
        validate_endpoint_url(&endpoint.base_url, "Ollama")?;
        if let Some(key) = &endpoint.api_key {
            sensitive_header(&format!("Bearer {key}"))?;
        }
        Ok(Base {
            client: build_client(endpoint.location(), endpoint.timeouts.connect)?,
            root: ollama_root(&endpoint.base_url),
            host: endpoint.host(),
            base_url: endpoint.base_url,
            api_key: endpoint.api_key,
            timeouts: endpoint.timeouts,
        })
    }

    fn request(&self, method: Method, path: &str) -> Result<RequestBuilder, ProviderError> {
        let mut builder = self
            .client
            .request(method, format!("{}{}", self.root, path));
        if let Some(key) = &self.api_key {
            builder = builder.header(AUTHORIZATION, sensitive_header(&format!("Bearer {key}"))?);
        }
        Ok(builder)
    }

    async fn failure(
        &self,
        response: reqwest::Response,
        cancel: &CancellationToken,
    ) -> HttpFailure {
        HttpFailure::read(response, cancel, self.api_key.as_deref(), &self.host).await
    }

    /// `GET /api/tags` → model names.
    async fn tags(&self, cancel: &CancellationToken) -> Result<Vec<String>, ProviderError> {
        let builder = self
            .request(Method::GET, "/api/tags")?
            .header(ACCEPT, "application/json");
        let response = send(builder, cancel, self.timeouts.health, &self.host).await?;
        if !response.status().is_success() {
            return Err(self.failure(response, cancel).await.into_error());
        }
        let value: Value = read_json(
            response,
            "model list",
            cancel,
            self.timeouts.health,
            &self.host,
        )
        .await?;
        let models = value
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ProviderError::InvalidResponse("the model list has no `models` array".to_string())
            })?;
        let mut names: Vec<String> = models
            .iter()
            .filter_map(|m| m.get("name").or_else(|| m.get("model")))
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct Params {
    num_predict: u32,
    temperature: Option<f32>,
    think: Option<Value>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Quirks {
    no_think: bool,
}

/// Streaming chat against an Ollama server.
pub struct OllamaLlm {
    base: Base,
    model: String,
    reasoning_effort: ReasoningEffort,
    supports_images: bool,
    quirks: Mutex<Quirks>,
}

impl OllamaLlm {
    /// Creates the adapter. Fails for an invalid base URL or an empty model.
    pub fn new(endpoint: Endpoint, model: impl Into<String>) -> Result<Self, ProviderError> {
        let model = model.into().trim().to_string();
        if model.is_empty() {
            return Err(ProviderError::NotConfigured(
                "LLM model is not set".to_string(),
            ));
        }
        Ok(OllamaLlm {
            base: Base::new(endpoint)?,
            supports_images: model_supports_images(&model),
            model,
            reasoning_effort: ReasoningEffort::Low,
            quirks: Mutex::new(Quirks::default()),
        })
    }

    /// Effort for requests that are not latency-sensitive: `low` disables
    /// thinking, higher levels leave it to the model.
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

    fn quirks(&self) -> MutexGuard<'_, Quirks> {
        self.quirks.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn params_for(&self, request: &LlmRequest) -> Params {
        let quirks = *self.quirks();
        let think = if quirks.no_think {
            None
        } else {
            think_value(
                &self.model,
                request.latency_sensitive,
                self.reasoning_effort,
            )
        };
        // Thinking tokens count against `num_predict`: reserve room whenever
        // the model may think.
        let may_think = think.as_ref() != Some(&json!(false));
        let mut num_predict = request.max_output_tokens.max(1);
        if may_think {
            num_predict = num_predict.saturating_add(THINKING_HEADROOM_TOKENS);
        }
        Params {
            num_predict,
            temperature: request
                .temperature
                .filter(|t| t.is_finite())
                .map(|t| t.clamp(0.0, 2.0)),
            think,
        }
    }

    fn downgrade(&self, params: &Params, message: &str) -> Option<Params> {
        if params.think.is_some() && message.contains("think") {
            self.quirks().no_think = true;
            // The server rejected the field because the model cannot think,
            // so no thinking headroom is needed after dropping it.
            let mut next = params.clone();
            next.think = None;
            return Some(next);
        }
        None
    }
}

/// Builds the `/api/chat` body.
fn chat_body(model: &str, request: &LlmRequest, params: &Params) -> Value {
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if !request.system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    for message in &request.messages {
        let mut m = Map::new();
        let role = match message.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        m.insert("role".into(), json!(role));
        m.insert("content".into(), json!(message.content));
        if message.role == ChatRole::User && !message.images.is_empty() {
            let images: Vec<&str> = message
                .images
                .iter()
                .map(|i| i.data_base64.as_str())
                .collect();
            m.insert("images".into(), json!(images));
        }
        messages.push(Value::Object(m));
    }
    let mut options = Map::new();
    options.insert("num_predict".into(), json!(params.num_predict));
    if let Some(t) = params.temperature {
        options.insert("temperature".into(), json!(t));
    }
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), json!(true));
    body.insert("options".into(), Value::Object(options));
    if let Some(think) = &params.think {
        body.insert("think".into(), think.clone());
    }
    Value::Object(body)
}

fn count(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
}

/// Decodes the `/api/chat` NDJSON stream.
struct OllamaDecoder {
    lines: LineSplitter,
    think: ThinkFilter,
    done: bool,
    skipped_lines: usize,
}

impl OllamaDecoder {
    fn new() -> Self {
        OllamaDecoder {
            lines: LineSplitter::new(),
            think: ThinkFilter::new(),
            done: false,
            skipped_lines: 0,
        }
    }

    fn handle(&mut self, line: &str, out: &mut ChunkQueue) {
        if self.done {
            return;
        }
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            self.skipped_lines += 1;
            return;
        };
        if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
            let message = error
                .as_str()
                .or_else(|| error.get("message").and_then(Value::as_str))
                .map(|m| sanitize_message(m, None))
                .unwrap_or_else(|| "unknown error".to_string());
            out.push_back(Err(ProviderError::Other(format!(
                "the provider reported an error: {message}"
            ))));
            self.done = true;
            return;
        }
        // Only `message.content` is visible; `message.thinking` is ignored.
        if let Some(text) = value
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
        {
            let visible = self.think.push(text);
            if !visible.is_empty() {
                out.push_back(Ok(LlmChunk::Delta(visible)));
            }
        }
        if value.get("done").and_then(Value::as_bool) == Some(true) {
            let rest = self.think.finish();
            if !rest.is_empty() {
                out.push_back(Ok(LlmChunk::Delta(rest)));
            }
            let input = count(&value, "prompt_eval_count");
            let output = count(&value, "eval_count");
            let usage = (input.is_some() || output.is_some()).then(|| TokenUsage {
                input_tokens: input.unwrap_or(0),
                output_tokens: output.unwrap_or(0),
            });
            out.push_back(Ok(LlmChunk::Done {
                stop_reason: value
                    .get("done_reason")
                    .and_then(Value::as_str)
                    .filter(|r| !r.is_empty())
                    .map(str::to_string),
                usage,
            }));
            self.done = true;
        }
    }
}

impl ChunkDecoder for OllamaDecoder {
    fn decode(&mut self, bytes: &[u8], out: &mut ChunkQueue) {
        for line in self.lines.push(bytes) {
            self.handle(&line, out);
            if self.done {
                return;
            }
        }
    }

    fn finish(&mut self, out: &mut ChunkQueue) {
        if let Some(line) = self.lines.finish() {
            self.handle(&line, out);
        }
        if self.skipped_lines > 0 {
            tracing::debug!(
                provider = PROVIDER_ID,
                count = self.skipped_lines,
                "skipped malformed stream lines"
            );
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

#[async_trait]
impl LlmProvider for OllamaLlm {
    fn info(&self) -> ProviderInfo {
        provider_info(PROVIDER_ID, DISPLAY_NAME, &self.model, &self.base.base_url)
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
                num_predict = params.num_predict,
                attempt,
                "sending chat request"
            );
            let body = serde_json::to_vec(&chat_body(&self.model, &request, &params))
                .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
            let builder = self
                .base
                .request(Method::POST, "/api/chat")?
                .header(CONTENT_TYPE, "application/json")
                .header(ACCEPT, "application/x-ndjson")
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
                    log_failure(&request.request_id, &e);
                    return Err(e);
                }
            };
            if response.status().is_success() {
                tracing::debug!(
                    provider = PROVIDER_ID,
                    request_id = %request.request_id,
                    headers_ms = started.elapsed().as_millis() as u64,
                    "chat stream opened"
                );
                let meta = StreamMeta {
                    provider: PROVIDER_ID,
                    request_id: request.request_id.clone(),
                    host: self.base.host.clone(),
                    started,
                };
                return Ok(into_llm_stream(
                    response,
                    OllamaDecoder::new(),
                    cancel,
                    self.base.timeouts.stream_idle,
                    meta,
                ));
            }
            let failure = self.base.failure(response, &cancel).await;
            if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }
            if failure.status == StatusCode::BAD_REQUEST && attempt < MAX_ATTEMPTS {
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
        self.base.tags(&CancellationToken::new()).await.map(|_| ())
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        self.base.tags(&CancellationToken::new()).await
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// Embeddings through `POST /api/embed` (falling back to the legacy
/// `POST /api/embeddings` on servers that predate it).
pub struct OllamaEmbeddings {
    base: Base,
    model: String,
    legacy: Mutex<bool>,
}

impl OllamaEmbeddings {
    /// Creates the adapter. Fails for an invalid base URL or an empty model.
    pub fn new(endpoint: Endpoint, model: impl Into<String>) -> Result<Self, ProviderError> {
        let model = model.into().trim().to_string();
        if model.is_empty() {
            return Err(ProviderError::NotConfigured(
                "embedding model is not set".to_string(),
            ));
        }
        Ok(OllamaEmbeddings {
            base: Base::new(endpoint)?,
            model,
            legacy: Mutex::new(false),
        })
    }

    /// Configured model.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn uses_legacy(&self) -> bool {
        *self.legacy.lock().unwrap_or_else(|e| e.into_inner())
    }

    async fn post_json(
        &self,
        path: &str,
        body: &Value,
        cancel: &CancellationToken,
    ) -> Result<Result<Value, HttpFailure>, ProviderError> {
        let bytes = serde_json::to_vec(body)
            .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
        let builder = self
            .base
            .request(Method::POST, path)?
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json")
            .body(bytes);
        let started = Instant::now();
        let deadline = self.base.timeouts.request;
        let response = send(builder, cancel, deadline, &self.base.host).await?;
        if !response.status().is_success() {
            return Ok(Err(self.base.failure(response, cancel).await));
        }
        let remaining = deadline
            .saturating_sub(started.elapsed())
            .max(Duration::from_secs(1));
        let value = read_json(response, "embeddings", cancel, remaining, &self.base.host).await?;
        Ok(Ok(value))
    }

    async fn embed_batch(
        &self,
        batch: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        if self.uses_legacy() {
            return self.embed_legacy(batch, cancel).await;
        }
        let body = json!({"model": self.model, "input": batch});
        match self.post_json("/api/embed", &body, cancel).await? {
            Ok(value) => {
                let items = value
                    .get("embeddings")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        ProviderError::InvalidResponse(
                            "the embeddings response has no `embeddings` array".to_string(),
                        )
                    })?;
                let vectors = items
                    .iter()
                    .map(parse_vector)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        ProviderError::InvalidResponse(
                            "an embedding in the response is malformed".to_string(),
                        )
                    })?;
                if vectors.len() != batch.len() {
                    return Err(ProviderError::InvalidResponse(format!(
                        "expected {} embeddings, received {}",
                        batch.len(),
                        vectors.len()
                    )));
                }
                Ok(vectors)
            }
            // Servers older than /api/embed answer with a plain-text 404
            // (a missing model is a JSON 404 with an error message).
            Err(failure)
                if failure.status == StatusCode::NOT_FOUND && failure.message.is_none() =>
            {
                tracing::debug!(
                    provider = PROVIDER_ID,
                    "/api/embed unavailable; using the legacy embeddings endpoint"
                );
                *self.legacy.lock().unwrap_or_else(|e| e.into_inner()) = true;
                self.embed_legacy(batch, cancel).await
            }
            Err(failure) => Err(failure.into_error()),
        }
    }

    async fn embed_legacy(
        &self,
        batch: &[String],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let mut vectors = Vec::with_capacity(batch.len());
        for text in batch {
            let body = json!({"model": self.model, "prompt": text});
            let value = self
                .post_json("/api/embeddings", &body, cancel)
                .await?
                .map_err(HttpFailure::into_error)?;
            let vector = value
                .get("embedding")
                .and_then(parse_vector)
                .ok_or_else(|| {
                    ProviderError::InvalidResponse(
                        "the embeddings response has no `embedding` array".to_string(),
                    )
                })?;
            vectors.push(vector);
        }
        Ok(vectors)
    }
}

fn parse_vector(value: &Value) -> Option<Vec<f32>> {
    let vector = value
        .as_array()?
        .iter()
        .map(|x| x.as_f64().map(|f| f as f32))
        .collect::<Option<Vec<f32>>>()?;
    (!vector.is_empty() && vector.iter().all(|x| x.is_finite())).then_some(vector)
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbeddings {
    fn info(&self) -> ProviderInfo {
        provider_info(PROVIDER_ID, DISPLAY_NAME, &self.model, &self.base.base_url)
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
                    provider = PROVIDER_ID,
                    texts = texts.len(),
                    error_kind = error_kind(e),
                    "embedding request failed"
                )
            })?;
            vectors.extend(batch_vectors);
        }
        check_dimensions(&vectors)?;
        tracing::debug!(
            provider = PROVIDER_ID,
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

    fn llm(model: &str) -> OllamaLlm {
        OllamaLlm::new(Endpoint::new("http://127.0.0.1:11434"), model).unwrap()
    }

    fn request(latency: bool) -> LlmRequest {
        let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "sys", "hi");
        r.temperature = Some(0.3);
        r.max_output_tokens = 500;
        r.latency_sensitive = latency;
        r
    }

    #[test]
    fn roots() {
        assert_eq!(
            ollama_root("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            ollama_root("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            ollama_root("http://127.0.0.1:11434/api"),
            "http://127.0.0.1:11434"
        );
        assert_eq!(
            ollama_root("http://h/proxy/ollama"),
            "http://h/proxy/ollama"
        );
    }

    #[test]
    fn vision_guess() {
        for m in [
            "llava:13b",
            "llama3.2-vision",
            "gemma3:4b",
            "qwen2.5vl:7b",
            "minicpm-v",
        ] {
            assert!(model_supports_images(m), "{m}");
        }
        for m in ["llama3.2", "qwen2.5:7b", "mistral", "deepseek-r1:8b"] {
            assert!(!model_supports_images(m), "{m}");
        }
    }

    #[test]
    fn think_rules() {
        let l = llm("qwen3:8b");
        let p = l.params_for(&request(true));
        assert_eq!(p.think, Some(json!(false)));
        assert_eq!(p.num_predict, 500);
        let l = llm("qwen3:8b").with_reasoning_effort(ReasoningEffort::High);
        let p = l.params_for(&request(false));
        assert_eq!(p.think, None);
        assert_eq!(p.num_predict, 500 + THINKING_HEADROOM_TOKENS);
        let g = llm("gpt-oss:20b").with_reasoning_effort(ReasoningEffort::Medium);
        assert_eq!(g.params_for(&request(true)).think, Some(json!("low")));
        assert_eq!(g.params_for(&request(false)).think, Some(json!("medium")));

        let l = llm("llama3.2");
        let p = l.params_for(&request(true));
        let next = l
            .downgrade(&p, "\"llama3.2\" does not support thinking")
            .unwrap();
        assert_eq!(next.think, None);
        assert_eq!(l.params_for(&request(true)).think, None);
        assert!(l.downgrade(&next, "does not support thinking").is_none());
    }

    #[test]
    fn body_shape() {
        let l = llm("llava");
        let mut r = request(true);
        r.messages = vec![ChatMessage::user("what is this").with_image(ImageInput {
            mime_type: "image/png".into(),
            data_base64: "QUJD".into(),
        })];
        let body = chat_body("llava", &r, &l.params_for(&r));
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["images"][0], "QUJD");
        assert_eq!(body["options"]["num_predict"], 500);
        assert!((body["options"]["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(body["think"], false);
    }

    #[test]
    fn decoder_any_split() {
        let input = concat!(
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"hmm\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"<think>x</think>Hé\"},\"done\":false}\n",
            "\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"llo\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":12,\"eval_count\":3}"
        )
        .as_bytes();
        for split in 0..=input.len() {
            let mut d = OllamaDecoder::new();
            let mut q = ChunkQueue::new();
            d.decode(&input[..split], &mut q);
            d.decode(&input[split..], &mut q);
            d.finish(&mut q);
            let items: Vec<LlmChunk> = q.into_iter().map(Result::unwrap).collect();
            let text: String = items
                .iter()
                .filter_map(|c| match c {
                    LlmChunk::Delta(t) => Some(t.as_str()),
                    LlmChunk::Done { .. } => None,
                })
                .collect();
            assert_eq!(text, "Héllo", "split {split}");
            assert_eq!(
                items.last().unwrap(),
                &LlmChunk::Done {
                    stop_reason: Some("stop".into()),
                    usage: Some(TokenUsage {
                        input_tokens: 12,
                        output_tokens: 3
                    }),
                },
                "split {split}"
            );
        }
    }

    #[test]
    fn decoder_error_line() {
        let mut d = OllamaDecoder::new();
        let mut q = ChunkQueue::new();
        d.decode(b"{\"error\":\"out of memory\"}\n", &mut q);
        assert!(
            matches!(q.pop_front(), Some(Err(ProviderError::Other(m))) if m.contains("out of memory"))
        );
    }

    #[test]
    fn vectors() {
        assert_eq!(parse_vector(&json!([1.0, 2.5])), Some(vec![1.0, 2.5]));
        assert_eq!(parse_vector(&json!([])), None);
        assert_eq!(parse_vector(&json!("x")), None);
    }
}
