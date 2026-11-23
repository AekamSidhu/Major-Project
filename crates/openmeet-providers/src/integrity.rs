//! Client for the local AI sidecar's answer-integrity endpoints
//! (`local-ai/integrity`), implementing [`IntegrityService`]:
//!
//! | Call | Endpoint |
//! |---|---|
//! | [`IntegrityService::status`] | `GET /v1/integrity/status` (+ `GET /health` for the embedding model) |
//! | [`IntegrityService::ai_text`] | `POST /v1/integrity/echo` `{"text"}` |
//! | [`IntegrityService::start_gaze`] | `POST /v1/integrity/gaze/screen` `{"left","top","width","height"}` |
//! | [`IntegrityService::stop_gaze`] | `POST /v1/integrity/gaze/stop` |
//! | [`IntegrityService::reset_gaze`] | `POST /v1/integrity/gaze/reset` |
//! | [`IntegrityService::reading`] | `POST /v1/integrity/reading` `{"start","end"}` (epoch seconds) |
//!
//! Requests carry no meeting content in logs or errors: only the endpoint
//! name, sizes and error kinds are logged.

use std::time::Duration;

use async_trait::async_trait;
use openmeet_core::ProviderError;
use openmeet_core::integrity::{
    AiTextResult, GazeStatus, IntegrityLevel, IntegrityService, IntegrityServiceStatus,
    ReadingResult,
};
use openmeet_core::ipc::Rect;
use openmeet_core::types::ProviderInfo;
use reqwest::Method;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::http::{
    Endpoint, HEALTH_CHECK_TIMEOUT, HttpFailure, build_client, error_kind, provider_info,
    read_json, root_url, send, validate_endpoint_url,
};

/// Display name of the sidecar's integrity service.
pub const LOCAL_AI_INTEGRITY_NAME: &str = "Local AI (integrity check)";
/// Longest transcript the sidecar accepts (`MAX_TEXT_CHARS`).
pub const MAX_AI_TEXT_CHARS: usize = 20_000;
/// Longest window the reading endpoint accepts (seconds).
pub const MAX_READING_SPAN_S: f64 = 600.0;
/// Timeout of a scoring request (the reading model runs on the CPU).
pub const SCORING_TIMEOUT: Duration = Duration::from_secs(30);

/// The sidecar's `/v1/integrity` endpoints.
#[derive(Debug, Clone)]
pub struct LocalAiIntegrity {
    root: String,
    base_url: String,
    host: String,
    client: reqwest::Client,
}

impl LocalAiIntegrity {
    /// Client for the sidecar at `endpoint` (with or without `/v1`).
    pub fn new(endpoint: Endpoint) -> Result<Self, ProviderError> {
        validate_endpoint_url(&endpoint.base_url, "Local AI")?;
        let client = build_client(endpoint.location(), endpoint.timeouts.connect)?;
        Ok(LocalAiIntegrity {
            root: root_url(&endpoint.base_url),
            host: endpoint.host(),
            base_url: endpoint.base_url,
            client,
        })
    }

    async fn call(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
        timeout: Duration,
        cancel: &CancellationToken,
    ) -> Result<Value, ProviderError> {
        let mut builder = self
            .client
            .request(method, format!("{}{path}", self.root))
            .header(ACCEPT, "application/json");
        if let Some(body) = body {
            let bytes = serde_json::to_vec(&body)
                .map_err(|_| ProviderError::Other("could not encode the request".to_string()))?;
            builder = builder.header(CONTENT_TYPE, "application/json").body(bytes);
        }
        let result = async {
            let response = send(builder, cancel, timeout, &self.host).await?;
            if !response.status().is_success() {
                return Err(HttpFailure::read(response, cancel, None, &self.host)
                    .await
                    .into_error());
            }
            read_json::<Value>(response, "integrity", cancel, timeout, &self.host).await
        }
        .await;
        if let Err(e) = &result
            && !matches!(e, ProviderError::Cancelled)
        {
            tracing::debug!(
                endpoint = path,
                error_kind = error_kind(e),
                "integrity request failed"
            );
        }
        result
    }

    async fn gaze_call(
        &self,
        path: &str,
        body: Option<Value>,
        cancel: &CancellationToken,
    ) -> Result<GazeStatus, ProviderError> {
        let value = self
            .call(
                Method::POST,
                path,
                body.or(Some(json!({}))),
                HEALTH_CHECK_TIMEOUT,
                cancel,
            )
            .await?;
        Ok(parse_gaze(&value))
    }
}

fn opt_text(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

/// Parses the sidecar's gaze status object.
pub fn parse_gaze(value: &Value) -> GazeStatus {
    GazeStatus {
        source: opt_text(value, "source").unwrap_or_else(|| "none".to_string()),
        capturing_screen: value
            .get("capturing_screen")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        frames_per_second: number(value, "frames_per_second").unwrap_or(0.0),
        face_found_fraction: number(value, "face_found_fraction").unwrap_or(0.0),
        tracked_fraction: number(value, "tracked_fraction").unwrap_or(0.0),
        error: opt_text(value, "error"),
    }
}

/// Maps a sidecar level name; `not_enough_gaze` is the spec's `no_gaze`.
pub fn parse_sidecar_level(level: &str) -> Option<IntegrityLevel> {
    match level.trim() {
        "not_enough_gaze" => Some(IntegrityLevel::NoGaze),
        other => IntegrityLevel::parse(other),
    }
}

/// Parses `POST /v1/integrity/echo`.
pub fn parse_ai_text(value: &Value) -> Result<AiTextResult, ProviderError> {
    let level = value
        .get("level")
        .and_then(Value::as_str)
        .and_then(parse_sidecar_level)
        .filter(|l| l.is_scored() || *l == IntegrityLevel::TooShort)
        .ok_or_else(|| {
            ProviderError::InvalidResponse("the AI-text response has no valid level".to_string())
        })?;
    let probability = number(value, "ai_probability").ok_or_else(|| {
        ProviderError::InvalidResponse("the AI-text response has no probability".to_string())
    })?;
    Ok(AiTextResult {
        probability: probability.clamp(0.0, 1.0),
        level,
        words: value.get("words").and_then(Value::as_u64).unwrap_or(0) as u32,
        model: opt_text(value, "model").unwrap_or_default(),
    })
}

/// Parses `POST /v1/integrity/reading`.
pub fn parse_reading(value: &Value) -> Result<ReadingResult, ProviderError> {
    let level = value
        .get("level")
        .and_then(Value::as_str)
        .and_then(parse_sidecar_level)
        .filter(|l| l.is_scored() || *l == IntegrityLevel::NoGaze)
        .ok_or_else(|| {
            ProviderError::InvalidResponse("the reading response has no valid level".to_string())
        })?;
    let probability = number(value, "reading_probability").map(|p| p.clamp(0.0, 1.0));
    if level.is_scored() && probability.is_none() {
        return Err(ProviderError::InvalidResponse(
            "the reading response has a level but no probability".to_string(),
        ));
    }
    Ok(ReadingResult {
        probability,
        level,
        valid_fraction: number(value, "valid_fraction"),
        windows: value
            .get("window_probabilities")
            .and_then(Value::as_array)
            .map_or(0, |w| w.len() as u32),
        seconds: number(value, "seconds").unwrap_or(0.0),
        calibrated: value
            .get("calibrated")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

/// Parses `GET /v1/integrity/status`.
pub fn parse_status(value: &Value) -> IntegrityServiceStatus {
    IntegrityServiceStatus {
        ai_text_ready: value
            .get("echo_ready")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        reading_ready: value
            .get("reading_ready")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        ai_text_error: opt_text(value, "echo_error"),
        reading_error: opt_text(value, "reading_error"),
        gaze: value.get("gaze").map(parse_gaze).unwrap_or_default(),
        embedding_model: None,
    }
}

/// Characters of `text` the sidecar accepts (the end of a very long answer
/// is dropped).
fn bounded_text(text: &str) -> String {
    if text.chars().count() <= MAX_AI_TEXT_CHARS {
        text.to_string()
    } else {
        text.chars().take(MAX_AI_TEXT_CHARS).collect()
    }
}

#[async_trait]
impl IntegrityService for LocalAiIntegrity {
    fn info(&self) -> ProviderInfo {
        provider_info(
            "local_ai",
            LOCAL_AI_INTEGRITY_NAME,
            "AI-text + reading models",
            &self.base_url,
        )
    }

    async fn status(
        &self,
        cancel: CancellationToken,
    ) -> Result<IntegrityServiceStatus, ProviderError> {
        let value = self
            .call(
                Method::GET,
                "/v1/integrity/status",
                None,
                HEALTH_CHECK_TIMEOUT,
                &cancel,
            )
            .await?;
        let mut status = parse_status(&value);
        // The embedding model the sidecar serves decides whether the echo
        // check can run; an older sidecar without it is not an error.
        if let Ok(health) = self
            .call(Method::GET, "/health", None, HEALTH_CHECK_TIMEOUT, &cancel)
            .await
        {
            status.embedding_model = opt_text(&health, "embedding_model");
        }
        Ok(status)
    }

    async fn ai_text(
        &self,
        text: String,
        cancel: CancellationToken,
    ) -> Result<AiTextResult, ProviderError> {
        if text.trim().is_empty() {
            return Err(ProviderError::Other(
                "there is no answer text to check".to_string(),
            ));
        }
        let value = self
            .call(
                Method::POST,
                "/v1/integrity/echo",
                Some(json!({ "text": bounded_text(&text) })),
                SCORING_TIMEOUT,
                &cancel,
            )
            .await?;
        parse_ai_text(&value)
    }

    async fn start_gaze(
        &self,
        region: Rect,
        cancel: CancellationToken,
    ) -> Result<GazeStatus, ProviderError> {
        let body = json!({
            "left": region.x,
            "top": region.y,
            "width": region.width,
            "height": region.height,
        });
        self.gaze_call("/v1/integrity/gaze/screen", Some(body), &cancel)
            .await
    }

    async fn stop_gaze(&self, cancel: CancellationToken) -> Result<GazeStatus, ProviderError> {
        self.gaze_call("/v1/integrity/gaze/stop", None, &cancel)
            .await
    }

    async fn reset_gaze(&self, cancel: CancellationToken) -> Result<GazeStatus, ProviderError> {
        self.gaze_call("/v1/integrity/gaze/reset", None, &cancel)
            .await
    }

    async fn reading(
        &self,
        start_epoch_s: f64,
        end_epoch_s: f64,
        cancel: CancellationToken,
    ) -> Result<ReadingResult, ProviderError> {
        if !(start_epoch_s.is_finite() && end_epoch_s.is_finite()) || end_epoch_s <= start_epoch_s {
            return Err(ProviderError::Other(
                "the answer has no time window to check".to_string(),
            ));
        }
        // The sidecar scores at most 600 s: keep the end of a longer answer.
        let start = start_epoch_s.max(end_epoch_s - MAX_READING_SPAN_S);
        let value = self
            .call(
                Method::POST,
                "/v1/integrity/reading",
                Some(json!({ "start": start, "end": end_epoch_s })),
                SCORING_TIMEOUT,
                &cancel,
            )
            .await?;
        parse_reading(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_map_to_the_spec() {
        assert_eq!(
            parse_sidecar_level("not_enough_gaze"),
            Some(IntegrityLevel::NoGaze)
        );
        assert_eq!(
            parse_sidecar_level("too_short"),
            Some(IntegrityLevel::TooShort)
        );
        assert_eq!(parse_sidecar_level("high"), Some(IntegrityLevel::High));
        assert_eq!(parse_sidecar_level("weird"), None);
    }

    #[test]
    fn ai_text_and_reading_parsing() {
        let a = parse_ai_text(
            &json!({"ai_probability": 0.93, "level": "high", "words": 40, "model": "echo-tfidf"}),
        )
        .unwrap();
        assert_eq!(a.level, IntegrityLevel::High);
        assert_eq!(a.words, 40);
        assert!(parse_ai_text(&json!({"ai_probability": 0.9, "level": "no_gaze"})).is_err());
        assert!(parse_ai_text(&json!({"level": "low"})).is_err());

        let r = parse_reading(
            &json!({"reading_probability": null, "level": "not_enough_gaze", "seconds": 8.0}),
        )
        .unwrap();
        assert_eq!(r.level, IntegrityLevel::NoGaze);
        assert_eq!(r.probability, None);
        let r = parse_reading(&json!({"reading_probability": 0.8, "level": "possible", "window_probabilities": [0.9, 0.7], "valid_fraction": 0.9, "seconds": 20.0})).unwrap();
        assert_eq!(r.windows, 2);
        assert!(parse_reading(&json!({"level": "high"})).is_err());
    }

    #[test]
    fn long_text_is_bounded() {
        let long = "a".repeat(MAX_AI_TEXT_CHARS + 50);
        assert_eq!(bounded_text(&long).chars().count(), MAX_AI_TEXT_CHARS);
        assert_eq!(bounded_text("short"), "short");
    }
}
