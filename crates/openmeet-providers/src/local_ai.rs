//! Client for the optional local AI sidecar (`local-ai/server.py`):
//! faster-whisper transcription and fastembed embeddings behind
//! OpenAI-compatible endpoints, plus `GET /health` for status reporting.
//!
//! Transcription and embeddings go through [`crate::openai`] with
//! [`OpenAiFlavor::LocalAi`](crate::openai::OpenAiFlavor::LocalAi); this module
//! only covers the sidecar-specific health endpoint.

use std::time::Duration;

use openmeet_core::ProviderError;
use openmeet_core::providers::{display_host, location_for_url};
use reqwest::header::ACCEPT;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::http::{
    HEALTH_CHECK_TIMEOUT, HttpFailure, build_client, normalize_base_url, read_json, root_url, send,
    validate_endpoint_url,
};

/// Result of the sidecar's `GET /health`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalAiHealth {
    /// The service reported `"status": "ok"`.
    pub ok: bool,
    /// Whisper model the service uses (loaded or configured default).
    pub transcription_model: Option<String>,
    /// Embedding model the service uses (loaded or configured default).
    pub embedding_model: Option<String>,
}

/// Status client for the local AI sidecar.
#[derive(Debug, Clone)]
pub struct LocalAiClient {
    base_url: String,
    host: String,
    timeout: Duration,
    /// Construction problem (invalid URL, HTTP client initialisation),
    /// reported by every call instead of panicking in [`LocalAiClient::new`].
    client: Result<reqwest::Client, ProviderError>,
}

impl LocalAiClient {
    /// Client for the sidecar at `base_url` (with or without a trailing
    /// `/v1`). Invalid URLs are reported by [`LocalAiClient::health`].
    pub fn new(base_url: &str) -> Self {
        let base_url = normalize_base_url(base_url);
        let client = validate_endpoint_url(&base_url, "Local AI")
            .and_then(|()| build_client(location_for_url(&base_url), HEALTH_CHECK_TIMEOUT));
        LocalAiClient {
            host: display_host(&base_url),
            base_url,
            timeout: HEALTH_CHECK_TIMEOUT,
            client,
        }
    }

    /// Overrides the health-check timeout (default 5 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Normalized base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `GET /health` → `{"status":"ok","transcription_model":..,"embedding_model":..}`.
    pub async fn health(&self, cancel: CancellationToken) -> Result<LocalAiHealth, ProviderError> {
        let client = self.client.as_ref().map_err(Clone::clone)?;
        fetch_health(
            client,
            &root_url(&self.base_url),
            self.timeout,
            &self.host,
            &cancel,
        )
        .await
    }
}

/// Shared implementation of the health request (also used by the local AI
/// transcriber's health check).
pub(crate) async fn fetch_health(
    client: &reqwest::Client,
    root: &str,
    timeout: Duration,
    host: &str,
    cancel: &CancellationToken,
) -> Result<LocalAiHealth, ProviderError> {
    let builder = client
        .get(format!("{root}/health"))
        .header(ACCEPT, "application/json");
    let response = send(builder, cancel, timeout, host).await?;
    if !response.status().is_success() {
        return Err(HttpFailure::read(response, cancel, None, host)
            .await
            .into_error());
    }
    let value: Value = read_json(response, "health", cancel, timeout, host).await?;
    Ok(parse_health(&value))
}

fn parse_health(value: &Value) -> LocalAiHealth {
    let text = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    LocalAiHealth {
        ok: value
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("ok")),
        transcription_model: text("transcription_model"),
        embedding_model: text("embedding_model"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn health_parsing() {
        let h = parse_health(&json!({
            "status": "ok",
            "version": "0.1.0",
            "transcription_model": "base.en",
            "embedding_model": "BAAI/bge-small-en-v1.5"
        }));
        assert_eq!(
            h,
            LocalAiHealth {
                ok: true,
                transcription_model: Some("base.en".into()),
                embedding_model: Some("BAAI/bge-small-en-v1.5".into()),
            }
        );
        let h = parse_health(&json!({"status": "loading", "transcription_model": ""}));
        assert!(!h.ok);
        assert_eq!(h.transcription_model, None);
    }

    #[tokio::test]
    async fn invalid_url_is_reported_not_panicked() {
        let c = LocalAiClient::new("not a url");
        let err = c.health(CancellationToken::new()).await.unwrap_err();
        assert!(matches!(err, ProviderError::NotConfigured(_)), "{err:?}");
    }

    #[test]
    fn base_url_is_normalized() {
        let c = LocalAiClient::new("http://127.0.0.1:8765/v1/ ");
        assert_eq!(c.base_url(), "http://127.0.0.1:8765/v1");
        assert_eq!(root_url(c.base_url()), "http://127.0.0.1:8765");
    }
}
