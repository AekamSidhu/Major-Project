use serde::{Deserialize, Serialize};
use thiserror::Error;
use ts_rs::TS;

/// Errors produced by AI / transcription / vision / embedding providers.
///
/// Messages must never contain API keys, raw transcripts or prompt contents.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
#[ts(export)]
pub enum ProviderError {
    #[error("provider is not configured: {0}")]
    NotConfigured(String),
    #[error("authentication failed: {0}")]
    Unauthorized(String),
    #[error("rate limited: {0}")]
    RateLimited(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("request timed out")]
    Timeout,
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("request cancelled")]
    Cancelled,
    #[error("not supported: {0}")]
    Unsupported(String),
    #[error("the model declined this request: {0}")]
    Refused(String),
    #[error("blocked by local-only mode: {0}")]
    BlockedByLocalOnly(String),
    #[error("{0}")]
    Other(String),
}

impl ProviderError {
    /// Whether retrying the same request later could succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited(_) | ProviderError::Network(_) | ProviderError::Timeout
        )
    }

    /// Short message suitable for the overlay.
    pub fn user_message(&self) -> String {
        match self {
            ProviderError::NotConfigured(_) => {
                "AI provider unavailable. Check provider settings.".to_string()
            }
            ProviderError::Unauthorized(_) => {
                "AI provider rejected the API key. Check provider settings.".to_string()
            }
            ProviderError::RateLimited(_) => {
                "AI provider rate limit reached. Try again shortly.".to_string()
            }
            ProviderError::Network(_) | ProviderError::Timeout => {
                "AI provider unavailable. Check your connection and provider settings.".to_string()
            }
            ProviderError::Cancelled => "Cancelled.".to_string(),
            ProviderError::Refused(_) => "The model declined to answer this request.".to_string(),
            ProviderError::BlockedByLocalOnly(_) => {
                "Blocked: local-only mode is on and this provider is remote.".to_string()
            }
            other => other.to_string(),
        }
    }
}

/// Persistence errors.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize)]
pub enum StoreError {
    #[error("database error: {0}")]
    Database(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("migration error: {0}")]
    Migration(String),
}

/// Errors returned across the IPC boundary. Serialized as a plain string-ish
/// object the frontend can display.
#[derive(Debug, Clone, PartialEq, Error, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
#[ts(export)]
pub enum CoreError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid state: {0}")]
    InvalidState(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Provider(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("platform error: {0}")]
    Platform(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("unsupported on this system: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<StoreError> for CoreError {
    fn from(e: StoreError) -> Self {
        match e {
            StoreError::NotFound(m) => CoreError::NotFound(m),
            other => CoreError::Storage(other.to_string()),
        }
    }
}

impl From<ProviderError> for CoreError {
    fn from(e: ProviderError) -> Self {
        CoreError::Provider(e.user_message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_serializes_tagged() {
        let v = serde_json::to_value(ProviderError::Timeout).unwrap();
        assert_eq!(v["kind"], "timeout");
        let v = serde_json::to_value(ProviderError::Http("500".into())).unwrap();
        assert_eq!(v["kind"], "http");
        assert_eq!(v["message"], "500");
    }

    #[test]
    fn retryable() {
        assert!(ProviderError::Timeout.is_retryable());
        assert!(!ProviderError::Unauthorized("x".into()).is_retryable());
    }

    #[test]
    fn store_not_found_maps() {
        let e: CoreError = StoreError::NotFound("m1".into()).into();
        assert_eq!(e, CoreError::NotFound("m1".into()));
    }
}
