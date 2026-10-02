//! Shared HTTP plumbing for every adapter: client construction, timeouts,
//! base-URL handling, cancellable sends, bounded body reads and
//! provider-neutral error mapping.
//!
//! Privacy rules enforced here:
//! * error messages never contain URLs (only `host:port`), API keys or
//!   request/response content — provider error bodies are reduced to their
//!   `message` field, sanitized and truncated;
//! * redirects are never followed (a redirect could re-send a meeting
//!   transcript, or a custom auth header, to another host);
//! * clients for loopback endpoints ignore proxy settings and only connect to
//!   loopback addresses, so "local" really means "on this device".

use std::error::Error as StdError;
use std::fmt;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use openmeet_core::ProviderError;
use openmeet_core::providers::{display_host, location_for_url};
use openmeet_core::types::{ProviderInfo, ProviderLocation};
use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use tokio_util::sync::CancellationToken;

/// `User-Agent` sent with every request.
pub const USER_AGENT: &str = concat!("OpenMeet/", env!("CARGO_PKG_VERSION"));
/// TCP + TLS connect timeout.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout for health checks and model listings.
pub const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);
/// Error bodies are read up to this size (the rest is ignored).
pub const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
/// Upper bound for non-streaming JSON responses (model lists, embeddings,
/// transcriptions).
pub const MAX_JSON_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Maximum length (in characters) of a provider error message kept in a
/// [`ProviderError`].
pub const MAX_ERROR_MESSAGE_CHARS: usize = 240;

/// Timeouts applied by the adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// TCP + TLS connect.
    pub connect: Duration,
    /// From sending a streaming request until the response headers arrive
    /// (covers model loading in local servers and queueing at the provider).
    pub response: Duration,
    /// Maximum silence between two body chunks of a streaming response.
    pub stream_idle: Duration,
    /// Whole non-streaming request (transcription, embeddings), body included.
    pub request: Duration,
    /// Health checks and model listing.
    pub health: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Timeouts {
            connect: CONNECT_TIMEOUT,
            response: Duration::from_secs(120),
            stream_idle: Duration::from_secs(120),
            request: Duration::from_secs(120),
            health: HEALTH_CHECK_TIMEOUT,
        }
    }
}

/// Where and how to reach a provider.
#[derive(Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// Base URL, with or without a trailing `/v1` (both are accepted).
    pub base_url: String,
    /// API key, when the service needs one. Never logged or shown.
    pub api_key: Option<String>,
    /// Network timeouts.
    pub timeouts: Timeouts,
}

impl Endpoint {
    /// Endpoint without credentials and with default timeouts.
    pub fn new(base_url: impl Into<String>) -> Self {
        Endpoint {
            base_url: normalize_base_url(&base_url.into()),
            api_key: None,
            timeouts: Timeouts::default(),
        }
    }

    /// Sets the API key. Blank keys are treated as absent.
    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty());
        self
    }

    /// Replaces the timeouts.
    pub fn with_timeouts(mut self, timeouts: Timeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Where requests to this endpoint are processed.
    pub fn location(&self) -> ProviderLocation {
        location_for_url(&self.base_url)
    }

    /// `host[:port]` for display and error messages.
    pub fn host(&self) -> String {
        display_host(&self.base_url)
    }
}

impl fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Endpoint")
            .field("host", &self.host())
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("timeouts", &self.timeouts)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Clients
// ---------------------------------------------------------------------------

/// Shared client configuration: connect timeout, `User-Agent`, no redirects.
///
/// For loopback endpoints ([`ProviderLocation::Local`]) proxies are disabled
/// (an `HTTP_PROXY` would otherwise route local traffic off the device) and
/// host names must resolve to loopback addresses.
pub fn client_builder(location: ProviderLocation, connect_timeout: Duration) -> reqwest::ClientBuilder {
    let builder = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none())
        .pool_idle_timeout(Duration::from_secs(60))
        .tcp_nodelay(true);
    if location == ProviderLocation::Remote {
        builder
    } else {
        builder.no_proxy().dns_resolver(LoopbackOnlyResolver)
    }
}

/// Builds a client with [`client_builder`].
pub fn build_client(
    location: ProviderLocation,
    connect_timeout: Duration,
) -> Result<reqwest::Client, ProviderError> {
    client_builder(location, connect_timeout)
        .build()
        .map_err(|_| ProviderError::Other("could not initialise the HTTP client".to_string()))
}

/// Error returned by [`LoopbackOnlyResolver`] when a name resolves to a
/// non-loopback address.
#[derive(Debug)]
struct NonLoopbackAddress;

impl fmt::Display for NonLoopbackAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("host name did not resolve to a loopback address")
    }
}

impl StdError for NonLoopbackAddress {}

/// DNS resolver for endpoints classified as local: only loopback addresses
/// are returned, so a host name such as `localhost` can never be redirected
/// to another machine by a hosts file or DNS.
#[derive(Debug, Clone, Copy)]
struct LoopbackOnlyResolver;

impl reqwest::dns::Resolve for LoopbackOnlyResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let resolved = tokio::net::lookup_host((host.as_str(), 0)).await?;
            let loopback: Vec<SocketAddr> = resolved.filter(|a| a.ip().is_loopback()).collect();
            if loopback.is_empty() {
                return Err(Box::new(NonLoopbackAddress) as Box<dyn StdError + Send + Sync>);
            }
            Ok(Box::new(loopback.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Marks a header value as sensitive (excluded from debug output).
pub(crate) fn sensitive_header(value: &str) -> Result<HeaderValue, ProviderError> {
    let mut v = HeaderValue::from_str(value).map_err(|_| {
        ProviderError::NotConfigured("the API key contains characters that are not allowed".to_string())
    })?;
    v.set_sensitive(true);
    Ok(v)
}

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------

/// Trims whitespace and trailing slashes.
pub fn normalize_base_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

/// Service root: [`normalize_base_url`] plus removal of a trailing `/v1`
/// (`http://localhost:1234/v1` → `http://localhost:1234`).
pub fn root_url(url: &str) -> String {
    let u = normalize_base_url(url);
    let lower = u.to_ascii_lowercase();
    if lower.ends_with("/v1") {
        u[..u.len() - 3].to_string()
    } else {
        u
    }
}

/// Base for OpenAI-style APIs: a URL whose last path segment is a version
/// (`/v1`, `/v1beta`, `/v2`, …) is used as is, otherwise `/v1` is appended
/// (`http://localhost:1234` → `http://localhost:1234/v1`).
pub fn openai_api_base(url: &str) -> String {
    let u = normalize_base_url(url);
    if ends_with_version_segment(&u) {
        u
    } else {
        format!("{u}/v1")
    }
}

fn ends_with_version_segment(url: &str) -> bool {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let Some((_, path)) = after_scheme.split_once('/') else {
        return false;
    };
    let last = path.rsplit('/').next().unwrap_or("");
    let mut chars = last.chars();
    matches!(chars.next(), Some('v' | 'V')) && chars.next().is_some_and(|c| c.is_ascii_digit())
}

/// Validates a base URL: http(s), a host, no embedded credentials, no query
/// or fragment. `what` names the component in the error ("LLM", ...).
/// Error messages never echo the URL itself.
pub fn validate_endpoint_url(url: &str, what: &str) -> Result<(), ProviderError> {
    if url.trim().is_empty() {
        return Err(ProviderError::NotConfigured(format!("{what} base URL is not set")));
    }
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| ProviderError::NotConfigured(format!("{what} base URL is not a valid URL")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ProviderError::NotConfigured(format!(
            "{what} base URL must start with http:// or https://"
        )));
    }
    if parsed.host_str().is_none_or(str::is_empty) {
        return Err(ProviderError::NotConfigured(format!("{what} base URL is missing a host")));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(ProviderError::NotConfigured(format!(
            "credentials must not be embedded in the {what} base URL"
        )));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(ProviderError::NotConfigured(format!(
            "{what} base URL must not contain a query string or fragment"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sending and reading
// ---------------------------------------------------------------------------

/// Sends a request, honouring cancellation and a deadline for the response
/// headers.
pub(crate) async fn send(
    request: RequestBuilder,
    cancel: &CancellationToken,
    timeout: Duration,
    host: &str,
) -> Result<Response, ProviderError> {
    if cancel.is_cancelled() {
        return Err(ProviderError::Cancelled);
    }
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(ProviderError::Cancelled),
        result = tokio::time::timeout(timeout, request.send()) => match result {
            Err(_) => Err(ProviderError::Timeout),
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(transport_error(e, host)),
        }
    }
}

/// Reads a response body, failing beyond `limit` bytes (or truncating when
/// `truncate` is set), with cancellation and an overall deadline.
pub(crate) async fn read_body(
    response: &mut Response,
    limit: usize,
    truncate: bool,
    cancel: &CancellationToken,
    timeout: Duration,
    host: &str,
) -> Result<Vec<u8>, ProviderError> {
    let read = async {
        let mut body = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    if body.len() + chunk.len() > limit {
                        if truncate {
                            let room = limit - body.len();
                            body.extend_from_slice(&chunk[..room]);
                            return Ok(body);
                        }
                        return Err(ProviderError::InvalidResponse(format!(
                            "response body exceeds {limit} bytes"
                        )));
                    }
                    body.extend_from_slice(&chunk);
                }
                Ok(None) => return Ok(body),
                Err(e) => return Err(transport_error(e, host)),
            }
        }
    };
    tokio::select! {
        biased;
        _ = cancel.cancelled() => Err(ProviderError::Cancelled),
        result = tokio::time::timeout(timeout, read) => match result {
            Err(_) => Err(ProviderError::Timeout),
            Ok(r) => r,
        }
    }
}

/// Parses JSON without ever echoing body content in the error.
pub(crate) fn parse_json<T: DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, ProviderError> {
    serde_json::from_slice(bytes).map_err(|e| {
        ProviderError::InvalidResponse(format!(
            "could not parse the {what} response ({:?} error at line {}, column {})",
            e.classify(),
            e.line(),
            e.column()
        ))
    })
}

/// Reads and parses a successful JSON response.
pub(crate) async fn read_json<T: DeserializeOwned>(
    mut response: Response,
    what: &str,
    cancel: &CancellationToken,
    timeout: Duration,
    host: &str,
) -> Result<T, ProviderError> {
    let body = read_body(&mut response, MAX_JSON_BODY_BYTES, false, cancel, timeout, host).await?;
    parse_json(&body, what)
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// A non-success HTTP response, reduced to what is safe to surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpFailure {
    /// Response status.
    pub status: StatusCode,
    /// Sanitized provider error message, if the body had one.
    pub message: Option<String>,
    /// Seconds from `Retry-After` / `retry-after-ms`.
    pub retry_after_secs: Option<u64>,
}

impl HttpFailure {
    /// Reads (a bounded prefix of) the error body. Never fails: an unreadable
    /// body just yields no message.
    pub(crate) async fn read(
        mut response: Response,
        cancel: &CancellationToken,
        secret: Option<&str>,
        host: &str,
    ) -> HttpFailure {
        let status = response.status();
        let retry_after_secs = parse_retry_after(response.headers(), SystemTime::now());
        let body = read_body(
            &mut response,
            MAX_ERROR_BODY_BYTES,
            true,
            cancel,
            HEALTH_CHECK_TIMEOUT,
            host,
        )
        .await
        .unwrap_or_default();
        let message = extract_error_message(&body)
            .map(|m| sanitize_message(&m, secret))
            .filter(|m| !m.is_empty());
        HttpFailure {
            status,
            message,
            retry_after_secs,
        }
    }

    /// Lower-cased message for rule matching ("" when absent).
    pub(crate) fn message_lower(&self) -> String {
        self.message.as_deref().unwrap_or("").to_ascii_lowercase()
    }

    /// Maps the failure to a [`ProviderError`]:
    /// 401/403 → `Unauthorized`, 429 → `RateLimited`, 408/504 → `Timeout`,
    /// anything else → `Http("<status> <message>")`.
    pub fn into_error(self) -> ProviderError {
        let code = self.status.as_u16();
        let detail = self.message.clone().unwrap_or_else(|| {
            self.status
                .canonical_reason()
                .unwrap_or("error")
                .to_string()
        });
        match code {
            401 => ProviderError::Unauthorized(format!(
                "401 {}",
                self.message.as_deref().unwrap_or("invalid or missing API key")
            )),
            403 => ProviderError::Unauthorized(format!(
                "403 {}",
                self.message.as_deref().unwrap_or("access denied")
            )),
            429 => ProviderError::RateLimited(match self.retry_after_secs {
                Some(secs) => format!("429 {detail} (retry after {secs} s)"),
                None => format!("429 {detail}"),
            }),
            408 | 504 => ProviderError::Timeout,
            _ => ProviderError::Http(format!("{code} {detail}")),
        }
    }
}

/// Maps a transport-level reqwest error. The message names the host only;
/// the reqwest error text (which contains the full URL) is never used.
pub(crate) fn transport_error(error: reqwest::Error, host: &str) -> ProviderError {
    let error = error.without_url();
    if error.is_timeout() {
        return ProviderError::Timeout;
    }
    let mut chain = String::new();
    let mut io_kind = None;
    let mut non_loopback = false;
    let mut source: Option<&(dyn StdError + 'static)> = error.source();
    while let Some(err) = source {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            io_kind.get_or_insert(io.kind());
        }
        if err.is::<NonLoopbackAddress>() {
            non_loopback = true;
        }
        chain.push_str(&err.to_string().to_ascii_lowercase());
        chain.push(' ');
        source = err.source();
    }
    if non_loopback || chain.contains("did not resolve to a loopback") {
        return ProviderError::Network(format!(
            "{host} does not resolve to this device; refusing to send data to it"
        ));
    }
    if chain.contains("timed out") || io_kind == Some(std::io::ErrorKind::TimedOut) {
        return ProviderError::Timeout;
    }
    if error.is_dns() || chain.contains("dns error") || chain.contains("failed to lookup") {
        return ProviderError::Network(format!("could not resolve {host}"));
    }
    if io_kind == Some(std::io::ErrorKind::ConnectionRefused) || chain.contains("refused") {
        return ProviderError::Network(format!(
            "connection refused by {host} (is the service running?)"
        ));
    }
    if chain.contains("certificate") {
        return ProviderError::Network(format!(
            "the TLS certificate of {host} could not be verified"
        ));
    }
    if chain.contains("tls") || chain.contains("handshake") {
        return ProviderError::Network(format!("TLS handshake with {host} failed"));
    }
    if error.is_connect() {
        return ProviderError::Network(format!("could not connect to {host}"));
    }
    if error.is_body() || error.is_decode() {
        return ProviderError::Network(format!("the connection to {host} was interrupted"));
    }
    if matches!(
        io_kind,
        Some(std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted)
    ) {
        return ProviderError::Network(format!("the connection to {host} was reset"));
    }
    ProviderError::Network(format!("the request to {host} failed"))
}

/// Extracts the human-readable message from common provider error bodies:
/// `{"error":{"message":..}}` (OpenAI, Anthropic, llama.cpp),
/// `{"error":".."}` (Ollama, LM Studio), `{"message":..}` (vLLM),
/// `{"detail":".."}` / `{"detail":[{"loc":[..],"msg":..}]}` (FastAPI).
/// Non-JSON bodies (HTML error pages, plain text) yield `None`; other fields
/// (which could echo request content) are never read.
pub fn extract_error_message(body: &[u8]) -> Option<String> {
    use serde_json::Value;
    fn text(v: &Value) -> Option<String> {
        v.as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
    let v: Value = serde_json::from_slice(body).ok()?;
    if let Some(e) = v.get("error") {
        if let Some(m) = text(e) {
            return Some(m);
        }
        if let Some(m) = e.get("message").and_then(text) {
            return Some(m);
        }
        if let Some(m) = e.get("msg").and_then(text) {
            return Some(m);
        }
    }
    if let Some(m) = v.get("message").and_then(text) {
        return Some(m);
    }
    if let Some(detail) = v.get("detail") {
        if let Some(m) = text(detail) {
            return Some(m);
        }
        if let Some(items) = detail.as_array() {
            let parts: Vec<String> = items
                .iter()
                .take(3)
                .filter_map(|item| {
                    let msg = item.get("msg").and_then(text)?;
                    let loc = item
                        .get("loc")
                        .and_then(Value::as_array)
                        .map(|l| {
                            l.iter()
                                .filter_map(|x| {
                                    x.as_str()
                                        .map(str::to_string)
                                        .or_else(|| x.as_u64().map(|n| n.to_string()))
                                })
                                .collect::<Vec<_>>()
                                .join(".")
                        })
                        .unwrap_or_default();
                    Some(if loc.is_empty() {
                        msg
                    } else {
                        format!("{loc}: {msg}")
                    })
                })
                .collect();
            if !parts.is_empty() {
                return Some(parts.join("; "));
            }
        }
    }
    None
}

/// Makes a provider message safe to show and store: removes the configured
/// secret and anything that looks like an API key or token, collapses
/// whitespace and truncates to [`MAX_ERROR_MESSAGE_CHARS`].
pub fn sanitize_message(raw: &str, secret: Option<&str>) -> String {
    let mut s = raw.to_string();
    if let Some(secret) = secret.map(str::trim).filter(|k| k.len() >= 6) {
        s = s.replace(secret, "[redacted]");
    }
    let words: Vec<String> = s.split_whitespace().map(redact_token).collect();
    truncate_chars(&words.join(" "), MAX_ERROR_MESSAGE_CHARS)
}

fn redact_token(token: &str) -> String {
    let core = token.trim_matches(|c: char| {
        !(c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '*' || c == '.')
    });
    let core = core.trim_end_matches('.');
    if !core.is_empty() && looks_like_secret(core) {
        token.replacen(core, "[redacted]", 1)
    } else {
        token.to_string()
    }
}

fn looks_like_secret(token: &str) -> bool {
    if token.len() < 12 {
        return false;
    }
    let lower = token.to_ascii_lowercase();
    const PREFIXES: [&str; 6] = ["sk-", "sk_", "gsk_", "xai-", "pk-", "rk-"];
    if PREFIXES.iter().any(|p| lower.starts_with(p)) {
        return true;
    }
    if token.contains("****") {
        return true;
    }
    // Long random-looking runs (letters and digits, no separators).
    token
        .split(['-', '_', '.'])
        .any(|run| {
            run.len() >= 20
                && run.chars().all(|c| c.is_ascii_alphanumeric())
                && run.chars().any(|c| c.is_ascii_digit())
                && run.chars().any(|c| c.is_ascii_alphabetic())
        })
}

/// Truncates to `max` characters (on a char boundary), appending "…".
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        Some((idx, _)) => format!("{}…", &s[..idx]),
        None => s.to_string(),
    }
}

/// Seconds to wait from `retry-after-ms` (OpenAI) or `Retry-After`
/// (delta-seconds or an IMF-fixdate HTTP date).
pub fn parse_retry_after(headers: &HeaderMap, now: SystemTime) -> Option<u64> {
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::trim);
    if let Some(ms) = header("retry-after-ms").and_then(|v| v.parse::<f64>().ok()) {
        if ms.is_finite() && ms >= 0.0 {
            return Some((ms / 1000.0).ceil() as u64);
        }
    }
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(secs) = value.parse::<u64>() {
        return Some(secs);
    }
    if let Ok(secs) = value.parse::<f64>() {
        if secs.is_finite() && secs >= 0.0 {
            return Some(secs.ceil() as u64);
        }
    }
    let at = parse_http_date(value)?;
    let now = now.duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(at.saturating_sub(now))
}

/// Parses an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) into Unix seconds.
pub(crate) fn parse_http_date(value: &str) -> Option<u64> {
    let (_, rest) = value.split_once(',')?;
    let mut parts = rest.split_ascii_whitespace();
    let day: u32 = parts.next()?.parse().ok()?;
    let month: u32 = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut hms = parts.next()?.split(':');
    let hour: i64 = hms.next()?.parse().ok()?;
    let minute: i64 = hms.next()?.parse().ok()?;
    let second: i64 = hms.next()?.parse().ok()?;
    if !matches!(parts.next()?, "GMT" | "UTC") {
        return None;
    }
    if !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let secs = days * 86_400 + hour * 3600 + minute * 60 + second;
    u64::try_from(secs).ok()
}

/// Days since 1970-01-01 for a proleptic Gregorian date (H. Hinnant).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(month) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// [`ProviderInfo`] for an endpoint; the location is derived from the actual
/// base URL (a LAN host is `Remote`).
pub(crate) fn provider_info(id: &str, display_name: &str, model: &str, base_url: &str) -> ProviderInfo {
    ProviderInfo {
        id: id.to_string(),
        display_name: display_name.to_string(),
        model: model.to_string(),
        location: location_for_url(base_url),
        endpoint: Some(display_host(base_url)),
    }
}

/// Extracts the allowed output-token limit from a provider message such as
/// "…supports at most 16384 completion tokens…" (OpenAI),
/// "max_tokens: 70000 > 64000, which is the maximum allowed…" (Anthropic) or
/// "Expected a value <= 4096…".
pub(crate) fn parse_token_limit(message: &str) -> Option<u32> {
    for marker in ["at most ", "<= ", "> ", "maximum of ", "up to "] {
        let mut search = message;
        while let Some(idx) = search.find(marker) {
            let after = &search[idx + marker.len()..];
            let digits: String = after
                .trim_start_matches(['`', '\'', '"'])
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == ',')
                .filter(|c| *c != ',')
                .collect();
            if let Ok(n) = digits.parse::<u32>() {
                if n > 0 {
                    return Some(n);
                }
            }
            search = after;
        }
    }
    None
}

/// Short machine-readable kind of an error, for logs.
pub(crate) fn error_kind(error: &ProviderError) -> &'static str {
    match error {
        ProviderError::NotConfigured(_) => "not_configured",
        ProviderError::Unauthorized(_) => "unauthorized",
        ProviderError::RateLimited(_) => "rate_limited",
        ProviderError::Network(_) => "network",
        ProviderError::Timeout => "timeout",
        ProviderError::Http(_) => "http",
        ProviderError::InvalidResponse(_) => "invalid_response",
        ProviderError::Cancelled => "cancelled",
        ProviderError::Unsupported(_) => "unsupported",
        ProviderError::Refused(_) => "refused",
        ProviderError::BlockedByLocalOnly(_) => "blocked_by_local_only",
        ProviderError::Other(_) => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_normalization() {
        assert_eq!(normalize_base_url(" http://h:1/ "), "http://h:1");
        assert_eq!(root_url("http://localhost:1234/v1"), "http://localhost:1234");
        assert_eq!(root_url("http://localhost:1234/v1/"), "http://localhost:1234");
        assert_eq!(root_url("http://localhost:1234"), "http://localhost:1234");
        assert_eq!(root_url("https://api.anthropic.com/V1"), "https://api.anthropic.com");
        assert_eq!(openai_api_base("http://localhost:1234"), "http://localhost:1234/v1");
        assert_eq!(openai_api_base("http://localhost:1234/v1/"), "http://localhost:1234/v1");
        assert_eq!(
            openai_api_base("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            openai_api_base("https://openrouter.ai/api"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(openai_api_base("http://h/v1beta"), "http://h/v1beta");
        assert_eq!(openai_api_base("http://h/vendor"), "http://h/vendor/v1");
    }

    #[test]
    fn endpoint_validation_never_echoes_the_url() {
        assert!(validate_endpoint_url("https://api.openai.com", "LLM").is_ok());
        assert!(validate_endpoint_url("http://127.0.0.1:11434/v1", "LLM").is_ok());
        let cases = [
            "",
            "ftp://x",
            "not a url",
            "https://user:secretpw@host",
            "https://host/v1?key=supersecret",
        ];
        for c in cases {
            let err = validate_endpoint_url(c, "LLM").unwrap_err();
            let msg = err.to_string();
            assert!(matches!(err, ProviderError::NotConfigured(_)), "{c}");
            assert!(!msg.contains("secret"), "{msg}");
        }
    }

    #[test]
    fn endpoint_debug_redacts_key() {
        let e = Endpoint::new("https://api.openai.com/").with_api_key(Some(" sk-test-123 ".into()));
        assert_eq!(e.api_key.as_deref(), Some("sk-test-123"));
        assert_eq!(e.base_url, "https://api.openai.com");
        let dbg = format!("{e:?}");
        assert!(!dbg.contains("sk-test"), "{dbg}");
        assert!(dbg.contains("api.openai.com"));
        assert_eq!(Endpoint::new("x").with_api_key(Some("  ".into())).api_key, None);
    }

    #[test]
    fn error_message_extraction() {
        let openai = br#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#;
        assert_eq!(extract_error_message(openai).as_deref(), Some("Incorrect API key provided"));
        let ollama = br#"{"error":"model \"llama3.2\" not found, try pulling it first"}"#;
        assert_eq!(
            extract_error_message(ollama).as_deref(),
            Some("model \"llama3.2\" not found, try pulling it first")
        );
        let vllm = br#"{"object":"error","message":"bad thing","code":400}"#;
        assert_eq!(extract_error_message(vllm).as_deref(), Some("bad thing"));
        let fastapi = br#"{"detail":[{"loc":["body","max_completion_tokens"],"msg":"Extra inputs are not permitted","input":"SECRET CONTENT"}]}"#;
        let m = extract_error_message(fastapi).unwrap();
        assert_eq!(m, "body.max_completion_tokens: Extra inputs are not permitted");
        assert!(!m.contains("SECRET"));
        assert_eq!(extract_error_message(br#"{"detail":"Not Found"}"#).as_deref(), Some("Not Found"));
        assert_eq!(extract_error_message(b"<html>bad gateway</html>"), None);
        assert_eq!(extract_error_message(b""), None);
    }

    #[test]
    fn sanitizer_redacts_keys_but_keeps_model_names() {
        let m = sanitize_message(
            "Incorrect API key provided: sk-proj-********************abcd. You can find your API key at https://platform.openai.com/account/api-keys.",
            None,
        );
        assert!(!m.contains("sk-proj"), "{m}");
        assert!(m.contains("[redacted]"));
        let m = sanitize_message("bad key k3Jd8sLq0aZx91mNbVcX7tYu used", None);
        assert!(!m.contains("k3Jd8sLq0aZx91mNbVcX7tYu"), "{m}");
        let m = sanitize_message("my-custom-key-value leaked", Some("my-custom-key-value"));
        assert_eq!(m, "[redacted] leaked");
        let m = sanitize_message(
            "model claude-sonnet-4-5-20250929 and gpt-4o-mini-transcribe-2025-03-20 not found",
            None,
        );
        assert!(m.contains("claude-sonnet-4-5-20250929"), "{m}");
        assert!(m.contains("gpt-4o-mini-transcribe-2025-03-20"), "{m}");
        let long = "x ".repeat(500);
        assert!(sanitize_message(&long, None).chars().count() <= MAX_ERROR_MESSAGE_CHARS + 1);
        assert_eq!(sanitize_message("a\n\tb   c", None), "a b c");
    }

    #[test]
    fn retry_after_parsing() {
        let now = UNIX_EPOCH + Duration::from_secs(784_111_700);
        let mut h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h, now), None);
        h.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        assert_eq!(parse_retry_after(&h, now), Some(7));
        h.insert(RETRY_AFTER, HeaderValue::from_static("1.5"));
        assert_eq!(parse_retry_after(&h, now), Some(2));
        h.insert(RETRY_AFTER, HeaderValue::from_static("Sun, 06 Nov 1994 08:49:37 GMT"));
        assert_eq!(parse_retry_after(&h, now), Some(77));
        h.insert("retry-after-ms", HeaderValue::from_static("2500"));
        assert_eq!(parse_retry_after(&h, now), Some(3));
        let mut h = HeaderMap::new();
        h.insert(RETRY_AFTER, HeaderValue::from_static("soon"));
        assert_eq!(parse_retry_after(&h, now), None);
    }

    #[test]
    fn http_dates() {
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"), Some(784_111_777));
        assert_eq!(parse_http_date("Thu, 01 Jan 1970 00:00:00 GMT"), Some(0));
        assert_eq!(parse_http_date("Tue, 29 Feb 2028 12:00:00 GMT"), Some(1_835_438_400));
        assert_eq!(parse_http_date("Sun, 06 Foo 1994 08:49:37 GMT"), None);
        assert_eq!(parse_http_date("Sun, 06 Nov 1994 08:49:37 PST"), None);
        assert_eq!(parse_http_date("garbage"), None);
    }

    #[test]
    fn status_mapping() {
        let f = |status: u16, message: Option<&str>, retry: Option<u64>| HttpFailure {
            status: StatusCode::from_u16(status).unwrap(),
            message: message.map(str::to_string),
            retry_after_secs: retry,
        }
        .into_error();
        assert!(matches!(f(401, None, None), ProviderError::Unauthorized(_)));
        assert!(matches!(f(403, Some("no access"), None), ProviderError::Unauthorized(m) if m == "403 no access"));
        assert_eq!(
            f(429, Some("slow down"), Some(12)),
            ProviderError::RateLimited("429 slow down (retry after 12 s)".into())
        );
        assert_eq!(f(408, None, None), ProviderError::Timeout);
        assert_eq!(f(504, None, None), ProviderError::Timeout);
        assert_eq!(f(500, None, None), ProviderError::Http("500 Internal Server Error".into()));
        assert_eq!(
            f(404, Some("model not found"), None),
            ProviderError::Http("404 model not found".into())
        );
    }

    #[test]
    fn truncation_is_char_safe() {
        assert_eq!(truncate_chars("héllo", 2), "hé…");
        assert_eq!(truncate_chars("hi", 5), "hi");
    }
}
