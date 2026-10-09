//! Local AI sidecar status client against a local mock server.

mod support;

use std::time::{Duration, Instant};

use openmeet_core::ProviderError;
use openmeet_providers::local_ai::{LocalAiClient, LocalAiHealth};
use serde_json::json;
use support::{MockServer, Reply, closed_port};
use tokio_util::sync::CancellationToken;

fn health_reply() -> Reply {
    Reply::json(
        200,
        json!({
            "status": "ok",
            "version": "0.1.0",
            "transcription_model": "base.en",
            "embedding_model": "BAAI/bge-small-en-v1.5"
        }),
    )
}

#[tokio::test]
async fn health_reports_models_with_or_without_v1() {
    for suffix in ["", "/", "/v1", "/v1/"] {
        let server = MockServer::with_reply(health_reply()).await;
        let client = LocalAiClient::new(&format!("{}{suffix}", server.url));
        let health = client.health(CancellationToken::new()).await.unwrap();
        assert_eq!(
            health,
            LocalAiHealth {
                ok: true,
                transcription_model: Some("base.en".into()),
                embedding_model: Some("BAAI/bge-small-en-v1.5".into()),
            },
            "suffix {suffix:?}"
        );
        let req = server.request(0);
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/health", "suffix {suffix:?}");
        assert_eq!(req.header("authorization"), None);
        assert!(req.header("user-agent").unwrap().starts_with("OpenMeet/"));
    }
}

#[tokio::test]
async fn not_ready_service_is_reported_not_ok() {
    let server = MockServer::with_reply(Reply::json(200, json!({"status": "starting"}))).await;
    let health = LocalAiClient::new(&server.url)
        .health(CancellationToken::new())
        .await
        .unwrap();
    assert!(!health.ok);
    assert_eq!(health.transcription_model, None);
    assert_eq!(health.embedding_model, None);
}

#[tokio::test]
async fn errors_are_typed() {
    let server =
        MockServer::with_reply(Reply::json(500, json!({"detail": "model load failed"}))).await;
    let err = LocalAiClient::new(&server.url)
        .health(CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(err, ProviderError::Http("500 model load failed".into()));

    let garbage = MockServer::with_reply(Reply::new(
        200,
        "application/json",
        support::Body::Full(b"not json".to_vec()),
    ))
    .await;
    assert!(matches!(
        LocalAiClient::new(&garbage.url)
            .health(CancellationToken::new())
            .await,
        Err(ProviderError::InvalidResponse(_))
    ));

    let port = closed_port().await;
    let err = LocalAiClient::new(&format!("http://127.0.0.1:{port}"))
        .health(CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ProviderError::Network(m) if m.contains(&format!("127.0.0.1:{port}"))),
        "{err:?}"
    );
}

#[tokio::test]
async fn health_times_out_and_cancels_promptly() {
    let slow = MockServer::with_reply(health_reply().delayed(Duration::from_secs(20))).await;
    let started = Instant::now();
    let err = LocalAiClient::new(&slow.url)
        .with_timeout(Duration::from_millis(300))
        .health(CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(err, ProviderError::Timeout);
    assert!(started.elapsed() < Duration::from_secs(3));

    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let err = LocalAiClient::new(&slow.url)
        .health(cancel)
        .await
        .unwrap_err();
    assert_eq!(err, ProviderError::Cancelled);
    assert!(started.elapsed() < Duration::from_millis(600));

    let already = CancellationToken::new();
    already.cancel();
    assert_eq!(
        LocalAiClient::new(&slow.url).health(already).await,
        Err(ProviderError::Cancelled)
    );
}

#[tokio::test]
async fn invalid_urls_are_not_configured() {
    for url in ["", "ftp://127.0.0.1:8765", "http://user:pw@127.0.0.1:8765"] {
        let err = LocalAiClient::new(url)
            .health(CancellationToken::new())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ProviderError::NotConfigured(_)),
            "{url}: {err:?}"
        );
        assert!(!err.to_string().contains("pw"), "{err}");
    }
}
