//! Local AI sidecar answer-integrity client against a local mock server.

mod support;

use std::time::{Duration, Instant};

use openmeet_core::ProviderError;
use openmeet_core::integrity::{IntegrityLevel, IntegrityService};
use openmeet_core::ipc::Rect;
use openmeet_core::providers::StaticSecrets;
use openmeet_core::settings::{Settings, TranscriptionProviderKind};
use openmeet_core::types::{Component, ProviderLocation};
use openmeet_providers::factory::{BuildContext, build_integrity, component_base_url};
use openmeet_providers::http::Endpoint;
use openmeet_providers::integrity::LocalAiIntegrity;
use serde_json::json;
use support::{MockServer, Recorded, Reply};
use tokio_util::sync::CancellationToken;

fn status_body() -> serde_json::Value {
    json!({
        "echo_ready": true,
        "reading_ready": false,
        "echo_error": null,
        "reading_error": "ModuleNotFoundError: onnxruntime",
        "gaze": {
            "source": "screen",
            "capturing_screen": true,
            "region": {"left": 10, "top": 20, "width": 640, "height": 360},
            "frames_per_second": 9.5,
            "face_found_fraction": 0.9,
            "tracked_fraction": 0.85,
            "seconds_since_last_frame": 0.1,
            "now": 1_760_000_000.0,
            "error": null
        }
    })
}

/// Answers like the sidecar does (routes by path).
fn sidecar(req: &Recorded, _: usize) -> Reply {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/v1/integrity/status") => Reply::json(200, status_body()),
        ("GET", "/health") => Reply::json(
            200,
            json!({"status": "ok", "embedding_model": "BAAI/bge-small-en-v1.5"}),
        ),
        ("POST", "/v1/integrity/echo") => Reply::json(
            200,
            json!({"ai_probability": 0.91, "level": "high", "words": 42,
                   "thresholds": {"possible": 0.5, "high": 0.8}, "model": "echo-tfidf"}),
        ),
        ("POST", "/v1/integrity/gaze/screen")
        | ("POST", "/v1/integrity/gaze/stop")
        | ("POST", "/v1/integrity/gaze/reset") => Reply::json(200, status_body()["gaze"].clone()),
        ("POST", "/v1/integrity/reading") => Reply::json(
            200,
            json!({"reading_probability": null, "level": "not_enough_gaze", "seconds": 8.0,
                   "detail": "need at least 10 s with the face tracked in most frames"}),
        ),
        _ => Reply::json(404, json!({"detail": "not found"})),
    }
}

fn client(url: &str) -> LocalAiIntegrity {
    LocalAiIntegrity::new(Endpoint::new(url)).unwrap()
}

#[tokio::test]
async fn status_includes_the_sidecar_embedding_model() {
    for suffix in ["", "/v1"] {
        let server = MockServer::start(sidecar).await;
        let c = client(&format!("{}{suffix}", server.url));
        let info = c.info();
        assert_eq!(info.id, "local_ai");
        assert_eq!(info.location, ProviderLocation::Local);
        let s = c.status(CancellationToken::new()).await.unwrap();
        assert!(s.ai_text_ready);
        assert!(!s.reading_ready);
        assert_eq!(
            s.reading_error.as_deref(),
            Some("ModuleNotFoundError: onnxruntime")
        );
        assert!(s.gaze.capturing_screen);
        assert!((s.gaze.tracked_fraction - 0.85).abs() < 1e-9);
        assert_eq!(s.embedding_model.as_deref(), Some("BAAI/bge-small-en-v1.5"));
        let paths: Vec<String> = server.requests().into_iter().map(|r| r.path).collect();
        assert_eq!(paths, vec!["/v1/integrity/status", "/health"], "{suffix:?}");
    }
}

#[tokio::test]
async fn ai_text_posts_the_transcript() {
    let server = MockServer::start(sidecar).await;
    let c = client(&server.url);
    let r = c
        .ai_text(
            "We cached the results in Redis and invalidated them on writes.".into(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(r.level, IntegrityLevel::High);
    assert!((r.probability - 0.91).abs() < 1e-9);
    assert_eq!(r.words, 42);
    let req = server.request(0);
    assert_eq!(req.path, "/v1/integrity/echo");
    assert_eq!(req.header("content-type"), Some("application/json"));
    assert_eq!(
        req.json(),
        json!({"text": "We cached the results in Redis and invalidated them on writes."})
    );
    // Empty answers are not sent.
    assert!(
        c.ai_text("   ".into(), CancellationToken::new())
            .await
            .is_err()
    );
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn gaze_control_and_reading_window() {
    let server = MockServer::start(sidecar).await;
    let c = client(&server.url);
    let region = Rect {
        x: -1920,
        y: 40,
        width: 640,
        height: 360,
    };
    let g = c
        .start_gaze(region, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(g.source, "screen");
    c.stop_gaze(CancellationToken::new()).await.unwrap();
    c.reset_gaze(CancellationToken::new()).await.unwrap();
    // A 900 s answer is cut to its last 600 s.
    let r = c
        .reading(1_000.0, 1_900.0, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r.level, IntegrityLevel::NoGaze);
    assert_eq!(r.probability, None);
    let reqs = server.requests();
    assert_eq!(reqs[0].path, "/v1/integrity/gaze/screen");
    assert_eq!(
        reqs[0].json(),
        json!({"left": -1920, "top": 40, "width": 640, "height": 360})
    );
    assert_eq!(reqs[1].path, "/v1/integrity/gaze/stop");
    assert_eq!(reqs[2].path, "/v1/integrity/gaze/reset");
    assert_eq!(reqs[3].path, "/v1/integrity/reading");
    assert_eq!(reqs[3].json(), json!({"start": 1_300.0, "end": 1_900.0}));
    // An empty window is refused locally.
    assert!(c.reading(5.0, 5.0, CancellationToken::new()).await.is_err());
}

#[tokio::test]
async fn errors_are_typed_and_cancellation_is_prompt() {
    let server = MockServer::with_reply(Reply::json(
        503,
        json!({"detail": "model unavailable: ModuleNotFoundError: onnxruntime"}),
    ))
    .await;
    let err = client(&server.url)
        .reading(0.0, 30.0, CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(
        err,
        ProviderError::Http("503 model unavailable: ModuleNotFoundError: onnxruntime".into())
    );

    let bad = MockServer::with_reply(Reply::json(200, json!({"level": "maybe"}))).await;
    assert!(matches!(
        client(&bad.url)
            .ai_text("some text".into(), CancellationToken::new())
            .await,
        Err(ProviderError::InvalidResponse(_))
    ));

    let slow = MockServer::with_reply(
        Reply::json(200, json!({"ai_probability": 0.1, "level": "low"}))
            .delayed(Duration::from_secs(20)),
    )
    .await;
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    assert_eq!(
        client(&slow.url)
            .ai_text("some text".into(), cancel)
            .await
            .unwrap_err(),
        ProviderError::Cancelled
    );
    assert!(started.elapsed() < Duration::from_millis(600));
}

#[test]
fn factory_builds_only_when_on_and_respects_local_only() {
    let secrets = StaticSecrets::default();
    let mut s = Settings::default();
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    assert!(build_integrity(&ctx).unwrap().is_none(), "off by default");

    s.integrity.enabled = true;
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    let svc = build_integrity(&ctx).unwrap().unwrap();
    assert_eq!(svc.info().location, ProviderLocation::Local);

    // A sidecar on another machine is remote and refused in local-only mode.
    s.transcription.provider = TranscriptionProviderKind::LocalAi;
    s.transcription.base_url = Some("http://192.168.1.50:8765".into());
    s.privacy.local_only = true;
    assert_eq!(
        component_base_url(&s, Component::Integrity).as_deref(),
        Some("http://192.168.1.50:8765")
    );
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    assert!(matches!(
        build_integrity(&ctx),
        Err(ProviderError::BlockedByLocalOnly(_))
    ));
    s.privacy.local_only = false;
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    assert_eq!(
        build_integrity(&ctx).unwrap().unwrap().info().location,
        ProviderLocation::Remote
    );
}
