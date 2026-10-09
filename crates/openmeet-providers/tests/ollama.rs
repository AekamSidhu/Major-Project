//! Ollama chat and embeddings adapters against a local mock server.

mod support;

use std::time::{Duration, Instant};

use futures::StreamExt;
use openmeet_core::ProviderError;
use openmeet_core::providers::{
    ChatMessage, EmbeddingProvider, ImageInput, LlmChunk, LlmProvider, LlmRequest, TokenUsage,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::{ProviderLocation, RequestKind};
use openmeet_providers::http::{Endpoint, Timeouts};
use openmeet_providers::ollama::{OllamaEmbeddings, OllamaLlm, THINKING_HEADROOM_TOKENS};
use serde_json::json;
use support::{MockServer, Reply, collect};
use tokio_util::sync::CancellationToken;

/// A realistic `/api/chat` stream: thinking (ignored), visible content with a
/// multi-byte character, a blank line, and the final statistics line.
const STREAM: &str = concat!(
    "{\"model\":\"qwen3:8b\",\"created_at\":\"2026-10-03T10:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"\",\"thinking\":\"Let me think\"},\"done\":false}\n",
    "{\"model\":\"qwen3:8b\",\"created_at\":\"2026-10-03T10:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"Use a \"},\"done\":false}\n",
    "\n",
    "{\"model\":\"qwen3:8b\",\"created_at\":\"2026-10-03T10:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"token bucket ✓\"},\"done\":false}\n",
    "{\"model\":\"qwen3:8b\",\"created_at\":\"2026-10-03T10:00:01Z\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"total_duration\":123456,\"prompt_eval_count\":31,\"eval_count\":7}\n"
);

fn request(latency: bool) -> LlmRequest {
    let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "System prompt.", "Question?");
    r.temperature = Some(0.3);
    r.max_output_tokens = 600;
    r.latency_sensitive = latency;
    r
}

fn make(url: &str, model: &str) -> OllamaLlm {
    OllamaLlm::new(Endpoint::new(url), model).unwrap()
}

#[tokio::test]
async fn chat_request_shape_and_stream_across_chunk_boundaries() {
    for chunk_size in [1, 3, 17, 4096] {
        let server = MockServer::with_reply(Reply::ndjson(STREAM, chunk_size)).await;
        let llm = make(&server.url, "qwen3:8b");
        let out = collect(
            llm.stream(request(true), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.text, "Use a token bucket ✓", "chunk size {chunk_size}");
        assert!(
            !out.text.contains("Let me think"),
            "thinking is never surfaced"
        );
        assert_eq!(
            out.done,
            Some(LlmChunk::Done {
                stop_reason: Some("stop".into()),
                usage: Some(TokenUsage {
                    input_tokens: 31,
                    output_tokens: 7
                }),
            })
        );
        assert!(out.error.is_none());

        let req = server.request(0);
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/api/chat");
        assert_eq!(req.header("authorization"), None);
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert!(req.header("user-agent").unwrap().starts_with("OpenMeet/"));
        let body = req.json();
        assert_eq!(body["model"], "qwen3:8b");
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Question?"}
            ])
        );
        assert_eq!(body["options"]["num_predict"], 600);
        assert!((body["options"]["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        // Latency-sensitive requests turn thinking off.
        assert_eq!(body["think"], false);
    }
}

#[tokio::test]
async fn thinking_allowed_for_non_latency_requests_with_headroom() {
    let server = MockServer::with_reply(Reply::ndjson(STREAM, 4096)).await;
    let llm = make(&server.url, "qwen3:8b").with_reasoning_effort(ReasoningEffort::High);
    collect(
        llm.stream(request(false), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    let body = server.request(0).json();
    assert!(body.get("think").is_none());
    assert_eq!(
        body["options"]["num_predict"],
        600 + THINKING_HEADROOM_TOKENS
    );

    // gpt-oss cannot disable thinking: it gets an effort level instead.
    let server = MockServer::with_reply(Reply::ndjson(STREAM, 4096)).await;
    let llm = make(&server.url, "gpt-oss:20b");
    collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(server.request(0).json()["think"], "low");
}

#[tokio::test]
async fn images_are_sent_as_base64_on_the_message() {
    let server = MockServer::with_reply(Reply::ndjson(STREAM, 4096)).await;
    let llm = make(&server.url, "llava:13b");
    assert!(llm.supports_images());
    let mut r = request(true);
    r.messages = vec![
        ChatMessage::user("What is on screen?").with_image(ImageInput {
            mime_type: "image/png".into(),
            data_base64: "iVBORw0KGgo=".into(),
        }),
    ];
    collect(llm.stream(r, CancellationToken::new()).await.unwrap()).await;
    let body = server.request(0).json();
    assert_eq!(body["messages"][1]["content"], "What is on screen?");
    assert_eq!(body["messages"][1]["images"], json!(["iVBORw0KGgo="]));
    assert!(
        !OllamaLlm::new(Endpoint::new(&server.url), "llama3.2")
            .unwrap()
            .supports_images()
    );
}

#[tokio::test]
async fn base_url_with_v1_or_api_suffix_targets_the_native_api() {
    for suffix in ["/v1", "/v1/", "/api", ""] {
        let server = MockServer::with_reply(Reply::ndjson(STREAM, 4096)).await;
        let llm = make(&format!("{}{suffix}", server.url), "qwen3:8b");
        collect(
            llm.stream(request(true), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(server.request(0).path, "/api/chat", "suffix {suffix:?}");
    }
}

#[tokio::test]
async fn inline_think_block_is_removed() {
    let stream = concat!(
        "{\"message\":{\"role\":\"assistant\",\"content\":\"<thi\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"nk>private</thi\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"nk>\\n\\nVisible answer.\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"length\"}\n"
    );
    let server = MockServer::with_reply(Reply::ndjson(stream, 7)).await;
    let llm = make(&server.url, "deepseek-r1:8b");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Visible answer.");
    assert_eq!(
        out.done,
        Some(LlmChunk::Done {
            stop_reason: Some("length".into()),
            usage: None
        })
    );
}

#[tokio::test]
async fn stream_errors_and_truncation() {
    let stream = concat!(
        "{\"message\":{\"role\":\"assistant\",\"content\":\"partial\"},\"done\":false}\n",
        "{\"error\":\"an error was encountered while running the model: unexpected EOF\"}\n"
    );
    let server = MockServer::with_reply(Reply::ndjson(stream, 9)).await;
    let llm = make(&server.url, "qwen3:8b");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "partial");
    assert!(out.done.is_none());
    assert!(
        matches!(&out.error, Some(ProviderError::Other(m)) if m.contains("unexpected EOF")),
        "{:?}",
        out.error
    );

    // The body ends without a `done: true` line.
    let truncated = &STREAM[..STREAM.rfind("{\"model\"").unwrap()];
    let server = MockServer::with_reply(Reply::ndjson(truncated, 4096)).await;
    let llm = make(&server.url, "qwen3:8b");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert!(out.done.is_none());
    assert!(
        matches!(out.error, Some(ProviderError::Network(_))),
        "{:?}",
        out.error
    );
}

#[tokio::test]
async fn http_errors_are_mapped() {
    let cases: Vec<(Reply, ProviderError)> = vec![
        (
            Reply::json(
                404,
                json!({"error": "model \"qwen3:8b\" not found, try pulling it first"}),
            ),
            ProviderError::Http("404 model \"qwen3:8b\" not found, try pulling it first".into()),
        ),
        (
            Reply::json(401, json!({"error": "unauthorized"})),
            ProviderError::Unauthorized("401 unauthorized".into()),
        ),
        (
            Reply::json(403, json!({"error": "forbidden"})),
            ProviderError::Unauthorized("403 forbidden".into()),
        ),
        (
            Reply::json(429, json!({"error": "server busy, please try again"}))
                .with_header("retry-after", "4"),
            ProviderError::RateLimited(
                "429 server busy, please try again (retry after 4 s)".into(),
            ),
        ),
        (
            Reply::json(500, json!({"error": "llama runner process has terminated"})),
            ProviderError::Http("500 llama runner process has terminated".into()),
        ),
        (Reply::text(504, ""), ProviderError::Timeout),
    ];
    for (reply, expected) in cases {
        let server = MockServer::with_reply(reply).await;
        let llm = make(&server.url, "qwen3:8b");
        let err = llm
            .stream(request(true), CancellationToken::new())
            .await
            .err()
            .unwrap();
        assert_eq!(err, expected);
    }
}

#[tokio::test]
async fn rejected_think_field_is_dropped_and_remembered() {
    let server = MockServer::with_replies(vec![
        Reply::json(
            400,
            json!({"error": "\"llama3.2\" does not support thinking"}),
        ),
        Reply::ndjson(STREAM, 4096),
    ])
    .await;
    let llm = make(&server.url, "llama3.2");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Use a token bucket ✓");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].json()["think"], false);
    assert!(requests[1].json().get("think").is_none());
    collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert!(server.request(2).json().get("think").is_none());
}

#[tokio::test]
async fn cancellation_mid_stream_is_prompt() {
    let server = MockServer::with_reply(Reply::drip(
        "application/x-ndjson",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"first\"},\"done\":false}\n",
        "{\"message\":{\"role\":\"assistant\",\"content\":\" more\"},\"done\":false}\n",
        Duration::from_millis(40),
    ))
    .await;
    let llm = make(&server.url, "qwen3:8b");
    let cancel = CancellationToken::new();
    let mut stream = llm.stream(request(true), cancel.clone()).await.unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        LlmChunk::Delta("first".into())
    );
    assert!(matches!(stream.next().await, Some(Ok(LlmChunk::Delta(_)))));
    cancel.cancel();
    let cancelled_at = Instant::now();
    let mut rest = Vec::new();
    while let Some(item) = stream.next().await {
        rest.push(item);
    }
    let elapsed = cancelled_at.elapsed();
    assert!(elapsed < Duration::from_millis(500), "took {elapsed:?}");
    assert_eq!(rest.last(), Some(&Err(ProviderError::Cancelled)));
}

#[tokio::test]
async fn stream_idle_timeout_ends_the_stream() {
    let server = MockServer::with_reply(Reply::drip(
        "application/x-ndjson",
        "{\"message\":{\"role\":\"assistant\",\"content\":\"a\"},\"done\":false}\n",
        "\n",
        Duration::from_secs(5),
    ))
    .await;
    let timeouts = Timeouts {
        stream_idle: Duration::from_millis(300),
        ..Timeouts::default()
    };
    let llm = OllamaLlm::new(Endpoint::new(&server.url).with_timeouts(timeouts), "m").unwrap();
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "a");
    assert_eq!(out.error, Some(ProviderError::Timeout));
}

#[tokio::test]
async fn list_models_and_health_check_use_tags() {
    let server = MockServer::start(|req, _| {
        if req.path == "/api/tags" {
            Reply::json(
                200,
                json!({"models": [
                    {"name": "qwen3:8b", "model": "qwen3:8b", "size": 5_000_000_000u64},
                    {"name": "llama3.2:latest", "model": "llama3.2:latest"},
                    {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest"}
                ]}),
            )
        } else {
            Reply::text(404, "404 page not found")
        }
    })
    .await;
    let ollama = make(&server.url, "qwen3:8b");
    assert_eq!(
        ollama.list_models().await.unwrap(),
        vec!["llama3.2:latest", "nomic-embed-text:latest", "qwen3:8b"]
    );
    ollama.health_check().await.unwrap();
    let req = server.request(0);
    assert_eq!(req.method, "GET");
    assert_eq!(req.path, "/api/tags");

    let port = support::closed_port().await;
    let down = make(&format!("http://127.0.0.1:{port}"), "qwen3:8b");
    let err = down.health_check().await.unwrap_err();
    assert!(
        matches!(&err, ProviderError::Network(m) if m.contains("127.0.0.1")),
        "{err:?}"
    );
}

#[tokio::test]
async fn embeddings_use_api_embed() {
    let server = MockServer::with_reply(Reply::json(
        200,
        json!({"model": "nomic-embed-text", "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}),
    ))
    .await;
    let e = OllamaEmbeddings::new(Endpoint::new(&server.url), "nomic-embed-text").unwrap();
    let info = e.info();
    assert_eq!(info.id, "ollama");
    assert_eq!(info.model, "nomic-embed-text");
    assert_eq!(info.location, ProviderLocation::Local);
    let vectors = e
        .embed(
            vec!["alpha".into(), "beta".into()],
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(vectors.len(), 2);
    assert!((vectors[1][2] - 0.6).abs() < 1e-6);
    let req = server.request(0);
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/api/embed");
    assert_eq!(
        req.json(),
        json!({"model": "nomic-embed-text", "input": ["alpha", "beta"]})
    );
    assert!(
        e.embed(Vec::new(), CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn embeddings_fall_back_to_legacy_endpoint() {
    let server = MockServer::start(|req, _| {
        if req.path == "/api/embed" {
            Reply::text(404, "404 page not found")
        } else {
            let prompt = serde_json::from_slice::<serde_json::Value>(&req.body).unwrap()["prompt"]
                .as_str()
                .unwrap()
                .len() as f64;
            Reply::json(200, json!({"embedding": [prompt, 1.0]}))
        }
    })
    .await;
    let e = OllamaEmbeddings::new(Endpoint::new(&server.url), "nomic-embed-text").unwrap();
    let vectors = e
        .embed(vec!["a".into(), "bbb".into()], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 1.0], vec![3.0, 1.0]]);
    let paths: Vec<String> = server.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(
        paths,
        vec!["/api/embed", "/api/embeddings", "/api/embeddings"]
    );

    // A JSON 404 (missing model) is an error, not a reason to fall back.
    let missing = MockServer::with_reply(Reply::json(
        404,
        json!({"error": "model \"nomic-embed-text\" not found, try pulling it first"}),
    ))
    .await;
    let e = OllamaEmbeddings::new(Endpoint::new(&missing.url), "nomic-embed-text").unwrap();
    let err = e
        .embed(vec!["a".into()], CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ProviderError::Http(m) if m.starts_with("404 model")));
    assert_eq!(missing.requests().len(), 1);
}

#[tokio::test]
async fn embeddings_cancellation_is_prompt() {
    let slow = MockServer::with_reply(
        Reply::json(200, json!({"embeddings": [[1.0]]})).delayed(Duration::from_secs(20)),
    )
    .await;
    let e = OllamaEmbeddings::new(Endpoint::new(&slow.url), "nomic-embed-text").unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    assert_eq!(
        e.embed(vec!["a".into()], cancel).await.unwrap_err(),
        ProviderError::Cancelled
    );
    assert!(started.elapsed() < Duration::from_millis(600));
}

#[test]
fn lan_host_is_remote() {
    let llm = OllamaLlm::new(Endpoint::new("http://192.168.1.20:11434"), "qwen3:8b").unwrap();
    assert_eq!(llm.info().location, ProviderLocation::Remote);
    assert_eq!(llm.info().endpoint.as_deref(), Some("192.168.1.20:11434"));
    let local = OllamaLlm::new(Endpoint::new("http://localhost:11434"), "qwen3:8b").unwrap();
    assert_eq!(local.info().location, ProviderLocation::Local);
    assert_eq!(local.info().display_name, "Ollama");
    assert!(matches!(
        OllamaLlm::new(Endpoint::new("http://127.0.0.1:11434"), "  "),
        Err(ProviderError::NotConfigured(_))
    ));
}
