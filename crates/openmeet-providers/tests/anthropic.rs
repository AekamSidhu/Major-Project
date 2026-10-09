//! Anthropic Messages adapter against a local mock server.

mod support;

use std::time::{Duration, Instant};

use futures::StreamExt;
use openmeet_core::ProviderError;
use openmeet_core::providers::{
    ChatMessage, ImageInput, LlmChunk, LlmProvider, LlmRequest, TokenUsage, complete_text,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::{ProviderLocation, RequestKind};
use openmeet_providers::anthropic::{
    ANTHROPIC_VERSION, AnthropicLlm, REFUSAL_FALLBACK_BETA, THINKING_HEADROOM_TOKENS,
};
use openmeet_providers::http::Endpoint;
use serde_json::json;
use support::{MockServer, Reply, collect};
use tokio_util::sync::CancellationToken;

/// A realistic stream: thinking block (omitted display), a fallback switch
/// marker, two text blocks, pings, usage and stop reason.
const STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":20,\"cache_read_input_tokens\":5,\"output_tokens\":1}}}\n\n",
    "event: ping\n",
    "data: {\"type\":\"ping\"}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"private reasoning\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"fallback\",\"from\":{\"model\":\"claude-opus-5-5\"},\"to\":{\"model\":\"claude-opus-4-8\"}}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"Use a \"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"token bucket ✓\"}}\n\n",
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":42}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n"
);

fn llm(url: &str, model: &str) -> AnthropicLlm {
    AnthropicLlm::new(
        Endpoint::new(url).with_api_key(Some("sk-ant-test-0123456789abcdefghij".into())),
        model,
    )
    .unwrap()
}

fn request(latency: bool) -> LlmRequest {
    let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "System prompt.", "Question?");
    r.temperature = Some(0.3);
    r.max_output_tokens = 1000;
    r.latency_sensitive = latency;
    r
}

#[tokio::test]
async fn opus_5_5_request_rules_and_stream() {
    for chunk_size in [1, 11, 97, 8192] {
        let server = MockServer::with_reply(Reply::sse(STREAM, chunk_size)).await;
        let llm = llm(&format!("{}/v1", server.url), "claude-opus-5-5");
        let out = collect(
            llm.stream(request(true), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.text, "Use a token bucket ✓", "chunk size {chunk_size}");
        assert_eq!(
            out.done,
            Some(LlmChunk::Done {
                stop_reason: Some("end_turn".into()),
                usage: Some(TokenUsage {
                    input_tokens: 25,
                    output_tokens: 42
                }),
            })
        );
        assert!(out.error.is_none());

        let req = server.request(0);
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/messages");
        assert_eq!(
            req.header("x-api-key"),
            Some("sk-ant-test-0123456789abcdefghij")
        );
        assert_eq!(req.header("authorization"), None);
        assert_eq!(req.header("anthropic-version"), Some(ANTHROPIC_VERSION));
        assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("anthropic-beta"), Some(REFUSAL_FALLBACK_BETA));
        assert_eq!(
            req.header("anthropic-beta"),
            Some("server-side-fallback-2026-07-01")
        );
        let body = req.json();
        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "System prompt.");
        assert_eq!(body["max_tokens"], 1000 + THINKING_HEADROOM_TOKENS);
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["output_config"], json!({"effort": "low"}));
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": "Question?"}])
        );
        for absent in ["temperature", "top_p", "top_k", "thinking"] {
            assert!(body.get(absent).is_none(), "{absent} must not be sent");
        }
    }
}

#[tokio::test]
async fn per_model_parameters() {
    struct Case {
        model: &'static str,
        latency: bool,
        temperature: bool,
        effort: Option<&'static str>,
        fallbacks: bool,
        headroom: bool,
    }
    let cases = [
        Case {
            model: "claude-haiku-4-5",
            latency: true,
            temperature: true,
            effort: None,
            fallbacks: false,
            headroom: false,
        },
        Case {
            model: "claude-sonnet-4-6",
            latency: false,
            temperature: true,
            effort: Some("medium"),
            fallbacks: false,
            headroom: false,
        },
        Case {
            model: "claude-opus-4-6",
            latency: true,
            temperature: true,
            effort: Some("low"),
            fallbacks: false,
            headroom: false,
        },
        Case {
            model: "claude-opus-4-7",
            latency: true,
            temperature: false,
            effort: Some("low"),
            fallbacks: false,
            headroom: false,
        },
        Case {
            model: "claude-opus-4-8",
            latency: false,
            temperature: false,
            effort: Some("medium"),
            fallbacks: false,
            headroom: false,
        },
        Case {
            model: "claude-opus-5",
            latency: true,
            temperature: false,
            effort: Some("low"),
            fallbacks: true,
            headroom: true,
        },
        Case {
            model: "claude-sonnet-5",
            latency: true,
            temperature: false,
            effort: Some("low"),
            fallbacks: false,
            headroom: true,
        },
        Case {
            model: "claude-sonnet-5-5",
            latency: false,
            temperature: false,
            effort: Some("medium"),
            fallbacks: true,
            headroom: true,
        },
        Case {
            model: "claude-fable-5-1",
            latency: true,
            temperature: false,
            effort: Some("low"),
            fallbacks: true,
            headroom: true,
        },
        Case {
            model: "claude-mythos-5-1",
            latency: true,
            temperature: false,
            effort: Some("low"),
            fallbacks: false,
            headroom: true,
        },
        Case {
            model: "claude-sonnet-4-5",
            latency: true,
            temperature: true,
            effort: None,
            fallbacks: false,
            headroom: false,
        },
    ];
    for case in cases {
        let server = MockServer::with_reply(Reply::sse(STREAM, 4096)).await;
        let llm = llm(&server.url, case.model).with_reasoning_effort(ReasoningEffort::Medium);
        collect(
            llm.stream(request(case.latency), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        let req = server.request(0);
        let body = req.json();
        let m = case.model;
        assert_eq!(
            body.get("temperature").is_some(),
            case.temperature,
            "{m} temperature"
        );
        if case.temperature {
            assert!(
                (body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6,
                "{m}"
            );
        }
        assert_eq!(
            body.get("output_config")
                .and_then(|o| o.get("effort"))
                .and_then(|e| e.as_str()),
            case.effort,
            "{m} effort"
        );
        assert_eq!(
            body.get("fallbacks").is_some(),
            case.fallbacks,
            "{m} fallbacks body"
        );
        assert_eq!(
            req.header("anthropic-beta").is_some(),
            case.fallbacks,
            "{m} fallbacks header"
        );
        let expected_max = if case.headroom {
            1000 + THINKING_HEADROOM_TOKENS
        } else {
            1000
        };
        assert_eq!(body["max_tokens"], expected_max, "{m} max_tokens");
        assert!(body.get("thinking").is_none(), "{m} thinking");
        assert!(
            body.get("top_p").is_none() && body.get("top_k").is_none(),
            "{m}"
        );
    }
}

#[tokio::test]
async fn images_precede_text() {
    let server = MockServer::with_reply(Reply::sse(STREAM, 4096)).await;
    let llm = llm(&server.url, "claude-opus-5-5");
    assert!(llm.supports_images());
    let mut r = request(true);
    r.messages = vec![
        ChatMessage::user("Describe the diagram").with_image(ImageInput {
            mime_type: "image/png".into(),
            data_base64: "iVBORw0KGgo=".into(),
        }),
    ];
    collect(llm.stream(r, CancellationToken::new()).await.unwrap()).await;
    let body = server.request(0).json();
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
            {"type": "text", "text": "Describe the diagram"}
        ])
    );
}

#[tokio::test]
async fn refusal_is_done_with_refusal_stop_reason() {
    let stream = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":9,\"output_tokens\":0}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\"}},\"usage\":{\"output_tokens\":0}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );
    let server = MockServer::with_reply(Reply::sse(stream, 17)).await;
    let llm = llm(&server.url, "claude-opus-5-5");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "");
    assert_eq!(
        out.done,
        Some(LlmChunk::Done {
            stop_reason: Some("refusal".into()),
            usage: Some(TokenUsage {
                input_tokens: 9,
                output_tokens: 0
            }),
        })
    );
    let err = complete_text(&llm, request(true), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(matches!(err, ProviderError::Refused(_)));
}

#[tokio::test]
async fn error_event_mid_stream() {
    let stream = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
    );
    let server = MockServer::with_reply(Reply::sse(stream, 23)).await;
    let llm = llm(&server.url, "claude-sonnet-5-5");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "partial");
    assert_eq!(out.done, None);
    assert_eq!(
        out.error,
        Some(ProviderError::Http("529 Overloaded".into()))
    );
}

#[tokio::test]
async fn truncated_stream_is_an_error() {
    let truncated = &STREAM[..STREAM.find("event: message_delta").unwrap()];
    let server = MockServer::with_reply(Reply::sse(truncated, 4096)).await;
    let llm = llm(&server.url, "claude-opus-5-5");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Use a token bucket ✓");
    assert!(out.done.is_none());
    assert!(
        matches!(out.error, Some(ProviderError::Network(_))),
        "{:?}",
        out.error
    );
}

#[tokio::test]
async fn http_errors() {
    let key_echo = "sk-ant-test-0123456789abcdefghij";
    let cases: Vec<(Reply, ProviderError)> = vec![
        (
            Reply::json(401, json!({"type": "error", "error": {"type": "authentication_error", "message": format!("invalid x-api-key {key_echo}")}})),
            ProviderError::Unauthorized("401 invalid x-api-key [redacted]".into()),
        ),
        (
            Reply::json(403, json!({"type": "error", "error": {"type": "permission_error", "message": "Your API key does not have permission"}})),
            ProviderError::Unauthorized("403 Your API key does not have permission".into()),
        ),
        (
            Reply::json(429, json!({"type": "error", "error": {"type": "rate_limit_error", "message": "Number of request tokens has exceeded your rate limit"}}))
                .with_header("retry-after", "30"),
            ProviderError::RateLimited(
                "429 Number of request tokens has exceeded your rate limit (retry after 30 s)".into(),
            ),
        ),
        (
            Reply::json(529, json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})),
            ProviderError::Http("529 Overloaded".into()),
        ),
        (
            Reply::json(404, json!({"type": "error", "error": {"type": "not_found_error", "message": "model: claude-nope"}})),
            ProviderError::Http("404 model: claude-nope".into()),
        ),
        (Reply::text(504, ""), ProviderError::Timeout),
    ];
    for (reply, expected) in cases {
        let server = MockServer::with_reply(reply).await;
        let llm = llm(&server.url, "claude-opus-5-5");
        let err = llm
            .stream(request(true), CancellationToken::new())
            .await
            .err()
            .unwrap();
        assert_eq!(err, expected);
    }
}

#[tokio::test]
async fn rejected_fallbacks_are_dropped_and_remembered() {
    let server = MockServer::with_replies(vec![
        Reply::json(
            400,
            json!({"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks: Extra inputs are not permitted"}}),
        ),
        Reply::sse(STREAM, 4096),
    ])
    .await;
    let llm = llm(&server.url, "claude-opus-5-5");
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Use a token bucket ✓");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].header("anthropic-beta").is_some());
    assert!(requests[1].header("anthropic-beta").is_none());
    assert!(requests[1].json().get("fallbacks").is_none());
    // Everything else is unchanged.
    assert_eq!(requests[1].json()["output_config"]["effort"], "low");
}

#[tokio::test]
async fn invalid_conversations_are_rejected_locally() {
    let server = MockServer::with_reply(Reply::sse(STREAM, 4096)).await;
    let llm = llm(&server.url, "claude-opus-5-5");
    let mut prefill = request(true);
    prefill.messages.push(ChatMessage::assistant("Sure, "));
    assert!(matches!(
        llm.stream(prefill, CancellationToken::new()).await,
        Err(ProviderError::Unsupported(_))
    ));
    let mut leading = request(true);
    leading.messages.insert(0, ChatMessage::assistant("Hello"));
    assert!(llm.stream(leading, CancellationToken::new()).await.is_err());
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn cancellation_mid_stream_is_prompt() {
    let initial = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"first\"}}\n\n"
    );
    let drip = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" more\"}}\n\n";
    let server = MockServer::with_reply(Reply::drip(
        "text/event-stream",
        initial,
        drip,
        Duration::from_millis(40),
    ))
    .await;
    let llm = llm(&server.url, "claude-opus-5-5");
    let cancel = CancellationToken::new();
    let mut stream = llm.stream(request(true), cancel.clone()).await.unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        LlmChunk::Delta("first".into())
    );
    // Let a few more deltas arrive, then cancel.
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
async fn list_models_paginates_and_health_check() {
    let server = MockServer::start(|req, _| {
        if req.path.contains("after_id=claude-opus-5") {
            Reply::json(
                200,
                json!({"data": [{"id": "claude-haiku-4-5", "type": "model"}], "has_more": false, "last_id": "claude-haiku-4-5"}),
            )
        } else {
            Reply::json(
                200,
                json!({"data": [{"id": "claude-opus-5-5", "type": "model"}, {"id": "claude-opus-5", "type": "model"}], "has_more": true, "last_id": "claude-opus-5"}),
            )
        }
    })
    .await;
    let llm = llm(&format!("{}/v1/", server.url), "claude-opus-5-5");
    assert_eq!(
        llm.list_models().await.unwrap(),
        vec!["claude-opus-5-5", "claude-opus-5", "claude-haiku-4-5"]
    );
    let requests = server.requests();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[0].path, "/v1/models?limit=1000");
    assert_eq!(
        requests[1].path,
        "/v1/models?limit=1000&after_id=claude-opus-5"
    );
    assert!(requests[0].header("x-api-key").is_some());
    assert_eq!(requests[0].header("anthropic-version"), Some("2023-06-01"));
    llm.health_check().await.unwrap();

    let denied = MockServer::with_reply(Reply::json(
        401,
        json!({"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
    ))
    .await;
    let llm = AnthropicLlm::new(
        Endpoint::new(&denied.url).with_api_key(Some("bad".into())),
        "",
    )
    .unwrap();
    assert_eq!(llm.model(), "claude-opus-5-5");
    assert!(matches!(
        llm.health_check().await,
        Err(ProviderError::Unauthorized(_))
    ));
}

#[test]
fn info_reflects_endpoint() {
    let remote = AnthropicLlm::new(
        Endpoint::new("https://api.anthropic.com").with_api_key(Some("k".into())),
        "claude-opus-5-5",
    )
    .unwrap();
    let info = remote.info();
    assert_eq!(info.id, "anthropic");
    assert_eq!(info.location, ProviderLocation::Remote);
    assert_eq!(info.endpoint.as_deref(), Some("api.anthropic.com"));
    let local =
        AnthropicLlm::new(Endpoint::new("http://127.0.0.1:4000"), "claude-opus-5-5").unwrap();
    assert_eq!(local.info().location, ProviderLocation::Local);
}
