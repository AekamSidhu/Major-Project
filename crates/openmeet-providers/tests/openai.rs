//! OpenAI, OpenAI-compatible and local AI sidecar adapters against a local
//! mock server.

mod support;

use std::time::{Duration, Instant};

use futures::StreamExt;
use openmeet_core::ProviderError;
use openmeet_core::audio::{AudioClip, decode_wav_pcm16};
use openmeet_core::providers::{
    ChatMessage, EmbeddingProvider, ImageInput, LlmChunk, LlmProvider, LlmRequest, TokenUsage,
    Transcriber, TranscriptionRequest,
};
use openmeet_core::settings::ReasoningEffort;
use openmeet_core::types::{AudioSourceKind, ProviderLocation, RequestKind};
use openmeet_providers::http::{Endpoint, Timeouts};
use openmeet_providers::openai::{
    OpenAiEmbeddings, OpenAiFlavor, OpenAiLlm, OpenAiTranscriber, REASONING_HEADROOM_TOKENS,
};
use serde_json::json;
use support::{MockServer, Reply, closed_port, collect};
use tokio_util::sync::CancellationToken;

const STREAM: &str = concat!(
    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    ": keep-alive\n\n",
    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hidden plan\"}}]}\n\n",
    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Héllo\"}}]}\n\n",
    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" wörld\"}}]}\n\n",
    "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":4,\"total_tokens\":15}}\n\n",
    "data: [DONE]\n\n"
);

fn request(latency: bool) -> LlmRequest {
    let mut r = LlmRequest::new(RequestKind::CandidateAnswer, "You are helpful.", "Hi there");
    r.temperature = Some(0.4);
    r.max_output_tokens = 1000;
    r.latency_sensitive = latency;
    r
}

fn done(stop: &str, input: u32, output: u32) -> LlmChunk {
    LlmChunk::Done {
        stop_reason: Some(stop.to_string()),
        usage: Some(TokenUsage {
            input_tokens: input,
            output_tokens: output,
        }),
    }
}

#[tokio::test]
async fn openai_reasoning_model_request_and_stream() {
    for chunk_size in [1, 7, 64, 4096] {
        let server = MockServer::with_reply(Reply::sse(STREAM, chunk_size)).await;
        let llm = OpenAiLlm::new(
            OpenAiFlavor::OpenAi,
            Endpoint::new(format!("{}/v1/", server.url)).with_api_key(Some("sk-test-key".into())),
            "gpt-5-mini",
        )
        .unwrap();
        let out = collect(
            llm.stream(request(true), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(out.text, "Héllo wörld", "chunk size {chunk_size}");
        assert_eq!(out.done, Some(done("stop", 11, 4)));
        assert!(out.error.is_none());

        let req = server.request(0);
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.header("authorization"), Some("Bearer sk-test-key"));
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("accept"), Some("text/event-stream"));
        assert!(req.header("user-agent").unwrap().starts_with("OpenMeet/"));
        let body = req.json();
        assert_eq!(body["model"], "gpt-5-mini");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(
            body["max_completion_tokens"],
            1000 + REASONING_HEADROOM_TOKENS
        );
        assert!(body.get("max_tokens").is_none());
        assert!(
            body.get("temperature").is_none(),
            "reasoning models reject temperature"
        );
        assert_eq!(body["reasoning_effort"], "minimal");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "You are helpful.");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "Hi there");
    }
}

#[tokio::test]
async fn openai_non_latency_request_uses_configured_effort() {
    let server = MockServer::with_reply(Reply::sse(STREAM, 4096)).await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("k".into())),
        "o4-mini",
    )
    .unwrap()
    .with_reasoning_effort(ReasoningEffort::High);
    collect(
        llm.stream(request(false), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    let body = server.request(0).json();
    assert_eq!(body["reasoning_effort"], "high");
    assert!(body.get("temperature").is_none());
}

#[tokio::test]
async fn compatible_server_request_shape_and_images() {
    let server = MockServer::with_reply(Reply::sse(STREAM, 13)).await;
    // No `/v1` in the configured URL: it is added.
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(&server.url),
        "qwen2.5-vl-7b",
    )
    .unwrap();
    assert!(llm.supports_images());
    let info = llm.info();
    assert_eq!(info.id, "openai_compatible");
    assert_eq!(info.location, ProviderLocation::Local);
    assert_eq!(
        info.endpoint.as_deref(),
        Some(format!("127.0.0.1:{}", server.port).as_str())
    );

    let mut r = request(true);
    r.messages = vec![
        ChatMessage::user("What is on screen?").with_image(ImageInput {
            mime_type: "image/png".into(),
            data_base64: "iVBORw0KGgo=".into(),
        }),
    ];
    let out = collect(llm.stream(r, CancellationToken::new()).await.unwrap()).await;
    assert_eq!(out.text, "Héllo wörld");

    let req = server.request(0);
    assert_eq!(req.path, "/v1/chat/completions");
    assert_eq!(
        req.header("authorization"),
        None,
        "no key → no Authorization header"
    );
    let body = req.json();
    assert_eq!(body["max_tokens"], 1000);
    assert!(body.get("max_completion_tokens").is_none());
    assert!((body["temperature"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    assert!(body.get("reasoning_effort").is_none());
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(
        content[0]["image_url"]["url"],
        "data:image/png;base64,iVBORw0KGgo="
    );
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "What is on screen?");
}

#[tokio::test]
async fn compatible_server_strips_inline_think_blocks() {
    let stream = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"<thi\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"nk>secret reasoning</th\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ink>\\n\\nAnswer.\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let server = MockServer::with_reply(Reply::sse(stream, 5)).await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(&server.url),
        "deepseek-r1",
    )
    .unwrap();
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Answer.");
    assert_eq!(
        out.done,
        Some(LlmChunk::Done {
            stop_reason: Some("length".into()),
            usage: None
        })
    );
}

#[tokio::test]
async fn non_streaming_json_reply_is_accepted() {
    let server = MockServer::with_reply(Reply::json(
        200,
        json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Whole answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        }),
    ))
    .await;
    let llm = OpenAiLlm::new(OpenAiFlavor::Compatible, Endpoint::new(&server.url), "m").unwrap();
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Whole answer");
    assert_eq!(out.done, Some(done("stop", 5, 2)));
}

#[tokio::test]
async fn refusal_is_surfaced_as_stop_reason() {
    let stream = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't help with that.\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let server = MockServer::with_reply(Reply::sse(stream, 9)).await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("k".into())),
        "gpt-4o",
    )
    .unwrap();
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "", "refusal text is not answer text");
    assert_eq!(
        out.done,
        Some(LlmChunk::Done {
            stop_reason: Some("refusal".into()),
            usage: None
        })
    );
    let err =
        openmeet_core::providers::complete_text(&llm, request(true), CancellationToken::new())
            .await
            .unwrap_err();
    assert!(matches!(err, ProviderError::Refused(_)));
}

#[tokio::test]
async fn http_errors_are_mapped_without_leaking_secrets() {
    let secret = "sk-proj-SECRETabc123def456ghi789";
    type Check = Box<dyn Fn(&ProviderError) -> bool>;
    let cases: Vec<(Reply, Check)> = vec![
        (
            Reply::json(
                401,
                json!({"error": {"message": format!("Incorrect API key provided: {secret}"), "type": "invalid_request_error"}}),
            ),
            Box::new(|e| matches!(e, ProviderError::Unauthorized(_))),
        ),
        (
            Reply::json(403, json!({"error": {"message": "Country not supported"}})),
            Box::new(|e| matches!(e, ProviderError::Unauthorized(m) if m.contains("403"))),
        ),
        (
            Reply::json(429, json!({"error": {"message": "Rate limit reached"}}))
                .with_header("retry-after", "7"),
            Box::new(
                |e| matches!(e, ProviderError::RateLimited(m) if m.contains("retry after 7 s") && m.contains("Rate limit reached")),
            ),
        ),
        (
            Reply::json(
                500,
                json!({"error": {"message": "The server had an error"}}),
            ),
            Box::new(|e| *e == ProviderError::Http("500 The server had an error".into())),
        ),
        (
            Reply::text(502, "<html>Bad gateway</html>"),
            Box::new(|e| *e == ProviderError::Http("502 Bad Gateway".into())),
        ),
        (
            Reply::text(504, "gateway timeout"),
            Box::new(|e| *e == ProviderError::Timeout),
        ),
        (
            Reply::text(408, ""),
            Box::new(|e| *e == ProviderError::Timeout),
        ),
    ];
    for (reply, check) in cases {
        let status = reply.status;
        let server = MockServer::with_reply(reply).await;
        let llm = OpenAiLlm::new(
            OpenAiFlavor::OpenAi,
            Endpoint::new(&server.url).with_api_key(Some(secret.into())),
            "gpt-4o",
        )
        .unwrap();
        let err = match llm.stream(request(true), CancellationToken::new()).await {
            Ok(_) => panic!("status {status} must fail"),
            Err(e) => e,
        };
        assert!(check(&err), "status {status}: {err:?}");
        let text = format!("{err} {err:?}");
        assert!(
            !text.contains("SECRET"),
            "status {status} leaked the key: {text}"
        );
        assert!(
            !text.contains("http://"),
            "status {status} leaked the URL: {text}"
        );
    }
}

#[tokio::test]
async fn rejected_parameter_is_dropped_and_retried() {
    let server = MockServer::with_replies(vec![
        Reply::json(
            400,
            json!({"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model.", "param": "temperature"}}),
        ),
        Reply::sse(STREAM, 4096),
    ])
    .await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("k".into())),
        "gpt-4.1-mini",
    )
    .unwrap();
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Héllo wörld");
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].json().get("temperature").is_some());
    assert!(requests[1].json().get("temperature").is_none());
    // Remembered for the next request.
    collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(server.requests().len(), 3);
    assert!(server.request(2).json().get("temperature").is_none());
}

#[tokio::test]
async fn response_timeout_and_connection_refused() {
    let server =
        MockServer::with_reply(Reply::sse(STREAM, 64).delayed(Duration::from_secs(20))).await;
    let timeouts = Timeouts {
        response: Duration::from_millis(300),
        ..Timeouts::default()
    };
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(&server.url).with_timeouts(timeouts),
        "m",
    )
    .unwrap();
    let started = Instant::now();
    let err = llm
        .stream(request(true), CancellationToken::new())
        .await
        .err()
        .unwrap();
    assert_eq!(err, ProviderError::Timeout);
    assert!(started.elapsed() < Duration::from_secs(3));

    let port = closed_port().await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(format!("http://127.0.0.1:{port}/v1")),
        "m",
    )
    .unwrap();
    let err = llm
        .stream(request(true), CancellationToken::new())
        .await
        .err()
        .unwrap();
    assert!(
        matches!(&err, ProviderError::Network(m) if m.contains(&format!("127.0.0.1:{port}"))),
        "{err:?}"
    );
}

#[tokio::test]
async fn stream_idle_timeout() {
    let server = MockServer::with_reply(Reply::drip(
        "text/event-stream",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n",
        ": ping\n\n",
        Duration::from_secs(5),
    ))
    .await;
    let timeouts = Timeouts {
        stream_idle: Duration::from_millis(300),
        ..Timeouts::default()
    };
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(&server.url).with_timeouts(timeouts),
        "m",
    )
    .unwrap();
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
async fn cancellation_mid_stream_is_prompt() {
    let server = MockServer::with_reply(Reply::drip(
        "text/event-stream",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" more\"}}]}\n\n",
        Duration::from_millis(40),
    ))
    .await;
    let llm = OpenAiLlm::new(OpenAiFlavor::Compatible, Endpoint::new(&server.url), "m").unwrap();
    let cancel = CancellationToken::new();
    let mut stream = llm.stream(request(true), cancel.clone()).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first, LlmChunk::Delta("first".into()));
    cancel.cancel();
    let cancelled_at = Instant::now();
    let mut rest = Vec::new();
    while let Some(item) = stream.next().await {
        rest.push(item);
    }
    let elapsed = cancelled_at.elapsed();
    assert!(elapsed < Duration::from_millis(500), "took {elapsed:?}");
    assert_eq!(rest.last(), Some(&Err(ProviderError::Cancelled)));
    assert!(rest.len() <= 2, "{rest:?}");
}

#[tokio::test]
async fn cancellation_before_headers_is_prompt() {
    let server =
        MockServer::with_reply(Reply::sse(STREAM, 64).delayed(Duration::from_secs(20))).await;
    let llm = OpenAiLlm::new(OpenAiFlavor::Compatible, Endpoint::new(&server.url), "m").unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    let err = llm.stream(request(true), cancel).await.err().unwrap();
    assert_eq!(err, ProviderError::Cancelled);
    assert!(started.elapsed() < Duration::from_millis(600));
}

#[tokio::test]
async fn localhost_names_resolve_for_local_endpoints() {
    let server = MockServer::with_reply(Reply::sse(STREAM, 4096)).await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new(format!("http://localhost:{}/v1", server.port)),
        "m",
    )
    .unwrap();
    assert_eq!(llm.info().location, ProviderLocation::Local);
    let out = collect(
        llm.stream(request(true), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "Héllo wörld");
}

#[tokio::test]
async fn list_models_and_health_check() {
    let server = MockServer::with_reply(Reply::json(
        200,
        json!({"object": "list", "data": [
            {"id": "gpt-5-mini"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"},
            {"id": "gpt-4o"}, {"id": "gpt-4o-mini-transcribe"}
        ]}),
    ))
    .await;
    let openai = OpenAiLlm::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("k".into())),
        "gpt-4o",
    )
    .unwrap();
    assert_eq!(
        openai.list_models().await.unwrap(),
        vec!["gpt-4o", "gpt-5-mini"]
    );
    openai.health_check().await.unwrap();
    let req = server.request(0);
    assert_eq!(req.method, "GET");
    assert_eq!(req.path, "/v1/models");
    assert_eq!(req.header("authorization"), Some("Bearer k"));

    let compat = OpenAiLlm::new(OpenAiFlavor::Compatible, Endpoint::new(&server.url), "m").unwrap();
    assert_eq!(compat.list_models().await.unwrap().len(), 5);

    let unauthorized = MockServer::with_reply(Reply::json(
        401,
        json!({"error": {"message": "Invalid key"}}),
    ))
    .await;
    let llm = OpenAiLlm::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&unauthorized.url).with_api_key(Some("bad".into())),
        "gpt-4o",
    )
    .unwrap();
    assert!(matches!(
        llm.health_check().await,
        Err(ProviderError::Unauthorized(_))
    ));
    assert!(matches!(
        llm.list_models().await,
        Err(ProviderError::Unauthorized(_))
    ));
}

fn clip(ms: usize) -> AudioClip {
    let samples: Vec<f32> = (0..16 * ms)
        .map(|i| ((i as f32) * 0.05).sin() * 0.3)
        .collect();
    AudioClip {
        utterance_id: "utt-1".into(),
        source: AudioSourceKind::System,
        samples,
        sample_rate: 16_000,
        start_ms: 1_000,
        end_ms: 1_000 + ms as i64,
        is_final: true,
    }
}

#[tokio::test]
async fn transcription_multipart_and_confidence() {
    let server = MockServer::with_reply(Reply::json(
        200,
        json!({"text": "  How would you design a rate limiter?  ", "confidence": 0.83}),
    ))
    .await;
    let t = OpenAiTranscriber::new(
        OpenAiFlavor::LocalAi,
        Endpoint::new(format!("{}/v1", server.url)),
        "base.en",
    )
    .unwrap();
    let info = t.info();
    assert_eq!(info.id, "local_ai");
    assert_eq!(info.display_name, "Local AI (faster-whisper)");
    assert_eq!(info.location, ProviderLocation::Local);
    assert_eq!(info.model, "base.en");

    let out = t
        .transcribe(
            TranscriptionRequest {
                clip: clip(500),
                language: Some("EN".into()),
                prompt: Some("Kafka, PostgreSQL".into()),
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(out.text, "How would you design a rate limiter?");
    assert!((out.confidence.unwrap() - 0.83).abs() < 1e-6);

    let req = server.request(0);
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/v1/audio/transcriptions");
    assert!(
        req.header("content-type")
            .unwrap()
            .starts_with("multipart/form-data")
    );
    let parts = req.multipart();
    let field = |name: &str| {
        parts
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("missing field {name}"))
    };
    let file = field("file");
    assert_eq!(file.filename.as_deref(), Some("audio.wav"));
    assert_eq!(file.content_type.as_deref(), Some("audio/wav"));
    let (samples, rate) = decode_wav_pcm16(&file.data).unwrap();
    assert_eq!(rate, 16_000);
    assert_eq!(samples.len(), 16 * 500);
    assert_eq!(field("model").data, b"base.en");
    assert_eq!(field("language").data, b"en");
    assert_eq!(field("prompt").data, b"Kafka, PostgreSQL");
    assert_eq!(field("response_format").data, b"json");
}

#[tokio::test]
async fn transcription_without_optional_fields_and_with_key() {
    let server = MockServer::with_reply(Reply::json(200, json!({"text": "hello"}))).await;
    let t = OpenAiTranscriber::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("sk-x".into())),
        "gpt-4o-mini-transcribe",
    )
    .unwrap();
    assert_eq!(t.info().display_name, "OpenAI");
    let out = t
        .transcribe(
            TranscriptionRequest {
                clip: clip(300),
                language: None,
                prompt: Some("   ".into()),
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(out.text, "hello");
    assert_eq!(out.confidence, None);
    let req = server.request(0);
    assert_eq!(req.header("authorization"), Some("Bearer sk-x"));
    let names: Vec<String> = req.multipart().into_iter().map(|p| p.name).collect();
    assert_eq!(names, vec!["file", "model", "response_format"]);

    // Clips too short to contain speech are not uploaded.
    let empty = t
        .transcribe(
            TranscriptionRequest {
                clip: clip(50),
                language: None,
                prompt: None,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(empty.text, "");
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn transcription_errors_and_cancellation() {
    let server = MockServer::with_reply(Reply::json(
        400,
        json!({"detail": "could not decode audio: ValueError"}),
    ))
    .await;
    let t = OpenAiTranscriber::new(OpenAiFlavor::LocalAi, Endpoint::new(&server.url), "base.en")
        .unwrap();
    let req = || TranscriptionRequest {
        clip: clip(400),
        language: None,
        prompt: None,
    };
    let err = t
        .transcribe(req(), CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(
        err,
        ProviderError::Http("400 could not decode audio: ValueError".into())
    );

    let slow = MockServer::with_reply(
        Reply::json(200, json!({"text": "late"})).delayed(Duration::from_secs(20)),
    )
    .await;
    let t =
        OpenAiTranscriber::new(OpenAiFlavor::LocalAi, Endpoint::new(&slow.url), "base.en").unwrap();
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        trigger.cancel();
    });
    let started = Instant::now();
    assert_eq!(
        t.transcribe(req(), cancel).await.unwrap_err(),
        ProviderError::Cancelled
    );
    assert!(started.elapsed() < Duration::from_millis(600));
}

#[tokio::test]
async fn local_ai_transcriber_health_uses_health_endpoint() {
    let server = MockServer::start(|req, _| {
        if req.path == "/health" {
            Reply::json(200, json!({"status": "ok", "version": "0.1.0", "transcription_model": "base.en", "embedding_model": "BAAI/bge-small-en-v1.5"}))
        } else {
            Reply::text(404, "not found")
        }
    })
    .await;
    let t = OpenAiTranscriber::new(
        OpenAiFlavor::LocalAi,
        Endpoint::new(format!("{}/v1", server.url)),
        "base.en",
    )
    .unwrap();
    t.health_check().await.unwrap();
    assert_eq!(server.request(0).path, "/health");

    let loading = MockServer::with_reply(Reply::json(200, json!({"status": "loading"}))).await;
    let t = OpenAiTranscriber::new(
        OpenAiFlavor::LocalAi,
        Endpoint::new(&loading.url),
        "base.en",
    )
    .unwrap();
    assert!(t.health_check().await.is_err());
}

#[tokio::test]
async fn embeddings_request_order_and_sidecar_default_model() {
    let reply = Reply::json(
        200,
        json!({"object": "list", "data": [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.5]},
            {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.25]}
        ], "model": "text-embedding-3-small"}),
    );
    let server = MockServer::with_reply(reply).await;
    let e = OpenAiEmbeddings::new(
        OpenAiFlavor::OpenAi,
        Endpoint::new(&server.url).with_api_key(Some("k".into())),
        "text-embedding-3-small",
    )
    .unwrap();
    let vectors = e
        .embed(
            vec!["first".into(), "second".into()],
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0, 0.25], vec![0.0, 1.0, 0.5]]);
    let req = server.request(0);
    assert_eq!(req.path, "/v1/embeddings");
    assert_eq!(
        req.json(),
        json!({"model": "text-embedding-3-small", "input": ["first", "second"]})
    );
    assert!(
        e.embed(Vec::new(), CancellationToken::new())
            .await
            .unwrap()
            .is_empty()
    );

    // The sidecar with no model configured: `model` is omitted.
    let local =
        OpenAiEmbeddings::new(OpenAiFlavor::LocalAi, Endpoint::new(&server.url), "").unwrap();
    let info = local.info();
    assert_eq!(info.display_name, "Local AI (fastembed)");
    assert_eq!(info.location, ProviderLocation::Local);
    local
        .embed(vec!["a".into(), "b".into()], CancellationToken::new())
        .await
        .unwrap();
    let body = server.request(1).json();
    assert!(body.get("model").is_none());
    assert_eq!(body["input"], json!(["a", "b"]));

    // A count mismatch is an invalid response.
    let short = MockServer::with_reply(Reply::json(
        200,
        json!({"data": [{"index": 0, "embedding": [1.0]}]}),
    ))
    .await;
    let e =
        OpenAiEmbeddings::new(OpenAiFlavor::Compatible, Endpoint::new(&short.url), "m").unwrap();
    assert!(matches!(
        e.embed(vec!["a".into(), "b".into()], CancellationToken::new())
            .await,
        Err(ProviderError::InvalidResponse(_))
    ));
}

#[test]
fn remote_endpoint_is_reported_remote() {
    let llm = OpenAiLlm::new(
        OpenAiFlavor::Compatible,
        Endpoint::new("http://192.168.1.20:1234/v1"),
        "m",
    )
    .unwrap();
    assert_eq!(llm.info().location, ProviderLocation::Remote);
    assert_eq!(llm.info().endpoint.as_deref(), Some("192.168.1.20:1234"));
}
