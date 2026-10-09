//! Settings → providers: every provider kind, credentials, local-only mode,
//! base-URL inheritance and end-to-end requests through factory-built
//! providers (against a local mock server).

mod support;

use std::sync::Arc;

use openmeet_core::ProviderError;
use openmeet_core::audio::AudioClip;
use openmeet_core::ipc::SecretProvider;
use openmeet_core::providers::{
    EmbeddingProvider, LlmProvider, LlmRequest, StaticSecrets, Transcriber, TranscriptionRequest,
};
use openmeet_core::settings::{
    DEFAULT_ANTHROPIC_URL, DEFAULT_LOCAL_AI_URL, DEFAULT_OLLAMA_URL, DEFAULT_OPENAI_URL,
    EmbeddingProviderKind, LlmProviderKind, Settings, TranscriptionProviderKind,
    VisionProviderKind,
};
use openmeet_core::types::{AudioSourceKind, ProviderLocation, RequestKind};
use openmeet_providers::factory::{
    self, BuildContext, DEFAULT_OLLAMA_EMBEDDING_MODEL, DEFAULT_OPENAI_EMBEDDING_MODEL,
};
use serde_json::json;
use support::{MockServer, Reply, collect};
use tokio_util::sync::CancellationToken;

const OPENAI_KEY: &str = "sk-openai-test-SECRET-0123456789";
const ANTHROPIC_KEY: &str = "sk-ant-test-SECRET-0123456789";
const COMPAT_KEY: &str = "compat-SECRET-key-0123456789";

fn no_secrets() -> StaticSecrets {
    StaticSecrets::default()
}

fn all_secrets() -> StaticSecrets {
    StaticSecrets(vec![
        (SecretProvider::Openai, OPENAI_KEY.into()),
        (SecretProvider::Anthropic, ANTHROPIC_KEY.into()),
        (SecretProvider::OpenaiCompatible, COMPAT_KEY.into()),
    ])
}

fn llm(s: &Settings, secrets: &StaticSecrets) -> Result<Arc<dyn LlmProvider>, ProviderError> {
    factory::build_llm(&BuildContext {
        settings: s,
        secrets,
    })
}

fn transcriber(
    s: &Settings,
    secrets: &StaticSecrets,
) -> Result<Arc<dyn Transcriber>, ProviderError> {
    factory::build_transcriber(&BuildContext {
        settings: s,
        secrets,
    })
}

fn vision(
    s: &Settings,
    secrets: &StaticSecrets,
) -> Result<Option<Arc<dyn LlmProvider>>, ProviderError> {
    factory::build_vision_llm(&BuildContext {
        settings: s,
        secrets,
    })
}

fn embeddings(
    s: &Settings,
    secrets: &StaticSecrets,
) -> Result<Option<Arc<dyn EmbeddingProvider>>, ProviderError> {
    factory::build_embeddings(&BuildContext {
        settings: s,
        secrets,
    })
}

/// Unwraps an expected error (the providers are not `Debug`).
fn expect_err<T>(result: Result<T, ProviderError>) -> ProviderError {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

fn assert_no_secret(error: &ProviderError) {
    let text = format!("{error} {error:?} {}", error.user_message());
    assert!(!text.contains("SECRET"), "error leaks a key: {text}");
}

fn with_llm(provider: LlmProviderKind, base_url: Option<&str>, model: &str) -> Settings {
    let mut s = Settings::default();
    s.ai.llm.provider = provider;
    s.ai.llm.base_url = base_url.map(str::to_string);
    s.ai.llm.model = model.to_string();
    s
}

#[test]
fn default_settings_are_local_first() {
    let s = Settings::default();
    let secrets = no_secrets();
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };

    let l = factory::build_llm(&ctx).unwrap();
    let info = l.info();
    assert_eq!(info.id, "ollama");
    assert_eq!(info.model, "llama3.2");
    assert_eq!(info.location, ProviderLocation::Local);
    assert_eq!(info.endpoint.as_deref(), Some("127.0.0.1:11434"));

    assert!(factory::build_fast_llm(&ctx).is_none());

    let t = factory::build_transcriber(&ctx).unwrap().info();
    assert_eq!(t.id, "local_ai");
    assert_eq!(t.display_name, "Local AI (faster-whisper)");
    assert_eq!(t.model, "base.en");
    assert_eq!(t.location, ProviderLocation::Local);
    assert_eq!(t.endpoint.as_deref(), Some("127.0.0.1:8765"));

    assert!(factory::build_vision_llm(&ctx).unwrap().is_none());
    assert!(factory::build_embeddings(&ctx).unwrap().is_none());
}

#[test]
fn openai_llm_needs_a_key_for_the_remote_api() {
    // The base URL left over from the default Ollama settings is ignored.
    let s = with_llm(
        LlmProviderKind::Openai,
        Some(DEFAULT_OLLAMA_URL),
        "gpt-5-mini",
    );
    let err = expect_err(llm(&s, &no_secrets()));
    assert_eq!(
        err,
        ProviderError::NotConfigured("OpenAI API key is not set".into())
    );

    let p = llm(&s, &all_secrets()).unwrap();
    let info = p.info();
    assert_eq!(info.id, "openai");
    assert_eq!(info.model, "gpt-5-mini");
    assert_eq!(info.location, ProviderLocation::Remote);
    assert_eq!(info.endpoint.as_deref(), Some("api.openai.com"));
    assert!(p.supports_images());

    // A blank key counts as missing.
    let blank = StaticSecrets(vec![(SecretProvider::Openai, "   ".into())]);
    assert!(matches!(
        expect_err(llm(&s, &blank)),
        ProviderError::NotConfigured(_)
    ));
}

#[test]
fn anthropic_llm_needs_a_key_and_defaults_its_model() {
    let s = with_llm(LlmProviderKind::Anthropic, None, "");
    let err = expect_err(llm(&s, &no_secrets()));
    assert_eq!(
        err,
        ProviderError::NotConfigured("Anthropic API key is not set".into())
    );
    // The OpenAI key does not unlock Anthropic.
    let openai_only = StaticSecrets(vec![(SecretProvider::Openai, OPENAI_KEY.into())]);
    assert!(matches!(
        expect_err(llm(&s, &openai_only)),
        ProviderError::NotConfigured(_)
    ));

    let p = llm(&s, &all_secrets()).unwrap();
    let info = p.info();
    assert_eq!(info.id, "anthropic");
    assert_eq!(info.model, "claude-opus-5-5");
    assert_eq!(info.location, ProviderLocation::Remote);
    assert_eq!(info.endpoint.as_deref(), Some("api.anthropic.com"));
    assert!(p.supports_images());
}

#[test]
fn ollama_location_comes_from_the_endpoint() {
    let lan = with_llm(
        LlmProviderKind::Ollama,
        Some("http://192.168.1.20:11434"),
        "qwen3:8b",
    );
    let info = llm(&lan, &no_secrets()).unwrap().info();
    assert_eq!(info.id, "ollama");
    assert_eq!(info.location, ProviderLocation::Remote);
    assert_eq!(info.endpoint.as_deref(), Some("192.168.1.20:11434"));

    let local = with_llm(
        LlmProviderKind::Ollama,
        Some("http://localhost:11434/v1"),
        "qwen3:8b",
    );
    assert_eq!(
        llm(&local, &no_secrets()).unwrap().info().location,
        ProviderLocation::Local
    );

    // No base URL: the default local Ollama endpoint.
    let default = with_llm(LlmProviderKind::Ollama, None, "qwen3:8b");
    assert_eq!(
        llm(&default, &no_secrets())
            .unwrap()
            .info()
            .endpoint
            .as_deref(),
        Some("127.0.0.1:11434")
    );
}

#[test]
fn openai_compatible_needs_a_base_url_but_no_key() {
    let s = with_llm(LlmProviderKind::OpenaiCompatible, None, "local-model");
    let err = expect_err(llm(&s, &no_secrets()));
    assert!(
        matches!(&err, ProviderError::NotConfigured(m) if m.contains("base URL")),
        "{err:?}"
    );

    let s = with_llm(
        LlmProviderKind::OpenaiCompatible,
        Some("http://localhost:1234/v1"),
        "qwen2.5-7b-instruct",
    );
    let info = llm(&s, &no_secrets()).unwrap().info();
    assert_eq!(info.id, "openai_compatible");
    assert_eq!(info.location, ProviderLocation::Local);

    // A remote compatible service without a key is allowed (some need none).
    let s = with_llm(
        LlmProviderKind::OpenaiCompatible,
        Some("https://llm.example.com/v1"),
        "m",
    );
    assert_eq!(
        llm(&s, &no_secrets()).unwrap().info().location,
        ProviderLocation::Remote
    );
}

#[test]
fn invalid_base_urls_are_not_configured_and_not_echoed() {
    for url in [
        "ftp://127.0.0.1:1234",
        "http://user:SECRETpw@127.0.0.1:1234",
        "http://127.0.0.1:1234/v1?key=SECRET",
        "not a url",
    ] {
        let s = with_llm(LlmProviderKind::OpenaiCompatible, Some(url), "m");
        let err = expect_err(llm(&s, &all_secrets()));
        assert!(
            matches!(err, ProviderError::NotConfigured(_)),
            "{url}: {err:?}"
        );
        assert_no_secret(&err);
    }
}

#[test]
fn keys_that_cannot_be_sent_are_rejected_without_echo() {
    let s = with_llm(LlmProviderKind::Openai, None, "gpt-4o");
    let bad = StaticSecrets(vec![(
        SecretProvider::Openai,
        "sk-SECRET\nInjected: x".into(),
    )]);
    let err = expect_err(llm(&s, &bad));
    assert!(matches!(err, ProviderError::NotConfigured(_)), "{err:?}");
    assert_no_secret(&err);
}

#[test]
fn local_only_blocks_every_remote_endpoint() {
    let secrets = all_secrets();
    let blocked = |e: ProviderError| {
        assert!(matches!(&e, ProviderError::BlockedByLocalOnly(_)), "{e:?}");
        assert_no_secret(&e);
        e
    };

    // Local-only is checked before credentials.
    let mut s = with_llm(LlmProviderKind::Openai, None, "gpt-5-mini");
    s.privacy.local_only = true;
    let e = blocked(expect_err(llm(&s, &no_secrets())));
    assert!(e.to_string().contains("api.openai.com"), "{e}");
    blocked(expect_err(llm(&s, &secrets)));

    let mut s = with_llm(LlmProviderKind::Anthropic, None, "claude-opus-5-5");
    s.privacy.local_only = true;
    blocked(expect_err(llm(&s, &secrets)));

    let mut s = with_llm(
        LlmProviderKind::Ollama,
        Some("http://192.168.1.20:11434"),
        "qwen3:8b",
    );
    s.privacy.local_only = true;
    blocked(expect_err(llm(&s, &secrets)));
    s.ai.llm.fast_model = Some("qwen3:1.7b".into());
    blocked(expect_err(
        factory::build_fast_llm(&BuildContext {
            settings: &s,
            secrets: &secrets,
        })
        .unwrap(),
    ));

    // A "localhost" look-alike domain is remote.
    let mut s = with_llm(
        LlmProviderKind::OpenaiCompatible,
        Some("http://localhost.example.com:1234/v1"),
        "m",
    );
    s.privacy.local_only = true;
    blocked(expect_err(llm(&s, &secrets)));

    // Loopback endpoints are fine, for every kind.
    for (kind, url) in [
        (LlmProviderKind::Ollama, "http://127.0.0.1:11434"),
        (
            LlmProviderKind::OpenaiCompatible,
            "http://localhost:1234/v1",
        ),
        (LlmProviderKind::OpenaiCompatible, "http://[::1]:8080"),
        (LlmProviderKind::Openai, "http://127.0.0.1:4000/v1"),
        (LlmProviderKind::Anthropic, "http://127.0.0.1:4001"),
    ] {
        let mut s = with_llm(kind, Some(url), "model-x");
        s.privacy.local_only = true;
        let p = llm(&s, &no_secrets()).unwrap_or_else(|e| panic!("{url}: {e:?}"));
        assert_eq!(p.info().location, ProviderLocation::Local, "{url}");
    }

    // Transcription.
    let mut s = Settings::default();
    s.privacy.local_only = true;
    assert!(
        transcriber(&s, &secrets).is_ok(),
        "local AI sidecar is local"
    );
    s.transcription.provider = TranscriptionProviderKind::Openai;
    s.transcription.base_url = None;
    blocked(expect_err(transcriber(&s, &secrets)));

    // Vision.
    let mut s = Settings::default();
    s.privacy.local_only = true;
    s.ai.vision.provider = VisionProviderKind::Anthropic;
    blocked(expect_err(vision(&s, &secrets)));
    s.ai.vision.provider = VisionProviderKind::OcrOnly;
    assert!(vision(&s, &secrets).unwrap().is_none());

    // Embeddings.
    let mut s = Settings::default();
    s.privacy.local_only = true;
    s.ai.embeddings.provider = EmbeddingProviderKind::Openai;
    blocked(expect_err(embeddings(&s, &secrets)));
    s.ai.embeddings.provider = EmbeddingProviderKind::LocalAi;
    assert!(embeddings(&s, &secrets).unwrap().is_some());
    s.ai.embeddings.provider = EmbeddingProviderKind::LocalHash;
    assert!(embeddings(&s, &secrets).unwrap().is_none());
}

#[test]
fn fast_llm_uses_the_main_provider() {
    let mut s = with_llm(LlmProviderKind::Openai, None, "gpt-5");
    let secrets = all_secrets();
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    assert!(factory::build_fast_llm(&ctx).is_none());

    s.ai.llm.fast_model = Some("  ".into());
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    assert!(
        factory::build_fast_llm(&ctx).is_none(),
        "blank = not configured"
    );

    s.ai.llm.fast_model = Some("gpt-5-nano".into());
    let ctx = BuildContext {
        settings: &s,
        secrets: &secrets,
    };
    let fast = factory::build_fast_llm(&ctx).unwrap().unwrap();
    assert_eq!(fast.info().id, "openai");
    assert_eq!(fast.info().model, "gpt-5-nano");

    let none = no_secrets();
    let ctx = BuildContext {
        settings: &s,
        secrets: &none,
    };
    assert!(matches!(
        expect_err(factory::build_fast_llm(&ctx).unwrap()),
        ProviderError::NotConfigured(_)
    ));
}

#[test]
fn transcriber_for_every_kind() {
    // OpenAI: key required; an empty model gets the provider default; the
    // local AI URL left in the settings is ignored.
    let mut s = Settings::default();
    s.transcription.provider = TranscriptionProviderKind::Openai;
    s.transcription.model = String::new();
    assert_eq!(
        expect_err(transcriber(&s, &no_secrets())),
        ProviderError::NotConfigured("OpenAI API key is not set".into())
    );
    let info = transcriber(&s, &all_secrets()).unwrap().info();
    assert_eq!(info.id, "openai");
    assert_eq!(info.model, "gpt-4o-mini-transcribe");
    assert_eq!(info.location, ProviderLocation::Remote);
    assert_eq!(info.endpoint.as_deref(), Some("api.openai.com"));

    // OpenAI-compatible: needs a URL, not a key.
    s.transcription.provider = TranscriptionProviderKind::OpenaiCompatible;
    s.transcription.base_url = None;
    assert!(matches!(
        expect_err(transcriber(&s, &no_secrets())),
        ProviderError::NotConfigured(_)
    ));
    s.transcription.base_url = Some("http://127.0.0.1:9000/v1".into());
    let info = transcriber(&s, &no_secrets()).unwrap().info();
    assert_eq!(info.id, "openai_compatible");
    assert_eq!(info.model, "whisper-1");
    assert_eq!(info.location, ProviderLocation::Local);

    // Local AI sidecar without a URL uses the default port.
    s.transcription.provider = TranscriptionProviderKind::LocalAi;
    s.transcription.base_url = None;
    s.transcription.model = "small.en".into();
    let info = transcriber(&s, &no_secrets()).unwrap().info();
    assert_eq!(info.id, "local_ai");
    assert_eq!(info.display_name, "Local AI (faster-whisper)");
    assert_eq!(info.model, "small.en");
    assert_eq!(
        info.endpoint.as_deref(),
        Some(DEFAULT_LOCAL_AI_URL.trim_start_matches("http://"))
    );
}

#[test]
fn vision_for_every_kind() {
    let secrets = all_secrets();
    let mut s = Settings::default();
    s.ai.vision.provider = VisionProviderKind::Disabled;
    assert!(vision(&s, &secrets).unwrap().is_none());
    s.ai.vision.provider = VisionProviderKind::OcrOnly;
    assert!(vision(&s, &secrets).unwrap().is_none());

    // Same provider as the LLM: the LLM model is reused.
    let mut s = with_llm(LlmProviderKind::Openai, None, "gpt-4.1-mini");
    s.ai.vision.provider = VisionProviderKind::Openai;
    let v = vision(&s, &secrets).unwrap().unwrap();
    assert_eq!(v.info().model, "gpt-4.1-mini");
    assert!(v.supports_images());
    assert!(matches!(
        expect_err(vision(&s, &no_secrets())),
        ProviderError::NotConfigured(_)
    ));

    // Different provider: that provider's default model.
    let mut s = Settings::default();
    s.ai.vision.provider = VisionProviderKind::Anthropic;
    let v = vision(&s, &secrets).unwrap().unwrap();
    assert_eq!(v.info().id, "anthropic");
    assert_eq!(v.info().model, "claude-opus-5-5");
    assert_eq!(
        expect_err(vision(&s, &no_secrets())),
        ProviderError::NotConfigured("Anthropic API key is not set".into())
    );

    // Ollama: a text-only LLM model is not reused for vision.
    let mut s = with_llm(
        LlmProviderKind::Ollama,
        Some("http://127.0.0.1:11999"),
        "llama3.2",
    );
    s.ai.vision.provider = VisionProviderKind::Ollama;
    assert!(matches!(
        expect_err(vision(&s, &secrets)),
        ProviderError::NotConfigured(m) if m.contains("Vision model")
    ));
    s.ai.vision.model = "llava:13b".into();
    let v = vision(&s, &secrets).unwrap().unwrap();
    assert_eq!(v.info().model, "llava:13b");
    // The LLM's Ollama URL is inherited.
    assert_eq!(v.info().endpoint.as_deref(), Some("127.0.0.1:11999"));
    assert!(v.supports_images());

    // OpenAI-compatible with an explicit vision model whose name gives no
    // hint: the user's choice wins.
    let mut s = with_llm(
        LlmProviderKind::OpenaiCompatible,
        Some("http://localhost:1234/v1"),
        "text-model",
    );
    s.ai.vision.provider = VisionProviderKind::OpenaiCompatible;
    assert!(
        vision(&s, &secrets).is_err(),
        "text-only LLM model is not reused"
    );
    s.ai.vision.model = "my-custom-model".into();
    let v = vision(&s, &secrets).unwrap().unwrap();
    assert!(v.supports_images());
    assert_eq!(v.info().endpoint.as_deref(), Some("localhost:1234"));
}

#[test]
fn embeddings_for_every_kind() {
    let secrets = all_secrets();
    let mut s = Settings::default();
    assert!(embeddings(&s, &secrets).unwrap().is_none(), "LocalHash");

    s.ai.embeddings.provider = EmbeddingProviderKind::LocalAi;
    let info = embeddings(&s, &no_secrets()).unwrap().unwrap().info();
    assert_eq!(info.id, "local_ai");
    assert_eq!(info.display_name, "Local AI (fastembed)");
    assert_eq!(info.location, ProviderLocation::Local);
    // Inherits the sidecar URL configured for transcription.
    s.transcription.base_url = Some("http://127.0.0.1:9876".into());
    let info = embeddings(&s, &no_secrets()).unwrap().unwrap().info();
    assert_eq!(info.endpoint.as_deref(), Some("127.0.0.1:9876"));

    s.ai.embeddings.provider = EmbeddingProviderKind::Openai;
    assert_eq!(
        expect_err(embeddings(&s, &no_secrets())),
        ProviderError::NotConfigured("OpenAI API key is not set".into())
    );
    let info = embeddings(&s, &secrets).unwrap().unwrap().info();
    assert_eq!(info.id, "openai");
    assert_eq!(info.model, DEFAULT_OPENAI_EMBEDDING_MODEL);
    assert_eq!(info.location, ProviderLocation::Remote);

    s.ai.embeddings.provider = EmbeddingProviderKind::Ollama;
    s.ai.llm.base_url = Some("http://127.0.0.1:11500".into());
    let info = embeddings(&s, &no_secrets()).unwrap().unwrap().info();
    assert_eq!(info.id, "ollama");
    assert_eq!(info.model, DEFAULT_OLLAMA_EMBEDDING_MODEL);
    assert_eq!(info.endpoint.as_deref(), Some("127.0.0.1:11500"));
    s.ai.embeddings.model = "mxbai-embed-large".into();
    assert_eq!(
        embeddings(&s, &no_secrets()).unwrap().unwrap().info().model,
        "mxbai-embed-large"
    );
}

#[test]
fn resolved_base_urls() {
    use openmeet_providers::resolved_base_url;
    assert_eq!(
        resolved_base_url(LlmProviderKind::Openai, None),
        DEFAULT_OPENAI_URL
    );
    assert_eq!(
        resolved_base_url(LlmProviderKind::Anthropic, Some(DEFAULT_OLLAMA_URL)),
        DEFAULT_ANTHROPIC_URL
    );
    assert_eq!(
        resolved_base_url(LlmProviderKind::Ollama, Some("http://10.0.0.5:11434/")),
        "http://10.0.0.5:11434"
    );
    assert_eq!(
        resolved_base_url(TranscriptionProviderKind::LocalAi, None),
        DEFAULT_LOCAL_AI_URL
    );
    assert_eq!(
        resolved_base_url(EmbeddingProviderKind::LocalHash, None),
        ""
    );
    assert_eq!(resolved_base_url(VisionProviderKind::Disabled, None), "");
}

// ---------------------------------------------------------------------------
// End to end through factory-built providers
// ---------------------------------------------------------------------------

const OPENAI_STREAM: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n"
);

const ANTHROPIC_STREAM: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n"
);

fn chat_request() -> LlmRequest {
    LlmRequest::new(RequestKind::AskAi, "system", "hello")
}

#[tokio::test]
async fn compatible_key_from_secrets_is_sent_as_bearer() {
    let server = MockServer::with_reply(Reply::sse(OPENAI_STREAM, 4096)).await;
    let s = with_llm(
        LlmProviderKind::OpenaiCompatible,
        Some(format!("{}/v1", server.url).as_str()),
        "local-model",
    );
    let p = llm(&s, &all_secrets()).unwrap();
    let out = collect(
        p.stream(chat_request(), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "ok");
    let req = server.request(0);
    assert_eq!(req.path, "/v1/chat/completions");
    assert_eq!(
        req.header("authorization"),
        Some(format!("Bearer {COMPAT_KEY}").as_str())
    );
    // Compatible servers get `max_tokens`.
    assert!(req.json().get("max_tokens").is_some());

    // Without a key no Authorization header is sent.
    let p = llm(&s, &no_secrets()).unwrap();
    collect(
        p.stream(chat_request(), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(server.request(1).header("authorization"), None);
}

#[tokio::test]
async fn anthropic_built_from_settings_sends_its_headers() {
    let server = MockServer::with_reply(Reply::sse(ANTHROPIC_STREAM, 4096)).await;
    let s = with_llm(
        LlmProviderKind::Anthropic,
        Some(format!("{}/v1", server.url).as_str()),
        "claude-sonnet-5-5",
    );
    let p = llm(&s, &all_secrets()).unwrap();
    assert_eq!(p.info().location, ProviderLocation::Local);
    let out = collect(
        p.stream(chat_request(), CancellationToken::new())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(out.text, "ok");
    let req = server.request(0);
    assert_eq!(req.path, "/v1/messages");
    assert_eq!(req.header("x-api-key"), Some(ANTHROPIC_KEY));
    assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(
        req.header("anthropic-beta"),
        Some("server-side-fallback-2026-07-01")
    );
    let body = req.json();
    assert_eq!(body["fallbacks"], "default");
    assert!(body.get("temperature").is_none());
    assert!(body.get("thinking").is_none());
}

#[tokio::test]
async fn local_ai_transcription_and_embeddings_end_to_end() {
    let server = MockServer::start(|req, _| match req.path.as_str() {
        "/v1/audio/transcriptions" => Reply::json(
            200,
            json!({"text": " Tell me about Kafka. ", "confidence": 0.91}),
        ),
        "/v1/embeddings" => Reply::json(
            200,
            json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.5]}]}),
        ),
        "/health" => Reply::json(
            200,
            json!({"status": "ok", "version": "0.1.0", "transcription_model": "base.en", "embedding_model": "BAAI/bge-small-en-v1.5"}),
        ),
        _ => Reply::text(404, "not found"),
    })
    .await;
    let mut s = Settings::default();
    s.transcription.base_url = Some(server.url.clone());
    s.ai.embeddings.provider = EmbeddingProviderKind::LocalAi;
    s.privacy.local_only = true;

    let t = transcriber(&s, &no_secrets()).unwrap();
    t.health_check().await.unwrap();
    let clip = AudioClip {
        utterance_id: "u1".into(),
        source: AudioSourceKind::System,
        samples: vec![0.1; 16_000],
        sample_rate: 16_000,
        start_ms: 0,
        end_ms: 1000,
        is_final: true,
    };
    let out = t
        .transcribe(
            TranscriptionRequest {
                clip,
                language: s.transcription.language.clone(),
                prompt: None,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(out.text, "Tell me about Kafka.");
    assert!((out.confidence.unwrap() - 0.91).abs() < 1e-6);

    let e = embeddings(&s, &no_secrets()).unwrap().unwrap();
    let vectors = e
        .embed(vec!["kafka".into()], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![0.5, 0.5]]);

    let paths: Vec<String> = server.requests().into_iter().map(|r| r.path).collect();
    assert_eq!(
        paths,
        vec!["/health", "/v1/audio/transcriptions", "/v1/embeddings"]
    );
    let parts = server.request(1).multipart();
    let language = parts.iter().find(|p| p.name == "language").unwrap();
    assert_eq!(language.data, b"en");
}
