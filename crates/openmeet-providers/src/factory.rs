//! Settings → provider instances.
//!
//! Every builder resolves the endpoint (configured base URL or the provider
//! default), then enforces, in this order:
//! 1. a valid http(s) base URL (`NotConfigured` otherwise);
//! 2. local-only mode: a remote endpoint (per
//!    [`location_for_url`]) yields `BlockedByLocalOnly`;
//! 3. credentials: OpenAI and Anthropic need an API key when their endpoint
//!    is remote (`NotConfigured("… API key is not set")`); OpenAI-compatible
//!    servers, Ollama and the local AI sidecar work without one.
//!
//! Keys come from the [`SecretResolver`] and never appear in errors or logs.
//!
//! Base-URL inheritance (so users configure a server once):
//! * vision with the same provider kind as the LLM reuses the LLM base URL
//!   (and, for OpenAI / Anthropic, the LLM model) unless set explicitly;
//! * Ollama embeddings reuse the LLM base URL when the LLM is Ollama;
//! * local AI embeddings reuse the transcription base URL when transcription
//!   also uses the local AI sidecar.
//!
//! [`component_base_url`] reports the URL each builder actually uses.

use std::sync::Arc;

use openmeet_core::ProviderError;
use openmeet_core::ipc::SecretProvider;
use openmeet_core::providers::{
    EmbeddingProvider, LlmProvider, SecretResolver, Transcriber, display_host, location_for_url,
};
use openmeet_core::settings::{
    DEFAULT_ANTHROPIC_URL, DEFAULT_LOCAL_AI_URL, DEFAULT_OLLAMA_URL, DEFAULT_OPENAI_URL,
    EmbeddingProviderKind, LlmProviderKind, ReasoningEffort, Settings, TranscriptionProviderKind,
    VisionProviderKind, default_llm_model, default_transcription_model,
};
use openmeet_core::types::{Component, ProviderLocation};

use crate::anthropic::AnthropicLlm;
use crate::http::{Endpoint, normalize_base_url, validate_endpoint_url};
use crate::ollama::{OllamaEmbeddings, OllamaLlm};
use crate::openai::{OpenAiEmbeddings, OpenAiFlavor, OpenAiLlm, OpenAiTranscriber};

/// Default OpenAI embedding model.
pub const DEFAULT_OPENAI_EMBEDDING_MODEL: &str = "text-embedding-3-small";
/// Default Ollama embedding model.
pub const DEFAULT_OLLAMA_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// Inputs for building providers.
pub struct BuildContext<'a> {
    /// Current settings.
    pub settings: &'a Settings,
    /// API key source.
    pub secrets: &'a dyn SecretResolver,
}

/// A provider kind from settings with a well-known default endpoint.
pub trait ProviderKind: Copy {
    /// Default base URL; `None` when the user must configure one (or the kind
    /// has no HTTP endpoint).
    fn default_base_url(self) -> Option<&'static str>;
}

impl ProviderKind for LlmProviderKind {
    fn default_base_url(self) -> Option<&'static str> {
        match self {
            LlmProviderKind::Openai => Some(DEFAULT_OPENAI_URL),
            LlmProviderKind::Anthropic => Some(DEFAULT_ANTHROPIC_URL),
            LlmProviderKind::Ollama => Some(DEFAULT_OLLAMA_URL),
            LlmProviderKind::OpenaiCompatible => None,
        }
    }
}

impl ProviderKind for VisionProviderKind {
    fn default_base_url(self) -> Option<&'static str> {
        vision_llm_kind(self).and_then(ProviderKind::default_base_url)
    }
}

impl ProviderKind for TranscriptionProviderKind {
    fn default_base_url(self) -> Option<&'static str> {
        match self {
            TranscriptionProviderKind::LocalAi => Some(DEFAULT_LOCAL_AI_URL),
            TranscriptionProviderKind::Openai => Some(DEFAULT_OPENAI_URL),
            TranscriptionProviderKind::OpenaiCompatible => None,
        }
    }
}

impl ProviderKind for EmbeddingProviderKind {
    fn default_base_url(self) -> Option<&'static str> {
        match self {
            EmbeddingProviderKind::LocalHash => None,
            EmbeddingProviderKind::LocalAi => Some(DEFAULT_LOCAL_AI_URL),
            EmbeddingProviderKind::Openai => Some(DEFAULT_OPENAI_URL),
            EmbeddingProviderKind::Ollama => Some(DEFAULT_OLLAMA_URL),
        }
    }
}

const WELL_KNOWN_DEFAULTS: [&str; 4] = [
    DEFAULT_OPENAI_URL,
    DEFAULT_ANTHROPIC_URL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_LOCAL_AI_URL,
];

/// Resolved base URL for a component (defaults applied).
///
/// The configured URL wins (normalized: trimmed, no trailing `/`), except
/// when it is another provider's well-known default left over from switching
/// providers in the UI (e.g. the Ollama URL while the provider is OpenAI); the
/// kind's own default is used then. Returns an empty string when nothing is
/// configured and the kind has no default (OpenAI-compatible servers).
pub fn resolved_base_url<K: ProviderKind>(kind: K, configured: Option<&str>) -> String {
    let default = kind.default_base_url();
    let configured = configured.map(normalize_base_url).filter(|u| !u.is_empty());
    match (configured, default) {
        (Some(url), Some(default)) => {
            let stale = WELL_KNOWN_DEFAULTS
                .iter()
                .any(|d| *d != default && url.eq_ignore_ascii_case(d));
            if stale { default.to_string() } else { url }
        }
        (Some(url), None) => url,
        (None, Some(default)) => default.to_string(),
        (None, None) => String::new(),
    }
}

/// The base URL the builder for `component` uses with these settings, `None`
/// for components without an HTTP endpoint here (disabled / OCR-only vision,
/// built-in hashing embeddings, capture components).
pub fn component_base_url(settings: &Settings, component: Component) -> Option<String> {
    let url = match component {
        Component::Llm => llm_base_url(settings),
        Component::Vision => vision_base_url(settings)?,
        Component::Transcription => transcription_base_url(settings),
        Component::Embeddings => embeddings_base_url(settings)?,
        _ => return None,
    };
    (!url.is_empty()).then_some(url)
}

fn llm_base_url(settings: &Settings) -> String {
    let llm = &settings.ai.llm;
    resolved_base_url(llm.provider, llm.base_url.as_deref())
}

fn vision_llm_kind(kind: VisionProviderKind) -> Option<LlmProviderKind> {
    match kind {
        VisionProviderKind::Disabled | VisionProviderKind::OcrOnly => None,
        VisionProviderKind::Openai => Some(LlmProviderKind::Openai),
        VisionProviderKind::Anthropic => Some(LlmProviderKind::Anthropic),
        VisionProviderKind::Ollama => Some(LlmProviderKind::Ollama),
        VisionProviderKind::OpenaiCompatible => Some(LlmProviderKind::OpenaiCompatible),
    }
}

fn vision_base_url(settings: &Settings) -> Option<String> {
    let vision = &settings.ai.vision;
    let kind = vision_llm_kind(vision.provider)?;
    let configured = non_empty(vision.base_url.as_deref()).or_else(|| {
        (settings.ai.llm.provider == kind)
            .then(|| non_empty(settings.ai.llm.base_url.as_deref()))
            .flatten()
    });
    Some(resolved_base_url(kind, configured))
}

fn transcription_base_url(settings: &Settings) -> String {
    let t = &settings.transcription;
    resolved_base_url(t.provider, t.base_url.as_deref())
}

fn embeddings_base_url(settings: &Settings) -> Option<String> {
    let e = &settings.ai.embeddings;
    let inherited = match e.provider {
        EmbeddingProviderKind::LocalHash => return None,
        EmbeddingProviderKind::Ollama if settings.ai.llm.provider == LlmProviderKind::Ollama => {
            non_empty(settings.ai.llm.base_url.as_deref())
        }
        EmbeddingProviderKind::LocalAi
            if settings.transcription.provider == TranscriptionProviderKind::LocalAi =>
        {
            non_empty(settings.transcription.base_url.as_deref())
        }
        _ => None,
    };
    let configured = non_empty(e.base_url.as_deref()).or(inherited);
    Some(resolved_base_url(e.provider, configured))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

/// Which service an endpoint belongs to (decides the API key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Service {
    OpenAi,
    Anthropic,
    OpenAiCompatible,
    Ollama,
    LocalAi,
}

impl Service {
    fn secret(self) -> Option<SecretProvider> {
        match self {
            Service::OpenAi => Some(SecretProvider::Openai),
            Service::Anthropic => Some(SecretProvider::Anthropic),
            Service::OpenAiCompatible => Some(SecretProvider::OpenaiCompatible),
            Service::Ollama | Service::LocalAi => None,
        }
    }

    /// Remote endpoints of these services cannot work without a key.
    fn key_required_when_remote(self) -> Option<&'static str> {
        match self {
            Service::OpenAi => Some("OpenAI"),
            Service::Anthropic => Some("Anthropic"),
            _ => None,
        }
    }

    fn for_llm(kind: LlmProviderKind) -> Service {
        match kind {
            LlmProviderKind::Openai => Service::OpenAi,
            LlmProviderKind::Anthropic => Service::Anthropic,
            LlmProviderKind::Ollama => Service::Ollama,
            LlmProviderKind::OpenaiCompatible => Service::OpenAiCompatible,
        }
    }
}

/// Validates the URL, enforces local-only mode and resolves the API key.
fn endpoint_for(
    ctx: &BuildContext,
    component: Component,
    service: Service,
    base_url: &str,
) -> Result<Endpoint, ProviderError> {
    let label = component.label();
    if base_url.trim().is_empty() {
        return Err(ProviderError::NotConfigured(format!(
            "{label} base URL is not set"
        )));
    }
    validate_endpoint_url(base_url, label)?;
    let location = location_for_url(base_url);
    if ctx.settings.privacy.local_only && location == ProviderLocation::Remote {
        return Err(ProviderError::BlockedByLocalOnly(format!(
            "the {label} endpoint {} is not on this device",
            display_host(base_url)
        )));
    }
    let api_key = service
        .secret()
        .and_then(|s| ctx.secrets.api_key(s))
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    if api_key.is_none() && location == ProviderLocation::Remote {
        if let Some(name) = service.key_required_when_remote() {
            return Err(ProviderError::NotConfigured(format!(
                "{name} API key is not set"
            )));
        }
    }
    tracing::debug!(
        component = label,
        service = ?service,
        location = ?location,
        host = %display_host(base_url),
        has_key = api_key.is_some(),
        "provider endpoint resolved"
    );
    Ok(Endpoint::new(base_url).with_api_key(api_key))
}

fn build_llm_for(
    ctx: &BuildContext,
    component: Component,
    kind: LlmProviderKind,
    base_url: &str,
    model: &str,
    effort: ReasoningEffort,
    force_images: bool,
) -> Result<Arc<dyn LlmProvider>, ProviderError> {
    let endpoint = endpoint_for(ctx, component, Service::for_llm(kind), base_url)?;
    let provider: Arc<dyn LlmProvider> = match kind {
        LlmProviderKind::Openai | LlmProviderKind::OpenaiCompatible => {
            let flavor = if kind == LlmProviderKind::Openai {
                OpenAiFlavor::OpenAi
            } else {
                OpenAiFlavor::Compatible
            };
            let mut llm = OpenAiLlm::new(flavor, endpoint, model)?.with_reasoning_effort(effort);
            if force_images {
                llm = llm.with_image_support(true);
            }
            Arc::new(llm)
        }
        LlmProviderKind::Anthropic => {
            Arc::new(AnthropicLlm::new(endpoint, model)?.with_reasoning_effort(effort))
        }
        LlmProviderKind::Ollama => {
            let mut llm = OllamaLlm::new(endpoint, model)?.with_reasoning_effort(effort);
            if force_images {
                llm = llm.with_image_support(true);
            }
            Arc::new(llm)
        }
    };
    Ok(provider)
}

fn llm_model(settings: &Settings) -> String {
    let llm = &settings.ai.llm;
    non_empty(Some(&llm.model))
        .unwrap_or_else(|| default_llm_model(llm.provider))
        .to_string()
}

/// The main LLM.
pub fn build_llm(ctx: &BuildContext) -> Result<Arc<dyn LlmProvider>, ProviderError> {
    let llm = &ctx.settings.ai.llm;
    build_llm_for(
        ctx,
        Component::Llm,
        llm.provider,
        &llm_base_url(ctx.settings),
        &llm_model(ctx.settings),
        llm.reasoning_effort,
        false,
    )
}

/// The optional faster model for classification / compression (same provider
/// and endpoint as the main LLM). `None` when no fast model is configured.
pub fn build_fast_llm(ctx: &BuildContext) -> Option<Result<Arc<dyn LlmProvider>, ProviderError>> {
    let llm = &ctx.settings.ai.llm;
    let model = non_empty(llm.fast_model.as_deref())?;
    Some(build_llm_for(
        ctx,
        Component::Llm,
        llm.provider,
        &llm_base_url(ctx.settings),
        model,
        llm.reasoning_effort,
        false,
    ))
}

/// Utterance transcriber (local AI sidecar, OpenAI or an OpenAI-compatible
/// server).
pub fn build_transcriber(ctx: &BuildContext) -> Result<Arc<dyn Transcriber>, ProviderError> {
    let t = &ctx.settings.transcription;
    let (flavor, service) = match t.provider {
        TranscriptionProviderKind::LocalAi => (OpenAiFlavor::LocalAi, Service::LocalAi),
        TranscriptionProviderKind::Openai => (OpenAiFlavor::OpenAi, Service::OpenAi),
        TranscriptionProviderKind::OpenaiCompatible => {
            (OpenAiFlavor::Compatible, Service::OpenAiCompatible)
        }
    };
    let model =
        non_empty(Some(&t.model)).unwrap_or_else(|| default_transcription_model(t.provider));
    let endpoint = endpoint_for(
        ctx,
        Component::Transcription,
        service,
        &transcription_base_url(ctx.settings),
    )?;
    Ok(Arc::new(OpenAiTranscriber::new(flavor, endpoint, model)?))
}

/// Vision-capable LLM for `VisionProviderKind::{Openai, Anthropic, Ollama,
/// OpenaiCompatible}`; `Ok(None)` for `Disabled` / `OcrOnly`.
///
/// Without an explicit vision model, OpenAI and Anthropic reuse the LLM model
/// when the LLM uses the same provider (else the provider's default model);
/// Ollama and OpenAI-compatible servers reuse the LLM model only when its
/// name indicates image support, and otherwise need a vision model.
pub fn build_vision_llm(ctx: &BuildContext) -> Result<Option<Arc<dyn LlmProvider>>, ProviderError> {
    let settings = ctx.settings;
    let vision = &settings.ai.vision;
    let Some(kind) = vision_llm_kind(vision.provider) else {
        return Ok(None);
    };
    let same_as_llm = settings.ai.llm.provider == kind;
    let llm_model = same_as_llm
        .then(|| non_empty(Some(&settings.ai.llm.model)))
        .flatten();
    let model = match non_empty(Some(&vision.model)) {
        Some(m) => m.to_string(),
        None => match kind {
            LlmProviderKind::Openai | LlmProviderKind::Anthropic => llm_model
                .unwrap_or_else(|| default_llm_model(kind))
                .to_string(),
            LlmProviderKind::Ollama => llm_model
                .filter(|m| crate::ollama::model_supports_images(m))
                .map(str::to_string)
                .ok_or_else(vision_model_missing)?,
            LlmProviderKind::OpenaiCompatible => llm_model
                .filter(|m| crate::openai::name_suggests_images(m))
                .map(str::to_string)
                .ok_or_else(vision_model_missing)?,
        },
    };
    let base_url = vision_base_url(settings).unwrap_or_default();
    build_llm_for(
        ctx,
        Component::Vision,
        kind,
        &base_url,
        &model,
        settings.ai.llm.reasoning_effort,
        true,
    )
    .map(Some)
}

fn vision_model_missing() -> ProviderError {
    ProviderError::NotConfigured(
        "Vision model is not set; choose a vision-capable model".to_string(),
    )
}

/// Embedding provider; `Ok(None)` for `EmbeddingProviderKind::LocalHash` (the
/// engine uses the built-in embedder).
pub fn build_embeddings(
    ctx: &BuildContext,
) -> Result<Option<Arc<dyn EmbeddingProvider>>, ProviderError> {
    let e = &ctx.settings.ai.embeddings;
    let Some(base_url) = embeddings_base_url(ctx.settings) else {
        return Ok(None);
    };
    let model = non_empty(Some(&e.model));
    let provider: Arc<dyn EmbeddingProvider> = match e.provider {
        EmbeddingProviderKind::LocalHash => return Ok(None),
        EmbeddingProviderKind::LocalAi => {
            let endpoint = endpoint_for(ctx, Component::Embeddings, Service::LocalAi, &base_url)?;
            Arc::new(OpenAiEmbeddings::new(
                OpenAiFlavor::LocalAi,
                endpoint,
                model.unwrap_or_default(),
            )?)
        }
        EmbeddingProviderKind::Openai => {
            let endpoint = endpoint_for(ctx, Component::Embeddings, Service::OpenAi, &base_url)?;
            Arc::new(OpenAiEmbeddings::new(
                OpenAiFlavor::OpenAi,
                endpoint,
                model.unwrap_or(DEFAULT_OPENAI_EMBEDDING_MODEL),
            )?)
        }
        EmbeddingProviderKind::Ollama => {
            let endpoint = endpoint_for(ctx, Component::Embeddings, Service::Ollama, &base_url)?;
            Arc::new(OllamaEmbeddings::new(
                endpoint,
                model.unwrap_or(DEFAULT_OLLAMA_EMBEDDING_MODEL),
            )?)
        }
    };
    Ok(Some(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_resolution() {
        assert_eq!(
            resolved_base_url(LlmProviderKind::Openai, None),
            DEFAULT_OPENAI_URL
        );
        assert_eq!(
            resolved_base_url(LlmProviderKind::Ollama, Some(" http://10.0.0.2:11434/ ")),
            "http://10.0.0.2:11434"
        );
        assert_eq!(
            resolved_base_url(LlmProviderKind::OpenaiCompatible, None),
            ""
        );
        assert_eq!(
            resolved_base_url(
                LlmProviderKind::OpenaiCompatible,
                Some("http://localhost:1234/v1")
            ),
            "http://localhost:1234/v1"
        );
        // A leftover default of another provider is not used.
        assert_eq!(
            resolved_base_url(LlmProviderKind::Openai, Some(DEFAULT_OLLAMA_URL)),
            DEFAULT_OPENAI_URL
        );
        assert_eq!(
            resolved_base_url(
                TranscriptionProviderKind::Openai,
                Some(DEFAULT_LOCAL_AI_URL)
            ),
            DEFAULT_OPENAI_URL
        );
        // ...but compatible servers keep whatever was configured.
        assert_eq!(
            resolved_base_url(LlmProviderKind::OpenaiCompatible, Some(DEFAULT_OLLAMA_URL)),
            DEFAULT_OLLAMA_URL
        );
        assert_eq!(resolved_base_url(VisionProviderKind::OcrOnly, None), "");
        assert_eq!(
            resolved_base_url(VisionProviderKind::Anthropic, None),
            DEFAULT_ANTHROPIC_URL
        );
        assert_eq!(
            resolved_base_url(EmbeddingProviderKind::LocalHash, None),
            ""
        );
        assert_eq!(
            resolved_base_url(EmbeddingProviderKind::LocalAi, None),
            DEFAULT_LOCAL_AI_URL
        );
    }

    #[test]
    fn component_urls_follow_inheritance() {
        let mut s = Settings::default();
        s.ai.llm.provider = LlmProviderKind::Ollama;
        s.ai.llm.base_url = Some("http://192.168.1.5:11434".into());
        s.ai.vision.provider = VisionProviderKind::Ollama;
        s.ai.embeddings.provider = EmbeddingProviderKind::Ollama;
        assert_eq!(
            component_base_url(&s, Component::Vision).as_deref(),
            Some("http://192.168.1.5:11434")
        );
        assert_eq!(
            component_base_url(&s, Component::Embeddings).as_deref(),
            Some("http://192.168.1.5:11434")
        );
        s.transcription.base_url = Some("http://127.0.0.1:9000".into());
        s.ai.embeddings.provider = EmbeddingProviderKind::LocalAi;
        assert_eq!(
            component_base_url(&s, Component::Embeddings).as_deref(),
            Some("http://127.0.0.1:9000")
        );
        s.ai.embeddings.provider = EmbeddingProviderKind::LocalHash;
        assert_eq!(component_base_url(&s, Component::Embeddings), None);
        s.ai.vision.provider = VisionProviderKind::Disabled;
        assert_eq!(component_base_url(&s, Component::Vision), None);
        assert_eq!(component_base_url(&s, Component::Microphone), None);
    }
}
