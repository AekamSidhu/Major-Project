//! Builds the engine's provider set from settings and describes where meeting
//! data goes (data-flow / privacy view, spec §46–§48).

use std::sync::Arc;

use openmeet_core::ipc::{DataFlowNode, DataFlowReport, DataLocation};
use openmeet_core::providers::{
    OcrEngine, ProviderIssue, ProviderSet, SecretResolver, TranscriptionProvider, VisionProvider,
};
use openmeet_core::settings::{RetentionPolicy, Settings, VisionProviderKind};
use openmeet_core::types::{Component, ProviderInfo, ProviderLocation};
use openmeet_engine::VadTranscriptionProvider;
use openmeet_intel::vision::{LlmVisionProvider, OcrVisionProvider};
use openmeet_providers::factory::{self, BuildContext};

/// Builds every provider; failures become `issues` (the engine degrades
/// gracefully, spec §92).
pub fn build(settings: &Settings, secrets: &dyn SecretResolver) -> ProviderSet {
    let ctx = BuildContext { settings, secrets };
    let mut set = ProviderSet::default();

    match factory::build_llm(&ctx) {
        Ok(p) => set.llm = Some(p),
        Err(e) => set.issues.push(ProviderIssue {
            component: Component::Llm,
            message: e.user_message(),
        }),
    }
    if let Some(r) = factory::build_fast_llm(&ctx) {
        match r {
            Ok(p) => set.fast_llm = Some(p),
            Err(e) => tracing::warn!(error = %e, "fast model unavailable; using the main model"),
        }
    }
    match factory::build_transcriber(&ctx) {
        Ok(t) => {
            set.transcription =
                Some(Arc::new(VadTranscriptionProvider::new(t)) as Arc<dyn TranscriptionProvider>)
        }
        Err(e) => set.issues.push(ProviderIssue {
            component: Component::Transcription,
            message: e.user_message(),
        }),
    }

    // Screen understanding: local OCR first, then an optional vision model.
    let vision_kind = settings.ai.vision.provider;
    let want_ocr = match vision_kind {
        VisionProviderKind::Disabled => false,
        VisionProviderKind::OcrOnly => true,
        _ => settings.ai.vision.use_ocr,
    };
    if want_ocr {
        set.ocr = openmeet_platform::ocr::system_ocr();
    }
    match factory::build_vision_llm(&ctx) {
        Ok(Some(llm)) => {
            set.vision = Some(Arc::new(LlmVisionProvider::new(llm)) as Arc<dyn VisionProvider>);
        }
        Ok(None) => {
            if vision_kind == VisionProviderKind::OcrOnly {
                match set.ocr.clone() {
                    Some(ocr) => {
                        set.vision = Some(Arc::new(OcrVisionProvider::new(ocr as Arc<dyn OcrEngine>)) as Arc<dyn VisionProvider>)
                    }
                    None => set.issues.push(ProviderIssue {
                        component: Component::Vision,
                        message: "Local OCR is not available on this system. Choose a vision model for screen context."
                            .to_string(),
                    }),
                }
            }
        }
        Err(e) => set.issues.push(ProviderIssue {
            component: Component::Vision,
            message: e.user_message(),
        }),
    }

    match factory::build_embeddings(&ctx) {
        Ok(e) => set.embeddings = e,
        Err(e) => set.issues.push(ProviderIssue {
            component: Component::Embeddings,
            message: format!("{} Using built-in local embeddings.", e.user_message()),
        }),
    }
    set
}

fn location_of(info: &ProviderInfo) -> DataLocation {
    match info.location {
        ProviderLocation::Local => DataLocation::Local,
        ProviderLocation::Remote => DataLocation::Remote,
        ProviderLocation::Mock => DataLocation::Mock,
    }
}

fn node(
    component: Component,
    label: &str,
    info: Option<ProviderInfo>,
    fallback: (DataLocation, &str),
) -> DataFlowNode {
    match info {
        Some(info) => DataFlowNode {
            component,
            label: label.to_string(),
            location: location_of(&info),
            provider: if info.model.is_empty() {
                info.display_name.clone()
            } else {
                format!("{} · {}", info.display_name, info.model)
            },
            endpoint: info.endpoint.clone(),
        },
        None => DataFlowNode {
            component,
            label: label.to_string(),
            location: fallback.0,
            provider: fallback.1.to_string(),
            endpoint: None,
        },
    }
}

/// Describes where meeting data is processed. "No meeting data leaves this
/// device." is only stated when every active component is local.
pub fn data_flow(settings: &Settings, providers: &ProviderSet) -> DataFlowReport {
    let mic = DataFlowNode {
        component: Component::Microphone,
        label: "Microphone".into(),
        location: if settings.audio.microphone_enabled {
            DataLocation::Local
        } else {
            DataLocation::Disabled
        },
        provider: settings
            .audio
            .microphone_device
            .clone()
            .unwrap_or_else(|| "Default input device".into()),
        endpoint: None,
    };
    let system = DataFlowNode {
        component: Component::SystemAudio,
        label: "System audio".into(),
        location: if settings.audio.system_audio_enabled {
            DataLocation::Local
        } else {
            DataLocation::Disabled
        },
        provider: settings
            .audio
            .system_audio_device
            .clone()
            .unwrap_or_else(|| "Default output device".into()),
        endpoint: None,
    };
    let audio_enabled = settings.audio.microphone_enabled || settings.audio.system_audio_enabled;
    let transcription = node(
        Component::Transcription,
        "Transcription",
        providers
            .transcription
            .as_ref()
            .map(|p| p.info())
            .filter(|_| audio_enabled),
        if audio_enabled {
            (DataLocation::Disabled, "Not configured")
        } else {
            (DataLocation::Disabled, "Audio off")
        },
    );
    let context = DataFlowNode {
        component: Component::QuestionDetector,
        label: "Context engine".into(),
        location: DataLocation::Local,
        provider: "OpenMeet (on this device)".into(),
        endpoint: None,
    };
    let llm = node(
        Component::Llm,
        "AI provider",
        providers.llm.as_ref().map(|p| p.info()),
        (DataLocation::Disabled, "Not configured"),
    );
    let screen = DataFlowNode {
        component: Component::Screen,
        label: "Screen".into(),
        location: if settings.screen.enabled {
            DataLocation::Local
        } else {
            DataLocation::Disabled
        },
        provider: settings
            .screen
            .source
            .as_ref()
            .map(|s| s.label.clone())
            .unwrap_or_else(|| "Off (manual capture only)".into()),
        endpoint: None,
    };
    let vision_info = providers
        .vision
        .as_ref()
        .map(|v| v.info())
        .or_else(|| providers.ocr.as_ref().map(|o| o.info()));
    let vision = node(
        Component::Vision,
        "Vision / OCR",
        vision_info,
        (DataLocation::Disabled, "Off"),
    );
    let embeddings = node(
        Component::Embeddings,
        "Embeddings",
        providers.embeddings.as_ref().map(|e| e.info()),
        (DataLocation::Local, "Built-in hashing embeddings"),
    );
    let database = DataFlowNode {
        component: Component::Database,
        label: "Storage".into(),
        location: if settings.privacy.retention == RetentionPolicy::None {
            DataLocation::Disabled
        } else {
            DataLocation::Local
        },
        provider: match settings.privacy.retention {
            RetentionPolicy::None => "Not retained".into(),
            RetentionPolicy::OneDay => "Local SQLite · 1 day".into(),
            RetentionPolicy::SevenDays => "Local SQLite · 7 days".into(),
            RetentionPolicy::ThirtyDays => "Local SQLite · 30 days".into(),
            RetentionPolicy::Forever => "Local SQLite · until deleted".into(),
        },
        endpoint: None,
    };
    let nodes = vec![
        mic,
        system,
        transcription,
        context,
        llm,
        screen,
        vision,
        embeddings,
        database,
    ];
    let remote: Vec<&DataFlowNode> = nodes
        .iter()
        .filter(|n| n.location == DataLocation::Remote)
        .collect();
    let all_local = remote.is_empty();
    let statement = if all_local {
        "No meeting data leaves this device.".to_string()
    } else {
        let parts: Vec<String> = remote
            .iter()
            .map(|n| match &n.endpoint {
                Some(host) => format!("{} ({})", n.label, host),
                None => n.label.clone(),
            })
            .collect();
        format!(
            "Meeting data is sent off this device for: {}.",
            parts.join(", ")
        )
    };
    DataFlowReport {
        nodes,
        local_only_mode: settings.privacy.local_only,
        all_local,
        statement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::providers::StaticSecrets;
    use openmeet_core::settings::{LlmProviderKind, TranscriptionProviderKind};

    #[test]
    fn default_settings_are_local_and_statement_is_honest() {
        let s = Settings::default();
        let set = build(&s, &StaticSecrets::default());
        let report = data_flow(&s, &set);
        assert!(report.all_local, "{:?}", report.nodes);
        assert_eq!(report.statement, "No meeting data leaves this device.");
    }

    #[test]
    fn cloud_llm_is_reported_as_remote() {
        let mut s = Settings::default();
        s.ai.llm.provider = LlmProviderKind::Openai;
        s.ai.llm.base_url = None;
        s.ai.llm.model = "gpt-5-mini".into();
        s.transcription.provider = TranscriptionProviderKind::LocalAi;
        let secrets = StaticSecrets(vec![(
            openmeet_core::ipc::SecretProvider::Openai,
            "sk-test".into(),
        )]);
        let set = build(&s, &secrets);
        let report = data_flow(&s, &set);
        assert!(!report.all_local);
        assert!(report.statement.contains("AI provider"));
        assert!(!report.statement.contains("No meeting data leaves"));
    }

    #[test]
    fn missing_key_is_an_issue_not_a_crash() {
        let mut s = Settings::default();
        s.ai.llm.provider = LlmProviderKind::Anthropic;
        s.ai.llm.base_url = None;
        s.ai.llm.model = "claude-opus-5-5".into();
        let set = build(&s, &StaticSecrets::default());
        assert!(set.llm.is_none());
        assert!(set.issue_for(Component::Llm).is_some());
    }
}
