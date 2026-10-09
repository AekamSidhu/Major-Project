use std::time::{Duration, Instant};

use openmeet_core::CoreError;
use openmeet_core::ipc::{
    ApiKeyStatus, DataFlowReport, LocalAiStatus, ProviderTestResult, SecretProvider,
};
use openmeet_core::types::Component;
use openmeet_providers::factory::{self, BuildContext};
use tokio_util::sync::CancellationToken;

use super::{AppStateRef, CmdResult};
use crate::state::blocking;

const TEST_TIMEOUT: Duration = Duration::from_secs(20);

#[tauri::command]
pub async fn get_api_key_status(state: AppStateRef<'_>) -> CmdResult<Vec<ApiKeyStatus>> {
    let secrets = state.secrets.clone();
    blocking(move || Ok(secrets.all_status())).await
}

#[tauri::command]
pub async fn set_api_key(
    state: AppStateRef<'_>,
    provider: SecretProvider,
    key: String,
) -> CmdResult<ApiKeyStatus> {
    let secrets = state.secrets.clone();
    let status = blocking(move || secrets.set(provider, &key)).await?;
    state.rebuild_providers().await;
    Ok(status)
}

#[tauri::command]
pub async fn clear_api_key(
    state: AppStateRef<'_>,
    provider: SecretProvider,
) -> CmdResult<ApiKeyStatus> {
    let secrets = state.secrets.clone();
    let status = blocking(move || secrets.clear(provider)).await?;
    state.rebuild_providers().await;
    Ok(status)
}

fn result(
    component: Component,
    started: Instant,
    outcome: Result<String, String>,
    info: Option<openmeet_core::types::ProviderInfo>,
) -> ProviderTestResult {
    let latency = started.elapsed().as_millis() as u64;
    match outcome {
        Ok(message) => ProviderTestResult {
            component,
            ok: true,
            latency_ms: Some(latency),
            message,
            info,
        },
        Err(message) => ProviderTestResult {
            component,
            ok: false,
            latency_ms: None,
            message,
            info,
        },
    }
}

#[tauri::command]
pub async fn test_provider(
    state: AppStateRef<'_>,
    component: Component,
) -> CmdResult<ProviderTestResult> {
    let providers = state.engine.provider_set();
    let started = Instant::now();
    let issue = providers.issue_for(component).map(|i| i.message.clone());
    let timed = |r: Result<
        Result<(), openmeet_core::ProviderError>,
        tokio::time::error::Elapsed,
    >| match r {
        Ok(Ok(())) => Ok("Connected.".to_string()),
        Ok(Err(e)) => Err(e.user_message()),
        Err(_) => Err("The provider did not respond in time.".to_string()),
    };
    let out = match component {
        Component::Llm => match providers.llm.clone() {
            Some(llm) => {
                let info = llm.info();
                let r = timed(tokio::time::timeout(TEST_TIMEOUT, llm.health_check()).await);
                result(component, started, r, Some(info))
            }
            None => result(
                component,
                started,
                Err(issue.unwrap_or_else(|| "No AI provider is configured.".into())),
                None,
            ),
        },
        Component::Transcription => match providers.transcription.clone() {
            Some(t) => {
                let info = t.info();
                let r = timed(tokio::time::timeout(TEST_TIMEOUT, t.health_check()).await);
                result(component, started, r, Some(info))
            }
            None => result(
                component,
                started,
                Err(issue.unwrap_or_else(|| "No transcription provider is configured.".into())),
                None,
            ),
        },
        Component::Embeddings => {
            let Some(e) = providers.embeddings.clone() else {
                return Ok(result(
                    component,
                    started,
                    Ok("Built-in local embeddings (always available).".into()),
                    None,
                ));
            };
            let info = e.info();
            let r = match tokio::time::timeout(
                TEST_TIMEOUT,
                e.embed(
                    vec!["OpenMeet connection test".into()],
                    CancellationToken::new(),
                ),
            )
            .await
            {
                Ok(Ok(v)) if v.first().is_some_and(|x| !x.is_empty()) => {
                    Ok(format!("Connected ({} dimensions).", v[0].len()))
                }
                Ok(Ok(_)) => Err("The provider returned no embedding.".into()),
                Ok(Err(e)) => Err(e.user_message()),
                Err(_) => Err("The provider did not respond in time.".into()),
            };
            result(component, started, r, Some(info))
        }
        Component::Vision => {
            let settings = state.settings();
            let secrets = state.secrets.clone();
            let vision_llm = blocking(move || {
                factory::build_vision_llm(&BuildContext {
                    settings: &settings,
                    secrets: secrets.as_ref(),
                })
                .map_err(|e| CoreError::Provider(e.user_message()))
            })
            .await;
            match vision_llm {
                Ok(Some(llm)) => {
                    let info = llm.info();
                    let r = timed(tokio::time::timeout(TEST_TIMEOUT, llm.health_check()).await);
                    result(component, started, r, Some(info))
                }
                Ok(None) => match providers.ocr.clone() {
                    Some(ocr) => result(
                        component,
                        started,
                        Ok("Local OCR is available.".into()),
                        Some(ocr.info()),
                    ),
                    None => result(
                        component,
                        started,
                        Err(issue.unwrap_or_else(|| "Screen understanding is disabled.".into())),
                        None,
                    ),
                },
                Err(e) => result(component, started, Err(e.to_string()), None),
            }
        }
        other => {
            return Err(CoreError::InvalidInput(format!(
                "{} cannot be tested here.",
                other.label()
            )));
        }
    };
    Ok(out)
}

#[tauri::command]
pub async fn list_models(state: AppStateRef<'_>, component: Component) -> CmdResult<Vec<String>> {
    let llm = match component {
        Component::Llm => state.engine.provider_set().llm,
        Component::Vision => {
            let settings = state.settings();
            let secrets = state.secrets.clone();
            blocking(move || {
                factory::build_vision_llm(&BuildContext {
                    settings: &settings,
                    secrets: secrets.as_ref(),
                })
                .map_err(|e| CoreError::Provider(e.user_message()))
            })
            .await?
        }
        _ => {
            return Err(CoreError::InvalidInput(
                "Model lists are available for the AI and vision providers.".into(),
            ));
        }
    };
    let llm = llm.ok_or_else(|| CoreError::InvalidState("Configure the provider first.".into()))?;
    let mut models = tokio::time::timeout(TEST_TIMEOUT, llm.list_models())
        .await
        .map_err(|_| CoreError::Provider("The provider did not respond in time.".into()))??;
    models.sort();
    models.dedup();
    Ok(models)
}

#[tauri::command]
pub async fn get_data_flow(state: AppStateRef<'_>) -> CmdResult<DataFlowReport> {
    Ok(crate::providers_build::data_flow(
        &state.settings(),
        &state.engine.provider_set(),
    ))
}

#[tauri::command]
pub async fn get_local_ai_status(state: AppStateRef<'_>) -> CmdResult<LocalAiStatus> {
    Ok(state.local_ai.status(&state.settings()).await)
}

#[tauri::command]
pub async fn start_local_ai(state: AppStateRef<'_>) -> CmdResult<LocalAiStatus> {
    let status = state.local_ai.start(&state.settings()).await?;
    Ok(status)
}

#[tauri::command]
pub async fn stop_local_ai(state: AppStateRef<'_>) -> CmdResult<LocalAiStatus> {
    if state.engine.is_live() && crate::local_ai::sidecar_in_use(&state.settings()) {
        return Err(CoreError::InvalidState(
            "Local transcription is in use by the running meeting.".into(),
        ));
    }
    state.local_ai.stop();
    Ok(state.local_ai.status(&state.settings()).await)
}
