use openmeet_core::CoreError;
use openmeet_core::ipc::{AudioDevices, DisplayInfo, ScreenSourceInfo, ScreenSourceSpec};
use openmeet_core::types::AudioSourceKind;
use tauri::AppHandle;

use super::{AppStateRef, CmdResult};
use crate::state::blocking;

#[tauri::command]
pub async fn list_audio_devices() -> CmdResult<AudioDevices> {
    blocking(|| openmeet_platform::audio::list_devices().map_err(CoreError::from)).await
}

#[tauri::command]
pub async fn start_audio_test(
    state: AppStateRef<'_>,
    source: AudioSourceKind,
    device: Option<String>,
) -> CmdResult<()> {
    if device.as_ref().is_some_and(|d| d.len() > 512) {
        return Err(CoreError::InvalidInput("Invalid device.".into()));
    }
    state.capture.start_test(source, device)
}

#[tauri::command]
pub async fn stop_audio_test(state: AppStateRef<'_>) -> CmdResult<()> {
    state.capture.stop_test();
    Ok(())
}

#[tauri::command]
pub async fn list_screen_sources(state: AppStateRef<'_>) -> CmdResult<Vec<ScreenSourceInfo>> {
    let db = state.db.clone();
    blocking(move || {
        let rules = db.list_privacy_rules().map_err(CoreError::from)?;
        openmeet_platform::screen::list_sources(&rules).map_err(CoreError::from)
    })
    .await
}

/// Enables/disables screen context (and remembers the source). Takes effect
/// immediately when a meeting is running.
#[tauri::command]
pub async fn set_screen_capture(
    state: AppStateRef<'_>,
    enabled: bool,
    source: Option<ScreenSourceSpec>,
) -> CmdResult<()> {
    if enabled && source.is_none() && state.settings().screen.source.is_none() {
        return Err(CoreError::InvalidInput(
            "Choose what to capture first.".into(),
        ));
    }
    if let Some(s) = &source {
        if s.id.len() > 256 || s.label.chars().count() > 300 {
            return Err(CoreError::InvalidInput("Invalid screen source.".into()));
        }
    }
    state
        .inner()
        .modify_settings(|s| {
            s.screen.enabled = enabled;
            if source.is_some() {
                s.screen.source = source.clone();
            }
        })
        .await?;
    state.inner().sync_screen_capture().await;
    Ok(())
}

#[tauri::command]
pub async fn capture_screen_now(state: AppStateRef<'_>) -> CmdResult<()> {
    state.inner().capture_screen_now().await
}

#[tauri::command]
pub async fn list_displays(app: AppHandle) -> CmdResult<Vec<DisplayInfo>> {
    Ok(crate::windows::list_displays(&app))
}
