use openmeet_core::CoreError;
use openmeet_core::ipc::{MeetingAppInfo, StartMeetingRequest};
use openmeet_core::types::{Meeting, MeetingState};

use super::{AppStateRef, CmdResult};
use crate::state::blocking;

fn validate_start(req: &StartMeetingRequest) -> CmdResult<()> {
    if let Some(t) = &req.title {
        if t.chars().count() > 200 {
            return Err(CoreError::InvalidInput("The title is too long.".into()));
        }
    }
    if let Some(id) = &req.focus_app_id {
        if id.len() > 64
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':')
        {
            return Err(CoreError::InvalidInput(
                "Invalid meeting application.".into(),
            ));
        }
    }
    if let Some(s) = &req.screen {
        if s.id.len() > 256 || s.label.chars().count() > 300 {
            return Err(CoreError::InvalidInput("Invalid screen source.".into()));
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn start_meeting(
    state: AppStateRef<'_>,
    request: StartMeetingRequest,
) -> CmdResult<Meeting> {
    validate_start(&request)?;
    crate::tray::set_tooltip(&state.app, "OpenMeet — meeting in progress");
    let r = state.inner().start_meeting(request).await;
    if r.is_err() {
        crate::tray::set_tooltip(&state.app, "OpenMeet");
    }
    r
}

#[tauri::command]
pub async fn pause_meeting(state: AppStateRef<'_>) -> CmdResult<()> {
    state.inner().pause().await?;
    crate::tray::set_tooltip(&state.app, "OpenMeet — paused");
    Ok(())
}

#[tauri::command]
pub async fn resume_meeting(state: AppStateRef<'_>) -> CmdResult<()> {
    state.inner().resume().await?;
    crate::tray::set_tooltip(&state.app, "OpenMeet — meeting in progress");
    Ok(())
}

#[tauri::command]
pub async fn toggle_pause(state: AppStateRef<'_>) -> CmdResult<MeetingState> {
    let s = state.inner().toggle_pause().await?;
    crate::tray::set_tooltip(
        &state.app,
        if s == MeetingState::Paused {
            "OpenMeet — paused"
        } else {
            "OpenMeet — meeting in progress"
        },
    );
    Ok(s)
}

#[tauri::command]
pub async fn stop_meeting(state: AppStateRef<'_>) -> CmdResult<()> {
    let r = state.inner().stop_meeting().await;
    crate::tray::set_tooltip(&state.app, "OpenMeet");
    r
}

#[tauri::command]
pub async fn emergency_stop(state: AppStateRef<'_>) -> CmdResult<()> {
    state.emergency_stop();
    Ok(())
}

#[tauri::command]
pub async fn detect_meeting_apps() -> CmdResult<Vec<MeetingAppInfo>> {
    blocking(|| Ok(openmeet_platform::apps::detect_meeting_apps())).await
}

#[tauri::command]
pub async fn focus_meeting_app(window_id: String) -> CmdResult<()> {
    if window_id.is_empty() || window_id.len() > 64 {
        return Err(CoreError::InvalidInput("Invalid window.".into()));
    }
    blocking(move || openmeet_platform::focus::focus_window(&window_id).map_err(CoreError::from))
        .await
}
