//! Demo mode (available to everyone) and developer simulation tools
//! (developer mode only).

use openmeet_core::CoreError;
use openmeet_core::ipc::{DemoScenario, SimulationAction};
use openmeet_core::types::Meeting;

use super::{AppStateRef, CmdResult};

#[tauri::command]
pub async fn list_demo_scenarios(state: AppStateRef<'_>) -> CmdResult<Vec<DemoScenario>> {
    Ok(state.engine.demo_scenarios())
}

#[tauri::command]
pub async fn start_demo(state: AppStateRef<'_>, scenario_id: String) -> CmdResult<Meeting> {
    if scenario_id.len() > 64 {
        return Err(CoreError::InvalidInput("Invalid scenario.".into()));
    }
    let meeting = state.engine.start_demo(&scenario_id).await?;
    let settings = state.settings();
    state.overlay.show(&settings.overlay);
    crate::tray::set_tooltip(&state.app, "OpenMeet — demo meeting");
    Ok(meeting)
}

#[tauri::command]
pub async fn stop_demo(state: AppStateRef<'_>) -> CmdResult<()> {
    let r = state.engine.stop_demo().await;
    crate::tray::set_tooltip(&state.app, "OpenMeet");
    r
}

fn require_dev(state: &AppStateRef<'_>) -> CmdResult<()> {
    if state.settings().general.developer_mode {
        Ok(())
    } else {
        Err(CoreError::PermissionDenied(
            "Enable developer mode to use simulation tools.".into(),
        ))
    }
}

#[tauri::command]
pub async fn simulate(state: AppStateRef<'_>, action: SimulationAction) -> CmdResult<()> {
    require_dev(&state)?;
    let too_long = match &action {
        SimulationAction::Transcript { text, .. }
        | SimulationAction::Question { text }
        | SimulationAction::CandidateAnswer { text }
        | SimulationAction::AiResponse { text } => text.chars().count() > 8000,
        SimulationAction::ScreenChange {
            description,
            visible_text,
        } => {
            description.chars().count() > 4000
                || visible_text
                    .as_ref()
                    .is_some_and(|t| t.chars().count() > 8000)
        }
        SimulationAction::AiFailure { message } => message.chars().count() > 500,
        SimulationAction::NetworkLatency { ms } => *ms > 120_000,
    };
    if too_long {
        return Err(CoreError::InvalidInput(
            "Simulation input is too large.".into(),
        ));
    }
    state.engine.simulate(action)
}

#[tauri::command]
pub async fn clear_simulation(state: AppStateRef<'_>) -> CmdResult<()> {
    state.engine.clear_simulation();
    Ok(())
}
