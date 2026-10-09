use openmeet_core::CoreError;
use openmeet_core::ids::now_ms;
use openmeet_core::ipc::{AppInfo, BenchmarkReport, Diagnostics, EngineSnapshot};
use openmeet_core::settings::Settings;
use openmeet_core::types::LatencyStage;
use tauri::AppHandle;

use super::{AppStateRef, CmdResult};
use crate::windows;

#[tauri::command]
pub async fn get_app_info(app: AppHandle, state: AppStateRef<'_>) -> CmdResult<AppInfo> {
    Ok(AppInfo {
        version: app.package_info().version.to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        developer_mode: state.settings().general.developer_mode,
        debug_build: cfg!(debug_assertions),
    })
}

#[tauri::command]
pub async fn get_settings(state: AppStateRef<'_>) -> CmdResult<Settings> {
    Ok(state.settings())
}

#[tauri::command]
pub async fn update_settings(state: AppStateRef<'_>, settings: Settings) -> CmdResult<Settings> {
    state.inner().update_settings(settings).await
}

/// Restores default settings (onboarding stays completed; the user profile is kept).
#[tauri::command]
pub async fn reset_settings(state: AppStateRef<'_>) -> CmdResult<Settings> {
    let current = state.settings();
    let mut s = Settings::default();
    s.general.onboarding_complete = current.general.onboarding_complete;
    s.profile = current.profile;
    state.inner().update_settings(s).await
}

#[tauri::command]
pub async fn get_snapshot(state: AppStateRef<'_>) -> CmdResult<EngineSnapshot> {
    Ok(state.engine.snapshot())
}

#[tauri::command]
pub async fn open_main_window(app: AppHandle, section: Option<String>) -> CmdResult<()> {
    if let Some(s) = &section {
        if s.len() > 64
            || !s
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '/' || c == '-')
        {
            return Err(CoreError::InvalidInput("Invalid section.".into()));
        }
    }
    windows::show_main(&app, section.as_deref());
    Ok(())
}

#[tauri::command]
pub async fn report_render_latency(state: AppStateRef<'_>, ms: f64) -> CmdResult<()> {
    if ms.is_finite() && (0.0..60_000.0).contains(&ms) {
        state.engine.record_latency(LatencyStage::OverlayRender, ms);
    }
    Ok(())
}

#[tauri::command]
pub async fn get_diagnostics(app: AppHandle, state: AppStateRef<'_>) -> CmdResult<Diagnostics> {
    let (memory_mb, cpu_percent) = state.stats.lock().sample();
    let settings = state.settings();
    Ok(Diagnostics {
        components: state.engine.components(),
        latency: state.engine.latency_stats(),
        memory_mb,
        cpu_percent,
        version: app.package_info().version.to_string(),
        platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        developer_mode: settings.general.developer_mode,
        meeting_state: state.engine.meeting_state(),
        mode: state.engine.mode(),
        active_requests: state.engine.active_request_count(),
    })
}

#[tauri::command]
pub async fn run_benchmark(state: AppStateRef<'_>) -> CmdResult<BenchmarkReport> {
    let mut items = state.engine.run_benchmark().await;
    let (memory_mb, cpu_percent) = state.stats.lock().sample();
    items.push(openmeet_core::ipc::BenchmarkItem {
        name: "Process memory".into(),
        value: (memory_mb * 10.0).round() / 10.0,
        unit: "MB".into(),
        note: None,
    });
    items.push(openmeet_core::ipc::BenchmarkItem {
        name: "Process CPU".into(),
        value: (f64::from(cpu_percent) * 10.0).round() / 10.0,
        unit: "%".into(),
        note: Some("Since the previous sample".into()),
    });
    Ok(BenchmarkReport {
        items,
        ran_at: now_ms(),
    })
}
