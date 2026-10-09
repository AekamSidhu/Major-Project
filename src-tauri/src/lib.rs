//! OpenMeet desktop application (Tauri 2).
//!
//! Wires the engine, providers, OS integration and storage together, owns the
//! two windows (secondary main window + always-on-top overlay), the tray and
//! the global hotkeys, and exposes typed commands to the frontend.

mod capture;
mod commands;
mod local_ai;
mod logging;
mod overlay;
mod providers_build;
mod secrets;
mod shortcuts;
mod sink;
mod state;
mod tray;
mod windows;

use std::sync::Arc;

use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::now_ms;
use openmeet_core::settings::Settings;
use openmeet_core::store::MeetingStore;
use openmeet_core::types::{Component, ComponentStatus};
use openmeet_engine::{Engine, EngineDeps};
use openmeet_store::Database;
use tauri::{Manager, RunEvent, WindowEvent};

use crate::secrets::SecretStore;
use crate::sink::TauriSink;
use crate::state::AppState;

const AUTOSTART_ARG: &str = "--autostart";

pub fn run() {
    logging::init();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            windows::show_main(app, None);
        }))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_autostart::Builder::new()
                .arg(AUTOSTART_ARG)
                .build(),
        )
        .setup(|app| {
            setup(app.handle())?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                // OpenMeet lives in the tray: closing a window hides it.
                api.prevent_close();
                let _ = window.hide();
                if window.label() == windows::OVERLAY {
                    if let Some(state) = window.app_handle().try_state::<Arc<AppState>>() {
                        state.overlay.hide();
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::app::get_app_info,
            commands::app::get_settings,
            commands::app::update_settings,
            commands::app::reset_settings,
            commands::app::get_snapshot,
            commands::app::open_main_window,
            commands::app::report_render_latency,
            commands::app::get_diagnostics,
            commands::app::run_benchmark,
            commands::providers::get_api_key_status,
            commands::providers::set_api_key,
            commands::providers::clear_api_key,
            commands::providers::test_provider,
            commands::providers::list_models,
            commands::providers::get_data_flow,
            commands::providers::get_local_ai_status,
            commands::providers::start_local_ai,
            commands::providers::stop_local_ai,
            commands::devices::list_audio_devices,
            commands::devices::start_audio_test,
            commands::devices::stop_audio_test,
            commands::devices::list_screen_sources,
            commands::devices::set_screen_capture,
            commands::devices::capture_screen_now,
            commands::devices::list_displays,
            commands::meeting::start_meeting,
            commands::meeting::pause_meeting,
            commands::meeting::resume_meeting,
            commands::meeting::toggle_pause,
            commands::meeting::stop_meeting,
            commands::meeting::emergency_stop,
            commands::meeting::detect_meeting_apps,
            commands::meeting::focus_meeting_app,
            commands::assistant::generate_answer,
            commands::assistant::generate_followups,
            commands::assistant::ask_openmeet,
            commands::assistant::clarify_question,
            commands::assistant::refine_answer,
            commands::assistant::cancel_request,
            commands::assistant::dismiss_suggestion,
            commands::assistant::pin_suggestion,
            commands::assistant::set_mode,
            commands::assistant::set_answer_style,
            commands::overlay::overlay_show,
            commands::overlay::overlay_hide,
            commands::overlay::overlay_toggle,
            commands::overlay::set_click_through,
            commands::overlay::overlay_set_size,
            commands::overlay::overlay_set_position,
            commands::overlay::overlay_save_position,
            commands::overlay::overlay_move_to_display,
            commands::overlay::overlay_begin_input,
            commands::overlay::overlay_end_input,
            commands::privacy::get_capture_privacy_status,
            commands::privacy::set_capture_exclusion,
            commands::privacy::list_privacy_rules,
            commands::privacy::add_privacy_rule,
            commands::privacy::remove_privacy_rule,
            commands::privacy::suggested_privacy_apps,
            commands::history::list_meetings,
            commands::history::get_meeting,
            commands::history::delete_meeting,
            commands::history::delete_all_data,
            commands::history::export_meeting,
            commands::history::regenerate_summary,
            commands::shortcuts::get_shortcut_status,
            commands::shortcuts::update_shortcut,
            commands::shortcuts::reset_shortcuts,
            commands::dev::list_demo_scenarios,
            commands::dev::start_demo,
            commands::dev::stop_demo,
            commands::dev::simulate,
            commands::dev::clear_simulation,
        ])
        .build(tauri::generate_context!())
        .expect("error while building OpenMeet");

    app.run(|app, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = app.try_state::<Arc<AppState>>() {
                state.capture.stop_all(&state.engine);
                state.local_ai.stop();
            }
        }
    });
}

fn setup(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = app.path().app_data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    // Storage. A broken database must not prevent the assistant from working.
    let (db, db_error) = match Database::open(&data_dir.join("openmeet.db")) {
        Ok(db) => (db, None),
        Err(e) => {
            tracing::error!(error = %e, "database unavailable; using a temporary in-memory database");
            (Database::open_in_memory()?, Some(e.to_string()))
        }
    };
    let db = Arc::new(db);
    let settings = match db.load_settings() {
        Ok(Some(s)) => s.sanitized(),
        Ok(None) => Settings::default(),
        Err(e) => {
            tracing::warn!(error = %e, "settings unreadable; using defaults");
            Settings::default()
        }
    };
    let now = now_ms();
    if let Err(e) = db.recover_interrupted_meetings(now) {
        tracing::warn!(error = %e, "meeting recovery failed");
    }
    match db.apply_retention(settings.privacy.retention, now) {
        Ok(n) if n > 0 => tracing::info!(deleted = n, "retention applied at startup"),
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "retention failed"),
    }

    let secrets = Arc::new(SecretStore::new());
    let providers = providers_build::build(&settings, secrets.as_ref());
    let sink = Arc::new(TauriSink::new(app.clone(), settings.general.notifications));
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone() as Arc<dyn EventSink>,
            store: db.clone() as Arc<dyn MeetingStore>,
            providers,
            settings: settings.clone(),
        },
        tauri::async_runtime::handle().inner().clone(),
    );
    let state = Arc::new(AppState::new(app.clone(), engine, db, secrets, sink.clone(), data_dir));
    app.manage(state.clone());

    // Windows: the main window is secondary; the overlay starts hidden.
    let autostarted = std::env::args().any(|a| a == AUTOSTART_ARG);
    let show_main = !autostarted || !settings.general.onboarding_complete;
    windows::create_main(app, show_main)?;
    windows::create_overlay(app, &settings.overlay)?;
    state.overlay.apply_layout(&settings.overlay);
    if settings.overlay.click_through {
        state.overlay.set_click_through(true);
    }
    let privacy = state.overlay.apply_capture_privacy(settings.privacy.capture_exclusion);
    sink.emit(MeetingEvent::PrivacyStateChanged {
        enabled: privacy.requested,
        supported: privacy.supported,
        active: privacy.active,
    });
    state.engine.set_component_status(
        Component::CapturePrivacy,
        if !privacy.supported {
            ComponentStatus::Unsupported
        } else if privacy.active {
            ComponentStatus::Active
        } else {
            ComponentStatus::Disabled
        },
        Some(privacy.mechanism.clone()),
    );
    state
        .engine
        .set_component_status(Component::Overlay, ComponentStatus::Active, None);
    match &db_error {
        None => state
            .engine
            .set_component_status(Component::Database, ComponentStatus::Active, None),
        Some(_) => state.engine.set_component_status(
            Component::Database,
            ComponentStatus::Error,
            Some("Database unavailable: meetings are kept in memory only".into()),
        ),
    }

    tray::create(app)?;
    state.shortcuts.register_all(app, &settings.shortcuts);
    state.spawn_retention_task();
    tracing::info!(autostarted, "OpenMeet started");
    Ok(())
}
