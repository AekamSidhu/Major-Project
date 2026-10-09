//! System tray / menu bar (spec §69).

use std::sync::Arc;

use openmeet_core::events::NoticeLevel;
use openmeet_core::types::Mode;
use tauri::menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager};

use crate::state::AppState;
use crate::windows;

pub const TRAY_ID: &str = "openmeet";

pub fn create(app: &AppHandle) -> tauri::Result<()> {
    let title = MenuItemBuilder::with_id("title", "OpenMeet").enabled(false).build(app)?;
    let start = MenuItemBuilder::with_id("start", "Start Meeting…").build(app)?;
    let pause = MenuItemBuilder::with_id("pause", "Pause").build(app)?;
    let resume = MenuItemBuilder::with_id("resume", "Resume").build(app)?;
    let show_overlay = MenuItemBuilder::with_id("show_overlay", "Show Overlay").build(app)?;
    let hide_overlay = MenuItemBuilder::with_id("hide_overlay", "Hide Overlay").build(app)?;
    let candidate = MenuItemBuilder::with_id("mode_candidate", "Candidate Mode").build(app)?;
    let interviewer = MenuItemBuilder::with_id("mode_interviewer", "Interviewer Mode").build(app)?;
    let settings = MenuItemBuilder::with_id("settings", "Settings…").build(app)?;
    let end = MenuItemBuilder::with_id("end", "End Meeting").build(app)?;
    let stop = MenuItemBuilder::with_id("emergency", "Emergency Stop (stop all processing)").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit OpenMeet").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&title, &PredefinedMenuItem::separator(app)?])
        .items(&[&start, &pause, &resume])
        .item(&PredefinedMenuItem::separator(app)?)
        .items(&[&show_overlay, &hide_overlay])
        .item(&PredefinedMenuItem::separator(app)?)
        .items(&[&candidate, &interviewer])
        .item(&PredefinedMenuItem::separator(app)?)
        .items(&[&settings, &end, &stop])
        .item(&PredefinedMenuItem::separator(app)?)
        .item(&quit)
        .build()?;

    let mut builder = TrayIconBuilder::with_id(TRAY_ID)
        .tooltip("OpenMeet")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| handle_menu(app, event.id().as_ref()))
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                windows::show_main(tray.app_handle(), None);
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

pub fn set_tooltip(app: &AppHandle, text: &str) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_tooltip(Some(text));
    }
}

fn handle_menu(app: &AppHandle, id: &str) {
    if id == "quit" {
        if let Some(state) = app.try_state::<Arc<AppState>>() {
            state.capture.stop_all(&state.engine);
            state.local_ai.stop();
        }
        app.exit(0);
        return;
    }
    let Some(state) = app.try_state::<Arc<AppState>>().map(|s| s.inner().clone()) else {
        return;
    };
    let id = id.to_string();
    tauri::async_runtime::spawn(async move {
        let settings = state.settings();
        let result = match id.as_str() {
            "start" => {
                windows::show_main(&state.app, Some("home"));
                Ok(())
            }
            "pause" => state.pause().await,
            "resume" => state.resume().await,
            "show_overlay" => {
                state.overlay.show(&settings.overlay);
                Ok(())
            }
            "hide_overlay" => {
                state.overlay.hide();
                Ok(())
            }
            "mode_candidate" | "mode_interviewer" => {
                let mode = if id == "mode_candidate" { Mode::Candidate } else { Mode::Interviewer };
                state.engine.set_mode(mode);
                state.modify_settings(|s| s.mode = mode).await.map(|_| ())
            }
            "settings" => {
                windows::show_main(&state.app, Some("settings/ai"));
                Ok(())
            }
            "end" => state.stop_meeting().await,
            "emergency" => {
                state.emergency_stop();
                Ok(())
            }
            _ => Ok(()),
        };
        if let Err(e) = result {
            state.notice(NoticeLevel::Warning, e.to_string());
        }
    });
}
