//! Global hotkeys (spec §33): registration, conflict detection and dispatch.

use std::collections::BTreeMap;
use std::sync::Arc;

use openmeet_core::events::NoticeLevel;
use openmeet_core::ipc::{GenerateAnswerRequest, ShortcutStatus, UiCommand};
use openmeet_core::settings::{ShortcutAction, ShortcutSettings, normalize_accelerator};
use openmeet_core::types::MeetingState;
use parking_lot::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

use crate::state::AppState;
use crate::windows;

/// Shortcuts known to be used by common meeting apps and browsers. A global
/// hotkey takes precedence over those apps while OpenMeet runs, so we warn.
const KNOWN_CONFLICTS: &[(&str, &str)] = &[
    ("cmdorctrl+shift+m", "Microsoft Teams: mute / unmute"),
    ("cmdorctrl+shift+m", "Chrome / Edge: switch profile"),
    ("cmdorctrl+shift+c", "Chrome / Edge: inspect element"),
    ("cmdorctrl+shift+a", "Microsoft Teams: accept audio call"),
    ("cmdorctrl+shift+a", "Zoom (macOS): mute / unmute"),
    ("cmdorctrl+shift+a", "Slack: all unreads"),
    ("cmdorctrl+shift+s", "Microsoft Teams: accept video call"),
    ("cmdorctrl+shift+s", "Zoom (macOS): share screen"),
    ("cmdorctrl+shift+e", "Microsoft Teams: share screen"),
    ("cmdorctrl+shift+p", "Microsoft Teams: background blur"),
    ("cmdorctrl+shift+p", "Browsers: private window"),
    ("cmdorctrl+shift+space", "Slack: mute / unmute in a huddle"),
    ("cmdorctrl+shift+q", "Firefox: quit"),
    ("cmdorctrl+shift+h", "Microsoft Teams: leave call"),
    ("cmdorctrl+shift+o", "Microsoft Teams: camera on / off"),
    ("cmdorctrl+shift+k", "Microsoft Teams: raise hand"),
    ("cmdorctrl+shift+d", "Microsoft Teams: decline call"),
    ("cmdorctrl+shift+v", "Zoom (macOS): start / stop video"),
];

#[cfg(windows)]
const RESERVED: &[(&str, &str)] = &[(
    "cmdorctrl+shift+escape",
    "Reserved by Windows (Task Manager)",
)];
#[cfg(not(windows))]
const RESERVED: &[(&str, &str)] = &[];

/// Normalizes an accelerator so "CommandOrControl" and its aliases compare
/// equal to the `cmdorctrl` entries above.
fn key_of(acc: &str) -> String {
    normalize_accelerator(acc)
}

pub fn warnings_for(acc: &str) -> Vec<String> {
    let key = key_of(acc);
    let mut out: Vec<String> = KNOWN_CONFLICTS
        .iter()
        .chain(RESERVED.iter())
        .filter(|(k, _)| key_of(k) == key)
        .map(|(_, w)| format!("May conflict with {w}"))
        .collect();
    out.dedup();
    out
}

pub struct ShortcutManager {
    status: Mutex<BTreeMap<ShortcutAction, ShortcutStatus>>,
}

impl Default for ShortcutManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortcutManager {
    pub fn new() -> Self {
        ShortcutManager {
            status: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn status(&self) -> Vec<ShortcutStatus> {
        self.status.lock().values().cloned().collect()
    }

    /// (Re-)registers every shortcut. Failures (e.g. the combination is
    /// already taken by another application) are recorded, never fatal.
    pub fn register_all(&self, app: &AppHandle, settings: &ShortcutSettings) {
        let gs = app.global_shortcut();
        if let Err(e) = gs.unregister_all() {
            tracing::warn!(error = %e, "failed to unregister shortcuts");
        }
        let duplicates = settings.duplicates();
        let mut status = BTreeMap::new();
        for action in ShortcutAction::ALL {
            let accelerator = settings.accelerator(action);
            let mut warnings = warnings_for(&accelerator);
            for (a, b) in &duplicates {
                if *a == action || *b == action {
                    let other = if *a == action { *b } else { *a };
                    warnings.push(format!("Same keys as \"{}\"", other.label()));
                }
            }
            let (registered, error) = if accelerator.trim().is_empty() {
                (false, Some("Not assigned".to_string()))
            } else if duplicates.iter().any(|(_, b)| *b == action) {
                // The first action with these keys wins.
                (
                    false,
                    Some("Already used by another OpenMeet shortcut".to_string()),
                )
            } else {
                let handle = app.clone();
                match gs.on_shortcut(accelerator.as_str(), move |_app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        dispatch(&handle, action);
                    }
                }) {
                    Ok(()) => (true, None),
                    Err(e) => {
                        tracing::warn!(action = ?action, error = %e, "shortcut registration failed");
                        (
                            false,
                            Some("Could not register: the keys may be in use by another application.".to_string()),
                        )
                    }
                }
            };
            status.insert(
                action,
                ShortcutStatus {
                    action,
                    accelerator,
                    registered,
                    error,
                    warnings,
                },
            );
        }
        let failed = status
            .values()
            .filter(|s| !s.registered && s.error.as_deref() != Some("Not assigned"))
            .count();
        *self.status.lock() = status;
        if let Some(state) = app.try_state::<Arc<AppState>>() {
            state.engine.set_component_status(
                openmeet_core::types::Component::Hotkeys,
                if failed == 0 {
                    openmeet_core::types::ComponentStatus::Active
                } else {
                    openmeet_core::types::ComponentStatus::Degraded
                },
                (failed > 0).then(|| format!("{failed} shortcut(s) could not be registered")),
            );
        }
    }
}

/// Executes a shortcut action off the event-loop thread.
pub fn dispatch(app: &AppHandle, action: ShortcutAction) {
    let Some(state) = app.try_state::<Arc<AppState>>().map(|s| s.inner().clone()) else {
        return;
    };
    tauri::async_runtime::spawn(async move {
        let settings = state.settings();
        let result = match action {
            ShortcutAction::ToggleOverlay => {
                state.overlay.toggle(&settings.overlay);
                Ok(())
            }
            ShortcutAction::GenerateAnswer => {
                state.overlay.show(&settings.overlay);
                state
                    .engine
                    .generate_answer(GenerateAnswerRequest::default())
                    .map(|_| ())
            }
            ShortcutAction::GenerateFollowups => {
                state.overlay.show(&settings.overlay);
                state.engine.generate_followups().map(|_| ())
            }
            ShortcutAction::AskAi => {
                state.overlay.show(&settings.overlay);
                state.overlay.begin_input();
                windows::emit_ui_to(&state.app, windows::OVERLAY, UiCommand::FocusAsk);
                Ok(())
            }
            ShortcutAction::ToggleMode => {
                let mode = state.engine.toggle_mode();
                let _ = state.modify_settings(|s| s.mode = mode).await;
                Ok(())
            }
            ShortcutAction::ToggleClickThrough => {
                let enabled = !settings.overlay.click_through;
                state
                    .modify_settings(|s| s.overlay.click_through = enabled)
                    .await
                    .map(|_| ())
            }
            ShortcutAction::CaptureScreen => state.capture_screen_now().await,
            ShortcutAction::PauseResume => {
                if state.engine.meeting_state() == MeetingState::Idle {
                    Ok(())
                } else {
                    state.toggle_pause().await.map(|_| ())
                }
            }
            ShortcutAction::EmergencyStop => {
                state.emergency_stop();
                Ok(())
            }
            ShortcutAction::ExpandHud => {
                state.overlay.show(&settings.overlay);
                windows::emit_ui_to(&state.app, windows::OVERLAY, UiCommand::ToggleExpanded);
                Ok(())
            }
            ShortcutAction::CycleStyle => {
                state.engine.cycle_style();
                Ok(())
            }
        };
        if let Err(e) = result {
            state.notice(NoticeLevel::Warning, e.to_string());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conflict_warnings() {
        let w = warnings_for("CommandOrControl+Shift+M");
        assert!(w.iter().any(|w| w.contains("Teams")));
        assert!(warnings_for("CmdOrCtrl+Shift+Y").is_empty());
        let w = warnings_for("Shift+CmdOrCtrl+Space");
        assert!(w.iter().any(|w| w.contains("Slack")));
    }

    #[test]
    fn defaults_avoid_teams_share_screen() {
        let s = ShortcutSettings::default();
        assert!(warnings_for(&s.accelerator(ShortcutAction::ExpandHud)).is_empty());
        assert!(warnings_for(&s.accelerator(ShortcutAction::CycleStyle)).is_empty());
    }
}
