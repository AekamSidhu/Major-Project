//! Bridges the engine's event bus to the webviews (and to OS notifications
//! for important post-meeting events, spec §64).

use std::sync::atomic::{AtomicBool, Ordering};

use openmeet_core::events::{EventSink, MEETING_EVENT_CHANNEL, MeetingEvent};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_notification::NotificationExt;

pub struct TauriSink {
    app: AppHandle,
    /// Mirrors `settings.general.notifications`.
    notifications: AtomicBool,
}

impl TauriSink {
    pub fn new(app: AppHandle, notifications: bool) -> Self {
        TauriSink {
            app,
            notifications: AtomicBool::new(notifications),
        }
    }

    pub fn set_notifications(&self, enabled: bool) {
        self.notifications.store(enabled, Ordering::Relaxed);
    }

    fn main_window_visible(&self) -> bool {
        self.app
            .get_webview_window(crate::windows::MAIN)
            .and_then(|w| w.is_visible().ok())
            .unwrap_or(false)
    }
}

impl EventSink for TauriSink {
    fn emit(&self, event: MeetingEvent) {
        if let MeetingEvent::SummaryReady { .. } = &event {
            if self.notifications.load(Ordering::Relaxed) && !self.main_window_visible() {
                let _ = self
                    .app
                    .notification()
                    .builder()
                    .title("OpenMeet")
                    .body("Your meeting summary is ready.")
                    .show();
            }
        }
        if let Err(e) = self.app.emit(MEETING_EVENT_CHANNEL, &event) {
            tracing::warn!(event = event.name(), error = %e, "failed to deliver event to windows");
        }
    }
}
