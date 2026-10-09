//! Application state shared by commands, hotkeys and the tray.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use openmeet_core::events::{EventSink, MeetingEvent, NoticeLevel};
use openmeet_core::ids::now_ms;
use openmeet_core::ipc::{ScreenSourceKind, ScreenSourceSpec, StartMeetingRequest, UiCommand};
use openmeet_core::settings::Settings;
use openmeet_core::types::{AudioSourceKind, Component, ComponentStatus, Meeting, MeetingState};
use openmeet_core::CoreError;
use openmeet_engine::Engine;
use openmeet_platform::system::ProcessStats;
use openmeet_store::Database;
use parking_lot::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_autostart::ManagerExt as AutostartExt;

use crate::capture::{CaptureManager, MeetingSources};
use crate::local_ai::{self, LocalAiManager};
use crate::overlay::OverlayController;
use crate::secrets::SecretStore;
use crate::shortcuts::ShortcutManager;
use crate::sink::TauriSink;
use crate::windows;

pub struct AppState {
    pub app: AppHandle,
    pub engine: Engine,
    pub db: Arc<Database>,
    pub secrets: Arc<SecretStore>,
    pub sink: Arc<TauriSink>,
    pub capture: CaptureManager,
    pub overlay: OverlayController,
    pub shortcuts: ShortcutManager,
    pub local_ai: LocalAiManager,
    pub stats: Mutex<ProcessStats>,
    pub data_dir: PathBuf,
    /// Serializes settings updates (read-modify-write).
    settings_lock: tokio::sync::Mutex<()>,
}

/// Runs blocking work (database, device enumeration) off the async workers.
pub async fn blocking<T, F>(f: F) -> Result<T, CoreError>
where
    F: FnOnce() -> Result<T, CoreError> + Send + 'static,
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| CoreError::Internal(format!("background task failed: {e}")))?
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        app: AppHandle,
        engine: Engine,
        db: Arc<Database>,
        secrets: Arc<SecretStore>,
        sink: Arc<TauriSink>,
        data_dir: PathBuf,
    ) -> Self {
        AppState {
            capture: CaptureManager::new(sink.clone() as Arc<dyn EventSink>),
            overlay: OverlayController::new(app.clone()),
            shortcuts: ShortcutManager::new(),
            local_ai: LocalAiManager::new(&app),
            stats: Mutex::new(ProcessStats::new()),
            settings_lock: tokio::sync::Mutex::new(()),
            app,
            engine,
            db,
            secrets,
            sink,
            data_dir,
        }
    }

    pub fn settings(&self) -> Settings {
        self.engine.settings()
    }

    pub fn notice(&self, level: NoticeLevel, message: impl Into<String>) {
        self.sink.emit(MeetingEvent::Notice {
            level,
            message: message.into(),
            component: None,
        });
    }

    // ------------------------------------------------------------------
    // Settings
    // ------------------------------------------------------------------

    /// Validates, persists and applies new settings; returns the stored value.
    pub async fn update_settings(self: &Arc<Self>, requested: Settings) -> Result<Settings, CoreError> {
        let _guard = self.settings_lock.lock().await;
        let next = requested.sanitized();
        next.validate()
            .map_err(|errors| CoreError::InvalidInput(errors.join("; ")))?;
        let previous = self.settings();
        let db = self.db.clone();
        let to_save = next.clone();
        blocking(move || db.save_settings(&to_save).map_err(CoreError::from)).await?;
        self.apply_settings(&previous, &next).await;
        windows::emit_ui(&self.app, UiCommand::SettingsChanged);
        Ok(next)
    }

    /// Applies a mutation to the current settings.
    pub async fn modify_settings(self: &Arc<Self>, f: impl FnOnce(&mut Settings)) -> Result<Settings, CoreError> {
        let mut s = self.settings();
        f(&mut s);
        self.update_settings(s).await
    }

    async fn apply_settings(self: &Arc<Self>, prev: &Settings, next: &Settings) {
        self.engine.update_settings(next.clone());
        self.sink.set_notifications(next.general.notifications);

        let providers_changed = prev.ai != next.ai
            || prev.transcription != next.transcription
            || prev.privacy.local_only != next.privacy.local_only;
        if providers_changed {
            self.rebuild_providers().await;
        }
        if prev.overlay != next.overlay {
            if prev.overlay.click_through != next.overlay.click_through {
                self.overlay.set_click_through(next.overlay.click_through);
            }
            let layout_changed = prev.overlay.width != next.overlay.width
                || prev.overlay.max_height != next.overlay.max_height
                || prev.overlay.position != next.overlay.position
                || prev.overlay.target_display != next.overlay.target_display
                || prev.overlay.layout != next.overlay.layout
                || prev.overlay.margin != next.overlay.margin;
            if layout_changed {
                self.overlay.apply_layout(&next.overlay);
            }
        }
        if prev.privacy.capture_exclusion != next.privacy.capture_exclusion {
            let status = self.overlay.apply_capture_privacy(next.privacy.capture_exclusion);
            self.sink.emit(MeetingEvent::PrivacyStateChanged {
                enabled: status.requested,
                supported: status.supported,
                active: status.active,
            });
        }
        if prev.shortcuts != next.shortcuts {
            self.shortcuts.register_all(&self.app, &next.shortcuts);
        }
        if prev.general.launch_at_startup != next.general.launch_at_startup {
            let autostart = self.app.autolaunch();
            let r = if next.general.launch_at_startup {
                autostart.enable()
            } else {
                autostart.disable()
            };
            if let Err(e) = r {
                tracing::warn!(error = %e, "autostart change failed");
                self.notice(NoticeLevel::Warning, "Could not change the launch-at-startup setting.");
            }
        }
        if prev.privacy.retention != next.privacy.retention {
            let db = self.db.clone();
            let policy = next.privacy.retention;
            let _ = blocking(move || db.apply_retention(policy, now_ms()).map(|_| ()).map_err(CoreError::from)).await;
        }
        // Screen settings while a meeting is live.
        if prev.screen != next.screen && self.engine.is_live() {
            self.sync_screen_capture().await;
        }
    }

    /// Rebuilds providers from the current settings and hands them to the engine.
    pub async fn rebuild_providers(&self) {
        let settings = self.settings();
        let secrets = self.secrets.clone();
        let built = tauri::async_runtime::spawn_blocking(move || {
            crate::providers_build::build(&settings, secrets.as_ref())
        })
        .await;
        match built {
            Ok(set) => self.engine.set_providers(set),
            Err(e) => tracing::error!(error = %e, "provider rebuild failed"),
        }
    }

    // ------------------------------------------------------------------
    // Meeting lifecycle (spec §39, §68)
    // ------------------------------------------------------------------

    pub async fn start_meeting(self: &Arc<Self>, req: StartMeetingRequest) -> Result<Meeting, CoreError> {
        let settings = self.settings();
        if !req.microphone && !req.system_audio {
            self.notice(
                NoticeLevel::Info,
                "No audio source selected: transcription is off. Ask OpenMeet still works.",
            );
        }
        // Make sure the local sidecar is up when transcription depends on it.
        if (req.microphone || req.system_audio) && local_ai::sidecar_in_use(&settings) {
            match self.local_ai.start(&settings).await {
                Ok(_) => {}
                Err(e) => {
                    self.engine.set_component_status(
                        Component::Transcription,
                        ComponentStatus::Error,
                        Some(e.to_string()),
                    );
                    self.notice(NoticeLevel::Warning, format!("Local transcription unavailable: {e}"));
                }
            }
        }
        let app_name = match &req.focus_app_id {
            Some(id) => {
                let id = id.clone();
                blocking(move || {
                    Ok(openmeet_platform::apps::detect_meeting_apps()
                        .into_iter()
                        .find(|a| a.window_id.as_deref() == Some(id.as_str()) || a.id == id)
                        .map(|a| a.name))
                })
                .await
                .unwrap_or(None)
            }
            None => None,
        };
        let meeting = self.engine.start_meeting(req.clone()).await?;
        let sources = MeetingSources {
            microphone: req.microphone,
            system_audio: req.system_audio,
            screen: req.screen.clone(),
        };
        self.capture.set_sources(Some(sources.clone()));
        self.start_capture(&sources).await;

        self.overlay.set_app_name(app_name);
        if settings.overlay.show_on_meeting_start {
            self.overlay.show(&settings.overlay);
        }
        if let Some(id) = &req.focus_app_id {
            if let Err(e) = openmeet_platform::focus::focus_window(id) {
                tracing::debug!(error = %e, "could not focus the meeting app");
            } else if let Some(main) = self.app.get_webview_window(windows::MAIN) {
                // The meeting app is the primary surface now (spec §99).
                let _ = main.minimize();
            }
        }
        Ok(meeting)
    }

    async fn start_capture(self: &Arc<Self>, sources: &MeetingSources) {
        let settings = self.settings();
        if sources.microphone {
            let _ = self.capture.start_audio(
                &self.engine,
                AudioSourceKind::Microphone,
                settings.audio.microphone_device.clone(),
            );
        }
        if sources.system_audio {
            let _ = self.capture.start_audio(
                &self.engine,
                AudioSourceKind::System,
                settings.audio.system_audio_device.clone(),
            );
        }
        if let Some(spec) = sources.screen.clone() {
            self.start_screen(spec).await;
        }
    }

    async fn start_screen(self: &Arc<Self>, spec: ScreenSourceSpec) {
        let db = self.db.clone();
        let rules = blocking(move || db.list_privacy_rules().map_err(CoreError::from))
            .await
            .unwrap_or_default();
        let settings = self.settings();
        if let Err(e) = self.capture.start_screen(&self.engine, spec, &settings, rules) {
            self.notice(NoticeLevel::Error, format!("Screen capture unavailable: {e}"));
        }
    }

    /// Starts/stops the sampler to match the screen settings of a live meeting.
    pub async fn sync_screen_capture(self: &Arc<Self>) {
        let settings = self.settings();
        if !self.engine.is_live() || self.engine.meeting_state() == MeetingState::Paused {
            return;
        }
        let spec = settings.screen.source.clone().filter(|_| settings.screen.enabled);
        if let Some(mut sources) = self.capture.sources() {
            sources.screen = spec.clone();
            self.capture.set_sources(Some(sources));
        }
        match spec {
            Some(spec) => self.start_screen(spec).await,
            None => self.capture.stop_screen(&self.engine),
        }
    }

    pub async fn pause(self: &Arc<Self>) -> Result<(), CoreError> {
        self.engine.pause()?;
        // Paused means not capturing: release the devices.
        self.capture.stop_all(&self.engine);
        Ok(())
    }

    pub async fn resume(self: &Arc<Self>) -> Result<(), CoreError> {
        self.engine.resume()?;
        if let Some(sources) = self.capture.sources() {
            if !self.capture.is_capturing_audio() {
                self.start_capture(&sources).await;
            }
        }
        Ok(())
    }

    pub async fn toggle_pause(self: &Arc<Self>) -> Result<MeetingState, CoreError> {
        match self.engine.meeting_state() {
            MeetingState::Active => self.pause().await?,
            MeetingState::Paused => self.resume().await?,
            _ => return Err(CoreError::InvalidState("No meeting is running.".into())),
        }
        Ok(self.engine.meeting_state())
    }

    pub async fn stop_meeting(self: &Arc<Self>) -> Result<(), CoreError> {
        self.capture.stop_all(&self.engine);
        self.capture.set_sources(None);
        let result = if self.engine.demo_running() {
            self.engine.stop_demo().await
        } else {
            self.engine.stop_meeting().await
        };
        self.overlay.set_app_name(None);
        result
    }

    /// Emergency privacy stop (spec §40): immediate, no confirmation.
    pub fn emergency_stop(&self) {
        self.capture.stop_all(&self.engine);
        self.engine.emergency_stop();
        self.overlay.hide();
        if let Some(tray) = self.app.tray_by_id(crate::tray::TRAY_ID) {
            let _ = tray.set_tooltip(Some("OpenMeet — PROCESSING STOPPED"));
        }
    }

    /// Manual screen capture (hotkey / overlay button).
    pub async fn capture_screen_now(self: &Arc<Self>) -> Result<(), CoreError> {
        if !self.engine.is_live() {
            return Err(CoreError::InvalidState("Start a meeting to capture the screen.".into()));
        }
        if self.capture.capture_now() {
            return Ok(());
        }
        // No sampler running: capture the configured source, or the primary display.
        let settings = self.settings();
        let spec = settings.screen.source.clone().unwrap_or(ScreenSourceSpec {
            kind: ScreenSourceKind::Display,
            id: String::new(),
            label: "Primary display".into(),
            region: None,
        });
        let db = self.db.clone();
        let frame = blocking(move || {
            let rules = db.list_privacy_rules().map_err(CoreError::from)?;
            let spec = if spec.id.is_empty() {
                openmeet_platform::screen::list_sources(&rules)
                    .map_err(CoreError::from)?
                    .into_iter()
                    .find(|s| s.kind == ScreenSourceKind::Display && s.is_primary)
                    .map(|s| ScreenSourceSpec {
                        kind: s.kind,
                        id: s.id,
                        label: s.label,
                        region: None,
                    })
                    .ok_or_else(|| CoreError::NotFound("No display found to capture.".into()))?
            } else {
                spec
            };
            openmeet_platform::screen::capture_once(&spec, &rules, 1280).map_err(CoreError::from)
        })
        .await?;
        let mut frame = frame;
        frame.manual = true;
        self.engine.push_screen_frame(frame);
        Ok(())
    }

    /// Periodic retention enforcement (spec §90).
    pub fn spawn_retention_task(self: &Arc<Self>) {
        let state = Arc::downgrade(self);
        tauri::async_runtime::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                let Some(state) = state.upgrade() else { break };
                let policy = state.settings().privacy.retention;
                let db = state.db.clone();
                match blocking(move || db.apply_retention(policy, now_ms()).map_err(CoreError::from)).await {
                    Ok(n) if n > 0 => tracing::info!(deleted = n, "retention applied"),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "retention failed"),
                }
            }
        });
    }
}
