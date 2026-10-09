//! Overlay window behaviour: visibility, click-through, content sizing,
//! positioning, focus hand-off for typing and capture privacy (spec §4–§8,
//! §35–§37).

use openmeet_core::ipc::{CapturePrivacyStatus, DisplayInfo, Point, UiCommand};
use openmeet_core::settings::OverlaySettings;
use openmeet_platform::privacy::{self, CapturePrivacy};
use parking_lot::Mutex;
use raw_window_handle::HasWindowHandle;
use tauri::{AppHandle, LogicalSize, Manager, PhysicalPosition, WebviewWindow};

use crate::windows::{self, OVERLAY, OVERLAY_MAX_WIDTH, OVERLAY_MIN_HEIGHT, OVERLAY_MIN_WIDTH};

#[derive(Default)]
struct OverlayState {
    input_mode: bool,
    /// Foreground window (meeting app) before the overlay took focus.
    previous_foreground: Option<String>,
    /// Meeting application name, for per-app remembered positions.
    app_name: Option<String>,
    /// Last content size requested by the UI (logical px).
    content_size: Option<(f64, f64)>,
    /// Capture exclusion handles for the OpenMeet windows.
    privacy: Vec<(String, Box<dyn CapturePrivacy>)>,
    privacy_requested: bool,
}

pub struct OverlayController {
    app: AppHandle,
    state: Mutex<OverlayState>,
}

impl OverlayController {
    pub fn new(app: AppHandle) -> Self {
        OverlayController {
            app,
            state: Mutex::new(OverlayState::default()),
        }
    }

    fn window(&self) -> Option<WebviewWindow> {
        self.app.get_webview_window(OVERLAY)
    }

    pub fn is_visible(&self) -> bool {
        self.window().and_then(|w| w.is_visible().ok()).unwrap_or(false)
    }

    pub fn set_app_name(&self, name: Option<String>) {
        self.state.lock().app_name = name;
    }

    pub fn show(&self, settings: &OverlaySettings) -> bool {
        let Some(w) = self.window() else { return false };
        if !self.is_visible() {
            self.apply_layout(settings);
        }
        let ok = w.show().is_ok();
        // Some window managers drop the topmost flag while hidden.
        let _ = w.set_always_on_top(true);
        windows::emit_ui(&self.app, UiCommand::OverlayVisibilityChanged { visible: ok });
        ok
    }

    pub fn hide(&self) -> bool {
        if self.state.lock().input_mode {
            self.end_input();
        }
        let Some(w) = self.window() else { return false };
        let ok = w.hide().is_ok();
        windows::emit_ui(&self.app, UiCommand::OverlayVisibilityChanged { visible: false });
        ok
    }

    /// Returns the new visibility.
    pub fn toggle(&self, settings: &OverlaySettings) -> bool {
        if self.is_visible() {
            self.hide();
            false
        } else {
            self.show(settings)
        }
    }

    /// Click-through: the overlay stays visible but mouse events reach the
    /// meeting application underneath (spec §6).
    pub fn set_click_through(&self, enabled: bool) -> bool {
        let ok = self
            .window()
            .map(|w| w.set_ignore_cursor_events(enabled).is_ok())
            .unwrap_or(false);
        windows::emit_ui(&self.app, UiCommand::ClickThroughChanged { enabled });
        ok
    }

    /// Content-driven resize (logical px), clamped to the configured limits.
    pub fn set_content_size(&self, width: f64, height: f64, settings: &OverlaySettings) {
        if !width.is_finite() || !height.is_finite() {
            return;
        }
        let w = width.clamp(OVERLAY_MIN_WIDTH, OVERLAY_MAX_WIDTH.min(f64::from(settings.width).max(OVERLAY_MIN_WIDTH)));
        let h = height.clamp(OVERLAY_MIN_HEIGHT, f64::from(settings.max_height));
        {
            let mut st = self.state.lock();
            if let Some((pw, ph)) = st.content_size {
                if (pw - w).abs() < 1.0 && (ph - h).abs() < 1.0 {
                    return;
                }
            }
            st.content_size = Some((w, h));
        }
        if let Some(win) = self.window() {
            let _ = win.set_size(LogicalSize::new(w, h));
            // Keep the anchored corner in place when the content grows.
            self.apply_position(settings);
        }
    }

    /// Applies size + position for the current settings.
    pub fn apply_layout(&self, settings: &OverlaySettings) {
        let Some(win) = self.window() else { return };
        let height = self
            .state
            .lock()
            .content_size
            .map(|(_, h)| h)
            .unwrap_or(windows::OVERLAY_INITIAL_HEIGHT)
            .min(f64::from(settings.max_height));
        let _ = win.set_size(LogicalSize::new(f64::from(settings.width), height));
        self.state.lock().content_size = Some((f64::from(settings.width), height));
        self.apply_position(settings);
    }

    fn apply_position(&self, settings: &OverlaySettings) {
        let Some(win) = self.window() else { return };
        let displays = windows::list_displays(&self.app);
        let Some(display) = windows::pick_display(&displays, settings) else { return };
        let size = win.outer_size().map(|s| (s.width, s.height)).unwrap_or((
            (f64::from(settings.width) * display.scale_factor) as u32,
            (windows::OVERLAY_INITIAL_HEIGHT * display.scale_factor) as u32,
        ));
        let app_name = self.state.lock().app_name.clone();
        let (x, y) = windows::overlay_origin(display, size, settings, app_name.as_deref());
        let _ = win.set_position(PhysicalPosition::new(x, y));
    }

    /// Current position relative to the display it is on, for remembering a
    /// dragged position. Returns (display key, position key, point).
    pub fn current_relative_position(&self) -> Option<(String, String, Point)> {
        let win = self.window()?;
        let pos = win.outer_position().ok()?;
        let size = win.outer_size().ok()?;
        let displays = windows::list_displays(&self.app);
        let cx = pos.x + size.width as i32 / 2;
        let cy = pos.y + size.height as i32 / 2;
        let display: &DisplayInfo = windows::display_at(&displays, cx, cy)
            .or_else(|| displays.iter().find(|d| d.is_primary))?;
        let app_name = self.state.lock().app_name.clone();
        let key = windows::position_key(&display.key, app_name.as_deref());
        Some((display.key.clone(), key, windows::relative_point(display, (pos.x, pos.y))))
    }

    /// Lets the overlay take keyboard focus for typing (Ask OpenMeet),
    /// remembering the meeting app so focus can be returned.
    pub fn begin_input(&self) {
        let Some(win) = self.window() else { return };
        {
            let mut st = self.state.lock();
            if st.input_mode {
                let _ = win.set_focus();
                return;
            }
            st.previous_foreground = openmeet_platform::focus::foreground_window_id();
            st.input_mode = true;
        }
        let _ = win.set_focusable(true);
        let _ = win.set_focus();
    }

    /// Restores non-activating behaviour and returns focus to the meeting app.
    pub fn end_input(&self) {
        let previous = {
            let mut st = self.state.lock();
            if !st.input_mode {
                return;
            }
            st.input_mode = false;
            st.previous_foreground.take()
        };
        if let Some(win) = self.window() {
            let _ = win.set_focusable(false);
        }
        if let Some(prev) = previous {
            if !openmeet_platform::focus::restore_foreground(&prev) {
                tracing::debug!("could not restore the previous foreground window");
            }
        }
    }

    // ------------------------------------------------------------------
    // Capture privacy (spec §8)
    // ------------------------------------------------------------------

    /// Applies (or removes) capture exclusion on the OpenMeet windows using the
    /// documented OS API, and reports the verified state.
    pub fn apply_capture_privacy(&self, enabled: bool) -> CapturePrivacyStatus {
        let labels = [windows::OVERLAY, windows::MAIN];
        let mut st = self.state.lock();
        st.privacy_requested = enabled;
        for label in labels {
            if st.privacy.iter().any(|(l, _)| l == label) {
                continue;
            }
            let Some(win) = self.app.get_webview_window(label) else { continue };
            match win.window_handle() {
                Ok(handle) => st.privacy.push((label.to_string(), privacy::for_window(handle.as_raw()))),
                Err(e) => tracing::warn!(window = label, error = %e, "no native window handle"),
            }
        }
        for (label, p) in &st.privacy {
            if !p.is_supported() {
                continue;
            }
            if let Err(e) = p.set_excluded_from_capture(enabled) {
                tracing::warn!(window = %label, error = %e, "capture exclusion failed");
            }
        }
        drop(st);
        self.capture_privacy_status()
    }

    pub fn capture_privacy_status(&self) -> CapturePrivacyStatus {
        let (supported, mechanism, note) = privacy::platform_support();
        let st = self.state.lock();
        let overlay = st.privacy.iter().find(|(l, _)| l == windows::OVERLAY);
        let active = match overlay {
            Some((_, p)) => p.is_excluded().unwrap_or(st.privacy_requested && supported),
            None => false,
        };
        CapturePrivacyStatus {
            supported,
            requested: st.privacy_requested,
            active: supported && active,
            platform: std::env::consts::OS.to_string(),
            mechanism: mechanism.to_string(),
            note: note.to_string(),
        }
    }
}
