//! Window creation and display geometry.

use openmeet_core::events::UI_EVENT_CHANNEL;
use openmeet_core::ipc::UiCommand;
use openmeet_core::ipc::{DisplayInfo, DisplayOrientation, Point, Rect};
use openmeet_core::settings::{DisplayLayout, OverlayPosition, OverlaySettings};
use tauri::{
    AppHandle, Emitter, Manager, Monitor, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
};

pub const MAIN: &str = "main";
pub const OVERLAY: &str = "overlay";

/// Initial overlay height before the UI reports its content size.
pub const OVERLAY_INITIAL_HEIGHT: f64 = 120.0;
pub const OVERLAY_MIN_WIDTH: f64 = 260.0;
pub const OVERLAY_MAX_WIDTH: f64 = 900.0;
pub const OVERLAY_MIN_HEIGHT: f64 = 40.0;

pub fn create_main(app: &AppHandle, visible: bool) -> tauri::Result<WebviewWindow> {
    WebviewWindowBuilder::new(app, MAIN, WebviewUrl::App("index.html".into()))
        .title("OpenMeet")
        .inner_size(1120.0, 760.0)
        .min_inner_size(860.0, 560.0)
        .visible(visible)
        .center()
        .build()
}

/// The overlay: transparent, frameless, always on top, not in the taskbar and
/// non-activating (`focusable(false)` → WS_EX_NOACTIVATE on Windows), so the
/// meeting application keeps keyboard focus (spec §4, §7).
pub fn create_overlay(app: &AppHandle, s: &OverlaySettings) -> tauri::Result<WebviewWindow> {
    WebviewWindowBuilder::new(app, OVERLAY, WebviewUrl::App("overlay.html".into()))
        .title("OpenMeet Overlay")
        .transparent(true)
        .decorations(false)
        .shadow(false)
        .always_on_top(true)
        .visible_on_all_workspaces(true)
        .skip_taskbar(true)
        .resizable(true)
        .maximizable(false)
        .minimizable(false)
        .focusable(false)
        .focused(false)
        .visible(false)
        .accept_first_mouse(true)
        .inner_size(f64::from(s.width), OVERLAY_INITIAL_HEIGHT)
        .min_inner_size(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT)
        .build()
}

pub fn emit_ui(app: &AppHandle, command: UiCommand) {
    if let Err(e) = app.emit(UI_EVENT_CHANNEL, &command) {
        tracing::warn!(error = %e, "failed to emit UI command");
    }
}

pub fn emit_ui_to(app: &AppHandle, window: &str, command: UiCommand) {
    if let Err(e) = app.emit_to(window, UI_EVENT_CHANNEL, &command) {
        tracing::warn!(error = %e, "failed to emit UI command");
    }
}

/// Shows (creating if needed) and focuses the main window, optionally on a
/// section.
pub fn show_main(app: &AppHandle, section: Option<&str>) {
    let window = match app.get_webview_window(MAIN) {
        Some(w) => w,
        None => match create_main(app, true) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(error = %e, "failed to create main window");
                return;
            }
        },
    };
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_focus();
    if let Some(section) = section {
        emit_ui_to(
            app,
            MAIN,
            UiCommand::Navigate {
                section: section.to_string(),
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Displays & geometry (spec §35–§37)
// ---------------------------------------------------------------------------

/// Stable key for a display: name + resolution, so swapping an external
/// monitor yields a different remembered position.
pub fn display_key(name: Option<&str>, width: u32, height: u32) -> String {
    format!("{}@{}x{}", name.unwrap_or("display"), width, height)
}

pub fn display_info(m: &Monitor, primary_key: Option<&str>) -> DisplayInfo {
    let size = m.size();
    let pos = m.position();
    let work = m.work_area();
    let key = display_key(m.name().map(String::as_str), size.width, size.height);
    DisplayInfo {
        is_primary: primary_key == Some(key.as_str()),
        name: m
            .name()
            .cloned()
            .map(|n| n.trim_start_matches(r"\\.\").to_string())
            .unwrap_or_else(|| "Display".into()),
        key,
        x: pos.x,
        y: pos.y,
        width: size.width,
        height: size.height,
        work_area: Rect {
            x: work.position.x,
            y: work.position.y,
            width: work.size.width,
            height: work.size.height,
        },
        scale_factor: m.scale_factor(),
        orientation: if size.height > size.width {
            DisplayOrientation::Portrait
        } else {
            DisplayOrientation::Landscape
        },
    }
}

pub fn list_displays(app: &AppHandle) -> Vec<DisplayInfo> {
    let primary_key = app.primary_monitor().ok().flatten().map(|m| {
        display_key(
            m.name().map(String::as_str),
            m.size().width,
            m.size().height,
        )
    });
    let mut out: Vec<DisplayInfo> = app
        .available_monitors()
        .unwrap_or_default()
        .iter()
        .map(|m| display_info(m, primary_key.as_deref()))
        .collect();
    if !out.iter().any(|d| d.is_primary) {
        if let Some(first) = out.first_mut() {
            first.is_primary = true;
        }
    }
    out
}

/// Chooses the display the overlay should live on.
pub fn pick_display<'a>(
    displays: &'a [DisplayInfo],
    s: &OverlaySettings,
) -> Option<&'a DisplayInfo> {
    if let Some(key) = &s.target_display {
        if let Some(d) = displays.iter().find(|d| &d.key == key) {
            return Some(d);
        }
    }
    let primary = displays
        .iter()
        .find(|d| d.is_primary)
        .or_else(|| displays.first());
    match s.layout {
        DisplayLayout::SeparateDisplay => displays.iter().find(|d| !d.is_primary).or(primary),
        DisplayLayout::SameDisplay => primary,
    }
}

/// Key under which a custom position is remembered: per display, and per
/// meeting application on that display when known.
pub fn position_key(display_key: &str, app_name: Option<&str>) -> String {
    match app_name {
        Some(app) if !app.trim().is_empty() => format!("{display_key}|{}", app.trim()),
        _ => display_key.to_string(),
    }
}

/// Top-left corner (physical px) for the overlay of `size` on `display`.
pub fn overlay_origin(
    display: &DisplayInfo,
    size: (u32, u32),
    s: &OverlaySettings,
    app_name: Option<&str>,
) -> (i32, i32) {
    let work = display.work_area;
    let margin = (f64::from(s.margin) * display.scale_factor).round() as i32;
    let (w, h) = (size.0 as i32, size.1 as i32);
    let left = work.x + margin;
    let right = work.x + work.width as i32 - w - margin;
    let top = work.y + margin;
    let bottom = work.y + work.height as i32 - h - margin;
    let corner = |p: OverlayPosition| match p {
        OverlayPosition::TopLeft => (left, top),
        OverlayPosition::BottomRight => (right, bottom),
        OverlayPosition::BottomLeft => (left, bottom),
        OverlayPosition::TopRight | OverlayPosition::Custom => (right, top),
    };
    let (x, y) = match s.position {
        OverlayPosition::Custom => {
            let remembered = s
                .custom_positions
                .get(&position_key(&display.key, app_name))
                .or_else(|| s.custom_positions.get(&display.key));
            match remembered {
                Some(p) => (work.x + p.x, work.y + p.y),
                None => corner(OverlayPosition::TopRight),
            }
        }
        p => corner(p),
    };
    clamp_into(work, (w, h), (x, y))
}

/// Keeps a window of `size` inside `work` (as much as possible).
pub fn clamp_into(work: Rect, size: (i32, i32), pos: (i32, i32)) -> (i32, i32) {
    let max_x = work.x + (work.width as i32 - size.0).max(0);
    let max_y = work.y + (work.height as i32 - size.1).max(0);
    (pos.0.clamp(work.x, max_x), pos.1.clamp(work.y, max_y))
}

/// Position relative to the display's work area (stored in settings).
pub fn relative_point(display: &DisplayInfo, pos: (i32, i32)) -> Point {
    Point {
        x: pos.0 - display.work_area.x,
        y: pos.1 - display.work_area.y,
    }
}

/// Display containing the point (physical px), if any.
pub fn display_at(displays: &[DisplayInfo], x: i32, y: i32) -> Option<&DisplayInfo> {
    displays.iter().find(|d| {
        Rect {
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
        }
        .contains(Point { x, y })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn display(key: &str, x: i32, primary: bool, scale: f64) -> DisplayInfo {
        DisplayInfo {
            key: key.into(),
            name: key.into(),
            x,
            y: 0,
            width: 1920,
            height: 1080,
            work_area: Rect {
                x,
                y: 0,
                width: 1920,
                height: 1040,
            },
            scale_factor: scale,
            is_primary: primary,
            orientation: DisplayOrientation::Landscape,
        }
    }

    #[test]
    fn corners_respect_margin_scale_and_taskbar() {
        let d = display("A", 0, true, 1.5);
        let mut s = OverlaySettings {
            margin: 16,
            position: OverlayPosition::TopRight,
            ..Default::default()
        };
        assert_eq!(
            overlay_origin(&d, (570, 300), &s, None),
            (1920 - 570 - 24, 24)
        );
        s.position = OverlayPosition::BottomLeft;
        assert_eq!(
            overlay_origin(&d, (570, 300), &s, None),
            (24, 1040 - 300 - 24)
        );
    }

    #[test]
    fn custom_positions_per_display_and_app() {
        let d = display("B", 1920, false, 1.0);
        let mut s = OverlaySettings {
            position: OverlayPosition::Custom,
            ..Default::default()
        };
        s.custom_positions
            .insert("B".into(), Point { x: 100, y: 50 });
        s.custom_positions
            .insert("B|Zoom".into(), Point { x: 300, y: 70 });
        assert_eq!(overlay_origin(&d, (380, 200), &s, None), (2020, 50));
        assert_eq!(overlay_origin(&d, (380, 200), &s, Some("Zoom")), (2220, 70));
        // Unknown display falls back to the top-right corner.
        let other = display("C", 0, true, 1.0);
        assert_eq!(
            overlay_origin(&other, (380, 200), &s, None),
            (1920 - 380 - 16, 16)
        );
    }

    #[test]
    fn clamps_inside_work_area() {
        let work = Rect {
            x: 0,
            y: 0,
            width: 1000,
            height: 800,
        };
        assert_eq!(clamp_into(work, (300, 200), (900, -50)), (700, 0));
        assert_eq!(clamp_into(work, (1200, 200), (50, 50)), (0, 50));
    }

    #[test]
    fn display_selection() {
        let displays = vec![display("A", 0, true, 1.0), display("B", 1920, false, 1.0)];
        let mut s = OverlaySettings::default();
        assert_eq!(pick_display(&displays, &s).unwrap().key, "A");
        s.layout = DisplayLayout::SeparateDisplay;
        assert_eq!(pick_display(&displays, &s).unwrap().key, "B");
        s.target_display = Some("A".into());
        assert_eq!(pick_display(&displays, &s).unwrap().key, "A");
        s.target_display = Some("gone".into());
        assert_eq!(pick_display(&displays, &s).unwrap().key, "B");
        assert_eq!(display_at(&displays, 2500, 10).unwrap().key, "B");
        assert!(display_at(&displays, -5, 10).is_none());
    }

    #[test]
    fn keys() {
        assert_eq!(
            display_key(Some(r"\\.\DISPLAY1"), 2560, 1440),
            r"\\.\DISPLAY1@2560x1440"
        );
        assert_eq!(position_key("A", Some(" Zoom ")), "A|Zoom");
        assert_eq!(position_key("A", Some("")), "A");
    }
}
