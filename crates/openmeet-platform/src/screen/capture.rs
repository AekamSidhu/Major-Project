//! OS-facing capture: window snapshots, target resolution, privacy evaluation
//! and masking. Pure decision logic lives in [`evaluate_area_privacy`] so it can
//! be unit tested without a display.

use std::collections::{HashMap, HashSet};

use image::RgbaImage;
use openmeet_core::ipc::{ScreenSourceKind, ScreenSourceSpec};
use openmeet_core::types::{PrivacyRule, PrivacyRuleType};

use super::geometry::{IRect, visible_parts};
use super::rules::{apply_region_masks, excluded_by, fill_black, normalize_app};
use crate::PlatformError;

/// User-facing message for a missing macOS screen-recording permission.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) const SCREEN_PERMISSION_MESSAGE: &str =
    "Screen permission required. Open System Settings → Privacy & Security → Screen Recording.";

/// Minimum share of the captured area a window must cover to be considered
/// the "front" window when the focused window is not on that area.
const FRONT_MIN_COVERAGE: f64 = 0.2;

/// Plain data about a top-level window (no OS handle), used by pure logic.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WinMeta {
    pub id: u32,
    pub pid: u32,
    pub app: String,
    pub title: String,
    /// Global (virtual desktop) coordinates, same units as monitor geometry.
    pub rect: IRect,
    pub minimized: bool,
    pub focused: bool,
    /// Higher = closer to the viewer.
    pub z: i64,
    /// False for layered / click-through windows that may not hide what is
    /// behind them (never treated as occluders).
    pub opaque: bool,
}

impl WinMeta {
    fn app_opt(&self) -> Option<&str> {
        Some(self.app.as_str()).filter(|a| !a.trim().is_empty())
    }

    fn title_opt(&self) -> Option<&str> {
        Some(self.title.as_str()).filter(|t| !t.trim().is_empty())
    }

    /// Privacy-rule match for this window.
    pub fn excluded(&self, rules: &[PrivacyRule]) -> Option<String> {
        excluded_by(self.app_opt(), self.title_opt(), rules)
    }
}

/// A window snapshot entry: metadata plus the handle used to capture it.
pub(crate) struct WinInfo {
    pub meta: WinMeta,
    pub handle: xcap::Window,
}

/// Caches application names per process id (resolving them can involve
/// reading version resources from the executable).
#[derive(Debug, Default)]
pub(crate) struct AppNameCache {
    names: HashMap<u32, String>,
}

impl AppNameCache {
    fn name(&mut self, pid: u32, window: &xcap::Window) -> String {
        if pid == 0 {
            return window.app_name().unwrap_or_default();
        }
        self.names
            .entry(pid)
            .or_insert_with(|| window.app_name().unwrap_or_default())
            .clone()
    }

    fn retain(&mut self, live: &HashSet<u32>) {
        self.names.retain(|pid, _| live.contains(pid));
    }
}

/// Lists top-level windows, front-most first.
pub(crate) fn snapshot_windows(cache: &mut AppNameCache) -> Result<Vec<WinInfo>, PlatformError> {
    let windows = xcap::Window::all().map_err(|e| xcap_error("listing windows", e))?;
    let count = windows.len();
    let mut out = Vec::with_capacity(count);
    for (index, handle) in windows.into_iter().enumerate() {
        let Ok(id) = handle.id() else { continue };
        let (Ok(x), Ok(y), Ok(w), Ok(h)) =
            (handle.x(), handle.y(), handle.width(), handle.height())
        else {
            continue;
        };
        let pid = handle.pid().unwrap_or(0);
        let app = cache.name(pid, &handle);
        let title = handle.title().unwrap_or_default();
        let minimized = handle.is_minimized().unwrap_or(false);
        let focused = handle.is_focused().unwrap_or(false);
        // xcap enumerates Windows windows in Z order (front first); computing
        // `z()` per window would re-enumerate every window each time.
        let z = if cfg!(windows) {
            (count - index) as i64
        } else {
            i64::from(handle.z().unwrap_or(0))
        };
        let opaque = window_is_opaque(id);
        out.push(WinInfo {
            meta: WinMeta {
                id,
                pid,
                app,
                title,
                rect: IRect::from_xywh(i64::from(x), i64::from(y), i64::from(w), i64::from(h)),
                minimized,
                focused,
                z,
                opaque,
            },
            handle,
        });
    }
    out.sort_by_key(|w| std::cmp::Reverse(w.meta.z));
    let live: HashSet<u32> = out.iter().map(|w| w.meta.pid).collect();
    cache.retain(&live);
    Ok(out)
}

#[cfg(windows)]
fn window_is_opaque(id: u32) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{WS_EX_LAYERED, WS_EX_TRANSPARENT};
    match crate::win::hwnd_from_id(&id.to_string()) {
        Some(hwnd) => {
            crate::win::window_ex_style(hwnd) & (WS_EX_LAYERED.0 | WS_EX_TRANSPARENT.0) == 0
        }
        None => false,
    }
}

#[cfg(not(windows))]
fn window_is_opaque(_id: u32) -> bool {
    // xcap only lists normal application windows on macOS / X11; assume they
    // hide what is behind them.
    true
}

/// Result of evaluating privacy rules for a display (or display region).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct AreaPrivacy {
    /// Index (into the evaluated slice) of the window treated as the front
    /// window of the area.
    pub front: Option<usize>,
    /// Set when the front window matches a rule: the frame must be skipped.
    pub skip_reason: Option<String>,
    /// Visible parts of other excluded windows (global coordinates) that must
    /// be blacked out.
    pub masks: Vec<IRect>,
}

/// Decides how privacy rules apply to a capture of `area` (global
/// coordinates). `windows` must be sorted front-most first.
///
/// * The front window is the focused window when it is on the area,
///   otherwise the top-most window covering at least 20 % of the area.
///   OpenMeet's own windows (`own_pid`) and minimized windows are ignored.
/// * A front window matching a rule skips the whole frame.
/// * Any other matching window is masked where it is visible (its rectangle
///   minus opaque windows above it).
pub(crate) fn evaluate_area_privacy(
    windows: &[WinMeta],
    area: IRect,
    own_pid: u32,
    rules: &[PrivacyRule],
) -> AreaPrivacy {
    let candidates: Vec<usize> = windows
        .iter()
        .enumerate()
        .filter(|(_, w)| w.pid != own_pid && !w.minimized && w.rect.intersects(&area))
        .map(|(i, _)| i)
        .collect();
    let area_size = area.area().max(1) as f64;
    let front = candidates
        .iter()
        .copied()
        .find(|&i| windows[i].focused)
        .or_else(|| {
            candidates.iter().copied().find(|&i| {
                let covered = windows[i]
                    .rect
                    .intersect(&area)
                    .map(|r| r.area())
                    .unwrap_or(0);
                covered as f64 / area_size >= FRONT_MIN_COVERAGE
            })
        });

    let mut result = AreaPrivacy {
        front,
        ..AreaPrivacy::default()
    };
    if !has_window_rules(rules) {
        return result;
    }
    if let Some(f) = front {
        if let Some(reason) = windows[f].excluded(rules) {
            result.skip_reason = Some(reason);
            return result;
        }
    }
    for (pos, &i) in candidates.iter().enumerate() {
        let w = &windows[i];
        if w.excluded(rules).is_none() {
            continue;
        }
        let Some(target) = w.rect.intersect(&area) else {
            continue;
        };
        let occluders: Vec<IRect> = candidates[..pos]
            .iter()
            .map(|&j| &windows[j])
            .filter(|o| o.opaque)
            .map(|o| o.rect)
            .collect();
        result.masks.extend(visible_parts(target, &occluders));
    }
    result
}

/// Whether any rule can exclude a window (application / window / domain).
pub(crate) fn has_window_rules(rules: &[PrivacyRule]) -> bool {
    rules
        .iter()
        .any(|r| r.rule_type != PrivacyRuleType::Region && !r.value.trim().is_empty())
}

/// Blacks out global-coordinate rectangles on an image whose top-left corner
/// is at `origin` and whose pixels per global unit are `scale`.
pub(crate) fn apply_global_masks(
    img: &mut RgbaImage,
    masks: &[IRect],
    origin: (i64, i64),
    scale: (f64, f64),
) {
    for m in masks {
        let x0 = ((m.x0 - origin.0) as f64 * scale.0).floor();
        let y0 = ((m.y0 - origin.1) as f64 * scale.1).floor();
        let x1 = ((m.x1 - origin.0) as f64 * scale.0).ceil();
        let y1 = ((m.y1 - origin.1) as f64 * scale.1).ceil();
        fill_black(img, x0, y0, x1, y1);
    }
}

/// A captured, privacy-filtered image at source resolution.
pub(crate) struct Captured {
    pub image: RgbaImage,
    pub application: Option<String>,
    pub window_title: Option<String>,
    pub label: String,
}

/// What a capture attempt produced.
pub(crate) enum CaptureOutcome {
    Frame(Captured),
    /// Skipped because of a privacy rule (the reason names the rule).
    Skipped(String),
}

/// Stateful capturer (keeps the application-name cache between frames).
#[derive(Debug, Default)]
pub(crate) struct Capturer {
    apps: AppNameCache,
}

impl Capturer {
    pub fn new() -> Self {
        Capturer::default()
    }

    /// Captures `spec`, applying privacy rules. Never captures windows of
    /// `own_pid`.
    pub fn capture(
        &mut self,
        spec: &ScreenSourceSpec,
        rules: &[PrivacyRule],
        own_pid: u32,
    ) -> Result<CaptureOutcome, PlatformError> {
        #[cfg(windows)]
        let _dpi = crate::win::DpiAwareGuard::new();
        ensure_screen_permission()?;
        match spec.kind {
            ScreenSourceKind::Display | ScreenSourceKind::Region => {
                self.capture_display(spec, rules, own_pid)
            }
            ScreenSourceKind::Window => {
                let id = parse_window_id(&spec.id)?;
                let windows = snapshot_windows(&mut self.apps)?;
                let info = windows
                    .into_iter()
                    .find(|w| w.meta.id == id)
                    .ok_or_else(|| {
                        PlatformError::DeviceNotFound(
                            "The selected window is no longer open".into(),
                        )
                    })?;
                capture_window(info, spec, rules, own_pid)
            }
            ScreenSourceKind::Application => {
                let windows = snapshot_windows(&mut self.apps)?;
                let info = pick_application_window(windows, &spec.id, own_pid)?;
                capture_window(info, spec, rules, own_pid)
            }
        }
    }

    fn capture_display(
        &mut self,
        spec: &ScreenSourceSpec,
        rules: &[PrivacyRule],
        own_pid: u32,
    ) -> Result<CaptureOutcome, PlatformError> {
        let monitor = find_monitor(&spec.id)?;
        let geo = MonitorGeometry::of(&monitor)?;
        let region = if spec.kind == ScreenSourceKind::Region {
            spec.region
        } else {
            None
        };
        let area = match region {
            Some(r) => {
                let est = geo.estimated_scale.max(f64::EPSILON);
                IRect::from_xywh(
                    geo.x + (f64::from(r.x) / est).floor() as i64,
                    geo.y + (f64::from(r.y) / est).floor() as i64,
                    (f64::from(r.width) / est).ceil() as i64,
                    (f64::from(r.height) / est).ceil() as i64,
                )
                .intersect(&geo.rect())
                .ok_or_else(|| {
                    PlatformError::Capture("The selected region is outside the display".into())
                })?
            }
            None => geo.rect(),
        };

        let check_privacy = has_window_rules(rules);
        // Snapshot before capturing so excluded content is never grabbed.
        let before = match snapshot_windows(&mut self.apps) {
            Ok(w) => w,
            Err(e) if check_privacy => {
                return Err(PlatformError::Capture(format!(
                    "cannot verify privacy rules, frame not captured ({e})"
                )));
            }
            Err(_) => Vec::new(),
        };
        let before_meta: Vec<WinMeta> = before.iter().map(|w| w.meta.clone()).collect();
        let pre = evaluate_area_privacy(&before_meta, area, own_pid, rules);
        if let Some(reason) = pre.skip_reason {
            return Ok(CaptureOutcome::Skipped(reason));
        }

        let mut image = monitor
            .capture_image()
            .map_err(|e| xcap_error("capturing the display", e))?;

        let mut masks = pre.masks;
        if check_privacy {
            // Re-check after capturing: focus or titles may have changed while
            // the frame was grabbed (e.g. a password manager came to front).
            let after = snapshot_windows(&mut self.apps).map_err(|e| {
                PlatformError::Capture(format!(
                    "cannot verify privacy rules, frame discarded ({e})"
                ))
            })?;
            let after_meta: Vec<WinMeta> = after.iter().map(|w| w.meta.clone()).collect();
            let post = evaluate_area_privacy(&after_meta, area, own_pid, rules);
            if let Some(reason) = post.skip_reason {
                return Ok(CaptureOutcome::Skipped(reason));
            }
            masks.extend(post.masks);
        }

        let scale = (
            f64::from(image.width()) / geo.width.max(1) as f64,
            f64::from(image.height()) / geo.height.max(1) as f64,
        );
        apply_global_masks(&mut image, &masks, (geo.x, geo.y), scale);
        apply_region_masks(&mut image, rules, 1.0);

        if let Some(r) = region {
            let (iw, ih) = image.dimensions();
            let x = (r.x.max(0) as u32).min(iw);
            let y = (r.y.max(0) as u32).min(ih);
            let w = r.width.min(iw - x);
            let h = r.height.min(ih - y);
            if w == 0 || h == 0 {
                return Err(PlatformError::Capture(
                    "The selected region is outside the display".into(),
                ));
            }
            image = image::imageops::crop_imm(&image, x, y, w, h).to_image();
        }

        let front = pre.front.map(|i| &before_meta[i]);
        Ok(CaptureOutcome::Frame(Captured {
            image,
            application: front.and_then(|w| w.app_opt().map(str::to_string)),
            window_title: front.and_then(|w| w.title_opt().map(str::to_string)),
            label: label_or(spec, || geo.label.clone()),
        }))
    }
}

/// Captures a single window (Window / Application sources).
fn capture_window(
    info: WinInfo,
    spec: &ScreenSourceSpec,
    rules: &[PrivacyRule],
    own_pid: u32,
) -> Result<CaptureOutcome, PlatformError> {
    let meta = &info.meta;
    if meta.pid == own_pid {
        return Err(PlatformError::Unsupported(
            "OpenMeet's own windows are never captured".into(),
        ));
    }
    if let Some(reason) = meta.excluded(rules) {
        return Ok(CaptureOutcome::Skipped(reason));
    }
    if meta.minimized {
        return Err(PlatformError::Capture(
            "The selected window is minimized; restore it to resume screen context".into(),
        ));
    }
    let mut image = info
        .handle
        .capture_image()
        .map_err(|e| xcap_error("capturing the window", e))?;
    // Titles change with content (browser tabs, documents): re-check now that
    // the pixels were grabbed.
    let title_after = info.handle.title().unwrap_or_default();
    let title_after_opt = Some(title_after.as_str()).filter(|t| !t.trim().is_empty());
    if let Some(reason) = excluded_by(meta.app_opt(), title_after_opt, rules) {
        return Ok(CaptureOutcome::Skipped(reason));
    }
    apply_region_masks(&mut image, rules, 1.0);
    let app = meta.app_opt().map(str::to_string);
    Ok(CaptureOutcome::Frame(Captured {
        image,
        label: label_or(spec, || match &app {
            Some(a) => format!("Window: {a}"),
            None => "Window".to_string(),
        }),
        application: app,
        window_title: title_after_opt.map(str::to_string),
    }))
}

/// The front-most non-minimized window of an application (by name).
fn pick_application_window(
    windows: Vec<WinInfo>,
    app: &str,
    own_pid: u32,
) -> Result<WinInfo, PlatformError> {
    let wanted = normalize_app(app);
    if wanted.is_empty() {
        return Err(PlatformError::DeviceNotFound(
            "No application selected".into(),
        ));
    }
    let mut matching: Vec<WinInfo> = windows
        .into_iter()
        .filter(|w| w.meta.pid != own_pid && normalize_app(&w.meta.app) == wanted)
        .collect();
    if matching.is_empty() {
        return Err(PlatformError::DeviceNotFound(format!(
            "{app} has no open windows"
        )));
    }
    // `windows` is sorted front-most first; prefer a visible window.
    match matching.iter().position(|w| !w.meta.minimized) {
        Some(pos) => Ok(matching.swap_remove(pos)),
        None => Ok(matching.swap_remove(0)),
    }
}

fn label_or(spec: &ScreenSourceSpec, fallback: impl FnOnce() -> String) -> String {
    if spec.label.trim().is_empty() {
        fallback()
    } else {
        spec.label.clone()
    }
}

pub(crate) fn parse_window_id(id: &str) -> Result<u32, PlatformError> {
    id.trim()
        .parse::<u32>()
        .map_err(|_| PlatformError::DeviceNotFound("Invalid window id".into()))
}

/// Geometry of a monitor in global units plus a display label.
pub(crate) struct MonitorGeometry {
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
    /// Pixels per global unit before capturing (2.0 on Retina macOS
    /// displays, 1.0 on Windows where geometry is already in pixels).
    pub estimated_scale: f64,
    pub label: String,
    pub is_primary: bool,
}

impl MonitorGeometry {
    pub fn of(monitor: &xcap::Monitor) -> Result<MonitorGeometry, PlatformError> {
        let x = monitor
            .x()
            .map_err(|e| xcap_error("reading display geometry", e))?;
        let y = monitor
            .y()
            .map_err(|e| xcap_error("reading display geometry", e))?;
        let width = monitor
            .width()
            .map_err(|e| xcap_error("reading display geometry", e))?;
        let height = monitor
            .height()
            .map_err(|e| xcap_error("reading display geometry", e))?;
        let estimated_scale = if cfg!(target_os = "macos") {
            f64::from(monitor.scale_factor().unwrap_or(1.0)).max(1.0)
        } else {
            1.0
        };
        Ok(MonitorGeometry {
            x: i64::from(x),
            y: i64::from(y),
            width: i64::from(width),
            height: i64::from(height),
            estimated_scale,
            label: display_label(monitor),
            is_primary: monitor.is_primary().unwrap_or(false),
        })
    }

    pub fn rect(&self) -> IRect {
        IRect::from_xywh(self.x, self.y, self.width, self.height)
    }

    /// Size in physical pixels.
    pub fn pixel_size(&self) -> (u32, u32) {
        (
            (self.width as f64 * self.estimated_scale).round() as u32,
            (self.height as f64 * self.estimated_scale).round() as u32,
        )
    }
}

/// "Display 2 (DELL U2720Q)" style label; the number is the 1-based position
/// in the OS display list.
fn display_label(monitor: &xcap::Monitor) -> String {
    let index = xcap::Monitor::all()
        .ok()
        .and_then(|all| {
            let id = monitor.id().ok()?;
            all.iter().position(|m| m.id().ok() == Some(id))
        })
        .map(|i| i + 1);
    let friendly = monitor
        .friendly_name()
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty() && !n.starts_with("Unknown Monitor"));
    match (index, friendly) {
        (Some(i), Some(f)) => format!("Display {i} ({f})"),
        (Some(i), None) => format!("Display {i}"),
        (None, Some(f)) => f,
        (None, None) => "Display".to_string(),
    }
}

/// Finds a monitor by numeric id, device name or friendly name; "" or
/// "primary" selects the primary display.
pub(crate) fn find_monitor(id: &str) -> Result<xcap::Monitor, PlatformError> {
    let monitors = xcap::Monitor::all().map_err(|e| xcap_error("listing displays", e))?;
    if monitors.is_empty() {
        return Err(PlatformError::DeviceNotFound(
            "No display is available".into(),
        ));
    }
    let wanted = id.trim();
    if wanted.is_empty() || wanted.eq_ignore_ascii_case("primary") {
        let pos = monitors
            .iter()
            .position(|m| m.is_primary().unwrap_or(false))
            .unwrap_or(0);
        return monitors
            .into_iter()
            .nth(pos)
            .ok_or_else(|| PlatformError::DeviceNotFound("No display is available".into()));
    }
    if let Ok(num) = wanted.parse::<u32>() {
        if let Some(m) = monitors.iter().find(|m| m.id().ok() == Some(num)) {
            return Ok(m.clone());
        }
    }
    monitors
        .into_iter()
        .find(|m| {
            m.name().is_ok_and(|n| n.eq_ignore_ascii_case(wanted))
                || m.friendly_name()
                    .is_ok_and(|n| n.eq_ignore_ascii_case(wanted))
        })
        .ok_or_else(|| {
            PlatformError::DeviceNotFound("The selected display is no longer connected".into())
        })
}

/// Maps an xcap error to a capture error with context (xcap errors carry
/// OS error codes / messages, never window contents).
pub(crate) fn xcap_error(context: &str, e: xcap::XCapError) -> PlatformError {
    PlatformError::Capture(format!("{context}: {e}"))
}

/// Ensures the process may record the screen (macOS Screen Recording
/// permission). Always Ok elsewhere.
pub(crate) fn ensure_screen_permission() -> Result<(), PlatformError> {
    #[cfg(target_os = "macos")]
    {
        if !objc2_core_graphics::CGPreflightScreenCaptureAccess() {
            return Err(PlatformError::PermissionDenied(
                SCREEN_PERMISSION_MESSAGE.to_string(),
            ));
        }
    }
    Ok(())
}

/// Asks macOS to show the Screen Recording permission prompt (once per
/// process). No-op elsewhere.
pub(crate) fn request_screen_permission_once() {
    #[cfg(target_os = "macos")]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static REQUESTED: AtomicBool = AtomicBool::new(false);
        if !REQUESTED.swap(true, Ordering::SeqCst) {
            let _ = objc2_core_graphics::CGRequestScreenCaptureAccess();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::types::PrivacyAction;

    const OWN: u32 = 999;

    fn win(
        id: u32,
        pid: u32,
        app: &str,
        title: &str,
        rect: (i64, i64, i64, i64),
        z: i64,
    ) -> WinMeta {
        WinMeta {
            id,
            pid,
            app: app.into(),
            title: title.into(),
            rect: IRect::from_xywh(rect.0, rect.1, rect.2, rect.3),
            minimized: false,
            focused: false,
            z,
            opaque: true,
        }
    }

    fn rule(t: PrivacyRuleType, v: &str) -> PrivacyRule {
        PrivacyRule {
            id: v.into(),
            rule_type: t,
            value: v.into(),
            action: PrivacyAction::Exclude,
            created_at: 0,
        }
    }

    fn display() -> IRect {
        IRect::from_xywh(0, 0, 1920, 1080)
    }

    #[test]
    fn focused_excluded_window_skips_frame() {
        let mut pm = win(1, 10, "1Password", "Vault", (100, 100, 800, 600), 5);
        pm.focused = true;
        let windows = vec![pm, win(2, 11, "Code", "main.rs", (0, 0, 1920, 1080), 4)];
        let rules = [rule(PrivacyRuleType::Application, "1Password")];
        let p = evaluate_area_privacy(&windows, display(), OWN, &rules);
        assert_eq!(p.front, Some(0));
        assert_eq!(
            p.skip_reason.as_deref(),
            Some("application rule \"1Password\"")
        );
    }

    #[test]
    fn background_excluded_window_is_masked_where_visible() {
        let mut code = win(1, 11, "Code", "main.rs", (0, 0, 1000, 1080), 5);
        code.focused = true;
        let slack = win(2, 12, "Slack", "general", (800, 0, 1120, 1080), 4);
        let windows = vec![code, slack];
        let rules = [rule(PrivacyRuleType::Application, "Slack")];
        let p = evaluate_area_privacy(&windows, display(), OWN, &rules);
        assert_eq!(p.front, Some(0));
        assert!(p.skip_reason.is_none());
        // Only the part of Slack not covered by Code (x >= 1000) is masked.
        let area: i64 = p.masks.iter().map(IRect::area).sum();
        assert_eq!(area, 920 * 1080);
        assert!(p.masks.iter().all(|m| m.x0 >= 1000));
    }

    #[test]
    fn fully_hidden_excluded_window_needs_no_mask() {
        let mut browser = win(1, 11, "Google Chrome", "Docs", (0, 0, 1920, 1080), 5);
        browser.focused = true;
        let slack = win(2, 12, "Slack", "general", (0, 0, 1920, 1080), 4);
        let rules = [rule(PrivacyRuleType::Application, "Slack")];
        let p = evaluate_area_privacy(&[browser, slack], display(), OWN, &rules);
        assert!(p.skip_reason.is_none());
        assert!(p.masks.is_empty());
    }

    #[test]
    fn transparent_windows_do_not_hide_sensitive_content() {
        let mut overlay = win(
            1,
            20,
            "GeForce Overlay",
            "NVIDIA GeForce Overlay",
            (0, 0, 1920, 1080),
            9,
        );
        overlay.opaque = false;
        let mut code = win(2, 11, "Code", "main.rs", (0, 0, 1000, 1080), 5);
        code.focused = true;
        let signal = win(3, 12, "Signal", "Chats", (1000, 0, 920, 1080), 4);
        let rules = [rule(PrivacyRuleType::Application, "Signal")];
        let p = evaluate_area_privacy(&[overlay, code, signal], display(), OWN, &rules);
        let area: i64 = p.masks.iter().map(IRect::area).sum();
        assert_eq!(area, 920 * 1080);
    }

    #[test]
    fn own_and_minimized_windows_are_ignored() {
        let mut own = win(1, OWN, "OpenMeet", "Overlay", (1500, 0, 400, 600), 9);
        own.focused = true;
        let mut minimized = win(2, 12, "1Password", "Vault", (0, 0, 800, 600), 8);
        minimized.minimized = true;
        let docs = win(3, 13, "Google Chrome", "Design doc", (0, 0, 1920, 1080), 5);
        let rules = [rule(PrivacyRuleType::Application, "1Password")];
        let p = evaluate_area_privacy(&[own, minimized, docs], display(), OWN, &rules);
        // Focused window is our own overlay -> the large browser window is front.
        assert_eq!(p.front, Some(2));
        assert!(p.skip_reason.is_none());
        assert!(p.masks.is_empty());
    }

    #[test]
    fn small_topmost_windows_do_not_become_front() {
        let taskbar = win(1, 30, "Windows Explorer", "", (0, 1040, 1920, 40), 9);
        let bank = win(
            2,
            31,
            "Google Chrome",
            "Example Bank - Google Chrome",
            (0, 0, 1920, 1040),
            5,
        );
        let rules = [rule(PrivacyRuleType::Domain, "examplebank.com")];
        let p = evaluate_area_privacy(&[taskbar, bank], display(), OWN, &rules);
        assert_eq!(p.front, Some(1));
        assert_eq!(
            p.skip_reason.as_deref(),
            Some("domain rule \"examplebank.com\"")
        );
    }

    #[test]
    fn windows_on_other_displays_are_ignored() {
        let mut other = win(1, 12, "1Password", "Vault", (1920, 0, 1920, 1080), 9);
        other.focused = true;
        let code = win(2, 11, "Code", "main.rs", (0, 0, 1920, 1080), 5);
        let rules = [rule(PrivacyRuleType::Application, "1Password")];
        let p = evaluate_area_privacy(&[other, code], display(), OWN, &rules);
        assert_eq!(p.front, Some(1));
        assert!(p.skip_reason.is_none());
        assert!(p.masks.is_empty());
    }

    #[test]
    fn no_window_rules_means_no_privacy_work() {
        let mut pm = win(1, 10, "1Password", "Vault", (0, 0, 800, 600), 5);
        pm.focused = true;
        let rules = [rule(PrivacyRuleType::Region, "0,0,10,10")];
        let p = evaluate_area_privacy(&[pm], display(), OWN, &rules);
        assert_eq!(p.front, Some(0));
        assert!(p.skip_reason.is_none());
        assert!(!has_window_rules(&rules));
        assert!(has_window_rules(&[rule(PrivacyRuleType::Window, "x")]));
        assert!(!has_window_rules(&[rule(PrivacyRuleType::Window, "  ")]));
    }

    #[test]
    fn region_area_limits_evaluation() {
        // Excluded window outside the captured region is irrelevant.
        let mut pm = win(1, 10, "1Password", "Vault", (1500, 0, 400, 400), 9);
        pm.focused = true;
        let code = win(2, 11, "Code", "main.rs", (0, 0, 1920, 1080), 5);
        let rules = [rule(PrivacyRuleType::Application, "1Password")];
        let region = IRect::from_xywh(0, 500, 800, 500);
        let p = evaluate_area_privacy(&[pm, code], region, OWN, &rules);
        assert!(p.skip_reason.is_none());
        assert!(p.masks.is_empty());
    }

    #[test]
    fn global_masks_respect_origin_and_scale() {
        let mut img = RgbaImage::from_pixel(200, 100, image::Rgba([255, 255, 255, 255]));
        // Display at x=1920 with 2x pixel density; mask global (1930,10)-(1940,20).
        let masks = [IRect::from_xywh(1930, 10, 10, 10)];
        apply_global_masks(&mut img, &masks, (1920, 0), (2.0, 2.0));
        assert_eq!(img.get_pixel(20, 20).0, [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(39, 39).0, [0, 0, 0, 255]);
        assert_eq!(img.get_pixel(40, 40).0, [255, 255, 255, 255]);
        assert_eq!(img.get_pixel(19, 19).0, [255, 255, 255, 255]);
    }

    #[test]
    fn window_id_parsing() {
        assert_eq!(parse_window_id(" 42 ").unwrap(), 42);
        assert!(matches!(
            parse_window_id("abc"),
            Err(PlatformError::DeviceNotFound(_))
        ));
    }
}
