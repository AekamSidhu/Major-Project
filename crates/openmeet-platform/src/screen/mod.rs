//! Screen context capture.
//!
//! Screen capture is always explicit: the user selects a display, region,
//! window or application ([`ScreenSourceSpec`]). Frames go through a fixed
//! pipeline:
//!
//! 1. **Privacy rules** – application / window / domain rules are checked
//!    *before* pixels are grabbed (and re-checked right after). For display
//!    and region sources the focused (or top-most large) window decides: when
//!    it matches a rule the frame is skipped ([`ScreenStatus::SkippedPrivacy`]).
//!    Other matching windows are blacked out where they are visible. Window
//!    and application sources refuse excluded targets. OpenMeet's own windows
//!    are never used as a source.
//! 2. **Region masks** – region rules are blacked out at source resolution.
//! 3. **Downscale** – aspect-preserving, to at most `max_width` pixels wide.
//! 4. **Change detection** – [`diff::ChangeDetector`] drops frames that did not
//!    change meaningfully (manual captures bypass it).
//! 5. **PNG encode** – RGB PNG in memory ([`ScreenFrame`]); nothing is
//!    written to disk.

mod capture;
pub mod diff;
mod geometry;
pub mod rules;

use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use image::RgbaImage;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use openmeet_core::ipc::{ScreenSourceInfo, ScreenSourceKind, ScreenSourceSpec};
use openmeet_core::settings::ScreenFrequency;
use openmeet_core::types::{PrivacyRule, ScreenFrame};
use openmeet_core::{new_id, now_ms};

use self::capture::{
    AppNameCache, CaptureOutcome, Captured, Capturer, MonitorGeometry, ensure_screen_permission,
    find_monitor, parse_window_id, request_screen_permission_once, snapshot_windows, xcap_error,
};
use self::diff::ChangeDetector;
use self::rules::{excluded_by, normalize_app};
use crate::PlatformError;
use crate::error::guard_panics;

/// Default downscale target width.
pub const DEFAULT_MAX_WIDTH: u32 = 1280;

/// Lists selectable screen sources: displays (primary first), windows
/// (front-most first) and applications with open windows (alphabetical).
/// OpenMeet's own windows are never listed. Sources matching a privacy rule
/// are still listed but flagged with `excluded_by_rule`.
pub fn list_sources(rules: &[PrivacyRule]) -> Result<Vec<ScreenSourceInfo>, PlatformError> {
    guard_panics("listing screen sources", || {
        #[cfg(windows)]
        let _dpi = crate::win::DpiAwareGuard::new();
        let own_pid = std::process::id();
        let monitors = xcap::Monitor::all().map_err(|e| xcap_error("listing displays", e))?;

        let mut displays = Vec::with_capacity(monitors.len());
        for monitor in &monitors {
            let Ok(id) = monitor.id() else { continue };
            let Ok(geo) = MonitorGeometry::of(monitor) else {
                continue;
            };
            let (width, height) = geo.pixel_size();
            displays.push(ScreenSourceInfo {
                kind: ScreenSourceKind::Display,
                id: id.to_string(),
                label: geo.label.clone(),
                application: None,
                title: None,
                width,
                height,
                is_primary: geo.is_primary,
                excluded_by_rule: false,
            });
        }
        displays.sort_by_key(|d| !d.is_primary);

        let windows = match snapshot_windows(&mut AppNameCache::default()) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!(error = %e, "window enumeration failed; listing displays only");
                Vec::new()
            }
        };
        let mut window_sources = Vec::new();
        let mut apps: Vec<ScreenSourceInfo> = Vec::new();
        for w in windows.iter().map(|w| &w.meta) {
            if w.pid == own_pid || w.rect.is_empty() {
                continue;
            }
            let title = w.title.trim();
            let app = w.app.trim();
            if title.is_empty() {
                continue;
            }
            let app_opt = Some(app).filter(|a| !a.is_empty());
            window_sources.push(ScreenSourceInfo {
                kind: ScreenSourceKind::Window,
                id: w.id.to_string(),
                label: match app_opt {
                    Some(a) if !title.to_lowercase().contains(&a.to_lowercase()) => {
                        format!("{} — {a}", truncate(title, 80))
                    }
                    _ => truncate(title, 80),
                },
                application: app_opt.map(str::to_string),
                title: Some(title.to_string()),
                width: w.rect.width() as u32,
                height: w.rect.height() as u32,
                is_primary: false,
                excluded_by_rule: excluded_by(app_opt, Some(title), rules).is_some(),
            });
            if let Some(a) = app_opt {
                let key = normalize_app(a);
                if !apps.iter().any(|s| normalize_app(&s.id) == key) {
                    apps.push(ScreenSourceInfo {
                        kind: ScreenSourceKind::Application,
                        id: a.to_string(),
                        label: a.to_string(),
                        application: Some(a.to_string()),
                        title: None,
                        width: w.rect.width() as u32,
                        height: w.rect.height() as u32,
                        is_primary: false,
                        excluded_by_rule: excluded_by(Some(a), None, rules).is_some(),
                    });
                }
            }
        }
        apps.sort_by_key(|a| a.label.to_lowercase());

        let mut out = displays;
        out.extend(window_sources);
        out.extend(apps);
        Ok(out)
    })
}

/// Configuration of a [`ScreenSampler`].
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// What to capture.
    pub spec: ScreenSourceSpec,
    /// Sampling frequency ([`ScreenFrequency::Manual`] = only [`ScreenSampler::capture_now`]).
    pub frequency: ScreenFrequency,
    /// Minimum change score (0..1) for an automatic frame to be emitted.
    pub change_threshold: f32,
    /// Privacy rules (can be replaced with [`ScreenSampler::update_rules`]).
    pub rules: Vec<PrivacyRule>,
    /// Downscale target, default 1280 (0 = keep source resolution).
    pub max_width: u32,
    /// Never capture OpenMeet's own windows as a source.
    pub own_process_id: u32,
}

impl SamplerConfig {
    /// Defaults for `spec`: 1 fps, threshold 0.08, no rules, 1280 px, the
    /// current process as OpenMeet's own process.
    pub fn new(spec: ScreenSourceSpec) -> Self {
        SamplerConfig {
            spec,
            frequency: ScreenFrequency::Fps1,
            change_threshold: 0.08,
            rules: Vec::new(),
            max_width: DEFAULT_MAX_WIDTH,
            own_process_id: std::process::id(),
        }
    }
}

/// Sampler status updates (deduplicated: emitted when the status changes, and
/// after every manual capture).
#[derive(Debug, Clone, PartialEq)]
pub enum ScreenStatus {
    /// A frame was captured and emitted.
    Capturing,
    /// Captured, but the content did not change enough to emit a frame.
    Unchanged,
    /// Skipped because a privacy rule matched (names the rule).
    SkippedPrivacy(String),
    /// The OS denied screen capture (message tells the user how to fix it).
    PermissionDenied(String),
    /// Capture failed (display disconnected, window closed / minimized, ...).
    Error(String),
}

enum Command {
    CaptureNow,
    UpdateRules(Vec<PrivacyRule>),
    Stop,
}

/// Periodic, privacy-aware screen sampler running on a dedicated thread.
/// Dropping it stops the thread.
pub struct ScreenSampler {
    tx: mpsc::Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

impl ScreenSampler {
    /// Validates the source (and, on macOS, the Screen Recording permission)
    /// and starts sampling. The first automatic capture happens immediately.
    /// `on_frame` / `on_status` are called on the sampler thread.
    pub fn start(
        config: SamplerConfig,
        on_frame: Box<dyn FnMut(ScreenFrame) + Send>,
        on_status: Box<dyn FnMut(ScreenStatus) + Send>,
    ) -> Result<ScreenSampler, PlatformError> {
        preflight(&config.spec, config.own_process_id)?;
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("openmeet-screen".to_string())
            .spawn(move || run_sampler(config, rx, on_frame, on_status))
            .map_err(|e| {
                PlatformError::Other(format!("could not start the screen sampler thread: {e}"))
            })?;
        tracing::info!("screen sampler started");
        Ok(ScreenSampler {
            tx,
            thread: Some(thread),
        })
    }

    /// Manual capture: bypasses change detection (ScreenFrame.manual = true), still applies privacy rules.
    pub fn capture_now(&self) {
        let _ = self.tx.send(Command::CaptureNow);
    }

    /// Replaces the privacy rules used for subsequent captures.
    pub fn update_rules(&self, rules: Vec<PrivacyRule>) {
        let _ = self.tx.send(Command::UpdateRules(rules));
    }

    /// Stops the sampler and waits for its thread to finish.
    pub fn stop(self) {
        drop(self);
    }
}

impl Drop for ScreenSampler {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Stop);
        if let Some(thread) = self.thread.take() {
            // Dropping from inside a callback must not join our own thread.
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
        tracing::info!("screen sampler stopped");
    }
}

/// Captures one frame (manual = true, change score 1.0) without starting a
/// sampler. Privacy rules apply; a skipped capture returns
/// [`PlatformError::Capture`] naming the matching rule.
pub fn capture_once(
    spec: &ScreenSourceSpec,
    rules: &[PrivacyRule],
    max_width: u32,
) -> Result<ScreenFrame, PlatformError> {
    guard_panics("screen capture", || {
        let captured_at = now_ms();
        let started = Instant::now();
        match Capturer::new().capture(spec, rules, std::process::id())? {
            CaptureOutcome::Skipped(reason) => Err(PlatformError::Capture(format!(
                "skipped because it matches the privacy {reason}"
            ))),
            CaptureOutcome::Frame(captured) => {
                let mut detector = ChangeDetector::new(0.0);
                let frame = finish_frame(captured, max_width, &mut detector, true, captured_at)?
                    .ok_or_else(|| {
                        PlatformError::Other("manual capture produced no frame".into())
                    })?;
                log_frame(&frame, started);
                Ok(frame)
            }
        }
    })
}

/// Checks that the source exists and can be captured before a sampler starts.
fn preflight(spec: &ScreenSourceSpec, own_pid: u32) -> Result<(), PlatformError> {
    guard_panics("checking the screen source", || {
        if let Err(e) = ensure_screen_permission() {
            request_screen_permission_once();
            return Err(e);
        }
        #[cfg(windows)]
        let _dpi = crate::win::DpiAwareGuard::new();
        match spec.kind {
            ScreenSourceKind::Display | ScreenSourceKind::Region => {
                find_monitor(&spec.id).map(|_| ())
            }
            ScreenSourceKind::Window => {
                let id = parse_window_id(&spec.id)?;
                let windows = snapshot_windows(&mut AppNameCache::default())?;
                match windows.iter().find(|w| w.meta.id == id) {
                    None => Err(PlatformError::DeviceNotFound(
                        "The selected window is no longer open".into(),
                    )),
                    Some(w) if w.meta.pid == own_pid => Err(PlatformError::Unsupported(
                        "OpenMeet's own windows cannot be used as a screen source".into(),
                    )),
                    Some(_) => Ok(()),
                }
            }
            ScreenSourceKind::Application => {
                let wanted = normalize_app(&spec.id);
                let windows = snapshot_windows(&mut AppNameCache::default())?;
                if !wanted.is_empty()
                    && windows
                        .iter()
                        .any(|w| w.meta.pid != own_pid && normalize_app(&w.meta.app) == wanted)
                {
                    Ok(())
                } else {
                    Err(PlatformError::DeviceNotFound(format!(
                        "{} has no open windows",
                        spec.id.trim()
                    )))
                }
            }
        }
    })
}

fn run_sampler(
    config: SamplerConfig,
    rx: mpsc::Receiver<Command>,
    mut on_frame: Box<dyn FnMut(ScreenFrame) + Send>,
    mut on_status: Box<dyn FnMut(ScreenStatus) + Send>,
) {
    #[cfg(windows)]
    let _dpi = crate::win::DpiAwareGuard::new();
    let SamplerConfig {
        spec,
        frequency,
        change_threshold,
        mut rules,
        max_width,
        own_process_id,
    } = config;
    let interval = frequency.interval_ms().map(Duration::from_millis);
    let mut capturer = Capturer::new();
    let mut detector = ChangeDetector::new(change_threshold);
    let mut last_status: Option<ScreenStatus> = None;
    let mut next_tick = Instant::now();

    loop {
        let timeout = match interval {
            Some(_) => next_tick.saturating_duration_since(Instant::now()),
            None => Duration::from_secs(3600),
        };
        let manual = match rx.recv_timeout(timeout) {
            Ok(Command::CaptureNow) => true,
            Ok(Command::UpdateRules(new_rules)) => {
                rules = new_rules;
                continue;
            }
            Ok(Command::Stop) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {
                if interval.is_none() {
                    continue;
                }
                false
            }
        };
        if let (false, Some(step)) = (manual, interval) {
            next_tick += step;
            let now = Instant::now();
            if next_tick < now {
                // Capturing is slower than the requested rate: skip missed ticks.
                next_tick = now + step;
            }
        }

        let started = Instant::now();
        let captured_at = now_ms();
        let result = guard_panics("screen capture", || {
            match capturer.capture(&spec, &rules, own_process_id)? {
                CaptureOutcome::Skipped(reason) => Ok(Err(ScreenStatus::SkippedPrivacy(reason))),
                CaptureOutcome::Frame(captured) => {
                    finish_frame(captured, max_width, &mut detector, manual, captured_at).map(Ok)
                }
            }
        });
        let status = match result {
            Ok(Ok(Some(frame))) => {
                log_frame(&frame, started);
                on_frame(frame);
                ScreenStatus::Capturing
            }
            Ok(Ok(None)) => ScreenStatus::Unchanged,
            Ok(Err(skipped)) => {
                tracing::debug!("screen frame skipped by a privacy rule");
                skipped
            }
            Err(PlatformError::PermissionDenied(m)) => ScreenStatus::PermissionDenied(m),
            Err(e) => {
                tracing::debug!(error = %e, "screen capture failed");
                ScreenStatus::Error(e.to_string())
            }
        };
        if manual || last_status.as_ref() != Some(&status) {
            on_status(status.clone());
            last_status = Some(status);
        }
    }
}

/// Downscale → change detection → PNG. Returns None when the frame did not
/// change enough (never for manual captures).
fn finish_frame(
    captured: Captured,
    max_width: u32,
    detector: &mut ChangeDetector,
    manual: bool,
    captured_at: i64,
) -> Result<Option<ScreenFrame>, PlatformError> {
    let Captured {
        image,
        application,
        window_title,
        label,
    } = captured;
    if image.width() == 0 || image.height() == 0 {
        return Err(PlatformError::Capture("the captured image is empty".into()));
    }
    let image = downscale(image, max_width);
    let (changed, change_score, hash) = if manual {
        let (score, hash) = detector.force_accept(&image);
        (true, score, hash)
    } else {
        detector.evaluate(&image)
    };
    if !changed {
        return Ok(None);
    }
    let png = encode_png(&image)?;
    Ok(Some(ScreenFrame {
        id: new_id(),
        captured_at,
        source_label: label,
        application,
        window_title,
        width: image.width(),
        height: image.height(),
        png,
        hash,
        change_score,
        manual,
    }))
}

/// Aspect-preserving downscale to at most `max_width` pixels (0 = no limit).
fn downscale(image: RgbaImage, max_width: u32) -> RgbaImage {
    if max_width == 0 || image.width() <= max_width {
        return image;
    }
    let height = ((u64::from(image.height()) * u64::from(max_width) + u64::from(image.width()) / 2)
        / u64::from(image.width()))
    .max(1) as u32;
    image::imageops::thumbnail(&image, max_width, height)
}

/// Encodes an RGBA image as an RGB PNG (screens are opaque; dropping alpha
/// makes the file smaller and faster to encode).
fn encode_png(image: &RgbaImage) -> Result<Vec<u8>, PlatformError> {
    let mut rgb = Vec::with_capacity(image.width() as usize * image.height() as usize * 3);
    for px in image.as_raw().chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    let mut out = Vec::with_capacity(rgb.len() / 4);
    PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::Adaptive)
        .write_image(&rgb, image.width(), image.height(), ExtendedColorType::Rgb8)
        .map_err(|e| PlatformError::Capture(format!("PNG encoding failed: {e}")))?;
    Ok(out)
}

fn log_frame(frame: &ScreenFrame, started: Instant) {
    tracing::debug!(
        frame_id = %frame.id,
        width = frame.width,
        height = frame.height,
        png_bytes = frame.png.len(),
        change_score = frame.change_score,
        manual = frame.manual,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "screen frame captured"
    );
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn captured(w: u32, h: u32) -> Captured {
        let mut image = RgbaImage::from_pixel(w, h, Rgba([240, 240, 240, 255]));
        for x in 0..w / 2 {
            for y in 0..h / 3 {
                image.put_pixel(x, y, Rgba([20, 30, 40, 255]));
            }
        }
        Captured {
            image,
            application: Some("Code".into()),
            window_title: Some("main.rs".into()),
            label: "Display 1".into(),
        }
    }

    #[test]
    fn downscale_keeps_aspect_ratio() {
        let img = RgbaImage::new(2560, 1440);
        let small = downscale(img, 1280);
        assert_eq!(small.dimensions(), (1280, 720));
        let odd = downscale(RgbaImage::new(1999, 1001), 1000);
        assert_eq!(odd.dimensions(), (1000, 501));
        let untouched = downscale(RgbaImage::new(800, 600), 1280);
        assert_eq!(untouched.dimensions(), (800, 600));
        let unlimited = downscale(RgbaImage::new(4000, 100), 0);
        assert_eq!(unlimited.dimensions(), (4000, 100));
        let thin = downscale(RgbaImage::new(5000, 1), 100);
        assert_eq!(thin.dimensions(), (100, 1));
    }

    #[test]
    fn png_roundtrip() {
        let c = captured(64, 48);
        let png = encode_png(&c.image).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        let decoded = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), (64, 48));
        assert_eq!(decoded.get_pixel(0, 0), &Rgba([20, 30, 40, 255]));
        assert_eq!(decoded.get_pixel(63, 47), &Rgba([240, 240, 240, 255]));
    }

    #[test]
    fn finish_frame_pipeline() {
        let mut detector = ChangeDetector::new(0.08);
        let frame = finish_frame(captured(2560, 1440), 1280, &mut detector, false, 42)
            .unwrap()
            .expect("first frame is always emitted");
        assert_eq!((frame.width, frame.height), (1280, 720));
        assert_eq!(frame.captured_at, 42);
        assert_eq!(frame.change_score, 1.0);
        assert!(!frame.manual);
        assert_eq!(frame.source_label, "Display 1");
        assert_eq!(frame.application.as_deref(), Some("Code"));
        assert!(!frame.png.is_empty());
        // Identical content: dropped by change detection...
        assert!(
            finish_frame(captured(2560, 1440), 1280, &mut detector, false, 43)
                .unwrap()
                .is_none()
        );
        // ...unless it is a manual capture.
        let manual = finish_frame(captured(2560, 1440), 1280, &mut detector, true, 44)
            .unwrap()
            .expect("manual frames bypass change detection");
        assert!(manual.manual);
        assert_eq!(manual.change_score, 0.0);
        assert_eq!(manual.hash, frame.hash);
        assert_ne!(manual.id, frame.id);
    }

    #[test]
    fn finish_frame_rejects_empty_images() {
        let mut detector = ChangeDetector::new(0.08);
        let empty = Captured {
            image: RgbaImage::new(0, 0),
            application: None,
            window_title: None,
            label: "x".into(),
        };
        assert!(finish_frame(empty, 1280, &mut detector, true, 0).is_err());
    }

    #[test]
    fn sampler_config_defaults() {
        let spec = ScreenSourceSpec {
            kind: ScreenSourceKind::Display,
            id: "primary".into(),
            label: "Display 1".into(),
            region: None,
        };
        let c = SamplerConfig::new(spec);
        assert_eq!(c.max_width, 1280);
        assert_eq!(c.frequency, ScreenFrequency::Fps1);
        assert_eq!(c.own_process_id, std::process::id());
        assert!((c.change_threshold - 0.08).abs() < f32::EPSILON);
    }

    #[test]
    fn truncation() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("abcdefghij", 5), "abcd…");
        assert_eq!(truncate("ééééé", 5), "ééééé");
    }
}
