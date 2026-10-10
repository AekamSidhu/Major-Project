//! Hardware tests: real audio devices, displays, OCR and capture exclusion.
//!
//! Ignored by default (CI has no devices). Run on a desktop with:
//!
//! ```text
//! cargo test -p openmeet-platform --test hardware -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Output is limited to counts, sizes and durations — never OCR text, window
//! titles or audio content.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use openmeet_core::audio::PIPELINE_SAMPLE_RATE;
use openmeet_core::ipc::{ScreenSourceKind, ScreenSourceSpec};
use openmeet_core::settings::ScreenFrequency;
use openmeet_core::types::AudioSourceKind;
use openmeet_platform::audio::{self, AudioCapture, AudioCaptureConfig};
use openmeet_platform::screen::{self, SamplerConfig, ScreenSampler, ScreenStatus};
use parking_lot::Mutex;

fn primary_display() -> ScreenSourceSpec {
    let sources = screen::list_sources(&[]).expect("list screen sources");
    let d = sources
        .iter()
        .find(|s| s.kind == ScreenSourceKind::Display && s.is_primary)
        .or_else(|| sources.iter().find(|s| s.kind == ScreenSourceKind::Display))
        .expect("at least one display");
    ScreenSourceSpec {
        kind: d.kind,
        id: d.id.clone(),
        label: d.label.clone(),
        region: None,
    }
}

struct Captured {
    samples: usize,
    frames: usize,
    monotonic: bool,
    errors: usize,
}

fn capture_for(
    source: AudioSourceKind,
    dur: Duration,
    during: impl FnOnce(),
) -> Result<Captured, String> {
    let samples = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let last_ts = Arc::new(Mutex::new((i64::MIN, true)));
    let (s, f, e, l) = (
        samples.clone(),
        frames.clone(),
        errors.clone(),
        last_ts.clone(),
    );
    let cap = AudioCapture::start(
        AudioCaptureConfig {
            source,
            device: None,
        },
        Box::new(move |frame| {
            s.fetch_add(frame.samples.len(), Ordering::Relaxed);
            f.fetch_add(1, Ordering::Relaxed);
            let mut last = l.lock();
            if frame.captured_at_ms < last.0 {
                last.1 = false;
            }
            last.0 = frame.captured_at_ms;
        }),
        Box::new(move |_err| {
            e.fetch_add(1, Ordering::Relaxed);
        }),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{source:?}: device opened ({} chars in name)",
        cap.device_name().len()
    );
    during();
    std::thread::sleep(dur);
    cap.stop();
    let monotonic = last_ts.lock().1;
    Ok(Captured {
        samples: samples.load(Ordering::Relaxed),
        frames: frames.load(Ordering::Relaxed),
        monotonic,
        errors: errors.load(Ordering::Relaxed),
    })
}

#[test]
#[ignore = "needs audio hardware"]
fn lists_audio_devices() {
    let devices = audio::list_devices().expect("list devices");
    println!(
        "inputs: {}, outputs: {}, system audio supported: {} ({})",
        devices.inputs.len(),
        devices.outputs.len(),
        devices.system_audio.supported,
        devices.system_audio.mechanism
    );
    assert!(!devices.inputs.is_empty(), "no input device found");
    assert!(devices.inputs.iter().filter(|d| d.is_default).count() <= 1);
}

#[test]
#[ignore = "needs a microphone"]
fn captures_microphone_at_16khz() {
    let c = capture_for(AudioSourceKind::Microphone, Duration::from_secs(2), || {})
        .expect("microphone capture");
    let seconds = c.samples as f64 / PIPELINE_SAMPLE_RATE as f64;
    println!(
        "microphone: {} frames, {:.2} s of audio, errors {}, monotonic {}",
        c.frames, seconds, c.errors, c.monotonic
    );
    assert!(c.frames > 10, "too few frames");
    assert!(
        (1.5..=2.8).contains(&seconds),
        "unexpected amount of audio: {seconds:.2}s"
    );
    assert!(c.monotonic, "timestamps went backwards");
    assert_eq!(c.errors, 0);
}

#[cfg(windows)]
#[test]
#[ignore = "needs audio output hardware"]
fn captures_system_audio_loopback_while_a_sound_plays() {
    // WASAPI loopback only delivers packets while something is playing.
    let player = || {
        let wav = r"C:\Windows\Media\Windows Notify System Generic.wav";
        let wav = if std::path::Path::new(wav).exists() {
            wav
        } else {
            r"C:\Windows\Media\chimes.wav"
        };
        std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(New-Object Media.SoundPlayer '{wav}').PlaySync(); (New-Object Media.SoundPlayer '{wav}').PlaySync()"),
            ])
            .spawn()
            .ok();
    };
    let c = capture_for(AudioSourceKind::System, Duration::from_millis(2500), player)
        .expect("loopback capture");
    let seconds = c.samples as f64 / PIPELINE_SAMPLE_RATE as f64;
    println!(
        "loopback: {} frames, {:.2} s of audio, errors {}",
        c.frames, seconds, c.errors
    );
    assert!(
        c.frames > 0,
        "no loopback audio received while a sound was playing"
    );
    assert_eq!(c.errors, 0);
}

#[test]
#[ignore = "needs a display"]
fn lists_screen_sources_and_captures_primary_display() {
    let sources = screen::list_sources(&[]).expect("list sources");
    let displays = sources
        .iter()
        .filter(|s| s.kind == ScreenSourceKind::Display)
        .count();
    let windows = sources
        .iter()
        .filter(|s| s.kind == ScreenSourceKind::Window)
        .count();
    println!(
        "displays: {displays}, windows: {windows}, total sources: {}",
        sources.len()
    );
    assert!(displays >= 1);
    let started = Instant::now();
    let frame =
        screen::capture_once(&primary_display(), &[], 1280).expect("capture primary display");
    println!(
        "captured {}x{} PNG of {} KB in {} ms",
        frame.width,
        frame.height,
        frame.png.len() / 1024,
        started.elapsed().as_millis()
    );
    assert!(frame.width > 0 && frame.width <= 1280);
    assert!(frame.png.starts_with(&[0x89, b'P', b'N', b'G']));
    assert!(frame.manual);
}

#[test]
#[ignore = "needs a display"]
fn sampler_emits_first_frame_then_deduplicates() {
    let frames = Arc::new(AtomicUsize::new(0));
    let statuses = Arc::new(Mutex::new(Vec::<ScreenStatus>::new()));
    let (f, s) = (frames.clone(), statuses.clone());
    let mut config = SamplerConfig::new(primary_display());
    config.frequency = ScreenFrequency::Fps2;
    let sampler = ScreenSampler::start(
        config,
        Box::new(move |_frame| {
            f.fetch_add(1, Ordering::Relaxed);
        }),
        Box::new(move |status| s.lock().push(status)),
    )
    .expect("start sampler");
    std::thread::sleep(Duration::from_millis(2600));
    let auto = frames.load(Ordering::Relaxed);
    sampler.capture_now();
    std::thread::sleep(Duration::from_millis(800));
    sampler.stop();
    let total = frames.load(Ordering::Relaxed);
    let st = statuses.lock().clone();
    println!(
        "automatic frames: {auto}, after manual capture: {total}, statuses: {}",
        st.iter()
            .map(|s| match s {
                ScreenStatus::Capturing => "capturing",
                ScreenStatus::Unchanged => "unchanged",
                ScreenStatus::SkippedPrivacy(_) => "skipped",
                ScreenStatus::PermissionDenied(_) => "denied",
                ScreenStatus::Error(_) => "error",
            })
            .collect::<Vec<_>>()
            .join(",")
    );
    assert!(auto >= 1, "the first capture must always be emitted");
    // ~5 samples at 2 fps: a mostly static screen must not emit them all.
    assert!(auto <= 5);
    assert_eq!(total, auto + 1, "manual capture bypasses change detection");
}

#[cfg(windows)]
#[test]
#[ignore = "needs a display and an OCR language"]
fn windows_ocr_reads_the_screen() {
    let Some(ocr) = openmeet_platform::ocr::system_ocr() else {
        println!("Windows OCR unavailable (no OCR language installed) — skipping");
        return;
    };
    let frame = screen::capture_once(&primary_display(), &[], 1600).expect("capture");
    let started = Instant::now();
    let out = ocr.recognize_png(&frame.png).expect("ocr");
    println!(
        "OCR: {} lines, {} chars in {} ms",
        out.lines.len(),
        out.text.chars().count(),
        started.elapsed().as_millis()
    );
    assert!(
        !out.lines.is_empty(),
        "a desktop screenshot should contain some text"
    );
}

#[cfg(windows)]
#[test]
#[ignore = "needs a desktop session"]
fn capture_exclusion_on_a_real_window() {
    use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
    };
    use windows::core::w;

    let hwnd = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("STATIC"),
            w!("OpenMeet capture exclusion test"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            100,
            100,
            320,
            200,
            None,
            None,
            None,
            None,
        )
    }
    .expect("create window");
    let handle =
        Win32WindowHandle::new(std::num::NonZeroIsize::new(hwnd.0 as isize).expect("hwnd"));
    let privacy = openmeet_platform::privacy::for_window(RawWindowHandle::Win32(handle));
    let (supported, mechanism, _) = openmeet_platform::privacy::platform_support();
    println!("supported: {supported}, mechanism: {mechanism}");
    if privacy.is_supported() {
        privacy.set_excluded_from_capture(true).expect("exclude");
        assert_eq!(privacy.is_excluded(), Some(true));
        privacy.set_excluded_from_capture(false).expect("include");
        assert_eq!(privacy.is_excluded(), Some(false));
    }
    unsafe {
        let _ = DestroyWindow(hwnd);
    }
}

#[test]
#[ignore = "needs a desktop session"]
fn meeting_app_detection_and_focus_queries() {
    let started = Instant::now();
    let apps = openmeet_platform::apps::detect_meeting_apps();
    println!(
        "meeting apps detected: {} in {} ms",
        apps.len(),
        started.elapsed().as_millis()
    );
    println!(
        "foreground window known: {}",
        openmeet_platform::focus::foreground_window_id().is_some()
    );
}

#[test]
#[ignore = "samples the current process"]
fn process_stats() {
    let mut stats = openmeet_platform::system::ProcessStats::new();
    let _ = stats.sample();
    std::thread::sleep(Duration::from_millis(300));
    let (mem, cpu) = stats.sample();
    println!("memory {mem:.1} MB, cpu {cpu:.1}%");
    assert!(mem > 1.0);
    assert!(cpu >= 0.0);
}
