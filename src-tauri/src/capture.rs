//! Owns the OS capture sessions (microphone, system audio, screen) and feeds
//! them into the engine. Capture only runs while a meeting is active (or
//! during an explicit device test); pausing or an emergency stop releases the
//! devices immediately (spec §39, §40, §60).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use openmeet_core::CoreError;
use openmeet_core::events::{EventSink, MeetingEvent, NoticeLevel};
use openmeet_core::ipc::ScreenSourceSpec;
use openmeet_core::settings::Settings;
use openmeet_core::types::{AudioSourceKind, Component, ComponentStatus, PrivacyRule};
use openmeet_engine::Engine;
use openmeet_platform::audio::{AudioCapture, AudioCaptureConfig};
use openmeet_platform::screen::{SamplerConfig, ScreenSampler, ScreenStatus};
use parking_lot::Mutex;

use crate::screenshots::ScreenshotStore;

/// Maximum duration of a device level test.
const AUDIO_TEST_SECONDS: u64 = 15;

/// What the running meeting captures (restored after pause / resume).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeetingSources {
    pub microphone: bool,
    pub system_audio: bool,
    pub screen: Option<ScreenSourceSpec>,
}

#[derive(Default)]
struct Inner {
    microphone: Option<AudioCapture>,
    system: Option<AudioCapture>,
    screen: Option<ScreenSampler>,
    test: Option<AudioCapture>,
    sources: Option<MeetingSources>,
}

pub struct CaptureManager {
    inner: Arc<Mutex<Inner>>,
    sink: Arc<dyn EventSink>,
    test_generation: Arc<AtomicU64>,
}

fn component_for(source: AudioSourceKind) -> Component {
    match source {
        AudioSourceKind::Microphone => Component::Microphone,
        AudioSourceKind::System => Component::SystemAudio,
    }
}

fn source_label(source: AudioSourceKind) -> &'static str {
    match source {
        AudioSourceKind::Microphone => "Microphone",
        AudioSourceKind::System => "System audio",
    }
}

impl CaptureManager {
    pub fn new(sink: Arc<dyn EventSink>) -> Self {
        CaptureManager {
            inner: Arc::new(Mutex::new(Inner::default())),
            sink,
            test_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn sources(&self) -> Option<MeetingSources> {
        self.inner.lock().sources.clone()
    }

    pub fn set_sources(&self, sources: Option<MeetingSources>) {
        self.inner.lock().sources = sources;
    }

    pub fn is_capturing_audio(&self) -> bool {
        let inner = self.inner.lock();
        inner.microphone.is_some() || inner.system.is_some()
    }

    fn notice(&self, level: NoticeLevel, message: String, component: Component) {
        self.sink.emit(MeetingEvent::Notice {
            level,
            message,
            component: Some(component),
        });
    }

    /// Starts audio capture for one source, feeding the engine.
    pub fn start_audio(
        &self,
        engine: &Engine,
        source: AudioSourceKind,
        device: Option<String>,
    ) -> Result<(), CoreError> {
        self.stop_test();
        let component = component_for(source);
        let feed = engine.clone();
        let err_engine = engine.clone();
        let err_sink = self.sink.clone();
        let last_error = Arc::new(Mutex::new(None::<Instant>));
        let config = AudioCaptureConfig { source, device };
        let capture = AudioCapture::start(
            config,
            Box::new(move |frame| feed.push_audio(frame)),
            Box::new(move |error| {
                let message = format!(
                    "{} unavailable: {error}. Select another audio source.",
                    source_label(source)
                );
                err_engine.set_component_status(
                    component,
                    ComponentStatus::Error,
                    Some(message.clone()),
                );
                let mut last = last_error.lock();
                if last.is_none_or(|t| t.elapsed() > Duration::from_secs(30)) {
                    *last = Some(Instant::now());
                    err_sink.emit(MeetingEvent::Notice {
                        level: NoticeLevel::Error,
                        message,
                        component: Some(component),
                    });
                }
            }),
        );
        match capture {
            Ok(c) => {
                engine.set_component_status(
                    component,
                    ComponentStatus::Active,
                    Some(c.device_name().to_string()),
                );
                let mut inner = self.inner.lock();
                match source {
                    AudioSourceKind::Microphone => inner.microphone = Some(c),
                    AudioSourceKind::System => inner.system = Some(c),
                }
                Ok(())
            }
            Err(e) => {
                let message = format!(
                    "{} unavailable: {e}. Select another audio source.",
                    source_label(source)
                );
                engine.set_component_status(
                    component,
                    ComponentStatus::Error,
                    Some(message.clone()),
                );
                self.notice(NoticeLevel::Error, message.clone(), component);
                Err(CoreError::from(e))
            }
        }
    }

    pub fn stop_audio(&self, engine: &Engine) {
        let (mic, sys) = {
            let mut inner = self.inner.lock();
            (inner.microphone.take(), inner.system.take())
        };
        if let Some(c) = mic {
            c.stop();
            engine.set_component_status(Component::Microphone, ComponentStatus::Inactive, None);
        }
        if let Some(c) = sys {
            c.stop();
            engine.set_component_status(Component::SystemAudio, ComponentStatus::Inactive, None);
        }
    }

    /// Starts the screen sampler for the explicitly selected source.
    pub fn start_screen(
        &self,
        engine: &Engine,
        spec: ScreenSourceSpec,
        settings: &Settings,
        rules: Vec<PrivacyRule>,
        keep_screenshots: Option<Arc<ScreenshotStore>>,
    ) -> Result<(), CoreError> {
        self.stop_screen(engine);
        let feed = engine.clone();
        let status_engine = engine.clone();
        let status_sink = self.sink.clone();
        let label = spec.label.clone();
        let config = SamplerConfig {
            spec,
            frequency: settings.screen.frequency,
            change_threshold: settings.screen.change_threshold,
            rules,
            max_width: 1280,
            own_process_id: std::process::id(),
        };
        let last_status = Arc::new(Mutex::new(None::<ScreenStatus>));
        let sampler = ScreenSampler::start(
            config,
            Box::new(move |frame| {
                if let Some(store) = &keep_screenshots {
                    keep_screenshot(&feed, store, &frame);
                }
                feed.push_screen_frame(frame)
            }),
            Box::new(move |status| {
                let mut last = last_status.lock();
                if last.as_ref() == Some(&status) {
                    return;
                }
                *last = Some(status.clone());
                match &status {
                    ScreenStatus::Capturing | ScreenStatus::Unchanged => status_engine
                        .set_component_status(Component::Screen, ComponentStatus::Active, None),
                    ScreenStatus::SkippedPrivacy(reason) => status_engine.set_component_status(
                        Component::Screen,
                        ComponentStatus::Paused,
                        Some(format!("Skipped by a privacy rule: {reason}")),
                    ),
                    ScreenStatus::PermissionDenied(msg) => {
                        status_engine.set_component_status(
                            Component::Screen,
                            ComponentStatus::Error,
                            Some(msg.clone()),
                        );
                        status_sink.emit(MeetingEvent::Notice {
                            level: NoticeLevel::Error,
                            message: msg.clone(),
                            component: Some(Component::Screen),
                        });
                    }
                    ScreenStatus::Error(msg) => status_engine.set_component_status(
                        Component::Screen,
                        ComponentStatus::Error,
                        Some(msg.clone()),
                    ),
                }
            }),
        )
        .map_err(|e| {
            engine.set_component_status(
                Component::Screen,
                ComponentStatus::Error,
                Some(e.to_string()),
            );
            CoreError::from(e)
        })?;
        engine.set_component_status(Component::Screen, ComponentStatus::Active, Some(label));
        self.inner.lock().screen = Some(sampler);
        Ok(())
    }

    pub fn stop_screen(&self, engine: &Engine) {
        let sampler = self.inner.lock().screen.take();
        if let Some(s) = sampler {
            s.stop();
            engine.set_component_status(Component::Screen, ComponentStatus::Inactive, None);
        }
    }

    pub fn update_rules(&self, rules: Vec<PrivacyRule>) {
        if let Some(s) = self.inner.lock().screen.as_ref() {
            s.update_rules(rules);
        }
    }

    /// Manual capture through the running sampler. Returns false when no
    /// sampler is running.
    pub fn capture_now(&self) -> bool {
        match self.inner.lock().screen.as_ref() {
            Some(s) => {
                s.capture_now();
                true
            }
            None => false,
        }
    }

    /// Stops every capture session (meeting end, pause, emergency stop).
    pub fn stop_all(&self, engine: &Engine) {
        self.stop_audio(engine);
        self.stop_screen(engine);
        self.stop_test();
    }

    // ------------------------------------------------------------------
    // Device test (level meter during setup)
    // ------------------------------------------------------------------

    pub fn start_test(
        &self,
        source: AudioSourceKind,
        device: Option<String>,
    ) -> Result<(), CoreError> {
        self.stop_test();
        {
            let inner = self.inner.lock();
            if inner.microphone.is_some() || inner.system.is_some() {
                return Err(CoreError::InvalidState(
                    "Audio is in use by the running meeting; the level meter is shown there."
                        .into(),
                ));
            }
        }
        let sink = self.sink.clone();
        let mut last_emit = Instant::now() - Duration::from_secs(1);
        let mut smoothed = 0.0f32;
        let capture = AudioCapture::start(
            AudioCaptureConfig { source, device },
            Box::new(move |frame| {
                if frame.samples.is_empty() {
                    return;
                }
                let rms = (frame.samples.iter().map(|s| s * s).sum::<f32>()
                    / frame.samples.len() as f32)
                    .sqrt();
                // Map -60..0 dBFS to 0..1.
                let db = 20.0 * rms.max(1e-6).log10();
                let level = ((db + 60.0) / 60.0).clamp(0.0, 1.0);
                smoothed = smoothed.max(level) * 0.85 + level * 0.15;
                if last_emit.elapsed() >= Duration::from_millis(100) {
                    last_emit = Instant::now();
                    sink.emit(MeetingEvent::AudioLevel {
                        source,
                        level: smoothed,
                    });
                    smoothed *= 0.6;
                }
            }),
            Box::new(|e| tracing::warn!(error = %e, "audio test capture error")),
        )
        .map_err(CoreError::from)?;
        self.inner.lock().test = Some(capture);
        let generation = self.test_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let current = self.test_generation.clone();
        let inner = self.inner.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_secs(AUDIO_TEST_SECONDS)).await;
            if current.load(Ordering::SeqCst) == generation {
                if let Some(c) = inner.lock().test.take() {
                    c.stop();
                }
            }
        });
        Ok(())
    }

    pub fn stop_test(&self) {
        self.test_generation.fetch_add(1, Ordering::SeqCst);
        if let Some(c) = self.inner.lock().test.take() {
            c.stop();
        }
    }
}

/// Saves an analyzed frame when the user opted into keeping screenshots and
/// the current meeting is persisted (never for "do not retain" meetings).
pub fn keep_screenshot(
    engine: &Engine,
    store: &ScreenshotStore,
    frame: &openmeet_core::types::ScreenFrame,
) {
    if engine.unpersisted_meeting().is_some() {
        return;
    }
    let Some(meeting_id) = engine.current_meeting_id() else {
        return;
    };
    if let Err(e) = store.save(&meeting_id, frame) {
        tracing::warn!(error = %e, "could not keep screenshot");
    }
}
