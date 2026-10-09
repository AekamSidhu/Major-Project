//! Microphone and system-audio capture.
//!
//! Capture uses cpal on a dedicated thread (cpal streams are `!Send` on some
//! hosts). Device buffers in any PCM sample format are converted to mono f32,
//! resampled to 16 kHz and delivered as [`AudioFrame`]s of 40 ms with
//! wall-clock `captured_at_ms` timestamps.
//!
//! System audio:
//! * **Windows** – WASAPI loopback: an *input* stream built on the *output*
//!   (render) endpoint, which cpal 0.18 turns into a loopback capture.
//!   WASAPI delivers no packets while nothing is playing; those periods are
//!   filled with silence so the timeline (and voice activity detection) keeps
//!   moving.
//! * **macOS 14.6+** – cpal's Core Audio process tap on the output device
//!   (the user is asked for the System Audio Recording permission). Older
//!   versions need a virtual input device such as BlackHole, which can be
//!   selected as the system-audio device.
//! * **Linux** – PulseAudio / PipeWire (pipewire-pulse) "Monitor of …"
//!   sources.
//!
//! Device disconnects and stream errors are reported through `on_error` and
//! never panic; the capture thread keeps trying to reopen the device (the
//! current default device when none was selected) and frames resume when it
//! succeeds.

mod convert;

use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ErrorKind, SampleFormat, SupportedStreamConfig};
use openmeet_core::audio::AudioFrame;
use openmeet_core::ipc::{AudioDeviceInfo, AudioDevices, SystemAudioSupport};
use openmeet_core::types::AudioSourceKind;

use self::convert::FrameAssembler;
use crate::PlatformError;
use crate::error::guard_panics;

/// How long `start` waits for the device to open.
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);
/// Backend initialization timeout passed to cpal.
const BACKEND_TIMEOUT: Duration = Duration::from_secs(5);
/// Wake-up interval of the capture thread.
const TICK: Duration = Duration::from_millis(50);
/// System audio: no packets for this long means "nothing is playing".
const IDLE_FILL_AFTER: Duration = Duration::from_millis(200);
/// Silence is filled up to `now - IDLE_FILL_LAG_MS` so late packets still fit.
const IDLE_FILL_LAG_MS: f64 = 100.0;
/// Microphone: no packets for this long is reported as a stalled device.
const MIC_STALL_AFTER: Duration = Duration::from_secs(5);
/// Delay between attempts to reopen a failed device.
const RETRY_INTERVAL: Duration = Duration::from_secs(2);
/// A stream that fails sooner than this after opening is not reopened
/// immediately (avoids tight failure loops).
const MIN_STREAM_LIFETIME: Duration = Duration::from_secs(1);
/// Larger capture timestamps latencies are treated as bogus.
const MAX_PLAUSIBLE_LATENCY_MS: f64 = 2_000.0;

/// Lists capture devices: microphones (`inputs`) and devices usable for
/// system-audio capture (`outputs`: loopback-capable output devices on
/// Windows / macOS 14.6+, monitor sources on Linux, virtual input devices
/// such as BlackHole on macOS).
///
/// Device ids are cpal device ids (`"<host>:<device id>"`); [`AudioCapture`]
/// also accepts a device name.
pub fn list_devices() -> Result<AudioDevices, PlatformError> {
    guard_panics("listing audio devices", || {
        let host = cpal::default_host();
        let system_audio = system_audio_support();
        let default_in = host.default_input_device().and_then(|d| device_id(&d));
        let default_out = host.default_output_device().and_then(|d| device_id(&d));

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let input_devices = host
            .input_devices()
            .map_err(|e| map_cpal_error(&e, AudioSourceKind::Microphone))?;
        for device in input_devices {
            let Some(info) = describe(&device, default_in.as_deref(), true) else {
                continue;
            };
            if is_monitor_source(&info.id) {
                // Linux "Monitor of …" sources capture an output device.
                if system_audio.supported {
                    let default_monitor = default_out.as_deref().map(|o| format!("{o}.monitor"));
                    outputs.push(AudioDeviceInfo {
                        is_default: default_monitor.as_deref() == Some(info.id.as_str()),
                        ..info
                    });
                }
                continue;
            }
            if cfg!(target_os = "macos") && is_virtual(&device) {
                // BlackHole & co. route system audio into an input device.
                outputs.push(AudioDeviceInfo {
                    is_default: false,
                    ..info.clone()
                });
            }
            inputs.push(info);
        }

        if system_audio.supported && loopback_on_output_devices() {
            let output_devices = host
                .output_devices()
                .map_err(|e| map_cpal_error(&e, AudioSourceKind::System))?;
            for device in output_devices {
                if let Some(info) = describe(&device, default_out.as_deref(), false) {
                    outputs.push(info);
                }
            }
        }
        inputs.sort_by_key(|d| !d.is_default);
        outputs.sort_by_key(|d| !d.is_default);
        tracing::debug!(
            inputs = inputs.len(),
            outputs = outputs.len(),
            "audio devices listed"
        );
        Ok(AudioDevices {
            inputs,
            outputs,
            system_audio,
        })
    })
}

/// Whether (and how) system audio can be captured on this machine.
pub fn system_audio_support() -> SystemAudioSupport {
    #[cfg(windows)]
    {
        SystemAudioSupport {
            supported: true,
            mechanism: "WASAPI loopback".to_string(),
            note: "Captures what plays on the selected output device (default: the system \
                   default output). Audio played in exclusive mode is not captured."
                .to_string(),
        }
    }
    #[cfg(target_os = "macos")]
    {
        if macos_supports_process_tap() {
            SystemAudioSupport {
                supported: true,
                mechanism: "Core Audio process tap".to_string(),
                note: "Captures what plays on the selected output device. macOS asks for the \
                       System Audio Recording permission on first use."
                    .to_string(),
            }
        } else {
            SystemAudioSupport {
                supported: false,
                mechanism: "Virtual audio device".to_string(),
                note: "System audio capture requires a virtual audio device such as BlackHole \
                       on macOS (built-in capture needs macOS 14.6 or later)."
                    .to_string(),
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        let host = guard_panics("opening the audio host", || Ok(cpal::default_host()));
        let pulse = host.is_ok_and(|h| h.id().name().eq_ignore_ascii_case("pulseaudio"));
        if pulse {
            SystemAudioSupport {
                supported: true,
                mechanism: "PulseAudio monitor source".to_string(),
                note: "Captures the \"Monitor of …\" source of an output device. Works with \
                       PulseAudio and PipeWire (pipewire-pulse)."
                    .to_string(),
            }
        } else {
            SystemAudioSupport {
                supported: false,
                mechanism: "PulseAudio monitor source".to_string(),
                note: "System audio capture on Linux requires a PulseAudio or PipeWire \
                       (pipewire-pulse) sound server; plain ALSA cannot capture it."
                    .to_string(),
            }
        }
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        SystemAudioSupport {
            supported: false,
            mechanism: "none".to_string(),
            note: "System audio capture is not supported on this operating system.".to_string(),
        }
    }
}

/// What to capture.
#[derive(Debug, Clone)]
pub struct AudioCaptureConfig {
    /// Microphone or system audio.
    pub source: AudioSourceKind,
    /// Device id (from [`list_devices`]) or device name; `None` = the default
    /// device (followed when the default changes).
    pub device: Option<String>,
}

/// A running capture. Capture runs on a dedicated thread (cpal streams are
/// `!Send` on some hosts). Frames are converted to 16 kHz mono f32 and
/// delivered in 40 ms chunks with `captured_at_ms` timestamps. Dropping it
/// stops the capture.
pub struct AudioCapture {
    device_name: String,
    source: AudioSourceKind,
    tx: mpsc::Sender<Msg>,
    thread: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for AudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCapture")
            .field("device_name", &self.device_name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl AudioCapture {
    /// Opens the device and starts capturing. Returns once the stream is
    /// running (or failed to open). `on_frame` and `on_error` are called on
    /// the capture thread; they should not block.
    pub fn start(
        config: AudioCaptureConfig,
        on_frame: Box<dyn FnMut(AudioFrame) + Send>,
        on_error: Box<dyn FnMut(PlatformError) + Send>,
    ) -> Result<AudioCapture, PlatformError> {
        let source = config.source;
        let (tx, rx) = mpsc::channel::<Msg>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<String, PlatformError>>(1);
        let thread_tx = tx.clone();
        let thread = std::thread::Builder::new()
            .name(format!("openmeet-audio-{}", source_slug(source)))
            .spawn(move || {
                CaptureWorker::run(config, thread_tx, rx, ready_tx, on_frame, on_error);
            })
            .map_err(|e| PlatformError::Other(format!("could not start the audio thread: {e}")))?;

        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(device_name)) => {
                tracing::info!(source = source_slug(source), "audio capture started");
                Ok(AudioCapture {
                    device_name,
                    source,
                    tx,
                    thread: Some(thread),
                })
            }
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(RecvTimeoutError::Timeout) => {
                // The worker exits on its own once the backend returns.
                let _ = tx.send(Msg::Stop);
                Err(PlatformError::DeviceUnavailable(format!(
                    "The {} did not respond",
                    source_noun(source)
                )))
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = thread.join();
                Err(PlatformError::Other(
                    "the audio thread stopped unexpectedly".to_string(),
                ))
            }
        }
    }

    /// Stops the capture and waits for the capture thread to finish.
    pub fn stop(self) {
        drop(self);
    }

    /// Name of the device being captured (as opened at start).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The captured source.
    pub fn source(&self) -> AudioSourceKind {
        self.source
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Stop);
        if let Some(thread) = self.thread.take() {
            // Dropping from inside a callback must not join our own thread.
            if thread.thread().id() != std::thread::current().id() {
                let _ = thread.join();
            }
        }
        tracing::info!(source = source_slug(self.source), "audio capture stopped");
    }
}

/// Messages to the capture thread (from cpal callbacks and the handle).
enum Msg {
    Audio {
        generation: u64,
        samples: Vec<f32>,
        start_ms: f64,
    },
    StreamError {
        generation: u64,
        error: cpal::Error,
    },
    Unconvertible {
        generation: u64,
        format: SampleFormat,
    },
    Stop,
}

/// An open cpal stream plus what the worker needs to know about it.
struct OpenStream {
    /// Kept alive, never read: dropping it stops the callbacks. Declared
    /// first so it is dropped first.
    _stream: cpal::Stream,
    name: String,
    sample_rate: u32,
    opened_at: Instant,
}

struct CaptureWorker {
    config: AudioCaptureConfig,
    tx: mpsc::Sender<Msg>,
    on_frame: Box<dyn FnMut(AudioFrame) + Send>,
    on_error: Box<dyn FnMut(PlatformError) + Send>,
    assembler: FrameAssembler,
    generation: u64,
    current: Option<OpenStream>,
    last_data: Instant,
    retry_at: Option<Instant>,
    /// An error was reported and the device has not recovered since.
    error_reported: bool,
    stall_reported: bool,
}

impl CaptureWorker {
    fn run(
        config: AudioCaptureConfig,
        tx: mpsc::Sender<Msg>,
        rx: mpsc::Receiver<Msg>,
        ready: mpsc::SyncSender<Result<String, PlatformError>>,
        on_frame: Box<dyn FnMut(AudioFrame) + Send>,
        on_error: Box<dyn FnMut(PlatformError) + Send>,
    ) {
        let opened = open_stream(&config, &tx, 1);
        let stream = match opened {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(source = source_slug(config.source), error = %e, "audio device could not be opened");
                let _ = ready.send(Err(e));
                return;
            }
        };
        if ready.send(Ok(stream.name.clone())).is_err() {
            // `start` gave up waiting; nobody is listening.
            return;
        }
        let mut worker = CaptureWorker {
            assembler: FrameAssembler::new(config.source, stream.sample_rate),
            config,
            tx,
            on_frame,
            on_error,
            generation: 1,
            current: Some(stream),
            last_data: Instant::now(),
            retry_at: None,
            error_reported: false,
            stall_reported: false,
        };
        worker.event_loop(&rx);
        // Stop callbacks before flushing the tail.
        worker.current = None;
        let tail = worker.assembler.flush();
        worker.emit(tail);
    }

    fn event_loop(&mut self, rx: &mpsc::Receiver<Msg>) {
        loop {
            match rx.recv_timeout(TICK) {
                Ok(Msg::Stop) | Err(RecvTimeoutError::Disconnected) => return,
                Ok(Msg::Audio {
                    generation,
                    samples,
                    start_ms,
                }) if generation == self.generation => {
                    self.last_data = Instant::now();
                    if self.stall_reported || self.error_reported {
                        tracing::info!(
                            source = source_slug(self.config.source),
                            "audio capture recovered"
                        );
                        self.stall_reported = false;
                        self.error_reported = false;
                    }
                    let frames = self.assembler.push(&samples, start_ms);
                    self.emit(frames);
                }
                Ok(Msg::StreamError { generation, error }) if generation == self.generation => {
                    self.on_stream_error(error);
                }
                Ok(Msg::Unconvertible { generation, format }) if generation == self.generation => {
                    tracing::warn!(%format, "audio device delivers a non-PCM sample format");
                    self.current = None;
                    self.retry_at = None;
                    self.report(PlatformError::Unsupported(format!(
                        "The {} delivers audio in a format OpenMeet cannot convert ({format})",
                        source_noun(self.config.source)
                    )));
                }
                // Messages from a stream that was already replaced.
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.on_tick();
        }
    }

    fn on_tick(&mut self) {
        let now = Instant::now();
        if self.current.is_some() {
            let idle = now.duration_since(self.last_data);
            match self.config.source {
                AudioSourceKind::System if idle >= IDLE_FILL_AFTER => {
                    // Nothing is playing: loopback delivers no packets.
                    let frames = self
                        .assembler
                        .fill_silence_until(wall_ms() - IDLE_FILL_LAG_MS);
                    self.emit(frames);
                }
                AudioSourceKind::Microphone if idle >= MIC_STALL_AFTER && !self.stall_reported => {
                    self.stall_reported = true;
                    tracing::warn!("microphone stopped delivering audio");
                    let msg = "The microphone stopped delivering audio".to_string();
                    self.call_on_error(PlatformError::DeviceUnavailable(msg));
                }
                _ => {}
            }
        } else if self.retry_at.is_some_and(|at| now >= at) {
            match self.reopen() {
                Ok(()) => tracing::info!(
                    source = source_slug(self.config.source),
                    "audio device reopened"
                ),
                Err(e) => {
                    tracing::debug!(error = %e, "audio device still unavailable");
                    self.retry_at = Some(now + RETRY_INTERVAL);
                }
            }
        }
    }

    fn on_stream_error(&mut self, error: cpal::Error) {
        match error.kind() {
            ErrorKind::Xrun => {
                tracing::debug!("audio buffer overrun/underrun");
                return;
            }
            ErrorKind::RealtimeDenied => {
                tracing::debug!("real-time audio thread priority was refused");
                return;
            }
            ErrorKind::DeviceChanged => {
                tracing::info!(
                    source = source_slug(self.config.source),
                    "audio route changed; stream rerouted"
                );
                return;
            }
            _ => {}
        }
        tracing::warn!(source = source_slug(self.config.source), kind = ?error.kind(), "audio stream failed");
        let lived = self.current.as_ref().map(|s| s.opened_at.elapsed());
        self.current = None;
        // Default device changed / device re-plugged: reopening usually
        // works right away and is not worth an error in the UI.
        if lived.is_some_and(|l| l >= MIN_STREAM_LIFETIME) {
            match self.reopen() {
                Ok(()) => {
                    tracing::info!(
                        source = source_slug(self.config.source),
                        "audio stream reopened"
                    );
                    return;
                }
                Err(e) => tracing::debug!(error = %e, "immediate reopen failed"),
            }
        }
        self.retry_at = Some(Instant::now() + RETRY_INTERVAL);
        self.report(map_cpal_error(&error, self.config.source));
    }

    fn reopen(&mut self) -> Result<(), PlatformError> {
        let generation = self.generation + 1;
        let stream = guard_panics("reopening the audio device", || {
            open_stream(&self.config, &self.tx, generation)
        })?;
        self.generation = generation;
        self.assembler.set_input_rate(stream.sample_rate);
        self.current = Some(stream);
        self.retry_at = None;
        self.last_data = Instant::now();
        self.stall_reported = false;
        Ok(())
    }

    /// Reports an error once until the capture recovers.
    fn report(&mut self, error: PlatformError) {
        if !self.error_reported {
            self.error_reported = true;
            self.call_on_error(error);
        }
    }

    fn call_on_error(&mut self, error: PlatformError) {
        let on_error = &mut self.on_error;
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_error(error))).is_err() {
            tracing::error!("audio on_error callback panicked");
        }
    }

    fn emit(&mut self, frames: Vec<AudioFrame>) {
        for frame in frames {
            let on_frame = &mut self.on_frame;
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_frame(frame))).is_err() {
                tracing::error!("audio on_frame callback panicked");
            }
        }
    }
}

/// Resolves the configured device and starts a stream whose callbacks send
/// [`Msg`]s tagged with `generation`.
fn open_stream(
    config: &AudioCaptureConfig,
    tx: &mpsc::Sender<Msg>,
    generation: u64,
) -> Result<OpenStream, PlatformError> {
    guard_panics("opening the audio device", || {
        let source = config.source;
        let host = cpal::default_host();
        let (device, name) = resolve_device(&host, source, config.device.as_deref())?;
        let supported = stream_config(&device, source)?;
        let stream_config = supported.config();
        let channels = stream_config.channels.max(1);
        let sample_rate = stream_config.sample_rate.max(1);

        let data_tx = tx.clone();
        let mut warned = false;
        let data_callback = move |data: &cpal::Data, info: &cpal::InputCallbackInfo| {
            let arrival = wall_ms();
            match convert::data_to_mono(data, channels) {
                Some(samples) if !samples.is_empty() => {
                    let ts = info.timestamp();
                    let latency_ms = ts.callback.duration_since(ts.capture).as_secs_f64() * 1000.0;
                    let start_ms = if latency_ms > 0.0 && latency_ms < MAX_PLAUSIBLE_LATENCY_MS {
                        arrival - latency_ms
                    } else {
                        arrival - samples.len() as f64 * 1000.0 / f64::from(sample_rate)
                    };
                    let _ = data_tx.send(Msg::Audio {
                        generation,
                        samples,
                        start_ms,
                    });
                }
                Some(_) => {}
                None if !warned => {
                    warned = true;
                    let _ = data_tx.send(Msg::Unconvertible {
                        generation,
                        format: data.sample_format(),
                    });
                }
                None => {}
            }
        };
        let error_tx = tx.clone();
        let error_callback = move |error: cpal::Error| {
            let _ = error_tx.send(Msg::StreamError { generation, error });
        };
        let stream = device
            .build_input_stream_raw(
                stream_config,
                supported.sample_format(),
                data_callback,
                error_callback,
                Some(BACKEND_TIMEOUT),
            )
            .map_err(|e| map_cpal_error(&e, source))?;
        stream.play().map_err(|e| map_cpal_error(&e, source))?;
        tracing::debug!(
            source = source_slug(source),
            channels,
            sample_rate,
            format = %supported.sample_format(),
            "audio stream opened"
        );
        Ok(OpenStream {
            _stream: stream,
            name,
            sample_rate,
            opened_at: Instant::now(),
        })
    })
}

/// Picks the device for `source`. Returns the device and its display name.
fn resolve_device(
    host: &cpal::Host,
    source: AudioSourceKind,
    wanted: Option<&str>,
) -> Result<(cpal::Device, String), PlatformError> {
    let wanted = wanted.map(str::trim).filter(|w| !w.is_empty());
    match source {
        AudioSourceKind::Microphone => {
            let device = match wanted {
                Some(w) => find_device(input_list(host)?, w).ok_or_else(|| {
                    PlatformError::DeviceNotFound(
                        "The selected microphone is not connected".to_string(),
                    )
                })?,
                None => host.default_input_device().ok_or_else(|| {
                    PlatformError::DeviceNotFound("No microphone was found".to_string())
                })?,
            };
            let name = device_name(&device);
            Ok((device, name))
        }
        AudioSourceKind::System => {
            let support = system_audio_support();
            // An explicitly selected input device (BlackHole, Stereo Mix, a
            // monitor source) works wherever inputs work.
            if let Some(w) = wanted {
                if loopback_on_output_devices() && support.supported {
                    if let Some(d) = find_device(output_list(host)?, w) {
                        let name = device_name(&d);
                        return Ok((d, name));
                    }
                }
                if let Some(d) = find_device(input_list(host)?, w) {
                    let name = device_name(&d);
                    return Ok((d, name));
                }
                // A sink id selects its monitor source (Linux).
                let monitor = format!("{w}.monitor");
                if let Some(d) = find_device(input_list(host)?, &monitor) {
                    let name = device_name(&d);
                    return Ok((d, name));
                }
                if !support.supported {
                    return Err(PlatformError::Unsupported(support.note));
                }
                return Err(PlatformError::DeviceNotFound(
                    "The selected system audio device is not connected".to_string(),
                ));
            }
            if !support.supported {
                return Err(PlatformError::Unsupported(support.note));
            }
            if loopback_on_output_devices() {
                let device = host.default_output_device().ok_or_else(|| {
                    PlatformError::DeviceNotFound("No audio output device was found".to_string())
                })?;
                let name = device_name(&device);
                return Ok((device, name));
            }
            // Linux: the monitor of the default sink, else any monitor.
            let inputs = input_list(host)?;
            let default_monitor = host
                .default_output_device()
                .and_then(|d| device_id(&d))
                .map(|id| format!("{id}.monitor"));
            let mut fallback = None;
            for d in inputs {
                let Some(id) = device_id(&d) else { continue };
                if default_monitor.as_deref() == Some(id.as_str()) {
                    let name = device_name(&d);
                    return Ok((d, name));
                }
                if fallback.is_none() && is_monitor_source(&id) {
                    fallback = Some(d);
                }
            }
            let d = fallback.ok_or_else(|| {
                PlatformError::DeviceNotFound("No output monitor source was found".to_string())
            })?;
            let name = device_name(&d);
            Ok((d, name))
        }
    }
}

fn input_list(host: &cpal::Host) -> Result<Vec<cpal::Device>, PlatformError> {
    Ok(host
        .input_devices()
        .map_err(|e| map_cpal_error(&e, AudioSourceKind::Microphone))?
        .collect())
}

fn output_list(host: &cpal::Host) -> Result<Vec<cpal::Device>, PlatformError> {
    Ok(host
        .output_devices()
        .map_err(|e| map_cpal_error(&e, AudioSourceKind::System))?
        .collect())
}

/// Finds a device by cpal id (`host:id` or bare id) or by name (case-insensitive).
fn find_device(devices: Vec<cpal::Device>, wanted: &str) -> Option<cpal::Device> {
    let mut by_name = None;
    for d in devices {
        if let Some(id) = device_id(&d) {
            if id_matches(&id, wanted) {
                return Some(d);
            }
        }
        if by_name.is_none() && device_name(&d).eq_ignore_ascii_case(wanted) {
            by_name = Some(d);
        }
    }
    by_name
}

fn id_matches(id: &str, wanted: &str) -> bool {
    id == wanted || id.split_once(':').is_some_and(|(_, bare)| bare == wanted)
}

/// The stream configuration: the device's default (for loopback on an
/// output device, its default output format), falling back to the first PCM
/// configuration when the default is unusable.
fn stream_config(
    device: &cpal::Device,
    source: AudioSourceKind,
) -> Result<SupportedStreamConfig, PlatformError> {
    let input = device.supports_input();
    let default = if input {
        device.default_input_config()
    } else {
        device.default_output_config()
    };
    match default {
        Ok(c) if !c.sample_format().is_dsd() => return Ok(c),
        Ok(_) => {}
        Err(e) => tracing::debug!(error = %e, "no default stream configuration"),
    }
    let ranges: Vec<cpal::SupportedStreamConfigRange> = if input {
        device
            .supported_input_configs()
            .map_err(|e| map_cpal_error(&e, source))?
            .collect()
    } else {
        device
            .supported_output_configs()
            .map_err(|e| map_cpal_error(&e, source))?
            .collect()
    };
    ranges
        .into_iter()
        .filter(|r| !r.sample_format().is_dsd())
        .max_by(|a, b| a.cmp_default_heuristics(b))
        .map(|r| {
            r.try_with_standard_sample_rate()
                .unwrap_or_else(|| r.with_max_sample_rate())
        })
        .ok_or_else(|| {
            PlatformError::Unsupported(format!(
                "The {} offers no supported audio format",
                source_noun(source)
            ))
        })
}

fn describe(
    device: &cpal::Device,
    default_id: Option<&str>,
    input: bool,
) -> Option<AudioDeviceInfo> {
    let name = device.description().ok()?.name().trim().to_string();
    let id = device_id(device).unwrap_or_else(|| name.clone());
    let config = if input {
        device.default_input_config()
    } else {
        device.default_output_config()
    };
    let (channels, sample_rate) = config
        .map(|c| (c.channels(), c.sample_rate()))
        .unwrap_or((0, 0));
    Some(AudioDeviceInfo {
        is_default: default_id == Some(id.as_str()),
        id,
        name,
        channels,
        sample_rate,
    })
}

fn device_id(device: &cpal::Device) -> Option<String> {
    device.id().ok().map(|id| id.to_string())
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().trim().to_string())
        .ok()
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "Unknown audio device".to_string())
}

/// PulseAudio monitor sources are named `<sink>.monitor`.
fn is_monitor_source(id: &str) -> bool {
    cfg!(target_os = "linux") && id.ends_with(".monitor")
}

#[cfg(target_os = "macos")]
fn is_virtual(device: &cpal::Device) -> bool {
    device
        .description()
        .is_ok_and(|d| d.device_type() == cpal::DeviceType::Virtual)
}

#[cfg(not(target_os = "macos"))]
fn is_virtual(_device: &cpal::Device) -> bool {
    false
}

/// Hosts where an input stream on an output device captures its output
/// (WASAPI loopback, Core Audio process tap).
fn loopback_on_output_devices() -> bool {
    cfg!(any(windows, target_os = "macos"))
}

/// Core Audio process taps (used by cpal for output-device capture) need
/// macOS 14.6 or later.
#[cfg(target_os = "macos")]
fn macos_supports_process_tap() -> bool {
    sysinfo::System::os_version()
        .and_then(|v| parse_version(&v))
        .is_some_and(|(major, minor)| (major, minor) >= (14, 6))
}

/// "14.6.1" → (14, 6).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn parse_version(v: &str) -> Option<(u32, u32)> {
    let mut parts = v.trim().split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next().map_or(Some(0), |m| m.trim().parse().ok())?;
    Some((major, minor))
}

/// Maps a cpal error to a user-facing platform error (cpal messages carry
/// device / OS error details, never audio content).
fn map_cpal_error(error: &cpal::Error, source: AudioSourceKind) -> PlatformError {
    let noun = source_noun(source);
    let detail = error.to_string();
    match error.kind() {
        ErrorKind::DeviceNotAvailable => PlatformError::DeviceUnavailable(format!(
            "The {noun} was disconnected or is unavailable"
        )),
        ErrorKind::DeviceBusy => {
            PlatformError::DeviceUnavailable(format!("The {noun} is in use by another application"))
        }
        ErrorKind::StreamInvalidated => PlatformError::DeviceUnavailable(format!(
            "The {noun} changed and could not be reopened"
        )),
        ErrorKind::HostUnavailable => {
            PlatformError::DeviceUnavailable("The audio service is not running".to_string())
        }
        ErrorKind::PermissionDenied => PlatformError::PermissionDenied(permission_message(source)),
        ErrorKind::UnsupportedConfig | ErrorKind::UnsupportedOperation => {
            PlatformError::Unsupported(format!("The {noun} cannot be captured: {detail}"))
        }
        _ if looks_like_access_denied(&detail) => {
            PlatformError::PermissionDenied(permission_message(source))
        }
        _ => PlatformError::Capture(format!("{noun}: {detail}")),
    }
}

/// WASAPI reports a blocked microphone as `E_ACCESSDENIED` (0x80070005),
/// which cpal classifies as a generic backend error.
fn looks_like_access_denied(detail: &str) -> bool {
    let lower = detail.to_ascii_lowercase();
    lower.contains("0x80070005")
        || lower.contains("-2147024891")
        || lower.contains("access is denied")
}

fn permission_message(source: AudioSourceKind) -> String {
    match source {
        AudioSourceKind::Microphone => {
            if cfg!(windows) {
                "Microphone access is blocked. Open Settings → Privacy & security → Microphone and allow \
                 desktop apps to access the microphone."
                    .to_string()
            } else if cfg!(target_os = "macos") {
                "Microphone permission required. Open System Settings → Privacy & Security → Microphone.".to_string()
            } else {
                "Microphone access was denied by the operating system.".to_string()
            }
        }
        AudioSourceKind::System => {
            if cfg!(target_os = "macos") {
                "System audio recording permission required. Open System Settings → Privacy & Security → \
                 Screen & System Audio Recording."
                    .to_string()
            } else {
                "System audio capture was denied by the operating system.".to_string()
            }
        }
    }
}

fn source_noun(source: AudioSourceKind) -> &'static str {
    match source {
        AudioSourceKind::Microphone => "microphone",
        AudioSourceKind::System => "system audio device",
    }
}

fn source_slug(source: AudioSourceKind) -> &'static str {
    match source {
        AudioSourceKind::Microphone => "mic",
        AudioSourceKind::System => "system",
    }
}

/// Wall-clock Unix milliseconds with sub-millisecond precision.
fn wall_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_matching() {
        assert!(id_matches(
            "wasapi:{0.0.1.00000000}.{abc}",
            "wasapi:{0.0.1.00000000}.{abc}"
        ));
        assert!(id_matches(
            "wasapi:{0.0.1.00000000}.{abc}",
            "{0.0.1.00000000}.{abc}"
        ));
        assert!(id_matches(
            "pulseaudio:alsa_output.pci.monitor",
            "alsa_output.pci.monitor"
        ));
        assert!(!id_matches("wasapi:{a}", "{b}"));
        assert!(!id_matches("wasapi:{a}", "wasapi"));
    }

    #[test]
    fn version_parsing() {
        assert_eq!(parse_version("14.6.1"), Some((14, 6)));
        assert_eq!(parse_version("15"), Some((15, 0)));
        assert_eq!(parse_version(" 13.5 "), Some((13, 5)));
        assert_eq!(parse_version("abc"), None);
        assert!(parse_version("14.6").unwrap() >= (14, 6));
        assert!(parse_version("14.5.9").unwrap() < (14, 6));
    }

    #[test]
    fn error_mapping_is_user_facing() {
        let mic = AudioSourceKind::Microphone;
        let e = map_cpal_error(&cpal::Error::new(ErrorKind::DeviceNotAvailable), mic);
        assert!(matches!(e, PlatformError::DeviceUnavailable(m) if m.contains("microphone")));
        let e = map_cpal_error(&cpal::Error::new(ErrorKind::PermissionDenied), mic);
        assert!(matches!(e, PlatformError::PermissionDenied(_)));
        let denied = cpal::Error::with_message(
            ErrorKind::BackendError,
            "Failed to build audio client: Access is denied. (os error -2147024891)",
        );
        assert!(matches!(
            map_cpal_error(&denied, mic),
            PlatformError::PermissionDenied(_)
        ));
        let e = map_cpal_error(
            &cpal::Error::new(ErrorKind::DeviceBusy),
            AudioSourceKind::System,
        );
        assert!(matches!(e, PlatformError::DeviceUnavailable(m) if m.contains("in use")));
        let e = map_cpal_error(&cpal::Error::new(ErrorKind::UnsupportedConfig), mic);
        assert!(matches!(e, PlatformError::Unsupported(_)));
        let e = map_cpal_error(&cpal::Error::with_message(ErrorKind::Other, "boom"), mic);
        assert_eq!(e, PlatformError::Capture("microphone: boom".to_string()));
    }

    #[test]
    fn monitor_sources_only_on_linux() {
        assert_eq!(
            is_monitor_source("alsa_output.pci.analog-stereo.monitor"),
            cfg!(target_os = "linux")
        );
        assert!(!is_monitor_source("alsa_input.pci.analog-stereo"));
    }

    #[test]
    fn system_audio_support_is_described() {
        let s = system_audio_support();
        assert!(!s.mechanism.is_empty());
        assert!(!s.note.is_empty());
        if cfg!(windows) {
            assert!(s.supported);
            assert_eq!(s.mechanism, "WASAPI loopback");
        }
    }

    #[test]
    fn wall_clock_is_sane() {
        let a = wall_ms();
        assert!(a > 1_704_067_200_000.0);
        assert!(wall_ms() >= a);
    }
}
