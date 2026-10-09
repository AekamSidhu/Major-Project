//! Energy-based voice activity detection and utterance segmentation.
//!
//! [`Segmenter`] turns a stream of 16 kHz mono [`AudioFrame`]s of any length
//! into utterances:
//!
//! * audio is analysed in fixed frames (`frame_ms`, 20 ms by default) and the
//!   RMS level of each frame is compared with an **adaptive noise floor**
//!   (fast adaptation downward, slow EMA of non-speech frames upward);
//! * speech starts after `min_speech_ms` of consecutive speech frames, so
//!   clicks and short bumps never open an utterance;
//! * `pre_roll_ms` of audio before the detected onset is prepended so the
//!   first phoneme is not clipped;
//! * an utterance ends after `silence_ms` of non-speech (hangover); utterances
//!   whose voiced span is shorter than `min_utterance_ms` are discarded;
//! * long monologues are cut at `max_utterance_ms`, preferring the quietest
//!   frame of the last 1.5 s as the cut point;
//! * optional interim clips are emitted every `interim_interval_ms` of speech;
//! * timestamps are sample-accurate: every frame carries the capture time of
//!   its first sample derived from `AudioFrame::captured_at_ms` plus the
//!   sample offset inside the frame.
//!
//! A stationarity guard recovers from a sudden, constant noise source (a fan
//! switching on) that would otherwise look like never-ending speech.

use std::collections::VecDeque;

use openmeet_core::audio::{AudioClip, AudioFrame, PIPELINE_SAMPLE_RATE};
use openmeet_core::ids::new_id;
use openmeet_core::types::AudioSourceKind;

/// Samples per millisecond at the pipeline rate.
const SAMPLES_PER_MS: f64 = PIPELINE_SAMPLE_RATE as f64 / 1000.0;
/// Level reported for digital silence.
const MIN_DB: f32 = -100.0;
/// Bounds of the adaptive noise floor.
const FLOOR_MIN_DB: f32 = -90.0;
const FLOOR_MAX_DB: f32 = -10.0;
/// The first frame seeds the noise floor, but never above this level (a
/// stream that starts mid-sentence must still be detected as speech).
const INITIAL_FLOOR_MAX_DB: f32 = -35.0;
/// Per-frame smoothing factor when a frame is quieter than the floor.
const FLOOR_DOWN_ALPHA: f32 = 0.3;
/// Time constant of the upward EMA over non-speech frames while idle.
const FLOOR_UP_TAU_IDLE_MS: f32 = 1_500.0;
/// Very slow upward creep while an utterance is open (safety net against a
/// floor that is stuck far below a new, permanent noise level).
const FLOOR_UP_TAU_SPEECH_MS: f32 = 30_000.0;
/// Speech continues with a threshold this much lower than the one needed to
/// start it (hysteresis keeps soft word endings inside the utterance).
const HYSTERESIS_DB: f32 = 3.0;
/// Minimum effective threshold above the noise floor.
const MIN_THRESHOLD_DB: f32 = 3.0;
/// Trailing audio kept after the last speech frame (soft word endings).
const TRAILING_KEEP_MS: f64 = 200.0;
/// Window searched for the quietest frame when a long utterance is cut.
const CUT_SEARCH_MS: f64 = 1_500.0;
/// Input timestamps that deviate more than this from the expected sample
/// clock are treated as a discontinuity.
const DISCONTINUITY_MS: f64 = 100.0;
/// Stationarity guard: minimum speech duration before it may trigger, the
/// analysis window, the maximum level standard deviation and the maximum
/// level at which a constant sound is considered noise.
const STATIONARY_MIN_SPEECH_MS: f64 = 4_000.0;
const STATIONARY_WINDOW_MS: f64 = 3_000.0;
const STATIONARY_MAX_STD_DB: f32 = 1.0;
const STATIONARY_MAX_LEVEL_DB: f32 = -30.0;
/// A whole utterance this stationary is discarded as noise.
const NOISE_UTTERANCE_MAX_STD_DB: f32 = 1.5;
/// Level meter range and smoothing.
const METER_FLOOR_DB: f32 = -60.0;
const METER_ATTACK: f32 = 0.6;
const METER_RELEASE: f32 = 0.15;

/// Voice-activity detection parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// Analysis frame length in milliseconds (20).
    pub frame_ms: u32,
    /// Consecutive speech required before an utterance starts (100–150 ms).
    pub min_speech_ms: u32,
    /// Non-speech duration that ends an utterance.
    pub silence_ms: u32,
    /// Audio kept before the detected speech onset.
    pub pre_roll_ms: u32,
    /// Utterances with a shorter voiced span are discarded.
    pub min_utterance_ms: u32,
    /// Utterances are force-cut at this length (preferring a quiet frame).
    pub max_utterance_ms: u32,
    /// `Some(n)`: emit interim clips every `n` ms of speech.
    pub interim_interval_ms: Option<u32>,
    /// Level above the adaptive noise floor needed to count as speech.
    pub threshold_db_above_noise: f32,
    /// Frames quieter than this are never speech.
    pub absolute_floor_dbfs: f32,
}

impl VadConfig {
    /// Interval between interim clips when interim results are enabled.
    pub const DEFAULT_INTERIM_INTERVAL_MS: u32 = 1_500;

    /// Builds a configuration from user settings.
    ///
    /// `sensitivity` is 0..1 (higher = more sensitive: lower thresholds and a
    /// shorter onset requirement); non-finite values fall back to 0.6.
    pub fn from_settings(sensitivity: f32, silence_ms: u32, interim_results: bool) -> Self {
        let s = if sensitivity.is_finite() {
            sensitivity.clamp(0.0, 1.0)
        } else {
            0.6
        };
        VadConfig {
            frame_ms: 20,
            min_speech_ms: (150.0 - 50.0 * s).round() as u32,
            silence_ms: silence_ms.clamp(100, 10_000),
            pre_roll_ms: 240,
            min_utterance_ms: 350,
            max_utterance_ms: 20_000,
            interim_interval_ms: interim_results.then_some(Self::DEFAULT_INTERIM_INTERVAL_MS),
            threshold_db_above_noise: 18.0 - 12.0 * s,
            absolute_floor_dbfs: -45.0 - 50.0 * s / 3.0,
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig::from_settings(0.6, 700, false)
    }
}

/// Output of [`Segmenter::push`] / [`Segmenter::flush`].
#[derive(Debug, Clone, PartialEq)]
pub enum SegmenterEvent {
    /// Speech onset (after `min_speech_ms` of consecutive speech). `at_ms` is
    /// the capture time of the first speech frame.
    SpeechStarted {
        /// Shared by the interim clips and the final clip of this utterance.
        utterance_id: String,
        /// Unix milliseconds of the first speech sample.
        at_ms: i64,
    },
    /// In-progress clip (is_final = false), only when interim is enabled.
    Interim(AudioClip),
    /// Completed utterance (is_final = true).
    Utterance(AudioClip),
    /// Speech ended but was shorter than min_utterance_ms (or was constant
    /// noise rather than speech).
    Discarded {
        /// Id announced by the matching `SpeechStarted`.
        utterance_id: String,
    },
}

/// One analysis frame.
#[derive(Debug, Clone)]
struct Frame {
    /// Unix ms (fractional) of the first sample.
    start_ms: f64,
    samples: Vec<f32>,
    db: f32,
    speech: bool,
}

impl Frame {
    fn duration_ms(&self) -> f64 {
        self.samples.len() as f64 / SAMPLES_PER_MS
    }

    fn end_ms(&self) -> f64 {
        self.start_ms + self.duration_ms()
    }
}

/// An open utterance.
#[derive(Debug)]
struct Utterance {
    id: String,
    frames: Vec<Frame>,
    /// Start of the first speech frame.
    speech_start_ms: f64,
    /// End of the last speech frame.
    last_speech_end_ms: f64,
    /// Trailing non-speech duration.
    silence_run_ms: f64,
    /// Speech-relative time covered by the last interim clip.
    last_interim_ms: f64,
    /// Levels of recent frames (stationarity guard).
    recent_db: VecDeque<(f64, f32)>,
    /// Running statistics over speech frames.
    speech_db_sum: f64,
    speech_db_sq_sum: f64,
    speech_frames: u32,
    /// The stationarity guard fired for this utterance.
    noise_suspect: bool,
}

impl Utterance {
    fn new(frames: Vec<Frame>) -> Self {
        let speech_start_ms = frames
            .iter()
            .find(|f| f.speech)
            .map(|f| f.start_ms)
            .or_else(|| frames.first().map(|f| f.start_ms))
            .unwrap_or(0.0);
        let mut u = Utterance {
            id: new_id(),
            frames: Vec::with_capacity(frames.len() + 64),
            speech_start_ms,
            last_speech_end_ms: speech_start_ms,
            silence_run_ms: 0.0,
            last_interim_ms: 0.0,
            recent_db: VecDeque::new(),
            speech_db_sum: 0.0,
            speech_db_sq_sum: 0.0,
            speech_frames: 0,
            noise_suspect: false,
        };
        for f in frames {
            u.track(&f);
            u.frames.push(f);
        }
        u
    }

    /// Updates speech bookkeeping for a frame that is being appended.
    fn track(&mut self, f: &Frame) {
        if f.speech {
            if self.speech_frames == 0 {
                self.speech_start_ms = f.start_ms;
            }
            self.last_speech_end_ms = f.end_ms();
            self.silence_run_ms = 0.0;
            self.speech_db_sum += f.db as f64;
            self.speech_db_sq_sum += (f.db as f64) * (f.db as f64);
            self.speech_frames += 1;
        } else if self.speech_frames > 0 {
            self.silence_run_ms += f.duration_ms();
        }
        self.recent_db.push_back((f.end_ms(), f.db));
        while let Some((t, _)) = self.recent_db.front() {
            if f.end_ms() - t > STATIONARY_WINDOW_MS {
                self.recent_db.pop_front();
            } else {
                break;
            }
        }
    }

    fn start_ms(&self) -> f64 {
        self.frames
            .first()
            .map(|f| f.start_ms)
            .unwrap_or(self.speech_start_ms)
    }

    fn end_ms(&self) -> f64 {
        self.frames
            .last()
            .map(|f| f.end_ms())
            .unwrap_or(self.speech_start_ms)
    }

    fn duration_ms(&self) -> f64 {
        self.end_ms() - self.start_ms()
    }

    fn voiced_ms(&self) -> f64 {
        if self.speech_frames == 0 {
            0.0
        } else {
            self.last_speech_end_ms - self.speech_start_ms
        }
    }

    fn speech_db_std(&self) -> f32 {
        if self.speech_frames < 2 {
            return f32::INFINITY;
        }
        let n = self.speech_frames as f64;
        let mean = self.speech_db_sum / n;
        let var = (self.speech_db_sq_sum / n - mean * mean).max(0.0);
        var.sqrt() as f32
    }

    /// (mean, std) of the recent level window when it spans the full window.
    fn recent_stats(&self) -> Option<(f32, f32)> {
        let (first, _) = *self.recent_db.front()?;
        let (last, _) = *self.recent_db.back()?;
        // Frames are stored by end time; the window is "full" when it spans
        // nearly the whole analysis window.
        if last - first < STATIONARY_WINDOW_MS * 0.95 {
            return None;
        }
        let n = self.recent_db.len() as f32;
        let mean = self.recent_db.iter().map(|(_, d)| *d).sum::<f32>() / n;
        let var = self
            .recent_db
            .iter()
            .map(|(_, d)| (d - mean) * (d - mean))
            .sum::<f32>()
            / n;
        Some((mean, var.max(0.0).sqrt()))
    }
}

/// Utterance segmenter: adaptive noise floor, onset detection, hangover,
/// pre-roll, min/max utterance length and interim clips.
#[derive(Debug)]
pub struct Segmenter {
    source: AudioSourceKind,
    config: VadConfig,
    frame_samples: usize,
    /// Samples of the next (incomplete) analysis frame.
    pending: Vec<f32>,
    /// Unix ms of `pending[0]`.
    pending_start_ms: f64,
    noise_floor: Option<f32>,
    level: f32,
    /// Recent frames while idle (pre-roll + onset candidates).
    history: VecDeque<Frame>,
    /// Duration of the current run of consecutive speech frames while idle.
    onset_ms: f64,
    utterance: Option<Utterance>,
}

impl Segmenter {
    /// Creates a segmenter for one audio source.
    pub fn new(source: AudioSourceKind, config: VadConfig) -> Self {
        let frame_ms = config.frame_ms.clamp(5, 100);
        let frame_samples = (PIPELINE_SAMPLE_RATE * frame_ms / 1000) as usize;
        Segmenter {
            source,
            config,
            frame_samples,
            pending: Vec::with_capacity(frame_samples),
            pending_start_ms: 0.0,
            noise_floor: None,
            level: 0.0,
            history: VecDeque::new(),
            onset_ms: 0.0,
            utterance: None,
        }
    }

    /// Feed 16 kHz mono audio of any length. Timestamps come from
    /// `frame.captured_at_ms` (sample-accurate offsets inside the frame).
    pub fn push(&mut self, frame: &AudioFrame) -> Vec<SegmenterEvent> {
        let mut out = Vec::new();
        if frame.samples.is_empty() {
            return out;
        }
        let t0 = frame.captured_at_ms as f64;
        if self.pending.is_empty() {
            self.pending_start_ms = t0;
        } else {
            let expected = self.pending_start_ms + self.pending.len() as f64 / SAMPLES_PER_MS;
            if (t0 - expected).abs() > DISCONTINUITY_MS {
                // The input clock jumped (device restart, pause): analyse the
                // partial frame on its own and restart the sample clock.
                let samples = std::mem::take(&mut self.pending);
                let start = self.pending_start_ms;
                self.process_frame(samples, start, &mut out);
                self.pending_start_ms = t0;
            }
        }
        let mut consumed = 0usize;
        let input = &frame.samples;
        while consumed < input.len() {
            let need = self.frame_samples - self.pending.len();
            let take = need.min(input.len() - consumed);
            self.pending.extend(
                input[consumed..consumed + take]
                    .iter()
                    .map(|s| if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 }),
            );
            consumed += take;
            if self.pending.len() == self.frame_samples {
                let samples = std::mem::replace(
                    &mut self.pending,
                    Vec::with_capacity(self.frame_samples),
                );
                let start = self.pending_start_ms;
                self.process_frame(samples, start, &mut out);
                // Re-synchronise with the input clock: the next frame starts
                // at the next unconsumed sample of this input frame.
                self.pending_start_ms = t0 + consumed as f64 / SAMPLES_PER_MS;
            }
        }
        out
    }

    /// Force-end the current utterance (pause / stop / turn end). Buffered
    /// audio that does not fill a whole analysis frame is included.
    pub fn flush(&mut self) -> Vec<SegmenterEvent> {
        let mut out = Vec::new();
        if !self.pending.is_empty() {
            let samples = std::mem::take(&mut self.pending);
            let start = self.pending_start_ms;
            self.process_frame(samples, start, &mut out);
        }
        if self.utterance.is_some() {
            self.finish_utterance(&mut out);
        }
        self.onset_ms = 0.0;
        self.history.clear();
        out
    }

    /// True while an utterance is open.
    pub fn is_speaking(&self) -> bool {
        self.utterance.is_some()
    }

    /// Smoothed level 0..1 for UI meters.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// Current adaptive noise floor in dBFS.
    pub fn noise_floor_dbfs(&self) -> f32 {
        self.noise_floor.unwrap_or(
            (self.config.absolute_floor_dbfs - self.config.threshold_db_above_noise)
                .clamp(FLOOR_MIN_DB, FLOOR_MAX_DB),
        )
    }

    /// The configuration in use.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    // -- internals ---------------------------------------------------------

    fn frame_ms(&self) -> f64 {
        self.frame_samples as f64 / SAMPLES_PER_MS
    }

    fn process_frame(&mut self, samples: Vec<f32>, start_ms: f64, out: &mut Vec<SegmenterEvent>) {
        if samples.is_empty() {
            return;
        }
        let db = frame_db(&samples);
        self.update_level(db);
        if self.noise_floor.is_none() {
            self.noise_floor = Some(db.clamp(FLOOR_MIN_DB, INITIAL_FLOOR_MAX_DB));
        }
        let speaking = self.utterance.is_some();
        let speech = self.is_speech(db, speaking);
        let duration = samples.len() as f64 / SAMPLES_PER_MS;
        self.adapt_floor(db, speech, speaking, duration as f32);
        let frame = Frame {
            start_ms,
            samples,
            db,
            speech,
        };
        if speaking {
            self.speaking_frame(frame, out);
        } else {
            self.idle_frame(frame, out);
        }
    }

    fn update_level(&mut self, db: f32) {
        let target = ((db - METER_FLOOR_DB) / -METER_FLOOR_DB).clamp(0.0, 1.0);
        let alpha = if target > self.level {
            METER_ATTACK
        } else {
            METER_RELEASE
        };
        self.level = (self.level + (target - self.level) * alpha).clamp(0.0, 1.0);
    }

    fn is_speech(&self, db: f32, speaking: bool) -> bool {
        let floor = self.noise_floor_dbfs();
        let (threshold, absolute) = if speaking {
            (
                (self.config.threshold_db_above_noise - HYSTERESIS_DB).max(MIN_THRESHOLD_DB),
                self.config.absolute_floor_dbfs - HYSTERESIS_DB,
            )
        } else {
            (
                self.config.threshold_db_above_noise.max(MIN_THRESHOLD_DB),
                self.config.absolute_floor_dbfs,
            )
        };
        db > absolute && db > floor + threshold
    }

    fn adapt_floor(&mut self, db: f32, speech: bool, speaking: bool, duration_ms: f32) {
        let floor = self.noise_floor.get_or_insert(db);
        if db < *floor {
            *floor += (db - *floor) * FLOOR_DOWN_ALPHA;
        } else if !speaking && !speech {
            let alpha = (duration_ms / FLOOR_UP_TAU_IDLE_MS).min(1.0);
            *floor += (db - *floor) * alpha;
        } else if speaking {
            let alpha = (duration_ms / FLOOR_UP_TAU_SPEECH_MS).min(1.0);
            *floor += (db - *floor) * alpha;
        }
        *floor = floor.clamp(FLOOR_MIN_DB, FLOOR_MAX_DB);
    }

    fn idle_frame(&mut self, frame: Frame, out: &mut Vec<SegmenterEvent>) {
        let speech = frame.speech;
        let duration = frame.duration_ms();
        self.history.push_back(frame);
        if speech {
            self.onset_ms += duration;
        } else {
            self.onset_ms = 0.0;
        }
        if speech && self.onset_ms + 1e-6 >= self.config.min_speech_ms as f64 {
            self.start_utterance(out);
            return;
        }
        // Keep enough history for the pre-roll plus the onset candidates.
        let keep_ms = self.config.pre_roll_ms as f64 + self.onset_ms + self.frame_ms();
        while self.history.len() > 1 {
            let span = self.history.back().map(|f| f.end_ms()).unwrap_or(0.0)
                - self.history.front().map(|f| f.start_ms).unwrap_or(0.0);
            if span > keep_ms {
                self.history.pop_front();
            } else {
                break;
            }
        }
    }

    fn start_utterance(&mut self, out: &mut Vec<SegmenterEvent>) {
        // Onset frames are the trailing run of speech frames.
        let mut onset_index = self.history.len();
        while onset_index > 0 && self.history[onset_index - 1].speech {
            onset_index -= 1;
        }
        let onset_start = self
            .history
            .get(onset_index)
            .map(|f| f.start_ms)
            .unwrap_or(0.0);
        let pre_roll_from = onset_start - self.config.pre_roll_ms as f64 - 0.5;
        let frames: Vec<Frame> = self
            .history
            .drain(..)
            .enumerate()
            .filter(|(i, f)| *i >= onset_index || f.start_ms >= pre_roll_from)
            .map(|(_, f)| f)
            .collect();
        self.onset_ms = 0.0;
        let utterance = Utterance::new(frames);
        tracing::trace!(utterance = %utterance.id, "speech started");
        out.push(SegmenterEvent::SpeechStarted {
            utterance_id: utterance.id.clone(),
            at_ms: onset_start.round() as i64,
        });
        self.utterance = Some(utterance);
    }

    fn speaking_frame(&mut self, frame: Frame, out: &mut Vec<SegmenterEvent>) {
        let silence_ms = self.config.silence_ms as f64;
        let max_ms = self.config.max_utterance_ms.max(1_000) as f64;
        let interim = self.config.interim_interval_ms;
        let Some(utt) = self.utterance.as_mut() else {
            return;
        };
        let speech = frame.speech;
        utt.track(&frame);
        utt.frames.push(frame);

        // Stationarity guard: a constant, moderately loud sound that has been
        // "speech" for several seconds is a new noise source.
        if !utt.noise_suspect && utt.voiced_ms() >= STATIONARY_MIN_SPEECH_MS {
            if let Some((mean, std)) = utt.recent_stats() {
                if std < STATIONARY_MAX_STD_DB && mean < STATIONARY_MAX_LEVEL_DB {
                    utt.noise_suspect = true;
                    self.noise_floor = Some((mean + 1.0).clamp(FLOOR_MIN_DB, FLOOR_MAX_DB));
                    tracing::debug!(level_db = mean, "stationary noise detected; noise floor raised");
                }
            }
        }

        if utt.silence_run_ms + 1e-6 >= silence_ms {
            self.finish_utterance(out);
            return;
        }
        if utt.duration_ms() + 1e-6 >= max_ms {
            self.cut_utterance(out);
            return;
        }
        if let Some(interval) = interim {
            let interval = interval.max(200) as f64;
            let elapsed = utt.end_ms() - utt.speech_start_ms;
            if speech && elapsed - utt.last_interim_ms + 1e-6 >= interval {
                utt.last_interim_ms = (elapsed / interval).floor() * interval;
                let clip = self.build_clip(None, false);
                out.push(SegmenterEvent::Interim(clip));
            }
        }
    }

    /// Ends the open utterance (silence or flush).
    fn finish_utterance(&mut self, out: &mut Vec<SegmenterEvent>) {
        let Some(utt) = self.utterance.as_ref() else {
            return;
        };
        let voiced = utt.voiced_ms();
        let is_noise = utt.noise_suspect && utt.speech_db_std() < NOISE_UTTERANCE_MAX_STD_DB;
        if voiced + 1e-6 < self.config.min_utterance_ms as f64 || is_noise {
            let id = utt.id.clone();
            tracing::trace!(utterance = %id, voiced_ms = voiced, noise = is_noise, "utterance discarded");
            self.utterance = None;
            out.push(SegmenterEvent::Discarded { utterance_id: id });
            return;
        }
        let keep_until = utt.last_speech_end_ms + TRAILING_KEEP_MS;
        let cut = utt
            .frames
            .iter()
            .rposition(|f| f.start_ms < keep_until - 1e-6)
            .unwrap_or(0);
        let clip = self.build_clip(Some(cut), true);
        let Some(mut utt) = self.utterance.take() else {
            return;
        };
        tracing::trace!(utterance = %utt.id, samples = clip.samples.len(), "utterance finished");
        out.push(SegmenterEvent::Utterance(clip));
        // Unused trailing frames are silence: keep them as pre-roll history.
        let rest: Vec<Frame> = utt.frames.drain(cut + 1..).collect();
        self.history.clear();
        self.history.extend(rest);
        let keep_ms = self.config.pre_roll_ms as f64 + self.frame_ms();
        while self.history.len() > 1 {
            let span = self.history.back().map(|f| f.end_ms()).unwrap_or(0.0)
                - self.history.front().map(|f| f.start_ms).unwrap_or(0.0);
            if span > keep_ms {
                self.history.pop_front();
            } else {
                break;
            }
        }
        self.onset_ms = 0.0;
    }

    /// Cuts a long utterance at the quietest recent frame and continues the
    /// speech in a new utterance.
    fn cut_utterance(&mut self, out: &mut Vec<SegmenterEvent>) {
        let Some(utt) = self.utterance.as_ref() else {
            return;
        };
        let end = utt.end_ms();
        let earliest = (end - CUT_SEARCH_MS).max(utt.speech_start_ms + 1.0);
        let mut cut = utt.frames.len() - 1;
        let mut best_db = f32::INFINITY;
        for (i, f) in utt.frames.iter().enumerate() {
            if f.start_ms < earliest || i + 1 == utt.frames.len() {
                continue;
            }
            // Prefer later frames on ties so the clip is as long as allowed.
            if f.db <= best_db {
                best_db = f.db;
                cut = i;
            }
        }
        let clip = self.build_clip(Some(cut), true);
        let Some(mut utt) = self.utterance.take() else {
            return;
        };
        tracing::debug!(utterance = %utt.id, samples = clip.samples.len(), "long utterance cut");
        out.push(SegmenterEvent::Utterance(clip));
        let rest: Vec<Frame> = utt.frames.drain(cut + 1..).collect();
        let at_ms = rest
            .iter()
            .find(|f| f.speech)
            .or(rest.first())
            .map(|f| f.start_ms)
            .unwrap_or(end);
        let mut next = Utterance::new(rest);
        if next.speech_frames == 0 {
            // Speech continues; anchor the new utterance at the cut.
            next.speech_start_ms = at_ms;
            next.last_speech_end_ms = at_ms;
            next.speech_frames = 1;
            next.speech_db_sum = best_db as f64;
            next.speech_db_sq_sum = (best_db as f64) * (best_db as f64);
        }
        next.noise_suspect = utt.noise_suspect;
        out.push(SegmenterEvent::SpeechStarted {
            utterance_id: next.id.clone(),
            at_ms: at_ms.round() as i64,
        });
        self.utterance = Some(next);
    }

    /// Clip of the open utterance's frames `0..=last` (all frames when `None`).
    fn build_clip(&self, last: Option<usize>, is_final: bool) -> AudioClip {
        let Some(utt) = self.utterance.as_ref() else {
            return AudioClip {
                utterance_id: String::new(),
                source: self.source,
                samples: Vec::new(),
                sample_rate: PIPELINE_SAMPLE_RATE,
                start_ms: 0,
                end_ms: 0,
                is_final,
            };
        };
        let last = last
            .unwrap_or(utt.frames.len().saturating_sub(1))
            .min(utt.frames.len().saturating_sub(1));
        let frames = &utt.frames[..utt.frames.len().min(last + 1)];
        let total: usize = frames.iter().map(|f| f.samples.len()).sum();
        let mut samples = Vec::with_capacity(total);
        for f in frames {
            samples.extend_from_slice(&f.samples);
        }
        let start = frames.first().map(|f| f.start_ms).unwrap_or(0.0);
        let end = frames
            .last()
            .map(|f| f.start_ms + (f.samples.len().saturating_sub(1)) as f64 / SAMPLES_PER_MS)
            .unwrap_or(start);
        AudioClip {
            utterance_id: utt.id.clone(),
            source: self.source,
            samples,
            sample_rate: PIPELINE_SAMPLE_RATE,
            start_ms: start.round() as i64,
            end_ms: end.round() as i64,
            is_final,
        }
    }
}

/// RMS level of a frame in dBFS (`MIN_DB` for digital silence).
fn frame_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return MIN_DB;
    }
    let energy: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    let rms = (energy / samples.len() as f64).sqrt();
    if rms <= 1e-5 {
        MIN_DB
    } else {
        (20.0 * rms.log10()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = PIPELINE_SAMPLE_RATE as usize;

    fn n(ms: u32) -> usize {
        RATE * ms as usize / 1000
    }

    /// +/-0.0005 alternating "room noise" (-66 dBFS).
    fn noise(ms: u32) -> Vec<f32> {
        (0..n(ms))
            .map(|i| if i % 2 == 0 { 0.0005 } else { -0.0005 })
            .collect()
    }

    fn tone(ms: u32, amp: f32) -> Vec<f32> {
        (0..n(ms))
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / RATE as f32).sin() * amp)
            .collect()
    }

    /// Speech-like signal: a 180 Hz carrier with a 4 Hz syllabic envelope.
    fn speechlike(ms: u32, amp: f32) -> Vec<f32> {
        (0..n(ms))
            .map(|i| {
                let t = i as f32 / RATE as f32;
                let env = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
                (2.0 * std::f32::consts::PI * 180.0 * t).sin() * amp * env
            })
            .collect()
    }

    /// Feeds `samples` in chunks of `chunk` samples starting at `*t` (Unix ms,
    /// fractional clock kept exactly) and returns `(events, ms_fed_before_event)`.
    struct Feeder {
        seg: Segmenter,
        /// Samples fed so far.
        fed: usize,
        t0: i64,
        events: Vec<(usize, SegmenterEvent)>,
    }

    impl Feeder {
        fn new(config: VadConfig) -> Self {
            Feeder {
                seg: Segmenter::new(AudioSourceKind::System, config),
                fed: 0,
                t0: 1_000_000,
                events: Vec::new(),
            }
        }

        fn now_ms(&self) -> f64 {
            self.t0 as f64 + self.fed as f64 / SAMPLES_PER_MS
        }

        fn feed(&mut self, samples: &[f32], chunk: usize) {
            for c in samples.chunks(chunk) {
                let frame = AudioFrame {
                    source: AudioSourceKind::System,
                    samples: c.to_vec(),
                    captured_at_ms: self.now_ms().round() as i64,
                };
                self.fed += c.len();
                for e in self.seg.push(&frame) {
                    self.events.push((self.fed, e));
                }
            }
        }

        fn flush(&mut self) {
            for e in self.seg.flush() {
                self.events.push((self.fed, e));
            }
        }

        fn started(&self) -> Vec<(usize, String, i64)> {
            self.events
                .iter()
                .filter_map(|(at, e)| match e {
                    SegmenterEvent::SpeechStarted {
                        utterance_id,
                        at_ms,
                    } => Some((*at, utterance_id.clone(), *at_ms)),
                    _ => None,
                })
                .collect()
        }

        fn finals(&self) -> Vec<AudioClip> {
            self.events
                .iter()
                .filter_map(|(_, e)| match e {
                    SegmenterEvent::Utterance(c) => Some(c.clone()),
                    _ => None,
                })
                .collect()
        }

        fn interims(&self) -> Vec<AudioClip> {
            self.events
                .iter()
                .filter_map(|(_, e)| match e {
                    SegmenterEvent::Interim(c) => Some(c.clone()),
                    _ => None,
                })
                .collect()
        }

        fn discarded(&self) -> Vec<String> {
            self.events
                .iter()
                .filter_map(|(_, e)| match e {
                    SegmenterEvent::Discarded { utterance_id } => Some(utterance_id.clone()),
                    _ => None,
                })
                .collect()
        }
    }

    #[test]
    fn config_from_settings() {
        let c = VadConfig::from_settings(0.6, 500, false);
        assert_eq!(c.frame_ms, 20);
        assert_eq!(c.min_speech_ms, 120);
        assert_eq!(c.silence_ms, 500);
        assert_eq!(c.pre_roll_ms, 240);
        assert_eq!(c.min_utterance_ms, 350);
        assert_eq!(c.max_utterance_ms, 20_000);
        assert_eq!(c.interim_interval_ms, None);
        assert!((c.absolute_floor_dbfs + 55.0).abs() < 0.01, "{}", c.absolute_floor_dbfs);
        assert!((100..=150).contains(&c.min_speech_ms));
        assert_eq!(VadConfig::default(), VadConfig::from_settings(0.6, 700, false));
        assert_eq!(
            VadConfig::from_settings(0.6, 500, true).interim_interval_ms,
            Some(VadConfig::DEFAULT_INTERIM_INTERVAL_MS)
        );
        // Higher sensitivity: lower thresholds, faster onset.
        let lo = VadConfig::from_settings(0.0, 700, false);
        let hi = VadConfig::from_settings(1.0, 700, false);
        assert!(hi.threshold_db_above_noise < lo.threshold_db_above_noise);
        assert!(hi.absolute_floor_dbfs < lo.absolute_floor_dbfs);
        assert!(hi.min_speech_ms < lo.min_speech_ms);
        // Out-of-range input is clamped; NaN falls back to the default.
        assert_eq!(VadConfig::from_settings(5.0, 700, false), hi);
        assert_eq!(
            VadConfig::from_settings(f32::NAN, 700, false),
            VadConfig::default()
        );
        assert_eq!(VadConfig::from_settings(0.6, 0, false).silence_ms, 100);
    }

    #[test]
    fn speech_starts_fast_and_ends_after_silence() {
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
        f.feed(&noise(400), 320);
        let tone_start_samples = f.fed;
        let tone_start_ms = f.now_ms();
        assert!(!f.seg.is_speaking());
        f.feed(&tone(1000, 0.3), 320);
        let tone_end_ms = f.now_ms();

        let started = f.started();
        assert_eq!(started.len(), 1, "{:?}", f.events);
        let (fed_at, id, at_ms) = started[0].clone();
        let latency_ms = (fed_at - tone_start_samples) as f64 / SAMPLES_PER_MS;
        assert!(latency_ms <= 150.0, "speech start took {latency_ms} ms");
        assert!((at_ms as f64 - tone_start_ms).abs() <= 20.0, "{at_ms} vs {tone_start_ms}");
        assert!(f.seg.is_speaking());

        // 480 ms of noise: still within the 500 ms hangover.
        f.feed(&noise(480), 320);
        assert!(f.finals().is_empty());
        assert!(f.seg.is_speaking());
        f.feed(&noise(20), 320);
        let finals = f.finals();
        assert_eq!(finals.len(), 1, "500 ms of silence ends the utterance");
        assert!(!f.seg.is_speaking());

        let clip = &finals[0];
        assert!(clip.is_final);
        assert_eq!(clip.utterance_id, id);
        assert_eq!(clip.source, AudioSourceKind::System);
        assert_eq!(clip.sample_rate, PIPELINE_SAMPLE_RATE);
        let pre_roll = 240.0;
        assert!(clip.start_ms as f64 >= tone_start_ms - pre_roll - 1.0, "{}", clip.start_ms);
        assert!(clip.start_ms as f64 <= tone_start_ms);
        assert!(clip.end_ms as f64 >= tone_end_ms - 1.0);
        assert!(clip.end_ms as f64 <= tone_end_ms + TRAILING_KEEP_MS + 21.0);
        let expected_len = ((clip.end_ms - clip.start_ms) as f64 * SAMPLES_PER_MS) as i64;
        assert!((clip.samples.len() as i64 - expected_len).abs() <= 32);
        assert!(f.discarded().is_empty());

        // Nothing else happens during further silence.
        let before = f.events.len();
        f.feed(&noise(2000), 320);
        assert_eq!(f.events.len(), before);
    }

    #[test]
    fn one_second_burst_is_never_discarded() {
        for sensitivity in [0.0, 0.3, 0.6, 1.0] {
            for silence in [300, 500, 700, 1500] {
                for amp in [0.05, 0.3, 0.9] {
                    let mut f = Feeder::new(VadConfig::from_settings(sensitivity, silence, false));
                    f.feed(&noise(500), 320);
                    f.feed(&tone(1000, amp), 320);
                    f.feed(&noise(silence + 100), 320);
                    assert_eq!(
                        f.finals().len(),
                        1,
                        "s={sensitivity} silence={silence} amp={amp}: {:?}",
                        f.events.iter().map(|(_, e)| e).collect::<Vec<_>>()
                    );
                    assert!(f.discarded().is_empty());
                    let clip = &f.finals()[0];
                    assert!(clip.duration_ms() >= 1000, "{}", clip.duration_ms());
                }
            }
        }
    }

    #[test]
    fn clicks_never_start_speech() {
        let config = VadConfig::from_settings(0.6, 500, false);
        let min_speech = config.min_speech_ms;
        let mut f = Feeder::new(config);
        f.feed(&noise(500), 320);
        for click_ms in [1, 5, 20, 40, 60, 100] {
            assert!(click_ms < min_speech);
            f.feed(&tone(click_ms, 0.9), 320);
            f.feed(&noise(300), 320);
        }
        // A click train with gaps never accumulates consecutive speech.
        for _ in 0..20 {
            f.feed(&tone(40, 0.9), 320);
            f.feed(&noise(40), 320);
        }
        assert!(f.started().is_empty(), "{:?}", f.events);
        assert!(!f.seg.is_speaking());
    }

    #[test]
    fn short_blips_are_discarded() {
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
        f.feed(&noise(400), 320);
        f.feed(&tone(200, 0.3), 320);
        f.feed(&noise(600), 320);
        let started = f.started();
        assert_eq!(started.len(), 1);
        assert_eq!(f.discarded(), vec![started[0].1.clone()]);
        assert!(f.finals().is_empty());
    }

    #[test]
    fn long_speech_is_cut_at_max_utterance_preferring_a_quiet_frame() {
        let config = VadConfig::from_settings(0.6, 500, false);
        let max = config.max_utterance_ms as i64;
        let mut f = Feeder::new(config);
        f.feed(&noise(500), 320);
        let speech_start = f.now_ms();
        f.feed(&speechlike(19_000, 0.3), 320);
        // A quieter (but still voiced) stretch shortly before the limit.
        let dip_start = f.now_ms();
        f.feed(&speechlike(120, 0.03), 320);
        let dip_end = f.now_ms();
        f.feed(&speechlike(6_000, 0.3), 320);
        f.feed(&noise(800), 320);

        let finals = f.finals();
        assert_eq!(finals.len(), 2, "{:?}", f.started());
        let first = &finals[0];
        assert!(first.duration_ms() <= max + 20, "{}", first.duration_ms());
        assert!(first.duration_ms() >= max - 1_600, "{}", first.duration_ms());
        assert!(
            (first.end_ms as f64) >= dip_start - 1.0 && (first.end_ms as f64) <= dip_end + 1.0,
            "cut at {} expected within the dip {dip_start}..{dip_end}",
            first.end_ms
        );
        assert!(first.start_ms as f64 <= speech_start);

        let started = f.started();
        assert_eq!(started.len(), 2);
        assert_ne!(started[0].1, started[1].1);
        assert_eq!(finals[1].utterance_id, started[1].1);
        // No audio is lost or duplicated at the cut.
        let gap = finals[1].start_ms - first.end_ms;
        assert!((0..=2).contains(&gap), "gap {gap}");
        assert!(f.discarded().is_empty());
    }

    #[test]
    fn interim_clips_every_interval_while_speaking() {
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, true));
        f.feed(&noise(400), 320);
        f.feed(&speechlike(4_000, 0.3), 320);
        f.feed(&noise(700), 320);
        let started = f.started();
        assert_eq!(started.len(), 1);
        let interims = f.interims();
        assert_eq!(interims.len(), 2, "1.5 s and 3.0 s into a 4 s utterance");
        for (i, clip) in interims.iter().enumerate() {
            assert!(!clip.is_final);
            assert_eq!(clip.utterance_id, started[0].1);
            let since_speech = clip.end_ms - started[0].2;
            let expected = 1_500 * (i as i64 + 1);
            assert!(
                (since_speech - expected).abs() <= 40,
                "interim {i} at {since_speech} ms"
            );
        }
        assert!(interims[1].samples.len() > interims[0].samples.len());
        let finals = f.finals();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].utterance_id, started[0].1);
        // Interim clips must be prefixes of the final clip.
        assert_eq!(finals[0].start_ms, interims[0].start_ms);

        // Disabled: no interim clips.
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
        f.feed(&noise(400), 320);
        f.feed(&speechlike(4_000, 0.3), 320);
        f.feed(&noise(700), 320);
        assert!(f.interims().is_empty());
        assert_eq!(f.finals().len(), 1);
    }

    #[test]
    fn flush_finalizes_the_current_utterance() {
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
        f.feed(&noise(400), 320);
        f.feed(&tone(800, 0.3), 320);
        assert!(f.seg.is_speaking());
        f.flush();
        assert_eq!(f.finals().len(), 1);
        assert!(!f.seg.is_speaking());
        // Flushing while idle does nothing.
        assert!(f.seg.flush().is_empty());
        // Speech right after a flush starts a new utterance.
        f.feed(&tone(1000, 0.3), 320);
        f.flush();
        let finals = f.finals();
        assert_eq!(finals.len(), 2);
        assert_ne!(finals[0].utterance_id, finals[1].utterance_id);
        assert!(finals[1].start_ms > finals[0].end_ms, "no overlap after flush");
        // A flushed blip shorter than min_utterance_ms is discarded.
        f.feed(&noise(600), 320);
        f.feed(&tone(200, 0.3), 320);
        f.flush();
        assert_eq!(f.discarded().len(), 1);
    }

    #[test]
    fn timestamps_are_sample_accurate_for_odd_chunk_sizes() {
        for chunk in [97usize, 160, 333, 480, 1_000, 4_096] {
            let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
            f.feed(&noise(1_000), chunk);
            let tone_start = f.now_ms();
            f.feed(&tone(900, 0.3), chunk);
            let tone_end = f.now_ms();
            f.feed(&noise(900), chunk);
            let started = f.started();
            assert_eq!(started.len(), 1, "chunk {chunk}");
            // Onset detection is quantised to analysis frames (20 ms).
            assert!(
                (started[0].2 as f64 - tone_start).abs() <= 21.0,
                "chunk {chunk}: {} vs {tone_start}",
                started[0].2
            );
            let clip = &f.finals()[0];
            assert!(clip.end_ms as f64 >= tone_end - 21.0, "chunk {chunk}");
            let span = (clip.end_ms - clip.start_ms) as f64;
            let len_ms = clip.samples.len() as f64 / SAMPLES_PER_MS;
            assert!((span - len_ms).abs() <= 2.0, "chunk {chunk}: {span} vs {len_ms}");
        }
    }

    #[test]
    fn level_meter_rises_and_decays() {
        let mut f = Feeder::new(VadConfig::default());
        assert_eq!(f.seg.level(), 0.0);
        f.feed(&noise(300), 320);
        assert!(f.seg.level() < 0.15, "{}", f.seg.level());
        f.feed(&tone(300, 0.3), 320);
        let loud = f.seg.level();
        assert!(loud > 0.6 && loud <= 1.0, "{loud}");
        f.feed(&noise(1_000), 320);
        assert!(f.seg.level() < 0.15, "{}", f.seg.level());
    }

    #[test]
    fn noise_floor_adapts_to_the_room() {
        // A noisier room (-50 dBFS, above the absolute floor) is learned and
        // not mistaken for speech.
        let room: Vec<f32> = (0..n(2_000))
            .map(|i| if i % 2 == 0 { 0.003 } else { -0.003 })
            .collect();
        let mut f = Feeder::new(VadConfig::default());
        f.feed(&room, 320);
        assert!(f.started().is_empty());
        assert!((f.seg.noise_floor_dbfs() + 50.5).abs() < 2.0, "{}", f.seg.noise_floor_dbfs());
        // Speech on top of it is still detected.
        f.feed(&tone(800, 0.3), 320);
        assert_eq!(f.started().len(), 1);
        // Fast adaptation downward when the room gets quiet.
        f.feed(&noise(300), 320);
        assert!(f.seg.noise_floor_dbfs() < -60.0, "{}", f.seg.noise_floor_dbfs());
    }

    #[test]
    fn stationary_noise_onset_is_recovered() {
        let mut f = Feeder::new(VadConfig::from_settings(0.6, 500, false));
        f.feed(&noise(1_000), 320);
        // A fan switches on: constant -43 dBFS hum that first looks like speech.
        let hum = tone(8_000, 0.01);
        f.feed(&hum, 320);
        assert_eq!(f.started().len(), 1);
        assert_eq!(f.discarded().len(), 1, "constant noise is not an utterance");
        assert!(f.finals().is_empty());
        assert!(!f.seg.is_speaking());
        // Speech over the hum is detected afterwards.
        let voice: Vec<f32> = speechlike(1_500, 0.3)
            .iter()
            .zip(tone(1_500, 0.01))
            .map(|(a, b)| a + b)
            .collect();
        f.feed(&voice, 320);
        let hum_tail = tone(1_000, 0.01);
        f.feed(&hum_tail, 320);
        assert_eq!(f.finals().len(), 1, "{:?}", f.events.iter().map(|(_, e)| e).collect::<Vec<_>>());
    }

    #[test]
    fn discontinuities_and_bad_samples_are_handled() {
        let mut seg = Segmenter::new(AudioSourceKind::Microphone, VadConfig::default());
        // Partial frame, then a timestamp jump.
        assert!(seg
            .push(&AudioFrame {
                source: AudioSourceKind::Microphone,
                samples: vec![0.0005; 100],
                captured_at_ms: 0,
            })
            .is_empty());
        let mut events = seg.push(&AudioFrame {
            source: AudioSourceKind::Microphone,
            samples: vec![f32::NAN; 320],
            captured_at_ms: 10_000,
        });
        events.extend(seg.push(&AudioFrame {
            source: AudioSourceKind::Microphone,
            samples: vec![f32::INFINITY, -2.0, 0.5],
            captured_at_ms: 10_020,
        }));
        assert!(events.is_empty());
        assert!(seg.level().is_finite());
        assert!(seg.noise_floor_dbfs().is_finite());
        assert!(seg
            .push(&AudioFrame {
                source: AudioSourceKind::Microphone,
                samples: Vec::new(),
                captured_at_ms: 10_040,
            })
            .is_empty());
        assert_eq!(seg.config(), &VadConfig::default());
    }
}
