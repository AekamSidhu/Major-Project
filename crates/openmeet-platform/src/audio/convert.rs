//! Pure audio conversion: every PCM sample format cpal can deliver → mono
//! f32, and the [`FrameAssembler`] that resamples to 16 kHz, keeps a
//! continuous wall-clock timeline and cuts fixed-size [`AudioFrame`]s.
//!
//! Nothing here touches the OS, so all of it is unit tested.

use cpal::{Data, I24, SampleFormat, U24};
use openmeet_core::audio::{
    AudioFrame, PIPELINE_SAMPLE_RATE, Resampler, downmix_to_mono, i16_to_f32, i32_to_f32,
    u16_to_f32,
};
use openmeet_core::types::AudioSourceKind;

/// Samples per emitted frame: 40 ms at 16 kHz (the contract allows 20–100 ms).
pub(crate) const FRAME_SAMPLES: usize = 640;

/// A gap between the end of the timeline and the start of a new packet
/// larger than this is filled with silence (loopback capture delivers no
/// packets while nothing is playing). Smaller differences are callback
/// jitter and are ignored.
const GAP_FILL_THRESHOLD_MS: f64 = 80.0;

/// Gaps longer than this are not filled; the timeline is re-anchored instead
/// (e.g. after the machine was suspended).
const MAX_GAP_FILL_MS: f64 = 5_000.0;

/// When the sample-counted timeline runs ahead of wall-clock time by more
/// than this (device clock drift or late, bursty delivery), it is
/// re-anchored at the next frame boundary.
const REANCHOR_AHEAD_MS: f64 = 500.0;

const PIPELINE_SAMPLES_PER_MS: f64 = PIPELINE_SAMPLE_RATE as f64 / 1000.0;

/// Converts one cpal input buffer (any PCM sample format, interleaved
/// `channels`) to mono f32 in [-1, 1]. Returns `None` for formats that are not
/// PCM (DSD bitstreams), which cannot be converted sample by sample.
pub(crate) fn data_to_mono(data: &Data, channels: u16) -> Option<Vec<f32>> {
    let interleaved = data_to_f32(data)?;
    Some(downmix_to_mono(&interleaved, channels))
}

/// Interleaved PCM of any supported format → f32 in [-1, 1].
pub(crate) fn data_to_f32(data: &Data) -> Option<Vec<f32>> {
    fn map<T: cpal::SizedSample + Copy>(data: &Data, f: impl Fn(T) -> f32) -> Option<Vec<f32>> {
        data.as_slice::<T>()
            .map(|s| s.iter().map(|&v| sanitize(f(v))).collect())
    }
    const I24_SCALE: f32 = 8_388_608.0;
    const I64_SCALE: f64 = 9_223_372_036_854_775_808.0;
    match data.sample_format() {
        SampleFormat::I8 => map::<i8>(data, |s| f32::from(s) / 128.0),
        SampleFormat::I16 => map::<i16>(data, i16_to_f32),
        SampleFormat::I24 => map::<I24>(data, |s| s.inner() as f32 / I24_SCALE),
        SampleFormat::I32 => map::<i32>(data, i32_to_f32),
        SampleFormat::I64 => map::<i64>(data, |s| (s as f64 / I64_SCALE) as f32),
        SampleFormat::U8 => map::<u8>(data, |s| (f32::from(s) - 128.0) / 128.0),
        SampleFormat::U16 => map::<u16>(data, u16_to_f32),
        SampleFormat::U24 => map::<U24>(data, |s| (s.inner() as f32 - I24_SCALE) / I24_SCALE),
        SampleFormat::U32 => map::<u32>(data, |s| {
            ((f64::from(s) - 2_147_483_648.0) / 2_147_483_648.0) as f32
        }),
        SampleFormat::U64 => map::<u64>(data, |s| ((s as f64 - I64_SCALE) / I64_SCALE) as f32),
        SampleFormat::F32 => map::<f32>(data, |s| s),
        SampleFormat::F64 => map::<f64>(data, |s| s as f32),
        _ => None,
    }
}

/// Clamps to [-1, 1]; NaN / infinities (broken drivers) become silence.
#[inline]
fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Turns device-rate mono packets into 16 kHz [`AudioFrame`]s of
/// [`FRAME_SAMPLES`] samples with wall-clock timestamps.
///
/// The timeline is sample-accurate between packets. It is anchored at the
/// first packet's capture time; later gaps (no packets delivered) are filled
/// with silence so downstream voice activity detection sees time pass, and
/// large clock disagreements re-anchor it.
#[derive(Debug, Clone)]
pub(crate) struct FrameAssembler {
    source: AudioSourceKind,
    input_rate: u32,
    resampler: Resampler,
    /// 16 kHz samples not yet emitted.
    pending: Vec<f32>,
    /// Wall-clock time (Unix ms) of `pending[0]`; `None` until anchored.
    pending_start_ms: Option<f64>,
}

impl FrameAssembler {
    pub(crate) fn new(source: AudioSourceKind, input_rate: u32) -> Self {
        FrameAssembler {
            source,
            input_rate: input_rate.max(1),
            resampler: Resampler::new(input_rate, PIPELINE_SAMPLE_RATE),
            pending: Vec::with_capacity(FRAME_SAMPLES * 2),
            pending_start_ms: None,
        }
    }

    /// Switches to a new device sample rate (stream rebuilt on another
    /// device). The timeline is kept.
    pub(crate) fn set_input_rate(&mut self, input_rate: u32) {
        let input_rate = input_rate.max(1);
        if input_rate != self.input_rate {
            self.input_rate = input_rate;
            self.resampler = Resampler::new(input_rate, PIPELINE_SAMPLE_RATE);
        }
    }

    /// Wall-clock time of the next sample on the timeline.
    pub(crate) fn timeline_end_ms(&self) -> Option<f64> {
        self.pending_start_ms
            .map(|start| start + self.pending.len() as f64 / PIPELINE_SAMPLES_PER_MS)
    }

    /// Feeds one device-rate mono packet whose first sample was captured at
    /// `packet_start_ms` (Unix ms). Returns the completed frames.
    pub(crate) fn push(&mut self, mono: &[f32], packet_start_ms: f64) -> Vec<AudioFrame> {
        let mut out = Vec::new();
        if mono.is_empty() {
            return out;
        }
        match self.timeline_end_ms() {
            None => self.anchor(packet_start_ms),
            Some(end) => {
                let delta = packet_start_ms - end;
                if delta > GAP_FILL_THRESHOLD_MS && delta <= MAX_GAP_FILL_MS {
                    self.append_silence(delta);
                } else if delta > MAX_GAP_FILL_MS || delta < -REANCHOR_AHEAD_MS {
                    out.extend(self.flush());
                    self.resampler = Resampler::new(self.input_rate, PIPELINE_SAMPLE_RATE);
                    self.anchor(packet_start_ms);
                }
            }
        }
        let resampled = self.resampler.process(mono);
        self.pending.extend_from_slice(&resampled);
        out.extend(self.drain_full_frames());
        out
    }

    /// Extends the timeline with silence up to `until_ms` (used while a
    /// loopback stream delivers no packets because nothing is playing).
    /// Anchors the timeline at `until_ms` when no audio was seen yet.
    pub(crate) fn fill_silence_until(&mut self, until_ms: f64) -> Vec<AudioFrame> {
        match self.timeline_end_ms() {
            None => {
                self.anchor(until_ms);
                Vec::new()
            }
            Some(end) if until_ms > end => {
                self.append_silence((until_ms - end).min(MAX_GAP_FILL_MS));
                self.drain_full_frames()
            }
            Some(_) => Vec::new(),
        }
    }

    /// Emits everything pending as a final (possibly short) frame.
    pub(crate) fn flush(&mut self) -> Vec<AudioFrame> {
        let Some(start) = self.pending_start_ms else {
            return Vec::new();
        };
        if self.pending.is_empty() {
            return Vec::new();
        }
        let samples = std::mem::take(&mut self.pending);
        let frame = AudioFrame {
            source: self.source,
            captured_at_ms: start.round() as i64,
            samples,
        };
        self.pending_start_ms = Some(start + frame.samples.len() as f64 / PIPELINE_SAMPLES_PER_MS);
        vec![frame]
    }

    fn anchor(&mut self, start_ms: f64) {
        debug_assert!(self.pending.is_empty() || self.pending_start_ms.is_none());
        self.pending.clear();
        self.pending_start_ms = Some(start_ms);
    }

    fn append_silence(&mut self, ms: f64) {
        let n = (ms * PIPELINE_SAMPLES_PER_MS).round().max(0.0) as usize;
        self.pending.resize(self.pending.len() + n, 0.0);
    }

    fn drain_full_frames(&mut self) -> Vec<AudioFrame> {
        let Some(mut start) = self.pending_start_ms else {
            return Vec::new();
        };
        let mut frames = Vec::with_capacity(self.pending.len() / FRAME_SAMPLES);
        let mut offset = 0;
        while self.pending.len() - offset >= FRAME_SAMPLES {
            frames.push(AudioFrame {
                source: self.source,
                samples: self.pending[offset..offset + FRAME_SAMPLES].to_vec(),
                captured_at_ms: start.round() as i64,
            });
            offset += FRAME_SAMPLES;
            start += FRAME_SAMPLES as f64 / PIPELINE_SAMPLES_PER_MS;
        }
        if offset > 0 {
            self.pending.drain(..offset);
            self.pending_start_ms = Some(start);
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a cpal `Data` view over a typed slice.
    fn data<T: cpal::SizedSample>(buf: &mut [T]) -> Data {
        // SAFETY: `buf` outlives the returned view within each test, the
        // pointer and length describe `buf` exactly and the format matches T.
        unsafe { Data::from_parts(buf.as_mut_ptr().cast(), buf.len(), T::FORMAT) }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-3, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn converts_signed_integer_formats() {
        let mut i8s = [i8::MIN, 0, 64];
        assert_close(&data_to_f32(&data(&mut i8s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut i16s = [i16::MIN, 0, 16_384];
        assert_close(&data_to_f32(&data(&mut i16s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut i24s = [
            I24::new_unchecked(-8_388_608),
            I24::new_unchecked(0),
            I24::new_unchecked(4_194_304),
        ];
        assert_close(&data_to_f32(&data(&mut i24s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut i32s = [i32::MIN, 0, 1 << 30];
        assert_close(&data_to_f32(&data(&mut i32s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut i64s = [i64::MIN, 0, 1 << 62];
        assert_close(&data_to_f32(&data(&mut i64s)).unwrap(), &[-1.0, 0.0, 0.5]);
    }

    #[test]
    fn converts_unsigned_integer_formats() {
        let mut u8s = [0u8, 128, 192];
        assert_close(&data_to_f32(&data(&mut u8s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut u16s = [0u16, 32_768, 49_152];
        assert_close(&data_to_f32(&data(&mut u16s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut u24s = [
            U24::new_unchecked(0),
            U24::new_unchecked(8_388_608),
            U24::new_unchecked(12_582_912),
        ];
        assert_close(&data_to_f32(&data(&mut u24s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut u32s = [0u32, 1 << 31, 3 << 30];
        assert_close(&data_to_f32(&data(&mut u32s)).unwrap(), &[-1.0, 0.0, 0.5]);
        let mut u64s = [0u64, 1 << 63, 3 << 62];
        assert_close(&data_to_f32(&data(&mut u64s)).unwrap(), &[-1.0, 0.0, 0.5]);
    }

    #[test]
    fn converts_and_sanitizes_float_formats() {
        let mut f32s = [0.25f32, -2.0, f32::NAN, f32::INFINITY];
        assert_close(
            &data_to_f32(&data(&mut f32s)).unwrap(),
            &[0.25, -1.0, 0.0, 0.0],
        );
        let mut f64s = [0.5f64, 1.5, -0.125];
        assert_close(&data_to_f32(&data(&mut f64s)).unwrap(), &[0.5, 1.0, -0.125]);
    }

    #[test]
    fn dsd_is_not_convertible() {
        let mut raw = [0u8; 4];
        // SAFETY: the view covers `raw` exactly; DSD bytes are never read as
        // PCM because conversion refuses the format.
        let dsd =
            unsafe { Data::from_parts(raw.as_mut_ptr().cast(), raw.len(), SampleFormat::DsdU8) };
        assert!(data_to_f32(&dsd).is_none());
        assert!(data_to_mono(&dsd, 1).is_none());
    }

    #[test]
    fn downmixes_interleaved_channels() {
        let mut stereo = [i16::MAX, 0, 0, i16::MIN];
        let mono = data_to_mono(&data(&mut stereo), 2).unwrap();
        assert_close(&mono, &[0.5, -0.5]);
        let mut six = [0.6f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_close(&data_to_mono(&data(&mut six), 6).unwrap(), &[0.1]);
    }

    fn total_samples(frames: &[AudioFrame]) -> usize {
        frames.iter().map(|f| f.samples.len()).sum()
    }

    #[test]
    fn assembles_fixed_size_frames_with_continuous_timestamps() {
        let mut asm = FrameAssembler::new(AudioSourceKind::Microphone, 48_000);
        let t0 = 1_700_000_000_000.0;
        let mut frames = Vec::new();
        // One second of 10 ms packets at 48 kHz with ±3 ms delivery jitter.
        for i in 0..100 {
            let jitter = if i % 2 == 0 { 3.0 } else { -3.0 };
            frames.extend(asm.push(&[0.1; 480], t0 + f64::from(i) * 10.0 + jitter));
        }
        assert!(frames.iter().all(|f| f.samples.len() == FRAME_SAMPLES));
        assert!(
            frames
                .iter()
                .all(|f| f.source == AudioSourceKind::Microphone)
        );
        assert_eq!(frames[0].captured_at_ms, (t0 + 3.0) as i64);
        for pair in frames.windows(2) {
            assert_eq!(pair[1].captured_at_ms - pair[0].captured_at_ms, 40);
        }
        let flushed = asm.flush();
        let total = total_samples(&frames) + total_samples(&flushed);
        assert!((15_990..=16_010).contains(&total), "{total}");
        assert!(asm.flush().is_empty());
    }

    #[test]
    fn fills_gaps_with_silence() {
        let mut asm = FrameAssembler::new(AudioSourceKind::System, 16_000);
        let t0 = 1_000_000.0;
        let mut frames = asm.push(&[0.5; 1600], t0); // 100 ms
        // Next packet arrives 300 ms after the end of the first one.
        frames.extend(asm.push(&[0.5; 1600], t0 + 400.0));
        frames.extend(asm.flush());
        let samples: Vec<f32> = frames.iter().flat_map(|f| f.samples.clone()).collect();
        // 100 ms audio + 300 ms silence + 100 ms audio.
        assert_eq!(samples.len(), 8_000);
        assert!(samples[1_700..6_300].iter().all(|&s| s == 0.0));
        assert!(samples[6_500..].iter().all(|&s| (s - 0.5).abs() < 1e-3));
        // Timestamps stay continuous across the filled gap.
        for pair in frames.windows(2) {
            let expected = pair[0].captured_at_ms + pair[0].duration_ms();
            assert!((pair[1].captured_at_ms - expected).abs() <= 1);
        }
    }

    #[test]
    fn small_jitter_is_not_a_gap() {
        let mut asm = FrameAssembler::new(AudioSourceKind::Microphone, 16_000);
        let mut frames = asm.push(&[0.2; 640], 0.0);
        frames.extend(asm.push(&[0.2; 640], 40.0 + 50.0));
        assert_eq!(total_samples(&frames), 1_280);
        assert!(frames.iter().flat_map(|f| &f.samples).all(|&s| s > 0.1));
    }

    #[test]
    fn long_gaps_and_clock_lead_reanchor_the_timeline() {
        let mut asm = FrameAssembler::new(AudioSourceKind::System, 16_000);
        let mut frames = asm.push(&[0.3; 320], 0.0);
        assert!(frames.is_empty());
        // 60 s later (suspend / resume): no minute of silence, new anchor.
        frames.extend(asm.push(&[0.3; 640], 60_000.0));
        assert_eq!(total_samples(&frames), 960);
        assert_eq!(frames[0].captured_at_ms, 0);
        assert_eq!(frames[1].captured_at_ms, 60_000);
        // Timeline 1 s ahead of the packet clock: re-anchored, not shifted.
        let mut asm = FrameAssembler::new(AudioSourceKind::System, 16_000);
        asm.push(&[0.3; 16_000], 0.0);
        let frames = asm.push(&[0.3; 640], 0.0);
        assert_eq!(frames.last().unwrap().captured_at_ms, 0);
    }

    #[test]
    fn idle_silence_fill() {
        let mut asm = FrameAssembler::new(AudioSourceKind::System, 48_000);
        // Before any audio: only anchors.
        assert!(asm.fill_silence_until(1_000.0).is_empty());
        let frames = asm.fill_silence_until(1_200.0);
        assert_eq!(
            total_samples(&frames),
            3_200 / FRAME_SAMPLES * FRAME_SAMPLES
        );
        assert_eq!(frames[0].captured_at_ms, 1_000);
        assert!(frames.iter().flat_map(|f| &f.samples).all(|&s| s == 0.0));
        // Going backwards is a no-op.
        assert!(asm.fill_silence_until(500.0).is_empty());
        // Real audio continues the timeline.
        let frames = asm.push(&[0.4; 4_800], 1_200.0);
        assert_eq!(frames[0].captured_at_ms, 1_200);
    }

    #[test]
    fn rate_change_keeps_timeline() {
        let mut asm = FrameAssembler::new(AudioSourceKind::System, 48_000);
        let first = asm.push(&[0.1; 4_800], 0.0);
        asm.set_input_rate(44_100);
        let second = asm.push(&[0.1; 4_410], 100.0);
        let all: Vec<AudioFrame> = first.into_iter().chain(second).chain(asm.flush()).collect();
        let total = total_samples(&all);
        assert!((3_195..=3_205).contains(&total), "{total}");
        assert_eq!(all[0].captured_at_ms, 0);
    }

    #[test]
    fn empty_packets_are_ignored() {
        let mut asm = FrameAssembler::new(AudioSourceKind::Microphone, 16_000);
        assert!(asm.push(&[], 5.0).is_empty());
        assert!(asm.timeline_end_ms().is_none());
        assert!(asm.flush().is_empty());
    }
}
