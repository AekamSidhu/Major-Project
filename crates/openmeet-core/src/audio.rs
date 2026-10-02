//! Audio data types and small DSP helpers shared by capture, VAD and
//! transcription.

use serde::{Deserialize, Serialize};

use crate::types::AudioSourceKind;

/// Canonical pipeline sample rate. Capture adapters resample to this rate,
/// mono, f32 in [-1, 1].
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// A short chunk of captured audio, already converted to 16 kHz mono f32.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFrame {
    pub source: AudioSourceKind,
    /// 16 kHz mono samples in [-1, 1].
    pub samples: Vec<f32>,
    /// Unix milliseconds at which the first sample was captured.
    pub captured_at_ms: i64,
}

impl AudioFrame {
    pub fn duration_ms(&self) -> i64 {
        (self.samples.len() as i64 * 1000) / PIPELINE_SAMPLE_RATE as i64
    }
}

/// A complete (or in-progress, for interim results) utterance cut out by the
/// voice activity detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioClip {
    /// Stable identifier of the utterance. Interim clips and the final clip of
    /// the same utterance share this id.
    pub utterance_id: String,
    pub source: AudioSourceKind,
    /// 16 kHz mono samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Unix milliseconds of the first sample.
    pub start_ms: i64,
    /// Unix milliseconds of the last sample.
    pub end_ms: i64,
    /// True when the utterance ended (silence detected or max length reached).
    pub is_final: bool,
}

impl AudioClip {
    pub fn duration_ms(&self) -> i64 {
        (self.samples.len() as i64 * 1000) / self.sample_rate.max(1) as i64
    }
}

// ---------------------------------------------------------------------------
// Sample conversion
// ---------------------------------------------------------------------------

#[inline]
pub fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

#[inline]
pub fn u16_to_f32(s: u16) -> f32 {
    (s as f32 - 32768.0) / 32768.0
}

#[inline]
pub fn i32_to_f32(s: i32) -> f32 {
    s as f32 / 2_147_483_648.0
}

/// Averages interleaved channels into mono.
pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Streaming sample-rate converter for speech.
///
/// Uses a one-pole low-pass before decimation (to limit aliasing) and linear
/// interpolation. State is kept across calls so chunk boundaries are seamless.
/// This is intentionally simple: speech recognizers are robust to the small
/// spectral artifacts, and it costs a few operations per sample.
#[derive(Debug, Clone)]
pub struct Resampler {
    input_rate: u32,
    output_rate: u32,
    /// Fractional read position into the (virtual) input stream, relative to
    /// `prev`.
    pos: f64,
    step: f64,
    /// Last filtered input sample of the previous chunk.
    prev: f32,
    has_prev: bool,
    lp_state: f32,
    lp_alpha: f32,
}

impl Resampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        let input_rate = input_rate.max(1);
        let output_rate = output_rate.max(1);
        // One-pole low-pass with cutoff ~0.45 * output Nyquist when
        // downsampling; bypassed when upsampling.
        let lp_alpha = if output_rate < input_rate {
            let cutoff = 0.45 * output_rate as f32;
            let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
            let dt = 1.0 / input_rate as f32;
            dt / (rc + dt)
        } else {
            1.0
        };
        Resampler {
            input_rate,
            output_rate,
            pos: 0.0,
            step: input_rate as f64 / output_rate as f64,
            prev: 0.0,
            has_prev: false,
            lp_state: 0.0,
            lp_alpha,
        }
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        // Low-pass filter the chunk.
        let mut filtered = Vec::with_capacity(input.len() + 1);
        if self.has_prev {
            filtered.push(self.prev);
        }
        for &x in input {
            self.lp_state += self.lp_alpha * (x - self.lp_state);
            filtered.push(self.lp_state);
        }
        if self.input_rate == self.output_rate {
            let out = if self.has_prev {
                filtered[1..].to_vec()
            } else {
                filtered.clone()
            };
            self.prev = *filtered.last().unwrap_or(&0.0);
            self.has_prev = true;
            return out;
        }
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 2);
        // `filtered[0]` corresponds to position 0.0 (prev sample when present).
        let last_index = (filtered.len() - 1) as f64;
        while self.pos <= last_index {
            let i = self.pos.floor() as usize;
            let frac = (self.pos - i as f64) as f32;
            let a = filtered[i];
            let b = if i + 1 < filtered.len() {
                filtered[i + 1]
            } else {
                a
            };
            if i + 1 >= filtered.len() && frac > 0.0 {
                break;
            }
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }
        // Re-base position so that the last filtered sample becomes index 0
        // of the next call.
        self.pos -= last_index;
        self.prev = *filtered.last().unwrap_or(&0.0);
        self.has_prev = true;
        out
    }
}

// ---------------------------------------------------------------------------
// WAV encoding (for utterance upload to transcription providers)
// ---------------------------------------------------------------------------

/// Encodes mono f32 samples as a 16-bit PCM WAV file.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decodes a 16-bit PCM mono/stereo WAV produced by [`encode_wav_pcm16`] (or
/// any simple PCM16 WAV). Returns mono samples and the sample rate.
pub fn decode_wav_pcm16(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }
    let mut pos = 12;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.checked_add(size).ok_or("chunk size overflow")?;
        if body_end > bytes.len() {
            return Err("truncated chunk".to_string());
        }
        let body = &bytes[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err("fmt chunk too short".to_string());
            }
            let format = u16::from_le_bytes([body[0], body[1]]);
            if format != 1 {
                return Err(format!("unsupported WAV format {format}"));
            }
            channels = u16::from_le_bytes([body[2], body[3]]);
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if bits != 16 || channels == 0 {
                return Err("missing or unsupported fmt chunk".to_string());
            }
            let samples: Vec<f32> = body
                .chunks_exact(2)
                .map(|b| i16_to_f32(i16::from_le_bytes([b[0], b[1]])))
                .collect();
            return Ok((downmix_to_mono(&samples, channels), rate));
        }
        pos = body_end + (size & 1);
    }
    Err("no data chunk".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_duration() {
        let f = AudioFrame {
            source: AudioSourceKind::Microphone,
            samples: vec![0.0; 3200],
            captured_at_ms: 0,
        };
        assert_eq!(f.duration_ms(), 200);
    }

    #[test]
    fn conversions() {
        assert_eq!(i16_to_f32(0), 0.0);
        assert!((i16_to_f32(i16::MIN) + 1.0).abs() < 1e-6);
        assert!((u16_to_f32(32768)).abs() < 1e-6);
        assert!(i32_to_f32(i32::MAX) > 0.999);
    }

    #[test]
    fn downmix() {
        assert_eq!(downmix_to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(downmix_to_mono(&[0.2, 0.4], 1), vec![0.2, 0.4]);
    }

    #[test]
    fn resample_length_48k_to_16k_across_chunks() {
        let mut r = Resampler::new(48_000, 16_000);
        let mut total = 0;
        for _ in 0..100 {
            total += r.process(&vec![0.25; 480]).len(); // 10 ms chunks
        }
        // 1 second of input -> ~16000 samples (+/- a couple at boundaries)
        assert!((15_990..=16_010).contains(&total), "got {total}");
    }

    #[test]
    fn resample_preserves_dc_level() {
        let mut r = Resampler::new(44_100, 16_000);
        let mut out = Vec::new();
        for _ in 0..50 {
            out.extend(r.process(&vec![0.5; 441]));
        }
        let tail = &out[out.len() - 100..];
        for v in tail {
            assert!((v - 0.5).abs() < 0.01, "{v}");
        }
    }

    #[test]
    fn resample_upsamples() {
        let mut r = Resampler::new(8_000, 16_000);
        let out = r.process(&vec![0.1; 800]);
        assert!((1590..=1602).contains(&out.len()), "{}", out.len());
    }

    #[test]
    fn resample_identity() {
        let mut r = Resampler::new(16_000, 16_000);
        let a = r.process(&[0.0, 0.1, 0.2]);
        let b = r.process(&[0.3]);
        assert_eq!(a.len() + b.len(), 4);
    }

    #[test]
    fn wav_roundtrip() {
        let samples: Vec<f32> = (0..1600).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let wav = encode_wav_pcm16(&samples, 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(wav.len(), 44 + samples.len() * 2);
        let (decoded, rate) = decode_wav_pcm16(&wav).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(decoded.len(), samples.len());
        for (a, b) in samples.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
        assert!(decode_wav_pcm16(b"nope").is_err());
    }
}
