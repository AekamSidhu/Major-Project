//! Local OS OCR.
//!
//! * **Windows** – `Windows.Media.Ocr` (documented WinRT API, runs fully on
//!   the device with the OCR languages installed for the user profile).
//! * Other platforms – `None` (the engine falls back to vision models or no
//!   OCR).
//!
//! OCR text is screen content: it is returned to the caller and never logged.

use std::sync::Arc;

use openmeet_core::providers::OcrEngine;

/// The system OCR engine, when the OS provides one usable from a worker
/// thread. Windows: `Windows.Media.Ocr` with the user's profile languages
/// (`None` when no OCR language is installed). Other platforms: `None`.
pub fn system_ocr() -> Option<Arc<dyn OcrEngine>> {
    #[cfg(windows)]
    {
        match windows_ocr::WindowsOcr::new() {
            Ok(engine) => Some(Arc::new(engine)),
            Err(e) => {
                tracing::info!(error = %e, "Windows OCR is unavailable");
                None
            }
        }
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Joins recognized lines into the output text: one line per row, empty
/// lines dropped, whitespace inside a line collapsed.
#[cfg_attr(not(windows), allow(dead_code))]
fn assemble_output(lines: impl IntoIterator<Item = String>) -> openmeet_core::providers::OcrOutput {
    let lines: Vec<String> = lines
        .into_iter()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect();
    openmeet_core::providers::OcrOutput {
        text: lines.join("\n"),
        lines,
    }
}

/// Size `(w, h)` scaled down (aspect preserved) so neither side exceeds
/// `max_dim`; unchanged when it already fits. Never returns a zero side.
#[cfg_attr(not(windows), allow(dead_code))]
fn fit_within(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    let max_dim = max_dim.max(1);
    if w <= max_dim && h <= max_dim {
        return (w, h);
    }
    let scale = f64::from(max_dim) / f64::from(w.max(h));
    (
        ((f64::from(w) * scale).floor() as u32).clamp(1, max_dim),
        ((f64::from(h) * scale).floor() as u32).clamp(1, max_dim),
    )
}

#[cfg(windows)]
mod windows_ocr {
    use std::time::Instant;

    use image::RgbaImage;
    use openmeet_core::ProviderError;
    use openmeet_core::providers::{OcrEngine, OcrOutput};
    use openmeet_core::types::{ProviderInfo, ProviderLocation};
    use windows::Graphics::Imaging::{BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Ocr::OcrEngine as WinOcrEngine;
    use windows::Storage::Streams::DataWriter;

    use super::{assemble_output, fit_within};

    /// Smallest image side worth recognizing.
    const MIN_SIDE: u32 = 8;
    /// Used when the OS does not report `MaxImageDimension`.
    const FALLBACK_MAX_DIMENSION: u32 = 2600;

    /// `Windows.Media.Ocr` engine for the user's profile languages.
    ///
    /// Usable from any worker thread: WinRT activation joins the process-wide
    /// multithreaded apartment when the thread has no COM apartment, and the
    /// asynchronous recognition is awaited with a blocking `join`.
    pub(super) struct WindowsOcr {
        engine: WinOcrEngine,
        language: String,
        max_dimension: u32,
    }

    impl WindowsOcr {
        pub(super) fn new() -> windows::core::Result<WindowsOcr> {
            let engine = WinOcrEngine::TryCreateFromUserProfileLanguages()?;
            let language = engine
                .RecognizerLanguage()
                .and_then(|l| l.LanguageTag())
                .map(|t| t.to_string_lossy())
                .unwrap_or_default();
            let max_dimension = WinOcrEngine::MaxImageDimension().unwrap_or(FALLBACK_MAX_DIMENSION);
            Ok(WindowsOcr {
                engine,
                language,
                max_dimension: if max_dimension == 0 {
                    FALLBACK_MAX_DIMENSION
                } else {
                    max_dimension
                },
            })
        }

        fn recognize(&self, image: RgbaImage) -> windows::core::Result<OcrOutput> {
            let (w, h) = image.dimensions();
            let (tw, th) = fit_within(w, h, self.max_dimension);
            let image = if (tw, th) != (w, h) {
                image::imageops::resize(&image, tw, th, image::imageops::FilterType::Triangle)
            } else {
                image
            };
            let (w, h) = image.dimensions();
            // RGBA → BGRA (premultiplied; screen frames are opaque so the
            // color values are unchanged).
            let mut bgra = image.into_raw();
            for px in bgra.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            let writer = DataWriter::new()?;
            writer.WriteBytes(&bgra)?;
            let buffer = writer.DetachBuffer()?;
            let bitmap = SoftwareBitmap::CreateCopyWithAlphaFromBuffer(
                &buffer,
                BitmapPixelFormat::Bgra8,
                w as i32,
                h as i32,
                BitmapAlphaMode::Premultiplied,
            )?;
            let result = self.engine.RecognizeAsync(&bitmap)?.join();
            let _ = bitmap.Close();
            let result = result?;
            let mut lines = Vec::new();
            for line in &result.Lines()? {
                lines.push(line.Text()?.to_string_lossy());
            }
            Ok(assemble_output(lines))
        }
    }

    impl OcrEngine for WindowsOcr {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                id: "windows_ocr".to_string(),
                display_name: "Windows OCR".to_string(),
                model: if self.language.is_empty() {
                    "Windows.Media.Ocr".to_string()
                } else {
                    format!("Windows.Media.Ocr ({})", self.language)
                },
                location: ProviderLocation::Local,
                endpoint: None,
            }
        }

        fn recognize_png(&self, png: &[u8]) -> Result<OcrOutput, ProviderError> {
            let started = Instant::now();
            let image = image::load_from_memory_with_format(png, image::ImageFormat::Png)
                .map_err(|_| {
                    ProviderError::InvalidResponse(
                        "the screen image could not be decoded".to_string(),
                    )
                })?
                .to_rgba8();
            let (w, h) = image.dimensions();
            if w < MIN_SIDE || h < MIN_SIDE {
                return Ok(OcrOutput::default());
            }
            let output = self.recognize(image).map_err(|e| {
                ProviderError::Other(format!("Windows OCR failed ({:#010x})", e.code().0))
            })?;
            tracing::debug!(
                width = w,
                height = h,
                lines = output.lines.len(),
                chars = output.text.chars().count(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "OCR completed"
            );
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_assembly() {
        let out = assemble_output(vec![
            "  fn   main() {".to_string(),
            "".to_string(),
            "   ".to_string(),
            "println!(\"hi\");".to_string(),
        ]);
        assert_eq!(out.lines, vec!["fn main() {", "println!(\"hi\");"]);
        assert_eq!(out.text, "fn main() {\nprintln!(\"hi\");");
        assert_eq!(assemble_output(Vec::new()).text, "");
    }

    #[test]
    fn fitting_images_into_the_ocr_limit() {
        assert_eq!(fit_within(1280, 720, 2600), (1280, 720));
        assert_eq!(fit_within(5200, 1300, 2600), (2600, 650));
        assert_eq!(fit_within(1000, 4000, 2000), (500, 2000));
        assert_eq!(fit_within(10_000, 1, 100), (100, 1));
        assert_eq!(fit_within(10, 10, 0), (1, 1));
    }

    #[test]
    fn system_ocr_availability_matches_platform() {
        let engine = system_ocr();
        if cfg!(not(windows)) {
            assert!(engine.is_none());
        }
        if let Some(engine) = engine {
            let info = engine.info();
            assert_eq!(info.id, "windows_ocr");
            assert_eq!(info.location, openmeet_core::types::ProviderLocation::Local);
            // Undecodable input is an error, not a panic.
            assert!(engine.recognize_png(b"not a png").is_err());
        }
    }
}
