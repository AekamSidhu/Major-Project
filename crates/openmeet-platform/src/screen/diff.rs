//! Perceptual change detection for screen frames (pure, no OS access).
//!
//! Two complementary signals are combined:
//!
//! * a 64-bit **difference hash** (dHash, 9x8 luminance grid) that captures
//!   the global structure of the frame – slide changes, window switches and
//!   scrolling flip many bits;
//! * a **32x32 luminance thumbnail** compared cell by cell (mean absolute
//!   difference with a small noise floor). Each cell covers roughly 40x22 px
//!   of a 1280x720 frame, so a new line of text changes a whole row of cells
//!   while a blinking caret only touches one or two.
//!
//! The change score is `max(hamming / 64, cell_score)`; with the default
//! threshold (0.08) roughly 7 fully changed cells (~0.7 % of the frame) or 6
//! flipped hash bits count as a meaningful change, whereas a caret blink
//! scores below 0.01.

use image::{GrayImage, RgbaImage};

/// Side of the comparison thumbnail.
const THUMB: usize = 32;
/// Per-cell luminance difference ignored as noise (0..255 scale).
const CELL_NOISE_FLOOR: f32 = 2.0;
/// Difference at which a cell counts as fully changed.
const CELL_FULL_CHANGE: f32 = 16.0;
/// Gain applied to the fraction of changed cells. 12 => about 7 fully changed
/// cells out of 1024 reach the default threshold of 0.08.
const CELL_GAIN: f32 = 12.0;

/// 64-bit difference hash: the image is reduced (area average) to a 9x8
/// luminance grid and each bit records whether a cell is brighter than its
/// right neighbour (row-major, most significant bit first). An empty image
/// hashes to 0.
pub fn dhash(gray: &GrayImage) -> u64 {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return 0;
    }
    let grid = area_average(gray.as_raw(), w as usize, h as usize, 1, 9, 8, |p| {
        f32::from(p[0])
    });
    hash_from_grid(&grid)
}

/// Number of differing bits between two hashes.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Decides whether a frame differs meaningfully from the last *accepted*
/// frame. Slow gradual changes therefore accumulate until they cross the
/// threshold.
#[derive(Debug, Clone)]
pub struct ChangeDetector {
    threshold: f32,
    reference: Option<Reference>,
}

#[derive(Debug, Clone)]
struct Reference {
    hash: u64,
    thumb: Vec<f32>,
}

impl ChangeDetector {
    /// `threshold` is the minimum change score (0..1) for a frame to count as
    /// changed. Non-finite values fall back to the default 0.08.
    pub fn new(threshold: f32) -> Self {
        let threshold = if threshold.is_finite() {
            threshold.clamp(0.0, 1.0)
        } else {
            0.08
        };
        ChangeDetector {
            threshold,
            reference: None,
        }
    }

    /// The configured threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// (changed, change_score 0..1, hash). First frame is always changed.
    /// Combines dHash distance with a 32x32 thumbnail mean-abs-diff so small
    /// but meaningful text changes are detected while cursor blinks are not.
    /// The frame becomes the new reference only when it is reported changed.
    pub fn evaluate(&mut self, frame: &RgbaImage) -> (bool, f32, u64) {
        let (thumb, hash) = signature(frame);
        let score = self.score_against_reference(&thumb, hash);
        let changed = self.reference.is_none() || score >= self.threshold;
        if changed {
            self.reference = Some(Reference { hash, thumb });
        }
        (changed, score, hash)
    }

    /// Unconditionally accepts `frame` as the new reference (manual captures)
    /// and returns its (change_score, hash) relative to the previous one.
    pub fn force_accept(&mut self, frame: &RgbaImage) -> (f32, u64) {
        let (thumb, hash) = signature(frame);
        let score = self.score_against_reference(&thumb, hash);
        self.reference = Some(Reference { hash, thumb });
        (score, hash)
    }

    /// Forgets the reference frame: the next frame is reported as changed.
    pub fn reset(&mut self) {
        self.reference = None;
    }

    fn score_against_reference(&self, thumb: &[f32], hash: u64) -> f32 {
        match &self.reference {
            None => 1.0,
            Some(r) => change_score(&r.thumb, r.hash, thumb, hash),
        }
    }
}

/// Change score between two signatures (see module docs).
fn change_score(thumb_a: &[f32], hash_a: u64, thumb_b: &[f32], hash_b: u64) -> f32 {
    let hash_component = hamming(hash_a, hash_b) as f32 / 64.0;
    let mut weighted = 0.0f32;
    for (a, b) in thumb_a.iter().zip(thumb_b.iter()) {
        let d = (a - b).abs();
        if d > CELL_NOISE_FLOOR {
            weighted += (d / CELL_FULL_CHANGE).min(1.0);
        }
    }
    let cells = thumb_a.len().max(1) as f32;
    let cell_component = (weighted / cells * CELL_GAIN).min(1.0);
    hash_component.max(cell_component).clamp(0.0, 1.0)
}

/// 32x32 luminance thumbnail and dHash of an RGBA frame.
fn signature(frame: &RgbaImage) -> (Vec<f32>, u64) {
    let (w, h) = frame.dimensions();
    if w == 0 || h == 0 {
        return (vec![0.0; THUMB * THUMB], 0);
    }
    let thumb = area_average(
        frame.as_raw(),
        w as usize,
        h as usize,
        4,
        THUMB,
        THUMB,
        |p| luma(p[0], p[1], p[2]),
    );
    // dHash of the thumbnail: averaging 32x32 down to 9x8 is equivalent to
    // averaging the full frame (up to cell-boundary rounding) and much cheaper.
    let grid = area_average_f32(&thumb, THUMB, THUMB, 9, 8);
    (thumb, hash_from_grid(&grid))
}

#[inline]
fn luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)
}

fn hash_from_grid(grid: &[f32]) -> u64 {
    let mut hash = 0u64;
    for row in 0..8 {
        for col in 0..8 {
            let left = grid[row * 9 + col];
            let right = grid[row * 9 + col + 1];
            hash = (hash << 1) | u64::from(left > right);
        }
    }
    hash
}

/// Cell bounds `[start, end)` of cell `i` out of `cells` over `len` pixels.
/// Every cell gets at least one pixel, even when `len < cells`.
#[inline]
fn cell_bounds(i: usize, cells: usize, len: usize) -> (usize, usize) {
    let start = (i * len / cells).min(len - 1);
    let end = ((i + 1) * len / cells).max(start + 1).min(len);
    (start, end)
}

/// Area-average downsampling of an interleaved 8-bit image into an
/// `out_w` x `out_h` grid of f32 values produced by `value(pixel)`.
fn area_average(
    raw: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    out_w: usize,
    out_h: usize,
    value: impl Fn(&[u8]) -> f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_w * out_h];
    let stride = w * channels;
    for cy in 0..out_h {
        let (y0, y1) = cell_bounds(cy, out_h, h);
        for cx in 0..out_w {
            let (x0, x1) = cell_bounds(cx, out_w, w);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                let row = &raw[y * stride..(y + 1) * stride];
                for x in x0..x1 {
                    sum += value(&row[x * channels..x * channels + channels]);
                }
            }
            out[cy * out_w + cx] = sum / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }
    out
}

fn area_average_f32(src: &[f32], w: usize, h: usize, out_w: usize, out_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_w * out_h];
    for cy in 0..out_h {
        let (y0, y1) = cell_bounds(cy, out_h, h);
        for cx in 0..out_w {
            let (x0, x1) = cell_bounds(cx, out_w, w);
            let mut sum = 0.0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += src[y * w + x];
                }
            }
            out[cy * out_w + cx] = sum / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgba};

    const W: u32 = 1280;
    const H: u32 = 720;

    fn canvas() -> RgbaImage {
        RgbaImage::from_pixel(W, H, Rgba([250, 250, 250, 255]))
    }

    fn fill(img: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, c: [u8; 4]) {
        for yy in y..(y + h).min(img.height()) {
            for xx in x..(x + w).min(img.width()) {
                img.put_pixel(xx, yy, Rgba(c));
            }
        }
    }

    /// Draws a line of "text": glyph-like dark boxes (6x12 ink in 8 px cells).
    fn text_line(img: &mut RgbaImage, x: u32, y: u32, chars: u32) {
        for i in 0..chars {
            // Leave every 5th char blank like word spacing.
            if i % 5 == 4 {
                continue;
            }
            fill(img, x + i * 8, y, 6, 12, [30, 30, 30, 255]);
        }
    }

    fn document() -> RgbaImage {
        let mut img = canvas();
        for line in 0..20 {
            text_line(&mut img, 40, 40 + line * 24, 100);
        }
        img
    }

    #[test]
    fn dhash_basics() {
        let empty = GrayImage::new(0, 0);
        assert_eq!(dhash(&empty), 0);
        // Horizontal gradient getting darker to the right: left > right everywhere.
        let grad = GrayImage::from_fn(90, 80, |x, _| Luma([255 - (x as u8 * 2)]));
        assert_eq!(dhash(&grad), u64::MAX);
        // Gradient getting brighter: no bit set.
        let grad = GrayImage::from_fn(90, 80, |x, _| Luma([x as u8 * 2]));
        assert_eq!(dhash(&grad), 0);
        // Tiny images still hash deterministically.
        let tiny = GrayImage::from_fn(3, 2, |x, y| Luma([(x * 40 + y * 7) as u8]));
        assert_eq!(dhash(&tiny), dhash(&tiny.clone()));
    }

    #[test]
    fn dhash_is_stable_under_resize() {
        // Non-monotonic 9x8 block pattern.
        let img = GrayImage::from_fn(360, 320, |x, y| {
            Luma([(((x / 40) * 37 % 11) * 20 + (y / 40) * 3) as u8])
        });
        let small = image::imageops::thumbnail(&img, 180, 160);
        assert!(hamming(dhash(&img), dhash(&small)) <= 4);
    }

    #[test]
    fn hamming_counts_bits() {
        assert_eq!(hamming(0, 0), 0);
        assert_eq!(hamming(0b1011, 0b0001), 2);
        assert_eq!(hamming(u64::MAX, 0), 64);
    }

    #[test]
    fn first_frame_always_changed_then_identical_unchanged() {
        let mut d = ChangeDetector::new(0.08);
        let img = document();
        let (changed, score, hash) = d.evaluate(&img);
        assert!(changed);
        assert_eq!(score, 1.0);
        let (changed, score, hash2) = d.evaluate(&img);
        assert!(!changed);
        assert_eq!(score, 0.0);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn caret_blink_is_not_a_change() {
        let mut d = ChangeDetector::new(0.08);
        let base = document();
        d.evaluate(&base);
        let mut caret = base.clone();
        // 2x18 px caret at the end of a line.
        fill(&mut caret, 845, 38, 2, 18, [0, 0, 0, 255]);
        let (changed, score, _) = d.evaluate(&caret);
        assert!(!changed, "caret scored {score}");
        assert!(score < 0.02, "caret scored {score}");
    }

    #[test]
    fn new_line_of_text_is_a_change() {
        let mut d = ChangeDetector::new(0.08);
        let base = document();
        d.evaluate(&base);
        let mut more = base.clone();
        // A new 60-character line below the existing text (~480 px wide).
        text_line(&mut more, 40, 40 + 21 * 24, 60);
        let (changed, score, _) = d.evaluate(&more);
        assert!(changed, "new line scored {score}");
    }

    #[test]
    fn edited_line_is_a_change() {
        let mut d = ChangeDetector::new(0.08);
        let base = document();
        d.evaluate(&base);
        let mut edited = base.clone();
        // Rewrite one line with different text: clear it and draw a sparser
        // pattern (different words / spacing).
        fill(&mut edited, 40, 40 + 5 * 24, 800, 12, [250, 250, 250, 255]);
        for i in 0..100u32 {
            if i % 2 == 0 {
                fill(
                    &mut edited,
                    40 + i * 8,
                    40 + 5 * 24,
                    6,
                    12,
                    [30, 30, 30, 255],
                );
            }
        }
        let (changed, score, _) = d.evaluate(&edited);
        assert!(changed, "edited line scored {score}");
    }

    #[test]
    fn slide_change_scores_high() {
        let mut d = ChangeDetector::new(0.08);
        d.evaluate(&document());
        let mut slide = RgbaImage::from_pixel(W, H, Rgba([20, 40, 90, 255]));
        fill(&mut slide, 100, 100, 1080, 120, [240, 240, 240, 255]);
        let (changed, score, _) = d.evaluate(&slide);
        assert!(changed);
        assert!(score > 0.5, "{score}");
    }

    #[test]
    fn gradual_changes_accumulate_against_accepted_reference() {
        let mut d = ChangeDetector::new(0.08);
        let mut img = document();
        d.evaluate(&img);
        let mut changed_at = None;
        // Type one short word per step on a new line; individually tiny.
        for step in 0..20u32 {
            text_line(&mut img, 40 + step * 40, 40 + 21 * 24, 4);
            let (changed, _, _) = d.evaluate(&img);
            if changed {
                changed_at = Some(step);
                break;
            }
        }
        let step = changed_at.expect("accumulated typing must eventually count as a change");
        assert!(step >= 1, "a single short word should not trigger");
    }

    #[test]
    fn reset_and_force_accept() {
        let mut d = ChangeDetector::new(0.5);
        let img = document();
        d.evaluate(&img);
        assert!(!d.evaluate(&img).0);
        d.reset();
        assert!(d.evaluate(&img).0);

        let mut other = img.clone();
        text_line(&mut other, 40, 40 + 21 * 24, 30);
        let (score, _) = d.force_accept(&other);
        assert!(score < 0.5);
        // `other` is now the reference.
        let (changed, score, _) = d.evaluate(&other);
        assert!(!changed);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn threshold_is_sanitized() {
        assert_eq!(ChangeDetector::new(f32::NAN).threshold(), 0.08);
        assert_eq!(ChangeDetector::new(7.0).threshold(), 1.0);
        assert_eq!(ChangeDetector::new(-1.0).threshold(), 0.0);
    }

    #[test]
    fn tiny_and_empty_frames_do_not_panic() {
        let mut d = ChangeDetector::new(0.08);
        assert!(d.evaluate(&RgbaImage::new(0, 0)).0);
        let tiny = RgbaImage::from_pixel(3, 2, Rgba([10, 20, 30, 255]));
        assert!(d.evaluate(&tiny).0);
        assert!(!d.evaluate(&tiny).0);
    }
}
