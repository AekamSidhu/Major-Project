//! Integer rectangle math for privacy masking (pure).

/// Axis-aligned rectangle `[x0, x1) x [y0, y1)` in a common coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IRect {
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}

impl IRect {
    /// Rectangle from origin + size.
    pub fn from_xywh(x: i64, y: i64, w: i64, h: i64) -> IRect {
        IRect {
            x0: x,
            y0: y,
            x1: x + w.max(0),
            y1: y + h.max(0),
        }
    }

    pub fn width(&self) -> i64 {
        (self.x1 - self.x0).max(0)
    }

    pub fn height(&self) -> i64 {
        (self.y1 - self.y0).max(0)
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    pub fn area(&self) -> i64 {
        self.width() * self.height()
    }

    pub fn intersect(&self, other: &IRect) -> Option<IRect> {
        let r = IRect {
            x0: self.x0.max(other.x0),
            y0: self.y0.max(other.y0),
            x1: self.x1.min(other.x1),
            y1: self.y1.min(other.y1),
        };
        (!r.is_empty()).then_some(r)
    }

    pub fn intersects(&self, other: &IRect) -> bool {
        self.intersect(other).is_some()
    }

    /// `self` minus `cut`: up to four disjoint rectangles.
    pub fn subtract(&self, cut: &IRect) -> Vec<IRect> {
        let Some(c) = self.intersect(cut) else {
            return vec![*self];
        };
        let mut out = Vec::with_capacity(4);
        // Top band.
        if c.y0 > self.y0 {
            out.push(IRect { x0: self.x0, y0: self.y0, x1: self.x1, y1: c.y0 });
        }
        // Bottom band.
        if c.y1 < self.y1 {
            out.push(IRect { x0: self.x0, y0: c.y1, x1: self.x1, y1: self.y1 });
        }
        // Left and right parts of the middle band.
        if c.x0 > self.x0 {
            out.push(IRect { x0: self.x0, y0: c.y0, x1: c.x0, y1: c.y1 });
        }
        if c.x1 < self.x1 {
            out.push(IRect { x0: c.x1, y0: c.y0, x1: self.x1, y1: c.y1 });
        }
        out
    }
}

/// Parts of `target` not covered by any of `occluders` (disjoint rectangles).
pub(crate) fn visible_parts(target: IRect, occluders: &[IRect]) -> Vec<IRect> {
    let mut parts = if target.is_empty() { Vec::new() } else { vec![target] };
    for occ in occluders {
        if parts.is_empty() {
            break;
        }
        parts = parts.iter().flat_map(|p| p.subtract(occ)).collect();
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: i64, y: i64, w: i64, h: i64) -> IRect {
        IRect::from_xywh(x, y, w, h)
    }

    fn total_area(parts: &[IRect]) -> i64 {
        parts.iter().map(IRect::area).sum()
    }

    #[test]
    fn basics() {
        let a = r(0, 0, 10, 10);
        assert_eq!(a.area(), 100);
        assert!(!a.is_empty());
        assert!(r(5, 5, 0, 3).is_empty());
        assert_eq!(r(0, 0, -5, 3).width(), 0);
        assert_eq!(a.intersect(&r(5, 5, 10, 10)), Some(r(5, 5, 5, 5)));
        assert_eq!(a.intersect(&r(10, 0, 5, 5)), None);
        assert!(!a.intersects(&r(20, 20, 1, 1)));
    }

    #[test]
    fn subtract_cases() {
        let a = r(0, 0, 10, 10);
        // No overlap.
        assert_eq!(a.subtract(&r(20, 20, 5, 5)), vec![a]);
        // Full cover.
        assert!(a.subtract(&r(-1, -1, 20, 20)).is_empty());
        // Hole in the middle -> 4 parts.
        let parts = a.subtract(&r(3, 3, 4, 4));
        assert_eq!(parts.len(), 4);
        assert_eq!(total_area(&parts), 100 - 16);
        // Corner cut.
        let parts = a.subtract(&r(5, 5, 10, 10));
        assert_eq!(total_area(&parts), 75);
        // Parts are disjoint.
        for (i, p) in parts.iter().enumerate() {
            for q in parts.iter().skip(i + 1) {
                assert!(!p.intersects(q));
            }
        }
    }

    #[test]
    fn visible_parts_of_partially_covered_window() {
        let sensitive = r(100, 100, 400, 300);
        // A window above covers the left half; another covers the bottom strip.
        let parts = visible_parts(sensitive, &[r(0, 0, 300, 1000), r(0, 350, 2000, 100)]);
        assert_eq!(total_area(&parts), 200 * 250);
        for p in &parts {
            assert!(p.x0 >= 300 && p.y1 <= 350);
        }
        // Fully covered -> nothing visible.
        assert!(visible_parts(sensitive, &[r(0, 0, 1920, 1080)]).is_empty());
        // No occluders -> whole window.
        assert_eq!(visible_parts(sensitive, &[]), vec![sensitive]);
        // Empty target.
        assert!(visible_parts(r(0, 0, 0, 0), &[]).is_empty());
    }
}
