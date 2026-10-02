//! Small, dependency-free time formatting helpers (UTC only).

/// Meeting-relative milliseconds as `mm:ss` (or `h:mm:ss` past one hour).
pub(crate) fn clock(ms: i64) -> String {
    let total = ms.max(0) / 1000;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Human duration such as `42 min`, `1 h 05 min` or `35 s`.
pub(crate) fn duration(ms: i64) -> String {
    let total = ms.max(0) / 1000;
    if total < 60 {
        return format!("{total} s");
    }
    let minutes = total / 60;
    if minutes < 60 {
        format!("{minutes} min")
    } else {
        format!("{} h {:02} min", minutes / 60, minutes % 60)
    }
}

/// Converts days since 1970-01-01 to a (year, month, day) civil date
/// (proleptic Gregorian calendar; Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Unix milliseconds as `YYYY-MM-DD HH:MM UTC`.
pub(crate) fn utc_datetime(unix_ms: i64) -> String {
    let secs = unix_ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, mo, d) = civil_from_days(days);
    format!(
        "{y:04}-{mo:02}-{d:02} {:02}:{:02} UTC",
        rem / 3600,
        (rem % 3600) / 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks() {
        assert_eq!(clock(0), "00:00");
        assert_eq!(clock(61_500), "01:01");
        assert_eq!(clock(3_725_000), "1:02:05");
        assert_eq!(clock(-5), "00:00");
    }

    #[test]
    fn durations() {
        assert_eq!(duration(35_000), "35 s");
        assert_eq!(duration(42 * 60_000 + 10), "42 min");
        assert_eq!(duration(65 * 60_000), "1 h 05 min");
    }

    #[test]
    fn dates() {
        assert_eq!(utc_datetime(0), "1970-01-01 00:00 UTC");
        // 2024-02-29 12:34:00 UTC
        assert_eq!(utc_datetime(1_709_210_040_000), "2024-02-29 12:34 UTC");
        // 2026-10-02 15:00:00 UTC
        assert_eq!(utc_datetime(1_790_953_200_000), "2026-10-02 15:00 UTC");
    }
}
