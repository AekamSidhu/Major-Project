use std::time::{SystemTime, UNIX_EPOCH};

/// Opaque identifier. Time-ordered (UUIDv7, simple hex form) so that ids sort by
/// creation time in SQLite and in the UI.
pub type Id = String;

/// Generates a new time-ordered identifier.
pub fn new_id() -> Id {
    uuid::Uuid::now_v7().simple().to_string()
}

/// Current wall-clock time in Unix milliseconds.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_ordered() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a <= b, "UUIDv7 ids must be monotonic: {a} > {b}");
    }

    #[test]
    fn now_is_after_2024() {
        assert!(now_ms() > 1_704_067_200_000);
    }
}
