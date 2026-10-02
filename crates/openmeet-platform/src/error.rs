//! Error type shared by every platform module.

use openmeet_core::CoreError;

/// Errors returned by OS integration code.
///
/// Messages are user-facing: they describe the device / OS condition and never
/// contain transcript text, screen contents or secrets.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlatformError {
    /// The requested device, display or window does not exist (any more).
    #[error("not found: {0}")]
    DeviceNotFound(String),
    /// The device exists but cannot be used right now (busy, disconnected,
    /// audio service not running, ...).
    #[error("unavailable: {0}")]
    DeviceUnavailable(String),
    /// The operating system denied access (microphone / screen recording
    /// permission).
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    /// The capability does not exist on this OS / version / configuration.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A capture operation failed.
    #[error("capture failed: {0}")]
    Capture(String),
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

impl From<PlatformError> for CoreError {
    fn from(e: PlatformError) -> Self {
        match e {
            PlatformError::PermissionDenied(m) => CoreError::PermissionDenied(m),
            PlatformError::Unsupported(m) => CoreError::Unsupported(m),
            PlatformError::DeviceNotFound(m) => CoreError::NotFound(m),
            other => CoreError::Platform(other.to_string()),
        }
    }
}

/// Runs `f`, converting a panic inside third-party OS bindings into an error
/// instead of tearing down the calling thread. Our own code does not panic;
/// this guards against `expect()`s inside backend crates (e.g. when a device
/// disappears mid-query).
pub(crate) fn guard_panics<T>(
    what: &str,
    f: impl FnOnce() -> Result<T, PlatformError>,
) -> Result<T, PlatformError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            tracing::error!(operation = what, "OS backend panicked; recovered");
            Err(PlatformError::Other(format!(
                "{what} failed unexpectedly in the OS backend"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_to_core_errors() {
        assert_eq!(
            CoreError::from(PlatformError::PermissionDenied("mic".into())),
            CoreError::PermissionDenied("mic".into())
        );
        assert_eq!(
            CoreError::from(PlatformError::Unsupported("x".into())),
            CoreError::Unsupported("x".into())
        );
        assert_eq!(
            CoreError::from(PlatformError::DeviceNotFound("Display 3".into())),
            CoreError::NotFound("Display 3".into())
        );
        assert_eq!(
            CoreError::from(PlatformError::Capture("boom".into())),
            CoreError::Platform("capture failed: boom".into())
        );
    }

    #[test]
    fn guard_panics_converts_panics() {
        let r: Result<(), PlatformError> = guard_panics("probe", || panic!("backend bug"));
        assert!(matches!(r, Err(PlatformError::Other(m)) if m.contains("probe")));
        let ok = guard_panics("probe", || Ok(5));
        assert_eq!(ok, Ok(5));
    }
}
