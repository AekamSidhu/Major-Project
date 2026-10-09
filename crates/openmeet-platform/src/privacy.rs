//! Excluding OpenMeet's overlay from screen capture (spec §8).
//!
//! Only the documented OS mechanism is used:
//! * **Windows 10 2004+** – `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`,
//!   verified with `GetWindowDisplayAffinity`.
//! * **macOS** – `NSWindow.sharingType = .none`.
//! * **Linux / others** – unsupported: X11 and Wayland have no such flag.
//!
//! The protection is best effort by nature; [`CapturePrivacy::limitations`]
//! states honestly what it does not cover.

use raw_window_handle::RawWindowHandle;

/// Errors from changing the capture-exclusion state.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CapturePrivacyError {
    /// The OS (version) or window type has no capture-exclusion mechanism.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The mechanism exists but the call failed or did not take effect.
    #[error("failed: {0}")]
    Failed(String),
}

/// Capture exclusion for one window.
pub trait CapturePrivacy: Send + Sync {
    /// Excludes (or re-includes) the window from screen capture and verifies
    /// the result where the OS allows reading it back.
    fn set_excluded_from_capture(&self, enabled: bool) -> Result<(), CapturePrivacyError>;
    /// Whether this OS / window supports capture exclusion.
    fn is_supported(&self) -> bool;
    /// Verified current state when the OS allows reading it back.
    fn is_excluded(&self) -> Option<bool>;
    /// The OS mechanism used ("SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)", ...).
    fn mechanism(&self) -> &'static str;
    /// Honest description of what the protection does not cover.
    fn limitations(&self) -> &'static str;
}

/// Shared part of every limitations statement.
macro_rules! common_limitations {
    () => {
        "Capture exclusion only affects screen-capture tools that honor the operating-system flag. \
         Hardware capture devices, phone cameras, some remote-desktop tools and older capture APIs \
         are not affected."
    };
}

#[cfg_attr(not(windows), allow(dead_code))]
const WINDOWS_MECHANISM: &str = "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)";
#[cfg_attr(not(windows), allow(dead_code))]
const WINDOWS_LIMITATIONS: &str = concat!(
    common_limitations!(),
    " Requires Windows 10 version 2004 or later; on older versions the overlay is visible in \
     screen shares and recordings."
);
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const MACOS_MECHANISM: &str = "NSWindow.sharingType = .none";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const MACOS_LIMITATIONS: &str = concat!(
    common_limitations!(),
    " On macOS 15 and later, ScreenCaptureKit may not honor sharingType for all capture clients, \
     so the overlay can still appear in some screen shares and recordings."
);
const UNSUPPORTED_MECHANISM: &str = "none";
const UNSUPPORTED_LIMITATIONS: &str = concat!(
    "This operating system / window system provides no capture-exclusion flag (X11 and Wayland \
     have none): the overlay is visible in screen shares and recordings. ",
    common_limitations!()
);

/// Capture-privacy control for a native window. Windows:
/// `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` (Windows 10 2004+,
/// verified via `GetWindowDisplayAffinity`); macOS: `NSWindow.sharingType =
/// .none`; Linux/others: Unsupported.
///
/// Call it from the process that owns the window (Windows refuses windows
/// of other processes); on macOS the change is dispatched to the main thread.
pub fn for_window(handle: RawWindowHandle) -> Box<dyn CapturePrivacy> {
    match handle {
        #[cfg(windows)]
        RawWindowHandle::Win32(h) => {
            Box::new(windows_impl::WindowsCapturePrivacy::new(h.hwnd.get()))
        }
        #[cfg(target_os = "macos")]
        RawWindowHandle::AppKit(h) => Box::new(macos_impl::MacCapturePrivacy::new(
            h.ns_view.as_ptr() as usize,
        )),
        _ => Box::new(UnsupportedCapturePrivacy),
    }
}

/// (supported, mechanism, limitations) for this OS.
pub fn platform_support() -> (bool, &'static str, &'static str) {
    #[cfg(windows)]
    {
        (
            crate::win::os_build_number() >= crate::win::BUILD_EXCLUDE_FROM_CAPTURE,
            WINDOWS_MECHANISM,
            WINDOWS_LIMITATIONS,
        )
    }
    #[cfg(target_os = "macos")]
    {
        (true, MACOS_MECHANISM, MACOS_LIMITATIONS)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        (false, UNSUPPORTED_MECHANISM, UNSUPPORTED_LIMITATIONS)
    }
}

/// Used for window systems without a capture-exclusion flag.
#[derive(Debug, Clone, Copy, Default)]
struct UnsupportedCapturePrivacy;

impl CapturePrivacy for UnsupportedCapturePrivacy {
    fn set_excluded_from_capture(&self, _enabled: bool) -> Result<(), CapturePrivacyError> {
        Err(CapturePrivacyError::Unsupported(
            "This window system provides no way to exclude a window from screen capture"
                .to_string(),
        ))
    }

    fn is_supported(&self) -> bool {
        false
    }

    fn is_excluded(&self) -> Option<bool> {
        None
    }

    fn mechanism(&self) -> &'static str {
        UNSUPPORTED_MECHANISM
    }

    fn limitations(&self) -> &'static str {
        UNSUPPORTED_LIMITATIONS
    }
}

#[cfg(windows)]
mod windows_impl {
    use std::ffi::c_void;

    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetWindowDisplayAffinity, IsWindow, SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
        WDA_NONE,
    };

    use super::{CapturePrivacy, CapturePrivacyError, WINDOWS_LIMITATIONS, WINDOWS_MECHANISM};
    use crate::win::{BUILD_EXCLUDE_FROM_CAPTURE, os_build_number};

    /// Capture exclusion for one top-level window of this process.
    pub(super) struct WindowsCapturePrivacy {
        /// HWND value (handles are plain identifiers, valid from any thread).
        hwnd: isize,
    }

    impl WindowsCapturePrivacy {
        pub(super) fn new(hwnd: isize) -> Self {
            WindowsCapturePrivacy { hwnd }
        }

        fn hwnd(&self) -> HWND {
            HWND(self.hwnd as *mut c_void)
        }
    }

    impl CapturePrivacy for WindowsCapturePrivacy {
        fn set_excluded_from_capture(&self, enabled: bool) -> Result<(), CapturePrivacyError> {
            if !self.is_supported() {
                return Err(CapturePrivacyError::Unsupported(
                    "Capture exclusion requires Windows 10 version 2004 or later".to_string(),
                ));
            }
            let hwnd = self.hwnd();
            // SAFETY: IsWindow accepts any handle value.
            if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
                return Err(CapturePrivacyError::Failed(
                    "the window no longer exists".to_string(),
                ));
            }
            let affinity = if enabled {
                WDA_EXCLUDEFROMCAPTURE
            } else {
                WDA_NONE
            };
            // SAFETY: documented call on a valid top-level window handle.
            unsafe { SetWindowDisplayAffinity(hwnd, affinity) }.map_err(|e| {
                CapturePrivacyError::Failed(format!(
                    "SetWindowDisplayAffinity failed ({:#010x})",
                    e.code().0
                ))
            })?;
            match self.is_excluded() {
                Some(state) if state == enabled => {
                    tracing::info!(enabled, "capture exclusion applied");
                    Ok(())
                }
                Some(_) => Err(CapturePrivacyError::Failed(
                    "Windows did not apply the capture exclusion".to_string(),
                )),
                None => Err(CapturePrivacyError::Failed(
                    "the capture exclusion state could not be verified".to_string(),
                )),
            }
        }

        fn is_supported(&self) -> bool {
            os_build_number() >= BUILD_EXCLUDE_FROM_CAPTURE
        }

        fn is_excluded(&self) -> Option<bool> {
            let mut affinity = 0u32;
            // SAFETY: valid out pointer; invalid handles make the call fail.
            unsafe { GetWindowDisplayAffinity(self.hwnd(), &mut affinity) }.ok()?;
            Some(affinity == WDA_EXCLUDEFROMCAPTURE.0)
        }

        fn mechanism(&self) -> &'static str {
            WINDOWS_MECHANISM
        }

        fn limitations(&self) -> &'static str {
            WINDOWS_LIMITATIONS
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use objc2_app_kit::{NSView, NSWindowSharingType};

    use super::{CapturePrivacy, CapturePrivacyError, MACOS_LIMITATIONS, MACOS_MECHANISM};

    /// Capture exclusion through the `NSWindow` hosting an `NSView`.
    pub(super) struct MacCapturePrivacy {
        /// `NSView*` from the window handle; only dereferenced on the main
        /// thread while the window is alive (the caller owns the window).
        ns_view: usize,
    }

    impl MacCapturePrivacy {
        pub(super) fn new(ns_view: usize) -> Self {
            MacCapturePrivacy { ns_view }
        }

        /// Runs `f` with the view's window on the main thread.
        fn with_window<R: Send>(
            &self,
            f: impl FnOnce(&objc2_app_kit::NSWindow) -> R + Send,
        ) -> Option<R> {
            let ptr = self.ns_view;
            dispatch2::run_on_main(move |_mtm| {
                // SAFETY: `ptr` is the NSView of a live AppKit window handle
                // and we are on the main thread, where AppKit objects may be
                // used.
                let view: &NSView = unsafe { &*(ptr as *const NSView) };
                view.window().map(|w| f(&w))
            })
        }
    }

    impl CapturePrivacy for MacCapturePrivacy {
        fn set_excluded_from_capture(&self, enabled: bool) -> Result<(), CapturePrivacyError> {
            let wanted = if enabled {
                NSWindowSharingType::None
            } else {
                NSWindowSharingType::ReadOnly
            };
            let applied = self
                .with_window(move |w| {
                    w.setSharingType(wanted);
                    w.sharingType() == wanted
                })
                .ok_or_else(|| {
                    CapturePrivacyError::Failed("the view is not attached to a window".to_string())
                })?;
            if applied {
                tracing::info!(enabled, "capture exclusion applied");
                Ok(())
            } else {
                Err(CapturePrivacyError::Failed(
                    "macOS did not apply the window sharing type".to_string(),
                ))
            }
        }

        fn is_supported(&self) -> bool {
            true
        }

        fn is_excluded(&self) -> Option<bool> {
            self.with_window(|w| w.sharingType() == NSWindowSharingType::None)
        }

        fn mechanism(&self) -> &'static str {
            MACOS_MECHANISM
        }

        fn limitations(&self) -> &'static str {
            MACOS_LIMITATIONS
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limitations_are_honest() {
        for text in [
            WINDOWS_LIMITATIONS,
            MACOS_LIMITATIONS,
            UNSUPPORTED_LIMITATIONS,
        ] {
            assert!(text.contains("honor the operating-system flag"), "{text}");
            assert!(text.contains("phone cameras"), "{text}");
            assert!(text.contains("remote-desktop"), "{text}");
            assert!(text.contains("older capture APIs"), "{text}");
        }
        assert!(MACOS_LIMITATIONS.contains("ScreenCaptureKit"));
        assert!(WINDOWS_LIMITATIONS.contains("2004"));
    }

    #[test]
    fn unsupported_handles_report_unsupported() {
        let handle = RawWindowHandle::Xlib(raw_window_handle::XlibWindowHandle::new(42));
        let privacy = for_window(handle);
        assert!(!privacy.is_supported());
        assert_eq!(privacy.is_excluded(), None);
        assert_eq!(privacy.mechanism(), "none");
        assert!(matches!(
            privacy.set_excluded_from_capture(true),
            Err(CapturePrivacyError::Unsupported(_))
        ));
    }

    #[test]
    fn platform_support_matches_os() {
        let (supported, mechanism, limitations) = platform_support();
        assert!(!limitations.is_empty());
        #[cfg(windows)]
        {
            assert_eq!(mechanism, WINDOWS_MECHANISM);
            assert_eq!(
                supported,
                crate::win::os_build_number() >= crate::win::BUILD_EXCLUDE_FROM_CAPTURE
            );
        }
        #[cfg(target_os = "macos")]
        {
            assert!(supported);
            assert_eq!(mechanism, MACOS_MECHANISM);
        }
        #[cfg(not(any(windows, target_os = "macos")))]
        {
            assert!(!supported);
            assert_eq!(mechanism, "none");
        }
    }

    #[cfg(windows)]
    #[test]
    fn invalid_window_fails_cleanly() {
        let handle = raw_window_handle::Win32WindowHandle::new(
            std::num::NonZeroIsize::new(0x7fff_fff0).unwrap(),
        );
        let privacy = for_window(RawWindowHandle::Win32(handle));
        assert_eq!(privacy.mechanism(), WINDOWS_MECHANISM);
        assert_eq!(privacy.is_excluded(), None);
        if privacy.is_supported() {
            assert!(matches!(
                privacy.set_excluded_from_capture(true),
                Err(CapturePrivacyError::Failed(_))
            ));
        }
    }
}
