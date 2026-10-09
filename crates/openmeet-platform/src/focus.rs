//! Keeping the meeting application focused (spec §7).
//!
//! Window ids are strings:
//! * **Windows** – the `HWND` value as a decimal string (the same value
//!   xcap uses as a window id).
//! * **macOS** – the `CGWindowID` (xcap window id). Focusing activates the
//!   owning application (macOS has no public API to raise another
//!   application's individual window).
//! * **Linux** – the X11 window id where available; changing focus is not
//!   supported (Wayland forbids it and X11 has no portable request).
//!
//! Only documented, focus-related APIs are used: no hooks, no input
//! injection.

use raw_window_handle::RawWindowHandle;

use crate::PlatformError;

/// The current foreground (focused) top-level window, if any.
pub fn foreground_window_id() -> Option<String> {
    imp::foreground_window_id()
}

/// Brings the window `id` back to the foreground (e.g. after the overlay
/// was clicked). Returns whether the OS accepted the request.
pub fn restore_foreground(id: &str) -> bool {
    match imp::focus_window(id) {
        Ok(()) => true,
        Err(e) => {
            tracing::debug!(error = %e, "could not restore the foreground window");
            false
        }
    }
}

/// Windows: toggles `WS_EX_NOACTIVATE` so clicks on the overlay do not steal
/// focus from the meeting application. Other platforms: Unsupported (macOS
/// uses non-activating panels configured by the window layer).
pub fn set_no_activate(handle: RawWindowHandle, enabled: bool) -> Result<(), PlatformError> {
    imp::set_no_activate(handle, enabled)
}

/// Brings the window `id` (restored if minimized) to the foreground.
pub fn focus_window(id: &str) -> Result<(), PlatformError> {
    imp::focus_window(id)
}

#[cfg(windows)]
mod imp {
    use raw_window_handle::RawWindowHandle;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, IsIconic, IsWindow, SW_RESTORE, SetForegroundWindow, ShowWindow,
        WS_EX_NOACTIVATE,
    };

    use crate::PlatformError;
    use crate::win::{hwnd_from_id, hwnd_to_id, set_window_ex_style, window_ex_style};

    pub(super) fn foreground_window_id() -> Option<String> {
        // SAFETY: no preconditions.
        let hwnd = unsafe { GetForegroundWindow() };
        (!hwnd.0.is_null()).then(|| hwnd_to_id(hwnd))
    }

    pub(super) fn focus_window(id: &str) -> Result<(), PlatformError> {
        let hwnd = hwnd_from_id(id)
            .ok_or_else(|| PlatformError::DeviceNotFound("Invalid window id".to_string()))?;
        // SAFETY: IsWindow accepts any handle value.
        if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
            return Err(PlatformError::DeviceNotFound(
                "The window is no longer open".to_string(),
            ));
        }
        // SAFETY: documented calls on a valid top-level window handle.
        unsafe {
            if IsIconic(hwnd).as_bool() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            }
            if SetForegroundWindow(hwnd).as_bool() {
                return Ok(());
            }
        }
        // Windows only lets the foreground process (or one that received the
        // last input) change the foreground window.
        Err(PlatformError::Other(
            "Windows did not allow OpenMeet to bring the window to the front; click it to focus it"
                .to_string(),
        ))
    }

    pub(super) fn set_no_activate(
        handle: RawWindowHandle,
        enabled: bool,
    ) -> Result<(), PlatformError> {
        let RawWindowHandle::Win32(h) = handle else {
            return Err(PlatformError::Unsupported(
                "not a Win32 window handle".to_string(),
            ));
        };
        let hwnd = HWND(h.hwnd.get() as *mut std::ffi::c_void);
        // SAFETY: IsWindow accepts any handle value.
        if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
            return Err(PlatformError::DeviceNotFound(
                "The window no longer exists".to_string(),
            ));
        }
        let style = window_ex_style(hwnd);
        let wanted = if enabled {
            style | WS_EX_NOACTIVATE.0
        } else {
            style & !WS_EX_NOACTIVATE.0
        };
        if wanted == style {
            return Ok(());
        }
        set_window_ex_style(hwnd, wanted).map_err(|e| {
            PlatformError::Other(format!(
                "could not change the window style ({:#010x})",
                e.code().0
            ))
        })?;
        if (window_ex_style(hwnd) & WS_EX_NOACTIVATE.0 != 0) != enabled {
            return Err(PlatformError::Other(
                "Windows did not apply the window style".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2_app_kit::{NSApplicationActivationOptions, NSRunningApplication, NSWorkspace};
    use raw_window_handle::RawWindowHandle;

    use crate::PlatformError;

    pub(super) fn foreground_window_id() -> Option<String> {
        let app = NSWorkspace::sharedWorkspace().frontmostApplication()?;
        let pid = u32::try_from(app.processIdentifier()).ok()?;
        // xcap lists windows front-most first.
        let windows = xcap::Window::all().ok()?;
        windows
            .iter()
            .filter(|w| w.pid().ok() == Some(pid))
            .find(|w| !w.is_minimized().unwrap_or(false))
            .and_then(|w| w.id().ok())
            .map(|id| id.to_string())
    }

    pub(super) fn focus_window(id: &str) -> Result<(), PlatformError> {
        let wanted: u32 = id
            .trim()
            .parse()
            .map_err(|_| PlatformError::DeviceNotFound("Invalid window id".to_string()))?;
        let windows = xcap::Window::all()
            .map_err(|e| PlatformError::Other(format!("listing windows failed: {e}")))?;
        let pid = windows
            .iter()
            .find(|w| w.id().ok() == Some(wanted))
            .and_then(|w| w.pid().ok())
            .ok_or_else(|| {
                PlatformError::DeviceNotFound("The window is no longer open".to_string())
            })?;
        let pid = i32::try_from(pid)
            .map_err(|_| PlatformError::DeviceNotFound("Invalid process id".to_string()))?;
        let app = NSRunningApplication::runningApplicationWithProcessIdentifier(pid).ok_or_else(
            || PlatformError::DeviceNotFound("The application is no longer running".to_string()),
        )?;
        if app.isHidden() {
            app.unhide();
        }
        if app.activateWithOptions(NSApplicationActivationOptions::empty()) {
            Ok(())
        } else {
            Err(PlatformError::Other(
                "macOS did not allow OpenMeet to activate the application".to_string(),
            ))
        }
    }

    pub(super) fn set_no_activate(
        _handle: RawWindowHandle,
        _enabled: bool,
    ) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported(
            "On macOS the overlay avoids taking focus through a non-activating panel".to_string(),
        ))
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod imp {
    use raw_window_handle::RawWindowHandle;

    use crate::PlatformError;

    pub(super) fn foreground_window_id() -> Option<String> {
        // Best effort: X11 reports the focused window; Wayland does not.
        let windows = xcap::Window::all().ok()?;
        windows
            .iter()
            .find(|w| w.is_focused().unwrap_or(false))
            .and_then(|w| w.id().ok())
            .map(|id| id.to_string())
    }

    pub(super) fn focus_window(_id: &str) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported(
            "Bringing another application's window to the front is not supported on this desktop"
                .to_string(),
        ))
    }

    pub(super) fn set_no_activate(
        _handle: RawWindowHandle,
        _enabled: bool,
    ) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported(
            "Non-activating windows are not supported on this desktop".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_ids_are_rejected() {
        assert!(focus_window("not-a-window").is_err());
        assert!(!restore_foreground(""));
        if cfg!(windows) {
            assert!(matches!(
                focus_window("abc"),
                Err(PlatformError::DeviceNotFound(_))
            ));
            // A handle value that is not a window.
            assert!(matches!(
                focus_window("2147483632"),
                Err(PlatformError::DeviceNotFound(_))
            ));
        }
    }

    #[test]
    fn foreground_id_is_numeric_when_present() {
        if let Some(id) = foreground_window_id() {
            assert!(id.parse::<u64>().is_ok(), "{id}");
        }
    }

    #[test]
    fn no_activate_rejects_foreign_handles() {
        let handle = RawWindowHandle::Xlib(raw_window_handle::XlibWindowHandle::new(7));
        assert!(set_no_activate(handle, true).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn no_activate_on_missing_window() {
        let handle = raw_window_handle::Win32WindowHandle::new(
            std::num::NonZeroIsize::new(0x7fff_fff0).unwrap(),
        );
        assert!(matches!(
            set_no_activate(RawWindowHandle::Win32(handle), true),
            Err(PlatformError::DeviceNotFound(_))
        ));
    }
}
