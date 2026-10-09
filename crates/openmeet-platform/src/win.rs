//! Small Windows helpers shared by several modules.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Wdk::System::SystemServices::RtlGetVersion;
use windows::Win32::Foundation::{GetLastError, HWND, SetLastError, WIN32_ERROR};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::Win32::System::SystemInformation::OSVERSIONINFOW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GWL_EXSTYLE, GetWindowLongPtrW, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
    SWP_NOZORDER, SetWindowLongPtrW, SetWindowPos,
};
use windows::core::{s, w};

/// First Windows build with `WDA_EXCLUDEFROMCAPTURE` (Windows 10 version 2004).
pub(crate) const BUILD_EXCLUDE_FROM_CAPTURE: u32 = 19041;

/// The real OS build number (`RtlGetVersion` is not subject to the manifest
/// based version lie of `GetVersionEx`). 0 when it cannot be determined.
pub(crate) fn os_build_number() -> u32 {
    static BUILD: OnceLock<u32> = OnceLock::new();
    *BUILD.get_or_init(|| {
        let mut info = OSVERSIONINFOW {
            dwOSVersionInfoSize: std::mem::size_of::<OSVERSIONINFOW>() as u32,
            ..Default::default()
        };
        // SAFETY: `info` is a properly sized, initialized OSVERSIONINFOW.
        let status = unsafe { RtlGetVersion(&mut info) };
        if status.is_ok() {
            info.dwBuildNumber
        } else {
            0
        }
    })
}

/// Converts a window id string (decimal, as produced by xcap / this crate) to
/// an `HWND`. Window handles are 32-bit significant; per the Windows
/// interoperability rules a 32-bit handle value is sign-extended to 64 bits.
pub(crate) fn hwnd_from_id(id: &str) -> Option<HWND> {
    let id = id.trim();
    let value: isize = if let Ok(v) = id.parse::<u32>() {
        v as i32 as isize
    } else {
        id.parse::<i64>().ok()? as isize
    };
    if value == 0 {
        return None;
    }
    Some(HWND(value as *mut c_void))
}

/// The id string for an `HWND` (same representation as `xcap::Window::id`).
pub(crate) fn hwnd_to_id(hwnd: HWND) -> String {
    (hwnd.0 as usize as u32).to_string()
}

/// The extended window style (`GWL_EXSTYLE`) of a window, 0 on failure.
pub(crate) fn window_ex_style(hwnd: HWND) -> u32 {
    // SAFETY: reading a window long has no preconditions; invalid handles
    // simply return 0.
    unsafe { GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32 }
}

/// Replaces the extended window style and applies it (frame change).
pub(crate) fn set_window_ex_style(hwnd: HWND, style: u32) -> windows::core::Result<()> {
    // SAFETY: documented calls on a caller-provided window handle. The last
    // error is cleared first because 0 is also a valid previous style value.
    unsafe {
        SetLastError(WIN32_ERROR(0));
        let previous = SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style as i32 as _);
        if previous == 0 {
            let err = GetLastError();
            if err.0 != 0 {
                return Err(windows::core::Error::from_hresult(err.to_hresult()));
            }
        }
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )
    }
}

type SetThreadDpiAwarenessContextFn =
    unsafe extern "system" fn(DPI_AWARENESS_CONTEXT) -> DPI_AWARENESS_CONTEXT;

fn set_thread_dpi_awareness_fn() -> Option<SetThreadDpiAwarenessContextFn> {
    static FUNC: OnceLock<Option<usize>> = OnceLock::new();
    let addr = *FUNC.get_or_init(|| {
        // SAFETY: user32.dll is always loaded in GUI-capable processes; looking
        // up an export by name has no side effects.
        unsafe {
            let module = GetModuleHandleW(w!("user32.dll")).ok()?;
            GetProcAddress(module, s!("SetThreadDpiAwarenessContext")).map(|f| f as usize)
        }
    });
    // SAFETY: the address was resolved from user32's export of exactly this
    // name, whose documented signature matches `SetThreadDpiAwarenessContextFn`.
    addr.map(|a| unsafe { std::mem::transmute::<usize, SetThreadDpiAwarenessContextFn>(a) })
}

/// Makes the current thread per-monitor DPI aware (v2) so window rectangles,
/// monitor sizes and GDI captures all use physical pixels, then restores the
/// previous awareness on drop. A no-op before Windows 10 1607.
pub(crate) struct DpiAwareGuard {
    previous: Option<DPI_AWARENESS_CONTEXT>,
}

impl DpiAwareGuard {
    pub(crate) fn new() -> Self {
        let previous = set_thread_dpi_awareness_fn().and_then(|f| {
            // SAFETY: documented API, valid predefined context value.
            let prev = unsafe { f(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
            (!prev.0.is_null()).then_some(prev)
        });
        DpiAwareGuard { previous }
    }
}

impl Drop for DpiAwareGuard {
    fn drop(&mut self) {
        if let (Some(prev), Some(f)) = (self.previous, set_thread_dpi_awareness_fn()) {
            // SAFETY: restoring the context value previously returned by the OS.
            unsafe {
                f(prev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hwnd_id_roundtrip() {
        let hwnd = hwnd_from_id("1312334").unwrap();
        assert_eq!(hwnd.0 as isize, 1_312_334);
        assert_eq!(hwnd_to_id(hwnd), "1312334");
        assert!(hwnd_from_id("0").is_none());
        assert!(hwnd_from_id("abc").is_none());
        // 32-bit values with the top bit set are sign-extended.
        let high = hwnd_from_id("4294967294").unwrap();
        assert_eq!(high.0 as isize, -2);
        assert_eq!(hwnd_to_id(high), "4294967294");
    }

    #[test]
    fn build_number_is_known_on_windows() {
        assert!(os_build_number() >= 7600, "{}", os_build_number());
    }

    #[test]
    fn dpi_guard_is_reentrant() {
        let a = DpiAwareGuard::new();
        let b = DpiAwareGuard::new();
        drop(b);
        drop(a);
    }
}
