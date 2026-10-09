//! Meeting-application detection (UX only).
//!
//! Lists running meeting applications (Zoom, Microsoft Teams, Slack, Discord,
//! Webex, ...) and browser windows whose title shows a web meeting (Google
//! Meet, Teams, Zoom, Webex, Jitsi, Whereby) so the user can pick the window
//! OpenMeet keeps focused. Detection only reads the process list (sysinfo) and
//! top-level window titles (xcap); nothing is injected into, hooked into or
//! read from other processes.
//!
//! Browsers expose only the *active* tab's title, so a meeting in a
//! background tab is not detected. Window titles are returned to the caller
//! (they are shown in the UI) and never logged.

use std::collections::{HashMap, HashSet};

use openmeet_core::ipc::MeetingAppInfo;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

use crate::error::guard_panics;
use crate::screen::rules::{is_browser, normalize_app};

/// A native meeting application recognized by its process name.
#[derive(Debug, Clone, Copy)]
struct NativeApp {
    slug: &'static str,
    name: &'static str,
    /// Normalized process names (see [`normalize_app`]) on any OS.
    processes: &'static [&'static str],
    /// Lowercase window-title fragments of an in-progress meeting / call.
    meeting_hints: &'static [&'static str],
}

const NATIVE_APPS: &[NativeApp] = &[
    NativeApp {
        slug: "zoom",
        name: "Zoom",
        processes: &["zoom", "zoom.us"],
        meeting_hints: &["zoom meeting", "zoom webinar"],
    },
    NativeApp {
        slug: "teams",
        name: "Microsoft Teams",
        processes: &[
            "ms-teams",
            "msteams",
            "teams",
            "microsoft teams",
            "microsoft teams (work or school)",
            "microsoft teams classic",
            "teams-for-linux",
        ],
        meeting_hints: &["meeting", "call with", "| call"],
    },
    NativeApp {
        slug: "slack",
        name: "Slack",
        processes: &["slack"],
        meeting_hints: &["huddle"],
    },
    NativeApp {
        slug: "discord",
        name: "Discord",
        processes: &["discord", "discordptb", "discordcanary"],
        meeting_hints: &[],
    },
    NativeApp {
        slug: "webex",
        name: "Webex",
        processes: &[
            "webex",
            "ciscocollabhost",
            "webexmta",
            "atmgr",
            "ciscowebexstart",
            "cisco webex meetings",
        ],
        meeting_hints: &["meeting"],
    },
    NativeApp {
        slug: "goto",
        name: "GoTo Meeting",
        processes: &["gotomeeting", "g2mcomm", "g2mstart", "goto"],
        meeting_hints: &["meeting", "session"],
    },
    NativeApp {
        slug: "ringcentral",
        name: "RingCentral",
        processes: &["ringcentral", "ringcentral meetings", "ringcentral video"],
        meeting_hints: &["meeting", "video"],
    },
    NativeApp {
        slug: "facetime",
        name: "FaceTime",
        processes: &["facetime"],
        meeting_hints: &[],
    },
];

/// A web meeting service recognized from a browser window title.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WebService {
    slug: &'static str,
    name: &'static str,
}

/// A process as seen by the detector (plain data, testable).
#[derive(Debug, Clone, PartialEq)]
struct ProcEntry {
    pid: u32,
    parent: Option<u32>,
    name: String,
}

/// A top-level window as seen by the detector (front-most first).
#[derive(Debug, Clone, PartialEq)]
struct WindowEntry {
    id: String,
    pid: u32,
    /// Application name reported by the window system (may be empty).
    app: String,
    title: String,
    minimized: bool,
}

/// Running meeting applications and browser windows showing a web meeting,
/// in-progress meetings first. OpenMeet's own process is never listed. Never
/// fails: when the process or window list is unavailable the corresponding
/// part is simply empty.
pub fn detect_meeting_apps() -> Vec<MeetingAppInfo> {
    let result = guard_panics("detecting meeting applications", || {
        let processes = list_processes();
        let windows = list_windows();
        let apps = detect(&processes, &windows, std::process::id());
        tracing::debug!(
            processes = processes.len(),
            windows = windows.len(),
            apps = apps.len(),
            "meeting applications detected"
        );
        Ok(apps)
    });
    result.unwrap_or_default()
}

fn list_processes() -> Vec<ProcEntry> {
    let mut system = System::new();
    system.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing());
    system
        .processes()
        .values()
        .map(|p| ProcEntry {
            pid: p.pid().as_u32(),
            parent: p.parent().map(|pid| pid.as_u32()),
            name: p.name().to_string_lossy().into_owned(),
        })
        .collect()
}

fn list_windows() -> Vec<WindowEntry> {
    #[cfg(windows)]
    let _dpi = crate::win::DpiAwareGuard::new();
    let windows = match xcap::Window::all() {
        Ok(w) => w,
        Err(e) => {
            tracing::debug!(error = %e, "window enumeration failed");
            return Vec::new();
        }
    };
    let mut out: Vec<(i64, WindowEntry)> = Vec::with_capacity(windows.len());
    let count = windows.len();
    for (index, w) in windows.iter().enumerate() {
        let (Ok(id), Ok(pid)) = (w.id(), w.pid()) else {
            continue;
        };
        let title = w.title().unwrap_or_default();
        if title.trim().is_empty() {
            continue;
        }
        // xcap enumerates Windows windows front-most first; elsewhere use z().
        let z = if cfg!(windows) {
            (count - index) as i64
        } else {
            i64::from(w.z().unwrap_or(0))
        };
        // The window-system app name is only needed when the process list is
        // unavailable; on Windows it costs a version-resource read, so it is
        // left empty there (the process name is used instead).
        let app = if cfg!(windows) {
            String::new()
        } else {
            w.app_name().unwrap_or_default()
        };
        out.push((
            z,
            WindowEntry {
                id: id.to_string(),
                pid,
                app,
                title,
                minimized: w.is_minimized().unwrap_or(false),
            },
        ));
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.into_iter().map(|(_, w)| w).collect()
}

/// Pure detection over snapshots. `windows` must be sorted front-most first.
fn detect(processes: &[ProcEntry], windows: &[WindowEntry], own_pid: u32) -> Vec<MeetingAppInfo> {
    let by_pid: HashMap<u32, &ProcEntry> = processes.iter().map(|p| (p.pid, p)).collect();
    let windows: Vec<&WindowEntry> = windows
        .iter()
        .filter(|w| w.pid != own_pid && !w.title.trim().is_empty())
        .collect();
    // (in_meeting, info), kept in discovery order.
    let mut found: Vec<(bool, MeetingAppInfo)> = Vec::new();

    for app in NATIVE_APPS {
        let pids: HashSet<u32> = processes
            .iter()
            .filter(|p| {
                p.pid != own_pid && app.processes.contains(&normalize_app(&p.name).as_str())
            })
            .map(|p| p.pid)
            .collect();
        // Without a process list, fall back to the window system's app name.
        let app_windows: Vec<&WindowEntry> = windows
            .iter()
            .copied()
            .filter(|w| {
                pids.contains(&w.pid)
                    || (!by_pid.contains_key(&w.pid)
                        && app.processes.contains(&normalize_app(&w.app).as_str()))
            })
            .collect();
        if pids.is_empty() && app_windows.is_empty() {
            continue;
        }
        let best = best_window(&app_windows, app.meeting_hints);
        let in_meeting = best.is_some_and(|w| has_hint(&w.title, app.meeting_hints));
        let pid = best.map(|w| w.pid).or_else(|| root_process(&pids, &by_pid));
        let Some(pid) = pid else { continue };
        let process_name = by_pid
            .get(&pid)
            .map(|p| p.name.clone())
            .or_else(|| best.map(|w| w.app.clone()))
            .unwrap_or_default();
        found.push((
            in_meeting,
            MeetingAppInfo {
                id: format!("{}:{pid}", app.slug),
                name: app.name.to_string(),
                process_name,
                pid,
                window_title: best.map(|w| w.title.trim().to_string()),
                window_id: best.map(|w| w.id.clone()),
            },
        ));
    }

    for w in &windows {
        let process_name = by_pid.get(&w.pid).map(|p| p.name.as_str());
        let browser_name = match process_name {
            Some(name) if is_browser(name) => name,
            _ if is_browser(&w.app) => w.app.as_str(),
            _ => continue,
        };
        let Some(service) = classify_browser_title(&w.title) else {
            continue;
        };
        found.push((
            true,
            MeetingAppInfo {
                id: format!("{}-web:{}", service.slug, w.id),
                name: format!("{} ({})", service.name, browser_label(browser_name)),
                process_name: process_name.unwrap_or(browser_name).to_string(),
                pid: w.pid,
                window_title: Some(w.title.trim().to_string()),
                window_id: Some(w.id.clone()),
            },
        ));
    }

    // In-progress meetings first; otherwise keep discovery order.
    found.sort_by_key(|(in_meeting, _)| !in_meeting);
    found.into_iter().map(|(_, info)| info).collect()
}

/// The window to focus for a native app: a meeting window if one exists,
/// else a non-minimized window, else any window (front-most first).
fn best_window<'a>(windows: &[&'a WindowEntry], hints: &[&str]) -> Option<&'a WindowEntry> {
    windows
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|(index, w)| {
            let score = u8::from(has_hint(&w.title, hints)) * 2 + u8::from(!w.minimized);
            // Earlier (front-most) wins ties.
            (score, std::cmp::Reverse(*index))
        })
        .map(|(_, w)| w)
}

/// The main process of a multi-process app: one whose parent is not part of
/// the same app (lowest pid on ties).
fn root_process(pids: &HashSet<u32>, by_pid: &HashMap<u32, &ProcEntry>) -> Option<u32> {
    let mut sorted: Vec<u32> = pids.iter().copied().collect();
    sorted.sort_unstable();
    sorted
        .iter()
        .copied()
        .find(|pid| {
            by_pid
                .get(pid)
                .and_then(|p| p.parent)
                .is_none_or(|parent| !pids.contains(&parent))
        })
        .or_else(|| sorted.first().copied())
}

fn has_hint(title: &str, hints: &[&str]) -> bool {
    let lower = title.to_lowercase();
    hints.iter().any(|h| lower.contains(h))
}

/// Recognizes a web meeting from a browser window (active tab) title.
fn classify_browser_title(title: &str) -> Option<WebService> {
    let t = title.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }
    const MEET: WebService = WebService {
        slug: "meet",
        name: "Google Meet",
    };
    const TEAMS: WebService = WebService {
        slug: "teams",
        name: "Microsoft Teams",
    };
    const ZOOM: WebService = WebService {
        slug: "zoom",
        name: "Zoom",
    };
    const WEBEX: WebService = WebService {
        slug: "webex",
        name: "Webex",
    };
    const JITSI: WebService = WebService {
        slug: "jitsi",
        name: "Jitsi Meet",
    };
    const WHEREBY: WebService = WebService {
        slug: "whereby",
        name: "Whereby",
    };
    // Google Meet tabs are titled "Meet - abc-defg-hij" / "Meet – …".
    let meet_prefix = ["meet - ", "meet – ", "meet — "]
        .iter()
        .any(|p| t.starts_with(p));
    if meet_prefix || t.contains("google meet") || t.contains("meet.google.com") {
        return Some(MEET);
    }
    if t.contains("microsoft teams")
        || t.contains("teams.microsoft.com")
        || t.contains("teams.live.com")
    {
        return Some(TEAMS);
    }
    if [
        "zoom meeting",
        "zoom webinar",
        "zoom workplace",
        "zoom.us",
        "zoom.com/wc",
    ]
    .iter()
    .any(|p| t.contains(p))
    {
        return Some(ZOOM);
    }
    if t.contains("webex") {
        return Some(WEBEX);
    }
    if t.contains("jitsi meet") || t.contains("meet.jit.si") {
        return Some(JITSI);
    }
    if t.contains("whereby") {
        return Some(WHEREBY);
    }
    None
}

/// Short browser name for labels ("Google Meet (Chrome)").
fn browser_label(app: &str) -> &'static str {
    let n = normalize_app(app);
    if n.contains("edge") || n == "msedge" {
        "Edge"
    } else if n.contains("chrome") {
        "Chrome"
    } else if n.contains("chromium") {
        "Chromium"
    } else if n.contains("firefox") || n == "librewolf" || n == "waterfox" {
        "Firefox"
    } else if n.contains("brave") {
        "Brave"
    } else if n.contains("opera") {
        "Opera"
    } else if n.contains("vivaldi") {
        "Vivaldi"
    } else if n.contains("safari") {
        "Safari"
    } else if n == "arc" {
        "Arc"
    } else if n.starts_with("zen") {
        "Zen"
    } else {
        "Browser"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWN: u32 = 1;

    fn proc_(pid: u32, parent: Option<u32>, name: &str) -> ProcEntry {
        ProcEntry {
            pid,
            parent,
            name: name.to_string(),
        }
    }

    fn win(id: u32, pid: u32, title: &str) -> WindowEntry {
        WindowEntry {
            id: id.to_string(),
            pid,
            app: String::new(),
            title: title.to_string(),
            minimized: false,
        }
    }

    #[test]
    fn web_meeting_titles() {
        let slug = |t: &str| classify_browser_title(t).map(|s| s.slug);
        assert_eq!(slug("Meet - abc-defg-hij - Google Chrome"), Some("meet"));
        assert_eq!(slug("Meet – xyz-abcd-efg"), Some("meet"));
        assert_eq!(slug("Google Meet"), Some("meet"));
        assert_eq!(
            slug("Meeting with Alex | Microsoft Teams - Microsoft\u{200b} Edge"),
            Some("teams")
        );
        assert_eq!(slug("Zoom Meeting - Google Chrome"), Some("zoom"));
        assert_eq!(
            slug("Launch Meeting - Zoom.us - Mozilla Firefox"),
            Some("zoom")
        );
        assert_eq!(slug("Cisco Webex Meetings"), Some("webex"));
        assert_eq!(slug("Jitsi Meet"), Some("jitsi"));
        assert_eq!(slug("Team sync | Whereby"), Some("whereby"));
        // Not meetings.
        assert_eq!(slug("How to zoom in on a photo - Google Search"), None);
        assert_eq!(slug("Meeting notes.docx - Word"), None);
        assert_eq!(slug("Meetup - Find events"), None);
        assert_eq!(slug("  "), None);
    }

    #[test]
    fn browser_labels() {
        assert_eq!(browser_label("chrome.exe"), "Chrome");
        assert_eq!(browser_label("Google Chrome"), "Chrome");
        assert_eq!(browser_label("msedge.exe"), "Edge");
        assert_eq!(browser_label("Microsoft Edge"), "Edge");
        assert_eq!(browser_label("firefox"), "Firefox");
        assert_eq!(browser_label("Brave Browser"), "Brave");
        assert_eq!(browser_label("Safari"), "Safari");
        assert_eq!(browser_label("Arc"), "Arc");
        assert_eq!(browser_label("something"), "Browser");
    }

    #[test]
    fn native_apps_with_meeting_windows_come_first() {
        let processes = vec![
            proc_(OWN, None, "openmeet.exe"),
            proc_(10, None, "Slack.exe"),
            proc_(11, Some(10), "Slack.exe"),
            proc_(20, None, "Zoom.exe"),
            proc_(30, None, "notepad.exe"),
        ];
        let windows = vec![
            win(100, 30, "notes.txt - Notepad"),
            win(200, 11, "general - Acme - Slack"),
            win(300, 20, "Zoom Workplace"),
            win(301, 20, "Zoom Meeting"),
        ];
        let apps = detect(&processes, &windows, OWN);
        assert_eq!(apps.len(), 2, "{apps:?}");
        // Zoom is in a meeting: listed first, with the meeting window.
        assert_eq!(apps[0].name, "Zoom");
        assert_eq!(apps[0].id, "zoom:20");
        assert_eq!(apps[0].pid, 20);
        assert_eq!(apps[0].process_name, "Zoom.exe");
        assert_eq!(apps[0].window_id.as_deref(), Some("301"));
        assert_eq!(apps[0].window_title.as_deref(), Some("Zoom Meeting"));
        // Slack: the window's process (renderer child) owns the window.
        assert_eq!(apps[1].name, "Slack");
        assert_eq!(apps[1].pid, 11);
        assert_eq!(apps[1].window_id.as_deref(), Some("200"));
    }

    #[test]
    fn running_app_without_windows_uses_the_root_process() {
        let processes = vec![
            proc_(50, Some(4), "Discord.exe"),
            proc_(51, Some(50), "Discord.exe"),
            proc_(49, Some(50), "Discord.exe"),
        ];
        let apps = detect(&processes, &[], OWN);
        assert_eq!(apps.len(), 1);
        assert_eq!(apps[0].name, "Discord");
        assert_eq!(apps[0].pid, 50);
        assert_eq!(apps[0].id, "discord:50");
        assert!(apps[0].window_id.is_none());
        assert!(apps[0].window_title.is_none());
    }

    #[test]
    fn minimized_windows_are_a_last_resort() {
        let processes = vec![proc_(60, None, "ms-teams.exe")];
        let mut minimized = win(600, 60, "Chat | Microsoft Teams");
        minimized.minimized = true;
        let visible = win(601, 60, "Calendar | Microsoft Teams");
        let apps = detect(&processes, &[minimized.clone(), visible], OWN);
        assert_eq!(apps[0].window_id.as_deref(), Some("601"));
        // A minimized meeting window still beats a visible non-meeting one.
        let mut meeting = win(602, 60, "Meeting with Sam | Microsoft Teams");
        meeting.minimized = true;
        let apps = detect(&processes, &[minimized, win(603, 60, "Chat"), meeting], OWN);
        assert_eq!(apps[0].window_id.as_deref(), Some("602"));
        assert_eq!(apps[0].name, "Microsoft Teams");
    }

    #[test]
    fn browser_meetings_are_detected_per_window() {
        let processes = vec![
            proc_(70, None, "chrome.exe"),
            proc_(80, None, "msedge.exe"),
            proc_(90, None, "Code.exe"),
        ];
        let windows = vec![
            win(700, 70, "Meet - abc-defg-hij - Google Chrome"),
            win(701, 70, "Inbox - Google Chrome"),
            win(800, 80, "Microsoft Teams - Microsoft\u{200b} Edge"),
            // Not a browser: never treated as a web meeting.
            win(900, 90, "meet - notes.md - Visual Studio Code"),
        ];
        let apps = detect(&processes, &windows, OWN);
        assert_eq!(apps.len(), 2, "{apps:?}");
        assert_eq!(apps[0].name, "Google Meet (Chrome)");
        assert_eq!(apps[0].id, "meet-web:700");
        assert_eq!(apps[0].process_name, "chrome.exe");
        assert_eq!(apps[0].window_id.as_deref(), Some("700"));
        assert_eq!(apps[1].name, "Microsoft Teams (Edge)");
        assert_eq!(apps[1].pid, 80);
    }

    #[test]
    fn own_process_is_never_listed() {
        let processes = vec![proc_(OWN, None, "Zoom.exe"), proc_(70, None, "chrome.exe")];
        let windows = vec![
            win(1, OWN, "Zoom Meeting"),
            win(2, OWN, "Meet - abc - Google Chrome"),
        ];
        assert!(detect(&processes, &windows, OWN).is_empty());
    }

    #[test]
    fn falls_back_to_window_app_names_without_a_process_list() {
        let mut zoom = win(5, 77, "Zoom Meeting");
        zoom.app = "zoom.us".to_string();
        let mut browser = win(6, 78, "Meet - abc-defg-hij");
        browser.app = "Safari".to_string();
        let apps = detect(&[], &[zoom, browser], OWN);
        assert_eq!(apps.len(), 2, "{apps:?}");
        assert_eq!(apps[0].name, "Zoom");
        assert_eq!(apps[0].process_name, "zoom.us");
        assert_eq!(apps[1].name, "Google Meet (Safari)");
        assert_eq!(apps[1].process_name, "Safari");
    }

    #[test]
    fn process_names_are_matched_exactly_after_normalization() {
        let processes = vec![
            proc_(1000, None, "ZoomIt64.exe"),
            proc_(1001, None, "teamspeak3.exe"),
            proc_(1002, None, "slack-helper"),
        ];
        assert!(detect(&processes, &[], OWN).is_empty());
        let processes = vec![proc_(1003, None, "Microsoft Teams (work or school)")];
        assert_eq!(detect(&processes, &[], OWN)[0].name, "Microsoft Teams");
    }
}
