//! Typed payloads exchanged between the Rust backend and the React frontend
//! (Tauri command arguments / return values). Exported to TypeScript.

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::ids::Id;
use crate::settings::ShortcutAction;
use crate::types::*;

// ---------------------------------------------------------------------------
// Geometry & displays
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// Parses "x,y,width,height" (privacy region rules).
    pub fn parse_csv(s: &str) -> Option<Rect> {
        let parts: Vec<&str> = s.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return None;
        }
        Some(Rect {
            x: parts[0].parse().ok()?,
            y: parts[1].parse().ok()?,
            width: parts[2].parse().ok()?,
            height: parts[3].parse().ok()?,
        })
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.x
            && p.y >= self.y
            && (p.x as i64) < self.x as i64 + self.width as i64
            && (p.y as i64) < self.y as i64 + self.height as i64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DisplayOrientation {
    Landscape,
    Portrait,
}

/// A connected display as seen by the overlay window manager.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayInfo {
    /// Stable key used to remember per-display overlay positions.
    pub key: String,
    pub name: String,
    /// Physical pixels, virtual-desktop coordinates.
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    /// Usable area excluding taskbar / dock / menu bar (physical pixels).
    pub work_area: Rect,
    pub scale_factor: f64,
    pub is_primary: bool,
    pub orientation: DisplayOrientation,
}

// ---------------------------------------------------------------------------
// Audio & screen sources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AudioDeviceInfo {
    /// Identifier passed back when selecting the device (the device name on
    /// most hosts).
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub channels: u16,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct SystemAudioSupport {
    pub supported: bool,
    /// e.g. "WASAPI loopback", "PulseAudio monitor source".
    pub mechanism: String,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AudioDevices {
    pub inputs: Vec<AudioDeviceInfo>,
    /// Output devices usable for system-audio capture (loopback / monitor).
    pub outputs: Vec<AudioDeviceInfo>,
    pub system_audio: SystemAudioSupport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ScreenSourceKind {
    /// Entire display / specific monitor.
    Display,
    /// A specific window.
    Window,
    /// The front-most window of a specific application.
    Application,
    /// A rectangle of a display.
    Region,
}

/// What the user explicitly selected for screen context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ScreenSourceSpec {
    pub kind: ScreenSourceKind,
    /// Display id (Display/Region), window id (Window) or application name
    /// (Application).
    pub id: String,
    pub label: String,
    /// Region within the display, physical pixels relative to the display.
    pub region: Option<Rect>,
}

/// A selectable screen source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ScreenSourceInfo {
    pub kind: ScreenSourceKind,
    pub id: String,
    pub label: String,
    pub application: Option<String>,
    pub title: Option<String>,
    pub width: u32,
    pub height: u32,
    pub is_primary: bool,
    /// True when a privacy rule excludes this source.
    pub excluded_by_rule: bool,
}

// ---------------------------------------------------------------------------
// Meetings & assistant actions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct StartMeetingRequest {
    pub mode: Mode,
    pub title: Option<String>,
    pub microphone: bool,
    pub system_audio: bool,
    /// `None` keeps screen context off (the default).
    pub screen: Option<ScreenSourceSpec>,
    /// Meeting application window to bring to the front after start.
    pub focus_app_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct GenerateAnswerRequest {
    /// Explicit question; defaults to the latest detected question or the
    /// recent conversation.
    pub question: Option<String>,
    pub style: Option<AnswerStyle>,
}

/// Snapshot of the in-flight / latest primary request for UI hydration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ActiveRequestView {
    pub request_id: Id,
    pub kind: RequestKind,
    pub slot: RequestSlot,
    pub state: AnswerState,
    pub question_text: Option<String>,
    pub category: Option<QuestionCategory>,
    pub style: Option<AnswerStyle>,
    pub text: String,
    pub error: Option<String>,
}

/// Full engine state for windows that (re)load mid-meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct EngineSnapshot {
    pub meeting: Option<Meeting>,
    pub state: MeetingState,
    pub paused_reason: Option<PauseReason>,
    pub mode: Mode,
    pub style: AnswerStyle,
    pub transcript: Vec<TranscriptSegment>,
    pub questions: Vec<DetectedQuestion>,
    pub active: Option<ActiveRequestView>,
    pub followups: Vec<FollowUpQuestion>,
    pub followups_request_id: Option<Id>,
    pub pinned: Vec<Suggestion>,
    pub topics: Vec<Topic>,
    pub suggested_next_topic: Option<String>,
    pub components: Vec<ComponentHealth>,
    pub last_observation: Option<ScreenObservation>,
    pub last_summary: Option<MeetingSummary>,
    pub demo_running: bool,
    pub processing_stopped: bool,
}

/// Developer tools: inject synthetic events without joining a meeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "action", rename_all = "snake_case")]
#[ts(export)]
pub enum SimulationAction {
    /// A final transcript line from the given speaker.
    Transcript { speaker: SpeakerRole, text: String },
    /// A question from the remote participant (candidate mode).
    Question { text: String },
    /// A candidate answer from the remote participant (interviewer mode).
    CandidateAnswer { text: String },
    /// A screen change with the given content.
    ScreenChange {
        description: String,
        #[serde(rename = "visibleText")]
        visible_text: Option<String>,
    },
    /// Force the next AI response to be this text (mock provider).
    AiResponse { text: String },
    /// Force the next AI request to fail with this message.
    AiFailure { message: String },
    /// Add artificial latency (ms) to mock provider responses.
    NetworkLatency { ms: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DemoScenario {
    pub id: String,
    pub title: String,
    pub mode: Mode,
    pub description: String,
}

/// A detected meeting application (UX only; nothing is injected into it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct MeetingAppInfo {
    pub id: String,
    /// "Zoom", "Microsoft Teams", "Google Meet (Chrome)", ...
    pub name: String,
    pub process_name: String,
    pub pid: u32,
    pub window_title: Option<String>,
    /// Native window id usable with `focus_meeting_app`.
    pub window_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Shortcuts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ShortcutStatus {
    pub action: ShortcutAction,
    pub accelerator: String,
    pub registered: bool,
    /// Registration error (e.g. already taken by another application).
    pub error: Option<String>,
    /// Known conflicts with common meeting / browser shortcuts.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Privacy & providers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct CapturePrivacyStatus {
    /// The OS offers a documented capture-exclusion API.
    pub supported: bool,
    /// The user setting.
    pub requested: bool,
    /// Exclusion is currently applied to the overlay (verified when possible).
    pub active: bool,
    pub platform: String,
    /// e.g. "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)".
    pub mechanism: String,
    /// Honest limitation statement for the UI.
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DataLocation {
    Local,
    Remote,
    Disabled,
    Mock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DataFlowNode {
    pub component: Component,
    pub label: String,
    pub location: DataLocation,
    pub provider: String,
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DataFlowReport {
    pub nodes: Vec<DataFlowNode>,
    /// The local-only setting is on (remote providers are refused).
    pub local_only_mode: bool,
    /// Every active component that touches meeting data is local.
    pub all_local: bool,
    /// Plain-language statement. "No meeting data leaves this device." only
    /// when `all_local` is true.
    pub statement: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum SecretProvider {
    Openai,
    Anthropic,
    OpenaiCompatible,
}

impl SecretProvider {
    pub const ALL: [SecretProvider; 3] = [
        SecretProvider::Openai,
        SecretProvider::Anthropic,
        SecretProvider::OpenaiCompatible,
    ];

    /// Environment variable consulted as a development fallback.
    pub fn env_var(self) -> &'static str {
        match self {
            SecretProvider::Openai => "OPENAI_API_KEY",
            SecretProvider::Anthropic => "ANTHROPIC_API_KEY",
            SecretProvider::OpenaiCompatible => "OPENAI_COMPATIBLE_API_KEY",
        }
    }

    /// Keychain account name.
    pub fn account(self) -> &'static str {
        match self {
            SecretProvider::Openai => "openai",
            SecretProvider::Anthropic => "anthropic",
            SecretProvider::OpenaiCompatible => "openai_compatible",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum SecretSource {
    Keychain,
    Environment,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ApiKeyStatus {
    pub provider: SecretProvider,
    pub configured: bool,
    pub source: SecretSource,
    pub env_var: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ProviderTestResult {
    pub component: Component,
    pub ok: bool,
    #[ts(type = "number | null")]
    pub latency_ms: Option<u64>,
    pub message: String,
    pub info: Option<ProviderInfo>,
}

/// Status of the optional local AI sidecar (faster-whisper + local embeddings
/// in a Python virtual environment).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct LocalAiStatus {
    /// The virtual environment exists.
    pub installed: bool,
    /// The server answers its health check.
    pub running: bool,
    /// The server process was started by OpenMeet.
    pub managed: bool,
    pub base_url: String,
    pub transcription_model: Option<String>,
    pub embedding_model: Option<String>,
    pub message: String,
    /// Command the user can run to install it.
    pub setup_hint: String,
}

// ---------------------------------------------------------------------------
// History & export
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ExportFormat {
    Markdown,
    Json,
    Text,
}

impl ExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Markdown => "md",
            ExportFormat::Json => "json",
            ExportFormat::Text => "txt",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ExportParts {
    pub summary: bool,
    pub transcript: bool,
    pub questions: bool,
    pub followups: bool,
}

impl Default for ExportParts {
    fn default() -> Self {
        ExportParts {
            summary: true,
            transcript: true,
            questions: true,
            followups: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ExportRequest {
    pub meeting_id: Id,
    pub format: ExportFormat,
    pub parts: ExportParts,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct ExportResult {
    /// `None` when the user cancelled the save dialog.
    pub path: Option<String>,
    #[ts(type = "number")]
    pub bytes: u64,
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct Diagnostics {
    pub components: Vec<ComponentHealth>,
    pub latency: Vec<LatencyStats>,
    pub memory_mb: f64,
    pub cpu_percent: f32,
    pub version: String,
    pub platform: String,
    pub developer_mode: bool,
    pub meeting_state: MeetingState,
    pub mode: Mode,
    pub active_requests: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BenchmarkItem {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct BenchmarkReport {
    pub items: Vec<BenchmarkItem>,
    #[ts(type = "number")]
    pub ran_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct AppInfo {
    pub version: String,
    pub platform: String,
    pub arch: String,
    pub developer_mode: bool,
    pub debug_build: bool,
}

// ---------------------------------------------------------------------------
// Backend -> window UI commands (event "openmeet://ui")
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
#[ts(export)]
pub enum UiCommand {
    /// Overlay: open and focus the "Ask OpenMeet" input.
    FocusAsk,
    /// Overlay: toggle between minimal HUD and expanded view.
    ToggleExpanded,
    ClickThroughChanged {
        enabled: bool,
    },
    OverlayVisibilityChanged {
        visible: bool,
    },
    /// Settings were changed; windows should re-fetch them.
    SettingsChanged,
    /// Main window: navigate to a section ("settings", "history", ...).
    Navigate {
        section: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_parse_and_contains() {
        let r = Rect::parse_csv("10, 20, 100, 50").unwrap();
        assert_eq!(
            r,
            Rect {
                x: 10,
                y: 20,
                width: 100,
                height: 50
            }
        );
        assert!(r.contains(Point { x: 10, y: 20 }));
        assert!(r.contains(Point { x: 109, y: 69 }));
        assert!(!r.contains(Point { x: 110, y: 20 }));
        assert!(Rect::parse_csv("1,2,3").is_none());
        assert!(Rect::parse_csv("a,b,c,d").is_none());
    }

    #[test]
    fn simulation_action_shape() {
        let a = SimulationAction::Transcript {
            speaker: SpeakerRole::Remote,
            text: "hi".into(),
        };
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["action"], "transcript");
        assert_eq!(v["speaker"], "remote");
        let back: SimulationAction = serde_json::from_value(v).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn ui_command_shape() {
        let v = serde_json::to_value(UiCommand::ClickThroughChanged { enabled: true }).unwrap();
        assert_eq!(v["type"], "CLICK_THROUGH_CHANGED");
        assert_eq!(v["enabled"], true);
    }
}
