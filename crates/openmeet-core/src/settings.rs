//! Persisted user settings.
//!
//! Every field has a serde default so settings written by older versions keep
//! loading. `Settings::sanitized` clamps values into safe ranges and is applied
//! to every settings update coming from the UI.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::ipc::{Point, ScreenSourceSpec};
use crate::types::{AnswerStyle, Mode, ResponseLength, UserProfile};

pub const SETTINGS_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct Settings {
    pub version: u32,
    pub mode: Mode,
    pub ai: AiSettings,
    pub transcription: TranscriptionSettings,
    pub assistant: AssistantSettings,
    pub audio: AudioSettings,
    pub screen: ScreenSettings,
    pub privacy: PrivacySettings,
    pub overlay: OverlaySettings,
    pub shortcuts: ShortcutSettings,
    pub general: GeneralSettings,
    pub profile: UserProfile,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            version: SETTINGS_VERSION,
            mode: Mode::Candidate,
            ai: AiSettings::default(),
            transcription: TranscriptionSettings::default(),
            assistant: AssistantSettings::default(),
            audio: AudioSettings::default(),
            screen: ScreenSettings::default(),
            privacy: PrivacySettings::default(),
            overlay: OverlaySettings::default(),
            shortcuts: ShortcutSettings::default(),
            general: GeneralSettings::default(),
            profile: UserProfile::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// AI providers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum LlmProviderKind {
    Openai,
    Anthropic,
    /// Local models through Ollama (default: local-first).
    #[default]
    Ollama,
    /// Any OpenAI-compatible server (LM Studio, vLLM, llama.cpp server, ...).
    OpenaiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ReasoningEffort {
    /// Lowest latency (default for live suggestions).
    #[default]
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct LlmSettings {
    pub provider: LlmProviderKind,
    pub model: String,
    /// Required for Ollama / OpenAI-compatible; optional override otherwise.
    pub base_url: Option<String>,
    /// Ignored by models that do not accept sampling parameters.
    pub temperature: f32,
    /// Visible-output token budget for live suggestions.
    pub max_output_tokens: u32,
    /// Applied where the provider supports it.
    pub reasoning_effort: ReasoningEffort,
    /// Optional cheaper/faster model for classification and compression.
    pub fast_model: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            provider: LlmProviderKind::Ollama,
            model: default_llm_model(LlmProviderKind::Ollama).to_string(),
            base_url: Some(DEFAULT_OLLAMA_URL.to_string()),
            temperature: 0.3,
            max_output_tokens: 1200,
            reasoning_effort: ReasoningEffort::Low,
            fast_model: None,
        }
    }
}

pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
pub const DEFAULT_LOCAL_AI_URL: &str = "http://127.0.0.1:8765";
pub const DEFAULT_OPENAI_URL: &str = "https://api.openai.com";
pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com";

/// Suggested default model per provider (always user-editable; the settings UI
/// can list models from the provider).
pub fn default_llm_model(kind: LlmProviderKind) -> &'static str {
    match kind {
        LlmProviderKind::Openai => "gpt-5-mini",
        LlmProviderKind::Anthropic => "claude-opus-5-5",
        LlmProviderKind::Ollama => "llama3.2",
        LlmProviderKind::OpenaiCompatible => "local-model",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum VisionProviderKind {
    /// No screen understanding.
    Disabled,
    /// Local OS OCR only (Windows.Media.Ocr where available).
    #[default]
    OcrOnly,
    Openai,
    Anthropic,
    Ollama,
    OpenaiCompatible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct VisionSettings {
    pub provider: VisionProviderKind,
    pub model: String,
    pub base_url: Option<String>,
    /// Run local OCR first and pass the text to the vision model.
    pub use_ocr: bool,
}

impl Default for VisionSettings {
    fn default() -> Self {
        VisionSettings {
            provider: VisionProviderKind::OcrOnly,
            model: String::new(),
            base_url: None,
            use_ocr: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum EmbeddingProviderKind {
    /// Built-in deterministic feature-hashing embeddings (no model needed).
    #[default]
    LocalHash,
    /// The local AI sidecar (fastembed).
    LocalAi,
    Openai,
    Ollama,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct EmbeddingSettings {
    pub provider: EmbeddingProviderKind,
    pub model: String,
    pub base_url: Option<String>,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        EmbeddingSettings {
            provider: EmbeddingProviderKind::LocalHash,
            model: String::new(),
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
#[derive(Default)]
pub struct AiSettings {
    pub llm: LlmSettings,
    pub vision: VisionSettings,
    pub embeddings: EmbeddingSettings,
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum TranscriptionProviderKind {
    /// Local faster-whisper sidecar in a Python venv (default: local-first).
    #[default]
    LocalAi,
    Openai,
    /// Any server implementing POST /v1/audio/transcriptions.
    OpenaiCompatible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct TranscriptionSettings {
    pub provider: TranscriptionProviderKind,
    pub model: String,
    pub base_url: Option<String>,
    /// ISO-639-1 language hint, `None` = auto-detect.
    pub language: Option<String>,
    /// Emit interim (partial) transcripts while someone is still speaking.
    /// Costs extra requests with cloud providers.
    pub interim_results: bool,
    /// Domain vocabulary passed as a transcription prompt (names, jargon).
    pub vocabulary: String,
}

impl Default for TranscriptionSettings {
    fn default() -> Self {
        TranscriptionSettings {
            provider: TranscriptionProviderKind::LocalAi,
            model: default_transcription_model(TranscriptionProviderKind::LocalAi).to_string(),
            base_url: Some(DEFAULT_LOCAL_AI_URL.to_string()),
            language: Some("en".to_string()),
            interim_results: false,
            vocabulary: String::new(),
        }
    }
}

pub fn default_transcription_model(kind: TranscriptionProviderKind) -> &'static str {
    match kind {
        TranscriptionProviderKind::LocalAi => "base.en",
        TranscriptionProviderKind::Openai => "gpt-4o-mini-transcribe",
        TranscriptionProviderKind::OpenaiCompatible => "whisper-1",
    }
}

// ---------------------------------------------------------------------------
// Assistant behaviour
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct AssistantSettings {
    pub answer_style: AnswerStyle,
    pub response_length: ResponseLength,
    /// Candidate mode: generate a suggestion as soon as a question is detected.
    pub auto_answer: bool,
    /// Interviewer mode: generate follow-ups when the candidate finishes.
    pub auto_followups: bool,
    /// Use the LLM to confirm ambiguous question detections.
    pub semantic_detection: bool,
    /// Minimum detector confidence (0..1) for automatic answers.
    pub detection_threshold: f32,
    /// Interviewer mode: minimum words in a candidate turn before analysis.
    pub followup_min_words: u32,
    /// Silence after the candidate's last utterance that ends their turn.
    pub turn_end_silence_ms: u32,
    /// Raw transcript window sent with every request (seconds).
    pub context_window_secs: u32,
    /// Approximate token budget for assembled context.
    pub context_token_budget: u32,
}

impl Default for AssistantSettings {
    fn default() -> Self {
        AssistantSettings {
            answer_style: AnswerStyle::Concise,
            response_length: ResponseLength::Medium,
            auto_answer: true,
            auto_followups: true,
            semantic_detection: true,
            detection_threshold: 0.6,
            followup_min_words: 20,
            turn_end_silence_ms: 1500,
            context_window_secs: 180,
            context_token_budget: 3500,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct AudioSettings {
    pub microphone_enabled: bool,
    /// `None` = OS default input device.
    pub microphone_device: Option<String>,
    pub system_audio_enabled: bool,
    /// `None` = OS default output device (loopback / monitor).
    pub system_audio_device: Option<String>,
    /// 0 (least sensitive) ..= 1 (most sensitive).
    pub vad_sensitivity: f32,
    /// Silence that ends an utterance.
    pub silence_ms: u32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings {
            microphone_enabled: true,
            microphone_device: None,
            system_audio_enabled: cfg!(target_os = "windows"),
            system_audio_device: None,
            vad_sensitivity: 0.6,
            silence_ms: 700,
        }
    }
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ScreenFrequency {
    /// Only explicit manual captures.
    Manual,
    #[default]
    Fps1,
    Fps2,
    Fps5,
}

impl ScreenFrequency {
    /// Sampling interval, `None` for manual.
    pub fn interval_ms(self) -> Option<u64> {
        match self {
            ScreenFrequency::Manual => None,
            ScreenFrequency::Fps1 => Some(1000),
            ScreenFrequency::Fps2 => Some(500),
            ScreenFrequency::Fps5 => Some(200),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct ScreenSettings {
    /// Screen context is OFF by default and requires explicit activation.
    pub enabled: bool,
    pub source: Option<ScreenSourceSpec>,
    pub frequency: ScreenFrequency,
    /// Minimum perceptual change (0..1) before a frame is processed again.
    pub change_threshold: f32,
    /// Keep raw screenshots on disk. Off by default; observations are kept.
    pub store_screenshots: bool,
    /// Upper bound on vision-model calls per minute (cost control).
    pub max_vision_per_minute: u32,
}

impl Default for ScreenSettings {
    fn default() -> Self {
        ScreenSettings {
            enabled: false,
            source: None,
            frequency: ScreenFrequency::Fps1,
            change_threshold: 0.08,
            store_screenshots: false,
            max_vision_per_minute: 6,
        }
    }
}

// ---------------------------------------------------------------------------
// Privacy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum RetentionPolicy {
    /// Nothing about meetings is written to disk.
    None,
    OneDay,
    #[default]
    SevenDays,
    ThirtyDays,
    Forever,
}

impl RetentionPolicy {
    /// Maximum age in milliseconds, `None` = keep until deleted.
    pub fn max_age_ms(self) -> Option<i64> {
        const DAY: i64 = 24 * 60 * 60 * 1000;
        match self {
            RetentionPolicy::None => Some(0),
            RetentionPolicy::OneDay => Some(DAY),
            RetentionPolicy::SevenDays => Some(7 * DAY),
            RetentionPolicy::ThirtyDays => Some(30 * DAY),
            RetentionPolicy::Forever => None,
        }
    }

    pub fn persists(self) -> bool {
        self != RetentionPolicy::None
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct PrivacySettings {
    /// Ask the OS to exclude the overlay from screen capture where supported.
    pub capture_exclusion: bool,
    /// Refuse to construct any provider that sends data off the device.
    pub local_only: bool,
    pub retention: RetentionPolicy,
}

impl Default for PrivacySettings {
    fn default() -> Self {
        PrivacySettings {
            capture_exclusion: true,
            local_only: false,
            retention: RetentionPolicy::SevenDays,
        }
    }
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum OverlayPosition {
    #[default]
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Theme {
    #[default]
    Dark,
    Light,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum Density {
    Compact,
    #[default]
    Comfortable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum DisplayLayout {
    /// Overlay on the same display as the meeting.
    #[default]
    SameDisplay,
    /// Overlay on a different display than the meeting ("assistant display").
    SeparateDisplay,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct OverlaySettings {
    /// 0.3 ..= 1.0, applied to the overlay background.
    pub opacity: f32,
    /// Logical pixels.
    pub width: u32,
    /// Logical pixels; content scrolls beyond this.
    pub max_height: u32,
    /// Base font size in px.
    pub font_size: u32,
    pub density: Density,
    pub theme: Theme,
    /// Blurred translucent background.
    pub translucent: bool,
    /// Start in the minimal HUD state.
    pub minimal: bool,
    pub position: OverlayPosition,
    /// Remembered custom positions keyed by display key (or
    /// "display-key|app" for per-meeting-app positions).
    pub custom_positions: BTreeMap<String, Point>,
    /// Display key the overlay should live on (`None` = primary / meeting
    /// display).
    pub target_display: Option<String>,
    pub layout: DisplayLayout,
    pub click_through: bool,
    pub reduced_motion: bool,
    pub show_on_meeting_start: bool,
    /// Distance from the display edge for corner positions (logical px).
    pub margin: u32,
}

impl Default for OverlaySettings {
    fn default() -> Self {
        OverlaySettings {
            opacity: 0.92,
            width: 380,
            max_height: 560,
            font_size: 14,
            density: Density::Comfortable,
            theme: Theme::Dark,
            translucent: true,
            minimal: false,
            position: OverlayPosition::TopRight,
            custom_positions: BTreeMap::new(),
            target_display: None,
            layout: DisplayLayout::SameDisplay,
            click_through: false,
            reduced_motion: false,
            show_on_meeting_start: true,
            margin: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Shortcuts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum ShortcutAction {
    ToggleOverlay,
    GenerateAnswer,
    GenerateFollowups,
    AskAi,
    ToggleMode,
    ToggleClickThrough,
    CaptureScreen,
    PauseResume,
    EmergencyStop,
    ExpandHud,
    CycleStyle,
}

impl ShortcutAction {
    pub const ALL: [ShortcutAction; 11] = [
        ShortcutAction::ToggleOverlay,
        ShortcutAction::GenerateAnswer,
        ShortcutAction::GenerateFollowups,
        ShortcutAction::AskAi,
        ShortcutAction::ToggleMode,
        ShortcutAction::ToggleClickThrough,
        ShortcutAction::CaptureScreen,
        ShortcutAction::PauseResume,
        ShortcutAction::EmergencyStop,
        ShortcutAction::ExpandHud,
        ShortcutAction::CycleStyle,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ShortcutAction::ToggleOverlay => "Show / hide overlay",
            ShortcutAction::GenerateAnswer => "Generate answer",
            ShortcutAction::GenerateFollowups => "Generate follow-ups",
            ShortcutAction::AskAi => "Ask OpenMeet",
            ShortcutAction::ToggleMode => "Toggle candidate / interviewer mode",
            ShortcutAction::ToggleClickThrough => "Toggle click-through",
            ShortcutAction::CaptureScreen => "Capture screen manually",
            ShortcutAction::PauseResume => "Pause / resume processing",
            ShortcutAction::EmergencyStop => "Emergency privacy stop",
            ShortcutAction::ExpandHud => "Expand / collapse HUD",
            ShortcutAction::CycleStyle => "Cycle answer style",
        }
    }

    /// Platform default accelerator (Tauri global-shortcut syntax).
    pub fn default_accelerator(self) -> &'static str {
        match self {
            ShortcutAction::ToggleOverlay => "CommandOrControl+Shift+Space",
            ShortcutAction::GenerateAnswer => "CommandOrControl+Shift+Enter",
            ShortcutAction::GenerateFollowups => "CommandOrControl+Shift+Q",
            ShortcutAction::AskAi => "CommandOrControl+Shift+A",
            ShortcutAction::ToggleMode => "CommandOrControl+Shift+M",
            ShortcutAction::ToggleClickThrough => "CommandOrControl+Shift+C",
            ShortcutAction::CaptureScreen => "CommandOrControl+Shift+S",
            ShortcutAction::PauseResume => "CommandOrControl+Shift+P",
            // Ctrl+Shift+Esc is reserved by Windows for Task Manager and cannot
            // be registered there; use Ctrl+Shift+Backspace on Windows.
            ShortcutAction::EmergencyStop => {
                if cfg!(target_os = "windows") {
                    "CommandOrControl+Shift+Backspace"
                } else {
                    "CommandOrControl+Shift+Escape"
                }
            }
            ShortcutAction::ExpandHud => "CommandOrControl+Shift+E",
            ShortcutAction::CycleStyle => "CommandOrControl+Shift+Y",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct ShortcutSettings {
    pub bindings: BTreeMap<ShortcutAction, String>,
}

impl Default for ShortcutSettings {
    fn default() -> Self {
        ShortcutSettings {
            bindings: ShortcutAction::ALL
                .into_iter()
                .map(|a| (a, a.default_accelerator().to_string()))
                .collect(),
        }
    }
}

impl ShortcutSettings {
    /// Accelerator for an action (falls back to the default).
    pub fn accelerator(&self, action: ShortcutAction) -> String {
        self.bindings
            .get(&action)
            .cloned()
            .unwrap_or_else(|| action.default_accelerator().to_string())
    }

    /// Pairs of actions bound to the same (normalized) accelerator.
    pub fn duplicates(&self) -> Vec<(ShortcutAction, ShortcutAction)> {
        let mut out = Vec::new();
        let actions: Vec<(ShortcutAction, String)> = ShortcutAction::ALL
            .into_iter()
            .map(|a| (a, normalize_accelerator(&self.accelerator(a))))
            .filter(|(_, acc)| !acc.is_empty())
            .collect();
        for i in 0..actions.len() {
            for j in (i + 1)..actions.len() {
                if actions[i].1 == actions[j].1 {
                    out.push((actions[i].0, actions[j].0));
                }
            }
        }
        out
    }
}

/// Canonical form of an accelerator for comparison: modifiers sorted,
/// aliases unified, case-insensitive. An empty string means "unbound".
pub fn normalize_accelerator(acc: &str) -> String {
    let mut mods: Vec<String> = Vec::new();
    let mut key = String::new();
    for raw in acc.split('+').map(str::trim).filter(|s| !s.is_empty()) {
        let lower = raw.to_ascii_lowercase();
        let m = match lower.as_str() {
            "commandorcontrol" | "cmdorctrl" | "cmdorcontrol" | "commandorctrl" => {
                Some("cmdorctrl")
            }
            "control" | "ctrl" => Some("ctrl"),
            "command" | "cmd" | "super" | "meta" => Some("super"),
            "shift" => Some("shift"),
            "alt" | "option" => Some("alt"),
            _ => None,
        };
        match m {
            Some(m) => mods.push(m.to_string()),
            None => {
                key = match lower.as_str() {
                    "esc" => "escape".to_string(),
                    "return" => "enter".to_string(),
                    other => other.to_string(),
                }
            }
        }
    }
    if key.is_empty() {
        return String::new();
    }
    mods.sort();
    mods.dedup();
    mods.push(key);
    mods.join("+")
}

// ---------------------------------------------------------------------------
// General
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export)]
pub struct GeneralSettings {
    pub onboarding_complete: bool,
    /// Launch OpenMeet at login. Never starts any capture by itself.
    pub launch_at_startup: bool,
    /// Shows diagnostics, latency metrics and simulation tools.
    pub developer_mode: bool,
    /// "OpenMeet, explain that" style commands from the user's microphone.
    pub voice_commands: bool,
    /// OS notifications for important post-meeting events only.
    pub notifications: bool,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        GeneralSettings {
            onboarding_complete: false,
            launch_at_startup: false,
            developer_mode: cfg!(debug_assertions),
            voice_commands: false,
            notifications: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn clamp_f32(v: f32, lo: f32, hi: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

fn clean_url(url: &Option<String>) -> Option<String> {
    url.as_ref()
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
}

/// Returns an error message when `url` is not an http(s) URL.
pub fn validate_base_url(url: &str) -> Result<(), String> {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err(format!("'{url}' must start with http:// or https://"));
    }
    if url.contains(char::is_whitespace) {
        return Err(format!("'{url}' must not contain whitespace"));
    }
    let rest = &url[url.find("://").map(|i| i + 3).unwrap_or(0)..];
    if rest.is_empty() || rest.starts_with('/') {
        return Err(format!("'{url}' is missing a host"));
    }
    if rest.split('/').next().unwrap_or("").contains('@') {
        return Err("credentials must not be embedded in URLs".to_string());
    }
    Ok(())
}

impl Settings {
    /// Returns a copy with every numeric value clamped into its supported range
    /// and URLs normalized.
    pub fn sanitized(&self) -> Settings {
        let mut s = self.clone();
        s.version = SETTINGS_VERSION;

        s.ai.llm.temperature = clamp_f32(s.ai.llm.temperature, 0.0, 2.0, 0.3);
        s.ai.llm.max_output_tokens = s.ai.llm.max_output_tokens.clamp(128, 16_000);
        s.ai.llm.base_url = clean_url(&s.ai.llm.base_url);
        s.ai.llm.model = s.ai.llm.model.trim().to_string();
        s.ai.llm.fast_model =
            s.ai.llm
                .fast_model
                .as_ref()
                .map(|m| m.trim().to_string())
                .filter(|m| !m.is_empty());
        s.ai.vision.base_url = clean_url(&s.ai.vision.base_url);
        s.ai.vision.model = s.ai.vision.model.trim().to_string();
        s.ai.embeddings.base_url = clean_url(&s.ai.embeddings.base_url);
        s.ai.embeddings.model = s.ai.embeddings.model.trim().to_string();

        s.transcription.base_url = clean_url(&s.transcription.base_url);
        s.transcription.model = s.transcription.model.trim().to_string();
        s.transcription.language = s
            .transcription
            .language
            .as_ref()
            .map(|l| l.trim().to_ascii_lowercase())
            .filter(|l| !l.is_empty() && l != "auto");
        s.transcription.vocabulary = s.transcription.vocabulary.chars().take(1000).collect();

        let a = &mut s.assistant;
        a.detection_threshold = clamp_f32(a.detection_threshold, 0.1, 0.99, 0.6);
        a.followup_min_words = a.followup_min_words.clamp(3, 400);
        a.turn_end_silence_ms = a.turn_end_silence_ms.clamp(400, 10_000);
        a.context_window_secs = a.context_window_secs.clamp(30, 900);
        a.context_token_budget = a.context_token_budget.clamp(800, 32_000);

        s.audio.vad_sensitivity = clamp_f32(s.audio.vad_sensitivity, 0.0, 1.0, 0.6);
        s.audio.silence_ms = s.audio.silence_ms.clamp(300, 3000);

        s.screen.change_threshold = clamp_f32(s.screen.change_threshold, 0.01, 0.9, 0.08);
        s.screen.max_vision_per_minute = s.screen.max_vision_per_minute.clamp(1, 60);

        let o = &mut s.overlay;
        o.opacity = clamp_f32(o.opacity, 0.3, 1.0, 0.92);
        o.width = o.width.clamp(260, 900);
        o.max_height = o.max_height.clamp(120, 1600);
        o.font_size = o.font_size.clamp(11, 24);
        o.margin = o.margin.min(200);

        for action in ShortcutAction::ALL {
            s.shortcuts
                .bindings
                .entry(action)
                .or_insert_with(|| action.default_accelerator().to_string());
        }
        for v in s.shortcuts.bindings.values_mut() {
            *v = v.trim().to_string();
        }

        s.profile.background = s.profile.background.chars().take(20_000).collect();
        s.profile.role_context = s.profile.role_context.chars().take(10_000).collect();
        s.profile.planned_topics = s.profile.planned_topics.chars().take(4_000).collect();
        s.profile.custom_instructions = s.profile.custom_instructions.chars().take(4_000).collect();
        s
    }

    /// Validation errors that must be fixed before settings are saved.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let urls = [
            ("LLM base URL", &self.ai.llm.base_url),
            ("Vision base URL", &self.ai.vision.base_url),
            ("Embeddings base URL", &self.ai.embeddings.base_url),
            ("Transcription base URL", &self.transcription.base_url),
        ];
        for (label, url) in urls {
            if let Some(u) = url {
                if let Err(e) = validate_base_url(u) {
                    errors.push(format!("{label}: {e}"));
                }
            }
        }
        if self.ai.llm.model.trim().is_empty() {
            errors.push("LLM model must not be empty".to_string());
        }
        if matches!(
            self.ai.llm.provider,
            LlmProviderKind::Ollama | LlmProviderKind::OpenaiCompatible
        ) && self.ai.llm.base_url.is_none()
        {
            errors.push("LLM base URL is required for this provider".to_string());
        }
        if self.transcription.model.trim().is_empty() {
            errors.push("Transcription model must not be empty".to_string());
        }
        for (a, b) in self.shortcuts.duplicates() {
            errors.push(format!(
                "Shortcut conflict: '{}' and '{}' use the same keys",
                a.label(),
                b.label()
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_local_first_and_conservative() {
        let s = Settings::default();
        assert_eq!(s.ai.llm.provider, LlmProviderKind::Ollama);
        assert_eq!(s.transcription.provider, TranscriptionProviderKind::LocalAi);
        assert_eq!(s.ai.embeddings.provider, EmbeddingProviderKind::LocalHash);
        assert!(!s.screen.enabled, "screen capture must be off by default");
        assert!(!s.screen.store_screenshots);
        assert_eq!(s.privacy.retention, RetentionPolicy::SevenDays);
        assert!(!s.general.launch_at_startup);
        assert!(s.validate().is_ok(), "{:?}", s.validate());
    }

    #[test]
    fn deserializes_partial_json_with_defaults() {
        let s: Settings =
            serde_json::from_str(r#"{"mode":"interviewer","overlay":{"opacity":0.5}}"#).unwrap();
        assert_eq!(s.mode, Mode::Interviewer);
        assert_eq!(s.overlay.opacity, 0.5);
        assert_eq!(s.overlay.width, 380);
        assert_eq!(s.shortcuts.bindings.len(), ShortcutAction::ALL.len());
    }

    #[test]
    fn sanitize_clamps() {
        let mut s = Settings::default();
        s.overlay.opacity = 7.0;
        s.overlay.width = 10;
        s.assistant.detection_threshold = f32::NAN;
        s.ai.llm.base_url = Some("http://localhost:11434/ ".into());
        s.shortcuts.bindings.clear();
        let c = s.sanitized();
        assert_eq!(c.overlay.opacity, 1.0);
        assert_eq!(c.overlay.width, 260);
        assert_eq!(c.assistant.detection_threshold, 0.6);
        assert_eq!(c.ai.llm.base_url.as_deref(), Some("http://localhost:11434"));
        assert_eq!(c.shortcuts.bindings.len(), ShortcutAction::ALL.len());
    }

    #[test]
    fn validates_urls() {
        assert!(validate_base_url("https://api.openai.com").is_ok());
        assert!(validate_base_url("http://127.0.0.1:11434").is_ok());
        assert!(validate_base_url("ftp://x").is_err());
        assert!(validate_base_url("https://user:pw@host").is_err());
        assert!(validate_base_url("https://").is_err());
        let mut s = Settings::default();
        s.ai.llm.base_url = Some("javascript:alert(1)".into());
        assert!(s.validate().is_err());
    }

    #[test]
    fn detects_duplicate_shortcuts() {
        let mut s = Settings::default();
        s.shortcuts
            .bindings
            .insert(ShortcutAction::AskAi, "ctrl+shift+q".into());
        s.shortcuts
            .bindings
            .insert(ShortcutAction::GenerateFollowups, "Shift+Control+Q".into());
        let d = s.shortcuts.duplicates();
        assert_eq!(d.len(), 1);
        assert!(s.validate().is_err());
    }

    #[test]
    fn normalize_accelerator_aliases() {
        assert_eq!(
            normalize_accelerator("CmdOrCtrl+Shift+Esc"),
            normalize_accelerator("Shift+CommandOrControl+Escape")
        );
        assert_eq!(normalize_accelerator(""), "");
        assert_eq!(normalize_accelerator("Shift+"), "");
    }

    #[test]
    fn retention_ages() {
        assert_eq!(RetentionPolicy::Forever.max_age_ms(), None);
        assert!(!RetentionPolicy::None.persists());
        assert_eq!(RetentionPolicy::OneDay.max_age_ms(), Some(86_400_000));
    }

    #[test]
    fn screen_frequency_intervals() {
        assert_eq!(ScreenFrequency::Manual.interval_ms(), None);
        assert_eq!(ScreenFrequency::Fps5.interval_ms(), Some(200));
    }
}
