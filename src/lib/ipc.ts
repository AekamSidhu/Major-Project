// Typed IPC contract between the React windows and the Rust backend.
//
// Every backend command is wrapped here; components never call `invoke`
// directly. Argument names are camelCase (Tauri maps them to the snake_case
// Rust parameters). Errors are `CoreError` objects ({ kind, message }).

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  ApiKeyStatus,
  AppInfo,
  AnswerStyle,
  AudioDevices,
  AudioSourceKind,
  BenchmarkReport,
  CapturePrivacyStatus,
  Component,
  CoreError,
  DataFlowReport,
  DemoScenario,
  Diagnostics,
  DisplayInfo,
  EngineSnapshot,
  ExportRequest,
  ExportResult,
  GenerateAnswerRequest,
  LocalAiStatus,
  Meeting,
  MeetingAppInfo,
  MeetingEvent,
  MeetingListItem,
  MeetingRecord,
  MeetingState,
  MeetingSummary,
  Mode,
  OverlayPosition,
  PrivacyRule,
  PrivacyRuleType,
  ProviderTestResult,
  Refinement,
  ScreenSourceInfo,
  ScreenSourceSpec,
  SecretProvider,
  Settings,
  ShortcutAction,
  ShortcutStatus,
  SimulationAction,
  StartMeetingRequest,
  UiCommand,
} from "@/bindings";

export const MEETING_EVENT = "openmeet://event";
export const UI_EVENT = "openmeet://ui";

/** Window labels created by the backend. */
export const WINDOW_MAIN = "main";
export const WINDOW_OVERLAY = "overlay";

function isCoreError(e: unknown): e is CoreError {
  return typeof e === "object" && e !== null && "kind" in e;
}

/** Human-readable message for any error thrown by `invoke`. */
export function errorMessage(e: unknown): string {
  if (isCoreError(e)) {
    return "message" in e && typeof e.message === "string" ? e.message : String(e.kind);
  }
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return "Unexpected error";
}

export const api = {
  // App & settings ---------------------------------------------------------
  getAppInfo: () => invoke<AppInfo>("get_app_info"),
  getSettings: () => invoke<Settings>("get_settings"),
  /** Validates, sanitizes, persists and applies settings; returns the stored value. */
  updateSettings: (settings: Settings) => invoke<Settings>("update_settings", { settings }),
  resetSettings: () => invoke<Settings>("reset_settings"),

  // Secrets (OS keychain; never returned to the UI) ------------------------
  getApiKeyStatus: () => invoke<ApiKeyStatus[]>("get_api_key_status"),
  setApiKey: (provider: SecretProvider, key: string) =>
    invoke<ApiKeyStatus>("set_api_key", { provider, key }),
  clearApiKey: (provider: SecretProvider) => invoke<ApiKeyStatus>("clear_api_key", { provider }),

  // Providers ----------------------------------------------------------------
  /** component: "llm" | "transcription" | "vision" | "embeddings" */
  testProvider: (component: Component) =>
    invoke<ProviderTestResult>("test_provider", { component }),
  listModels: (component: Component) => invoke<string[]>("list_models", { component }),
  getDataFlow: () => invoke<DataFlowReport>("get_data_flow"),
  getLocalAiStatus: () => invoke<LocalAiStatus>("get_local_ai_status"),
  startLocalAi: () => invoke<LocalAiStatus>("start_local_ai"),
  stopLocalAi: () => invoke<LocalAiStatus>("stop_local_ai"),

  // Audio & screen -----------------------------------------------------------
  listAudioDevices: () => invoke<AudioDevices>("list_audio_devices"),
  /** Level meter for device setup: emits AUDIO_LEVEL events for up to 15 s. */
  startAudioTest: (source: AudioSourceKind, device: string | null) =>
    invoke<void>("start_audio_test", { source, device }),
  stopAudioTest: () => invoke<void>("stop_audio_test"),
  listScreenSources: () => invoke<ScreenSourceInfo[]>("list_screen_sources"),
  /** Enables/disables screen context for the running meeting and saves the choice. */
  setScreenCapture: (enabled: boolean, source: ScreenSourceSpec | null) =>
    invoke<void>("set_screen_capture", { enabled, source }),
  captureScreenNow: () => invoke<void>("capture_screen_now"),

  // Meeting lifecycle ----------------------------------------------------------
  getSnapshot: () => invoke<EngineSnapshot>("get_snapshot"),
  startMeeting: (request: StartMeetingRequest) => invoke<Meeting>("start_meeting", { request }),
  pauseMeeting: () => invoke<void>("pause_meeting"),
  resumeMeeting: () => invoke<void>("resume_meeting"),
  togglePause: () => invoke<MeetingState>("toggle_pause"),
  stopMeeting: () => invoke<void>("stop_meeting"),
  emergencyStop: () => invoke<void>("emergency_stop"),
  detectMeetingApps: () => invoke<MeetingAppInfo[]>("detect_meeting_apps"),
  focusMeetingApp: (windowId: string) => invoke<void>("focus_meeting_app", { windowId }),

  // Assistant ---------------------------------------------------------------
  /** Returns the request id rendered in the overlay's primary slot. */
  generateAnswer: (request: GenerateAnswerRequest) =>
    invoke<string>("generate_answer", { request }),
  generateFollowups: () => invoke<string>("generate_followups"),
  askOpenMeet: (prompt: string) => invoke<string>("ask_openmeet", { prompt }),
  clarifyQuestion: () => invoke<string>("clarify_question"),
  refineAnswer: (requestId: string, refinement: Refinement) =>
    invoke<string>("refine_answer", { requestId, refinement }),
  cancelRequest: (requestId: string) => invoke<void>("cancel_request", { requestId }),
  dismissSuggestion: (requestId: string) => invoke<void>("dismiss_suggestion", { requestId }),
  pinSuggestion: (requestId: string, pinned: boolean) =>
    invoke<void>("pin_suggestion", { requestId, pinned }),
  setMode: (mode: Mode) => invoke<void>("set_mode", { mode }),
  setAnswerStyle: (style: AnswerStyle) => invoke<void>("set_answer_style", { style }),

  // Overlay window -----------------------------------------------------------
  overlayShow: () => invoke<boolean>("overlay_show"),
  overlayHide: () => invoke<boolean>("overlay_hide"),
  overlayToggle: () => invoke<boolean>("overlay_toggle"),
  setClickThrough: (enabled: boolean) => invoke<boolean>("set_click_through", { enabled }),
  /** Content-driven resize (logical pixels); clamped to the configured limits. */
  overlaySetSize: (width: number, height: number) =>
    invoke<void>("overlay_set_size", { width, height }),
  overlaySetPosition: (position: OverlayPosition) =>
    invoke<void>("overlay_set_position", { position }),
  /** Remembers the current (dragged) position for the current display. */
  overlaySavePosition: () => invoke<void>("overlay_save_position"),
  overlayMoveToDisplay: (displayKey: string) =>
    invoke<void>("overlay_move_to_display", { displayKey }),
  /** Makes the overlay focusable for typing (Ask OpenMeet) and remembers the previous app. */
  overlayBeginInput: () => invoke<void>("overlay_begin_input"),
  /** Restores non-activating behaviour and returns focus to the meeting app. */
  overlayEndInput: () => invoke<void>("overlay_end_input"),
  listDisplays: () => invoke<DisplayInfo[]>("list_displays"),
  reportRenderLatency: (ms: number) => invoke<void>("report_render_latency", { ms }),
  /** Opens (or focuses) the main window, optionally on a section. */
  openMainWindow: (section: string | null) => invoke<void>("open_main_window", { section }),

  // Privacy -----------------------------------------------------------------
  getCapturePrivacyStatus: () => invoke<CapturePrivacyStatus>("get_capture_privacy_status"),
  setCaptureExclusion: (enabled: boolean) =>
    invoke<CapturePrivacyStatus>("set_capture_exclusion", { enabled }),
  listPrivacyRules: () => invoke<PrivacyRule[]>("list_privacy_rules"),
  addPrivacyRule: (ruleType: PrivacyRuleType, value: string) =>
    invoke<PrivacyRule>("add_privacy_rule", { ruleType, value }),
  removePrivacyRule: (id: string) => invoke<void>("remove_privacy_rule", { id }),
  suggestedPrivacyApps: () => invoke<string[]>("suggested_privacy_apps"),

  // History & export ----------------------------------------------------------
  listMeetings: (limit: number, offset: number) =>
    invoke<MeetingListItem[]>("list_meetings", { limit, offset }),
  getMeeting: (meetingId: string) => invoke<MeetingRecord>("get_meeting", { meetingId }),
  deleteMeeting: (meetingId: string) => invoke<void>("delete_meeting", { meetingId }),
  /** Deletes every meeting, transcript, summary and embedding (keeps settings). */
  deleteAllData: () => invoke<void>("delete_all_data"),
  /** Opens a native save dialog in the backend; `path` is null when cancelled. */
  exportMeeting: (request: ExportRequest) => invoke<ExportResult>("export_meeting", { request }),
  regenerateSummary: (meetingId: string) =>
    invoke<MeetingSummary>("regenerate_summary", { meetingId }),

  // Shortcuts ---------------------------------------------------------------
  getShortcutStatus: () => invoke<ShortcutStatus[]>("get_shortcut_status"),
  updateShortcut: (action: ShortcutAction, accelerator: string) =>
    invoke<ShortcutStatus[]>("update_shortcut", { action, accelerator }),
  resetShortcuts: () => invoke<ShortcutStatus[]>("reset_shortcuts"),

  // Diagnostics & developer tools -------------------------------------------
  getDiagnostics: () => invoke<Diagnostics>("get_diagnostics"),
  runBenchmark: () => invoke<BenchmarkReport>("run_benchmark"),
  listDemoScenarios: () => invoke<DemoScenario[]>("list_demo_scenarios"),
  startDemo: (scenarioId: string) => invoke<Meeting>("start_demo", { scenarioId }),
  stopDemo: () => invoke<void>("stop_demo"),
  simulate: (action: SimulationAction) => invoke<void>("simulate", { action }),
  clearSimulation: () => invoke<void>("clear_simulation"),
} as const;

export type Api = typeof api;

/** Subscribes to the meeting event bus. */
export function onMeetingEvent(handler: (event: MeetingEvent) => void): Promise<UnlistenFn> {
  return listen<MeetingEvent>(MEETING_EVENT, (e) => handler(e.payload));
}

/** Subscribes to backend → window UI commands. */
export function onUiCommand(handler: (command: UiCommand) => void): Promise<UnlistenFn> {
  return listen<UiCommand>(UI_EVENT, (e) => handler(e.payload));
}
