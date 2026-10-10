// Pure helpers for the meeting (Home) section. Unit-tested; no React here.

import type {
  AudioDeviceInfo,
  Component,
  ComponentHealth,
  ComponentStatus,
  DataLocation,
  DetectedQuestion,
  Meeting,
  MeetingState,
  Mode,
  PauseReason,
  Rect,
  RetentionPolicy,
  ScreenSourceInfo,
  ScreenSourceKind,
  ScreenSourceSpec,
  Settings,
  SpeakerRole,
  StartMeetingRequest,
  TranscriptSegment,
} from "@/bindings";
import { formatOffset } from "@/lib/format";
import type { SectionId } from "@/main/store";

export type Tone = "neutral" | "accent" | "ok" | "warn" | "danger";

// ---------------------------------------------------------------------------
// Meeting state
// ---------------------------------------------------------------------------

const LIVE_STATES: ReadonlySet<MeetingState> = new Set<MeetingState>([
  "starting",
  "active",
  "paused",
  "stopping",
]);

/** The meeting is running (capture may be on or paused). */
export function isLiveState(state: MeetingState): boolean {
  return LIVE_STATES.has(state);
}

export type HomeView = "start" | "live" | "completed";

/**
 * Which Home page to show: the live meeting, the summary of the meeting that
 * just ended (until dismissed), or the start flow.
 */
export function homeView(
  meeting: Meeting | null,
  state: MeetingState,
  dismissedId: string | null,
): HomeView {
  if (!meeting) return "start";
  if (isLiveState(state)) return "live";
  if ((state === "summarizing" || state === "completed") && meeting.id !== dismissedId)
    return "completed";
  return "start";
}

export function modeLabel(mode: Mode): string {
  return mode === "candidate" ? "Candidate" : "Interviewer";
}

/** Badge text + tone for the live meeting state. */
export function meetingStateBadge(
  state: MeetingState,
  processingStopped: boolean,
  pausedReason: PauseReason | null,
): { label: string; tone: Tone } {
  if (processingStopped || pausedReason === "emergency")
    return { label: "Processing stopped", tone: "danger" };
  switch (state) {
    case "starting":
      return { label: "Starting…", tone: "neutral" };
    case "active":
      return { label: "Listening", tone: "ok" };
    case "paused":
      return { label: "Paused", tone: "warn" };
    case "stopping":
      return { label: "Ending…", tone: "neutral" };
    case "summarizing":
      return { label: "Summarizing", tone: "accent" };
    case "completed":
      return { label: "Completed", tone: "neutral" };
    case "idle":
      return { label: "Idle", tone: "neutral" };
  }
}

/** Milliseconds since the meeting started (never negative). */
export function elapsedMs(startedAt: number, now: number): number {
  return Math.max(0, now - startedAt);
}

export function formatElapsed(startedAt: number, now: number): string {
  return formatOffset(elapsedMs(startedAt, now));
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

/** Spec: user → "You"; remote → the other side of the conversation; unknown → "Speaker". */
export function speakerLabel(role: SpeakerRole, mode: Mode): string {
  switch (role) {
    case "user":
      return "You";
    case "remote":
      return mode === "candidate" ? "Interviewer" : "Candidate";
    case "unknown":
      return "Speaker";
  }
}

/**
 * Segment times are meeting-relative. Guard against absolute (Unix-ms)
 * timestamps so the UI never shows absurd offsets.
 */
export function segmentOffset(time: number, meetingStartedAt: number | null): number {
  if (meetingStartedAt !== null && time >= 1e12 && time >= meetingStartedAt)
    return time - meetingStartedAt;
  return Math.max(0, time);
}

export interface TranscriptGroup {
  /** Stable React key (id of the first segment). */
  key: string;
  role: SpeakerRole;
  speakerId: string | null;
  label: string;
  /** Start of the first segment (as stored). */
  startTime: number;
  segments: TranscriptSegment[];
}

/** A pause longer than this starts a new block even for the same speaker. */
export const GROUP_GAP_MS = 45_000;

/**
 * Orders segments chronologically and merges consecutive segments of the same
 * speaker into blocks (one label per block). Partials stay in place so the
 * in-progress utterance is shown where it started.
 */
export function groupTranscript(
  segments: ReadonlyArray<TranscriptSegment>,
  mode: Mode,
): TranscriptGroup[] {
  const ordered = segments
    .map((s, i) => ({ s, i }))
    .sort((a, b) => a.s.startTime - b.s.startTime || a.i - b.i)
    .map((x) => x.s);
  const groups: TranscriptGroup[] = [];
  for (const seg of ordered) {
    const last = groups[groups.length - 1];
    const prev = last?.segments[last.segments.length - 1];
    if (
      last &&
      prev &&
      last.role === seg.speakerRole &&
      last.speakerId === seg.speakerId &&
      seg.startTime - prev.endTime <= GROUP_GAP_MS
    ) {
      last.segments.push(seg);
      continue;
    }
    groups.push({
      key: seg.id,
      role: seg.speakerRole,
      speakerId: seg.speakerId,
      label: speakerLabel(seg.speakerRole, mode),
      startTime: seg.startTime,
      segments: [seg],
    });
  }
  return groups;
}

/** Scroll position is close enough to the bottom to keep auto-scrolling. */
export function isNearBottom(
  m: { scrollTop: number; scrollHeight: number; clientHeight: number },
  threshold = 40,
): boolean {
  return m.scrollHeight - m.scrollTop - m.clientHeight <= threshold;
}

/**
 * Questions to list in the live view: real questions only, newest first,
 * hiding questions that a later detection extended (`supersedes`).
 */
export function visibleQuestions(questions: ReadonlyArray<DetectedQuestion>): DetectedQuestion[] {
  const superseded = new Set(
    questions.map((q) => q.supersedes).filter((id): id is string => id !== null),
  );
  return questions
    .filter((q) => q.isQuestion && !superseded.has(q.id))
    .slice()
    .reverse();
}

// ---------------------------------------------------------------------------
// Components & health
// ---------------------------------------------------------------------------

const COMPONENT_LABELS: Record<Component, string> = {
  meeting: "Meeting",
  microphone: "Microphone",
  system_audio: "System audio",
  transcription: "Transcription",
  question_detector: "Question detection",
  llm: "AI model",
  vision: "Vision",
  screen: "Screen capture",
  embeddings: "Embeddings",
  database: "Storage",
  overlay: "Overlay",
  capture_privacy: "Capture privacy",
  hotkeys: "Global shortcuts",
};

export function componentLabel(c: Component): string {
  return COMPONENT_LABELS[c];
}

/** Where the user can fix a problem with a component. */
export function componentSection(c: Component): SectionId {
  switch (c) {
    case "microphone":
    case "system_audio":
    case "transcription":
      return "settings/transcription";
    case "llm":
    case "embeddings":
      return "settings/ai";
    case "vision":
    case "screen":
      return "settings/screen";
    case "capture_privacy":
      return "privacy";
    case "hotkeys":
      return "settings/shortcuts";
    case "overlay":
      return "settings/appearance";
    case "meeting":
    case "question_detector":
    case "database":
      return "diagnostics";
  }
}

const WARNING_RANK: Partial<Record<ComponentStatus, number>> = {
  error: 0,
  degraded: 1,
  unsupported: 2,
};

/** Components in error / degraded / unsupported state, most severe first. */
export function componentWarnings(list: ReadonlyArray<ComponentHealth>): ComponentHealth[] {
  return list
    .filter((h) => WARNING_RANK[h.status] !== undefined)
    .slice()
    .sort(
      (a, b) =>
        (WARNING_RANK[a.status] ?? 9) - (WARNING_RANK[b.status] ?? 9) ||
        componentLabel(a.component).localeCompare(componentLabel(b.component)),
    );
}

export function healthOf(
  list: ReadonlyArray<ComponentHealth>,
  c: Component,
): ComponentHealth | undefined {
  return list.find((h) => h.component === c);
}

/** Short status text for an audio/screen source chip. */
export function sourceStatus(status: ComponentStatus | undefined): { label: string; tone: Tone } {
  switch (status) {
    case "active":
    case "streaming":
      return { label: "On", tone: "ok" };
    case "starting":
      return { label: "Starting…", tone: "neutral" };
    case "paused":
      return { label: "Paused", tone: "warn" };
    case "degraded":
      return { label: "Degraded", tone: "warn" };
    case "error":
      return { label: "Error", tone: "danger" };
    case "unsupported":
      return { label: "Unsupported", tone: "warn" };
    case "inactive":
    case "disabled":
    case undefined:
      return { label: "Off", tone: "neutral" };
  }
}

const SCREEN_ON: ReadonlySet<ComponentStatus> = new Set<ComponentStatus>([
  "starting",
  "active",
  "streaming",
  "paused",
  "degraded",
]);

/** Screen context is on for the running meeting (health wins over the saved setting). */
export function screenContextOn(
  components: ReadonlyArray<ComponentHealth>,
  settings: Settings,
): boolean {
  const h = healthOf(components, "screen");
  if (h) return SCREEN_ON.has(h.status);
  return settings.screen.enabled;
}

/**
 * Display name of the configured device: the named device when connected,
 * otherwise the OS default (`null` id).
 */
export function deviceName(
  id: string | null,
  devices: ReadonlyArray<AudioDeviceInfo> | null,
  fallback: string,
): string {
  if (id !== null) {
    if (!devices) return id;
    const d = devices.find((x) => x.id === id);
    return d ? d.name : `${id} (not connected)`;
  }
  const def = devices?.find((x) => x.isDefault);
  return def ? `${fallback} (${def.name})` : fallback;
}

// ---------------------------------------------------------------------------
// Privacy summary
// ---------------------------------------------------------------------------

export function locationBadge(location: DataLocation): { label: string; tone: Tone } {
  switch (location) {
    case "local":
      return { label: "Local", tone: "ok" };
    case "remote":
      return { label: "Cloud", tone: "warn" };
    case "disabled":
      return { label: "Off", tone: "neutral" };
    case "mock":
      return { label: "Simulated", tone: "neutral" };
  }
}

export function retentionLabel(r: RetentionPolicy): string {
  switch (r) {
    case "none":
      return "Not saved — kept in memory only until the next meeting";
    case "one_day":
      return "Saved on this device for 1 day";
    case "seven_days":
      return "Saved on this device for 7 days";
    case "thirty_days":
      return "Saved on this device for 30 days";
    case "forever":
      return "Saved on this device until you delete it";
  }
}

// ---------------------------------------------------------------------------
// Screen source selection
// ---------------------------------------------------------------------------

export const SCREEN_KINDS: ReadonlyArray<{
  value: Exclude<ScreenSourceKind, "region">;
  label: string;
}> = [
  { value: "display", label: "Display" },
  { value: "window", label: "Window" },
  { value: "application", label: "Application" },
];

/** Kind used by the picker tabs (a region is a part of a display). */
export function pickerKind(kind: ScreenSourceKind): Exclude<ScreenSourceKind, "region"> {
  return kind === "region" ? "display" : kind;
}

export function formatRegion(r: Rect): string {
  return `${r.x}, ${r.y}, ${r.width}, ${r.height}`;
}

/**
 * Parses "x, y, width, height" (physical pixels). Empty text → no region.
 * When `bounds` is given the region must fit inside it.
 */
export function parseRegion(
  text: string,
  bounds?: { width: number; height: number },
): { rect: Rect | null; error: string | null } {
  const t = text.trim();
  if (!t) return { rect: null, error: null };
  const parts = t.split(/[\s,x×]+/i).filter(Boolean);
  if (parts.length !== 4 || parts.some((p) => !/^\d+$/.test(p))) {
    return { rect: null, error: "Enter the region as four whole numbers: x, y, width, height." };
  }
  const [x, y, width, height] = parts.map(Number) as [number, number, number, number];
  if (width < 16 || height < 16)
    return { rect: null, error: "The region must be at least 16 × 16 pixels." };
  if (
    bounds &&
    bounds.width > 0 &&
    bounds.height > 0 &&
    (x + width > bounds.width || y + height > bounds.height)
  ) {
    return {
      rect: null,
      error: `The region must fit inside the display (${bounds.width} × ${bounds.height}).`,
    };
  }
  return { rect: { x, y, width, height }, error: null };
}

/** Converts a picked source (+ optional display region) to the request spec. */
export function sourceToSpec(info: ScreenSourceInfo, region: Rect | null): ScreenSourceSpec {
  const useRegion = region !== null && (info.kind === "display" || info.kind === "region");
  return {
    kind: useRegion ? "region" : info.kind,
    id: info.id,
    label: info.label,
    region: useRegion ? region : null,
  };
}

/** Human description of a source for selects ("Display 1 · 2560 × 1440 · primary"). */
export function sourceOptionLabel(info: ScreenSourceInfo): string {
  const parts: string[] = [];
  if (info.kind === "window") {
    parts.push(info.title || info.label);
    if (info.application && info.application !== info.title) parts.push(info.application);
  } else {
    parts.push(info.label);
    if (info.width > 0 && info.height > 0 && info.kind === "display")
      parts.push(`${info.width} × ${info.height}`);
    if (info.isPrimary) parts.push("primary");
  }
  if (info.excludedByRule) parts.push("excluded by a privacy rule");
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------
// Start meeting form
// ---------------------------------------------------------------------------

export interface StartForm {
  mode: Mode;
  title: string;
  microphone: boolean;
  systemAudio: boolean;
  screenEnabled: boolean;
  screen: ScreenSourceSpec | null;
  /** Error from the screen picker (e.g. an invalid region). */
  screenError: string | null;
  /** `MeetingAppInfo.id` to bring to the front after start. */
  focusAppId: string | null;
}

export function initialStartForm(settings: Settings): StartForm {
  return {
    mode: settings.mode,
    title: "",
    microphone: settings.audio.microphoneEnabled,
    systemAudio: settings.audio.systemAudioEnabled,
    // Screen context stays off unless the user explicitly turned it on.
    screenEnabled: settings.screen.enabled && settings.screen.source !== null,
    screen: settings.screen.source,
    screenError: null,
    focusAppId: null,
  };
}

/** Returns a user-facing problem with the form, or null when it can be submitted. */
export function validateStartForm(form: StartForm, systemAudioSupported: boolean): string | null {
  const systemAudio = form.systemAudio && systemAudioSupported;
  if (form.screenEnabled) {
    if (form.screenError) return form.screenError;
    if (!form.screen) return "Choose a screen source, or turn screen context off.";
  }
  if (!form.microphone && !systemAudio && !form.screenEnabled) {
    return "Turn on the microphone, system audio or screen context so OpenMeet has something to follow.";
  }
  return null;
}

export function buildStartRequest(
  form: StartForm,
  systemAudioSupported: boolean,
): StartMeetingRequest {
  const title = form.title.trim();
  return {
    mode: form.mode,
    title: title ? title : null,
    microphone: form.microphone,
    systemAudio: form.systemAudio && systemAudioSupported,
    screen: form.screenEnabled ? form.screen : null,
    focusAppId: form.focusAppId,
  };
}
