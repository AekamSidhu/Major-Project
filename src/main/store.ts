// Main-window state (Zustand): settings, navigation, live meeting state and
// notices. Section components read from here and call `api` for actions.

import { create } from "zustand";
import type {
  AppInfo,
  ComponentHealth,
  DetectedQuestion,
  EngineSnapshot,
  Meeting,
  MeetingEvent,
  MeetingState,
  MeetingSummary,
  Mode,
  AnswerStyle,
  PauseReason,
  Settings,
  Topic,
  TranscriptSegment,
  AudioSourceKind,
} from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";

export type SectionId =
  | "home"
  | "history"
  | "privacy"
  | "settings/ai"
  | "settings/transcription"
  | "settings/screen"
  | "settings/appearance"
  | "settings/shortcuts"
  | "settings/profile"
  | "settings/general"
  | "diagnostics";

export const SECTIONS: ReadonlyArray<SectionId> = [
  "home",
  "history",
  "privacy",
  "settings/ai",
  "settings/transcription",
  "settings/screen",
  "settings/appearance",
  "settings/shortcuts",
  "settings/profile",
  "settings/general",
  "diagnostics",
];

export function isSectionId(s: string): s is SectionId {
  return (SECTIONS as ReadonlyArray<string>).includes(s);
}

export interface Notice {
  id: number;
  level: "info" | "warning" | "error";
  message: string;
}

const MAX_TRANSCRIPT = 400;
const MAX_NOTICES = 4;

export interface LiveState {
  meeting: Meeting | null;
  state: MeetingState;
  pausedReason: PauseReason | null;
  mode: Mode;
  style: AnswerStyle;
  /** Final segments plus the latest partial per utterance (partials replaced by finals). */
  transcript: TranscriptSegment[];
  questions: DetectedQuestion[];
  topics: Topic[];
  suggestedNextTopic: string | null;
  components: ComponentHealth[];
  lastSummary: MeetingSummary | null;
  processingStopped: boolean;
  demoRunning: boolean;
  levels: Partial<Record<AudioSourceKind, number>>;
}

const EMPTY_LIVE: LiveState = {
  meeting: null,
  state: "idle",
  pausedReason: null,
  mode: "candidate",
  style: "concise",
  transcript: [],
  questions: [],
  topics: [],
  suggestedNextTopic: null,
  components: [],
  lastSummary: null,
  processingStopped: false,
  demoRunning: false,
  levels: {},
};

export function liveFromSnapshot(s: EngineSnapshot): LiveState {
  return {
    meeting: s.meeting,
    state: s.state,
    pausedReason: s.pausedReason,
    mode: s.mode,
    style: s.style,
    transcript: s.transcript,
    questions: s.questions,
    topics: s.topics,
    suggestedNextTopic: s.suggestedNextTopic,
    components: s.components,
    lastSummary: s.lastSummary,
    processingStopped: s.processingStopped,
    demoRunning: s.demoRunning,
    levels: {},
  };
}

function upsertSegment(list: TranscriptSegment[], seg: TranscriptSegment): TranscriptSegment[] {
  // A final replaces the partial of the same utterance.
  const idx = list.findIndex(
    (s) => s.utteranceId === seg.utteranceId && (!s.isFinal || s.id === seg.id),
  );
  let next: TranscriptSegment[];
  if (idx >= 0) {
    next = list.slice();
    next[idx] = seg;
  } else {
    next = [...list, seg];
  }
  if (seg.isFinal) next.sort((a, b) => a.startTime - b.startTime);
  return next.length > MAX_TRANSCRIPT ? next.slice(next.length - MAX_TRANSCRIPT) : next;
}

/** Pure reducer for meeting events (unit-testable). */
export function reduceLive(live: LiveState, ev: MeetingEvent): LiveState {
  switch (ev.type) {
    case "MEETING_STATE_CHANGED":
      // Idle without a meeting id: the meeting was deleted; forget it.
      if (ev.state === "idle" && ev.meetingId === null) {
        return {
          ...EMPTY_LIVE,
          mode: ev.mode,
          style: live.style,
          components: live.components,
          processingStopped: live.processingStopped,
        };
      }
      return {
        ...live,
        state: ev.state,
        mode: ev.mode,
        pausedReason: ev.pausedReason,
        meeting:
          live.meeting && ev.meetingId === live.meeting.id
            ? { ...live.meeting, state: ev.state }
            : live.meeting,
        processingStopped: ev.state === "active" ? false : live.processingStopped,
      };
    case "MEETING_STARTED":
      return {
        ...EMPTY_LIVE,
        meeting: ev.meeting,
        state: ev.meeting.state,
        mode: ev.meeting.mode,
        style: live.style,
        components: live.components,
        demoRunning: ev.meeting.title.startsWith("Demo:"),
      };
    case "MEETING_STOPPED":
      return { ...live, demoRunning: false };
    case "TRANSCRIPT_PARTIAL":
    case "TRANSCRIPT_FINAL":
      if (live.meeting && ev.segment.meetingId !== live.meeting.id) return live;
      return { ...live, transcript: upsertSegment(live.transcript, ev.segment) };
    case "QUESTION_DETECTED":
      return { ...live, questions: [...live.questions, ev.question].slice(-50) };
    case "TOPICS_UPDATED":
      return { ...live, topics: ev.topics, suggestedNextTopic: ev.suggestedNext };
    case "COMPONENT_STATUS_CHANGED": {
      const others = live.components.filter((c) => c.component !== ev.health.component);
      return { ...live, components: [...others, ev.health] };
    }
    case "SUMMARY_READY":
      return { ...live, lastSummary: ev.summary };
    case "PROCESSING_STOPPED":
      return { ...live, processingStopped: true, demoRunning: false };
    case "MODE_CHANGED":
      return { ...live, mode: ev.mode };
    case "STYLE_CHANGED":
      return { ...live, style: ev.style };
    case "AUDIO_LEVEL":
      return { ...live, levels: { ...live.levels, [ev.source]: ev.level } };
    default:
      return live;
  }
}

interface MainStore {
  ready: boolean;
  loadError: string | null;
  appInfo: AppInfo | null;
  settings: Settings | null;
  savingError: string | null;
  section: SectionId;
  live: LiveState;
  notices: Notice[];

  init: () => Promise<void>;
  reloadSettings: () => Promise<void>;
  /**
   * Applies `mutate` to a copy of the settings, updates the UI optimistically
   * and persists. On failure the previous settings are restored and the
   * error is shown. Resolves to true on success.
   */
  updateSettings: (mutate: (draft: Settings) => void) => Promise<boolean>;
  setSection: (section: SectionId) => void;
  refreshSnapshot: () => Promise<void>;
  applyEvent: (ev: MeetingEvent) => void;
  notify: (level: Notice["level"], message: string) => void;
  dismissNotice: (id: number) => void;
}

let noticeSeq = 1;
let saveSeq = 0;

export const useMain = create<MainStore>((set, get) => ({
  ready: false,
  loadError: null,
  appInfo: null,
  settings: null,
  savingError: null,
  section: "home",
  live: EMPTY_LIVE,
  notices: [],

  init: async () => {
    try {
      const [appInfo, settings, snapshot] = await Promise.all([
        api.getAppInfo(),
        api.getSettings(),
        api.getSnapshot(),
      ]);
      set({ appInfo, settings, live: liveFromSnapshot(snapshot), ready: true, loadError: null });
    } catch (e) {
      set({ loadError: errorMessage(e), ready: true });
    }
  },

  reloadSettings: async () => {
    try {
      set({ settings: await api.getSettings() });
    } catch (e) {
      get().notify("error", errorMessage(e));
    }
  },

  updateSettings: async (mutate) => {
    const prev = get().settings;
    if (!prev) return false;
    const draft = structuredClone(prev);
    mutate(draft);
    const seq = ++saveSeq;
    set({ settings: draft, savingError: null });
    try {
      const stored = await api.updateSettings(draft);
      // Ignore responses of superseded saves.
      if (seq === saveSeq) set({ settings: stored });
      return true;
    } catch (e) {
      const msg = errorMessage(e);
      if (seq === saveSeq) set({ settings: prev, savingError: msg });
      get().notify("error", `Settings not saved: ${msg}`);
      return false;
    }
  },

  setSection: (section) => set({ section }),

  refreshSnapshot: async () => {
    try {
      const snapshot = await api.getSnapshot();
      set((s) => ({ live: { ...liveFromSnapshot(snapshot), levels: s.live.levels } }));
    } catch (e) {
      get().notify("error", errorMessage(e));
    }
  },

  applyEvent: (ev) => {
    if (ev.type === "NOTICE") {
      get().notify(ev.level, ev.message);
      return;
    }
    set((s) => ({ live: reduceLive(s.live, ev) }));
  },

  notify: (level, message) =>
    set((s) => ({
      notices: [...s.notices, { id: noticeSeq++, level, message }].slice(-MAX_NOTICES),
    })),

  dismissNotice: (id) => set((s) => ({ notices: s.notices.filter((n) => n.id !== id) })),
}));
