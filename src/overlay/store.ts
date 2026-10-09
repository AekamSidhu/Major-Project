// Overlay store (Zustand). Holds the reduced meeting state (`core`), the
// settings that drive appearance and the overlay-local UI state (expanded /
// minimal, Ask input, click-through). Meeting events go through the pure
// `reduceOverlay`; this module adds hydration, render-latency reporting and
// UI-command handling.

import { create } from "zustand";
import type { MeetingEvent, NoticeLevel, Settings, UiCommand } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import {
  INITIAL_STATE,
  hydrateFromSnapshot,
  isFirstContent,
  reduceOverlay,
  replayAfterHydration,
  type OverlayAction,
  type OverlayState,
} from "./state";

export interface OverlayStore {
  core: OverlayState;
  /** Snapshot applied; events are buffered until then. */
  hydrated: boolean;
  settings: Settings | null;
  settingsError: string | null;
  /** Expanded card (true) or minimal one-line HUD (false). */
  expanded: boolean;
  askOpen: boolean;
  /** Incremented whenever the Ask input should (re)take keyboard focus. */
  askFocusNonce: number;
  clickThrough: boolean;
  visible: boolean;
  /** Incremented to force the next size report (e.g. after the window is shown). */
  resizeNonce: number;

  dispatch: (action: OverlayAction) => void;
  applyEvent: (ev: MeetingEvent) => void;
  applyUiCommand: (cmd: UiCommand) => void;
  init: () => Promise<void>;
  loadSettings: () => Promise<void>;
  setExpanded: (expanded: boolean) => void;
  notify: (level: NoticeLevel, message: string) => void;
  openAsk: () => Promise<void>;
  closeAsk: () => Promise<void>;
}

let buffered: MeetingEvent[] = [];
let initGeneration = 0;

function nextFrame(cb: () => void): void {
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => cb());
  else setTimeout(cb, 16);
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/** Reports event-arrival → next animation frame for the first content of a request. */
function reportLatency(start: number): void {
  nextFrame(() => {
    const ms = Math.max(0, Math.round((now() - start) * 10) / 10);
    // Telemetry only: a failed report must not disturb the meeting UI.
    api.reportRenderLatency(ms).catch(() => undefined);
  });
}

export const useOverlay = create<OverlayStore>((set, get) => ({
  core: INITIAL_STATE,
  hydrated: false,
  settings: null,
  settingsError: null,
  expanded: true,
  askOpen: false,
  askFocusNonce: 0,
  clickThrough: false,
  visible: true,
  resizeNonce: 0,

  dispatch: (action) => {
    const core = get().core;
    const next = reduceOverlay(core, action);
    if (next !== core) set({ core: next });
  },

  applyEvent: (ev) => {
    if (!get().hydrated) {
      buffered.push(ev);
      return;
    }
    const start = now();
    const core = get().core;
    const first = isFirstContent(core, ev);
    const next = reduceOverlay(core, ev);
    if (next !== core) set({ core: next });
    if (first) reportLatency(start);
  },

  applyUiCommand: (cmd) => {
    switch (cmd.type) {
      case "FOCUS_ASK":
        void get().openAsk();
        break;
      case "TOGGLE_EXPANDED":
        set((s) => ({ expanded: !s.expanded }));
        break;
      case "CLICK_THROUGH_CHANGED":
        set({ clickThrough: cmd.enabled });
        break;
      case "OVERLAY_VISIBILITY_CHANGED":
        set((s) => ({ visible: cmd.visible, resizeNonce: cmd.visible ? s.resizeNonce + 1 : s.resizeNonce }));
        if (!cmd.visible && get().askOpen) void get().closeAsk();
        break;
      case "SETTINGS_CHANGED":
        void get().loadSettings();
        break;
      case "NAVIGATE":
        // Main-window navigation; nothing to do in the overlay.
        break;
    }
  },

  init: async () => {
    const generation = ++initGeneration;
    buffered = [];
    set({ hydrated: false });
    const [snapshot, privacy] = await Promise.allSettled([
      api.getSnapshot(),
      api.getCapturePrivacyStatus(),
      get().loadSettings(),
    ]);
    if (generation !== initGeneration) return;
    let core = get().core;
    let activeId: string | null = null;
    if (snapshot.status === "fulfilled") {
      core = hydrateFromSnapshot(core, snapshot.value);
      activeId = snapshot.value.active?.requestId ?? null;
    } else {
      core = reduceOverlay(core, {
        type: "LOCAL_NOTICE",
        level: "error",
        message: `Could not load the meeting state: ${errorMessage(snapshot.reason)}`,
      });
    }
    if (privacy.status === "fulfilled") {
      core = reduceOverlay(core, { type: "LOCAL_PRIVACY_STATUS", status: privacy.value });
    }
    core = replayAfterHydration(core, buffered, activeId);
    buffered = [];
    set({ core, hydrated: true });
  },

  loadSettings: async () => {
    try {
      const settings = await api.getSettings();
      const prev = get().settings;
      set((s) => ({
        settings,
        settingsError: null,
        clickThrough: settings.overlay.clickThrough,
        // Follow the "minimal" preference when it is first read or changed.
        expanded:
          !prev || prev.overlay.minimal !== settings.overlay.minimal
            ? !settings.overlay.minimal
            : s.expanded,
      }));
    } catch (e) {
      const message = errorMessage(e);
      set({ settingsError: message });
      get().notify("error", `Could not load settings: ${message}`);
    }
  },

  setExpanded: (expanded) => set({ expanded }),

  notify: (level, message) => get().dispatch({ type: "LOCAL_NOTICE", level, message }),

  openAsk: async () => {
    set((s) => ({ askOpen: true, expanded: true, askFocusNonce: s.askFocusNonce + 1 }));
    try {
      // Makes the non-activating overlay focusable so the input can take keys.
      await api.overlayBeginInput();
      set((s) => ({ askFocusNonce: s.askFocusNonce + 1 }));
    } catch (e) {
      set({ askOpen: false });
      get().notify("error", `Can't open Ask OpenMeet: ${errorMessage(e)}`);
    }
  },

  closeAsk: async () => {
    if (!get().askOpen) return;
    set({ askOpen: false });
    try {
      await api.overlayEndInput();
    } catch (e) {
      get().notify("error", `Couldn't return focus to the meeting: ${errorMessage(e)}`);
    }
  },
}));

/** Test helper: resets the store and module state. */
export function resetOverlayStore(): void {
  buffered = [];
  initGeneration = 0;
  useOverlay.setState({
    core: INITIAL_STATE,
    hydrated: false,
    settings: null,
    settingsError: null,
    expanded: true,
    askOpen: false,
    askFocusNonce: 0,
    clickThrough: false,
    visible: true,
    resizeNonce: 0,
  });
}
