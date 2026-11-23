// Screenshot harness: renders the real overlay and main-window components in
// a browser with a mocked Tauri backend serving demo data. Used by
// capture.mjs to produce docs/screenshots/*.png.
//
//   ?scene=overlay-candidate | overlay-interviewer | overlay-integrity | main-live
//          | overlay-workplace | main-minutes | main-privacy | main-integrity
//          | main-home | main-history | main-ai | main-appearance (previews only)

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { EngineSnapshot } from "@/bindings";
import "./shots.css";
import { OverlayApp } from "@/overlay/OverlayApp";
import { App } from "@/main/App";
import { useMain } from "@/main/store";
import { useOverlay } from "@/overlay/store";
import {
  DATA_FLOW_INTEGRITY,
  DATA_FLOW_LOCAL,
  DISPLAYS,
  INTEGRITY_STATUS,
  MEETINGS,
  candidateSnapshot,
  integritySnapshot,
  interviewerSnapshot,
  minutesSnapshot,
  settings,
  workplaceSnapshot,
} from "./fixtures";

const scene = new URLSearchParams(location.search).get("scene") ?? "overlay-candidate";
const overlay = scene.startsWith("overlay");
const integrity = scene === "overlay-integrity" || scene === "main-integrity";
const interviewer =
  scene === "overlay-interviewer" || scene === "main-live" || scene === "overlay-integrity";
// Scenes that show the main window between meetings get an idle engine.
const SECTION_SCENES: Record<
  string,
  "home" | "history" | "privacy" | "settings/ai" | "settings/appearance" | "settings/integrity"
> = {
  "main-integrity": "settings/integrity",
  "main-privacy": "privacy",
  "main-home": "home",
  "main-history": "history",
  "main-ai": "settings/ai",
  "main-appearance": "settings/appearance",
};
const workplace = scene === "overlay-workplace";
const minutes = scene === "main-minutes";
const idle = scene in SECTION_SCENES && scene !== "main-privacy";
const liveSnapshot: EngineSnapshot = integrity
  ? integritySnapshot()
  : workplace
    ? workplaceSnapshot()
    : minutes
      ? minutesSnapshot()
      : interviewer
        ? interviewerSnapshot()
        : candidateSnapshot();
const snapshot: EngineSnapshot = idle
  ? {
      ...liveSnapshot,
      meeting: null,
      state: "idle",
      transcript: [],
      questions: [],
      topics: [],
      active: null,
    }
  : liveSnapshot;
// Tall enough that the candidate answer's controls are in the picture.
const appSettings = settings(
  workplace || minutes ? "workplace" : interviewer || integrity ? "interviewer" : "candidate",
  scene === "overlay-candidate" || scene === "overlay-integrity" ? 760 : 560,
);
if (integrity) {
  appSettings.integrity.enabled = true;
  appSettings.ai.embeddings.provider = "local_ai";
  appSettings.integrity.candidateRegion = { x: 3200, y: 0, width: 1280, height: 720 };
}

mockWindows(overlay ? "overlay" : "main");
mockIPC(
  (cmd) => {
    switch (cmd) {
      case "get_app_info":
        return {
          version: "0.1.0",
          platform: "windows",
          arch: "x86_64",
          developerMode: false,
          debugBuild: false,
        };
      case "get_settings":
        return appSettings;
      case "get_snapshot":
        return snapshot;
      case "get_capture_privacy_status":
        return {
          supported: true,
          requested: true,
          active: true,
          platform: "windows",
          mechanism: "SetWindowDisplayAffinity (WDA_EXCLUDEFROMCAPTURE)",
          note: "Windows 10 version 2004 or later.",
        };
      case "get_data_flow":
        return integrity ? DATA_FLOW_INTEGRITY : DATA_FLOW_LOCAL;
      case "get_integrity_status":
        return INTEGRITY_STATUS;
      case "list_displays":
        return DISPLAYS;
      case "list_privacy_rules":
        return [
          {
            id: "r1",
            type: "application",
            value: "1Password",
            action: "exclude",
            createdAt: Date.UTC(2026, 9, 1),
          },
          {
            id: "r2",
            type: "domain",
            value: "bank.example.com",
            action: "exclude",
            createdAt: Date.UTC(2026, 9, 1),
          },
        ];
      case "suggested_privacy_apps":
        return ["Bitwarden", "KeePassXC"];
      case "get_api_key_status":
        return [];
      case "list_demo_scenarios":
      case "detect_meeting_apps":
        return [];
      case "list_meetings":
        return MEETINGS;
      default:
        return null;
    }
  },
  { shouldMockEvents: true },
);

if (overlay) {
  document.documentElement.classList.add("overlay", "shot-overlay");
} else {
  document.documentElement.classList.add("shot-main");
  const target = SECTION_SCENES[scene];
  if (target) {
    // Open the section once the shell has loaded.
    const unsub = useMain.subscribe((s) => {
      if (s.ready) {
        unsub();
        s.setSection(target);
      }
    });
  }
}

// capture.mjs waits for this flag before taking the screenshot.
declare global {
  interface Window {
    __shotReady?: boolean;
  }
}
function markReadyWhen(ready: () => boolean, settleMs: number) {
  const check = () => {
    if (ready()) window.setTimeout(() => (window.__shotReady = true), settleMs);
    else window.setTimeout(check, 50);
  };
  check();
}
if (overlay)
  markReadyWhen(() => useOverlay.getState().hydrated && !!useOverlay.getState().settings, 400);
else markReadyWhen(() => useMain.getState().ready, 1200);

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      {overlay ? (
        <div className="shot-stage">
          <div className="shot-overlay-slot">
            <OverlayApp />
          </div>
        </div>
      ) : (
        <App />
      )}
    </StrictMode>,
  );
}
