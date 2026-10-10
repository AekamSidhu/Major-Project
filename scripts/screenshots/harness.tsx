// Screenshot harness: renders the real overlay and main-window components in
// a browser with a mocked Tauri backend serving demo data. Used by
// capture.mjs to produce docs/screenshots/*.png.
//
//   ?scene=overlay-candidate | overlay-interviewer | main-live | main-privacy

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { EngineSnapshot } from "@/bindings";
import "./shots.css";
import { OverlayApp } from "@/overlay/OverlayApp";
import { App } from "@/main/App";
import { useMain } from "@/main/store";
import { useOverlay } from "@/overlay/store";
import { DATA_FLOW_LOCAL, MEETINGS, candidateSnapshot, interviewerSnapshot, settings } from "./fixtures";

const scene = new URLSearchParams(location.search).get("scene") ?? "overlay-candidate";
const overlay = scene.startsWith("overlay");
const interviewer = scene === "overlay-interviewer" || scene === "main-live";
const snapshot: EngineSnapshot = interviewer ? interviewerSnapshot() : candidateSnapshot();
// Tall enough that the candidate answer's controls are in the picture.
const appSettings = settings(interviewer ? "interviewer" : "candidate", scene === "overlay-candidate" ? 760 : 560);

mockWindows(overlay ? "overlay" : "main");
mockIPC(
  (cmd) => {
    switch (cmd) {
      case "get_app_info":
        return { version: "0.1.0", platform: "windows", arch: "x86_64", developerMode: false, debugBuild: false };
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
        return DATA_FLOW_LOCAL;
      case "list_privacy_rules":
        return [
          { id: "r1", type: "application", value: "1Password", action: "exclude", createdAt: Date.UTC(2026, 9, 1) },
          { id: "r2", type: "domain", value: "bank.example.com", action: "exclude", createdAt: Date.UTC(2026, 9, 1) },
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
  if (scene === "main-privacy") {
    // Open the section once the shell has loaded.
    const unsub = useMain.subscribe((s) => {
      if (s.ready) {
        unsub();
        s.setSection("privacy");
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
if (overlay) markReadyWhen(() => useOverlay.getState().hydrated && !!useOverlay.getState().settings, 400);
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
