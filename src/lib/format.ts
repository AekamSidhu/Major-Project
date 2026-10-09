// Small formatting helpers shared by both windows.

import type { QuestionCategory, ShortcutAction } from "@/bindings";

/** "mm:ss" (or "h:mm:ss") for meeting-relative milliseconds. */
export function formatOffset(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(h > 0 ? 2 : 1, "0");
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Locale date-time for a Unix-ms timestamp. */
export function formatDateTime(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Human duration between two Unix-ms timestamps ("42 min", "1 h 5 min"). */
export function formatDuration(startMs: number, endMs: number | null): string {
  if (endMs === null) return "in progress";
  const min = Math.max(0, Math.round((endMs - startMs) / 60000));
  if (min < 1) return "< 1 min";
  if (min < 60) return `${min} min`;
  return `${Math.floor(min / 60)} h ${min % 60} min`;
}

const CATEGORY_LABELS: Record<QuestionCategory, string> = {
  technical: "Technical",
  coding: "Coding",
  system_design: "System Design",
  behavioral: "Behavioral",
  experience: "Experience",
  product: "Product",
  leadership: "Leadership",
  clarification: "Clarification",
  architecture: "Architecture",
  tradeoff: "Trade-off",
  culture: "Culture",
  general: "General",
  follow_up: "Follow-up",
};

export function categoryLabel(c: QuestionCategory): string {
  return CATEGORY_LABELS[c];
}

const SHORTCUT_LABELS: Record<ShortcutAction, string> = {
  toggle_overlay: "Show / hide overlay",
  generate_answer: "Generate answer",
  generate_followups: "Generate follow-ups",
  ask_ai: "Ask OpenMeet",
  toggle_mode: "Toggle candidate / interviewer mode",
  toggle_click_through: "Toggle click-through",
  capture_screen: "Capture screen manually",
  pause_resume: "Pause / resume processing",
  emergency_stop: "Emergency privacy stop",
  expand_hud: "Expand / collapse HUD",
  cycle_style: "Cycle answer style",
};

export function shortcutLabel(a: ShortcutAction): string {
  return SHORTCUT_LABELS[a];
}

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

/** Renders a Tauri accelerator for display ("CommandOrControl+Shift+A" -> "Ctrl+Shift+A" / "⌘⇧A"). */
export function formatAccelerator(acc: string): string {
  const parts = acc.split("+").map((p) => p.trim()).filter(Boolean);
  const map = (p: string): string => {
    const l = p.toLowerCase();
    if (["commandorcontrol", "cmdorctrl", "cmdorcontrol", "commandorctrl"].includes(l))
      return isMac ? "⌘" : "Ctrl";
    if (l === "control" || l === "ctrl") return isMac ? "⌃" : "Ctrl";
    if (l === "command" || l === "cmd" || l === "super" || l === "meta") return isMac ? "⌘" : "Win";
    if (l === "shift") return isMac ? "⇧" : "Shift";
    if (l === "alt" || l === "option") return isMac ? "⌥" : "Alt";
    if (l === "escape" || l === "esc") return "Esc";
    if (l === "return" || l === "enter") return "Enter";
    if (l === "backspace") return isMac ? "⌫" : "Backspace";
    if (l === "space") return "Space";
    return p.length === 1 ? p.toUpperCase() : p;
  };
  return parts.map(map).join(isMac ? "" : "+");
}
