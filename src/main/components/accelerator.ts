// Shortcut recording: converts browser keyboard events into Tauri global
// shortcut accelerators ("CommandOrControl+Shift+K") and detects conflicts.
// Key names follow the `global-hotkey` crate parser used by Tauri.

import type { ShortcutAction } from "@/bindings";

const NAMED_CODES: Record<string, string> = {
  Space: "Space",
  Enter: "Enter",
  NumpadEnter: "NumpadEnter",
  Escape: "Escape",
  Backspace: "Backspace",
  Tab: "Tab",
  Delete: "Delete",
  Insert: "Insert",
  Home: "Home",
  End: "End",
  PageUp: "PageUp",
  PageDown: "PageDown",
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
  Minus: "Minus",
  Equal: "Equal",
  Comma: "Comma",
  Period: "Period",
  Slash: "Slash",
  Backslash: "Backslash",
  Semicolon: "Semicolon",
  Quote: "Quote",
  Backquote: "Backquote",
  BracketLeft: "BracketLeft",
  BracketRight: "BracketRight",
  PrintScreen: "PrintScreen",
  ScrollLock: "ScrollLock",
  Pause: "Pause",
  NumpadAdd: "NumpadAdd",
  NumpadSubtract: "NumpadSubtract",
  NumpadMultiply: "NumpadMultiply",
  NumpadDivide: "NumpadDivide",
  NumpadDecimal: "NumpadDecimal",
  NumpadEqual: "NumpadEqual",
  AudioVolumeMute: "AudioVolumeMute",
  AudioVolumeDown: "AudioVolumeDown",
  AudioVolumeUp: "AudioVolumeUp",
  MediaPlayPause: "MediaPlayPause",
  MediaStop: "MediaStop",
  MediaTrackNext: "MediaTrackNext",
  MediaTrackPrevious: "MediaTrackPrevious",
};

const MODIFIER_CODES = new Set([
  "ShiftLeft",
  "ShiftRight",
  "ControlLeft",
  "ControlRight",
  "AltLeft",
  "AltRight",
  "MetaLeft",
  "MetaRight",
  "OSLeft",
  "OSRight",
  "CapsLock",
  "NumLock",
  "Fn",
  "FnLock",
]);

/**
 * Tauri key name for a `KeyboardEvent.code`; null for modifier keys and keys
 * a global shortcut cannot use.
 */
export function keyFromCode(code: string): string | null {
  if (MODIFIER_CODES.has(code)) return null;
  let m = /^Key([A-Z])$/.exec(code);
  if (m) return m[1] ?? null;
  m = /^Digit([0-9])$/.exec(code);
  if (m) return m[1] ?? null;
  m = /^F([1-9]|1[0-9]|2[0-4])$/.exec(code);
  if (m) return code;
  m = /^Numpad([0-9])$/.exec(code);
  if (m) return code;
  return NAMED_CODES[code] ?? null;
}

export function isFunctionKey(key: string): boolean {
  return /^F([1-9]|1[0-9]|2[0-4])$/.test(key);
}

export interface KeyComboInput {
  code: string;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

export type RecordResult =
  | { kind: "partial"; modifiers: string[] }
  | { kind: "unsupported"; code: string; modifiers: string[] }
  | { kind: "complete"; accelerator: string; modifiers: string[]; key: string };

/**
 * Builds an accelerator from a keydown. On macOS ⌘ maps to CommandOrControl
 * and ⌃ to Control; elsewhere Ctrl maps to CommandOrControl and the Windows /
 * Super key to Super.
 */
export function acceleratorFromEvent(e: KeyComboInput, isMac: boolean): RecordResult {
  const modifiers: string[] = [];
  if (isMac) {
    if (e.metaKey) modifiers.push("CommandOrControl");
    if (e.ctrlKey) modifiers.push("Control");
  } else if (e.ctrlKey) {
    modifiers.push("CommandOrControl");
  }
  if (e.altKey) modifiers.push("Alt");
  if (e.shiftKey) modifiers.push("Shift");
  if (!isMac && e.metaKey) modifiers.push("Super");
  if (MODIFIER_CODES.has(e.code) || e.code === "") return { kind: "partial", modifiers };
  const key = keyFromCode(e.code);
  if (!key) return { kind: "unsupported", code: e.code, modifiers };
  return { kind: "complete", accelerator: [...modifiers, key].join("+"), modifiers, key };
}

/**
 * A global shortcut must not hijack normal typing: it needs Ctrl/⌘, Alt or
 * Super, unless the key is a function key. Null when acceptable.
 */
export function validateAccelerator(accelerator: string): string | null {
  const parts = accelerator.split("+").map((p) => p.trim().toLowerCase()).filter(Boolean);
  if (parts.length === 0) return "Press a key combination.";
  const key = parts[parts.length - 1] ?? "";
  const mods = parts.slice(0, -1);
  const strong = mods.some((m) =>
    ["commandorcontrol", "cmdorctrl", "cmdorcontrol", "commandorctrl", "control", "ctrl", "alt", "option", "super", "command", "cmd", "meta"].includes(m),
  );
  if (!strong && !isFunctionKey(key.toUpperCase())) {
    return "Include Ctrl, Alt or the Windows/⌘ key so the shortcut doesn't interfere with typing.";
  }
  return null;
}

/**
 * Canonical form for comparison (mirror of `normalize_accelerator` in
 * openmeet-core): modifier aliases unified and sorted, case-insensitive.
 * Empty string = unbound.
 */
export function normalizeAccelerator(acc: string): string {
  const mods: string[] = [];
  let key = "";
  for (const raw of acc.split("+").map((s) => s.trim()).filter(Boolean)) {
    const lower = raw.toLowerCase();
    let m: string | null = null;
    if (["commandorcontrol", "cmdorctrl", "cmdorcontrol", "commandorctrl"].includes(lower)) m = "cmdorctrl";
    else if (lower === "control" || lower === "ctrl") m = "ctrl";
    else if (["command", "cmd", "super", "meta"].includes(lower)) m = "super";
    else if (lower === "shift") m = "shift";
    else if (lower === "alt" || lower === "option") m = "alt";
    if (m) mods.push(m);
    else key = lower === "esc" ? "escape" : lower === "return" ? "enter" : lower;
  }
  if (!key) return "";
  const unique = [...new Set(mods)].sort();
  return [...unique, key].join("+");
}

/** Actions (other than `action`) already bound to `accelerator`. */
export function conflictsFor(
  action: ShortcutAction,
  accelerator: string,
  bindings: ReadonlyArray<{ action: ShortcutAction; accelerator: string }>,
): ShortcutAction[] {
  const target = normalizeAccelerator(accelerator);
  if (!target) return [];
  return bindings
    .filter((b) => b.action !== action && normalizeAccelerator(b.accelerator) === target)
    .map((b) => b.action);
}

/** Map of action → actions sharing its accelerator. */
export function duplicateMap(
  bindings: ReadonlyArray<{ action: ShortcutAction; accelerator: string }>,
): Partial<Record<ShortcutAction, ShortcutAction[]>> {
  const out: Partial<Record<ShortcutAction, ShortcutAction[]>> = {};
  for (const b of bindings) {
    const others = conflictsFor(b.action, b.accelerator, bindings);
    if (others.length > 0) out[b.action] = others;
  }
  return out;
}

/** True on macOS (⌘ is CommandOrControl there). */
export const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
