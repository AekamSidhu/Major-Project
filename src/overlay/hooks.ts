// Window-integration hooks for the overlay: content-driven window size,
// remembering the position after a drag, and hotkey labels.

import {
  useEffect,
  useMemo,
  useRef,
  type MouseEvent as ReactMouseEvent,
  type RefObject,
} from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { ShortcutAction } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatAccelerator } from "@/lib/format";
import { useOverlay } from "./store";

// ---------------------------------------------------------------------------
// Auto-size
// ---------------------------------------------------------------------------

export interface Size {
  w: number;
  h: number;
}

/** Minimum change (logical px) worth a resize call. */
export const SIZE_THRESHOLD = 4;
/** Minimum interval between two resize calls. */
export const SIZE_THROTTLE_MS = 60;

/** True when `next` differs from the last reported size by at least the threshold. */
export function sizeChanged(last: Size | null, next: Size, threshold = SIZE_THRESHOLD): boolean {
  if (!last) return true;
  return Math.abs(next.w - last.w) >= threshold || Math.abs(next.h - last.h) >= threshold;
}

/**
 * Observes `ref` and reports its size with `api.overlaySetSize` (throttled,
 * only on changes ≥ SIZE_THRESHOLD). `nonce` changes force a fresh report.
 */
export function useAutoSize(ref: RefObject<HTMLElement | null>, nonce: number): void {
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    let last: Size | null = null;
    let pending: Size | null = null;
    let lastSentAt = 0;
    let lastError: string | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const flush = () => {
      timer = undefined;
      if (!pending || !sizeChanged(last, pending)) return;
      const size = pending;
      last = size;
      lastSentAt = Date.now();
      api
        .overlaySetSize(size.w, size.h)
        .then(() => {
          lastError = null;
        })
        .catch((e: unknown) => {
          last = null; // retry on the next change
          const message = errorMessage(e);
          if (message !== lastError) {
            lastError = message;
            useOverlay.getState().notify("error", `Couldn't resize the overlay: ${message}`);
          }
        });
    };

    const measure = () => {
      const rect = el.getBoundingClientRect();
      pending = { w: Math.ceil(rect.width), h: Math.ceil(rect.height) };
      if (timer !== undefined) return;
      timer = setTimeout(flush, Math.max(0, lastSentAt + SIZE_THROTTLE_MS - Date.now()));
    };

    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => {
      observer.disconnect();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [ref, nonce]);
}

// ---------------------------------------------------------------------------
// Drag → save position
// ---------------------------------------------------------------------------

export interface DragSaver {
  /** A drag may start (mouse down on a drag region). */
  arm(): void;
  /** The window moved. */
  moved(): void;
  /** The mouse button was released inside the webview. */
  released(): void;
  dispose(): void;
}

/** Quiet period after the last move before the position is saved. */
export const DRAG_SETTLE_MS = 350;
/** An armed drag that never moves is forgotten after this long. */
export const DRAG_ARM_TIMEOUT_MS = 15000;

/**
 * Small state machine: after `arm()`, window moves are debounced and `save`
 * runs once the window settles. Moves that were not preceded by `arm()`
 * (programmatic placement by the backend) never trigger a save.
 */
export function createDragSaver(save: () => void): DragSaver {
  let armed = false;
  let movedSinceArm = false;
  let settle: ReturnType<typeof setTimeout> | undefined;
  let expiry: ReturnType<typeof setTimeout> | undefined;
  const clear = () => {
    if (settle !== undefined) clearTimeout(settle);
    if (expiry !== undefined) clearTimeout(expiry);
    settle = undefined;
    expiry = undefined;
  };
  return {
    arm() {
      clear();
      armed = true;
      movedSinceArm = false;
      expiry = setTimeout(() => {
        if (!movedSinceArm) armed = false;
      }, DRAG_ARM_TIMEOUT_MS);
    },
    moved() {
      if (!armed) return;
      movedSinceArm = true;
      if (settle !== undefined) clearTimeout(settle);
      settle = setTimeout(() => {
        armed = false;
        clear();
        save();
      }, DRAG_SETTLE_MS);
    },
    released() {
      if (armed && !movedSinceArm) {
        armed = false;
        clear();
      }
    },
    dispose() {
      armed = false;
      clear();
    },
  };
}

const INTERACTIVE = "button, a, input, textarea, select, [role='button'], [contenteditable='true']";

/**
 * Returns a mousedown handler for drag regions (`data-tauri-drag-region`).
 * Saves the position with `api.overlaySavePosition` after the drag ends and
 * blocks the double-click maximize of the drag script.
 */
export function useDragSave(): (e: ReactMouseEvent<HTMLElement>) => void {
  const saver = useMemo(
    () =>
      createDragSaver(() => {
        api.overlaySavePosition().catch((e: unknown) => {
          useOverlay
            .getState()
            .notify("error", `Couldn't remember the position: ${errorMessage(e)}`);
        });
      }),
    [],
  );
  const saverRef = useRef(saver);

  useEffect(() => {
    const current = saverRef.current;
    let disposed = false;
    let unlisten: (() => void) | null = null;
    try {
      getCurrentWindow()
        .onMoved(() => current.moved())
        .then((un) => {
          if (disposed) un();
          else unlisten = un;
        })
        .catch(() => undefined);
    } catch {
      // Not running inside Tauri (e.g. a plain browser preview): nothing to save.
    }
    const onUp = () => current.released();
    window.addEventListener("mouseup", onUp);
    return () => {
      disposed = true;
      unlisten?.();
      window.removeEventListener("mouseup", onUp);
      current.dispose();
    };
  }, []);

  return (e) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest(INTERACTIVE)) return;
    if (e.detail >= 2) {
      // The overlay must never maximize on a double click.
      e.stopPropagation();
      e.preventDefault();
      return;
    }
    saverRef.current.arm();
  };
}

// ---------------------------------------------------------------------------
// Hotkeys
// ---------------------------------------------------------------------------

/** Display label of a configured global hotkey, or null when unbound. */
export function useHotkey(action: ShortcutAction): string | null {
  const acc = useOverlay((s) => s.settings?.shortcuts.bindings[action] ?? null);
  return acc ? formatAccelerator(acc) : null;
}

/** "Label (Ctrl+Shift+A)" tooltip helper. */
export function withHotkey(label: string, hotkey: string | null): string {
  return hotkey ? `${label} (${hotkey})` : label;
}
