// Settings writes from the settings UI. Every write goes through the store's
// optimistic `updateSettings`; this module additionally tracks in-flight saves
// and debounced edits so actions that depend on the stored settings (test
// connection, load models) can wait until the user's latest input is saved.

import type { Settings } from "@/bindings";
import { useMain } from "../store";

const inFlight = new Set<Promise<boolean>>();
const flushers = new Set<() => void>();

/** Applies `mutate` to the settings and persists it. Resolves to true on success. */
export function saveSettings(mutate: (draft: Settings) => void): Promise<boolean> {
  const p = useMain.getState().updateSettings(mutate);
  inFlight.add(p);
  void p.finally(() => inFlight.delete(p));
  return p;
}

/** Registers a function that commits a pending debounced edit immediately. */
export function registerFlusher(flush: () => void): () => void {
  flushers.add(flush);
  return () => {
    flushers.delete(flush);
  };
}

/** Commits pending debounced edits and waits for every in-flight save. */
export async function settleSettings(): Promise<void> {
  for (const flush of [...flushers]) flush();
  await Promise.allSettled([...inFlight]);
}

/** Message of the last failed save (set by the store), for inline errors. */
export function lastSaveError(): string {
  return useMain.getState().savingError ?? "the settings could not be saved";
}
