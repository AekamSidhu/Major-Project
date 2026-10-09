// Small React hooks shared by the meeting (Home) and History sections.

import { useCallback, useEffect, useRef, useState } from "react";
import type { MeetingEvent, UiCommand } from "@/bindings";
import { errorMessage, onMeetingEvent, onUiCommand } from "@/lib/ipc";

export interface AsyncData<T> {
  /** Last loaded value for the current `key` (kept while reloading). */
  data: T | null;
  /** Error of the latest load attempt, if it failed. */
  error: string | null;
  loading: boolean;
  reload: () => void;
}

interface AsyncResult<T> {
  key: string;
  token: string;
  data: T | null;
  error: string | null;
}

/**
 * Loads data with `load()` whenever `key` changes or `reload()` is called.
 * Stale responses (for an older key or reload) are ignored.
 */
export function useAsyncData<T>(load: () => Promise<T>, key = ""): AsyncData<T> {
  const loadRef = useRef(load);
  useEffect(() => {
    loadRef.current = load;
  });
  const [nonce, setNonce] = useState(0);
  const token = `${key}#${nonce}`;
  const [result, setResult] = useState<AsyncResult<T> | null>(null);

  useEffect(() => {
    let alive = true;
    loadRef.current().then(
      (data) => {
        if (alive) setResult({ key, token, data, error: null });
      },
      (e: unknown) => {
        if (alive) {
          setResult((prev) => ({
            key,
            token,
            data: prev && prev.key === key ? prev.data : null,
            error: errorMessage(e),
          }));
        }
      },
    );
    return () => {
      alive = false;
    };
  }, [key, token]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const loading = result === null || result.token !== token;
  return {
    data: result && result.key === key ? result.data : null,
    error: !loading && result ? result.error : null,
    loading,
    reload,
  };
}

function useSubscription<T>(
  subscribe: (handler: (value: T) => void) => Promise<() => void>,
  handler: (value: T) => void,
): void {
  const ref = useRef(handler);
  useEffect(() => {
    ref.current = handler;
  });
  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | null = null;
    subscribe((value) => ref.current(value))
      .then((un) => {
        if (disposed) un();
        else unlisten = un;
      })
      // The App shell already reports a broken event bus; nothing to add here.
      .catch(() => {});
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [subscribe]);
}

/** Subscribes to backend meeting events while the component is mounted. */
export function useMeetingEvents(handler: (event: MeetingEvent) => void): void {
  useSubscription(onMeetingEvent, handler);
}

/** Subscribes to backend UI commands while the component is mounted. */
export function useUiCommands(handler: (command: UiCommand) => void): void {
  useSubscription(onUiCommand, handler);
}

/** Current wall-clock time, refreshed every `intervalMs`. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}
