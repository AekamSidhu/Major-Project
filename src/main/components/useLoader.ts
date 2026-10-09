// Loads data from the backend on mount (and on demand), keeping the last
// good value while reloading and exposing the error for inline display.

import { useCallback, useEffect, useState } from "react";
import { errorMessage } from "@/lib/ipc";

export interface Loader<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
  /** Replaces the data locally (e.g. with a command's returned value). */
  setData: (data: T) => void;
}

interface LoadState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** `load` must be stable (an `api.*` function or a memoized callback). */
export function useLoader<T>(load: () => Promise<T>): Loader<T> {
  const [state, setState] = useState<LoadState<T>>({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    load().then(
      (data) => {
        if (alive) setState({ data, error: null, loading: false });
      },
      (e: unknown) => {
        if (alive) setState((s) => ({ data: s.data, error: errorMessage(e), loading: false }));
      },
    );
    return () => {
      alive = false;
    };
  }, [load, nonce]);

  const reload = useCallback(() => {
    setState((s) => ({ ...s, loading: true }));
    setNonce((n) => n + 1);
  }, []);

  const setData = useCallback((data: T) => setState({ data, error: null, loading: false }), []);

  return { ...state, reload, setData };
}
