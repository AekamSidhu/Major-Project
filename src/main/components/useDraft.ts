// Local draft state for settings inputs: shows what the user types, validates
// it, and commits it to the settings after a short debounce (or on blur).
// While the user is editing, stored values coming back from the backend (which
// may be sanitized, e.g. trimmed) never overwrite the text being typed.

import { useCallback, useEffect, useRef, useState } from "react";
import { lastSaveError, registerFlusher } from "./save";

export interface DraftOptions<T> {
  /** The stored value. */
  value: T;
  /** Persists a value; resolves to false when the save failed. */
  commit: (value: T) => Promise<boolean>;
  /** Returns an error message for invalid values (never committed). */
  validate?: (value: T) => string | null;
  /** Debounce delay in ms. */
  delay?: number;
  /**
   * Drop the draft right after committing (sliders: the store value is
   * updated optimistically). Text inputs keep their draft until blur.
   */
  resetOnCommit?: boolean;
}

export interface Draft<T> {
  value: T;
  error: string | null;
  editing: boolean;
  onChange: (value: T) => void;
  /** Commits immediately and ends editing (when valid). */
  onBlur: () => void;
}

interface Box<T> {
  commit: (value: T) => Promise<boolean>;
  timer: number | null;
  pending: { value: T } | null;
}

export function useDraft<T>({
  value,
  commit,
  validate,
  delay = 600,
  resetOnCommit = false,
}: DraftOptions<T>): Draft<T> {
  const [draft, setDraft] = useState<{ value: T } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<Box<T>>({ commit, timer: null, pending: null });

  useEffect(() => {
    box.current.commit = commit;
  });

  const flush = useCallback(() => {
    const b = box.current;
    if (b.timer !== null) {
      window.clearTimeout(b.timer);
      b.timer = null;
    }
    const pending = b.pending;
    if (!pending) return;
    b.pending = null;
    void b.commit(pending.value).then((ok) => {
      if (!ok) setError(`Not saved: ${lastSaveError()}`);
    });
    if (resetOnCommit) setDraft(null);
  }, [resetOnCommit]);

  // Pending edits are committed when another action needs the stored
  // settings, and when the input unmounts.
  useEffect(() => {
    const unregister = registerFlusher(flush);
    return () => {
      unregister();
      flush();
    };
  }, [flush]);

  const onChange = (next: T) => {
    setDraft({ value: next });
    const err = validate ? validate(next) : null;
    setError(err);
    const b = box.current;
    if (b.timer !== null) window.clearTimeout(b.timer);
    b.timer = null;
    b.pending = null;
    if (err === null) {
      b.pending = { value: next };
      b.timer = window.setTimeout(flush, delay);
    }
  };

  const onBlur = () => {
    flush();
    if (error === null) setDraft(null);
  };

  return { value: draft ? draft.value : value, error, editing: draft !== null, onChange, onBlur };
}
