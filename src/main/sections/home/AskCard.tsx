// "Ask OpenMeet" hint: the global hotkey (registered by the backend) and a
// button to bring up the overlay. Ask works without a meeting.

import { useId, useState } from "react";
import type { Settings, ShortcutAction, ShortcutStatus } from "@/bindings";
import { formatAccelerator } from "@/lib/format";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "@/main/store";
import { Button, Kbd } from "@/main/ui";
import { InlineError, LinkButton } from "./controls";
import { useAsyncData } from "./hooks";
import { Icon } from "./icons";

interface ShortcutInfo {
  accelerator: string | null;
  problem: string | null;
}

function shortcutInfo(
  action: ShortcutAction,
  statuses: ShortcutStatus[] | null,
  settings: Settings,
): ShortcutInfo {
  const st = statuses?.find((s) => s.action === action);
  if (st) {
    return {
      accelerator: st.accelerator || null,
      problem: st.registered ? null : (st.error ?? "This shortcut is not registered."),
    };
  }
  return { accelerator: settings.shortcuts.bindings[action] ?? null, problem: null };
}

export function AskCard({ settings }: { settings: Settings }) {
  const setSection = useMain((s) => s.setSection);
  const statuses = useAsyncData(api.getShortcutStatus);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const headingId = useId();

  const ask = shortcutInfo("ask_ai", statuses.data, settings);
  const overlay = shortcutInfo("toggle_overlay", statuses.data, settings);

  const openOverlay = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.overlayShow();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      aria-labelledby={headingId}
      className="rounded-lg border border-border bg-surface px-4 py-3.5"
    >
      <h2 id={headingId} className="text-[0.92rem] font-semibold text-fg">
        Ask OpenMeet
      </h2>
      <p className="mt-1 text-[0.8rem] leading-relaxed text-fg-soft">
        {ask.accelerator ? (
          <>
            Press <Kbd>{formatAccelerator(ask.accelerator)}</Kbd> anywhere to ask a question in the
            overlay — it works without a meeting, too.
          </>
        ) : (
          <>Assign a shortcut to ask questions from anywhere — it works without a meeting, too.</>
        )}
      </p>
      {ask.problem && (
        <p className="mt-1.5 text-[0.78rem] text-warn">Shortcut unavailable: {ask.problem}</p>
      )}
      {overlay.accelerator && (
        <p className="mt-2 text-[0.78rem] text-muted">
          Show or hide the overlay with <Kbd>{formatAccelerator(overlay.accelerator)}</Kbd>.
        </p>
      )}
      {statuses.error && (
        <p className="mt-1.5 text-[0.78rem] text-muted">
          Shortcut status unavailable: {statuses.error}
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button size="sm" busy={busy} onClick={() => void openOverlay()}>
          <Icon name="eye" className="size-3.5" />
          Open overlay
        </Button>
        <LinkButton onClick={() => setSection("settings/shortcuts")}>Shortcuts</LinkButton>
      </div>
      {error && <InlineError className="mt-2">Could not open the overlay: {error}</InlineError>}
    </section>
  );
}
