// Shortcuts: every global hotkey with its registration status, conflict
// warnings and a key-combination recorder (spec §33). Hotkeys are registered
// by the backend; this page only edits and displays them.

import { useState, type KeyboardEvent } from "react";
import type { ShortcutAction, ShortcutStatus } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatAccelerator, shortcutLabel } from "@/lib/format";
import {
  IS_MAC,
  acceleratorFromEvent,
  conflictsFor,
  validateAccelerator,
} from "../../components/accelerator";
import { useLoader } from "../../components/useLoader";
import { Badge, Banner, Button, ConfirmDialog, Kbd, PageHeader, Spinner, cx } from "../../ui";

const ORDER: ReadonlyArray<ShortcutAction> = [
  "toggle_overlay",
  "generate_answer",
  "generate_followups",
  "ask_ai",
  "toggle_mode",
  "cycle_style",
  "expand_hud",
  "toggle_click_through",
  "capture_screen",
  "pause_resume",
  "emergency_stop",
];

const DESCRIPTIONS: Partial<Record<ShortcutAction, string>> = {
  emergency_stop: "Immediately stops audio, screen and AI processing. Always keep this assigned.",
  ask_ai: "Opens the Ask box in the overlay for typing.",
  toggle_click_through: "Lets clicks pass through the overlay to the meeting window.",
};

function Recorder({
  action,
  statuses,
  onDone,
  onCancel,
}: {
  action: ShortcutAction;
  statuses: ShortcutStatus[];
  onDone: (list: ShortcutStatus[]) => void;
  onCancel: () => void;
}) {
  const [preview, setPreview] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onKeyDown = async (e: KeyboardEvent<HTMLButtonElement>) => {
    if (busy) return;
    if (e.key === "Tab") return;
    e.preventDefault();
    e.stopPropagation();
    const noMods = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
    if (e.code === "Escape" && noMods) {
      onCancel();
      return;
    }
    const r = acceleratorFromEvent(e, IS_MAC);
    if (r.kind === "partial") {
      setPreview(r.modifiers.join("+"));
      setError(null);
      return;
    }
    if (r.kind === "unsupported") {
      setError(`That key (${r.code}) can't be used in a global shortcut.`);
      return;
    }
    setPreview(r.accelerator);
    const invalid = validateAccelerator(r.accelerator);
    if (invalid) {
      setError(invalid);
      return;
    }
    const clash = conflictsFor(action, r.accelerator, statuses);
    if (clash.length > 0 && clash[0]) {
      setError(`Already used for “${shortcutLabel(clash[0])}”.`);
      return;
    }
    setBusy(true);
    try {
      onDone(await api.updateShortcut(action, r.accelerator));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-1">
      <button
        type="button"
        autoFocus
        onKeyDown={(e) => void onKeyDown(e)}
        onKeyUp={(e) => {
          if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && !error) setPreview("");
        }}
        onBlur={onCancel}
        aria-label={`Press the new keys for ${shortcutLabel(action)}`}
        className="flex h-8 min-w-48 items-center gap-2 rounded-md border border-accent bg-accent-soft px-2.5 text-[0.84rem] text-fg focus:outline-none"
      >
        {busy && <Spinner />}
        {preview ? (
          <Kbd>{formatAccelerator(preview)}</Kbd>
        ) : (
          <span className="text-muted">Press keys… (Esc to cancel)</span>
        )}
      </button>
      {error && (
        <p role="alert" className="text-[0.76rem] text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

function Row({
  status,
  statuses,
  recording,
  onRecord,
  onUpdate,
  onError,
}: {
  status: ShortcutStatus;
  statuses: ShortcutStatus[];
  recording: boolean;
  onRecord: (on: boolean) => void;
  onUpdate: (list: ShortcutStatus[]) => void;
  onError: (msg: string) => void;
}) {
  const unassigned = !status.accelerator.trim();
  const clear = async () => {
    try {
      onUpdate(await api.updateShortcut(status.action, ""));
    } catch (e) {
      onError(errorMessage(e));
    }
  };
  return (
    <li className="flex items-start gap-4 py-3">
      <div className="min-w-0 flex-1">
        <div className="text-[0.86rem] font-medium text-fg">{shortcutLabel(status.action)}</div>
        {DESCRIPTIONS[status.action] && (
          <div className="mt-0.5 text-[0.76rem] text-muted">{DESCRIPTIONS[status.action]}</div>
        )}
        {!unassigned && status.error && (
          <div className="mt-0.5 text-[0.76rem] text-danger" role="alert">
            {status.error}
          </div>
        )}
        {status.warnings.map((w) => (
          <div key={w} className="mt-0.5 text-[0.76rem] text-warn">
            {w}
          </div>
        ))}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {recording ? (
          <Recorder
            action={status.action}
            statuses={statuses}
            onCancel={() => onRecord(false)}
            onDone={(list) => {
              onUpdate(list);
              onRecord(false);
            }}
          />
        ) : (
          <>
            {unassigned ? (
              <Badge>Not assigned</Badge>
            ) : (
              <span
                className={cx(
                  "inline-flex items-center gap-1.5",
                  !status.registered && "opacity-70",
                )}
              >
                <Kbd>{formatAccelerator(status.accelerator)}</Kbd>
                {status.registered ? (
                  <Badge tone="ok">Active</Badge>
                ) : (
                  <Badge tone="danger">Not active</Badge>
                )}
              </span>
            )}
            <Button size="sm" onClick={() => onRecord(true)}>
              {unassigned ? "Assign" : "Change"}
            </Button>
            {!unassigned && status.action !== "emergency_stop" && (
              <Button size="sm" variant="ghost" onClick={() => void clear()}>
                Clear
              </Button>
            )}
          </>
        )}
      </div>
    </li>
  );
}

export function ShortcutsSection() {
  const loader = useLoader(api.getShortcutStatus);
  const [recording, setRecording] = useState<ShortcutAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetting, setResetting] = useState(false);
  const statuses = loader.data ?? [];
  const byAction = new Map(statuses.map((s) => [s.action, s]));

  const reset = async () => {
    setResetting(true);
    try {
      loader.setData(await api.resetShortcuts());
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setResetting(false);
      setConfirmReset(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Shortcuts"
        description="Global hotkeys work while the meeting app has focus. They take precedence over the same keys in other apps while OpenMeet runs."
        actions={
          <Button variant="ghost" onClick={() => setConfirmReset(true)}>
            Reset to defaults
          </Button>
        }
      />
      {(error || loader.error) && <Banner tone="danger">{error ?? loader.error}</Banner>}
      {loader.loading && !loader.data ? (
        <div className="flex items-center gap-2 py-6 text-[0.86rem] text-muted">
          <Spinner /> Loading shortcuts…
        </div>
      ) : (
        <ul className="divide-y divide-border rounded-lg border border-border bg-surface px-4">
          {ORDER.map((a) => {
            const s = byAction.get(a);
            if (!s) return null;
            return (
              <Row
                key={a}
                status={s}
                statuses={statuses}
                recording={recording === a}
                onRecord={(on) => setRecording(on ? a : null)}
                onUpdate={(list) => {
                  loader.setData(list);
                  setError(null);
                }}
                onError={setError}
              />
            );
          })}
        </ul>
      )}
      <ConfirmDialog
        open={confirmReset}
        title="Reset all shortcuts?"
        message="Every hotkey returns to its default keys."
        confirmLabel="Reset shortcuts"
        busy={resetting}
        onConfirm={() => void reset()}
        onCancel={() => setConfirmReset(false)}
      />
    </>
  );
}
