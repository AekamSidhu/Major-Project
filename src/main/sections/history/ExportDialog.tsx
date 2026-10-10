// Export dialog: format + parts → exportMeeting (the backend shows the native
// save dialog; a null path means the user cancelled it).

import { useId, useState } from "react";
import type { ExportFormat, ExportParts } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "@/main/store";
import { Button, Dialog, cx } from "@/main/ui";
import { Checkbox, InlineError } from "@/main/sections/home/controls";
import { Icon } from "@/main/sections/home/icons";
import {
  buildExportRequest,
  DEFAULT_EXPORT_PARTS,
  EXPORT_FORMATS,
  EXPORT_PARTS,
  formatBytes,
} from "./helpers";

export interface ExportDialogProps {
  meetingId: string;
  meetingTitle: string;
  onClose: () => void;
}

/** Mount it only while open: each opening starts from fresh choices. */
export function ExportDialog({ meetingId, meetingTitle, onClose }: ExportDialogProps) {
  const notify = useMain((s) => s.notify);
  const [format, setFormat] = useState<ExportFormat>("markdown");
  const [parts, setParts] = useState<ExportParts>(DEFAULT_EXPORT_PARTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState(false);
  const formatName = useId();
  const built = buildExportRequest(meetingId, format, parts);

  const submit = async () => {
    if (!built.request || busy) return;
    setBusy(true);
    setError(null);
    setCancelled(false);
    try {
      const result = await api.exportMeeting(built.request);
      if (result.path === null) {
        setCancelled(true);
        return;
      }
      notify("info", `Exported ${formatBytes(result.bytes)} to ${result.path}`);
      onClose();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open
      title="Export meeting"
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            busy={busy}
            disabled={!built.request}
            onClick={() => void submit()}
          >
            <Icon name="download" />
            Export…
          </Button>
        </>
      }
    >
      <p className="text-[0.82rem] text-muted">
        “{meetingTitle}” — you’ll choose where to save the file next. Exports stay on your device.
      </p>
      <fieldset className="mt-4">
        <legend className="mb-1.5 text-[0.8rem] font-semibold text-fg">Format</legend>
        <div className="space-y-1">
          {EXPORT_FORMATS.map((f) => (
            <label
              key={f.value}
              className={cx(
                "flex cursor-pointer items-start gap-2.5 rounded-md border px-2.5 py-1.5",
                format === f.value
                  ? "border-accent bg-accent-soft"
                  : "border-border hover:border-border-strong",
              )}
            >
              <input
                type="radio"
                name={formatName}
                value={f.value}
                checked={format === f.value}
                onChange={() => setFormat(f.value)}
                className="mt-1 size-3.5 shrink-0 accent-[var(--om-accent)]"
              />
              <span>
                <span className="block text-[0.84rem] font-medium text-fg">{f.label}</span>
                <span className="block text-[0.76rem] text-muted">{f.description}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset className="mt-4">
        <legend className="mb-0.5 text-[0.8rem] font-semibold text-fg">Include</legend>
        {EXPORT_PARTS.map((p) => (
          <Checkbox
            key={p.key}
            label={p.label}
            description={p.description}
            checked={parts[p.key]}
            onChange={(v) => {
              setParts((prev) => ({ ...prev, [p.key]: v }));
              setCancelled(false);
            }}
          />
        ))}
      </fieldset>
      <div className="mt-2 min-h-5">
        {built.error && <p className="text-[0.8rem] text-warn">{built.error}</p>}
        {cancelled && (
          <p role="status" className="text-[0.8rem] text-muted">
            Save cancelled — nothing was exported.
          </p>
        )}
        {error && <InlineError>Export failed: {error}</InlineError>}
      </div>
    </Dialog>
  );
}
