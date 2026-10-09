// Status and controls for the optional local AI sidecar (faster-whisper
// transcription + local embeddings).

import { useState } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import type { LocalAiStatus } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { Banner, Button, StatusDot } from "../ui";
import { CheckIcon, CopyIcon, RefreshIcon } from "./icons";
import { useLoader } from "./useLoader";

function summary(s: LocalAiStatus): { tone: "ok" | "warn" | "neutral"; text: string } {
  if (s.running) return { tone: "ok", text: s.managed ? "Running (started by OpenMeet)" : "Running (started outside OpenMeet)" };
  if (s.installed) return { tone: "warn", text: "Installed, not running" };
  return { tone: "neutral", text: "Not installed" };
}

export function LocalAiPanel({ compact = false }: { compact?: boolean }) {
  const status = useLoader(api.getLocalAiStatus);
  const [busy, setBusy] = useState<"start" | "stop" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const act = async (kind: "start" | "stop") => {
    setBusy(kind);
    setError(null);
    try {
      const next = kind === "start" ? await api.startLocalAi() : await api.stopLocalAi();
      status.setData(next);
      if (kind === "start" && !next.running) setError(next.message || "The local AI server did not start.");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const copyHint = async (hint: string) => {
    try {
      await writeText(hint);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      setError(`Could not copy: ${errorMessage(e)}`);
    }
  };

  const s = status.data;
  const sum = s ? summary(s) : null;

  return (
    <div className="py-2.5" aria-busy={status.loading || undefined}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[0.86rem] font-medium text-fg">Local AI sidecar</div>
          <div role="status" className="mt-0.5 flex items-center gap-1.5 text-[0.8rem] text-fg-soft">
            {sum ? (
              <>
                <StatusDot tone={sum.tone} />
                {sum.text}
                {s?.running && <span className="font-mono text-[0.74rem] text-muted">{s.baseUrl}</span>}
              </>
            ) : status.error ? (
              <span className="text-danger">Status unavailable: {status.error}</span>
            ) : (
              <span className="text-muted">Checking…</span>
            )}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {s?.installed && !s.running && (
            <Button variant="primary" size="sm" busy={busy === "start"} disabled={busy !== null} onClick={() => void act("start")}>
              Start
            </Button>
          )}
          {s?.running && s.managed && (
            <Button size="sm" busy={busy === "stop"} disabled={busy !== null} onClick={() => void act("stop")}>
              Stop
            </Button>
          )}
          <Button variant="ghost" size="sm" aria-label="Refresh local AI status" disabled={status.loading || busy !== null} onClick={status.reload}>
            <RefreshIcon />
          </Button>
        </div>
      </div>

      {s && !compact && (s.transcriptionModel || s.embeddingModel) && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[0.78rem]">
          {s.transcriptionModel && (
            <>
              <dt className="text-muted">Transcription model</dt>
              <dd className="font-mono text-fg-soft">{s.transcriptionModel}</dd>
            </>
          )}
          {s.embeddingModel && (
            <>
              <dt className="text-muted">Embedding model</dt>
              <dd className="font-mono text-fg-soft">{s.embeddingModel}</dd>
            </>
          )}
        </dl>
      )}
      {s?.message && !s.running && <p className="mt-1.5 text-[0.78rem] text-muted">{s.message}</p>}

      {s && !s.installed && s.setupHint && (
        <div className="mt-2.5 rounded-md border border-border bg-surface-2 px-3 py-2">
          <p className="text-[0.78rem] text-fg-soft">
            Install it once with this command (requires Python). Speech and embedding models then run entirely on this computer:
          </p>
          <div className="mt-1.5 flex items-start gap-2">
            <code className="min-w-0 flex-1 overflow-x-auto font-mono text-[0.76rem] whitespace-pre text-fg">{s.setupHint}</code>
            <Button size="sm" variant="ghost" aria-label="Copy setup command" onClick={() => void copyHint(s.setupHint)}>
              {copied ? <CheckIcon /> : <CopyIcon />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-2">
          <Banner tone="danger">{error}</Banner>
        </div>
      )}
    </div>
  );
}
