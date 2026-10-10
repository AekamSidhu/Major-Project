// "Test connection" for an AI component: waits for pending settings edits,
// calls the backend health check and reports latency, model and location.

import { useState } from "react";
import type { Component, ProviderTestResult } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { Button, StatusDot } from "../ui";
import { LocationBadge } from "./LocationBadge";
import { settleSettings } from "./save";

type Outcome = { kind: "result"; result: ProviderTestResult } | { kind: "error"; message: string };

export function TestConnection({
  component,
  label = "Test connection",
  disabled,
}: {
  component: Extract<Component, "llm" | "transcription" | "vision" | "embeddings">;
  label?: string;
  disabled?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  const run = async () => {
    setBusy(true);
    setOutcome(null);
    try {
      await settleSettings();
      const result = await api.testProvider(component);
      setOutcome({ kind: "result", result });
    } catch (e) {
      setOutcome({ kind: "error", message: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-start gap-3 py-2.5">
      <Button onClick={() => void run()} busy={busy} disabled={disabled}>
        {busy ? "Testing…" : label}
      </Button>
      <div role="status" aria-live="polite" className="min-w-0 flex-1 pt-1.5 text-[0.8rem]">
        {outcome?.kind === "error" && (
          <span className="flex items-start gap-1.5 text-danger">
            <span className="mt-1.5">
              <StatusDot tone="danger" />
            </span>
            {outcome.message}
          </span>
        )}
        {outcome?.kind === "result" && <TestResult result={outcome.result} />}
      </div>
    </div>
  );
}

function TestResult({ result }: { result: ProviderTestResult }) {
  const info = result.info;
  return (
    <span className="flex flex-col gap-0.5">
      <span
        className={
          result.ok
            ? "flex items-center gap-1.5 font-medium text-ok"
            : "flex items-center gap-1.5 font-medium text-danger"
        }
      >
        <StatusDot tone={result.ok ? "ok" : "danger"} />
        {result.ok ? "Connected" : "Connection failed"}
        {result.latencyMs !== null && (
          <span className="font-normal text-muted tabular-nums">
            · {Math.round(result.latencyMs)} ms
          </span>
        )}
      </span>
      {result.message && (
        <span className={result.ok ? "text-fg-soft" : "text-danger"}>{result.message}</span>
      )}
      {info && (
        <span className="flex flex-wrap items-center gap-1.5 text-muted">
          {info.displayName}
          {info.model && <span className="font-mono text-[0.76rem]">{info.model}</span>}
          {info.endpoint && <span>· {info.endpoint}</span>}
          <LocationBadge location={info.location} endpoint={info.endpoint} />
        </span>
      )}
    </span>
  );
}
