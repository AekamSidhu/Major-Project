// Demo mode: scripted meetings with simulated AI that work without any
// microphone, provider account or network access.

import { useId, useState } from "react";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "@/main/store";
import { Badge, Button, Spinner } from "@/main/ui";
import { InlineError } from "./controls";
import { modeLabel } from "./helpers";
import { useAsyncData } from "./hooks";
import { Icon } from "./icons";

export function DemoCard() {
  const scenarios = useAsyncData(api.listDemoScenarios);
  const [startingId, setStartingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const headingId = useId();

  const run = async (scenarioId: string) => {
    setStartingId(scenarioId);
    setError(null);
    try {
      await api.startDemo(scenarioId);
      await useMain.getState().refreshSnapshot();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setStartingId(null);
    }
  };

  return (
    <section
      aria-labelledby={headingId}
      className="rounded-lg border border-border bg-surface px-4 py-3.5"
    >
      <div className="flex items-center gap-2">
        <Icon name="sparkle" className="text-accent" />
        <h2 id={headingId} className="text-[0.92rem] font-semibold text-fg">
          Demo mode
        </h2>
      </div>
      <p className="mt-1 text-[0.8rem] text-muted">
        See the whole experience with a scripted conversation and simulated AI. No microphone,
        account or network needed — demo meetings are not saved.
      </p>

      {scenarios.loading && !scenarios.data && (
        <p role="status" className="mt-3 flex items-center gap-2 text-[0.8rem] text-muted">
          <Spinner /> Loading scenarios…
        </p>
      )}
      {scenarios.error && (
        <div className="mt-3 flex items-center gap-3">
          <InlineError>Could not load demo scenarios: {scenarios.error}</InlineError>
          <Button size="sm" variant="ghost" onClick={scenarios.reload}>
            Retry
          </Button>
        </div>
      )}
      {scenarios.data && scenarios.data.length === 0 && (
        <p className="mt-3 text-[0.8rem] text-muted">
          No demo scenarios are available in this build.
        </p>
      )}
      {scenarios.data && scenarios.data.length > 0 && (
        <ul className="mt-2 divide-y divide-border">
          {scenarios.data.map((s) => (
            <li key={s.id} className="flex items-start gap-3 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[0.86rem] font-medium text-fg">{s.title}</span>
                  <Badge>{modeLabel(s.mode)}</Badge>
                </div>
                <p className="mt-0.5 text-[0.78rem] leading-snug text-muted">{s.description}</p>
              </div>
              <Button
                size="sm"
                busy={startingId === s.id}
                disabled={startingId !== null && startingId !== s.id}
                onClick={() => void run(s.id)}
                aria-label={`Run demo: ${s.title}`}
              >
                <Icon name="play" className="size-3.5" />
                Run
              </Button>
            </li>
          ))}
        </ul>
      )}
      {error && <InlineError className="mt-2">Could not start the demo: {error}</InlineError>}
    </section>
  );
}
