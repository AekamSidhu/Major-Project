// Diagnostics: component health, pipeline latency, resource use, a local
// benchmark and — in developer mode — simulation tools (spec §57, §73–§75).

import { useEffect, useState, type FormEvent } from "react";
import type {
  BenchmarkReport,
  ComponentHealth,
  ComponentStatus,
  LatencyStage,
  SimulationAction,
  SpeakerRole,
} from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatDateTime } from "@/lib/format";
import { COMPONENT_LABELS } from "../components/settingsMeta";
import { useLoader } from "../components/useLoader";
import { useMeetingLive, useSettings } from "../components/useSettings";
import { useMain } from "../store";
import {
  Badge,
  Banner,
  Button,
  Field,
  PageHeader,
  Section,
  Select,
  Spinner,
  StatusDot,
  TextArea,
  TextInput,
} from "../ui";

type Tone = "neutral" | "accent" | "ok" | "warn" | "danger";

const STATUS_TONE: Record<ComponentStatus, Tone> = {
  inactive: "neutral",
  starting: "accent",
  active: "ok",
  streaming: "ok",
  paused: "warn",
  degraded: "warn",
  error: "danger",
  unsupported: "neutral",
  disabled: "neutral",
};

const STAGE_LABELS: Record<LatencyStage, string> = {
  audio_to_transcript: "Speech end → transcript",
  transcript_to_detection: "Transcript → question detected",
  detection_to_first_token: "Question → first words of answer",
  first_token_to_complete: "First words → complete answer",
  screen_processing: "Screen frame → observation",
  overlay_render: "Event → overlay render",
};

const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

function Components({ list }: { list: ComponentHealth[] }) {
  if (list.length === 0)
    return <p className="py-3 text-[0.82rem] text-muted">No components reported yet.</p>;
  return (
    <ul className="grid grid-cols-1 gap-2 py-3 sm:grid-cols-2 lg:grid-cols-3">
      {list.map((c) => (
        <li key={c.component} className="rounded-md border border-border px-3 py-2">
          <div className="flex items-center gap-2">
            <StatusDot tone={STATUS_TONE[c.status]} pulse={c.status === "streaming"} />
            <span className="text-[0.84rem] font-medium text-fg">
              {COMPONENT_LABELS[c.component]}
            </span>
            <span className="ml-auto text-[0.72rem] text-muted">{c.status}</span>
          </div>
          {c.message && (
            <p className="mt-1 text-[0.76rem] leading-snug text-muted" title={c.message}>
              {c.message}
            </p>
          )}
        </li>
      ))}
    </ul>
  );
}

function Benchmark() {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    try {
      setReport(await api.runBenchmark());
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Section
      title="Benchmark"
      description="Measures question detection, context assembly and your configured providers with short test requests. Uses no meeting data."
      actions={
        <Button onClick={() => void run()} busy={busy}>
          Run benchmark
        </Button>
      }
    >
      {error && <Banner tone="danger">{error}</Banner>}
      {report ? (
        <table className="my-2 w-full text-left text-[0.82rem]">
          <thead className="text-[0.74rem] text-muted">
            <tr>
              <th className="py-1.5 font-medium">Measurement</th>
              <th className="py-1.5 text-right font-medium">Result</th>
              <th className="py-1.5 pl-4 font-medium">Note</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {report.items.map((i) => (
              <tr key={i.name}>
                <td className="py-1.5 text-fg">{i.name}</td>
                <td className="py-1.5 text-right font-mono tabular-nums text-fg">
                  {Number.isInteger(i.value) ? i.value : i.value.toFixed(1)} {i.unit}
                </td>
                <td className="py-1.5 pl-4 text-muted">{i.note}</td>
              </tr>
            ))}
          </tbody>
          <caption className="caption-bottom pt-2 text-left text-[0.72rem] text-muted">
            Ran {formatDateTime(report.ranAt)}
          </caption>
        </table>
      ) : (
        !busy && <p className="py-3 text-[0.82rem] text-muted">Not run yet.</p>
      )}
    </Section>
  );
}

function DevTools() {
  const live = useMeetingLive();
  const notify = useMain((s) => s.notify);
  const [speaker, setSpeaker] = useState<SpeakerRole>("remote");
  const [line, setLine] = useState("");
  const [question, setQuestion] = useState("How would you design a rate limiter for a public API?");
  const [answer, setAnswer] = useState(
    "I'd start with a token bucket per API key stored in Redis, since it allows short bursts. We used a sliding window at my last job but it was expensive at scale.",
  );
  const [screen, setScreen] = useState(
    "Code editor showing a Python function rate_limit(key, limit) using redis.incr",
  );
  const [aiText, setAiText] = useState("");
  const [failure, setFailure] = useState("Simulated provider timeout");
  const [latency, setLatency] = useState("1500");

  const run = async (action: SimulationAction, done?: () => void) => {
    try {
      await api.simulate(action);
      done?.();
    } catch (e) {
      notify("error", `Simulation failed: ${errorMessage(e)}`);
    }
  };
  const submit = (action: () => SimulationAction | null, done?: () => void) => (e: FormEvent) => {
    e.preventDefault();
    const a = action();
    if (a) void run(a, done);
  };

  return (
    <Section
      title="Developer tools"
      description="Inject events into the live pipeline. Simulated AI responses and failures replace the language model with a mock until cleared."
      actions={
        <Button
          variant="ghost"
          onClick={() =>
            void api.clearSimulation().catch((e: unknown) => notify("error", errorMessage(e)))
          }
        >
          Clear simulation
        </Button>
      }
    >
      {!live && (
        <div className="py-3">
          <Banner tone="accent">
            Transcript, question and answer simulations need a running meeting — start one (or a
            demo) from Home.
          </Banner>
        </div>
      )}
      <form
        onSubmit={submit(
          () => (line.trim() ? { action: "transcript", speaker, text: line } : null),
          () => setLine(""),
        )}
        className="py-3"
      >
        <Field label="Transcript line">
          <div className="flex gap-2">
            <Select
              aria-label="Speaker"
              value={speaker}
              className="w-36"
              options={[
                { value: "remote", label: "Other person" },
                { value: "user", label: "You" },
                { value: "unknown", label: "Unknown" },
              ]}
              onChange={setSpeaker}
            />
            <TextInput
              aria-label="Transcript text"
              value={line}
              onChange={(e) => setLine(e.target.value)}
              className="h-8.5 flex-1"
            />
            <Button type="submit" disabled={!live || !line.trim()}>
              Add
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(() => (question.trim() ? { action: "question", text: question } : null))}
        className="py-3"
      >
        <Field
          label="Incoming question"
          description="Spoken by the other person; triggers detection and (in candidate mode) a suggestion."
        >
          <div className="flex gap-2">
            <TextInput
              aria-label="Question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="h-8.5 flex-1"
            />
            <Button type="submit" disabled={!live || !question.trim()}>
              Ask
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(() =>
          answer.trim() ? { action: "candidate_answer", text: answer } : null,
        )}
        className="py-3"
      >
        <Field
          label="Candidate answer"
          description="Interviewer mode: a finished candidate answer that triggers follow-ups."
        >
          <TextArea
            aria-label="Candidate answer"
            rows={3}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
          />
          <div className="mt-2">
            <Button type="submit" disabled={!live || !answer.trim()}>
              Send answer
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(() =>
          screen.trim()
            ? { action: "screen_change", description: screen, visibleText: null }
            : null,
        )}
        className="py-3"
      >
        <Field
          label="Screen change"
          description="A described screen observation, as if produced by the vision pipeline."
        >
          <div className="flex gap-2">
            <TextInput
              aria-label="Screen description"
              value={screen}
              onChange={(e) => setScreen(e.target.value)}
              className="h-8.5 flex-1"
            />
            <Button type="submit" disabled={!screen.trim()}>
              Inject
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(
          () => (aiText.trim() ? { action: "ai_response", text: aiText } : null),
          () => setAiText(""),
        )}
        className="py-3"
      >
        <Field
          label="Next AI response"
          description="Queued for the next request. Answers use [[SECTION]] markers, follow-ups one JSON object per line."
        >
          <TextArea
            aria-label="AI response"
            rows={3}
            value={aiText}
            onChange={(e) => setAiText(e.target.value)}
          />
          <div className="mt-2">
            <Button type="submit" disabled={!aiText.trim()}>
              Queue response
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(() =>
          failure.trim() ? { action: "ai_failure", message: failure } : null,
        )}
        className="py-3"
      >
        <Field label="AI failure" description="The next request fails with this network error.">
          <div className="flex gap-2">
            <TextInput
              aria-label="Failure message"
              value={failure}
              onChange={(e) => setFailure(e.target.value)}
              className="h-8.5 flex-1"
            />
            <Button type="submit" disabled={!failure.trim()}>
              Fail next
            </Button>
          </div>
        </Field>
      </form>
      <form
        onSubmit={submit(() => {
          const n = Number(latency);
          return Number.isInteger(n) && n >= 0 && n <= 60000
            ? { action: "network_latency", ms: n }
            : null;
        })}
        className="py-3"
      >
        <Field
          label="Network latency"
          description="Delay before the first token of every mock response (0–60000 ms)."
          inline
        >
          <div className="flex gap-2">
            <TextInput
              aria-label="Latency in milliseconds"
              inputMode="numeric"
              value={latency}
              onChange={(e) => setLatency(e.target.value)}
              className="h-8.5 w-28"
            />
            <Button type="submit">Set</Button>
          </div>
        </Field>
      </form>
    </Section>
  );
}

export function DiagnosticsSection() {
  const dev = useSettings().general.developerMode;
  const liveComponents = useMain((s) => s.live.components);
  const diag = useLoader(api.getDiagnostics);
  const { reload } = diag;

  // Latency and resource figures refresh while the page is open.
  useEffect(() => {
    const t = window.setInterval(reload, 3000);
    return () => window.clearInterval(t);
  }, [reload]);

  const d = diag.data;
  // Health from live events is newer than the last poll.
  const merged = new Map((d?.components ?? []).map((c) => [c.component, c]));
  for (const c of liveComponents) {
    const prev = merged.get(c.component);
    if (!prev || c.updatedAt >= prev.updatedAt) merged.set(c.component, c);
  }
  const components = [...merged.values()].sort((a, b) =>
    COMPONENT_LABELS[a.component].localeCompare(COMPONENT_LABELS[b.component]),
  );

  return (
    <>
      <PageHeader
        title="Diagnostics"
        description="Everything here is measured locally and never sent anywhere."
      />
      {diag.error && (
        <Banner tone="danger" title="Diagnostics unavailable">
          {diag.error}
        </Banner>
      )}
      {!d && !diag.error ? (
        <div className="flex items-center gap-2 py-6 text-[0.86rem] text-muted">
          <Spinner /> Collecting diagnostics…
        </div>
      ) : (
        d && (
          <>
            <Section title="System">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1.5 py-3 text-[0.82rem] sm:grid-cols-3">
                {[
                  ["Version", d.version],
                  ["Platform", d.platform],
                  ["Meeting", `${d.meetingState} · ${d.mode}`],
                  ["Memory", `${d.memoryMb.toFixed(0)} MB`],
                  ["CPU", `${d.cpuPercent.toFixed(1)}%`],
                  ["Active AI requests", String(d.activeRequests)],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-[0.74rem] text-muted">{k}</dt>
                    <dd className="text-fg">{v}</dd>
                  </div>
                ))}
              </dl>
            </Section>
            <Section title="Components">
              <Components list={components} />
            </Section>
            <Section title="Latency" description="Measured during meetings since OpenMeet started.">
              {d.latency.every((l) => l.count === 0) ? (
                <p className="py-3 text-[0.82rem] text-muted">
                  No measurements yet — they appear once a meeting (or demo) runs.
                </p>
              ) : (
                <table className="my-2 w-full text-left text-[0.82rem]">
                  <thead className="text-[0.74rem] text-muted">
                    <tr>
                      <th className="py-1.5 font-medium">Stage</th>
                      <th className="py-1.5 text-right font-medium">Samples</th>
                      <th className="py-1.5 text-right font-medium">Last</th>
                      <th className="py-1.5 text-right font-medium">p50</th>
                      <th className="py-1.5 text-right font-medium">p95</th>
                      <th className="py-1.5 text-right font-medium">Max</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border font-mono tabular-nums">
                    {d.latency
                      .filter((l) => l.count > 0)
                      .map((l) => (
                        <tr key={l.stage}>
                          <td className="py-1.5 font-sans text-fg">{STAGE_LABELS[l.stage]}</td>
                          <td className="py-1.5 text-right text-fg-soft">{l.count}</td>
                          <td className="py-1.5 text-right text-fg">{ms(l.lastMs)}</td>
                          <td className="py-1.5 text-right text-fg">{ms(l.p50Ms)}</td>
                          <td className="py-1.5 text-right text-fg">{ms(l.p95Ms)}</td>
                          <td className="py-1.5 text-right text-fg-soft">{ms(l.maxMs)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              )}
            </Section>
          </>
        )
      )}
      <Benchmark />
      {dev ? (
        <DevTools />
      ) : (
        <p className="text-[0.78rem] text-muted">
          <Badge>Tip</Badge> Turn on developer mode in General to simulate questions, screen changes
          and provider failures.
        </p>
      )}
    </>
  );
}
