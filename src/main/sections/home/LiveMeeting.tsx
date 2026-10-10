// Home — meeting live: header, big controls, sources, transcript, detected
// questions, topics and component health.

import { useId, useMemo, useState, type ReactNode } from "react";
import type {
  ComponentHealth,
  DetectedQuestion,
  Mode,
  ScreenSourceSpec,
  Settings,
  Topic,
} from "@/bindings";
import { categoryLabel, formatOffset } from "@/lib/format";
import { api, errorMessage } from "@/lib/ipc";
import { useMain, type LiveState } from "@/main/store";
import { Badge, Banner, Button, StatusDot, Toggle, cx } from "@/main/ui";
import { InlineError, LevelMeter, LinkButton } from "./controls";
import {
  componentLabel,
  componentSection,
  componentWarnings,
  formatElapsed,
  healthOf,
  meetingStateBadge,
  modeLabel,
  screenContextOn,
  segmentOffset,
  sourceStatus,
  speakerLabel,
  visibleQuestions,
} from "./helpers";
import { useNow, useUiCommands } from "./hooks";
import { Icon, type IconName } from "./icons";
import { LiveTranscript } from "./LiveTranscript";
import { ScreenSourcePicker } from "./ScreenSourcePicker";

type Action = "pause" | "end" | "emergency" | "overlay" | "screen" | "capture" | "assist";

function Card({
  title,
  actions,
  children,
  className,
  bodyClassName,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  const id = useId();
  return (
    <section
      aria-labelledby={id}
      className={cx("rounded-lg border border-border bg-surface", className)}
    >
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <h2 id={id} className="text-[0.84rem] font-semibold text-fg">
          {title}
        </h2>
        {actions}
      </div>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

function ElapsedClock({ startedAt }: { startedAt: number }) {
  const now = useNow(1000);
  return (
    <div className="text-right">
      <div className="text-[0.68rem] font-semibold tracking-wider text-muted uppercase">
        Elapsed
      </div>
      <time aria-label="Elapsed time" className="text-[1.25rem] font-semibold text-fg tabular-nums">
        {formatElapsed(startedAt, now)}
      </time>
    </div>
  );
}

function HealthWarnings({ components }: { components: ReadonlyArray<ComponentHealth> }) {
  const setSection = useMain((s) => s.setSection);
  const warnings = useMemo(() => componentWarnings(components), [components]);
  if (warnings.length === 0) return null;
  const severe = warnings.some((w) => w.status === "error");
  return (
    <div
      role="alert"
      className={cx(
        "mb-4 rounded-lg border bg-surface px-3.5 py-2.5",
        severe ? "border-danger/40" : "border-warn/40",
      )}
    >
      <ul className="space-y-1.5">
        {warnings.map((w) => (
          <li key={w.component} className="flex items-start gap-2.5 text-[0.82rem]">
            <span className="mt-1.5">
              <StatusDot tone={w.status === "error" ? "danger" : "warn"} />
            </span>
            <span className="min-w-0 flex-1 text-fg-soft">
              <span className="font-medium text-fg">{componentLabel(w.component)}</span>
              {" · "}
              {w.status === "error"
                ? "Error"
                : w.status === "degraded"
                  ? "Degraded"
                  : "Unsupported"}
              {w.message ? ` — ${w.message}` : ""}
            </span>
            <LinkButton onClick={() => setSection(componentSection(w.component))}>
              {componentSection(w.component) === "diagnostics" ? "Diagnostics" : "Settings"}
            </LinkButton>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SourceLine({
  icon,
  title,
  status,
  level,
  control,
  children,
}: {
  icon: IconName;
  title: string;
  status: { label: string; tone: "neutral" | "accent" | "ok" | "warn" | "danger" };
  level?: number;
  control?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="px-4 py-2.5">
      <div className="flex items-center gap-2.5">
        <Icon name={icon} className="text-fg-soft" />
        <span className="flex-1 text-[0.84rem] font-medium text-fg">{title}</span>
        {level !== undefined && status.tone === "ok" && (
          <LevelMeter level={level} label={`${title} level`} />
        )}
        <Badge tone={status.tone}>{status.label}</Badge>
        {control}
      </div>
      {children}
    </div>
  );
}

function ScreenControl({
  settings,
  on,
  busy,
  run,
}: {
  settings: Settings;
  on: boolean;
  busy: Action | null;
  run: (a: Action, fn: () => Promise<unknown>) => Promise<boolean>;
}) {
  const [picking, setPicking] = useState(false);
  const [pick, setPick] = useState<{ spec: ScreenSourceSpec | null; error: string | null }>({
    spec: settings.screen.source,
    error: null,
  });
  const reloadSettings = useMain((s) => s.reloadSettings);

  const enable = async (spec: ScreenSourceSpec) => {
    const ok = await run("screen", async () => {
      await api.setScreenCapture(true, spec);
      await reloadSettings();
    });
    if (ok) setPicking(false);
  };
  const disable = () =>
    run("screen", async () => {
      await api.setScreenCapture(false, settings.screen.source);
      await reloadSettings();
    });

  const source = settings.screen.source;
  return (
    <SourceLine
      icon="monitor"
      title="Screen context"
      status={on ? { label: "On", tone: "ok" } : { label: "Off", tone: "neutral" }}
      control={
        <Toggle
          label="Screen context"
          checked={on}
          disabled={busy === "screen"}
          onChange={(v) => {
            if (!v) void disable();
            else if (source) void enable(source);
            else setPicking(true);
          }}
        />
      }
    >
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 pl-6.5 text-[0.76rem] text-muted">
        {source ? <span className="truncate">{source.label}</span> : <span>No source chosen</span>}
        {!picking && (
          <LinkButton onClick={() => setPicking(true)} className="text-[0.76rem]">
            {source ? "Change" : "Choose source"}
          </LinkButton>
        )}
        {on && (
          <LinkButton
            className="text-[0.76rem]"
            disabled={busy === "capture"}
            onClick={() => void run("capture", () => api.captureScreenNow())}
          >
            <Icon name="camera" className="size-3.5" />
            Capture now
          </LinkButton>
        )}
      </div>
      {picking && (
        <div className="mt-3 rounded-md border border-border bg-bg/40 p-3">
          <ScreenSourcePicker
            value={pick.spec}
            onChange={(spec, error) => setPick({ spec, error })}
          />
          <div className="mt-3 flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setPicking(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              busy={busy === "screen"}
              disabled={!pick.spec || pick.error !== null}
              onClick={() => pick.spec && void enable(pick.spec)}
            >
              {on ? "Use this source" : "Start screen context"}
            </Button>
          </div>
        </div>
      )}
    </SourceLine>
  );
}

function QuestionItem({
  q,
  mode,
  startedAt,
  onSuggest,
  busy,
}: {
  q: DetectedQuestion;
  mode: Mode;
  startedAt: number;
  onSuggest: ((q: DetectedQuestion) => void) | null;
  busy: boolean;
}) {
  return (
    <li className="om-fade-in px-4 py-2.5">
      <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[0.72rem] text-muted">
        <Badge tone="accent">{categoryLabel(q.category)}</Badge>
        {q.requiresResponse && mode === "candidate" && <Badge tone="warn">Needs your answer</Badge>}
        <span>
          {speakerLabel(q.speakerRole, mode)} ·{" "}
          {formatOffset(segmentOffset(q.detectedAt, startedAt))}
        </span>
      </div>
      <p className="text-[0.84rem] leading-snug text-fg">{q.question}</p>
      {q.parts.length > 1 && (
        <ul className="mt-1 list-disc pl-4 text-[0.78rem] text-fg-soft">
          {q.parts.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}
      {onSuggest && (
        <Button
          size="sm"
          variant="ghost"
          className="mt-1 -ml-2"
          disabled={busy}
          onClick={() => onSuggest(q)}
        >
          <Icon name="sparkle" className="size-3.5" />
          Suggest answer in overlay
        </Button>
      )}
    </li>
  );
}

const TOPIC_STYLE: Record<Topic["status"], { mark: string; className: string; label: string }> = {
  covered: { mark: "✓", className: "text-ok", label: "covered" },
  current: { mark: "→", className: "text-accent", label: "current" },
  pending: { mark: "○", className: "text-muted", label: "pending" },
};

function Topics({ topics, next }: { topics: ReadonlyArray<Topic>; next: string | null }) {
  if (topics.length === 0) {
    return (
      <p className="px-4 py-3 text-[0.8rem] text-muted">
        Topics appear here as the conversation develops.
      </p>
    );
  }
  return (
    <div className="px-4 py-3">
      <ul className="flex flex-wrap gap-1.5">
        {topics.map((t) => {
          const s = TOPIC_STYLE[t.status];
          return (
            <li
              key={t.id}
              className={cx(
                "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[0.78rem]",
                t.status === "current" ? "border-accent/50 bg-accent-soft" : "border-border",
              )}
            >
              <span aria-hidden className={s.className}>
                {s.mark}
              </span>
              <span className={t.status === "pending" ? "text-fg-soft" : "text-fg"}>{t.name}</span>
              <span className="sr-only">({s.label})</span>
            </li>
          );
        })}
      </ul>
      {next && (
        <p className="mt-2 text-[0.78rem] text-fg-soft">
          Suggested next area: <span className="font-medium text-fg">{next}</span>
        </p>
      )}
    </div>
  );
}

export function LiveMeeting({ live, settings }: { live: LiveState; settings: Settings }) {
  const setSection = useMain((s) => s.setSection);
  const notify = useMain((s) => s.notify);
  const meeting = live.meeting;
  const [busy, setBusy] = useState<Action | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overlayVisible, setOverlayVisible] = useState<boolean | null>(null);
  useUiCommands((cmd) => {
    if (cmd.type === "OVERLAY_VISIBILITY_CHANGED") setOverlayVisible(cmd.visible);
  });

  const questions = useMemo(() => visibleQuestions(live.questions), [live.questions]);
  const latestQuestion = questions[0] ?? null;

  if (!meeting) return null;

  const run = async (action: Action, fn: () => Promise<unknown>): Promise<boolean> => {
    setBusy(action);
    setError(null);
    try {
      await fn();
      return true;
    } catch (e) {
      setError(errorMessage(e));
      return false;
    } finally {
      setBusy(null);
    }
  };

  const state = live.state;
  const stopped = live.processingStopped || live.pausedReason === "emergency";
  const paused = state === "paused";
  const transitioning = state === "starting" || state === "stopping";
  const badge = meetingStateBadge(state, live.processingStopped, live.pausedReason);
  const mode = live.mode;
  const mic = healthOf(live.components, "microphone");
  const sys = healthOf(live.components, "system_audio");
  const screenOn = screenContextOn(live.components, settings);

  const togglePause = () => run("pause", () => (paused ? api.resumeMeeting() : api.pauseMeeting()));
  const end = () => run("end", () => (live.demoRunning ? api.stopDemo() : api.stopMeeting()));
  const emergency = () => run("emergency", () => api.emergencyStop());
  const toggleOverlay = () =>
    run("overlay", async () => {
      if (overlayVisible === true) {
        await api.overlayHide();
        setOverlayVisible(false);
      } else if (overlayVisible === false) {
        await api.overlayShow();
        setOverlayVisible(true);
      } else {
        setOverlayVisible(await api.overlayToggle());
      }
    });
  const suggest = async (q: DetectedQuestion) => {
    const ok = await run("assist", () => api.generateAnswer({ question: q.question, style: null }));
    if (ok) notify("info", "The suggested answer is streaming in the overlay.");
  };
  const followups = async () => {
    const ok = await run("assist", () => api.generateFollowups());
    if (ok) notify("info", "Follow-up questions are streaming in the overlay.");
  };

  return (
    <div>
      <header className="mb-5 flex items-start justify-between gap-6">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center gap-1.5">
            <Badge tone={badge.tone}>
              <StatusDot tone={badge.tone} pulse={state === "active" && !stopped} />
              {badge.label}
            </Badge>
            <Badge>{modeLabel(mode)}</Badge>
            {live.demoRunning && <Badge tone="accent">Demo</Badge>}
            {!meeting.persisted && !live.demoRunning && <Badge>Not saved</Badge>}
          </div>
          <h1 className="truncate text-[1.25rem] font-semibold tracking-tight text-fg">
            {meeting.title}
          </h1>
        </div>
        <ElapsedClock startedAt={meeting.startedAt} />
      </header>

      {stopped && (
        <Banner
          tone="danger"
          title="PROCESSING STOPPED"
          actions={
            <Button
              size="sm"
              variant="primary"
              busy={busy === "pause"}
              onClick={() => void run("pause", () => api.resumeMeeting())}
            >
              Resume
            </Button>
          }
        >
          Audio, screen and AI processing are halted. Nothing is captured or sent until you resume.
        </Banner>
      )}
      {paused && !stopped && (
        <Banner tone="neutral" title="Paused">
          Nothing new is transcribed or sent to AI until you resume.
        </Banner>
      )}
      <HealthWarnings components={live.components} />

      <div
        role="group"
        aria-label="Meeting controls"
        className="mb-5 flex flex-wrap items-center gap-2"
      >
        {!stopped && (
          <Button
            className="h-10 px-4"
            busy={busy === "pause"}
            disabled={transitioning}
            onClick={() => void togglePause()}
          >
            <Icon name={paused ? "play" : "pause"} />
            {paused ? "Resume" : "Pause"}
          </Button>
        )}
        <Button
          className="h-10 px-4"
          busy={busy === "end"}
          disabled={state === "stopping"}
          onClick={() => void end()}
        >
          <Icon name="stop" />
          End meeting
        </Button>
        <Button
          variant="danger"
          className="h-10 px-4"
          busy={busy === "emergency"}
          disabled={stopped}
          onClick={() => void emergency()}
        >
          <Icon name="octagon" />
          Emergency stop
        </Button>
        <span className="mx-1 h-6 w-px bg-border" aria-hidden />
        <Button
          variant="ghost"
          className="h-10"
          busy={busy === "overlay"}
          onClick={() => void toggleOverlay()}
        >
          <Icon name={overlayVisible === true ? "eyeOff" : "eye"} />
          {overlayVisible === true
            ? "Hide overlay"
            : overlayVisible === false
              ? "Show overlay"
              : "Show / hide overlay"}
        </Button>
        {mode === "interviewer" && (
          <Button
            variant="ghost"
            className="h-10"
            disabled={busy === "assist" || state !== "active"}
            onClick={() => void followups()}
          >
            <Icon name="sparkle" />
            Suggest follow-ups
          </Button>
        )}
      </div>
      {error && <InlineError className="-mt-3 mb-4">{error}</InlineError>}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <Card title="Transcript" className="min-w-0">
          <LiveTranscript
            segments={live.transcript}
            mode={mode}
            meetingStartedAt={meeting.startedAt}
            className="h-[26rem]"
            emptyText={
              stopped || paused
                ? "Paused — nothing is being transcribed."
                : state === "starting"
                  ? "Starting audio…"
                  : "Listening… the transcript appears here as people speak."
            }
          />
        </Card>

        <div className="min-w-0 space-y-5">
          <Card
            title="Sources"
            bodyClassName="divide-y divide-border"
            actions={<LinkButton onClick={() => setSection("settings/ai")}>AI provider</LinkButton>}
          >
            <SourceLine
              icon="mic"
              title="Microphone"
              status={sourceStatus(mic?.status)}
              level={live.levels.microphone ?? 0}
            />
            <SourceLine
              icon="speaker"
              title="System audio"
              status={sourceStatus(sys?.status)}
              level={live.levels.system ?? 0}
            />
            <ScreenControl settings={settings} on={screenOn} busy={busy} run={run} />
            <p className="px-4 py-2 text-[0.74rem] text-muted">
              Audio sources are chosen when the meeting starts. Pause to stop listening.
            </p>
          </Card>

          <Card
            title={`Questions${questions.length ? ` · ${questions.length}` : ""}`}
            bodyClassName="max-h-80 overflow-y-auto"
          >
            <p role="status" className="sr-only">
              {latestQuestion ? `Question detected: ${latestQuestion.question}` : ""}
            </p>
            {questions.length === 0 ? (
              <p className="px-4 py-3 text-[0.8rem] text-muted">
                {mode === "candidate"
                  ? "Questions you’re asked will appear here."
                  : "Questions detected in the conversation will appear here."}
              </p>
            ) : (
              <ul className="divide-y divide-border">
                {questions.map((q) => (
                  <QuestionItem
                    key={q.id}
                    q={q}
                    mode={mode}
                    startedAt={meeting.startedAt}
                    busy={busy === "assist" || state !== "active"}
                    onSuggest={mode === "candidate" ? (x) => void suggest(x) : null}
                  />
                ))}
              </ul>
            )}
          </Card>

          <Card title="Topics">
            <Topics topics={live.topics} next={live.suggestedNextTopic} />
          </Card>
        </div>
      </div>
    </div>
  );
}
