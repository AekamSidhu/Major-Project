// Home — not in a meeting: the calm, single-page Start Meeting flow
// (mode → audio → optional screen → privacy check → meeting app → start).

import { useId, useMemo, useState, type ReactNode } from "react";
import type { AudioDevices, CapturePrivacyStatus, MeetingAppInfo, Mode, Settings } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "@/main/store";
import { Badge, Banner, Button, PageHeader, Spinner, TextInput, Toggle, cx } from "@/main/ui";
import { AskCard } from "./AskCard";
import { InlineError, LinkButton } from "./controls";
import { DemoCard } from "./DemoCard";
import {
  buildStartRequest,
  deviceName,
  initialStartForm,
  locationBadge,
  retentionLabel,
  validateStartForm,
  type StartForm,
} from "./helpers";
import { useAsyncData, useMeetingEvents } from "./hooks";
import { Icon, type IconName } from "./icons";
import { ScreenSourcePicker } from "./ScreenSourcePicker";

const MODES: ReadonlyArray<{ value: Mode; title: string; description: string }> = [
  {
    value: "candidate",
    title: "Candidate",
    description: "You’re being interviewed. OpenMeet suggests answers to the questions you’re asked.",
  },
  {
    value: "interviewer",
    title: "Interviewer",
    description: "You’re interviewing. OpenMeet suggests follow-up questions grounded in what the candidate said.",
  },
];

function Step({
  n,
  title,
  description,
  actions,
  children,
}: {
  n: number;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className="grid grid-cols-[1.5rem_minmax(0,1fr)] gap-x-3 px-5 py-4">
      <span
        aria-hidden
        className="mt-px flex size-5 items-center justify-center rounded-full border border-border-strong text-[0.68rem] font-semibold text-muted tabular-nums"
      >
        {n}
      </span>
      <div className="min-w-0">
        <div className="mb-2.5 flex items-start justify-between gap-4">
          <div>
            <h2 id={headingId} className="text-[0.92rem] font-semibold text-fg">
              {title}
            </h2>
            {description && <p className="mt-0.5 text-[0.8rem] text-muted">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
        {children}
      </div>
    </section>
  );
}

function SourceRow({
  icon,
  title,
  detail,
  control,
  note,
}: {
  icon: IconName;
  title: string;
  detail: ReactNode;
  control: ReactNode;
  note?: ReactNode;
}) {
  return (
    <div className="flex items-start gap-3 py-2">
      <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-md bg-surface-2 text-fg-soft">
        <Icon name={icon} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[0.86rem] font-medium text-fg">{title}</div>
        <div className="truncate text-[0.78rem] text-muted">{detail}</div>
        {note}
      </div>
      <div className="shrink-0 pt-1">{control}</div>
    </div>
  );
}

function ModePicker({ value, onChange }: { value: Mode; onChange: (m: Mode) => void }) {
  const name = useId();
  return (
    <div role="radiogroup" aria-label="Meeting mode" className="grid gap-2 sm:grid-cols-2">
      {MODES.map((m) => {
        const checked = value === m.value;
        return (
          <label
            key={m.value}
            className={cx(
              "flex cursor-pointer gap-2.5 rounded-lg border px-3 py-2.5 transition-colors",
              "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent",
              checked ? "border-accent bg-accent-soft" : "border-border hover:border-border-strong",
            )}
          >
            <input
              type="radio"
              name={name}
              value={m.value}
              checked={checked}
              onChange={() => onChange(m.value)}
              className="mt-1 size-3.5 shrink-0 accent-[var(--om-accent)]"
            />
            <span className="min-w-0">
              <span className="block text-[0.86rem] font-semibold text-fg">{m.title}</span>
              <span className="block text-[0.78rem] leading-snug text-fg-soft">{m.description}</span>
            </span>
          </label>
        );
      })}
    </div>
  );
}

function AudioStep({
  form,
  update,
  settings,
  devices,
  devicesError,
  devicesLoading,
  reloadDevices,
}: {
  form: StartForm;
  update: (patch: Partial<StartForm>) => void;
  settings: Settings;
  devices: AudioDevices | null;
  devicesError: string | null;
  devicesLoading: boolean;
  reloadDevices: () => void;
}) {
  const setSection = useMain((s) => s.setSection);
  const support = devices?.systemAudio ?? null;
  const unsupported = support !== null && !support.supported;
  const micName = deviceName(settings.audio.microphoneDevice, devices?.inputs ?? null, "Default microphone");
  const sysName = deviceName(settings.audio.systemAudioDevice, devices?.outputs ?? null, "Default output device");
  return (
    <Step
      n={2}
      title="Audio"
      description="What OpenMeet listens to. Nothing is captured until you press Start."
      actions={
        <LinkButton onClick={() => setSection("settings/transcription")}>
          Devices
          <Icon name="arrowRight" className="size-3.5" />
        </LinkButton>
      }
    >
      <div className="divide-y divide-border">
        <SourceRow
          icon="mic"
          title="Microphone"
          detail={<>Your voice · {micName}</>}
          control={
            <Toggle label="Microphone" checked={form.microphone} onChange={(v) => update({ microphone: v })} />
          }
        />
        <SourceRow
          icon="speaker"
          title="System audio"
          detail={
            <>
              Other participants · {sysName}
              {support?.supported && support.mechanism ? ` · ${support.mechanism}` : ""}
            </>
          }
          control={
            <Toggle
              label="System audio"
              checked={form.systemAudio && !unsupported}
              disabled={unsupported}
              onChange={(v) => update({ systemAudio: v })}
            />
          }
          note={
            unsupported ? (
              <p className="mt-1 text-[0.78rem] text-warn">
                Not available on this system{support.note ? `: ${support.note}` : "."}
              </p>
            ) : null
          }
        />
      </div>
      {devicesLoading && devices === null && (
        <p role="status" className="mt-1 flex items-center gap-2 text-[0.78rem] text-muted">
          <Spinner /> Checking audio devices…
        </p>
      )}
      {devicesError && (
        <div className="mt-1 flex items-center gap-3">
          <InlineError>Could not list audio devices: {devicesError}</InlineError>
          <Button size="sm" variant="ghost" onClick={reloadDevices}>
            Retry
          </Button>
        </div>
      )}
    </Step>
  );
}

function captureSummary(status: CapturePrivacyStatus): { text: string; tone: "ok" | "warn" } {
  if (!status.supported) return { text: "Not available on this platform — the overlay can appear in screen shares", tone: "warn" };
  if (status.active) return { text: "Overlay hidden from screen sharing and recordings", tone: "ok" };
  if (status.requested) return { text: "Requested, but not active right now", tone: "warn" };
  return { text: "Off — the overlay is visible in screen shares", tone: "warn" };
}

function PrivacyStep({ settings }: { settings: Settings }) {
  const setSection = useMain((s) => s.setSection);
  const flowKey = useMemo(
    () => JSON.stringify([settings.ai, settings.transcription, settings.privacy, settings.audio, settings.screen]),
    [settings],
  );
  const flow = useAsyncData(api.getDataFlow, flowKey);
  const capture = useAsyncData(api.getCapturePrivacyStatus);
  const reloadCapture = capture.reload;
  useMeetingEvents((ev) => {
    if (ev.type === "PRIVACY_STATE_CHANGED") reloadCapture();
  });
  const nodes = flow.data?.nodes.filter((n) => n.location !== "disabled") ?? [];
  const cap = capture.data ? captureSummary(capture.data) : null;

  return (
    <Step
      n={4}
      title="Privacy check"
      description="Where meeting data goes with your current settings."
      actions={
        <LinkButton onClick={() => setSection("privacy")}>
          Privacy & data
          <Icon name="arrowRight" className="size-3.5" />
        </LinkButton>
      }
    >
      <dl className="space-y-3 text-[0.82rem]">
        <div>
          <dt className="mb-1 text-[0.72rem] font-semibold tracking-wide text-muted uppercase">Data flow</dt>
          <dd>
            {flow.loading && !flow.data && (
              <p role="status" className="flex items-center gap-2 text-muted">
                <Spinner /> Checking providers…
              </p>
            )}
            {flow.error && (
              <div className="flex items-center gap-3">
                <InlineError>Could not load the data flow: {flow.error}</InlineError>
                <Button size="sm" variant="ghost" onClick={flow.reload}>
                  Retry
                </Button>
              </div>
            )}
            {flow.data && (
              <>
                <ul className="flex flex-wrap gap-1.5">
                  {nodes.map((n) => {
                    const b = locationBadge(n.location);
                    return (
                      <li
                        key={n.component}
                        className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1"
                        title={n.endpoint ? `${n.provider} · ${n.endpoint}` : n.provider}
                      >
                        <span className="text-fg-soft">{n.label}</span>
                        <Badge tone={b.tone}>{b.label}</Badge>
                      </li>
                    );
                  })}
                </ul>
                <p className={cx("mt-1.5 text-[0.8rem]", flow.data.allLocal ? "text-ok" : "text-fg-soft")}>
                  {flow.data.statement}
                  {flow.data.localOnlyMode && <span className="ml-1.5 text-muted">(local-only mode is on)</span>}
                </p>
              </>
            )}
          </dd>
        </div>
        <div className="flex items-start gap-2">
          <Icon name="shield" className={cx("mt-0.5", cap?.tone === "ok" ? "text-ok" : "text-warn")} />
          <div className="min-w-0">
            <dt className="font-medium text-fg">Capture privacy</dt>
            <dd className="text-fg-soft">
              {capture.loading && !capture.data && <span className="text-muted">Checking…</span>}
              {capture.error && <InlineError>{capture.error}</InlineError>}
              {cap && capture.data && (
                <>
                  <span className={cap.tone === "ok" ? "text-ok" : "text-warn"}>{cap.text}</span>
                  {capture.data.note && <span className="block text-[0.76rem] text-muted">{capture.data.note}</span>}
                </>
              )}
            </dd>
          </div>
        </div>
        <div className="flex items-start gap-2">
          <Icon name="info" className="mt-0.5 text-muted" />
          <div>
            <dt className="font-medium text-fg">Retention</dt>
            <dd className="text-fg-soft">{retentionLabel(settings.privacy.retention)}</dd>
          </div>
        </div>
      </dl>
    </Step>
  );
}

function appLabel(app: MeetingAppInfo): string {
  return app.windowTitle && app.windowTitle !== app.name ? `${app.name} — ${app.windowTitle}` : app.name;
}

function MeetingAppStep({ value, onChange }: { value: string | null; onChange: (id: string | null) => void }) {
  const apps = useAsyncData(api.detectMeetingApps);
  const name = useId();
  const list = apps.data ?? [];
  const options: Array<{ id: string | null; label: string; detail?: string }> = [
    { id: null, label: "Stay in OpenMeet" },
    ...list.map((a) => ({ id: a.id, label: appLabel(a), detail: a.processName })),
  ];
  return (
    <Step
      n={5}
      title="Meeting app"
      description="Bring your meeting window to the front after start. Nothing is injected into it."
      actions={
        <Button
          size="sm"
          variant="ghost"
          onClick={apps.reload}
          busy={apps.loading && apps.data !== null}
          aria-label="Detect meeting apps again"
        >
          <Icon name="refresh" className="size-3.5" />
          Detect
        </Button>
      }
    >
      {apps.loading && apps.data === null ? (
        <p role="status" className="flex items-center gap-2 text-[0.8rem] text-muted">
          <Spinner /> Looking for meeting apps…
        </p>
      ) : (
        <div role="radiogroup" aria-label="Meeting app to focus after start" className="space-y-0.5">
          {options.map((o) => (
            <label
              key={o.id ?? "none"}
              className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 text-[0.84rem] hover:bg-surface-2"
            >
              <input
                type="radio"
                name={name}
                checked={value === o.id}
                onChange={() => onChange(o.id)}
                className="size-3.5 shrink-0 accent-[var(--om-accent)]"
              />
              <span className="min-w-0 truncate text-fg">{o.label}</span>
              {o.detail && <span className="shrink-0 text-[0.74rem] text-muted">{o.detail}</span>}
            </label>
          ))}
          {apps.data !== null && list.length === 0 && !apps.error && (
            <p className="px-2 pt-1 text-[0.78rem] text-muted">
              No meeting apps detected. Open Zoom, Teams or your browser meeting first, then press Detect.
            </p>
          )}
        </div>
      )}
      {apps.error && <InlineError className="mt-1.5">Could not detect meeting apps: {apps.error}</InlineError>}
    </Step>
  );
}

export function StartMeeting({ settings }: { settings: Settings }) {
  const processingStopped = useMain((s) => s.live.processingStopped);
  const [form, setForm] = useState<StartForm>(() => initialStartForm(settings));
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const devices = useAsyncData(api.listAudioDevices);
  const titleId = useId();

  const update = (patch: Partial<StartForm>) => {
    setForm((f) => ({ ...f, ...patch }));
    setStartError(null);
  };

  // Unknown support (still loading / failed) is left to the backend to decide.
  const systemAudioSupported = devices.data?.systemAudio.supported ?? true;
  const problem = validateStartForm(form, systemAudioSupported);

  const start = async () => {
    if (problem || starting) return;
    setStarting(true);
    setStartError(null);
    try {
      await api.startMeeting(buildStartRequest(form, systemAudioSupported));
      await useMain.getState().refreshSnapshot();
    } catch (e) {
      setStartError(errorMessage(e));
    } finally {
      setStarting(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Start a meeting"
        description="OpenMeet only listens after you press Start, and only to the sources you choose here."
      />
      {processingStopped && (
        <Banner tone="danger" title="Processing stopped">
          The emergency stop halted audio, screen and AI processing. Starting a meeting turns processing back on.
        </Banner>
      )}
      <form
        aria-label="Start meeting"
        onSubmit={(e) => {
          e.preventDefault();
          void start();
        }}
        className="mb-5 divide-y divide-border rounded-lg border border-border bg-surface"
      >
        <Step n={1} title="Mode" description="Which side of the conversation OpenMeet assists.">
          <ModePicker value={form.mode} onChange={(mode) => update({ mode })} />
          <div className="mt-3">
            <label htmlFor={titleId} className="mb-1 block text-[0.8rem] font-medium text-fg-soft">
              Title <span className="font-normal text-muted">(optional)</span>
            </label>
            <TextInput
              id={titleId}
              value={form.title}
              maxLength={120}
              placeholder={form.mode === "candidate" ? "e.g. Acme — system design round" : "e.g. Backend engineer — J. Doe"}
              onChange={(e) => update({ title: e.target.value })}
              className="max-w-md"
            />
          </div>
        </Step>

        <AudioStep
          form={form}
          update={update}
          settings={settings}
          devices={devices.data}
          devicesError={devices.error}
          devicesLoading={devices.loading}
          reloadDevices={devices.reload}
        />

        <Step
          n={3}
          title="Screen context"
          description="Optional and off by default. OpenMeet only looks at the one source you pick."
          actions={
            <Toggle
              label="Use screen context"
              checked={form.screenEnabled}
              onChange={(v) => update({ screenEnabled: v, screenError: v ? form.screenError : null })}
            />
          }
        >
          {form.screenEnabled ? (
            <ScreenSourcePicker
              value={form.screen}
              onChange={(screen, screenError) => update({ screen, screenError })}
            />
          ) : (
            <p className="text-[0.8rem] text-muted">
              Off. Turn it on to let suggestions use what’s on a display, window or application (e.g. code or a
              diagram).
            </p>
          )}
        </Step>

        <PrivacyStep settings={settings} />

        <MeetingAppStep value={form.focusAppId} onChange={(focusAppId) => update({ focusAppId })} />

        <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-4">
          <p className="flex max-w-md items-start gap-2 text-[0.8rem] text-fg-soft">
            <Icon name="info" className="mt-0.5 text-accent" />
            <span>
              Make sure recording and AI assistance are allowed in this meeting, and tell participants where that’s
              required.
            </span>
          </p>
          <Button type="submit" variant="primary" busy={starting} disabled={problem !== null} className="h-10 px-5 text-[0.92rem]">
            <Icon name="play" />
            Start meeting
          </Button>
          {(problem || startError) && (
            <div className="w-full">
              {startError ? (
                <InlineError>Could not start the meeting: {startError}</InlineError>
              ) : (
                <p className="text-[0.8rem] text-warn">{problem}</p>
              )}
            </div>
          )}
        </div>
      </form>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <DemoCard />
        <AskCard settings={settings} />
      </div>
    </div>
  );
}
