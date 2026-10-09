// Privacy & data: where meeting data goes, capture privacy, screen privacy
// rules, retention and deletion (spec §48, §50–§53, §59, §90, §91). The
// backend's data-flow report is the source of truth.

import { useCallback, useState, type FormEvent } from "react";
import type { CapturePrivacyStatus, DataFlowReport, PrivacyRule, PrivacyRuleType } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatDateTime } from "@/lib/format";
import { audioFlow, otherNodes, privacyStatement, screenFlow, screenFlowDisabled, type FlowStep } from "../components/dataFlow";
import { LocationBadge } from "../components/LocationBadge";
import { saveSettings } from "../components/save";
import { RETENTION_OPTIONS } from "../components/settingsMeta";
import { useLoader } from "../components/useLoader";
import { useMeetingLive, useSettings } from "../components/useSettings";
import { RULE_TYPE_LABELS, RULE_TYPE_PLACEHOLDERS, normalizeRuleValue } from "../components/validation";
import { useMain } from "../store";
import { Badge, Banner, Button, ConfirmDialog, Field, PageHeader, Section, Select, Spinner, TextInput, ToggleField, cx } from "../ui";

const RULE_TYPES: ReadonlyArray<PrivacyRuleType> = ["application", "window", "domain", "region"];

function Flow({ steps, muted }: { steps: FlowStep[]; muted?: boolean }) {
  return (
    <ol className={cx("flex flex-wrap items-stretch gap-1.5", muted && "opacity-55")}>
      {steps.map((s, i) => (
        <li key={`${s.key}-${i}`} className="flex items-center gap-1.5">
          <div
            className={cx(
              "min-w-28 rounded-md border px-2.5 py-1.5",
              s.location === "remote" ? "border-warn/50" : "border-border",
              s.location === "disabled" && "border-dashed",
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-[0.8rem] font-medium text-fg">{s.label}</span>
              <LocationBadge location={s.location} endpoint={s.node?.endpoint ?? null} />
            </div>
            <div className="mt-0.5 max-w-48 truncate text-[0.72rem] text-muted" title={s.detail}>
              {s.detail}
            </div>
          </div>
          {i < steps.length - 1 && (
            <span aria-hidden className="text-muted">
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

function DataFlow({ report }: { report: DataFlowReport }) {
  const statement = privacyStatement(report);
  const others = otherNodes(report);
  return (
    <Section
      title="Where your meeting data goes"
      description="Based on the providers that are configured right now."
    >
      <div className="py-3">
        <Banner tone={statement.tone === "ok" ? "ok" : "warn"} title={statement.tone === "ok" ? "Everything stays on this device" : "Some data is sent to cloud providers"}>
          {statement.text}
        </Banner>
      </div>
      <div className="space-y-3 py-3">
        <div>
          <div className="mb-1.5 text-[0.78rem] font-medium text-fg-soft">Audio</div>
          <Flow steps={audioFlow(report)} />
        </div>
        <div>
          <div className="mb-1.5 text-[0.78rem] font-medium text-fg-soft">
            Screen {screenFlowDisabled(report) && <span className="font-normal text-muted">(off)</span>}
          </div>
          <Flow steps={screenFlow(report)} muted={screenFlowDisabled(report)} />
        </div>
        {others.length > 0 && (
          <ul className="flex flex-wrap gap-2 pt-1">
            {others.map((n) => (
              <li key={n.component} className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[0.76rem]">
                <span className="text-fg-soft">{n.label}</span>
                <span className="text-muted">{n.endpoint ? `${n.provider} · ${n.endpoint}` : n.provider}</span>
                <LocationBadge location={n.location} endpoint={n.endpoint} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </Section>
  );
}

function QuickControls() {
  const settings = useSettings();
  const reloadSettings = useMain((s) => s.reloadSettings);
  const [error, setError] = useState<string | null>(null);
  return (
    <Section title="Quick controls" description="Turn data sources and cloud processing off at any time, also during a meeting.">
      {error && <Banner tone="danger">{error}</Banner>}
      <ToggleField
        label="Local-only mode (no cloud processing)"
        description="OpenMeet refuses to use any provider that sends data off this device. Cloud providers stop working until you turn this off."
        checked={settings.privacy.localOnly}
        onChange={(v) =>
          void saveSettings((d) => {
            d.privacy.localOnly = v;
          })
        }
      />
      <ToggleField
        label="Microphone"
        checked={settings.audio.microphoneEnabled}
        onChange={(v) =>
          void saveSettings((d) => {
            d.audio.microphoneEnabled = v;
          })
        }
      />
      <ToggleField
        label="System audio"
        checked={settings.audio.systemAudioEnabled}
        onChange={(v) =>
          void saveSettings((d) => {
            d.audio.systemAudioEnabled = v;
          })
        }
      />
      <ToggleField
        label="Screen context"
        description={settings.screen.source ? undefined : "Choose a source in Screen context settings to enable it."}
        checked={settings.screen.enabled}
        disabled={!settings.screen.enabled && !settings.screen.source}
        onChange={(v) =>
          void api
            .setScreenCapture(v, null)
            .then(() => setError(null), (e: unknown) => setError(errorMessage(e)))
            .finally(() => void reloadSettings())
        }
      />
    </Section>
  );
}

function CapturePrivacy() {
  const settings = useSettings();
  const loader = useLoader(api.getCapturePrivacyStatus);
  const [error, setError] = useState<string | null>(null);
  const status: CapturePrivacyStatus | null = loader.data;

  const toggle = async (enabled: boolean) => {
    try {
      loader.setData(await api.setCaptureExclusion(enabled));
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  let badge: { tone: "ok" | "warn" | "neutral"; text: string } = { tone: "neutral", text: "Checking…" };
  if (status) {
    if (!status.supported) badge = { tone: "warn", text: "Not supported here" };
    else if (status.active) badge = { tone: "ok", text: "Active" };
    else if (status.requested) badge = { tone: "warn", text: "Requested, not active" };
    else badge = { tone: "neutral", text: "Off" };
  }

  return (
    <Section
      title="Capture privacy"
      description="Asks the operating system to leave the overlay out of screen sharing, screenshots and recordings."
      actions={<Badge tone={badge.tone}>{badge.text}</Badge>}
    >
      {(error || loader.error) && <Banner tone="danger">{error ?? loader.error}</Banner>}
      <ToggleField
        label="Hide the overlay from screen capture"
        description={status?.mechanism ? `Uses ${status.mechanism}.` : undefined}
        checked={settings.privacy.captureExclusion}
        disabled={status !== null && !status.supported}
        onChange={(v) => void toggle(v)}
      />
      <div className="space-y-1.5 py-2.5 text-[0.8rem] text-muted">
        {status?.note && <p>{status.note}</p>}
        <p>
          This is not a guarantee. It relies on documented operating-system features that capture tools may ignore —
          for example hardware capture, a phone camera pointed at the screen, some remote-desktop tools, or older
          operating-system versions. Treat the overlay as potentially visible.
        </p>
      </div>
    </Section>
  );
}

function PrivacyRules() {
  const rules = useLoader(api.listPrivacyRules);
  const suggestions = useLoader(api.suggestedPrivacyApps);
  const [type, setType] = useState<PrivacyRuleType>("application");
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const list: PrivacyRule[] = rules.data ?? [];

  const add = async (t: PrivacyRuleType, raw: string) => {
    const norm = normalizeRuleValue(t, raw);
    if (!norm.ok) {
      setError(norm.error);
      return false;
    }
    setBusy(true);
    try {
      const rule = await api.addPrivacyRule(t, norm.value);
      rules.setData([...list.filter((r) => r.id !== rule.id), rule]);
      setError(null);
      return true;
    } catch (e) {
      setError(errorMessage(e));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    try {
      await api.removePrivacyRule(id);
      rules.setData(list.filter((r) => r.id !== id));
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (await add(type, value)) setValue("");
  };

  const existing = new Set(list.filter((r) => r.type === "application").map((r) => r.value.toLowerCase()));
  const suggested = (suggestions.data ?? []).filter((a) => !existing.has(a.toLowerCase()));

  return (
    <Section
      title="Screen privacy rules"
      description="Screen context skips frames that show these applications, windows or websites, and blanks out excluded regions before anything is processed."
    >
      {rules.error && <Banner tone="danger">{rules.error}</Banner>}
      {rules.loading && !rules.data ? (
        <div className="flex items-center gap-2 py-3 text-[0.82rem] text-muted">
          <Spinner /> Loading rules…
        </div>
      ) : list.length === 0 ? (
        <p className="py-3 text-[0.82rem] text-muted">No rules yet.</p>
      ) : (
        <ul className="py-1.5">
          {list.map((r) => (
            <li key={r.id} className="flex items-center gap-3 py-1.5">
              <Badge>{RULE_TYPE_LABELS[r.type]}</Badge>
              <span className="min-w-0 flex-1 truncate font-mono text-[0.82rem] text-fg">{r.value}</span>
              <span className="text-[0.72rem] text-muted">{formatDateTime(r.createdAt)}</span>
              <Button size="sm" variant="ghost" aria-label={`Remove rule ${r.value}`} onClick={() => void remove(r.id)}>
                Remove
              </Button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={(e) => void submit(e)} className="py-3">
        <div className="flex items-center gap-2">
          <Select
            aria-label="Rule type"
            value={type}
            className="w-44"
            options={RULE_TYPES.map((t) => ({ value: t, label: RULE_TYPE_LABELS[t] }))}
            onChange={(t) => {
              setType(t);
              setError(null);
            }}
          />
          <TextInput
            aria-label="Rule value"
            value={value}
            placeholder={RULE_TYPE_PLACEHOLDERS[type]}
            onChange={(e) => setValue(e.target.value)}
            className="h-8.5 flex-1"
          />
          <Button type="submit" variant="primary" busy={busy} disabled={!value.trim()}>
            Add rule
          </Button>
        </div>
        {error && (
          <p role="alert" className="mt-1.5 text-[0.78rem] text-danger">
            {error}
          </p>
        )}
        {type === "region" && (
          <p className="mt-1.5 text-[0.76rem] text-muted">Pixels from the top-left corner of the captured display or window. The region is blacked out before any processing.</p>
        )}
      </form>
      {suggested.length > 0 && (
        <div className="py-3">
          <div className="mb-1.5 text-[0.78rem] text-muted">Suggested sensitive apps installed or running on this computer:</div>
          <div className="flex flex-wrap gap-1.5">
            {suggested.map((a) => (
              <Button key={a} size="sm" onClick={() => void add("application", a)}>
                + {a}
              </Button>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}

function Retention() {
  const settings = useSettings();
  const live = useMeetingLive();
  const notify = useMain((s) => s.notify);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const option = RETENTION_OPTIONS.find((o) => o.value === settings.privacy.retention);

  const deleteAll = async () => {
    setBusy(true);
    try {
      await api.deleteAllData();
      notify("info", "All meeting data was deleted.");
    } catch (e) {
      notify("error", `Could not delete meeting data: ${errorMessage(e)}`);
    } finally {
      setBusy(false);
      setConfirm(false);
    }
  };

  return (
    <Section title="Stored data" description="Meetings are stored in a local database on this device. Nothing is uploaded for storage.">
      <Field label="Keep meetings for" description={option?.description} inline>
        <Select
          value={settings.privacy.retention}
          className="w-56"
          options={RETENTION_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
          onChange={(v) =>
            void saveSettings((d) => {
              d.privacy.retention = v;
            })
          }
        />
      </Field>
      <Field
        label="Delete all meeting data"
        description={
          live
            ? "End the current meeting first."
            : "Permanently deletes every transcript, question, suggestion, summary, screen observation and kept screenshot. Settings, profile and API keys are kept."
        }
        inline
      >
        <Button variant="danger" disabled={live} onClick={() => setConfirm(true)}>
          Delete all…
        </Button>
      </Field>
      <ConfirmDialog
        open={confirm}
        title="Delete all meeting data?"
        message="Every meeting stored on this device is permanently deleted. This cannot be undone."
        confirmLabel="Delete everything"
        danger
        busy={busy}
        onConfirm={() => void deleteAll()}
        onCancel={() => setConfirm(false)}
      />
    </Section>
  );
}

export function PrivacySection() {
  const settings = useSettings();
  // Re-read the report whenever settings that affect providers change.
  const fetchFlow = useCallback(
    () => api.getDataFlow(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [settings.ai, settings.transcription, settings.privacy, settings.audio, settings.screen],
  );
  const flow = useLoader(fetchFlow);
  return (
    <>
      <PageHeader
        title="Privacy & data"
        description="OpenMeet has no telemetry or analytics. API keys are kept in your operating system's credential store."
      />
      {flow.error && <Banner tone="danger" title="Could not build the data-flow report">{flow.error}</Banner>}
      {flow.data ? (
        <DataFlow report={flow.data} />
      ) : (
        !flow.error && (
          <div className="flex items-center gap-2 py-6 text-[0.86rem] text-muted">
            <Spinner /> Checking providers…
          </div>
        )
      )}
      <QuickControls />
      <CapturePrivacy />
      <PrivacyRules />
      <Retention />
    </>
  );
}
