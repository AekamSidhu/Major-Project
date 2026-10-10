// Audio input setup: enable toggle, device picker and a live level meter.
// Shared by Settings → Transcription & audio and onboarding.

import { useEffect, useId, useRef, useState } from "react";
import { create } from "zustand";
import type { AudioDeviceInfo, AudioDevices, AudioSourceKind } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "../store";
import { Banner, Button, Field, Select, ToggleField, cx } from "../ui";
import { RefreshIcon } from "./icons";
import { saveSettings } from "./save";
import { useLoader } from "./useLoader";
import { useMeetingLive, useSettings } from "./useSettings";

/** Which source the backend's single level test is running for. */
const useAudioTest = create<{ testing: AudioSourceKind | null }>(() => ({ testing: null }));

const TEST_DURATION_MS = 15_000;

export function useAudioDevices() {
  return useLoader(api.listAudioDevices);
}

/** Horizontal level meter (0..1). */
export function LevelMeter({
  level,
  label,
  active,
}: {
  level: number;
  label: string;
  active: boolean;
}) {
  const clamped = Math.max(0, Math.min(1, level));
  // Perceptual scaling: speech sits around 0.05–0.3 linear.
  const shown = active ? Math.sqrt(clamped) : 0;
  const pct = Math.round(shown * 100);
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-valuetext={active ? `${pct}%` : "Not measuring"}
      className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2"
    >
      <div
        className={cx(
          "h-full rounded-full transition-[width] duration-100",
          clamped > 0.9 ? "bg-warn" : "bg-ok",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function LevelTest({ source, device }: { source: AudioSourceKind; device: string | null }) {
  const testing = useAudioTest((s) => s.testing === source);
  const meetingLive = useMeetingLive();
  const level = useMain((s) => s.live.levels[source] ?? 0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [silent, setSilent] = useState(false);
  const heard = useRef(false);
  const timers = useRef<{ stop: number | null; silence: number | null }>({
    stop: null,
    silence: null,
  });

  useEffect(() => {
    if (testing && level > 0.02) heard.current = true;
  }, [testing, level]);

  // Stop the test when leaving the page.
  useEffect(() => {
    const t = timers.current;
    return () => {
      if (t.stop !== null) window.clearTimeout(t.stop);
      if (t.silence !== null) window.clearTimeout(t.silence);
      if (useAudioTest.getState().testing === source) {
        useAudioTest.setState({ testing: null });
        api
          .stopAudioTest()
          .catch((e: unknown) =>
            useMain.getState().notify("error", `Audio test: ${errorMessage(e)}`),
          );
      }
    };
  }, [source]);

  const clearTimers = () => {
    const t = timers.current;
    if (t.stop !== null) window.clearTimeout(t.stop);
    if (t.silence !== null) window.clearTimeout(t.silence);
    t.stop = null;
    t.silence = null;
  };

  const stop = async () => {
    clearTimers();
    setBusy(true);
    try {
      await api.stopAudioTest();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      if (useAudioTest.getState().testing === source) useAudioTest.setState({ testing: null });
      setBusy(false);
    }
  };

  const start = async () => {
    clearTimers();
    setBusy(true);
    setError(null);
    setSilent(false);
    heard.current = false;
    try {
      await api.startAudioTest(source, device);
      useAudioTest.setState({ testing: source });
      const t = timers.current;
      t.stop = window.setTimeout(() => {
        t.stop = null;
        if (useAudioTest.getState().testing === source) useAudioTest.setState({ testing: null });
      }, TEST_DURATION_MS);
      t.silence = window.setTimeout(() => {
        t.silence = null;
        if (!heard.current && useAudioTest.getState().testing === source) setSilent(true);
      }, 3500);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const label = source === "microphone" ? "Microphone level" : "System audio level";
  const active = testing || meetingLive;
  return (
    <div className="pb-2.5">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <LevelMeter level={level} label={label} active={active} />
        </div>
        {meetingLive ? (
          <span className="shrink-0 text-[0.76rem] text-muted">Live from the current meeting</span>
        ) : testing ? (
          <Button size="sm" busy={busy} onClick={() => void stop()}>
            Stop test
          </Button>
        ) : (
          <Button size="sm" busy={busy} onClick={() => void start()}>
            Test level
          </Button>
        )}
      </div>
      <div role="status" aria-live="polite" className="mt-1 text-[0.76rem] text-muted">
        {testing &&
          !silent &&
          (source === "microphone"
            ? "Speak normally — the bar should move."
            : "Play some audio — the bar should move.")}
        {testing && silent && (
          <span className="text-warn">
            No sound detected yet. Check that the right device is selected and that OpenMeet may use
            it in your system's privacy settings.
          </span>
        )}
      </div>
      {error && (
        <p role="alert" className="mt-1 text-[0.78rem] text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

function deviceOptions(list: AudioDeviceInfo[], current: string | null, defaultLabel: string) {
  const def = list.find((d) => d.isDefault);
  const options: Array<{ value: string; label: string }> = [
    { value: "", label: def ? `${defaultLabel} (${def.name})` : defaultLabel },
    ...list.map((d) => ({ value: d.id, label: d.isDefault ? `${d.name} — default` : d.name })),
  ];
  if (current && !list.some((d) => d.id === current)) {
    options.push({ value: current, label: `${current} (not connected)` });
  }
  return options;
}

export interface AudioSourceSettingsProps {
  source: AudioSourceKind;
  devices: AudioDevices | null;
  /** Hide the enable toggle (onboarding microphone step). */
  hideToggle?: boolean;
}

/** Enable toggle + device select + level test for one audio source. */
export function AudioSourceSettings({ source, devices, hideToggle }: AudioSourceSettingsProps) {
  const settings = useSettings();
  const selectId = useId();
  const audio = settings.audio;
  const isMic = source === "microphone";
  const enabled = isMic ? audio.microphoneEnabled : audio.systemAudioEnabled;
  const device = isMic ? audio.microphoneDevice : audio.systemAudioDevice;
  const support = devices?.systemAudio;
  const unsupported = !isMic && support !== undefined && !support.supported;
  const list = devices ? (isMic ? devices.inputs : devices.outputs) : [];

  return (
    <div>
      {!hideToggle && (
        <ToggleField
          label={isMic ? "Microphone" : "System audio"}
          description={
            isMic
              ? "Your voice. Transcribed as “You”."
              : unsupported
                ? `Not available on this system: ${support?.note || "system audio capture is unsupported"}`
                : `What other participants say, captured from your speakers${support?.mechanism ? ` (${support.mechanism})` : ""}.`
          }
          checked={enabled && !unsupported}
          disabled={unsupported}
          onChange={(v) =>
            void saveSettings((d) => {
              if (isMic) d.audio.microphoneEnabled = v;
              else d.audio.systemAudioEnabled = v;
            })
          }
        />
      )}
      {!unsupported && (
        <>
          <Field
            label={isMic ? "Input device" : "Output device to capture"}
            description={!isMic && support?.note ? support.note : undefined}
            htmlFor={selectId}
          >
            <Select
              id={selectId}
              value={device ?? ""}
              disabled={!devices || (!enabled && !hideToggle)}
              options={
                devices
                  ? deviceOptions(
                      list,
                      device,
                      isMic ? "System default microphone" : "System default output",
                    )
                  : [{ value: device ?? "", label: "Loading devices…" }]
              }
              onChange={(v) =>
                void saveSettings((d) => {
                  if (isMic) d.audio.microphoneDevice = v === "" ? null : v;
                  else d.audio.systemAudioDevice = v === "" ? null : v;
                })
              }
            />
          </Field>
          {(enabled || hideToggle) && <LevelTest source={source} device={device} />}
        </>
      )}
    </div>
  );
}

/** Device list status + refresh. */
export function AudioDevicesStatus({ loader }: { loader: ReturnType<typeof useAudioDevices> }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1.5 text-[0.78rem] text-muted">
      <span role="status">
        {loader.loading && !loader.data
          ? "Looking for audio devices…"
          : loader.data
            ? `${loader.data.inputs.length} input${loader.data.inputs.length === 1 ? "" : "s"}, ${loader.data.outputs.length} output${loader.data.outputs.length === 1 ? "" : "s"} found`
            : null}
      </span>
      <Button size="sm" variant="ghost" onClick={loader.reload} disabled={loader.loading}>
        <RefreshIcon /> Refresh devices
      </Button>
      {loader.error && (
        <div className="sr-only" role="alert">
          {loader.error}
        </div>
      )}
    </div>
  );
}

export function AudioDevicesError({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <Banner tone="danger" title="Audio devices unavailable">
      {error}
    </Banner>
  );
}
