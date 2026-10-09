// General: default mode, startup, notifications, voice commands, developer
// mode and resetting settings (spec §67–§70, §75).

import { useState } from "react";
import { api, errorMessage } from "@/lib/ipc";
import { ChoiceCards } from "../../components/fields";
import { saveSettings } from "../../components/save";
import { MODE_OPTIONS } from "../../components/settingsMeta";
import { useSettings } from "../../components/useSettings";
import { useMain } from "../../store";
import { Banner, Button, ConfirmDialog, Field, PageHeader, Section, ToggleField } from "../../ui";

export function GeneralSection() {
  const settings = useSettings();
  const g = settings.general;
  const appInfo = useMain((s) => s.appInfo);
  const reloadSettings = useMain((s) => s.reloadSettings);
  const notify = useMain((s) => s.notify);
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reset = async () => {
    setResetting(true);
    try {
      await api.resetSettings();
      await reloadSettings();
      setError(null);
      notify("info", "Settings were reset to their defaults.");
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setResetting(false);
      setConfirmReset(false);
    }
  };

  return (
    <>
      <PageHeader title="General" />
      {error && <Banner tone="danger">{error}</Banner>}
      <Section title="Default mode" description="Also switchable from the overlay, the tray menu or the mode hotkey.">
        <ChoiceCards
          label="Mode"
          hideLabel
          value={settings.mode}
          options={MODE_OPTIONS}
          onChange={(m) => {
            // Switches the live meeting too; the backend saves the setting.
            void api.setMode(m).then(reloadSettings, (e: unknown) => setError(errorMessage(e)));
          }}
        />
      </Section>
      <Section title="Startup & notifications">
        <ToggleField
          label="Launch OpenMeet at login"
          description="Starts OpenMeet in the tray. It never starts listening, recording or capturing the screen by itself."
          checked={g.launchAtStartup}
          onChange={(v) =>
            void saveSettings((d) => {
              d.general.launchAtStartup = v;
            })
          }
        />
        <ToggleField
          label="Notifications"
          description="System notifications for important events only, such as a finished meeting summary."
          checked={g.notifications}
          onChange={(v) =>
            void saveSettings((d) => {
              d.general.notifications = v;
            })
          }
        />
      </Section>
      <Section title="Voice commands">
        <ToggleField
          label="Voice commands from your microphone"
          description="Say “OpenMeet, explain that”, “OpenMeet, give me follow-ups” or “OpenMeet, summarize” during a meeting. Only your own microphone is listened to for commands; nothing extra is recorded."
          checked={g.voiceCommands}
          onChange={(v) =>
            void saveSettings((d) => {
              d.general.voiceCommands = v;
            })
          }
        />
      </Section>
      <Section title="Advanced">
        <ToggleField
          label="Developer mode"
          description="Shows latency metrics and the simulation tools in Diagnostics."
          checked={g.developerMode}
          onChange={(v) =>
            void saveSettings((d) => {
              d.general.developerMode = v;
            })
          }
        />
        <Field
          label="Setup guide"
          description="Walk through the first-run setup again. Your settings are kept."
          inline
        >
          <Button
            onClick={() =>
              void saveSettings((d) => {
                d.general.onboardingComplete = false;
              })
            }
          >
            Run setup again
          </Button>
        </Field>
        <Field
          label="Reset settings"
          description="Restores every setting to its default. Your profile, API keys and meetings are kept."
          inline
        >
          <Button variant="danger" onClick={() => setConfirmReset(true)}>
            Reset settings…
          </Button>
        </Field>
      </Section>
      {appInfo && (
        <p className="text-[0.76rem] text-muted">
          OpenMeet {appInfo.version} · {appInfo.platform} {appInfo.arch}
          {appInfo.debugBuild ? " · debug build" : ""} · Apache-2.0 · No telemetry.
        </p>
      )}
      <ConfirmDialog
        open={confirmReset}
        title="Reset all settings?"
        message="Providers, audio, screen, overlay, shortcuts and privacy settings return to their defaults. Your profile, saved API keys and meeting history are not affected."
        confirmLabel="Reset settings"
        danger
        busy={resetting}
        onConfirm={() => void reset()}
        onCancel={() => setConfirmReset(false)}
      />
    </>
  );
}
