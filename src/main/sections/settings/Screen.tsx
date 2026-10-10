// Screen context: explicit opt-in, source selection, sampling and cost limits
// (spec §21–§26). Off by default; nothing is captured until enabled here or
// when starting a meeting.

import { useState } from "react";
import type { ScreenSourceSpec } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatAccelerator } from "@/lib/format";
import { NumberSetting, SelectSetting, SliderSetting } from "../../components/fields";
import { saveSettings } from "../../components/save";
import { SCREEN_FREQUENCY_OPTIONS } from "../../components/settingsMeta";
import { useMeetingLive, useSettings } from "../../components/useSettings";
import { ScreenSourcePicker } from "../home/ScreenSourcePicker";
import { useMain } from "../../store";
import { Banner, Button, Field, Kbd, PageHeader, Section, ToggleField } from "../../ui";
import { LinkButton } from "../home/controls";

export function ScreenSection() {
  const settings = useSettings();
  const screen = settings.screen;
  const live = useMeetingLive();
  const setSection = useMain((s) => s.setSection);
  const reloadSettings = useMain((s) => s.reloadSettings);
  const [error, setError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const captureKey = settings.shortcuts.bindings.capture_screen ?? "";
  const visionOff = settings.ai.vision.provider === "disabled";

  const apply = async (enabled: boolean, source: ScreenSourceSpec | null) => {
    try {
      await api.setScreenCapture(enabled, source);
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      await reloadSettings();
    }
  };

  const captureNow = async () => {
    setCapturing(true);
    try {
      await api.captureScreenNow();
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setCapturing(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Screen context"
        description="Lets suggestions refer to what is being shared — code, slides, diagrams. Off by default. Only the source you choose is captured, and only during a meeting."
      />
      {error && (
        <Banner tone="danger" title="Screen context">
          {error}
        </Banner>
      )}
      {visionOff && (
        <Banner tone="warn" title="Screen understanding is off">
          Frames would be captured but not interpreted.{" "}
          <LinkButton onClick={() => setSection("settings/ai")}>
            Choose a vision provider
          </LinkButton>
        </Banner>
      )}
      <Section>
        <ToggleField
          label="Use screen context"
          description={
            screen.source
              ? `Source: ${screen.source.label}. Capture runs only while a meeting is active and not paused.`
              : "Choose a source below first."
          }
          checked={screen.enabled}
          disabled={!screen.source && !screen.enabled}
          onChange={(v) => void apply(v, null)}
        />
        <div className="py-2.5">
          <ScreenSourcePicker
            value={screen.source}
            onChange={(spec, err) => {
              if (err || !spec) return;
              void apply(screen.enabled, spec);
            }}
          />
        </div>
        {live && screen.enabled && (
          <Field
            label="Capture now"
            description="Grabs one frame immediately, even in manual mode."
          >
            <Button onClick={() => void captureNow()} busy={capturing}>
              Capture now
            </Button>
          </Field>
        )}
      </Section>

      <Section
        title="Sampling"
        description="Frames are compared first; only meaningful changes are interpreted."
      >
        <SelectSetting
          label="Frequency"
          description={
            captureKey ? (
              <>
                In any mode, <Kbd>{formatAccelerator(captureKey)}</Kbd> captures the screen on
                demand.
              </>
            ) : undefined
          }
          value={screen.frequency}
          options={SCREEN_FREQUENCY_OPTIONS}
          onChange={(v) =>
            void saveSettings((d) => {
              d.screen.frequency = v;
            })
          }
        />
        <SliderSetting
          label="Change threshold"
          description="How different a frame must look before it is processed again. Higher ignores cursor movement and small scrolls."
          value={screen.changeThreshold}
          min={0.01}
          max={0.9}
          step={0.01}
          format={(v) => `${Math.round(v * 100)}%`}
          onCommit={(v) =>
            saveSettings((d) => {
              d.screen.changeThreshold = v;
            })
          }
        />
        <NumberSetting
          label="Vision requests per minute (max)"
          description="Upper bound on calls to the vision model, to limit cost. Text recognition runs locally and is not counted."
          value={screen.maxVisionPerMinute}
          min={1}
          max={60}
          unit="/ min"
          onCommit={(v) =>
            saveSettings((d) => {
              d.screen.maxVisionPerMinute = v;
            })
          }
        />
      </Section>

      <Section title="Storage">
        <ToggleField
          label="Keep screenshots"
          description="Off: frames are processed in memory and discarded; only the text description of each meaningful screen change is kept with the meeting. On: the frames are also saved in the app's data folder and deleted with the meeting."
          checked={screen.storeScreenshots}
          onChange={(v) =>
            void saveSettings((d) => {
              d.screen.storeScreenshots = v;
            })
          }
        />
        <div className="py-2.5 text-[0.8rem] text-muted">
          Sensitive applications, windows, sites or regions can be excluded in{" "}
          <LinkButton onClick={() => setSection("privacy")}>Privacy &amp; data</LinkButton>.
        </div>
      </Section>
    </>
  );
}
