// Overlay & appearance: look, size, placement and behaviour of the overlay
// (spec §34–§37, §61–§66). The live preview is the real overlay.

import { useState } from "react";
import type { OverlayPosition } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { NumberSetting, SelectSetting, SliderSetting } from "../../components/fields";
import { useLoader } from "../../components/useLoader";
import { saveSettings } from "../../components/save";
import { DENSITY_OPTIONS, LAYOUT_OPTIONS, POSITION_LABELS, THEME_OPTIONS } from "../../components/settingsMeta";
import { useSettings } from "../../components/useSettings";
import { Banner, Button, Field, PageHeader, Section, Select, ToggleField } from "../../ui";

const POSITIONS: ReadonlyArray<OverlayPosition> = ["top_left", "top_right", "bottom_left", "bottom_right", "custom"];

function Placement() {
  const o = useSettings().overlay;
  const displays = useLoader(api.listDisplays);
  const [error, setError] = useState<string | null>(null);

  const run = async (f: () => Promise<unknown>) => {
    try {
      await f();
      setError(null);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const list = displays.data ?? [];
  const known = o.targetDisplay !== null && list.some((d) => d.key === o.targetDisplay);
  const displayOptions = [
    { value: "", label: o.layout === "separate_display" ? "Automatic (a secondary display)" : "Automatic (primary display)" },
    ...list.map((d) => ({
      value: d.key,
      label: `${d.name}${d.isPrimary ? " (primary)" : ""} — ${d.width}×${d.height}, ${Math.round(d.scaleFactor * 100)}%, ${d.orientation}`,
    })),
    ...(o.targetDisplay !== null && !known ? [{ value: o.targetDisplay, label: "Disconnected display" }] : []),
  ];

  return (
    <Section title="Placement" description="Drag the overlay by its top bar to move it; choose Custom to keep that spot.">
      {error && <Banner tone="danger">{error}</Banner>}
      <Field label="Position" description="Custom remembers the dragged position per display and meeting app." inline>
        <Select
          value={o.position}
          className="w-56"
          options={POSITIONS.map((p) => ({ value: p, label: POSITION_LABELS[p] }))}
          onChange={(p) => void run(() => api.overlaySetPosition(p))}
        />
      </Field>
      <SelectSetting
        label="Display layout"
        description={LAYOUT_OPTIONS.find((l) => l.value === o.layout)?.description}
        value={o.layout}
        options={LAYOUT_OPTIONS}
        onChange={(v) =>
          void saveSettings((d) => {
            d.overlay.layout = v;
          })
        }
      />
      <Field
        label="Display"
        description={displays.error ? `Could not list displays: ${displays.error}` : `${list.length} display${list.length === 1 ? "" : "s"} connected.`}
      >
        <div className="flex items-center gap-2">
          <Select
            value={o.targetDisplay ?? ""}
            options={displayOptions}
            onChange={(key) =>
              void run(() =>
                key === ""
                  ? saveSettings((d) => {
                      d.overlay.targetDisplay = null;
                    })
                  : api.overlayMoveToDisplay(key),
              )
            }
          />
          <Button size="sm" variant="ghost" onClick={displays.reload} disabled={displays.loading}>
            Refresh
          </Button>
        </div>
      </Field>
      <NumberSetting
        label="Distance from the edge"
        value={o.margin}
        min={0}
        max={200}
        unit="px"
        onCommit={(v) =>
          saveSettings((d) => {
            d.overlay.margin = v;
          })
        }
      />
    </Section>
  );
}

export function AppearanceSection() {
  const o = useSettings().overlay;
  const [previewError, setPreviewError] = useState<string | null>(null);

  const showOverlay = async () => {
    try {
      await api.overlayShow();
      setPreviewError(null);
    } catch (e) {
      setPreviewError(errorMessage(e));
    }
  };

  return (
    <>
      <PageHeader
        title="Overlay & appearance"
        description="Changes apply to the overlay immediately."
        actions={<Button onClick={() => void showOverlay()}>Show overlay</Button>}
      />
      {previewError && <Banner tone="danger">{previewError}</Banner>}

      <Section title="Look">
        <SelectSetting
          label="Theme"
          value={o.theme}
          options={THEME_OPTIONS}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.theme = v;
            })
          }
        />
        <SliderSetting
          label="Opacity"
          value={o.opacity}
          min={0.3}
          max={1}
          step={0.01}
          format={(v) => `${Math.round(v * 100)}%`}
          onCommit={(v) =>
            saveSettings((d) => {
              d.overlay.opacity = v;
            })
          }
        />
        <ToggleField
          label="Translucent background"
          description="Blurs what is behind the overlay."
          checked={o.translucent}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.translucent = v;
            })
          }
        />
        <SliderSetting
          label="Text size"
          value={o.fontSize}
          min={11}
          max={24}
          format={(v) => `${v}px`}
          onCommit={(v) =>
            saveSettings((d) => {
              d.overlay.fontSize = v;
            })
          }
        />
        <SelectSetting
          label="Density"
          value={o.density}
          options={DENSITY_OPTIONS}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.density = v;
            })
          }
        />
        <ToggleField
          label="Reduce motion"
          description="No pulsing indicators or fade-ins."
          checked={o.reducedMotion}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.reducedMotion = v;
            })
          }
        />
      </Section>

      <Section title="Size">
        <SliderSetting
          label="Width"
          value={o.width}
          min={260}
          max={900}
          step={10}
          format={(v) => `${v}px`}
          onCommit={(v) =>
            saveSettings((d) => {
              d.overlay.width = v;
            })
          }
        />
        <SliderSetting
          label="Maximum height"
          description="Longer content scrolls inside the overlay."
          value={o.maxHeight}
          min={120}
          max={1600}
          step={10}
          format={(v) => `${v}px`}
          onCommit={(v) =>
            saveSettings((d) => {
              d.overlay.maxHeight = v;
            })
          }
        />
      </Section>

      <Placement />

      <Section title="Behaviour">
        <ToggleField
          label="Show when a meeting starts"
          checked={o.showOnMeetingStart}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.showOnMeetingStart = v;
            })
          }
        />
        <ToggleField
          label="Start as minimal HUD"
          description="Only the status line is shown until you expand it."
          checked={o.minimal}
          onChange={(v) =>
            void saveSettings((d) => {
              d.overlay.minimal = v;
            })
          }
        />
        <ToggleField
          label="Click-through"
          description="Mouse clicks pass through the overlay to the window below. Use the click-through hotkey to interact with it again."
          checked={o.clickThrough}
          onChange={(v) => void api.setClickThrough(v)}
        />
      </Section>
    </>
  );
}
