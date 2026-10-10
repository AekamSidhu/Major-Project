// Explicit screen-source picker: display / window / application, optional
// display region. Nothing is captured here; the caller decides when to start.

import { useId, useMemo, useState } from "react";
import type { ScreenSourceInfo, ScreenSourceKind, ScreenSourceSpec } from "@/bindings";
import { api } from "@/lib/ipc";
import { Button, Select, TextInput } from "@/main/ui";
import { InlineError, Segmented } from "./controls";
import {
  formatRegion,
  parseRegion,
  pickerKind,
  SCREEN_KINDS,
  sourceOptionLabel,
  sourceToSpec,
} from "./helpers";
import { useAsyncData } from "./hooks";
import { Icon } from "./icons";

type PickerKind = Exclude<ScreenSourceKind, "region">;

export interface ScreenSourcePickerProps {
  value: ScreenSourceSpec | null;
  /** Called with the new spec (null = nothing valid selected) and a validation error, if any. */
  onChange: (spec: ScreenSourceSpec | null, error: string | null) => void;
  disabled?: boolean;
}

export function ScreenSourcePicker({ value, onChange, disabled }: ScreenSourcePickerProps) {
  const sources = useAsyncData(api.listScreenSources);
  const [kind, setKind] = useState<PickerKind>(value ? pickerKind(value.kind) : "display");
  const [regionText, setRegionText] = useState(value?.region ? formatRegion(value.region) : "");
  const selectId = useId();
  const regionId = useId();
  const regionHelpId = useId();

  const ofKind = useMemo(
    () => (sources.data ?? []).filter((s) => pickerKind(s.kind) === kind),
    [sources.data, kind],
  );
  const selectedId = value && pickerKind(value.kind) === kind ? value.id : "";
  const selected = ofKind.find((s) => s.id === selectedId) ?? null;
  const missing = value !== null && selectedId !== "" && sources.data !== null && selected === null;
  const region = parseRegion(
    regionText,
    selected ? { width: selected.width, height: selected.height } : undefined,
  );

  const emit = (info: ScreenSourceInfo | null, text: string) => {
    if (!info) {
      onChange(null, null);
      return;
    }
    const r = parseRegion(text, { width: info.width, height: info.height });
    onChange(
      sourceToSpec(info, info.kind === "display" ? r.rect : null),
      info.kind === "display" ? r.error : null,
    );
  };

  const options = [
    {
      value: "",
      label:
        kind === "display"
          ? "Choose a display…"
          : kind === "window"
            ? "Choose a window…"
            : "Choose an application…",
      disabled: true,
    },
    ...ofKind.map((s) => ({
      value: s.id,
      label: sourceOptionLabel(s),
      disabled: s.excludedByRule,
    })),
  ];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          label="Screen source type"
          value={kind}
          options={SCREEN_KINDS}
          disabled={disabled}
          onChange={(k) => {
            setKind(k);
            emit(null, regionText);
          }}
        />
        <Button
          size="sm"
          variant="ghost"
          onClick={sources.reload}
          busy={sources.loading && sources.data !== null}
          disabled={disabled}
          aria-label="Refresh screen sources"
        >
          <Icon name="refresh" className="size-3.5" />
          Refresh
        </Button>
      </div>

      <div>
        <label htmlFor={selectId} className="mb-1 block text-[0.8rem] font-medium text-fg-soft">
          {kind === "display" ? "Display" : kind === "window" ? "Window" : "Application"}
        </label>
        {sources.loading && sources.data === null ? (
          <p role="status" className="text-[0.8rem] text-muted">
            Looking for screen sources…
          </p>
        ) : (
          <Select<string>
            id={selectId}
            value={selected ? selected.id : ""}
            options={options}
            disabled={disabled || ofKind.length === 0}
            onChange={(id) => emit(ofKind.find((s) => s.id === id) ?? null, regionText)}
          />
        )}
        {sources.error && (
          <InlineError className="mt-1.5">
            Could not list screen sources: {sources.error}
          </InlineError>
        )}
        {!sources.loading && !sources.error && ofKind.length === 0 && (
          <p className="mt-1.5 text-[0.78rem] text-muted">
            No {kind === "display" ? "displays" : kind === "window" ? "windows" : "applications"}{" "}
            found.
          </p>
        )}
        {missing && (
          <p className="mt-1.5 text-[0.78rem] text-warn">
            The saved source “{value?.label}” is not available right now. Choose another one.
          </p>
        )}
        {ofKind.some((s) => s.excludedByRule) && (
          <p className="mt-1.5 text-[0.78rem] text-muted">
            Sources excluded by your privacy rules can’t be selected.
          </p>
        )}
      </div>

      {kind === "display" && (
        <div>
          <label htmlFor={regionId} className="mb-1 block text-[0.8rem] font-medium text-fg-soft">
            Region <span className="font-normal text-muted">(optional)</span>
          </label>
          <TextInput
            id={regionId}
            value={regionText}
            placeholder="x, y, width, height"
            inputMode="numeric"
            disabled={disabled}
            aria-describedby={regionHelpId}
            aria-invalid={region.error ? true : undefined}
            className="max-w-56 font-mono"
            onChange={(e) => {
              setRegionText(e.target.value);
              emit(selected, e.target.value);
            }}
          />
          <p id={regionHelpId} className="mt-1 text-[0.76rem] text-muted">
            Physical pixels within the display. Leave empty to use the whole display.
          </p>
          {region.error && <InlineError className="mt-1">{region.error}</InlineError>}
        </div>
      )}
    </div>
  );
}
