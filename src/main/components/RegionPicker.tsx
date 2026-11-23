// Picks the candidate's video tile for the answer-integrity reading signal:
// a display, a rectangle on it (presets or exact pixels) and a preview
// captured the way the local AI sidecar will see it.

import { useState } from "react";
import type { DisplayInfo, Rect } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { Banner, Button, Field, Select, TextInput } from "../ui";
import {
  REGION_PRESETS,
  describeRegion,
  presetRegion,
  toAbsolute,
  toRelative,
  validateRegion,
  type RelativeRegion,
} from "./region";
import { useLoader } from "./useLoader";

function initialRegion(
  value: Rect | null,
  displays: ReadonlyArray<DisplayInfo>,
): RelativeRegion | null {
  if (value) {
    const rel = toRelative(value, displays);
    if (rel) return rel;
  }
  const primary = displays.find((d) => d.isPrimary) ?? displays[0];
  return primary ? presetRegion(primary, "right") : null;
}

const FIELDS: ReadonlyArray<{ key: "x" | "y" | "width" | "height"; label: string }> = [
  { key: "x", label: "Left" },
  { key: "y", label: "Top" },
  { key: "width", label: "Width" },
  { key: "height", label: "Height" },
];

interface RegionPickerProps {
  value: Rect | null;
  onSave: (rect: Rect | null) => Promise<boolean>;
}

export function RegionPicker({ value, onSave }: RegionPickerProps) {
  const displays = useLoader(api.listDisplays);
  if (displays.error) {
    return (
      <Banner tone="danger" title="Displays">
        {displays.error}
      </Banner>
    );
  }
  const list = displays.data ?? [];
  const start = list.length > 0 ? initialRegion(value, list) : null;
  if (!start) {
    return (
      <p className="py-2.5 text-[0.82rem] text-muted">
        {displays.loading ? "Loading displays…" : "No display found."}
      </p>
    );
  }
  return <RegionEditor list={list} start={start} value={value} onSave={onSave} />;
}

function RegionEditor({
  list,
  start,
  value,
  onSave,
}: RegionPickerProps & { list: ReadonlyArray<DisplayInfo>; start: RelativeRegion }) {
  const [region, setRegion] = useState<RelativeRegion>(start);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const error = validateRegion(region, list);
  const absolute = error ? null : toAbsolute(region, list);
  const display = list.find((d) => d.key === region.displayKey) ?? null;
  const unchanged =
    !!value &&
    !!absolute &&
    value.x === absolute.x &&
    value.y === absolute.y &&
    value.width === absolute.width &&
    value.height === absolute.height;

  const update = (patch: Partial<RelativeRegion>) => {
    setRegion({ ...region, ...patch });
    setPreview(null);
  };

  const runPreview = async () => {
    if (!absolute) return;
    setBusy(true);
    try {
      setPreview(await api.previewIntegrityRegion(absolute));
      setPreviewError(null);
    } catch (e) {
      setPreview(null);
      setPreviewError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3 py-2.5">
      <Field label="Display" description="Where the meeting window shows the candidate's video.">
        <Select
          value={region.displayKey}
          options={list.map((d) => ({
            value: d.key,
            label: `${d.name}${d.isPrimary ? " (primary)" : ""} · ${d.width} × ${d.height}`,
          }))}
          onChange={(key) => {
            const d = list.find((x) => x.key === key);
            if (d) update(presetRegion(d, "right"));
          }}
        />
      </Field>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Region presets">
        {REGION_PRESETS.map((p) => (
          <Button
            key={p.value}
            size="sm"
            variant="ghost"
            onClick={() => display && update(presetRegion(display, p.value))}
          >
            {p.label}
          </Button>
        ))}
      </div>
      <div className="grid grid-cols-4 gap-2">
        {FIELDS.map((f) => (
          <label key={f.key} className="text-[0.76rem] text-muted">
            {f.label} (px)
            <TextInput
              inputMode="numeric"
              value={String(region[f.key])}
              aria-label={`${f.label} in pixels`}
              onChange={(e) => {
                const n = Number.parseInt(e.target.value.replace(/[^\d]/g, "") || "0", 10);
                update({ [f.key]: n } as Partial<RelativeRegion>);
              }}
              className="mt-1 text-right tabular-nums"
            />
          </label>
        ))}
      </div>
      {error && <p className="text-[0.78rem] text-danger">{error}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={() => void runPreview()} disabled={!absolute} busy={busy}>
          Preview
        </Button>
        <Button
          size="sm"
          variant="primary"
          disabled={!absolute || unchanged}
          onClick={() => absolute && void onSave(absolute)}
        >
          Use this region
        </Button>
        {value && (
          <Button size="sm" variant="ghost" onClick={() => void onSave(null)}>
            Clear region
          </Button>
        )}
        <span className="text-[0.76rem] text-muted">
          {value ? `Saved: ${describeRegion(value)}` : "No region saved"}
        </span>
      </div>
      {previewError && <p className="text-[0.78rem] text-danger">{previewError}</p>}
      {preview && (
        <figure className="space-y-1">
          <img
            src={preview}
            alt="Preview of the candidate video region"
            className="max-h-56 rounded-md border border-border"
          />
          <figcaption className="text-[0.74rem] text-muted">
            The candidate's face should be clearly visible in this area. The preview is not stored.
          </figcaption>
        </figure>
      )}
    </div>
  );
}
