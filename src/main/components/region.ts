// Candidate video region for the answer-integrity reading signal: the
// rectangle is stored in absolute screen coordinates (physical pixels of the
// virtual desktop, as the local AI sidecar captures it) and edited relative
// to one display. Pure helpers (unit-tested).

import type { DisplayInfo, Rect } from "@/bindings";

/** Smallest / largest region the sidecar accepts (pixels). */
export const REGION_MIN = 17;
export const REGION_MAX = 8192;

export interface RelativeRegion {
  displayKey: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export type RegionPreset = "full" | "left" | "right" | "top_right";

export const REGION_PRESETS: ReadonlyArray<{ value: RegionPreset; label: string }> = [
  { value: "right", label: "Right half" },
  { value: "left", label: "Left half" },
  { value: "top_right", label: "Top-right quarter" },
  { value: "full", label: "Whole display" },
];

/** The display holding the centre of `rect` (absolute coordinates). */
export function displayOf(rect: Rect, displays: ReadonlyArray<DisplayInfo>): DisplayInfo | null {
  const cx = rect.x + rect.width / 2;
  const cy = rect.y + rect.height / 2;
  return (
    displays.find((d) => cx >= d.x && cx < d.x + d.width && cy >= d.y && cy < d.y + d.height) ??
    null
  );
}

/** `rect` relative to the display that holds it. */
export function toRelative(
  rect: Rect,
  displays: ReadonlyArray<DisplayInfo>,
): RelativeRegion | null {
  const d = displayOf(rect, displays);
  if (!d) return null;
  return {
    displayKey: d.key,
    x: rect.x - d.x,
    y: rect.y - d.y,
    width: rect.width,
    height: rect.height,
  };
}

/** Absolute rectangle of a relative region; null when its display is gone. */
export function toAbsolute(r: RelativeRegion, displays: ReadonlyArray<DisplayInfo>): Rect | null {
  const d = displays.find((x) => x.key === r.displayKey);
  if (!d) return null;
  return { x: d.x + r.x, y: d.y + r.y, width: r.width, height: r.height };
}

/** A preset region on `display`. */
export function presetRegion(display: DisplayInfo, preset: RegionPreset): RelativeRegion {
  const w = display.width;
  const h = display.height;
  const half = Math.floor(w / 2);
  switch (preset) {
    case "full":
      return { displayKey: display.key, x: 0, y: 0, width: w, height: h };
    case "left":
      return { displayKey: display.key, x: 0, y: 0, width: half, height: h };
    case "right":
      return { displayKey: display.key, x: half, y: 0, width: w - half, height: h };
    case "top_right":
      return {
        displayKey: display.key,
        x: half,
        y: 0,
        width: w - half,
        height: Math.floor(h / 2),
      };
  }
}

/** Why a region cannot be used, or null when it is fine. */
export function validateRegion(
  r: RelativeRegion,
  displays: ReadonlyArray<DisplayInfo>,
): string | null {
  const d = displays.find((x) => x.key === r.displayKey);
  if (!d) return "Choose a display.";
  const nums = [r.x, r.y, r.width, r.height];
  if (nums.some((n) => !Number.isInteger(n))) return "Use whole pixels.";
  if (r.x < 0 || r.y < 0) return "Left and top must be 0 or more.";
  if (r.width < REGION_MIN || r.height < REGION_MIN) {
    return `The region must be at least ${REGION_MIN} × ${REGION_MIN} pixels.`;
  }
  if (r.width > REGION_MAX || r.height > REGION_MAX) {
    return `The region must be at most ${REGION_MAX} pixels wide and high.`;
  }
  if (r.x + r.width > d.width || r.y + r.height > d.height) {
    return `The region must fit on the display (${d.width} × ${d.height}).`;
  }
  return null;
}

/** "640 × 360 at 1920, 120". */
export function describeRegion(rect: Rect): string {
  return `${rect.width} × ${rect.height} at ${rect.x}, ${rect.y}`;
}
