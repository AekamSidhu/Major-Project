// Settings accessor for settings pages. The shell renders them only after the
// settings have loaded, so a missing value is a programming error.

import type { Settings } from "@/bindings";
import { useMain } from "../store";

export function useSettings(): Settings {
  const settings = useMain((s) => s.settings);
  if (!settings) throw new Error("Settings are not loaded yet");
  return settings;
}

/** True while a meeting is running or paused. */
export function useMeetingLive(): boolean {
  return useMain((s) => s.live.state === "active" || s.live.state === "paused");
}
