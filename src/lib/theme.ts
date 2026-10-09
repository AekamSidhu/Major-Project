// Applies appearance settings (theme, density, font size, reduced motion) to
// the document. Shared by the main window and the overlay.

import type { Density, Theme } from "@/bindings";

export interface Appearance {
  theme: Theme;
  density: Density;
  fontSize: number;
  reducedMotion: boolean;
}

function systemPrefersDark(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-color-scheme: dark)").matches
    : true;
}

/** Resolves "system" to the concrete theme. */
export function resolveTheme(theme: Theme): "dark" | "light" {
  if (theme === "system") return systemPrefersDark() ? "dark" : "light";
  return theme;
}

export function applyAppearance(a: Appearance, root: HTMLElement = document.documentElement): void {
  root.dataset.theme = resolveTheme(a.theme);
  root.dataset.density = a.density;
  root.dataset.reducedMotion = String(a.reducedMotion);
  const size = Math.min(24, Math.max(11, Math.round(a.fontSize)));
  root.style.setProperty("--om-font-size", `${size}px`);
}

/** Re-applies "system" theme when the OS preference changes. Returns an unsubscribe fn. */
export function watchSystemTheme(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => {};
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}
