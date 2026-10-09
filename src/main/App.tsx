// Main window shell: navigation, live capture indicator, notices, onboarding gate.

import { useEffect, type ReactNode } from "react";
import { api, errorMessage, onMeetingEvent, onUiCommand } from "@/lib/ipc";
import { applyAppearance, watchSystemTheme } from "@/lib/theme";
import { isSectionId, useMain, type SectionId } from "./store";
import { Button, StatusDot, cx } from "./ui";
import { HomeSection } from "./sections/Home";
import { HistorySection } from "./sections/History";
import { PrivacySection } from "./sections/Privacy";
import { DiagnosticsSection } from "./sections/Diagnostics";
import { AiProvidersSection } from "./sections/settings/AiProviders";
import { TranscriptionSection } from "./sections/settings/Transcription";
import { ScreenSection } from "./sections/settings/Screen";
import { AppearanceSection } from "./sections/settings/Appearance";
import { ShortcutsSection } from "./sections/settings/Shortcuts";
import { ProfileSection } from "./sections/settings/Profile";
import { GeneralSection } from "./sections/settings/General";
import { Onboarding } from "./onboarding/Onboarding";

interface NavItem {
  id: SectionId;
  label: string;
}

const NAV: ReadonlyArray<{ group: string; items: ReadonlyArray<NavItem> }> = [
  {
    group: "Meeting",
    items: [
      { id: "home", label: "Home" },
      { id: "history", label: "History" },
    ],
  },
  { group: "Privacy", items: [{ id: "privacy", label: "Privacy & data" }] },
  {
    group: "Settings",
    items: [
      { id: "settings/ai", label: "AI providers" },
      { id: "settings/transcription", label: "Transcription & audio" },
      { id: "settings/screen", label: "Screen context" },
      { id: "settings/appearance", label: "Overlay & appearance" },
      { id: "settings/shortcuts", label: "Shortcuts" },
      { id: "settings/profile", label: "Profile & context" },
      { id: "settings/general", label: "General" },
    ],
  },
  { group: "System", items: [{ id: "diagnostics", label: "Diagnostics" }] },
];

function renderSection(id: SectionId): ReactNode {
  switch (id) {
    case "home":
      return <HomeSection />;
    case "history":
      return <HistorySection />;
    case "privacy":
      return <PrivacySection />;
    case "settings/ai":
      return <AiProvidersSection />;
    case "settings/transcription":
      return <TranscriptionSection />;
    case "settings/screen":
      return <ScreenSection />;
    case "settings/appearance":
      return <AppearanceSection />;
    case "settings/shortcuts":
      return <ShortcutsSection />;
    case "settings/profile":
      return <ProfileSection />;
    case "settings/general":
      return <GeneralSection />;
    case "diagnostics":
      return <DiagnosticsSection />;
  }
}

/** Always-visible statement of what is being captured (spec §60). */
function CaptureIndicator() {
  const live = useMain((s) => s.live);
  const setSection = useMain((s) => s.setSection);
  if (live.processingStopped) {
    return (
      <button
        type="button"
        onClick={() => setSection("home")}
        className="flex w-full items-center gap-2 rounded-md border border-danger/40 bg-danger-soft px-2.5 py-2 text-left text-[0.78rem] font-medium text-danger"
      >
        <StatusDot tone="danger" />
        Processing stopped
      </button>
    );
  }
  if (live.state === "active" || live.state === "paused") {
    const active = live.state === "active";
    return (
      <button
        type="button"
        onClick={() => setSection("home")}
        className="flex w-full items-center gap-2 rounded-md border border-border bg-surface-2 px-2.5 py-2 text-left text-[0.78rem]"
      >
        <StatusDot tone={active ? "danger" : "warn"} pulse={active} />
        <span className="font-medium text-fg">{active ? "Meeting in progress" : "Meeting paused"}</span>
      </button>
    );
  }
  return null;
}

function Notices() {
  const notices = useMain((s) => s.notices);
  const dismiss = useMain((s) => s.dismissNotice);
  useEffect(() => {
    if (notices.length === 0) return;
    const timers = notices
      .filter((n) => n.level === "info")
      .map((n) => window.setTimeout(() => dismiss(n.id), 6000));
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [notices, dismiss]);
  if (notices.length === 0) return null;
  return (
    <div aria-live="polite" className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-80 flex-col gap-2">
      {notices.map((n) => (
        <div
          key={n.id}
          role={n.level === "error" ? "alert" : "status"}
          className="om-fade-in pointer-events-auto flex items-start gap-2.5 rounded-lg border border-border bg-raised px-3 py-2.5 text-[0.82rem] shadow-lg"
        >
          <span className="mt-1.5">
            <StatusDot tone={n.level === "error" ? "danger" : n.level === "warning" ? "warn" : "accent"} />
          </span>
          <span className="min-w-0 flex-1 text-fg-soft">{n.message}</span>
          <button
            type="button"
            aria-label="Dismiss notification"
            onClick={() => dismiss(n.id)}
            className="text-muted hover:text-fg"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

export function App() {
  const ready = useMain((s) => s.ready);
  const loadError = useMain((s) => s.loadError);
  const settings = useMain((s) => s.settings);
  const appInfo = useMain((s) => s.appInfo);
  const section = useMain((s) => s.section);
  const setSection = useMain((s) => s.setSection);

  // Initial load + event subscriptions.
  useEffect(() => {
    const store = useMain.getState();
    void store.init();
    const unsubs: Array<Promise<() => void>> = [
      onMeetingEvent((ev) => useMain.getState().applyEvent(ev)),
      onUiCommand((cmd) => {
        const s = useMain.getState();
        if (cmd.type === "NAVIGATE" && isSectionId(cmd.section)) s.setSection(cmd.section);
        if (cmd.type === "SETTINGS_CHANGED") void s.reloadSettings();
      }),
    ];
    return () => {
      unsubs.forEach((p) => void p.then((un) => un()).catch(() => {}));
    };
  }, []);

  // Appearance follows settings (main window uses the overlay's theme choice).
  useEffect(() => {
    if (!settings) return;
    const apply = () =>
      applyAppearance({
        theme: settings.overlay.theme,
        density: "comfortable",
        fontSize: 14,
        reducedMotion: settings.overlay.reducedMotion,
      });
    apply();
    return watchSystemTheme(apply);
  }, [settings]);

  if (!ready) {
    return <div className="flex h-full items-center justify-center text-[0.86rem] text-muted">Loading…</div>;
  }
  if (loadError || !settings) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <p className="font-medium">OpenMeet could not load its settings.</p>
        <p className="max-w-md text-[0.86rem] text-muted">{loadError}</p>
        <Button
          onClick={() => {
            void useMain.getState().init();
          }}
        >
          Try again
        </Button>
      </div>
    );
  }
  if (!settings.general.onboardingComplete) {
    return (
      <>
        <Onboarding />
        <Notices />
      </>
    );
  }

  const devMode = settings.general.developerMode;
  return (
    <div className="flex h-full">
      <nav aria-label="Sections" className="flex w-56 shrink-0 flex-col border-r border-border bg-surface px-3 py-4">
        <div className="mb-4 flex items-center gap-2 px-2">
          <span aria-hidden className="inline-block size-3 rounded-full border-[3px] border-accent" />
          <span className="text-[0.95rem] font-semibold tracking-tight">OpenMeet</span>
          {devMode && <span className="ml-auto text-[0.68rem] text-muted">dev</span>}
        </div>
        <div className="mb-3">
          <CaptureIndicator />
        </div>
        <div className="flex-1 overflow-y-auto">
          {NAV.map((g) => (
            <div key={g.group} className="mb-3">
              <div className="px-2 pb-1 text-[0.68rem] font-semibold tracking-wider text-muted uppercase">{g.group}</div>
              <ul>
                {g.items.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      aria-current={section === item.id ? "page" : undefined}
                      onClick={() => setSection(item.id)}
                      className={cx(
                        "w-full rounded-md px-2 py-1.5 text-left text-[0.84rem] transition-colors",
                        section === item.id ? "bg-accent-soft font-medium text-fg" : "text-fg-soft hover:bg-surface-2",
                      )}
                    >
                      {item.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="px-2 pt-2 text-[0.7rem] text-muted">
          {appInfo ? `v${appInfo.version}` : null}
          <button
            type="button"
            className="ml-2 underline-offset-2 hover:underline"
            onClick={() => {
              api.overlayToggle().catch((e) => useMain.getState().notify("error", errorMessage(e)));
            }}
          >
            Toggle overlay
          </button>
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-8 py-7">{renderSection(section)}</div>
      </main>
      <Notices />
    </div>
  );
}
