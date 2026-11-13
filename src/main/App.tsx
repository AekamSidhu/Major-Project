// Main window shell: navigation, live capture indicator, notices, onboarding gate.

import { useEffect, type ReactNode } from "react";
import { api, errorMessage, onMeetingEvent, onUiCommand } from "@/lib/ipc";
import { applyAppearance, watchSystemTheme } from "@/lib/theme";
import { isSectionId, useMain, type SectionId } from "./store";
import { Button, StatusDot, cx } from "./ui";
import { Icon, type IconName } from "./sections/home/icons";
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
  icon: IconName;
}

const NAV: ReadonlyArray<{ group: string; items: ReadonlyArray<NavItem> }> = [
  {
    group: "Meeting",
    items: [
      { id: "home", label: "Home", icon: "home" },
      { id: "history", label: "History", icon: "clock" },
    ],
  },
  { group: "Privacy", items: [{ id: "privacy", label: "Privacy & data", icon: "lock" }] },
  {
    group: "Settings",
    items: [
      { id: "settings/ai", label: "AI providers", icon: "cpu" },
      { id: "settings/transcription", label: "Transcription & audio", icon: "wave" },
      { id: "settings/screen", label: "Screen context", icon: "monitor" },
      { id: "settings/appearance", label: "Overlay & appearance", icon: "palette" },
      { id: "settings/shortcuts", label: "Shortcuts", icon: "keyboard" },
      { id: "settings/profile", label: "Profile & context", icon: "user" },
      { id: "settings/general", label: "General", icon: "sliders" },
    ],
  },
  { group: "System", items: [{ id: "diagnostics", label: "Diagnostics", icon: "activity" }] },
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

/** The OpenMeet mark: two overlapping speech rings on the accent gradient. */
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <span
      aria-hidden
      className="om-gradient inline-flex shrink-0 items-center justify-center rounded-[9px] shadow-glow"
      style={{ width: size, height: size }}
    >
      <svg viewBox="0 0 24 24" width={size * 0.62} height={size * 0.62} fill="none">
        <circle cx="9.5" cy="12" r="5.5" stroke="white" strokeWidth="2.2" />
        <circle cx="14.5" cy="12" r="5.5" stroke="white" strokeOpacity="0.65" strokeWidth="2.2" />
      </svg>
    </span>
  );
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
        className="flex w-full items-center gap-2.5 rounded-xl border border-danger/30 bg-danger-soft px-3 py-2.5 text-left text-[0.8rem] font-medium text-danger"
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
        className="group flex w-full items-center gap-2.5 rounded-xl border border-border bg-surface-2 px-3 py-2.5 text-left text-[0.8rem] shadow-card transition-colors hover:border-border-strong"
      >
        <StatusDot tone={active ? "danger" : "warn"} pulse={active} />
        <span className="min-w-0 flex-1">
          <span className="block font-semibold text-fg">
            {active ? "Meeting in progress" : "Meeting paused"}
          </span>
          <span className="block text-[0.72rem] text-muted">Open the live view</span>
        </span>
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
    <div
      aria-live="polite"
      className="pointer-events-none fixed right-4 bottom-4 z-50 flex w-80 flex-col gap-2"
    >
      {notices.map((n) => (
        <div
          key={n.id}
          role={n.level === "error" ? "alert" : "status"}
          className="om-rise pointer-events-auto flex items-start gap-2.5 rounded-xl border border-border bg-raised px-3.5 py-3 text-[0.82rem] shadow-pop"
        >
          <span className="mt-1.5">
            <StatusDot
              tone={n.level === "error" ? "danger" : n.level === "warning" ? "warn" : "accent"}
            />
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
    return (
      <div className="flex h-full items-center justify-center text-[0.86rem] text-muted">
        Loading…
      </div>
    );
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
      <nav
        aria-label="Sections"
        className="flex w-60 shrink-0 flex-col border-r border-border bg-surface/80 px-3 py-4 backdrop-blur-xl"
      >
        <div className="mb-5 flex items-center gap-2.5 px-2">
          <BrandMark />
          <span className="text-[1.02rem] font-semibold tracking-[-0.02em]">OpenMeet</span>
          {devMode && (
            <span className="ml-auto rounded-full border border-border px-1.5 text-[0.62rem] font-medium text-muted">
              dev
            </span>
          )}
        </div>
        <div className="mb-4">
          <CaptureIndicator />
        </div>
        <div className="-mx-1 flex-1 overflow-y-auto px-1">
          {NAV.map((g) => (
            <div key={g.group} className="mb-4">
              <div className="om-eyebrow px-2.5 pb-1.5">{g.group}</div>
              <ul className="space-y-0.5">
                {g.items.map((item) => {
                  const current = section === item.id;
                  return (
                    <li key={item.id}>
                      <button
                        type="button"
                        aria-current={current ? "page" : undefined}
                        onClick={() => setSection(item.id)}
                        className={cx(
                          "group relative flex h-8.5 w-full items-center gap-2.5 rounded-lg px-2.5 text-left text-[0.85rem] transition-colors duration-150",
                          current
                            ? "bg-accent-soft font-medium text-fg"
                            : "text-fg-soft hover:bg-surface-2 hover:text-fg",
                        )}
                      >
                        {current && (
                          <span
                            aria-hidden
                            className="om-gradient absolute top-1.5 bottom-1.5 left-0 w-[3px] rounded-full"
                          />
                        )}
                        <Icon
                          name={item.icon}
                          className={cx(
                            "transition-colors",
                            current ? "text-accent" : "text-muted group-hover:text-fg-soft",
                          )}
                        />
                        {item.label}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-2 flex items-center justify-between gap-2 border-t border-border px-1 pt-3">
          <button
            type="button"
            className="flex h-8 items-center gap-2 rounded-lg px-2 text-[0.8rem] text-fg-soft transition-colors hover:bg-surface-2 hover:text-fg"
            onClick={() => {
              api.overlayToggle().catch((e) => useMain.getState().notify("error", errorMessage(e)));
            }}
          >
            <Icon name="layers" className="text-muted" />
            Toggle overlay
          </button>
          <span className="text-[0.7rem] text-muted tabular-nums">
            {appInfo ? `v${appInfo.version}` : null}
          </span>
        </div>
      </nav>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-10 py-9">{renderSection(section)}</div>
      </main>
      <Notices />
    </div>
  );
}
