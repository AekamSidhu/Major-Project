// The overlay: a compact, translucent HUD that floats above the meeting.
// It prioritizes the current piece of information: status → question →
// answer / follow-ups → actions (spec §61).

import { useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { onMeetingEvent, onUiCommand } from "@/lib/ipc";
import { parseAnswer } from "@/lib/answer";
import { applyAppearance, watchSystemTheme } from "@/lib/theme";
import {
  generateAnswer,
  hideOverlay,
  openMainWindow,
  pauseMeeting,
  resumeMeeting,
  setClickThrough,
  setMode,
  setPinned,
  submitAsk,
} from "./actions";
import { FollowupsCard, ListeningActions, PrimaryCard, TopicsStrip } from "./cards";
import { useAutoSize, useDragSave, useHotkey, withHotkey } from "./hooks";
import {
  CollapseIcon,
  ChevronDownIcon,
  ExternalIcon,
  EyeOffIcon,
  GripIcon,
  MonitorIcon,
  PauseIcon,
  PlayIcon,
  PointerIcon,
  SendIcon,
  ShieldIcon,
  SparkIcon,
} from "./icons";
import {
  componentLabel,
  componentSection,
  deriveStatus,
  hudLabel,
  isLive,
  screenActive,
  speakerLabel,
  suggestionTitle,
  tickerText,
  worstComponent,
} from "./state";
import { useOverlay } from "./store";
import { ActionButton, Dot, IconButton, LinkButton, PAD_X, SectionLabel, cx } from "./ui";

function useBackendSubscriptions(): void {
  useEffect(() => {
    const store = useOverlay.getState();
    const subs = [
      onMeetingEvent((ev) => useOverlay.getState().applyEvent(ev)),
      onUiCommand((cmd) => useOverlay.getState().applyUiCommand(cmd)),
    ];
    void store.init();
    return () => {
      subs.forEach((p) => void p.then((un) => un()).catch(() => undefined));
    };
  }, []);
}

function useAppearance(): CSSProperties {
  const overlay = useOverlay((s) => s.settings?.overlay ?? null);
  useEffect(() => {
    if (!overlay) return;
    const apply = () =>
      applyAppearance({
        theme: overlay.theme,
        density: overlay.density,
        fontSize: overlay.fontSize,
        reducedMotion: overlay.reducedMotion,
      });
    apply();
    return watchSystemTheme(apply);
  }, [overlay]);
  return useMemo<CSSProperties>(() => {
    const opacity = overlay?.opacity ?? 0.92;
    const translucent = overlay?.translucent ?? true;
    const alpha = translucent ? Math.min(opacity, 0.9) : opacity;
    return {
      width: `${overlay?.width ?? 380}px`,
      maxHeight: `${overlay?.maxHeight ?? 560}px`,
      background: `rgb(var(--om-glass) / ${alpha})`,
      backdropFilter: translucent ? "blur(14px) saturate(140%)" : undefined,
      WebkitBackdropFilter: translucent ? "blur(14px) saturate(140%)" : undefined,
    };
  }, [overlay]);
}

function Header() {
  const core = useOverlay((s) => s.core);
  const expanded = useOverlay((s) => s.expanded);
  const clickThrough = useOverlay((s) => s.clickThrough);
  const setExpanded = useOverlay((s) => s.setExpanded);
  const onDragStart = useDragSave();
  const status = deriveStatus(core);
  const live = isLive(core.meetingState);
  const pauseKey = useHotkey("pause_resume");
  const hideKey = useHotkey("toggle_overlay");
  const clickKey = useHotkey("toggle_click_through");
  const expandKey = useHotkey("expand_hud");
  const modeKey = useHotkey("toggle_mode");
  const privacy = core.privacy;
  const screen = screenActive(core.components);

  return (
    <header
      data-tauri-drag-region
      onMouseDown={onDragStart}
      className={cx("flex items-center gap-1.5 select-none", PAD_X, "py-[calc(var(--om-space)*0.4rem)]")}
    >
      <span data-tauri-drag-region className="cursor-grab text-muted" aria-hidden>
        <GripIcon />
      </span>
      <span data-tauri-drag-region className="flex min-w-0 items-center gap-1.5" role="status" aria-live="polite">
        <Dot tone={status.tone} pulse={status.pulse} />
        <span data-tauri-drag-region className="truncate text-[0.78rem] font-medium text-fg-soft">
          {expanded ? status.label : hudLabel(core)}
        </span>
      </span>
      <button
        type="button"
        onClick={() => void setMode(core.mode === "candidate" ? "interviewer" : "candidate")}
        title={withHotkey("Switch mode", modeKey)}
        className="ml-1 rounded border border-border px-1.5 py-px text-[0.68rem] font-medium tracking-wide text-muted uppercase hover:text-fg"
      >
        {core.mode === "candidate" ? "Candidate" : "Interviewer"}
      </button>
      <span data-tauri-drag-region className="flex-1" />
      {screen && (
        <span title="Screen context is on" className="text-muted">
          <MonitorIcon />
        </span>
      )}
      {privacy && (
        <span
          title={
            !privacy.supported
              ? "Capture privacy is not supported on this system"
              : privacy.active
                ? "Hidden from screen capture (where the OS supports it)"
                : "Visible to screen capture"
          }
          className={cx(privacy.active ? "text-ok" : privacy.supported ? "text-muted" : "text-warn")}
        >
          <ShieldIcon variant={privacy.active ? "on" : privacy.supported ? "off" : "warn"} />
        </span>
      )}
      <IconButton
        label={withHotkey(clickThrough ? "Click-through is on — turn off" : "Turn on click-through", clickKey)}
        tone={clickThrough ? "accent" : "default"}
        onClick={() => void setClickThrough(!clickThrough)}
      >
        <PointerIcon off={clickThrough} />
      </IconButton>
      {live && (
        <IconButton
          label={withHotkey(core.meetingState === "paused" ? "Resume" : "Pause", pauseKey)}
          tone={core.meetingState === "paused" ? "warn" : "default"}
          onClick={() => void (core.meetingState === "paused" ? resumeMeeting() : pauseMeeting())}
        >
          {core.meetingState === "paused" ? <PlayIcon /> : <PauseIcon />}
        </IconButton>
      )}
      <IconButton
        label={withHotkey(expanded ? "Collapse to minimal HUD" : "Expand", expandKey)}
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? <CollapseIcon /> : <ChevronDownIcon />}
      </IconButton>
      <IconButton label="Open OpenMeet" onClick={() => void openMainWindow(null)}>
        <ExternalIcon />
      </IconButton>
      <IconButton label={withHotkey("Hide overlay", hideKey)} onClick={() => void hideOverlay()}>
        <EyeOffIcon />
      </IconButton>
    </header>
  );
}

function StoppedBanner() {
  const reason = useOverlay((s) => s.core.stopReason);
  return (
    <div role="alert" className="space-y-1.5 rounded-md border border-danger/50 bg-danger-soft px-2.5 py-2">
      <p className="text-[0.78rem] font-semibold tracking-wider text-danger">PROCESSING STOPPED</p>
      <p className="text-[0.8rem] text-fg-soft">{reason ?? "Audio, screen and AI processing are halted."}</p>
      <ActionButton onClick={() => void resumeMeeting()}>
        <PlayIcon /> Resume
      </ActionButton>
    </div>
  );
}

function Notices() {
  const notices = useOverlay((s) => s.core.notices);
  const dispatch = useOverlay((s) => s.dispatch);
  useEffect(() => {
    const timers = notices
      .filter((n) => n.level === "info")
      .map((n) => window.setTimeout(() => dispatch({ type: "LOCAL_DISMISS_NOTICE", id: n.id }), 5000));
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [notices, dispatch]);
  if (notices.length === 0) return null;
  return (
    <ul className="space-y-1" aria-live="polite">
      {notices.map((n) => (
        <li
          key={n.id}
          role={n.level === "error" ? "alert" : "status"}
          className={cx(
            "flex items-start gap-1.5 rounded-md px-2 py-1 text-[0.78rem]",
            n.level === "error" ? "bg-danger-soft text-danger" : n.level === "warning" ? "text-warn" : "text-fg-soft",
          )}
        >
          <span className="min-w-0 flex-1">{n.message}</span>
          <button
            type="button"
            aria-label="Dismiss notice"
            className="text-muted hover:text-fg"
            onClick={() => dispatch({ type: "LOCAL_DISMISS_NOTICE", id: n.id })}
          >
            ×
          </button>
        </li>
      ))}
    </ul>
  );
}

function ComponentWarning() {
  const components = useOverlay((s) => s.core.components);
  const worst = worstComponent(components);
  if (!worst) return null;
  return (
    <div className="flex items-center gap-1.5 text-[0.76rem] text-warn">
      <Dot tone={worst.status === "error" ? "danger" : "warn"} />
      <span className="min-w-0 flex-1 truncate" title={worst.message ?? undefined}>
        {componentLabel(worst.component)}: {worst.message ?? worst.status}
      </span>
      <LinkButton onClick={() => void openMainWindow(componentSection(worst.component))}>Fix</LinkButton>
    </div>
  );
}

function Ticker() {
  const ticker = useOverlay((s) => s.core.ticker);
  const mode = useOverlay((s) => s.core.mode);
  if (!ticker) return null;
  return (
    <p className={cx("truncate text-[0.74rem] text-muted", !ticker.isFinal && "italic")} title={ticker.text}>
      <span className="font-medium">{speakerLabel(ticker.speakerRole, mode)}:</span> {tickerText(ticker.text, 120)}
    </p>
  );
}

function Pinned() {
  const pinned = useOverlay((s) => s.core.pinned);
  const [open, setOpen] = useState(false);
  if (pinned.length === 0) return null;
  return (
    <section aria-label="Pinned suggestions">
      <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)} className="flex items-center gap-1">
        <SectionLabel>{`Pinned (${pinned.length}) ${open ? "▾" : "▸"}`}</SectionLabel>
      </button>
      {open && (
        <ul className="mt-1 space-y-1">
          {pinned.map((s) => {
            const a = parseAnswer(s.content);
            const body = a.answer || s.followups.map((f) => f.question).join(" · ") || s.content;
            return (
              <li key={s.id} className="rounded-md border border-border px-2 py-1">
                <div className="flex items-start gap-1">
                  <p className="min-w-0 flex-1 text-[0.78rem] font-medium text-fg-soft">{suggestionTitle(s)}</p>
                  <button
                    type="button"
                    aria-label="Unpin"
                    className="text-[0.72rem] text-muted hover:text-fg"
                    onClick={() => void setPinned(s.id, false)}
                  >
                    Unpin
                  </button>
                </div>
                <p className="mt-0.5 line-clamp-3 text-[0.78rem] text-muted">{body}</p>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

function AskBox() {
  const askOpen = useOverlay((s) => s.askOpen);
  const nonce = useOverlay((s) => s.askFocusNonce);
  const closeAsk = useOverlay((s) => s.closeAsk);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (askOpen) inputRef.current?.focus();
  }, [askOpen, nonce]);
  if (!askOpen) return null;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const prompt = text.trim();
    if (!prompt || busy) return;
    setBusy(true);
    const err = await submitAsk(prompt);
    setBusy(false);
    if (err) setError(err);
    else {
      setText("");
      setError(null);
    }
  };
  return (
    <form onSubmit={(e) => void submit(e)} className="space-y-1">
      <div className="flex items-center gap-1 rounded-md border border-border-strong bg-[rgb(var(--om-glass)/0.6)] px-1.5 focus-within:border-accent">
        <SparkIcon className="text-accent" />
        <input
          ref={inputRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              e.preventDefault();
              void closeAsk();
            }
          }}
          aria-label="Ask OpenMeet"
          placeholder="Ask OpenMeet…  (Enter to send, Esc to close)"
          maxLength={4000}
          className="h-[1.9rem] min-w-0 flex-1 bg-transparent text-[0.86rem] text-fg placeholder:text-muted focus:outline-none"
        />
        <IconButton label="Send" type="submit" disabled={!text.trim() || busy}>
          <SendIcon />
        </IconButton>
      </div>
      {error && (
        <p role="alert" className="text-[0.76rem] text-danger">
          {error}
        </p>
      )}
    </form>
  );
}

function Body() {
  const core = useOverlay((s) => s.core);
  const openAsk = useOverlay((s) => s.openAsk);
  const askOpen = useOverlay((s) => s.askOpen);
  const askKey = useHotkey("ask_ai");
  const live = isLive(core.meetingState);
  const p = core.primary;
  const f = core.followups;
  const interviewer = core.mode === "interviewer";

  return (
    <div className={cx("min-h-0 flex-1 space-y-[calc(var(--om-space)*0.7rem)] overflow-y-auto", PAD_X, "pb-[calc(var(--om-space)*0.7rem)]")}>
      {core.processingStopped && <StoppedBanner />}
      <Notices />
      {!core.processingStopped && <ComponentWarning />}

      {core.pendingQuestion && (!p || p.questionId !== core.pendingQuestion.id) && !interviewer && (
        <div className="om-fade-in">
          <SectionLabel>Question detected</SectionLabel>
          <p className="mt-0.5 text-[0.88rem] font-medium text-fg">“{core.pendingQuestion.question}”</p>
          {!core.pendingQuestion.requiresResponse && (
            <p className="mt-0.5 text-[0.74rem] text-muted">Asked by you — no answer suggested.</p>
          )}
          {core.pendingQuestion.requiresResponse && (
            <div className="mt-1">
              <ActionButton onClick={() => void generateAnswer(core.pendingQuestion?.question ?? null)}>
                Suggest an answer
              </ActionButton>
            </div>
          )}
        </div>
      )}

      {interviewer && f && <FollowupsCard f={f} />}
      {p && <PrimaryCard p={p} style={core.style} mode={core.mode} />}

      {!p && !(interviewer && f) && !core.pendingQuestion && !core.processingStopped && (
        <div className="space-y-2">
          {live ? (
            <p className="text-[0.82rem] text-fg-soft">
              {core.meetingState === "paused"
                ? "Paused — nothing is being captured."
                : interviewer
                  ? "Listening. Follow-up questions appear when the candidate finishes an answer."
                  : "Listening. Suggestions appear when a question is detected."}
            </p>
          ) : (
            <p className="text-[0.82rem] text-fg-soft">
              {core.meetingState === "completed" || core.summaryReady
                ? "Meeting ended. The summary is in OpenMeet → History."
                : "Ready. Start a meeting in OpenMeet, or ask anything."}
            </p>
          )}
          {live && core.meetingState !== "paused" && <ListeningActions mode={core.mode} />}
          {!live && (
            <LinkButton onClick={() => void openMainWindow(core.summaryReady ? "history" : "home")}>
              {core.summaryReady ? "Open summary" : "Start a meeting"}
            </LinkButton>
          )}
        </div>
      )}

      {interviewer && <TopicsStrip topics={core.topics} suggestedNext={core.suggestedNextTopic} />}
      <Pinned />
      <Ticker />
      <AskBox />
      {!askOpen && !core.processingStopped && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => void openAsk()}
            title={withHotkey("Ask OpenMeet", askKey)}
            className="flex items-center gap-1 text-[0.76rem] text-muted hover:text-fg"
          >
            <SparkIcon /> Ask OpenMeet{askKey ? ` · ${askKey}` : ""}
          </button>
        </div>
      )}
    </div>
  );
}

export function OverlayApp() {
  useBackendSubscriptions();
  const style = useAppearance();
  const expanded = useOverlay((s) => s.expanded);
  const hydrated = useOverlay((s) => s.hydrated);
  const resizeNonce = useOverlay((s) => s.resizeNonce);
  const askOpen = useOverlay((s) => s.askOpen);
  const ref = useRef<HTMLDivElement>(null);
  useAutoSize(ref, resizeNonce + (expanded ? 1 : 0) + (askOpen ? 2 : 0));

  return (
    <div
      ref={ref}
      style={style}
      className="flex flex-col overflow-hidden rounded-xl border border-border-strong text-fg"
    >
      <Header />
      {expanded && hydrated && <Body />}
    </div>
  );
}
