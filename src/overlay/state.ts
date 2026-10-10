// Overlay state: a pure reducer over meeting events (plus a few local
// actions) and the derived views the HUD renders. Everything here is free of
// side effects so it can be unit-tested exhaustively.
//
// Core rule (docs/FRONTEND.md): each UI slot shows only its latest request.
// Deltas, completions and state changes for any other request id are ignored,
// and ids that were superseded or dismissed are remembered ("retired") so a
// late event can never resurrect them.

import type {
  AnswerState,
  AnswerStyle,
  AudioSourceKind,
  CapturePrivacyStatus,
  Component,
  ComponentHealth,
  DetectedQuestion,
  EngineSnapshot,
  FollowUpQuestion,
  MeetingEvent,
  MeetingState,
  Mode,
  NoticeLevel,
  PauseReason,
  QuestionCategory,
  Refinement,
  RequestKind,
  SpeakerRole,
  Suggestion,
  Topic,
  TranscriptSegment,
  VoiceCommand,
} from "@/bindings";
import { categoryLabel } from "@/lib/format";

export const MAX_RETIRED = 64;
export const MAX_NOTICES = 3;

/** The request rendered in the primary slot (answers, Ask, refinements, clarifications). */
export interface PrimaryRequest {
  requestId: string;
  kind: RequestKind;
  questionId: string | null;
  questionText: string | null;
  category: QuestionCategory | null;
  style: AnswerStyle | null;
  parentId: string | null;
  /** Which answer control produced this request, when it was started from the overlay. */
  refinement: Refinement | null;
  state: AnswerState;
  error: string | null;
  /** Streamed text, replaced by the final suggestion content on completion. */
  text: string;
  /** ANSWER_COMPLETE arrived; `text` is the final content. */
  final: boolean;
  pinned: boolean;
  /** Some content has been rendered (used for render-latency reporting). */
  firstContent: boolean;
}

/** The request rendered in the follow-ups slot (interviewer mode). */
export interface FollowupsRequest {
  requestId: string;
  state: AnswerState;
  error: string | null;
  questions: FollowUpQuestion[];
  /** The backend reported the list as final. */
  done: boolean;
  /** Short preview of the candidate turn that triggered the request. */
  turnPreview: string | null;
  firstContent: boolean;
}

export interface Notice {
  id: number;
  level: NoticeLevel;
  message: string;
}

export interface PrivacyState {
  /** Capture exclusion requested by the user. */
  enabled: boolean;
  supported: boolean;
  /** Exclusion currently applied to the overlay. */
  active: boolean;
}

export interface OverlayState {
  meetingId: string | null;
  meetingTitle: string | null;
  meetingState: MeetingState;
  pausedReason: PauseReason | null;
  mode: Mode;
  style: AnswerStyle;
  processingStopped: boolean;
  stopReason: string | null;
  demoRunning: boolean;
  summaryReady: boolean;
  primary: PrimaryRequest | null;
  followups: FollowupsRequest | null;
  /** Superseded / dismissed request ids (bounded). */
  retired: string[];
  /** Refinement the user just asked for; attached to the matching ANSWER_STARTED. */
  pendingRefinement: { parentId: string; refinement: Refinement } | null;
  /** Detected question (candidate mode) that has no answer yet. */
  pendingQuestion: DetectedQuestion | null;
  /** Latest transcript line (partial or final) for the live ticker. */
  ticker: TranscriptSegment | null;
  topics: Topic[];
  suggestedNextTopic: string | null;
  components: ComponentHealth[];
  pinned: Suggestion[];
  privacy: PrivacyState | null;
  levels: Partial<Record<AudioSourceKind, number>>;
  notices: Notice[];
  noticeSeq: number;
}

/** Actions that originate in the overlay itself (not from the backend). */
export type LocalAction =
  | { type: "LOCAL_REFINE_REQUESTED"; parentId: string; refinement: Refinement }
  | { type: "LOCAL_DISMISS_QUESTION" }
  | { type: "LOCAL_NOTICE"; level: NoticeLevel; message: string }
  | { type: "LOCAL_DISMISS_NOTICE"; id: number }
  | { type: "LOCAL_PRIVACY_STATUS"; status: CapturePrivacyStatus }
  | { type: "LOCAL_CLEAR_STOPPED" };

export type OverlayAction = MeetingEvent | LocalAction;

export const INITIAL_STATE: OverlayState = {
  meetingId: null,
  meetingTitle: null,
  meetingState: "idle",
  pausedReason: null,
  mode: "candidate",
  style: "concise",
  processingStopped: false,
  stopReason: null,
  demoRunning: false,
  summaryReady: false,
  primary: null,
  followups: null,
  retired: [],
  pendingRefinement: null,
  pendingQuestion: null,
  ticker: null,
  topics: [],
  suggestedNextTopic: null,
  components: [],
  pinned: [],
  privacy: null,
  levels: {},
  notices: [],
  noticeSeq: 0,
};

// ---------------------------------------------------------------------------
// Small predicates
// ---------------------------------------------------------------------------

/** Request is waiting for its first token. */
export function isThinking(state: AnswerState): boolean {
  return state === "created" || state === "context_building" || state === "requesting";
}

/** Request is still running (thinking or streaming). */
export function isRunning(state: AnswerState): boolean {
  return isThinking(state) || state === "streaming";
}

export function isTerminal(state: AnswerState): boolean {
  return state === "complete" || state === "failed" || state === "cancelled";
}

/** Meeting is running (capture may be on). */
export function isLive(state: MeetingState): boolean {
  return state === "starting" || state === "active" || state === "paused";
}

function retire(retired: string[], id: string | null | undefined): string[] {
  if (!id || retired.includes(id)) return retired;
  const next = [...retired, id];
  return next.length > MAX_RETIRED ? next.slice(next.length - MAX_RETIRED) : next;
}

/** Appends a notice (bounded); identical consecutive messages are not repeated. */
export function pushNotice(state: OverlayState, level: NoticeLevel, message: string): OverlayState {
  const text = message.trim();
  if (!text) return state;
  const last = state.notices[state.notices.length - 1];
  if (last && last.message === text && last.level === level) return state;
  const id = state.noticeSeq + 1;
  const notices = [...state.notices, { id, level, message: text }];
  return {
    ...state,
    noticeSeq: id,
    notices: notices.length > MAX_NOTICES ? notices.slice(notices.length - MAX_NOTICES) : notices,
  };
}

/**
 * Next ticker line. Partials and the final of one utterance share
 * `utteranceId`; a late partial never replaces its final, and an older
 * utterance never replaces a newer one.
 */
export function nextTicker(
  current: TranscriptSegment | null,
  seg: TranscriptSegment,
): TranscriptSegment | null {
  if (!seg.text.trim()) return current;
  if (!current) return seg;
  if (seg.utteranceId === current.utteranceId) {
    if (current.isFinal && !seg.isFinal) return current;
    return seg;
  }
  return seg.startTime >= current.startTime ? seg : current;
}

function voiceCommandLabel(c: VoiceCommand): string {
  switch (c.kind) {
    case "explain":
      return "explain that";
    case "follow_up":
      return "follow-up";
    case "answer":
      return "answer";
    case "concise":
      return "concise";
    case "detailed":
      return "detailed";
    case "ask":
      return `ask “${c.text}”`;
  }
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function onAnswerStarted(
  state: OverlayState,
  ev: Extract<MeetingEvent, { type: "ANSWER_STARTED" }>,
): OverlayState {
  if (state.retired.includes(ev.requestId)) return state;
  if (ev.slot === "primary") {
    if (state.primary?.requestId === ev.requestId) return state;
    const pr = state.pendingRefinement;
    const refinement =
      ev.kind === "refinement" && pr && ev.parentId === pr.parentId ? pr.refinement : null;
    const clearsQuestion =
      ev.kind === "candidate_answer" ||
      ev.kind === "clarification" ||
      (ev.questionId !== null && ev.questionId === state.pendingQuestion?.id);
    return {
      ...state,
      retired: retire(state.retired, state.primary?.requestId),
      pendingRefinement: refinement ? null : state.pendingRefinement,
      pendingQuestion: clearsQuestion ? null : state.pendingQuestion,
      primary: {
        requestId: ev.requestId,
        kind: ev.kind,
        questionId: ev.questionId,
        questionText: ev.questionText,
        category: ev.category,
        style: ev.style,
        parentId: ev.parentId,
        refinement,
        state: "created",
        error: null,
        text: "",
        final: false,
        pinned: false,
        firstContent: false,
      },
    };
  }
  if (ev.slot === "followups") {
    if (state.followups?.requestId === ev.requestId) return state;
    return {
      ...state,
      retired: retire(state.retired, state.followups?.requestId),
      followups: {
        requestId: ev.requestId,
        state: "created",
        error: null,
        questions: [],
        done: false,
        turnPreview: ev.questionText,
        firstContent: false,
      },
    };
  }
  // Background requests never render in the overlay.
  return state;
}

function onStateChanged(
  state: OverlayState,
  ev: Extract<MeetingEvent, { type: "ANSWER_STATE_CHANGED" }>,
): OverlayState {
  // A failure is never silent: fall back to a generic message.
  const error = ev.state === "failed" ? ev.error?.trim() || "The request failed." : ev.error;
  const p = state.primary;
  if (p && p.requestId === ev.requestId) {
    if (p.state === ev.state && p.error === error) return state;
    return { ...state, primary: { ...p, state: ev.state, error } };
  }
  const f = state.followups;
  if (f && f.requestId === ev.requestId) {
    if (f.state === ev.state && f.error === error) return state;
    return { ...state, followups: { ...f, state: ev.state, error } };
  }
  return state;
}

/** Pure reducer: (state, event) → next state. Returns `state` itself for no-ops. */
export function reduceOverlay(state: OverlayState, ev: OverlayAction): OverlayState {
  switch (ev.type) {
    // -- requests ------------------------------------------------------------
    case "ANSWER_STARTED":
      return onAnswerStarted(state, ev);

    case "ANSWER_DELTA": {
      const p = state.primary;
      // Follow-up deltas are raw NDJSON: only FOLLOWUPS_READY renders them.
      if (!p || p.requestId !== ev.requestId || p.final || isTerminal(p.state) || !ev.text) {
        return state;
      }
      return {
        ...state,
        primary: {
          ...p,
          text: p.text + ev.text,
          state: isThinking(p.state) ? "streaming" : p.state,
          firstContent: true,
        },
      };
    }

    case "ANSWER_STATE_CHANGED":
      return onStateChanged(state, ev);

    case "ANSWER_COMPLETE": {
      const s = ev.suggestion;
      const p = state.primary;
      if (p && p.requestId === ev.requestId) {
        return {
          ...state,
          primary: {
            ...p,
            text: s.content.trim() ? s.content : p.text,
            final: true,
            state: "complete",
            error: null,
            pinned: s.pinned,
            questionText: p.questionText ?? s.questionText,
            category: p.category ?? s.category,
            style: p.style ?? s.style,
            firstContent: true,
          },
        };
      }
      const f = state.followups;
      if (f && f.requestId === ev.requestId) {
        return {
          ...state,
          followups: {
            ...f,
            questions: s.followups.length > 0 ? s.followups : f.questions,
            done: true,
            state: "complete",
            error: null,
            firstContent: f.firstContent || s.followups.length > 0,
          },
        };
      }
      return state;
    }

    case "FOLLOWUPS_READY": {
      const f = state.followups;
      if (!f || f.requestId !== ev.requestId || f.state === "cancelled" || f.state === "failed") {
        return state;
      }
      return {
        ...state,
        followups: {
          ...f,
          questions: ev.questions,
          done: f.done || ev.done,
          state: isThinking(f.state) ? "streaming" : f.state,
          firstContent: f.firstContent || ev.questions.length > 0,
        },
      };
    }

    case "SUGGESTION_PINNED": {
      let pinned = state.pinned.filter((s) => s.id !== ev.requestId);
      if (ev.pinned && ev.suggestion) pinned = [{ ...ev.suggestion, pinned: true }, ...pinned];
      const p = state.primary;
      return {
        ...state,
        pinned,
        primary: p && p.requestId === ev.requestId ? { ...p, pinned: ev.pinned } : p,
      };
    }

    case "SUGGESTION_DISMISSED": {
      const hitPrimary = state.primary?.requestId === ev.requestId;
      const hitFollowups = state.followups?.requestId === ev.requestId;
      if (!hitPrimary && !hitFollowups)
        return { ...state, retired: retire(state.retired, ev.requestId) };
      return {
        ...state,
        primary: hitPrimary ? null : state.primary,
        followups: hitFollowups ? null : state.followups,
        retired: retire(state.retired, ev.requestId),
      };
    }

    // -- conversation ----------------------------------------------------------
    case "QUESTION_DETECTED": {
      const q = ev.question;
      if (state.meetingId && q.meetingId !== state.meetingId) return state;
      if (state.mode !== "candidate" || !q.requiresResponse || !q.isQuestion) return state;
      if (state.primary && state.primary.questionId === q.id) return state;
      return { ...state, pendingQuestion: q };
    }

    case "TRANSCRIPT_PARTIAL":
    case "TRANSCRIPT_FINAL": {
      if (state.meetingId && ev.segment.meetingId !== state.meetingId) return state;
      const ticker = nextTicker(state.ticker, ev.segment);
      return ticker === state.ticker ? state : { ...state, ticker };
    }

    case "TOPICS_UPDATED":
      return { ...state, topics: ev.topics, suggestedNextTopic: ev.suggestedNext };

    // -- meeting lifecycle -------------------------------------------------------
    case "MEETING_STARTED":
      return {
        ...state,
        meetingId: ev.meeting.id,
        meetingTitle: ev.meeting.title,
        meetingState: ev.meeting.state,
        pausedReason: null,
        mode: ev.meeting.mode,
        processingStopped: false,
        stopReason: null,
        demoRunning: ev.meeting.title.startsWith("Demo:"),
        summaryReady: false,
        retired: retire(
          retire(state.retired, state.primary?.requestId),
          state.followups?.requestId,
        ),
        primary: null,
        followups: null,
        pendingRefinement: null,
        pendingQuestion: null,
        ticker: null,
        topics: [],
        suggestedNextTopic: null,
        pinned: [],
        levels: {},
      };

    case "MEETING_STATE_CHANGED": {
      // Idle without a meeting id: the meeting was deleted; drop everything
      // that belonged to it.
      if (ev.state === "idle" && ev.meetingId === null && state.meetingId !== null) {
        return {
          ...state,
          meetingId: null,
          meetingTitle: null,
          meetingState: "idle",
          mode: ev.mode,
          pausedReason: null,
          summaryReady: false,
          retired: retire(
            retire(state.retired, state.primary?.requestId),
            state.followups?.requestId,
          ),
          primary: null,
          followups: null,
          pendingRefinement: null,
          pendingQuestion: null,
          ticker: null,
          topics: [],
          suggestedNextTopic: null,
          pinned: [],
          levels: {},
        };
      }
      const ended = ev.state === "idle" || ev.state === "completed";
      const meetingId = ev.meetingId ?? (ev.state === "idle" ? null : state.meetingId);
      return {
        ...state,
        meetingId,
        meetingTitle: meetingId === state.meetingId ? state.meetingTitle : null,
        meetingState: ev.state,
        mode: ev.mode,
        pausedReason: ev.pausedReason,
        processingStopped: ev.state === "active" ? false : state.processingStopped,
        stopReason: ev.state === "active" ? null : state.stopReason,
        pendingQuestion: ended || ev.mode === "interviewer" ? null : state.pendingQuestion,
        ticker: ended ? null : state.ticker,
        levels: ev.state === "active" ? state.levels : {},
      };
    }

    case "MEETING_STOPPED":
      return { ...state, demoRunning: false, pendingQuestion: null, ticker: null, levels: {} };

    case "PROCESSING_STOPPED":
      return {
        ...state,
        processingStopped: true,
        stopReason: ev.reason,
        demoRunning: false,
        pendingQuestion: null,
        ticker: null,
        levels: {},
      };

    case "LOCAL_CLEAR_STOPPED":
      return { ...state, processingStopped: false, stopReason: null };

    case "SUMMARY_READY":
      return { ...state, summaryReady: true };

    case "SUMMARY_FAILED":
      return pushNotice(state, "error", `Summary failed: ${ev.error}`);

    // -- settings-ish ------------------------------------------------------------
    case "MODE_CHANGED":
      return {
        ...state,
        mode: ev.mode,
        pendingQuestion: ev.mode === "interviewer" ? null : state.pendingQuestion,
      };

    case "STYLE_CHANGED":
      return state.style === ev.style ? state : { ...state, style: ev.style };

    case "PRIVACY_STATE_CHANGED":
      return {
        ...state,
        privacy: { enabled: ev.enabled, supported: ev.supported, active: ev.active },
      };

    case "LOCAL_PRIVACY_STATUS":
      return {
        ...state,
        privacy: {
          enabled: ev.status.requested,
          supported: ev.status.supported,
          active: ev.status.active,
        },
      };

    case "COMPONENT_STATUS_CHANGED": {
      const others = state.components.filter((c) => c.component !== ev.health.component);
      return { ...state, components: [...others, ev.health] };
    }

    case "AUDIO_LEVEL":
      return { ...state, levels: { ...state.levels, [ev.source]: ev.level } };

    // -- notices ---------------------------------------------------------------
    case "NOTICE":
      return pushNotice(state, ev.level, ev.message);

    case "VOICE_COMMAND_RECOGNIZED":
      return pushNotice(state, "info", `Voice command: ${voiceCommandLabel(ev.command)}`);

    case "LOCAL_NOTICE":
      return pushNotice(state, ev.level, ev.message);

    case "LOCAL_DISMISS_NOTICE":
      return { ...state, notices: state.notices.filter((n) => n.id !== ev.id) };

    // -- local intents ---------------------------------------------------------
    case "LOCAL_REFINE_REQUESTED":
      return {
        ...state,
        pendingRefinement: { parentId: ev.parentId, refinement: ev.refinement },
      };

    case "LOCAL_DISMISS_QUESTION":
      return state.pendingQuestion ? { ...state, pendingQuestion: null } : state;

    // Not rendered by the overlay.
    case "SCREEN_UPDATED":
    case "CLAIMS_EXTRACTED":
    case "ACTION_ITEMS_UPDATED":
    case "LATENCY_RECORDED":
      return state;
  }
}

/** True when `ev` delivers the first visible content of the current request in a slot. */
export function isFirstContent(state: OverlayState, ev: MeetingEvent): boolean {
  if (ev.type === "ANSWER_DELTA") {
    const p = state.primary;
    return !!p && p.requestId === ev.requestId && !p.firstContent && !p.final && !!ev.text;
  }
  if (ev.type === "FOLLOWUPS_READY") {
    const f = state.followups;
    return !!f && f.requestId === ev.requestId && !f.firstContent && ev.questions.length > 0;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Snapshot hydration
// ---------------------------------------------------------------------------

/**
 * Builds the state for a window that (re)loads mid-meeting. Overlay-local
 * fields (notices, privacy, retired ids) are kept from `prev`.
 */
export function hydrateFromSnapshot(prev: OverlayState, s: EngineSnapshot): OverlayState {
  const a = s.active;
  const primary: PrimaryRequest | null =
    a && a.slot === "primary" && !prev.retired.includes(a.requestId)
      ? {
          requestId: a.requestId,
          kind: a.kind,
          questionId: null,
          questionText: a.questionText,
          category: a.category,
          style: a.style,
          parentId: null,
          refinement: null,
          state: a.state,
          error: a.state === "failed" ? a.error?.trim() || "The request failed." : a.error,
          text: a.text,
          final: a.state === "complete",
          pinned: s.pinned.some((p) => p.id === a.requestId),
          firstContent: a.text.length > 0,
        }
      : null;
  const followups: FollowupsRequest | null =
    s.followupsRequestId && !prev.retired.includes(s.followupsRequestId)
      ? {
          requestId: s.followupsRequestId,
          // The snapshot does not carry the follow-up request state; treat
          // what we have as settled. Later events for this id still apply.
          state: "complete",
          error: null,
          questions: s.followups,
          done: true,
          turnPreview: null,
          firstContent: s.followups.length > 0,
        }
      : null;
  const live = isLive(s.state);
  const last = s.transcript.length > 0 ? s.transcript[s.transcript.length - 1] : undefined;
  const pinned = [...s.pinned].sort(
    (x, y) => (y.completedAt ?? y.createdAt) - (x.completedAt ?? x.createdAt),
  );
  return {
    ...prev,
    meetingId: s.meeting?.id ?? null,
    meetingTitle: s.meeting?.title ?? null,
    meetingState: s.state,
    pausedReason: s.pausedReason,
    mode: s.mode,
    style: s.style,
    processingStopped: s.processingStopped,
    stopReason: s.processingStopped ? prev.stopReason : null,
    demoRunning: s.demoRunning,
    summaryReady: s.lastSummary !== null,
    primary,
    followups,
    pendingRefinement: null,
    pendingQuestion: null,
    ticker: live && last ? last : null,
    topics: s.topics,
    suggestedNextTopic: s.suggestedNextTopic,
    components: s.components,
    pinned,
    levels: {},
  };
}

/**
 * Events received while the snapshot was loading are replayed on top of it.
 * Deltas of the hydrated active request may already be contained in the
 * snapshot text, so they are dropped (ANSWER_COMPLETE restores the full text).
 */
export function replayAfterHydration(
  state: OverlayState,
  buffered: ReadonlyArray<MeetingEvent>,
  hydratedActiveId: string | null,
): OverlayState {
  let next = state;
  for (const ev of buffered) {
    if (ev.type === "ANSWER_DELTA" && ev.requestId === hydratedActiveId) continue;
    next = reduceOverlay(next, ev);
  }
  return next;
}

// ---------------------------------------------------------------------------
// Derived views
// ---------------------------------------------------------------------------

export type Tone = "neutral" | "accent" | "ok" | "warn" | "danger";

export type StatusKind =
  | "stopped"
  | "paused"
  | "thinking"
  | "answering"
  | "listening"
  | "starting"
  | "ending"
  | "ended"
  | "ready";

export interface StatusView {
  kind: StatusKind;
  label: string;
  tone: Tone;
  pulse: boolean;
}

/** Header status (dot + label). */
export function deriveStatus(s: OverlayState): StatusView {
  if (s.processingStopped) {
    return { kind: "stopped", label: "Processing stopped", tone: "danger", pulse: false };
  }
  if (s.meetingState === "paused") {
    return { kind: "paused", label: "Paused", tone: "warn", pulse: false };
  }
  const p = s.primary;
  const f = s.mode === "interviewer" ? s.followups : null;
  if ((p && isThinking(p.state)) || (f && isThinking(f.state))) {
    return { kind: "thinking", label: "Thinking…", tone: "accent", pulse: true };
  }
  if (p && p.state === "streaming") {
    return { kind: "answering", label: "Answering…", tone: "accent", pulse: true };
  }
  if (f && f.state === "streaming" && !f.done) {
    return { kind: "answering", label: "Drafting follow-ups…", tone: "accent", pulse: true };
  }
  switch (s.meetingState) {
    case "active":
      return { kind: "listening", label: "Listening", tone: "ok", pulse: true };
    case "starting":
      return { kind: "starting", label: "Starting…", tone: "neutral", pulse: true };
    case "stopping":
      return { kind: "ending", label: "Ending meeting…", tone: "neutral", pulse: true };
    case "summarizing":
      return { kind: "ending", label: "Writing summary…", tone: "neutral", pulse: true };
    case "completed":
      return { kind: "ended", label: "Meeting ended", tone: "neutral", pulse: false };
    default:
      return { kind: "ready", label: "Ready", tone: "neutral", pulse: false };
  }
}

/** One-line text of the minimal HUD. */
export function hudLabel(s: OverlayState): string {
  const status = deriveStatus(s);
  if (status.kind === "stopped" || status.kind === "paused") return status.label;
  const p = s.primary;
  if (p && p.state === "failed") return "Answer failed";
  if (status.kind === "thinking") return status.label;
  const category = p?.category ?? s.pendingQuestion?.category ?? null;
  const suffix = category ? ` · ${categoryLabel(category)}` : "";
  if (s.pendingQuestion && (!p || !isRunning(p.state))) return `Q detected${suffix}`;
  if (p && p.state === "streaming") {
    return p.kind === "candidate_answer" ? `Q detected${suffix}` : "Answering…";
  }
  if (s.mode === "interviewer" && s.followups && s.followups.questions.length > 0) {
    const n = s.followups.questions.length;
    return `${n} follow-up${n === 1 ? "" : "s"}`;
  }
  if (p && p.state === "complete" && p.text.trim()) return `Answer ready${suffix}`;
  return status.label;
}

export type RetryPlan =
  | { kind: "refine"; requestId: string; refinement: Refinement }
  | { kind: "answer"; question: string | null }
  | { kind: "ask"; prompt: string }
  | { kind: "clarify" };

/** What "Retry" re-runs for a failed / cancelled primary request. */
export function retryPlan(p: PrimaryRequest): RetryPlan {
  switch (p.kind) {
    case "ask_ai":
      return p.questionText?.trim()
        ? { kind: "ask", prompt: p.questionText }
        : { kind: "refine", requestId: p.requestId, refinement: "regenerate" };
    case "clarification":
      return { kind: "clarify" };
    case "refinement":
      return p.parentId
        ? { kind: "refine", requestId: p.parentId, refinement: p.refinement ?? "regenerate" }
        : { kind: "answer", question: p.questionText };
    default:
      return { kind: "answer", question: p.questionText };
  }
}

/** Speaker label for the ticker. */
export function speakerLabel(role: SpeakerRole, mode: Mode): string {
  if (role === "user") return "You";
  if (role === "remote") return mode === "candidate" ? "Interviewer" : "Candidate";
  return "Speaker";
}

/** Tail of a long line for the one-line ticker. */
export function tickerText(text: string, max = 140): string {
  const t = text.trim().replace(/\s+/g, " ");
  if (t.length <= max) return t;
  const tail = t.slice(t.length - max);
  const space = tail.indexOf(" ");
  return `…${space > 0 && space < 20 ? tail.slice(space + 1) : tail}`;
}

const COMPONENT_LABELS: Record<Component, string> = {
  meeting: "Meeting",
  microphone: "Microphone",
  system_audio: "System audio",
  transcription: "Transcription",
  question_detector: "Question detection",
  llm: "AI provider",
  vision: "Vision",
  screen: "Screen context",
  embeddings: "Embeddings",
  database: "Storage",
  overlay: "Overlay",
  capture_privacy: "Capture privacy",
  hotkeys: "Hotkeys",
};

export function componentLabel(c: Component): string {
  return COMPONENT_LABELS[c];
}

/** Main-window section that fixes a component problem. */
export function componentSection(c: Component): string {
  switch (c) {
    case "llm":
    case "embeddings":
    case "question_detector":
      return "settings/ai";
    case "microphone":
    case "system_audio":
    case "transcription":
      return "settings/transcription";
    case "screen":
    case "vision":
      return "settings/screen";
    case "hotkeys":
      return "settings/shortcuts";
    case "capture_privacy":
      return "privacy";
    default:
      return "diagnostics";
  }
}

/** Most important unhealthy component (errors before degradations, newest first). */
export function worstComponent(components: ReadonlyArray<ComponentHealth>): ComponentHealth | null {
  let worst: ComponentHealth | null = null;
  const rank = (c: ComponentHealth) => (c.status === "error" ? 2 : c.status === "degraded" ? 1 : 0);
  for (const c of components) {
    if (c.component === "meeting" || rank(c) === 0) continue;
    if (
      !worst ||
      rank(c) > rank(worst) ||
      (rank(c) === rank(worst) && c.updatedAt > worst.updatedAt)
    ) {
      worst = c;
    }
  }
  return worst;
}

/** Screen context is currently being sampled. */
export function screenActive(components: ReadonlyArray<ComponentHealth>): boolean {
  return components.some(
    (c) => c.component === "screen" && (c.status === "active" || c.status === "streaming"),
  );
}

const TOPIC_ORDER: Record<Topic["status"], number> = { covered: 0, current: 1, pending: 2 };

/** Topics in tracker order: covered → current → pending (stable within a group). */
export function orderTopics(topics: ReadonlyArray<Topic>): Topic[] {
  return topics
    .map((t, i) => ({ t, i }))
    .sort((a, b) => TOPIC_ORDER[a.t.status] - TOPIC_ORDER[b.t.status] || a.i - b.i)
    .map(({ t }) => t);
}

/** Short title for a pinned suggestion. */
export function suggestionTitle(s: Suggestion): string {
  if (s.questionText?.trim()) return s.questionText.trim();
  switch (s.kind) {
    case "ask":
      return "Ask OpenMeet";
    case "followups":
      return "Follow-up questions";
    case "clarification":
      return "Clarifying questions";
    default:
      return "Suggested answer";
  }
}
