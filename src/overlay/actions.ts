// User actions of the overlay. Every backend call goes through `api` and
// reports failures visibly as an overlay notice (never silently).

import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import type { AnswerStyle, Mode, Refinement } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { retryPlan, type PrimaryRequest } from "./state";
import { useOverlay } from "./store";

/** Runs `fn`; on failure shows "<context>: <message>". Resolves to success. */
export async function attempt(context: string, fn: () => Promise<unknown>): Promise<boolean> {
  try {
    await fn();
    return true;
  } catch (e) {
    useOverlay.getState().notify("error", `${context}: ${errorMessage(e)}`);
    return false;
  }
}

const dispatch: ReturnType<typeof useOverlay.getState>["dispatch"] = (a) =>
  useOverlay.getState().dispatch(a);

export function generateAnswer(question: string | null, style: AnswerStyle | null = null) {
  return attempt("Couldn't generate an answer", () => api.generateAnswer({ question, style }));
}

export function generateFollowups() {
  return attempt("Couldn't suggest follow-ups", () => api.generateFollowups());
}

export function refine(requestId: string, refinement: Refinement) {
  dispatch({ type: "LOCAL_REFINE_REQUESTED", parentId: requestId, refinement });
  return attempt("Couldn't refine the answer", () => api.refineAnswer(requestId, refinement));
}

/** Re-runs a failed or cancelled primary request. */
export function retry(p: PrimaryRequest) {
  const plan = retryPlan(p);
  switch (plan.kind) {
    case "refine":
      return refine(plan.requestId, plan.refinement);
    case "answer":
      return generateAnswer(plan.question);
    case "ask":
      return attempt("Couldn't ask OpenMeet", () => api.askOpenMeet(plan.prompt));
    case "clarify":
      return attempt("Couldn't clarify the question", () => api.clarifyQuestion());
  }
}

/** Dismisses a suggestion (removed from the HUD immediately). */
export function dismiss(requestId: string) {
  dispatch({ type: "SUGGESTION_DISMISSED", requestId });
  return attempt("Couldn't dismiss", () => api.dismissSuggestion(requestId));
}

export function setPinned(requestId: string, pinned: boolean) {
  return attempt(pinned ? "Couldn't pin" : "Couldn't unpin", () =>
    api.pinSuggestion(requestId, pinned),
  );
}

export function copyText(text: string) {
  return attempt("Couldn't copy", () => writeText(text));
}

/**
 * Switches the answer style. When a candidate answer is on screen it is
 * rewritten in the new style right away.
 */
export async function chooseStyle(style: AnswerStyle, current: PrimaryRequest | null) {
  const ok = await attempt("Couldn't change the answer style", () => api.setAnswerStyle(style));
  if (!ok || !current) return ok;
  const rewritable =
    (current.kind === "candidate_answer" || current.kind === "refinement") &&
    !!current.questionText?.trim() &&
    current.style !== style;
  if (rewritable) await generateAnswer(current.questionText, style);
  return ok;
}

export function setMode(mode: Mode) {
  return attempt("Couldn't switch mode", () => api.setMode(mode));
}

export function pauseMeeting() {
  return attempt("Couldn't pause", () => api.pauseMeeting());
}

/** Resumes after a pause or an emergency stop. */
export async function resumeMeeting() {
  const { core } = useOverlay.getState();
  const noMeeting =
    core.meetingState !== "active" &&
    core.meetingState !== "paused" &&
    core.meetingState !== "starting";
  try {
    await api.resumeMeeting();
    if (noMeeting) dispatch({ type: "LOCAL_CLEAR_STOPPED" });
    return true;
  } catch (e) {
    // Without a meeting there is nothing to resume ("invalid_state"), but the
    // backend still lifts the processing stop; clear the banner instead.
    const invalidState =
      typeof e === "object" && e !== null && "kind" in e && e.kind === "invalid_state";
    if (noMeeting && core.processingStopped && invalidState) {
      dispatch({ type: "LOCAL_CLEAR_STOPPED" });
      return true;
    }
    useOverlay.getState().notify("error", `Couldn't resume: ${errorMessage(e)}`);
    return false;
  }
}

export function hideOverlay() {
  return attempt("Couldn't hide the overlay", () => api.overlayHide());
}

export function openMainWindow(section: string | null = null) {
  return attempt("Couldn't open OpenMeet", () => api.openMainWindow(section));
}

export function setClickThrough(enabled: boolean) {
  return attempt("Couldn't change click-through", async () => {
    const applied = await api.setClickThrough(enabled);
    useOverlay.setState({ clickThrough: applied });
  });
}

/** Submits an Ask OpenMeet prompt; resolves to an error message or null. */
export async function submitAsk(prompt: string): Promise<string | null> {
  try {
    await api.askOpenMeet(prompt);
    await useOverlay.getState().closeAsk();
    return null;
  } catch (e) {
    return errorMessage(e);
  }
}
