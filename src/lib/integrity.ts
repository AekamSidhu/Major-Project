// Answer-integrity flags (interviewer mode): pure helpers shared by the
// overlay, the live view and history. A flag is a signal for the
// interviewer, never a verdict — the wording here and in the backend's
// evidence lines follows that rule.

import type {
  IntegrityFlag,
  IntegrityLevel,
  IntegritySignal,
  IntegritySignalName,
} from "@/bindings";
import { formatOffset } from "./format";

export const SIGNAL_LABELS: Record<IntegritySignalName, string> = {
  ai_text: "AI-written text",
  echo_similarity: "Echo of an LLM answer",
  delivery: "Delivery",
  reading: "Reading",
};

export const LEVEL_LABELS: Record<IntegrityLevel, string> = {
  low: "Low",
  possible: "Possible",
  high: "High",
  too_short: "Too short",
  baseline: "Learning baseline",
  no_gaze: "No gaze",
  unavailable: "Not available",
};

/** One-line explanation shown with every flag. */
export const FLAG_NOTE = "A reason to ask a follow-up, never a verdict.";

export type FlagTone = "neutral" | "ok" | "warn";

/** `possible` or `high`. */
export function isElevated(level: IntegrityLevel): boolean {
  return level === "possible" || level === "high";
}

/** `low`, `possible` or `high` (the others say why a signal was not scored). */
export function isScored(level: IntegrityLevel): boolean {
  return level === "low" || isElevated(level);
}

export function levelTone(level: IntegrityLevel): FlagTone {
  if (isElevated(level)) return "warn";
  if (level === "low") return "ok";
  return "neutral";
}

/** "Answer at 23:54–25:00". */
export function answerRangeLabel(flag: Pick<IntegrityFlag, "answerStartMs" | "answerEndMs">) {
  return `Answer at ${formatOffset(flag.answerStartMs)}–${formatOffset(flag.answerEndMs)}`;
}

/** Signals with something to say first: elevated, then scored, then the rest. */
export function orderedSignals(flag: IntegrityFlag): IntegritySignal[] {
  const rank = (s: IntegritySignal) => (isElevated(s.level) ? 0 : isScored(s.level) ? 1 : 2);
  return flag.signals
    .map((s, i) => ({ s, i }))
    .sort((a, b) => rank(a.s) - rank(b.s) || a.i - b.i)
    .map(({ s }) => s);
}

/** Short status of the latest check for compact views ("2 signals raised"). */
export function flagSummary(flag: IntegrityFlag): string {
  const n = flag.elevated.length;
  if (isElevated(flag.level)) {
    return n === 1 ? "1 signal raised" : `${n} signals raised`;
  }
  if (flag.level === "low") return "Nothing unusual";
  return "Not enough to check";
}

/**
 * Inserts or replaces `flag` by id; an older revision never replaces a newer
 * one. Latest answer first.
 */
export function upsertFlag(
  list: ReadonlyArray<IntegrityFlag>,
  flag: IntegrityFlag,
): IntegrityFlag[] {
  const existing = list.find((f) => f.id === flag.id);
  if (existing && existing.revision > flag.revision) return list.slice();
  const next = list.filter((f) => f.id !== flag.id);
  next.push(flag);
  return sortFlags(next);
}

/** Latest answer first. */
export function sortFlags(list: ReadonlyArray<IntegrityFlag>): IntegrityFlag[] {
  return list
    .slice()
    .sort((a, b) => b.answerStartMs - a.answerStartMs || b.createdAt - a.createdAt);
}
