// Pure helpers for the workplace board (decisions, action items, open
// questions, clarifying questions), kept apart from the components so they
// can be unit tested.

import type { OpenQuestion, WorkplaceBoard } from "@/bindings";

export interface BoardCounts {
  decisions: number;
  actionItems: number;
  openQuestions: number;
  clarifications: number;
}

export function boardCounts(board: WorkplaceBoard): BoardCounts {
  return {
    decisions: board.decisions.length,
    actionItems: board.actionItems.length,
    openQuestions: board.openQuestions.filter((q) => !q.answered).length,
    clarifications: board.clarifications.length,
  };
}

/** How a question on the board reads: answered, deferred, or still open. */
export function questionStatus(q: OpenQuestion): "answered" | "deferred" | "open" {
  if (q.answered) return "answered";
  if (q.deferred) return "deferred";
  return "open";
}

/** "owner: Priya · due: by Friday", with "not stated" for what was not said. */
export function ownerDueLine(owner: string | null, due: string | null): string {
  return `owner: ${owner?.trim() ? owner : "not stated"} · due: ${due?.trim() ? due : "not stated"}`;
}
