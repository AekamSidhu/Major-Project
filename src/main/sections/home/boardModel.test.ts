import { describe, expect, it } from "vitest";
import type { OpenQuestion, WorkplaceBoard } from "@/bindings";
import { boardCounts, ownerDueLine, questionStatus } from "./boardModel";

function question(patch: Partial<OpenQuestion>): OpenQuestion {
  return {
    id: "q",
    question: "Can you share the plan?",
    speakerRole: "remote",
    addressedToYou: true,
    askedAtMs: 1000,
    answered: false,
    answer: null,
    deferred: false,
    ...patch,
  };
}

describe("workplace board model", () => {
  it("counts only open questions as open", () => {
    const board: WorkplaceBoard = {
      decisions: [],
      actionItems: [],
      openQuestions: [
        question({ id: "a" }),
        question({ id: "b", answered: true, answer: "Monday" }),
      ],
      clarifications: [],
    };
    expect(boardCounts(board)).toEqual({
      decisions: 0,
      actionItems: 0,
      openQuestions: 1,
      clarifications: 0,
    });
  });

  it.each([
    [{ answered: true }, "answered"],
    [{ deferred: true }, "deferred"],
    [{}, "open"],
  ])("reads a question's status %#", (patch, expected) => {
    expect(questionStatus(question(patch))).toBe(expected);
  });

  it.each([
    ["Priya", "by Friday", "owner: Priya · due: by Friday"],
    [null, null, "owner: not stated · due: not stated"],
    ["  ", "next sprint", "owner: not stated · due: next sprint"],
  ])("writes owner and due date (%s, %s)", (owner, due, expected) => {
    expect(ownerDueLine(owner, due)).toBe(expected);
  });
});
