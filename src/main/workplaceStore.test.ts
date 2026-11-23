import { describe, expect, it } from "vitest";
import type { LiveState } from "./store";
import { reduceLive } from "./store";
import type { MeetingEvent, WorkplaceBoard } from "@/bindings";

const EMPTY_BOARD: WorkplaceBoard = {
  decisions: [],
  actionItems: [],
  openQuestions: [],
  clarifications: [],
};

describe("workplace board in the main store", () => {
  it("replaces the board on every workplace update", () => {
    const live = { workplace: null } as unknown as LiveState;
    const board: WorkplaceBoard = {
      ...EMPTY_BOARD,
      decisions: [
        {
          id: "c1",
          meetingId: "m1",
          kind: "decision",
          text: "We decided to freeze the schema.",
          createdAt: 1,
          sourceSegmentId: null,
        } as unknown as WorkplaceBoard["decisions"][number],
      ],
    };
    const ev = { type: "WORKPLACE_UPDATED", board } as unknown as MeetingEvent;
    const next = reduceLive(live, ev);
    expect(next.workplace).toEqual(board);
    expect(reduceLive(next, ev).workplace).toEqual(board);
  });
});
