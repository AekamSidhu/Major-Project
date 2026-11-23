import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PrimaryCard } from "./cards";
import { catchUpMinutes, retryPlan, type PrimaryRequest } from "./state";

afterEach(cleanup);

function catchUp(patch: Partial<PrimaryRequest> = {}): PrimaryRequest {
  return {
    requestId: "req-catch-up",
    kind: "catch_up",
    slot: "primary",
    state: "complete",
    questionText: "Last 5 minutes",
    category: null,
    style: "concise",
    parentId: null,
    refinement: null,
    error: null,
    text: "[[ANSWER]]\nThe team agreed to move the export job to Kafka.\n[[KEY_POINTS]]\n- Decided: move the export job to Kafka",
    ...patch,
  } as unknown as PrimaryRequest;
}

describe("catch-up card", () => {
  it("is headed by its window and reads as a recap", () => {
    render(<PrimaryCard p={catchUp()} style="concise" mode="workplace" />);
    expect(screen.getByRole("region", { name: "Last 5 minutes" })).toBeTruthy();
    expect(screen.getByText("Recap")).toBeTruthy();
    expect(screen.queryByText("Suggested answer")).toBeNull();
    expect(screen.getAllByText(/move the export job to Kafka/).length).toBeGreaterThan(0);
  });

  it("has no answer refinement chips but keeps the copy control", () => {
    render(<PrimaryCard p={catchUp()} style="concise" mode="workplace" />);
    expect(screen.queryByRole("group", { name: "Answer controls" })).toBeNull();
    for (const label of ["Regenerate", "Shorter", "Longer", "Technical", "Simplify"]) {
      expect(screen.queryByText(label)).toBeNull();
    }
    const card = screen.getByRole("region", { name: "Last 5 minutes" });
    expect(within(card).queryByText("Regenerate")).toBeNull();
  });

  it("still shows refinement chips for an ordinary answer", () => {
    const answer = catchUp({ kind: "candidate_answer", questionText: "Why Kafka?" });
    render(<PrimaryCard p={answer} style="concise" mode="candidate" />);
    expect(screen.getByRole("group", { name: "Answer controls" })).toBeTruthy();
    expect(screen.getByText("Shorter")).toBeTruthy();
  });
});

describe("catch-up window and retry", () => {
  it("reads the window from the heading and defaults to five minutes", () => {
    expect(catchUpMinutes("Last 10 minutes")).toBe(10);
    expect(catchUpMinutes("Last 1 minutes")).toBe(1);
    expect(catchUpMinutes("Why Kafka?")).toBe(5);
    expect(catchUpMinutes(null)).toBe(5);
    expect(catchUpMinutes("Last 0 minutes")).toBe(5);
  });

  it("retries a catch-up as a catch-up, not as an answer", () => {
    expect(retryPlan(catchUp({ questionText: "Last 10 minutes" }))).toEqual({
      kind: "catch_up",
      minutes: 10,
    });
  });
});
