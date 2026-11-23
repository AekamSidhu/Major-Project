import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { quietFlag, raisedFlag } from "@/lib/integrity.fixtures";
import { IntegrityCard } from "./cards";
import {
  INITIAL_STATE,
  hudLabel,
  reduceOverlay,
  visibleIntegrity,
  type OverlayAction,
  type OverlayState,
} from "./state";

afterEach(cleanup);

describe("IntegrityCard", () => {
  it("shows a raised flag with its evidence as a signal, not a verdict", () => {
    const onDismiss = vi.fn();
    render(<IntegrityCard flag={raisedFlag()} onDismiss={onDismiss} />);
    const card = screen.getByRole("region", { name: "Answer integrity" });
    const text = card.textContent ?? "";
    expect(text).toContain("Several signals point to AI assistance — worth a follow-up");
    expect(text).toContain("Answer at 23:54–25:00");
    expect(text).toContain("AI-written text score 0.91 (high)");
    expect(text).toContain("Close to a local LLM's answer: similarity 0.88 (uncalibrated)");
    expect(text).toContain("Started 3.1 s later than usual");
    expect(text).toContain("Eye movements look like reading");
    expect(text).toContain("never a verdict");
    // Uncalibrated signals are marked as such (echo and delivery).
    expect(within(card).getAllByText("uncalibrated")).toHaveLength(2);
    expect(text.toLowerCase()).not.toContain("cheat");
    expect(text.toLowerCase()).not.toContain("detected");

    fireEvent.click(screen.getByRole("button", { name: "Hide this check" }));
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("keeps a quiet result to one line until expanded", () => {
    render(<IntegrityCard flag={quietFlag()} onDismiss={() => {}} />);
    const card = screen.getByRole("region", { name: "Answer integrity" });
    expect(card.textContent).toContain("Nothing unusual · Answer at 1:04–1:15");
    expect(card.textContent).not.toContain("AI-written text score");
    fireEvent.click(within(card).getByText("Details"));
    expect(card.textContent).toContain("AI-written text score 0.12 (low)");
    expect(card.textContent).toContain("Learning this candidate's usual delivery");
  });

  it("shows only raised signals until all are requested", () => {
    const flag = raisedFlag({
      level: "possible",
      elevated: ["ai_text"],
      headline: "Possible AI-assisted answer — worth a follow-up",
    });
    flag.signals = flag.signals.map((s) =>
      s.name === "ai_text" ? s : { ...s, level: "low", evidence: [`${s.name} quiet`] },
    );
    render(<IntegrityCard flag={flag} onDismiss={() => {}} />);
    const card = screen.getByRole("region", { name: "Answer integrity" });
    expect(card.textContent).toContain("Possible AI-assisted answer — worth a follow-up");
    expect(card.textContent).not.toContain("reading quiet");
    fireEvent.click(within(card).getByText("Show all signals (3 more)"));
    expect(card.textContent).toContain("reading quiet");
  });
});

describe("overlay reducer: integrity", () => {
  const run = (actions: OverlayAction[], from: OverlayState = INITIAL_STATE) =>
    actions.reduce(reduceOverlay, from);
  const interviewer: OverlayState = { ...INITIAL_STATE, meetingId: "m1", mode: "interviewer" };

  it("keeps the latest revision and the latest answer", () => {
    const r2 = raisedFlag({ revision: 2, level: "possible" });
    let s = run([{ type: "INTEGRITY_CHECKED", flag: r2 }], interviewer);
    expect(s.integrity).toBe(r2);
    // A slower, older revision of the same answer is ignored…
    s = run([{ type: "INTEGRITY_CHECKED", flag: raisedFlag({ revision: 1 }) }], s);
    expect(s.integrity?.revision).toBe(2);
    // …and so is a late check of an earlier answer.
    s = run([{ type: "INTEGRITY_CHECKED", flag: quietFlag() }], s);
    expect(s.integrity?.id).toBe("answer-4");
    // Other meetings never leak in.
    s = run([{ type: "INTEGRITY_CHECKED", flag: raisedFlag({ meetingId: "m2", id: "x" }) }], s);
    expect(s.integrity?.id).toBe("answer-4");
  });

  it("dismissal hides a revision until the answer is re-checked; candidate mode hides it", () => {
    let s = run([{ type: "INTEGRITY_CHECKED", flag: raisedFlag() }], interviewer);
    expect(visibleIntegrity(s)).not.toBeNull();
    expect(hudLabel(s)).toBe("Answer flagged · worth a follow-up");
    s = run([{ type: "LOCAL_DISMISS_INTEGRITY" }], s);
    expect(visibleIntegrity(s)).toBeNull();
    s = run([{ type: "INTEGRITY_CHECKED", flag: raisedFlag({ revision: 2 }) }], s);
    expect(visibleIntegrity(s)?.revision).toBe(2);
    expect(visibleIntegrity({ ...s, mode: "candidate" })).toBeNull();
    // A new meeting starts clean.
    s = run(
      [
        {
          type: "MEETING_STARTED",
          meeting: {
            id: "m3",
            title: "Interview",
            mode: "interviewer",
            state: "active",
            startedAt: 1,
            endedAt: null,
            participants: [],
            persisted: true,
          },
        },
      ],
      s,
    );
    expect(s.integrity).toBeNull();
  });
});
