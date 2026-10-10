import { describe, expect, it } from "vitest";
import type { FollowUpQuestion, MeetingEvent, RequestKind, Suggestion } from "@/bindings";
import {
  INITIAL_STATE,
  deriveStatus,
  hudLabel,
  reduceOverlay,
  retryPlan,
  tickerText,
  type OverlayAction,
  type OverlayState,
} from "./state";

function run(actions: OverlayAction[], from: OverlayState = INITIAL_STATE): OverlayState {
  return actions.reduce(reduceOverlay, from);
}

function started(
  requestId: string,
  kind: RequestKind = "candidate_answer",
  questionText = "How would you shard it?",
): MeetingEvent {
  return {
    type: "ANSWER_STARTED",
    requestId,
    kind,
    slot: kind === "interviewer_followups" ? "followups" : "primary",
    questionId: `q-${requestId}`,
    questionText,
    category: "system_design",
    style: "concise",
    parentId: null,
  };
}

const delta = (requestId: string, text: string): MeetingEvent => ({
  type: "ANSWER_DELTA",
  requestId,
  text,
});

function suggestion(id: string, content: string, followups: FollowUpQuestion[] = []): Suggestion {
  return {
    id,
    meetingId: "m1",
    kind: followups.length ? "followups" : "answer",
    questionId: null,
    questionText: null,
    category: null,
    style: null,
    parentId: null,
    promptId: "candidate_answer",
    promptVersion: 1,
    content,
    followups,
    provider: "mock",
    model: "mock",
    createdAt: 1,
    completedAt: 2,
    pinned: false,
  };
}

function followup(id: string, requestId: string, question: string): FollowUpQuestion {
  return {
    id,
    requestId,
    question,
    category: "technical",
    reason: "Candidate mentioned Redis but not failover",
    priority: "high",
    trigger: null,
    createdAt: 1,
  };
}

describe("reduceOverlay — request slots", () => {
  it("streams deltas into the current primary request and completes it", () => {
    const s = run([
      started("r1"),
      { type: "ANSWER_STATE_CHANGED", requestId: "r1", state: "requesting", error: null },
      delta("r1", "[[ANSWER]]\nUse consistent "),
      delta("r1", "hashing."),
    ]);
    expect(s.primary?.text).toBe("[[ANSWER]]\nUse consistent hashing.");
    expect(s.primary?.state).toBe("streaming");
    expect(deriveStatus(s).kind).toBe("answering");

    const done = reduceOverlay(s, {
      type: "ANSWER_COMPLETE",
      requestId: "r1",
      suggestion: suggestion("r1", "[[ANSWER]]\nUse consistent hashing on the user id."),
    });
    expect(done.primary?.state).toBe("complete");
    expect(done.primary?.final).toBe(true);
    expect(done.primary?.text).toContain("on the user id");
  });

  it("ignores deltas, completions and state changes of a superseded request", () => {
    const s = run([
      started("old"),
      delta("old", "old text"),
      started("new"),
      delta("old", " late delta"),
      { type: "ANSWER_STATE_CHANGED", requestId: "old", state: "failed", error: "boom" },
      { type: "ANSWER_COMPLETE", requestId: "old", suggestion: suggestion("old", "stale final") },
      delta("new", "fresh"),
    ]);
    expect(s.primary?.requestId).toBe("new");
    expect(s.primary?.text).toBe("fresh");
    expect(s.primary?.state).toBe("streaming");
    expect(s.primary?.error).toBeNull();
    // A late ANSWER_STARTED for the retired id cannot resurrect it.
    const again = reduceOverlay(s, started("old"));
    expect(again.primary?.requestId).toBe("new");
  });

  it("keeps the follow-ups slot independent and only accepts its latest request", () => {
    const s = run([
      { type: "MODE_CHANGED", mode: "interviewer" },
      started("f1", "interviewer_followups", ""),
      {
        type: "FOLLOWUPS_READY",
        requestId: "f1",
        questions: [followup("a", "f1", "How does it fail over?")],
        done: false,
      },
      started("f2", "interviewer_followups", ""),
      {
        type: "FOLLOWUPS_READY",
        requestId: "f1",
        questions: [followup("b", "f1", "stale")],
        done: true,
      },
      {
        type: "FOLLOWUPS_READY",
        requestId: "f2",
        questions: [followup("c", "f2", "What was the p99 latency?")],
        done: true,
      },
    ]);
    expect(s.followups?.requestId).toBe("f2");
    expect(s.followups?.questions.map((q) => q.question)).toEqual(["What was the p99 latency?"]);
    expect(s.followups?.done).toBe(true);
    expect(hudLabel(s)).toBe("1 follow-up");
  });

  it("shows provider failures instead of failing silently", () => {
    const s = run([
      started("r1"),
      {
        type: "ANSWER_STATE_CHANGED",
        requestId: "r1",
        state: "failed",
        error: "Ollama is not running",
      },
    ]);
    expect(s.primary?.state).toBe("failed");
    expect(s.primary?.error).toBe("Ollama is not running");
    expect(hudLabel(s)).toBe("Answer failed");
    expect(retryPlan(s.primary!)).toEqual({ kind: "answer", question: "How would you shard it?" });
  });

  it("removes a dismissed request and ignores its later events", () => {
    const s = run([
      started("r1"),
      { type: "SUGGESTION_DISMISSED", requestId: "r1" },
      delta("r1", "late"),
    ]);
    expect(s.primary).toBeNull();
  });
});

describe("reduceOverlay — emergency stop and lifecycle", () => {
  it("shows PROCESSING STOPPED until the meeting is active again", () => {
    const stopped = run([
      {
        type: "MEETING_STATE_CHANGED",
        meetingId: "m1",
        state: "active",
        mode: "candidate",
        pausedReason: null,
      },
      { type: "PROCESSING_STOPPED", reason: "Emergency stop" },
    ]);
    expect(stopped.processingStopped).toBe(true);
    expect(deriveStatus(stopped).kind).toBe("stopped");
    const resumed = reduceOverlay(stopped, {
      type: "MEETING_STATE_CHANGED",
      meetingId: "m1",
      state: "active",
      mode: "candidate",
      pausedReason: null,
    });
    expect(resumed.processingStopped).toBe(false);
    expect(deriveStatus(resumed).kind).toBe("listening");
  });

  it("only offers questions that need the user's response in candidate mode", () => {
    const base = run([
      {
        type: "MEETING_STATE_CHANGED",
        meetingId: "m1",
        state: "active",
        mode: "candidate",
        pausedReason: null,
      },
    ]);
    const q = {
      id: "q1",
      meetingId: "m1",
      segmentIds: [],
      isQuestion: true,
      confidence: 0.9,
      question: "Can you walk me through your last project?",
      parts: [],
      category: "behavioral" as const,
      speakerId: null,
      speakerRole: "remote" as const,
      requiresResponse: true,
      method: "heuristic" as const,
      supersedes: null,
      detectedAt: 1,
    };
    const asked = reduceOverlay(base, { type: "QUESTION_DETECTED", question: q });
    expect(asked.pendingQuestion?.id).toBe("q1");
    const own = reduceOverlay(base, {
      type: "QUESTION_DETECTED",
      question: { ...q, requiresResponse: false },
    });
    expect(own.pendingQuestion).toBeNull();
    const interviewer = reduceOverlay(
      reduceOverlay(base, { type: "MODE_CHANGED", mode: "interviewer" }),
      {
        type: "QUESTION_DETECTED",
        question: q,
      },
    );
    expect(interviewer.pendingQuestion).toBeNull();
  });

  it("drops everything of a meeting that was deleted", () => {
    const s = run([
      {
        type: "MEETING_STATE_CHANGED",
        meetingId: "m1",
        state: "summarizing",
        mode: "candidate",
        pausedReason: null,
      },
      started("r1"),
      delta("r1", "[[ANSWER]]\nsecret answer"),
      {
        type: "MEETING_STATE_CHANGED",
        meetingId: null,
        state: "idle",
        mode: "candidate",
        pausedReason: null,
      },
      delta("r1", " late"),
    ]);
    expect(s.meetingId).toBeNull();
    expect(s.primary).toBeNull();
    expect(s.meetingState).toBe("idle");
  });

  it("caps notices", () => {
    const s = run(
      Array.from({ length: 6 }, (_, i) => ({
        type: "LOCAL_NOTICE" as const,
        level: "info" as const,
        message: `n${i}`,
      })),
    );
    expect(s.notices.length).toBeLessThanOrEqual(3);
    expect(s.notices.at(-1)?.message).toBe("n5");
  });
});

describe("tickerText", () => {
  it("keeps short lines and tails long ones at a word boundary", () => {
    expect(tickerText("  hello   world ")).toBe("hello world");
    const long = Array.from({ length: 60 }, (_, i) => `word${i}`).join(" ");
    const t = tickerText(long, 40);
    expect(t.startsWith("…")).toBe(true);
    expect(t.length).toBeLessThanOrEqual(41);
    expect(long.endsWith(t.slice(1))).toBe(true);
  });
});
