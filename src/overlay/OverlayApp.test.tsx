import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { clearMocks, mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import { emit } from "@tauri-apps/api/event";
import type { EngineSnapshot, MeetingEvent, Settings } from "@/bindings";
import { MEETING_EVENT } from "@/lib/ipc";
import { OverlayApp } from "./OverlayApp";
import { resetOverlayStore } from "./store";

const SETTINGS = {
  mode: "candidate",
  overlay: {
    opacity: 0.92,
    width: 380,
    maxHeight: 560,
    fontSize: 14,
    density: "comfortable",
    theme: "dark",
    translucent: true,
    minimal: false,
    position: "top_right",
    customPositions: {},
    targetDisplay: null,
    layout: "same_display",
    clickThrough: false,
    reducedMotion: true,
    showOnMeetingStart: true,
    margin: 16,
  },
  shortcuts: {
    bindings: {
      generate_answer: "CommandOrControl+Shift+A",
      ask_ai: "CommandOrControl+Shift+Space",
    },
  },
} as unknown as Settings;

function snapshot(patch: Partial<EngineSnapshot> = {}): EngineSnapshot {
  return {
    meeting: {
      id: "m1",
      title: "Interview",
      mode: "candidate",
      state: "active",
      startedAt: 1,
      endedAt: null,
      participants: [],
      persisted: true,
    },
    state: "active",
    pausedReason: null,
    mode: "candidate",
    style: "concise",
    transcript: [],
    questions: [],
    active: null,
    followups: [],
    followupsRequestId: null,
    pinned: [],
    topics: [],
    suggestedNextTopic: null,
    components: [],
    lastObservation: null,
    lastSummary: null,
    demoRunning: false,
    processingStopped: false,
    ...patch,
  };
}

let calls: Array<{ cmd: string; args: unknown }> = [];

function backend(snap: EngineSnapshot) {
  calls = [];
  mockWindows("overlay");
  mockIPC(
    (cmd, args) => {
      calls.push({ cmd, args });
      switch (cmd) {
        case "get_snapshot":
          return snap;
        case "get_settings":
          return SETTINGS;
        case "get_capture_privacy_status":
          return {
            supported: true,
            requested: true,
            active: true,
            platform: "windows",
            mechanism: "WDA_EXCLUDEFROMCAPTURE",
            note: "",
          };
        default:
          return null;
      }
    },
    { shouldMockEvents: true },
  );
}

async function sendEvent(ev: MeetingEvent) {
  await act(async () => {
    await emit(MEETING_EVENT, ev);
  });
}

beforeEach(() => {
  resetOverlayStore();
});

afterEach(() => {
  cleanup();
  clearMocks();
});

describe("OverlayApp", () => {
  it("hydrates an in-flight answer from the snapshot", async () => {
    backend(
      snapshot({
        active: {
          requestId: "r1",
          kind: "candidate_answer",
          slot: "primary",
          state: "streaming",
          questionText: "How would you scale the database?",
          category: "system_design",
          style: "concise",
          text: "[[ANSWER]]\nStart with read replicas",
          error: null,
        },
      }),
    );
    render(<OverlayApp />);
    expect(await screen.findByText(/Start with read replicas/)).toBeTruthy();
    expect(screen.getByText(/How would you scale the database\?/)).toBeTruthy();
    expect(screen.getByText("Candidate")).toBeTruthy();
  });

  it("streams a new answer from live events and ignores a superseded one", async () => {
    backend(snapshot());
    render(<OverlayApp />);
    expect(await screen.findByText(/Suggestions appear when a question is detected/)).toBeTruthy();

    const startedEv = (requestId: string, questionText: string): MeetingEvent => ({
      type: "ANSWER_STARTED",
      requestId,
      kind: "candidate_answer",
      slot: "primary",
      questionId: null,
      questionText,
      category: "technical",
      style: "concise",
      parentId: null,
    });
    await sendEvent(startedEv("old", "First question?"));
    await sendEvent(startedEv("new", "Second question?"));
    await sendEvent({ type: "ANSWER_DELTA", requestId: "old", text: "[[ANSWER]]\nSTALE" });
    await sendEvent({ type: "ANSWER_DELTA", requestId: "new", text: "[[ANSWER]]\nFresh answer" });

    expect(await screen.findByText(/Fresh answer/)).toBeTruthy();
    expect(screen.queryByText(/STALE/)).toBeNull();
    expect(screen.queryByText(/First question/)).toBeNull();
  });

  it("shows provider errors with a working Retry", async () => {
    backend(snapshot());
    render(<OverlayApp />);
    await screen.findByText(/Suggestions appear when a question is detected/);
    await sendEvent({
      type: "ANSWER_STARTED",
      requestId: "r1",
      kind: "candidate_answer",
      slot: "primary",
      questionId: null,
      questionText: "What is a B-tree?",
      category: "technical",
      style: "concise",
      parentId: null,
    });
    await sendEvent({
      type: "ANSWER_STATE_CHANGED",
      requestId: "r1",
      state: "failed",
      error: "Ollama is not running",
    });
    expect(await screen.findByText("Ollama is not running")).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Retry/ }));
    });
    const retry = calls.find((c) => c.cmd === "generate_answer");
    expect(retry?.args).toMatchObject({ request: { question: "What is a B-tree?" } });
  });

  it("shows PROCESSING STOPPED after an emergency stop and resumes on request", async () => {
    backend(snapshot({ processingStopped: true }));
    render(<OverlayApp />);
    expect(await screen.findByText("PROCESSING STOPPED")).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Resume" }));
    });
    expect(calls.some((c) => c.cmd === "resume_meeting")).toBe(true);
  });

  it("makes the overlay focusable only while typing an Ask prompt", async () => {
    backend(snapshot());
    render(<OverlayApp />);
    await screen.findByText(/Suggestions appear when a question is detected/);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Ask OpenMeet/ }));
    });
    expect(calls.some((c) => c.cmd === "overlay_begin_input")).toBe(true);
    const input = screen.getByRole("textbox", { name: "Ask OpenMeet" });
    fireEvent.change(input, { target: { value: "Explain CAP theorem briefly" } });
    await act(async () => {
      fireEvent.submit(input.closest("form")!);
    });
    expect(calls.find((c) => c.cmd === "ask_openmeet")?.args).toMatchObject({
      prompt: "Explain CAP theorem briefly",
    });
    expect(calls.some((c) => c.cmd === "overlay_end_input")).toBe(true);
    expect(screen.queryByRole("textbox", { name: "Ask OpenMeet" })).toBeNull();
  });
});
