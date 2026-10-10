import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { clearMocks, mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { DataFlowReport, EngineSnapshot, Settings, TranscriptSegment } from "@/bindings";
import { liveFromSnapshot, reduceLive, useMain } from "./store";
import { buildExportRequest, pageRequest, splitPage } from "./sections/history/helpers";
import { PrivacySection } from "./sections/Privacy";

function seg(
  id: string,
  utteranceId: string,
  text: string,
  isFinal: boolean,
  startTime: number,
): TranscriptSegment {
  return {
    id,
    meetingId: "m1",
    utteranceId,
    speakerId: null,
    speakerRole: "remote",
    source: "system",
    startTime,
    endTime: startTime + 500,
    text,
    isFinal,
    confidence: null,
  };
}

const EMPTY_SNAPSHOT: EngineSnapshot = {
  meeting: null,
  state: "idle",
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
};

describe("reduceLive", () => {
  it("replaces a partial transcript with the final of the same utterance", () => {
    let live = liveFromSnapshot(EMPTY_SNAPSHOT);
    live = reduceLive(live, {
      type: "TRANSCRIPT_PARTIAL",
      segment: seg("p1", "u1", "how would", false, 1000),
    });
    live = reduceLive(live, {
      type: "TRANSCRIPT_FINAL",
      segment: seg("f0", "u0", "hello", true, 0),
    });
    live = reduceLive(live, {
      type: "TRANSCRIPT_FINAL",
      segment: seg("f1", "u1", "How would you test it?", true, 1000),
    });
    expect(live.transcript.map((s) => s.text)).toEqual(["hello", "How would you test it?"]);
  });
});

describe("reduceLive — deleted meeting", () => {
  it("forgets the meeting when the backend reports idle without a meeting id", () => {
    let live = liveFromSnapshot(EMPTY_SNAPSHOT);
    live = reduceLive(live, {
      type: "TRANSCRIPT_FINAL",
      segment: seg("f1", "u1", "private words", true, 0),
    });
    live = reduceLive(live, {
      type: "MEETING_STATE_CHANGED",
      meetingId: null,
      state: "idle",
      mode: "interviewer",
      pausedReason: null,
    });
    expect(live.transcript).toEqual([]);
    expect(live.meeting).toBeNull();
    expect(live.mode).toBe("interviewer");
  });
});

describe("history helpers", () => {
  it("pages with one look-ahead row", () => {
    expect(pageRequest(2, 20)).toEqual({ limit: 21, offset: 40 });
    expect(splitPage([1, 2, 3], 2)).toEqual({ items: [1, 2], hasMore: true });
    expect(splitPage([1, 2], 2)).toEqual({ items: [1, 2], hasMore: false });
  });

  it("requires at least one export part", () => {
    const none = { summary: false, transcript: false, questions: false, followups: false };
    expect(buildExportRequest("m1", "markdown", none).error).not.toBeNull();
    expect(buildExportRequest("m1", "json", { ...none, transcript: true }).request).toEqual({
      meetingId: "m1",
      format: "json",
      parts: { ...none, transcript: true },
    });
  });
});

describe("PrivacySection", () => {
  const settings = {
    ai: {
      llm: { provider: "openai", model: "gpt-5-mini", baseUrl: null },
      vision: { provider: "ocr_only", model: "", baseUrl: null, useOcr: true },
      embeddings: { provider: "local_hash", model: "", baseUrl: null },
    },
    transcription: { provider: "local_ai", model: "base.en", baseUrl: null },
    audio: {
      microphoneEnabled: true,
      systemAudioEnabled: true,
      microphoneDevice: null,
      systemAudioDevice: null,
    },
    screen: { enabled: false, source: null },
    privacy: { captureExclusion: true, localOnly: false, retention: "thirty_days" },
  } as unknown as Settings;

  function mount(report: DataFlowReport) {
    mockWindows("main");
    mockIPC((cmd) => {
      switch (cmd) {
        case "get_data_flow":
          return report;
        case "get_capture_privacy_status":
          return {
            supported: true,
            requested: true,
            active: true,
            platform: "windows",
            mechanism: "SetWindowDisplayAffinity",
            note: "",
          };
        case "list_privacy_rules":
        case "suggested_privacy_apps":
          return [];
        default:
          return null;
      }
    });
    useMain.setState({ settings });
    render(<PrivacySection />);
  }

  beforeEach(() => {
    useMain.setState({ settings: null });
  });
  afterEach(() => {
    cleanup();
    clearMocks();
  });

  it("names the cloud provider instead of claiming that data stays local", async () => {
    mount({
      nodes: [
        {
          component: "transcription",
          label: "Transcription",
          location: "local",
          provider: "Local AI",
          endpoint: "127.0.0.1:8765",
        },
        {
          component: "llm",
          label: "AI provider",
          location: "remote",
          provider: "OpenAI",
          endpoint: "api.openai.com",
        },
      ],
      localOnlyMode: false,
      allLocal: false,
      statement: "Transcript excerpts are sent to OpenAI (api.openai.com).",
    });
    expect(await screen.findByText("Some data is sent to cloud providers")).toBeTruthy();
    expect(screen.queryByText("No meeting data leaves this device.")).toBeNull();
  });

  it("shows the all-local statement when everything is local", async () => {
    mount({
      nodes: [
        {
          component: "llm",
          label: "AI provider",
          location: "local",
          provider: "Ollama",
          endpoint: "127.0.0.1:11434",
        },
      ],
      localOnlyMode: false,
      allLocal: true,
      statement: "No meeting data leaves this device.",
    });
    expect(await screen.findByText("No meeting data leaves this device.")).toBeTruthy();
  });
});
