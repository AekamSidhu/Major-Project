import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { clearMocks, mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import type { DataFlowReport, EngineSnapshot, Settings, TranscriptSegment } from "@/bindings";
import { liveFromSnapshot, reduceLive, useMain } from "./store";
import { buildExportRequest, pageRequest, splitPage } from "./sections/history/helpers";
import { PrivacySection } from "./sections/Privacy";
import { IntegritySection } from "./sections/settings/Integrity";

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
  integrityFlags: [],
  workplace: null,
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
    const none = {
      summary: false,
      transcript: false,
      questions: false,
      followups: false,
      integrity: false,
    };
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

describe("Answer integrity", () => {
  const base = {
    ai: {
      llm: { provider: "ollama", model: "llama3.2", baseUrl: "http://127.0.0.1:11434" },
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
    integrity: {
      enabled: false,
      aiText: true,
      echoSimilarity: true,
      delivery: true,
      reading: true,
      candidateRegion: null,
    },
  } as unknown as Settings;

  let saved: Settings[] = [];

  function mount(node: "settings" | "privacy", report: DataFlowReport) {
    saved = [];
    mockWindows("main");
    mockIPC((cmd, args) => {
      switch (cmd) {
        case "get_data_flow":
          return report;
        case "get_integrity_status":
          return {
            reachable: true,
            aiTextReady: true,
            readingReady: false,
            embeddingModel: "BAAI/bge-small-en-v1.5",
            requiredEmbeddingModel: "BAAI/bge-small-en-v1.5",
            echoCalibrated: true,
            echoReferenceModel: "llama3.2",
            deliveryCalibrated: false,
            message:
              "Local AI is running but the reading model (needs onnxruntime) is not available.",
          };
        case "list_displays":
          return [
            {
              key: "d1",
              name: "DISPLAY1",
              x: 0,
              y: 0,
              width: 1920,
              height: 1080,
              workArea: { x: 0, y: 0, width: 1920, height: 1040 },
              scaleFactor: 1,
              isPrimary: true,
              orientation: "landscape",
            },
          ];
        case "update_settings": {
          const s = (args as { settings: Settings }).settings;
          saved.push(s);
          return s;
        }
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
    useMain.setState({ settings: structuredClone(base) });
    render(node === "settings" ? <IntegritySection /> : <PrivacySection />);
  }

  const offReport: DataFlowReport = {
    nodes: [],
    localOnlyMode: false,
    allLocal: true,
    statement: "No meeting data leaves this device.",
  };

  afterEach(() => {
    cleanup();
    clearMocks();
  });

  it("is off by default, reminds to tell the candidate and turns on", async () => {
    mount("settings", offReport);
    expect(screen.getByText("Tell the candidate")).toBeTruthy();
    const toggle = screen.getByRole("switch", { name: "Check answers in interviewer mode" });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    expect(await screen.findByText(/reading model \(needs onnxruntime\)/)).toBeTruthy();
    // The echo check needs the local embedder; the page says so.
    expect(screen.getByText(/echo check is unavailable/)).toBeTruthy();
    // Calibrated echo thresholds hold for the configured reference model.
    expect(await screen.findByText(/Calibrated with reference answers from/)).toBeTruthy();
    fireEvent.click(toggle);
    await waitFor(() => expect(saved.length).toBe(1));
    expect(saved[0]?.integrity.enabled).toBe(true);
    // A region on the display can be chosen and saved (absolute coordinates).
    fireEvent.click(await screen.findByRole("button", { name: "Use this region" }));
    await waitFor(() => expect(saved.length).toBe(2));
    expect(saved[1]?.integrity.candidateRegion).toEqual({
      x: 960,
      y: 0,
      width: 960,
      height: 1080,
    });
  });

  it("lists the integrity parts as Local or Cloud on the Privacy page", async () => {
    mount("privacy", offReport);
    expect(await screen.findByText(/answer check is off/)).toBeTruthy();
    cleanup();
    clearMocks();
    mount("privacy", {
      nodes: [
        {
          component: "integrity",
          label: "Integrity: AI-text and reading check",
          location: "local",
          provider: "Local AI (integrity check)",
          endpoint: "127.0.0.1:8765",
        },
        {
          component: "integrity",
          label: "Integrity: reference answers (AI provider)",
          location: "remote",
          provider: "OpenAI · gpt-5-mini",
          endpoint: "api.openai.com",
        },
      ],
      localOnlyMode: false,
      allLocal: false,
      statement:
        "Meeting data is sent off this device for: Integrity: reference answers (AI provider) (api.openai.com).",
    });
    const list = await screen.findByRole("list", { name: "Answer integrity data flow" });
    expect(list.textContent).toContain("AI-text and reading check");
    expect(list.textContent).toContain("Local");
    expect(list.textContent).toContain("Reference answers (AI provider)");
    expect(list.textContent).toContain("Cloud");
  });
});
