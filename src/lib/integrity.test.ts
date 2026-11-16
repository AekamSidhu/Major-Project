import { describe, expect, it } from "vitest";
import type { DisplayInfo } from "@/bindings";
import { reduceLive, liveFromSnapshot } from "@/main/store";
import {
  REGION_MIN,
  presetRegion,
  toAbsolute,
  toRelative,
  validateRegion,
} from "@/main/components/region";
import { integrityNodes, integrityPartLabel, otherNodes } from "@/main/components/dataFlow";
import {
  answerRangeLabel,
  flagSummary,
  levelTone,
  orderedSignals,
  sortFlags,
  upsertFlag,
} from "./integrity";
import { quietFlag, raisedFlag } from "./integrity.fixtures";

describe("integrity helpers", () => {
  it("formats the answer range and summarizes", () => {
    expect(answerRangeLabel(raisedFlag())).toBe("Answer at 23:54–25:00");
    expect(flagSummary(raisedFlag())).toBe("4 signals raised");
    expect(flagSummary(quietFlag())).toBe("Nothing unusual");
    expect(flagSummary(quietFlag({ level: "unavailable" }))).toBe("Not enough to check");
    expect(levelTone("high")).toBe("warn");
    expect(levelTone("low")).toBe("ok");
    expect(levelTone("baseline")).toBe("neutral");
  });

  it("orders raised signals first and upserts by revision", () => {
    const flag = quietFlag();
    flag.signals = [...flag.signals].reverse();
    expect(orderedSignals(flag).map((s) => s.name)).toEqual(["ai_text", "delivery"]);

    const a = quietFlag();
    const b = raisedFlag();
    let list = upsertFlag([], a);
    list = upsertFlag(list, b);
    expect(list.map((f) => f.id)).toEqual(["answer-4", "answer-1"]);
    list = upsertFlag(list, raisedFlag({ revision: 3, level: "possible" }));
    expect(list[0]?.revision).toBe(3);
    list = upsertFlag(list, raisedFlag({ revision: 2 }));
    expect(list[0]?.revision).toBe(3);
    expect(sortFlags([a, b])[0]?.id).toBe("answer-4");
  });
});

describe("live view reducer", () => {
  it("collects the meeting's flags from events and snapshots", () => {
    const snap = liveFromSnapshot({
      meeting: {
        id: "m1",
        title: "Interview",
        mode: "interviewer",
        state: "active",
        startedAt: 1,
        endedAt: null,
        participants: [],
        persisted: true,
      },
      state: "active",
      pausedReason: null,
      mode: "interviewer",
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
      integrityFlags: [quietFlag()],
    });
    let live = reduceLive(snap, { type: "INTEGRITY_CHECKED", flag: raisedFlag() });
    expect(live.integrityFlags.map((f) => f.id)).toEqual(["answer-4", "answer-1"]);
    live = reduceLive(live, { type: "INTEGRITY_CHECKED", flag: raisedFlag({ meetingId: "zz" }) });
    expect(live.integrityFlags).toHaveLength(2);
  });
});

describe("candidate region", () => {
  const displays: [DisplayInfo, DisplayInfo] = [
    {
      key: "a",
      name: "Display 1",
      x: 0,
      y: 0,
      width: 1920,
      height: 1080,
      workArea: { x: 0, y: 0, width: 1920, height: 1040 },
      scaleFactor: 1,
      isPrimary: true,
      orientation: "landscape",
    },
    {
      key: "b",
      name: "Display 2",
      x: 1920,
      y: -200,
      width: 2560,
      height: 1440,
      workArea: { x: 1920, y: -200, width: 2560, height: 1400 },
      scaleFactor: 1,
      isPrimary: false,
      orientation: "landscape",
    },
  ];

  it("converts between display-relative and absolute coordinates", () => {
    const rel = presetRegion(displays[1], "right");
    expect(rel).toEqual({ displayKey: "b", x: 1280, y: 0, width: 1280, height: 1440 });
    const abs = toAbsolute(rel, displays)!;
    expect(abs).toEqual({ x: 3200, y: -200, width: 1280, height: 1440 });
    expect(toRelative(abs, displays)).toEqual(rel);
    expect(toAbsolute({ ...rel, displayKey: "gone" }, displays)).toBeNull();
  });

  it("validates size and bounds", () => {
    const ok = presetRegion(displays[0], "top_right");
    expect(validateRegion(ok, displays)).toBeNull();
    expect(validateRegion({ ...ok, width: REGION_MIN - 1 }, displays)).toMatch(/at least/);
    expect(validateRegion({ ...ok, x: 1900 }, displays)).toMatch(/fit on the display/);
    expect(validateRegion({ ...ok, displayKey: "x" }, displays)).toBe("Choose a display.");
  });
});

describe("privacy data flow", () => {
  it("lists the integrity parts separately from the other nodes", () => {
    const report = {
      nodes: [
        {
          component: "embeddings" as const,
          label: "Embeddings",
          location: "local" as const,
          provider: "Built-in",
          endpoint: null,
        },
        {
          component: "integrity" as const,
          label: "Integrity: reference answers (AI provider)",
          location: "remote" as const,
          provider: "OpenAI · gpt-5-mini",
          endpoint: "api.openai.com",
        },
      ],
      localOnlyMode: false,
      allLocal: false,
      statement: "Meeting data is sent off this device for: Integrity: reference answers.",
    };
    expect(integrityNodes(report)).toHaveLength(1);
    expect(integrityPartLabel("Integrity: answer embeddings")).toBe("Answer embeddings");
    expect(otherNodes(report).map((n) => n.component)).toEqual(["embeddings"]);
  });
});
