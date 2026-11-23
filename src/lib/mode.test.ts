import { describe, expect, it } from "vitest";
import type { Mode } from "@/bindings";
import { modeLabel, nextMode } from "./mode";

describe("mode labels and cycle", () => {
  it("labels every mode", () => {
    const labels: Record<Mode, string> = {
      candidate: "Candidate",
      interviewer: "Interviewer",
      workplace: "Workplace",
    };
    for (const mode of Object.keys(labels) as Mode[]) {
      expect(modeLabel(mode)).toBe(labels[mode]);
    }
  });

  it("cycles candidate, interviewer, workplace and back", () => {
    expect(nextMode("candidate")).toBe("interviewer");
    expect(nextMode("interviewer")).toBe("workplace");
    expect(nextMode("workplace")).toBe("candidate");
  });
});
