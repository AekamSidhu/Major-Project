import { describe, expect, it } from "vitest";
import type { DataFlowNode, DataFlowReport, Settings } from "@/bindings";
import { acceleratorFromEvent, conflictsFor, normalizeAccelerator, validateAccelerator } from "./accelerator";
import { ALL_LOCAL_STATEMENT, audioFlow, privacyStatement } from "./dataFlow";
import { effectiveEndpoint, locationForUrl, resolvedBaseUrl, DEFAULT_OPENAI_URL, DEFAULT_OLLAMA_URL } from "./settingsMeta";
import { normalizeRuleValue, parseRegion, validateBaseUrl } from "./validation";

function node(component: DataFlowNode["component"], location: DataFlowNode["location"], provider = "x", endpoint: string | null = null): DataFlowNode {
  return { component, label: component, location, provider, endpoint };
}

function report(nodes: DataFlowNode[], allLocal: boolean, statement = ""): DataFlowReport {
  return { nodes, localOnlyMode: false, allLocal, statement };
}

describe("privacyStatement", () => {
  it("claims that nothing leaves the device only when the report says so", () => {
    const local = report([node("transcription", "local"), node("llm", "local")], true);
    expect(privacyStatement(local)).toMatchObject({ tone: "ok", text: ALL_LOCAL_STATEMENT });
  });

  it("never repeats an all-local claim when a node is remote", () => {
    const r = report([node("llm", "remote", "OpenAI", "api.openai.com")], false, "No meeting data leaves this device.");
    const s = privacyStatement(r);
    expect(s.tone).toBe("warn");
    expect(s.text).not.toMatch(/no meeting data leaves/i);
    expect(s.text).toContain("api.openai.com");
  });

  it("is cautious when the report is inconsistent", () => {
    const s = privacyStatement(report([node("llm", "remote")], true));
    expect(s.tone).toBe("warn");
  });

  it("marks audio as off when both sources are disabled", () => {
    const steps = audioFlow(report([node("microphone", "disabled"), node("system_audio", "disabled")], true));
    expect(steps[0]).toMatchObject({ location: "disabled", detail: "Off" });
    expect(steps.map((s) => s.key)).toEqual(["audio", "transcription", "context", "llm", "overlay"]);
  });
});

describe("accelerators", () => {
  const key = (code: string, mods: Partial<Record<"ctrlKey" | "metaKey" | "altKey" | "shiftKey", boolean>> = {}) => ({
    code,
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    shiftKey: false,
    ...mods,
  });

  it("builds Tauri accelerators from key events", () => {
    expect(acceleratorFromEvent(key("KeyK", { ctrlKey: true, shiftKey: true }), false)).toMatchObject({
      kind: "complete",
      accelerator: "CommandOrControl+Shift+K",
    });
    expect(acceleratorFromEvent(key("ShiftLeft", { shiftKey: true }), false).kind).toBe("partial");
    expect(acceleratorFromEvent(key("KeyK", { metaKey: true }), true)).toMatchObject({ accelerator: "CommandOrControl+K" });
  });

  it("rejects shortcuts that would block typing", () => {
    expect(validateAccelerator("Shift+K")).not.toBeNull();
    expect(validateAccelerator("K")).not.toBeNull();
    expect(validateAccelerator("CommandOrControl+Shift+K")).toBeNull();
    expect(validateAccelerator("F9")).toBeNull();
  });

  it("detects conflicts regardless of modifier spelling and order", () => {
    expect(normalizeAccelerator("Shift+CmdOrCtrl+k")).toBe(normalizeAccelerator("CommandOrControl+Shift+K"));
    const bindings = [
      { action: "toggle_overlay" as const, accelerator: "CommandOrControl+Shift+O" },
      { action: "ask_ai" as const, accelerator: "CommandOrControl+Shift+Space" },
    ];
    expect(conflictsFor("ask_ai", "Shift+CmdOrCtrl+O", bindings)).toEqual(["toggle_overlay"]);
    expect(conflictsFor("toggle_overlay", "CommandOrControl+Shift+O", bindings)).toEqual([]);
  });
});

describe("endpoints", () => {
  it("classifies loopback hosts as local and everything else as remote", () => {
    expect(locationForUrl("http://127.0.0.1:11434")).toBe("local");
    expect(locationForUrl("http://localhost:8765/v1")).toBe("local");
    expect(locationForUrl("http://[::1]:1234")).toBe("local");
    expect(locationForUrl("http://192.168.1.20:11434")).toBe("remote");
    expect(locationForUrl("https://api.openai.com")).toBe("remote");
    expect(locationForUrl("http://localhost.evil.com")).toBe("remote");
  });

  it("ignores another provider's default URL left over from switching", () => {
    expect(resolvedBaseUrl(DEFAULT_OPENAI_URL, DEFAULT_OLLAMA_URL)).toBe(DEFAULT_OPENAI_URL);
    expect(resolvedBaseUrl(DEFAULT_OLLAMA_URL, "http://gpu-box:11434")).toBe("http://gpu-box:11434");
  });

  it("reports the built-in embeddings as local and needs an address for compatible servers", () => {
    const settings = {
      ai: {
        llm: { provider: "openai_compatible", model: "m", baseUrl: null },
        vision: { provider: "disabled", model: "", baseUrl: null, useOcr: true },
        embeddings: { provider: "local_hash", model: "", baseUrl: null },
      },
      transcription: { provider: "local_ai", model: "base.en", baseUrl: null },
    } as unknown as Settings;
    expect(effectiveEndpoint(settings, "embeddings")).toEqual({ url: "", location: "local" });
    expect(effectiveEndpoint(settings, "llm")).toEqual({ url: "", location: "unset" });
    expect(effectiveEndpoint(settings, "vision").location).toBe("disabled");
    expect(effectiveEndpoint(settings, "transcription").location).toBe("local");
  });
});

describe("validation", () => {
  it("validates base URLs", () => {
    expect(validateBaseUrl("http://127.0.0.1:11434")).toBeNull();
    expect(validateBaseUrl("ftp://example.com")).not.toBeNull();
    expect(validateBaseUrl("https://user:pass@example.com")).not.toBeNull();
  });

  it("normalizes privacy rule values", () => {
    expect(normalizeRuleValue("domain", "https://Bank.Example.com/login?x=1")).toEqual({ ok: true, value: "bank.example.com" });
    expect(normalizeRuleValue("domain", "not a domain").ok).toBe(false);
    expect(normalizeRuleValue("region", " 0, 10 ,800,600 ")).toEqual({ ok: true, value: "0,10,800,600" });
    expect(normalizeRuleValue("region", "0,0,0,10").ok).toBe(false);
    expect(normalizeRuleValue("application", "  ").ok).toBe(false);
    expect(parseRegion("1,2,3")).toMatchObject({ ok: false });
  });
});
