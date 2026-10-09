// Pure helpers for the privacy data-flow view. The backend report is the
// source of truth; these functions only arrange it and make sure the UI never
// claims more privacy than the report supports.

import type { Component, DataFlowNode, DataFlowReport, DataLocation } from "@/bindings";

export const ALL_LOCAL_STATEMENT = "No meeting data leaves this device.";

export type StatementTone = "ok" | "warn";

export interface PrivacyStatement {
  tone: StatementTone;
  text: string;
  /** Active components that send data off the device. */
  remote: DataFlowNode[];
}

/** Nodes whose data leaves the device. */
export function remoteNodes(report: DataFlowReport): DataFlowNode[] {
  return report.nodes.filter((n) => n.location === "remote");
}

function describeRemote(nodes: DataFlowNode[]): string {
  const parts = nodes.map((n) => {
    const where = n.endpoint ? `${n.provider} (${n.endpoint})` : n.provider;
    return `${n.label} → ${where}`;
  });
  return `Some meeting data leaves this device: ${parts.join("; ")}.`;
}

/**
 * The statement to show. "No meeting data leaves this device." is claimed
 * only when the report says `allLocal` and no node is remote; a backend
 * statement making that claim otherwise is replaced by an honest one.
 */
export function privacyStatement(report: DataFlowReport): PrivacyStatement {
  const remote = remoteNodes(report);
  if (report.allLocal && remote.length === 0) {
    return { tone: "ok", text: ALL_LOCAL_STATEMENT, remote };
  }
  const backend = report.statement.trim();
  const claimsAllLocal = /no meeting data leaves/i.test(backend);
  let text: string;
  if (backend && !claimsAllLocal) text = backend;
  else if (remote.length > 0) text = describeRemote(remote);
  else text = "OpenMeet cannot confirm that all meeting data stays on this device.";
  return { tone: "warn", text, remote };
}

export interface FlowStep {
  key: string;
  label: string;
  /** Node from the report; fixed in-app steps have none. */
  node: DataFlowNode | null;
  /** Location used for the badge. */
  location: DataLocation | "unknown";
  detail: string;
}

/** Steps that always run inside OpenMeet on this device. */
const IN_APP_LABELS = { context: "Context engine", overlay: "Overlay" } as const;

function find(report: DataFlowReport, component: Component): DataFlowNode | null {
  return report.nodes.find((n) => n.component === component) ?? null;
}

function stepFor(report: DataFlowReport, component: Component, fallbackLabel: string): FlowStep {
  const node = find(report, component);
  if (!node) {
    return { key: component, label: fallbackLabel, node: null, location: "unknown", detail: "Not reported" };
  }
  const detail =
    node.location === "disabled"
      ? "Off"
      : node.endpoint
        ? `${node.provider} · ${node.endpoint}`
        : node.provider;
  return { key: component, label: node.label || fallbackLabel, node, location: node.location, detail };
}

function inApp(key: keyof typeof IN_APP_LABELS): FlowStep {
  return { key, label: IN_APP_LABELS[key], node: null, location: "local", detail: "Inside OpenMeet" };
}

/** Audio input step: microphone and system audio combined. */
function audioStep(report: DataFlowReport): FlowStep {
  const mic = find(report, "microphone");
  const sys = find(report, "system_audio");
  if (!mic && !sys) return stepFor(report, "microphone", "Microphone");
  const active = [mic, sys].filter((n): n is DataFlowNode => !!n && n.location !== "disabled");
  const names = active.map((n) => n.label);
  const location: DataLocation = active.length === 0 ? "disabled" : active.some((n) => n.location === "remote") ? "remote" : "local";
  return {
    key: "audio",
    label: "Audio input",
    node: mic ?? sys,
    location,
    detail: names.length > 0 ? names.join(" + ") : "Off",
  };
}

/** Microphone/system audio → Transcription → Context engine → AI provider → Overlay. */
export function audioFlow(report: DataFlowReport): FlowStep[] {
  return [
    audioStep(report),
    stepFor(report, "transcription", "Transcription"),
    inApp("context"),
    stepFor(report, "llm", "AI provider"),
    inApp("overlay"),
  ];
}

/** Screen → Vision → Context engine → AI provider. */
export function screenFlow(report: DataFlowReport): FlowStep[] {
  return [
    stepFor(report, "screen", "Screen"),
    stepFor(report, "vision", "Vision"),
    inApp("context"),
    stepFor(report, "llm", "AI provider"),
  ];
}

const SHOWN_IN_FLOWS: ReadonlyArray<Component> = ["microphone", "system_audio", "transcription", "llm", "screen", "vision", "overlay"];

/** Report nodes not drawn in the two flows (embeddings, storage, …). */
export function otherNodes(report: DataFlowReport): DataFlowNode[] {
  return report.nodes.filter((n) => !SHOWN_IN_FLOWS.includes(n.component));
}

/** True when the screen pipeline is entirely off. */
export function screenFlowDisabled(report: DataFlowReport): boolean {
  const screen = find(report, "screen");
  return !!screen && screen.location === "disabled";
}
