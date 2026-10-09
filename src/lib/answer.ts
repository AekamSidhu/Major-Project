// Progressive parser for the streamed [[SECTION]] answer format produced by
// the candidate-answer / ask / refine / clarification prompts (see
// docs/INTERNALS.md). Works on partial text: call it on every delta.

export type SectionKey =
  | "structure"
  | "answer"
  | "keyPoints"
  | "example"
  | "followUp"
  | "basis"
  | "caveat"
  | "clarify";

export interface ParsedAnswer {
  structure: string[];
  answer: string;
  keyPoints: string[];
  example: string;
  followUp: string;
  basis: string;
  caveat: string;
  clarify: string[];
  /** Section currently being streamed (last marker seen). */
  current: SectionKey | null;
  /** Text before any marker (models occasionally omit markers). */
  preamble: string;
}

const MARKERS: Record<string, SectionKey> = {
  STRUCTURE: "structure",
  ANSWER: "answer",
  KEY_POINTS: "keyPoints",
  KEYPOINTS: "keyPoints",
  EXAMPLE: "example",
  FOLLOW_UP: "followUp",
  FOLLOWUP: "followUp",
  BASIS: "basis",
  CAVEAT: "caveat",
  CLARIFY: "clarify",
};

const MARKER_RE = /^\s*\[\[\s*([A-Z_ ]+?)\s*\]\]\s*(.*)$/;
const BULLET_RE = /^\s*(?:[-*•]|\d+[.)])\s+/;

function listItems(lines: string[]): string[] {
  const out: string[] = [];
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) continue;
    if (BULLET_RE.test(line)) {
      out.push(line.replace(BULLET_RE, "").trim());
    } else if (out.length > 0 && /^\s{2,}/.test(raw)) {
      out[out.length - 1] += ` ${line.trim()}`;
    } else {
      out.push(line.trim());
    }
  }
  return out;
}

export function emptyAnswer(): ParsedAnswer {
  return {
    structure: [],
    answer: "",
    keyPoints: [],
    example: "",
    followUp: "",
    basis: "",
    caveat: "",
    clarify: [],
    current: null,
    preamble: "",
  };
}

/** Parses (possibly partial) streamed text into sections. */
export function parseAnswer(text: string): ParsedAnswer {
  const result = emptyAnswer();
  const buckets = new Map<SectionKey | "preamble", string[]>();
  let current: SectionKey | "preamble" = "preamble";
  const lines = text.replace(/\r\n?/g, "\n").replace(/```[a-z]*\n?/gi, "").split("\n");
  for (const line of lines) {
    const m = MARKER_RE.exec(line);
    const key = m ? MARKERS[(m[1] ?? "").replace(/ /g, "_").toUpperCase()] : undefined;
    if (m && key) {
      current = key;
      result.current = key;
      if (!buckets.has(key)) buckets.set(key, []);
      const rest = m[2] ?? "";
      if (rest.trim()) buckets.get(key)?.push(rest);
      continue;
    }
    // Hide an incomplete marker at the very end of a partial stream ("[[KEY_PO").
    if (/^\s*\[\[[A-Z_ ]*\]?$/.test(line)) continue;
    if (!buckets.has(current)) buckets.set(current, []);
    buckets.get(current)?.push(line);
  }
  const join = (k: SectionKey | "preamble") => (buckets.get(k) ?? []).join("\n").trim();
  result.preamble = join("preamble");
  result.answer = join("answer");
  result.example = join("example");
  result.followUp = join("followUp");
  result.basis = join("basis");
  result.caveat = join("caveat");
  result.structure = listItems(buckets.get("structure") ?? []);
  result.keyPoints = listItems(buckets.get("keyPoints") ?? []);
  result.clarify = listItems(buckets.get("clarify") ?? []);
  // Unmarked output: treat it as the answer.
  if (!result.answer && result.preamble && result.current === null) {
    result.answer = result.preamble;
    result.preamble = "";
  }
  return result;
}

/** Plain-text rendering for "Copy". */
export function answerToPlainText(a: ParsedAnswer): string {
  const parts: string[] = [];
  if (a.structure.length) parts.push(a.structure.map((s, i) => `${i + 1}. ${s}`).join("\n"));
  if (a.answer) parts.push(a.answer);
  if (a.keyPoints.length) parts.push(a.keyPoints.map((p) => `• ${p}`).join("\n"));
  if (a.example) parts.push(`Example: ${a.example}`);
  if (a.clarify.length) parts.push(a.clarify.map((p) => `• ${p}`).join("\n"));
  if (a.followUp) parts.push(`You could ask: ${a.followUp}`);
  if (a.caveat) parts.push(a.caveat);
  return parts.join("\n\n");
}
