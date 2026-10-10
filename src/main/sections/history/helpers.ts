// Pure helpers for the History section (pagination, export, Q&A assembly,
// summary sections). Unit-tested; no React here.

import type {
  AnswerRecord,
  AnswerStyle,
  Claim,
  ClaimKind,
  DetectedQuestion,
  ExportFormat,
  ExportParts,
  ExportRequest,
  FollowUpQuestion,
  MeetingListItem,
  MeetingRecord,
  MeetingSummary,
  Mode,
  Priority,
  ScreenObservation,
  SpeakerRole,
  Suggestion,
  SuggestionKind,
  TopicStatus,
} from "@/bindings";
import type { Tone } from "@/main/sections/home/helpers";

// ---------------------------------------------------------------------------
// Pagination
// ---------------------------------------------------------------------------

export const PAGE_SIZE = 20;

/** Asks for one extra row so the UI knows whether a next page exists. */
export function pageRequest(page: number, pageSize = PAGE_SIZE): { limit: number; offset: number } {
  return { limit: pageSize + 1, offset: Math.max(0, page) * pageSize };
}

export function splitPage<T>(
  rows: ReadonlyArray<T>,
  pageSize = PAGE_SIZE,
): { items: T[]; hasMore: boolean } {
  return { items: rows.slice(0, pageSize), hasMore: rows.length > pageSize };
}

/** "21–40" for page 1 with 20 rows. */
export function pageRangeLabel(page: number, count: number, pageSize = PAGE_SIZE): string {
  if (count === 0) return "0";
  const first = page * pageSize + 1;
  return `${first}–${first + count - 1}`;
}

/** Status badge for a list row; null for normal completed meetings. */
export function listStateBadge(
  item: MeetingListItem,
  liveMeetingId: string | null,
): { label: string; tone: Tone } | null {
  if (item.state === "completed") return null;
  if (item.id === liveMeetingId) {
    return item.state === "summarizing"
      ? { label: "Summarizing", tone: "accent" }
      : { label: "In progress", tone: "ok" };
  }
  return { label: "Incomplete", tone: "warn" };
}

export function countLabel(n: number, singular: string, plural = `${singular}s`): string {
  return `${n} ${n === 1 ? singular : plural}`;
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export const EXPORT_FORMATS: ReadonlyArray<{
  value: ExportFormat;
  label: string;
  description: string;
}> = [
  { value: "markdown", label: "Markdown", description: "Readable document with headings (.md)" },
  { value: "json", label: "JSON", description: "Structured data for other tools (.json)" },
  { value: "text", label: "Plain text", description: "Simple text without formatting (.txt)" },
];

export const EXPORT_PARTS: ReadonlyArray<{
  key: keyof ExportParts;
  label: string;
  description: string;
}> = [
  {
    key: "summary",
    label: "Summary",
    description: "Overview, decisions, action items and open questions",
  },
  {
    key: "transcript",
    label: "Transcript",
    description: "Everything that was said, with timestamps",
  },
  {
    key: "questions",
    label: "Questions & answers",
    description: "Detected questions, spoken answers and AI suggestions",
  },
  { key: "followups", label: "Follow-ups", description: "Suggested follow-up questions" },
];

export const DEFAULT_EXPORT_PARTS: ExportParts = {
  summary: true,
  transcript: true,
  questions: true,
  followups: true,
};

export function buildExportRequest(
  meetingId: string,
  format: ExportFormat,
  parts: ExportParts,
): { request: ExportRequest | null; error: string | null } {
  if (!meetingId) return { request: null, error: "No meeting selected." };
  if (!parts.summary && !parts.transcript && !parts.questions && !parts.followups) {
    return { request: null, error: "Choose at least one part to export." };
  }
  return { request: { meetingId, format, parts: { ...parts } }, error: null };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// Questions & answers
// ---------------------------------------------------------------------------

export interface QaItem {
  question: DetectedQuestion;
  answers: AnswerRecord[];
  suggestions: Suggestion[];
}

export interface QaView {
  items: QaItem[];
  /** AI suggestions not tied to a question (e.g. Ask OpenMeet). */
  otherSuggestions: Suggestion[];
  /** Spoken answers whose question is unknown. */
  otherAnswers: AnswerRecord[];
}

const byCreated = <T extends { createdAt: number }>(a: T, b: T) => a.createdAt - b.createdAt;

/**
 * Pairs every detected question with the spoken answers and the AI
 * suggestions for it. Questions extended by a later detection (`supersedes`)
 * are folded into the newer one; refinements follow their parent suggestion.
 */
export function buildQa(record: MeetingRecord): QaView {
  const supersededBy = new Map<string, string>();
  for (const q of record.questions) if (q.supersedes) supersededBy.set(q.supersedes, q.id);
  const resolveQuestion = (id: string | null): string | null => {
    let cur = id;
    const seen = new Set<string>();
    while (cur && supersededBy.has(cur) && !seen.has(cur)) {
      seen.add(cur);
      cur = supersededBy.get(cur) ?? cur;
    }
    return cur;
  };

  const questions = record.questions
    .filter((q) => !supersededBy.has(q.id))
    .slice()
    .sort((a, b) => a.detectedAt - b.detectedAt);
  const items = new Map<string, QaItem>(
    questions.map((q) => [q.id, { question: q, answers: [], suggestions: [] }]),
  );

  const suggestionById = new Map(record.suggestions.map((s) => [s.id, s]));
  const suggestionQuestion = (s: Suggestion): string | null => {
    let cur: Suggestion | undefined = s;
    const seen = new Set<string>();
    while (cur && !seen.has(cur.id)) {
      if (cur.questionId) return resolveQuestion(cur.questionId);
      seen.add(cur.id);
      cur = cur.parentId ? suggestionById.get(cur.parentId) : undefined;
    }
    return null;
  };

  const otherSuggestions: Suggestion[] = [];
  for (const s of record.suggestions) {
    if (s.kind === "followups") continue;
    const qid = suggestionQuestion(s);
    const item = qid ? items.get(qid) : undefined;
    if (item) item.suggestions.push(s);
    else otherSuggestions.push(s);
  }
  const otherAnswers: AnswerRecord[] = [];
  for (const a of record.answers) {
    const qid = resolveQuestion(a.questionId);
    const item = qid ? items.get(qid) : undefined;
    if (item) item.answers.push(a);
    else otherAnswers.push(a);
  }
  for (const it of items.values()) {
    it.answers.sort(byCreated);
    it.suggestions.sort(byCreated);
  }
  return {
    items: questions.map((q) => items.get(q.id)).filter((x): x is QaItem => x !== undefined),
    otherSuggestions: otherSuggestions.sort(byCreated),
    otherAnswers: otherAnswers.sort(byCreated),
  };
}

/** Heading for a spoken answer: the user's own answer or the other side's. */
export function spokenAnswerLabel(role: SpeakerRole, mode: Mode): string {
  if (role === "user") return "Your answer";
  if (role === "remote") return mode === "candidate" ? "Interviewer’s reply" : "Candidate’s answer";
  return "Answer";
}

const SUGGESTION_KIND_LABELS: Record<SuggestionKind, string> = {
  answer: "Suggested answer",
  followups: "Follow-ups",
  ask: "Ask OpenMeet",
  refinement: "Refined answer",
  clarification: "Clarifying questions",
};

export function suggestionKindLabel(kind: SuggestionKind): string {
  return SUGGESTION_KIND_LABELS[kind];
}

const STYLE_LABELS: Record<AnswerStyle, string> = {
  concise: "Concise",
  detailed: "Detailed",
  technical: "Technical",
  simple: "Simple",
  star: "STAR",
  example_driven: "Example-driven",
  step_by_step: "Step-by-step",
};

export function answerStyleLabel(style: AnswerStyle): string {
  return STYLE_LABELS[style];
}

// ---------------------------------------------------------------------------
// Follow-ups
// ---------------------------------------------------------------------------

export interface FollowupGroup {
  suggestion: Suggestion;
  items: FollowUpQuestion[];
}

/** Follow-up suggestions in order, each follow-up listed once. */
export function followupGroups(record: MeetingRecord): FollowupGroup[] {
  const seen = new Set<string>();
  const groups: FollowupGroup[] = [];
  for (const s of record.suggestions.filter((x) => x.kind === "followups").sort(byCreated)) {
    const items = s.followups.filter((f) => {
      if (seen.has(f.id)) return false;
      seen.add(f.id);
      return true;
    });
    if (items.length > 0) groups.push({ suggestion: s, items });
  }
  return groups;
}

export function priorityBadge(p: Priority): { label: string; tone: Tone } | null {
  switch (p) {
    case "high":
      return { label: "High priority", tone: "accent" };
    case "optional":
      return { label: "Optional", tone: "neutral" };
    case "normal":
      return null;
  }
}

// ---------------------------------------------------------------------------
// Topics, claims, observations
// ---------------------------------------------------------------------------

const CLAIM_KINDS: ReadonlyArray<{ kind: ClaimKind; label: string }> = [
  { kind: "claim", label: "Claims" },
  { kind: "metric", label: "Metrics" },
  { kind: "ownership", label: "Ownership" },
  { kind: "decision", label: "Decisions" },
  { kind: "tradeoff", label: "Trade-offs" },
  { kind: "implementation", label: "Implementation details" },
  { kind: "technology", label: "Technologies" },
  { kind: "risk", label: "Risks" },
  { kind: "unknown", label: "Other statements" },
];

export function groupClaims(
  claims: ReadonlyArray<Claim>,
): Array<{ kind: ClaimKind; label: string; claims: Claim[] }> {
  return CLAIM_KINDS.map(({ kind, label }) => ({
    kind,
    label,
    claims: claims.filter((c) => c.kind === kind).sort(byCreated),
  })).filter((g) => g.claims.length > 0);
}

export function topicStatusLabel(s: TopicStatus): string {
  switch (s) {
    case "covered":
      return "Covered";
    case "current":
      return "Discussed last";
    case "pending":
      return "Not covered";
  }
}

export function sortedObservations(obs: ReadonlyArray<ScreenObservation>): ScreenObservation[] {
  return obs.slice().sort((a, b) => a.timestamp - b.timestamp);
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

export interface SummarySection {
  key: string;
  title: string;
  items: string[];
}

/** Non-empty list sections of a summary, in reading order (interviewer extras last). */
export function summarySections(s: MeetingSummary): SummarySection[] {
  const all: SummarySection[] = [
    { key: "keyTopics", title: "Key topics", items: s.keyTopics },
    { key: "questionsAsked", title: "Questions asked", items: s.questionsAsked },
    { key: "answers", title: "Answers", items: s.answers },
    { key: "decisions", title: "Decisions", items: s.decisions },
    { key: "actionItems", title: "Action items", items: s.actionItems },
    { key: "openQuestions", title: "Open questions", items: s.openQuestions },
    { key: "technicalTopics", title: "Technical topics", items: s.technicalTopics },
    { key: "followUpSuggestions", title: "Follow-up suggestions", items: s.followUpSuggestions },
  ];
  if (s.interviewer) {
    all.push(
      {
        key: "questionCoverage",
        title: "Question coverage",
        items: s.interviewer.questionCoverage,
      },
      {
        key: "clarificationAreas",
        title: "Areas to clarify",
        items: s.interviewer.clarificationAreas,
      },
      { key: "claimsToVerify", title: "Claims to verify", items: s.interviewer.claimsToVerify },
      {
        key: "unexploredTopics",
        title: "Unexplored topics",
        items: s.interviewer.unexploredTopics,
      },
    );
  }
  return all
    .map((x) => ({ ...x, items: x.items.map((i) => i.trim()).filter(Boolean) }))
    .filter((x) => x.items.length > 0);
}

export function generatedByLabel(generatedBy: string): string {
  if (!generatedBy || generatedBy === "extractive") return "Extracted from the transcript (no AI)";
  return `Generated by ${generatedBy}`;
}
