// Demo data for the screenshot harness. The meeting content mirrors the
// built-in demo scenarios (crates/openmeet-intel/src/demo.rs), so the images
// show what demo mode produces.

import type {
  Claim,
  DataFlowNode,
  DataFlowReport,
  DisplayInfo,
  IntegrityServiceView,
  DetectedQuestion,
  EngineSnapshot,
  FollowUpQuestion,
  Meeting,
  MeetingListItem,
  MeetingSummary,
  Settings,
  Topic,
  TranscriptSegment,
  WorkplaceBoard,
} from "@/bindings";
import defaults from "./default-settings.json";
import { quietFlag, raisedFlag } from "@/lib/integrity.fixtures";

/** The demo meeting started a few minutes before the screenshot. */
const T0 = Date.now() - 6 * 60_000 - 42_000;

export function settings(
  mode: "candidate" | "interviewer" | "workplace",
  overlayMaxHeight = 560,
): Settings {
  const s = structuredClone(defaults) as unknown as Settings;
  s.mode = mode;
  s.general.onboardingComplete = true;
  s.general.developerMode = false;
  s.audio.systemAudioEnabled = true;
  s.overlay.reducedMotion = true;
  s.overlay.maxHeight = overlayMaxHeight;
  s.profile.plannedTopics =
    "System design\nReliability & failure handling\nObservability\nSecurity\nTesting & quality";
  return s;
}

function meeting(mode: "candidate" | "interviewer" | "workplace", title: string): Meeting {
  return {
    id: "demo-meeting",
    title,
    mode,
    state: "active",
    startedAt: T0,
    endedAt: null,
    participants: [],
    persisted: true,
  };
}

let segSeq = 0;
function seg(role: "user" | "remote", text: string, atSec: number): TranscriptSegment {
  segSeq += 1;
  return {
    id: `s${segSeq}`,
    meetingId: "demo-meeting",
    utteranceId: `u${segSeq}`,
    speakerId: null,
    speakerRole: role,
    source: role === "user" ? "microphone" : "system",
    startTime: atSec * 1000,
    endTime: atSec * 1000 + 4000,
    text,
    isFinal: true,
    confidence: 0.94,
  };
}

const BASE_SNAPSHOT: Omit<EngineSnapshot, "meeting" | "mode"> = {
  state: "active",
  pausedReason: null,
  style: "concise",
  transcript: [],
  questions: [],
  active: null,
  followups: [],
  followupsRequestId: null,
  pinned: [],
  topics: [],
  suggestedNextTopic: null,
  components: [
    { component: "microphone", status: "streaming", message: null, updatedAt: T0 },
    { component: "system_audio", status: "streaming", message: null, updatedAt: T0 },
    { component: "transcription", status: "active", message: "Local AI · base.en", updatedAt: T0 },
    { component: "llm", status: "active", message: "Ollama · llama3.2", updatedAt: T0 },
    { component: "capture_privacy", status: "active", message: null, updatedAt: T0 },
  ],
  lastObservation: null,
  lastSummary: null,
  demoRunning: true,
  processingStopped: false,
  integrityFlags: [],
};

// --- Candidate: URL shortener ------------------------------------------------

export const URL_SHORTENER_ANSWER = `[[STRUCTURE]]
1. Requirements
2. API design
3. Short-code generation
4. Storage
5. Caching
6. Scaling
[[ANSWER]]
I'd start by clarifying the expected scale — new links per day versus redirects per second, since reads usually dominate by 100 to 1. Then a tiny API: POST to create a short link, GET on the code to redirect with a 301 or 302. For codes I'd encode a unique ID in base62, store code-to-URL mappings in a key-value store, and put a cache in front because popular links get hit constantly.
[[KEY_POINTS]]
- Reads dominate: optimize the redirect path first
- Base62 of a unique ID avoids collisions by construction
- Cache hot links close to users; the store stays simple
- Plan for abuse: rate limits and malicious URL checks
[[EXAMPLE]]
At 100 million new links a month, 7 base62 characters give about 3.5 trillion codes — decades of headroom.
[[FOLLOW_UP]]
What read and write volumes should I design for?
[[BASIS]]
Suggested framing: standard URL-shortener design; the numbers are illustrative assumptions.
`;

export function candidateSnapshot(): EngineSnapshot {
  return {
    ...BASE_SNAPSHOT,
    meeting: meeting("candidate", "Demo: System design interview"),
    mode: "candidate",
    transcript: [
      seg(
        "remote",
        "Hi, thanks for joining! To get started, could you give me a quick overview of your background?",
        2,
      ),
      seg(
        "user",
        "Sure. I'm a backend engineer, and for the last few years I've mostly worked on APIs and data pipelines.",
        9,
      ),
      seg("remote", "Great. Let's do a design exercise. How would you design a URL shortener?", 21),
    ],
    active: {
      requestId: "req-url",
      kind: "candidate_answer",
      slot: "primary",
      state: "complete",
      questionText: "How would you design a URL shortener?",
      category: "system_design",
      style: "concise",
      text: URL_SHORTENER_ANSWER,
      error: null,
    },
  };
}

// --- Interviewer: rate limiter -------------------------------------------------

function followup(
  id: string,
  question: string,
  category: FollowUpQuestion["category"],
  reason: string,
  priority: FollowUpQuestion["priority"],
): FollowUpQuestion {
  return {
    id,
    requestId: "req-fu",
    question,
    category,
    reason,
    priority,
    trigger: null,
    createdAt: T0,
  };
}

export const RATE_LIMITER_FOLLOWUPS: FollowUpQuestion[] = [
  followup(
    "f1",
    "What consistency guarantees do you need when several gateway nodes update the same bucket?",
    "tradeoff",
    "Atomic Lua updates were described, but not consistency across nodes or replicas.",
    "high",
  ),
  followup(
    "f2",
    "What happens when Redis becomes unavailable?",
    "technical",
    "Redis sits on every request path, but failure handling wasn't mentioned.",
    "high",
  ),
  followup(
    "f3",
    "How would this behave across multiple regions?",
    "system_design",
    "A single Redis implies one region; cross-region latency and global limits were not covered.",
    "normal",
  ),
];

function topic(id: string, name: string, status: Topic["status"]): Topic {
  return {
    id,
    meetingId: "demo-meeting",
    name,
    status,
    mentions: status === "pending" ? 0 : 2,
    firstMentionedAt: null,
    lastMentionedAt: null,
    planned: true,
    summary: null,
  };
}

const RATE_LIMITER_QUESTION: DetectedQuestion = {
  id: "q1",
  meetingId: "demo-meeting",
  segmentIds: [],
  isQuestion: true,
  confidence: 0.93,
  question: "How would you design a rate limiter for a public API?",
  parts: [],
  category: "system_design",
  speakerId: null,
  speakerRole: "user",
  requiresResponse: false,
  method: "heuristic",
  supersedes: null,
  detectedAt: T0 + 3_000,
};

export function interviewerSnapshot(): EngineSnapshot {
  return {
    ...BASE_SNAPSHOT,
    meeting: meeting("interviewer", "Demo: Rate limiter deep-dive"),
    mode: "interviewer",
    transcript: [
      seg(
        "user",
        "Thanks for taking the time today. Let's start with a design question: how would you design a rate limiter for a public API?",
        3,
      ),
      seg(
        "remote",
        "I'd probably use Redis. Each API key gets a token bucket stored in Redis, and every request runs a small Lua script that refills tokens based on the elapsed time and takes one atomically. If the bucket is empty we return a 429 with a Retry-After header. I'd run the check in the API gateway so every service gets it for free.",
        11,
      ),
    ],
    questions: [RATE_LIMITER_QUESTION],
    followups: RATE_LIMITER_FOLLOWUPS,
    followupsRequestId: "req-fu",
    topics: [
      topic("system-design", "System design", "current"),
      topic("reliability", "Reliability & failure handling", "pending"),
      topic("observability", "Observability", "pending"),
      topic("security", "Security", "pending"),
      topic("testing", "Testing & quality", "pending"),
    ],
    suggestedNextTopic: "Reliability & failure handling",
  };
}

// --- Main window -----------------------------------------------------------------

export const DATA_FLOW_LOCAL: DataFlowReport = {
  nodes: [
    {
      component: "microphone",
      label: "Microphone",
      location: "local",
      provider: "WASAPI",
      endpoint: null,
    },
    {
      component: "system_audio",
      label: "System audio",
      location: "local",
      provider: "WASAPI loopback",
      endpoint: null,
    },
    {
      component: "transcription",
      label: "Transcription",
      location: "local",
      provider: "Local AI (Whisper base.en)",
      endpoint: "127.0.0.1:8765",
    },
    {
      component: "llm",
      label: "AI provider",
      location: "local",
      provider: "Ollama (llama3.2)",
      endpoint: "127.0.0.1:11434",
    },
    { component: "screen", label: "Screen", location: "disabled", provider: "Off", endpoint: null },
    {
      component: "vision",
      label: "Vision",
      location: "local",
      provider: "Windows OCR",
      endpoint: null,
    },
    {
      component: "embeddings",
      label: "Embeddings",
      location: "local",
      provider: "Built-in",
      endpoint: null,
    },
    {
      component: "database",
      label: "Storage",
      location: "local",
      provider: "SQLite",
      endpoint: null,
    },
  ],
  localOnlyMode: false,
  allLocal: true,
  statement: "No meeting data leaves this device.",
};

export const MEETINGS: MeetingListItem[] = [];

// --- Answer integrity (the integrity demo scenario) -----------------------------

/** Interviewer meeting after the integrity demo's fourth answer. */
export function integritySnapshot(): EngineSnapshot {
  const snap = interviewerSnapshot();
  return {
    ...snap,
    integrityFlags: [
      raisedFlag({ meetingId: "demo-meeting", answerStartMs: 310_000, answerEndMs: 352_000 }),
      quietFlag({ meetingId: "demo-meeting", answerStartMs: 64_000, answerEndMs: 75_000 }),
    ],
  };
}

export const DISPLAYS: DisplayInfo[] = [
  {
    key: "DISPLAY1@1920x1080",
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
  {
    key: "DISPLAY2@2560x1440",
    name: "DISPLAY2",
    x: 1920,
    y: 0,
    width: 2560,
    height: 1440,
    workArea: { x: 1920, y: 0, width: 2560, height: 1400 },
    scaleFactor: 1,
    isPrimary: false,
    orientation: "landscape",
  },
];

export const INTEGRITY_STATUS: IntegrityServiceView = {
  reachable: true,
  aiTextReady: true,
  readingReady: true,
  embeddingModel: "BAAI/bge-small-en-v1.5",
  requiredEmbeddingModel: "BAAI/bge-small-en-v1.5",
  echoCalibrated: false,
  echoReferenceModel: "llama3.2",
  deliveryCalibrated: false,
  message: "Ready.",
};

const INTEGRITY_NODES: DataFlowNode[] = [
  {
    component: "integrity",
    label: "Integrity: AI-text and reading check",
    location: "local",
    provider: "Local AI (integrity check) · AI-text + reading models",
    endpoint: "127.0.0.1:8765",
  },
  {
    component: "integrity",
    label: "Integrity: candidate video region (gaze)",
    location: "local",
    provider: "Captured by Local AI; frames are turned into gaze and dropped",
    endpoint: "127.0.0.1:8765",
  },
  {
    component: "integrity",
    label: "Integrity: reference answers (AI provider)",
    location: "local",
    provider: "Ollama · llama3.2",
    endpoint: "127.0.0.1:11434",
  },
  {
    component: "integrity",
    label: "Integrity: answer embeddings",
    location: "local",
    provider: "Local AI (fastembed) · default",
    endpoint: "127.0.0.1:8765",
  },
];

export const DATA_FLOW_INTEGRITY: DataFlowReport = {
  ...DATA_FLOW_LOCAL,
  nodes: [...DATA_FLOW_LOCAL.nodes, ...INTEGRITY_NODES],
};

// --- Workplace: sprint planning ------------------------------------------------

function boardClaim(id: string, text: string): Claim {
  return {
    id,
    meetingId: "demo-meeting",
    text,
    kind: "decision",
    segmentId: null,
    speakerRole: "remote",
    source: "heuristic",
    createdAt: T0,
  } as Claim;
}

/** The live board of the sprint planning demo, as the engine would send it. */
export const WORKPLACE_BOARD: WorkplaceBoard = {
  decisions: [boardClaim("d1", "We decided to move the export job to Kafka for this sprint.")],
  actionItems: [
    {
      id: "a1",
      meetingId: "demo-meeting",
      text: "Priya will update the runbook.",
      owner: "Priya",
      due: "by Friday",
      sourceSegmentId: null,
      createdAt: T0,
    },
    {
      id: "a2",
      meetingId: "demo-meeting",
      text: "Someone should look into the alerting thresholds.",
      owner: null,
      due: null,
      sourceSegmentId: null,
      createdAt: T0,
    },
  ],
  openQuestions: [
    {
      id: "q1",
      question: "Can you share the rollout plan by Monday?",
      speakerRole: "remote",
      addressedToYou: true,
      askedAtMs: 60_000,
      answered: false,
      answer: null,
      deferred: false,
    },
    {
      id: "q2",
      question: "Do we know if the staging data is ready?",
      speakerRole: "remote",
      addressedToYou: false,
      askedAtMs: 95_000,
      answered: false,
      answer: null,
      deferred: true,
    },
  ],
  clarifications: [
    {
      id: "c1",
      targetId: "a2",
      targetKind: "action_item",
      targetText: "Someone should look into the alerting thresholds.",
      issues: ["no_owner", "no_due_date", "unclear_scope"],
      question: "Who will take “look into the alerting thresholds”, and by when?",
      createdAt: T0,
    },
  ],
};

export function workplaceSnapshot(): EngineSnapshot {
  return {
    ...BASE_SNAPSHOT,
    meeting: meeting("workplace", "Demo: Sprint planning"),
    mode: "workplace",
    transcript: [
      seg(
        "user",
        "Morning, everyone. Let's run sprint planning, starting with the billing export.",
        2,
      ),
      seg("remote", "We decided to move the export job to Kafka for this sprint.", 14),
      seg("remote", "Priya will update the runbook by Friday.", 26),
      seg("remote", "Can you share the rollout plan by Monday?", 60),
      seg("user", "Yes, I'll share the rollout plan on Monday.", 74),
    ],
    workplace: WORKPLACE_BOARD,
    active: {
      requestId: "req-catch-up",
      kind: "catch_up",
      slot: "primary",
      state: "complete",
      questionText: null,
      category: null,
      style: "concise",
      text: "[[ANSWER]]\nThe team agreed to move the export job to Kafka this sprint, and Priya will update the runbook by Friday.\n[[KEY_POINTS]]\n- Decided: move the export job to Kafka\n- Action: Priya updates the runbook by Friday",
      error: null,
    },
  };
}

/** A finished sprint planning meeting with its minutes. */
export function minutesSnapshot(): EngineSnapshot {
  const summary: MeetingSummary = {
    meetingId: "demo-meeting",
    overview:
      "The team moved the billing export to Kafka for this sprint, agreed on the schema freeze, and left the alerting thresholds to be tuned.",
    keyTopics: ["Billing export", "Kafka", "Alerting"],
    questionsAsked: ["Can you share the rollout plan by Monday?"],
    answers: ["Yes, I’ll share the rollout plan on Monday."],
    decisions: [
      "Move the export job to Kafka for this sprint.",
      "Freeze the schema for this sprint.",
    ],
    actionItems: [
      "Update the runbook — Priya, by Friday",
      "Tune the alert thresholds — owner to be confirmed",
    ],
    openQuestions: ["Is the staging data ready? (deferred)"],
    technicalTopics: [],
    followUpSuggestions: [],
    interviewer: null,
    workplace: {
      actionItems: [
        { task: "Update the runbook", owner: "Priya", due: "by Friday" },
        { task: "Tune the alerting thresholds", owner: null, due: null },
      ],
      followUpMessage:
        "Hi all,\n\nThanks for the sprint planning. Here is what we agreed:\n- Move the export job to Kafka this sprint.\n- Freeze the schema this sprint.\n\nAction items:\n- Update the runbook: Priya, by Friday\n- Tune the alerting thresholds: owner to be confirmed\n\nLet me know if I missed anything.",
    },
    markdown: "",
    generatedAt: T0,
    generatedBy: "extractive",
  };
  return {
    ...BASE_SNAPSHOT,
    meeting: {
      ...meeting("workplace", "Demo: Sprint planning"),
      state: "completed",
      endedAt: T0 + 1_800_000,
    },
    state: "completed",
    mode: "workplace",
    transcript: workplaceSnapshot().transcript,
    workplace: WORKPLACE_BOARD,
    lastSummary: summary,
  };
}
