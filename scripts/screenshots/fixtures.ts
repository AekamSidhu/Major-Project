// Demo data for the screenshot harness. The meeting content mirrors the
// built-in demo scenarios (crates/openmeet-intel/src/demo.rs), so the images
// show what demo mode produces.

import type {
  DataFlowReport,
  DetectedQuestion,
  EngineSnapshot,
  FollowUpQuestion,
  Meeting,
  MeetingListItem,
  Settings,
  Topic,
  TranscriptSegment,
} from "@/bindings";
import defaults from "./default-settings.json";

/** The demo meeting started a few minutes before the screenshot. */
const T0 = Date.now() - 6 * 60_000 - 42_000;

export function settings(mode: "candidate" | "interviewer", overlayMaxHeight = 560): Settings {
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

function meeting(mode: "candidate" | "interviewer", title: string): Meeting {
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
      seg("remote", "Hi, thanks for joining! To get started, could you give me a quick overview of your background?", 2),
      seg("user", "Sure. I'm a backend engineer, and for the last few years I've mostly worked on APIs and data pipelines.", 9),
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

function followup(id: string, question: string, category: FollowUpQuestion["category"], reason: string, priority: FollowUpQuestion["priority"]): FollowUpQuestion {
  return { id, requestId: "req-fu", question, category, reason, priority, trigger: null, createdAt: T0 };
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
      seg("user", "Thanks for taking the time today. Let's start with a design question: how would you design a rate limiter for a public API?", 3),
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
    { component: "microphone", label: "Microphone", location: "local", provider: "WASAPI", endpoint: null },
    { component: "system_audio", label: "System audio", location: "local", provider: "WASAPI loopback", endpoint: null },
    { component: "transcription", label: "Transcription", location: "local", provider: "Local AI (Whisper base.en)", endpoint: "127.0.0.1:8765" },
    { component: "llm", label: "AI provider", location: "local", provider: "Ollama (llama3.2)", endpoint: "127.0.0.1:11434" },
    { component: "screen", label: "Screen", location: "disabled", provider: "Off", endpoint: null },
    { component: "vision", label: "Vision", location: "local", provider: "Windows OCR", endpoint: null },
    { component: "embeddings", label: "Embeddings", location: "local", provider: "Built-in", endpoint: null },
    { component: "database", label: "Storage", location: "local", provider: "SQLite", endpoint: null },
  ],
  localOnlyMode: false,
  allLocal: true,
  statement: "No meeting data leaves this device.",
};

export const MEETINGS: MeetingListItem[] = [];
