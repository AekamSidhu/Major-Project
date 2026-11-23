// Answer-integrity flags for tests and the screenshot harness (synthetic).

import type { IntegrityFlag } from "@/bindings";

/** A raised flag: the candidate read an AI-written answer (demo dialogue). */
export function raisedFlag(patch: Partial<IntegrityFlag> = {}): IntegrityFlag {
  return {
    id: "answer-4",
    meetingId: "m1",
    questionId: "q4",
    questionText: "How does consistent hashing help when you add or remove cache nodes?",
    answerStartMs: 1_434_000,
    answerEndMs: 1_500_000,
    answerWords: 31,
    level: "high",
    elevated: ["ai_text", "delivery", "echo_similarity", "reading"],
    headline: "Several signals point to AI assistance — worth a follow-up",
    signals: [
      {
        name: "ai_text",
        level: "high",
        calibrated: true,
        score: 0.91,
        evidence: ["AI-written text score 0.91 (high)"],
        note: null,
        delivery: null,
      },
      {
        name: "echo_similarity",
        level: "high",
        calibrated: false,
        score: 0.88,
        evidence: ["Close to a local LLM's answer: similarity 0.88 (uncalibrated)"],
        note: null,
        delivery: null,
      },
      {
        name: "delivery",
        level: "high",
        calibrated: false,
        score: 3.4,
        evidence: [
          "Delivery index 3.4 (high, uncalibrated)",
          "Started 3.1 s later than usual",
          "Spoke faster than usual (212 vs 150 words/min)",
          "Fewer filler words than usual (0.0 vs 11.8 per 100 words)",
        ],
        note: null,
        delivery: {
          granularity: "word",
          wordCount: 31,
          latencyS: 4.3,
          baselineLatencyS: 1.2,
          rateWpm: 212,
          baselineRateWpm: 150,
          pausePerMin: 0,
          baselinePausePerMin: 0,
          fillerPer100w: 0,
          baselineFillerPer100w: 11.8,
          baselineAnswers: 3,
        },
      },
      {
        name: "reading",
        level: "high",
        calibrated: true,
        score: 0.87,
        evidence: ["Eye movements look like reading: score 0.87 (high)"],
        note: null,
        delivery: null,
      },
    ],
    revision: 1,
    specVersion: 1,
    createdAt: 1_760_000_000_000,
    updatedAt: 1_760_000_000_500,
    ...patch,
  };
}

/** A quiet flag: nothing unusual, one signal not scored yet. */
export function quietFlag(patch: Partial<IntegrityFlag> = {}): IntegrityFlag {
  return {
    id: "answer-1",
    meetingId: "m1",
    questionId: "q1",
    questionText: "Can you tell me about a project you're proud of?",
    answerStartMs: 64_000,
    answerEndMs: 75_000,
    answerWords: 36,
    level: "low",
    elevated: [],
    headline: "Nothing unusual in this answer",
    signals: [
      {
        name: "ai_text",
        level: "low",
        calibrated: true,
        score: 0.12,
        evidence: ["AI-written text score 0.12 (low)"],
        note: null,
        delivery: null,
      },
      {
        name: "delivery",
        level: "baseline",
        calibrated: false,
        score: null,
        evidence: ["Learning this candidate's usual delivery (answer 1 of 3)"],
        note: "Learning this candidate's usual delivery (answer 1 of 3)",
        delivery: null,
      },
    ],
    revision: 1,
    specVersion: 1,
    createdAt: 1_760_000_000_000,
    updatedAt: 1_760_000_000_100,
    ...patch,
  };
}
