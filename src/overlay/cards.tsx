// Content cards of the overlay: the primary answer card (candidate answers,
// Ask OpenMeet, refinements, clarifications), interviewer follow-ups and the
// topic tracker.

import { useMemo, useState } from "react";
import type { AnswerStyle, FollowUpQuestion, Refinement, Topic } from "@/bindings";
import { answerToPlainText, parseAnswer, type ParsedAnswer } from "@/lib/answer";
import { categoryLabel } from "@/lib/format";
import {
  chooseStyle,
  copyText,
  dismiss,
  generateAnswer,
  generateFollowups,
  openMainWindow,
  refine,
  retry,
  setPinned,
} from "./actions";
import { CheckIcon, CopyIcon, PinIcon, RefreshIcon } from "./icons";
import {
  isRunning,
  isThinking,
  orderTopics,
  type FollowupsRequest,
  type PrimaryRequest,
} from "./state";
import { ActionButton, ChipButton, Dot, IconButton, LinkButton, SectionLabel, Tag, cx } from "./ui";
import { useHotkey, withHotkey } from "./hooks";

const STYLES: ReadonlyArray<{ value: AnswerStyle; label: string }> = [
  { value: "concise", label: "Concise" },
  { value: "detailed", label: "Detailed" },
  { value: "technical", label: "Technical" },
  { value: "simple", label: "Simple" },
  { value: "star", label: "STAR" },
  { value: "example_driven", label: "Example" },
  { value: "step_by_step", label: "Steps" },
];

const REFINEMENTS: ReadonlyArray<{ value: Refinement; label: string; title: string }> = [
  { value: "shorter", label: "Shorter", title: "Make the answer shorter" },
  { value: "longer", label: "Longer", title: "Expand the answer" },
  { value: "more_technical", label: "Technical", title: "More technical depth" },
  { value: "simplify", label: "Simplify", title: "Simpler wording" },
  { value: "give_example", label: "Example", title: "Add a concrete example" },
  {
    value: "show_rationale",
    label: "Reasoning",
    title: "Show the key considerations behind this answer",
  },
];

const REFINEMENT_LABEL: Record<Refinement, string> = {
  regenerate: "Regenerated",
  shorter: "Shorter",
  longer: "Longer",
  more_technical: "More technical",
  simplify: "Simplified",
  give_example: "With example",
  show_rationale: "Reasoning",
};

function useCopied(): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);
  return [
    copied,
    (text: string) => {
      void copyText(text).then((ok) => {
        if (!ok) return;
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      });
    },
  ];
}

function headerFor(p: PrimaryRequest): { label: string; text: string | null } {
  switch (p.kind) {
    case "ask_ai":
      return { label: "Ask OpenMeet", text: p.questionText };
    case "clarification":
      return { label: "Clarify the question", text: p.questionText };
    case "refinement":
      return {
        label: p.refinement ? `${REFINEMENT_LABEL[p.refinement]} answer` : "Refined answer",
        text: p.questionText,
      };
    default:
      return {
        label: p.category ? `Question · ${categoryLabel(p.category)}` : "Question",
        text: p.questionText,
      };
  }
}

function Thinking({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 py-1 text-[0.82rem] text-muted" role="status">
      <Dot tone="accent" pulse />
      {label}
    </div>
  );
}

function Sections({ a, streaming }: { a: ParsedAnswer; streaming: boolean }) {
  const [showExample, setShowExample] = useState(true);
  const caretOn = (key: ParsedAnswer["current"]) => streaming && a.current === key;
  return (
    <div className="space-y-[calc(var(--om-space)*0.65rem)]">
      {a.structure.length > 0 && (
        <div>
          <SectionLabel>Suggested structure</SectionLabel>
          <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-[0.86rem] text-fg-soft">
            {a.structure.map((s, i) => (
              <li
                key={i}
                className={cx(caretOn("structure") && i === a.structure.length - 1 && "om-caret")}
              >
                {s}
              </li>
            ))}
          </ol>
        </div>
      )}
      {(a.answer || (streaming && a.current === "answer")) && (
        <div>
          <SectionLabel>Suggested answer</SectionLabel>
          <p
            className={cx(
              "mt-1 text-[0.92rem] leading-relaxed whitespace-pre-wrap text-fg",
              caretOn("answer") && "om-caret",
            )}
          >
            {a.answer}
          </p>
        </div>
      )}
      {a.clarify.length > 0 && (
        <div>
          <SectionLabel>Clarifying questions</SectionLabel>
          <ul className="mt-1 space-y-0.5 text-[0.86rem] text-fg-soft">
            {a.clarify.map((c, i) => (
              <li key={i} className="flex gap-1.5">
                <span aria-hidden className="text-accent">
                  →
                </span>
                <span
                  className={cx(caretOn("clarify") && i === a.clarify.length - 1 && "om-caret")}
                >
                  {c}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {a.keyPoints.length > 0 && (
        <div>
          <SectionLabel>Key points</SectionLabel>
          <ul className="mt-1 space-y-0.5 text-[0.86rem] text-fg-soft">
            {a.keyPoints.map((k, i) => (
              <li key={i} className="flex gap-1.5">
                <span aria-hidden className="text-muted">
                  •
                </span>
                <span
                  className={cx(caretOn("keyPoints") && i === a.keyPoints.length - 1 && "om-caret")}
                >
                  {k}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {a.example && (
        <div>
          <button
            type="button"
            className="flex items-center gap-1 text-left"
            aria-expanded={showExample}
            onClick={() => setShowExample((v) => !v)}
          >
            <SectionLabel>{showExample ? "Example ▾" : "Example ▸"}</SectionLabel>
          </button>
          {showExample && (
            <p
              className={cx(
                "mt-1 text-[0.84rem] leading-relaxed text-fg-soft",
                caretOn("example") && "om-caret",
              )}
            >
              {a.example}
            </p>
          )}
        </div>
      )}
      {a.followUp && (
        <div className="rounded-md bg-accent-soft px-2 py-1.5 text-[0.82rem] text-fg-soft">
          <span className="font-medium text-accent">You could ask: </span>
          {a.followUp}
        </div>
      )}
      {a.basis && (
        <div>
          <SectionLabel>Why this answer</SectionLabel>
          <p className="mt-1 text-[0.8rem] leading-relaxed text-muted">{a.basis}</p>
        </div>
      )}
      {a.caveat && (
        <div
          role="note"
          className="rounded-md border border-warn/40 px-2 py-1.5 text-[0.8rem] text-warn"
        >
          {a.caveat}
        </div>
      )}
    </div>
  );
}

export function PrimaryCard({
  p,
  style,
  mode,
}: {
  p: PrimaryRequest;
  style: AnswerStyle;
  mode: "candidate" | "interviewer";
}) {
  const parsed = useMemo(() => parseAnswer(p.text), [p.text]);
  const [copied, copy] = useCopied();
  const running = isRunning(p.state);
  const thinking = isThinking(p.state) && !p.text.trim();
  const header = headerFor(p);
  const canRefine = p.state === "complete" && !!p.text.trim();
  const regenerateKey = useHotkey("generate_answer");

  return (
    <section
      aria-label={header.label}
      className="om-fade-in space-y-[calc(var(--om-space)*0.6rem)]"
    >
      <div>
        <SectionLabel>{header.label}</SectionLabel>
        {header.text && (
          <p className="mt-0.5 text-[0.88rem] leading-snug font-medium text-fg">
            {p.kind === "candidate_answer" || p.kind === "refinement"
              ? `“${header.text}”`
              : header.text}
          </p>
        )}
      </div>

      {thinking && <Thinking label="Thinking…" />}
      {!thinking && p.text.trim() && <Sections a={parsed} streaming={p.state === "streaming"} />}

      {p.state === "failed" && (
        <div
          role="alert"
          className="space-y-1.5 rounded-md border border-danger/40 bg-danger-soft px-2 py-1.5"
        >
          <p className="text-[0.82rem] text-danger">{p.error ?? "The request failed."}</p>
          <div className="flex items-center gap-2">
            <ActionButton onClick={() => void retry(p)}>
              <RefreshIcon /> Retry
            </ActionButton>
            <LinkButton onClick={() => void openMainWindow("settings/ai")}>
              Provider settings
            </LinkButton>
          </div>
        </div>
      )}
      {p.state === "cancelled" && !p.text.trim() && (
        <div className="flex items-center gap-2 text-[0.8rem] text-muted">
          Cancelled.
          <LinkButton onClick={() => void retry(p)}>Retry</LinkButton>
        </div>
      )}

      {!thinking && (p.text.trim() || p.state === "complete") && (
        <div className="space-y-1.5">
          <div
            className="flex flex-wrap items-center gap-1"
            role="group"
            aria-label="Answer controls"
          >
            <ChipButton
              disabled={running}
              title={withHotkey("Regenerate", regenerateKey)}
              onClick={() => void refine(p.requestId, "regenerate")}
            >
              <RefreshIcon /> Regenerate
            </ChipButton>
            {REFINEMENTS.map((r) => (
              <ChipButton
                key={r.value}
                disabled={!canRefine}
                title={r.title}
                onClick={() => void refine(p.requestId, r.value)}
              >
                {r.label}
              </ChipButton>
            ))}
          </div>
          <div className="flex items-center justify-between gap-2">
            {mode === "candidate" && (p.kind === "candidate_answer" || p.kind === "refinement") ? (
              <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Answer style">
                {STYLES.map((s) => (
                  <ChipButton
                    key={s.value}
                    role="radio"
                    aria-checked={style === s.value}
                    active={style === s.value}
                    disabled={running}
                    onClick={() => void chooseStyle(s.value, p)}
                  >
                    {s.label}
                  </ChipButton>
                ))}
              </div>
            ) : (
              <span />
            )}
            <div className="flex shrink-0 items-center">
              <IconButton
                label={copied ? "Copied" : "Copy answer"}
                tone={copied ? "accent" : "default"}
                disabled={!p.text.trim()}
                onClick={() => copy(answerToPlainText(parsed) || p.text)}
              >
                {copied ? <CheckIcon /> : <CopyIcon />}
              </IconButton>
              <IconButton
                label={p.pinned ? "Unpin" : "Pin"}
                tone={p.pinned ? "accent" : "default"}
                disabled={p.state !== "complete"}
                onClick={() => void setPinned(p.requestId, !p.pinned)}
              >
                <PinIcon filled={p.pinned} />
              </IconButton>
              <IconButton label="Dismiss" onClick={() => void dismiss(p.requestId)}>
                <span aria-hidden className="text-[0.95rem] leading-none">
                  ×
                </span>
              </IconButton>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

const PRIORITY_TONE: Record<FollowUpQuestion["priority"], "accent" | "neutral" | "warn"> = {
  high: "accent",
  normal: "neutral",
  optional: "neutral",
};

function FollowupItem({ f }: { f: FollowUpQuestion }) {
  const [copied, copy] = useCopied();
  return (
    <li>
      <button
        type="button"
        onClick={() => copy(f.question)}
        title="Copy question"
        className="group w-full rounded-md px-1.5 py-1 text-left transition-colors hover:bg-[color-mix(in_srgb,var(--om-fg)_7%,transparent)]"
      >
        <div className="flex items-start gap-1.5">
          <span
            aria-hidden
            className={cx("mt-px", f.priority === "high" ? "text-accent" : "text-muted")}
          >
            →
          </span>
          <span
            className={cx(
              "text-[0.88rem] leading-snug",
              f.priority === "optional" ? "text-fg-soft" : "text-fg",
            )}
          >
            {f.question}
          </span>
          <span className="ml-auto shrink-0 pl-1 text-[0.7rem] text-accent opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100">
            {copied ? "Copied" : "Copy"}
          </span>
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-1 pl-4">
          <Tag tone={PRIORITY_TONE[f.priority]}>{categoryLabel(f.category)}</Tag>
          {f.priority === "high" && <Tag tone="accent">High</Tag>}
          {f.reason && <span className="text-[0.74rem] leading-snug text-muted">{f.reason}</span>}
        </div>
      </button>
    </li>
  );
}

export function FollowupsCard({ f }: { f: FollowupsRequest }) {
  const thinking = isThinking(f.state) || (f.state === "streaming" && f.questions.length === 0);
  return (
    <section aria-label="Follow-up questions" className="om-fade-in space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <SectionLabel>Follow-up questions</SectionLabel>
        {f.turnPreview && (
          <span className="truncate text-[0.72rem] text-muted" title={f.turnPreview}>
            on “{f.turnPreview}”
          </span>
        )}
      </div>
      {thinking && <Thinking label="Reading the answer…" />}
      {f.questions.length > 0 && (
        <ol className="space-y-0.5" aria-live="polite">
          {f.questions.map((q) => (
            <FollowupItem key={q.id} f={q} />
          ))}
        </ol>
      )}
      {f.state === "complete" && f.done && f.questions.length === 0 && (
        <p className="text-[0.8rem] text-muted">
          No new follow-ups — everything relevant was already asked.
        </p>
      )}
      {f.state === "failed" && (
        <div
          role="alert"
          className="flex flex-wrap items-center gap-2 rounded-md border border-danger/40 bg-danger-soft px-2 py-1.5 text-[0.8rem] text-danger"
        >
          {f.error ?? "Couldn't suggest follow-ups."}
          <LinkButton onClick={() => void generateFollowups()}>Retry</LinkButton>
          <LinkButton onClick={() => void openMainWindow("settings/ai")}>
            Provider settings
          </LinkButton>
        </div>
      )}
    </section>
  );
}

const TOPIC_MARK: Record<Topic["status"], string> = { covered: "✓", current: "→", pending: "○" };

export function TopicsStrip({
  topics,
  suggestedNext,
}: {
  topics: Topic[];
  suggestedNext: string | null;
}) {
  const ordered = orderTopics(topics);
  if (ordered.length === 0 && !suggestedNext) return null;
  return (
    <section aria-label="Topics" className="space-y-1">
      <SectionLabel>Topics</SectionLabel>
      <ul className="flex flex-wrap gap-x-2.5 gap-y-0.5 text-[0.76rem]">
        {ordered.map((t) => (
          <li
            key={t.id}
            className={cx(
              t.status === "current"
                ? "font-medium text-fg"
                : t.status === "covered"
                  ? "text-fg-soft"
                  : "text-muted",
            )}
          >
            <span
              aria-hidden
              className={cx(
                "mr-0.5",
                t.status === "current" && "text-accent",
                t.status === "covered" && "text-ok",
              )}
            >
              {TOPIC_MARK[t.status]}
            </span>
            <span className="sr-only">{t.status}: </span>
            {t.name}
          </li>
        ))}
      </ul>
      {suggestedNext && (
        <p className="text-[0.76rem] text-fg-soft">
          <span className="text-muted">Suggested next area: </span>
          {suggestedNext}
        </p>
      )}
    </section>
  );
}

/** Empty state while listening, with manual triggers (spec §93). */
export function ListeningActions({ mode }: { mode: "candidate" | "interviewer" }) {
  const answerKey = useHotkey("generate_answer");
  const followKey = useHotkey("generate_followups");
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {mode === "candidate" ? (
        <ActionButton
          title={withHotkey("Answer the latest question", answerKey)}
          onClick={() => void generateAnswer(null)}
        >
          Answer latest question
        </ActionButton>
      ) : (
        <ActionButton
          title={withHotkey("Suggest follow-ups for the latest answer", followKey)}
          onClick={() => void generateFollowups()}
        >
          Suggest follow-ups
        </ActionButton>
      )}
    </div>
  );
}
