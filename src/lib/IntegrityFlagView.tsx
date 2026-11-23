// One answer-integrity flag with its evidence. Used by the overlay card, the
// live view and history (both windows share the design tokens).

import { useState } from "react";
import type { IntegrityFlag, IntegrityLevel, IntegritySignal } from "@/bindings";
import {
  FLAG_NOTE,
  LEVEL_LABELS,
  SIGNAL_LABELS,
  answerRangeLabel,
  isElevated,
  levelTone,
  orderedSignals,
  type FlagTone,
} from "./integrity";

const TAG_TONES: Record<FlagTone, string> = {
  neutral: "border-border text-fg-soft",
  ok: "border-transparent bg-[color-mix(in_srgb,var(--om-ok)_14%,transparent)] text-ok",
  warn: "border-transparent bg-[color-mix(in_srgb,var(--om-warn)_16%,transparent)] text-warn",
};

function Tag({ tone, children, title }: { tone: FlagTone; children: string; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center rounded-full border px-[0.45rem] py-[0.05rem] text-[0.68rem] leading-tight font-medium whitespace-nowrap ${TAG_TONES[tone]}`}
    >
      {children}
    </span>
  );
}

export function LevelTag({ level }: { level: IntegrityLevel }) {
  return <Tag tone={levelTone(level)}>{LEVEL_LABELS[level]}</Tag>;
}

function SignalRow({ s }: { s: IntegritySignal }) {
  return (
    <li className="space-y-0.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[0.8rem] font-medium text-fg">{SIGNAL_LABELS[s.name]}</span>
        <LevelTag level={s.level} />
        {!s.calibrated && (
          <Tag
            tone="neutral"
            title="Thresholds not yet measured on labelled data; this signal counts at most as possible."
          >
            uncalibrated
          </Tag>
        )}
      </div>
      <ul className="space-y-0.5 pl-0.5">
        {s.evidence.map((line, i) => (
          <li key={i} className="text-[0.76rem] leading-snug text-fg-soft">
            {line}
          </li>
        ))}
      </ul>
    </li>
  );
}

export interface IntegrityFlagViewProps {
  flag: IntegrityFlag;
  /** Overlay: show only the raised signals until expanded. */
  compact?: boolean;
}

export function IntegrityFlagView({ flag, compact }: IntegrityFlagViewProps) {
  const signals = orderedSignals(flag);
  const raised = signals.filter((s) => isElevated(s.level));
  const [showAll, setShowAll] = useState(!compact);
  const shown = showAll ? signals : raised;
  const hidden = signals.length - shown.length;
  return (
    <div className="space-y-1.5" data-testid="integrity-flag">
      <div className="flex items-start gap-2">
        <LevelTag level={flag.level} />
        <p
          className={`text-[0.86rem] leading-snug ${isElevated(flag.level) ? "font-medium text-fg" : "text-fg-soft"}`}
        >
          {flag.headline}
        </p>
      </div>
      <p className="text-[0.74rem] text-muted">
        {answerRangeLabel(flag)}
        {flag.questionText ? ` · “${flag.questionText}”` : ""}
      </p>
      {shown.length > 0 && (
        <ul className="space-y-1.5">
          {shown.map((s) => (
            <SignalRow key={s.name} s={s} />
          ))}
        </ul>
      )}
      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="text-[0.74rem] text-accent underline-offset-2 hover:underline"
        >
          {raised.length > 0 ? `Show all signals (${hidden} more)` : "Show signals"}
        </button>
      )}
      {isElevated(flag.level) && <p className="text-[0.72rem] text-muted">{FLAG_NOTE}</p>}
    </div>
  );
}
