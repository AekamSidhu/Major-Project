// Tab panels of the meeting detail view in History.

import { useMemo, useState, type ReactNode } from "react";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import type { FollowUpQuestion, MeetingRecord, MeetingSummary, Mode, Suggestion } from "@/bindings";
import { categoryLabel, formatDateTime, formatOffset } from "@/lib/format";
import { api, errorMessage } from "@/lib/ipc";
import { useMain } from "@/main/store";
import { Badge, Button, EmptyState, TextInput, cx } from "@/main/ui";
import { InlineError } from "@/main/sections/home/controls";
import { segmentOffset, speakerLabel } from "@/main/sections/home/helpers";
import { Icon } from "@/main/sections/home/icons";
import { TranscriptBlocks } from "@/main/sections/home/LiveTranscript";
import {
  answerStyleLabel,
  buildQa,
  countLabel,
  followupGroups,
  groupClaims,
  priorityBadge,
  sortedObservations,
  spokenAnswerLabel,
  suggestionKindLabel,
  topicStatusLabel,
} from "./helpers";
import { SuggestionContent } from "./SuggestionContent";
import { SummaryView } from "./SummaryView";

function Panel({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-border bg-surface px-5 py-4">{children}</div>;
}

function SubHeading({ children, count }: { children: ReactNode; count?: number }) {
  return (
    <h3 className="mb-2 flex items-center gap-2 text-[0.72rem] font-semibold tracking-wider text-muted uppercase">
      {children}
      {count !== undefined && <span className="font-normal tabular-nums">{count}</span>}
    </h3>
  );
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

export function SummaryTab({
  record,
  summary,
  onSummary,
}: {
  record: MeetingRecord;
  summary: MeetingSummary | null;
  onSummary: (s: MeetingSummary) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const regenerate = async () => {
    setBusy(true);
    setError(null);
    try {
      onSummary(await api.regenerateSummary(record.meeting.id));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  const button = (
    <Button size="sm" busy={busy} onClick={() => void regenerate()}>
      <Icon name="refresh" className="size-3.5" />
      {summary ? "Regenerate summary" : "Generate summary"}
    </Button>
  );
  return (
    <div>
      {error && <InlineError className="mb-3">Could not generate the summary: {error}</InlineError>}
      {summary ? (
        <Panel>
          <div className="mb-3 flex justify-end">{button}</div>
          <div aria-busy={busy || undefined} className={cx(busy && "opacity-60")}>
            <SummaryView summary={summary} />
          </div>
        </Panel>
      ) : (
        <EmptyState title="No summary yet" action={button}>
          Generate one from the saved transcript, questions and topics. With a cloud AI provider
          this sends the meeting text to that provider.
        </EmptyState>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

export function TranscriptTab({ record }: { record: MeetingRecord }) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const segments = useMemo(
    () => (q ? record.segments.filter((s) => s.text.toLowerCase().includes(q)) : record.segments),
    [record.segments, q],
  );
  if (record.segments.length === 0) {
    return <EmptyState title="No transcript">Nothing was transcribed in this meeting.</EmptyState>;
  }
  return (
    <div>
      <div className="mb-3 flex items-center gap-3">
        <div className="relative w-full max-w-xs">
          <Icon
            name="search"
            className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted"
          />
          <TextInput
            type="search"
            aria-label="Search transcript"
            placeholder="Search transcript"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-8"
          />
        </div>
        {q && (
          <span role="status" className="text-[0.78rem] text-muted">
            {countLabel(segments.length, "match", "matches")}
          </span>
        )}
      </div>
      <div className="space-y-3 rounded-lg border border-border bg-surface px-4 py-3">
        {segments.length === 0 ? (
          <p className="py-4 text-center text-[0.84rem] text-muted">
            No lines match “{query.trim()}”.
          </p>
        ) : (
          <TranscriptBlocks
            segments={segments}
            mode={record.meeting.mode}
            meetingStartedAt={record.meeting.startedAt}
          />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Questions & answers
// ---------------------------------------------------------------------------

function SuggestionCard({ s }: { s: Suggestion }) {
  const meta = [s.style ? answerStyleLabel(s.style) : null, s.model || s.provider || null]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="rounded-md border border-border bg-bg/40 px-3.5 py-3">
      <div className="mb-2 flex flex-wrap items-center gap-1.5 text-[0.74rem] text-muted">
        <Icon name="sparkle" className="size-3.5 text-accent" />
        <span className="font-medium text-fg-soft">{suggestionKindLabel(s.kind)}</span>
        {s.pinned && (
          <Badge tone="accent">
            <Icon name="pin" className="size-3" />
            Pinned
          </Badge>
        )}
        {meta && <span>· {meta}</span>}
      </div>
      {s.kind === "ask" && s.questionText && (
        <p className="mb-2 text-[0.8rem] text-fg-soft">
          <span className="text-muted">You asked:</span> {s.questionText}
        </p>
      )}
      <SuggestionContent content={s.content} />
    </div>
  );
}

function SpokenAnswer({ label, text }: { label: string; text: string }) {
  return (
    <div>
      <div className="mb-1 text-[0.68rem] font-semibold tracking-wider text-muted uppercase">
        {label}
      </div>
      <blockquote className="border-l-2 border-border-strong pl-3 text-[0.86rem] leading-relaxed whitespace-pre-line text-fg">
        {text}
      </blockquote>
    </div>
  );
}

export function QaTab({ record }: { record: MeetingRecord }) {
  const qa = useMemo(() => buildQa(record), [record]);
  const mode = record.meeting.mode;
  if (qa.items.length === 0 && qa.otherSuggestions.length === 0 && qa.otherAnswers.length === 0) {
    return (
      <EmptyState title="No questions">No questions were detected in this meeting.</EmptyState>
    );
  }
  return (
    <div className="space-y-4">
      {qa.items.map(({ question: q, answers, suggestions }) => (
        <article key={q.id} className="rounded-lg border border-border bg-surface px-5 py-4">
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5 text-[0.74rem] text-muted">
            <Badge tone="accent">{categoryLabel(q.category)}</Badge>
            <span className="tabular-nums">
              {formatOffset(segmentOffset(q.detectedAt, record.meeting.startedAt))}
            </span>
            <span>
              · asked by{" "}
              {speakerLabel(q.speakerRole, mode).toLowerCase() === "you"
                ? "you"
                : speakerLabel(q.speakerRole, mode)}
            </span>
          </div>
          <h3 className="text-[0.92rem] leading-snug font-semibold text-fg">{q.question}</h3>
          {q.parts.length > 1 && (
            <ul className="mt-1 list-disc pl-5 text-[0.82rem] text-fg-soft">
              {q.parts.map((p, i) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          )}
          <div className="mt-3 space-y-3">
            {answers.map((a) => (
              <SpokenAnswer
                key={a.id}
                label={spokenAnswerLabel(a.speakerRole, mode)}
                text={a.text}
              />
            ))}
            {answers.length === 0 && (
              <p className="text-[0.8rem] text-muted">No spoken answer was recorded.</p>
            )}
            {suggestions.map((s) => (
              <SuggestionCard key={s.id} s={s} />
            ))}
          </div>
        </article>
      ))}
      {qa.otherAnswers.length > 0 && (
        <Panel>
          <SubHeading count={qa.otherAnswers.length}>Other answers</SubHeading>
          <div className="space-y-3">
            {qa.otherAnswers.map((a) => (
              <SpokenAnswer
                key={a.id}
                label={spokenAnswerLabel(a.speakerRole, mode)}
                text={a.text}
              />
            ))}
          </div>
        </Panel>
      )}
      {qa.otherSuggestions.length > 0 && (
        <Panel>
          <SubHeading count={qa.otherSuggestions.length}>Other AI suggestions</SubHeading>
          <div className="space-y-3">
            {qa.otherSuggestions.map((s) => (
              <SuggestionCard key={s.id} s={s} />
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Follow-ups
// ---------------------------------------------------------------------------

function FollowupItem({ f }: { f: FollowUpQuestion }) {
  const notify = useMain((s) => s.notify);
  const [copied, setCopied] = useState(false);
  const priority = priorityBadge(f.priority);
  const copy = async () => {
    try {
      await writeText(f.question);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      notify("error", `Could not copy: ${errorMessage(e)}`);
    }
  };
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span
        aria-hidden
        className={cx("mt-px text-[0.9rem]", f.priority === "high" ? "text-accent" : "text-muted")}
      >
        →
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[0.88rem] leading-snug text-fg">{f.question}</p>
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          <Badge>{categoryLabel(f.category)}</Badge>
          {priority && <Badge tone={priority.tone}>{priority.label}</Badge>}
          {f.reason && <span className="text-[0.78rem] text-muted">{f.reason}</span>}
        </div>
        {f.trigger && (
          <p className="mt-1 text-[0.76rem] text-muted">
            Triggered by: <q className="text-fg-soft">{f.trigger}</q>
          </p>
        )}
      </div>
      <Button
        size="sm"
        variant="ghost"
        aria-label={copied ? "Copied" : `Copy question: ${f.question}`}
        onClick={() => void copy()}
      >
        <Icon name={copied ? "check" : "copy"} className="size-3.5" />
        <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
      </Button>
    </li>
  );
}

export function FollowupsTab({ record }: { record: MeetingRecord }) {
  const groups = useMemo(() => followupGroups(record), [record]);
  if (groups.length === 0) {
    return (
      <EmptyState title="No follow-ups">
        {record.meeting.mode === "interviewer"
          ? "No follow-up questions were suggested in this meeting."
          : "Follow-up questions are suggested in interviewer mode."}
      </EmptyState>
    );
  }
  return (
    <div className="space-y-4">
      {groups.map((g) => (
        <Panel key={g.suggestion.id}>
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
            <SubHeading count={g.items.length}>
              {g.suggestion.questionText ? `After “${g.suggestion.questionText}”` : "Follow-ups"}
            </SubHeading>
            <span className="text-[0.74rem] text-muted tabular-nums">
              {formatOffset(segmentOffset(g.suggestion.createdAt, record.meeting.startedAt))}
            </span>
          </div>
          <ul className="divide-y divide-border">
            {g.items.map((f) => (
              <FollowupItem key={f.id} f={f} />
            ))}
          </ul>
        </Panel>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Topics & claims
// ---------------------------------------------------------------------------

export function TopicsTab({ record }: { record: MeetingRecord }) {
  const claims = useMemo(() => groupClaims(record.claims), [record.claims]);
  const mode: Mode = record.meeting.mode;
  if (record.topics.length === 0 && claims.length === 0) {
    return (
      <EmptyState title="No topics or claims">Nothing was tracked for this meeting.</EmptyState>
    );
  }
  return (
    <div className="space-y-4">
      {record.topics.length > 0 && (
        <Panel>
          <SubHeading count={record.topics.length}>Topics</SubHeading>
          <ul className="divide-y divide-border">
            {record.topics.map((t) => (
              <li key={t.id} className="flex items-start gap-3 py-2">
                <span
                  aria-hidden
                  className={cx(
                    "mt-px w-4 text-center",
                    t.status === "covered"
                      ? "text-ok"
                      : t.status === "current"
                        ? "text-accent"
                        : "text-muted",
                  )}
                >
                  {t.status === "covered" ? "✓" : t.status === "current" ? "→" : "○"}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[0.86rem] font-medium text-fg">{t.name}</span>
                    {t.planned && <Badge>Planned</Badge>}
                  </div>
                  {t.summary && <p className="mt-0.5 text-[0.8rem] text-fg-soft">{t.summary}</p>}
                </div>
                <div className="shrink-0 text-right text-[0.76rem] text-muted">
                  <div>{topicStatusLabel(t.status)}</div>
                  <div className="tabular-nums">{countLabel(t.mentions, "mention")}</div>
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      )}
      {claims.map((g) => (
        <Panel key={g.kind}>
          <SubHeading count={g.claims.length}>{g.label}</SubHeading>
          <ul className="space-y-1.5">
            {g.claims.map((c) => (
              <li key={c.id} className="flex items-start gap-2 text-[0.86rem]">
                <span className="mt-0.5 w-20 shrink-0 text-[0.74rem] text-muted">
                  {speakerLabel(c.speakerRole, mode)}
                </span>
                <span className="min-w-0 flex-1 text-fg">{c.text}</span>
                <span className="shrink-0 text-[0.7rem] text-muted">
                  {c.source === "llm" ? "AI" : "Heuristic"}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen observations
// ---------------------------------------------------------------------------

export function ScreenTab({ record }: { record: MeetingRecord }) {
  const observations = useMemo(
    () => sortedObservations(record.observations),
    [record.observations],
  );
  if (observations.length === 0) {
    return (
      <EmptyState title="No screen observations">
        Screen context was off, or nothing on screen changed enough to be analysed.
      </EmptyState>
    );
  }
  return (
    <div className="space-y-3">
      <p className="text-[0.78rem] text-muted">
        Text descriptions only — raw screenshots are not part of the record.
      </p>
      {observations.map((o) => (
        <article key={o.id} className="rounded-lg border border-border bg-surface px-5 py-3.5">
          <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[0.74rem] text-muted">
            <span className="tabular-nums" title={formatDateTime(o.timestamp)}>
              {formatOffset(segmentOffset(o.timestamp, record.meeting.startedAt))}
            </span>
            <span>· {o.source}</span>
            {(o.application || o.title) && (
              <span>· {[o.application, o.title].filter(Boolean).join(" — ")}</span>
            )}
            {o.contentType && <Badge>{o.contentType}</Badge>}
          </div>
          <p className="text-[0.86rem] leading-relaxed whitespace-pre-line text-fg">
            {o.description}
          </p>
          {o.entities.length > 0 && (
            <ul aria-label="Entities" className="mt-2 flex flex-wrap gap-1">
              {o.entities.map((e, i) => (
                <li
                  key={i}
                  className="rounded border border-border px-1.5 py-px text-[0.74rem] text-fg-soft"
                >
                  {e}
                </li>
              ))}
            </ul>
          )}
          {o.visibleText && (
            <details className="mt-2 text-[0.8rem]">
              <summary className="cursor-pointer text-muted select-none hover:text-fg-soft">
                Visible text
              </summary>
              <pre className="mt-1.5 max-h-60 overflow-auto rounded-md bg-surface-2 p-2.5 font-mono text-[0.76rem] whitespace-pre-wrap text-fg-soft">
                {o.visibleText}
              </pre>
            </details>
          )}
        </article>
      ))}
    </div>
  );
}
