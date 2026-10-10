// Renders a MeetingSummary as quiet, readable sections (never scores).

import { useMemo } from "react";
import type { MeetingSummary } from "@/bindings";
import { formatDateTime } from "@/lib/format";
import { generatedByLabel, summarySections } from "./helpers";

const CHIP_SECTIONS = new Set(["keyTopics", "technicalTopics"]);

function Heading({ children }: { children: string }) {
  return (
    <h3 className="mb-1.5 text-[0.72rem] font-semibold tracking-wider text-muted uppercase">
      {children}
    </h3>
  );
}

export function SummaryView({ summary }: { summary: MeetingSummary }) {
  const sections = useMemo(() => summarySections(summary), [summary]);
  const overview = summary.overview.trim();
  const empty = !overview && sections.length === 0;
  return (
    <article className="space-y-5">
      {empty && (
        <p className="text-[0.86rem] text-muted">
          The summary has no content — the meeting may have been very short.
        </p>
      )}
      {overview && (
        <section>
          <Heading>Overview</Heading>
          <p className="text-[0.88rem] leading-relaxed whitespace-pre-line text-fg">{overview}</p>
        </section>
      )}
      {sections.map((s) => (
        <section key={s.key}>
          <Heading>{s.title}</Heading>
          {CHIP_SECTIONS.has(s.key) ? (
            <ul className="flex flex-wrap gap-1.5">
              {s.items.map((item, i) => (
                <li
                  key={i}
                  className="rounded-md border border-border px-2 py-0.5 text-[0.8rem] text-fg-soft"
                >
                  {item}
                </li>
              ))}
            </ul>
          ) : (
            <ul className="list-disc space-y-1 pl-5 text-[0.86rem] leading-relaxed text-fg marker:text-muted">
              {s.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          )}
        </section>
      ))}
      <p className="border-t border-border pt-3 text-[0.76rem] text-muted">
        {generatedByLabel(summary.generatedBy)}
        {summary.generatedAt > 0 && ` · ${formatDateTime(summary.generatedAt)}`}
      </p>
    </article>
  );
}
