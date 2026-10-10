// Home — meeting completed: the summary (or "Generating summary…"), with
// Export / Open in History, and a way back to the start flow.

import { useEffect, useState } from "react";
import { formatDateTime, formatDuration } from "@/lib/format";
import { useMain, type LiveState } from "@/main/store";
import { Badge, Banner, Button, PageHeader, Spinner } from "@/main/ui";
import { ExportDialog } from "@/main/sections/history/ExportDialog";
import { openInHistory } from "@/main/sections/history/historyNav";
import { SummaryView } from "@/main/sections/history/SummaryView";
import { InlineError } from "./controls";
import { meetingStateBadge, modeLabel } from "./helpers";
import { useHomeUi } from "./homeState";
import { useMeetingEvents } from "./hooks";
import { Icon } from "./icons";

export function CompletedMeeting({ live }: { live: LiveState }) {
  const meeting = live.meeting;
  const dismiss = useHomeUi((s) => s.dismissCompleted);
  const [exportOpen, setExportOpen] = useState(false);
  const [summaryError, setSummaryError] = useState<{ meetingId: string; error: string } | null>(
    null,
  );
  const meetingId = meeting?.id ?? null;

  useMeetingEvents((ev) => {
    if (ev.type === "SUMMARY_FAILED") setSummaryError({ meetingId: ev.meetingId, error: ev.error });
  });

  // Events do not carry the end time; a snapshot does.
  const needsEndTime = meeting !== null && meeting.endedAt === null;
  useEffect(() => {
    if (needsEndTime) void useMain.getState().refreshSnapshot();
  }, [needsEndTime]);

  if (!meeting || !meetingId) return null;

  const summary =
    live.lastSummary && live.lastSummary.meetingId === meeting.id ? live.lastSummary : null;
  const failure = summaryError && summaryError.meetingId === meeting.id ? summaryError.error : null;
  const badge = meetingStateBadge(live.state, false, null);
  const isDemo = meeting.title.startsWith("Demo:");
  const details = [
    modeLabel(meeting.mode),
    formatDateTime(meeting.startedAt),
    meeting.endedAt !== null ? formatDuration(meeting.startedAt, meeting.endedAt) : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div>
      <PageHeader
        title={meeting.title}
        description={details}
        actions={
          <>
            <Button onClick={() => setExportOpen(true)}>
              <Icon name="download" />
              Export
            </Button>
            <Button
              onClick={() => openInHistory(meeting.id)}
              disabled={!meeting.persisted}
              title={meeting.persisted ? undefined : "This meeting was not saved"}
            >
              Open in History
              <Icon name="arrowRight" />
            </Button>
          </>
        }
      />

      {!meeting.persisted && (
        <Banner tone="neutral" title="Not saved">
          {isDemo
            ? "Demo meetings are kept in memory only and disappear when you start another meeting."
            : "Your retention setting keeps meetings in memory only. Export it now if you need a copy — it disappears when you start another meeting."}
        </Banner>
      )}

      <section
        aria-labelledby="om-completed-summary"
        className="mb-5 rounded-lg border border-border bg-surface px-5 py-4"
      >
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 id="om-completed-summary" className="text-[0.92rem] font-semibold text-fg">
            Summary
          </h2>
          <Badge tone={badge.tone}>{badge.label}</Badge>
        </div>
        {summary ? (
          <SummaryView summary={summary} />
        ) : failure ? (
          <InlineError>Summary failed: {failure}</InlineError>
        ) : live.state === "summarizing" ? (
          <p role="status" className="flex items-center gap-2 py-4 text-[0.86rem] text-muted">
            <Spinner /> Generating summary…
          </p>
        ) : (
          <p className="py-2 text-[0.86rem] text-muted">
            No summary was generated for this meeting.
            {meeting.persisted ? " You can generate one from History." : ""}
          </p>
        )}
      </section>

      <div className="flex justify-end">
        <Button variant="primary" onClick={() => dismiss(meeting.id)}>
          <Icon name="play" />
          Start a new meeting
        </Button>
      </div>

      {exportOpen && (
        <ExportDialog
          meetingId={meeting.id}
          meetingTitle={meeting.title}
          onClose={() => setExportOpen(false)}
        />
      )}
    </div>
  );
}
