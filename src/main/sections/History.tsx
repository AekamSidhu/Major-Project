// History: past meetings (paginated) and a meeting detail view with summary,
// transcript, Q&A, follow-ups, topics & claims and screen observations;
// export and delete (spec §41, §42, §90, §91).

import { useCallback, useState } from "react";
import type { MeetingListItem, MeetingRecord, MeetingSummary } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { formatDateTime, formatDuration } from "@/lib/format";
import { useLoader } from "../components/useLoader";
import { useMain } from "../store";
import { Badge, Banner, Button, ConfirmDialog, EmptyState, PageHeader, Spinner, Tabs } from "../ui";
import { FollowupsTab, QaTab, ScreenTab, SummaryTab, TopicsTab, TranscriptTab } from "./history/DetailTabs";
import { ExportDialog } from "./history/ExportDialog";
import { countLabel, listStateBadge, pageRangeLabel, pageRequest, splitPage } from "./history/helpers";
import { useHistoryNav } from "./history/historyNav";

type DetailTab = "summary" | "transcript" | "qa" | "followups" | "topics" | "screen";

function MeetingList() {
  const page = useHistoryNav((s) => s.page);
  const setPage = useHistoryNav((s) => s.setPage);
  const openMeeting = useHistoryNav((s) => s.openMeeting);
  const liveId = useMain((s) => s.live.meeting?.id ?? null);
  const liveState = useMain((s) => s.live.state);
  const fetchPage = useCallback(async () => {
    const { limit, offset } = pageRequest(page);
    return splitPage(await api.listMeetings(limit, offset));
    // `liveState` is a deliberate dependency: re-fetch when a meeting starts or ends.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, liveState]);
  const loader = useLoader(fetchPage);
  const rows: MeetingListItem[] | null = loader.data?.items ?? (loader.error ? [] : null);
  const hasMore = loader.data?.hasMore ?? false;
  const error = loader.error;

  if (rows === null) {
    return (
      <div className="flex items-center gap-2 py-8 text-[0.86rem] text-muted">
        <Spinner /> Loading meetings…
      </div>
    );
  }
  return (
    <>
      {error && (
        <Banner tone="danger" title="Could not load your meetings" actions={<Button size="sm" onClick={loader.reload}>Retry</Button>}>
          {error}
        </Banner>
      )}
      {rows.length === 0 && !error ? (
        <EmptyState title={page === 0 ? "No meetings yet" : "No more meetings"}>
          {page === 0
            ? "Meetings you hold with OpenMeet appear here (depending on your retention setting). Try a demo meeting from Home."
            : "You've reached the end of your history."}
        </EmptyState>
      ) : (
        <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-surface">
          {rows.map((m) => {
            const badge = listStateBadge(m, liveId);
            return (
              <li key={m.id}>
                <button
                  type="button"
                  onClick={() => openMeeting(m.id)}
                  className="flex w-full items-center gap-4 px-4 py-3 text-left transition-colors hover:bg-surface-2"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[0.9rem] font-medium text-fg">{m.title}</span>
                      <Badge>{m.mode === "candidate" ? "Candidate" : "Interviewer"}</Badge>
                      {badge && <Badge tone={badge.tone}>{badge.label}</Badge>}
                      {m.hasSummary && <Badge tone="accent">Summary</Badge>}
                    </div>
                    <div className="mt-0.5 text-[0.78rem] text-muted">
                      {formatDateTime(m.startedAt)} · {formatDuration(m.startedAt, m.endedAt)} ·{" "}
                      {countLabel(m.questionCount, "question")} · {countLabel(m.segmentCount, "transcript line")}
                    </div>
                  </div>
                  <span aria-hidden className="text-muted">›</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {(page > 0 || hasMore) && (
        <div className="mt-3 flex items-center justify-between text-[0.8rem] text-muted">
          <Button size="sm" variant="ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
            ← Newer
          </Button>
          <span>Meetings {pageRangeLabel(page, rows.length)}</span>
          <Button size="sm" variant="ghost" disabled={!hasMore} onClick={() => setPage(page + 1)}>
            Older →
          </Button>
        </div>
      )}
    </>
  );
}

function MeetingDetail({ meetingId }: { meetingId: string }) {
  const closeMeeting = useHistoryNav((s) => s.closeMeeting);
  const notify = useMain((s) => s.notify);
  const liveSummary = useMain((s) => (s.live.meeting?.id === meetingId ? s.live.lastSummary : null));
  const fetchMeeting = useCallback(() => api.getMeeting(meetingId), [meetingId]);
  const loader = useLoader(fetchMeeting);
  const record: MeetingRecord | null = loader.data;
  const error = loader.error;
  const [summaryOverride, setSummary] = useState<MeetingSummary | null>(null);
  const [tab, setTab] = useState<DetailTab>("summary");
  const [exporting, setExporting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  // A regenerated summary, else the stored one, else one that finished while open.
  const shownSummary = summaryOverride ?? record?.summary ?? liveSummary;

  const remove = async () => {
    setDeleting(true);
    try {
      await api.deleteMeeting(meetingId);
      notify("info", "Meeting deleted.");
      closeMeeting();
    } catch (e) {
      notify("error", `Could not delete the meeting: ${errorMessage(e)}`);
    } finally {
      setDeleting(false);
      setConfirmDelete(false);
    }
  };

  if (error) {
    return (
      <>
        <Button variant="ghost" size="sm" onClick={closeMeeting}>
          ← All meetings
        </Button>
        <Banner tone="danger" title="Could not open this meeting" actions={<Button size="sm" onClick={loader.reload}>Retry</Button>}>
          {error}
        </Banner>
      </>
    );
  }
  if (!record) {
    return (
      <div className="flex items-center gap-2 py-8 text-[0.86rem] text-muted">
        <Spinner /> Loading meeting…
      </div>
    );
  }
  const m = record.meeting;
  const followupCount = record.suggestions.reduce((n, s) => n + s.followups.length, 0);
  const tabs: ReadonlyArray<{ value: DetailTab; label: string }> = [
    { value: "summary", label: "Summary" },
    { value: "transcript", label: `Transcript (${record.segments.length})` },
    { value: "qa", label: `Questions & answers (${record.questions.length})` },
    { value: "followups", label: `Follow-ups (${followupCount})` },
    { value: "topics", label: "Topics & claims" },
    { value: "screen", label: `Screen (${record.observations.length})` },
  ];
  return (
    <>
      <Button variant="ghost" size="sm" onClick={closeMeeting} className="-ml-2 mb-2">
        ← All meetings
      </Button>
      <PageHeader
        title={m.title}
        description={
          <>
            {formatDateTime(m.startedAt)} · {formatDuration(m.startedAt, m.endedAt)} ·{" "}
            {m.mode === "candidate" ? "Candidate mode" : "Interviewer mode"}
            {!m.persisted && " · not saved to disk (retention: do not retain)"}
          </>
        }
        actions={
          <>
            <Button onClick={() => setExporting(true)}>Export…</Button>
            <Button variant="danger" onClick={() => setConfirmDelete(true)} disabled={!m.persisted}>
              Delete
            </Button>
          </>
        }
      />
      <Tabs label="Meeting details" value={tab} tabs={tabs} onChange={setTab} />
      {tab === "summary" && <SummaryTab record={record} summary={shownSummary} onSummary={setSummary} />}
      {tab === "transcript" && <TranscriptTab record={record} />}
      {tab === "qa" && <QaTab record={record} />}
      {tab === "followups" && <FollowupsTab record={record} />}
      {tab === "topics" && <TopicsTab record={record} />}
      {tab === "screen" && <ScreenTab record={record} />}
      {exporting && <ExportDialog meetingId={m.id} meetingTitle={m.title} onClose={() => setExporting(false)} />}
      <ConfirmDialog
        open={confirmDelete}
        title="Delete this meeting?"
        message="The transcript, questions, suggestions, summary and screen observations of this meeting are permanently deleted from this device."
        confirmLabel="Delete meeting"
        danger
        busy={deleting}
        onConfirm={() => void remove()}
        onCancel={() => setConfirmDelete(false)}
      />
    </>
  );
}

export function HistorySection() {
  const selectedId = useHistoryNav((s) => s.selectedId);
  if (selectedId) return <MeetingDetail meetingId={selectedId} />;
  return (
    <>
      <PageHeader
        title="History"
        description="Past meetings stored on this device. Retention and deletion are under Privacy & data."
      />
      <MeetingList />
    </>
  );
}
