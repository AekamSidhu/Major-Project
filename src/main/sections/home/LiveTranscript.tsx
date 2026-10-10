// Live transcript: speaker blocks, partials in muted italics, and auto-scroll
// that pauses while the user reads earlier lines.

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Mode, TranscriptSegment } from "@/bindings";
import { formatOffset } from "@/lib/format";
import { cx } from "@/main/ui";
import { groupTranscript, isNearBottom, segmentOffset } from "./helpers";
import { Icon } from "./icons";

export interface TranscriptViewProps {
  segments: ReadonlyArray<TranscriptSegment>;
  mode: Mode;
  meetingStartedAt: number | null;
  emptyText: string;
  className?: string;
  label?: string;
}

export function TranscriptBlocks({
  segments,
  mode,
  meetingStartedAt,
}: Pick<TranscriptViewProps, "segments" | "mode" | "meetingStartedAt">) {
  const groups = useMemo(() => groupTranscript(segments, mode), [segments, mode]);
  return (
    <>
      {groups.map((g) => (
        <div key={g.key} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-3">
          <div className="pt-px text-[0.76rem] leading-snug">
            <div
              className={cx("font-semibold", g.role === "user" ? "text-accent" : "text-fg-soft")}
            >
              {g.label}
            </div>
            <div className="text-muted tabular-nums">
              {formatOffset(segmentOffset(g.startTime, meetingStartedAt))}
            </div>
          </div>
          <p className="text-[0.86rem] leading-relaxed text-fg">
            {g.segments.map((s, i) => (
              <span key={s.id} className={s.isFinal ? undefined : "text-muted italic"}>
                {i > 0 ? " " : ""}
                {s.text}
              </span>
            ))}
          </p>
        </div>
      ))}
    </>
  );
}

export function LiveTranscript({
  segments,
  mode,
  meetingStartedAt,
  emptyText,
  className,
  label = "Live transcript",
}: TranscriptViewProps) {
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);

  // Keep the newest line in view while following.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && following) el.scrollTop = el.scrollHeight;
  }, [segments, following]);

  return (
    <div className="relative">
      <div
        ref={scroller}
        role="log"
        aria-label={label}
        aria-live="off"
        tabIndex={0}
        onScroll={(e) => setFollowing(isNearBottom(e.currentTarget))}
        className={cx(
          "space-y-3 overflow-y-auto px-4 py-3 focus-visible:outline-offset-[-2px]",
          className,
        )}
      >
        {segments.length === 0 ? (
          <p className="py-6 text-center text-[0.84rem] text-muted">{emptyText}</p>
        ) : (
          <TranscriptBlocks segments={segments} mode={mode} meetingStartedAt={meetingStartedAt} />
        )}
      </div>
      {!following && segments.length > 0 && (
        <button
          type="button"
          onClick={() => setFollowing(true)}
          className="om-fade-in absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-1 rounded-full border border-border-strong bg-raised px-3 py-1 text-[0.76rem] font-medium text-fg-soft shadow-md hover:text-fg"
        >
          <Icon name="arrowDown" className="size-3.5" />
          Jump to latest
        </button>
      )}
    </div>
  );
}
