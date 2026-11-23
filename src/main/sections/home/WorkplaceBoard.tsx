// Home — workplace board: the running lists of decisions, action items (with
// owner and due date when stated), open questions and clarifying questions.

import type { ReactNode } from "react";

import type { WorkplaceBoard as Board } from "@/bindings";
import { formatOffset } from "@/lib/format";
import { Badge } from "@/main/ui";
import { ownerDueLine, questionStatus } from "./boardModel";

function BoardSection({
  title,
  count,
  empty,
  children,
}: {
  title: string;
  count: number;
  empty: string;
  children: ReactNode[];
}) {
  return (
    <section className="px-5 py-3">
      <h3 className="mb-1.5 flex items-center gap-2 text-[0.78rem] font-semibold text-fg-soft">
        {title}
        <Badge>{count}</Badge>
      </h3>
      {count === 0 ? (
        <p className="text-[0.8rem] text-muted">{empty}</p>
      ) : (
        <ul className="space-y-2">{children}</ul>
      )}
    </section>
  );
}

export function WorkplaceBoardView({ board }: { board: Board }) {
  return (
    <div className="divide-y divide-border">
      <BoardSection
        title="Decisions"
        count={board.decisions.length}
        empty="Decisions the team makes will appear here."
      >
        {board.decisions.map((d) => (
          <li key={d.id} className="text-[0.84rem] text-fg">
            {d.text}
          </li>
        ))}
      </BoardSection>
      <BoardSection
        title="Action items"
        count={board.actionItems.length}
        empty="Commitments and next steps will appear here."
      >
        {board.actionItems.map((a) => (
          <li key={a.id}>
            <p className="text-[0.84rem] text-fg">{a.text}</p>
            <p className="text-[0.74rem] text-muted">{ownerDueLine(a.owner, a.due)}</p>
          </li>
        ))}
      </BoardSection>
      <BoardSection
        title="Open questions"
        count={board.openQuestions.filter((q) => !q.answered).length}
        empty="Questions asked in the meeting appear here until they are answered."
      >
        {board.openQuestions.map((q) => {
          const status = questionStatus(q);
          return (
            <li key={q.id} className="text-[0.84rem] text-fg">
              <span className="text-muted tabular-nums">{formatOffset(q.askedAtMs)} </span>
              {q.question}{" "}
              <span className="text-[0.74rem] text-muted">
                ({status === "answered" ? `answered${q.answer ? `: ${q.answer}` : ""}` : status})
              </span>
            </li>
          );
        })}
      </BoardSection>
      {board.clarifications.length > 0 && (
        <BoardSection title="Clarify" count={board.clarifications.length} empty="">
          {board.clarifications.map((c) => (
            <li key={c.id} className="text-[0.84rem] text-fg">
              {c.question}
            </li>
          ))}
        </BoardSection>
      )}
    </div>
  );
}
