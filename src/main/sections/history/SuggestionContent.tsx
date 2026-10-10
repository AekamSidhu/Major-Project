// Static rendering of a stored AI suggestion in the [[SECTION]] format.

import { useMemo, type ReactNode } from "react";
import { parseAnswer } from "@/lib/answer";

function Label({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 text-[0.68rem] font-semibold tracking-wider text-muted uppercase">
      {children}
    </div>
  );
}

export function SuggestionContent({ content }: { content: string }) {
  const a = useMemo(() => parseAnswer(content), [content]);
  const hasAny =
    a.answer ||
    a.structure.length ||
    a.keyPoints.length ||
    a.example ||
    a.followUp ||
    a.caveat ||
    a.clarify.length;
  if (!hasAny && !a.preamble) {
    return <p className="text-[0.82rem] text-muted italic">No text was generated.</p>;
  }
  return (
    <div className="space-y-3 text-[0.86rem] leading-relaxed text-fg">
      {a.preamble && <p className="whitespace-pre-line text-fg-soft">{a.preamble}</p>}
      {a.structure.length > 0 && (
        <div>
          <Label>Suggested structure</Label>
          <ol className="list-decimal space-y-0.5 pl-5 marker:text-muted">
            {a.structure.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      )}
      {a.answer && (
        <div>
          <Label>Suggested answer</Label>
          <p className="whitespace-pre-line">{a.answer}</p>
        </div>
      )}
      {a.clarify.length > 0 && (
        <div>
          <Label>Clarifying questions</Label>
          <ul className="list-disc space-y-0.5 pl-5 marker:text-muted">
            {a.clarify.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      {a.keyPoints.length > 0 && (
        <div>
          <Label>Key points</Label>
          <ul className="list-disc space-y-0.5 pl-5 marker:text-muted">
            {a.keyPoints.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      {a.example && (
        <div>
          <Label>Example</Label>
          <p className="whitespace-pre-line text-fg-soft">{a.example}</p>
        </div>
      )}
      {a.followUp && (
        <div>
          <Label>Follow-up you could ask</Label>
          <p className="text-fg-soft">{a.followUp}</p>
        </div>
      )}
      {a.caveat && (
        <p className="rounded-md border border-warn/40 bg-[color-mix(in_srgb,var(--om-warn)_10%,transparent)] px-2.5 py-1.5 text-[0.82rem] text-warn">
          {a.caveat}
        </p>
      )}
      {a.basis && (
        <details className="group text-[0.8rem]">
          <summary className="cursor-pointer text-muted select-none hover:text-fg-soft">
            Reasoning
          </summary>
          <p className="mt-1 whitespace-pre-line text-fg-soft">{a.basis}</p>
        </details>
      )}
    </div>
  );
}
