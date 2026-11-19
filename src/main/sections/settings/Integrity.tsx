// Answer integrity (interviewer mode): an opt-in check of each candidate
// answer with four signals, shown privately as a flag with its evidence — a
// reason to ask a follow-up, never a verdict. Off by default.

import { useCallback } from "react";
import type { DataFlowNode } from "@/bindings";
import { api } from "@/lib/ipc";
import { integrityPartLabel } from "../../components/dataFlow";
import { LocationBadge } from "../../components/LocationBadge";
import { RegionPicker } from "../../components/RegionPicker";
import { saveSettings } from "../../components/save";
import { useLoader } from "../../components/useLoader";
import { useSettings } from "../../components/useSettings";
import { useMain } from "../../store";
import { Badge, Banner, Button, PageHeader, Section, ToggleField } from "../../ui";
import { LinkButton } from "../home/controls";

function WhereItRuns({ nodes }: { nodes: DataFlowNode[] }) {
  if (nodes.length === 0) return null;
  return (
    <ul className="flex flex-wrap gap-2 py-2.5">
      {nodes.map((n) => (
        <li
          key={n.label}
          className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[0.76rem]"
        >
          <span className="text-fg-soft">{integrityPartLabel(n.label)}</span>
          <LocationBadge location={n.location} endpoint={n.endpoint} />
        </li>
      ))}
    </ul>
  );
}

/** Same LLM as the thresholds' reference model (an Ollama `:latest` tag is ignored). */
function sameModel(model: string, reference: string): boolean {
  const norm = (m: string) =>
    m
      .trim()
      .toLowerCase()
      .replace(/:latest$/, "");
  return norm(model) !== "" && norm(model) === norm(reference);
}

export function IntegritySection() {
  const settings = useSettings();
  const i = settings.integrity;
  const setSection = useMain((s) => s.setSection);
  const status = useLoader(api.getIntegrityStatus);
  const loadFlow = useCallback(() => api.getDataFlow(), []);
  const flow = useLoader(loadFlow);
  const reloadFlow = flow.reload;
  const integrityNodes = (flow.data?.nodes ?? []).filter((n) => n.component === "integrity");
  const required = status.data?.requiredEmbeddingModel ?? null;
  // Calibration comes from the spec thresholds (via the backend); unknown
  // while loading, shown as uncalibrated then.
  const echoCalibrated = status.data?.echoCalibrated ?? false;
  const referenceModel = status.data?.echoReferenceModel ?? null;
  const deliveryCalibrated = status.data?.deliveryCalibrated ?? false;
  const llmModel = settings.ai.llm.model;
  const embedder = settings.ai.embeddings.provider;

  const save = async (mutate: Parameters<typeof saveSettings>[0]) => {
    const ok = await saveSettings(mutate);
    reloadFlow();
    return ok;
  };

  return (
    <>
      <PageHeader
        title="Answer integrity"
        description="Interviewer mode only. After each candidate answer, OpenMeet checks four signals in the background and shows you a private flag with its evidence. A flag is a reason to ask a follow-up question, never a verdict. Off by default."
      />
      <Banner tone="accent" title="Tell the candidate">
        Let candidates know before the interview that their answers are checked this way — and, with
        Reading on, that their video tile is analysed for eye movements. Check what your local law
        and your organisation's policy require.
      </Banner>

      <Section>
        <ToggleField
          label="Check answers in interviewer mode"
          description="Runs while an interviewer meeting is active. Suggestions and follow-ups are never slowed down by it."
          checked={i.enabled}
          onChange={(v) =>
            void save((d) => {
              d.integrity.enabled = v;
            })
          }
        />
        {i.enabled && <WhereItRuns nodes={integrityNodes} />}
      </Section>

      <Section
        title="Signals"
        description="Each signal can be turned off. Two raised signals, one of them calibrated, are needed for the strongest flag."
      >
        <ToggleField
          label="AI-written text"
          description="Local AI scores the answer's transcript with a model trained on public data (calibrated)."
          checked={i.aiText}
          onChange={(v) =>
            void save((d) => {
              d.integrity.aiText = v;
            })
          }
        />
        <ToggleField
          label="Echo of an LLM answer"
          description={
            <>
              Your AI provider answers the interviewer's question itself; the candidate's answer is
              compared with those answers by embedding similarity. Works only with the{" "}
              {required ? <code className="text-[0.78rem]">{required}</code> : "local"} embedder (AI
              providers → Embeddings: Local AI).{" "}
              {echoCalibrated && referenceModel ? (
                <>
                  Calibrated with reference answers from{" "}
                  <code className="text-[0.78rem]">{referenceModel}</code>
                  {sameModel(llmModel, referenceModel)
                    ? "."
                    : `; with ${llmModel || "another model"} it counts as uncalibrated.`}
                </>
              ) : (
                <Badge tone="neutral">uncalibrated</Badge>
              )}
            </>
          }
          checked={i.echoSimilarity}
          onChange={(v) =>
            void save((d) => {
              d.integrity.echoSimilarity = v;
            })
          }
        />
        <ToggleField
          label="Delivery"
          description={
            <>
              Answer timing — how long the candidate takes to start, pace, pauses and filler words —
              compared with their own first three answers. Uses word timestamps from Local AI
              transcription when available.{" "}
              {!deliveryCalibrated && <Badge tone="neutral">uncalibrated</Badge>}
            </>
          }
          checked={i.delivery}
          onChange={(v) =>
            void save((d) => {
              d.integrity.delivery = v;
            })
          }
        />
        <ToggleField
          label="Reading"
          description="Local AI captures the candidate's video region (10 frames per second), turns each frame into a gaze point and drops it; a model trained on public eye-tracking data scores whether the eyes move like reading. Needs the region below."
          checked={i.reading}
          onChange={(v) =>
            void save((d) => {
              d.integrity.reading = v;
            })
          }
        />
        {i.echoSimilarity && embedder !== "local_ai" && (
          <div className="py-2.5 text-[0.8rem] text-muted">
            The echo check is unavailable with the current embeddings provider.{" "}
            <LinkButton onClick={() => setSection("settings/ai")}>Change embeddings</LinkButton>
          </div>
        )}
      </Section>

      <Section
        title="Candidate video region"
        description="The part of your screen where the meeting app shows the candidate's video. Only this rectangle is captured, only during interviewer meetings with Reading on, and no frame is stored."
      >
        <RegionPicker
          value={i.candidateRegion}
          onSave={(rect) =>
            save((d) => {
              d.integrity.candidateRegion = rect;
            })
          }
        />
      </Section>

      <Section
        title="Local AI"
        description="The AI-written text and Reading signals run in the local AI sidecar."
        actions={
          <Button size="sm" variant="ghost" onClick={status.reload} busy={status.loading}>
            Check again
          </Button>
        }
      >
        <div className="space-y-1 py-2.5 text-[0.82rem]">
          {status.error && <p className="text-danger">{status.error}</p>}
          {status.data && (
            <>
              <p className={status.data.reachable ? "text-fg-soft" : "text-muted"}>
                {status.data.message}
              </p>
              {status.data.reachable && (
                <p className="text-muted">
                  AI-written text model: {status.data.aiTextReady ? "ready" : "not available"} ·
                  Reading model: {status.data.readingReady ? "ready" : "not available"}
                  {status.data.embeddingModel ? ` · Embeddings: ${status.data.embeddingModel}` : ""}
                </p>
              )}
            </>
          )}
        </div>
      </Section>

      <Section title="Limits">
        <ul className="list-disc space-y-1 py-2.5 pl-5 text-[0.82rem] text-fg-soft">
          <li>
            Thresholds are not calibrated on real interviews yet; the calibrated ones were measured
            on public data. Signals marked uncalibrated count at most as “possible”.
          </li>
          <li>
            Fluent speakers, prepared answers, accents and transcription errors can raise signals
            without any AI help; nervous candidates can look unusual too.
          </li>
          <li>
            Delivery needs three answers of at least ten words as this candidate's baseline before
            it scores anything.
          </li>
          <li>Reading needs about ten seconds of a clearly visible face during the answer.</li>
        </ul>
      </Section>
    </>
  );
}
