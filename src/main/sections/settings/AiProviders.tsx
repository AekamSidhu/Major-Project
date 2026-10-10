// AI providers: the language model, screen understanding (vision) and
// retrieval embeddings. Every choice is labeled Local or Cloud (spec §44–§48).

import type { ReactNode } from "react";
import type { EmbeddingProviderKind, VisionProviderKind } from "@/bindings";
import { ApiKeyField } from "../../components/ApiKeyField";
import { ChoiceCards, TextSetting } from "../../components/fields";
import { LlmProviderForm, useModelList } from "../../components/LlmProviderForm";
import { LocationBadge } from "../../components/LocationBadge";
import { saveSettings } from "../../components/save";
import {
  DEFAULT_EMBEDDING_MODEL,
  EMBEDDING_PROVIDERS,
  VISION_PROVIDERS,
  choiceFor,
  cleanUrl,
  displayHost,
  effectiveEndpoint,
} from "../../components/settingsMeta";
import { TestConnection } from "../../components/TestConnection";
import { useSettings } from "../../components/useSettings";
import { validateOptionalBaseUrl } from "../../components/validation";
import { Banner, PageHeader, Section, ToggleField } from "../../ui";

function Endpoint({ component }: { component: "vision" | "embeddings" }) {
  const settings = useSettings();
  const ep = effectiveEndpoint(settings, component);
  if (ep.location === "disabled") return null;
  return (
    <div className="flex items-center gap-2 py-1 text-[0.8rem] text-muted" aria-live="polite">
      <span>{ep.url ? "Requests go to" : "Runs"}</span>
      {ep.url ? (
        <span className="font-mono text-fg-soft">{displayHost(ep.url)}</span>
      ) : (
        <span>inside OpenMeet</span>
      )}
      <LocationBadge location={ep.location} endpoint={ep.url ? displayHost(ep.url) : null} />
    </div>
  );
}

function LocalOnlyWarning({ component }: { component: "vision" | "embeddings" }) {
  const settings = useSettings();
  if (!settings.privacy.localOnly || effectiveEndpoint(settings, component).location !== "remote")
    return null;
  return (
    <Banner tone="warn" title="Blocked by local-only mode">
      This provider sends data off this device, so OpenMeet will not use it while local-only mode is
      on.
    </Banner>
  );
}

function VisionForm() {
  const settings = useSettings();
  const v = settings.ai.vision;
  const choice = choiceFor(VISION_PROVIDERS, v.provider);
  const usesModel = v.provider !== "disabled" && v.provider !== "ocr_only";
  const models = useModelList("vision", v.provider);
  const sameAsLlm = settings.ai.llm.provider === v.provider;
  const selfHosted = v.provider === "ollama" || v.provider === "openai_compatible";

  const select = (next: VisionProviderKind) => {
    if (next === v.provider) return;
    void saveSettings((d) => {
      d.ai.vision.provider = next;
      d.ai.vision.model = "";
      d.ai.vision.baseUrl = null;
    });
  };

  let modelHint: ReactNode;
  if (models.error)
    modelHint = <span className="text-danger">Could not load models: {models.error}</span>;
  else if (models.status) modelHint = models.status;
  else if (sameAsLlm && selfHosted)
    modelHint =
      "Leave empty to use the language model if it accepts images; otherwise name a vision model (e.g. llava, qwen2.5vl).";
  else if (sameAsLlm) modelHint = "Leave empty to use the same model as the language model.";
  else if (selfHosted) modelHint = "A vision-capable model, e.g. llava or qwen2.5vl.";
  else modelHint = "Leave empty to use the provider's default model.";

  return (
    <Section
      title="Screen understanding"
      description="How shared-screen frames are interpreted when screen context is on. Screen context itself stays off until you enable it."
    >
      <ChoiceCards
        label="Vision provider"
        value={v.provider}
        onChange={select}
        columns={3}
        options={VISION_PROVIDERS.map((p) => ({
          value: p.value,
          label: p.label,
          description: p.description,
          location: p.location,
        }))}
      />
      <Endpoint component="vision" />
      <LocalOnlyWarning component="vision" />
      {usesModel && (
        <>
          <TextSetting
            label="Vision model"
            description={modelHint}
            value={v.model}
            list={models.listId}
            monospace
            placeholder={sameAsLlm ? settings.ai.llm.model : undefined}
            onCommit={(m) =>
              saveSettings((d) => {
                d.ai.vision.model = m.trim();
              })
            }
            addon={models.button}
          />
          {models.datalist}
          {selfHosted && (
            <TextSetting
              label="Server address (optional)"
              description={
                sameAsLlm
                  ? "Leave empty to use the language model's server."
                  : `Leave empty to use ${choice.defaultBaseUrl ?? "the default address"}.`
              }
              value={v.baseUrl ?? ""}
              placeholder={choice.defaultBaseUrl ?? "http://localhost:1234/v1"}
              type="url"
              monospace
              validate={(u) => validateOptionalBaseUrl(u, false)}
              onCommit={(u) =>
                saveSettings((d) => {
                  d.ai.vision.baseUrl = cleanUrl(u);
                })
              }
            />
          )}
          <ToggleField
            label="Read on-screen text locally first"
            description="Runs the operating system's text recognition before the vision model and includes the recognized text. Helps with code and slides."
            checked={v.useOcr}
            onChange={(on) =>
              void saveSettings((d) => {
                d.ai.vision.useOcr = on;
              })
            }
          />
          {choice.secret && (
            <ApiKeyField provider={choice.secret} optional={v.provider === "openai_compatible"} />
          )}
          <TestConnection component="vision" />
        </>
      )}
      {v.provider === "ocr_only" && (
        <p className="py-2.5 text-[0.8rem] text-muted">
          OpenMeet reads text from the shared screen with the operating system's built-in text
          recognition (Windows) and summarizes it without an AI model. Diagrams and images are not
          interpreted.
        </p>
      )}
    </Section>
  );
}

function EmbeddingsForm() {
  const settings = useSettings();
  const e = settings.ai.embeddings;
  const choice = choiceFor(EMBEDDING_PROVIDERS, e.provider);
  const hasModel = e.provider === "ollama" || e.provider === "openai";
  const models = useModelList("embeddings", e.provider);

  const select = (next: EmbeddingProviderKind) => {
    if (next === e.provider) return;
    void saveSettings((d) => {
      d.ai.embeddings.provider = next;
      d.ai.embeddings.model = DEFAULT_EMBEDDING_MODEL[next] ?? "";
      d.ai.embeddings.baseUrl = null;
    });
  };

  let modelHint: ReactNode;
  if (models.error)
    modelHint = <span className="text-danger">Could not load models: {models.error}</span>;
  else if (models.status) modelHint = models.status;
  else if (e.provider === "ollama") modelHint = "Pull it first, e.g. ollama pull nomic-embed-text.";

  return (
    <Section
      title="Retrieval embeddings"
      description="Used to find earlier parts of a long meeting that are relevant to the current question."
    >
      <ChoiceCards
        label="Embeddings provider"
        value={e.provider}
        onChange={select}
        options={EMBEDDING_PROVIDERS.map((p) => ({
          value: p.value,
          label: p.label,
          description: p.description,
          location: p.location,
        }))}
      />
      <Endpoint component="embeddings" />
      <LocalOnlyWarning component="embeddings" />
      {hasModel && (
        <>
          <TextSetting
            label="Embedding model"
            description={modelHint}
            value={e.model}
            list={models.listId}
            monospace
            placeholder={DEFAULT_EMBEDDING_MODEL[e.provider] ?? undefined}
            onCommit={(m) =>
              saveSettings((d) => {
                d.ai.embeddings.model = m.trim();
              })
            }
            addon={models.button}
          />
          {models.datalist}
        </>
      )}
      {(e.provider === "ollama" || e.provider === "local_ai") && (
        <TextSetting
          label="Server address (optional)"
          description={`Leave empty to use ${choice.defaultBaseUrl ?? "the default address"}.`}
          value={e.baseUrl ?? ""}
          placeholder={choice.defaultBaseUrl ?? undefined}
          type="url"
          monospace
          validate={(u) => validateOptionalBaseUrl(u, false)}
          onCommit={(u) =>
            saveSettings((d) => {
              d.ai.embeddings.baseUrl = cleanUrl(u);
            })
          }
        />
      )}
      {choice.secret && <ApiKeyField provider={choice.secret} />}
      {e.provider !== "local_hash" && <TestConnection component="embeddings" />}
    </Section>
  );
}

export function AiProvidersSection() {
  return (
    <>
      <PageHeader
        title="AI providers"
        description="Choose where suggestions are generated. Local providers keep meeting data on this computer; cloud providers receive the transcript excerpts needed for each request."
      />
      <Section
        title="Language model"
        description="Generates answers, follow-up questions and summaries."
      >
        <LlmProviderForm />
      </Section>
      <VisionForm />
      <EmbeddingsForm />
    </>
  );
}
