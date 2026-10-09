// Language-model provider form, shared by Settings → AI providers and
// onboarding. Changes are saved immediately (text fields debounced).

import { useId, useState } from "react";
import type { LlmProviderKind } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { Banner, Button } from "../ui";
import { ApiKeyField } from "./ApiKeyField";
import { ChoiceCards, NumberSetting, SelectSetting, SliderSetting, TextSetting } from "./fields";
import { LocationBadge } from "./LocationBadge";
import { saveSettings, settleSettings } from "./save";
import {
  LLM_PROVIDERS,
  REASONING_EFFORT_OPTIONS,
  choiceFor,
  cleanUrl,
  displayHost,
  effectiveEndpoint,
  switchLlmProvider,
} from "./settingsMeta";
import { TestConnection } from "./TestConnection";
import { useSettings } from "./useSettings";
import { validateOptionalBaseUrl, validateRequired } from "./validation";

export interface ModelListState {
  provider: string;
  list: string[];
}

/** "Load models" button + datalist for a model input. */
export function useModelList(component: "llm" | "transcription" | "vision" | "embeddings", provider: string) {
  const listId = useId();
  const [models, setModels] = useState<ModelListState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setBusy(true);
    setError(null);
    try {
      await settleSettings();
      const list = await api.listModels(component);
      setModels({ provider, list: [...new Set(list)].sort((a, b) => a.localeCompare(b)) });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const current = models && models.provider === provider ? models.list : null;
  const button = (
    <Button onClick={() => void load()} busy={busy} className="shrink-0">
      Load models
    </Button>
  );
  const datalist = current ? (
    <datalist id={listId}>
      {current.map((m) => (
        <option key={m} value={m} />
      ))}
    </datalist>
  ) : null;
  let status: string | null = null;
  if (current) {
    status =
      current.length === 0
        ? "The provider reported no models."
        : `${current.length} model${current.length === 1 ? "" : "s"} available — pick one from the list or type a name.`;
  }
  return { listId: current ? listId : undefined, button, datalist, status, error };
}

export function LlmProviderForm({ variant = "full" }: { variant?: "full" | "compact" }) {
  const settings = useSettings();
  const llm = settings.ai.llm;
  const [pendingCompat, setPendingCompat] = useState(false);
  const shown: LlmProviderKind = pendingCompat ? "openai_compatible" : llm.provider;
  const choice = choiceFor(LLM_PROVIDERS, shown);
  const models = useModelList("llm", shown);
  const ep = pendingCompat ? { url: "", location: "unset" as const } : effectiveEndpoint(settings, "llm");
  const urlRequired = shown === "ollama" || shown === "openai_compatible";
  const showBaseUrl = variant === "full" || urlRequired;

  const selectProvider = (next: LlmProviderKind) => {
    if (next === llm.provider) {
      setPendingCompat(false);
      return;
    }
    const patch = switchLlmProvider(llm, next);
    if (next === "openai_compatible" && !patch.baseUrl) {
      // The backend requires an address for this provider; switch once it is entered.
      setPendingCompat(true);
      return;
    }
    setPendingCompat(false);
    void saveSettings((d) => {
      d.ai.llm = switchLlmProvider(d.ai.llm, next);
    });
  };

  const commitBaseUrl = async (v: string) => {
    if (pendingCompat) {
      const ok = await saveSettings((d) => {
        d.ai.llm = { ...switchLlmProvider(d.ai.llm, "openai_compatible"), baseUrl: cleanUrl(v) };
      });
      if (ok) setPendingCompat(false);
      return ok;
    }
    return saveSettings((d) => {
      d.ai.llm.baseUrl = cleanUrl(v);
    });
  };

  return (
    <div>
      <ChoiceCards
        label="Language model provider"
        value={shown}
        onChange={selectProvider}
        options={LLM_PROVIDERS.map((p) => ({
          value: p.value,
          label: p.label,
          description: p.description,
          location: p.location,
        }))}
      />

      {pendingCompat && <Banner tone="accent">Enter your server's address to switch to an OpenAI-compatible server.</Banner>}

      <div className="flex items-center gap-2 py-1 text-[0.8rem] text-muted" aria-live="polite">
        <span>Requests go to</span>
        {ep.url ? <span className="font-mono text-fg-soft">{displayHost(ep.url)}</span> : <span>—</span>}
        <LocationBadge location={ep.location} endpoint={ep.url ? displayHost(ep.url) : null} />
      </div>

      {settings.privacy.localOnly && ep.location === "remote" && (
        <Banner tone="warn" title="Blocked by local-only mode">
          This provider sends data off this device, so OpenMeet will refuse to use it while local-only mode is on.
          Choose a local provider or turn off local-only mode in Privacy &amp; data.
        </Banner>
      )}

      {showBaseUrl && (
        <TextSetting
          label={urlRequired ? "Server address" : "Custom endpoint (optional)"}
          description={
            urlRequired
              ? shown === "ollama"
                ? "Where Ollama listens. The default is this computer."
                : "Base URL of the server, with or without /v1 — e.g. http://localhost:1234/v1."
              : `Leave empty to use ${choice.defaultBaseUrl ?? "the provider's default"}.`
          }
          value={pendingCompat ? "" : (llm.baseUrl ?? "")}
          placeholder={choice.defaultBaseUrl ?? "http://localhost:1234/v1"}
          type="url"
          monospace
          validate={(v) => validateOptionalBaseUrl(v, urlRequired)}
          onCommit={commitBaseUrl}
          autoFocus={pendingCompat}
        />
      )}

      {!pendingCompat && (
        <>
          <TextSetting
            label="Model"
            description={
              models.error ? (
                <span className="text-danger">Could not load models: {models.error}</span>
              ) : (
                (models.status ??
                (shown === "ollama"
                  ? "A model you have pulled, e.g. llama3.2 (run `ollama pull llama3.2` first)."
                  : "The model name exactly as the provider spells it."))
              )
            }
            value={llm.model}
            list={models.listId}
            monospace
            validate={(v) => validateRequired(v, "A model")}
            onCommit={(v) =>
              saveSettings((d) => {
                d.ai.llm.model = v.trim();
              })
            }
            addon={models.button}
          />
          {models.datalist}

          {choice.secret && <ApiKeyField provider={choice.secret} optional={shown === "openai_compatible"} />}

          <TestConnection component="llm" />
        </>
      )}

      {variant === "full" && !pendingCompat && (
        <details className="group border-t border-border pt-1">
          <summary className="cursor-pointer py-2 text-[0.84rem] font-medium text-fg-soft select-none hover:text-fg">
            Advanced model options
          </summary>
          <div className="divide-y divide-border">
            <SliderSetting
              label="Temperature"
              description="Lower is more focused. Ignored by models that don't accept sampling parameters."
              value={llm.temperature}
              min={0}
              max={2}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onCommit={(v) =>
                saveSettings((d) => {
                  d.ai.llm.temperature = v;
                })
              }
            />
            <NumberSetting
              label="Max output tokens"
              description="Budget for the visible part of each live suggestion."
              value={llm.maxOutputTokens}
              min={128}
              max={16000}
              onCommit={(v) =>
                saveSettings((d) => {
                  d.ai.llm.maxOutputTokens = v;
                })
              }
            />
            <SelectSetting
              label="Reasoning effort"
              description="Applied where the model supports it. Higher effort is slower during live meetings."
              value={llm.reasoningEffort}
              options={REASONING_EFFORT_OPTIONS}
              onChange={(v) =>
                void saveSettings((d) => {
                  d.ai.llm.reasoningEffort = v;
                })
              }
            />
            <TextSetting
              label="Fast model (optional)"
              description="A cheaper, faster model on the same provider for question classification and context compression. Leave empty to use the main model."
              value={llm.fastModel ?? ""}
              list={models.listId}
              placeholder="Same as the main model"
              monospace
              onCommit={(v) =>
                saveSettings((d) => {
                  d.ai.llm.fastModel = v.trim() === "" ? null : v.trim();
                })
              }
            />
          </div>
        </details>
      )}
    </div>
  );
}
