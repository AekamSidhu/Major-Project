// Speech-to-text provider form, shared by Settings → Transcription & audio and
// onboarding.

import { useId, useState } from "react";
import type { TranscriptionProviderKind } from "@/bindings";
import { Banner, Field, Select, TextInput, ToggleField } from "../ui";
import { ApiKeyField } from "./ApiKeyField";
import { ChoiceCards, TextAreaSetting, TextSetting } from "./fields";
import { LocalAiPanel } from "./LocalAiPanel";
import { LocationBadge } from "./LocationBadge";
import { useModelList } from "./LlmProviderForm";
import { saveSettings } from "./save";
import {
  LANGUAGE_OPTIONS,
  TRANSCRIPTION_PROVIDERS,
  choiceFor,
  cleanUrl,
  displayHost,
  effectiveEndpoint,
  isEnglishOnlyModel,
  switchTranscriptionProvider,
} from "./settingsMeta";
import { TestConnection } from "./TestConnection";
import { useDraft } from "./useDraft";
import { useSettings } from "./useSettings";
import { validateLanguageCode, validateOptionalBaseUrl, validateRequired } from "./validation";

const OTHER = "__other";

/** Language hint: common languages, auto-detect, or any ISO code. */
export function LanguageSetting() {
  const settings = useSettings();
  const id = useId();
  const otherId = useId();
  const language = settings.transcription.language ?? "";
  const known = LANGUAGE_OPTIONS.some((o) => o.value === language);
  const [otherChosen, setOtherChosen] = useState(false);
  const showOther = otherChosen || !known;
  const draft = useDraft<string>({
    value: known ? "" : language,
    validate: validateLanguageCode,
    commit: (v) =>
      saveSettings((d) => {
        d.transcription.language = v.trim().toLowerCase();
      }),
  });
  const options = [...LANGUAGE_OPTIONS, { value: OTHER, label: "Other (language code)…" }];
  return (
    <Field
      label="Spoken language"
      description="Telling the recognizer the language improves accuracy and speed."
      htmlFor={id}
      error={showOther ? draft.error : null}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Select
          id={id}
          value={showOther ? OTHER : language}
          options={options}
          className="w-56"
          onChange={(v) => {
            if (v === OTHER) {
              setOtherChosen(true);
              return;
            }
            setOtherChosen(false);
            void saveSettings((d) => {
              d.transcription.language = v === "" ? null : v;
            });
          }}
        />
        {showOther && (
          <>
            <label htmlFor={otherId} className="sr-only">
              Language code
            </label>
            <TextInput
              id={otherId}
              value={draft.value}
              placeholder="e.g. fi"
              maxLength={3}
              className="w-24 font-mono"
              aria-invalid={draft.error ? true : undefined}
              onChange={(e) => draft.onChange(e.target.value)}
              onBlur={draft.onBlur}
            />
          </>
        )}
      </div>
    </Field>
  );
}

export function TranscriptionProviderForm({ variant = "full" }: { variant?: "full" | "compact" }) {
  const settings = useSettings();
  const t = settings.transcription;
  const choice = choiceFor(TRANSCRIPTION_PROVIDERS, t.provider);
  const models = useModelList("transcription", t.provider);
  const ep = effectiveEndpoint(settings, "transcription");
  const urlRequired = t.provider === "openai_compatible";
  const showBaseUrl = variant === "full" || urlRequired;
  const language = t.language ?? "";

  const selectProvider = (next: TranscriptionProviderKind) => {
    if (next === t.provider) return;
    void saveSettings((d) => {
      d.transcription = switchTranscriptionProvider(d.transcription, next);
    });
  };

  return (
    <div>
      <ChoiceCards
        label="Transcription provider"
        value={t.provider}
        onChange={selectProvider}
        options={TRANSCRIPTION_PROVIDERS.map((p) => ({
          value: p.value,
          label: p.label,
          description: p.description,
          location: p.location,
        }))}
      />

      <div className="flex items-center gap-2 py-1 text-[0.8rem] text-muted" aria-live="polite">
        <span>Audio goes to</span>
        {ep.url ? <span className="font-mono text-fg-soft">{displayHost(ep.url)}</span> : <span>—</span>}
        <LocationBadge location={ep.location} endpoint={ep.url ? displayHost(ep.url) : null} />
      </div>

      {settings.privacy.localOnly && ep.location === "remote" && (
        <Banner tone="warn" title="Blocked by local-only mode">
          This provider sends audio off this device, so OpenMeet will refuse to use it while local-only mode is on.
        </Banner>
      )}
      {urlRequired && !t.baseUrl && (
        <Banner tone="warn">Enter the server address below — transcription can't start without it.</Banner>
      )}

      {t.provider === "local_ai" && <LocalAiPanel compact={variant === "compact"} />}

      {showBaseUrl && (
        <TextSetting
          label={urlRequired ? "Server address" : "Custom endpoint (optional)"}
          description={
            urlRequired
              ? "Base URL of a server implementing /v1/audio/transcriptions."
              : `Leave empty to use ${choice.defaultBaseUrl ?? "the default"}.`
          }
          value={t.baseUrl ?? ""}
          placeholder={choice.defaultBaseUrl ?? "http://localhost:8000/v1"}
          type="url"
          monospace
          validate={(v) => validateOptionalBaseUrl(v, urlRequired)}
          onCommit={(v) =>
            saveSettings((d) => {
              d.transcription.baseUrl = cleanUrl(v);
            })
          }
        />
      )}

      <TextSetting
        label="Model"
        description={
          models.error ? (
            <span className="text-danger">Could not load models: {models.error}</span>
          ) : (
            (models.status ??
            (t.provider === "local_ai"
              ? "A faster-whisper model: tiny, base, small, medium (add .en for English-only)."
              : "The transcription model name."))
          )
        }
        value={t.model}
        list={models.listId}
        monospace
        validate={(v) => validateRequired(v, "A model")}
        onCommit={(v) =>
          saveSettings((d) => {
            d.transcription.model = v.trim();
          })
        }
        addon={models.button}
      />
      {models.datalist}

      <LanguageSetting />
      {isEnglishOnlyModel(t.model) && language !== "en" && (
        <Banner tone="warn">
          {t.model} only understands English. Choose a multilingual model (e.g. without “.en”) for other languages.
        </Banner>
      )}

      {choice.secret && <ApiKeyField provider={choice.secret} optional={t.provider === "openai_compatible"} />}

      {variant === "full" && (
        <>
          <ToggleField
            label="Interim results"
            description="Show words while someone is still speaking. With cloud providers this costs extra requests."
            checked={t.interimResults}
            onChange={(v) =>
              void saveSettings((d) => {
                d.transcription.interimResults = v;
              })
            }
          />
          <TextAreaSetting
            label="Vocabulary"
            description="Names, products and jargon the recognizer should expect, separated by commas."
            value={t.vocabulary}
            maxLength={1000}
            rows={3}
            placeholder="e.g. Kubernetes, PostgreSQL, Priya Raman"
            onCommit={(v) =>
              saveSettings((d) => {
                d.transcription.vocabulary = v;
              })
            }
          />
        </>
      )}

      <TestConnection component="transcription" />
    </div>
  );
}
