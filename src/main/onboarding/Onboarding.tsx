// First-run setup (spec §67): welcome → mode → AI provider → transcription →
// microphone → optional screen context → privacy → finish. Every step can be
// skipped; nothing is captured during setup except an explicit level test.

import { useState, type ReactNode } from "react";
import type { Mode } from "@/bindings";
import { api, errorMessage } from "@/lib/ipc";
import { AudioDevicesError, AudioDevicesStatus, AudioSourceSettings, useAudioDevices } from "../components/AudioDevicePicker";
import { ChoiceCards } from "../components/fields";
import { LlmProviderForm } from "../components/LlmProviderForm";
import { saveSettings, settleSettings } from "../components/save";
import { LLM_PROVIDERS, MODE_OPTIONS, RETENTION_OPTIONS, TRANSCRIPTION_PROVIDERS, choiceFor } from "../components/settingsMeta";
import { TranscriptionProviderForm } from "../components/TranscriptionProviderForm";
import { useSettings } from "../components/useSettings";
import { ScreenSourcePicker } from "../sections/home/ScreenSourcePicker";
import { useMain } from "../store";
import { Banner, Button, Field, Select, ToggleField, cx } from "../ui";

type StepId = "welcome" | "mode" | "ai" | "transcription" | "microphone" | "screen" | "privacy" | "finish";

const STEPS: ReadonlyArray<{ id: StepId; label: string }> = [
  { id: "welcome", label: "Welcome" },
  { id: "mode", label: "Mode" },
  { id: "ai", label: "AI provider" },
  { id: "transcription", label: "Transcription" },
  { id: "microphone", label: "Microphone" },
  { id: "screen", label: "Screen" },
  { id: "privacy", label: "Privacy" },
  { id: "finish", label: "Finish" },
];

function StepTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-4">
      <h1 className="text-[1.35rem] font-semibold tracking-tight text-fg">{title}</h1>
      {children && <p className="mt-1.5 max-w-xl text-[0.88rem] leading-relaxed text-muted">{children}</p>}
    </div>
  );
}

function Welcome() {
  return (
    <>
      <StepTitle title="Welcome to OpenMeet">
        A real-time assistant for interviews and meetings. It listens only when you start a meeting, shows suggestions
        in a small overlay, and never speaks or acts on your behalf.
      </StepTitle>
      <ul className="max-w-xl space-y-2 text-[0.86rem] text-fg-soft">
        <li>• Runs locally by default. Cloud AI providers are optional and clearly labeled.</li>
        <li>• No telemetry or analytics. API keys stay in your operating system's credential store.</li>
        <li>• Suggestions use only what is said in the meeting and what you tell OpenMeet — never invented experience.</li>
        <li>• Screen context is off unless you turn it on.</li>
      </ul>
      <p className="mt-4 max-w-xl text-[0.8rem] text-muted">
        Recording laws differ by place. Tell the other participants when you use transcription or an AI assistant, and
        follow the rules of the interview or meeting you are in.
      </p>
      <p className="mt-2 max-w-xl text-[0.8rem] text-muted">Each step saves as you go and none is required.</p>
    </>
  );
}

function ModeStep() {
  const settings = useSettings();
  const reloadSettings = useMain((s) => s.reloadSettings);
  const [error, setError] = useState<string | null>(null);
  const choose = (m: Mode) =>
    void api.setMode(m).then(
      () => {
        setError(null);
        return reloadSettings();
      },
      (e: unknown) => setError(errorMessage(e)),
    );
  return (
    <>
      <StepTitle title="How will you use OpenMeet?">You can switch at any time with a hotkey or from the overlay.</StepTitle>
      {error && <Banner tone="danger">{error}</Banner>}
      <ChoiceCards label="Mode" hideLabel value={settings.mode} options={MODE_OPTIONS} onChange={choose} />
      {settings.mode === "interviewer" && (
        <p className="mt-2 max-w-xl text-[0.8rem] text-muted">
          Interviewer mode suggests follow-up questions and tracks topics. It does not score candidates or make hiring
          recommendations.
        </p>
      )}
    </>
  );
}

function AiStep() {
  return (
    <>
      <StepTitle title="Choose an AI provider">
        Ollama runs models on this computer, so nothing leaves it — install it from ollama.com and run{" "}
        <code className="font-mono text-fg-soft">ollama pull llama3.2</code>. Cloud providers need an API key and receive
        the parts of the conversation needed for each suggestion.
      </StepTitle>
      <LlmProviderForm variant="compact" />
    </>
  );
}

function TranscriptionStep() {
  return (
    <>
      <StepTitle title="Choose speech-to-text">
        The local AI sidecar transcribes on this computer with Whisper. A cloud provider can be faster on computers
        without a GPU, but receives meeting audio.
      </StepTitle>
      <TranscriptionProviderForm variant="compact" />
    </>
  );
}

function MicrophoneStep() {
  const devices = useAudioDevices();
  const settings = useSettings();
  return (
    <>
      <StepTitle title="Check your microphone">
        Choose the microphone you speak into and run the level test. The test listens for a few seconds and nothing is
        saved.
      </StepTitle>
      <AudioDevicesError error={devices.error} />
      <AudioSourceSettings source="microphone" devices={devices.data} hideToggle />
      <ToggleField
        label="Also capture system audio"
        description="What the other participants say, captured from your speakers or headphones. Needed to hear the interviewer or candidate."
        checked={settings.audio.systemAudioEnabled}
        onChange={(v) =>
          void saveSettings((d) => {
            d.audio.systemAudioEnabled = v;
          })
        }
      />
      <AudioDevicesStatus loader={devices} />
    </>
  );
}

function ScreenStep() {
  const settings = useSettings();
  const reloadSettings = useMain((s) => s.reloadSettings);
  const [want, setWant] = useState(settings.screen.enabled);
  const [error, setError] = useState<string | null>(null);

  const apply = (enabled: boolean, source: Parameters<typeof api.setScreenCapture>[1]) =>
    void api
      .setScreenCapture(enabled, source)
      .then(() => setError(null), (e: unknown) => setError(errorMessage(e)))
      .finally(() => void reloadSettings());

  return (
    <>
      <StepTitle title="Screen context (optional)">
        When on, OpenMeet looks at the window or display you choose during meetings, so suggestions can refer to shared
        code or slides. It stays off unless you turn it on here or when starting a meeting.
      </StepTitle>
      {error && <Banner tone="danger">{error}</Banner>}
      <ToggleField
        label="Use screen context"
        checked={want}
        onChange={(v) => {
          setWant(v);
          if (!v && settings.screen.enabled) apply(false, null);
        }}
      />
      {want && (
        <div className="py-2.5">
          <ScreenSourcePicker
            value={settings.screen.source}
            onChange={(spec, err) => {
              if (!err && spec) apply(true, spec);
            }}
          />
          {settings.screen.enabled && settings.screen.source && (
            <p className="mt-2 text-[0.8rem] text-ok">Screen context will use: {settings.screen.source.label}</p>
          )}
        </div>
      )}
    </>
  );
}

function PrivacyStep() {
  const settings = useSettings();
  const [captureError, setCaptureError] = useState<string | null>(null);
  return (
    <>
      <StepTitle title="Privacy">You can change all of this later in Privacy &amp; data.</StepTitle>
      <Field label="Keep meetings for" description={RETENTION_OPTIONS.find((o) => o.value === settings.privacy.retention)?.description} inline>
        <Select
          value={settings.privacy.retention}
          className="w-56"
          options={RETENTION_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
          onChange={(v) =>
            void saveSettings((d) => {
              d.privacy.retention = v;
            })
          }
        />
      </Field>
      <ToggleField
        label="Hide the overlay from screen capture"
        description="Asks the operating system to leave the overlay out of screen shares and recordings where supported. Not a guarantee — some capture tools ignore it."
        checked={settings.privacy.captureExclusion}
        onChange={(v) =>
          void api.setCaptureExclusion(v).then(
            () => setCaptureError(null),
            (e: unknown) => setCaptureError(errorMessage(e)),
          )
        }
      />
      {captureError && <Banner tone="danger">{captureError}</Banner>}
      <ToggleField
        label="Local-only mode"
        description="Refuse every provider that would send data off this computer."
        checked={settings.privacy.localOnly}
        onChange={(v) =>
          void saveSettings((d) => {
            d.privacy.localOnly = v;
          })
        }
      />
    </>
  );
}

function Finish() {
  const s = useSettings();
  const rows: Array<[string, string]> = [
    ["Mode", s.mode === "candidate" ? "Candidate" : "Interviewer"],
    ["AI provider", `${choiceFor(LLM_PROVIDERS, s.ai.llm.provider).label} · ${s.ai.llm.model}`],
    ["Transcription", `${choiceFor(TRANSCRIPTION_PROVIDERS, s.transcription.provider).label} · ${s.transcription.model}`],
    ["Audio", [s.audio.microphoneEnabled && "Microphone", s.audio.systemAudioEnabled && "System audio"].filter(Boolean).join(" + ") || "Off"],
    ["Screen context", s.screen.enabled && s.screen.source ? s.screen.source.label : "Off"],
    ["Local-only", s.privacy.localOnly ? "On" : "Off"],
  ];
  return (
    <>
      <StepTitle title="You're set">
        Start a meeting from Home when you are ready. Try a demo meeting first to see the overlay in action without
        using your microphone.
      </StepTitle>
      <dl className="max-w-md divide-y divide-border rounded-lg border border-border bg-surface px-4 text-[0.84rem]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4 py-2">
            <dt className="text-muted">{k}</dt>
            <dd className="truncate text-fg">{v}</dd>
          </div>
        ))}
      </dl>
    </>
  );
}

export function Onboarding() {
  const [index, setIndex] = useState(0);
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const step = STEPS[index] ?? STEPS[0]!;
  const last = index === STEPS.length - 1;

  const go = async (next: number) => {
    await settleSettings();
    setIndex(Math.max(0, Math.min(STEPS.length - 1, next)));
  };

  const finish = async () => {
    setFinishing(true);
    await settleSettings();
    const ok = await saveSettings((d) => {
      d.general.onboardingComplete = true;
    });
    if (!ok) setError("Could not save your setup. Check the highlighted settings and try again.");
    setFinishing(false);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-4 border-b border-border px-8 py-4">
        <span className="text-[0.95rem] font-semibold text-fg">OpenMeet setup</span>
        <ol className="flex flex-1 items-center gap-1.5" aria-label="Setup progress">
          {STEPS.map((s, i) => (
            <li key={s.id} className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => void go(i)}
                aria-current={i === index ? "step" : undefined}
                className={cx(
                  "rounded-full px-2 py-0.5 text-[0.74rem] transition-colors",
                  i === index ? "bg-accent text-accent-fg" : i < index ? "text-fg-soft hover:text-fg" : "text-muted hover:text-fg-soft",
                )}
              >
                {s.label}
              </button>
              {i < STEPS.length - 1 && <span aria-hidden className="h-px w-3 bg-border" />}
            </li>
          ))}
        </ol>
        {!last && (
          <Button variant="ghost" size="sm" onClick={() => void finish()} busy={finishing}>
            Skip setup
          </Button>
        )}
      </header>
      <main className="min-h-0 flex-1 overflow-y-auto px-8 py-8">
        <div className="mx-auto max-w-3xl">
          {error && <Banner tone="danger">{error}</Banner>}
          {step.id === "welcome" && <Welcome />}
          {step.id === "mode" && <ModeStep />}
          {step.id === "ai" && <AiStep />}
          {step.id === "transcription" && <TranscriptionStep />}
          {step.id === "microphone" && <MicrophoneStep />}
          {step.id === "screen" && <ScreenStep />}
          {step.id === "privacy" && <PrivacyStep />}
          {step.id === "finish" && <Finish />}
        </div>
      </main>
      <footer className="flex items-center justify-between border-t border-border px-8 py-4">
        <Button variant="ghost" onClick={() => void go(index - 1)} disabled={index === 0}>
          Back
        </Button>
        <div className="flex items-center gap-2">
          {last ? (
            <Button variant="primary" onClick={() => void finish()} busy={finishing}>
              Finish setup
            </Button>
          ) : (
            <Button variant="primary" onClick={() => void go(index + 1)}>
              {index === 0 ? "Get started" : "Next"}
            </Button>
          )}
        </div>
      </footer>
    </div>
  );
}
