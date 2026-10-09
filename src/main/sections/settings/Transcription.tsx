// Transcription & audio: speech-to-text provider, audio sources with a live
// level test, voice activity detection and assistant behaviour (spec §9–§16,
// §20, §27).

import { AudioDevicesError, AudioDevicesStatus, AudioSourceSettings, useAudioDevices } from "../../components/AudioDevicePicker";
import { NumberSetting, SelectSetting, SliderSetting } from "../../components/fields";
import { saveSettings } from "../../components/save";
import { ANSWER_STYLE_OPTIONS, RESPONSE_LENGTH_OPTIONS } from "../../components/settingsMeta";
import { TranscriptionProviderForm } from "../../components/TranscriptionProviderForm";
import { useSettings } from "../../components/useSettings";
import { PageHeader, Section, ToggleField } from "../../ui";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`;

function AudioSources() {
  const devices = useAudioDevices();
  return (
    <Section
      title="Audio sources"
      description="Nothing is captured until you start a meeting. Use the level test to check that OpenMeet hears the right device."
    >
      <AudioDevicesError error={devices.error} />
      <AudioSourceSettings source="microphone" devices={devices.data} />
      <AudioSourceSettings source="system" devices={devices.data} />
      <AudioDevicesStatus loader={devices} />
    </Section>
  );
}

function SpeechDetection() {
  const audio = useSettings().audio;
  return (
    <Section title="Speech detection" description="How OpenMeet decides that someone started and stopped speaking.">
      <SliderSetting
        label="Sensitivity"
        description="Higher picks up quieter speech but also more background noise."
        value={audio.vadSensitivity}
        min={0}
        max={1}
        step={0.05}
        format={pct}
        onCommit={(v) =>
          saveSettings((d) => {
            d.audio.vadSensitivity = v;
          })
        }
      />
      <SliderSetting
        label="End of utterance"
        description="Pause length that ends an utterance. Shorter is faster; longer avoids splitting sentences."
        value={audio.silenceMs}
        min={300}
        max={3000}
        step={50}
        format={secs}
        onCommit={(v) =>
          saveSettings((d) => {
            d.audio.silenceMs = v;
          })
        }
      />
    </Section>
  );
}

function AssistantBehaviour() {
  const a = useSettings().assistant;
  return (
    <Section title="Assistant behaviour" description="When suggestions appear and how much context they use.">
      <ToggleField
        label="Suggest answers automatically (candidate mode)"
        description="Starts a suggestion as soon as a question to you is detected. Off: press the Answer hotkey."
        checked={a.autoAnswer}
        onChange={(v) =>
          void saveSettings((d) => {
            d.assistant.autoAnswer = v;
          })
        }
      />
      <ToggleField
        label="Suggest follow-ups automatically (interviewer mode)"
        description="Suggests follow-up questions when the candidate finishes an answer."
        checked={a.autoFollowups}
        onChange={(v) =>
          void saveSettings((d) => {
            d.assistant.autoFollowups = v;
          })
        }
      />
      <ToggleField
        label="Confirm unclear questions with the AI model"
        description="Asks the language model about sentences the fast detector is unsure about. More accurate; one extra small request each time."
        checked={a.semanticDetection}
        onChange={(v) =>
          void saveSettings((d) => {
            d.assistant.semanticDetection = v;
          })
        }
      />
      <SliderSetting
        label="Detection confidence"
        description="Minimum confidence before a question triggers an automatic suggestion."
        value={a.detectionThreshold}
        min={0.1}
        max={0.99}
        step={0.01}
        format={pct}
        onCommit={(v) =>
          saveSettings((d) => {
            d.assistant.detectionThreshold = v;
          })
        }
      />
      <SelectSetting
        label="Default answer style"
        value={a.answerStyle}
        options={ANSWER_STYLE_OPTIONS}
        onChange={(v) =>
          void saveSettings((d) => {
            d.assistant.answerStyle = v;
          })
        }
      />
      <SelectSetting
        label="Response length"
        value={a.responseLength}
        options={RESPONSE_LENGTH_OPTIONS}
        onChange={(v) =>
          void saveSettings((d) => {
            d.assistant.responseLength = v;
          })
        }
      />
      <NumberSetting
        label="Recent transcript window"
        description="Seconds of verbatim conversation sent with every request. Older parts are summarized."
        value={a.contextWindowSecs}
        min={30}
        max={900}
        unit="s"
        onCommit={(v) =>
          saveSettings((d) => {
            d.assistant.contextWindowSecs = v;
          })
        }
      />
      <NumberSetting
        label="Context budget"
        description="Approximate tokens of context per request (transcript, profile, screen, retrieved history)."
        value={a.contextTokenBudget}
        min={800}
        max={32000}
        unit="tokens"
        onCommit={(v) =>
          saveSettings((d) => {
            d.assistant.contextTokenBudget = v;
          })
        }
      />
      <NumberSetting
        label="Minimum answer length for follow-ups"
        description="Interviewer mode: candidate answers shorter than this are not analyzed."
        value={a.followupMinWords}
        min={3}
        max={400}
        unit="words"
        onCommit={(v) =>
          saveSettings((d) => {
            d.assistant.followupMinWords = v;
          })
        }
      />
      <SliderSetting
        label="End of the candidate's turn"
        description="Interviewer mode: silence after the candidate's last sentence before follow-ups are suggested."
        value={a.turnEndSilenceMs}
        min={400}
        max={10000}
        step={100}
        format={secs}
        onCommit={(v) =>
          saveSettings((d) => {
            d.assistant.turnEndSilenceMs = v;
          })
        }
      />
    </Section>
  );
}

export function TranscriptionSection() {
  return (
    <>
      <PageHeader
        title="Transcription & audio"
        description="Speech is turned into text by the provider below. With the local AI sidecar, audio never leaves this computer."
      />
      <Section title="Speech-to-text">
        <TranscriptionProviderForm />
      </Section>
      <AudioSources />
      <SpeechDetection />
      <AssistantBehaviour />
    </>
  );
}
