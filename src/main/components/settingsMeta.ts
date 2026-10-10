// Static metadata for the settings UI: provider choices with their Local /
// Cloud classification, enum labels and the backend's default values. The
// endpoint helpers mirror `openmeet-providers::factory` so forms can label the
// *effective* endpoint before a test call is made.

import type {
  AnswerStyle,
  Component,
  DataLocation,
  Density,
  DisplayLayout,
  EmbeddingProviderKind,
  LlmProviderKind,
  LlmSettings,
  Mode,
  OverlayPosition,
  ReasoningEffort,
  ResponseLength,
  RetentionPolicy,
  ScreenFrequency,
  SecretProvider,
  Settings,
  Theme,
  TranscriptionProviderKind,
  VisionProviderKind,
} from "@/bindings";

export const DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434";
export const DEFAULT_LOCAL_AI_URL = "http://127.0.0.1:8765";
export const DEFAULT_OPENAI_URL = "https://api.openai.com";
export const DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com";

const WELL_KNOWN_DEFAULTS = [
  DEFAULT_OPENAI_URL,
  DEFAULT_ANTHROPIC_URL,
  DEFAULT_OLLAMA_URL,
  DEFAULT_LOCAL_AI_URL,
];

export const DEFAULT_LLM_MODEL: Record<LlmProviderKind, string> = {
  openai: "gpt-5-mini",
  anthropic: "claude-opus-5-5",
  ollama: "llama3.2",
  openai_compatible: "local-model",
};

export const DEFAULT_TRANSCRIPTION_MODEL: Record<TranscriptionProviderKind, string> = {
  local_ai: "base.en",
  openai: "gpt-4o-mini-transcribe",
  openai_compatible: "whisper-1",
};

export const DEFAULT_EMBEDDING_MODEL: Record<EmbeddingProviderKind, string | null> = {
  local_hash: null,
  local_ai: null,
  openai: "text-embedding-3-small",
  ollama: "nomic-embed-text",
};

/** Where a provider choice processes data, before looking at its address. */
export type ChoiceLocation = "local" | "cloud" | "configurable" | "off";

export interface ProviderChoice<T extends string> {
  value: T;
  label: string;
  description: string;
  location: ChoiceLocation;
  /** API key used by this provider (if any). */
  secret: SecretProvider | null;
  /** Built-in endpoint; null when the user must enter one. */
  defaultBaseUrl: string | null;
}

export const LLM_PROVIDERS: ReadonlyArray<ProviderChoice<LlmProviderKind>> = [
  {
    value: "ollama",
    label: "Ollama",
    description: "Runs open models on this computer. Recommended: nothing leaves your device.",
    location: "local",
    secret: null,
    defaultBaseUrl: DEFAULT_OLLAMA_URL,
  },
  {
    value: "openai",
    label: "OpenAI",
    description:
      "GPT models through the OpenAI API. Requires an API key; meeting context is sent to OpenAI.",
    location: "cloud",
    secret: "openai",
    defaultBaseUrl: DEFAULT_OPENAI_URL,
  },
  {
    value: "anthropic",
    label: "Anthropic",
    description:
      "Claude models through the Anthropic API. Requires an API key; meeting context is sent to Anthropic.",
    location: "cloud",
    secret: "anthropic",
    defaultBaseUrl: DEFAULT_ANTHROPIC_URL,
  },
  {
    value: "openai_compatible",
    label: "OpenAI-compatible server",
    description:
      "LM Studio, llama.cpp, vLLM or any server with the OpenAI chat API. Local or cloud depending on its address.",
    location: "configurable",
    secret: "openai_compatible",
    defaultBaseUrl: null,
  },
];

export const TRANSCRIPTION_PROVIDERS: ReadonlyArray<ProviderChoice<TranscriptionProviderKind>> = [
  {
    value: "local_ai",
    label: "Local AI (faster-whisper)",
    description:
      "Speech recognition on this computer through the optional OpenMeet local AI sidecar.",
    location: "local",
    secret: null,
    defaultBaseUrl: DEFAULT_LOCAL_AI_URL,
  },
  {
    value: "openai",
    label: "OpenAI",
    description: "OpenAI speech-to-text. Requires an API key; meeting audio is sent to OpenAI.",
    location: "cloud",
    secret: "openai",
    defaultBaseUrl: DEFAULT_OPENAI_URL,
  },
  {
    value: "openai_compatible",
    label: "OpenAI-compatible server",
    description:
      "Any server implementing /v1/audio/transcriptions. Local or cloud depending on its address.",
    location: "configurable",
    secret: "openai_compatible",
    defaultBaseUrl: null,
  },
];

export const VISION_PROVIDERS: ReadonlyArray<ProviderChoice<VisionProviderKind>> = [
  {
    value: "disabled",
    label: "Off",
    description: "Screen frames are not interpreted.",
    location: "off",
    secret: null,
    defaultBaseUrl: null,
  },
  {
    value: "ocr_only",
    label: "Text recognition only",
    description:
      "Reads on-screen text with the operating system's built-in OCR where available. Nothing leaves this device.",
    location: "local",
    secret: null,
    defaultBaseUrl: null,
  },
  {
    value: "ollama",
    label: "Ollama",
    description: "A vision-capable model running on this computer.",
    location: "local",
    secret: null,
    defaultBaseUrl: DEFAULT_OLLAMA_URL,
  },
  {
    value: "openai",
    label: "OpenAI",
    description: "Screen frames are sent to OpenAI for interpretation.",
    location: "cloud",
    secret: "openai",
    defaultBaseUrl: DEFAULT_OPENAI_URL,
  },
  {
    value: "anthropic",
    label: "Anthropic",
    description: "Screen frames are sent to Anthropic for interpretation.",
    location: "cloud",
    secret: "anthropic",
    defaultBaseUrl: DEFAULT_ANTHROPIC_URL,
  },
  {
    value: "openai_compatible",
    label: "OpenAI-compatible server",
    description: "A vision-capable model behind an OpenAI-compatible API.",
    location: "configurable",
    secret: "openai_compatible",
    defaultBaseUrl: null,
  },
];

export const EMBEDDING_PROVIDERS: ReadonlyArray<ProviderChoice<EmbeddingProviderKind>> = [
  {
    value: "local_hash",
    label: "Built-in",
    description:
      "Fast keyword-style embeddings computed inside OpenMeet. No model or server needed.",
    location: "local",
    secret: null,
    defaultBaseUrl: null,
  },
  {
    value: "local_ai",
    label: "Local AI sidecar",
    description: "Semantic embeddings from the OpenMeet local AI sidecar on this computer.",
    location: "local",
    secret: null,
    defaultBaseUrl: DEFAULT_LOCAL_AI_URL,
  },
  {
    value: "ollama",
    label: "Ollama",
    description: "An embedding model served by Ollama.",
    location: "local",
    secret: null,
    defaultBaseUrl: DEFAULT_OLLAMA_URL,
  },
  {
    value: "openai",
    label: "OpenAI",
    description: "OpenAI embeddings. Transcript snippets are sent to OpenAI.",
    location: "cloud",
    secret: "openai",
    defaultBaseUrl: DEFAULT_OPENAI_URL,
  },
];

export function choiceFor<T extends string>(
  choices: ReadonlyArray<ProviderChoice<T>>,
  value: T,
): ProviderChoice<T> {
  const found = choices.find((c) => c.value === value);
  if (found) return found;
  const first = choices[0];
  if (!first) throw new Error("empty provider list");
  return first;
}

// ---------------------------------------------------------------------------
// Endpoints (mirror of openmeet-providers::factory)
// ---------------------------------------------------------------------------

/** Trimmed URL without trailing slashes; null when empty. */
export function cleanUrl(url: string | null | undefined): string | null {
  if (url === null || url === undefined) return null;
  const t = url.trim().replace(/\/+$/, "");
  return t === "" ? null : t;
}

/** "local" for loopback hosts, "remote" for everything else (incl. LAN hosts). */
export function locationForUrl(url: string): "local" | "remote" {
  const rest = url.includes("://") ? url.slice(url.indexOf("://") + 3) : url;
  const authority = rest.split("/")[0] ?? "";
  const hostPart = authority.includes("@")
    ? authority.slice(authority.lastIndexOf("@") + 1)
    : authority;
  let host: string;
  if (hostPart.startsWith("[")) {
    host = hostPart.slice(1).split("]")[0] ?? "";
  } else {
    host = hostPart.split(":")[0] ?? "";
  }
  host = host.toLowerCase();
  const loopback =
    host === "localhost" ||
    host === "::1" ||
    host.startsWith("127.") ||
    host.endsWith(".localhost");
  return loopback ? "local" : "remote";
}

/** Host (and port) of a URL for display. */
export function displayHost(url: string): string {
  const rest = url.includes("://") ? url.slice(url.indexOf("://") + 3) : url;
  const authority = rest.split("/")[0] ?? "";
  return authority.slice(authority.lastIndexOf("@") + 1);
}

/**
 * The URL a provider actually uses: the configured one, unless it is another
 * provider's well-known default left over from switching providers.
 */
export function resolvedBaseUrl(defaultUrl: string | null, configured: string | null): string {
  const url = cleanUrl(configured);
  if (url && defaultUrl) {
    const stale = WELL_KNOWN_DEFAULTS.some(
      (d) => d !== defaultUrl && d.toLowerCase() === url.toLowerCase(),
    );
    return stale ? defaultUrl : url;
  }
  if (url) return url;
  return defaultUrl ?? "";
}

export interface EffectiveEndpoint {
  /** Resolved URL; "" when the component needs an address that is not set. */
  url: string;
  location: DataLocation | "unset";
}

function endpoint(url: string): EffectiveEndpoint {
  if (!url) return { url: "", location: "unset" };
  return { url, location: locationForUrl(url) };
}

const VISION_AS_LLM: Partial<Record<VisionProviderKind, LlmProviderKind>> = {
  openai: "openai",
  anthropic: "anthropic",
  ollama: "ollama",
  openai_compatible: "openai_compatible",
};

/** Effective endpoint and location of an AI component under `settings`. */
export function effectiveEndpoint(
  settings: Settings,
  component: Extract<Component, "llm" | "transcription" | "vision" | "embeddings">,
): EffectiveEndpoint {
  switch (component) {
    case "llm": {
      const llm = settings.ai.llm;
      return endpoint(
        resolvedBaseUrl(choiceFor(LLM_PROVIDERS, llm.provider).defaultBaseUrl, llm.baseUrl),
      );
    }
    case "transcription": {
      const t = settings.transcription;
      return endpoint(
        resolvedBaseUrl(choiceFor(TRANSCRIPTION_PROVIDERS, t.provider).defaultBaseUrl, t.baseUrl),
      );
    }
    case "vision": {
      const v = settings.ai.vision;
      const kind = VISION_AS_LLM[v.provider];
      if (!kind) return { url: "", location: v.provider === "disabled" ? "disabled" : "local" };
      const configured =
        cleanUrl(v.baseUrl) ??
        (settings.ai.llm.provider === kind ? cleanUrl(settings.ai.llm.baseUrl) : null);
      return endpoint(resolvedBaseUrl(choiceFor(LLM_PROVIDERS, kind).defaultBaseUrl, configured));
    }
    case "embeddings": {
      const e = settings.ai.embeddings;
      if (e.provider === "local_hash") return { url: "", location: "local" };
      let inherited: string | null = null;
      if (e.provider === "ollama" && settings.ai.llm.provider === "ollama")
        inherited = cleanUrl(settings.ai.llm.baseUrl);
      if (e.provider === "local_ai" && settings.transcription.provider === "local_ai")
        inherited = cleanUrl(settings.transcription.baseUrl);
      const configured = cleanUrl(e.baseUrl) ?? inherited;
      return endpoint(
        resolvedBaseUrl(choiceFor(EMBEDDING_PROVIDERS, e.provider).defaultBaseUrl, configured),
      );
    }
  }
}

/**
 * New LLM settings after choosing another provider: the provider's default
 * model and endpoint; the fast model is cleared (it belongs to the old
 * provider). For an OpenAI-compatible server a local address is kept (e.g.
 * Ollama also serves the OpenAI API); otherwise `baseUrl` is null and the
 * user must enter one before the change can be saved.
 */
export function switchLlmProvider(current: LlmSettings, next: LlmProviderKind): LlmSettings {
  if (current.provider === next) return current;
  const choice = choiceFor(LLM_PROVIDERS, next);
  // Cloud providers use their built-in endpoint unless overridden.
  let baseUrl: string | null = next === "ollama" ? choice.defaultBaseUrl : null;
  if (next === "openai_compatible") {
    const kept = cleanUrl(current.baseUrl);
    baseUrl = kept && locationForUrl(kept) === "local" ? kept : null;
  }
  return {
    ...current,
    provider: next,
    model: DEFAULT_LLM_MODEL[next],
    baseUrl,
    fastModel: null,
  };
}

// ---------------------------------------------------------------------------
// Enum labels & options
// ---------------------------------------------------------------------------

export const MODE_OPTIONS: ReadonlyArray<{ value: Mode; label: string; description: string }> = [
  {
    value: "candidate",
    label: "Candidate",
    description:
      "You are being interviewed. OpenMeet detects questions and suggests grounded answers.",
  },
  {
    value: "interviewer",
    label: "Interviewer",
    description:
      "You are interviewing. OpenMeet suggests follow-up questions and tracks covered topics.",
  },
];

export const ANSWER_STYLE_OPTIONS: ReadonlyArray<{ value: AnswerStyle; label: string }> = [
  { value: "concise", label: "Concise" },
  { value: "detailed", label: "Detailed" },
  { value: "technical", label: "Technical" },
  { value: "simple", label: "Simple" },
  { value: "star", label: "STAR" },
  { value: "example_driven", label: "Example-driven" },
  { value: "step_by_step", label: "Step-by-step" },
];

export const RESPONSE_LENGTH_OPTIONS: ReadonlyArray<{ value: ResponseLength; label: string }> = [
  { value: "short", label: "Short" },
  { value: "medium", label: "Medium" },
  { value: "long", label: "Long" },
];

export const REASONING_EFFORT_OPTIONS: ReadonlyArray<{ value: ReasoningEffort; label: string }> = [
  { value: "low", label: "Low — fastest" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High — slowest" },
];

export const SCREEN_FREQUENCY_OPTIONS: ReadonlyArray<{ value: ScreenFrequency; label: string }> = [
  { value: "manual", label: "Manual (hotkey or button only)" },
  { value: "fps1", label: "1 frame per second" },
  { value: "fps2", label: "2 frames per second" },
  { value: "fps5", label: "5 frames per second" },
];

export const RETENTION_OPTIONS: ReadonlyArray<{
  value: RetentionPolicy;
  label: string;
  description: string;
}> = [
  {
    value: "none",
    label: "Don't keep meetings",
    description: "Transcripts and summaries are deleted when a meeting ends.",
  },
  {
    value: "one_day",
    label: "1 day",
    description: "Meetings are deleted 24 hours after they end.",
  },
  {
    value: "seven_days",
    label: "7 days",
    description: "Meetings are deleted a week after they end.",
  },
  {
    value: "thirty_days",
    label: "30 days",
    description: "Meetings are deleted 30 days after they end.",
  },
  {
    value: "forever",
    label: "Keep until I delete them",
    description: "Meetings stay on this device until you delete them.",
  },
];

export const THEME_OPTIONS: ReadonlyArray<{ value: Theme; label: string }> = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
  { value: "system", label: "System" },
];

export const DENSITY_OPTIONS: ReadonlyArray<{ value: Density; label: string }> = [
  { value: "comfortable", label: "Comfortable" },
  { value: "compact", label: "Compact" },
];

export const LAYOUT_OPTIONS: ReadonlyArray<{
  value: DisplayLayout;
  label: string;
  description: string;
}> = [
  {
    value: "same_display",
    label: "Same display",
    description: "The overlay floats over the meeting window.",
  },
  {
    value: "separate_display",
    label: "Separate display",
    description: "The overlay lives on another monitor, away from the shared screen.",
  },
];

export const POSITION_LABELS: Record<OverlayPosition, string> = {
  top_left: "Top left",
  top_right: "Top right",
  bottom_left: "Bottom left",
  bottom_right: "Bottom right",
  custom: "Custom (where you dragged it)",
};

export const COMPONENT_LABELS: Record<Component, string> = {
  meeting: "Meeting",
  microphone: "Microphone",
  system_audio: "System audio",
  transcription: "Transcription",
  question_detector: "Question detection",
  llm: "Language model",
  vision: "Vision",
  screen: "Screen capture",
  embeddings: "Embeddings",
  database: "Database",
  overlay: "Overlay",
  capture_privacy: "Capture privacy",
  hotkeys: "Global shortcuts",
};

export interface LanguageOption {
  value: string;
  label: string;
}

/** Common transcription languages (ISO-639-1). "" = auto-detect. */
export const LANGUAGE_OPTIONS: ReadonlyArray<LanguageOption> = [
  { value: "", label: "Auto-detect" },
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "nl", label: "Dutch" },
  { value: "sv", label: "Swedish" },
  { value: "pl", label: "Polish" },
  { value: "uk", label: "Ukrainian" },
  { value: "ru", label: "Russian" },
  { value: "tr", label: "Turkish" },
  { value: "ar", label: "Arabic" },
  { value: "hi", label: "Hindi" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
];

/** True for Whisper-style English-only model names ("base.en", "small.en"). */
export function isEnglishOnlyModel(model: string): boolean {
  return /\.en$/i.test(model.trim());
}

/**
 * New transcription settings after choosing another provider: the provider's
 * default model and endpoint. An OpenAI-compatible server keeps a configured
 * address unless it is the OpenAI default.
 */
export function switchTranscriptionProvider(
  current: Settings["transcription"],
  next: TranscriptionProviderKind,
): Settings["transcription"] {
  if (current.provider === next) return current;
  let baseUrl: string | null = null;
  if (next === "local_ai") baseUrl = DEFAULT_LOCAL_AI_URL;
  if (next === "openai_compatible") {
    const kept = cleanUrl(current.baseUrl);
    baseUrl = kept && kept.toLowerCase() !== DEFAULT_OPENAI_URL ? kept : null;
  }
  return { ...current, provider: next, model: DEFAULT_TRANSCRIPTION_MODEL[next], baseUrl };
}
