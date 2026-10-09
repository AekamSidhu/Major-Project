# AI providers

OpenMeet separates *what* it does (detect questions, assemble context, suggest
answers and follow-ups) from *who* runs the models. Every model call goes
through a provider trait in `openmeet-core::providers`; adapters are
interchangeable and can be mixed (e.g. local transcription + cloud LLM).

| Component | Trait | Built-in providers |
|-----------|-------|--------------------|
| LLM (answers, follow-ups, summaries, classification) | `LlmProvider` (streaming) | **Ollama** (default, local), OpenAI, Anthropic, OpenAI-compatible |
| Transcription | `Transcriber` → wrapped into a streaming `TranscriptionProvider` by `VadTranscriptionProvider` | **Local AI sidecar** (faster-whisper, default), OpenAI, OpenAI-compatible |
| Screen understanding | `OcrEngine`, `VisionProvider` | **OCR only** (Windows.Media.Ocr, default), Ollama / OpenAI / Anthropic / compatible vision models |
| Embeddings (retrieval) | `EmbeddingProvider` | **Built-in hashing embeddings** (default), Local AI sidecar (fastembed), Ollama, OpenAI |

Every provider reports a `ProviderInfo` with its **location**, computed from
the actual endpoint: `localhost`, `127.0.0.0/8` and `::1` are *Local*; anything
else — including an Ollama server on another machine in your LAN — is
*Remote*. **Local-only mode** (Settings → Privacy) refuses to construct any
remote provider (`BlockedByLocalOnly`), and the Privacy page's data-flow report
is built from these locations.

Configuration lives in Settings → **AI providers** and **Transcription & audio**.
API keys are stored in the OS credential store (never in the database) and can
also come from environment variables for development (see `.env.example`).

---

## LLM providers

### Ollama (local, default)

* Install [Ollama](https://ollama.com), then pull a model, e.g.
  `ollama pull llama3.2` (fast, small) or `ollama pull qwen2.5:7b` /
  `llama3.1:8b` (better answers, needs more RAM/VRAM).
* Base URL: `http://127.0.0.1:11434` (default). With `/v1` or `/api` suffixes
  also accepted.
* Streaming via `POST /api/chat` (NDJSON). `temperature` and `num_predict` are
  passed as options; reasoning ("thinking") output of reasoning models is never
  shown; images are sent for vision-capable models (e.g. `llava`,
  `llama3.2-vision`, `qwen2.5vl`).
* *Load models* in settings lists installed models (`GET /api/tags`).

### OpenAI

* Key: Settings → AI providers → OpenAI key (or `OPENAI_API_KEY`).
* Endpoint: `https://api.openai.com/v1/chat/completions` (streaming, with usage).
* Reasoning models (`gpt-5*`, `o1*`, `o3*`, `o4*`): no temperature;
  `reasoning_effort` is `minimal` for live suggestions on `gpt-5*` (lowest
  latency) and the configured effort otherwise; extra output headroom so the
  visible answer is not truncated. Other models use `temperature`.
* Uses `max_completion_tokens` on api.openai.com, `max_tokens` on compatible
  servers. Images are sent as data URLs.
* Choose the model in settings (*Load models* lists what your key can use).

### Anthropic

* Key: Settings → AI providers → Anthropic key (or `ANTHROPIC_API_KEY`).
* Endpoint: `POST https://api.anthropic.com/v1/messages` (raw HTTP; there is no
  official Rust SDK), `anthropic-version: 2023-06-01`, server-sent events.
* Default model: `claude-opus-5-5`. Model-specific request rules are applied
  automatically:
  * sampling parameters (`temperature`/`top_p`/`top_k`) are not sent to models
    that reject them (Opus 4.7+, Opus 5.x, Sonnet 5.x, Fable, Mythos);
  * `thinking` is never sent; on models where thinking is always on, extra
    output headroom is added and only visible text is streamed;
  * `output_config.effort` is sent only to models that support it — `low` for
    live, latency-sensitive suggestions;
  * **refusal fallbacks** are enabled for `claude-opus-5*`, `claude-fable-5*`
    and `claude-sonnet-5-5` (`anthropic-beta: server-side-fallback-2026-07-01`,
    `"fallbacks": "default"`): if a model's safety classifier declines a
    request, the API re-runs it on Anthropic's recommended fallback model; a
    final `stop_reason: "refusal"` is shown as "The model declined to answer
    this request."
* Images are sent as base64 image blocks before the text.

### OpenAI-compatible

Any server implementing `POST /v1/chat/completions` with streaming — LM Studio,
llama.cpp server, vLLM, LocalAI, OpenRouter, Together, Groq, Azure OpenAI
proxies, etc. Set the base URL (with or without `/v1`) and the model name; an
API key is optional (`OPENAI_COMPATIBLE_API_KEY`). A server on localhost is
*Local*; anything else is *Remote*.

### Fast model (optional)

Settings → AI providers → *Fast model*: a cheaper/faster model on the same
provider for background work (semantic question classification, context
compression, topic analysis). Live answers always use the main model.

---

## Transcription providers

All transcription providers are utterance-based. OpenMeet's own voice activity
detection cuts speech into utterances; each is sent as a 16 kHz mono WAV, and
results are emitted in order with timestamps and speaker role (microphone =
you, system audio = other participants). Optional interim results transcribe
long utterances while they are still being spoken (more requests).

### Local AI sidecar (local, default)

A small Python service in `local-ai/` (faster-whisper + fastembed) running in
its own virtual environment, listening on `127.0.0.1:8765`, exposing
OpenAI-compatible endpoints.

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup-local-ai.ps1 -DownloadModels
```
```bash
# macOS / Linux
scripts/setup-local-ai.sh --download-models
```

* OpenMeet starts the sidecar automatically when a meeting needs it (or use
  *Start* in Settings → Transcription & audio) and stops it on exit.
* Models: `tiny.en`, `base.en` (default), `small.en`, `medium.en`,
  `distil-*`, `large-v3`, `large-v3-turbo`… — larger is more accurate and
  slower. `OPENMEET_WHISPER_DEVICE=auto|cpu|cuda` (auto uses an NVIDIA GPU when
  the CUDA runtime is installed and silently falls back to the CPU otherwise).
* Measured during development (Windows PC, CPU, int8, `base.en`): ~1 s to
  transcribe a 4–5 s utterance once the model is loaded.
* Nothing leaves the device; models are cached in `local-ai/.models/`.

### OpenAI

`POST /v1/audio/transcriptions` (multipart WAV). Default model
`gpt-4o-mini-transcribe`; `whisper-1` and `gpt-4o-transcribe` also work. A
vocabulary hint (names, jargon) can be set in settings.

### OpenAI-compatible

Any server implementing `/v1/audio/transcriptions` (e.g. faster-whisper-server,
speaches, LocalAI, Groq). Set base URL and model.

---

## Screen understanding

1. **Local OCR** — Windows.Media.Ocr (built into Windows 10/11, uses your
   installed OCR language packs). Produces structured observations (content
   type such as code / terminal / diagram / document / slides, recognized text,
   entities) without any model.
2. **Vision models** — Ollama (`llava`, `llama3.2-vision`, `qwen2.5vl`…),
   OpenAI, Anthropic or compatible vision models receive a downscaled frame
   (≤ 1280 px wide) plus the OCR text, and return a structured observation.
   Calls happen only for frames that changed meaningfully, at most
   *max vision calls per minute* (default 6), plus explicit manual captures.

Screen context is off until you enable it for a meeting.

## Embeddings

* **Built-in hashing embeddings** (default): deterministic feature hashing of
  words, word pairs and character trigrams. Fully local, instant, no model;
  combined with BM25 keyword search it gives solid retrieval within a meeting.
* **Local AI** (fastembed `BAAI/bge-small-en-v1.5`), **Ollama**
  (`nomic-embed-text` by default) or **OpenAI** (`text-embedding-3-small` by
  default) for semantic retrieval.

Vectors are stored with their model id; switching models never mixes vectors.

---

## Choosing a setup

| Goal | Transcription | LLM | Screen | Embeddings |
|------|---------------|-----|--------|------------|
| Fully local (no data leaves the device) | Local AI sidecar | Ollama | OCR only | Built-in or Local AI |
| Best answer quality | OpenAI or Local AI (`small.en`+) | Anthropic / OpenAI | Vision model | OpenAI or Local AI |
| Lowest latency | Local AI `base.en` on GPU, or OpenAI | Small/fast model, low effort | OCR only | Built-in |

## Testing a provider

Settings shows a **Test connection** button per component (latency included).
In developer mode, the Diagnostics page shows per-stage latency (audio →
transcript → detection → first token → complete) for real meetings.

## Adding a provider

1. Implement the relevant trait from `openmeet-core::providers`
   (`LlmProvider::stream` must stream visible text only, end with `Done`, honour
   the `CancellationToken`, and map HTTP errors to `ProviderError`).
2. Compute `ProviderInfo.location` from the real endpoint
   (`openmeet_core::providers::location_for_url`).
3. Add the provider kind to the settings enum in `openmeet-core::settings`
   (regenerate TypeScript bindings with `npm run bindings`) and to
   `openmeet-providers::factory`, enforcing local-only mode before credentials.
4. Add tests against a local mock HTTP server (see
   `crates/openmeet-providers/tests/`): streaming across chunk boundaries,
   error mapping, cancellation, request shape.
5. Document it here and in [PRIVACY.md](../PRIVACY.md) if it sends data
   somewhere new.

The provider traits are the intended seam for a future plugin system
(transcription, LLM, vision, knowledge-source and export plugins).
