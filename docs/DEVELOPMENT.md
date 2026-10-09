# Developing OpenMeet

## Prerequisites

| | Windows | macOS | Linux |
|---|---|---|---|
| Rust | stable ≥ 1.85 (`rustup`) | same | same |
| Node.js | 22 LTS + npm | same | same |
| Native toolchain | Visual Studio 2022 Build Tools with "Desktop development with C++" (MSVC + Windows 11 SDK) | Xcode Command Line Tools | `build-essential`, `clang` |
| WebView | WebView2 (preinstalled on Windows 10/11) | WKWebView (built in) | `libwebkit2gtk-4.1-dev` |
| Audio / screen | — | — | `libasound2-dev libpulse-dev libxcb1-dev libxrandr-dev libpipewire-0.3-dev libgbm-dev libegl1-mesa-dev libxdo-dev libappindicator3-dev librsvg2-dev libssl-dev libdbus-1-dev pkg-config` |
| Local AI (optional) | Python 3.10–3.12 | same | same |

`RUSTUP_HOME` / `CARGO_HOME` may point anywhere (e.g. a secondary drive); make
sure `$CARGO_HOME/bin` is on `PATH`.

## First run

```bash
npm install
npm run tauri dev        # starts Vite on :1420 and the Tauri app
```

The first Rust build takes a few minutes. The app opens the onboarding flow;
**demo mode** (Home → Demo mode) shows the full experience without any
microphone, provider or network.

### Local AI (local transcription + embeddings)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-local-ai.ps1 -Dev -DownloadModels   # Windows
```
```bash
scripts/setup-local-ai.sh --dev --download-models                                          # macOS/Linux
```

This creates `local-ai/.venv` (an isolated Python virtual environment),
installs faster-whisper, fastembed and the test tools, and downloads the
default models into `local-ai/.models/`. OpenMeet finds the sidecar in the
repository checkout during development; set `OPENMEET_LOCAL_AI_DIR` to use
another location. Run it manually with:

```bash
local-ai/.venv/Scripts/python local-ai/server.py --port 8765   # Windows
local-ai/.venv/bin/python local-ai/server.py --port 8765       # macOS / Linux
```

### Local LLM

Install [Ollama](https://ollama.com) and `ollama pull llama3.2` (or any chat
model). OpenMeet's default LLM setting points to `http://127.0.0.1:11434`.

## Repository layout

```
crates/
  openmeet-core/        domain types, events, traits, settings, IPC DTOs (→ src/bindings)
  openmeet-intel/       VAD, detection, topics, claims, prompts, export, mocks, demo
  openmeet-engine/      real-time orchestration (tokio)
  openmeet-providers/   OpenAI / Anthropic / Ollama / compatible / local AI adapters
  openmeet-platform/    audio & screen capture, OCR, capture privacy, focus, apps
  openmeet-store/       SQLite persistence
src-tauri/              desktop app: windows, overlay, tray, hotkeys, commands
src/
  bindings/             generated TypeScript types (do not edit)
  lib/                  IPC wrappers, answer parser, theme, formatting
  overlay/              overlay window (live HUD)
  main/                 main window (onboarding, settings, privacy, history, diagnostics)
local-ai/               Python sidecar (faster-whisper + fastembed)
docs/                   architecture and contracts
scripts/                setup and maintenance scripts
```

## Everyday commands

```bash
# Rust
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p openmeet-engine --test pipeline      # end-to-end pipeline with mocks

# Frontend
npx tsc -p tsconfig.json --noEmit
npx eslint .
npx vitest run
npx vite build

# TypeScript bindings after changing IPC types in openmeet-core
npm run bindings

# Local AI sidecar tests
cd local-ai && .venv/Scripts/python -m pytest -q   # Windows (.venv/bin/python elsewhere)
```

### Hardware tests

Tests that need real devices are `#[ignore]`d. Run them on a machine with a
microphone, speakers and a display:

```bash
cargo test -p openmeet-platform -- --ignored --nocapture
```

They capture a few seconds of microphone and loopback audio, enumerate and
capture screens, run Windows OCR and toggle capture exclusion on a test window.
They print counts and sizes only, never content.

## Developer mode

Settings → General → **Developer mode** (on by default in debug builds) adds:

* Diagnostics → **Developer tools**: simulate an incoming question, a
  transcript line from any speaker, a candidate answer, a screen change, a
  forced AI response, an AI failure and network latency — everything needed to
  develop the overlay without joining a meeting. Simulated AI responses swap in
  a deterministic local mock provider until *Clear simulation*.
* Per-stage latency statistics and the **benchmark** (detection, VAD
  throughput, embeddings, retrieval, prompt assembly, end-to-end pipeline
  overhead with a mock model, process memory/CPU).

## Logging

`OPENMEET_LOG` takes a `tracing` filter, e.g.
`OPENMEET_LOG=info,openmeet_engine=debug npm run tauri dev`. Logs never contain
transcript text, prompts, model output, OCR text or keys — keep it that way
(reviewers check).

## Data locations during development

* Database: `%APPDATA%\dev.openmeet.desktop\openmeet.db` (Windows),
  `~/Library/Application Support/dev.openmeet.desktop/` (macOS),
  `~/.local/share/dev.openmeet.desktop/` (Linux).
* Reset everything: delete that folder (settings, history, privacy rules).

## Packaging

```bash
npm run tauri build
```

Produces installers in `target/release/bundle/` (NSIS/MSI on Windows, DMG/app
on macOS, deb/AppImage/rpm on Linux). The local AI sidecar is not bundled by
default (it is a Python environment with large models); users install it with
the setup script, or set `OPENMEET_LOCAL_AI_DIR`. Release builds run without a
console window and with developer mode off by default.

The release workflow (`.github/workflows/release.yml`) builds all platforms
with `tauri-action` on version tags. macOS notarization and Windows code
signing require repository secrets (`APPLE_*`, `TAURI_SIGNING_*` / certificate
secrets) — see the Tauri distribution guides.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Local AI is not installed" | Run the setup script above |
| Local transcription slow | Use a smaller model (`base.en`, `tiny.en`), or a GPU with the CUDA runtime (`OPENMEET_WHISPER_DEVICE=cuda`) |
| `cublas64_12.dll is not found` in sidecar logs | Harmless with `OPENMEET_WHISPER_DEVICE=auto` (falls back to CPU); install the CUDA 12 runtime + cuDNN 9 to use the GPU |
| A hotkey does nothing | Settings → Shortcuts shows registration errors (another app owns the keys) and known conflicts; pick another combination |
| No system audio on macOS | Use a virtual loopback device (e.g. BlackHole) as the system audio device |
| Screen capture permission (macOS) | System Settings → Privacy & Security → Screen Recording → enable OpenMeet |
| Overlay steals focus / not on top | Report with your OS and window manager; Linux WMs differ in honouring always-on-top and focus hints |
