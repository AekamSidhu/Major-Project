# Running OpenMeet on Windows

This guide covers building the desktop app, starting the local AI sidecar, and
running the interview room, on Windows 11. Everything runs on your computer
unless a setting says otherwise. The commands below were run on a Windows 11
machine with the versions listed; other versions may work.

## Prerequisites

| Tool | Version used | Notes |
|---|---|---|
| Windows | 11 (build 26300) | WebView2 is already installed on Windows 11 |
| Rust | 1.99 stable (`rustup`) | 1.89 or newer is required; `CARGO_HOME` may be on another drive |
| Visual Studio 2022 Build Tools | "Desktop development with C++" (MSVC + Windows 11 SDK) | needed by Rust's MSVC toolchain |
| Node.js | 22.21 LTS, npm 11.6 | `npm ci` in the repository root |
| Python | 3.11 (3.10 to 3.12 work; 3.11 is preferred) | for the local AI sidecar and the room only; 3.14 is too new for MediaPipe |
| Ollama (optional) | any recent version, with `llama3.2` | only for the room's `--followups local` mode and Ollama answers |

Keep at least about 15 GB free on the drive with the Cargo target folder. The
release build alone takes several GB.

## First-time setup

```powershell
cd <repository>
npm ci                                                        # frontend tools and Tauri CLI
powershell -ExecutionPolicy Bypass -File scripts\setup-local-ai.ps1   # sidecar venv in local-ai\.venv
```

`setup-local-ai.ps1` picks Python 3.11 (then 3.12, 3.10) through the `py`
launcher, creates `local-ai\.venv`, installs `local-ai\requirements.txt` and
`requirements-integrity.txt`, and warns (without failing) if the gaze packages
(MediaPipe, mss) do not install. Add `-Dev` for the test tools and
`-DownloadModels` to fetch the default models in advance. Without
`-DownloadModels`, the first transcription downloads the Whisper model
(`base.en` by default) once, which needs internet access.

## Build

```powershell
npx tauri build
```

This runs the frontend build, then a release build of the Rust crates, then
creates the installers. On the test machine the Rust release build took about
11 minutes with `CARGO_BUILD_JOBS=4`. Outputs:

- `target\release\openmeet.exe` - the app, run directly (no install needed)
- `target\release\bundle\nsis\OpenMeet_0.1.0_x64-setup.exe` - NSIS installer
- `target\release\bundle\msi\OpenMeet_0.1.0_x64_en-US.msi` - MSI installer

To use a shared build folder, set `CARGO_TARGET_DIR` first; the start script
looks in the same place.

## Run

One command starts the sidecar (if it is installed and not already running)
and then the app:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-openmeet.ps1
```

- `-Dev` runs `npx tauri dev` (Vite and the app from source) instead of the built exe.
- `-NoSidecar` skips the sidecar.
- `-SidecarPort 8765` (default) must match the local AI URL in Settings.

When the app exits, the script stops the sidecar it started. Closing the main
window only hides OpenMeet to the tray; quit it from the tray menu ("Quit
OpenMeet").

To try the app without a microphone, provider or network, use demo mode
(Home → Demo mode). The interviewer integrity demo and the workplace demo are
available from there.

To run the sidecar by hand instead:
`local-ai\.venv\Scripts\python.exe local-ai\server.py --host 127.0.0.1 --port 8765`.
`GET /health` should return `{"status":"ok", ...}`.

## Interview room

The room is a browser-based interview page served from this computer.

```powershell
cd local-ai
.venv\Scripts\python.exe room_server.py --no-https --port 8463 --followups rules
```

- `http://localhost:8463/` redirects to `/host`, where you create a room.
- `POST /api/rooms` returns the room id, a host key, and invite links for the local network.
- `--followups rules` needs no model. `local` uses Ollama, `online` uses Gemini (see `room_server.py --help`).
- Without `--no-https`, the room uses HTTPS on port 8443 with a self-signed certificate; the browser warns once.
- `--tunnel` publishes a public link through Cloudflare. It is off by default.

`start-room.ps1` in the repository root runs the room with its own options
(`-Followups`, `-Transcription`, `-Tunnel`, `-Cloud`). Note that it runs
`python` from the PATH, not `local-ai\.venv`.

## Troubleshooting (problems we hit)

- **`WinError 32 ... websockets\server.py` during pip install.** Two `pip
  install` runs into the same venv at once lock the files. Run one install at a time.
- **Hugging Face warning about symlinks on Windows.** Without Developer Mode,
  the model cache works but uses more disk space. Turn on Developer Mode, or
  set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide the warning.
- **The app says "Local AI is not installed".** The app only looks for
  `local-ai\.venv`. A venv elsewhere must be linked there (for example with
  `mklink /J local-ai\.venv <path>`); the link is not in git.
- **Launching a new build upgrades your database.** The app opens
  `%APPDATA%\dev.openmeet.desktop\openmeet.db` and applies migrations (on the
  test machine, schema 2 to 3). Back up that file before the first run of a new build.
- **Screenshots of the app come out black or show other windows.** WebView2
  draws on the GPU, so `PrintWindow` returns a black image. Use a screen
  capture of a window that is not covered.
- **The sidecar did not start.** Check that port 8765 is free
  (`netstat -ano | findstr :8765`) and that `local-ai\.venv` exists. Running
  `local-ai\server.py` by hand shows the error.

## Where files go

| What | Where |
|---|---|
| Installers and app exe | `target\release\` (or `$CARGO_TARGET_DIR\release\`) |
| App data: meetings database, kept screenshots | `%APPDATA%\dev.openmeet.desktop\` (`openmeet.db`, `screenshots\`) |
| Sidecar virtual environment | `local-ai\.venv\` |
| Sidecar and room models, room certificate | `local-ai\.models\` |
| Room study data (optional, `--study`) | a folder you choose, which must be outside the repository |

Nothing here is committed to git: `local-ai/.venv/`, `local-ai/.models/` and the
target folder are ignored.
