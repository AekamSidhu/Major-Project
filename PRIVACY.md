# OpenMeet privacy

OpenMeet processes some of the most sensitive data a computer has: what people
say in meetings and what is on your screen. This document explains, precisely,
what OpenMeet can collect, where it is processed, what is stored, and how to
turn every part of it off or delete it.

**Short version**

* Nothing is captured until **you start a meeting**. Launching OpenMeet (even at
  login) never starts capture.
* **Screen context is off by default** and must be enabled explicitly for a
  meeting, for a source you choose.
* The default configuration is **local-first**: local transcription (faster-whisper
  in a Python virtual environment on your machine), a local LLM through
  [Ollama](https://ollama.com), local OCR (Windows) and local embeddings.
* **No telemetry, no analytics, no crash reporting, no update pings.** OpenMeet
  makes no network requests except to the AI providers *you* configure.
* Data only leaves your device if you configure a cloud provider. The Privacy
  page always shows which components are local and which are remote, and only
  says "No meeting data leaves this device." when that is actually true.
* Meeting data is kept for **7 days by default**, can be kept in memory only,
  and can be deleted with one click.

---

## 1. What OpenMeet can collect

| Data | When | Default |
|------|------|---------|
| Microphone audio | Only while a meeting is active and the microphone source is enabled | On for meetings |
| System / meeting audio (other participants) | Only while a meeting is active and system audio is enabled; Windows (WASAPI loopback), Linux (PulseAudio/PipeWire monitor sources), macOS where supported | On for meetings on Windows |
| Screen frames | Only while a meeting is active **and** you enabled screen context for a specific display / window / application / region, or when you press *Capture screen* | **Off** |
| Text you type into *Ask OpenMeet* | When you submit it | — |
| Your profile notes (background, role, planned topics, instructions) | Only what you type into Settings → Profile | Empty |

OpenMeet does **not**: record keystrokes, inject code into meeting apps, read
other applications' memory, capture audio or screen outside a meeting, or use
undocumented capture techniques.

Audio is never written to disk. It is held in memory only as long as needed to
cut an utterance and transcribe it (seconds).

## 2. What is processed where

```
Microphone / system audio ──▶ Transcription ──▶ Context engine ──▶ AI provider ──▶ Overlay
Screen (opt-in) ────────────▶ OCR / Vision  ──▶ Context engine ──▶ AI provider
```

| Component | Local options | Cloud options |
|-----------|---------------|---------------|
| Transcription | Local AI sidecar (faster-whisper, default) | OpenAI, any OpenAI-compatible endpoint |
| LLM (answers, follow-ups, summaries) | Ollama (default), any OpenAI-compatible server on localhost (LM Studio, llama.cpp, vLLM…) | OpenAI, Anthropic, remote OpenAI-compatible endpoints |
| Screen understanding | Windows OCR (Windows.Media.Ocr), Ollama vision models | OpenAI, Anthropic vision |
| Embeddings (retrieval) | Built-in hashing embeddings (default), local AI sidecar (fastembed), Ollama | OpenAI |
| Context engine, question detection, topic tracking, storage | Always local | — |

A provider is classified as **local** only if its endpoint is on this machine
(`localhost`, `127.0.0.0/8`, `::1`). An Ollama or LM Studio server on another
computer in your network is **remote** — it is labelled as such and blocked in
local-only mode.

### What is sent to a remote AI provider

When you configure a cloud provider, the following can be sent to it, over
HTTPS, with your API key:

* **Transcription provider**: short audio clips (one utterance each, WAV) and a
  short vocabulary/context hint.
* **LLM provider**: the detected question or your *Ask OpenMeet* text, the last
  few minutes of transcript, a compressed summary of earlier conversation,
  retrieved relevant snippets, recent screen observations (text descriptions,
  not images), detected topics/claims, and your profile notes.
* **Vision provider** (only with screen context enabled and a cloud vision
  model selected): downscaled screenshots of the selected source (after privacy
  rules and region masks are applied), rate-limited (default ≤ 6 per minute) and
  only when the screen content changed meaningfully.
* **Embedding provider** (only if you choose a cloud embedding model):
  transcript snippets and observations to compute retrieval vectors.

Each provider's own privacy policy and data-retention terms apply to what it
receives. Check them before using a cloud provider with confidential meetings.

### Local-only mode

Settings → Privacy → **Local-only mode** makes OpenMeet refuse to construct any
provider whose endpoint is remote. Affected components show "Blocked: local-only
mode is on" instead of silently falling back. With local-only mode on and all
components local, the Privacy page states **"No meeting data leaves this
device."** This statement is computed from the actual provider endpoints and is
never shown when any active component is remote.

## 3. What is stored

Storage location (a single SQLite database, `openmeet.db`):

| OS | Path |
|----|------|
| Windows | `%APPDATA%\dev.openmeet.desktop\` |
| macOS | `~/Library/Application Support/dev.openmeet.desktop/` |
| Linux | `$XDG_DATA_HOME/dev.openmeet.desktop/` (usually `~/.local/share/…`) |

Stored per meeting (subject to the retention policy):

* meeting metadata (title, mode, start/end time),
* transcript segments (text, timestamps, speaker role),
* detected questions, the answers given in the meeting, AI suggestions and
  follow-ups,
* topics, extracted claims, action items, the post-meeting summary,
* screen **observations** (structured text descriptions and recognized text —
  not images),
* retrieval embeddings (numeric vectors derived from the above).

Also stored: your settings (including profile notes) and screen privacy rules.

**Never stored**: raw audio, API keys (they live in the OS credential store —
Windows Credential Manager, macOS Keychain, Linux Secret Service — or come from
environment variables), and screenshots — unless you explicitly enable
*Settings → Screen context → Keep screenshots*, in which case downscaled PNG
frames that were analyzed are saved under `screenshots/<meeting id>/` next to
the database and are deleted together with their meeting.

The local AI sidecar keeps its downloaded models in `local-ai/.models/` and
stores no meeting data.

### Retention

Settings → Privacy → **Data retention**:

| Policy | Effect |
|--------|--------|
| Do not retain transcripts | Nothing about meetings is written to disk; the meeting lives in memory until the next meeting or app exit (you can still export it before then) |
| 1 day / **7 days (default)** / 30 days | Meetings older than this are deleted automatically (checked at start-up and hourly) |
| Keep until deleted | Kept until you delete them |

Deleting uses SQLite `secure_delete` and checkpoints the write-ahead log so
removed content is overwritten rather than left in free pages.

## 4. How to delete data

* One meeting: History → meeting → **Delete**.
* Everything: Privacy → **Delete all meeting data** (keeps settings and privacy
  rules). This removes all meetings, transcripts, suggestions, summaries,
  observations, embeddings and kept screenshots, then compacts the database.
* API keys: Settings → AI providers → **Remove** next to each key (removes it
  from the OS credential store).
* Complete removal: uninstall OpenMeet and delete the data folder listed above
  (and `local-ai/.venv` + `local-ai/.models` if you set up local AI).

## 5. Turning things off

You never have to hunt through settings to stop processing:

| Goal | How |
|------|-----|
| Stop everything immediately | **Emergency stop**: `Ctrl+Shift+Backspace` (Windows) / `Cmd+Shift+Esc` (macOS) / `Ctrl+Shift+Esc` (Linux), tray → *Emergency Stop*, or the Home page button. Stops audio capture, screen capture and all AI requests, cancels pending work and hides the overlay; shows **PROCESSING STOPPED** until you resume |
| Pause | `Ctrl/Cmd+Shift+P`, tray → *Pause*, overlay ❚❚ — paused meetings release the microphone and screen |
| Disable screen processing | Leave screen context off (default), or Home → live meeting → Screen toggle, or Settings → Screen context |
| Disable audio processing | Start a meeting with microphone and system audio off (manual *Ask OpenMeet* still works), or Settings → Transcription & audio |
| Disable cloud processing | Privacy → **Local-only mode** |
| Switch providers | Settings → AI providers / Transcription |

The main window always shows whether a meeting is capturing ("Meeting in
progress" / "Meeting paused" / "Processing stopped"), and the tray tooltip
reflects the state as well.

## 6. Screen privacy rules

Privacy → **Screen privacy rules** let you exclude content from screen capture:

* **Application** (e.g. `1Password`, `KeePassXC`, your banking app, messaging apps),
* **Window** title substring,
* **Domain** (matched against browser window titles, e.g. `bank.example.com`),
* **Region** (`x,y,width,height` in pixels — blacked out before analysis).

When an excluded application or window is the active window on a captured
display, the frame is skipped entirely. Rules cannot see inside other
applications, so domain rules depend on the browser showing the site in its
window title. OpenMeet's own windows are never captured as a screen source.

## 7. Capture exclusion (hiding the overlay from screen capture) — limitations

OpenMeet can ask the operating system to exclude its windows from screen
capture, using **documented** platform APIs only:

| Platform | Mechanism | Notes |
|----------|-----------|-------|
| Windows 10 2004+ / 11 | `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` | Verified with `GetWindowDisplayAffinity` |
| macOS | `NSWindow.sharingType = .none` | Newer capture APIs (ScreenCaptureKit, macOS 15+) may not honor it for every capture client |
| Linux | — | Not supported; shown as "Not supported on this system" |

This is **not** universal invisibility. It only affects capture software that
uses the OS capture paths that respect the flag. It does **not** affect:
hardware capture devices, cameras or phones pointed at the screen, some
remote-desktop / virtualization software, older or non-standard capture APIs,
accessibility tools, or a remote participant asking you to share your screen
while you move the overlay into view by other means. OpenMeet does not and will
not use undocumented techniques, driver hooks or anything designed to defeat
security, proctoring, DRM or anti-cheat software.

## 8. Logs

Logs contain identifiers, counts, sizes, durations and error kinds only — never
transcript text, prompts, model output, OCR text, screenshots or API keys. Log
verbosity is controlled with the `OPENMEET_LOG` environment variable.

## 9. Responsible use and consent

Recording or transcribing a conversation, or using AI assistance in it, may
require the consent of other participants and may be restricted by law, by your
employer, by the meeting platform's terms, or by the rules of an interview or
exam. **You are responsible for following those rules.** OpenMeet is designed to
make capture visible and controllable, not hidden:

* tell participants when you transcribe or use AI assistance where required,
* do not use OpenMeet where AI assistance is prohibited (for example, an
  interview or exam that forbids it),
* in interviewer mode, OpenMeet helps you ask better questions and take notes —
  it does not score, rank or make hiring decisions about anyone; the human
  decides.

## 10. Questions

See [SECURITY.md](SECURITY.md) to report a security or privacy vulnerability.
