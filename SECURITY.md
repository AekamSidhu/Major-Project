# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security or privacy vulnerabilities.

Report privately through GitHub's **[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)**
("Security" tab → "Report a vulnerability") on the OpenMeet repository. Include:

* affected version / commit and platform,
* a description of the issue and its impact,
* steps to reproduce (a minimal proof of concept is welcome — please do not
  include real meeting data or credentials),
* any suggested fix.

We aim to acknowledge reports within **3 business days**, to provide an initial
assessment within **10 business days**, and to coordinate disclosure with you
once a fix is available. We credit reporters who wish to be credited.

### Scope

In scope: the OpenMeet desktop app (Rust backend, React frontend, Tauri
configuration and capabilities), the local AI sidecar (`local-ai/`), build and
release scripts. Especially interesting:

* any way for web content, transcripts, screen text or model output to execute
  code, call privileged commands or read files (prompt injection that escapes
  the assistant's text output, IPC abuse, XSS in the webviews),
* leakage of API keys, transcripts or screen content (logs, error messages,
  temp files, crash dumps, network),
* data sent off-device when the UI says it is not (local-only mode bypass,
  incorrect local/remote classification),
* capture or processing that continues after pause / emergency stop,
* retention and deletion not removing data.

Out of scope: vulnerabilities in third-party AI providers, issues requiring a
compromised OS account, and the documented limitations of OS capture exclusion
(see [PRIVACY.md](PRIVACY.md#7-capture-exclusion-hiding-the-overlay-from-screen-capture--limitations)).

## Supported versions

Security fixes are released for the latest minor version.

## Security design

* **Process boundary.** All privileged work (audio and screen capture, network
  calls with API keys, database, keychain, file export, process management)
  happens in Rust. The webviews only call a fixed set of typed commands.
* **Least-privilege IPC.** Tauri capabilities grant the overlay window only the
  ~30 commands it needs; the main window gets the rest. No shell, filesystem or
  HTTP plugin is exposed to JavaScript, and `withGlobalTauri` is off.
* **Content Security Policy.** `default-src 'self'`, no remote scripts, no
  inline scripts, `object-src 'none'`, `frame-ancestors 'none'`; the only
  network origin the webviews may reach is Tauri IPC. Prototype freezing is on.
* **Input validation.** Every command validates its arguments (ids, sizes,
  URLs, accelerators, privacy rules); settings are sanitized and validated in
  Rust before being stored. Base URLs must be http(s) without embedded
  credentials.
* **Secrets.** API keys are stored in the OS credential store (Windows
  Credential Manager, macOS Keychain, Secret Service), never in SQLite, never
  returned to the UI, never logged; environment variables are a development
  fallback.
* **Prompt-injection hardening.** Transcripts, screen text, OCR and profile
  notes are passed to models inside clearly delimited data blocks with an
  explicit instruction that they are data, not instructions. Model output is
  rendered as text (no HTML injection), and the model has no tools — it cannot
  trigger actions.
* **No arbitrary execution.** The renderer cannot start processes. The only
  child process OpenMeet starts is the local AI sidecar, from a fixed path
  resolved by the backend (never from UI input), bound to `127.0.0.1`.
* **Sanitized logs.** Logs contain ids, counts, sizes, durations and error
  kinds — never transcript text, prompts, model output, OCR text or keys.
  Provider error messages are sanitized before display.
* **Local sidecar.** Binds to loopback only (refuses other interfaces unless
  explicitly overridden), restricts model names to an allow-list, bounds upload
  sizes, and supports an optional bearer token (`OPENMEET_LOCAL_AI_TOKEN`).
* **Dependency auditing.** CI runs `cargo audit` and `npm audit` on every
  change; dependencies are pinned through `Cargo.lock` and `package-lock.json`.
* **Updates.** OpenMeet does not auto-update or phone home. Install releases
  from the official repository and verify their checksums/signatures. If an
  updater is added, it must use Tauri's signed-update mechanism (Ed25519
  signatures verified before install) over HTTPS.
