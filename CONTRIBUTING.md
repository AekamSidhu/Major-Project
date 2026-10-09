# Contributing to OpenMeet

Thanks for helping build an open, privacy-first meeting assistant. This guide
covers how to set up the project, the quality bar, and how changes are
reviewed.

## Ground rules

* **Privacy first.** Never log transcript text, prompts, model output, OCR
  text, screenshots or API keys. Never add telemetry or network calls that the
  user did not configure. Anything that could send data off the device must be
  visible in the Privacy page's data-flow report.
* **No fake features.** Don't add buttons that only say "coming soon", stub
  implementations or simulated success. If a platform cannot support a
  capability, implement a real fallback and document the limitation.
* **No evasion techniques.** OpenMeet only uses documented OS APIs. Do not
  contribute code that injects into other processes, hooks drivers, defeats
  screen-capture protections, proctoring, DRM or anti-cheat, or hides capture
  from the user.
* **Assistive, not autonomous.** OpenMeet never speaks on the user's behalf and
  never scores or ranks people.

## Development setup

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for prerequisites, running the
app, the local AI sidecar and the test suites. Short version:

```bash
npm install
npm run tauri dev
```

## Before you open a pull request

Run all quality gates (CI runs the same):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
npx tsc -p tsconfig.json --noEmit
npx eslint .
npx vitest run
npx vite build
# local AI sidecar (if you changed local-ai/)
local-ai/.venv/Scripts/python -m pytest local-ai   # Windows
local-ai/.venv/bin/python -m pytest local-ai       # macOS / Linux
```

* Add tests for new behaviour. Tests must not need network access, API keys,
  audio hardware or a display; hardware-dependent tests are `#[ignore]` with a
  comment explaining how to run them.
* If you change a type that crosses IPC (anything in `crates/openmeet-core`
  with `#[ts(export)]`), regenerate the TypeScript bindings:
  `npm run bindings`.
* If you add a Tauri command, register it in `src-tauri/build.rs`, grant it in
  the right file under `src-tauri/capabilities/`, add the wrapper to
  `src/lib/ipc.ts`, and validate its inputs.
* Prompt changes (`crates/openmeet-intel/src/prompts/`) must bump the prompt's
  `VERSION` and keep the parser tests passing.
* Keep the architecture documents in sync (`docs/ARCHITECTURE.md`,
  `docs/INTERNALS.md`, `docs/FRONTEND.md`).

## Commit and PR style

* Small, focused pull requests with a clear description of *why*.
* Describe user-visible changes and any privacy impact in the PR template.
* Screenshots or short recordings for UI changes (use demo mode — never real
  meetings).

## Reporting bugs and requesting features

Use the issue templates. **Never paste real meeting transcripts, screenshots of
real meetings or API keys into issues.** Use demo mode or synthetic examples.
Security and privacy vulnerabilities go through
[private reporting](SECURITY.md), not public issues.

## Contact

Maintainers can be reached through the repository's GitHub discussions, or
privately through GitHub's private vulnerability reporting for anything
sensitive (including Code of Conduct reports).

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
