## What and why

<!-- What does this change do, and why is it needed? Link issues. -->

## User-visible changes

<!-- UI, behaviour, settings, shortcuts. Screenshots / recordings from demo mode only. -->

## Privacy and security impact

- [ ] No new data leaves the device, or the new data flow is shown in the Privacy page's data-flow report and documented in PRIVACY.md
- [ ] No transcript text, prompts, model output, OCR text, screenshots or API keys are logged
- [ ] New commands validate their inputs and are granted only to the window that needs them (`src-tauri/capabilities/`)
- [ ] Only documented OS APIs are used

## Checklist

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --exclude openmeet` and `cargo test -p openmeet`
- [ ] `npx tsc -p tsconfig.json --noEmit`, `npx eslint .`, `npx vitest run`, `npx vite build`
- [ ] Tests added or updated (no network / keys / hardware required)
- [ ] TypeScript bindings regenerated if IPC types changed (`npm run bindings`)
- [ ] Prompt `VERSION` bumped if a prompt changed
- [ ] Docs updated (README / docs/ / PRIVACY.md / SECURITY.md)
