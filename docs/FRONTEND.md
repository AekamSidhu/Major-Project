# OpenMeet frontend contract

React 19 + TypeScript 6 + Vite 8 + Tailwind CSS 4 + Zustand 5, inside Tauri 2.
Two windows, two entry points:

| window  | label     | entry                    | purpose |
|---------|-----------|--------------------------|---------|
| main    | `main`    | `index.html` → `src/main/main.tsx`       | onboarding, start-meeting flow, settings, privacy, history, diagnostics |
| overlay | `overlay` | `overlay.html` → `src/overlay/overlay.tsx` | the live AI HUD that floats over the meeting — **the primary product surface** |

## Shared building blocks (already written — use, do not fork)

* `src/bindings/*` — TypeScript types generated from Rust (`ts-rs`). Import via
  `@/bindings`. Never hand-edit; never redefine these types.
* `src/lib/ipc.ts` — the **only** way to talk to the backend: `api.*` wrappers
  for every Tauri command, `onMeetingEvent`, `onUiCommand`, `errorMessage`.
  Components never call `invoke` directly.
* `src/lib/answer.ts` — progressive `[[SECTION]]` answer parser (`parseAnswer`,
  `answerToPlainText`).
* `src/lib/theme.ts` — `applyAppearance`, `resolveTheme`, `watchSystemTheme`.
* `src/lib/format.ts` — time/duration/category/shortcut formatting.
* `src/styles/base.css` — design tokens. Tailwind color utilities map to tokens:
  `bg-bg`, `bg-surface`, `bg-surface-2`, `bg-raised`, `border-border`,
  `border-border-strong`, `text-fg`, `text-fg-soft`, `text-muted`,
  `bg-accent`/`text-accent`/`bg-accent-soft`/`text-accent-fg`, `text-ok`,
  `text-warn`, `text-danger`, `bg-danger-soft`. Utility classes: `om-pulse`,
  `om-fade-in`, `om-caret`. Overlay glass color: `rgb(var(--om-glass) / <alpha>)`.
* `src/main/ui.tsx` — main-window primitives: `Button`, `Spinner`, `Field`,
  `TextInput`, `TextArea`, `Select`, `Toggle`, `ToggleField`, `Slider`,
  `PageHeader`, `Section`, `Badge`, `StatusDot`, `Kbd`, `Banner`, `EmptyState`,
  `Dialog`, `ConfirmDialog`, `Tabs`, `cx`.
* `src/main/store.ts` — main-window Zustand store: `settings`,
  `updateSettings(mutator)` (optimistic, persisted, reverts + notifies on error),
  `section`/`setSection`, `live` (meeting state from events, `reduceLive`),
  `notify`, `appInfo`.
* `src/main/App.tsx` — shell (sidebar, capture indicator, notices, onboarding
  gate). Imports one component per section file (see ownership below).

## Backend behaviour the UI relies on

* **Settings** are a single `Settings` object (`api.getSettings` /
  `api.updateSettings(settings)` → stored, sanitized value). Validation errors
  come back as `CoreError { kind: "invalid_input", message }`. When settings
  change anywhere, the backend emits UI command `SETTINGS_CHANGED` to all
  windows.
* **Events** (`openmeet://event`, type `MeetingEvent`, discriminant `type`):
  every subsystem reports through them. Key semantics:
  * `ANSWER_STARTED { requestId, kind, slot, questionText, category, style, parentId }`
    starts a request in a UI slot (`primary` = answers / Ask / refinements /
    clarifications; `followups` = interviewer follow-ups). **The UI must show
    only the latest `requestId` per slot and ignore deltas/completions/state
    changes for any other id** (stale responses must never overwrite newer ones).
  * `ANSWER_DELTA { requestId, text }` — append; render progressively.
  * `ANSWER_STATE_CHANGED { requestId, state, error }` — `created →
    context_building → requesting → streaming → complete`, or `failed`
    (show `error`, never fail silently) / `cancelled`.
  * `ANSWER_COMPLETE { requestId, suggestion }` — final text in
    `suggestion.content` (use it to replace the streamed text).
  * `FOLLOWUPS_READY { requestId, questions, done }` — full list so far.
  * `QUESTION_DETECTED { question }` — `question.requiresResponse`,
    `category`, `speakerRole`, `method`.
  * `TRANSCRIPT_PARTIAL` / `TRANSCRIPT_FINAL { segment }` — partials and the
    final of an utterance share `utteranceId`.
  * `MEETING_STATE_CHANGED { meetingId, state, mode, pausedReason }`,
    `MEETING_STARTED`, `MEETING_STOPPED`, `SUMMARY_READY`, `SUMMARY_FAILED`.
  * `PROCESSING_STOPPED { reason }` — emergency stop; show **PROCESSING STOPPED**
    until the meeting is resumed (`MEETING_STATE_CHANGED` with `active`).
  * `PRIVACY_STATE_CHANGED { enabled, supported, active }`,
    `COMPONENT_STATUS_CHANGED { health }`, `NOTICE { level, message }`,
    `MODE_CHANGED`, `STYLE_CHANGED`, `TOPICS_UPDATED`, `SCREEN_UPDATED`,
    `CLAIMS_EXTRACTED`, `SUGGESTION_PINNED`, `SUGGESTION_DISMISSED`,
    `AUDIO_LEVEL { source, level }` (0..1, ~10 Hz), `LATENCY_RECORDED`,
    `VOICE_COMMAND_RECOGNIZED`.
* **UI commands** (`openmeet://ui`, `UiCommand`): `FOCUS_ASK` (overlay: open the
  Ask input), `TOGGLE_EXPANDED` (overlay: minimal ↔ expanded),
  `CLICK_THROUGH_CHANGED { enabled }`, `OVERLAY_VISIBILITY_CHANGED { visible }`,
  `SETTINGS_CHANGED`, `NAVIGATE { section }` (main window section id, e.g.
  `"settings/ai"`).
* **Global hotkeys are registered by the backend** (Rust), not the webview.
  The UI only displays them (`api.getShortcutStatus`).
* **Overlay window**: transparent, frameless, always-on-top, skip-taskbar,
  resizable by content (`api.overlaySetSize(width, height)` in logical px —
  backend clamps to `settings.overlay.width`/`maxHeight`). It is
  **non-activating**: clicking its buttons does not steal focus from the
  meeting app. To type (Ask OpenMeet), call `api.overlayBeginInput()` (makes it
  focusable, focuses it, remembers the meeting app) and `api.overlayEndInput()`
  when done (restores focus to the meeting app). Dragging: an element with
  `data-tauri-drag-region` moves the window; after a drag ends call
  `api.overlaySavePosition()` (backend remembers it per display). Click-through
  (`settings.overlay.clickThrough`, hotkey) makes the window ignore the mouse.
  The backend shows the overlay on meeting start when
  `settings.overlay.showOnMeetingStart`.
* `snapshot = api.getSnapshot()` (`EngineSnapshot`) hydrates any window that
  (re)loads mid-meeting: current meeting, state, mode, style, transcript tail,
  questions, `active` primary request (with text so far), follow-ups, pinned
  suggestions, topics, components, last observation/summary, flags.
* Errors from `api.*` reject with `CoreError`; show `errorMessage(e)`.

## Design language (spec §61–§65)

Premium, minimal, quiet, fast, professional. No giant dashboards, no heavy
gradients, no huge cards, no gratuitous animation (fades ≤160 ms; honour
`reducedMotion` and `prefers-reduced-motion`). Typography first; small type
scale (0.72–1.25 rem); restrained accent color; thin borders. Keyboard
accessible: every control reachable by Tab, visible focus, `aria-*` labels,
`role="status"/"alert"` for live regions, sufficient contrast in both themes.

## Overlay requirements (owner: overlay agent — everything in `src/overlay/`)

States (spec §5) rendered inside one compact card (default width 380 px):
* **Hidden** — window hidden by the backend (nothing to render).
* **Idle** (no meeting) — tiny "OpenMeet · Ready" pill with an *Ask* button
  (Ask OpenMeet works without a meeting) and a link to open the main window.
* **Listening** — minimal `● Listening` (mode badge `Candidate`/`Interviewer`,
  subtle pulsing dot; audio level shimmer optional).
* **Paused** — `❚❚ Paused` + Resume button.
* **Processing** — `● Thinking…` while a request is `context_building` /
  `requesting`.
* **Question detected / answer streaming** (candidate mode, primary slot):
  ```
  QUESTION · System Design
  "How did you handle scaling the database?"
  SUGGESTED STRUCTURE (if any) 1. … 2. …
  SUGGESTED ANSWER   streamed text with caret while streaming
  KEY POINTS         • …
  EXAMPLE (collapsible)  FOLLOW-UP you could ask  BASIS (shown only via "Show reasoning")
  CAVEAT (highlighted when present — "Context insufficient — verify before using.")
  ```
  Render progressively from `parseAnswer(text)` on every delta.
* **Follow-ups** (interviewer mode, followups slot): `FOLLOW-UP QUESTIONS` list,
  each with question (`→`), category chip, priority (high = accent), the short
  reason (muted), click-to-copy; progressive as `FOLLOWUPS_READY` arrives.
  Also show the topic tracker compactly (`✓ covered → current ○ pending`) and
  "Suggested next area: X" when provided.
* **Minimal HUD** (`settings.overlay.minimal` or `TOGGLE_EXPANDED`): one line,
  e.g. `● Q detected · System Design` / `● 3 follow-ups` / `● Listening`; click or
  hotkey expands.
* **PROCESSING STOPPED** banner after `PROCESSING_STOPPED` with Resume.
* **Errors** never silent: failed request shows its `error` with Retry
  (= `refineAnswer(id, "regenerate")` or re-run) and a "Settings" link
  (`api.openMainWindow("settings/ai")`); component errors (e.g. microphone,
  transcription) appear as a one-line warning.
* **Ask OpenMeet**: on `FOCUS_ASK` (or Ask button) → `api.overlayBeginInput()`,
  show a single-line input (placeholder "Ask OpenMeet…"); Enter →
  `api.askOpenMeet(text)`; Esc → close + `api.overlayEndInput()`; also end input
  after submit.
* **Answer controls** (spec §34), compact icon/text buttons under the answer,
  keyboard accessible: Regenerate, Shorter, Longer, More technical, Simplify,
  Give example, Show reasoning (→ `refineAnswer(id, "show_rationale")`), Copy
  (`@tauri-apps/plugin-clipboard-manager` `writeText`, using
  `answerToPlainText`), Pin (`pinSuggestion`), Dismiss (`dismissSuggestion`).
  Answer style switcher (Concise, Detailed, Technical, Simple, STAR,
  Example-driven, Step-by-step → `setAnswerStyle`). Generate-answer button when
  no answer is shown.
* **Header row**: drag handle (`data-tauri-drag-region`), status, mode badge
  (click toggles mode via `setMode`), click-through indicator, capture-privacy
  indicator (shield, from `getCapturePrivacyStatus` / `PRIVACY_STATE_CHANGED`),
  pause, hide (`overlayHide`), open main window.
* **Pinned** suggestions: small collapsible list.
* Only the most relevant card is shown; at most the primary card + follow-ups.
* Appearance from settings: theme, `opacity` (card background alpha),
  `translucent` (backdrop blur + lower alpha), `fontSize`, `density`,
  `reducedMotion`, `width`. Re-read settings on `SETTINGS_CHANGED`.
* Auto-size: observe the card size (ResizeObserver) and call
  `api.overlaySetSize(width, height)` (throttled, only when it changes by ≥4 px);
  content beyond `maxHeight` scrolls inside the card.
* Report render latency: for the first delta of each request, measure
  event-arrival → next animation frame and call `api.reportRenderLatency(ms)`.
* Performance: one Zustand store with a pure, unit-tested reducer
  (`reduceOverlay(state, event)`), memoized parsing, no heavy libraries.
* Tests (vitest + jsdom, `@tauri-apps/api/mocks` `mockIPC` for invoke): reducer
  (stale-request filtering per slot, partial→final, follow-ups progressive,
  processing-stopped, failures), parsing integration, rendering of the main
  states with Testing Library.

## Main window requirements

### Owner "main-meeting" — `src/main/sections/Home.tsx`, `src/main/sections/History.tsx` (+ any `src/main/sections/home/*`, `src/main/sections/history/*`)

* **Home — not in a meeting**: the Start Meeting flow (spec §68) on one calm
  page: 1) mode (Candidate / Interviewer, with one-line explanations);
  2) audio: microphone toggle (+ device name from settings), system audio
  toggle (disabled with the reason when `listAudioDevices().systemAudio.supported`
  is false); 3) optional screen context (off by default; pick a source from
  `listScreenSources()` — display / window / application; region optional);
  4) privacy check: compact data-flow summary from `getDataFlow()` (which
  components are Local / Cloud), capture-privacy status, retention policy,
  link to Privacy; 5) detected meeting apps (`detectMeetingApps()`), choose one
  to focus after start; a consent reminder ("Make sure recording/AI assistance
  is allowed in this meeting"); **Start meeting** → `startMeeting(request)`;
  errors shown inline. Plus **Demo mode** card: list `listDemoScenarios()` and
  start one (`startDemo`) — works without any provider. Also a quick
  "Ask OpenMeet" hint with the hotkey and an "Open overlay" button.
* **Home — meeting live**: header with title, mode, elapsed time, state badge;
  big, obvious controls: Pause/Resume, End meeting (`stopMeeting`), **Emergency
  stop** (danger), Show/Hide overlay; quick toggles for microphone/system
  audio/screen context (`setScreenCapture`) where possible, link to switch
  providers; live transcript (speaker labels via `SpeakerRole` and mode:
  user → "You", remote → "Interviewer"/"Candidate", unknown → "Speaker";
  partials in muted italics, auto-scroll that pauses when the user scrolls up);
  detected questions list; topics; component health warnings (from
  `live.components` with status error/degraded/unsupported).
* **Home — meeting completed**: summary (`live.lastSummary` or "Generating
  summary…" while `summarizing`), with Export and Open in History buttons.
* **History**: paginated meeting list (`listMeetings`) with title, date,
  duration, mode, counts, summary badge; empty state. Meeting detail
  (`getMeeting`): tabs Summary (rendered sections from `MeetingSummary` incl.
  interviewer extras; Regenerate summary), Transcript (timestamps
  `formatOffset`), Questions & answers (questions with the spoken answers and
  the AI suggestions — render suggestion content with `parseAnswer`),
  Follow-ups, Topics & claims, Screen observations. Export dialog: format
  (Markdown / JSON / Plain text) + parts checkboxes (summary, transcript,
  questions, follow-ups) → `exportMeeting` (backend shows the save dialog;
  `path === null` means cancelled). Delete meeting (confirm). Never render
  hiring scores (none exist).

### Owner "main-settings" — `src/main/sections/Privacy.tsx`, `src/main/sections/Diagnostics.tsx`, `src/main/sections/settings/*.tsx`, `src/main/onboarding/*` (+ shared settings components in `src/main/components/*`)

* **AI providers** (`settings.ai`): LLM provider (Ollama — local, default;
  OpenAI; Anthropic; OpenAI-compatible), model (text + "Load models" via
  `listModels("llm")`), base URL (Ollama / compatible / overrides), temperature,
  max output tokens, reasoning effort, optional fast model; API keys per
  provider (`getApiKeyStatus`, `setApiKey` — password input, never displayed
  back; shows source keychain/env; `clearApiKey`); "Test connection"
  (`testProvider("llm")`) with latency. Vision provider (Disabled / OCR only /
  OpenAI / Anthropic / Ollama / compatible) + model; embeddings provider
  (built-in / local AI / OpenAI / Ollama). Clearly label each choice **Local**
  or **Cloud**.
* **Transcription & audio**: provider (Local AI sidecar — default, OpenAI,
  OpenAI-compatible), model, base URL, language, interim results, vocabulary;
  local AI status (`getLocalAiStatus`, `startLocalAi`, `stopLocalAi`, setup
  hint shown when not installed); "Test" (`testProvider("transcription")`);
  microphone device select + system-audio device select from
  `listAudioDevices()`, enable toggles, live level meter
  (`startAudioTest`/`stopAudioTest` + `AUDIO_LEVEL` events), VAD sensitivity,
  silence length. Assistant behaviour (`settings.assistant`): auto-answer,
  auto-follow-ups, semantic detection, detection threshold, answer style,
  response length, context window, follow-up min words, turn-end silence.
* **Screen context** (`settings.screen`): explicit enable (off by default),
  source picker (`listScreenSources`), frequency (Manual / 1 / 2 / 5 fps),
  change threshold, max vision calls per minute, store screenshots (off;
  explain), manual capture hotkey hint.
* **Overlay & appearance** (`settings.overlay`): theme (dark/light/system),
  opacity, width, max height, font size, density, translucent, minimal mode,
  position (corners + custom via `overlaySetPosition`), target display
  (`listDisplays` with resolution/scale/orientation), layout (same display /
  separate display), click-through, reduced motion, show on meeting start,
  margin; live preview is the real overlay (offer "Show overlay").
* **Shortcuts**: table of all `ShortcutAction`s with the accelerator
  (`formatAccelerator`), registration status / errors / warnings from
  `getShortcutStatus()`; record-a-shortcut input (captures key combo, builds a
  Tauri accelerator like `CommandOrControl+Shift+K`) → `updateShortcut`;
  conflicts shown inline; reset all.
* **Profile & context** (`settings.profile`): background / resume notes (with
  the explicit explanation that suggestions only use facts from here and the
  conversation, never invented history), role / job description / agenda,
  planned interview topics (one per line), custom instructions. Debounced save.
* **General**: launch at startup (explicitly: never starts recording),
  voice commands (optional), notifications, developer mode, reset settings.
* **Privacy & data** (spec §48, §59, §90, §91): data-flow diagram from
  `getDataFlow()` (Microphone → Transcription → Context engine → AI provider →
  Overlay; Screen → Vision → Context engine → AI provider), each node marked
  Local / Cloud / Disabled; the statement from the report (show "No meeting data
  leaves this device." **only** when `allLocal` is true); local-only mode
  toggle; capture privacy (status, mechanism, honest limitation note,
  toggle via `setCaptureExclusion`); screen privacy rules (list / add by type:
  application, window, domain, region "x,y,w,h" / remove; suggested sensitive
  apps from `suggestedPrivacyApps()`); data retention select; **Delete all
  meeting data** (confirm dialog, `deleteAllData`); quick disable toggles for
  microphone, system audio, screen context and cloud processing (= local-only).
* **Diagnostics** (spec §57, §75): component health grid (`getDiagnostics`,
  live via events), latency stats table (p50/p95/last/max), memory & CPU,
  version/platform; "Run benchmark" (`runBenchmark`) results table.
  **Developer tools** (only when `developerMode`): simulate incoming question,
  transcript line (speaker select), candidate answer, screen change, AI
  response, AI failure, network latency (`simulate`), clear simulation.
* **Onboarding** (spec §67, first launch, full window): Welcome → choose mode →
  AI provider (Local via Ollama recommended; or cloud with key entry) →
  transcription provider (local sidecar status/setup hint, or cloud) →
  microphone (device + level test) → optional screen context (stays off unless
  chosen) → privacy (retention, capture exclusion, local-only, consent note) →
  Finish (sets `general.onboardingComplete = true`). Skippable steps, back/next,
  progress indicator; never forces screen capture.

## Testing & quality gates (all frontend agents)

```
npx tsc -p tsconfig.json --noEmit
npx eslint .
npx vitest run
npx vite build
```
All must pass. Unit-test reducers/pure helpers and render tests for key
states with Testing Library (`@tauri-apps/api/mocks` → `mockIPC`,
`mockWindows`; call `clearMocks()` in `afterEach`). Don't add heavy
dependencies (no UI kits, no icon packs — inline SVG is fine). No `console.log`.
Files you do not own are read-only; the shared files above may only be changed
by asking the integrator (describe the needed change in your final report).
Placeholders marked `AGENT-OWNED-PLACEHOLDER` must be fully replaced.
