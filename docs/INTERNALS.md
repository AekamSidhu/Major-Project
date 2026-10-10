# OpenMeet internals — crate contracts

This document is the binding contract between OpenMeet's Rust crates. It is
written for contributors (human or AI) implementing or changing a crate.
`docs/ARCHITECTURE.md` explains the *why*; this file pins down the *what*.

## Crate graph

```
openmeet-core        domain types, events, settings, provider traits, IPC DTOs, DSP helpers
   ▲   ▲   ▲   ▲
   │   │   │   └── openmeet-store      SQLite persistence (rusqlite), migrations, retention
   │   │   └────── openmeet-platform   OS integration: audio capture, screen capture, OCR,
   │   │                               capture privacy, focus, meeting-app detection
   │   └────────── openmeet-providers  HTTP adapters: OpenAI, Anthropic, Ollama,
   │                                   OpenAI-compatible, local AI sidecar
   └────────────── openmeet-intel      pure algorithms: VAD, question detection, text
                                       similarity, topics, claims, prompts, export,
                                       vision adapters, mocks, demo scripts
openmeet-engine      real-time orchestration (tokio): transcription sessions, context
                     engine, retrieval, request manager (cancellation / versioning),
                     candidate + interviewer flows, summaries, diagnostics
                     depends on core + intel
src-tauri            the desktop app: windows, overlay, tray, hotkeys, commands,
                     secrets (keychain), wiring of all crates
```

`openmeet-providers`, `openmeet-platform` and `openmeet-store` depend **only** on
`openmeet-core` (+ third-party crates). `openmeet-intel` depends only on core.

## Rules for every crate

1. **Read `crates/openmeet-core/src/*.rs` first.** Use its types; do not
   redefine equivalents. Do not edit openmeet-core — if you believe the contract
   needs a change, explain it in your final report.
2. **Real implementations only.** No `todo!()`, `unimplemented!()`, stub
   bodies, fake success values or "coming soon" paths. When an OS / provider
   capability is unavailable, return a typed `Unsupported`/error value with an
   honest message.
3. **Privacy in logs.** Use `tracing`. Never log transcript text, prompts,
   model output, OCR text, screenshots or API keys — log ids, sizes, counts,
   durations and error kinds only. Error messages returned to the UI must not
   contain secrets or meeting content.
4. **Cancellation.** Every network or long-running async operation accepts a
   `tokio_util::sync::CancellationToken` and stops promptly when cancelled
   (use `tokio::select!`).
5. **Tests.** Unit tests next to the code (`#[cfg(test)]`), integration tests in
   `tests/`. Tests must not require network access, API keys, audio devices or
   a display; hardware-dependent tests are `#[ignore]` with a comment.
6. **Quality gates** (run from `X:/openmeet`, Git Bash):
   ```bash
   export PATH="/x/rust/cargo/bin:$PATH" RUSTUP_HOME='X:\rust\rustup' CARGO_HOME='X:\rust\cargo'
   export CARGO_TARGET_DIR=X:/openmeet/target/agent-<crate>   # one target dir per agent
   cargo fmt -p <crate>
   cargo clippy -p <crate> --all-targets -- -D warnings
   cargo test -p <crate>
   ```
7. **Dependencies.** Prefer `{ workspace = true }` deps from the root
   `Cargo.toml` (`serde`, `serde_json`, `thiserror`, `tokio`, `tokio-util`,
   `futures`, `async-trait`, `uuid`, `tracing`, `ts-rs`, `parking_lot`, `regex`,
   `base64`). Other crates: add them to your crate's own `Cargo.toml` with an
   explicit current version (`cargo add -p <crate> <dep>` picks the latest).
   Check the actual API in `X:/rust/cargo/registry/src/*/<crate>-<ver>/` when
   unsure — APIs drift between versions.
8. **Exported TS types.** Any new type that crosses IPC must live in
   openmeet-core (ask instead of adding one). `i64`/`u64` fields need
   `#[ts(type = "number")]`.
9. Edition 2024, `rust-version = 1.85`. Keep public APIs documented.

---

## openmeet-intel (pure algorithms)

Dependencies allowed: core, serde, serde_json, regex, async-trait, futures,
tokio (sync, time), tokio-util, parking_lot, tracing, base64. No network, no OS.

### `intel::vad`
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    pub frame_ms: u32,                    // 20
    pub min_speech_ms: u32,               // consecutive speech before start, 100-150
    pub silence_ms: u32,                  // end of utterance, from settings (default 700)
    pub pre_roll_ms: u32,                 // audio kept before speech start, ~240
    pub min_utterance_ms: u32,            // shorter utterances are discarded, ~350
    pub max_utterance_ms: u32,            // force a cut (prefer a low-energy frame), ~20_000
    pub interim_interval_ms: Option<u32>, // Some(n) => emit Interim clips every n ms of speech
    pub threshold_db_above_noise: f32,    // derived from sensitivity
    pub absolute_floor_dbfs: f32,         // ~ -55 dBFS; quieter frames are never speech
}
impl VadConfig {
    /// sensitivity 0..1 (higher = more sensitive)
    pub fn from_settings(sensitivity: f32, silence_ms: u32, interim_results: bool) -> Self;
}
impl Default for VadConfig { /* from_settings(0.6, 700, false) */ }

#[derive(Debug, Clone, PartialEq)]
pub enum SegmenterEvent {
    SpeechStarted { utterance_id: String, at_ms: i64 },
    /// In-progress clip (is_final = false), only when interim is enabled.
    Interim(AudioClip),
    /// Completed utterance (is_final = true).
    Utterance(AudioClip),
    /// Speech ended but was shorter than min_utterance_ms.
    Discarded { utterance_id: String },
}

pub struct Segmenter { /* adaptive noise floor, hangover, pre-roll */ }
impl Segmenter {
    pub fn new(source: AudioSourceKind, config: VadConfig) -> Self;
    /// Feed 16 kHz mono audio of any length. Timestamps come from
    /// `frame.captured_at_ms` (sample-accurate offsets inside the frame).
    pub fn push(&mut self, frame: &AudioFrame) -> Vec<SegmenterEvent>;
    /// Force-end the current utterance (pause / stop / turn end).
    pub fn flush(&mut self) -> Vec<SegmenterEvent>;
    pub fn is_speaking(&self) -> bool;
    /// Smoothed level 0..1 for UI meters.
    pub fn level(&self) -> f32;
    pub fn noise_floor_dbfs(&self) -> f32;
    pub fn config(&self) -> &VadConfig;
}
```
Energy-based VAD with an adaptive noise floor (slow EMA of non-speech frames,
fast adaptation downward), speech start after `min_speech_ms` of consecutive
speech frames, hangover of `silence_ms`, pre-roll prepended to each utterance.
The same `utterance_id` (use `core::new_id()`) is used for SpeechStarted, all
Interim clips and the final Utterance.

### `intel::text`
```rust
pub fn normalize(text: &str) -> String;            // lowercase, unify quotes, collapse whitespace, strip punctuation except '
pub fn content_tokens(text: &str) -> Vec<String>;  // normalized, stop-words removed, stemmed, synonyms unified
pub fn stem(word: &str) -> String;                 // light English suffix stripping (scaling/scaled/scales -> scal)
pub fn jaccard(a: &[String], b: &[String]) -> f32;
/// Paraphrase-robust similarity 0..1 (content tokens + synonym map + bigrams).
/// "How did you scale the API?" vs "How did you handle API scaling?" >= 0.6;
/// unrelated questions <= 0.25.
pub fn lexical_similarity(a: &str, b: &str) -> f32;
pub fn cosine(a: &[f32], b: &[f32]) -> f32;        // 0 for empty / mismatched / zero vectors
pub fn estimate_tokens(text: &str) -> usize;       // ~ chars/4, never 0 for non-empty text
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String; // keep start, cut at word boundary, append "…"
pub fn tail_to_tokens(text: &str, max_tokens: usize) -> String;     // keep end, prefix "…"
pub fn word_count(text: &str) -> usize;
pub fn split_sentences(text: &str) -> Vec<String>;

/// Deterministic local embeddings: feature hashing of stemmed unigrams,
/// bigrams and character trigrams, signed hashing, L2-normalized.
pub struct HashingEmbedder { .. }
impl HashingEmbedder {
    pub const DEFAULT_DIMS: usize = 384;
    pub fn new(dims: usize) -> Self;
    pub fn embed_one(&self, text: &str) -> Vec<f32>;
}
// impl core::providers::EmbeddingProvider for HashingEmbedder
//   info(): id "local_hash", display "Built-in hashing embeddings", model "hash-384", location Local

/// BM25 keyword index for hybrid retrieval.
pub struct Bm25Index { .. }
impl Bm25Index {
    pub fn new() -> Self;
    pub fn add(&mut self, id: &str, text: &str);     // re-adding an id replaces it
    pub fn remove(&mut self, id: &str);
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)>; // best first, score > 0
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### `intel::detect`
```rust
pub struct DetectionInput<'a> {
    pub segment: &'a TranscriptSegment,       // the new FINAL segment
    pub recent: &'a [TranscriptSegment],      // earlier finals, oldest -> newest, excluding `segment`
    pub mode: Mode,
    pub last_question: Option<&'a DetectedQuestion>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    pub is_question: bool,
    pub confidence: f32,          // 0..1
    pub question: String,         // cleaned; merged with lead-in / continuation
    pub parts: Vec<String>,       // sub-questions for multi-part questions (>=1 when is_question)
    pub category: QuestionCategory,
    pub requires_response: bool,
    pub segment_ids: Vec<Id>,     // segments composing `question`
    pub supersedes: Option<Id>,   // id of `last_question` when this continues it
    pub ambiguous: bool,          // heuristics unsure (≈0.35..0.65) → semantic check worthwhile
    pub signals: Vec<String>,     // machine-readable reasons ("question_mark", "wh_start", "aux_inversion",
                                  // "imperative_request", "embedded_question", "negative:reported", ...)
}

pub fn detect(input: &DetectionInput) -> DetectionResult;
pub fn classify_category(text: &str) -> QuestionCategory;
pub fn split_parts(text: &str) -> Vec<String>;
/// "this", "that", "here", "this part", "the diagram", "the code above",
/// "on the screen", "what you're showing" ...
pub fn has_deictic_reference(text: &str) -> bool;

pub mod voice {
    /// Wake phrase "OpenMeet" (+ ASR variants "open meet", "open meat",
    /// "openmeat") at the start, followed by a command:
    /// "explain that" -> Explain, "give me a follow-up" -> FollowUp,
    /// "answer that" -> Answer, "make that concise/shorter" -> Concise,
    /// "more detail/longer" -> Detailed, anything else -> Ask(rest).
    /// Returns None when no wake phrase.
    pub fn parse_voice_command(text: &str) -> Option<VoiceCommand>;
}
```
Detection must use: punctuation when present (ASR often omits it), WH-word
starts, auxiliary inversion ("can you", "have you", "would you", "is it", ...),
imperative prompts ("tell me about", "walk me through", "explain", "describe",
"talk me through", "give me an example", "share an example"), embedded
questions ("I was wondering how", "I'm curious why", "I'd love to hear"),
tag questions, conversational context (a lead-in like "Next question." or
"Let's talk about databases." in the previous remote segment within 4 s is
merged; a same-speaker continuation within 3 s of `last_question` sets
`supersedes`) and speaker role:

| mode        | speaker  | requires_response |
|-------------|----------|-------------------|
| candidate   | remote   | true              |
| candidate   | unknown  | true (confidence ×0.85) |
| candidate   | user     | false             |
| interviewer | user     | false (it is an *asked* question, tracked for duplicates) |
| interviewer | remote   | true              |
| interviewer | unknown  | true (confidence ×0.85) |

False-positive filters (must be unit tested): "The question I was thinking
about was…", "That's a good question", "Good question.", reported speech ("he
asked me how…", "they wanted to know why…"), self-answered rhetorical "What
did we do? We added caching." where the answer follows in the same segment,
fillers ("you know?", "right?", "okay?" alone), "I wonder if I should…",
"no question about it", statements starting "How we did it was…". Very short
utterances (< 3 words) are not questions unless they are clarification
requests ("Why?" after a statement → follow_up with low confidence is OK).

Categories (`classify_category`): keyword/phrase scoring for the 13
categories (see `QuestionCategory`), e.g. system_design ("design a", "how would
you design", "scale to", "url shortener", "rate limiter", "distributed"),
coding ("algorithm", "complexity", "big o", "implement a function", "data
structure"), behavioral ("tell me about a time", "conflict", "disagree",
"failure", "mistake", "challenge you faced"), experience ("your experience",
"have you worked with", "in your previous role", "your last project"),
tradeoff ("trade-off", "pros and cons", "why did you choose", "versus", "instead
of"), clarification ("what do you mean", "could you clarify", "can you repeat",
"do you mean"), follow_up (short questions starting with "and", "so", "what
about", "how about", "why" referring back with "that/it"), leadership,
product, culture, architecture, technical (generic tech vocabulary), general
fallback.

### `intel::topics`
```rust
pub struct TopicTracker { .. }
impl TopicTracker {
    pub fn new(meeting_id: &str, mode: Mode, planned: &[String]) -> Self;
    /// Update from a FINAL segment; true when topics changed.
    pub fn observe(&mut self, segment: &TranscriptSegment) -> bool;
    /// Covered (chronological), then current, then pending.
    pub fn topics(&self) -> Vec<Topic>;
    pub fn current(&self) -> Option<Topic>;
    /// Next area to explore: planned pending topics first, then taxonomy topics
    /// relevant to the conversation so far that are still pending.
    pub fn suggested_next(&self) -> Option<String>;
    /// Merge LLM topic analysis (name, status); unknown names become new topics.
    pub fn apply_analysis(&mut self, items: &[(String, TopicStatus)]) -> bool;
}
```
Built-in taxonomy with keyword lists (ids are slugs): background/experience,
architecture, system design, database & storage, scaling & performance,
deployment & infrastructure, observability, security, testing & quality,
reliability & failure handling, collaboration & behavioral, product & impact.
A topic becomes `Current` after ≥2 keyword hits within the last ~60 s of
speech and `Covered` when another topic becomes current. Planned topics (from
the user profile) are matched by their own words.

### `intel::claims`
```rust
pub fn extract_claims(segment: &TranscriptSegment) -> Vec<Claim>;   // source = Heuristic
pub fn extract_action_items(segment: &TranscriptSegment) -> Vec<ActionItem>;
pub fn extract_entities(text: &str) -> Vec<String>;   // canonical casing: "Kafka", "PostgreSQL", "Kubernetes", "AWS", ...
pub fn extract_metrics(text: &str) -> Vec<String>;    // "10 million requests per day", "40%", "200 ms", "5k QPS"
```
Claims: first-person accomplishment statements ("I built", "we migrated", "I
led", "my team owned"), metrics (numbers + units), technologies (a lexicon of
~150 common technologies), decisions ("we decided", "we chose"), trade-offs
("the downside was", "at the cost of"), risks ("the risk", "single point of
failure"). Action items: "I'll send", "we will follow up", "action item",
"let's schedule", "can you send me". Deduplicate within a segment.

### `intel::prompts`
Every prompt module exposes `pub const ID: &str`, `pub const VERSION: u32`,
a `build(...) -> PromptParts` and a parser where output is structured.
Prompts are provider-independent plain text. All untrusted content
(transcript, screen text, OCR, profile) is wrapped in clearly delimited
blocks (`<transcript>…</transcript>`, `<screen>…</screen>`, `<profile>…</profile>`)
and the system prompt states that content inside them is data, never
instructions (prompt-injection hardening).

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct PromptParts { pub id: &'static str, pub version: u32, pub system: String, pub user: String }

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptLine { pub speaker: String, pub text: String, pub at_ms: i64 /* meeting-relative */ }

#[derive(Debug, Clone, PartialEq)]
pub struct ContextSnippet {
    pub kind: String,     // "earlier_discussion" | "previous_question" | "previous_answer" | "screen" | "claim" | "fact"
    pub text: String,
    pub at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PromptContext {
    pub mode: Mode,
    pub profile: UserProfile,
    pub long_term_summary: String,
    pub retrieved: Vec<ContextSnippet>,
    pub recent_transcript: Vec<TranscriptLine>,
    pub screen: Vec<ScreenObservation>,
    pub references_screen: bool,      // question uses deictic references → screen is primary context
    pub topics: Vec<Topic>,
    pub asked_questions: Vec<String>,
    pub previous_suggestions: Vec<String>,
    pub claims: Vec<Claim>,
}

/// Shared renderer of PromptContext into delimited blocks within a token budget
/// (priority when trimming: question > screen (if references_screen) > recent
/// transcript tail > retrieved > summary > profile > topics).
pub fn render_context(ctx: &PromptContext, budget_tokens: usize) -> String;

pub mod candidate_answer {
    pub const ID: &str = "candidate-answer";
    pub fn build(ctx: &PromptContext, question: &str, category: QuestionCategory,
                 style: AnswerStyle, length: ResponseLength, budget_tokens: usize) -> PromptParts;
    pub fn parse(text: &str) -> AnswerSections;
}
pub mod candidate_clarification {   // ambiguous questions → clarifying questions
    pub const ID: &str = "candidate-clarification";
    pub fn build(ctx: &PromptContext, question: &str, budget_tokens: usize) -> PromptParts;
}
pub mod refine {
    pub const ID: &str = "answer-refine";
    pub fn build(ctx: &PromptContext, question: Option<&str>, previous_answer: &str,
                 refinement: Refinement, style: AnswerStyle, budget_tokens: usize) -> PromptParts;
}
pub mod ask {
    pub const ID: &str = "ask-openmeet";
    pub fn build(ctx: &PromptContext, user_prompt: &str, budget_tokens: usize) -> PromptParts;
}
pub mod interviewer_followup {
    pub const ID: &str = "interviewer-followup";
    pub fn build(ctx: &PromptContext, candidate_turn: &str, heuristic_claims: &[Claim],
                 max_followups: usize, budget_tokens: usize) -> PromptParts;
    #[derive(Debug, Clone, PartialEq)]
    pub enum FollowupItem {
        Followup { question: String, category: QuestionCategory, reason: String,
                   priority: Priority, trigger: Option<String> },
        Claim { text: String, kind: ClaimKind },
        Topic { name: String, status: TopicStatus },
        Gap { text: String },
    }
    /// Incremental NDJSON parser: tolerant of code fences, partial lines,
    /// blank lines and malformed lines (skipped).
    pub struct FollowupStreamParser { .. }
    impl FollowupStreamParser {
        pub fn new() -> Self;
        pub fn push(&mut self, delta: &str) -> Vec<FollowupItem>;
        pub fn finish(&mut self) -> Vec<FollowupItem>;
    }
}
pub mod question_detection {
    pub const ID: &str = "question-detection";
    pub fn build(text: &str, recent: &[TranscriptLine], mode: Mode) -> PromptParts;
    #[derive(Debug, Clone, PartialEq)]
    pub struct SemanticClassification { pub is_question: bool, pub confidence: f32, pub question: String,
                                        pub category: QuestionCategory, pub requires_response: bool }
    pub fn parse(text: &str) -> Option<SemanticClassification>;   // tolerant JSON extraction
}
pub mod context_compression {
    pub const ID: &str = "context-compression";
    pub fn build(previous_summary: &str, transcript: &[TranscriptLine], mode: Mode) -> PromptParts;
    pub fn parse(text: &str) -> String;
}
pub mod screen_analysis {
    pub const ID: &str = "screen-analysis";
    pub fn build(ctx: &VisionContext, ocr_text: Option<&str>) -> PromptParts;   // the image is attached by the caller
    pub fn parse(text: &str) -> VisionResult;                                   // tolerant JSON → VisionResult; falls back to description = text
}
pub mod meeting_summary {
    pub const ID: &str = "meeting-summary";
    pub struct SummaryInput<'a> {
        pub mode: Mode, pub title: &'a str, pub long_term_summary: &'a str,
        pub transcript: &'a [TranscriptLine], pub questions: &'a [String], pub answers: &'a [String],
        pub followups: &'a [String], pub topics: &'a [Topic], pub claims: &'a [Claim],
        pub observations: &'a [String], pub action_items: &'a [String],
    }
    pub fn build(input: &SummaryInput, budget_tokens: usize) -> PromptParts;
    /// Parses the model's JSON; None if unusable.
    pub fn parse(text: &str, meeting_id: &str, mode: Mode, generated_by: &str) -> Option<MeetingSummary>;
    pub fn render_markdown(summary: &MeetingSummary, title: &str, mode: Mode) -> String;
}
pub mod topic_analysis {
    pub const ID: &str = "topic-analysis";
    pub fn build(transcript: &[TranscriptLine], topics: &[Topic], mode: Mode) -> PromptParts;
    pub fn parse(text: &str) -> Vec<(String, TopicStatus)>;
}
```

**Candidate answer output format** (streamed; the overlay renders sections
progressively, so the format must be line-oriented and stable). Each section
starts with a marker alone on its own line:

```
[[STRUCTURE]]      optional numbered outline (system design / multi-step questions)
[[ANSWER]]         the suggested answer, natural spoken language, first person only
                   when grounded in the profile
[[KEY_POINTS]]     "- " bullets (2-5)
[[EXAMPLE]]        optional
[[FOLLOW_UP]]      optional question the user could ask back
[[BASIS]]          optional: "Known from context: …", "Inference: …", "Suggested framing: …"
[[CAVEAT]]         required when context is insufficient:
                   "Context insufficient — verify before using."
```
Answer rules baked into the system prompt: relevance, factuality, brevity,
natural speech; **never invent** employment history, projects, credentials,
metrics, technologies or accomplishments — only use facts from `<profile>` and
the transcript; mark inference vs. framing; if there is not enough context say
"I don't have enough context to answer this reliably." in [[CAVEAT]]; never
reveal hidden reasoning; "show reasoning" (ShowRationale refinement) means a
short [[BASIS]] of key considerations. Styles: concise, detailed, technical,
simple, STAR (Situation/Task/Action/Result labels inside [[ANSWER]]),
example-driven, step-by-step. Clarification output uses `[[CLARIFY]]` with
"- " bullets plus `[[ANSWER]]` describing what is ambiguous. Ask OpenMeet
output uses `[[ANSWER]]` (+ optional `[[KEY_POINTS]]`).

**Interviewer follow-up output format**: NDJSON, one JSON object per line, no
prose:
```
{"type":"claim","text":"Uses Redis for counters","kind":"technology"}
{"type":"gap","text":"No consistency model discussed"}
{"type":"followup","question":"…","category":"tradeoff","reason":"…","priority":"high","trigger":"…"}
{"type":"topic","name":"Observability","status":"pending"}
```
Follow-ups must be grounded in what the candidate actually said (the system
prompt requires each follow-up to name its trigger statement and the missing
information), must avoid `asked_questions` / `previous_suggestions`
(semantically, not just verbatim), must go deeper instead of repeating, must
not be evaluative (no scores, no hiring recommendations). Max `max_followups`.

**Meeting summary output**: one JSON object with keys `overview`, `keyTopics`,
`questionsAsked`, `answers`, `decisions`, `actionItems`, `openQuestions`,
`technicalTopics`, `followUpSuggestions`, and in interviewer mode
`interviewer: {questionCoverage, clarificationAreas, claimsToVerify,
unexploredTopics}`. Explicitly forbid hiring recommendations, scores or
evaluative judgments of the candidate.

### `intel::vision`
```rust
/// VisionProvider backed by a vision-capable LLM: builds screen_analysis prompt,
/// attaches the PNG as base64 ImageInput, parses VisionResult.
pub struct LlmVisionProvider { .. }
impl LlmVisionProvider { pub fn new(llm: Arc<dyn LlmProvider>) -> Self; }
/// VisionProvider from local OCR only: description derived from OCR text
/// (content-type heuristics: code / terminal / diagram / document / slides),
/// entities via claims::extract_entities. Location Local.
pub struct OcrVisionProvider { .. }
impl OcrVisionProvider { pub fn new(ocr: Arc<dyn OcrEngine>) -> Self; }
/// Builds an observation from OCR text without an engine (shared logic).
pub fn describe_ocr_text(text: &str, ctx: &VisionContext) -> VisionResult;
```
(Both implement `core::providers::VisionProvider`; when `request.ocr_text` is
already present, OcrVisionProvider uses it instead of re-running OCR.)

### `intel::export`
```rust
pub fn render(record: &MeetingRecord, format: ExportFormat, parts: &ExportParts) -> String;
pub fn render_markdown(record: &MeetingRecord, parts: &ExportParts) -> String;
pub fn render_text(record: &MeetingRecord, parts: &ExportParts) -> String;
pub fn render_json(record: &MeetingRecord, parts: &ExportParts) -> String;   // pretty JSON of the selected parts
/// Summary without an LLM (fallback): questions, answers, topics, claims,
/// action items, decisions found heuristically. generated_by = "extractive".
pub fn extractive_summary(record: &MeetingRecord) -> MeetingSummary;
```

### `intel::mock` (deterministic, no network; used by tests, dev tools, demo)
```rust
pub struct MockLlmProvider { .. }   // Clone-able handle (Arc inside)
impl MockLlmProvider {
    /// Default: derives a plausible response from the request kind and prompt
    /// content (answers in [[SECTION]] format referencing the question; NDJSON
    /// follow-ups using entities from the candidate turn; summary JSON;
    /// classification JSON; compression text; screen JSON). Streams in small
    /// chunks.
    pub fn new() -> Self;
    pub fn with_script(responses: Vec<String>) -> Self;      // returned in order, then default behaviour
    pub fn push_response(&self, text: String);               // next response override
    pub fn push_failure(&self, error: ProviderError);        // next request fails
    pub fn set_first_token_latency_ms(&self, ms: u64);
    pub fn set_chunk_delay_ms(&self, ms: u64);
    pub fn requests(&self) -> Vec<LlmRequest>;               // recorded requests
    pub fn request_count(&self) -> usize;
}
// impl LlmProvider: info() id "mock", location Mock; supports_images true;
// stream respects the CancellationToken (yields Err(Cancelled) / ends).
pub struct MockTranscriber { .. }   // impl Transcriber: returns scripted texts in order, then ""
impl MockTranscriber { pub fn new(texts: Vec<String>) -> Self; pub fn with_latency_ms(self, ms: u64) -> Self; pub fn calls(&self) -> usize; }
pub struct MockVisionProvider;      // impl VisionProvider: deterministic from ocr_text / context
pub struct MockEmbeddingProvider;   // impl EmbeddingProvider via HashingEmbedder, location Mock
pub struct MockOcr { pub text: String }   // impl OcrEngine
```

### `intel::demo`
```rust
pub fn scenarios() -> Vec<DemoScenario>;
#[derive(Debug, Clone, PartialEq)]
pub enum DemoStep {
    Say { speaker: SpeakerRole, text: String, delay_ms: u64 },
    Screen { description: String, visible_text: Option<String>, content_type: Option<String>,
             application: Option<String>, delay_ms: u64 },
}
pub fn script(scenario_id: &str) -> Option<Vec<DemoStep>>;
/// Mock LLM preloaded with scenario-specific, high-quality responses.
pub fn demo_llm(scenario_id: &str) -> MockLlmProvider;
```
Scenarios: `rate-limiter-interviewer` (interviewer mode: "How would you design a
rate limiter?" → candidate: "I'd probably use Redis…" → follow-ups about
consistency guarantees, multi-region behaviour, Redis unavailability),
`url-shortener-candidate` (candidate mode: "How would you design a URL
shortener?" → structured answer: requirements, API, short-code generation,
storage, caching, scaling, failure handling), `architecture-screen` (candidate
mode with a screen showing "API → Queue → Workers → DB" and the question "Why did
you choose this architecture?" after earlier discussion of asynchronous
processing). Each script ~6–12 steps with realistic delays (1–4 s).

---

## openmeet-providers (HTTP adapters)

Depends only on core + reqwest (rustls, streaming) + serde/json + futures +
async-trait + tokio + tokio-util + base64 + tracing (+ `async-stream` if useful).

```rust
pub mod http;     // shared reqwest::Client builder: connect timeout 10 s, UA "OpenMeet/<version>"
pub mod sse;      // incremental text/event-stream parser
pub mod openai;   // OpenAiLlm, OpenAiTranscriber, OpenAiEmbeddings (also OpenAI-compatible & local AI sidecar)
pub mod anthropic;// AnthropicLlm
pub mod ollama;   // OllamaLlm, OllamaEmbeddings
pub mod local_ai; // LocalAiClient (sidecar health/status)
pub mod factory;  // settings -> providers

pub struct SseEvent { pub event: Option<String>, pub data: String, pub id: Option<String> }
pub struct SseParser { .. }
impl SseParser { pub fn new() -> Self; pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent>; pub fn finish(&mut self) -> Vec<SseEvent>; }

pub mod factory {
    pub struct BuildContext<'a> { pub settings: &'a Settings, pub secrets: &'a dyn SecretResolver }
    pub fn build_llm(ctx: &BuildContext) -> Result<Arc<dyn LlmProvider>, ProviderError>;
    /// None when no fast model is configured.
    pub fn build_fast_llm(ctx: &BuildContext) -> Option<Result<Arc<dyn LlmProvider>, ProviderError>>;
    pub fn build_transcriber(ctx: &BuildContext) -> Result<Arc<dyn Transcriber>, ProviderError>;
    /// Vision-capable LLM for VisionProviderKind::{Openai, Anthropic, Ollama, OpenaiCompatible};
    /// Ok(None) for Disabled / OcrOnly.
    pub fn build_vision_llm(ctx: &BuildContext) -> Result<Option<Arc<dyn LlmProvider>>, ProviderError>;
    /// Ok(None) for EmbeddingProviderKind::LocalHash (the engine uses the built-in embedder).
    pub fn build_embeddings(ctx: &BuildContext) -> Result<Option<Arc<dyn EmbeddingProvider>>, ProviderError>;
    /// Resolved base URL for a component (defaults applied).
    pub fn resolved_base_url(kind: /* provider kind */, configured: Option<&str>) -> String;
}

pub mod local_ai {
    pub struct LocalAiClient { .. }
    impl LocalAiClient {
        pub fn new(base_url: &str) -> Self;
        /// GET /health -> {"status":"ok","transcription_model":..,"embedding_model":..}
        pub async fn health(&self, cancel: CancellationToken) -> Result<LocalAiHealth, ProviderError>;
    }
    pub struct LocalAiHealth { pub ok: bool, pub transcription_model: Option<String>, pub embedding_model: Option<String> }
}
```

Requirements:
* **Local-only mode**: when `settings.privacy.local_only` and the resolved
  endpoint is remote (`core::providers::location_for_url`), return
  `ProviderError::BlockedByLocalOnly`. Provider `info().location` must be
  computed from the actual endpoint (Ollama on a LAN host is *Remote*).
* **Keys**: from `SecretResolver` (`SecretProvider::{Openai, Anthropic,
  OpenaiCompatible}`); missing key for a remote service →
  `NotConfigured("… API key is not set")`. OpenAI-compatible/local endpoints
  work without a key (send `Authorization` only if one exists). Never put keys
  in error messages or logs.
* **Base URLs**: accept with or without trailing `/v1`
  (`http://localhost:1234/v1` and `http://localhost:1234` both work).
* **Errors**: 401/403 → Unauthorized, 429 → RateLimited (include Retry-After
  seconds if present), 408/504/timeouts → Timeout, connect errors → Network,
  other non-2xx → Http("<status> <short provider error message>").
* **Streaming**: return the stream after the response headers are checked; the
  stream yields `LlmChunk::Delta` for visible text only (never thinking /
  reasoning content), then `LlmChunk::Done{stop_reason, usage}`; it ends
  promptly when the cancel token fires (drop the response to abort the
  connection).
* **OpenAI** (`/v1/chat/completions`, `stream: true`,
  `stream_options.include_usage`): use `max_completion_tokens` for
  api.openai.com, `max_tokens` for compatible servers; reasoning models
  (`gpt-5*`, `o1*`, `o3*`, `o4*`) → omit `temperature`, send
  `reasoning_effort` ("minimal" when latency_sensitive on gpt-5*, else the
  settings effort) and add ~2000 tokens of headroom; images as `image_url`
  data URLs. Transcription: multipart `POST /v1/audio/transcriptions` with
  WAV (`core::audio::encode_wav_pcm16`), `model`, optional `language` and
  `prompt`, `response_format=json`. Embeddings: `POST /v1/embeddings`.
  `list_models`: `GET /v1/models`.
* **Anthropic** (raw HTTP, no SDK exists for Rust): `POST {base}/v1/messages`
  with headers `x-api-key`, `anthropic-version: 2023-06-01`,
  `content-type: application/json`, body `stream: true`, `system`,
  `messages`, `max_tokens`. Parse SSE events `message_start`,
  `content_block_start`, `content_block_delta` (only `delta.type ==
  "text_delta"` produces text; ignore `thinking_delta` and non-text blocks such
  as `thinking` / `fallback`), `message_delta` (stop_reason, usage),
  `message_stop`, `ping`, `error` (→ ProviderError). `stop_reason ==
  "refusal"` → final `Done{stop_reason: Some("refusal")}`. Model-specific
  rules:
  - Default model `claude-opus-5-5`.
  - **Do not send `temperature`/`top_p`/`top_k`** to `claude-opus-4-7`,
    `claude-opus-4-8`, `claude-opus-5*`, `claude-sonnet-5*`, `claude-fable-*`,
    `claude-mythos-*` (400 otherwise). Send temperature only to older models
    (`claude-haiku-4-5`, `*-4-6`, `*-4-5`, 3.x).
  - Never send `thinking: {type: "disabled"}` or `budget_tokens` to
    `claude-opus-5*` / `claude-fable-*` / `claude-sonnet-5-5` (400). Omit
    `thinking` entirely. For latency, send `output_config: {"effort": "low"}`
    when `latency_sensitive`, else the settings effort — only for models that
    support effort (`claude-opus-4-6+`, `claude-sonnet-4-6+`,
    `claude-sonnet-5*`, `claude-opus-5*`, `claude-fable-*`); never for
    `claude-haiku-*`.
  - Because thinking may be always on, add ~4000 tokens of headroom to
    `max_tokens` for those models.
  - **Refusal fallbacks**: for `claude-opus-5*`, `claude-fable-5*` and
    `claude-sonnet-5-5` add header `anthropic-beta:
    server-side-fallback-2026-07-01` and body `"fallbacks": "default"`.
  - No assistant prefill; messages must start with `user` and alternate.
  - Images: content block `{"type":"image","source":{"type":"base64",
    "media_type":"image/png","data":…}}` before the text block.
  - `list_models`: `GET /v1/models` → `data[].id`.
* **Ollama**: `POST /api/chat` (`stream: true`, NDJSON lines
  `{message:{content}, done, done_reason, prompt_eval_count, eval_count}`),
  `options: {temperature, num_predict}`, images as `images: [base64]` on the
  message; ignore any `message.thinking`. Embeddings `POST /api/embed`
  (`{model, input:[..]}` → `embeddings`). `list_models`: `GET /api/tags`.
* **Health checks**: list-models style cheap calls with a 5 s timeout.
* Tests: a local mock HTTP server (e.g. a tiny tokio TCP server or
  `wiremock`) covering SSE parsing across chunk boundaries, streaming,
  error-status mapping, refusal, cancellation mid-stream (server drips data;
  cancel must end the stream < 500 ms), multipart transcription, base URL
  normalization, local-only blocking, model-rule helpers (temperature/effort
  /fallback selection).

---

## openmeet-platform (OS integration)

Depends on core + cpal + xcap + image + sysinfo + raw-window-handle 0.6 +
`windows` (cfg windows) + objc2 crates (cfg macos, only if needed).

```rust
pub enum PlatformError { DeviceNotFound(String), DeviceUnavailable(String), PermissionDenied(String),
                         Unsupported(String), Capture(String), Other(String) }
impl From<PlatformError> for CoreError;

pub mod audio {
    pub fn list_devices() -> Result<AudioDevices, PlatformError>;
    pub fn system_audio_support() -> SystemAudioSupport;
    #[derive(Debug, Clone)]
    pub struct AudioCaptureConfig { pub source: AudioSourceKind, pub device: Option<String> }
    /// Capture runs on a dedicated thread (cpal streams are !Send on some hosts).
    /// Frames are converted to 16 kHz mono f32 with core::audio::{downmix_to_mono, Resampler}
    /// and delivered in ~20–100 ms chunks with captured_at_ms timestamps.
    pub struct AudioCapture { .. }   // Drop = stop
    impl AudioCapture {
        pub fn start(config: AudioCaptureConfig,
                     on_frame: Box<dyn FnMut(AudioFrame) + Send>,
                     on_error: Box<dyn FnMut(PlatformError) + Send>) -> Result<AudioCapture, PlatformError>;
        pub fn stop(self);
        pub fn device_name(&self) -> &str;
        pub fn source(&self) -> AudioSourceKind;
    }
}
```
System audio: Windows → WASAPI loopback (build an *input* stream on the
*output* device; verify cpal's current API for loopback). Linux → PulseAudio /
PipeWire "Monitor of …" input devices. macOS → cpal's Core Audio process tap
on macOS 14.6+ (System Audio Recording permission); on older versions
`supported: false` with a note to use a virtual audio device such as BlackHole.
Device disconnects and stream errors go to `on_error`, never panic.

```rust
pub mod screen {
    pub fn list_sources(rules: &[PrivacyRule]) -> Result<Vec<ScreenSourceInfo>, PlatformError>;
    #[derive(Debug, Clone)]
    pub struct SamplerConfig {
        pub spec: ScreenSourceSpec,
        pub frequency: ScreenFrequency,
        pub change_threshold: f32,
        pub rules: Vec<PrivacyRule>,
        pub max_width: u32,                 // downscale target, default 1280
        pub own_process_id: u32,            // never capture OpenMeet's own windows as a source
    }
    #[derive(Debug, Clone, PartialEq)]
    pub enum ScreenStatus { Capturing, Unchanged, SkippedPrivacy(String), PermissionDenied(String), Error(String) }
    pub struct ScreenSampler { .. }   // dedicated thread; Drop = stop
    impl ScreenSampler {
        pub fn start(config: SamplerConfig,
                     on_frame: Box<dyn FnMut(ScreenFrame) + Send>,
                     on_status: Box<dyn FnMut(ScreenStatus) + Send>) -> Result<ScreenSampler, PlatformError>;
        /// Manual capture: bypasses change detection (ScreenFrame.manual = true), still applies privacy rules.
        pub fn capture_now(&self);
        pub fn update_rules(&self, rules: Vec<PrivacyRule>);
        pub fn stop(self);
    }
    pub fn capture_once(spec: &ScreenSourceSpec, rules: &[PrivacyRule], max_width: u32) -> Result<ScreenFrame, PlatformError>;

    pub mod diff {      // pure, unit-tested
        pub fn dhash(gray: &image::GrayImage) -> u64;           // 64-bit difference hash (9x8)
        pub fn hamming(a: u64, b: u64) -> u32;
        pub struct ChangeDetector { .. }
        impl ChangeDetector {
            pub fn new(threshold: f32) -> Self;
            /// (changed, change_score 0..1, hash). First frame is always changed.
            /// Combines dHash distance with a 32x32 thumbnail mean-abs-diff so small
            /// but meaningful text changes are detected while cursor blinks are not.
            pub fn evaluate(&mut self, frame: &image::RgbaImage) -> (bool, f32, u64);
            pub fn reset(&mut self);
        }
    }
    pub mod rules {     // pure, unit-tested
        /// Some(reason) when app/title matches an application, window or domain rule (case-insensitive).
        pub fn excluded_by(app: Option<&str>, title: Option<&str>, rules: &[PrivacyRule]) -> Option<String>;
        /// Blacks out all region rules ("x,y,w,h" in source pixels, scaled when the image was downscaled).
        pub fn apply_region_masks(img: &mut image::RgbaImage, rules: &[PrivacyRule], scale: f32);
        pub fn default_sensitive_apps() -> Vec<&'static str>;  // password managers, banking, messaging suggestions for the UI
    }
}
```
Display capture applies privacy rules by checking the focused/top-most visible
window: when it matches an exclusion, the frame is skipped
(`SkippedPrivacy`). Window/application capture refuses excluded targets.
Frames: RGBA → apply region masks → downscale (keep aspect, max_width) →
change detection → PNG encode → `ScreenFrame`. macOS screen recording
permission denial → `PermissionDenied("Screen permission required. Open
System Settings → Privacy & Security → Screen Recording.")`.

```rust
pub mod ocr {
    /// Windows: Windows.Media.Ocr (documented WinRT API, fully local).
    /// Other platforms: None.
    pub fn system_ocr() -> Option<Arc<dyn OcrEngine>>;
}

pub mod privacy {   // spec §8
    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    pub enum CapturePrivacyError { #[error("unsupported: {0}")] Unsupported(String), #[error("failed: {0}")] Failed(String) }
    pub trait CapturePrivacy: Send + Sync {
        fn set_excluded_from_capture(&self, enabled: bool) -> Result<(), CapturePrivacyError>;
        fn is_supported(&self) -> bool;
        /// Verified current state when the OS allows reading it back.
        fn is_excluded(&self) -> Option<bool>;
        fn mechanism(&self) -> &'static str;
        fn limitations(&self) -> &'static str;
    }
    /// Windows: SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) (Windows 10 2004+, verified via
    /// GetWindowDisplayAffinity); macOS: NSWindow.sharingType = .none; Linux/others: Unsupported.
    pub fn for_window(handle: raw_window_handle::RawWindowHandle) -> Box<dyn CapturePrivacy>;
    pub fn platform_support() -> (bool /*supported*/, &'static str /*mechanism*/, &'static str /*limitations*/);
}

pub mod focus {     // keep the meeting app focused (spec §7)
    pub fn foreground_window_id() -> Option<String>;
    pub fn restore_foreground(id: &str) -> bool;
    /// Windows: toggles WS_EX_NOACTIVATE so clicks on the overlay do not steal focus.
    pub fn set_no_activate(handle: raw_window_handle::RawWindowHandle, enabled: bool) -> Result<(), PlatformError>;
    pub fn focus_window(id: &str) -> Result<(), PlatformError>;
}

pub mod apps {      // UX only — never injects into other processes
    pub fn detect_meeting_apps() -> Vec<MeetingAppInfo>;   // Zoom, Teams, Slack, Discord, Webex, browsers with Meet/Teams/Zoom titles
}

pub mod system {
    pub struct ProcessStats { .. }    // wraps sysinfo; keep one instance to compute CPU deltas
    impl ProcessStats { pub fn new() -> Self; pub fn sample(&mut self) -> (f64 /*memory MB*/, f32 /*cpu %*/); }
}
```
Honest limitations text (used by docs and UI): capture exclusion only affects
capture tools that honor the OS flag; hardware capture, phone cameras, some
remote-desktop tools and older capture APIs are not affected; macOS 15+
ScreenCaptureKit may not honor `sharingType` for all capture clients.

---

## openmeet-store (SQLite)

Depends on core + rusqlite (`bundled`) + serde_json + parking_lot + tracing.

```rust
pub struct Database { .. }   // Mutex<rusqlite::Connection>; WAL; foreign_keys = ON; busy_timeout 5 s
impl Database {
    pub fn open(path: &std::path::Path) -> Result<Database, StoreError>;   // creates parent dirs, migrates
    pub fn open_in_memory() -> Result<Database, StoreError>;
    pub fn schema_version(&self) -> Result<u32, StoreError>;
    pub fn health_check(&self) -> Result<(), StoreError>;
    pub fn load_settings(&self) -> Result<Option<Settings>, StoreError>;   // tolerant: invalid JSON -> None + warn
    pub fn save_settings(&self, settings: &Settings) -> Result<(), StoreError>;
    pub fn list_privacy_rules(&self) -> Result<Vec<PrivacyRule>, StoreError>;
    pub fn add_privacy_rule(&self, rule: &PrivacyRule) -> Result<(), StoreError>;
    pub fn delete_privacy_rule(&self, id: &str) -> Result<(), StoreError>;
    pub fn list_meetings(&self, limit: u32, offset: u32) -> Result<Vec<MeetingListItem>, StoreError>; // newest first
    pub fn delete_meeting(&self, id: &str) -> Result<(), StoreError>;        // cascades to every child table
    /// Deletes all meeting data (keeps settings and privacy rules), then VACUUM.
    pub fn delete_all_meeting_data(&self) -> Result<(), StoreError>;
    /// Deletes meetings that started before now - max_age (RetentionPolicy::None deletes all).
    pub fn apply_retention(&self, policy: RetentionPolicy, now_ms: i64) -> Result<u32, StoreError>;
    pub fn load_embeddings(&self, meeting_id: &str) -> Result<Vec<EmbeddingRecord>, StoreError>;
    /// Crash recovery: meetings left in a live state become Completed (ended_at = last activity).
    pub fn recover_interrupted_meetings(&self, now_ms: i64) -> Result<u32, StoreError>;
}
impl core::store::MeetingStore for Database { .. }
```
Schema (versioned migrations via `PRAGMA user_version`, each migration in a
transaction, embedded SQL): `meetings`, `participants`, `transcript_segments`,
`questions`, `answers`, `topics`, `claims`, `screen_observations`,
`ai_suggestions`, `action_items`, `meeting_summaries`, `embeddings`,
`privacy_rules`, `settings`. Every meeting-owned row has `id`, `meeting_id`
(FK `ON DELETE CASCADE`), `created_at`/timestamps; indexes on
`(meeting_id, start_time)` etc. Lists (`segment_ids`, `entities`, `parts`,
`followups`) are JSON text columns. Embeddings are little-endian f32 BLOBs.
No raw audio and no screenshots are stored. Tests: migrations from empty,
re-open idempotence, every insert/upsert, `load_meeting` completeness,
ordering, cascade delete, delete-all, retention cutoffs, settings roundtrip
and tolerance, privacy rules CRUD, embeddings roundtrip, crash recovery.
