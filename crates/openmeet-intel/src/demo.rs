//! Scripted demo meetings (spec §74): the complete UX without microphones,
//! accounts or network access. The engine runs its real pipeline on scripted
//! dialogue; [`demo_llm`] answers with scenario-specific, hand-written
//! responses (falling back to the default mock for anything unexpected).
//! The answer-integrity scenario also gets a scripted sidecar
//! ([`demo_integrity`]) and a stand-in embedder ([`demo_embeddings`]).

use std::sync::Arc;

use openmeet_core::integrity::IntegrityLevel;
use openmeet_core::ipc::DemoScenario;
use openmeet_core::providers::LlmRequest;
use openmeet_core::types::{AnswerSections, Mode, RequestKind, SpeakerRole};
use serde_json::json;

use crate::mock::{MockEmbeddingProvider, MockIntegrityService, MockLlmProvider, render_sections};
use crate::prompts::shared::{extract_block, extract_question};

/// One step of a demo script.
#[derive(Debug, Clone, PartialEq)]
pub enum DemoStep {
    /// Someone speaks (streamed as partial transcripts, then a final).
    Say {
        /// Who speaks.
        speaker: SpeakerRole,
        /// What is said.
        text: String,
        /// Pause before this step.
        delay_ms: u64,
        /// Speaking pace; `None` streams the words quickly (no realistic
        /// timing needed).
        words_per_minute: Option<u32>,
    },
    /// The shared screen changes.
    Screen {
        /// What the screen shows.
        description: String,
        /// Text visible on screen.
        visible_text: Option<String>,
        /// "diagram", "code", …
        content_type: Option<String>,
        /// Application showing the content.
        application: Option<String>,
        /// Pause before this step.
        delay_ms: u64,
    },
}

/// Interviewer-mode rate limiter scenario.
pub const RATE_LIMITER: &str = "rate-limiter-interviewer";
/// Candidate-mode system design scenario.
pub const URL_SHORTENER: &str = "url-shortener-candidate";
/// Candidate-mode scenario with a shared diagram.
pub const ARCHITECTURE_SCREEN: &str = "architecture-screen";
/// Interviewer-mode answer-integrity scenario.
pub const INTEGRITY: &str = "integrity-interviewer";

/// Available demo scenarios.
pub fn scenarios() -> Vec<DemoScenario> {
    vec![
        DemoScenario {
            id: URL_SHORTENER.into(),
            title: "System design interview (candidate)".into(),
            mode: Mode::Candidate,
            description: "You are interviewing for a backend role. OpenMeet detects each question and suggests a structured answer — including a URL shortener design.".into(),
        },
        DemoScenario {
            id: RATE_LIMITER.into(),
            title: "Rate limiter deep-dive (interviewer)".into(),
            mode: Mode::Interviewer,
            description: "You are the interviewer. OpenMeet listens to the candidate and suggests grounded follow-up questions, tracking topics as you go.".into(),
        },
        DemoScenario {
            id: INTEGRITY.into(),
            title: "Answer integrity check (interviewer)".into(),
            mode: Mode::Interviewer,
            description: "You are the interviewer with the answer-integrity check on. Three natural answers set the candidate's baseline; the fourth sounds read out, and OpenMeet shows a private flag with its evidence — a reason to ask a follow-up, never a verdict.".into(),
        },
        DemoScenario {
            id: ARCHITECTURE_SCREEN.into(),
            title: "Architecture on screen (candidate)".into(),
            mode: Mode::Candidate,
            description: "A diagram is shared on screen. OpenMeet combines what is said with what is visible to answer \"Why did you choose this architecture?\"".into(),
        },
    ]
}

fn say(speaker: SpeakerRole, text: &str, delay_ms: u64) -> DemoStep {
    DemoStep::Say {
        speaker,
        text: text.to_string(),
        delay_ms,
        words_per_minute: None,
    }
}

fn say_paced(speaker: SpeakerRole, text: &str, delay_ms: u64, words_per_minute: u32) -> DemoStep {
    DemoStep::Say {
        speaker,
        text: text.to_string(),
        delay_ms,
        words_per_minute: Some(words_per_minute),
    }
}

/// The answer the integrity demo's candidate reads out (and the demo LLM's
/// reference answer to the same question).
const READ_ANSWER: &str = "Consistent hashing distributes keys across a logical ring, ensuring that adding or removing infrastructure only remaps a proportional fraction of keys, which significantly minimizes rebalancing overhead and maintains predictable performance characteristics.";
/// The question the integrity demo's candidate answers by reading.
const READ_QUESTION: &str =
    "Let's go deeper. How does consistent hashing help when you add or remove cache nodes?";

/// The script of a scenario.
pub fn script(scenario_id: &str) -> Option<Vec<DemoStep>> {
    use SpeakerRole::{Remote, User};
    Some(match scenario_id {
        RATE_LIMITER => vec![
            say(User, "Thanks for taking the time today. Let's start with a design question: how would you design a rate limiter for a public API?", 1_200),
            say(Remote, "I'd probably use Redis. Each API key gets a token bucket stored in Redis, and every request runs a small Lua script that refills tokens based on the elapsed time and takes one atomically. If the bucket is empty we return a 429 with a Retry-After header. I'd run the check in the API gateway so every service gets it for free.", 2_500),
            say(User, "What happens if Redis goes down?", 9_000),
            say(Remote, "Good question. For most endpoints I'd fail open and fall back to a local in-memory limiter on each gateway node, so a Redis outage doesn't take the whole API down. For expensive endpoints like bulk exports I'd fail closed instead. And we'd alert on Redis latency and error rates so we notice quickly.", 2_500),
            say(User, "Great. Let's switch to observability. How would you know the limiter is working correctly in production?", 9_000),
            say(Remote, "I'd emit metrics per key and per endpoint: allowed versus throttled requests, Redis script latency, and how often the fallback kicks in. Dashboards in Grafana, plus an alert if the throttle rate jumps suddenly, because that usually means a misconfigured limit rather than abuse.", 2_500),
        ],
        URL_SHORTENER => vec![
            say(Remote, "Hi, thanks for joining! To get started, could you give me a quick overview of your background?", 1_200),
            say(User, "Sure. I'm a backend engineer, and for the last few years I've mostly worked on APIs and data pipelines.", 7_000),
            say(Remote, "Great. Let's do a design exercise. How would you design a URL shortener?", 2_500),
            say(User, "I'd start by clarifying the expected scale, how many new links per day and how many redirects, and then sketch the API and the storage.", 9_000),
            say(Remote, "Makes sense. How would you generate the short codes so they never collide?", 3_000),
            say(User, "I'd hand out ranges from a counter and encode the IDs in base62, so codes are unique by construction.", 9_000),
            say(Remote, "We have a few minutes left. Do you have any questions for me?", 3_000),
        ],
        // Paced speech so delivery has realistic timing: three natural
        // answers (quick start, ~150 words/min, fillers) form the baseline;
        // the fourth starts late, runs fast and has no fillers.
        INTEGRITY => vec![
            say_paced(User, "Thanks for joining. Can you tell me about a project you're proud of?", 1_200, 170),
            say_paced(Remote, "Um, sure. So last year I rebuilt our billing exports. Uh, the old job kept timing out, you know, so I split it into small batches and we added retries. It mostly just worked after that.", 1_200, 150),
            say_paced(User, "Nice. How did you test it before launch?", 4_500, 170),
            say_paced(Remote, "Hmm, mostly with a copy of production data. I mean, we ran the old and the new job side by side for a week, um, and compared the totals every morning before we switched over.", 1_400, 145),
            say_paced(User, "What would you do differently next time?", 4_500, 170),
            say_paced(Remote, "Uh, honestly I'd add better alerts earlier. You know, we only noticed one slow batch because a customer asked, so, um, I'd watch the batch times from day one.", 1_000, 155),
            say_paced(User, READ_QUESTION, 4_500, 170),
            say_paced(Remote, READ_ANSWER, 4_600, 215),
        ],
        ARCHITECTURE_SCREEN => vec![
            say(Remote, "Can you tell me about the reporting system you worked on?", 1_200),
            say(User, "Sure. Large customers' reports were timing out, so we moved report generation into asynchronous jobs with a queue and a pool of workers.", 7_000),
            DemoStep::Screen {
                description: "System architecture diagram: API Gateway → Job Queue → Workers → PostgreSQL, with retries and a dead-letter queue".into(),
                visible_text: Some("API Gateway → Job Queue (SQS)\nJob Queue → Worker pool (autoscaling)\nWorkers → PostgreSQL\nRetries: 3 · Dead-letter queue".into()),
                content_type: Some("diagram".into()),
                application: Some("Excalidraw".into()),
                delay_ms: 3_000,
            },
            say(Remote, "Why did you choose this architecture?", 2_500),
            say(User, "Mainly because generation time was unpredictable, and we didn't want slow reports to hold HTTP requests open.", 9_000),
            say(Remote, "What happens if a worker crashes halfway through a report?", 3_000),
        ],
        _ => return None,
    })
}

fn sections(
    structure: &[&str],
    answer: &str,
    key_points: &[&str],
    example: Option<&str>,
    follow_up: Option<&str>,
    basis: Option<&str>,
    caveat: Option<&str>,
) -> String {
    render_sections(&AnswerSections {
        structure: structure.iter().map(|s| s.to_string()).collect(),
        answer: answer.to_string(),
        key_points: key_points.iter().map(|s| s.to_string()).collect(),
        example: example.map(str::to_string),
        follow_up: follow_up.map(str::to_string),
        basis: basis.map(str::to_string),
        caveat: caveat.map(str::to_string),
    })
}

fn followups(items: &[(&str, &str, &str, &str, &str)], extra: &[serde_json::Value]) -> String {
    let mut lines: Vec<String> = extra.iter().map(|v| v.to_string()).collect();
    for (q, cat, reason, prio, trigger) in items {
        lines.push(
            json!({"type": "followup", "question": q, "category": cat, "reason": reason, "priority": prio, "trigger": trigger})
                .to_string(),
        );
    }
    lines.join("\n") + "\n"
}

fn user_of(r: &LlmRequest) -> String {
    r.messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

fn rate_limiter(r: &LlmRequest) -> Option<String> {
    let user = user_of(r);
    match r.kind {
        RequestKind::InterviewerFollowups => {
            let turn = extract_block(&user, "candidate_answer").unwrap_or_default().to_lowercase();
            if turn.contains("grafana") || turn.contains("throttled") {
                Some(followups(
                    &[
                        ("How would you tell abuse apart from a misconfigured limit using those metrics?", "follow_up", "The candidate said a spike usually means misconfiguration but not how to distinguish the two.", "high", "that usually means a misconfigured limit rather than abuse"),
                        ("What throttle-rate change would page someone, and how would you choose that threshold?", "technical", "Alerting was mentioned without concrete thresholds or ownership.", "normal", "plus an alert if the throttle rate jumps suddenly"),
                        ("How would you trace one customer's 429s back to the exact limit that triggered them?", "technical", "Per-key metrics were described, but not request-level debugging.", "optional", "I'd emit metrics per key and per endpoint"),
                    ],
                    &[json!({"type": "claim", "text": "Monitors allowed vs throttled requests per key and endpoint in Grafana", "kind": "implementation"}), json!({"type": "topic", "name": "Observability", "status": "current"}), json!({"type": "topic", "name": "Security", "status": "pending"})],
                ))
            } else if turn.contains("fail open") || turn.contains("fallback") || turn.contains("fall back") {
                Some(followups(
                    &[
                        ("How would the local fallback limiters stay close to the global limit while Redis is down?", "tradeoff", "Per-node in-memory limits multiply the effective limit by the number of gateway nodes.", "high", "fall back to a local in-memory limiter on each gateway node"),
                        ("How do you decide which endpoints fail closed instead of open?", "tradeoff", "The candidate named exports as fail-closed but not the decision rule.", "normal", "For expensive endpoints like bulk exports I'd fail closed instead"),
                        ("How would you test the failover path before a real Redis outage happens?", "technical", "Failure handling was designed but its verification was not discussed.", "optional", "so a Redis outage doesn't take the whole API down"),
                    ],
                    &[json!({"type": "claim", "text": "Fails open with a per-node in-memory limiter when Redis is unavailable", "kind": "decision"}), json!({"type": "gap", "text": "Effective limit during fallback is not bounded"}), json!({"type": "topic", "name": "Reliability & failure handling", "status": "current"})],
                ))
            } else if turn.contains("redis") {
                Some(followups(
                    &[
                        ("What consistency guarantees do you need when several gateway nodes update the same bucket?", "tradeoff", "Atomic Lua updates were described, but not consistency across nodes or replicas.", "high", "every request runs a small Lua script that refills tokens based on the elapsed time and takes one atomically"),
                        ("How would this behave across multiple regions?", "system_design", "A single Redis implies one region; cross-region latency and global limits were not covered.", "normal", "Each API key gets a token bucket stored in Redis"),
                        ("What happens when Redis becomes unavailable?", "technical", "Redis sits on every request path, but failure handling wasn't mentioned.", "high", "I'd probably use Redis."),
                    ],
                    &[json!({"type": "claim", "text": "Token bucket per API key in Redis, updated atomically by a Lua script", "kind": "implementation"}), json!({"type": "claim", "text": "Limit enforced in the API gateway", "kind": "decision"}), json!({"type": "gap", "text": "No multi-region or failure behaviour described"}), json!({"type": "topic", "name": "System design", "status": "current"})],
                ))
            } else {
                None
            }
        }
        RequestKind::Summary => Some(
            json!({
                "overview": "Design discussion of a rate limiter for a public API. The candidate proposed per-key token buckets in Redis, enforced at the API gateway, then covered failure handling and observability.",
                "keyTopics": ["Rate limiting design", "Reliability & failure handling", "Observability"],
                "questionsAsked": ["How would you design a rate limiter for a public API?", "What happens if Redis goes down?", "How would you know the limiter is working correctly in production?"],
                "answers": ["Token bucket per API key in Redis, updated atomically with a Lua script; 429 with Retry-After; enforced in the API gateway.", "Fail open with a local in-memory limiter for most endpoints, fail closed for expensive ones; alert on Redis latency and errors.", "Per-key and per-endpoint metrics (allowed vs throttled, script latency, fallback rate) in Grafana with alerts on sudden throttle-rate changes."],
                "decisions": ["Enforce limits in the API gateway", "Fail open by default, fail closed for expensive endpoints"],
                "actionItems": [],
                "openQuestions": ["How is the effective limit bounded while running on local fallback limiters?"],
                "technicalTopics": ["Redis", "Lua", "API gateway", "Grafana"],
                "followUpSuggestions": ["Explore multi-region behaviour", "Ask how the failover path would be tested"],
                "interviewer": {
                    "questionCoverage": ["System design — discussed", "Reliability & failure handling — discussed", "Observability — discussed", "Security — not discussed"],
                    "clarificationAreas": ["Consistency of buckets across gateway nodes", "Effective limits during Redis fallback"],
                    "claimsToVerify": ["Lua script makes refill-and-take atomic", "Throttle-rate spikes usually indicate misconfiguration"],
                    "unexploredTopics": ["Security", "Testing & quality"]
                }
            })
            .to_string(),
        ),
        _ => None,
    }
}

fn url_shortener(r: &LlmRequest) -> Option<String> {
    let user = user_of(r);
    let q = extract_question(&user).unwrap_or_default().to_lowercase();
    match r.kind {
        RequestKind::CandidateAnswer if q.contains("background") || q.contains("yourself") => Some(sections(
            &[],
            "Keep it to about a minute: your current role and focus, one or two projects that match this job, and why you're interested in this team.",
            &["Current role and main focus", "One relevant project with a concrete outcome", "What draws you to this role"],
            None,
            None,
            Some("Suggested framing: a standard short introduction. OpenMeet does not know your history."),
            Some("Context insufficient — verify before using. Add your background in Settings → Profile so suggestions can use your real experience."),
        )),
        RequestKind::CandidateAnswer if q.contains("url shortener") || q.contains("shortener") => Some(sections(
            &["Requirements", "API design", "Short-code generation", "Storage", "Caching", "Scaling", "Failure handling"],
            "I'd start by clarifying the expected scale — new links per day versus redirects per second, since reads usually dominate by 100 to 1. Then a tiny API: POST to create a short link, GET on the code to redirect with a 301 or 302. For codes I'd encode a unique ID in base62, store code-to-URL mappings in a key-value store, and put a cache in front because popular links get hit constantly.",
            &["Reads dominate: optimize the redirect path first", "Base62 of a unique ID avoids collisions by construction", "Cache hot links close to users; the store stays simple", "Plan for abuse: rate limits and malicious URL checks"],
            Some("At 100 million new links a month, 7 base62 characters give about 3.5 trillion codes — decades of headroom."),
            Some("What read and write volumes should I design for?"),
            Some("Suggested framing: standard URL-shortener design; the numbers are illustrative assumptions."),
            None,
        )),
        RequestKind::CandidateAnswer if q.contains("collide") || q.contains("short code") => Some(sections(
            &[],
            "I'd avoid collisions by construction rather than by checking: a counter service hands out ID ranges to each app server, and each server encodes its next ID in base62. No two servers ever get the same range, so no lookups are needed on the write path.",
            &["Range allocation from a counter (or a key-generation service)", "Base62 encoding keeps codes short", "If codes must look random, shuffle the ID with a reversible permutation"],
            Some("A server grabs IDs 1,000,000–1,000,999, so its next code is base62(1000000) = \"4c92\"."),
            Some("Do the codes need to be unguessable, or is sequential acceptable?"),
            Some("Inference: builds on the base62 approach from the previous answer."),
            None,
        )),
        RequestKind::CandidateAnswer if q.contains("questions for me") || q.contains("any questions") => Some(sections(
            &[],
            "Yes — ask one or two questions that show you were listening and care about how the team works.",
            &["\"What does the read/write traffic look like for your services today?\"", "\"How do teams here decide when to take on a redesign like this?\"", "\"What would success look like for this role in the first six months?\""],
            None,
            None,
            Some("Suggested framing: questions grounded in this conversation's design topic."),
            None,
        )),
        RequestKind::Summary => Some(
            json!({
                "overview": "A candidate-side system design interview: a short introduction, then a URL shortener design covering API, short-code generation, storage, caching and scaling.",
                "keyTopics": ["Background", "System design", "Short-code generation"],
                "questionsAsked": ["Could you give me a quick overview of your background?", "How would you design a URL shortener?", "How would you generate the short codes so they never collide?", "Do you have any questions for me?"],
                "answers": ["Backend engineer focused on APIs and data pipelines.", "Clarify scale first, then API and storage.", "Counter-based ID ranges encoded in base62."],
                "decisions": [],
                "actionItems": [],
                "openQuestions": ["Expected read/write volumes were not specified"],
                "technicalTopics": ["Base62 encoding", "Caching", "Key-value storage"],
                "followUpSuggestions": ["Ask about the team's traffic patterns", "Ask how success is measured in the role"]
            })
            .to_string(),
        ),
        _ => None,
    }
}

fn architecture(r: &LlmRequest) -> Option<String> {
    let user = user_of(r);
    let q = extract_question(&user).unwrap_or_default().to_lowercase();
    match r.kind {
        RequestKind::CandidateAnswer if q.contains("reporting system") => Some(sections(
            &[],
            "Describe the problem first (reports timing out for large customers), then the change you made and the result. Keep the details to what you actually worked on.",
            &["The problem and who it affected", "The architectural change", "The measurable outcome"],
            None,
            None,
            Some("Suggested framing: problem → change → result."),
            Some("Context insufficient — verify before using. OpenMeet only knows what was said in this meeting."),
        )),
        RequestKind::CandidateAnswer if q.contains("architecture") => Some(sections(
            &[],
            "Mainly because report generation time was unpredictable. Large customers' reports were timing out on synchronous requests, so we put a job queue between the API and a pool of workers: the API responds immediately, workers scale with the backlog, and one slow report can't hold up everything else.",
            &["Decouples API latency from report generation time", "Workers scale independently with queue depth", "Failures are isolated and retried — the diagram shows 3 retries and a dead-letter queue"],
            None,
            Some("Would it help if I walked through how failed jobs are retried?"),
            Some("Known from context: the diagram shows API Gateway → Job Queue → Workers → PostgreSQL with retries and a dead-letter queue, and you said large reports were timing out. Suggested framing: the decoupling rationale."),
            None,
        )),
        RequestKind::CandidateAnswer if q.contains("crash") || q.contains("worker") => Some(sections(
            &[],
            "The job isn't lost: the message only leaves the queue when the worker acknowledges it, so if a worker crashes mid-report the message becomes visible again and another worker picks it up. After three failed attempts it moves to the dead-letter queue so someone can look at it. To make retries safe, report generation needs to be idempotent — for example writing results under the job ID so a retry overwrites instead of duplicating.",
            &["Acknowledge only after the report is written", "Retries are bounded (3) with a dead-letter queue", "Idempotent writes keyed by job ID make retries safe"],
            None,
            None,
            Some("Known from context: \"Retries: 3 · Dead-letter queue\" is visible on the shared diagram. Inference: visibility-timeout behaviour is how SQS-style queues usually work — confirm it matches your system."),
            None,
        )),
        RequestKind::Summary => Some(
            json!({
                "overview": "Discussion of an asynchronous reporting architecture shown on screen: API gateway, job queue, autoscaling workers and PostgreSQL, with retries and a dead-letter queue.",
                "keyTopics": ["Architecture", "Reliability & failure handling"],
                "questionsAsked": ["Can you tell me about the reporting system you worked on?", "Why did you choose this architecture?", "What happens if a worker crashes halfway through a report?"],
                "answers": ["Report generation moved to asynchronous jobs because large reports were timing out.", "Unpredictable generation time; avoid holding HTTP requests open."],
                "decisions": ["Move report generation to asynchronous jobs"],
                "actionItems": [],
                "openQuestions": ["How idempotency of report writes is guaranteed"],
                "technicalTopics": ["SQS", "PostgreSQL", "Dead-letter queue", "Autoscaling workers"],
                "followUpSuggestions": ["Walk through the retry and idempotency design"]
            })
            .to_string(),
        ),
        _ => None,
    }
}

/// Reference answers of the integrity demo: the read-out question gets the
/// very answer the candidate reads; other questions the default mock answer.
fn integrity(r: &LlmRequest) -> Option<String> {
    if r.kind != RequestKind::IntegrityReference {
        return None;
    }
    // The reference recipe sends the bare question as the user message.
    user_of(r)
        .to_lowercase()
        .contains("consistent hashing")
        .then(|| READ_ANSWER.to_string())
}

/// Scripted answer-integrity sidecar for a scenario (`None` for scenarios
/// without the check). Reading scores are queued per answer in script order;
/// AI-text scores come from the mock's deterministic text heuristic.
pub fn demo_integrity(scenario_id: &str) -> Option<MockIntegrityService> {
    if scenario_id != INTEGRITY {
        return None;
    }
    let svc = MockIntegrityService::new().with_embedding_model(
        crate::integrity::spec_thresholds()
            .echo_similarity
            .embedding_model
            .clone(),
    );
    for (p, level) in [
        (0.14, IntegrityLevel::Low),
        (0.22, IntegrityLevel::Low),
        (0.18, IntegrityLevel::Low),
        (0.91, IntegrityLevel::High),
    ] {
        svc.push_reading(Some(p), level);
    }
    Some(svc)
}

/// Demo embedder for a scenario: deterministic hashing embeddings reported
/// under the model name the echo thresholds are set for, so the demo can show
/// the echo signal without the local AI sidecar. `None` for other scenarios.
pub fn demo_embeddings(scenario_id: &str) -> Option<MockEmbeddingProvider> {
    (scenario_id == INTEGRITY).then(|| {
        MockEmbeddingProvider::new().with_model(
            crate::integrity::spec_thresholds()
                .echo_similarity
                .embedding_model
                .clone(),
        )
    })
}

/// Mock LLM preloaded with scenario-specific responses.
pub fn demo_llm(scenario_id: &str) -> MockLlmProvider {
    let responder: crate::mock::Responder = match scenario_id {
        RATE_LIMITER => Arc::new(rate_limiter),
        URL_SHORTENER => Arc::new(url_shortener),
        ARCHITECTURE_SCREEN => Arc::new(architecture),
        INTEGRITY => Arc::new(integrity),
        _ => Arc::new(|_: &LlmRequest| None),
    };
    let m = MockLlmProvider::with_responder(responder);
    // A natural streaming pace for the demo.
    m.set_chunk_delay_ms(28);
    m.set_first_token_latency_ms(350);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::interviewer_followup::{self, FollowupItem, FollowupStreamParser};
    use crate::prompts::{PromptContext, candidate_answer, meeting_summary};
    use crate::text::word_count;
    use openmeet_core::providers::{ChatMessage, complete_text};
    use openmeet_core::types::{AnswerStyle, QuestionCategory, ResponseLength};
    use tokio_util::sync::CancellationToken;

    #[test]
    fn scripts_are_well_formed() {
        for s in scenarios() {
            let steps = script(&s.id).expect("script exists");
            assert!((6..=12).contains(&steps.len()), "{}", s.id);
            for step in &steps {
                if let DemoStep::Say {
                    speaker,
                    text,
                    delay_ms,
                    ..
                } = step
                {
                    assert!((1_000..=10_000).contains(delay_ms));
                    let candidate_speaks = match s.mode {
                        Mode::Interviewer => *speaker == SpeakerRole::Remote,
                        Mode::Candidate | Mode::Workplace => *speaker == SpeakerRole::User,
                    };
                    if candidate_speaks && s.mode == Mode::Interviewer {
                        assert!(
                            word_count(text) >= 25,
                            "candidate turns must trigger follow-ups: {text}"
                        );
                    }
                }
            }
        }
        assert!(script("nope").is_none());
    }

    fn request(kind: RequestKind, system: String, user: String) -> LlmRequest {
        LlmRequest {
            request_id: "r".into(),
            kind,
            system,
            messages: vec![ChatMessage::user(user)],
            max_output_tokens: 1000,
            temperature: None,
            json_output: false,
            latency_sensitive: true,
            seed: None,
        }
    }

    #[tokio::test]
    async fn rate_limiter_followups_match_the_spec_example() {
        let m = demo_llm(RATE_LIMITER);
        m.set_chunk_delay_ms(0);
        m.set_first_token_latency_ms(0);
        let Some(DemoStep::Say { text, .. }) = script(RATE_LIMITER).unwrap().get(1).cloned() else {
            panic!()
        };
        let ctx = PromptContext {
            mode: Mode::Interviewer,
            ..Default::default()
        };
        let p = interviewer_followup::build(&ctx, &text, &[], 4, 3000);
        let out = complete_text(
            &m,
            request(RequestKind::InterviewerFollowups, p.system, p.user),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let mut parser = FollowupStreamParser::new();
        let mut items = parser.push(&out);
        items.extend(parser.finish());
        let qs: Vec<String> = items
            .iter()
            .filter_map(|i| match i {
                FollowupItem::Followup { question, .. } => Some(question.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(qs.len(), 3);
        assert!(qs.iter().any(|q| q.contains("consistency")));
        assert!(qs.iter().any(|q| q.contains("multiple regions")));
        assert!(qs.iter().any(|q| q.contains("Redis becomes unavailable")));
    }

    #[tokio::test]
    async fn candidate_scenarios_answer_with_sections() {
        let m = demo_llm(URL_SHORTENER);
        m.set_chunk_delay_ms(0);
        m.set_first_token_latency_ms(0);
        let p = candidate_answer::build(
            &PromptContext::default(),
            "How would you design a URL shortener?",
            QuestionCategory::SystemDesign,
            AnswerStyle::Concise,
            ResponseLength::Medium,
            2000,
        );
        let out = complete_text(
            &m,
            request(RequestKind::CandidateAnswer, p.system, p.user),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        let s = candidate_answer::parse(&out);
        assert_eq!(s.structure.len(), 7);
        assert!(
            s.answer
                .starts_with("I'd start by clarifying the expected scale")
        );
    }

    #[tokio::test]
    async fn demo_summaries_parse() {
        for id in [RATE_LIMITER, URL_SHORTENER, ARCHITECTURE_SCREEN] {
            let m = demo_llm(id);
            m.set_chunk_delay_ms(0);
            m.set_first_token_latency_ms(0);
            let mode = scenarios().into_iter().find(|s| s.id == id).unwrap().mode;
            let input = meeting_summary::SummaryInput {
                mode,
                title: "Demo",
                long_term_summary: "",
                transcript: &[],
                questions: &[],
                answers: &[],
                followups: &[],
                topics: &[],
                claims: &[],
                observations: &[],
                action_items: &[],
            };
            let p = meeting_summary::build(&input, 2000);
            let out = complete_text(
                &m,
                request(RequestKind::Summary, p.system, p.user),
                CancellationToken::new(),
            )
            .await
            .unwrap();
            let s = meeting_summary::parse(&out, "m", mode, "demo").expect(id);
            assert!(!s.overview.is_empty());
            assert_eq!(s.interviewer.is_some(), mode == Mode::Interviewer);
        }
    }

    #[test]
    fn integrity_demo_questions_are_detected_and_answers_are_not() {
        use crate::detect::{DetectionInput, detect};
        use openmeet_core::types::{AudioSourceKind, TranscriptSegment};
        for step in script(INTEGRITY).unwrap() {
            let DemoStep::Say { speaker, text, .. } = step else {
                continue;
            };
            let seg = TranscriptSegment {
                id: "s".into(),
                meeting_id: "m".into(),
                utterance_id: "s".into(),
                speaker_id: None,
                speaker_role: speaker,
                source: AudioSourceKind::System,
                start_time: 10_000,
                end_time: 14_000,
                text: text.clone(),
                is_final: true,
                confidence: None,
            };
            let r = detect(&DetectionInput {
                segment: &seg,
                recent: &[],
                mode: Mode::Interviewer,
                last_question: None,
            });
            // Interviewer questions are accepted at 0.85 x the default
            // detection threshold (0.6).
            let accepted = r.is_question && r.confidence >= 0.6 * 0.85;
            assert_eq!(accepted, speaker == SpeakerRole::User, "{text}: {r:?}");
        }
    }

    #[tokio::test]
    async fn integrity_demo_has_a_baseline_and_a_read_answer() {
        use crate::integrity::is_echo_embedding_model;
        use crate::prompts::integrity_reference;
        use openmeet_core::integrity::IntegrityService;
        use openmeet_core::providers::EmbeddingProvider;

        let steps = script(INTEGRITY).unwrap();
        let answers: Vec<&DemoStep> = steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    DemoStep::Say {
                        speaker: SpeakerRole::Remote,
                        ..
                    }
                )
            })
            .collect();
        // Three baseline answers plus the one that is checked.
        assert_eq!(answers.len(), 4);
        assert!(steps.iter().all(|s| matches!(
            s,
            DemoStep::Say {
                words_per_minute: Some(_),
                ..
            }
        )));

        // The mock sidecar scores the natural answers low and the read one high.
        let svc = demo_integrity(INTEGRITY).unwrap();
        let mut levels = Vec::new();
        for a in &answers {
            let DemoStep::Say { text, .. } = a else {
                unreachable!()
            };
            levels.push(
                svc.ai_text(text.clone(), CancellationToken::new())
                    .await
                    .unwrap()
                    .level,
            );
        }
        assert_eq!(&levels[..3], &[IntegrityLevel::Low; 3], "{levels:?}");
        assert_eq!(levels[3], IntegrityLevel::High);
        let status = svc.status(CancellationToken::new()).await.unwrap();
        assert!(is_echo_embedding_model(
            status.embedding_model.as_deref().unwrap(),
            crate::integrity::spec_thresholds()
        ));
        assert!(demo_integrity(RATE_LIMITER).is_none());

        // The demo LLM's reference for the read question is the read answer.
        let m = demo_llm(INTEGRITY);
        m.set_chunk_delay_ms(0);
        m.set_first_token_latency_ms(0);
        let p = integrity_reference::build(READ_QUESTION);
        let out = complete_text(
            &m,
            request(RequestKind::IntegrityReference, p.system, p.user),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(integrity_reference::parse(&out), READ_ANSWER);
        let e = demo_embeddings(INTEGRITY).unwrap();
        assert!(is_echo_embedding_model(
            &e.info().model,
            crate::integrity::spec_thresholds()
        ));
    }
}
