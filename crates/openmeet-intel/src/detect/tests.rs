use super::*;
use openmeet_core::types::{AudioSourceKind, DetectionMethod};

fn seg(id: &str, role: SpeakerRole, start: i64, end: i64, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        id: id.into(),
        meeting_id: "m".into(),
        utterance_id: id.into(),
        speaker_id: None,
        speaker_role: role,
        source: AudioSourceKind::System,
        start_time: start,
        end_time: end,
        text: text.into(),
        is_final: true,
        confidence: None,
    }
}

fn run(mode: Mode, role: SpeakerRole, text: &str) -> DetectionResult {
    let s = seg("s", role, 10_000, 12_000, text);
    detect(&DetectionInput {
        segment: &s,
        recent: &[],
        mode,
        last_question: None,
    })
}

fn remote(text: &str) -> DetectionResult {
    run(Mode::Candidate, SpeakerRole::Remote, text)
}

// ---------------------------------------------------------------------------
// Table-driven: (text, is_question)
// ---------------------------------------------------------------------------

const QUESTIONS: &[&str] = &[
    // Direct, with punctuation
    "How did you scale the database?",
    "What was the hardest bug you fixed last year?",
    "Why did you choose Kafka for the event pipeline?",
    "Can you explain how the cache invalidation worked?",
    "Have you ever worked with Kubernetes in production?",
    "Would you do anything differently today?",
    "Is there a reason you didn't use Postgres?",
    "Which component became the bottleneck?",
    "Where do you see yourself in five years?",
    "Who owned the deployment pipeline?",
    "Do you have any questions for me?",
    "What's your experience with distributed systems?",
    "How's the on-call rotation organized?",
    // Direct, no punctuation (typical ASR output)
    "how did you handle scaling the database",
    "what database did you choose and why",
    "can you walk me through your last project",
    "did you write tests for that",
    "so how would you design a url shortener",
    "and what happens when redis goes down",
    "um okay so what did the architecture look like",
    // Imperative interview prompts
    "Tell me about a time you disagreed with your manager.",
    "Walk me through how you would design a rate limiter.",
    "Talk me through the trade-offs of that approach.",
    "Explain the difference between a process and a thread.",
    "Describe the architecture of your current system.",
    "Give me an example of a project you're proud of.",
    "Design a notification system for a mobile app.",
    "Write a function that reverses a linked list.",
    "Imagine you have ten million users. How would the system change?",
    "Let's say you need to support multiple regions",
    "Please describe how you monitor the service.",
    // Embedded questions
    "I was wondering how you approached the migration.",
    "I'm curious why you picked a document database there.",
    "I'd love to hear how you measured the improvement.",
    "I'd like to understand how the workers recover from failures.",
    // Multi-part
    "What database did you use, and why did you choose it?",
    "How long did the migration take and what went wrong?",
    // Polite / prefixed
    "Great, thanks. How did you handle failures in the payment flow?",
    "Okay, next question: how do you prioritize technical debt?",
    "Could you clarify what you mean by eventual consistency?",
    "Sorry, can you repeat that?",
    // Short clarifications
    "Why?",
    "How so?",
    "Like what?",
];

const NOT_QUESTIONS: &[&str] = &[
    "The question I was thinking about was the schema design.",
    "That's a good question.",
    "Good question.",
    "Great question, let me think.",
    "He asked me how I would scale it.",
    "They wanted to know why we picked Kafka.",
    "My manager asked us whether we could ship it by Friday.",
    "What did we do? We added a cache in front of the database.",
    "How we did it was by sharding on customer id.",
    "What I did was rewrite the ingestion layer.",
    "Why we chose Rust is mostly about memory safety.",
    "You know?",
    "Right?",
    "Okay?",
    "I wonder if I should mention the outage.",
    "No question about it, that was the right call.",
    "I don't have any questions.",
    "We used Redis for rate limiting.",
    "I usually start with the tests.",
    "We needed durable, replayable events and our team already ran Kafka in production.",
    "Thanks, that makes sense.",
    "Let's move on.",
    "Next question.",
    "Yeah.",
    "Sure.",
    "Hello?",
];

#[test]
fn table_questions() {
    let mut failures = Vec::new();
    for text in QUESTIONS {
        let r = remote(text);
        if !r.is_question {
            failures.push(format!("{text:?} → conf {:.2} {:?}", r.confidence, r.signals));
        }
    }
    assert!(failures.is_empty(), "missed questions:\n{}", failures.join("\n"));
}

#[test]
fn table_not_questions() {
    let mut failures = Vec::new();
    for text in NOT_QUESTIONS {
        let r = remote(text);
        if r.is_question {
            failures.push(format!("{text:?} → conf {:.2} {:?}", r.confidence, r.signals));
        }
    }
    assert!(failures.is_empty(), "false positives:\n{}", failures.join("\n"));
}

#[test]
fn table_has_at_least_sixty_cases() {
    assert!(QUESTIONS.len() + NOT_QUESTIONS.len() >= 60);
}

// ---------------------------------------------------------------------------
// Engine-facing guarantees
// ---------------------------------------------------------------------------

#[test]
fn strong_questions_have_high_confidence() {
    let r = remote("How would you design a rate limiter for our public API?");
    assert!(r.is_question && r.confidence >= 0.75, "{r:?}");
    assert_eq!(r.category, QuestionCategory::SystemDesign);
    assert!(r.requires_response);
    assert!(!r.ambiguous);

    let r = remote("Why did you choose Kafka for the event pipeline?");
    assert!(r.confidence >= 0.75);
    assert!(matches!(r.category, QuestionCategory::Tradeoff | QuestionCategory::Architecture), "{:?}", r.category);

    let r = remote("What database did you choose and why?");
    assert!(r.is_question);
}

#[test]
fn interviewer_own_questions_do_not_require_response() {
    let r = run(Mode::Interviewer, SpeakerRole::User, "How would you design a rate limiter?");
    assert!(r.is_question && r.confidence >= 0.6, "{r:?}");
    assert!(!r.requires_response);
}

#[test]
fn speaker_role_table() {
    let q = "How did you test the migration?";
    assert!(run(Mode::Candidate, SpeakerRole::Remote, q).requires_response);
    assert!(!run(Mode::Candidate, SpeakerRole::User, q).requires_response);
    let unknown = run(Mode::Candidate, SpeakerRole::Unknown, q);
    assert!(unknown.requires_response);
    assert!(unknown.confidence < remote(q).confidence, "unknown speakers are discounted");
    assert!(run(Mode::Interviewer, SpeakerRole::Remote, q).requires_response);
    assert!(run(Mode::Interviewer, SpeakerRole::Unknown, q).requires_response);
    assert!(!run(Mode::Interviewer, SpeakerRole::User, q).requires_response);
}

#[test]
fn deictic_references() {
    let r = remote("Can you explain this part of the architecture?");
    assert!(r.is_question);
    assert!(has_deictic_reference(&r.question));
    assert_eq!(r.category, QuestionCategory::Architecture);
    assert!(has_deictic_reference("What does the code above do?"));
    assert!(has_deictic_reference("explain that"));
    assert!(has_deictic_reference("Why is there an arrow from the queue in the diagram?"));
    assert!(!has_deictic_reference("Why did you choose that approach for caching in the billing service?"));
    assert!(!has_deictic_reference("How did you scale the database?"));
}

#[test]
fn multi_part_questions_are_split() {
    let r = remote("What database did you use, and why did you choose it?");
    assert_eq!(r.parts, vec!["What database did you use?", "Why did you choose it?"]);
    let r = remote("How long did the migration take and what went wrong?");
    assert_eq!(r.parts.len(), 2, "{:?}", r.parts);
    let r = remote("How did you scale reads? Did you add replicas?");
    assert_eq!(r.parts.len(), 2);
    let r = remote("Do you prefer Postgres or MySQL?");
    assert_eq!(r.parts.len(), 1, "alternatives are one question: {:?}", r.parts);
}

#[test]
fn lead_in_is_merged() {
    let lead = seg("a", SpeakerRole::Remote, 0, 1_500, "Next question.");
    let q = seg("b", SpeakerRole::Remote, 2_500, 5_000, "How do you handle schema migrations?");
    let r = detect(&DetectionInput {
        segment: &q,
        recent: std::slice::from_ref(&lead),
        mode: Mode::Candidate,
        last_question: None,
    });
    assert!(r.is_question);
    assert_eq!(r.segment_ids, vec!["a", "b"]);
    assert!(r.question.starts_with("Next question."));
    assert!(r.signals.iter().any(|s| s == "lead_in_merged"));

    // A lead-in from another speaker or long ago is not merged.
    let far = seg("a", SpeakerRole::Remote, 0, 1_000, "Let's talk about databases.");
    let q2 = seg("b", SpeakerRole::Remote, 9_000, 11_000, "How did you shard it?");
    let r = detect(&DetectionInput {
        segment: &q2,
        recent: std::slice::from_ref(&far),
        mode: Mode::Candidate,
        last_question: None,
    });
    assert_eq!(r.segment_ids, vec!["b"]);
}

#[test]
fn split_utterance_fragment_is_joined() {
    let frag = seg("a", SpeakerRole::Remote, 0, 1_000, "So tell me");
    let rest = seg("b", SpeakerRole::Remote, 1_600, 3_500, "about the hardest outage you handled");
    let r = detect(&DetectionInput {
        segment: &rest,
        recent: std::slice::from_ref(&frag),
        mode: Mode::Candidate,
        last_question: None,
    });
    assert!(r.is_question, "{r:?}");
    assert_eq!(r.segment_ids, vec!["a", "b"]);
    assert_eq!(r.question, "So tell me about the hardest outage you handled");
}

#[test]
fn continuation_supersedes_previous_question() {
    let first = seg("a", SpeakerRole::Remote, 0, 2_000, "How did you scale reads?");
    let last = DetectedQuestion {
        id: "q1".into(),
        meeting_id: "m".into(),
        segment_ids: vec!["a".into()],
        is_question: true,
        confidence: 0.9,
        question: "How did you scale reads?".into(),
        parts: vec![],
        category: QuestionCategory::Technical,
        speaker_id: None,
        speaker_role: SpeakerRole::Remote,
        requires_response: true,
        method: DetectionMethod::Heuristic,
        supersedes: None,
        detected_at: 0,
    };
    let second = seg("b", SpeakerRole::Remote, 3_000, 4_500, "And what about writes?");
    let r = detect(&DetectionInput {
        segment: &second,
        recent: std::slice::from_ref(&first),
        mode: Mode::Candidate,
        last_question: Some(&last),
    });
    assert!(r.is_question);
    assert_eq!(r.supersedes.as_deref(), Some("q1"));
    assert_eq!(r.question, "How did you scale reads? And what about writes?");
    assert_eq!(r.segment_ids, vec!["a", "b"]);

    // Too late or another speaker: a new question.
    let late = seg("c", SpeakerRole::Remote, 10_000, 11_000, "And what about writes?");
    let r = detect(&DetectionInput {
        segment: &late,
        recent: std::slice::from_ref(&first),
        mode: Mode::Candidate,
        last_question: Some(&last),
    });
    assert!(r.supersedes.is_none());
}

#[test]
fn rhetorical_self_answer_is_not_a_question() {
    let r = remote("What did we do? We added a cache in front of the database.");
    assert!(!r.is_question);
    assert!(r.signals.iter().any(|s| s == "negative:rhetorical"));
    // A question followed by context for the listener is still a question.
    let r = remote("We use Kafka for events. How would you handle duplicates?");
    assert!(r.is_question);
}

#[test]
fn ambiguity_flag() {
    // Confirmation checks are low confidence and ambiguous.
    let r = remote("So you used Kafka for that, right?");
    assert!(!r.is_question || r.confidence < 0.6);
    assert!(r.ambiguous, "{r:?}");
    // Clear cases are never ambiguous.
    assert!(!remote("How did you scale the database?").ambiguous);
    assert!(!remote("We used Redis.").ambiguous);
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

#[test]
fn categories_cover_all_thirteen() {
    use QuestionCategory::*;
    let cases: &[(&str, QuestionCategory)] = &[
        ("How does a hash index differ from a B-tree index in Postgres?", Technical),
        ("Write a function that reverses a linked list in place. What's the time complexity?", Coding),
        ("How would you design a URL shortener?", SystemDesign),
        ("Tell me about a time you had a conflict with a teammate.", Behavioral),
        ("What's your experience with Kubernetes in your previous role?", Experience),
        ("How would you prioritize features for the next release with limited engineering time?", Product),
        ("How do you mentor junior engineers and grow your team?", Leadership),
        ("Could you clarify what you mean by eventual consistency?", Clarification),
        ("Can you explain how the microservices communicate in this architecture?", Architecture),
        ("What are the pros and cons of using a queue here instead of direct calls?", Tradeoff),
        ("Why do you want to work here, and what kind of team culture do you enjoy?", Culture),
        ("Is the weather nice where you are?", General),
        ("And why that?", FollowUp),
    ];
    let mut wrong = Vec::new();
    for (text, expected) in cases {
        let got = classify_category(text);
        if got != *expected {
            wrong.push(format!("{text:?}: expected {expected:?}, got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn technologies_imply_technical() {
    assert_eq!(classify_category("How do you use Redis?"), QuestionCategory::Technical);
}

#[test]
fn detection_is_fast() {
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = remote("Okay, so walk me through how you would design a rate limiter for our public API, and what happens when Redis is down?");
    }
    let per_call = start.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    // Generous bound for debug builds; release is ~10x faster.
    assert!(per_call < 2.0, "{per_call} ms per call");
}
