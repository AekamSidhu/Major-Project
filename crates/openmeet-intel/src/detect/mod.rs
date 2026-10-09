//! Heuristic question detection (spec §17).
//!
//! Runs on every FINAL transcript segment in well under a millisecond. It
//! scores linguistic features (punctuation when present — ASR often omits
//! it —, WH-starts, auxiliary inversion, imperative interview prompts,
//! embedded questions, tag questions), filters known false positives
//! (reported speech, "that's a good question", rhetorical self-answers, cleft
//! statements like "how we did it was…"), merges lead-ins and continuations
//! using the recent conversation, and applies the speaker / mode table from
//! `docs/INTERNALS.md`. Genuinely unclear cases are flagged `ambiguous` so the
//! engine can ask an LLM for a semantic check.

pub mod voice;

use openmeet_core::ids::Id;
use openmeet_core::types::{DetectedQuestion, Mode, QuestionCategory, SpeakerRole, TranscriptSegment};

use crate::lexicon::find_technologies;

/// Lead-in segments are merged when they end at most this long before the question.
const LEAD_IN_GAP_MS: i64 = 4_000;
/// Same-speaker continuation of the previous question.
const CONTINUATION_GAP_MS: i64 = 3_000;
/// An unterminated fragment from the same speaker is joined within this gap.
const FRAGMENT_GAP_MS: i64 = 2_500;
/// Final decision threshold on the confidence score.
const QUESTION_THRESHOLD: f32 = 0.5;

/// Input for [`detect`].
pub struct DetectionInput<'a> {
    /// The new FINAL segment.
    pub segment: &'a TranscriptSegment,
    /// Earlier final segments, oldest → newest, excluding `segment`.
    pub recent: &'a [TranscriptSegment],
    /// Assistance mode.
    pub mode: Mode,
    /// The most recently detected question, if any.
    pub last_question: Option<&'a DetectedQuestion>,
}

/// Detector output.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// Final decision.
    pub is_question: bool,
    /// Confidence 0..1.
    pub confidence: f32,
    /// Cleaned question text (merged with lead-in / continuation).
    pub question: String,
    /// Sub-questions for multi-part questions (≥ 1 when `is_question`).
    pub parts: Vec<String>,
    /// Question category.
    pub category: QuestionCategory,
    /// Whether the OpenMeet user is expected to respond.
    pub requires_response: bool,
    /// Segments composing `question`.
    pub segment_ids: Vec<Id>,
    /// Id of `last_question` when this continues it.
    pub supersedes: Option<Id>,
    /// Heuristics are unsure; a semantic check is worthwhile.
    pub ambiguous: bool,
    /// Machine-readable reasons.
    pub signals: Vec<String>,
}

// ---------------------------------------------------------------------------
// Lexicons
// ---------------------------------------------------------------------------

const WH_WORDS: &[&str] = &["what", "why", "how", "when", "where", "which", "who", "whom", "whose"];
const AUX_VERBS: &[&str] = &[
    "can", "could", "would", "will", "do", "does", "did", "have", "has", "had", "is", "are", "was",
    "were", "should", "shall", "may", "might", "am", "must", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "haven't", "hasn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't",
];
const SUBJECTS: &[&str] = &[
    "you", "we", "they", "it", "there", "this", "that", "these", "those", "the", "your", "i", "he",
    "she", "our", "their", "a", "an", "any", "anyone", "someone", "everyone", "y'all", "u",
];
/// Discourse markers stripped before analysing the start of a sentence.
const LEADING_FILLERS: &[&str] = &[
    "so", "okay", "ok", "alright", "all right", "and", "but", "well", "now", "right", "um", "uh",
    "uhm", "erm", "hmm", "great", "cool", "nice", "perfect", "thanks", "thank you", "sure", "yeah",
    "yes", "got it", "interesting", "awesome", "excellent", "fair enough", "makes sense",
    "i see", "gotcha", "alrighty", "then", "also", "next", "lastly", "finally", "and then",
];
const IMPERATIVE_PROMPTS: &[&str] = &[
    "tell me", "tell us", "walk me through", "walk us through", "talk me through", "talk us through",
    "take me through", "run me through", "run us through", "explain", "describe", "give me an example",
    "give me a example", "give us an example", "give me some examples", "share an example",
    "share a time", "talk about", "help me understand", "help us understand", "elaborate",
    "expand on", "go through", "go over", "show me", "show us", "define", "compare", "contrast",
    "design a", "design an", "design the", "implement a", "implement an", "write a function",
    "write a program", "write code", "write some code", "code up", "imagine you", "imagine that",
    "suppose you", "suppose that", "say you have", "let's say you", "assume you", "assume that",
    "pretend you", "list the", "name a", "name some", "summarize", "clarify", "remind me",
    "convince me", "estimate", "sketch", "outline", "please explain", "please describe",
];
const EMBEDDED_PROMPTS: &[&str] = &[
    "i was wondering", "i'm wondering", "i am wondering", "i'm curious", "i am curious",
    "i'd love to hear", "i would love to hear", "i'd like to hear", "i would like to hear",
    "i'd like to know", "i would like to know", "i want to know", "i want to understand",
    "i'd like to understand", "i would like to understand", "i'd love to understand",
    "i'm interested in", "i am interested in", "i'd be interested", "i want to hear",
    "i wonder if you", "i wonder whether you", "i wonder how you", "i wonder why you",
    "i wonder what you", "curious to hear", "curious how", "curious why", "curious what",
    "keen to hear", "keen to understand",
];
const LEAD_IN_PHRASES: &[&str] = &[
    "next question", "another question", "one more question", "follow up question",
    "follow-up question", "i have a question", "i've got a question", "quick question",
    "question for you", "my question is", "my next question", "let me ask you", "let me ask",
    "i want to ask", "i'd like to ask", "moving on", "let's move on", "let's talk about",
    "let's switch to", "let's switch gears", "switching gears", "changing topics",
    "let's go back to", "going back to", "let's dive into", "let's look at", "on to",
];
const FALSE_POSITIVE_PHRASES: &[&str] = &[
    "the question i was thinking about", "the question i had in mind was",
    "the question is whether", "that's a good question", "that is a good question",
    "good question", "great question", "excellent question", "interesting question",
    "fair question", "no question about it", "without question", "out of the question",
    "question for later", "i have no questions", "i don't have any questions",
    "i don't have questions", "no questions from me", "no further questions",
    "that was my question", "that answers my question", "answered my question",
];
const REPORTED_SPEECH: &[&str] = &[
    "he asked", "she asked", "they asked", "he asked me", "she asked me", "they asked me",
    "asked me how", "asked me why", "asked me what", "asked me whether", "asked me if",
    "asked us how", "asked us why", "asked us what", "asked us whether", "asked us if",
    "i asked him", "i asked her", "i asked them", "we asked them", "we asked him", "we asked her",
    "wanted to know", "the customer asked", "the client asked", "my manager asked",
    "people ask", "people often ask", "customers ask", "users ask", "they were asking",
    "he was asking", "she was asking", "someone asked", "everyone asks", "was wondering whether i",
    "i wonder if i", "i wonder whether i", "i wonder if we", "i wonder whether we",
];
const SHORT_CLARIFICATIONS: &[&str] = &[
    "why", "how", "how so", "meaning", "like what", "such as", "for example", "for instance",
    "and then", "then what", "what else", "anything else", "why not", "which one", "what for",
    "how come", "really", "in what way", "sorry", "pardon", "come again", "what do you mean",
];
const FILLER_QUESTIONS: &[&str] = &[
    "you know", "right", "okay", "ok", "yeah", "see", "you see", "huh", "isn't it", "know what i mean",
    "you know what i mean", "eh", "no", "yes", "sure", "alright", "all right",
];
/// "<wh> <subject> <verb…> is/was" statements: "how we did it was…".
const CLEFT_SUBJECTS: &[&str] = &["we", "i", "they", "he", "she", "it", "the team", "our team", "my team", "you"];

// ---------------------------------------------------------------------------
// Text utilities
// ---------------------------------------------------------------------------

fn words_of(s: &str) -> Vec<String> {
    s.to_lowercase()
        .replace(['’', '‘'], "'")
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '-'))
        .filter(|w| !w.is_empty())
        .map(|w| w.trim_matches(|c| c == '\'' || c == '-').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Lowercase words joined by single spaces (for phrase matching).
fn phrase_norm(s: &str) -> String {
    format!(" {} ", words_of(s).join(" "))
}

fn has_phrase(norm: &str, phrase: &str) -> bool {
    norm.contains(&format!(" {phrase} "))
}

fn starts_with_phrase(words: &[String], phrase: &str) -> bool {
    let p: Vec<&str> = phrase.split(' ').collect();
    words.len() >= p.len() && words.iter().zip(p.iter()).all(|(w, q)| w == q)
}

/// Removes leading discourse markers ("so, okay, um, thanks — …").
fn strip_fillers(words: &[String]) -> &[String] {
    let mut rest = words;
    loop {
        let mut stripped = false;
        for f in LEADING_FILLERS {
            if starts_with_phrase(rest, f) {
                let n = f.split(' ').count();
                // Keep at least one word.
                if rest.len() > n {
                    rest = &rest[n..];
                    stripped = true;
                    break;
                }
            }
        }
        if !stripped {
            return rest;
        }
    }
}

/// Splits into sentences on . ! ? … keeping the punctuation. A period only
/// ends a sentence when followed by whitespace (so "3.5" and "node.js" stay
/// intact).
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let terminal = matches!(c, '?' | '!' | '…' | '.');
        if terminal && chars.get(i + 1).is_none_or(|n| n.is_whitespace() || *n == '"') {
            let t = cur.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            cur.clear();
        }
    }
    let t = cur.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Sentence scoring
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SentenceScore {
    text: String,
    score: f32,
    signals: Vec<String>,
    negative: bool,
}

/// Probability-style combination of independent positive cues.
fn combine(weights: &[f32]) -> f32 {
    1.0 - weights.iter().fold(1.0, |acc, w| acc * (1.0 - w.clamp(0.0, 0.99)))
}

fn score_sentence(sentence: &str) -> SentenceScore {
    let all_words = words_of(sentence);
    let words = strip_fillers(&all_words);
    let norm = phrase_norm(sentence);
    let trimmed = sentence.trim();
    let ends_q = trimmed.ends_with('?') || trimmed.ends_with("?\"");
    let mut weights: Vec<f32> = Vec::new();
    let mut signals: Vec<String> = Vec::new();
    let mut negative = false;

    let first = words.first().map(String::as_str).unwrap_or("");
    let second = words.get(1).map(String::as_str).unwrap_or("");

    if ends_q {
        weights.push(0.6);
        signals.push("question_mark".into());
    } else if trimmed.contains('?') {
        weights.push(0.3);
        signals.push("inner_question_mark".into());
    }

    let wh_start = WH_WORDS.contains(&first)
        || (first == "and" && WH_WORDS.contains(&second))
        || matches!(first, "what's" | "how's" | "where's" | "who's" | "why's" | "when's" | "what're");
    if wh_start {
        weights.push(0.45);
        signals.push("wh_start".into());
        if AUX_VERBS.contains(&second)
            || matches!(second, "about" | "if" | "much" | "many" | "long" | "often" | "kind" | "sort" | "type" | "would" | "come")
            || first.ends_with("'s")
            || first.ends_with("'re")
        {
            weights.push(0.35);
            signals.push("wh_aux".into());
        } else {
            // "What happens…", "Which component became…": a WH-word followed
            // by something other than a subject pronoun (cleft statements such
            // as "how we did it was…" have the subject right after).
            let after_wh = if first == "and" { words.get(2) } else { words.get(1) }
                .map(String::as_str)
                .unwrap_or("");
            let subject_next = CLEFT_SUBJECTS.iter().any(|s| s.split(' ').next() == Some(after_wh));
            if !after_wh.is_empty() && !subject_next {
                weights.push(0.25);
                signals.push("wh_direct".into());
            }
        }
    }
    let aux_start = AUX_VERBS.contains(&first)
        && (SUBJECTS.contains(&second) || second.ends_with("'s") || first.contains('\''));
    if aux_start {
        weights.push(0.55);
        signals.push("aux_inversion".into());
    }
    // An imperative needs an object: "tell me" alone is an unfinished fragment.
    let imperative = IMPERATIVE_PROMPTS.iter().any(|p| {
        let n = p.split(' ').count();
        (starts_with_phrase(words, p) && words.len() > n)
            || (starts_with_phrase(words, "can you") && has_phrase(&norm, p) && words.len() > n + 2)
    });
    if imperative {
        weights.push(0.6);
        signals.push("imperative_request".into());
        if has_phrase(&norm, "tell me about a time")
            || has_phrase(&norm, "walk me through")
            || has_phrase(&norm, "give me an example")
            || has_phrase(&norm, "how would you design")
        {
            weights.push(0.3);
        }
    }
    let embedded = EMBEDDED_PROMPTS.iter().any(|p| has_phrase(&norm, p));
    if embedded {
        weights.push(0.55);
        signals.push("embedded_question".into());
    }
    let tag = ends_q
        && (norm.ends_with(" right ")
            || norm.ends_with(" correct ")
            || norm.ends_with(" isn't it ")
            || norm.ends_with(" don't you ")
            || norm.ends_with(" didn't you ")
            || norm.ends_with(" wasn't it ")
            || norm.ends_with(" no ")
            || norm.ends_with(" yeah "));
    if tag && !wh_start && !aux_start {
        signals.push("tag_question".into());
    }
    if !weights.is_empty() && (has_phrase(&norm, "you") || has_phrase(&norm, "your") || has_phrase(&norm, "you've") || has_phrase(&norm, "you'd") || has_phrase(&norm, "you're")) {
        weights.push(0.15);
        signals.push("addressed_to_listener".into());
    }
    let mut score = combine(&weights);

    // --- false positives ---
    if FALSE_POSITIVE_PHRASES.iter().any(|p| has_phrase(&norm, p)) {
        let explicit_question_after = ends_q && (aux_start || wh_start) && !has_phrase(&norm, "good question") && !has_phrase(&norm, "great question");
        if !explicit_question_after {
            score *= 0.1;
            negative = true;
            signals.push("negative:meta_question".into());
        }
    }
    if REPORTED_SPEECH.iter().any(|p| has_phrase(&norm, p)) && !(aux_start && ends_q) {
        score *= 0.15;
        negative = true;
        signals.push("negative:reported".into());
    }
    if !ends_q && !embedded && is_cleft_statement(words) {
        score *= 0.1;
        negative = true;
        signals.push("negative:cleft_statement".into());
    }
    let joined = words.join(" ");
    if FILLER_QUESTIONS.contains(&joined.as_str()) {
        score *= 0.05;
        negative = true;
        signals.push("negative:filler".into());
    }
    if tag && !wh_start && !aux_start && !imperative && !embedded {
        // Confirmation checks ("we used Kafka, right?") expect a short answer.
        score = score.min(0.45);
    }
    // Very short utterances: only clarification requests count.
    let wc = all_words.len();
    if wc < 3 {
        if SHORT_CLARIFICATIONS.contains(&joined.as_str()) && (ends_q || matches!(first, "why" | "how")) {
            score = score.max(0.55);
            signals.push("short_clarification".into());
        } else if !negative {
            score = score.min(if ends_q { 0.45 } else { 0.2 });
            signals.push("too_short".into());
        }
    }
    SentenceScore {
        text: trimmed.to_string(),
        score: score.clamp(0.0, 1.0),
        signals,
        negative,
    }
}

/// "How we did it was…", "What I did was…", "Why we chose X is…" (no '?').
fn is_cleft_statement(words: &[String]) -> bool {
    let Some(first) = words.first() else { return false };
    if !WH_WORDS.contains(&first.as_str()) {
        return false;
    }
    let rest = words[1..].join(" ");
    if !CLEFT_SUBJECTS.iter().any(|s| rest.starts_with(&format!("{s} "))) {
        return false;
    }
    // A copula later in the clause marks the statement.
    words
        .iter()
        .skip(2)
        .take(10)
        .any(|w| matches!(w.as_str(), "is" | "was" | "were" | "are" | "means" | "meant"))
}

fn is_lead_in(text: &str) -> bool {
    let words = words_of(text);
    let stripped = strip_fillers(&words);
    let norm = phrase_norm(text);
    let t = text.trim();
    !t.ends_with('?')
        && stripped.len() <= 10
        && LEAD_IN_PHRASES.iter().any(|p| has_phrase(&norm, p))
}

/// True when `text` contains references that point at shared visual context:
/// "this", "here", "this part", "the diagram", "the code above", "on the
/// screen", "what you're showing"…
pub fn has_deictic_reference(text: &str) -> bool {
    let norm = phrase_norm(text);
    const STRONG: &[&str] = &[
        "this", "these", "here", "the diagram", "this diagram", "that diagram", "the code above",
        "the code below", "this code", "that code", "on the screen", "on screen", "on your screen",
        "on my screen", "the screen", "what you're showing", "what you are showing", "you're showing",
        "being shown", "shown here", "the slide", "this slide", "that slide", "the chart",
        "this chart", "the graph", "this graph", "the image", "the picture", "the whiteboard",
        "the drawing", "the figure", "that part", "this part", "that box", "this box", "that arrow",
        "this arrow", "highlighted", "the error above", "this error", "that error", "this function",
        "that function", "this line", "that line", "this table", "that table", "this one", "that one",
    ];
    if STRONG.iter().any(|p| has_phrase(&norm, p)) {
        return true;
    }
    // "explain that" / "what is that": short requests referring to the latest
    // visible or spoken thing.
    let words = words_of(text);
    words.len() <= 5 && words.iter().any(|w| w == "that" || w == "those")
}

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

struct CategoryRule {
    category: QuestionCategory,
    phrases: &'static [(&'static str, u32)],
}

static CATEGORY_RULES: &[CategoryRule] = &[
    CategoryRule {
        category: QuestionCategory::Clarification,
        phrases: &[
            ("what do you mean", 4), ("could you clarify", 4), ("can you clarify", 4),
            ("can you repeat", 4), ("could you repeat", 4), ("do you mean", 4), ("say that again", 4),
            ("did you say", 3), ("what does that mean", 3), ("just to clarify", 4), ("to clarify", 2),
            ("just to confirm", 3), ("so you're saying", 3), ("are you saying", 3), ("come again", 3),
            ("sorry", 1), ("pardon", 2), ("rephrase", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::SystemDesign,
        phrases: &[
            ("how would you design", 5), ("design a", 3), ("design an", 3), ("design the", 2),
            ("system design", 4), ("url shortener", 4), ("rate limiter", 4), ("news feed", 4),
            ("chat system", 4), ("notification system", 4), ("distributed", 2), ("at scale", 2),
            ("scale to", 2), ("millions of users", 3), ("high availability", 3), ("load balanc", 2),
            ("sharding", 2), ("partition", 1), ("throughput", 1), ("scalable", 2), ("architect a", 3),
            ("would you build", 2), ("how would you build", 3), ("key value store", 3),
            ("design twitter", 4), ("design youtube", 4), ("parking lot", 3), ("ride sharing", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Coding,
        phrases: &[
            ("algorithm", 3), ("time complexity", 4), ("space complexity", 4), ("big o", 4),
            ("complexity", 2), ("implement a function", 4), ("write a function", 4),
            ("write code", 3), ("write a program", 4), ("data structure", 3), ("linked list", 4),
            ("binary tree", 4), ("binary search", 4), ("hash map", 3), ("hashmap", 3), ("array", 2),
            ("recursion", 3), ("recursive", 2), ("dynamic programming", 4), ("sort", 1),
            ("pseudocode", 3), ("function that", 2), ("leetcode", 4), ("palindrome", 4),
            ("reverse a", 3), ("o n", 2), ("edge cases", 2), ("unit test for", 2), ("code this", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Behavioral,
        phrases: &[
            ("tell me about a time", 5), ("describe a time", 5), ("a time when", 4),
            ("a time you", 4), ("a situation where", 4), ("situation when", 3), ("conflict", 3),
            ("disagree", 3), ("disagreement", 3), ("failure", 2), ("failed", 2), ("mistake", 3),
            ("challenge you faced", 4), ("biggest challenge", 3), ("difficult", 2), ("proud of", 3),
            ("feedback", 2), ("under pressure", 3), ("tight deadline", 3), ("deadline", 1),
            ("how do you handle", 2), ("weakness", 3), ("strengths", 2), ("teammate", 2),
            ("coworker", 2), ("difficult person", 4), ("went wrong", 2), ("learned from", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Experience,
        phrases: &[
            ("your experience", 4), ("experience with", 3), ("have you worked with", 4),
            ("have you used", 3), ("have you ever", 2), ("in your previous role", 4),
            ("your last project", 4), ("your current role", 4), ("your current job", 4),
            ("worked on", 2), ("tell me about yourself", 5), ("introduce yourself", 5),
            ("your background", 4), ("your resume", 4), ("your role", 3), ("your responsibilities", 3),
            ("your day to day", 3), ("in your career", 3), ("previous company", 3), ("your team at", 2),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Tradeoff,
        phrases: &[
            ("trade-off", 4), ("tradeoff", 4), ("trade off", 4), ("trade-offs", 4), ("tradeoffs", 4),
            ("pros and cons", 4), ("why did you choose", 4), ("why did you pick", 4),
            ("why did you go with", 4), ("why did you decide", 3), ("why not", 3), ("versus", 3),
            ("vs", 2), ("instead of", 3), ("compared to", 3), ("compare", 2), ("advantages", 2),
            ("disadvantages", 3), ("downside", 3), ("downsides", 3), ("benefits of", 2),
            ("when would you use", 3), ("rather than", 3), ("over the alternatives", 3),
            ("alternatives", 2), ("why would you use", 3), ("choose between", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Leadership,
        phrases: &[
            ("leadership", 4), ("lead a team", 4), ("led a team", 4), ("as a lead", 3), ("mentor", 3),
            ("mentoring", 3), ("manage a team", 4), ("managed a team", 4), ("managing people", 4),
            ("influence", 2), ("stakeholder", 2), ("stakeholders", 2), ("hiring", 2), ("delegate", 3),
            ("vision", 2), ("motivate", 3), ("grow your team", 3), ("underperforming", 4),
            ("direct reports", 4), ("technical direction", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Product,
        phrases: &[
            ("product", 2), ("customer", 2), ("customers", 2), ("user experience", 3), ("users want", 3),
            ("kpi", 3), ("metric would you", 3), ("success metric", 3), ("prioritize", 3),
            ("prioritise", 3), ("feature", 1), ("market", 2), ("roadmap", 2), ("requirements", 1),
            ("business impact", 3), ("business value", 3), ("go to market", 3), ("north star", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Culture,
        phrases: &[
            ("culture", 3), ("values", 2), ("why do you want to work", 5), ("why this company", 5),
            ("why us", 4), ("work environment", 3), ("team culture", 4), ("work-life", 3),
            ("work life balance", 3), ("what motivates you", 4), ("where do you see yourself", 5),
            ("ideal team", 4), ("ideal manager", 4), ("why are you leaving", 4),
            ("why are you looking", 4), ("what are you looking for", 3), ("questions for me", 4),
            ("questions for us", 4),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Architecture,
        phrases: &[
            ("architecture", 3), ("microservice", 3), ("microservices", 3), ("monolith", 3),
            ("service boundaries", 4), ("event driven", 3), ("event-driven", 3), ("message queue", 2),
            ("api gateway", 2), ("components", 2), ("diagram", 2), ("this part of the", 3),
            ("data flow", 3), ("pipeline", 1), ("infrastructure", 2), ("layer", 1), ("layers", 1),
            ("how does the", 1), ("how do these", 2), ("this component", 3),
        ],
    },
    CategoryRule {
        category: QuestionCategory::Technical,
        phrases: &[
            ("database", 2), ("sql", 2), ("index", 2), ("query", 2), ("cache", 2), ("caching", 2),
            ("api", 1), ("http", 2), ("latency", 2), ("memory", 2), ("thread", 2), ("concurrency", 3),
            ("deploy", 2), ("deployment", 2), ("test", 1), ("testing", 2), ("bug", 2), ("debug", 2),
            ("performance", 2), ("security", 2), ("scale", 1), ("scaling", 1), ("server", 1),
            ("network", 2), ("protocol", 2), ("encryption", 2), ("authentication", 2), ("garbage", 1),
            ("consistency", 2), ("transaction", 2), ("replication", 2), ("monitoring", 2),
        ],
    },
];

/// Category of a question (keyword / phrase scoring over 13 categories).
pub fn classify_category(text: &str) -> QuestionCategory {
    let norm = phrase_norm(text);
    let words = words_of(text);
    let stripped = strip_fillers(&words);
    let mut best = (QuestionCategory::General, 0u32);
    for rule in CATEGORY_RULES {
        let score: u32 = rule
            .phrases
            .iter()
            .filter(|(p, _)| has_phrase(&norm, p) || (p.ends_with(['c', 'e']) && norm.contains(&format!(" {p}"))))
            .map(|(_, w)| *w)
            .sum();
        if score > best.1 {
            best = (rule.category, score);
        }
    }
    // Technologies imply a technical question when nothing more specific matched.
    if best.1 < 2 && !find_technologies(text).is_empty() {
        best = (QuestionCategory::Technical, 2);
    }
    // Short questions that refer back to what was just said are follow-ups.
    let first = stripped.first().map(String::as_str).unwrap_or("");
    let refers_back = stripped.iter().any(|w| matches!(w.as_str(), "that" | "it" | "this" | "those" | "them"));
    let follow_start = matches!(first, "and" | "why" | "how" | "so")
        || starts_with_phrase(stripped, "what about")
        || starts_with_phrase(stripped, "how about")
        || starts_with_phrase(stripped, "and then")
        || starts_with_phrase(stripped, "then what");
    if stripped.len() <= 8 && follow_start && (refers_back || stripped.len() <= 3) && best.1 < 4 {
        return QuestionCategory::FollowUp;
    }
    best.0
}

// ---------------------------------------------------------------------------
// Multi-part questions
// ---------------------------------------------------------------------------

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn ensure_question_mark(s: &str) -> String {
    let t = s.trim().trim_end_matches(['.', ',', ';', ':']).trim();
    if t.ends_with('?') { t.to_string() } else { format!("{t}?") }
}

/// Sub-questions of a (possibly multi-part) question: "What database did you
/// use and why did you choose it?" → ["What database did you use?", "Why did
/// you choose it?"]. Alternatives joined by "or" stay one question.
pub fn split_parts(text: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for sentence in sentences(text) {
        if score_sentence(&sentence).score < QUESTION_THRESHOLD {
            continue;
        }
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        let bare = |t: &str| {
            t.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_lowercase()
        };
        // Cut before "and"/"also" when the following words start a new question.
        let mut cuts: Vec<usize> = Vec::new();
        for (i, tok) in tokens.iter().enumerate() {
            let w = bare(tok);
            if i == 0 || !(w == "and" || w == "also") {
                continue;
            }
            let next = tokens.get(i + 1).map(|t| bare(t)).unwrap_or_default();
            let after = tokens.get(i + 2).map(|t| bare(t)).unwrap_or_default();
            let starts_question = WH_WORDS.contains(&next.as_str())
                || (AUX_VERBS.contains(&next.as_str()) && SUBJECTS.contains(&after.as_str()));
            if starts_question && i >= 2 {
                cuts.push(i);
            }
        }
        if cuts.is_empty() {
            parts.push(ensure_question_mark(&sentence));
            continue;
        }
        let mut start = 0;
        for &cut in &cuts {
            let piece = tokens[start..cut].join(" ");
            let piece = piece.trim().trim_end_matches(',').trim();
            if !piece.is_empty() {
                parts.push(ensure_question_mark(&capitalize(piece)));
            }
            start = cut + 1; // skip the joiner word
        }
        let tail = tokens[start..].join(" ");
        if !tail.trim().is_empty() {
            parts.push(ensure_question_mark(&capitalize(tail.trim())));
        }
    }
    parts
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

fn requires_response(mode: Mode, role: SpeakerRole) -> (bool, f32) {
    match (mode, role) {
        (Mode::Candidate, SpeakerRole::Remote) => (true, 1.0),
        (Mode::Candidate, SpeakerRole::Unknown) => (true, 0.85),
        (Mode::Candidate, SpeakerRole::User) => (false, 1.0),
        (Mode::Interviewer, SpeakerRole::User) => (false, 1.0),
        (Mode::Interviewer, SpeakerRole::Remote) => (true, 1.0),
        (Mode::Interviewer, SpeakerRole::Unknown) => (true, 0.85),
    }
}

/// Detects whether a final transcript segment is a question.
pub fn detect(input: &DetectionInput) -> DetectionResult {
    let seg = input.segment;
    let text = clean(&seg.text);
    let sents = sentences(&text);
    let scored: Vec<SentenceScore> = sents.iter().map(|s| score_sentence(s)).collect();

    // Rhetorical self-answers: a question followed by a statement from the
    // same speaker in the same utterance ("What did we do? We added caching.").
    let mut best_idx: Option<usize> = None;
    for (i, s) in scored.iter().enumerate() {
        if s.score >= QUESTION_THRESHOLD {
            best_idx = Some(i);
        }
    }
    let mut signals: Vec<String> = Vec::new();
    let mut rhetorical = false;
    if let Some(i) = best_idx {
        let after: Vec<&SentenceScore> = scored[i + 1..].iter().collect();
        if !after.is_empty() && after.iter().all(|s| s.score < 0.3) {
            let answer_like = after.iter().any(|s| {
                let w = words_of(&s.text);
                let first = w.first().map(String::as_str).unwrap_or("");
                w.len() >= 3
                    && matches!(first, "we" | "i" | "it" | "the" | "basically" | "well" | "so" | "they" | "our" | "that's" | "it's" | "we're" | "i'm")
            });
            // "What did we do? We added caching." — the question is about the
            // speaker's own work and is answered immediately.
            let self_referential = has_phrase(&phrase_norm(&scored[i].text), "we") || has_phrase(&phrase_norm(&scored[i].text), "i");
            if answer_like && self_referential {
                rhetorical = true;
                signals.push("negative:rhetorical".into());
            }
        }
    }

    // Question sentences (everything scoring above the threshold) plus the
    // non-question context before them (kept for display).
    let q_sents: Vec<&SentenceScore> = scored.iter().filter(|s| s.score >= QUESTION_THRESHOLD).collect();
    let mut confidence = scored.iter().map(|s| s.score).fold(0.0f32, f32::max);
    for s in &scored {
        for sig in &s.signals {
            if !signals.contains(sig) {
                signals.push(sig.clone());
            }
        }
    }
    if rhetorical {
        confidence *= 0.2;
    }
    let mut question = if q_sents.is_empty() {
        text.clone()
    } else {
        // Keep the leading context sentence when it introduces the topic
        // ("Let's talk about caching. How do you invalidate?").
        let first_q = scored.iter().position(|s| s.score >= QUESTION_THRESHOLD).unwrap_or(0);
        let lead: Vec<&str> = scored[..first_q]
            .iter()
            .filter(|s| is_lead_in(&s.text) || words_of(&s.text).len() >= 4)
            .map(|s| s.text.as_str())
            .rev()
            .take(1)
            .collect();
        let mut parts: Vec<&str> = lead;
        parts.extend(scored[first_q..].iter().filter(|s| !s.negative).map(|s| s.text.as_str()));
        parts.join(" ")
    };
    let mut segment_ids = vec![seg.id.clone()];

    // Context: lead-ins and fragments from the same speaker just before.
    let prev_same: Option<&TranscriptSegment> = input
        .recent
        .iter()
        .rev()
        .find(|s| s.speaker_role == seg.speaker_role);
    if let Some(prev) = prev_same {
        let gap = seg.start_time - prev.end_time;
        let prev_text = clean(&prev.text);
        let prev_score = sentences(&prev_text).iter().map(|s| score_sentence(s).score).fold(0.0f32, f32::max);
        if gap <= LEAD_IN_GAP_MS && is_lead_in(&prev_text) && confidence >= QUESTION_THRESHOLD {
            question = format!("{prev_text} {question}");
            segment_ids.insert(0, prev.id.clone());
            confidence = combine(&[confidence, 0.2]);
            signals.push("lead_in_merged".into());
        } else if gap <= FRAGMENT_GAP_MS
            && !prev_text.ends_with(['.', '?', '!'])
            && prev_score < QUESTION_THRESHOLD
            && words_of(&prev_text).len() <= 12
        {
            // ASR split one question into two utterances ("So tell me" / "how you scaled it").
            let merged = format!("{prev_text} {text}");
            let merged_score = sentences(&merged).iter().map(|s| score_sentence(s).score).fold(0.0f32, f32::max);
            if merged_score > confidence && merged_score >= QUESTION_THRESHOLD {
                confidence = merged_score;
                question = merged;
                segment_ids.insert(0, prev.id.clone());
                signals.push("fragment_merged".into());
            }
        }
    }

    // Continuation of the previous question by the same speaker.
    let mut supersedes = None;
    if let Some(last) = input.last_question {
        if last.speaker_role == seg.speaker_role && confidence >= QUESTION_THRESHOLD {
            let last_end = last
                .segment_ids
                .last()
                .and_then(|id| input.recent.iter().find(|s| &s.id == id))
                .map(|s| s.end_time);
            if let Some(end) = last_end {
                if seg.start_time - end <= CONTINUATION_GAP_MS {
                    supersedes = Some(last.id.clone());
                    question = format!("{} {}", last.question.trim(), question);
                    for id in last.segment_ids.iter().rev() {
                        if !segment_ids.contains(id) {
                            segment_ids.insert(0, id.clone());
                        }
                    }
                    signals.push("continuation".into());
                }
            }
        }
    }

    let (requires, factor) = requires_response(input.mode, seg.speaker_role);
    confidence = (confidence * factor).clamp(0.0, 1.0);
    let is_question = confidence >= QUESTION_THRESHOLD;
    let question = clean(&question);
    let parts = if is_question {
        let p = split_parts(&question);
        if p.is_empty() { vec![ensure_question_mark(&question)] } else { p }
    } else {
        Vec::new()
    };
    let category = classify_category(&question);
    let ambiguous = (0.35..0.65).contains(&confidence) && !rhetorical;
    DetectionResult {
        is_question,
        confidence,
        question,
        parts,
        category,
        requires_response: is_question && requires,
        segment_ids,
        supersedes,
        ambiguous,
        signals,
    }
}

#[cfg(test)]
mod tests;
