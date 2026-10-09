//! Semantic confirmation of ambiguous heuristic question detections.

use openmeet_core::types::{Mode, QuestionCategory};

use super::shared::{
    DATA_RULE, TRANSCRIPT_NOTE, assemble, block, category_list, mode_line, one_line, question_line,
    sanitize, sanitize_line,
};
use super::{PromptParts, TranscriptLine};
use crate::json::{first_json_object, get_bool, get_f32, get_str};
use crate::text::{tail_to_tokens, truncate_to_tokens};
use crate::timefmt::clock;

/// Stable prompt identifier.
pub const ID: &str = "question-detection";
/// Prompt template version.
pub const VERSION: u32 = 1;

/// Recent lines included for context.
const MAX_RECENT_LINES: usize = 10;
/// Token cap for the recent transcript.
const MAX_RECENT_TOKENS: usize = 900;
/// Token cap for the utterance itself.
const MAX_UTTERANCE_TOKENS: usize = 600;

/// The model's verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticClassification {
    /// Whether the utterance is a question or request.
    pub is_question: bool,
    /// 0..1.
    pub confidence: f32,
    /// Cleaned, complete question text (may be empty).
    pub question: String,
    /// Question category.
    pub category: QuestionCategory,
    /// Whether the user is expected to respond.
    pub requires_response: bool,
}

fn response_rule(mode: Mode) -> &'static str {
    match mode {
        Mode::Candidate => {
            "Response expected: the user is being interviewed or is answering questions. \
             Questions and requests from the \"Interviewer\" (or an unidentified \"Speaker\") \
             require a response; the user's own questions do not."
        }
        Mode::Interviewer => {
            "Response expected: the user is the interviewer. Questions the \"Candidate\" (or an \
             unidentified \"Speaker\") asks the user — for example about the role, the team or \
             the process — require a response; the user's own questions do not."
        }
    }
}

/// Builds the classification prompt for the utterance `text`.
pub fn build(text: &str, recent: &[TranscriptLine], mode: Mode) -> PromptParts {
    let system = assemble(&[
        "You are the question detector of OpenMeet, a real-time meeting assistant. You classify \
         one utterance from a live, speech-recognized transcript."
            .to_string(),
        DATA_RULE.to_string(),
        TRANSCRIPT_NOTE.to_string(),
        format!(
            "Decide whether the utterance in <latest_utterance> is a question or request that the \
user (\"You\") is expected to respond to.
Counts as a question: direct questions, even without a question mark (\"how would you scale this\"); \
requests phrased as instructions (\"tell me about…\", \"walk me through…\", \"explain…\", \
\"describe…\", \"give me an example…\"); embedded questions (\"I'm curious how you…\", \"I'd love to \
hear about…\"); short clarification requests (\"why?\", \"what do you mean?\").
Does not count: statements and opinions; rhetorical questions the speaker answers right away; \
reported speech (\"he asked me how…\"); remarks about questions (\"good question\", \"the question I \
had was…\"); fillers (\"you know?\", \"right?\", \"okay?\"); thinking aloud (\"I wonder if I \
should…\").
Use <transcript> to merge a lead-in (\"Next question.\") or a continuation into the question and to \
resolve what it refers to.
{}
category is one of: {}.",
            response_rule(mode),
            category_list()
        ),
        "Output exactly one JSON object and nothing else (no prose, no code fences):
{\"is_question\": true|false, \"confidence\": 0.0-1.0, \"question\": \"the cleaned, complete \
question with obvious recognition errors fixed and lead-ins merged, or an empty string\", \
\"category\": \"<category>\", \"requires_response\": true|false}"
            .to_string(),
    ]);

    let lines: Vec<String> = recent
        .iter()
        .filter(|l| !l.text.trim().is_empty())
        .rev()
        .take(MAX_RECENT_LINES)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|l| {
            format!(
                "[{}] {}: {}",
                clock(l.at_ms),
                sanitize_line(&l.speaker),
                truncate_to_tokens(&sanitize_line(&l.text), 200)
            )
        })
        .collect();
    let transcript = if lines.is_empty() {
        String::new()
    } else {
        format!(
            "Earlier lines (oldest first):\n{}",
            block(
                "transcript",
                &tail_to_tokens(&lines.join("\n"), MAX_RECENT_TOKENS)
            )
        )
    };
    let utterance = tail_to_tokens(&one_line(&sanitize(text)), MAX_UTTERANCE_TOKENS);
    PromptParts {
        id: ID,
        version: VERSION,
        system,
        user: assemble(&[
            mode_line(mode).to_string(),
            transcript,
            format!(
                "The utterance to classify:\n{}",
                block("latest_utterance", &utterance)
            ),
            question_line(
                "Is the latest utterance a question or request the user should respond to?",
            ),
        ]),
    }
}

/// Tolerant JSON extraction of the classification; `None` when the output
/// contains no usable verdict.
pub fn parse(text: &str) -> Option<SemanticClassification> {
    let obj = first_json_object(text)?;
    let question = get_str(&obj, &["question", "cleaned_question", "text"])
        .map(|q| one_line(&q))
        .unwrap_or_default();
    let is_question = match get_bool(&obj, &["is_question", "isQuestion", "question?"]) {
        Some(b) => b,
        None if !question.is_empty() => true,
        None => return None,
    };
    let confidence = get_f32(&obj, &["confidence", "probability", "score"])
        .map(|c| if c > 1.0 && c <= 100.0 { c / 100.0 } else { c })
        .map(|c| c.clamp(0.0, 1.0))
        .unwrap_or(if is_question { 0.6 } else { 0.2 });
    let category = get_str(&obj, &["category", "type"])
        .and_then(|c| QuestionCategory::parse_lenient(&c))
        .unwrap_or_default();
    let requires_response = get_bool(&obj, &["requires_response", "requiresResponse", "respond"])
        .unwrap_or(is_question);
    Some(SemanticClassification {
        is_question,
        confidence,
        question,
        category,
        requires_response: requires_response && is_question,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::shared::{extract_block, extract_question};

    #[test]
    fn build_includes_utterance_and_recent_lines() {
        let recent = vec![TranscriptLine {
            speaker: "Interviewer".into(),
            text: "Next question.".into(),
            at_ms: 61_000,
        }];
        let p = build("how would you shard the users table", &recent, Mode::Candidate);
        assert_eq!(p.id, ID);
        assert!(p.system.contains("is_question"));
        assert!(p.system.contains("reported speech"));
        assert!(p.system.contains("\"Interviewer\""));
        assert_eq!(
            extract_block(&p.user, "latest_utterance").as_deref(),
            Some("how would you shard the users table")
        );
        assert!(p.user.contains("[01:01] Interviewer: Next question."));
        assert!(extract_question(&p.user).is_some());
        let i = build("what is the team like", &[], Mode::Interviewer);
        assert!(i.system.contains("\"Candidate\""));
        assert!(!i.user.contains("<transcript>"));
    }

    #[test]
    fn parses_fenced_and_lenient_json() {
        let c = parse("```json\n{\"is_question\": true, \"confidence\": 0.92, \"question\": \"How would you shard the users table?\", \"category\": \"system design\", \"requires_response\": true}\n```").unwrap();
        assert!(c.is_question);
        assert!((c.confidence - 0.92).abs() < 1e-6);
        assert_eq!(c.category, QuestionCategory::SystemDesign);
        assert!(c.requires_response);
        assert_eq!(c.question, "How would you shard the users table?");

        let c = parse("Result: {\"isQuestion\": \"no\", \"confidence\": \"85%\",}").unwrap();
        assert!(!c.is_question);
        assert!(!c.requires_response);
        assert!((c.confidence - 0.85).abs() < 1e-6);
        assert_eq!(c.category, QuestionCategory::General);

        let c = parse("{\"question\": \"Why Kafka?\", \"confidence\": 87}").unwrap();
        assert!(c.is_question);
        assert!((c.confidence - 0.87).abs() < 1e-6);

        assert!(parse("yes").is_none());
        assert!(parse("{\"confidence\": 0.5}").is_none());
    }
}
