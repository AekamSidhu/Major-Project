//! Rolling compression of conversation that leaves the raw transcript
//! window into a compact long-term summary.

use std::sync::LazyLock;

use openmeet_core::types::Mode;
use regex::Regex;

use super::shared::{
    DATA_RULE, NON_EVALUATIVE_RULE, TRANSCRIPT_NOTE, assemble, block, mode_line, question_line,
    sanitize, sanitize_line,
};
use super::{PromptParts, TranscriptLine};
use crate::json::strip_code_fences;
use crate::text::{estimate_tokens, tail_to_tokens, truncate_to_tokens};
use crate::timefmt::clock;

/// Stable prompt identifier.
pub const ID: &str = "context-compression";
/// Prompt template version.
pub const VERSION: u32 = 1;

/// Cap for the previous summary.
const MAX_PREVIOUS_TOKENS: usize = 1500;
/// Cap for the new transcript lines (oldest are dropped beyond it).
const MAX_TRANSCRIPT_TOKENS: usize = 10_000;

/// Builds the compression prompt.
pub fn build(previous_summary: &str, transcript: &[TranscriptLine], mode: Mode) -> PromptParts {
    let system = assemble(&[
        "You maintain the long-term memory of OpenMeet, a real-time meeting assistant: a compact \
         running summary of a live meeting that later prompts use as context."
            .to_string(),
        DATA_RULE.to_string(),
        TRANSCRIPT_NOTE.to_string(),
        format!(
            "Task: merge the previous summary in <summary> with the new lines in <transcript> into \
one updated summary.
Keep: every question that was asked (close to its wording) and who asked it; the key facts of each \
answer; names, numbers, technologies and systems exactly as stated; decisions, commitments and \
action items with their owners; open threads; what was shown on screen.
Drop: greetings, small talk, filler and repetition.
Rules: never add facts, interpretations or numbers that are not in the input. Attribute statements \
to speakers by their labels (You, Interviewer, Candidate, Speaker). {NON_EVALUATIVE_RULE}
Output plain text only: short lines starting with \"- \", in chronological order, at most about \
300 words. No heading, no preamble, no code fences."
        ),
    ]);

    let previous = truncate_to_tokens(sanitize(previous_summary.trim()).trim(), MAX_PREVIOUS_TOKENS);
    let previous_block = if previous.is_empty() {
        "There is no previous summary yet.".to_string()
    } else {
        format!("Previous summary:\n{}", block("summary", &previous))
    };
    let lines: Vec<String> = transcript
        .iter()
        .filter(|l| !l.text.trim().is_empty())
        .map(|l| {
            format!(
                "[{}] {}: {}",
                clock(l.at_ms),
                sanitize_line(&l.speaker),
                sanitize_line(&l.text)
            )
        })
        .collect();
    let mut kept: Vec<&String> = Vec::new();
    let mut used = 0usize;
    for line in lines.iter().rev() {
        let cost = estimate_tokens(line) + 1;
        if used + cost > MAX_TRANSCRIPT_TOKENS {
            break;
        }
        used += cost;
        kept.push(line);
    }
    kept.reverse();
    let mut body: Vec<String> = Vec::new();
    if kept.len() < lines.len() {
        body.push("[… earlier lines omitted …]".to_string());
    }
    body.extend(kept.into_iter().cloned());
    let transcript_block = if body.is_empty() {
        "There are no new transcript lines.".to_string()
    } else {
        format!(
            "New transcript lines (oldest first):\n{}",
            block("transcript", &body.join("\n"))
        )
    };
    PromptParts {
        id: ID,
        version: VERSION,
        system,
        user: assemble(&[
            mode_line(mode).to_string(),
            previous_block,
            transcript_block,
            question_line("Update the meeting summary with the new transcript lines."),
        ]),
    }
}

static PREAMBLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:#+\s*)?(?:\*\*)?(?:sure[,!.]?\s*)?(?:here(?:'s| is)\s+(?:the\s+)?(?:an?\s+)?(?:updated|new|revised)?\s*(?:meeting\s+)?summary|updated\s+(?:meeting\s+)?summary|(?:meeting\s+)?summary)\s*[:.]?\s*(?:\*\*)?\s*:?\s*$",
    )
    .expect("valid preamble regex")
});

/// Cleans the model output: code fences, a leading "Here is the summary:"
/// line and surrounding whitespace are removed. Bounded in size.
pub fn parse(text: &str) -> String {
    let cleaned = strip_code_fences(text);
    let mut lines: Vec<&str> = cleaned.lines().collect();
    while lines
        .first()
        .is_some_and(|l| l.trim().is_empty() || PREAMBLE_RE.is_match(l))
    {
        lines.remove(0);
    }
    let joined = lines
        .iter()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    tail_to_tokens(joined.trim(), 2_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::shared::{extract_block, extract_question};

    fn line(speaker: &str, text: &str, at: i64) -> TranscriptLine {
        TranscriptLine {
            speaker: speaker.into(),
            text: text.into(),
            at_ms: at,
        }
    }

    #[test]
    fn build_merges_previous_and_new_lines() {
        let p = build(
            "- Interviewer asked about Kafka.",
            &[
                line("Interviewer", "Why Kafka?", 60_000),
                line("You", "We needed replayable events.", 65_000),
            ],
            Mode::Candidate,
        );
        assert_eq!(p.id, ID);
        assert!(p.system.contains("never add facts"));
        assert_eq!(
            extract_block(&p.user, "summary").as_deref(),
            Some("- Interviewer asked about Kafka.")
        );
        let t = extract_block(&p.user, "transcript").unwrap();
        assert!(t.contains("[01:00] Interviewer: Why Kafka?"));
        assert!(extract_question(&p.user).is_some());
    }

    #[test]
    fn huge_transcripts_are_bounded() {
        let lines: Vec<TranscriptLine> = (0..5000)
            .map(|i| line("Speaker", &format!("sentence number {i} with some words"), i))
            .collect();
        let p = build("", &lines, Mode::Interviewer);
        assert!(estimate_tokens(&p.user) < MAX_TRANSCRIPT_TOKENS + 400);
        assert!(p.user.contains("sentence number 4999"));
        assert!(p.user.contains("earlier lines omitted"));
        assert!(p.user.contains("no previous summary"));
    }

    #[test]
    fn parse_strips_preamble_and_fences() {
        assert_eq!(
            parse("Here is the updated summary:\n```\n- A asked B.\n- B answered.\n```\n"),
            "- A asked B.\n- B answered."
        );
        assert_eq!(parse("**Summary:**\r\n- One"), "- One");
        assert_eq!(parse("- Kept as is"), "- Kept as is");
        assert_eq!(parse("  "), "");
    }
}
