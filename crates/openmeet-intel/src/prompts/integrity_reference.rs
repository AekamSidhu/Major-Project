//! Reference answers for the answer-integrity echo-similarity signal: what
//! the configured LLM itself answers to the interviewer's question.
//!
//! This is the shared reference **recipe v1** (`local-ai/integrity/signals.py`,
//! `REFERENCE_*`): the echo-similarity thresholds in
//! `spec/integrity/thresholds.json` were measured on HC3 with exactly this
//! system prompt, the bare question as the user message, temperature 0.7,
//! at most 400 tokens and reference `i` sampled with seed `i`. Changing any of
//! it invalidates those thresholds, so this prompt deliberately does not use
//! the shared data-block framing of the other prompts. The output is only
//! embedded and compared, never shown to anyone or acted on.

use super::PromptParts;

/// Stable prompt identifier.
pub const ID: &str = "integrity-reference";
/// Prompt template version (2 = the shared reference recipe v1).
pub const VERSION: u32 = 2;

/// The recipe's system prompt.
pub const SYSTEM: &str = "You are a helpful assistant. Answer the user's question directly, in plain prose without headings or lists, in about 150 words.";
/// Sampling temperature of the recipe.
pub const TEMPERATURE: f32 = 0.7;
/// Output budget of the recipe.
pub const MAX_OUTPUT_TOKENS: u32 = 400;

/// Seed of reference answer `i` (0-based) where the provider supports seeds.
pub fn seed(i: usize) -> u64 {
    i as u64
}

/// The recipe's prompt: the system prompt and the trimmed question.
pub fn build(question: &str) -> PromptParts {
    PromptParts {
        id: ID,
        version: VERSION,
        system: SYSTEM.to_string(),
        user: question.trim().to_string(),
    }
}

/// The reference answer as returned (trimmed). It goes through
/// [`crate::integrity::to_transcript`] before embedding, like the answer.
pub fn parse(text: &str) -> String {
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_follows_the_recipe() {
        let p = build("  How would you design a rate limiter?\n");
        assert_eq!(p.id, ID);
        assert_eq!(p.version, VERSION);
        assert_eq!(p.system, SYSTEM);
        assert_eq!(p.user, "How would you design a rate limiter?");
        assert_eq!(seed(0), 0);
        assert_eq!(seed(1), 1);
        assert_eq!(parse("  answer \n"), "answer");
    }
}
