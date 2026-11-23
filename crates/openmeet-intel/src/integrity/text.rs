//! An answer as a speech transcript of it would look: lower case, words only.
//!
//! Port of `to_transcript` in `local-ai/integrity/text.py`: the echo-
//! similarity thresholds were measured with both the candidate's answer and
//! the reference answers passed through it before embedding, so the app must
//! do exactly the same (`tests/integrity_spec.rs` checks parity with outputs
//! of the Python function).

use std::sync::LazyLock;

use regex::Regex;

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid to_transcript regex")
}

/// Pre-tokenised contractions ("do n't", "it 's") joined back, in order.
static TOKENISED: LazyLock<[(Regex, &'static str); 3]> = LazyLock::new(|| {
    [
        (re(r"\s+n't\b"), "n't"),
        (re(r"\s+'(s|re|ve|ll|d|m)\b"), "'${1}"),
        (
            re(
                r"\b(ca|wo|do|does|did|is|are|was|were|has|have|had|could|would|should|must)\s+n't\b",
            ),
            "${1}n't",
        ),
    ]
});
static URL: LazyLock<Regex> = LazyLock::new(|| re(r"(URL_\d+|https?://\S+|www\.\S+)"));
static MARKDOWN: LazyLock<Regex> = LazyLock::new(|| re(r"(^|\n)\s*(#+|[-*•]|\d+[.)])\s+"));
static CHARS: LazyLock<Regex> = LazyLock::new(|| re(r"[^a-z0-9' ]+"));
static SPACES: LazyLock<Regex> = LazyLock::new(|| re(r"\s+"));

/// Lower-case words separated by single spaces, as a transcript would give.
pub fn to_transcript(text: &str) -> String {
    let mut t = URL.replace_all(text, " ").into_owned();
    for (pattern, replacement) in TOKENISED.iter() {
        t = pattern.replace_all(&t, *replacement).into_owned();
    }
    t = MARKDOWN.replace_all(&t, " ").into_owned();
    t = t.replace("**", " ").replace("__", " ").replace('`', " ");
    t = t.to_lowercase().replace('\u{2019}', "'");
    t = CHARS.replace_all(&t, " ").into_owned();
    t = t.replace('\'', "");
    SPACES.replace_all(&t, " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_only() {
        assert_eq!(to_transcript("Hello, World!"), "hello world");
        assert_eq!(to_transcript(""), "");
    }
}
