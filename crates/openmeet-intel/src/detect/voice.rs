//! Optional spoken commands addressed to OpenMeet ("OpenMeet, explain that").
//!
//! Only utterances that *start* with the wake phrase are commands, so normal
//! conversation is never misinterpreted. Common speech-recognition variants of
//! the name are accepted.

use openmeet_core::types::VoiceCommand;

const WAKE_PREFIXES: &[&str] = &["hey ", "ok ", "okay ", "hi "];
const WAKE_WORDS: &[&str] = &[
    "openmeet", "open meet", "open meat", "openmeat", "open mead", "open-meet", "open mete",
];

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .replace(['’', '‘'], "'")
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' || c == '-' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parses a voice command. Returns `None` when the utterance does not start
/// with the wake phrase (or contains nothing after it).
pub fn parse_voice_command(text: &str) -> Option<VoiceCommand> {
    let norm = normalize(text);
    let mut rest = norm.as_str();
    for p in WAKE_PREFIXES {
        if let Some(r) = rest.strip_prefix(p) {
            rest = r;
            break;
        }
    }
    let wake = WAKE_WORDS.iter().find(|w| rest == **w || rest.starts_with(&format!("{w} ")))?;
    let command = rest[wake.len()..].trim();
    if command.is_empty() {
        return None;
    }
    let command = command.strip_prefix("please ").unwrap_or(command).trim_end_matches(" please");
    let is = |opts: &[&str]| opts.contains(&command);
    let starts = |opts: &[&str]| opts.iter().any(|o| command.starts_with(o));

    if is(&["explain", "explain that", "explain this", "explain it", "what is that", "what's that", "what does that mean"]) {
        return Some(VoiceCommand::Explain);
    }
    if is(&[
        "follow up", "follow-up", "give me a follow up", "give me a follow-up", "give me follow ups",
        "give me follow-ups", "give me some follow ups", "give me some follow-ups", "suggest a follow up",
        "suggest a follow-up", "suggest follow ups", "suggest follow-ups", "follow up question",
        "follow-up question", "follow up questions", "follow-up questions", "next question",
        "what should i ask", "what should i ask next",
    ]) {
        return Some(VoiceCommand::FollowUp);
    }
    if is(&[
        "answer", "answer that", "answer this", "answer it", "suggest an answer", "help me answer",
        "help me answer that", "what should i say", "how should i answer", "how do i answer that",
    ]) {
        return Some(VoiceCommand::Answer);
    }
    if starts(&["make that concise", "make it concise", "make that shorter", "make it shorter", "shorter", "be concise", "concise", "summarize that", "summarise that", "shorten", "tldr", "tl dr"]) {
        return Some(VoiceCommand::Concise);
    }
    if starts(&["more detail", "more details", "longer", "make it longer", "make that longer", "elaborate", "expand", "go deeper", "more depth", "in more detail"]) {
        return Some(VoiceCommand::Detailed);
    }
    // Anything else is a free-form request, in the user's original words.
    let original = text.trim();
    let start = original
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| normalize(&original[i..]) == command)
        .unwrap_or(0);
    let ask = original[start..].trim().trim_start_matches([',', ':', '-']).trim();
    Some(VoiceCommand::Ask(if ask.is_empty() { command.to_string() } else { ask.to_string() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wake_phrase_required() {
        assert_eq!(parse_voice_command("Explain that"), None);
        assert_eq!(parse_voice_command("We opened the meeting"), None);
        assert_eq!(parse_voice_command("OpenMeet"), None);
        assert_eq!(parse_voice_command("Open meet."), None);
    }

    #[test]
    fn commands() {
        assert_eq!(parse_voice_command("OpenMeet, explain that."), Some(VoiceCommand::Explain));
        assert_eq!(parse_voice_command("Hey open meet, give me a follow-up"), Some(VoiceCommand::FollowUp));
        assert_eq!(parse_voice_command("Open meat answer that"), Some(VoiceCommand::Answer));
        assert_eq!(parse_voice_command("OpenMeet, make that concise please"), Some(VoiceCommand::Concise));
        assert_eq!(parse_voice_command("Okay OpenMeet: more detail"), Some(VoiceCommand::Detailed));
        assert_eq!(parse_voice_command("openmeat shorter"), Some(VoiceCommand::Concise));
    }

    #[test]
    fn free_form_requests_keep_original_wording() {
        assert_eq!(
            parse_voice_command("OpenMeet, what does this code do?"),
            Some(VoiceCommand::Ask("what does this code do?".into()))
        );
        assert_eq!(
            parse_voice_command("Hey OpenMeet - Explain this architecture simply."),
            Some(VoiceCommand::Ask("Explain this architecture simply.".into()))
        );
    }
}
