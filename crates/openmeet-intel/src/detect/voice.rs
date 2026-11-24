//! Optional spoken commands addressed to OpenMeet ("OpenMeet, explain that").
//!
//! Only utterances that *start* with the wake phrase are commands, so normal
//! conversation is never misinterpreted. Common speech-recognition variants of
//! the name are accepted.

use openmeet_core::types::{Mode, VoiceCommand};

/// Minutes recapped by "catch me up" when no window is said.
const DEFAULT_CATCH_UP_MINUTES: u32 = 5;
/// Longest window a spoken catch-up can ask for.
const MAX_CATCH_UP_MINUTES: u32 = 60;

const WAKE_PREFIXES: &[&str] = &["hey ", "ok ", "okay ", "hi "];
const WAKE_WORDS: &[&str] = &[
    "openmeet",
    "open meet",
    "open meat",
    "openmeat",
    "open mead",
    "open-meet",
    "open mete",
];

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .replace(['’', '‘'], "'")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parses a voice command. Returns `None` when the utterance does not start
/// with the wake phrase (or contains nothing after it). "Catch me up" is a
/// [`VoiceCommand::CatchUp`] here; use [`parse_voice_command_in`] for a mode
/// where it is not a workplace command.
pub fn parse_voice_command(text: &str) -> Option<VoiceCommand> {
    parse(text, true)
}

/// Parses a voice command in `mode`. Outside workplace mode "catch me up" is
/// not a command: it is a free-form request like any other.
pub fn parse_voice_command_in(text: &str, mode: Mode) -> Option<VoiceCommand> {
    parse(text, mode == Mode::Workplace)
}

fn parse(text: &str, catch_up_enabled: bool) -> Option<VoiceCommand> {
    let norm = normalize(text);
    let mut rest = norm.as_str();
    for p in WAKE_PREFIXES {
        if let Some(r) = rest.strip_prefix(p) {
            rest = r;
            break;
        }
    }
    let wake = WAKE_WORDS
        .iter()
        .find(|w| rest == **w || rest.starts_with(&format!("{w} ")))?;
    let command = rest[wake.len()..].trim();
    if command.is_empty() {
        return None;
    }
    let command = command
        .strip_prefix("please ")
        .unwrap_or(command)
        .trim_end_matches(" please");
    let is = |opts: &[&str]| opts.contains(&command);
    let starts = |opts: &[&str]| opts.iter().any(|o| command.starts_with(o));

    if catch_up_enabled && let Some(minutes) = catch_up_window(command) {
        return Some(VoiceCommand::CatchUp(minutes));
    }

    if is(&[
        "explain",
        "explain that",
        "explain this",
        "explain it",
        "what is that",
        "what's that",
        "what does that mean",
    ]) {
        return Some(VoiceCommand::Explain);
    }
    if is(&[
        "follow up",
        "follow-up",
        "give me a follow up",
        "give me a follow-up",
        "give me follow ups",
        "give me follow-ups",
        "give me some follow ups",
        "give me some follow-ups",
        "suggest a follow up",
        "suggest a follow-up",
        "suggest follow ups",
        "suggest follow-ups",
        "follow up question",
        "follow-up question",
        "follow up questions",
        "follow-up questions",
        "next question",
        "what should i ask",
        "what should i ask next",
    ]) {
        return Some(VoiceCommand::FollowUp);
    }
    if is(&[
        "answer",
        "answer that",
        "answer this",
        "answer it",
        "suggest an answer",
        "help me answer",
        "help me answer that",
        "what should i say",
        "how should i answer",
        "how do i answer that",
    ]) {
        return Some(VoiceCommand::Answer);
    }
    if starts(&[
        "make that concise",
        "make it concise",
        "make that shorter",
        "make it shorter",
        "shorter",
        "be concise",
        "concise",
        "summarize that",
        "summarise that",
        "shorten",
        "tldr",
        "tl dr",
    ]) {
        return Some(VoiceCommand::Concise);
    }
    if starts(&[
        "more detail",
        "more details",
        "longer",
        "make it longer",
        "make that longer",
        "elaborate",
        "expand",
        "go deeper",
        "more depth",
        "in more detail",
    ]) {
        return Some(VoiceCommand::Detailed);
    }
    // Anything else is a free-form request, in the user's original words.
    let original = text.trim();
    let start = original
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| normalize(&original[i..]) == command)
        .unwrap_or(0);
    let ask = original[start..]
        .trim()
        .trim_start_matches([',', ':', '-'])
        .trim();
    Some(VoiceCommand::Ask(if ask.is_empty() {
        command.to_string()
    } else {
        ask.to_string()
    }))
}

/// The window of "catch me up [on the last N minutes]" (`command` is
/// normalized): `Some(5)` for a bare request, `None` when the phrase is
/// something else ("catch me up on the standup") or the window is not usable.
fn catch_up_window(command: &str) -> Option<u32> {
    let rest = command.strip_prefix("catch me up")?;
    let words: Vec<&str> = rest.split_whitespace().collect();
    if words.is_empty() {
        return Some(DEFAULT_CATCH_UP_MINUTES);
    }
    let mut i = 0;
    while i < words.len() && matches!(words[i], "on" | "for" | "over" | "the" | "last" | "past") {
        i += 1;
    }
    let minutes = match &words[i..] {
        ["minute"] => 1,
        [n, "minute" | "minutes" | "min" | "mins"] => number(n)?,
        _ => return None,
    };
    (1..=MAX_CATCH_UP_MINUTES)
        .contains(&minutes)
        .then_some(minutes)
}

/// A spoken or written number up to sixty ("5", "five", "twenty", "sixty").
fn number(word: &str) -> Option<u32> {
    if let Ok(n) = word.parse::<u32>() {
        return Some(n);
    }
    Some(match word {
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        "fifteen" => 15,
        "twenty" => 20,
        "thirty" => 30,
        "forty" => 40,
        "fifty" => 50,
        "sixty" => 60,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_me_up_with_an_optional_window() {
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up"),
            Some(VoiceCommand::CatchUp(5))
        );
        assert_eq!(
            parse_voice_command("Hey OpenMeet, catch me up on the last 10 minutes."),
            Some(VoiceCommand::CatchUp(10))
        );
        assert_eq!(
            parse_voice_command("OK OpenMeet catch me up for the last five minutes please"),
            Some(VoiceCommand::CatchUp(5))
        );
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up on the last minute"),
            Some(VoiceCommand::CatchUp(1))
        );
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up on the past twenty mins"),
            Some(VoiceCommand::CatchUp(20))
        );
        // A window that is not usable, or other wording, is a free-form request.
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up on the standup"),
            Some(VoiceCommand::Ask("catch me up on the standup".into()))
        );
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up on the last 0 minutes"),
            Some(VoiceCommand::Ask(
                "catch me up on the last 0 minutes".into()
            ))
        );
        assert_eq!(
            parse_voice_command("OpenMeet, catch me up on the last 90 minutes"),
            Some(VoiceCommand::Ask(
                "catch me up on the last 90 minutes".into()
            ))
        );
    }

    #[test]
    fn catch_me_up_is_a_workplace_command_only() {
        assert_eq!(
            parse_voice_command_in("OpenMeet, catch me up", Mode::Workplace),
            Some(VoiceCommand::CatchUp(5))
        );
        assert_eq!(
            parse_voice_command_in("OpenMeet, catch me up", Mode::Candidate),
            Some(VoiceCommand::Ask("catch me up".into()))
        );
        assert_eq!(
            parse_voice_command_in("OpenMeet, catch me up", Mode::Interviewer),
            Some(VoiceCommand::Ask("catch me up".into()))
        );
        assert_eq!(
            parse_voice_command_in("OpenMeet, give me a follow-up", Mode::Workplace),
            Some(VoiceCommand::FollowUp)
        );
    }

    #[test]
    fn wake_phrase_required() {
        assert_eq!(parse_voice_command("Explain that"), None);
        assert_eq!(parse_voice_command("We opened the meeting"), None);
        assert_eq!(parse_voice_command("OpenMeet"), None);
        assert_eq!(parse_voice_command("Open meet."), None);
    }

    #[test]
    fn commands() {
        assert_eq!(
            parse_voice_command("OpenMeet, explain that."),
            Some(VoiceCommand::Explain)
        );
        assert_eq!(
            parse_voice_command("Hey open meet, give me a follow-up"),
            Some(VoiceCommand::FollowUp)
        );
        assert_eq!(
            parse_voice_command("Open meat answer that"),
            Some(VoiceCommand::Answer)
        );
        assert_eq!(
            parse_voice_command("OpenMeet, make that concise please"),
            Some(VoiceCommand::Concise)
        );
        assert_eq!(
            parse_voice_command("Okay OpenMeet: more detail"),
            Some(VoiceCommand::Detailed)
        );
        assert_eq!(
            parse_voice_command("openmeat shorter"),
            Some(VoiceCommand::Concise)
        );
    }

    #[test]
    fn free_form_requests_keep_original_wording() {
        assert_eq!(
            parse_voice_command("OpenMeet, what does this code do?"),
            Some(VoiceCommand::Ask("what does this code do?".into()))
        );
        assert_eq!(
            parse_voice_command("Hey OpenMeet - Explain this architecture simply."),
            Some(VoiceCommand::Ask(
                "Explain this architecture simply.".into()
            ))
        );
    }
}
