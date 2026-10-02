//! Removes inline reasoning that some local models emit at the start of their
//! output (`<think>…</think>`, e.g. DeepSeek-R1 distills and Qwen3 served by
//! Ollama, llama.cpp or LM Studio). Reasoning is never surfaced as answer
//! text. Only a block that *starts* the response is removed, so answers that
//! merely mention the tag are left alone.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nothing visible yet: deciding whether the output starts with `<think>`.
    Start,
    /// Inside the reasoning block.
    Thinking,
    /// Visible text.
    Pass,
}

/// Streaming filter; tags may be split across chunks.
#[derive(Debug)]
pub(crate) struct ThinkFilter {
    state: State,
    buf: String,
    /// Drop whitespace right after `</think>` (until the first visible char).
    trim_leading: bool,
}

impl Default for ThinkFilter {
    fn default() -> Self {
        ThinkFilter {
            state: State::Start,
            buf: String::new(),
            trim_leading: false,
        }
    }
}

impl ThinkFilter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feeds streamed text; returns the visible part (possibly empty).
    pub(crate) fn push(&mut self, text: &str) -> String {
        match self.state {
            State::Pass => self.visible(text.to_string()),
            State::Start => {
                self.buf.push_str(text);
                let trimmed = self.buf.trim_start();
                if trimmed.is_empty() || OPEN.starts_with(trimmed) {
                    // Whitespace or a partial "<think" so far: wait.
                    return String::new();
                }
                if let Some(after) = trimmed.strip_prefix(OPEN) {
                    let after = after.to_string();
                    self.buf.clear();
                    self.state = State::Thinking;
                    return self.push_thinking(&after);
                }
                self.state = State::Pass;
                let all = std::mem::take(&mut self.buf);
                self.visible(all)
            }
            State::Thinking => self.push_thinking(text),
        }
    }

    /// End of stream: returns text held back while waiting for a tag.
    pub(crate) fn finish(&mut self) -> String {
        let out = match self.state {
            State::Start => std::mem::take(&mut self.buf),
            // An unterminated reasoning block is all reasoning.
            State::Thinking | State::Pass => String::new(),
        };
        *self = ThinkFilter::default();
        out
    }

    fn push_thinking(&mut self, text: &str) -> String {
        self.buf.push_str(text);
        if let Some(idx) = self.buf.find(CLOSE) {
            let after = self.buf[idx + CLOSE.len()..].to_string();
            self.buf.clear();
            self.state = State::Pass;
            self.trim_leading = true;
            return self.visible(after);
        }
        // Keep only a tail long enough to contain a split closing tag.
        let keep = CLOSE.len() - 1;
        if self.buf.len() > keep {
            let mut cut = self.buf.len() - keep;
            while !self.buf.is_char_boundary(cut) {
                cut -= 1;
            }
            self.buf.drain(..cut);
        }
        String::new()
    }

    fn visible(&mut self, text: String) -> String {
        if !self.trim_leading {
            return text;
        }
        let trimmed = text.trim_start();
        if trimmed.is_empty() {
            return String::new();
        }
        self.trim_leading = false;
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> String {
        let mut f = ThinkFilter::new();
        let mut out: String = chunks.iter().map(|c| f.push(c)).collect();
        out.push_str(&f.finish());
        out
    }

    fn all_splits(input: &str, expected: &str) {
        for (i, _) in input.char_indices().chain([(input.len(), ' ')]) {
            assert_eq!(run(&[&input[..i], &input[i..]]), expected, "split at {i}");
        }
        let chars: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(run(&refs), expected, "char by char");
    }

    #[test]
    fn strips_leading_think_block() {
        all_splits("<think>\nreasoning… über</think>\n\nThe answer.", "The answer.");
        all_splits("  <think>x</think>Answer", "Answer");
    }

    #[test]
    fn passes_normal_text() {
        all_splits("Hello world", "Hello world");
        all_splits("Use the <think> tag", "Use the <think> tag");
        all_splits("<thin", "<thin");
        all_splits("<b>bold</b>", "<b>bold</b>");
    }

    #[test]
    fn unterminated_block_is_hidden() {
        assert_eq!(run(&["<think>never closes"]), "");
    }

    #[test]
    fn whitespace_only_answer() {
        assert_eq!(run(&["<think>a</think>", "   "]), "");
        assert_eq!(run(&["<think>a</think>", "  ", " x "]), "x ");
    }
}
