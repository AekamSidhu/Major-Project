//! Incremental line splitter for newline-delimited JSON (Ollama streams).

/// Longest accepted line; longer lines are discarded.
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

/// Splits a byte stream into `\n`-terminated lines (a trailing `\r` is
/// removed). Lines are decoded only when complete, so chunks may split
/// multi-byte UTF-8 sequences anywhere.
#[derive(Debug, Default)]
pub(crate) struct LineSplitter {
    buf: Vec<u8>,
    discarding: bool,
}

impl LineSplitter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feeds bytes; returns the lines they completed.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = bytes;
        while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
            if self.discarding {
                self.discarding = false;
                self.buf.clear();
            } else {
                self.buf.extend_from_slice(&rest[..pos]);
                out.push(decode(&self.buf));
                self.buf.clear();
            }
            rest = &rest[pos + 1..];
        }
        if !rest.is_empty() && !self.discarding {
            if self.buf.len() + rest.len() > MAX_LINE_BYTES {
                self.buf.clear();
                self.discarding = true;
            } else {
                self.buf.extend_from_slice(rest);
            }
        }
        out
    }

    /// Returns the unterminated last line, if any.
    pub(crate) fn finish(&mut self) -> Option<String> {
        let discarding = std::mem::take(&mut self.discarding);
        let buf = std::mem::take(&mut self.buf);
        (!discarding && !buf.is_empty()).then(|| decode(&buf))
    }
}

fn decode(line: &[u8]) -> String {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    String::from_utf8_lossy(line).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_at_any_boundary() {
        let input = "{\"a\":\"é\"}\r\n\n{\"b\":2}\n{\"c\":3}".as_bytes();
        for split in 0..=input.len() {
            let mut s = LineSplitter::new();
            let mut lines = s.push(&input[..split]);
            lines.extend(s.push(&input[split..]));
            lines.extend(s.finish());
            assert_eq!(lines, vec!["{\"a\":\"é\"}", "", "{\"b\":2}", "{\"c\":3}"], "split {split}");
        }
    }

    #[test]
    fn overlong_line_is_dropped() {
        let mut s = LineSplitter::new();
        assert!(s.push(&vec![b'x'; MAX_LINE_BYTES + 1]).is_empty());
        assert_eq!(s.push(b"tail\nok\n"), vec!["ok"]);
        assert_eq!(s.finish(), None);
    }
}
