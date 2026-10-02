//! Incremental `text/event-stream` (Server-Sent Events) parser.
//!
//! Follows the WHATWG event-stream interpretation rules: lines end with CRLF,
//! LF or CR (also when the CR and LF arrive in different chunks), lines
//! starting with `:` are comments, a single space after the field colon is
//! removed, `data` lines are joined with `\n`, an empty line dispatches the
//! event, a leading UTF-8 BOM is ignored and `id` persists across events.
//! Bytes may be split anywhere — including inside a multi-byte UTF-8 sequence —
//! because lines are only decoded once complete.
//!
//! Two deliberate differences from a browser `EventSource`:
//! * [`SseParser::finish`] dispatches a final event that was not terminated
//!   by a blank line (some servers close the connection right after the last
//!   `data:` line);
//! * a single line longer than [`MAX_LINE_BYTES`] is dropped instead of
//!   growing the buffer without bound.

/// Longest accepted line; longer lines are discarded.
pub const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// One dispatched event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    /// `event:` field, `None` when absent (the default "message" type).
    pub event: Option<String>,
    /// `data:` lines joined with `\n`.
    pub data: String,
    /// Last event id seen on the stream, if any.
    pub id: Option<String>,
}

/// Incremental parser; feed it response body chunks as they arrive.
#[derive(Debug, Default)]
pub struct SseParser {
    /// Bytes of the current, not yet terminated line.
    line: Vec<u8>,
    /// Number of BOM bytes matched at the start of the stream; `None` once the
    /// BOM check is over.
    bom_matched: Option<usize>,
    bom_done: bool,
    /// The previous chunk ended with CR: skip a leading LF.
    skip_lf: bool,
    /// Discarding the rest of an over-long line.
    discarding: bool,
    event_type: String,
    data: String,
    last_event_id: String,
}

impl SseParser {
    /// New parser at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds bytes and returns every event completed by them.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        let mut rest = bytes;
        if !self.bom_done {
            rest = self.strip_bom(rest);
            if !self.bom_done {
                return out;
            }
        }
        while !rest.is_empty() {
            if self.skip_lf {
                self.skip_lf = false;
                if rest[0] == b'\n' {
                    rest = &rest[1..];
                    continue;
                }
            }
            match rest.iter().position(|&b| b == b'\n' || b == b'\r') {
                Some(pos) => {
                    if self.discarding {
                        self.discarding = false;
                        self.line.clear();
                    } else if self.line.is_empty() {
                        self.process_line(&rest[..pos], &mut out);
                    } else {
                        self.line.extend_from_slice(&rest[..pos]);
                        let line = std::mem::take(&mut self.line);
                        self.process_line(&line, &mut out);
                        self.line = line;
                        self.line.clear();
                    }
                    if rest[pos] == b'\r' {
                        self.skip_lf = true;
                    }
                    rest = &rest[pos + 1..];
                }
                None => {
                    if !self.discarding {
                        if self.line.len() + rest.len() > MAX_LINE_BYTES {
                            self.line.clear();
                            self.discarding = true;
                        } else {
                            self.line.extend_from_slice(rest);
                        }
                    }
                    rest = &[];
                }
            }
        }
        out
    }

    /// Ends the stream: processes an unterminated last line and dispatches a
    /// pending event. The parser is reset afterwards.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if !self.bom_done {
            // Fewer than three bytes arrived and they looked like a BOM prefix:
            // they are data after all.
            let pending: Vec<u8> = BOM[..self.bom_matched.unwrap_or(0)].to_vec();
            self.bom_done = true;
            self.line.extend_from_slice(&pending);
        }
        if !self.discarding && !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, &mut out);
        }
        self.dispatch(&mut out);
        *self = SseParser::default();
        out
    }

    fn strip_bom<'a>(&mut self, mut bytes: &'a [u8]) -> &'a [u8] {
        let mut matched = self.bom_matched.unwrap_or(0);
        while matched < BOM.len() {
            let Some(&b) = bytes.first() else {
                self.bom_matched = Some(matched);
                return bytes;
            };
            if b == BOM[matched] {
                matched += 1;
                bytes = &bytes[1..];
            } else {
                // Not a BOM: the bytes matched so far are regular data.
                self.line.extend_from_slice(&BOM[..matched]);
                self.bom_done = true;
                self.bom_matched = None;
                return bytes;
            }
        }
        self.bom_done = true;
        self.bom_matched = None;
        bytes
    }

    fn process_line(&mut self, line: &[u8], out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line[0] == b':' {
            return;
        }
        let (field, value) = match line.iter().position(|&b| b == b':') {
            Some(pos) => {
                let value = &line[pos + 1..];
                let value = value.strip_prefix(b" ").unwrap_or(value);
                (&line[..pos], value)
            }
            None => (line, &[][..]),
        };
        match field {
            b"event" => self.event_type = String::from_utf8_lossy(value).into_owned(),
            b"data" => {
                self.data.push_str(&String::from_utf8_lossy(value));
                self.data.push('\n');
            }
            b"id" => {
                if !value.contains(&0) {
                    self.last_event_id = String::from_utf8_lossy(value).into_owned();
                }
            }
            // `retry` only matters for reconnecting clients; unknown fields
            // are ignored per spec.
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if self.data.is_empty() {
            self.event_type.clear();
            return;
        }
        let mut data = std::mem::take(&mut self.data);
        if data.ends_with('\n') {
            data.pop();
        }
        let event = std::mem::take(&mut self.event_type);
        out.push(SseEvent {
            event: (!event.is_empty()).then_some(event),
            data,
            id: (!self.last_event_id.is_empty()).then(|| self.last_event_id.clone()),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event: Option<&str>, data: &str) -> SseEvent {
        SseEvent {
            event: event.map(str::to_string),
            data: data.to_string(),
            id: None,
        }
    }

    /// Feeds `input` split at every possible position (two chunks) and in
    /// single bytes; every variant must produce `expected`.
    fn assert_all_splits(input: &[u8], expected: &[SseEvent]) {
        for split in 0..=input.len() {
            let mut p = SseParser::new();
            let mut got = p.push(&input[..split]);
            got.extend(p.push(&input[split..]));
            got.extend(p.finish());
            assert_eq!(got, expected, "split at {split}");
        }
        let mut p = SseParser::new();
        let mut got = Vec::new();
        for b in input {
            got.extend(p.push(std::slice::from_ref(b)));
        }
        got.extend(p.finish());
        assert_eq!(got, expected, "byte by byte");
    }

    #[test]
    fn basic_events_any_split() {
        let input = b"event: message_start\ndata: {\"a\":1}\n\nevent: ping\ndata: {}\n\ndata: [DONE]\n\n";
        assert_all_splits(
            input,
            &[
                ev(Some("message_start"), "{\"a\":1}"),
                ev(Some("ping"), "{}"),
                ev(None, "[DONE]"),
            ],
        );
    }

    #[test]
    fn crlf_and_cr_line_endings() {
        let input = b"data: one\r\n\r\ndata: two\r\rdata: three\n\n";
        assert_all_splits(input, &[ev(None, "one"), ev(None, "two"), ev(None, "three")]);
    }

    #[test]
    fn multiline_data_comments_and_fields() {
        let input = b": keep-alive comment\ndata: line1\ndata:line2\ndata\nretry: 100\nunknown: x\n\n";
        assert_all_splits(input, &[ev(None, "line1\nline2\n")]);
    }

    #[test]
    fn utf8_split_inside_multibyte_sequence() {
        let input = "data: héllo wörld – ✓ 日本\n\n".as_bytes();
        assert_all_splits(input, &[ev(None, "héllo wörld – ✓ 日本")]);
    }

    #[test]
    fn bom_is_stripped_only_at_start() {
        let mut input = BOM.to_vec();
        input.extend_from_slice(b"data: x\n\n");
        assert_all_splits(&input, &[ev(None, "x")]);
    }

    #[test]
    fn empty_data_event_is_dispatched_but_no_data_is_not() {
        assert_all_splits(b"data:\n\n", &[ev(None, "")]);
        assert_all_splits(b"event: foo\n\ndata: x\n\n", &[ev(None, "x")]);
    }

    #[test]
    fn ids_persist() {
        let mut p = SseParser::new();
        let got = p.push(b"id: 7\ndata: a\n\ndata: b\n\nid:\ndata: c\n\n");
        assert_eq!(got[0].id.as_deref(), Some("7"));
        assert_eq!(got[1].id.as_deref(), Some("7"));
        assert_eq!(got[2].id, None);
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let mut p = SseParser::new();
        assert!(p.push(b"data: tail").is_empty());
        assert_eq!(p.finish(), vec![ev(None, "tail")]);
        // The parser is reusable after finish.
        assert_eq!(p.push(b"data: again\n\n"), vec![ev(None, "again")]);
    }

    #[test]
    fn overlong_lines_are_dropped() {
        let mut p = SseParser::new();
        let big = vec![b'a'; MAX_LINE_BYTES + 10];
        assert!(p.push(b"data: ").is_empty());
        assert!(p.push(&big).is_empty());
        let got = p.push(b"\ndata: ok\n\n");
        assert_eq!(got, vec![ev(None, "ok")]);
    }

    #[test]
    fn short_non_bom_prefix_is_data() {
        let mut p = SseParser::new();
        assert!(p.push(&[0xEF]).is_empty());
        let got = p.push(b"\n\ndata: x\n\n");
        assert_eq!(got, vec![ev(None, "x")]);
    }
}
