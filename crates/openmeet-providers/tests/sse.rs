//! Public `SseParser` behaviour on realistic provider streams split at
//! arbitrary byte boundaries.

use openmeet_providers::{SseEvent, SseParser};

const ANTHROPIC: &str = concat!(
    "event: message_start\r\n",
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\r\n",
    "\r\n",
    ": comment line\r\n",
    "event: content_block_delta\r\n",
    "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Grüße ✓ 日本\"}}\r\n",
    "\r\n",
    "event: message_stop\r\n",
    "data: {\"type\":\"message_stop\"}\r\n",
    "\r\n"
);

const OPENAI: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"multi\"}}]}\n",
    "data: line\n\n",
    "id: 42\n",
    "data: [DONE]\n\n"
);

fn expected_anthropic() -> Vec<SseEvent> {
    vec![
        SseEvent {
            event: Some("message_start".into()),
            data: "{\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}".into(),
            id: None,
        },
        SseEvent {
            event: Some("content_block_delta".into()),
            data: "{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Grüße ✓ 日本\"}}".into(),
            id: None,
        },
        SseEvent {
            event: Some("message_stop".into()),
            data: "{\"type\":\"message_stop\"}".into(),
            id: None,
        },
    ]
}

fn expected_openai() -> Vec<SseEvent> {
    vec![
        SseEvent {
            event: None,
            data: "{\"choices\":[{\"delta\":{\"content\":\"a\"}}]}".into(),
            id: None,
        },
        SseEvent {
            event: None,
            data: "{\"choices\":[{\"delta\":{\"content\":\"multi\"}}]}\nline".into(),
            id: None,
        },
        SseEvent {
            event: None,
            data: "[DONE]".into(),
            id: Some("42".into()),
        },
    ]
}

/// Small deterministic generator (no external crates).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % bound
    }
}

fn parse_in_random_chunks(input: &[u8], seed: u64) -> Vec<SseEvent> {
    let mut rng = Lcg(seed);
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < input.len() {
        let len = 1 + rng.next(12);
        let end = (pos + len).min(input.len());
        out.extend(parser.push(&input[pos..end]));
        pos = end;
    }
    out.extend(parser.finish());
    out
}

#[test]
fn random_chunk_boundaries_produce_identical_events() {
    for seed in 0..200 {
        assert_eq!(
            parse_in_random_chunks(ANTHROPIC.as_bytes(), seed),
            expected_anthropic(),
            "anthropic seed {seed}"
        );
        assert_eq!(
            parse_in_random_chunks(OPENAI.as_bytes(), seed),
            expected_openai(),
            "openai seed {seed}"
        );
    }
}

#[test]
fn every_two_way_split() {
    let input = ANTHROPIC.as_bytes();
    for split in 0..=input.len() {
        let mut parser = SseParser::new();
        let mut events = parser.push(&input[..split]);
        events.extend(parser.push(&input[split..]));
        events.extend(parser.finish());
        assert_eq!(events, expected_anthropic(), "split at {split}");
    }
}

#[test]
fn events_are_emitted_as_soon_as_they_complete() {
    let mut parser = SseParser::new();
    assert!(parser.push(b"data: one\n").is_empty());
    let events = parser.push(b"\ndata: tw");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, "one");
    assert!(parser.push(b"o").is_empty());
    let events = parser.push(b"\n\n");
    assert_eq!(events[0].data, "two");
    assert!(parser.finish().is_empty());
}

#[test]
fn unterminated_final_event_is_flushed_by_finish() {
    let mut parser = SseParser::new();
    assert!(parser.push(b"event: error\ndata: {\"x\":1}").is_empty());
    assert_eq!(
        parser.finish(),
        vec![SseEvent {
            event: Some("error".into()),
            data: "{\"x\":1}".into(),
            id: None,
        }]
    );
}
