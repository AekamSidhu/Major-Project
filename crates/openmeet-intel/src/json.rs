//! Tolerant JSON helpers for parsing model output.
//!
//! Models wrap JSON in code fences, prepend prose ("Sure, here is…"), leave
//! trailing commas or emit several objects. These helpers extract every
//! balanced JSON value they can find and read fields leniently.

use serde_json::{Map, Value};

/// Normalizes line endings and drops Markdown code-fence lines.
pub(crate) fn strip_code_fences(text: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");
    unified
        .lines()
        .filter(|l| !l.trim_start().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns the end (exclusive byte index) of the balanced JSON value that
/// starts at `start` (which must be `{` or `[`), honoring strings and escapes.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Removes trailing commas before `}` / `]` outside of strings.
fn repair_trailing_commas(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escaped = false;
    for (i, &c) in chars.iter().enumerate() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            continue;
        }
        if c == ',' {
            let next = chars[i + 1..].iter().find(|ch| !ch.is_whitespace());
            if matches!(next, Some('}') | Some(']')) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Parses a JSON slice, retrying once with trailing commas removed.
pub(crate) fn parse_lenient(slice: &str) -> Option<Value> {
    serde_json::from_str::<Value>(slice)
        .ok()
        .or_else(|| serde_json::from_str::<Value>(&repair_trailing_commas(slice)).ok())
}

/// Extracts every top-level JSON object or array embedded in `text`.
pub(crate) fn extract_json_values(text: &str) -> Vec<Value> {
    let cleaned = strip_code_fences(text);
    let bytes = cleaned.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'{' || b == b'[' {
            if let Some(end) = balanced_end(bytes, i) {
                if let Some(v) = parse_lenient(&cleaned[i..end]) {
                    out.push(v);
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// First JSON object in `text` (objects nested in a leading array count).
pub(crate) fn first_json_object(text: &str) -> Option<Map<String, Value>> {
    for v in extract_json_values(text) {
        match v {
            Value::Object(m) => return Some(m),
            Value::Array(items) => {
                if let Some(Value::Object(m)) = items.into_iter().find(|x| x.is_object()) {
                    return Some(m);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every JSON object in `text`, flattening arrays of objects.
pub(crate) fn all_json_objects(text: &str) -> Vec<Map<String, Value>> {
    let mut out = Vec::new();
    for v in extract_json_values(text) {
        match v {
            Value::Object(m) => out.push(m),
            Value::Array(items) => {
                out.extend(items.into_iter().filter_map(|x| match x {
                    Value::Object(m) => Some(m),
                    _ => None,
                }));
            }
            _ => {}
        }
    }
    out
}

fn key_norm(k: &str) -> String {
    k.chars()
        .filter(|c| *c != '_' && *c != '-' && *c != ' ')
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Looks a field up by any of `keys`, ignoring case, `_`, `-` and spaces.
pub(crate) fn get<'a>(obj: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            return Some(v);
        }
    }
    let wanted: Vec<String> = keys.iter().map(|k| key_norm(k)).collect();
    obj.iter()
        .find(|(k, _)| wanted.contains(&key_norm(k)))
        .map(|(_, v)| v)
}

/// String field (numbers / bools are stringified); empty strings are `None`.
pub(crate) fn get_str(obj: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    let v = get(obj, keys)?;
    let s = match v {
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => return None,
    };
    if s.is_empty() { None } else { Some(s) }
}

/// Boolean field; accepts `true`, `"yes"`, `"true"`, `1`, …
pub(crate) fn get_bool(obj: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    match get(obj, keys)? {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "y" | "1" => Some(true),
            "false" | "no" | "n" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Numeric field; accepts numbers, numeric strings and percentages.
pub(crate) fn get_f32(obj: &Map<String, Value>, keys: &[&str]) -> Option<f32> {
    match get(obj, keys)? {
        Value::Number(n) => n.as_f64().map(|f| f as f32),
        Value::String(s) => {
            let t = s.trim();
            if let Some(p) = t.strip_suffix('%') {
                p.trim().parse::<f32>().ok().map(|f| f / 100.0)
            } else {
                t.parse::<f32>().ok()
            }
        }
        _ => None,
    }
    .filter(|f| f.is_finite())
}

/// List of strings. Accepts an array of strings, an array of objects (their
/// `text` / `question` / `name` / `item` field) or a single newline / bullet
/// separated string.
pub(crate) fn get_str_list(obj: &Map<String, Value>, keys: &[&str]) -> Vec<String> {
    match get(obj, keys) {
        Some(v) => value_to_str_list(v),
        None => Vec::new(),
    }
}

pub(crate) fn value_to_str_list(v: &Value) -> Vec<String> {
    match v {
        Value::Array(items) => items
            .iter()
            .filter_map(|x| match x {
                Value::String(s) => Some(s.trim().to_string()),
                Value::Number(n) => Some(n.to_string()),
                Value::Object(m) => get_str(
                    m,
                    &[
                        "text",
                        "question",
                        "name",
                        "item",
                        "title",
                        "description",
                        "summary",
                    ],
                ),
                _ => None,
            })
            .filter(|s| !s.is_empty())
            .collect(),
        Value::String(s) => s
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches(['-', '*', '•'])
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_objects_from_prose_and_fences() {
        let text = "Sure! Here it is:\n```json\n{\"a\": 1, \"b\": \"x}\"}\n```\nThanks";
        let obj = first_json_object(text).unwrap();
        assert_eq!(obj["a"], 1);
        assert_eq!(obj["b"], "x}");
    }

    #[test]
    fn repairs_trailing_commas() {
        let obj = first_json_object("{\"a\": [1, 2,], \"b\": true,}").unwrap();
        assert_eq!(obj["a"].as_array().unwrap().len(), 2);
        assert_eq!(get_bool(&obj, &["b"]), Some(true));
    }

    #[test]
    fn lenient_keys_and_values() {
        let obj = first_json_object(
            "{\"is_question\": \"yes\", \"Confidence\": \"85%\", \"items\": \"- a\\n- b\"}",
        )
        .unwrap();
        assert_eq!(get_bool(&obj, &["isQuestion"]), Some(true));
        assert!((get_f32(&obj, &["confidence"]).unwrap() - 0.85).abs() < 1e-6);
        assert_eq!(get_str_list(&obj, &["items"]), vec!["a", "b"]);
    }

    #[test]
    fn multiple_objects_and_arrays() {
        let objs = all_json_objects("{\"x\":1}\n[{\"y\":2},{\"z\":3}] {broken");
        assert_eq!(objs.len(), 3);
    }

    #[test]
    fn unbalanced_input_is_ignored() {
        assert!(first_json_object("{\"a\": ").is_none());
        assert!(first_json_object("no json here").is_none());
    }
}
