//! Parity with the shared answer-integrity spec (`spec/integrity/`): the Rust
//! implementation must reproduce every fixture to within its tolerance.
//!
//! The fixtures are read from disk at test time, so a regenerated fixture
//! (`python -I spec/integrity/reference.py`) is picked up without code
//! changes.

use std::path::PathBuf;

use openmeet_core::integrity::{IntegrityLevel, IntegritySignalName};
use openmeet_intel::integrity::delivery::{
    BASELINE_ANSWERS, FILLER_BIGRAMS, FILLERS, MAD_TO_SD, MIN_SPAN_S, MIN_SPEAKING_S, MIN_WORDS,
    PAUSE_MIN_S, WEIGHTS,
};
use openmeet_intel::integrity::{
    DeliveryFeatures, DeliveryTracker, FusionInput, IntegrityThresholds, SPEC_VERSION, SpeechUnit,
    SpeechWord, delivery_features, echo_similarity, fuse, spec_thresholds,
};
use serde_json::Value;

const DEFAULT_TOLERANCE: f64 = 1e-6;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("spec")
        .join("integrity")
}

fn load(name: &str) -> Value {
    let path = spec_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("invalid JSON in {name}: {e}"))
}

fn tolerance(doc: &Value) -> f64 {
    doc["tolerance"].as_f64().unwrap_or(DEFAULT_TOLERANCE)
}

fn opt_f64(v: &Value) -> Option<f64> {
    if v.is_null() {
        None
    } else {
        Some(
            v.as_f64()
                .unwrap_or_else(|| panic!("expected a number, got {v}")),
        )
    }
}

fn close(label: &str, got: Option<f64>, want: &Value, tol: f64) {
    match (got, opt_f64(want)) {
        (None, None) => {}
        (Some(g), Some(w)) => assert!(
            (g - w).abs() <= tol,
            "{label}: got {g}, expected {w} (tolerance {tol})"
        ),
        (g, w) => panic!("{label}: got {g:?}, expected {w:?}"),
    }
}

fn unit(v: &Value) -> SpeechUnit {
    SpeechUnit {
        start: v["start"].as_f64().unwrap(),
        end: v["end"].as_f64().unwrap(),
        text: v["text"].as_str().unwrap_or_default().to_string(),
        words: v.get("words").and_then(Value::as_array).map(|ws| {
            ws.iter()
                .map(|w| SpeechWord {
                    word: w["word"].as_str().unwrap().to_string(),
                    start: w["start"].as_f64().unwrap(),
                    end: w["end"].as_f64().unwrap(),
                })
                .collect()
        }),
    }
}

fn units(v: &Value) -> Vec<SpeechUnit> {
    v.as_array().unwrap().iter().map(unit).collect()
}

fn check_features(label: &str, got: Option<&DeliveryFeatures>, want: &Value, tol: f64) {
    let Some(f) = got else {
        assert!(want.is_null(), "{label}: got no features, expected {want}");
        return;
    };
    assert!(!want.is_null(), "{label}: expected no features, got {f:?}");
    assert_eq!(
        f.granularity.as_str(),
        want["granularity"].as_str().unwrap(),
        "{label}: granularity"
    );
    assert_eq!(
        f.word_count as u64,
        want["word_count"].as_u64().unwrap(),
        "{label}: word_count"
    );
    close(
        &format!("{label}.speaking_s"),
        Some(f.speaking_s),
        &want["speaking_s"],
        tol,
    );
    close(
        &format!("{label}.span_s"),
        Some(f.span_s),
        &want["span_s"],
        tol,
    );
    close(
        &format!("{label}.latency_s"),
        f.latency_s,
        &want["latency_s"],
        tol,
    );
    close(
        &format!("{label}.rate_wpm"),
        f.rate_wpm,
        &want["rate_wpm"],
        tol,
    );
    close(
        &format!("{label}.pause_per_min"),
        f.pause_per_min,
        &want["pause_per_min"],
        tol,
    );
    close(
        &format!("{label}.mean_pause_s"),
        Some(f.mean_pause_s),
        &want["mean_pause_s"],
        tol,
    );
    close(
        &format!("{label}.filler_per_100w"),
        f.filler_per_100w,
        &want["filler_per_100w"],
        tol,
    );
}

#[test]
fn spec_version_matches() {
    let thresholds = spec_thresholds();
    assert_eq!(thresholds.spec_version, SPEC_VERSION);
    for name in [
        "delivery_cases.json",
        "fusion_cases.json",
        "echo_cases.json",
    ] {
        assert_eq!(
            load(name)["spec_version"].as_u64(),
            Some(u64::from(SPEC_VERSION)),
            "{name} was generated for another spec version"
        );
    }
}

#[test]
fn embedded_thresholds_equal_the_file_on_disk() {
    let path = spec_dir().join("thresholds.json");
    let on_disk = IntegrityThresholds::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(
        &on_disk,
        spec_thresholds(),
        "rebuild: thresholds.json changed"
    );
}

#[test]
fn delivery_constants_match_the_spec() {
    let doc = load("delivery_cases.json");
    let c = &doc["constants"];
    assert_eq!(c["pause_min_s"].as_f64(), Some(PAUSE_MIN_S));
    assert_eq!(c["min_speaking_s"].as_f64(), Some(MIN_SPEAKING_S));
    assert_eq!(c["min_span_s"].as_f64(), Some(MIN_SPAN_S));
    assert_eq!(c["min_words"].as_u64(), Some(MIN_WORDS as u64));
    assert_eq!(
        c["baseline_answers"].as_u64(),
        Some(BASELINE_ANSWERS as u64)
    );
    assert_eq!(c["mad_to_sd"].as_f64(), Some(MAD_TO_SD));
    let fillers: Vec<&str> = c["fillers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut ours = FILLERS.to_vec();
    ours.sort_unstable();
    assert_eq!(fillers, ours);
    let bigrams: Vec<&str> = c["filler_bigrams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut ours: Vec<String> = FILLER_BIGRAMS
        .iter()
        .map(|(a, b)| format!("{a} {b}"))
        .collect();
    ours.sort();
    assert_eq!(bigrams, ours);
    let weights = c["weights"].as_object().unwrap();
    let floors = c["floors"].as_object().unwrap();
    assert_eq!(weights.len(), WEIGHTS.len());
    assert_eq!(floors.len(), WEIGHTS.len());
    for (name, weight, floor) in WEIGHTS {
        assert_eq!(weights[name].as_f64(), Some(weight), "weight of {name}");
        assert_eq!(floors[name].as_f64(), Some(floor), "floor of {name}");
    }
}

#[test]
fn delivery_single_answers() {
    let doc = load("delivery_cases.json");
    let tol = tolerance(&doc);
    let cases = doc["single_answers"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let f = delivery_features(opt_f64(&case["question_end"]), &units(&case["utterances"]));
        check_features(name, f.as_ref(), &case["expected_features"], tol);
    }
}

#[test]
fn delivery_session_with_baseline() {
    let doc = load("delivery_cases.json");
    let tol = tolerance(&doc);
    let session = &doc["session"];
    let answers = session["answers"].as_array().unwrap();
    let expected = session["expected"].as_array().unwrap();
    assert_eq!(answers.len(), expected.len());
    let thresholds = &spec_thresholds().delivery;
    let mut tracker = DeliveryTracker::new();
    for (i, (answer, want)) in answers.iter().zip(expected).enumerate() {
        let label = format!("session answer {i}");
        let f = delivery_features(
            opt_f64(&answer["question_end"]),
            &units(&answer["utterances"]),
        );
        check_features(&label, f.as_ref(), &want["features"], tol);
        let score = tracker.observe(f.as_ref(), thresholds);
        let ws = &want["score"];
        assert_eq!(
            score.level.as_str(),
            ws["level"].as_str().unwrap(),
            "{label}: level"
        );
        assert_eq!(
            score.calibrated,
            ws["calibrated"].as_bool().unwrap(),
            "{label}: calibrated"
        );
        close(&format!("{label}.index"), score.index, &ws["index"], tol);
        let wz = ws["z"].as_object().unwrap();
        assert_eq!(score.z.len(), wz.len(), "{label}: z features");
        for (k, v) in wz {
            close(
                &format!("{label}.z.{k}"),
                score.z.get(k.as_str()).copied(),
                v,
                tol,
            );
        }
    }
}

#[test]
fn fusion_cases() {
    let doc = load("fusion_cases.json");
    let cases = doc["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for (i, case) in cases.iter().enumerate() {
        let inputs: Vec<FusionInput> = case["signals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| FusionInput {
                name: IntegritySignalName::parse(s["name"].as_str().unwrap()).unwrap(),
                level: IntegrityLevel::parse(s["level"].as_str().unwrap()).unwrap(),
                calibrated: s["calibrated"].as_bool().unwrap(),
            })
            .collect();
        let got = fuse(&inputs);
        let want = &case["expected"];
        assert_eq!(
            got.level.as_str(),
            want["level"].as_str().unwrap(),
            "fusion case {i}"
        );
        let elevated: Vec<&str> = got.elevated.iter().map(|n| n.as_str()).collect();
        let want_elevated: Vec<&str> = want["elevated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(elevated, want_elevated, "fusion case {i}: elevated");
    }
}

#[test]
fn echo_cases() {
    let doc = load("echo_cases.json");
    let tol = DEFAULT_TOLERANCE;
    // The fixture was generated with the thresholds in thresholds.json.
    let fixture_thresholds = &doc["thresholds"];
    let ours = &spec_thresholds().echo_similarity;
    assert_eq!(fixture_thresholds["possible"].as_f64(), Some(ours.possible));
    assert_eq!(fixture_thresholds["high"].as_f64(), Some(ours.high));
    assert_eq!(
        fixture_thresholds["embedding_model"].as_str(),
        Some(ours.embedding_model.as_str())
    );
    let vec_of = |v: &Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    for (i, case) in doc["cases"].as_array().unwrap().iter().enumerate() {
        let answer = vec_of(&case["answer"]);
        let refs: Vec<Vec<f64>> = case["references"]
            .as_array()
            .unwrap()
            .iter()
            .map(vec_of)
            .collect();
        let got = echo_similarity(&answer, &refs, ours);
        let want = &case["expected"];
        assert_eq!(
            got.level.as_str(),
            want["level"].as_str().unwrap(),
            "echo case {i}: level"
        );
        assert_eq!(
            got.calibrated,
            want["calibrated"].as_bool().unwrap(),
            "echo case {i}: calibrated"
        );
        close(
            &format!("echo case {i}.similarity"),
            got.similarity,
            &want["similarity"],
            tol,
        );
    }
}

/// `to_transcript` reproduces `local-ai/integrity/text.py` (the expected
/// strings were produced by that Python function).
#[test]
fn to_transcript_matches_the_sidecar() {
    use openmeet_intel::integrity::to_transcript;
    let cases: [(&str, &str); 9] = [
        (
            "## Caching\n- We used **Redis** for `hot` keys.\n2) Then __expired__ them.",
            "caching we used redis for hot keys then expired them",
        ),
        (
            "I do n't think it 's right , they 're wrong and I 'm sure .",
            "i dont think its right theyre wrong and im sure",
        ),
        (
            "It wo n't work and ca n't , could n't , should n't",
            "it wont work and cant couldnt shouldnt",
        ),
        (
            "See https://example.com/a?b=1 and www.test.org or URL_12 for details.",
            "see and or for details",
        ),
        (
            "Don\u{2019}t worry: it\u{2019}s 3.14\u{d7} faster \u{2014} na\u{ef}ve caf\u{e9} test!",
            "dont worry its 3 14 faster na ve caf test",
        ),
        (
            "* bullet one\n* bullet two\n10. ten",
            "bullet one bullet two ten",
        ),
        (
            "   Multiple   spaces\tand\nnewlines   ",
            "multiple spaces and newlines",
        ),
        ("HELLO World's END", "hello worlds end"),
        ("", ""),
    ];
    for (input, expected) in cases {
        assert_eq!(to_transcript(input), expected, "{input:?}");
    }
}
