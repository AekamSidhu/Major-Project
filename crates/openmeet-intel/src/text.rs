//! Text utilities: normalization, tokenization, light stemming,
//! paraphrase-robust similarity, token budgeting, sentence splitting,
//! deterministic local embeddings ([`HashingEmbedder`]) and a BM25 keyword
//! index ([`Bm25Index`]).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;

use async_trait::async_trait;
use openmeet_core::ProviderError;
use openmeet_core::providers::EmbeddingProvider;
use openmeet_core::types::{ProviderInfo, ProviderLocation};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

fn unify_quote(c: char) -> char {
    match c {
        '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' | '`' | '\u{00B4}' => '\'',
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{00AB}' | '\u{00BB}' | '\u{2033}' => '"',
        other => other,
    }
}

/// Lowercases, unifies quotes, strips punctuation (apostrophes inside words
/// such as "don't" are kept) and collapses whitespace.
pub fn normalize(text: &str) -> String {
    let chars: Vec<char> = text.chars().map(unify_quote).collect();
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for (i, &c) in chars.iter().enumerate() {
        let keep = if c.is_alphanumeric() {
            true
        } else if c == '\'' {
            let prev = i.checked_sub(1).map(|p| chars[p]);
            let next = chars.get(i + 1).copied();
            matches!(prev, Some(p) if p.is_alphanumeric())
                && matches!(next, Some(n) if n.is_alphanumeric())
        } else {
            false
        };
        if keep {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(c.to_lowercase());
        } else {
            pending_space = true;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Stop words, synonyms and stemming
// ---------------------------------------------------------------------------

/// Function words plus the generic scaffolding of spoken questions ("tell me
/// about", "how did you handle", "walk me through", …) so that paraphrases of
/// the same question share the same content tokens.
const STOP_WORDS: &[&str] = &[
    "a", "about", "above", "across", "actually", "after", "again", "all", "almost", "already",
    "also", "always", "am", "an", "and", "another", "any", "anybody", "anyone", "anything",
    "approach", "are", "aren't", "around", "as", "ask", "asked", "at", "basically", "be",
    "became", "because", "become", "becomes", "becoming", "been", "before", "being", "between",
    "bit", "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "currently", "d",
    "deal", "dealt", "describe", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "done", "during", "each", "either", "else", "even", "ever", "every", "everyone",
    "everything", "exactly", "example", "examples", "explain", "few", "for", "from", "generally",
    "get", "gets", "getting", "give", "given", "go", "goes", "going", "gone", "got", "gotten",
    "guess", "had", "handle", "has", "hasn't", "have", "haven't", "having", "he", "her", "here",
    "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "just", "kind", "kinds", "know", "knew", "known", "let", "lets", "like", "ll",
    "look", "looking", "lot", "lots", "m", "made", "make", "makes", "making", "many", "may",
    "maybe", "me", "mean", "meant", "might", "mine", "more", "most", "much", "must", "my",
    "myself", "need", "needs", "neither", "never", "no", "nor", "not", "now", "o", "of", "off",
    "often", "ok", "okay", "on", "once", "one", "ones", "only", "onto", "or", "other", "others",
    "our", "ours", "ourselves", "out", "over", "overall", "own", "particular", "perhaps",
    "please", "pretty", "probably", "quite", "rather", "re", "really", "right", "s", "said",
    "same", "say", "says", "see", "seen", "share", "she", "should", "shouldn't", "so", "some",
    "somebody", "someone", "something", "sometimes", "sort", "specific", "specifically",
    "still", "stuff", "such", "sure", "t", "tackle", "take", "talk", "tell", "than", "thank",
    "thanks", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "thing", "things", "think", "this", "those", "though", "thought", "through", "time",
    "times", "to", "too", "took", "tried", "try", "trying", "type", "types", "uh", "um", "under",
    "up", "us", "use", "used", "uses", "using", "usually", "ve", "very", "via", "walk", "want",
    "wanted", "was", "wasn't", "way", "ways", "we", "well", "went", "were", "weren't", "what",
    "whatever", "when", "where", "whether", "which", "while", "who", "whom", "whose", "why",
    "will", "with", "within", "without", "won't", "would", "wouldn't", "yeah", "yes", "yet",
    "you", "your", "yours", "yourself", "yourselves", "question", "questions", "able",
    "thinking", "happened", "explanation",
];

static STOP_SET: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOP_WORDS.iter().copied().collect());

/// Stems of stop words, so inflections ("handled", "describing") are dropped.
static STOP_STEMS: LazyLock<HashSet<String>> =
    LazyLock::new(|| STOP_WORDS.iter().map(|w| stem(w)).collect());

/// Irregular forms and abbreviations mapped before stemming.
static PRE_STEM: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    [
        ("built", "build"),
        ("builds", "build"),
        ("chose", "choos"),
        ("chosen", "choos"),
        ("led", "lead"),
        ("wrote", "build"),
        ("written", "build"),
        ("ran", "run"),
        ("grew", "grow"),
        ("grown", "grow"),
        ("broke", "break"),
        ("broken", "break"),
        ("kept", "keep"),
        ("began", "begin"),
        ("begun", "begin"),
        ("bought", "buy"),
        ("met", "meet"),
        ("held", "hold"),
        ("db", "databas"),
        ("dbs", "databas"),
        ("rdbms", "databas"),
        ("datastore", "databas"),
        ("datastores", "databas"),
        ("k8s", "kubernet"),
        ("pg", "postgr"),
        ("postgresql", "postgr"),
        ("perf", "perform"),
        ("js", "javascript"),
        ("ts", "typescript"),
        ("auth", "authentic"),
        ("authn", "authentic"),
        ("authz", "authoriz"),
        ("prod", "production"),
        ("infra", "infrastructur"),
        ("config", "configur"),
        ("configs", "configur"),
        ("repo", "repositori"),
        ("repos", "repositori"),
        ("ci", "ci"),
        ("cd", "cd"),
        ("ml", "ml"),
        ("ai", "ai"),
        ("ui", "ui"),
        ("ux", "ux"),
        ("qa", "qa"),
        ("sla", "sla"),
        ("slo", "slo"),
        ("kpi", "metric"),
        ("kpis", "metric"),
        ("rps", "qps"),
        ("tps", "qps"),
    ]
    .into_iter()
    .collect()
});

/// Synonyms unified after stemming (stem -> canonical stem).
static POST_STEM: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    [
        // storage
        ("datastor", "databas"),
        ("postgr", "postgr"),
        // monitoring / observability
        ("observ", "monitor"),
        ("telemetri", "monitor"),
        // failure
        ("failur", "fail"),
        ("outag", "fail"),
        ("downtim", "fail"),
        ("unavail", "fail"),
        ("crash", "fail"),
        ("mistak", "fail"),
        // conflict / colleagues
        ("disagr", "conflict"),
        ("coworker", "colleagu"),
        ("teammat", "colleagu"),
        ("peer", "colleagu"),
        // difficulty
        ("hard", "challeng"),
        ("difficult", "challeng"),
        ("difficulti", "challeng"),
        ("tough", "challeng"),
        // choice
        ("choic", "choos"),
        ("pick", "choos"),
        ("select", "choos"),
        ("decid", "choos"),
        ("decision", "choos"),
        ("opt", "choos"),
        // building
        ("imple", "build"),
        ("develop", "build"),
        ("creat", "build"),
        ("construct", "build"),
        ("writ", "build"),
        // design / architecture
        ("architectur", "design"),
        ("architect", "design"),
        // performance / optimization
        ("performant", "perform"),
        ("optimis", "optim"),
        // misc
        ("limiter", "limit"),
        ("throttl", "limit"),
        ("replica", "replic"),
        ("background", "experi"),
        ("leadership", "lead"),
        ("leader", "lead"),
        ("reduction", "reduc"),
        ("rollout", "deploy"),
        ("mainten", "maintain"),
        ("defect", "bug"),
        ("customer", "user"),
        ("enduser", "user"),
        ("asynchron", "async"),
        ("synchron", "sync"),
    ]
    .into_iter()
    .collect()
});

/// Multi-word phrases rewritten before tokenization (normalized form).
const PHRASES: &[(&str, &str)] = &[
    ("machine learning", "ml"),
    ("trade off", "tradeoff"),
    ("trade offs", "tradeoff"),
    ("data store", "databas"),
    ("data stores", "databas"),
    ("go wrong", "fail"),
    ("goes wrong", "fail"),
    ("went wrong", "fail"),
    ("goes down", "fail"),
    ("go down", "fail"),
    ("went down", "fail"),
    ("rate limiting", "rate limit"),
    ("end to end", "e2e"),
    ("ci cd", "ci"),
    ("pub sub", "pubsub"),
];

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn strip_suffix_min<'a>(w: &'a str, suffix: &str, min_rest: usize) -> Option<&'a str> {
    let rest = w.strip_suffix(suffix)?;
    if rest.len() >= min_rest { Some(rest) } else { None }
}

/// Light English stemmer: inflectional suffixes first (`-s`, `-es`, `-ies`,
/// `-ed`, `-ing`, `-ly`), then common derivational suffixes (`-ation`,
/// `-ability`, `-ment`, `-ness`, `-ency`, …), then final `e` / `y`
/// normalization. "scaling", "scaled", "scales" and "scale" all become
/// "scal". Tokens with digits or of three characters or fewer are returned
/// unchanged.
pub fn stem(word: &str) -> String {
    let lower = word.to_lowercase();
    if lower.len() <= 3 || !lower.bytes().all(|b| b.is_ascii_lowercase()) {
        return lower;
    }
    let mut w: String = lower;
    // Irregular inflections.
    match w.as_str() {
        "goes" | "going" => return "go".to_string(),
        "does" | "doing" => return "do".to_string(),
        "data" => return w,
        _ => {}
    }

    // --- pass 1: inflections ---
    let mut stripped_ed_ing = false;
    if let Some(r) = strip_suffix_min(&w, "ies", 3) {
        w = format!("{r}y");
    } else if let Some(r) = strip_suffix_min(&w, "ied", 3) {
        w = format!("{r}y");
    } else if let Some(r) = strip_suffix_min(&w, "ingly", 3) {
        w = r.to_string();
        stripped_ed_ing = true;
    } else if let Some(r) = strip_suffix_min(&w, "edly", 3) {
        w = r.to_string();
        stripped_ed_ing = true;
    } else if let Some(r) = strip_suffix_min(&w, "ings", 3) {
        w = r.to_string();
        stripped_ed_ing = true;
    } else if let Some(r) = strip_suffix_min(&w, "ing", 3) {
        w = r.to_string();
        stripped_ed_ing = true;
    } else if let Some(r) = strip_suffix_min(&w, "ed", 3) {
        w = r.to_string();
        stripped_ed_ing = true;
    } else if w.ends_with("es")
        && (w.ends_with("sses")
            || w.ends_with("xes")
            || w.ends_with("zes")
            || w.ends_with("ches")
            || w.ends_with("shes"))
        && w.len() > 4
    {
        w.truncate(w.len() - 2);
    } else if w.ends_with('s')
        && !w.ends_with("ss")
        && !w.ends_with("us")
        && !w.ends_with("is")
        && w.len() > 3
    {
        w.truncate(w.len() - 1);
    } else if let Some(r) = strip_suffix_min(&w, "ly", 5) {
        w = r.to_string();
    }
    if stripped_ed_ing {
        let b = w.as_bytes();
        let n = b.len();
        if n >= 2
            && b[n - 1] == b[n - 2]
            && !is_vowel(b[n - 1])
            && !matches!(b[n - 1], b'l' | b's' | b'z')
        {
            w.truncate(n - 1);
        } else if w.ends_with("at")
            || w.ends_with("bl")
            || w.ends_with("iz")
            || w.ends_with("nc")
        {
            w.push('e');
        }
    }

    // --- pass 2: derivations (at most twice) ---
    const DERIVATIONAL: &[(&str, &str, usize)] = &[
        ("izations", "", 4),
        ("ization", "", 4),
        ("isations", "", 4),
        ("isation", "", 4),
        ("abilities", "", 3),
        ("ability", "", 3),
        ("ibilities", "", 3),
        ("ibility", "", 3),
        ("ational", "", 4),
        ("ations", "", 4),
        ("ation", "", 4),
        ("itions", "", 4),
        ("ition", "", 4),
        ("utions", "ut", 4),
        ("ution", "ut", 4),
        ("nesses", "", 4),
        ("ness", "", 4),
        ("ments", "", 4),
        ("ment", "", 4),
        ("ity", "", 4),
        ("ables", "", 3),
        ("able", "", 3),
        ("ibles", "", 3),
        ("ible", "", 3),
        ("ancy", "", 4),
        ("ance", "", 4),
        ("ency", "", 4),
        ("ence", "", 4),
        ("ent", "", 4),
        ("izing", "", 4),
        ("ized", "", 4),
        ("izes", "", 4),
        ("ize", "", 4),
        ("ating", "", 4),
        ("ated", "", 4),
        ("ates", "", 4),
        ("ate", "", 4),
        ("ous", "", 4),
        ("ive", "", 4),
        ("al", "", 5),
    ];
    for _ in 0..2 {
        let mut changed = false;
        for (suffix, replacement, min_rest) in DERIVATIONAL {
            if let Some(r) = strip_suffix_min(&w, suffix, *min_rest) {
                w = format!("{r}{replacement}");
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }

    // --- final normalization ---
    if w.len() > 3 && w.ends_with('e') {
        w.pop();
    }
    if w.len() > 3 && w.ends_with('y') {
        let b = w.as_bytes();
        if !is_vowel(b[b.len() - 2]) {
            w.pop();
            w.push('i');
        }
    }
    w
}

/// Splits a contraction / possessive and returns the meaningful part.
fn strip_contraction(word: &str) -> &str {
    if let Some(base) = word.strip_suffix("n't") {
        return match base {
            "ca" => "can",
            "wo" => "will",
            "sha" => "shall",
            other => other,
        };
    }
    match word.find('\'') {
        Some(i) => &word[..i],
        None => word,
    }
}

fn apply_phrases(norm: &str) -> String {
    let mut padded = format!(" {norm} ");
    for (from, to) in PHRASES {
        let needle = format!(" {from} ");
        if padded.contains(&needle) {
            padded = padded.replace(&needle, &format!(" {to} "));
        }
    }
    padded.trim().to_string()
}

/// True for function words and generic question scaffolding.
pub(crate) fn is_stop_word(word: &str) -> bool {
    STOP_SET.contains(word)
}

/// Content tokens: normalized, stop words removed, stemmed and with
/// synonyms unified ("db" / "database" / "datastore" -> "databas").
pub fn content_tokens(text: &str) -> Vec<String> {
    let norm = apply_phrases(&normalize(text));
    let mut out = Vec::new();
    for raw in norm.split_whitespace() {
        let word = strip_contraction(raw);
        if word.is_empty() || is_stop_word(word) {
            continue;
        }
        if let Some(canon) = PRE_STEM.get(word) {
            out.push((*canon).to_string());
            continue;
        }
        let s = stem(word);
        if STOP_STEMS.contains(&s) {
            continue;
        }
        if s.chars().count() < 2 && !s.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let s = POST_STEM.get(s.as_str()).map(|c| c.to_string()).unwrap_or(s);
        out.push(s);
    }
    out
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

/// Jaccard similarity of two token lists (as sets). 0 when both are empty.
pub fn jaccard(a: &[String], b: &[String]) -> f32 {
    let sa: HashSet<&String> = a.iter().collect();
    let sb: HashSet<&String> = b.iter().collect();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0.0;
    }
    sa.intersection(&sb).count() as f32 / union as f32
}

fn dice<T: Eq + std::hash::Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f32 {
    let total = a.len() + b.len();
    if total == 0 {
        return 0.0;
    }
    2.0 * a.intersection(b).count() as f32 / total as f32
}

fn unordered_pairs(tokens: &[String]) -> HashSet<(String, String)> {
    tokens
        .windows(2)
        .filter(|w| w[0] != w[1])
        .map(|w| {
            if w[0] <= w[1] {
                (w[0].clone(), w[1].clone())
            } else {
                (w[1].clone(), w[0].clone())
            }
        })
        .collect()
}

fn token_trigrams(tokens: &[String]) -> HashSet<String> {
    let mut out = HashSet::new();
    for t in tokens {
        let padded: Vec<char> = format!("#{t}#").chars().collect();
        for w in padded.windows(3) {
            out.insert(w.iter().collect());
        }
    }
    out
}

/// Paraphrase-robust similarity in 0..1 combining content-token overlap
/// (stemmed, synonyms unified), order-insensitive adjacent token pairs and
/// character trigrams of the content tokens.
///
/// "How did you scale the API?" vs "How did you handle API scaling?" scores
/// 1.0, while unrelated questions stay well below 0.25.
pub fn lexical_similarity(a: &str, b: &str) -> f32 {
    let ta = content_tokens(a);
    let tb = content_tokens(b);
    if ta.is_empty() || tb.is_empty() {
        let (na, nb) = (normalize(a), normalize(b));
        return if !na.is_empty() && na == nb { 1.0 } else { 0.0 };
    }
    let sa: HashSet<String> = ta.iter().cloned().collect();
    let sb: HashSet<String> = tb.iter().cloned().collect();
    let uni = dice(&sa, &sb);
    let (pa, pb) = (unordered_pairs(&ta), unordered_pairs(&tb));
    let bi = if pa.is_empty() || pb.is_empty() {
        uni
    } else {
        dice(&pa, &pb)
    };
    let tri = dice(&token_trigrams(&ta), &token_trigrams(&tb));
    (0.6 * uni + 0.15 * bi + 0.25 * tri).clamp(0.0, 1.0)
}

/// Cosine similarity; 0 for empty, mismatched, zero or non-finite vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    let c = dot / (na.sqrt() * nb.sqrt());
    if c.is_finite() {
        c.clamp(-1.0, 1.0) as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Token budgeting
// ---------------------------------------------------------------------------

/// Rough token estimate (~4 characters per token); never 0 for non-empty
/// text.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 { 0 } else { chars.div_ceil(4).max(1) }
}

/// Keeps the start of `text` within `max_tokens`, cutting at a word boundary
/// and appending "…". Returns the text unchanged when it already fits.
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    if estimate_tokens(text) <= max_tokens {
        return text.to_string();
    }
    if max_tokens == 0 {
        return String::new();
    }
    let max_chars = max_tokens * 4 - 1;
    let prefix: String = text.chars().take(max_chars).collect();
    let cut = match prefix.rfind(char::is_whitespace) {
        Some(i) if prefix[..i].chars().count() * 2 >= max_chars => &prefix[..i],
        _ => prefix.as_str(),
    };
    let cut = cut.trim_end_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ':'));
    format!("{cut}…")
}

/// Keeps the end of `text` within `max_tokens`, starting at a word boundary
/// and prefixing "…". Returns the text unchanged when it already fits.
pub fn tail_to_tokens(text: &str, max_tokens: usize) -> String {
    if estimate_tokens(text) <= max_tokens {
        return text.to_string();
    }
    if max_tokens == 0 {
        return String::new();
    }
    let max_chars = max_tokens * 4 - 1;
    let total = text.chars().count();
    let suffix: String = text.chars().skip(total - max_chars).collect();
    let start = match suffix.find(char::is_whitespace) {
        Some(i) if suffix[i..].chars().count() * 2 >= max_chars => &suffix[i..],
        _ => suffix.as_str(),
    };
    format!("…{}", start.trim_start())
}

/// Number of words containing at least one letter or digit.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_alphanumeric()))
        .count()
}

const ABBREVIATIONS: &[&str] = &[
    "e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "inc", "ltd",
    "co", "corp", "approx", "fig", "a.m", "p.m", "u.s", "cf", "al", "esp", "dept", "est",
];

fn is_abbreviation(chars: &[char], dot_index: usize) -> bool {
    // Word immediately before the dot.
    let mut start = dot_index;
    while start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '.') {
        start -= 1;
    }
    let word: String = chars[start..dot_index].iter().collect::<String>().to_lowercase();
    if word.is_empty() {
        return false;
    }
    if ABBREVIATIONS.contains(&word.as_str()) {
        return true;
    }
    // Single-letter initials ("J. Smith").
    word.chars().count() == 1 && chars[start].is_uppercase()
}

/// Splits text into sentences on `.`, `!`, `?`, `…` and line breaks, keeping
/// the terminal punctuation. Decimals ("3.5"), abbreviations ("e.g.") and
/// dotted names ("bit.ly") do not end a sentence. Unpunctuated text yields a
/// single sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    let push = |s: &mut String, out: &mut Vec<String>| {
        let t = s.trim();
        if !t.is_empty() {
            out.push(t.to_string());
        }
        s.clear();
    };
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' || c == '\r' {
            push(&mut current, &mut out);
            i += 1;
            continue;
        }
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '…') {
            // Absorb runs of terminal punctuation and closing quotes/brackets.
            let mut j = i + 1;
            while j < chars.len()
                && matches!(chars[j], '.' | '!' | '?' | '…' | '"' | '\'' | ')' | ']' | '”' | '’')
            {
                current.push(chars[j]);
                j += 1;
            }
            let at_end = j >= chars.len();
            let followed_by_space = !at_end && chars[j].is_whitespace();
            let is_boundary = if at_end || followed_by_space {
                if c == '.' && j == i + 1 {
                    let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                    let next_digit = chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
                    !(prev_digit && next_digit) && !is_abbreviation(&chars, i)
                } else {
                    true
                }
            } else {
                false
            };
            if is_boundary {
                push(&mut current, &mut out);
            }
            i = j;
            continue;
        }
        i += 1;
    }
    push(&mut current, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Hashing embeddings
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit: stable across platforms and Rust versions, so persisted
/// embeddings stay comparable.
fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    // Final avalanche (splitmix64 finalizer) so low bits are well mixed.
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// Deterministic local embeddings: feature hashing of stemmed content
/// unigrams, adjacent bigrams and character trigrams with signed hashing,
/// L2-normalized. No model download, no network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashingEmbedder {
    dims: usize,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        HashingEmbedder::new(Self::DEFAULT_DIMS)
    }
}

impl HashingEmbedder {
    /// Default vector size.
    pub const DEFAULT_DIMS: usize = 384;

    /// Creates an embedder producing `dims`-dimensional vectors (minimum 8).
    pub fn new(dims: usize) -> Self {
        HashingEmbedder { dims: dims.max(8) }
    }

    /// Vector dimensionality.
    pub fn dims(&self) -> usize {
        self.dims
    }

    fn add(&self, v: &mut [f32], namespace: &[u8], feature: &str, weight: f32) {
        let h = fnv1a64(&[namespace, feature.as_bytes()]);
        let idx = (h % self.dims as u64) as usize;
        let sign = if (h >> 40) & 1 == 1 { -1.0 } else { 1.0 };
        v[idx] += sign * weight;
    }

    /// Embeds one text. Empty / content-free text yields a zero vector.
    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dims];
        let tokens = content_tokens(text);
        for t in &tokens {
            self.add(&mut v, b"u:", t, 1.0);
        }
        for w in tokens.windows(2) {
            let pair = if w[0] <= w[1] {
                format!("{} {}", w[0], w[1])
            } else {
                format!("{} {}", w[1], w[0])
            };
            self.add(&mut v, b"b:", &pair, 0.6);
        }
        for word in normalize(text).split_whitespace() {
            if is_stop_word(word) {
                continue;
            }
            let padded: Vec<char> = format!("#{word}#").chars().collect();
            for tri in padded.windows(3) {
                let s: String = tri.iter().collect();
                self.add(&mut v, b"c:", &s, 0.25);
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl EmbeddingProvider for HashingEmbedder {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            id: "local_hash".to_string(),
            display_name: "Built-in hashing embeddings".to_string(),
            model: format!("hash-{}", self.dims),
            location: ProviderLocation::Local,
            endpoint: None,
        }
    }

    async fn embed(
        &self,
        texts: Vec<String>,
        cancel: CancellationToken,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in &texts {
            if cancel.is_cancelled() {
                return Err(ProviderError::Cancelled);
            }
            out.push(self.embed_one(t));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Bm25Doc {
    terms: HashMap<String, u32>,
    len: u32,
}

/// In-memory BM25 (Okapi, k1 = 1.2, b = 0.75) keyword index over content
/// tokens, used together with embeddings for hybrid retrieval.
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    docs: HashMap<String, Bm25Doc>,
    postings: HashMap<String, HashMap<String, u32>>,
    total_len: u64,
}

impl Bm25Index {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// Creates an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Indexes `text` under `id`; re-adding an id replaces it.
    pub fn add(&mut self, id: &str, text: &str) {
        self.remove(id);
        let tokens = content_tokens(text);
        let mut terms: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *terms.entry(t).or_insert(0) += 1;
        }
        let len: u32 = terms.values().sum();
        for (term, tf) in &terms {
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id.to_string(), *tf);
        }
        self.total_len += len as u64;
        self.docs.insert(id.to_string(), Bm25Doc { terms, len });
    }

    /// Removes a document (no-op for unknown ids).
    pub fn remove(&mut self, id: &str) {
        if let Some(doc) = self.docs.remove(id) {
            self.total_len = self.total_len.saturating_sub(doc.len as u64);
            for term in doc.terms.keys() {
                if let Some(p) = self.postings.get_mut(term) {
                    p.remove(id);
                    if p.is_empty() {
                        self.postings.remove(term);
                    }
                }
            }
        }
    }

    /// Top `k` documents for `query`, best first, only scores > 0. Ties are
    /// broken by id for determinism.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        if k == 0 || self.docs.is_empty() {
            return Vec::new();
        }
        let n = self.docs.len() as f32;
        let avgdl = (self.total_len as f32 / n).max(1.0);
        let terms: BTreeSet<String> = content_tokens(query).into_iter().collect();
        let mut scores: HashMap<&str, f32> = HashMap::new();
        for term in &terms {
            let Some(postings) = self.postings.get(term) else {
                continue;
            };
            let df = postings.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (doc_id, tf) in postings {
                let dl = self.docs.get(doc_id).map(|d| d.len).unwrap_or(0) as f32;
                let tf = *tf as f32;
                let norm = tf + Self::K1 * (1.0 - Self::B + Self::B * dl / avgdl);
                *scores.entry(doc_id.as_str()).or_insert(0.0) += idf * tf * (Self::K1 + 1.0) / norm;
            }
        }
        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0 && s.is_finite())
            .map(|(id, s)| (id.to_string(), s))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k);
        ranked
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// True when nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basics() {
        assert_eq!(
            normalize("  Hello,   World!  Don’t “quote” me…  "),
            "hello world don't quote me"
        );
        assert_eq!(normalize("trade-off: 40% (p99)"), "trade off 40 p99");
        assert_eq!(normalize("'single quoted'"), "single quoted");
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("?!..."), "");
    }

    #[test]
    fn stemming_unifies_inflections() {
        for w in ["scaling", "scaled", "scales", "scale"] {
            assert_eq!(stem(w), "scal", "{w}");
        }
        for w in ["caching", "cached", "caches", "cache"] {
            assert_eq!(stem(w), "cach", "{w}");
        }
        for w in ["migrated", "migration", "migrating", "migrate", "migrations"] {
            assert_eq!(stem(w), "migr", "{w}");
        }
        for w in ["latency", "latencies"] {
            assert_eq!(stem(w), "latenci", "{w}");
        }
        for w in ["queue", "queues", "queuing", "queueing"] {
            assert_eq!(stem(w), "queu", "{w}");
        }
        for w in ["deployment", "deployed", "deploys", "deploying"] {
            assert_eq!(stem(w), "deploy", "{w}");
        }
        for w in ["databases", "database"] {
            assert_eq!(stem(w), "databas", "{w}");
        }
        for w in ["scalability", "scalable"] {
            assert_eq!(stem(w), "scal", "{w}");
        }
        for w in ["consistency", "consistent"] {
            assert_eq!(stem(w), "consist", "{w}");
        }
        for w in ["authentication", "authenticate", "authenticated"] {
            assert_eq!(stem(w), "authentic", "{w}");
        }
        for w in ["optimization", "optimize", "optimized", "optimizing"] {
            assert_eq!(stem(w), "optim", "{w}");
        }
        assert_eq!(stem("planned"), "plan");
        assert_eq!(stem("logging"), "log");
        assert_eq!(stem("process"), "process");
        assert_eq!(stem("status"), "status");
        assert_eq!(stem("api"), "api");
        assert_eq!(stem("s3"), "s3");
        assert_eq!(stem("k8s"), "k8s");
    }

    #[test]
    fn content_tokens_drop_scaffolding_and_unify_synonyms() {
        assert_eq!(content_tokens("How did you scale the API?"), vec!["scal", "api"]);
        assert_eq!(content_tokens("How did you handle API scaling?"), vec!["api", "scal"]);
        assert_eq!(
            content_tokens("Tell me about the DB you used"),
            vec!["databas"]
        );
        assert_eq!(content_tokens("What's the trade-off?"), vec!["tradeoff"]);
        assert!(content_tokens("so um what do you think about that").is_empty());
    }

    #[test]
    fn paraphrases_are_similar_and_different_questions_are_not() {
        let base = "How did you scale the API?";
        let para = lexical_similarity(base, "How did you handle API scaling?");
        assert!(para >= 0.6, "paraphrase {para}");
        let different = lexical_similarity(base, "Which component became the scaling bottleneck?");
        assert!(different < 0.4, "different {different}");
        assert!(para - different > 0.4);

        let paraphrases = [
            (
                "What database did you use for the orders service?",
                "Which DB backs the order service?",
            ),
            (
                "Tell me about a time you disagreed with a coworker.",
                "Describe a conflict you had with a colleague.",
            ),
            (
                "What happens if Redis goes down?",
                "How do you handle Redis failures?",
            ),
            (
                "Why did you choose Kafka?",
                "What made you pick Kafka?",
            ),
            (
                "How do you monitor the pipeline?",
                "What observability do you have for the pipeline?",
            ),
            (
                "Walk me through the architecture of the payment system.",
                "Can you explain how the payment system is designed?",
            ),
        ];
        for (a, b) in paraphrases {
            let s = lexical_similarity(a, b);
            assert!(s >= 0.6, "{a} | {b} -> {s}");
        }

        let unrelated = [
            (
                "How would you design a rate limiter?",
                "Tell me about a time you disagreed with your manager.",
            ),
            (
                "What is your experience with Kubernetes?",
                "How do you prioritize features on the roadmap?",
            ),
            (
                "How do you test asynchronous code?",
                "Where do you see yourself in five years?",
            ),
        ];
        for (a, b) in unrelated {
            let s = lexical_similarity(a, b);
            assert!(s <= 0.25, "{a} | {b} -> {s}");
        }
        assert_eq!(lexical_similarity("", ""), 0.0);
        assert_eq!(lexical_similarity("Why?", "why"), 1.0);
    }

    #[test]
    fn jaccard_and_cosine() {
        let a: Vec<String> = vec!["a".into(), "b".into()];
        let b: Vec<String> = vec!["b".into(), "c".into()];
        assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(jaccard(&[], &[]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine(&[f32::NAN, 1.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn token_estimates_and_truncation() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
        let text = "The quick brown fox jumps over the lazy dog and keeps running far away";
        let t = truncate_to_tokens(text, 5);
        assert!(t.ends_with('…'), "{t}");
        assert!(estimate_tokens(&t) <= 5, "{t}");
        assert!(text.starts_with(t.trim_end_matches('…')));
        assert!(!t.trim_end_matches('…').ends_with(' '));
        assert_eq!(truncate_to_tokens("short", 10), "short");
        assert_eq!(truncate_to_tokens(text, 0), "");

        let tail = tail_to_tokens(text, 5);
        assert!(tail.starts_with('…'), "{tail}");
        assert!(estimate_tokens(&tail) <= 5, "{tail}");
        assert!(text.ends_with(tail.trim_start_matches('…')));
        assert_eq!(tail_to_tokens("short", 10), "short");
    }

    #[test]
    fn words_and_sentences() {
        assert_eq!(word_count("  Hello there,  general - Kenobi! "), 3);
        assert_eq!(word_count(""), 0);
        assert_eq!(
            split_sentences("We use Redis. How do you scale it? Great!"),
            vec!["We use Redis.", "How do you scale it?", "Great!"]
        );
        assert_eq!(
            split_sentences("Latency went from 3.5 s to 1.2 s, e.g. on bit.ly links. Next"),
            vec!["Latency went from 3.5 s to 1.2 s, e.g. on bit.ly links.", "Next"]
        );
        assert_eq!(
            split_sentences("so how would you design it"),
            vec!["so how would you design it"]
        );
        assert_eq!(
            split_sentences("Really?! Yes...\nNew line"),
            vec!["Really?!", "Yes...", "New line"]
        );
        assert_eq!(split_sentences("Talk to Dr. Smith about it."), vec!["Talk to Dr. Smith about it."]);
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn hashing_embedder_is_deterministic_and_normalized() {
        let e = HashingEmbedder::default();
        let a = e.embed_one("How did you scale the API?");
        let b = e.embed_one("How did you scale the API?");
        assert_eq!(a, b);
        assert_eq!(a.len(), 384);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        let para = e.embed_one("How did you handle API scaling?");
        let other = e.embed_one("Tell me about your favourite holiday destination.");
        assert!(cosine(&a, &para) > cosine(&a, &other) + 0.3);
        assert!(e.embed_one("").iter().all(|x| *x == 0.0));
        assert_eq!(HashingEmbedder::new(1).dims(), 8);
    }

    #[tokio::test]
    async fn hashing_embedder_provider() {
        let e = HashingEmbedder::default();
        let info = e.info();
        assert_eq!(info.id, "local_hash");
        assert_eq!(info.display_name, "Built-in hashing embeddings");
        assert_eq!(info.model, "hash-384");
        assert_eq!(info.location, ProviderLocation::Local);
        let out = e
            .embed(vec!["a b c".into(), "d".into()], CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert_eq!(
            e.embed(vec!["x".into()], cancel).await,
            Err(ProviderError::Cancelled)
        );
    }

    #[test]
    fn bm25_ranks_and_replaces() {
        let mut idx = Bm25Index::new();
        assert!(idx.is_empty());
        idx.add("s1", "We store rate limit counters in Redis with a TTL.");
        idx.add("s2", "The frontend is written in React.");
        idx.add("s3", "Redis replication lag caused stale counters during failover.");
        assert_eq!(idx.len(), 3);
        let hits = idx.search("How are the Redis counters stored?", 5);
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|(_, s)| *s > 0.0));
        assert!(hits[0].0 == "s1" || hits[0].0 == "s3");
        assert!(!hits.iter().any(|(id, _)| id == "s2"));
        // Best first.
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
        // Re-adding replaces.
        idx.add("s2", "Redis counters everywhere");
        assert_eq!(idx.len(), 3);
        assert!(idx.search("Redis", 5).iter().any(|(id, _)| id == "s2"));
        idx.remove("s2");
        idx.remove("missing");
        assert_eq!(idx.len(), 2);
        assert!(idx.search("React frontend", 5).is_empty());
        assert!(idx.search("Redis", 0).is_empty());
        assert_eq!(idx.search("Redis", 1).len(), 1);
    }
}
