//! Meeting memory for retrieval (spec §18, §19, §52).
//!
//! Every finalized transcript segment, question, answer, screen observation
//! and claim becomes a [`MemoryItem`]. Retrieval is hybrid: BM25 keyword
//! scores plus cosine similarity of embeddings (when vectors from the active
//! embedding model exist), with a small recency bonus and a boost for screen
//! observations when the question refers to the screen.

use std::collections::HashMap;

use openmeet_intel::prompts::ContextSnippet;
use openmeet_intel::text::{Bm25Index, cosine};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Segment,
    Question,
    Answer,
    Observation,
    Claim,
}

impl MemoryKind {
    /// Snippet kind used in prompts.
    pub fn snippet_kind(self) -> &'static str {
        match self {
            MemoryKind::Segment => "earlier_discussion",
            MemoryKind::Question => "previous_question",
            MemoryKind::Answer => "previous_answer",
            MemoryKind::Observation => "screen",
            MemoryKind::Claim => "claim",
        }
    }

    /// Kind stored with persisted embeddings.
    pub fn store_kind(self) -> &'static str {
        match self {
            MemoryKind::Segment => "segment",
            MemoryKind::Question => "question",
            MemoryKind::Answer => "answer",
            MemoryKind::Observation => "observation",
            MemoryKind::Claim => "claim",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryItem {
    pub id: String,
    pub kind: MemoryKind,
    /// Display text (speaker-labelled for transcript segments).
    pub text: String,
    /// Milliseconds since meeting start.
    pub at_ms: i64,
}

#[derive(Default)]
pub struct ContextMemory {
    items: HashMap<String, MemoryItem>,
    bm25: Bm25Index,
    vectors: HashMap<String, Vec<f32>>,
    /// Embedding model the stored vectors belong to.
    embedding_model: Option<String>,
    /// Compressed summary of conversation older than the raw window.
    pub long_term_summary: String,
    /// Meeting-relative ms covered by `long_term_summary`.
    pub summarized_until_ms: i64,
}

/// Retrieval parameters.
#[derive(Debug, Clone, Copy)]
pub struct RetrievalQuery<'a> {
    pub text: &'a str,
    /// (model, vector) of the query embedding, if available.
    pub vector: Option<(&'a str, &'a [f32])>,
    pub k: usize,
    /// Items at or after this meeting-relative time are excluded (they are
    /// already present verbatim in the recent transcript window).
    pub exclude_from_ms: Option<i64>,
    pub now_ms: i64,
    pub boost_screen: bool,
}

impl ContextMemory {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn add(&mut self, item: MemoryItem) {
        self.bm25.add(&item.id, &item.text);
        self.items.insert(item.id.clone(), item);
    }

    /// Stores an embedding. Vectors from a different model invalidate the
    /// existing ones (cosine across models is meaningless).
    pub fn set_vector(&mut self, id: &str, model: &str, vector: Vec<f32>) {
        if self.embedding_model.as_deref() != Some(model) {
            self.vectors.clear();
            self.embedding_model = Some(model.to_string());
        }
        if self.items.contains_key(id) && !vector.is_empty() {
            self.vectors.insert(id.to_string(), vector);
        }
    }

    #[cfg(test)]
    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    #[cfg(test)]
    pub fn embedding_model(&self) -> Option<&str> {
        self.embedding_model.as_deref()
    }

    /// Ranked snippets, returned in chronological order.
    pub fn retrieve(&self, q: RetrievalQuery<'_>) -> Vec<ContextSnippet> {
        if self.items.is_empty() || q.k == 0 {
            return Vec::new();
        }
        let eligible = |item: &MemoryItem| match q.exclude_from_ms {
            Some(t) => item.at_ms < t,
            None => true,
        };

        let mut scores: HashMap<&str, (f32, f32)> = HashMap::new();

        // Keyword channel.
        let bm = self.bm25.search(q.text, 24);
        let bm_max = bm.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        if bm_max > 0.0 {
            for (id, s) in &bm {
                if let Some((key, _)) = self.items.get_key_value(id.as_str()) {
                    scores.entry(key.as_str()).or_default().0 = s / bm_max;
                }
            }
        }

        // Semantic channel.
        let mut semantic = false;
        if let Some((model, qv)) = q.vector {
            if self.embedding_model.as_deref() == Some(model) && !self.vectors.is_empty() {
                semantic = true;
                let mut sims: Vec<(&str, f32)> = self
                    .vectors
                    .iter()
                    .map(|(id, v)| (id.as_str(), cosine(qv, v).max(0.0)))
                    .collect();
                sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (id, s) in sims.into_iter().take(24) {
                    scores.entry(id).or_default().1 = s;
                }
            }
        }

        let mut ranked: Vec<(&MemoryItem, f32)> = scores
            .into_iter()
            .filter_map(|(id, (kw, sem))| {
                let item = self.items.get(id)?;
                if !eligible(item) {
                    return None;
                }
                let base = if semantic { 0.55 * sem + 0.45 * kw } else { kw };
                let age = (q.now_ms - item.at_ms).max(0) as f32;
                let recency = 0.08 * (-age / 600_000.0).exp();
                let kind_boost = match item.kind {
                    MemoryKind::Question | MemoryKind::Answer => 0.05,
                    MemoryKind::Observation if q.boost_screen => 0.25,
                    _ => 0.0,
                };
                let score = base + recency + kind_boost;
                (base >= 0.12).then_some((item, score))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(q.k);
        ranked.sort_by_key(|(item, _)| item.at_ms);
        ranked
            .into_iter()
            .map(|(item, _)| ContextSnippet {
                kind: item.kind.snippet_kind().to_string(),
                text: item.text.clone(),
                at_ms: Some(item.at_ms),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_intel::text::HashingEmbedder;

    fn item(id: &str, kind: MemoryKind, text: &str, at: i64) -> MemoryItem {
        MemoryItem {
            id: id.into(),
            kind,
            text: text.into(),
            at_ms: at,
        }
    }

    fn query<'a>(text: &'a str, vector: Option<(&'a str, &'a [f32])>) -> RetrievalQuery<'a> {
        RetrievalQuery {
            text,
            vector,
            k: 3,
            exclude_from_ms: None,
            now_ms: 600_000,
            boost_screen: false,
        }
    }

    #[test]
    fn keyword_retrieval_finds_relevant_discussion() {
        let mut m = ContextMemory::new();
        m.add(item(
            "1",
            MemoryKind::Segment,
            "Candidate: we used Kafka for event processing between services",
            10_000,
        ));
        m.add(item(
            "2",
            MemoryKind::Segment,
            "Interviewer: tell me about your hobbies",
            20_000,
        ));
        m.add(item(
            "3",
            MemoryKind::Segment,
            "Candidate: the database was PostgreSQL with read replicas",
            30_000,
        ));
        let r = m.retrieve(query("Why did you choose Kafka?", None));
        assert!(!r.is_empty());
        assert!(r[0].text.contains("Kafka"));
        assert!(r.iter().all(|s| !s.text.contains("hobbies")));
    }

    #[test]
    fn semantic_channel_and_model_switch() {
        let emb = HashingEmbedder::new(HashingEmbedder::DEFAULT_DIMS);
        let mut m = ContextMemory::new();
        let texts = [
            ("a", "we scaled the API horizontally behind a load balancer"),
            ("b", "my favourite food is pasta"),
        ];
        for (i, (id, t)) in texts.iter().enumerate() {
            m.add(item(id, MemoryKind::Segment, t, i as i64 * 1000));
            m.set_vector(id, "hash", emb.embed_one(t));
        }
        assert_eq!(m.vector_count(), 2);
        let qv = emb.embed_one("how did you scale the api");
        let r = m.retrieve(query("how did you scale the api", Some(("hash", &qv))));
        assert_eq!(r.len(), 1);
        assert!(r[0].text.contains("scaled"));
        // Switching model clears incompatible vectors.
        m.set_vector("a", "other-model", vec![1.0, 0.0]);
        assert_eq!(m.vector_count(), 1);
        assert_eq!(m.embedding_model(), Some("other-model"));
    }

    #[test]
    fn excludes_recent_window_and_orders_chronologically() {
        let mut m = ContextMemory::new();
        m.add(item(
            "old",
            MemoryKind::Question,
            "How did you shard the database?",
            1_000,
        ));
        m.add(item(
            "mid",
            MemoryKind::Answer,
            "We sharded the database by customer id",
            2_000,
        ));
        m.add(item(
            "new",
            MemoryKind::Segment,
            "database sharding came up again just now",
            590_000,
        ));
        let mut q = query("database sharding", None);
        q.exclude_from_ms = Some(500_000);
        let r = m.retrieve(q);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].kind, "previous_question");
        assert_eq!(r[1].kind, "previous_answer");
    }

    #[test]
    fn screen_boost() {
        let mut m = ContextMemory::new();
        m.add(item(
            "o",
            MemoryKind::Observation,
            "System diagram: API -> queue -> workers -> database",
            1_000,
        ));
        m.add(item(
            "s",
            MemoryKind::Segment,
            "the queue workers process jobs",
            1_500,
        ));
        let mut q = query("explain this part of the queue", None);
        q.boost_screen = true;
        q.k = 1;
        let r = m.retrieve(q);
        assert_eq!(r[0].kind, "screen");
    }

    #[test]
    fn empty_memory() {
        let m = ContextMemory::new();
        assert!(m.retrieve(query("anything", None)).is_empty());
        assert!(m.is_empty());
    }
}
