//! Persistence contract used by the engine. The SQLite implementation lives in
//! `openmeet-store`; [`NullStore`] implements the "do not retain" policy and
//! [`MemoryStore`] backs tests.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::error::StoreError;
use crate::ids::Id;
use crate::types::*;

/// A stored embedding vector for retrieval across the meeting.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingRecord {
    /// Id of the embedded item (segment, question, observation, ...).
    pub item_id: Id,
    pub meeting_id: Id,
    /// "segment" | "question" | "answer" | "observation" | "claim" | "summary"
    pub kind: String,
    pub model: String,
    pub vector: Vec<f32>,
    pub created_at: i64,
}

/// Writes are upserts keyed by id. Implementations must be fast (they are
/// called from a background persistence worker, never from the audio thread).
pub trait MeetingStore: Send + Sync {
    fn save_meeting(&self, meeting: &Meeting) -> Result<(), StoreError>;
    fn save_segment(&self, segment: &TranscriptSegment) -> Result<(), StoreError>;
    fn save_question(&self, question: &DetectedQuestion) -> Result<(), StoreError>;
    fn save_answer(&self, answer: &AnswerRecord) -> Result<(), StoreError>;
    fn save_suggestion(&self, suggestion: &Suggestion) -> Result<(), StoreError>;
    fn set_suggestion_pinned(&self, suggestion_id: &str, pinned: bool) -> Result<(), StoreError>;
    /// Replaces the full topic list of a meeting.
    fn save_topics(&self, meeting_id: &str, topics: &[Topic]) -> Result<(), StoreError>;
    fn save_claims(&self, claims: &[Claim]) -> Result<(), StoreError>;
    fn save_observation(&self, observation: &ScreenObservation) -> Result<(), StoreError>;
    fn save_action_items(&self, items: &[ActionItem]) -> Result<(), StoreError>;
    fn save_summary(&self, summary: &MeetingSummary) -> Result<(), StoreError>;
    fn save_embedding(&self, embedding: &EmbeddingRecord) -> Result<(), StoreError>;
    fn load_meeting(&self, meeting_id: &str) -> Result<MeetingRecord, StoreError>;
}

/// Retention policy "do not retain": nothing is written.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullStore;

impl MeetingStore for NullStore {
    fn save_meeting(&self, _: &Meeting) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_segment(&self, _: &TranscriptSegment) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_question(&self, _: &DetectedQuestion) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_answer(&self, _: &AnswerRecord) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_suggestion(&self, _: &Suggestion) -> Result<(), StoreError> {
        Ok(())
    }
    fn set_suggestion_pinned(&self, _: &str, _: bool) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_topics(&self, _: &str, _: &[Topic]) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_claims(&self, _: &[Claim]) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_observation(&self, _: &ScreenObservation) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_action_items(&self, _: &[ActionItem]) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_summary(&self, _: &MeetingSummary) -> Result<(), StoreError> {
        Ok(())
    }
    fn save_embedding(&self, _: &EmbeddingRecord) -> Result<(), StoreError> {
        Ok(())
    }
    fn load_meeting(&self, meeting_id: &str) -> Result<MeetingRecord, StoreError> {
        Err(StoreError::NotFound(meeting_id.to_string()))
    }
}

/// In-memory store for tests and benchmarks.
#[derive(Debug, Default)]
pub struct MemoryStore {
    records: Mutex<HashMap<Id, MeetingRecord>>,
    embeddings: Mutex<Vec<EmbeddingRecord>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn embedding_count(&self) -> usize {
        self.embeddings.lock().len()
    }

    fn with_record<F: FnOnce(&mut MeetingRecord)>(
        &self,
        meeting_id: &str,
        f: F,
    ) -> Result<(), StoreError> {
        let mut records = self.records.lock();
        let rec = records
            .get_mut(meeting_id)
            .ok_or_else(|| StoreError::NotFound(meeting_id.to_string()))?;
        f(rec);
        Ok(())
    }
}

fn upsert<T: Clone, F: Fn(&T) -> &str>(list: &mut Vec<T>, item: &T, key: F) {
    if let Some(existing) = list.iter_mut().find(|x| key(x) == key(item)) {
        *existing = item.clone();
    } else {
        list.push(item.clone());
    }
}

impl MeetingStore for MemoryStore {
    fn save_meeting(&self, meeting: &Meeting) -> Result<(), StoreError> {
        let mut records = self.records.lock();
        records
            .entry(meeting.id.clone())
            .and_modify(|r| r.meeting = meeting.clone())
            .or_insert_with(|| MeetingRecord {
                meeting: meeting.clone(),
                segments: Vec::new(),
                questions: Vec::new(),
                answers: Vec::new(),
                suggestions: Vec::new(),
                topics: Vec::new(),
                claims: Vec::new(),
                observations: Vec::new(),
                action_items: Vec::new(),
                summary: None,
            });
        Ok(())
    }

    fn save_segment(&self, segment: &TranscriptSegment) -> Result<(), StoreError> {
        self.with_record(&segment.meeting_id, |r| {
            upsert(&mut r.segments, segment, |s| &s.id)
        })
    }

    fn save_question(&self, question: &DetectedQuestion) -> Result<(), StoreError> {
        self.with_record(&question.meeting_id, |r| {
            upsert(&mut r.questions, question, |q| &q.id)
        })
    }

    fn save_answer(&self, answer: &AnswerRecord) -> Result<(), StoreError> {
        self.with_record(&answer.meeting_id, |r| {
            upsert(&mut r.answers, answer, |a| &a.id)
        })
    }

    fn save_suggestion(&self, suggestion: &Suggestion) -> Result<(), StoreError> {
        let Some(mid) = suggestion.meeting_id.clone() else {
            return Ok(());
        };
        self.with_record(&mid, |r| upsert(&mut r.suggestions, suggestion, |s| &s.id))
    }

    fn set_suggestion_pinned(&self, suggestion_id: &str, pinned: bool) -> Result<(), StoreError> {
        let mut records = self.records.lock();
        for r in records.values_mut() {
            if let Some(s) = r.suggestions.iter_mut().find(|s| s.id == suggestion_id) {
                s.pinned = pinned;
                return Ok(());
            }
        }
        Err(StoreError::NotFound(suggestion_id.to_string()))
    }

    fn save_topics(&self, meeting_id: &str, topics: &[Topic]) -> Result<(), StoreError> {
        self.with_record(meeting_id, |r| r.topics = topics.to_vec())
    }

    fn save_claims(&self, claims: &[Claim]) -> Result<(), StoreError> {
        for c in claims {
            self.with_record(&c.meeting_id, |r| upsert(&mut r.claims, c, |x| &x.id))?;
        }
        Ok(())
    }

    fn save_observation(&self, observation: &ScreenObservation) -> Result<(), StoreError> {
        let Some(mid) = observation.meeting_id.clone() else {
            return Ok(());
        };
        self.with_record(&mid, |r| {
            upsert(&mut r.observations, observation, |o| &o.id)
        })
    }

    fn save_action_items(&self, items: &[ActionItem]) -> Result<(), StoreError> {
        for i in items {
            self.with_record(&i.meeting_id, |r| upsert(&mut r.action_items, i, |x| &x.id))?;
        }
        Ok(())
    }

    fn save_summary(&self, summary: &MeetingSummary) -> Result<(), StoreError> {
        self.with_record(&summary.meeting_id, |r| r.summary = Some(summary.clone()))
    }

    fn save_embedding(&self, embedding: &EmbeddingRecord) -> Result<(), StoreError> {
        self.embeddings.lock().push(embedding.clone());
        Ok(())
    }

    fn load_meeting(&self, meeting_id: &str) -> Result<MeetingRecord, StoreError> {
        self.records
            .lock()
            .get(meeting_id)
            .cloned()
            .ok_or_else(|| StoreError::NotFound(meeting_id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meeting() -> Meeting {
        Meeting {
            id: "m1".into(),
            title: "t".into(),
            mode: Mode::Candidate,
            state: MeetingState::Active,
            started_at: 1,
            ended_at: None,
            participants: vec![],
            persisted: true,
        }
    }

    #[test]
    fn memory_store_roundtrip() {
        let s = MemoryStore::new();
        s.save_meeting(&meeting()).unwrap();
        let seg = TranscriptSegment {
            id: "s1".into(),
            meeting_id: "m1".into(),
            utterance_id: "u1".into(),
            speaker_id: None,
            speaker_role: SpeakerRole::Remote,
            source: AudioSourceKind::System,
            start_time: 0,
            end_time: 10,
            text: "hello".into(),
            is_final: true,
            confidence: None,
        };
        s.save_segment(&seg).unwrap();
        let mut seg2 = seg.clone();
        seg2.text = "hello there".into();
        s.save_segment(&seg2).unwrap();
        let rec = s.load_meeting("m1").unwrap();
        assert_eq!(rec.segments.len(), 1);
        assert_eq!(rec.segments[0].text, "hello there");
        assert!(s.load_meeting("nope").is_err());
    }

    #[test]
    fn null_store_is_noop() {
        let s = NullStore;
        s.save_meeting(&meeting()).unwrap();
        assert!(matches!(s.load_meeting("m1"), Err(StoreError::NotFound(_))));
    }
}
