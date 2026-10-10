//! Background persistence worker.
//!
//! The engine never calls the store from the audio / streaming hot paths.
//! Writes are queued and applied on a dedicated thread. Each command carries
//! the store chosen for its meeting, so switching the retention policy for a
//! new meeting can never drop writes of the previous one.

use std::sync::Arc;
use std::time::{Duration, Instant};

use openmeet_core::error::StoreError;
use openmeet_core::events::{EventSink, MeetingEvent, NoticeLevel};
use openmeet_core::store::{EmbeddingRecord, MeetingStore};
use openmeet_core::types::*;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::metrics::HealthRegistry;

pub enum PersistOp {
    Meeting(Meeting),
    Segment(TranscriptSegment),
    Question(DetectedQuestion),
    Answer(AnswerRecord),
    Suggestion(Suggestion),
    Pinned {
        id: String,
        pinned: bool,
    },
    Topics {
        meeting_id: String,
        topics: Vec<Topic>,
    },
    Claims(Vec<Claim>),
    Observation(ScreenObservation),
    ActionItems(Vec<ActionItem>),
    Summary(MeetingSummary),
    Embedding(EmbeddingRecord),
    /// Completes once every previously queued write has been applied.
    Barrier(oneshot::Sender<()>),
}

impl PersistOp {
    /// Meeting the write belongs to, when known.
    pub(crate) fn meeting_id(&self) -> Option<&str> {
        match self {
            PersistOp::Meeting(m) => Some(&m.id),
            PersistOp::Segment(s) => Some(&s.meeting_id),
            PersistOp::Question(q) => Some(&q.meeting_id),
            PersistOp::Answer(a) => Some(&a.meeting_id),
            PersistOp::Suggestion(s) => s.meeting_id.as_deref(),
            PersistOp::Topics { meeting_id, .. } => Some(meeting_id),
            PersistOp::Claims(c) => c.first().map(|c| c.meeting_id.as_str()),
            PersistOp::Observation(o) => o.meeting_id.as_deref(),
            PersistOp::ActionItems(a) => a.first().map(|a| a.meeting_id.as_str()),
            PersistOp::Summary(s) => Some(&s.meeting_id),
            PersistOp::Embedding(e) => Some(&e.meeting_id),
            PersistOp::Pinned { .. } | PersistOp::Barrier(_) => None,
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            PersistOp::Meeting(_) => "meeting",
            PersistOp::Segment(_) => "segment",
            PersistOp::Question(_) => "question",
            PersistOp::Answer(_) => "answer",
            PersistOp::Suggestion(_) => "suggestion",
            PersistOp::Pinned { .. } => "pinned",
            PersistOp::Topics { .. } => "topics",
            PersistOp::Claims(_) => "claims",
            PersistOp::Observation(_) => "observation",
            PersistOp::ActionItems(_) => "action_items",
            PersistOp::Summary(_) => "summary",
            PersistOp::Embedding(_) => "embedding",
            PersistOp::Barrier(_) => "barrier",
        }
    }
}

struct Cmd {
    store: Arc<dyn MeetingStore>,
    op: PersistOp,
}

#[derive(Clone)]
pub struct Persister {
    tx: mpsc::UnboundedSender<Cmd>,
}

impl Persister {
    pub fn spawn(sink: Arc<dyn EventSink>, health: Arc<HealthRegistry>) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Cmd>();
        let last_notice: Mutex<Option<Instant>> = Mutex::new(None);
        std::thread::Builder::new()
            .name("openmeet-persist".into())
            .spawn(move || {
                let mut healthy = true;
                while let Some(cmd) = rx.blocking_recv() {
                    let name = cmd.op.name();
                    let is_barrier = matches!(cmd.op, PersistOp::Barrier(_));
                    let result = apply(&*cmd.store, cmd.op);
                    match result {
                        Ok(()) => {
                            // A barrier is not a write: it must not clear an error.
                            if !healthy && !is_barrier {
                                healthy = true;
                                health.set(Component::Database, ComponentStatus::Active, None);
                            }
                        }
                        // The meeting was deleted (by the user or retention)
                            // while this write was queued: nothing to save.
                        Err(StoreError::NotFound(_)) => {
                            tracing::info!(op = name, "skipped a write for a meeting that no longer exists");
                        }
                        Err(e) => {
                            healthy = false;
                            tracing::error!(op = name, error = %e, "persistence failed");
                            health.set(
                                Component::Database,
                                ComponentStatus::Error,
                                Some("Saving meeting data failed".to_string()),
                            );
                            let mut last = last_notice.lock();
                            if last.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
                                *last = Some(Instant::now());
                                sink.emit(MeetingEvent::Notice {
                                    level: NoticeLevel::Error,
                                    message: "Could not save meeting data. Processing continues; check Diagnostics.".into(),
                                    component: Some(Component::Database),
                                });
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn persistence thread");
        Persister { tx }
    }

    pub fn send(&self, store: &Arc<dyn MeetingStore>, op: PersistOp) {
        let _ = self.tx.send(Cmd {
            store: store.clone(),
            op,
        });
    }

    /// Waits until all writes queued so far are applied.
    pub async fn flush(&self, store: &Arc<dyn MeetingStore>) {
        let (tx, rx) = oneshot::channel();
        self.send(store, PersistOp::Barrier(tx));
        let _ = tokio::time::timeout(Duration::from_secs(10), rx).await;
    }
}

fn apply(store: &dyn MeetingStore, op: PersistOp) -> Result<(), openmeet_core::StoreError> {
    match op {
        PersistOp::Meeting(m) => store.save_meeting(&m),
        PersistOp::Segment(s) => store.save_segment(&s),
        PersistOp::Question(q) => store.save_question(&q),
        PersistOp::Answer(a) => store.save_answer(&a),
        PersistOp::Suggestion(s) => store.save_suggestion(&s),
        PersistOp::Pinned { id, pinned } => store.set_suggestion_pinned(&id, pinned),
        PersistOp::Topics { meeting_id, topics } => store.save_topics(&meeting_id, &topics),
        PersistOp::Claims(c) => store.save_claims(&c),
        PersistOp::Observation(o) => store.save_observation(&o),
        PersistOp::ActionItems(a) => store.save_action_items(&a),
        PersistOp::Summary(s) => store.save_summary(&s),
        PersistOp::Embedding(e) => store.save_embedding(&e),
        PersistOp::Barrier(done) => {
            let _ = done.send(());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::events::CollectingSink;
    use openmeet_core::store::MemoryStore;

    #[tokio::test]
    async fn writes_are_applied_in_order() {
        let sink: Arc<dyn EventSink> = Arc::new(CollectingSink::new());
        let health = Arc::new(HealthRegistry::new(sink.clone()));
        let p = Persister::spawn(sink, health);
        let store = Arc::new(MemoryStore::new());
        let dyn_store: Arc<dyn MeetingStore> = store.clone();
        let meeting = Meeting {
            id: "m".into(),
            title: "t".into(),
            mode: Mode::Candidate,
            state: MeetingState::Active,
            started_at: 0,
            ended_at: None,
            participants: vec![],
            persisted: true,
        };
        p.send(&dyn_store, PersistOp::Meeting(meeting));
        p.send(
            &dyn_store,
            PersistOp::Topics {
                meeting_id: "m".into(),
                topics: vec![],
            },
        );
        p.flush(&dyn_store).await;
        assert!(store.load_meeting("m").is_ok());
    }

    /// A store whose disk is broken: every write fails.
    struct BrokenStore;

    fn broken<T>() -> Result<T, StoreError> {
        Err(StoreError::Database("disk I/O error".into()))
    }

    impl MeetingStore for BrokenStore {
        fn save_meeting(&self, _: &Meeting) -> Result<(), StoreError> {
            broken()
        }
        fn save_segment(&self, _: &TranscriptSegment) -> Result<(), StoreError> {
            broken()
        }
        fn save_question(&self, _: &DetectedQuestion) -> Result<(), StoreError> {
            broken()
        }
        fn save_answer(&self, _: &AnswerRecord) -> Result<(), StoreError> {
            broken()
        }
        fn save_suggestion(&self, _: &Suggestion) -> Result<(), StoreError> {
            broken()
        }
        fn set_suggestion_pinned(&self, _: &str, _: bool) -> Result<(), StoreError> {
            broken()
        }
        fn save_topics(&self, _: &str, _: &[Topic]) -> Result<(), StoreError> {
            broken()
        }
        fn save_claims(&self, _: &[Claim]) -> Result<(), StoreError> {
            broken()
        }
        fn save_observation(&self, _: &ScreenObservation) -> Result<(), StoreError> {
            broken()
        }
        fn save_action_items(&self, _: &[ActionItem]) -> Result<(), StoreError> {
            broken()
        }
        fn save_summary(&self, _: &MeetingSummary) -> Result<(), StoreError> {
            broken()
        }
        fn save_embedding(&self, _: &EmbeddingRecord) -> Result<(), StoreError> {
            broken()
        }
        fn load_meeting(&self, _: &str) -> Result<MeetingRecord, StoreError> {
            broken()
        }
    }

    fn topics(meeting_id: &str) -> PersistOp {
        PersistOp::Topics {
            meeting_id: meeting_id.into(),
            topics: vec![],
        }
    }

    #[tokio::test]
    async fn failures_report_database_health() {
        let sink = CollectingSink::new();
        let dyn_sink: Arc<dyn EventSink> = Arc::new(sink.clone());
        let health = Arc::new(HealthRegistry::new(dyn_sink.clone()));
        let p = Persister::spawn(dyn_sink, health.clone());
        let store: Arc<dyn MeetingStore> = Arc::new(BrokenStore);
        p.send(&store, topics("m"));
        p.flush(&store).await;
        assert_eq!(
            health.status(Component::Database),
            Some(ComponentStatus::Error)
        );
        assert_eq!(sink.count("NOTICE"), 1);
    }

    #[tokio::test]
    async fn writes_for_a_deleted_meeting_are_not_failures() {
        let sink = CollectingSink::new();
        let dyn_sink: Arc<dyn EventSink> = Arc::new(sink.clone());
        let health = Arc::new(HealthRegistry::new(dyn_sink.clone()));
        let p = Persister::spawn(dyn_sink, health.clone());
        let store: Arc<dyn MeetingStore> = Arc::new(MemoryStore::new());
        // The meeting row is gone (deleted while this write was queued).
        p.send(&store, topics("deleted"));
        p.flush(&store).await;
        assert_ne!(
            health.status(Component::Database),
            Some(ComponentStatus::Error)
        );
        assert_eq!(sink.count("NOTICE"), 0);
    }
}
