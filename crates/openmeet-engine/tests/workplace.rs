//! Workplace mode end to end with deterministic mock providers: the live
//! board (decisions, action items, open questions, clarifying questions),
//! answers to questions put to the user, "Catch me up", minutes at the end,
//! export, and the rule that answer-integrity never runs here.

use std::sync::Arc;
use std::time::Duration;

use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ipc::{ExportFormat, ExportParts, SimulationAction, StartMeetingRequest};
use openmeet_core::providers::ProviderSet;
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::{MeetingStore, MemoryStore};
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps};
use openmeet_intel::export::render;
use openmeet_intel::mock::MockLlmProvider;
use parking_lot::Mutex;
use tokio::sync::Notify;

#[derive(Default)]
struct TestSink {
    events: Mutex<Vec<MeetingEvent>>,
    notify: Notify,
}

impl EventSink for TestSink {
    fn emit(&self, event: MeetingEvent) {
        self.events.lock().push(event);
        self.notify.notify_waiters();
    }
}

impl TestSink {
    fn count(&self, name: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| e.name() == name)
            .count()
    }

    async fn wait_for(
        &self,
        timeout: Duration,
        pred: impl Fn(&MeetingEvent) -> bool,
    ) -> Option<MeetingEvent> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.notify.notified();
            if let Some(e) = self.events.lock().iter().find(|e| pred(e)).cloned() {
                return Some(e);
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return self.events.lock().iter().find(|e| pred(e)).cloned();
            }
        }
    }
}

struct Harness {
    engine: Engine,
    sink: Arc<TestSink>,
    store: Arc<MemoryStore>,
    llm: MockLlmProvider,
}

fn harness() -> Harness {
    let sink = Arc::new(TestSink::default());
    let store = Arc::new(MemoryStore::new());
    let llm = MockLlmProvider::new();
    llm.set_chunk_delay_ms(1);
    let mut settings = Settings::default();
    settings.assistant.semantic_detection = false;
    settings.assistant.turn_end_silence_ms = 400;
    settings.privacy.retention = RetentionPolicy::SevenDays;
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: store.clone(),
            providers: ProviderSet {
                llm: Some(Arc::new(llm.clone())),
                ..Default::default()
            },
            settings,
        },
        tokio::runtime::Handle::current(),
    );
    Harness {
        engine,
        sink,
        store,
        llm,
    }
}

fn start_request() -> StartMeetingRequest {
    StartMeetingRequest {
        mode: Mode::Workplace,
        title: Some("Sprint planning".into()),
        microphone: false,
        system_audio: true,
        screen: None,
        focus_app_id: None,
    }
}

const WAIT: Duration = Duration::from_secs(10);

fn transcript(h: &Harness, speaker: SpeakerRole, text: &str) {
    h.engine
        .simulate(SimulationAction::Transcript {
            speaker,
            text: text.into(),
        })
        .unwrap();
}

async fn wait_for_segment(h: &Harness, needle: &str) {
    let needle = needle.to_string();
    h.sink
        .wait_for(WAIT, move |e| {
            matches!(e, MeetingEvent::TranscriptFinal { segment } if segment.text.contains(&needle))
        })
        .await
        .expect("segment finalized");
}

fn board(h: &Harness) -> WorkplaceBoard {
    h.engine
        .snapshot()
        .workplace
        .expect("workplace board in the snapshot")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn workplace_meeting_tracks_the_board_and_answers_colleagues() {
    let h = harness();
    h.engine.start_meeting(start_request()).await.unwrap();

    // A decision and a named action item with a due date.
    transcript(
        &h,
        SpeakerRole::Remote,
        "We decided to move billing to Kafka.",
    );
    wait_for_segment(&h, "move billing").await;
    transcript(
        &h,
        SpeakerRole::Remote,
        "Priya will update the runbook by Friday.",
    );
    wait_for_segment(&h, "runbook").await;
    let b = board(&h);
    assert_eq!(b.decisions.len(), 1, "{b:?}");
    let item = b
        .action_items
        .iter()
        .find(|a| a.text.to_lowercase().contains("runbook"))
        .expect("action item tracked");
    assert_eq!(item.owner.as_deref(), Some("Priya"));
    assert_eq!(item.due.as_deref(), Some("by Friday"));

    // A vague action item gets a clarifying question.
    transcript(
        &h,
        SpeakerRole::Remote,
        "Someone should look into the alerts.",
    );
    wait_for_segment(&h, "alerts").await;
    let b = board(&h);
    assert!(
        b.clarifications
            .iter()
            .any(|c| c.target_text.to_lowercase().contains("alerts")),
        "clarifying question for the vague item: {:?}",
        b.clarifications
    );

    // A question put to the user gets a grounded suggestion for a colleague.
    h.engine
        .simulate(SimulationAction::Question {
            text: "Can you share the rollout plan by Monday?".into(),
        })
        .unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::AnswerComplete { suggestion, .. }
                if suggestion.kind == SuggestionKind::Answer)
        })
        .await
        .expect("answer suggestion for the question to the user");
    assert!(matches!(ev, MeetingEvent::AnswerComplete { .. }));
    let b = board(&h);
    let q = b
        .open_questions
        .iter()
        .find(|q| q.question.contains("rollout plan"))
        .expect("question on the board");
    assert!(!q.answered && q.addressed_to_you);

    // The user answers it later: the question is resolved on the board.
    transcript(
        &h,
        SpeakerRole::User,
        "Yes, I will share the rollout plan on Monday.",
    );
    wait_for_segment(&h, "rollout plan on Monday").await;
    let b = board(&h);
    assert!(
        b.open_questions
            .iter()
            .any(|q| q.question.contains("rollout plan") && q.answered),
        "answered question: {:?}",
        b.open_questions
    );

    // Catch me up summarizes the recent window, grounded in the board.
    h.engine.catch_up(5).unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::AnswerComplete { suggestion, .. }
                if suggestion.kind == SuggestionKind::CatchUp)
        })
        .await
        .expect("catch-up summary");
    let MeetingEvent::AnswerComplete { suggestion, .. } = ev else {
        unreachable!()
    };
    assert!(
        suggestion.content.contains("decision"),
        "{}",
        suggestion.content
    );
    let catch_up_req = h
        .llm
        .requests()
        .into_iter()
        .find(|r| r.kind == RequestKind::CatchUp)
        .expect("catch-up request");
    assert!(
        catch_up_req.messages[0]
            .content
            .contains("move billing to Kafka"),
        "catch-up prompt carries the board"
    );

    // Integrity never runs in workplace mode, and nothing is interviewer-only.
    assert_eq!(h.sink.count("INTEGRITY_CHECKED"), 0);
    assert!(h.llm.requests().iter().all(|r| {
        !matches!(
            r.kind,
            RequestKind::IntegrityCheck | RequestKind::IntegrityReference
        )
    }));
    assert!(
        h.engine.generate_followups().is_err(),
        "no interview follow-ups"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stopping_a_workplace_meeting_stores_and_exports_minutes() {
    let h = harness();
    let meeting = h.engine.start_meeting(start_request()).await.unwrap();
    transcript(&h, SpeakerRole::Remote, "We decided to freeze the schema.");
    wait_for_segment(&h, "freeze the schema").await;
    transcript(
        &h,
        SpeakerRole::Remote,
        "Sam will send the migration notes by Thursday.",
    );
    wait_for_segment(&h, "migration notes").await;
    h.engine.stop_meeting().await.unwrap();

    let ev = h
        .sink
        .wait_for(Duration::from_secs(20), |e| {
            matches!(e, MeetingEvent::SummaryReady { .. })
        })
        .await
        .expect("minutes generated");
    let MeetingEvent::SummaryReady { summary } = ev else {
        unreachable!()
    };
    assert_eq!(summary.meeting_id, meeting.id);
    let minutes = summary.workplace.as_ref().expect("workplace minutes");
    assert!(
        minutes
            .action_items
            .iter()
            .any(|a| a.owner.as_deref() == Some("Sam") && a.due.as_deref() == Some("by Thursday")),
        "{minutes:?}"
    );
    assert!(!minutes.follow_up_message.trim().is_empty());
    assert!(summary.markdown.contains("Sprint planning"));
    assert!(summary.interviewer.is_none(), "no interview notes");
    assert_eq!(h.sink.count("INTEGRITY_CHECKED"), 0);

    h.engine.flush_persistence().await;
    let record = h.store.load_meeting(&meeting.id).expect("persisted");
    assert_eq!(record.meeting.mode, Mode::Workplace);
    assert_eq!(
        record.summary.as_ref().and_then(|s| s.workplace.clone()),
        summary.workplace,
        "minutes survive the store"
    );
    let md = render(&record, ExportFormat::Markdown, &ExportParts::default());
    assert!(md.contains("Workplace mode"), "{md}");
    assert!(md.contains("Sam"), "minutes exported: {md}");
}
