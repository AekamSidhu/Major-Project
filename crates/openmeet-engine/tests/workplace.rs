//! Workplace mode end to end with deterministic mock providers: the live
//! board (decisions, action items, open questions, clarifying questions),
//! answers to questions put to the user, "Catch me up", minutes at the end,
//! export, and the rule that answer-integrity never runs here.

use std::sync::Arc;
use std::time::Duration;

use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::integrity::{IntegrityLevel, IntegrityService};
use openmeet_core::ipc::{ExportFormat, ExportParts, SimulationAction, StartMeetingRequest};
use openmeet_core::providers::{LlmProvider, ProviderSet};
use openmeet_core::settings::{RetentionPolicy, Settings};
use openmeet_core::store::{MeetingStore, MemoryStore};
use openmeet_core::types::*;
use openmeet_engine::{Engine, EngineDeps};
use openmeet_intel::export::render;
use openmeet_intel::integrity::spec_thresholds;
use openmeet_intel::mock::{MockIntegrityService, MockLlmProvider};
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

/// How a harness is wired: with or without an AI provider, whether the LLM
/// confirms ambiguous questions, and the heuristic detection threshold.
#[derive(Clone, Copy)]
struct Setup {
    model: bool,
    semantic: bool,
    threshold: f32,
}

const WITH_MODEL: Setup = Setup {
    model: true,
    semantic: false,
    threshold: 0.6,
};

fn harness() -> Harness {
    harness_with(WITH_MODEL)
}

fn harness_with(setup: Setup) -> Harness {
    let sink = Arc::new(TestSink::default());
    let store = Arc::new(MemoryStore::new());
    let llm = MockLlmProvider::new();
    llm.set_chunk_delay_ms(1);
    let mut settings = Settings::default();
    settings.assistant.semantic_detection = setup.semantic;
    settings.assistant.detection_threshold = setup.threshold;
    settings.assistant.turn_end_silence_ms = 400;
    settings.privacy.retention = RetentionPolicy::SevenDays;
    // Integrity is ON with every signal, and a mock service is wired in, so
    // the check would run if workplace mode were not gated out of it.
    settings.integrity.enabled = true;
    settings.integrity.ai_text = true;
    settings.integrity.echo_similarity = true;
    settings.integrity.delivery = true;
    settings.integrity.reading = true;
    let integrity = MockIntegrityService::new()
        .with_embedding_model(spec_thresholds().echo_similarity.embedding_model.clone());
    for p in [0.1, 0.2, 0.15, 0.9] {
        integrity.push_reading(Some(p), IntegrityLevel::Low);
    }
    let engine = Engine::new(
        EngineDeps {
            sink: sink.clone(),
            store: store.clone(),
            providers: ProviderSet {
                llm: setup
                    .model
                    .then(|| Arc::new(llm.clone()) as Arc<dyn LlmProvider>),
                integrity: Some(Arc::new(integrity) as Arc<dyn IntegrityService>),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn minutes_without_a_model_keep_the_board_state() {
    let h = harness_with(Setup {
        model: false,
        semantic: false,
        threshold: 0.6,
    });
    let meeting = h.engine.start_meeting(start_request()).await.unwrap();
    transcript(&h, SpeakerRole::Remote, "We decided to freeze the schema.");
    wait_for_segment(&h, "freeze the schema").await;
    transcript(
        &h,
        SpeakerRole::Remote,
        "Can you share the rollout plan by Monday?",
    );
    wait_for_segment(&h, "rollout plan").await;
    // The user puts the question off: it stays open, marked as deferred.
    transcript(&h, SpeakerRole::User, "Let's take that offline.");
    wait_for_segment(&h, "take that offline").await;
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
    assert_eq!(summary.generated_by, "extractive", "no model was used");
    assert_eq!(summary.decisions, vec!["We decided to freeze the schema."]);
    assert!(
        summary
            .open_questions
            .iter()
            .any(|q| q.contains("rollout plan") && q.ends_with("(deferred)")),
        "deferred question kept: {:?}",
        summary.open_questions
    );
    assert_eq!(
        summary.open_questions,
        vec!["Can you share the rollout plan by Monday? (deferred)"]
    );
    assert!(
        summary.markdown.contains("(deferred)"),
        "{}",
        summary.markdown
    );
    assert!(summary.workplace.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn catch_up_without_a_provider_is_extracted_from_the_transcript() {
    let h = harness_with(Setup {
        model: false,
        semantic: false,
        threshold: 0.6,
    });
    h.engine.start_meeting(start_request()).await.unwrap();
    transcript(
        &h,
        SpeakerRole::Remote,
        "We decided to move billing to Kafka.",
    );
    wait_for_segment(&h, "move billing").await;
    h.engine.catch_up(5).unwrap();
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::AnswerComplete { suggestion, .. }
                if suggestion.kind == SuggestionKind::CatchUp)
        })
        .await
        .expect("catch-up recap without a model");
    let MeetingEvent::AnswerComplete { suggestion, .. } = ev else {
        unreachable!()
    };
    assert_eq!(suggestion.provider, "extractive");
    assert!(
        suggestion.content.contains("[[ANSWER]]"),
        "{}",
        suggestion.content
    );
    assert!(
        suggestion
            .content
            .contains("Decided: We decided to move billing to Kafka."),
        "{}",
        suggestion.content
    );
    assert!(
        suggestion.content.contains("without an AI model"),
        "{}",
        suggestion.content
    );
    assert!(
        !h.sink.events.lock().iter().any(|e| matches!(
            e,
            MeetingEvent::AnswerStateChanged {
                state: AnswerState::Failed,
                ..
            }
        )),
        "no failure is reported for the recap"
    );
    assert!(h.llm.requests().is_empty(), "no provider was called");
}

/// "Could you walk us through the rollout" is ambiguous to the heuristics
/// (confidence 0.62): below a 0.7 threshold it is not a question by rule.
const AMBIGUOUS_REQUEST: &str = "Could you walk us through the rollout";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn semantic_detection_confirms_an_ambiguous_question_on_the_board() {
    let h = harness_with(Setup {
        model: true,
        semantic: true,
        threshold: 0.7,
    });
    h.engine.start_meeting(start_request()).await.unwrap();
    transcript(&h, SpeakerRole::Remote, AMBIGUOUS_REQUEST);
    let ev = h
        .sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::QuestionDetected { question }
                if question.method == DetectionMethod::Semantic)
        })
        .await
        .expect("the LLM confirms the question");
    let MeetingEvent::QuestionDetected { question } = ev else {
        unreachable!()
    };
    assert!(question.question.contains("rollout"), "{question:?}");
    assert!(question.requires_response);
    let b = board(&h);
    assert!(
        b.open_questions
            .iter()
            .any(|q| q.question.contains("rollout") && q.addressed_to_you && !q.answered),
        "confirmed question on the board: {:?}",
        b.open_questions
    );
    // Put to the user: a colleague-worded suggestion follows.
    h.sink
        .wait_for(WAIT, |e| {
            matches!(e, MeetingEvent::AnswerComplete { suggestion, .. }
                if suggestion.kind == SuggestionKind::Answer)
        })
        .await
        .expect("suggestion for the confirmed question");
    assert!(
        h.llm
            .requests()
            .iter()
            .any(|r| r.kind == RequestKind::QuestionClassification),
        "the semantic check ran"
    );
    assert_eq!(h.sink.count("INTEGRITY_CHECKED"), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn semantic_detection_off_leaves_an_ambiguous_question_alone() {
    let h = harness_with(Setup {
        model: true,
        semantic: false,
        threshold: 0.7,
    });
    h.engine.start_meeting(start_request()).await.unwrap();
    transcript(&h, SpeakerRole::Remote, AMBIGUOUS_REQUEST);
    wait_for_segment(&h, "walk us through").await;
    assert_eq!(h.sink.count("QUESTION_DETECTED"), 0);
    assert!(board(&h).open_questions.is_empty());
    assert!(
        !h.llm
            .requests()
            .iter()
            .any(|r| r.kind == RequestKind::QuestionClassification),
        "no semantic check when the setting is off"
    );
}
