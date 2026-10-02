//! The meeting event bus contract (spec §85).
//!
//! Every subsystem reports through [`MeetingEvent`]s. The Tauri layer forwards
//! them to all windows on the `openmeet://event` channel; tests collect them
//! with [`CollectingSink`].

use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::ids::Id;
use crate::types::*;

/// Tauri event name used for [`MeetingEvent`]s.
pub const MEETING_EVENT_CHANNEL: &str = "openmeet://event";
/// Tauri event name used for [`crate::ipc::UiCommand`]s.
pub const UI_EVENT_CHANNEL: &str = "openmeet://ui";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
#[ts(export)]
pub enum MeetingEvent {
    MeetingStateChanged {
        #[serde(rename = "meetingId")]
        meeting_id: Option<Id>,
        state: MeetingState,
        mode: Mode,
        #[serde(rename = "pausedReason")]
        paused_reason: Option<PauseReason>,
    },
    MeetingStarted {
        meeting: Meeting,
    },
    MeetingStopped {
        #[serde(rename = "meetingId")]
        meeting_id: Id,
    },
    TranscriptPartial {
        segment: TranscriptSegment,
    },
    TranscriptFinal {
        segment: TranscriptSegment,
    },
    QuestionDetected {
        question: DetectedQuestion,
    },
    /// A request started rendering into `slot`. The UI shows only the latest
    /// request id per slot and ignores events for any other id.
    AnswerStarted {
        #[serde(rename = "requestId")]
        request_id: Id,
        kind: RequestKind,
        slot: RequestSlot,
        #[serde(rename = "questionId")]
        question_id: Option<Id>,
        #[serde(rename = "questionText")]
        question_text: Option<String>,
        category: Option<QuestionCategory>,
        style: Option<AnswerStyle>,
        #[serde(rename = "parentId")]
        parent_id: Option<Id>,
    },
    AnswerStateChanged {
        #[serde(rename = "requestId")]
        request_id: Id,
        state: AnswerState,
        error: Option<String>,
    },
    AnswerDelta {
        #[serde(rename = "requestId")]
        request_id: Id,
        text: String,
    },
    AnswerComplete {
        #[serde(rename = "requestId")]
        request_id: Id,
        suggestion: Suggestion,
    },
    /// Follow-ups arrive progressively; `done` marks the final batch. Each
    /// event carries the full list so far for this request.
    FollowupsReady {
        #[serde(rename = "requestId")]
        request_id: Id,
        questions: Vec<FollowUpQuestion>,
        done: bool,
    },
    SuggestionPinned {
        #[serde(rename = "requestId")]
        request_id: Id,
        pinned: bool,
        suggestion: Option<Suggestion>,
    },
    SuggestionDismissed {
        #[serde(rename = "requestId")]
        request_id: Id,
    },
    ScreenUpdated {
        observation: ScreenObservation,
    },
    TopicsUpdated {
        topics: Vec<Topic>,
        #[serde(rename = "suggestedNext")]
        suggested_next: Option<String>,
    },
    ClaimsExtracted {
        claims: Vec<Claim>,
    },
    ActionItemsUpdated {
        items: Vec<ActionItem>,
    },
    SummaryReady {
        summary: MeetingSummary,
    },
    SummaryFailed {
        #[serde(rename = "meetingId")]
        meeting_id: Id,
        error: String,
    },
    PrivacyStateChanged {
        /// Capture exclusion requested by the user.
        enabled: bool,
        supported: bool,
        /// Exclusion currently applied.
        active: bool,
    },
    ComponentStatusChanged {
        health: ComponentHealth,
    },
    /// Emergency stop: all capture and AI processing halted.
    ProcessingStopped {
        reason: String,
    },
    LatencyRecorded {
        sample: LatencySample,
    },
    VoiceCommandRecognized {
        command: VoiceCommand,
    },
    ModeChanged {
        mode: Mode,
    },
    StyleChanged {
        style: AnswerStyle,
    },
    /// Normalized input level 0..1 (throttled, for meters).
    AudioLevel {
        source: AudioSourceKind,
        level: f32,
    },
    /// User-visible notice (errors are never silent).
    Notice {
        level: NoticeLevel,
        message: String,
        component: Option<Component>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, TS)]
#[serde(rename_all = "snake_case")]
#[ts(export)]
pub enum NoticeLevel {
    Info,
    Warning,
    Error,
}

impl MeetingEvent {
    /// Stable name of the variant (used in logs; never logs payloads).
    pub fn name(&self) -> &'static str {
        match self {
            MeetingEvent::MeetingStateChanged { .. } => "MEETING_STATE_CHANGED",
            MeetingEvent::MeetingStarted { .. } => "MEETING_STARTED",
            MeetingEvent::MeetingStopped { .. } => "MEETING_STOPPED",
            MeetingEvent::TranscriptPartial { .. } => "TRANSCRIPT_PARTIAL",
            MeetingEvent::TranscriptFinal { .. } => "TRANSCRIPT_FINAL",
            MeetingEvent::QuestionDetected { .. } => "QUESTION_DETECTED",
            MeetingEvent::AnswerStarted { .. } => "ANSWER_STARTED",
            MeetingEvent::AnswerStateChanged { .. } => "ANSWER_STATE_CHANGED",
            MeetingEvent::AnswerDelta { .. } => "ANSWER_DELTA",
            MeetingEvent::AnswerComplete { .. } => "ANSWER_COMPLETE",
            MeetingEvent::FollowupsReady { .. } => "FOLLOWUPS_READY",
            MeetingEvent::SuggestionPinned { .. } => "SUGGESTION_PINNED",
            MeetingEvent::SuggestionDismissed { .. } => "SUGGESTION_DISMISSED",
            MeetingEvent::ScreenUpdated { .. } => "SCREEN_UPDATED",
            MeetingEvent::TopicsUpdated { .. } => "TOPICS_UPDATED",
            MeetingEvent::ClaimsExtracted { .. } => "CLAIMS_EXTRACTED",
            MeetingEvent::ActionItemsUpdated { .. } => "ACTION_ITEMS_UPDATED",
            MeetingEvent::SummaryReady { .. } => "SUMMARY_READY",
            MeetingEvent::SummaryFailed { .. } => "SUMMARY_FAILED",
            MeetingEvent::PrivacyStateChanged { .. } => "PRIVACY_STATE_CHANGED",
            MeetingEvent::ComponentStatusChanged { .. } => "COMPONENT_STATUS_CHANGED",
            MeetingEvent::ProcessingStopped { .. } => "PROCESSING_STOPPED",
            MeetingEvent::LatencyRecorded { .. } => "LATENCY_RECORDED",
            MeetingEvent::VoiceCommandRecognized { .. } => "VOICE_COMMAND_RECOGNIZED",
            MeetingEvent::ModeChanged { .. } => "MODE_CHANGED",
            MeetingEvent::StyleChanged { .. } => "STYLE_CHANGED",
            MeetingEvent::AudioLevel { .. } => "AUDIO_LEVEL",
            MeetingEvent::Notice { .. } => "NOTICE",
        }
    }
}

/// Destination for meeting events. Implementations must be cheap and must not
/// block (they are called from the audio and streaming hot paths).
pub trait EventSink: Send + Sync + 'static {
    fn emit(&self, event: MeetingEvent);
}

/// Discards all events.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: MeetingEvent) {}
}

/// Records events in memory (tests, benchmarks).
#[derive(Debug, Default, Clone)]
pub struct CollectingSink {
    events: Arc<Mutex<Vec<MeetingEvent>>>,
}

impl CollectingSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<MeetingEvent> {
        self.events.lock().clone()
    }

    pub fn take(&self) -> Vec<MeetingEvent> {
        std::mem::take(&mut *self.events.lock())
    }

    pub fn count(&self, name: &str) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| e.name() == name)
            .count()
    }

    pub fn clear(&self) {
        self.events.lock().clear();
    }
}

impl EventSink for CollectingSink {
    fn emit(&self, event: MeetingEvent) {
        self.events.lock().push(event);
    }
}

/// Forwards every event to several sinks.
pub struct FanoutSink {
    sinks: Vec<Arc<dyn EventSink>>,
}

impl FanoutSink {
    pub fn new(sinks: Vec<Arc<dyn EventSink>>) -> Self {
        FanoutSink { sinks }
    }
}

impl EventSink for FanoutSink {
    fn emit(&self, event: MeetingEvent) {
        if let Some((last, rest)) = self.sinks.split_last() {
            for s in rest {
                s.emit(event.clone());
            }
            last.emit(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_json_shape() {
        let e = MeetingEvent::AnswerDelta {
            request_id: "r1".into(),
            text: "Hel".into(),
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["type"], "ANSWER_DELTA");
        assert_eq!(v["requestId"], "r1");
        assert_eq!(v["text"], "Hel");
        assert_eq!(e.name(), "ANSWER_DELTA");
        let back: MeetingEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn collecting_and_fanout() {
        let a = CollectingSink::new();
        let b = CollectingSink::new();
        let fan = FanoutSink::new(vec![Arc::new(a.clone()), Arc::new(b.clone())]);
        fan.emit(MeetingEvent::ModeChanged {
            mode: Mode::Interviewer,
        });
        assert_eq!(a.count("MODE_CHANGED"), 1);
        assert_eq!(b.events().len(), 1);
        assert_eq!(a.take().len(), 1);
        assert!(a.events().is_empty());
    }
}
