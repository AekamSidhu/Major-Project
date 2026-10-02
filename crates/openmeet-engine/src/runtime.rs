//! Per-meeting state owned by the engine.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use openmeet_core::ids::{Id, new_id, now_ms};
use openmeet_core::ipc::ActiveRequestView;
use openmeet_core::store::MeetingStore;
use openmeet_core::types::*;
use openmeet_intel::prompts::TranscriptLine;
use openmeet_intel::topics::TopicTracker;
use tokio::sync::mpsc;

use crate::conversation::ConvInput;
use crate::memory::ContextMemory;
use crate::state::MeetingStateMachine;

/// Maximum suggestions kept in memory per meeting (older ones stay in the DB).
const MAX_SUGGESTIONS: usize = 200;

/// Interviewer mode: the candidate's current turn (text since the last
/// interviewer question / follow-up generation).
#[derive(Debug, Default, Clone)]
pub struct CandidateTurn {
    pub segment_ids: Vec<Id>,
    pub text: String,
    pub version: u64,
}

impl CandidateTurn {
    pub fn push(&mut self, segment: &TranscriptSegment) {
        if !self.text.is_empty() {
            self.text.push(' ');
        }
        self.text.push_str(&segment.text);
        self.segment_ids.push(segment.id.clone());
        self.version += 1;
    }

    pub fn reset(&mut self) {
        self.segment_ids.clear();
        self.text.clear();
        self.version += 1;
    }
}

pub struct MeetingRuntime {
    pub meeting: Meeting,
    pub machine: MeetingStateMachine,
    pub paused_reason: Option<PauseReason>,
    /// Store chosen for this meeting (NullStore when not retained).
    pub store: Arc<dyn MeetingStore>,
    pub system_audio_active: bool,
    pub microphone_active: bool,
    pub demo: bool,
    /// Transcription sessions are running (false after an emergency stop).
    pub sessions_running: bool,
    /// Final segments ordered by start time.
    pub segments: Vec<TranscriptSegment>,
    pub questions: Vec<DetectedQuestion>,
    pub answers: Vec<AnswerRecord>,
    pub open_answer: Option<AnswerRecord>,
    pub suggestions: VecDeque<Suggestion>,
    pub latest_followups: Vec<FollowUpQuestion>,
    pub latest_followups_request: Option<Id>,
    pub all_followups: Vec<FollowUpQuestion>,
    pub claims: Vec<Claim>,
    pub gaps: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub observations: Vec<ScreenObservation>,
    pub topics: TopicTracker,
    pub memory: ContextMemory,
    pub turn: CandidateTurn,
    pub last_turn_text: String,
    /// Utterances submitted for transcription and not yet finalized.
    pub pending_utterances: HashSet<String>,
    pub speaking: HashSet<AudioSourceKind>,
    pub finals_since_topic_analysis: u32,
    pub last_topic_analysis_ms: i64,
    pub summary: Option<MeetingSummary>,
    pub active_primary: Option<ActiveRequestView>,
    /// Inputs into this meeting's conversation loop.
    pub conv_tx: mpsc::UnboundedSender<ConvInput>,
    /// Transcription events sender (cloned into every session).
    pub trans_tx: mpsc::UnboundedSender<openmeet_core::providers::TranscriptionEvent>,
    /// Timestamps (unix ms) of recent vision-model calls (rate limiting).
    pub vision_calls: VecDeque<i64>,
    /// Last transcription error notice (unix ms) for throttling.
    pub last_transcription_notice: i64,
    pub last_level_emit: HashMap<AudioSourceKind, i64>,
}

impl MeetingRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        meeting: Meeting,
        store: Arc<dyn MeetingStore>,
        planned_topics: &[String],
        system_audio_active: bool,
        microphone_active: bool,
        demo: bool,
        conv_tx: mpsc::UnboundedSender<ConvInput>,
        trans_tx: mpsc::UnboundedSender<openmeet_core::providers::TranscriptionEvent>,
    ) -> Self {
        let topics = TopicTracker::new(&meeting.id, meeting.mode, planned_topics);
        MeetingRuntime {
            machine: MeetingStateMachine::new(),
            paused_reason: None,
            store,
            system_audio_active,
            microphone_active,
            demo,
            sessions_running: false,
            segments: Vec::new(),
            questions: Vec::new(),
            answers: Vec::new(),
            open_answer: None,
            suggestions: VecDeque::new(),
            latest_followups: Vec::new(),
            latest_followups_request: None,
            all_followups: Vec::new(),
            claims: Vec::new(),
            gaps: Vec::new(),
            action_items: Vec::new(),
            observations: Vec::new(),
            topics,
            memory: ContextMemory::new(),
            turn: CandidateTurn::default(),
            last_turn_text: String::new(),
            pending_utterances: HashSet::new(),
            speaking: HashSet::new(),
            finals_since_topic_analysis: 0,
            last_topic_analysis_ms: 0,
            summary: None,
            active_primary: None,
            conv_tx,
            trans_tx,
            vision_calls: VecDeque::new(),
            last_transcription_notice: 0,
            last_level_emit: HashMap::new(),
            meeting,
        }
    }

    pub fn id(&self) -> &str {
        &self.meeting.id
    }

    pub fn state(&self) -> MeetingState {
        self.machine.state()
    }

    /// Validated state transition that keeps `meeting.state` in sync.
    pub fn transition(&mut self, next: MeetingState) -> Result<(), String> {
        self.machine.transition(next)?;
        self.meeting.state = next;
        Ok(())
    }

    /// Meeting is running (active or paused) and accepts transcript input.
    pub fn accepts_transcript(&self) -> bool {
        matches!(
            self.state(),
            MeetingState::Active | MeetingState::Paused | MeetingState::Stopping
        )
    }

    pub fn mode(&self) -> Mode {
        self.meeting.mode
    }

    /// Meeting-relative milliseconds for a unix timestamp.
    pub fn rel_ms(&self, unix_ms: i64) -> i64 {
        (unix_ms - self.meeting.started_at).max(0)
    }

    pub fn now_rel(&self) -> i64 {
        self.rel_ms(now_ms())
    }

    /// Speaker attribution from the audio source. Without system audio the
    /// microphone may hear everyone, so its speech is "unknown".
    pub fn role_for(&self, source: AudioSourceKind) -> SpeakerRole {
        match source {
            AudioSourceKind::System => SpeakerRole::Remote,
            AudioSourceKind::Microphone if self.system_audio_active => SpeakerRole::User,
            AudioSourceKind::Microphone => SpeakerRole::Unknown,
        }
    }

    /// Role whose speech answers a question asked by `asker`.
    pub fn answerer_for(&self, asker: SpeakerRole) -> SpeakerRole {
        match asker {
            SpeakerRole::Remote => SpeakerRole::User,
            SpeakerRole::User => SpeakerRole::Remote,
            SpeakerRole::Unknown => SpeakerRole::Unknown,
        }
    }

    pub fn speaker_label(&self, role: SpeakerRole) -> &'static str {
        role.label(self.meeting.mode)
    }

    pub fn participants_for(mode: Mode, system_audio: bool) -> Vec<Participant> {
        let mut p = vec![Participant {
            id: "user".into(),
            role: if system_audio { SpeakerRole::User } else { SpeakerRole::Unknown },
            label: if system_audio { "You".into() } else { "Microphone".into() },
        }];
        if system_audio {
            p.push(Participant {
                id: "remote".into(),
                role: SpeakerRole::Remote,
                label: SpeakerRole::Remote.label(mode).into(),
            });
        }
        p
    }

    /// Inserts a final segment keeping start-time order.
    pub fn add_final(&mut self, segment: TranscriptSegment) {
        let pos = self
            .segments
            .iter()
            .rposition(|s| s.start_time <= segment.start_time)
            .map(|p| p + 1)
            .unwrap_or(0);
        self.segments.insert(pos, segment);
    }

    pub fn line_for(&self, segment: &TranscriptSegment) -> TranscriptLine {
        TranscriptLine {
            speaker: self.speaker_label(segment.speaker_role).to_string(),
            text: segment.text.clone(),
            at_ms: segment.start_time,
        }
    }

    /// Transcript lines that started at or after `since_ms`.
    pub fn lines_since(&self, since_ms: i64) -> Vec<TranscriptLine> {
        self.segments
            .iter()
            .filter(|s| s.start_time >= since_ms)
            .map(|s| self.line_for(s))
            .collect()
    }

    pub fn all_lines(&self) -> Vec<TranscriptLine> {
        self.segments.iter().map(|s| self.line_for(s)).collect()
    }

    /// Latest question the user is expected to answer, if recent enough.
    pub fn latest_open_question(&self, max_age_ms: i64) -> Option<&DetectedQuestion> {
        let now = now_ms();
        self.questions
            .iter()
            .rev()
            .find(|q| q.requires_response && now - q.detected_at <= max_age_ms)
    }

    /// Text of remote speech in the last `window_ms`.
    pub fn recent_remote_text(&self, window_ms: i64) -> String {
        let since = self.now_rel() - window_ms;
        self.segments
            .iter()
            .filter(|s| s.start_time >= since && s.speaker_role != SpeakerRole::User)
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Questions already asked by the interviewer side (duplicate detection).
    pub fn asked_questions(&self) -> Vec<String> {
        let asker = match self.meeting.mode {
            Mode::Interviewer => SpeakerRole::User,
            Mode::Candidate => SpeakerRole::Remote,
        };
        self.questions
            .iter()
            .filter(|q| q.speaker_role == asker || q.speaker_role == SpeakerRole::Unknown)
            .map(|q| q.question.clone())
            .collect()
    }

    pub fn push_suggestion(&mut self, s: Suggestion) {
        if let Some(existing) = self.suggestions.iter_mut().find(|x| x.id == s.id) {
            *existing = s;
            return;
        }
        if self.suggestions.len() == MAX_SUGGESTIONS {
            // Never evict pinned suggestions.
            if let Some(pos) = self.suggestions.iter().position(|x| !x.pinned) {
                self.suggestions.remove(pos);
            }
        }
        self.suggestions.push_back(s);
    }

    pub fn suggestion(&self, id: &str) -> Option<&Suggestion> {
        self.suggestions.iter().find(|s| s.id == id)
    }

    pub fn suggestion_mut(&mut self, id: &str) -> Option<&mut Suggestion> {
        self.suggestions.iter_mut().find(|s| s.id == id)
    }

    /// Opens a new answer record for the response to `question`, returning the
    /// previous one when it contains text.
    pub fn open_answer_for(&mut self, question: &DetectedQuestion) -> Option<AnswerRecord> {
        let previous = self.close_answer();
        self.open_answer = Some(AnswerRecord {
            id: new_id(),
            meeting_id: self.meeting.id.clone(),
            question_id: Some(question.id.clone()),
            speaker_role: self.answerer_for(question.speaker_role),
            text: String::new(),
            segment_ids: Vec::new(),
            created_at: now_ms(),
        });
        previous
    }

    /// Appends a segment to the open answer when spoken by the answerer.
    pub fn extend_answer(&mut self, segment: &TranscriptSegment) {
        if let Some(open) = self.open_answer.as_mut() {
            if open.speaker_role == segment.speaker_role || open.speaker_role == SpeakerRole::Unknown {
                if !open.text.is_empty() {
                    open.text.push(' ');
                }
                open.text.push_str(&segment.text);
                open.segment_ids.push(segment.id.clone());
            }
        }
    }

    /// Closes the open answer; returns it when non-empty.
    pub fn close_answer(&mut self) -> Option<AnswerRecord> {
        let a = self.open_answer.take()?;
        if a.text.trim().is_empty() {
            return None;
        }
        self.answers.push(a.clone());
        Some(a)
    }

    pub fn to_record(&self) -> MeetingRecord {
        let mut answers = self.answers.clone();
        if let Some(open) = &self.open_answer {
            if !open.text.trim().is_empty() {
                answers.push(open.clone());
            }
        }
        MeetingRecord {
            meeting: self.meeting.clone(),
            segments: self.segments.clone(),
            questions: self.questions.clone(),
            answers,
            suggestions: self.suggestions.iter().cloned().collect(),
            topics: self.topics.topics(),
            claims: self.claims.clone(),
            observations: self.observations.clone(),
            action_items: self.action_items.clone(),
            summary: self.summary.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::store::NullStore;

    fn runtime(mode: Mode, system_audio: bool) -> MeetingRuntime {
        let (conv_tx, _r1) = mpsc::unbounded_channel();
        let (trans_tx, _r2) = mpsc::unbounded_channel();
        let meeting = Meeting {
            id: "m".into(),
            title: "t".into(),
            mode,
            state: MeetingState::Active,
            started_at: now_ms() - 60_000,
            ended_at: None,
            participants: MeetingRuntime::participants_for(mode, system_audio),
            persisted: false,
        };
        MeetingRuntime::new(meeting, Arc::new(NullStore), &[], system_audio, true, false, conv_tx, trans_tx)
    }

    fn seg(id: &str, role: SpeakerRole, start: i64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: id.into(),
            meeting_id: "m".into(),
            utterance_id: id.into(),
            speaker_id: None,
            speaker_role: role,
            source: AudioSourceKind::System,
            start_time: start,
            end_time: start + 500,
            text: text.into(),
            is_final: true,
            confidence: None,
        }
    }

    #[test]
    fn speaker_roles_depend_on_system_audio() {
        let r = runtime(Mode::Candidate, true);
        assert_eq!(r.role_for(AudioSourceKind::Microphone), SpeakerRole::User);
        assert_eq!(r.role_for(AudioSourceKind::System), SpeakerRole::Remote);
        let r = runtime(Mode::Candidate, false);
        assert_eq!(r.role_for(AudioSourceKind::Microphone), SpeakerRole::Unknown);
    }

    #[test]
    fn segments_stay_ordered() {
        let mut r = runtime(Mode::Candidate, true);
        r.add_final(seg("b", SpeakerRole::Remote, 2000, "second"));
        r.add_final(seg("a", SpeakerRole::User, 1000, "first"));
        r.add_final(seg("c", SpeakerRole::User, 3000, "third"));
        let order: Vec<_> = r.segments.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "c"]);
        let lines = r.lines_since(1500);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].speaker, "Interviewer");
    }

    #[test]
    fn answer_tracking() {
        let mut r = runtime(Mode::Candidate, true);
        let q = DetectedQuestion {
            id: "q1".into(),
            meeting_id: "m".into(),
            segment_ids: vec![],
            is_question: true,
            confidence: 0.9,
            question: "How did you scale it?".into(),
            parts: vec![],
            category: QuestionCategory::Technical,
            speaker_id: None,
            speaker_role: SpeakerRole::Remote,
            requires_response: true,
            method: DetectionMethod::Heuristic,
            supersedes: None,
            detected_at: now_ms(),
        };
        assert!(r.open_answer_for(&q).is_none());
        r.extend_answer(&seg("x", SpeakerRole::Remote, 1, "ignored, wrong speaker"));
        r.extend_answer(&seg("y", SpeakerRole::User, 2, "We sharded the database."));
        let closed = r.close_answer().unwrap();
        assert_eq!(closed.text, "We sharded the database.");
        assert_eq!(closed.question_id.as_deref(), Some("q1"));
        assert_eq!(r.answers.len(), 1);
        assert!(r.latest_open_question(60_000).is_none());
        r.questions.push(q);
        assert!(r.latest_open_question(60_000).is_some());
    }

    #[test]
    fn suggestions_are_bounded_and_keep_pinned() {
        let mut r = runtime(Mode::Candidate, true);
        for i in 0..(MAX_SUGGESTIONS + 5) {
            r.push_suggestion(Suggestion {
                id: format!("s{i}"),
                meeting_id: Some("m".into()),
                kind: SuggestionKind::Answer,
                question_id: None,
                question_text: None,
                category: None,
                style: None,
                parent_id: None,
                prompt_id: "p".into(),
                prompt_version: 1,
                content: String::new(),
                followups: vec![],
                provider: "mock".into(),
                model: "m".into(),
                created_at: 0,
                completed_at: None,
                pinned: i == 0,
            });
        }
        assert_eq!(r.suggestions.len(), MAX_SUGGESTIONS);
        assert!(r.suggestion("s0").is_some(), "pinned suggestion must survive");
    }

    #[test]
    fn candidate_turn() {
        let mut t = CandidateTurn::default();
        let v0 = t.version;
        t.push(&seg("a", SpeakerRole::Remote, 0, "We used Redis"));
        t.push(&seg("b", SpeakerRole::Remote, 1, "for rate limiting"));
        assert_eq!(t.text, "We used Redis for rate limiting");
        assert!(t.version > v0);
        t.reset();
        assert!(t.text.is_empty());
    }
}
