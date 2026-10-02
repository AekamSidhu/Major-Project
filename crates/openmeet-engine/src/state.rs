//! Explicit state machines (spec §86, §87).

use openmeet_core::types::{AnswerState, MeetingState};

/// Validated meeting lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeetingStateMachine {
    state: MeetingState,
}

impl Default for MeetingStateMachine {
    fn default() -> Self {
        MeetingStateMachine {
            state: MeetingState::Idle,
        }
    }
}

impl MeetingStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> MeetingState {
        self.state
    }

    /// Applies a transition; returns an error describing the illegal move.
    pub fn transition(&mut self, next: MeetingState) -> Result<MeetingState, String> {
        if self.state == next {
            return Ok(next);
        }
        if self.state.can_transition_to(next) {
            self.state = next;
            Ok(next)
        } else {
            Err(format!(
                "cannot move meeting from {:?} to {:?}",
                self.state, next
            ))
        }
    }
}

/// Whether an AI request may move from `from` to `to`.
pub fn answer_transition_allowed(from: AnswerState, to: AnswerState) -> bool {
    use AnswerState::*;
    if from == to {
        return true;
    }
    if from.is_terminal() {
        return false;
    }
    match to {
        Failed | Cancelled => true,
        ContextBuilding => from == Created,
        Requesting => matches!(from, Created | ContextBuilding),
        Streaming => matches!(from, Requesting),
        Complete => matches!(from, Requesting | Streaming),
        Created => false,
    }
}

/// Tracks one request's lifecycle and rejects illegal transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnswerStateMachine {
    state: AnswerState,
}

impl Default for AnswerStateMachine {
    fn default() -> Self {
        AnswerStateMachine {
            state: AnswerState::Created,
        }
    }
}

impl AnswerStateMachine {
    pub fn state(&self) -> AnswerState {
        self.state
    }

    /// Returns true when the state changed.
    pub fn advance(&mut self, to: AnswerState) -> bool {
        if self.state == to || !answer_transition_allowed(self.state, to) {
            return false;
        }
        self.state = to;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meeting_happy_path() {
        let mut m = MeetingStateMachine::new();
        for s in [
            MeetingState::Starting,
            MeetingState::Active,
            MeetingState::Paused,
            MeetingState::Active,
            MeetingState::Stopping,
            MeetingState::Summarizing,
            MeetingState::Completed,
            MeetingState::Starting,
        ] {
            m.transition(s).unwrap();
        }
        assert_eq!(m.state(), MeetingState::Starting);
    }

    #[test]
    fn meeting_rejects_illegal() {
        let mut m = MeetingStateMachine::new();
        assert!(m.transition(MeetingState::Active).is_err());
        assert!(m.transition(MeetingState::Summarizing).is_err());
        assert_eq!(m.state(), MeetingState::Idle);
    }

    #[test]
    fn answer_lifecycle() {
        let mut a = AnswerStateMachine::default();
        assert!(a.advance(AnswerState::ContextBuilding));
        assert!(a.advance(AnswerState::Requesting));
        assert!(a.advance(AnswerState::Streaming));
        assert!(!a.advance(AnswerState::Requesting));
        assert!(a.advance(AnswerState::Complete));
        assert!(!a.advance(AnswerState::Cancelled), "terminal states are final");
        assert_eq!(a.state(), AnswerState::Complete);
    }

    #[test]
    fn answer_can_fail_or_cancel_anytime_before_terminal() {
        for s in [
            AnswerState::Created,
            AnswerState::ContextBuilding,
            AnswerState::Requesting,
            AnswerState::Streaming,
        ] {
            assert!(answer_transition_allowed(s, AnswerState::Failed));
            assert!(answer_transition_allowed(s, AnswerState::Cancelled));
        }
        assert!(!answer_transition_allowed(AnswerState::Created, AnswerState::Streaming));
    }
}
