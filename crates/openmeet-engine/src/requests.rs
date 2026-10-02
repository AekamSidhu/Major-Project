//! Request orchestration: slots, cancellation and stale-response prevention
//! (spec §55, §56).
//!
//! * Starting a request in the `Primary` or `Followups` slot cancels the request
//!   currently occupying that slot.
//! * UI-visible events of a request are emitted only while it is still the
//!   current request of its slot, and the check-and-emit happens under the
//!   slot lock, so an old answer can never be interleaved after a newer
//!   `ANSWER_STARTED`.
//! * `Background` requests never cancel each other; they are cancelled only by
//!   `cancel_all` (pause, emergency stop).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::{Id, new_id};
use openmeet_core::types::{AnswerState, RequestKind, RequestSlot};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use crate::state::AnswerStateMachine;

/// Handle held by the task executing a request.
#[derive(Debug, Clone)]
pub struct RequestTicket {
    pub id: Id,
    pub slot: RequestSlot,
    pub kind: RequestKind,
    pub cancel: CancellationToken,
    pub started: Instant,
}

impl RequestTicket {
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

struct Entry {
    ticket: RequestTicket,
    machine: AnswerStateMachine,
}

#[derive(Default)]
struct Slots {
    current: HashMap<RequestSlot, Id>,
    entries: HashMap<Id, Entry>,
}

pub struct RequestManager {
    sink: Arc<dyn EventSink>,
    slots: Mutex<Slots>,
}

impl RequestManager {
    pub fn new(sink: Arc<dyn EventSink>) -> Self {
        RequestManager {
            sink,
            slots: Mutex::new(Slots::default()),
        }
    }

    /// Registers a new request. For exclusive slots the previous request is
    /// cancelled and `started_event` (typically `ANSWER_STARTED`) is emitted
    /// atomically with the slot switch.
    pub fn begin(
        &self,
        slot: RequestSlot,
        kind: RequestKind,
        started_event: impl FnOnce(&Id) -> Option<MeetingEvent>,
    ) -> RequestTicket {
        let ticket = RequestTicket {
            id: new_id(),
            slot,
            kind,
            cancel: CancellationToken::new(),
            started: Instant::now(),
        };
        let mut slots = self.slots.lock();
        if slot != RequestSlot::Background {
            if let Some(prev_id) = slots.current.insert(slot, ticket.id.clone()) {
                if let Some(mut prev) = slots.entries.remove(&prev_id) {
                    prev.ticket.cancel.cancel();
                    if prev.machine.advance(AnswerState::Cancelled) {
                        self.sink.emit(MeetingEvent::AnswerStateChanged {
                            request_id: prev_id.clone(),
                            state: AnswerState::Cancelled,
                            error: None,
                        });
                    }
                }
            }
        }
        slots.entries.insert(
            ticket.id.clone(),
            Entry {
                ticket: ticket.clone(),
                machine: AnswerStateMachine::default(),
            },
        );
        if let Some(ev) = started_event(&ticket.id) {
            self.sink.emit(ev);
        }
        ticket
    }

    /// True while `ticket` is live and (for exclusive slots) still the current
    /// request of its slot.
    pub fn is_current(&self, ticket: &RequestTicket) -> bool {
        if ticket.cancel.is_cancelled() {
            return false;
        }
        let slots = self.slots.lock();
        match ticket.slot {
            RequestSlot::Background => slots.entries.contains_key(&ticket.id),
            slot => slots.current.get(&slot) == Some(&ticket.id),
        }
    }

    /// Emits `event` only if the ticket is still current (atomic check+emit).
    /// Returns whether it was emitted.
    pub fn emit_if_current(&self, ticket: &RequestTicket, event: MeetingEvent) -> bool {
        if ticket.cancel.is_cancelled() {
            return false;
        }
        let slots = self.slots.lock();
        let current = match ticket.slot {
            RequestSlot::Background => slots.entries.contains_key(&ticket.id),
            slot => slots.current.get(&slot) == Some(&ticket.id),
        };
        if current {
            self.sink.emit(event);
        }
        current
    }

    /// Advances the request state machine and emits `ANSWER_STATE_CHANGED`
    /// (UI slots only, and only while current — except terminal states, which
    /// are always reported so the UI never hangs on a spinner).
    pub fn set_state(&self, ticket: &RequestTicket, state: AnswerState, error: Option<String>) {
        let mut slots = self.slots.lock();
        let is_current = match ticket.slot {
            RequestSlot::Background => true,
            slot => slots.current.get(&slot) == Some(&ticket.id),
        };
        let Some(entry) = slots.entries.get_mut(&ticket.id) else {
            return;
        };
        if !entry.machine.advance(state) {
            return;
        }
        let terminal = state.is_terminal();
        if ticket.slot != RequestSlot::Background && (is_current || terminal) {
            self.sink.emit(MeetingEvent::AnswerStateChanged {
                request_id: ticket.id.clone(),
                state,
                error,
            });
        }
        if terminal {
            // Keep the slot pointer (so stale events stay suppressed) but drop
            // the bookkeeping entry of finished non-current requests.
            let still_current = slots.current.values().any(|id| id == &ticket.id);
            if !still_current {
                slots.entries.remove(&ticket.id);
            }
        }
    }

    pub fn state_of(&self, request_id: &str) -> Option<AnswerState> {
        self.slots
            .lock()
            .entries
            .get(request_id)
            .map(|e| e.machine.state())
    }

    /// Cancels one request (dismiss / explicit cancel).
    pub fn cancel(&self, request_id: &str) -> bool {
        let mut slots = self.slots.lock();
        let Some(entry) = slots.entries.get_mut(request_id) else {
            return false;
        };
        entry.ticket.cancel.cancel();
        let changed = entry.machine.advance(AnswerState::Cancelled);
        let slot = entry.ticket.slot;
        if changed && slot != RequestSlot::Background {
            self.sink.emit(MeetingEvent::AnswerStateChanged {
                request_id: request_id.to_string(),
                state: AnswerState::Cancelled,
                error: None,
            });
        }
        changed
    }

    /// Cancels every in-flight request. `include_background` = false keeps
    /// background work such as an in-progress summary alive.
    pub fn cancel_all(&self, include_background: bool) {
        let mut slots = self.slots.lock();
        let ids: Vec<Id> = slots
            .entries
            .iter()
            .filter(|(_, e)| include_background || e.ticket.slot != RequestSlot::Background)
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            if let Some(entry) = slots.entries.get_mut(&id) {
                entry.ticket.cancel.cancel();
                if entry.machine.advance(AnswerState::Cancelled)
                    && entry.ticket.slot != RequestSlot::Background
                {
                    self.sink.emit(MeetingEvent::AnswerStateChanged {
                        request_id: id.clone(),
                        state: AnswerState::Cancelled,
                        error: None,
                    });
                }
            }
        }
    }

    /// Id of the current request in a UI slot.
    pub fn current(&self, slot: RequestSlot) -> Option<Id> {
        self.slots.lock().current.get(&slot).cloned()
    }

    /// Number of non-terminal requests.
    pub fn active_count(&self) -> usize {
        self.slots
            .lock()
            .entries
            .values()
            .filter(|e| !e.machine.state().is_terminal())
            .count()
    }

    /// Whether a background request of `kind` is still running.
    pub fn background_running(&self, kind: RequestKind) -> bool {
        self.slots.lock().entries.values().any(|e| {
            e.ticket.slot == RequestSlot::Background
                && e.ticket.kind == kind
                && !e.machine.state().is_terminal()
        })
    }

    /// Clears slot pointers (new meeting).
    pub fn reset(&self) {
        let mut slots = self.slots.lock();
        for e in slots.entries.values() {
            e.ticket.cancel.cancel();
        }
        slots.entries.clear();
        slots.current.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::events::CollectingSink;

    fn mgr() -> (RequestManager, CollectingSink) {
        let sink = CollectingSink::new();
        (RequestManager::new(Arc::new(sink.clone())), sink)
    }

    fn delta(t: &RequestTicket, s: &str) -> MeetingEvent {
        MeetingEvent::AnswerDelta {
            request_id: t.id.clone(),
            text: s.into(),
        }
    }

    #[test]
    fn new_request_cancels_previous_in_slot() {
        let (m, sink) = mgr();
        let a = m.begin(RequestSlot::Primary, RequestKind::CandidateAnswer, |_| None);
        let b = m.begin(RequestSlot::Primary, RequestKind::CandidateAnswer, |_| None);
        assert!(a.is_cancelled());
        assert!(!m.is_current(&a));
        assert!(m.is_current(&b));
        assert!(!m.emit_if_current(&a, delta(&a, "stale")));
        assert!(m.emit_if_current(&b, delta(&b, "fresh")));
        let deltas: Vec<_> = sink
            .events()
            .into_iter()
            .filter_map(|e| match e {
                MeetingEvent::AnswerDelta { request_id, .. } => Some(request_id),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec![b.id.clone()]);
        // The superseded request was reported as cancelled and forgotten.
        assert!(sink.events().iter().any(|e| matches!(e,
            MeetingEvent::AnswerStateChanged { request_id, state: AnswerState::Cancelled, .. }
                if request_id == &a.id)));
        assert_eq!(m.state_of(&a.id), None);
        // Late state updates from the cancelled task are ignored.
        m.set_state(&a, AnswerState::Failed, Some("late".into()));
        assert_eq!(sink.count("ANSWER_STATE_CHANGED"), 1);
    }

    #[test]
    fn slots_are_independent() {
        let (m, _) = mgr();
        let a = m.begin(RequestSlot::Primary, RequestKind::CandidateAnswer, |_| None);
        let f = m.begin(RequestSlot::Followups, RequestKind::InterviewerFollowups, |_| None);
        let bg1 = m.begin(RequestSlot::Background, RequestKind::Compression, |_| None);
        let bg2 = m.begin(RequestSlot::Background, RequestKind::Summary, |_| None);
        assert!(m.is_current(&a) && m.is_current(&f) && m.is_current(&bg1) && m.is_current(&bg2));
        assert!(m.background_running(RequestKind::Compression));
    }

    #[test]
    fn started_event_is_emitted_with_switch() {
        let (m, sink) = mgr();
        let t = m.begin(RequestSlot::Primary, RequestKind::AskAi, |id| {
            Some(MeetingEvent::AnswerDelta {
                request_id: id.clone(),
                text: String::new(),
            })
        });
        assert_eq!(sink.count("ANSWER_DELTA"), 1);
        assert!(m.is_current(&t));
    }

    #[test]
    fn state_transitions_and_terminal_reporting() {
        let (m, sink) = mgr();
        let a = m.begin(RequestSlot::Primary, RequestKind::CandidateAnswer, |_| None);
        m.set_state(&a, AnswerState::ContextBuilding, None);
        m.set_state(&a, AnswerState::Requesting, None);
        m.set_state(&a, AnswerState::Streaming, None);
        m.set_state(&a, AnswerState::Requesting, None); // illegal, ignored
        m.set_state(&a, AnswerState::Complete, None);
        assert_eq!(m.state_of(&a.id), Some(AnswerState::Complete));
        assert_eq!(sink.count("ANSWER_STATE_CHANGED"), 4);
        // completed request stays the slot's current -> no new state events for it
        m.set_state(&a, AnswerState::Failed, Some("x".into()));
        assert_eq!(sink.count("ANSWER_STATE_CHANGED"), 4);
    }

    #[test]
    fn background_states_are_not_emitted() {
        let (m, sink) = mgr();
        let bg = m.begin(RequestSlot::Background, RequestKind::Summary, |_| None);
        m.set_state(&bg, AnswerState::Requesting, None);
        m.set_state(&bg, AnswerState::Complete, None);
        assert_eq!(sink.count("ANSWER_STATE_CHANGED"), 0);
        assert!(!m.background_running(RequestKind::Summary));
    }

    #[test]
    fn cancel_all_respects_background_flag() {
        let (m, _) = mgr();
        let a = m.begin(RequestSlot::Primary, RequestKind::CandidateAnswer, |_| None);
        let bg = m.begin(RequestSlot::Background, RequestKind::Summary, |_| None);
        m.cancel_all(false);
        assert!(a.is_cancelled());
        assert!(!bg.is_cancelled());
        m.cancel_all(true);
        assert!(bg.is_cancelled());
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn explicit_cancel() {
        let (m, sink) = mgr();
        let a = m.begin(RequestSlot::Primary, RequestKind::AskAi, |_| None);
        assert!(m.cancel(&a.id));
        assert!(!m.cancel(&a.id));
        assert!(!m.is_current(&a));
        assert_eq!(sink.count("ANSWER_STATE_CHANGED"), 1);
    }
}
