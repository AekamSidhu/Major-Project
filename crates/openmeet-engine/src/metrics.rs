//! Latency tracking (spec §53) and component health (spec §57).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use openmeet_core::events::{EventSink, MeetingEvent};
use openmeet_core::ids::now_ms;
use openmeet_core::types::{
    Component, ComponentHealth, ComponentStatus, LatencySample, LatencyStage, LatencyStats,
};
use parking_lot::Mutex;

const MAX_SAMPLES: usize = 200;

/// Rolling latency samples per pipeline stage.
#[derive(Default)]
pub struct LatencyTracker {
    samples: Mutex<BTreeMap<LatencyStage, VecDeque<f64>>>,
    last: Mutex<BTreeMap<LatencyStage, f64>>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, stage: LatencyStage, ms: f64) {
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let mut samples = self.samples.lock();
        let q = samples.entry(stage).or_default();
        if q.len() == MAX_SAMPLES {
            q.pop_front();
        }
        q.push_back(ms);
        self.last.lock().insert(stage, ms);
    }

    pub fn stats(&self) -> Vec<LatencyStats> {
        let samples = self.samples.lock();
        let last = self.last.lock();
        samples
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(stage, q)| {
                let mut sorted: Vec<f64> = q.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                LatencyStats {
                    stage: *stage,
                    count: sorted.len() as u32,
                    last_ms: last.get(stage).copied().unwrap_or(0.0),
                    p50_ms: percentile(&sorted, 0.50),
                    p95_ms: percentile(&sorted, 0.95),
                    max_ms: *sorted.last().unwrap_or(&0.0),
                }
            })
            .collect()
    }

    pub fn clear(&self) {
        self.samples.lock().clear();
        self.last.lock().clear();
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Records a latency sample and emits a `LATENCY_RECORDED` event.
pub fn record_latency(
    tracker: &LatencyTracker,
    sink: &Arc<dyn EventSink>,
    stage: LatencyStage,
    ms: f64,
    request_id: Option<String>,
) {
    tracker.record(stage, ms);
    sink.emit(MeetingEvent::LatencyRecorded {
        sample: LatencySample {
            stage,
            ms,
            request_id,
            at: now_ms(),
        },
    });
}

/// Health of every subsystem. Emits `COMPONENT_STATUS_CHANGED` only when a
/// component's status or message actually changes.
pub struct HealthRegistry {
    sink: Arc<dyn EventSink>,
    map: Mutex<BTreeMap<Component, ComponentHealth>>,
}

impl HealthRegistry {
    pub fn new(sink: Arc<dyn EventSink>) -> Self {
        HealthRegistry {
            sink,
            map: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn set(&self, component: Component, status: ComponentStatus, message: Option<String>) {
        let changed = {
            let mut map = self.map.lock();
            match map.get(&component) {
                Some(h) if h.status == status && h.message == message => None,
                _ => {
                    let health = ComponentHealth {
                        component,
                        status,
                        message,
                        updated_at: now_ms(),
                    };
                    map.insert(component, health.clone());
                    Some(health)
                }
            }
        };
        if let Some(health) = changed {
            self.sink.emit(MeetingEvent::ComponentStatusChanged { health });
        }
    }

    pub fn get(&self, component: Component) -> Option<ComponentHealth> {
        self.map.lock().get(&component).cloned()
    }

    pub fn status(&self, component: Component) -> Option<ComponentStatus> {
        self.map.lock().get(&component).map(|h| h.status)
    }

    pub fn all(&self) -> Vec<ComponentHealth> {
        self.map.lock().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::events::CollectingSink;

    #[test]
    fn percentiles() {
        let t = LatencyTracker::new();
        for i in 1..=100 {
            t.record(LatencyStage::DetectionToFirstToken, i as f64);
        }
        t.record(LatencyStage::DetectionToFirstToken, f64::NAN);
        let s = &t.stats()[0];
        assert_eq!(s.count, 100);
        assert_eq!(s.last_ms, 100.0);
        assert!((s.p50_ms - 50.0).abs() <= 1.0);
        assert!((s.p95_ms - 95.0).abs() <= 1.0);
        assert_eq!(s.max_ms, 100.0);
    }

    #[test]
    fn ring_buffer_is_bounded() {
        let t = LatencyTracker::new();
        for i in 0..(MAX_SAMPLES + 50) {
            t.record(LatencyStage::AudioToTranscript, i as f64);
        }
        assert_eq!(t.stats()[0].count as usize, MAX_SAMPLES);
    }

    #[test]
    fn health_emits_only_on_change() {
        let sink = CollectingSink::new();
        let h = HealthRegistry::new(Arc::new(sink.clone()));
        h.set(Component::Llm, ComponentStatus::Active, None);
        h.set(Component::Llm, ComponentStatus::Active, None);
        h.set(Component::Llm, ComponentStatus::Error, Some("down".into()));
        assert_eq!(sink.count("COMPONENT_STATUS_CHANGED"), 2);
        assert_eq!(h.status(Component::Llm), Some(ComponentStatus::Error));
        assert_eq!(h.all().len(), 1);
    }
}
