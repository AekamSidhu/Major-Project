//! Conversation topic tracking (spec §25).
//!
//! A built-in taxonomy (plus the user's planned topics) is matched against
//! every final segment. A topic becomes **current** after ≥ 2 keyword hits
//! within the last ~60 s of speech and **covered** when another topic becomes
//! current. Topics mentioned only in passing stay **pending**. The tracker
//! suggests the next area to explore: planned topics first, then taxonomy
//! areas that naturally follow what has been discussed.
//!
//! Timestamps in [`Topic`] are meeting-relative milliseconds.

use std::collections::VecDeque;

use openmeet_core::types::{Mode, SpeakerRole, Topic, TopicStatus, TranscriptSegment};

use crate::text::{content_tokens, lexical_similarity};

/// Window in which hits count towards making a topic current.
const CURRENT_WINDOW_MS: i64 = 60_000;
/// Hits inside the window needed to become current.
const CURRENT_MIN_HITS: u32 = 2;

struct TaxonomyEntry {
    id: &'static str,
    name: &'static str,
    keywords: &'static [&'static str],
}

static TAXONOMY: &[TaxonomyEntry] = &[
    TaxonomyEntry {
        id: "background",
        name: "Background & experience",
        keywords: &[
            "experience",
            "background",
            "previous role",
            "resume",
            "career",
            "worked at",
            "years of",
            "my role",
            "current role",
            "joined",
            "introduce yourself",
        ],
    },
    TaxonomyEntry {
        id: "architecture",
        name: "Architecture",
        keywords: &[
            "architecture",
            "microservice",
            "microservices",
            "monolith",
            "service boundaries",
            "component",
            "components",
            "event-driven",
            "event driven",
            "api gateway",
            "design pattern",
            "queue",
            "workers",
            "pipeline",
        ],
    },
    TaxonomyEntry {
        id: "system-design",
        name: "System design",
        keywords: &[
            "design a",
            "system design",
            "scalability",
            "distributed",
            "high availability",
            "load balancer",
            "sharding",
            "partition",
            "replication",
            "consistency",
            "cap theorem",
            "rate limit",
            "rate limiter",
            "url shortener",
            "requirements",
        ],
    },
    TaxonomyEntry {
        id: "database",
        name: "Database & storage",
        keywords: &[
            "database",
            "sql",
            "nosql",
            "postgres",
            "postgresql",
            "mysql",
            "mongodb",
            "schema",
            "index",
            "query",
            "queries",
            "transaction",
            "storage",
            "dynamodb",
            "cassandra",
            "redis",
            "migration",
        ],
    },
    TaxonomyEntry {
        id: "performance",
        name: "Scaling & performance",
        keywords: &[
            "performance",
            "latency",
            "throughput",
            "bottleneck",
            "cache",
            "caching",
            "optimize",
            "optimization",
            "scale",
            "scaling",
            "load",
            "qps",
            "rps",
            "p99",
            "memory",
            "cpu",
        ],
    },
    TaxonomyEntry {
        id: "deployment",
        name: "Deployment & infrastructure",
        keywords: &[
            "deploy",
            "deployment",
            "ci",
            "cd",
            "kubernetes",
            "docker",
            "container",
            "terraform",
            "aws",
            "gcp",
            "azure",
            "cloud",
            "infrastructure",
            "release",
            "rollout",
            "canary",
            "blue-green",
            "rollback",
        ],
    },
    TaxonomyEntry {
        id: "observability",
        name: "Observability",
        keywords: &[
            "monitoring",
            "observability",
            "logging",
            "logs",
            "metrics",
            "tracing",
            "alert",
            "alerts",
            "alerting",
            "dashboard",
            "prometheus",
            "grafana",
            "datadog",
            "sentry",
            "on-call",
            "on call",
            "slo",
            "sla",
        ],
    },
    TaxonomyEntry {
        id: "security",
        name: "Security",
        keywords: &[
            "security",
            "auth",
            "authentication",
            "authorization",
            "oauth",
            "encryption",
            "vulnerability",
            "permissions",
            "access control",
            "secrets",
            "compliance",
            "gdpr",
            "pii",
            "xss",
            "injection",
            "threat",
        ],
    },
    TaxonomyEntry {
        id: "testing",
        name: "Testing & quality",
        keywords: &[
            "test",
            "tests",
            "testing",
            "unit test",
            "integration test",
            "end-to-end",
            "e2e",
            "qa",
            "coverage",
            "tdd",
            "regression",
            "code review",
            "quality",
        ],
    },
    TaxonomyEntry {
        id: "reliability",
        name: "Reliability & failure handling",
        keywords: &[
            "failure",
            "failures",
            "failover",
            "retry",
            "retries",
            "outage",
            "resilience",
            "redundancy",
            "backup",
            "disaster recovery",
            "circuit breaker",
            "idempotent",
            "idempotency",
            "fault tolerance",
            "downtime",
            "goes down",
            "dead letter",
        ],
    },
    TaxonomyEntry {
        id: "collaboration",
        name: "Collaboration & behavioral",
        keywords: &[
            "team",
            "teammate",
            "conflict",
            "disagree",
            "disagreement",
            "communication",
            "stakeholder",
            "stakeholders",
            "mentor",
            "feedback",
            "manager",
            "collaborate",
            "collaboration",
        ],
    },
    TaxonomyEntry {
        id: "product",
        name: "Product & impact",
        keywords: &[
            "customer",
            "customers",
            "user experience",
            "product",
            "business",
            "impact",
            "revenue",
            "kpi",
            "feature",
            "prioritize",
            "prioritization",
            "roadmap",
        ],
    },
];

/// Natural next areas after technical / behavioral discussions.
const TECHNICAL_NEXT: &[&str] = &[
    "reliability",
    "observability",
    "security",
    "testing",
    "deployment",
    "database",
    "performance",
];
const BEHAVIORAL_NEXT: &[&str] = &["collaboration", "product", "background"];
const TECHNICAL_IDS: &[&str] = &[
    "architecture",
    "system-design",
    "database",
    "performance",
    "deployment",
    "observability",
    "security",
    "testing",
    "reliability",
];

#[derive(Debug, Clone)]
struct TrackedTopic {
    topic: Topic,
    keywords: Vec<String>,
    taxonomy: bool,
    /// Ever current.
    was_current: bool,
}

/// Tracks the topics of one meeting.
pub struct TopicTracker {
    meeting_id: String,
    mode: Mode,
    topics: Vec<TrackedTopic>,
    /// (meeting ms, topic index, hits) of recent matches.
    recent: VecDeque<(i64, usize, u32)>,
    current: Option<usize>,
    /// Order in which topics were first mentioned (indices).
    order: Vec<usize>,
}

fn slug(name: &str) -> String {
    let s: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    s.split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn keyword_hits(norm: &str, keywords: &[String]) -> u32 {
    keywords
        .iter()
        .filter(|k| norm.contains(&format!(" {k} ")))
        .count() as u32
}

fn norm_text(text: &str) -> String {
    let lowered: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                ' '
            }
        })
        .collect();
    format!(
        " {} ",
        lowered.split_whitespace().collect::<Vec<_>>().join(" ")
    )
}

impl TopicTracker {
    /// Creates a tracker; `planned` are the user's planned topics (one per entry).
    pub fn new(meeting_id: &str, mode: Mode, planned: &[String]) -> Self {
        let mut topics: Vec<TrackedTopic> = TAXONOMY
            .iter()
            .map(|t| TrackedTopic {
                topic: Topic {
                    id: t.id.to_string(),
                    meeting_id: meeting_id.to_string(),
                    name: t.name.to_string(),
                    status: TopicStatus::Pending,
                    mentions: 0,
                    first_mentioned_at: None,
                    last_mentioned_at: None,
                    planned: false,
                    summary: None,
                },
                keywords: t.keywords.iter().map(|k| k.to_string()).collect(),
                taxonomy: true,
                was_current: false,
            })
            .collect();
        for name in planned {
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            // A planned topic that matches a taxonomy area marks that area.
            let existing = topics.iter().position(|t| {
                t.topic.name.eq_ignore_ascii_case(name)
                    || lexical_similarity(&t.topic.name, name) >= 0.5
                    || t.keywords.iter().any(|k| k.eq_ignore_ascii_case(name))
            });
            match existing {
                Some(i) => {
                    topics[i].topic.planned = true;
                    for tok in content_tokens(name) {
                        if !topics[i].keywords.contains(&tok) {
                            topics[i].keywords.push(tok);
                        }
                    }
                    let lower = name.to_lowercase();
                    if !topics[i].keywords.contains(&lower) {
                        topics[i].keywords.push(lower);
                    }
                }
                None => {
                    let mut keywords = vec![name.to_lowercase()];
                    for w in name.to_lowercase().split_whitespace() {
                        if w.len() >= 4 && !keywords.contains(&w.to_string()) {
                            keywords.push(w.to_string());
                        }
                    }
                    topics.push(TrackedTopic {
                        topic: Topic {
                            id: slug(name),
                            meeting_id: meeting_id.to_string(),
                            name: name.to_string(),
                            status: TopicStatus::Pending,
                            mentions: 0,
                            first_mentioned_at: None,
                            last_mentioned_at: None,
                            planned: true,
                            summary: None,
                        },
                        keywords,
                        taxonomy: false,
                        was_current: false,
                    });
                }
            }
        }
        TopicTracker {
            meeting_id: meeting_id.to_string(),
            mode,
            topics,
            recent: VecDeque::new(),
            current: None,
            order: Vec::new(),
        }
    }

    fn set_current(&mut self, idx: usize) -> bool {
        if self.current == Some(idx) {
            return false;
        }
        if let Some(prev) = self.current {
            self.topics[prev].topic.status = TopicStatus::Covered;
        }
        self.topics[idx].topic.status = TopicStatus::Current;
        self.topics[idx].was_current = true;
        self.current = Some(idx);
        true
    }

    /// Updates from a FINAL segment; true when the visible topic list changed.
    pub fn observe(&mut self, segment: &TranscriptSegment) -> bool {
        if !segment.is_final || segment.text.trim().is_empty() {
            return false;
        }
        let norm = norm_text(&segment.text);
        let at = segment.start_time;
        let mut changed = false;
        for (i, t) in self.topics.iter_mut().enumerate() {
            let hits = keyword_hits(&norm, &t.keywords);
            if hits == 0 {
                continue;
            }
            // In interviewer mode the candidate's speech defines the topic,
            // in candidate mode the interviewer's — but both count.
            let weight = if segment.speaker_role == SpeakerRole::Unknown {
                hits
            } else {
                hits.max(1)
            };
            if t.topic.mentions == 0 {
                t.topic.first_mentioned_at = Some(at);
                self.order.push(i);
                changed = true;
            }
            t.topic.mentions += weight;
            t.topic.last_mentioned_at = Some(at);
            self.recent.push_back((at, i, weight));
        }
        while let Some((t, _, _)) = self.recent.front() {
            if at - t > CURRENT_WINDOW_MS {
                self.recent.pop_front();
            } else {
                break;
            }
        }
        // Topic with the most recent hits becomes current.
        let mut totals: Vec<(usize, u32, i64)> = Vec::new();
        for &(t, i, h) in &self.recent {
            match totals.iter_mut().find(|(idx, _, _)| *idx == i) {
                Some(e) => {
                    e.1 += h;
                    e.2 = e.2.max(t);
                }
                None => totals.push((i, h, t)),
            }
        }
        totals.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        if let Some(&(idx, hits, _)) = totals.first() {
            if hits >= CURRENT_MIN_HITS {
                // Keep the current topic on ties to avoid flapping.
                let current_hits = self
                    .current
                    .and_then(|c| totals.iter().find(|(i, _, _)| *i == c))
                    .map(|e| e.1)
                    .unwrap_or(0);
                if Some(idx) != self.current && hits > current_hits {
                    changed |= self.set_current(idx);
                }
            }
        }
        changed
    }

    fn visible(&self, i: usize) -> bool {
        let t = &self.topics[i];
        t.topic.mentions > 0 || t.topic.planned || t.topic.status != TopicStatus::Pending
    }

    /// Covered (chronological), then current, then pending (planned first).
    pub fn topics(&self) -> Vec<Topic> {
        let mut covered: Vec<&TrackedTopic> = self
            .topics
            .iter()
            .filter(|t| t.topic.status == TopicStatus::Covered)
            .collect();
        covered.sort_by_key(|t| t.topic.first_mentioned_at.unwrap_or(i64::MAX));
        let current = self.current.map(|c| &self.topics[c]);
        let suggested = self.suggested_index();
        let mut pending: Vec<(usize, &TrackedTopic)> = self
            .topics
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                t.topic.status == TopicStatus::Pending
                    && (self.visible(*i) || Some(*i) == suggested)
            })
            .collect();
        pending.sort_by_key(|(i, t)| {
            (
                !t.topic.planned,
                Some(*i) != suggested,
                std::cmp::Reverse(t.topic.mentions),
                *i,
            )
        });
        covered
            .into_iter()
            .chain(current)
            .chain(pending.into_iter().map(|(_, t)| t))
            .map(|t| t.topic.clone())
            .collect()
    }

    /// The topic being discussed right now.
    pub fn current(&self) -> Option<Topic> {
        self.current.map(|c| self.topics[c].topic.clone())
    }

    fn suggested_index(&self) -> Option<usize> {
        let pending = |i: usize| self.topics[i].topic.status == TopicStatus::Pending;
        // 1. Planned topics in the order the user listed them.
        if let Some(i) =
            (0..self.topics.len()).find(|&i| self.topics[i].topic.planned && pending(i))
        {
            return Some(i);
        }
        // 2. Areas that naturally follow the discussion so far.
        let discussed: Vec<&str> = self
            .topics
            .iter()
            .filter(|t| t.taxonomy && t.topic.status != TopicStatus::Pending)
            .map(|t| t.topic.id.as_str())
            .collect();
        if discussed.is_empty() {
            return None;
        }
        let technical = discussed.iter().any(|d| TECHNICAL_IDS.contains(d));
        let order: &[&str] = if technical || self.mode == Mode::Interviewer && discussed.is_empty()
        {
            TECHNICAL_NEXT
        } else {
            BEHAVIORAL_NEXT
        };
        order
            .iter()
            .filter_map(|id| self.topics.iter().position(|t| t.topic.id == *id))
            .find(|&i| pending(i))
    }

    /// Next area to explore, if any.
    pub fn suggested_next(&self) -> Option<String> {
        self.suggested_index()
            .map(|i| self.topics[i].topic.name.clone())
    }

    /// Merges an LLM topic analysis. Unknown names become new topics. Never
    /// downgrades a covered / current topic to pending. True when changed.
    pub fn apply_analysis(&mut self, items: &[(String, TopicStatus)]) -> bool {
        let mut changed = false;
        for (name, status) in items {
            let name = name.trim();
            if name.is_empty() || name.chars().count() > 80 {
                continue;
            }
            let idx = self
                .topics
                .iter()
                .position(|t| {
                    t.topic.name.eq_ignore_ascii_case(name)
                        || t.topic.id == slug(name)
                        || lexical_similarity(&t.topic.name, name) >= 0.6
                })
                .unwrap_or_else(|| {
                    self.topics.push(TrackedTopic {
                        topic: Topic {
                            id: slug(name),
                            meeting_id: self.meeting_id.clone(),
                            name: name.to_string(),
                            status: TopicStatus::Pending,
                            mentions: 0,
                            first_mentioned_at: None,
                            last_mentioned_at: None,
                            planned: false,
                            summary: None,
                        },
                        keywords: vec![name.to_lowercase()],
                        taxonomy: false,
                        was_current: false,
                    });
                    changed = true;
                    self.topics.len() - 1
                });
            match status {
                TopicStatus::Current => changed |= self.set_current(idx),
                TopicStatus::Covered => {
                    if self.topics[idx].topic.status == TopicStatus::Pending {
                        self.topics[idx].topic.status = TopicStatus::Covered;
                        changed = true;
                    }
                }
                TopicStatus::Pending => {
                    // Mentioned as worth exploring: make sure it is visible.
                    let t = &self.topics[idx].topic;
                    if t.status == TopicStatus::Pending && t.mentions == 0 && !t.planned {
                        self.topics[idx].topic.mentions = 1;
                        changed = true;
                    }
                }
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::types::AudioSourceKind;

    fn seg(at: i64, role: SpeakerRole, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: format!("s{at}"),
            meeting_id: "m".into(),
            utterance_id: format!("u{at}"),
            speaker_id: None,
            speaker_role: role,
            source: AudioSourceKind::System,
            start_time: at,
            end_time: at + 2000,
            text: text.into(),
            is_final: true,
            confidence: None,
        }
    }

    fn names(t: &[Topic]) -> Vec<(String, TopicStatus)> {
        t.iter().map(|t| (t.name.clone(), t.status)).collect()
    }

    #[test]
    fn current_then_covered() {
        let mut tr = TopicTracker::new("m", Mode::Interviewer, &[]);
        assert!(tr.topics().is_empty());
        assert!(tr.observe(&seg(
            0,
            SpeakerRole::Remote,
            "We split the monolith into microservices behind an API gateway."
        )));
        assert_eq!(tr.current().unwrap().id, "architecture");
        tr.observe(&seg(
            70_000,
            SpeakerRole::Remote,
            "The database was Postgres with a read replica and careful schema migrations.",
        ));
        assert_eq!(tr.current().unwrap().id, "database");
        let topics = tr.topics();
        assert_eq!(topics[0].id, "architecture");
        assert_eq!(topics[0].status, TopicStatus::Covered);
        assert_eq!(topics[1].status, TopicStatus::Current);
        assert!(topics[0].first_mentioned_at.unwrap() < topics[1].first_mentioned_at.unwrap());
    }

    #[test]
    fn single_mentions_stay_pending_and_suggestion_follows_discussion() {
        let mut tr = TopicTracker::new("m", Mode::Interviewer, &[]);
        tr.observe(&seg(
            0,
            SpeakerRole::Remote,
            "We used caching and tuned latency until the bottleneck moved.",
        ));
        tr.observe(&seg(5_000, SpeakerRole::User, "Did you add any tests?"));
        let t = tr.topics();
        let testing = t.iter().find(|t| t.id == "testing").unwrap();
        assert_eq!(testing.status, TopicStatus::Pending);
        assert_eq!(tr.current().unwrap().id, "performance");
        assert_eq!(
            tr.suggested_next().as_deref(),
            Some("Reliability & failure handling")
        );
        assert!(t.iter().any(
            |x| x.name == "Reliability & failure handling" && x.status == TopicStatus::Pending
        ));
    }

    #[test]
    fn planned_topics_come_first() {
        let planned = vec!["Observability".to_string(), "Kafka internals".to_string()];
        let mut tr = TopicTracker::new("m", Mode::Interviewer, &planned);
        assert_eq!(tr.suggested_next().as_deref(), Some("Observability"));
        let t = tr.topics();
        assert!(t.iter().any(|x| x.name == "Observability" && x.planned));
        assert!(t.iter().any(|x| x.name == "Kafka internals" && x.planned));
        tr.observe(&seg(
            0,
            SpeakerRole::Remote,
            "Our monitoring used Prometheus metrics and Grafana dashboards with alerting.",
        ));
        assert_eq!(tr.current().unwrap().id, "observability");
        assert_eq!(tr.suggested_next().as_deref(), Some("Kafka internals"));
        tr.observe(&seg(
            70_000,
            SpeakerRole::Remote,
            "Kafka internals: partitions, consumer groups and kafka internals like offsets.",
        ));
        assert_eq!(tr.current().unwrap().name, "Kafka internals");
    }

    #[test]
    fn analysis_merges_and_never_downgrades() {
        let mut tr = TopicTracker::new("m", Mode::Candidate, &[]);
        tr.observe(&seg(
            0,
            SpeakerRole::Remote,
            "Tell me about your security practices: authentication, authorization and secrets.",
        ));
        assert_eq!(tr.current().unwrap().id, "security");
        assert!(tr.apply_analysis(&[
            ("Security".into(), TopicStatus::Pending),
            ("Hiring process".into(), TopicStatus::Covered),
            ("Testing & quality".into(), TopicStatus::Current),
        ]));
        let t = names(&tr.topics());
        assert!(
            t.contains(&("Security".into(), TopicStatus::Covered)),
            "{t:?}"
        );
        assert!(
            t.contains(&("Hiring process".into(), TopicStatus::Covered)),
            "{t:?}"
        );
        assert_eq!(tr.current().unwrap().id, "testing");
        assert!(!tr.apply_analysis(&[("Hiring process".into(), TopicStatus::Pending)]));
    }

    #[test]
    fn no_flapping_on_ties() {
        let mut tr = TopicTracker::new("m", Mode::Interviewer, &[]);
        tr.observe(&seg(
            0,
            SpeakerRole::Remote,
            "Our deployment used Kubernetes and Docker containers.",
        ));
        let first = tr.current().unwrap().id;
        tr.observe(&seg(1_000, SpeakerRole::Remote, "And the database index."));
        assert_eq!(tr.current().unwrap().id, first);
    }
}
