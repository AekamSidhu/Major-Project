//! Background embedding of memory items (spec §19).
//!
//! Items are batched (up to 16 or 50 ms) and embedded with the configured
//! embedding provider; the built-in hashing embedder is used when none is
//! configured. Failures leave keyword retrieval working and are reported as
//! degraded health — vectors from different models are never mixed.

use std::sync::Weak;
use std::time::Duration;

use openmeet_core::ids::now_ms;
use openmeet_core::store::EmbeddingRecord;
use openmeet_core::types::{Component, ComponentStatus};
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::engine::{Engine, Inner};
use crate::memory::MemoryKind;
use crate::persist::PersistOp;

const BATCH_SIZE: usize = 16;
const BATCH_WINDOW: Duration = Duration::from_millis(50);
const EMBED_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub(crate) struct EmbedJob {
    pub meeting_id: String,
    pub item_id: String,
    pub kind: MemoryKind,
    pub text: String,
}

pub(crate) fn spawn_embedder(
    weak: Weak<Inner>,
    mut rx: mpsc::UnboundedReceiver<EmbedJob>,
    handle: &Handle,
) {
    handle.spawn(async move {
        while let Some(first) = rx.recv().await {
            let mut batch = vec![first];
            let deadline = tokio::time::Instant::now() + BATCH_WINDOW;
            while batch.len() < BATCH_SIZE {
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(job)) => batch.push(job),
                    _ => break,
                }
            }
            let Some(inner) = weak.upgrade() else { break };
            Engine { inner }.embed_batch(batch).await;
        }
    });
}

impl Engine {
    async fn embed_batch(&self, batch: Vec<EmbedJob>) {
        let provider = self.embedder();
        let info = provider.info();
        let texts: Vec<String> = batch.iter().map(|j| j.text.clone()).collect();
        let vectors = if info.id == "local_hash" {
            let h = self.hashing();
            texts.iter().map(|t| h.embed_one(t)).collect::<Vec<_>>()
        } else {
            match tokio::time::timeout(
                EMBED_TIMEOUT,
                provider.embed(texts, CancellationToken::new()),
            )
            .await
            {
                Ok(Ok(v)) if v.len() == batch.len() => {
                    self.inner.health.set(
                        Component::Embeddings,
                        ComponentStatus::Active,
                        Some(info.display_name.clone()),
                    );
                    v
                }
                Ok(Ok(_)) | Ok(Err(_)) | Err(_) => {
                    self.inner.health.set(
                        Component::Embeddings,
                        ComponentStatus::Degraded,
                        Some("Embedding provider unavailable; using keyword retrieval".into()),
                    );
                    return;
                }
            }
        };
        let model = info.model.clone();
        let (store, current_meeting) = {
            let mut st = self.inner.state.lock();
            let Some(rt) = st.runtime.as_mut() else {
                return;
            };
            for (job, v) in batch.iter().zip(vectors.iter()) {
                if job.meeting_id == rt.meeting.id {
                    rt.memory.set_vector(&job.item_id, &model, v.clone());
                }
            }
            (rt.store.clone(), rt.meeting.id.clone())
        };
        let now = now_ms();
        for (job, vector) in batch
            .into_iter()
            .zip(vectors)
            .filter(|(job, _)| job.meeting_id == current_meeting)
        {
            self.persist(
                &store,
                PersistOp::Embedding(EmbeddingRecord {
                    item_id: job.item_id,
                    meeting_id: job.meeting_id,
                    kind: job.kind.store_kind().to_string(),
                    model: model.clone(),
                    vector,
                    created_at: now,
                }),
            );
        }
    }
}
