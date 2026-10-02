//! Turns a streaming HTTP response into an [`LlmStream`].
//!
//! The stream reads body chunks, hands them to a provider-specific
//! [`ChunkDecoder`] and yields the decoded items. It ends after the first
//! terminal item (`Done` or an error), when the cancellation token fires
//! (yielding `Err(Cancelled)`), or when no bytes arrive for the idle timeout
//! (`Err(Timeout)`). The response — and with it the connection — is dropped as
//! soon as the stream terminates or is dropped.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openmeet_core::ProviderError;
use openmeet_core::providers::{LlmChunk, LlmStream};
use reqwest::Response;
use tokio_util::sync::CancellationToken;

use crate::http::{error_kind, transport_error};

/// Queue of decoded items.
pub(crate) type ChunkQueue = VecDeque<Result<LlmChunk, ProviderError>>;

/// Provider-specific decoding of a streamed body.
pub(crate) trait ChunkDecoder: Send + 'static {
    /// Decodes one body chunk, appending items to `out`.
    fn decode(&mut self, bytes: &[u8], out: &mut ChunkQueue);
    /// End of body. Should append `Done` when the provider finished
    /// normally; appending nothing makes the stream fail with "ended
    /// unexpectedly".
    fn finish(&mut self, out: &mut ChunkQueue);
}

/// Identification for logs (never content).
#[derive(Debug, Clone)]
pub(crate) struct StreamMeta {
    pub provider: &'static str,
    pub request_id: String,
    pub host: String,
    pub started: Instant,
}

struct State<D> {
    response: Option<Response>,
    decoder: D,
    queue: ChunkQueue,
    cancel: CancellationToken,
    idle: Duration,
    finished: bool,
    meta: StreamMeta,
    deltas: usize,
    chars: usize,
    bytes: usize,
}

/// Wraps a successful streaming response.
pub(crate) fn into_llm_stream<D: ChunkDecoder>(
    response: Response,
    decoder: D,
    cancel: CancellationToken,
    idle: Duration,
    meta: StreamMeta,
) -> LlmStream {
    let state = State {
        response: Some(response),
        decoder,
        queue: ChunkQueue::new(),
        cancel,
        idle,
        finished: false,
        meta,
        deltas: 0,
        chars: 0,
        bytes: 0,
    };
    futures::stream::unfold(state, |mut state| async move {
        let item = state.next_item().await?;
        Some((item, state))
    })
    .boxed()
}

/// A stream over already-decoded items (non-streaming fallbacks).
pub(crate) fn from_items(items: Vec<Result<LlmChunk, ProviderError>>) -> LlmStream {
    futures::stream::iter(items).boxed()
}

impl<D: ChunkDecoder> State<D> {
    async fn next_item(&mut self) -> Option<Result<LlmChunk, ProviderError>> {
        loop {
            if self.finished {
                return None;
            }
            if self.cancel.is_cancelled() {
                return Some(self.terminate(Err(ProviderError::Cancelled)));
            }
            if let Some(item) = self.queue.pop_front() {
                match item {
                    Ok(LlmChunk::Delta(text)) => {
                        if text.is_empty() {
                            continue;
                        }
                        self.deltas += 1;
                        self.chars += text.chars().count();
                        return Some(Ok(LlmChunk::Delta(text)));
                    }
                    terminal => return Some(self.terminate(terminal)),
                }
            }
            let Some(response) = self.response.as_mut() else {
                let host = self.meta.host.clone();
                return Some(self.terminate(Err(ProviderError::Network(format!(
                    "the response stream from {host} ended before the reply was complete"
                )))));
            };
            let read = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => None,
                r = tokio::time::timeout(self.idle, response.chunk()) => Some(r),
            };
            match read {
                None => return Some(self.terminate(Err(ProviderError::Cancelled))),
                Some(Err(_elapsed)) => return Some(self.terminate(Err(ProviderError::Timeout))),
                Some(Ok(Ok(Some(bytes)))) => {
                    self.bytes += bytes.len();
                    self.decoder.decode(&bytes, &mut self.queue);
                }
                Some(Ok(Ok(None))) => {
                    self.response = None;
                    self.decoder.finish(&mut self.queue);
                }
                Some(Ok(Err(e))) => {
                    let err = transport_error(e, &self.meta.host);
                    return Some(self.terminate(Err(err)));
                }
            }
        }
    }

    fn terminate(
        &mut self,
        item: Result<LlmChunk, ProviderError>,
    ) -> Result<LlmChunk, ProviderError> {
        self.finished = true;
        self.response = None;
        self.queue.clear();
        let elapsed_ms = self.meta.started.elapsed().as_millis() as u64;
        match &item {
            Ok(LlmChunk::Done { stop_reason, usage }) => tracing::debug!(
                provider = self.meta.provider,
                request_id = %self.meta.request_id,
                deltas = self.deltas,
                chars = self.chars,
                bytes = self.bytes,
                elapsed_ms,
                stop_reason = stop_reason.as_deref().unwrap_or("none"),
                input_tokens = usage.map(|u| u.input_tokens),
                output_tokens = usage.map(|u| u.output_tokens),
                "LLM stream completed"
            ),
            Ok(LlmChunk::Delta(_)) => {}
            Err(ProviderError::Cancelled) => tracing::debug!(
                provider = self.meta.provider,
                request_id = %self.meta.request_id,
                deltas = self.deltas,
                elapsed_ms,
                "LLM stream cancelled"
            ),
            Err(e) => tracing::warn!(
                provider = self.meta.provider,
                request_id = %self.meta.request_id,
                deltas = self.deltas,
                elapsed_ms,
                error_kind = error_kind(e),
                "LLM stream failed"
            ),
        }
        item
    }
}
