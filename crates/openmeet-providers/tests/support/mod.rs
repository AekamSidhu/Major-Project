//! Minimal HTTP/1.1 mock server for adapter tests (no network beyond
//! loopback): records every request and answers with scripted replies, which
//! can be sent in delayed chunks or dripped forever to test cancellation.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use openmeet_core::ProviderError;
use openmeet_core::providers::{LlmChunk, LlmStream};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// A request as received by the server.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    /// Path including the query string.
    pub path: String,
    /// Header names are lower-cased.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("request body is JSON")
    }

    pub fn multipart(&self) -> Vec<FormPart> {
        let ct = self.header("content-type").expect("content-type");
        let boundary = ct
            .split(';')
            .map(str::trim)
            .find_map(|p| p.strip_prefix("boundary="))
            .expect("boundary")
            .trim_matches('"')
            .to_string();
        parse_multipart(&self.body, &boundary)
    }
}

/// One multipart/form-data part.
#[derive(Debug, Clone)]
pub struct FormPart {
    pub name: String,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub data: Vec<u8>,
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (from..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

fn parse_multipart(body: &[u8], boundary: &str) -> Vec<FormPart> {
    let delimiter = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut pos = find(body, delimiter.as_bytes(), 0).expect("first boundary") + delimiter.len();
    loop {
        if body[pos..].starts_with(b"--") {
            break;
        }
        pos += 2; // CRLF after the boundary
        let head_end = find(body, b"\r\n\r\n", pos).expect("part head");
        let head = String::from_utf8_lossy(&body[pos..head_end]).to_string();
        let data_start = head_end + 4;
        let next = find(body, format!("\r\n{delimiter}").as_bytes(), data_start).expect("next");
        let mut name = String::new();
        let mut filename = None;
        let mut content_type = None;
        for line in head.split("\r\n") {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("content-disposition:") {
                for attr in line.split(';').map(str::trim) {
                    if let Some(v) = attr.strip_prefix("name=") {
                        name = v.trim_matches('"').to_string();
                    } else if let Some(v) = attr.strip_prefix("filename=") {
                        filename = Some(v.trim_matches('"').to_string());
                    }
                }
            } else if lower.starts_with("content-type:") {
                content_type = Some(line["content-type:".len()..].trim().to_string());
            }
        }
        parts.push(FormPart {
            name,
            filename,
            content_type,
            data: body[data_start..next].to_vec(),
        });
        pos = next + 2 + delimiter.len();
    }
    parts
}

/// Response body shapes.
#[derive(Debug, Clone)]
pub enum Body {
    /// Sent at once with `Content-Length`.
    Full(Vec<u8>),
    /// Chunked transfer encoding; each chunk is written after its delay.
    Chunks(Vec<(Duration, Vec<u8>)>),
    /// Writes `initial`, then `chunk` every `interval` until the client goes
    /// away.
    Drip {
        initial: Vec<u8>,
        chunk: Vec<u8>,
        interval: Duration,
    },
}

/// A scripted reply.
#[derive(Debug, Clone)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Body,
    /// Delay before anything (status line included) is written.
    pub delay: Duration,
}

impl Reply {
    pub fn new(status: u16, content_type: &str, body: Body) -> Self {
        Reply {
            status,
            headers: vec![("content-type".into(), content_type.into())],
            body,
            delay: Duration::ZERO,
        }
    }

    pub fn json(status: u16, value: Value) -> Self {
        Reply::new(
            status,
            "application/json",
            Body::Full(serde_json::to_vec(&value).unwrap()),
        )
    }

    pub fn text(status: u16, text: &str) -> Self {
        Reply::new(status, "text/plain", Body::Full(text.as_bytes().to_vec()))
    }

    /// `text` split into HTTP chunks of `chunk_size` bytes (arbitrary
    /// boundaries, also inside UTF-8 sequences). Each chunk is written and
    /// flushed separately; the client's HTTP decoder hands every chunk to the
    /// adapter as its own body frame. No sleeps between chunks: timer
    /// granularity (~15 ms on Windows) would make byte-sized chunking slow.
    pub fn chunked(content_type: &str, text: &str, chunk_size: usize) -> Self {
        let chunks = text
            .as_bytes()
            .chunks(chunk_size.max(1))
            .map(|c| (Duration::ZERO, c.to_vec()))
            .collect();
        Reply::new(200, content_type, Body::Chunks(chunks))
    }

    pub fn sse(text: &str, chunk_size: usize) -> Self {
        Reply::chunked("text/event-stream", text, chunk_size)
    }

    pub fn ndjson(text: &str, chunk_size: usize) -> Self {
        Reply::chunked("application/x-ndjson", text, chunk_size)
    }

    pub fn drip(content_type: &str, initial: &str, chunk: &str, interval: Duration) -> Self {
        Reply::new(
            200,
            content_type,
            Body::Drip {
                initial: initial.as_bytes().to_vec(),
                chunk: chunk.as_bytes().to_vec(),
                interval,
            },
        )
    }

    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers
            .push((name.to_ascii_lowercase(), value.to_string()));
        self
    }

    pub fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

type Handler = dyn Fn(&Recorded, usize) -> Reply + Send + Sync;

/// Loopback HTTP server; aborted on drop.
pub struct MockServer {
    pub url: String,
    pub port: u16,
    requests: Arc<Mutex<Vec<Recorded>>>,
    task: JoinHandle<()>,
}

impl MockServer {
    /// Serves every request with `handler(request, index)`.
    pub async fn start<F>(handler: F) -> MockServer
    where
        F: Fn(&Recorded, usize) -> Reply + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let handler: Arc<Handler> = Arc::new(handler);
        let recorded = requests.clone();
        let task = tokio::spawn(async move {
            loop {
                let Ok((socket, _)) = listener.accept().await else {
                    return;
                };
                let handler = handler.clone();
                let recorded = recorded.clone();
                tokio::spawn(async move {
                    let _ = serve(socket, handler, recorded).await;
                });
            }
        });
        MockServer {
            url: format!("http://127.0.0.1:{port}"),
            port,
            requests,
            task,
        }
    }

    /// Answers requests with `replies` in order (the last one repeats).
    pub async fn with_replies(replies: Vec<Reply>) -> MockServer {
        MockServer::start(move |_, i| replies[i.min(replies.len() - 1)].clone()).await
    }

    pub async fn with_reply(reply: Reply) -> MockServer {
        MockServer::with_replies(vec![reply]).await
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    pub fn request(&self, index: usize) -> Recorded {
        self.requests()
            .get(index)
            .cloned()
            .unwrap_or_else(|| panic!("no request #{index}"))
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn read_request(socket: &mut TcpStream) -> std::io::Result<Option<Recorded>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n", 0) {
            break i;
        }
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut it = request_line.split(' ');
    let method = it.next().unwrap_or_default().to_string();
    let path = it.next().unwrap_or_default().to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let get = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    let mut rest = buf[head_end + 4..].to_vec();
    let body = if let Some(len) = get("content-length").and_then(|v| v.parse::<usize>().ok()) {
        while rest.len() < len {
            let n = socket.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            rest.extend_from_slice(&tmp[..n]);
        }
        rest.truncate(len);
        rest
    } else if get("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        let mut body = Vec::new();
        let mut pos = 0;
        loop {
            let line_end = loop {
                if let Some(i) = find(&rest, b"\r\n", pos) {
                    break i;
                }
                let n = socket.read(&mut tmp).await?;
                if n == 0 {
                    return Ok(None);
                }
                rest.extend_from_slice(&tmp[..n]);
            };
            let size_text = String::from_utf8_lossy(&rest[pos..line_end]).to_string();
            let size =
                usize::from_str_radix(size_text.split(';').next().unwrap().trim(), 16).unwrap_or(0);
            let data_start = line_end + 2;
            while rest.len() < data_start + size + 2 {
                let n = socket.read(&mut tmp).await?;
                if n == 0 {
                    return Ok(None);
                }
                rest.extend_from_slice(&tmp[..n]);
            }
            if size == 0 {
                break;
            }
            body.extend_from_slice(&rest[data_start..data_start + size]);
            pos = data_start + size + 2;
        }
        body
    } else {
        Vec::new()
    };
    Ok(Some(Recorded {
        method,
        path,
        headers,
        body,
    }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        529 => "Overloaded",
        _ => "Status",
    }
}

async fn serve(
    mut socket: TcpStream,
    handler: Arc<Handler>,
    recorded: Arc<Mutex<Vec<Recorded>>>,
) -> std::io::Result<()> {
    socket.set_nodelay(true)?;
    let Some(request) = read_request(&mut socket).await? else {
        return Ok(());
    };
    let index = {
        let mut all = recorded.lock().unwrap();
        all.push(request.clone());
        all.len() - 1
    };
    let reply = handler(&request, index);
    if !reply.delay.is_zero() {
        tokio::time::sleep(reply.delay).await;
    }
    let mut head = format!("HTTP/1.1 {} {}\r\n", reply.status, reason(reply.status));
    for (k, v) in &reply.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("connection: close\r\n");
    match &reply.body {
        Body::Full(bytes) => {
            head.push_str(&format!("content-length: {}\r\n\r\n", bytes.len()));
            socket.write_all(head.as_bytes()).await?;
            socket.write_all(bytes).await?;
        }
        Body::Chunks(chunks) => {
            head.push_str("transfer-encoding: chunked\r\n\r\n");
            socket.write_all(head.as_bytes()).await?;
            for (delay, chunk) in chunks {
                if delay.is_zero() {
                    tokio::task::yield_now().await;
                } else {
                    tokio::time::sleep(*delay).await;
                }
                write_chunk(&mut socket, chunk).await?;
            }
            socket.write_all(b"0\r\n\r\n").await?;
        }
        Body::Drip {
            initial,
            chunk,
            interval,
        } => {
            head.push_str("transfer-encoding: chunked\r\n\r\n");
            socket.write_all(head.as_bytes()).await?;
            write_chunk(&mut socket, initial).await?;
            // Bounded so a leaked connection cannot run forever.
            for _ in 0..2000 {
                tokio::time::sleep(*interval).await;
                write_chunk(&mut socket, chunk).await?;
            }
        }
    }
    socket.flush().await?;
    socket.shutdown().await
}

async fn write_chunk(socket: &mut TcpStream, chunk: &[u8]) -> std::io::Result<()> {
    if chunk.is_empty() {
        return Ok(());
    }
    socket
        .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
        .await?;
    socket.write_all(chunk).await?;
    socket.write_all(b"\r\n").await?;
    socket.flush().await
}

/// A loopback port with nothing listening on it.
pub async fn closed_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Collected stream output.
#[derive(Debug, Default)]
pub struct Collected {
    pub text: String,
    pub deltas: usize,
    pub done: Option<LlmChunk>,
    pub error: Option<ProviderError>,
}

/// Drains a stream (with a safety timeout).
pub async fn collect(mut stream: LlmStream) -> Collected {
    let mut out = Collected::default();
    let drain = async {
        while let Some(item) = stream.next().await {
            match item {
                Ok(LlmChunk::Delta(t)) => {
                    out.text.push_str(&t);
                    out.deltas += 1;
                }
                Ok(done @ LlmChunk::Done { .. }) => {
                    assert!(out.done.is_none(), "more than one Done");
                    out.done = Some(done);
                }
                Err(e) => {
                    assert!(out.error.is_none(), "more than one error");
                    out.error = Some(e);
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), drain)
        .await
        .expect("stream did not end");
    out
}
