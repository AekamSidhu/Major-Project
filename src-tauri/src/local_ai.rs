//! Manages the optional local AI sidecar (faster-whisper transcription and
//! fastembed embeddings in a Python virtual environment, see `local-ai/`).
//!
//! Security: the executable and script paths are resolved here from fixed
//! locations (environment override, the development checkout, or the bundled
//! resources) — never from the UI — and the server only binds to loopback.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use openmeet_core::ipc::LocalAiStatus;
use openmeet_core::providers::location_for_url;
use openmeet_core::settings::{
    DEFAULT_LOCAL_AI_URL, EmbeddingProviderKind, Settings, TranscriptionProviderKind,
};
use openmeet_core::types::ProviderLocation;
use openmeet_core::CoreError;
use openmeet_providers::local_ai::LocalAiClient;
use parking_lot::Mutex;
use tauri::{AppHandle, Manager};
use tokio_util::sync::CancellationToken;

const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);
const START_TIMEOUT: Duration = Duration::from_secs(25);

pub struct LocalAiManager {
    dir: Option<PathBuf>,
    child: Mutex<Option<Child>>,
}

#[cfg(windows)]
fn venv_python(dir: &Path) -> PathBuf {
    dir.join(".venv").join("Scripts").join("python.exe")
}

#[cfg(not(windows))]
fn venv_python(dir: &Path) -> PathBuf {
    dir.join(".venv").join("bin").join("python")
}

fn setup_hint() -> String {
    if cfg!(windows) {
        "Run scripts\\setup-local-ai.ps1 (creates local-ai\\.venv and installs faster-whisper + fastembed).".into()
    } else {
        "Run scripts/setup-local-ai.sh (creates local-ai/.venv and installs faster-whisper + fastembed).".into()
    }
}

/// Base URL used for the sidecar: the configured transcription / embedding
/// URL when that provider is the local sidecar, else the default.
pub fn sidecar_url(settings: &Settings) -> String {
    if settings.transcription.provider == TranscriptionProviderKind::LocalAi {
        if let Some(u) = &settings.transcription.base_url {
            return u.clone();
        }
    }
    if settings.ai.embeddings.provider == EmbeddingProviderKind::LocalAi {
        if let Some(u) = &settings.ai.embeddings.base_url {
            return u.clone();
        }
    }
    DEFAULT_LOCAL_AI_URL.to_string()
}

/// Whether the current settings use the sidecar at all.
pub fn sidecar_in_use(settings: &Settings) -> bool {
    settings.transcription.provider == TranscriptionProviderKind::LocalAi
        || settings.ai.embeddings.provider == EmbeddingProviderKind::LocalAi
}

fn port_of(url: &str) -> Option<u16> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next()?;
    authority.rsplit_once(':').and_then(|(_, p)| p.parse().ok())
}

impl LocalAiManager {
    pub fn new(app: &AppHandle) -> Self {
        let candidates: Vec<PathBuf> = [
            std::env::var_os("OPENMEET_LOCAL_AI_DIR").map(PathBuf::from),
            // Development checkout (compile-time path of this crate).
            Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("local-ai")),
            app.path().resource_dir().ok().map(|d| d.join("local-ai")),
        ]
        .into_iter()
        .flatten()
        .collect();
        let dir = candidates
            .into_iter()
            .find(|d| d.join("server.py").is_file())
            .and_then(|d| d.canonicalize().ok().or(Some(d)));
        LocalAiManager {
            dir,
            child: Mutex::new(None),
        }
    }

    fn installed(&self) -> bool {
        self.dir.as_ref().is_some_and(|d| venv_python(d).is_file())
    }

    fn child_alive(&self) -> bool {
        let mut child = self.child.lock();
        match child.as_mut() {
            Some(c) => match c.try_wait() {
                Ok(None) => true,
                _ => {
                    *child = None;
                    false
                }
            },
            None => false,
        }
    }

    pub async fn status(&self, settings: &Settings) -> LocalAiStatus {
        let base_url = sidecar_url(settings);
        let installed = self.installed();
        let managed = self.child_alive();
        let health = tokio::time::timeout(
            HEALTH_TIMEOUT,
            LocalAiClient::new(&base_url).health(CancellationToken::new()),
        )
        .await;
        let (running, transcription_model, embedding_model, message) = match health {
            Ok(Ok(h)) if h.ok => (
                true,
                h.transcription_model,
                h.embedding_model,
                "Local AI is running on this device.".to_string(),
            ),
            Ok(Ok(_)) => (false, None, None, "Local AI responded but is not healthy.".to_string()),
            Ok(Err(_)) | Err(_) => (
                false,
                None,
                None,
                if !installed {
                    "Local AI is not installed.".to_string()
                } else if managed {
                    "Local AI is starting…".to_string()
                } else {
                    "Local AI is not running.".to_string()
                },
            ),
        };
        LocalAiStatus {
            installed,
            running,
            managed,
            base_url,
            transcription_model,
            embedding_model,
            message,
            setup_hint: setup_hint(),
        }
    }

    /// Starts the sidecar (if installed and not already running) and waits
    /// until it answers its health check.
    pub async fn start(&self, settings: &Settings) -> Result<LocalAiStatus, CoreError> {
        let url = sidecar_url(settings);
        if location_for_url(&url) != ProviderLocation::Local {
            return Err(CoreError::InvalidState(
                "The local AI URL is not on this device; OpenMeet only starts the sidecar on localhost.".into(),
            ));
        }
        let current = self.status(settings).await;
        if current.running {
            return Ok(current);
        }
        let dir = self
            .dir
            .clone()
            .ok_or_else(|| CoreError::NotFound(format!("Local AI files were not found. {}", setup_hint())))?;
        let python = venv_python(&dir);
        if !python.is_file() {
            return Err(CoreError::NotFound(format!("Local AI is not installed. {}", setup_hint())));
        }
        if !self.child_alive() {
            let port = port_of(&url).unwrap_or(8765);
            let mut cmd = Command::new(&python);
            cmd.arg(dir.join("server.py"))
                .arg("--host")
                .arg("127.0.0.1")
                .arg("--port")
                .arg(port.to_string())
                .current_dir(&dir)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if let Some(model) = Some(settings.transcription.model.trim())
                .filter(|m| !m.is_empty() && settings.transcription.provider == TranscriptionProviderKind::LocalAi)
            {
                cmd.arg("--whisper-model").arg(model);
            }
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }
            let child = cmd.spawn().map_err(|e| {
                tracing::error!(error = %e, "failed to start local AI");
                CoreError::Platform("Could not start the local AI server.".into())
            })?;
            tracing::info!(pid = child.id(), port, "local AI started");
            *self.child.lock() = Some(child);
        }
        let deadline = tokio::time::Instant::now() + START_TIMEOUT;
        loop {
            let s = self.status(settings).await;
            if s.running {
                return Ok(s);
            }
            if !self.child_alive() {
                return Err(CoreError::Platform(
                    "The local AI server exited during start-up. Re-run the setup script and check its output.".into(),
                ));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(CoreError::Platform("The local AI server did not become ready in time.".into()));
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
    }

    /// Stops the sidecar if OpenMeet started it.
    pub fn stop(&self) {
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
            let _ = child.wait();
            tracing::info!("local AI stopped");
        }
    }
}

impl Drop for LocalAiManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ports() {
        assert_eq!(port_of("http://127.0.0.1:8765"), Some(8765));
        assert_eq!(port_of("http://localhost:9000/v1"), Some(9000));
        assert_eq!(port_of("http://localhost/v1"), None);
    }

    #[test]
    fn sidecar_url_follows_settings() {
        let mut s = Settings::default();
        assert_eq!(sidecar_url(&s), DEFAULT_LOCAL_AI_URL);
        s.transcription.base_url = Some("http://127.0.0.1:9999".into());
        assert_eq!(sidecar_url(&s), "http://127.0.0.1:9999");
        assert!(sidecar_in_use(&s));
        s.transcription.provider = TranscriptionProviderKind::Openai;
        assert!(!sidecar_in_use(&s));
    }
}
