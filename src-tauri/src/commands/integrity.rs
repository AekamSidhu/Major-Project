//! Answer-integrity settings helpers: is the local AI sidecar ready for the
//! check, and what does the candidate video region contain (preview).

use std::time::Duration;

use base64::Engine as _;
use openmeet_core::CoreError;
use openmeet_core::integrity::IntegrityServiceView;
use openmeet_core::ipc::Rect;
use openmeet_core::settings::{INTEGRITY_REGION_MAX, INTEGRITY_REGION_MIN};
use openmeet_intel::integrity::spec_thresholds;
use openmeet_providers::factory::{self, BuildContext};
use tokio_util::sync::CancellationToken;

use super::{AppStateRef, CmdResult};
use crate::state::blocking;

const STATUS_TIMEOUT: Duration = Duration::from_secs(6);
/// Width of the region preview image.
const PREVIEW_WIDTH: u32 = 480;

/// Whether the sidecar can run the integrity check (works while the check
/// is still off, so the user can see it before turning it on).
#[tauri::command]
pub async fn get_integrity_status(state: AppStateRef<'_>) -> CmdResult<IntegrityServiceView> {
    let mut settings = state.settings();
    settings.integrity.enabled = true;
    let built = factory::build_integrity(&BuildContext {
        settings: &settings,
        secrets: state.secrets.as_ref(),
    });
    let service = match built {
        Ok(Some(s)) => s,
        Ok(None) => return Ok(view_error("The integrity check is not available.".into())),
        Err(e) => return Ok(view_error(e.user_message())),
    };
    let status = match tokio::time::timeout(
        STATUS_TIMEOUT,
        service.status(CancellationToken::new()),
    )
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(_)) | Err(_) => {
            return Ok(view_error(
                "Local AI is not running (it starts with an interviewer meeting).".into(),
            ));
        }
    };
    let mut missing = Vec::new();
    if !status.ai_text_ready {
        missing.push("the AI-written text model");
    }
    if !status.reading_ready {
        missing.push("the reading model (needs onnxruntime)");
    }
    let message = if missing.is_empty() {
        "Ready.".to_string()
    } else {
        format!(
            "Local AI is running but {} is not available.",
            missing.join(" and ")
        )
    };
    Ok(IntegrityServiceView {
        reachable: true,
        ai_text_ready: status.ai_text_ready,
        reading_ready: status.reading_ready,
        embedding_model: status.embedding_model,
        required_embedding_model: required_embedding_model(),
        message,
    })
}

fn view_error(message: String) -> IntegrityServiceView {
    IntegrityServiceView {
        reachable: false,
        ai_text_ready: false,
        reading_ready: false,
        embedding_model: None,
        required_embedding_model: required_embedding_model(),
        message,
    }
}

/// The embedding model the echo-similarity thresholds hold for
/// (`spec/integrity/thresholds.json`).
fn required_embedding_model() -> String {
    spec_thresholds().echo_similarity.embedding_model.clone()
}

/// A PNG preview (data URL) of the candidate video region, captured the way
/// the sidecar will see it. Privacy rules apply; nothing is stored.
#[tauri::command]
pub async fn preview_integrity_region(state: AppStateRef<'_>, region: Rect) -> CmdResult<String> {
    if !(INTEGRITY_REGION_MIN..=INTEGRITY_REGION_MAX).contains(&region.width)
        || !(INTEGRITY_REGION_MIN..=INTEGRITY_REGION_MAX).contains(&region.height)
    {
        return Err(CoreError::InvalidInput(format!(
            "The region must be between {INTEGRITY_REGION_MIN} and {INTEGRITY_REGION_MAX} pixels wide and high."
        )));
    }
    let db = state.db.clone();
    let frame = blocking(move || {
        let rules = db.list_privacy_rules().map_err(CoreError::from)?;
        openmeet_platform::screen::capture_rect(region, &rules, PREVIEW_WIDTH)
            .map_err(CoreError::from)
    })
    .await?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&frame.png)
    ))
}
