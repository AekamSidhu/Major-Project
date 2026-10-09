//! Typed Tauri commands. Every command returns `Result<_, CoreError>` so the
//! frontend receives structured errors. Inputs are validated here (spec §58).
//! Keep this list in sync with `build.rs`, `capabilities/*.json` and
//! `src/lib/ipc.ts`.

pub mod app;
pub mod assistant;
pub mod dev;
pub mod devices;
pub mod history;
pub mod meeting;
pub mod overlay;
pub mod privacy;
pub mod providers;
pub mod shortcuts;

use std::sync::Arc;

use openmeet_core::CoreError;
use tauri::State;

use crate::state::AppState;

pub type CmdResult<T> = Result<T, CoreError>;
pub type AppStateRef<'a> = State<'a, Arc<AppState>>;

/// Validates a free-text identifier coming from the UI.
pub fn check_id(id: &str, what: &str) -> CmdResult<()> {
    if id.is_empty() || id.len() > 128 || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(CoreError::InvalidInput(format!("Invalid {what}.")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert!(check_id("0199aa3b5c7d4e2f9a1b2c3d4e5f6a7b", "id").is_ok());
        assert!(check_id("", "id").is_err());
        assert!(check_id("../etc/passwd", "id").is_err());
        assert!(check_id(&"a".repeat(200), "id").is_err());
    }
}
