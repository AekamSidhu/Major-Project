use std::str::FromStr;

use openmeet_core::ipc::ShortcutStatus;
use openmeet_core::settings::{ShortcutAction, ShortcutSettings};
use openmeet_core::CoreError;
use tauri_plugin_global_shortcut::Shortcut;

use super::{AppStateRef, CmdResult};

#[tauri::command]
pub async fn get_shortcut_status(state: AppStateRef<'_>) -> CmdResult<Vec<ShortcutStatus>> {
    Ok(state.shortcuts.status())
}

/// Validates an accelerator such as "CommandOrControl+Shift+K". An empty
/// string unassigns the action.
pub fn validate_accelerator(acc: &str) -> CmdResult<String> {
    let acc = acc.trim();
    if acc.is_empty() {
        return Ok(String::new());
    }
    if acc.len() > 64 {
        return Err(CoreError::InvalidInput("The shortcut is too long.".into()));
    }
    let shortcut = Shortcut::from_str(acc).map_err(|_| CoreError::InvalidInput(format!("'{acc}' is not a valid shortcut.")))?;
    if shortcut.mods.is_empty() {
        return Err(CoreError::InvalidInput(
            "Global shortcuts need at least one modifier (Ctrl/Cmd, Alt or Shift) so they don't block normal typing.".into(),
        ));
    }
    Ok(acc.to_string())
}

#[tauri::command]
pub async fn update_shortcut(state: AppStateRef<'_>, action: ShortcutAction, accelerator: String) -> CmdResult<Vec<ShortcutStatus>> {
    let acc = validate_accelerator(&accelerator)?;
    let current = state.settings();
    let mut probe = current.shortcuts.clone();
    probe.bindings.insert(action, acc.clone());
    if let Some((a, b)) = probe.duplicates().into_iter().find(|(a, b)| *a == action || *b == action) {
        let other = if a == action { b } else { a };
        return Err(CoreError::InvalidInput(format!(
            "These keys are already used for \"{}\".",
            other.label()
        )));
    }
    state
        .inner()
        .modify_settings(|s| {
            s.shortcuts.bindings.insert(action, acc.clone());
        })
        .await?;
    Ok(state.shortcuts.status())
}

#[tauri::command]
pub async fn reset_shortcuts(state: AppStateRef<'_>) -> CmdResult<Vec<ShortcutStatus>> {
    state
        .inner()
        .modify_settings(|s| s.shortcuts = ShortcutSettings::default())
        .await?;
    Ok(state.shortcuts.status())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerators() {
        assert_eq!(validate_accelerator(" CommandOrControl+Shift+K ").unwrap(), "CommandOrControl+Shift+K");
        assert_eq!(validate_accelerator("").unwrap(), "");
        assert!(validate_accelerator("K").is_err(), "needs a modifier");
        assert!(validate_accelerator("Ctrl+Shift+NotAKey").is_err());
        for action in ShortcutAction::ALL {
            assert!(validate_accelerator(action.default_accelerator()).is_ok(), "{action:?}");
        }
    }
}
