use openmeet_core::CoreError;
use openmeet_core::settings::OverlayPosition;

use super::{AppStateRef, CmdResult};

#[tauri::command]
pub async fn overlay_show(state: AppStateRef<'_>) -> CmdResult<bool> {
    Ok(state.overlay.show(&state.settings().overlay))
}

#[tauri::command]
pub async fn overlay_hide(state: AppStateRef<'_>) -> CmdResult<bool> {
    state.overlay.hide();
    Ok(false)
}

#[tauri::command]
pub async fn overlay_toggle(state: AppStateRef<'_>) -> CmdResult<bool> {
    Ok(state.overlay.toggle(&state.settings().overlay))
}

#[tauri::command]
pub async fn set_click_through(state: AppStateRef<'_>, enabled: bool) -> CmdResult<bool> {
    state
        .inner()
        .modify_settings(|s| s.overlay.click_through = enabled)
        .await?;
    Ok(enabled)
}

#[tauri::command]
pub async fn overlay_set_size(state: AppStateRef<'_>, width: f64, height: f64) -> CmdResult<()> {
    if !(width.is_finite() && height.is_finite())
        || width <= 0.0
        || height <= 0.0
        || width > 10_000.0
        || height > 10_000.0
    {
        return Err(CoreError::InvalidInput("Invalid overlay size.".into()));
    }
    state
        .overlay
        .set_content_size(width, height, &state.settings().overlay);
    Ok(())
}

#[tauri::command]
pub async fn overlay_set_position(
    state: AppStateRef<'_>,
    position: OverlayPosition,
) -> CmdResult<()> {
    if position == OverlayPosition::Custom {
        // "Custom" means: remember where the overlay is now.
        return overlay_save_position(state).await;
    }
    let settings = state
        .inner()
        .modify_settings(|s| s.overlay.position = position)
        .await?;
    state.overlay.apply_layout(&settings.overlay);
    Ok(())
}

/// Remembers the current (dragged) position for the display / meeting app.
#[tauri::command]
pub async fn overlay_save_position(state: AppStateRef<'_>) -> CmdResult<()> {
    let (display_key, key, point) = state
        .overlay
        .current_relative_position()
        .ok_or_else(|| CoreError::InvalidState("The overlay position is unavailable.".into()))?;
    state
        .inner()
        .modify_settings(|s| {
            s.overlay.position = OverlayPosition::Custom;
            s.overlay.target_display = Some(display_key.clone());
            s.overlay.custom_positions.insert(key.clone(), point);
            // Also remember it for the display in general.
            s.overlay
                .custom_positions
                .entry(display_key.clone())
                .or_insert(point);
            // Bound the map (displays and apps come and go).
            while s.overlay.custom_positions.len() > 32 {
                let first = s.overlay.custom_positions.keys().next().cloned();
                match first {
                    Some(k) if k != key && k != display_key => {
                        s.overlay.custom_positions.remove(&k);
                    }
                    _ => break,
                }
            }
        })
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn overlay_move_to_display(state: AppStateRef<'_>, display_key: String) -> CmdResult<()> {
    let displays = crate::windows::list_displays(&state.app);
    if !displays.iter().any(|d| d.key == display_key) {
        return Err(CoreError::NotFound("That display is not connected.".into()));
    }
    let settings = state
        .inner()
        .modify_settings(|s| s.overlay.target_display = Some(display_key.clone()))
        .await?;
    state.overlay.apply_layout(&settings.overlay);
    Ok(())
}

#[tauri::command]
pub async fn overlay_begin_input(state: AppStateRef<'_>) -> CmdResult<()> {
    state.overlay.begin_input();
    Ok(())
}

#[tauri::command]
pub async fn overlay_end_input(state: AppStateRef<'_>) -> CmdResult<()> {
    state.overlay.end_input();
    Ok(())
}
