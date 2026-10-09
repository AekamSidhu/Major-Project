use openmeet_core::ipc::{ExportRequest, ExportResult};
use openmeet_core::types::{MeetingListItem, MeetingRecord, MeetingSummary};
use openmeet_core::CoreError;
use tauri_plugin_dialog::DialogExt;

use super::{check_id, AppStateRef, CmdResult};
use crate::state::blocking;

#[tauri::command]
pub async fn list_meetings(state: AppStateRef<'_>, limit: u32, offset: u32) -> CmdResult<Vec<MeetingListItem>> {
    let limit = limit.clamp(1, 200);
    let db = state.db.clone();
    let mut items = blocking(move || db.list_meetings(limit, offset).map_err(CoreError::from)).await?;
    if offset == 0 {
        if let Some(current) = state.engine.unpersisted_meeting() {
            items.insert(0, current);
        }
    }
    Ok(items)
}

async fn load(state: &AppStateRef<'_>, meeting_id: &str) -> CmdResult<MeetingRecord> {
    if let Some(r) = state.engine.current_record(meeting_id) {
        if !r.meeting.persisted {
            return Ok(r);
        }
    }
    let db = state.db.clone();
    let id = meeting_id.to_string();
    blocking(move || db.load_meeting(&id).map_err(CoreError::from)).await
}

#[tauri::command]
pub async fn get_meeting(state: AppStateRef<'_>, meeting_id: String) -> CmdResult<MeetingRecord> {
    check_id(&meeting_id, "meeting")?;
    load(&state, &meeting_id).await
}

#[tauri::command]
pub async fn delete_meeting(state: AppStateRef<'_>, meeting_id: String) -> CmdResult<()> {
    check_id(&meeting_id, "meeting")?;
    if state.engine.is_live() && state.engine.current_meeting_id().as_deref() == Some(meeting_id.as_str()) {
        return Err(CoreError::InvalidState("End the meeting before deleting it.".into()));
    }
    let db = state.db.clone();
    blocking(move || db.delete_meeting(&meeting_id).map_err(CoreError::from)).await
}

/// Deletes every stored meeting, transcript, summary and embedding.
#[tauri::command]
pub async fn delete_all_data(state: AppStateRef<'_>) -> CmdResult<()> {
    if state.engine.is_live() {
        return Err(CoreError::InvalidState("End the current meeting before deleting all data.".into()));
    }
    state.engine.flush_persistence().await;
    let db = state.db.clone();
    blocking(move || db.delete_all_meeting_data().map_err(CoreError::from)).await?;
    tracing::info!("all meeting data deleted by the user");
    Ok(())
}

/// File-name-safe version of a meeting title.
pub fn safe_file_stem(title: &str) -> String {
    let cleaned: String = title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' { c } else { ' ' })
        .collect();
    let joined = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let stem: String = joined.chars().take(80).collect();
    if stem.is_empty() { "OpenMeet meeting".into() } else { stem }
}

#[tauri::command]
pub async fn export_meeting(state: AppStateRef<'_>, request: ExportRequest) -> CmdResult<ExportResult> {
    check_id(&request.meeting_id, "meeting")?;
    let parts = request.parts;
    if !(parts.summary || parts.transcript || parts.questions || parts.followups) {
        return Err(CoreError::InvalidInput("Choose at least one part to export.".into()));
    }
    let record = load(&state, &request.meeting_id).await?;
    let content = openmeet_intel::export::render(&record, request.format, &parts);
    let ext = request.format.extension();
    let date = {
        let secs = record.meeting.started_at / 1000;
        let days = secs.div_euclid(86_400);
        civil_date(days)
    };
    let file_name = format!("{} {}.{}", safe_file_stem(&record.meeting.title), date, ext);
    let filter_name = match request.format {
        openmeet_core::ipc::ExportFormat::Markdown => "Markdown",
        openmeet_core::ipc::ExportFormat::Json => "JSON",
        openmeet_core::ipc::ExportFormat::Text => "Plain text",
    };
    let dialog = state
        .app
        .dialog()
        .file()
        .set_title("Export meeting")
        .set_file_name(file_name)
        .add_filter(filter_name, &[ext]);
    let chosen = blocking(move || Ok(dialog.blocking_save_file())).await?;
    let Some(path) = chosen else {
        return Ok(ExportResult { path: None, bytes: 0 });
    };
    let path = path
        .into_path()
        .map_err(|_| CoreError::InvalidInput("Choose a file on this computer.".into()))?;
    let bytes = content.len() as u64;
    let display = path.display().to_string();
    blocking(move || {
        std::fs::write(&path, content.as_bytes())
            .map_err(|e| CoreError::Platform(format!("Could not write the file: {e}")))
    })
    .await?;
    Ok(ExportResult {
        path: Some(display),
        bytes,
    })
}

/// Days since 1970-01-01 → "YYYY-MM-DD" (UTC), without a date library.
pub fn civil_date(days: i64) -> String {
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[tauri::command]
pub async fn regenerate_summary(state: AppStateRef<'_>, meeting_id: String) -> CmdResult<MeetingSummary> {
    check_id(&meeting_id, "meeting")?;
    if state.engine.is_live() && state.engine.current_meeting_id().as_deref() == Some(meeting_id.as_str()) {
        return Err(CoreError::InvalidState("The summary is generated when the meeting ends.".into()));
    }
    let record = load(&state, &meeting_id).await?;
    state.engine.summarize_record(record).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_stems() {
        assert_eq!(safe_file_stem("Interview: Backend/Infra?"), "Interview Backend Infra");
        assert_eq!(safe_file_stem("***"), "OpenMeet meeting");
        assert_eq!(safe_file_stem(&"a".repeat(200)).len(), 80);
    }

    #[test]
    fn dates() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(20_729), "2026-10-03");
        assert_eq!(civil_date(11_016), "2000-02-29");
    }
}
