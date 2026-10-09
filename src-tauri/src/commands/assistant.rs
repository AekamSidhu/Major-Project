use openmeet_core::ipc::GenerateAnswerRequest;
use openmeet_core::types::{AnswerStyle, Mode, Refinement};
use openmeet_core::CoreError;

use super::{check_id, AppStateRef, CmdResult};

const MAX_QUESTION_CHARS: usize = 4000;

#[tauri::command]
pub async fn generate_answer(state: AppStateRef<'_>, request: GenerateAnswerRequest) -> CmdResult<String> {
    if request.question.as_ref().is_some_and(|q| q.chars().count() > MAX_QUESTION_CHARS) {
        return Err(CoreError::InvalidInput("The question is too long.".into()));
    }
    state.engine.generate_answer(request)
}

#[tauri::command]
pub async fn generate_followups(state: AppStateRef<'_>) -> CmdResult<String> {
    state.engine.generate_followups()
}

#[tauri::command]
pub async fn ask_openmeet(state: AppStateRef<'_>, prompt: String) -> CmdResult<String> {
    state.engine.ask(prompt)
}

#[tauri::command]
pub async fn clarify_question(state: AppStateRef<'_>) -> CmdResult<String> {
    state.engine.clarify()
}

#[tauri::command]
pub async fn refine_answer(state: AppStateRef<'_>, request_id: String, refinement: Refinement) -> CmdResult<String> {
    check_id(&request_id, "request")?;
    state.engine.refine(&request_id, refinement)
}

#[tauri::command]
pub async fn cancel_request(state: AppStateRef<'_>, request_id: String) -> CmdResult<()> {
    check_id(&request_id, "request")?;
    state.engine.cancel_request(&request_id);
    Ok(())
}

#[tauri::command]
pub async fn dismiss_suggestion(state: AppStateRef<'_>, request_id: String) -> CmdResult<()> {
    check_id(&request_id, "request")?;
    state.engine.dismiss(&request_id);
    Ok(())
}

#[tauri::command]
pub async fn pin_suggestion(state: AppStateRef<'_>, request_id: String, pinned: bool) -> CmdResult<()> {
    check_id(&request_id, "request")?;
    state.engine.pin(&request_id, pinned)
}

#[tauri::command]
pub async fn set_mode(state: AppStateRef<'_>, mode: Mode) -> CmdResult<()> {
    state.engine.set_mode(mode);
    state.inner().modify_settings(|s| s.mode = mode).await.map(|_| ())
}

#[tauri::command]
pub async fn set_answer_style(state: AppStateRef<'_>, style: AnswerStyle) -> CmdResult<()> {
    state.engine.set_style(style);
    state
        .inner()
        .modify_settings(|s| s.assistant.answer_style = style)
        .await
        .map(|_| ())
}
