use openmeet_core::ids::{new_id, now_ms};
use openmeet_core::ipc::{CapturePrivacyStatus, Rect};
use openmeet_core::types::{PrivacyAction, PrivacyRule, PrivacyRuleType};
use openmeet_core::CoreError;

use super::{check_id, AppStateRef, CmdResult};
use crate::state::blocking;

#[tauri::command]
pub async fn get_capture_privacy_status(state: AppStateRef<'_>) -> CmdResult<CapturePrivacyStatus> {
    Ok(state.overlay.capture_privacy_status())
}

#[tauri::command]
pub async fn set_capture_exclusion(state: AppStateRef<'_>, enabled: bool) -> CmdResult<CapturePrivacyStatus> {
    state
        .inner()
        .modify_settings(|s| s.privacy.capture_exclusion = enabled)
        .await?;
    Ok(state.overlay.capture_privacy_status())
}

#[tauri::command]
pub async fn list_privacy_rules(state: AppStateRef<'_>) -> CmdResult<Vec<PrivacyRule>> {
    let db = state.db.clone();
    blocking(move || db.list_privacy_rules().map_err(CoreError::from)).await
}

/// Validates and normalizes a rule value.
pub fn normalize_rule(rule_type: PrivacyRuleType, value: &str) -> CmdResult<String> {
    let v = value.trim();
    if v.is_empty() {
        return Err(CoreError::InvalidInput("The rule value is empty.".into()));
    }
    if v.chars().count() > 200 || v.chars().any(char::is_control) {
        return Err(CoreError::InvalidInput("The rule value is invalid or too long.".into()));
    }
    match rule_type {
        PrivacyRuleType::Region => {
            let r = Rect::parse_csv(v)
                .ok_or_else(|| CoreError::InvalidInput("Regions use the format x,y,width,height (pixels).".into()))?;
            if r.width == 0 || r.height == 0 {
                return Err(CoreError::InvalidInput("Region width and height must be positive.".into()));
            }
            Ok(format!("{},{},{},{}", r.x, r.y, r.width, r.height))
        }
        PrivacyRuleType::Domain => {
            let d = v
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .trim_end_matches('/')
                .to_ascii_lowercase();
            if d.contains(char::is_whitespace) || d.contains('/') {
                return Err(CoreError::InvalidInput("Enter a domain such as bank.example.com.".into()));
            }
            Ok(d)
        }
        PrivacyRuleType::Application | PrivacyRuleType::Window => Ok(v.to_string()),
    }
}

#[tauri::command]
pub async fn add_privacy_rule(state: AppStateRef<'_>, rule_type: PrivacyRuleType, value: String) -> CmdResult<PrivacyRule> {
    let value = normalize_rule(rule_type, &value)?;
    let rule = PrivacyRule {
        id: new_id(),
        rule_type,
        value,
        action: PrivacyAction::Exclude,
        created_at: now_ms(),
    };
    let db = state.db.clone();
    let to_add = rule.clone();
    let rules = blocking(move || {
        db.add_privacy_rule(&to_add).map_err(CoreError::from)?;
        db.list_privacy_rules().map_err(CoreError::from)
    })
    .await?;
    state.capture.update_rules(rules);
    Ok(rule)
}

#[tauri::command]
pub async fn remove_privacy_rule(state: AppStateRef<'_>, id: String) -> CmdResult<()> {
    check_id(&id, "rule")?;
    let db = state.db.clone();
    let rules = blocking(move || {
        db.delete_privacy_rule(&id).map_err(CoreError::from)?;
        db.list_privacy_rules().map_err(CoreError::from)
    })
    .await?;
    state.capture.update_rules(rules);
    Ok(())
}

#[tauri::command]
pub async fn suggested_privacy_apps() -> CmdResult<Vec<String>> {
    Ok(openmeet_platform::screen::rules::default_sensitive_apps()
        .into_iter()
        .map(str::to_string)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_normalization() {
        assert_eq!(normalize_rule(PrivacyRuleType::Region, " 1, 2 ,30,40").unwrap(), "1,2,30,40");
        assert!(normalize_rule(PrivacyRuleType::Region, "1,2,0,40").is_err());
        assert!(normalize_rule(PrivacyRuleType::Region, "nope").is_err());
        assert_eq!(normalize_rule(PrivacyRuleType::Domain, "https://Bank.Example.com/").unwrap(), "bank.example.com");
        assert!(normalize_rule(PrivacyRuleType::Domain, "bank.com/login").is_err());
        assert_eq!(normalize_rule(PrivacyRuleType::Application, " 1Password ").unwrap(), "1Password");
        assert!(normalize_rule(PrivacyRuleType::Window, "  ").is_err());
    }
}
