//! Screen-capture privacy rules: create, list, edit, delete and survival of
//! meeting-data deletion.

use openmeet_core::StoreError;
use openmeet_core::settings::RetentionPolicy;
use openmeet_core::types::{PrivacyAction, PrivacyRule, PrivacyRuleType};
use openmeet_store::Database;

use crate::fixtures::*;

#[test]
fn privacy_rules_crud() {
    let db = Database::open_in_memory().unwrap();
    assert!(db.list_privacy_rules().unwrap().is_empty());

    let app = privacy_rule("r-app", PrivacyRuleType::Application, "1Password", NOW + 3);
    let window = privacy_rule(
        "r-window",
        PrivacyRuleType::Window,
        "Private — Online Banking (日本語)",
        NOW + 1,
    );
    let domain = privacy_rule(
        "r-domain",
        PrivacyRuleType::Domain,
        "bank.example.com",
        NOW + 2,
    );
    let region = privacy_rule("r-region", PrivacyRuleType::Region, "0,0,400,300", NOW + 2);
    for rule in [&app, &window, &domain, &region] {
        db.add_privacy_rule(rule).unwrap();
    }
    // Oldest first; equal creation times ordered by id.
    assert_eq!(
        db.list_privacy_rules().unwrap(),
        vec![window.clone(), domain.clone(), region.clone(), app.clone()]
    );

    // Editing a rule is saving it again under the same id.
    let edited = PrivacyRule {
        rule_type: PrivacyRuleType::Window,
        value: "secure.bank.example.com".into(),
        action: PrivacyAction::Exclude,
        ..domain.clone()
    };
    db.add_privacy_rule(&edited).unwrap();
    assert_eq!(
        db.list_privacy_rules().unwrap(),
        vec![window.clone(), edited.clone(), region.clone(), app.clone()]
    );

    db.delete_privacy_rule("r-window").unwrap();
    assert_eq!(db.list_privacy_rules().unwrap(), vec![edited, region, app]);
    assert_eq!(
        db.delete_privacy_rule("r-window"),
        Err(StoreError::NotFound("r-window".into()))
    );
    assert_eq!(
        db.delete_privacy_rule("never-existed"),
        Err(StoreError::NotFound("never-existed".into()))
    );

    for id in ["r-domain", "r-region", "r-app"] {
        db.delete_privacy_rule(id).unwrap();
    }
    assert!(db.list_privacy_rules().unwrap().is_empty());
}

#[test]
fn privacy_rules_survive_reopen_retention_and_meeting_deletion() {
    let dir = tempfile::tempdir().unwrap();
    let rule = privacy_rule("r1", PrivacyRuleType::Application, "Signal", NOW - 90 * DAY);
    {
        let db = open_file_db(dir.path());
        db.add_privacy_rule(&rule).unwrap();
        populate(&db, "old", NOW - 60 * DAY);
        populate(&db, "new", NOW);
        assert_eq!(
            db.apply_retention(RetentionPolicy::SevenDays, NOW).unwrap(),
            1
        );
        db.delete_meeting("new").unwrap();
        assert_eq!(db.apply_retention(RetentionPolicy::None, NOW).unwrap(), 0);
    }
    let db = open_file_db(dir.path());
    assert_eq!(db.list_privacy_rules().unwrap(), vec![rule]);
}
