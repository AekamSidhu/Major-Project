//! Privacy rule evaluation for screen capture (pure, no OS access).
//!
//! Matching is deliberately biased towards privacy: when in doubt a window is
//! excluded (a skipped frame costs a little context; a leaked frame can expose
//! a password manager or a bank page to an AI provider).

use image::{Rgba, RgbaImage};
use openmeet_core::ipc::Rect;
use openmeet_core::types::{PrivacyRule, PrivacyRuleType};

/// Browser process / application names (normalized, see [`normalize_app`]).
const BROWSERS: &[&str] = &[
    "chrome",
    "google chrome",
    "chromium",
    "msedge",
    "microsoft edge",
    "firefox",
    "mozilla firefox",
    "firefox developer edition",
    "brave",
    "brave browser",
    "opera",
    "opera gx",
    "vivaldi",
    "arc",
    "safari",
    "zen",
    "zen browser",
    "librewolf",
    "waterfox",
    "yandex",
    "duckduckgo",
];

/// Window-title suffix markers appended by browsers (" - Google Chrome").
const BROWSER_TITLE_MARKERS: &[&str] = &[
    "google chrome",
    "chromium",
    "microsoft edge",
    "microsoft\u{200b} edge",
    "mozilla firefox",
    "firefox",
    "brave",
    "opera",
    "vivaldi",
    "safari",
    "arc",
    "zen browser",
    "librewolf",
    "waterfox",
];

/// Two-letter-country second-level labels that form public suffixes such as
/// `co.uk`, `com.au`, `gov.in`.
const SECOND_LEVEL_SUFFIXES: &[&str] = &["co", "com", "net", "org", "gov", "ac", "edu", "or", "ne"];

/// Some(reason) when app/title matches an application, window or domain rule
/// (case-insensitive). The reason names the rule (never the window title).
///
/// * Application rules match the application / process name, ignoring case,
///   `.exe` / `.app` suffixes and surrounding words ("1Password" matches
///   "1Password 8" and "1Password.exe"; "Code" matches "Visual Studio Code"
///   but not "Xcode"). When the application is unknown the window title is
///   checked instead.
/// * Window rules match a case-insensitive substring of the window title.
/// * Domain rules match browser window titles that contain the domain, or –
///   for browsers / unknown apps – the site name ("bank.example.com" matches
///   "Example Bank – Sign in - Google Chrome").
/// * Region rules never exclude a window (see [`apply_region_masks`]).
pub fn excluded_by(
    app: Option<&str>,
    title: Option<&str>,
    rules: &[PrivacyRule],
) -> Option<String> {
    let app = app.map(str::trim).filter(|a| !a.is_empty());
    let title = title.map(str::trim).filter(|t| !t.is_empty());
    for rule in rules {
        let value = rule.value.trim();
        if value.is_empty() {
            continue;
        }
        let matched = match rule.rule_type {
            PrivacyRuleType::Application => match (app, title) {
                (Some(app), _) => app_matches(app, value),
                (None, Some(title)) => contains_word_sequence(title, &normalize_app(value)),
                (None, None) => false,
            },
            PrivacyRuleType::Window => title.is_some_and(|t| contains_ci(t, value)),
            PrivacyRuleType::Domain => title.is_some_and(|t| domain_matches(app, t, value)),
            PrivacyRuleType::Region => false,
        };
        if matched {
            return Some(format!(
                "{} rule \"{}\"",
                rule_kind_label(rule.rule_type),
                value
            ));
        }
    }
    None
}

/// Blacks out all region rules ("x,y,w,h" in source pixels, scaled when the
/// image was downscaled). `scale` is `output / source` (1.0 when the image is
/// at source resolution). Invalid rules are ignored; rectangles are clipped to
/// the image and rounded outwards so a scaled mask never under-covers.
pub fn apply_region_masks(img: &mut RgbaImage, rules: &[PrivacyRule], scale: f32) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    for rule in rules {
        if rule.rule_type != PrivacyRuleType::Region {
            continue;
        }
        match Rect::parse_csv(&rule.value) {
            Some(rect) if rect.width > 0 && rect.height > 0 => {
                let x0 = (rect.x as f64 * scale as f64).floor();
                let y0 = (rect.y as f64 * scale as f64).floor();
                let x1 = ((rect.x as f64 + rect.width as f64) * scale as f64).ceil();
                let y1 = ((rect.y as f64 + rect.height as f64) * scale as f64).ceil();
                fill_black(img, x0, y0, x1, y1);
            }
            _ => tracing::debug!(rule_id = %rule.id, "ignoring malformed region privacy rule"),
        }
    }
}

/// Fills the clipped rectangle `[x0, x1) x [y0, y1)` (image coordinates) with
/// opaque black.
pub(crate) fn fill_black(img: &mut RgbaImage, x0: f64, y0: f64, x1: f64, y1: f64) {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let x0 = x0.clamp(0.0, w) as u32;
    let y0 = y0.clamp(0.0, h) as u32;
    let x1 = x1.clamp(0.0, w) as u32;
    let y1 = y1.clamp(0.0, h) as u32;
    let black = Rgba([0, 0, 0, 255]);
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, black);
        }
    }
}

/// Password managers, banking / finance and messaging apps users commonly
/// exclude (suggestions for the privacy-rules UI).
pub fn default_sensitive_apps() -> Vec<&'static str> {
    vec![
        // Password managers & authenticators
        "1Password",
        "Bitwarden",
        "Dashlane",
        "Enpass",
        "KeePass",
        "KeePassXC",
        "Keeper",
        "Keychain Access",
        "LastPass",
        "NordPass",
        "Passwords",
        "Proton Pass",
        "RoboForm",
        "Authy",
        "Microsoft Authenticator",
        // Banking, finance & crypto wallets
        "Quicken",
        "QuickBooks",
        "Moneydance",
        "Banktivity",
        "Ledger Live",
        "Exodus",
        "Electrum",
        "Trezor Suite",
        // Messaging
        "Signal",
        "WhatsApp",
        "Telegram",
        "Messages",
        "Messenger",
        "WeChat",
        "Threema",
        "Element",
        "Mail",
        "Outlook",
        "Thunderbird",
    ]
}

fn rule_kind_label(t: PrivacyRuleType) -> &'static str {
    match t {
        PrivacyRuleType::Application => "application",
        PrivacyRuleType::Window => "window",
        PrivacyRuleType::Domain => "domain",
        PrivacyRuleType::Region => "region",
    }
}

/// Lowercase, trimmed, whitespace-collapsed application name without a
/// trailing `.exe` / `.app`.
pub(crate) fn normalize_app(name: &str) -> String {
    let lower = name.trim().to_lowercase();
    let stripped = lower
        .strip_suffix(".exe")
        .or_else(|| lower.strip_suffix(".app"))
        .unwrap_or(&lower);
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `app` (display or process name) is a known web browser.
pub(crate) fn is_browser(app: &str) -> bool {
    let n = normalize_app(app);
    BROWSERS.iter().any(|b| *b == n)
        || n.starts_with("google chrome")
        || n.starts_with("microsoft edge")
        || n.starts_with("firefox")
        || n.starts_with("brave browser")
}

fn app_matches(app: &str, rule: &str) -> bool {
    let app_n = normalize_app(app);
    let rule_n = normalize_app(rule);
    if rule_n.is_empty() {
        return false;
    }
    app_n == rule_n || contains_word_sequence(&app_n, &rule_n)
}

fn contains_ci(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

/// Alphanumeric tokens, lowercased.
fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// True when the tokens of `needle` appear consecutively in `haystack`.
fn contains_word_sequence(haystack: &str, needle: &str) -> bool {
    let hay = tokens(haystack);
    let need = tokens(needle);
    if need.is_empty() || need.len() > hay.len() {
        return false;
    }
    hay.windows(need.len()).any(|w| w == need.as_slice())
}

/// Normalizes a user-entered domain: no scheme, path, port, credentials or
/// leading "www.", lowercase.
fn normalize_domain(value: &str) -> String {
    let v = value.trim().to_lowercase();
    let v = v
        .split_once("://")
        .map(|(_, rest)| rest.to_string())
        .unwrap_or(v);
    let host = v.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or("");
    let host = host.split(':').next().unwrap_or("");
    let host = host.trim_matches('.');
    host.strip_prefix("www.").unwrap_or(host).to_string()
}

/// The distinctive site label of a domain: "bank.example.com" -> "example",
/// "shop.example.co.uk" -> "example". None for single-label or very short
/// labels (too generic to match on their own).
fn site_label(domain: &str) -> Option<String> {
    let labels: Vec<&str> = domain.split('.').filter(|l| !l.is_empty()).collect();
    if labels.len() < 2 {
        return None;
    }
    let n = labels.len();
    let registrable_first =
        if n >= 3 && labels[n - 1].len() == 2 && SECOND_LEVEL_SUFFIXES.contains(&labels[n - 2]) {
            n - 3
        } else {
            n - 2
        };
    let label = labels[registrable_first];
    (label.chars().count() >= 3).then(|| label.to_string())
}

/// Removes a trailing " - Google Chrome" style browser marker from a title.
fn strip_browser_suffix(title: &str) -> String {
    let lower = title.to_lowercase();
    for sep in [" - ", " — ", " – "] {
        if let Some(idx) = lower.rfind(sep) {
            let tail = lower[idx + sep.len()..].trim();
            if BROWSER_TITLE_MARKERS
                .iter()
                .any(|m| tail == *m || tail.ends_with(m))
            {
                return lower[..idx].to_string();
            }
        }
    }
    lower
}

fn domain_matches(app: Option<&str>, title: &str, value: &str) -> bool {
    let domain = normalize_domain(value);
    if domain.is_empty() {
        return false;
    }
    let title_lower = title.to_lowercase();
    if title_lower.contains(&domain) {
        return true;
    }
    // Browsers usually show the page title, not the URL: match the site name
    // for browser windows (and windows whose application is unknown).
    let browser_like = app.is_none_or(is_browser);
    if !browser_like {
        return false;
    }
    match site_label(&domain) {
        Some(label) => {
            let page = strip_browser_suffix(title);
            contains_word_sequence(&page, &label) || contains_joined_words(&page, &label)
        }
        None => false,
    }
}

/// True when 2–4 consecutive words of `haystack`, joined without separators,
/// spell `label` ("Example Bank" ↔ "examplebank"). Only for labels of at
/// least 6 characters, to avoid accidental matches of short names.
fn contains_joined_words(haystack: &str, label: &str) -> bool {
    if label.chars().count() < 6 {
        return false;
    }
    let words = tokens(haystack);
    (0..words.len()).any(|i| {
        let mut joined = String::new();
        words[i..].iter().take(4).enumerate().any(|(k, w)| {
            joined.push_str(w);
            k >= 1 && joined == label
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::types::PrivacyAction;

    fn rule(t: PrivacyRuleType, v: &str) -> PrivacyRule {
        PrivacyRule {
            id: format!("r-{v}"),
            rule_type: t,
            value: v.to_string(),
            action: PrivacyAction::Exclude,
            created_at: 0,
        }
    }

    #[test]
    fn application_rules() {
        let rules = [rule(PrivacyRuleType::Application, "1Password")];
        assert!(excluded_by(Some("1Password"), None, &rules).is_some());
        assert!(excluded_by(Some("1password 8"), Some("Vault"), &rules).is_some());
        assert!(excluded_by(Some("1Password.exe"), None, &rules).is_some());
        assert!(excluded_by(Some("Bitwarden"), Some("1Password review"), &rules).is_none());
        let reason = excluded_by(Some("1Password"), Some("secret vault"), &rules).unwrap();
        assert_eq!(reason, "application rule \"1Password\"");
        assert!(!reason.contains("secret"), "reason must not leak the title");

        let code = [rule(PrivacyRuleType::Application, "Code")];
        assert!(excluded_by(Some("Visual Studio Code"), None, &code).is_some());
        assert!(excluded_by(Some("Xcode"), None, &code).is_none());

        let exe = [rule(PrivacyRuleType::Application, "signal.exe")];
        assert!(excluded_by(Some("Signal"), None, &exe).is_some());
    }

    #[test]
    fn application_rule_falls_back_to_title_when_app_unknown() {
        let rules = [rule(PrivacyRuleType::Application, "KeePassXC")];
        assert!(excluded_by(None, Some("Passwords.kdbx - KeePassXC"), &rules).is_some());
        assert!(excluded_by(Some(""), Some("KeePassXC"), &rules).is_some());
        assert!(excluded_by(None, Some("Notes"), &rules).is_none());
        assert!(excluded_by(None, None, &rules).is_none());
    }

    #[test]
    fn window_rules_are_case_insensitive_substrings() {
        let rules = [rule(PrivacyRuleType::Window, "Payroll")];
        assert!(excluded_by(Some("Excel"), Some("Q3 payroll.xlsx - Excel"), &rules).is_some());
        assert!(excluded_by(Some("Excel"), Some("Budget.xlsx"), &rules).is_none());
        assert!(excluded_by(Some("Excel"), None, &rules).is_none());
        let unicode = [rule(PrivacyRuleType::Window, "ÜBERSICHT")];
        assert!(excluded_by(None, Some("Kontoübersicht"), &unicode).is_some());
    }

    #[test]
    fn domain_rules() {
        let rules = [rule(
            PrivacyRuleType::Domain,
            "https://www.bank.example.com/login",
        )];
        // Full domain in the title.
        assert!(
            excluded_by(
                Some("Firefox"),
                Some("bank.example.com/accounts — Mozilla Firefox"),
                &rules
            )
            .is_some()
        );
        // Site name in a browser page title.
        assert!(
            excluded_by(
                Some("Google Chrome"),
                Some("Example Bank – Sign in - Google Chrome"),
                &rules
            )
            .is_some()
        );
        // Unknown app: treated like a browser.
        assert!(excluded_by(None, Some("Example Bank"), &rules).is_some());
        // Non-browser apps only match the full domain.
        assert!(excluded_by(Some("Slack"), Some("example channel"), &rules).is_none());
        assert!(excluded_by(Some("Slack"), Some("see bank.example.com"), &rules).is_some());
        // Unrelated browser tab.
        assert!(excluded_by(Some("Google Chrome"), Some("News - Google Chrome"), &rules).is_none());
        let reason = excluded_by(None, Some("Example Bank"), &rules).unwrap();
        assert!(reason.starts_with("domain rule"));
    }

    #[test]
    fn domain_rule_ignores_browser_name_suffix() {
        let rules = [rule(PrivacyRuleType::Domain, "google.com")];
        assert!(
            excluded_by(
                Some("Google Chrome"),
                Some("Weather - Google Chrome"),
                &rules
            )
            .is_none()
        );
        assert!(
            excluded_by(
                Some("Google Chrome"),
                Some("Google Docs - Google Chrome"),
                &rules
            )
            .is_some()
        );
        let edge = [rule(PrivacyRuleType::Domain, "microsoft.com")];
        assert!(
            excluded_by(
                Some("Microsoft Edge"),
                Some("Weather - Personal - Microsoft\u{200b} Edge"),
                &edge
            )
            .is_none()
        );
    }

    #[test]
    fn domain_helpers() {
        assert_eq!(
            normalize_domain("HTTPS://user:pw@WWW.Example.COM:8443/a?b"),
            "example.com"
        );
        assert_eq!(normalize_domain("bank.example.co.uk"), "bank.example.co.uk");
        assert_eq!(site_label("bank.example.com").as_deref(), Some("example"));
        assert_eq!(site_label("shop.example.co.uk").as_deref(), Some("example"));
        assert_eq!(site_label("example.com.au").as_deref(), Some("example"));
        assert_eq!(site_label("localhost"), None);
        assert_eq!(site_label("x.io"), None);
        assert_eq!(strip_browser_suffix("Inbox - Google Chrome"), "inbox");
        assert_eq!(strip_browser_suffix("A - B"), "a - b");
    }

    #[test]
    fn region_and_empty_rules_never_exclude() {
        let rules = [
            rule(PrivacyRuleType::Region, "0,0,10,10"),
            rule(PrivacyRuleType::Application, "   "),
            rule(PrivacyRuleType::Window, ""),
        ];
        assert!(excluded_by(Some("Anything"), Some("Any title"), &rules).is_none());
    }

    #[test]
    fn first_matching_rule_wins() {
        let rules = [
            rule(PrivacyRuleType::Window, "secret"),
            rule(PrivacyRuleType::Application, "Notes"),
        ];
        let reason = excluded_by(Some("Notes"), Some("secret plan"), &rules).unwrap();
        assert_eq!(reason, "window rule \"secret\"");
    }

    #[test]
    fn region_masks_full_scale() {
        let mut img = RgbaImage::from_pixel(100, 50, Rgba([200, 200, 200, 255]));
        let rules = [rule(PrivacyRuleType::Region, "10,5,20,10")];
        apply_region_masks(&mut img, &rules, 1.0);
        assert_eq!(img.get_pixel(10, 5), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(29, 14), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(30, 14), &Rgba([200, 200, 200, 255]));
        assert_eq!(img.get_pixel(29, 15), &Rgba([200, 200, 200, 255]));
        assert_eq!(img.get_pixel(9, 5), &Rgba([200, 200, 200, 255]));
    }

    #[test]
    fn region_masks_scaled_round_outwards() {
        let mut img = RgbaImage::from_pixel(50, 25, Rgba([200, 200, 200, 255]));
        // Source rect 11,5,21,11 at scale 0.5 -> [5.5, 16) x [2.5, 8) -> pixels 5..16 x 2..8
        let rules = [rule(PrivacyRuleType::Region, "11,5,21,11")];
        apply_region_masks(&mut img, &rules, 0.5);
        assert_eq!(img.get_pixel(5, 2), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(15, 7), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(16, 7), &Rgba([200, 200, 200, 255]));
        assert_eq!(img.get_pixel(4, 2), &Rgba([200, 200, 200, 255]));
    }

    #[test]
    fn region_masks_clip_and_ignore_invalid() {
        let mut img = RgbaImage::from_pixel(20, 20, Rgba([200, 200, 200, 255]));
        let rules = [
            rule(PrivacyRuleType::Region, "-5,-5,10,10"),
            rule(PrivacyRuleType::Region, "15,15,100,100"),
            rule(PrivacyRuleType::Region, "nonsense"),
            rule(PrivacyRuleType::Region, "1,1,0,5"),
            rule(PrivacyRuleType::Window, "1,1,5,5"),
        ];
        apply_region_masks(&mut img, &rules, f32::NAN);
        assert_eq!(img.get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(4, 4), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(5, 5), &Rgba([200, 200, 200, 255]));
        assert_eq!(img.get_pixel(19, 19), &Rgba([0, 0, 0, 255]));
        assert_eq!(img.get_pixel(14, 14), &Rgba([200, 200, 200, 255]));
        assert_eq!(img.get_pixel(2, 10), &Rgba([200, 200, 200, 255]));
    }

    #[test]
    fn sensitive_app_suggestions() {
        let apps = default_sensitive_apps();
        assert!(apps.contains(&"1Password"));
        assert!(apps.contains(&"Signal"));
        assert!(apps.len() >= 20);
        let mut sorted = apps.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), apps.len(), "no duplicates");
    }

    #[test]
    fn browser_detection() {
        assert!(is_browser("Google Chrome"));
        assert!(is_browser("chrome.exe"));
        assert!(is_browser("msedge"));
        assert!(is_browser("Firefox"));
        assert!(is_browser("Safari"));
        assert!(!is_browser("Slack"));
        assert!(!is_browser("Zoom"));
    }

    #[test]
    fn domain_label_split_into_words() {
        let rules = [rule(PrivacyRuleType::Domain, "examplebank.com")];
        assert!(
            excluded_by(
                Some("Google Chrome"),
                Some("Example Bank - Google Chrome"),
                &rules
            )
            .is_some()
        );
        assert!(
            excluded_by(
                Some("Google Chrome"),
                Some("Example news - Google Chrome"),
                &rules
            )
            .is_none()
        );
        // Short labels never match by concatenation.
        let short = [rule(PrivacyRuleType::Domain, "abcd.com")];
        assert!(
            excluded_by(Some("Google Chrome"), Some("ab cd - Google Chrome"), &short).is_none()
        );
    }
}
