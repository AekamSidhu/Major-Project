//! Screen understanding adapters (spec §29).
//!
//! * [`LlmVisionProvider`] — a vision-capable LLM analyses the (downscaled)
//!   frame together with local OCR text and returns a structured observation.
//! * [`OcrVisionProvider`] — fully local: the OS OCR engine reads the frame and
//!   [`describe_ocr_text`] derives a structured observation from the text.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use openmeet_core::error::ProviderError;
use openmeet_core::providers::{
    ChatMessage, ImageInput, LlmProvider, LlmRequest, OcrEngine, VisionContext, VisionProvider,
    VisionRequest, VisionResult, complete_text,
};
use openmeet_core::types::{ProviderInfo, RequestKind};
use tokio_util::sync::CancellationToken;

use crate::claims::extract_entities;
use crate::prompts::screen_analysis;

/// Maximum OCR text kept in an observation.
const MAX_VISIBLE_CHARS: usize = 4000;

/// Vision through a vision-capable LLM.
pub struct LlmVisionProvider {
    llm: Arc<dyn LlmProvider>,
}

impl LlmVisionProvider {
    /// Wraps a vision-capable LLM.
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        LlmVisionProvider { llm }
    }
}

#[async_trait]
impl VisionProvider for LlmVisionProvider {
    fn info(&self) -> ProviderInfo {
        let mut info = self.llm.info();
        info.display_name = format!("{} (vision)", info.display_name);
        info
    }

    async fn analyze(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<VisionResult, ProviderError> {
        let images = self.llm.supports_images();
        if !images
            && request
                .ocr_text
                .as_deref()
                .is_none_or(|t| t.trim().is_empty())
        {
            return Err(ProviderError::Unsupported(
                "the selected model cannot read images and no OCR text is available".to_string(),
            ));
        }
        let parts = screen_analysis::build(&request.context, request.ocr_text.as_deref());
        let mut message = ChatMessage::user(parts.user);
        if images {
            message = message.with_image(ImageInput {
                mime_type: "image/png".to_string(),
                data_base64: base64::engine::general_purpose::STANDARD
                    .encode(request.image_png.as_slice()),
            });
        }
        let llm_request = LlmRequest {
            request_id: openmeet_core::new_id(),
            kind: RequestKind::ScreenAnalysis,
            system: parts.system,
            messages: vec![message],
            max_output_tokens: 700,
            temperature: Some(0.1),
            json_output: true,
            latency_sensitive: false,
        };
        let text = complete_text(self.llm.as_ref(), llm_request, cancel).await?;
        let mut result = screen_analysis::parse(&text);
        if result.description.trim().is_empty() {
            return Err(ProviderError::InvalidResponse(
                "the model returned no screen description".to_string(),
            ));
        }
        if result.visible_text.is_none() {
            result.visible_text = request
                .ocr_text
                .as_deref()
                .map(|t| truncate_chars(t.trim(), MAX_VISIBLE_CHARS));
        }
        if result.entities.is_empty() {
            let mut source = result.description.clone();
            if let Some(v) = &result.visible_text {
                source.push(' ');
                source.push_str(v);
            }
            result.entities = extract_entities(&source);
        }
        if result.application.is_none() {
            result.application = request.context.application.clone();
        }
        Ok(result)
    }
}

/// Vision from local OCR only (no model).
pub struct OcrVisionProvider {
    ocr: Arc<dyn OcrEngine>,
}

impl OcrVisionProvider {
    /// Wraps a local OCR engine.
    pub fn new(ocr: Arc<dyn OcrEngine>) -> Self {
        OcrVisionProvider { ocr }
    }
}

#[async_trait]
impl VisionProvider for OcrVisionProvider {
    fn info(&self) -> ProviderInfo {
        self.ocr.info()
    }

    async fn analyze(
        &self,
        request: VisionRequest,
        cancel: CancellationToken,
    ) -> Result<VisionResult, ProviderError> {
        let text = match request.ocr_text.clone() {
            Some(t) => t,
            None => {
                let ocr = self.ocr.clone();
                let png = request.image_png.clone();
                let task = tokio::task::spawn_blocking(move || ocr.recognize_png(&png));
                tokio::select! {
                    _ = cancel.cancelled() => return Err(ProviderError::Cancelled),
                    r = task => r.map_err(|_| ProviderError::Other("OCR task failed".to_string()))??.text,
                }
            }
        };
        if text.trim().is_empty() {
            return Err(ProviderError::InvalidResponse(
                "no text was recognized on screen".to_string(),
            ));
        }
        Ok(describe_ocr_text(&text, &request.context))
    }
}

fn truncate_chars(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

const BROWSERS: &[&str] = &[
    "chrome", "edge", "firefox", "safari", "brave", "opera", "arc", "vivaldi", "chromium",
];
const TERMINALS: &[&str] = &[
    "terminal",
    "powershell",
    "cmd",
    "iterm",
    "warp",
    "konsole",
    "alacritty",
    "wezterm",
    "windows terminal",
    "kitty",
];
const EDITORS: &[&str] = &[
    "code",
    "visual studio",
    "intellij",
    "pycharm",
    "webstorm",
    "goland",
    "rider",
    "clion",
    "xcode",
    "android studio",
    "sublime",
    "vim",
    "neovim",
    "emacs",
    "zed",
    "cursor",
    "notepad++",
];
const SLIDES: &[&str] = &["powerpoint", "keynote", "slides", "impress"];
const DIAGRAMS: &[&str] = &[
    "excalidraw",
    "draw.io",
    "diagrams.net",
    "lucidchart",
    "miro",
    "figjam",
    "visio",
    "whimsical",
    "mermaid",
    "tldraw",
];
const DOCS: &[&str] = &[
    "word",
    "docs",
    "notion",
    "confluence",
    "pages",
    "acrobat",
    "pdf",
    "obsidian",
];

fn app_matches(app: &str, list: &[&str]) -> bool {
    let a = app.to_lowercase();
    list.iter().any(|x| a.contains(x))
}

/// Heuristic content type of OCR text (+ application hint).
fn content_type(text: &str, app: Option<&str>) -> &'static str {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n = lines.len().max(1) as f32;
    let code_markers = [
        "{", "}", ";", "=>", "->", "fn ", "def ", "function", "const ", "let ", "var ", "import ",
        "return", "class ", "public ", "private ", "#include", "func ", "</", "/>", "()", "==",
    ];
    let code_lines = lines
        .iter()
        .filter(|l| code_markers.iter().any(|m| l.contains(m)))
        .count() as f32;
    let terminal_lines = lines
        .iter()
        .filter(|l| {
            l.starts_with("$ ")
                || l.starts_with("PS ")
                || l.starts_with(">>> ")
                || l.starts_with("> ")
                || l.contains(":\\>")
                || l.starts_with("error:")
                || l.starts_with("warning:")
                || l.starts_with("npm ")
                || l.starts_with("cargo ")
                || l.starts_with("git ")
        })
        .count() as f32;
    let arrows = text.matches("→").count()
        + text.matches("->").count()
        + text.matches("-->").count()
        + text.matches("⇒").count();
    let short_lines = lines
        .iter()
        .filter(|l| l.split_whitespace().count() <= 3)
        .count() as f32;
    let bullets = lines
        .iter()
        .filter(|l| l.starts_with('•') || l.starts_with("- ") || l.starts_with("* "))
        .count() as f32;
    let long_lines = lines
        .iter()
        .filter(|l| l.split_whitespace().count() >= 12)
        .count() as f32;
    if let Some(app) = app {
        if app_matches(app, TERMINALS) {
            return "terminal";
        }
        if app_matches(app, DIAGRAMS) {
            return "diagram";
        }
        if app_matches(app, SLIDES) {
            return "slides";
        }
        if app_matches(app, EDITORS) && code_lines / n >= 0.15 {
            return "code";
        }
    }
    if terminal_lines / n >= 0.3 {
        return "terminal";
    }
    if code_lines / n >= 0.35 {
        return "code";
    }
    if arrows >= 2 && short_lines / n >= 0.5 {
        return "diagram";
    }
    if bullets / n >= 0.3 && lines.len() <= 15 {
        return "slides";
    }
    if let Some(app) = app {
        if app_matches(app, BROWSERS) {
            return "browser";
        }
        if app_matches(app, DOCS) {
            return "document";
        }
    }
    if long_lines / n >= 0.3 {
        return "document";
    }
    if arrows >= 2 {
        return "diagram";
    }
    "other"
}

fn title_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|l| {
            let words = l.split_whitespace().count();
            (2..=12).contains(&words) && l.chars().filter(|c| c.is_alphabetic()).count() >= 6
        })
        .map(|l| truncate_chars(l, 100))
}

/// Builds a structured observation from OCR text without any model.
pub fn describe_ocr_text(text: &str, ctx: &VisionContext) -> VisionResult {
    let text = text.trim();
    let app = ctx.application.as_deref();
    let kind = content_type(text, app);
    let entities = extract_entities(text);
    let what = match kind {
        "code" => "Code",
        "terminal" => "Terminal output",
        "diagram" => "Diagram",
        "slides" => "Slide",
        "document" => "Document",
        "browser" => "Web page",
        _ => "Screen content",
    };
    let mut description = String::from(what);
    if let Some(app) = app.filter(|a| !a.trim().is_empty()) {
        description.push_str(&format!(" in {}", app.trim()));
    }
    if kind == "diagram" {
        // Diagrams are usually short: their flow is the most useful summary.
        let flow: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.contains("→") || l.contains("->"))
            .take(3)
            .collect();
        if !flow.is_empty() {
            description.push_str(&format!(
                " showing {}",
                truncate_chars(&flow.join("; "), 200)
            ));
        }
    } else if let Some(title) = ctx
        .window_title
        .as_deref()
        .map(|t| truncate_chars(t.trim(), 100))
        .filter(|t| !t.is_empty())
        .or_else(|| title_line(text))
    {
        description.push_str(&format!(": \"{title}\""));
    }
    if !entities.is_empty() {
        description.push_str(&format!(
            " (mentions {})",
            entities
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    description.push('.');
    VisionResult {
        description,
        visible_text: Some(truncate_chars(text, MAX_VISIBLE_CHARS)).filter(|t| !t.is_empty()),
        entities,
        application: ctx.application.clone(),
        content_type: Some(kind.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openmeet_core::providers::OcrOutput;
    use openmeet_core::types::ProviderLocation;

    fn ctx(app: Option<&str>) -> VisionContext {
        VisionContext {
            source_label: "Display 1".into(),
            application: app.map(str::to_string),
            window_title: None,
            recent_transcript: String::new(),
            question: None,
        }
    }

    #[test]
    fn content_types() {
        assert_eq!(
            describe_ocr_text("import express from 'express';\nconst app = express();\napp.get('/', (req, res) => {\n  return res.send('ok');\n});", &ctx(Some("Code"))).content_type.as_deref(),
            Some("code")
        );
        assert_eq!(
            describe_ocr_text(
                "$ cargo test\nerror: could not compile\nwarning: unused",
                &ctx(None)
            )
            .content_type
            .as_deref(),
            Some("terminal")
        );
        let d = describe_ocr_text(
            "API Gateway → Job Queue\nJob Queue → Workers\nWorkers → PostgreSQL",
            &ctx(Some("Excalidraw")),
        );
        assert_eq!(d.content_type.as_deref(), Some("diagram"));
        assert!(
            d.description.contains("API Gateway → Job Queue"),
            "{}",
            d.description
        );
        assert!(d.entities.contains(&"PostgreSQL".to_string()));
        assert_eq!(
            describe_ocr_text(
                "Q3 Roadmap\n• Faster checkout\n• New onboarding\n• Billing v2",
                &ctx(None)
            )
            .content_type
            .as_deref(),
            Some("slides")
        );
    }

    #[test]
    fn description_mentions_app_and_entities() {
        let d = describe_ocr_text(
            "Kafka consumer lag dashboard\nTopic orders partition 3 lag 1200",
            &ctx(Some("Grafana")),
        );
        assert!(d.description.contains("Grafana"));
        assert!(d.description.contains("Kafka"));
        assert!(d.visible_text.unwrap().contains("lag 1200"));
    }

    struct FakeOcr;
    impl OcrEngine for FakeOcr {
        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                id: "fake_ocr".into(),
                display_name: "Fake OCR".into(),
                model: String::new(),
                location: ProviderLocation::Local,
                endpoint: None,
            }
        }
        fn recognize_png(&self, _png: &[u8]) -> Result<OcrOutput, ProviderError> {
            Ok(OcrOutput {
                text: "SELECT * FROM orders;".into(),
                lines: vec!["SELECT * FROM orders;".into()],
            })
        }
    }

    #[tokio::test]
    async fn ocr_provider_runs_ocr_when_needed() {
        let p = OcrVisionProvider::new(Arc::new(FakeOcr));
        let r = p
            .analyze(
                VisionRequest {
                    image_png: Arc::new(vec![]),
                    width: 1,
                    height: 1,
                    ocr_text: None,
                    context: ctx(None),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(r.visible_text.unwrap().contains("orders"));
    }

    #[tokio::test]
    async fn llm_vision_uses_model_output() {
        let llm = crate::mock::MockLlmProvider::new();
        llm.push_response(r#"{"description":"Architecture diagram of the billing service","visible_text":"API → Queue","entities":["Kafka"],"application":"Miro","content_type":"diagram"}"#.into());
        let p = LlmVisionProvider::new(Arc::new(llm.clone()));
        let r = p
            .analyze(
                VisionRequest {
                    image_png: Arc::new(vec![1, 2, 3]),
                    width: 1,
                    height: 1,
                    ocr_text: Some("API → Queue".into()),
                    context: ctx(Some("Miro")),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(r.description, "Architecture diagram of the billing service");
        assert_eq!(r.content_type.as_deref(), Some("diagram"));
        let req = &llm.requests()[0];
        assert_eq!(req.kind, RequestKind::ScreenAnalysis);
        assert_eq!(req.messages[0].images.len(), 1, "the frame is attached");
    }
}
