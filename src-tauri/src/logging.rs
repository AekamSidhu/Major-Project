//! Logging setup. Log lines carry ids, counts, sizes and durations only —
//! never transcripts, prompts, model output, OCR text or API keys (spec §57).

use tracing_subscriber::EnvFilter;

pub fn init() {
    let default = if cfg!(debug_assertions) {
        "warn,openmeet=debug,openmeet_lib=debug,openmeet_engine=debug,openmeet_providers=info,openmeet_platform=info,openmeet_store=info,openmeet_intel=info"
    } else {
        "warn,openmeet_lib=info,openmeet_engine=info,openmeet_providers=info,openmeet_platform=info,openmeet_store=info"
    };
    let filter =
        EnvFilter::try_from_env("OPENMEET_LOG").unwrap_or_else(|_| EnvFilter::new(default));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(cfg!(debug_assertions))
        .try_init();
}
