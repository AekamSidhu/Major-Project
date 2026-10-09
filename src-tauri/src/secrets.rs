//! API-key storage in the OS credential store (Windows Credential Manager,
//! macOS Keychain, Secret Service on Linux), with environment variables as a
//! development fallback (spec §44). Keys are never stored in SQLite, never
//! returned to the UI and never logged.

use std::collections::HashMap;

use openmeet_core::CoreError;
use openmeet_core::ipc::{ApiKeyStatus, SecretProvider, SecretSource};
use openmeet_core::providers::SecretResolver;
use parking_lot::Mutex;

const SERVICE: &str = "dev.openmeet.desktop";
const MAX_KEY_LEN: usize = 512;

pub struct SecretStore {
    /// Cached keychain lookups (avoids repeated OS prompts). `None` = absent.
    cache: Mutex<HashMap<SecretProvider, Option<String>>>,
}

impl Default for SecretStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretStore {
    pub fn new() -> Self {
        SecretStore {
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn entry(provider: SecretProvider) -> Result<keyring::Entry, CoreError> {
        keyring::Entry::new(SERVICE, provider.account()).map_err(|e| {
            tracing::warn!(provider = provider.account(), error = %e, "credential store unavailable");
            CoreError::Platform(
                "The system credential store is unavailable on this system.".to_string(),
            )
        })
    }

    fn keychain_value(&self, provider: SecretProvider) -> Option<String> {
        if let Some(cached) = self.cache.lock().get(&provider) {
            return cached.clone();
        }
        let value = Self::entry(provider)
            .ok()
            .and_then(|e| match e.get_password() {
                Ok(v) if !v.trim().is_empty() => Some(v),
                Ok(_) | Err(keyring::Error::NoEntry) => None,
                Err(e) => {
                    tracing::warn!(provider = provider.account(), error = %e, "credential read failed");
                    None
                }
            });
        self.cache.lock().insert(provider, value.clone());
        value
    }

    fn env_value(provider: SecretProvider) -> Option<String> {
        std::env::var(provider.env_var())
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    pub fn status(&self, provider: SecretProvider) -> ApiKeyStatus {
        let source = if self.keychain_value(provider).is_some() {
            SecretSource::Keychain
        } else if Self::env_value(provider).is_some() {
            SecretSource::Environment
        } else {
            SecretSource::None
        };
        ApiKeyStatus {
            provider,
            configured: source != SecretSource::None,
            source,
            env_var: provider.env_var().to_string(),
        }
    }

    pub fn all_status(&self) -> Vec<ApiKeyStatus> {
        SecretProvider::ALL
            .into_iter()
            .map(|p| self.status(p))
            .collect()
    }

    /// Stores a key in the OS credential store.
    pub fn set(&self, provider: SecretProvider, key: &str) -> Result<ApiKeyStatus, CoreError> {
        let key = validate_key(key)?;
        Self::entry(provider)?.set_password(&key).map_err(|e| {
            tracing::warn!(provider = provider.account(), error = %e, "credential write failed");
            CoreError::Platform(
                "Could not save the API key in the system credential store.".to_string(),
            )
        })?;
        self.cache.lock().insert(provider, Some(key));
        Ok(self.status(provider))
    }

    /// Removes a key from the OS credential store (environment variables are
    /// not affected).
    pub fn clear(&self, provider: SecretProvider) -> Result<ApiKeyStatus, CoreError> {
        match Self::entry(provider)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => {}
            Err(e) => {
                tracing::warn!(provider = provider.account(), error = %e, "credential delete failed");
                return Err(CoreError::Platform(
                    "Could not remove the API key from the system credential store.".to_string(),
                ));
            }
        }
        self.cache.lock().insert(provider, None);
        Ok(self.status(provider))
    }
}

impl SecretResolver for SecretStore {
    fn api_key(&self, provider: SecretProvider) -> Option<String> {
        self.keychain_value(provider)
            .or_else(|| Self::env_value(provider))
    }
}

/// Trims and validates an API key (no whitespace or control characters).
pub fn validate_key(key: &str) -> Result<String, CoreError> {
    let key = key.trim();
    if key.is_empty() {
        return Err(CoreError::InvalidInput("The API key is empty.".to_string()));
    }
    if key.len() > MAX_KEY_LEN {
        return Err(CoreError::InvalidInput(
            "The API key is too long.".to_string(),
        ));
    }
    if key.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(CoreError::InvalidInput(
            "The API key must not contain spaces or control characters.".to_string(),
        ));
    }
    Ok(key.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_validation() {
        assert_eq!(validate_key("  sk-abc123 ").unwrap(), "sk-abc123");
        assert!(validate_key("").is_err());
        assert!(validate_key("sk abc").is_err());
        assert!(validate_key("sk\u{0007}abc").is_err());
        assert!(validate_key(&"x".repeat(600)).is_err());
    }
}
