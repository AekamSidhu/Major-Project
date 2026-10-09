//! Optional screenshot retention (Settings → Screen context → Keep
//! screenshots, off by default, spec §29). Frames that were analyzed are kept
//! as PNG files in `<data dir>/screenshots/<meeting id>/` and deleted together
//! with their meeting (delete, delete-all, retention).

use std::collections::HashSet;
use std::io;
use std::path::{Path, PathBuf};

use openmeet_core::types::ScreenFrame;

pub struct ScreenshotStore {
    root: PathBuf,
}

/// Only plain ids may become directory / file names (no path traversal).
fn safe_name(id: &str) -> bool {
    !id.is_empty() && id.len() <= 128 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl ScreenshotStore {
    pub fn new(data_dir: &Path) -> Self {
        ScreenshotStore {
            root: data_dir.join("screenshots"),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn save(&self, meeting_id: &str, frame: &ScreenFrame) -> io::Result<PathBuf> {
        if !safe_name(meeting_id) || !safe_name(&frame.id) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid id"));
        }
        let dir = self.root.join(meeting_id);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.png", frame.id));
        std::fs::write(&path, &frame.png)?;
        Ok(path)
    }

    pub fn delete_meeting(&self, meeting_id: &str) -> io::Result<()> {
        if !safe_name(meeting_id) {
            return Ok(());
        }
        match std::fs::remove_dir_all(self.root.join(meeting_id)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }

    pub fn delete_all(&self) -> io::Result<()> {
        match std::fs::remove_dir_all(&self.root) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }

    /// Removes screenshot folders of meetings that no longer exist. Returns
    /// the number of folders removed.
    pub fn prune(&self, existing: &HashSet<String>) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.root) else { return 0 };
        let mut removed = 0;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if entry.path().is_dir() && !existing.contains(&name) && std::fs::remove_dir_all(entry.path()).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: &str) -> ScreenFrame {
        ScreenFrame {
            id: id.into(),
            captured_at: 0,
            source_label: "Display 1".into(),
            application: None,
            window_title: None,
            width: 1,
            height: 1,
            png: vec![1, 2, 3],
            hash: 0,
            change_score: 1.0,
            manual: false,
        }
    }

    #[test]
    fn save_delete_and_prune() {
        let dir = std::env::temp_dir().join(format!("openmeet-shots-{}", openmeet_core::new_id()));
        let store = ScreenshotStore::new(&dir);
        let p = store.save("m1", &frame("f1")).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), vec![1, 2, 3]);
        store.save("m2", &frame("f2")).unwrap();
        assert!(store.save("../evil", &frame("f3")).is_err());
        assert!(store.save("m1", &frame("..\\x")).is_err());
        let keep: HashSet<String> = ["m2".to_string()].into_iter().collect();
        assert_eq!(store.prune(&keep), 1);
        assert!(!store.root().join("m1").exists());
        store.delete_meeting("m2").unwrap();
        assert!(!store.root().join("m2").exists());
        store.delete_meeting("missing").unwrap();
        store.delete_all().unwrap();
        store.delete_all().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
