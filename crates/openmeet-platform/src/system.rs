//! Process resource statistics for the diagnostics view.

use sysinfo::{CpuRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Bytes per mebibyte.
const MIB: f64 = 1024.0 * 1024.0;

/// Memory and CPU usage of the current (OpenMeet) process.
///
/// Wraps a `sysinfo::System`; keep one instance alive and call
/// [`ProcessStats::sample`] periodically, because CPU usage is computed from
/// the difference between two samples.
pub struct ProcessStats {
    system: System,
    pid: Option<Pid>,
    /// Logical CPUs, used to normalize CPU usage to 0..100 % of the machine.
    cpus: usize,
}

impl std::fmt::Debug for ProcessStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessStats")
            .field("pid", &self.pid)
            .field("cpus", &self.cpus)
            .finish_non_exhaustive()
    }
}

impl Default for ProcessStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessStats {
    /// Creates the sampler for the current process. Cheap: only the CPU list
    /// is read (needed to compute process CPU usage).
    pub fn new() -> Self {
        let mut system = System::new();
        // sysinfo computes per-process CPU usage relative to the number of
        // known CPUs; without the list every process reports 0 %.
        system.refresh_cpu_list(CpuRefreshKind::nothing());
        let cpus = system.cpus().len().max(1);
        let pid = match sysinfo::get_current_pid() {
            Ok(pid) => Some(pid),
            Err(e) => {
                tracing::warn!(
                    error = e,
                    "current process id unavailable; process stats disabled"
                );
                None
            }
        };
        ProcessStats { system, pid, cpus }
    }

    /// Returns `(memory MB, cpu %)` of the current process.
    ///
    /// * Memory is the resident set / working set in MiB.
    /// * CPU is the average usage since the previous call as a share of the
    ///   whole machine (0..100 %, all logical CPUs together). The first call
    ///   returns 0 % because there is no previous sample yet; samples closer
    ///   together than ~200 ms repeat the previous value.
    ///
    /// Returns `(0.0, 0.0)` when the OS refuses to report the process.
    pub fn sample(&mut self) -> (f64, f32) {
        let Some(pid) = self.pid else {
            return (0.0, 0.0);
        };
        self.system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        match self.system.process(pid) {
            Some(process) => {
                let memory_mb = process.memory() as f64 / MIB;
                let cpu = normalize_cpu(process.cpu_usage(), self.cpus);
                (memory_mb, cpu)
            }
            None => (0.0, 0.0),
        }
    }
}

/// sysinfo reports up to `100 * cpus` %; scale to 0..100 % of the machine.
fn normalize_cpu(raw: f32, cpus: usize) -> f32 {
    if !raw.is_finite() || raw <= 0.0 {
        return 0.0;
    }
    (raw / cpus.max(1) as f32).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_normalization() {
        assert_eq!(normalize_cpu(0.0, 8), 0.0);
        assert_eq!(normalize_cpu(400.0, 8), 50.0);
        assert_eq!(normalize_cpu(1600.0, 8), 100.0);
        assert_eq!(normalize_cpu(50.0, 0), 50.0);
        assert_eq!(normalize_cpu(f32::NAN, 4), 0.0);
        assert_eq!(normalize_cpu(-3.0, 4), 0.0);
    }

    #[test]
    fn samples_the_current_process() {
        let mut stats = ProcessStats::new();
        assert!(stats.cpus >= 1);
        let (memory_mb, cpu) = stats.sample();
        // A running test binary uses at least some memory.
        assert!(memory_mb > 0.5, "{memory_mb}");
        assert!((0.0..=100.0).contains(&cpu), "{cpu}");

        // Burn a little CPU so the second sample has something to measure.
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        let started = std::time::Instant::now();
        let mut x = 0u64;
        while started.elapsed() < std::time::Duration::from_millis(250) {
            x = std::hint::black_box(x.wrapping_mul(31).wrapping_add(7));
        }
        let (memory_mb, cpu) = stats.sample();
        assert!(memory_mb > 0.5);
        assert!((0.0..=100.0).contains(&cpu), "{cpu}");
        assert!(format!("{stats:?}").contains("ProcessStats"));
    }
}
