/// Kernel tuner: detect and apply USB4/ASPM/BBR optimizations.
///
/// From paper §5.1.3: kernel-level optimizations that reduce P99 RTT
/// from 97 µs to 27.85 µs (71% reduction in tail jitter):
///   1. AMD P-State EPP scheduler
///   2. PCIe ASPM disablement (pcie_aspm=off)
///   3. BBR congestion control
///   4. Busy-poll sockets (SO_BUSY_POLL)
use std::fs;

pub struct KernelTuner;

impl KernelTuner {
    /// Detect current kernel tuning state and return a report.
    pub fn check() -> KernelTuneReport {
        KernelTuneReport {
            kernel: Self::read_string("/proc/sys/kernel/hostname").unwrap_or_default(),
            aspm: Self::aspm_status(),
            bbr: Self::bbr_status(),
            governor: Self::governor_status(),
            busy_poll: Self::busy_poll_status(),
            usb4_driver: Self::usb4_driver_loaded(),
            amdxdna: Self::amdxdna_loaded(),
        }
    }

    /// Apply recommended kernel tuning for USB4 P2P low latency.
    ///
    /// Requires root. Prints commands if not running as root.
    pub fn apply() -> Vec<String> {
        let cmds = vec![
            "sudo sh -c 'echo 0 > /sys/module/pcie_aspm/parameters/policy'".to_string(),
            "sudo sysctl -w net.core.busy_poll=50".to_string(),
            "sudo sysctl -w net.ipv4.tcp_congestion_control=bbr".to_string(),
            "sudo sysctl -w net.core.rmem_max=26214400".to_string(),
            "sudo sysctl -w net.core.wmem_max=26214400".to_string(),
            "sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'".to_string(),
        ];
        cmds
    }

    fn read_string(path: &str) -> Option<String> {
        fs::read_to_string(path).ok().map(|s| s.trim().to_string())
    }

    fn aspm_status() -> AspmStatus {
        let policy = Self::read_string("/sys/module/pcie_aspm/parameters/policy")
            .unwrap_or_default();
        if policy.contains("0") || policy.contains("off") {
            AspmStatus::Disabled
        } else if policy.is_empty() {
            AspmStatus::Unknown
        } else {
            AspmStatus::Enabled(policy)
        }
    }

    fn bbr_status() -> bool {
        Self::read_string("/proc/sys/net/ipv4/tcp_congestion_control")
            .map(|s| s.trim() == "bbr")
            .unwrap_or(false)
    }

    fn governor_status() -> String {
        Self::read_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .unwrap_or_default()
    }

    fn busy_poll_status() -> u32 {
        Self::read_string("/proc/sys/net/core/busy_poll")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    fn usb4_driver_loaded() -> bool {
        Self::read_string("/proc/modules")
            .map(|s| s.contains("thunderbolt"))
            .unwrap_or(false)
    }

    fn amdxdna_loaded() -> bool {
        Self::read_string("/proc/modules")
            .map(|s| s.contains("amdxdna"))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
pub struct KernelTuneReport {
    pub kernel: String,
    pub aspm: AspmStatus,
    pub bbr: bool,
    pub governor: String,
    pub busy_poll: u32,
    pub usb4_driver: bool,
    pub amdxdna: bool,
}

impl std::fmt::Display for KernelTuneReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Kernel: {}", self.kernel)?;
        writeln!(f, "ASPM: {}", self.aspm)?;
        writeln!(f, "BBR congestion: {}", if self.bbr { "✅" } else { "❌" })?;
        writeln!(f, "Governor: {}", self.governor)?;
        writeln!(f, "Busy poll: {} µs", self.busy_poll)?;
        writeln!(f, "USB4 driver: {}", if self.usb4_driver { "✅" } else { "❌" })?;
        writeln!(f, "amdxdna NPU: {}", if self.amdxdna { "✅" } else { "❌" })?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum AspmStatus {
    Disabled,
    Enabled(String),
    Unknown,
}

impl std::fmt::Display for AspmStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AspmStatus::Disabled => write!(f, "✅ Disabled"),
            AspmStatus::Enabled(v) => write!(f, "❌ Enabled ({})", v),
            AspmStatus::Unknown => write!(f, "❓ Unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_check_runs() {
        let report = KernelTuner::check();
        // Should not panic — reads from /proc and /sys
        println!("{}", report);
    }
}
