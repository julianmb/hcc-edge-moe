/// Kernel tuner: detect and apply USB4/ASPM/BBR optimizations.
///
/// From paper §5.1.3: kernel-level optimizations that reduce P99 RTT
/// from 97 µs to 27.85 µs (71% reduction in tail jitter):
///   1. AMD P-State EPP scheduler
///   2. PCIe ASPM disablement (pcie_aspm=off)
///   3. BBR congestion control
///   4. Busy-poll sockets (SO_BUSY_POLL)
///
/// Running on: Ubuntu 24.04.4 LTS, kernel 6.17.0-1020-oem (OEM for Strix Halo)
///   - amdxdna.ko: built-in, DMA_BUF import support
///   - ASPM: needs explicit disable (pcie_aspm=off kernel param)
///   - BBR: available in kernel, needs sysctl enable
use std::fs;

pub struct KernelTuner;

impl KernelTuner {
    pub fn check() -> KernelTuneReport {
        KernelTuneReport {
            kernel: Self::uname_release(),
            aspm: Self::aspm_status(),
            bbr: Self::bbr_status(),
            governor: Self::governor_status(),
            epp: Self::epp_status(),
            busy_poll: Self::busy_poll_status(),
            usb4_driver: Self::module_loaded("thunderbolt"),
            amdxdna: Self::module_loaded("amdxdna"),
            amdgpu: Self::module_loaded("amdgpu"),
            numa_nodes: Self::numa_count(),
        }
    }

    /// Print actionable kernel tuning commands for Strix Halo USB4 low latency.
    pub fn apply() -> Vec<String> {
        vec![
            // Boot params (add to GRUB_CMDLINE_LINUX_DEFAULT):
            "sudo sh -c 'echo GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash pcie_aspm=off amd_pstate_epp=performance\" >> /etc/default/grub'".into(),
            "sudo update-grub".into(),
            // Runtime tuning:
            "sudo sysctl -w net.core.busy_poll=50".into(),
            "sudo sysctl -w net.core.busy_read=50".into(),
            "sudo sysctl -w net.ipv4.tcp_congestion_control=bbr".into(),
            "sudo sysctl -w net.core.rmem_max=26214400".into(),
            "sudo sysctl -w net.core.wmem_max=26214400".into(),
            "sudo sysctl -w net.ipv4.tcp_rmem=\"4096 87380 26214400\"".into(),
            "sudo sysctl -w net.ipv4.tcp_wmem=\"4096 65536 26214400\"".into(),
            // CPU governor:
            "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor".into(),
            // USB4: set thunderbolt to low-latency mode:
            "sudo sh -c 'echo 0 > /sys/bus/thunderbolt/devices/0-0/power/pm_qos_resume_latency_us'".into(),
        ]
    }

    fn uname_release() -> String {
        fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default().trim().to_string()
    }

    fn module_loaded(name: &str) -> bool {
        fs::read_to_string("/proc/modules")
            .map(|s| s.lines().any(|l| l.starts_with(name)))
            .unwrap_or(false)
    }

    fn aspm_status() -> AspmStatus {
        let policy = fs::read_to_string("/sys/module/pcie_aspm/parameters/policy").unwrap_or_default();
        let active = fs::read_to_string("/sys/module/pcie_aspm/parameters/state").unwrap_or_default();
        if policy.contains("[default]") && !policy.contains("[performance]") {
            AspmStatus::Enabled(policy.trim().to_string(), active.trim().to_string())
        } else if policy.contains("[performance]") || policy.contains("off") {
            AspmStatus::Disabled
        } else {
            AspmStatus::Unknown(policy.trim().to_string())
        }
    }

    fn bbr_status() -> bool {
        fs::read_to_string("/proc/sys/net/ipv4/tcp_congestion_control")
            .map(|s| s.trim() == "bbr")
            .unwrap_or(false)
    }

    fn governor_status() -> String {
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .unwrap_or_default()
            .trim()
            .to_string()
    }

    fn epp_status() -> String {
        fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
            .unwrap_or_default()
            .trim()
            .to_string()
    }

    fn busy_poll_status() -> u32 {
        fs::read_to_string("/proc/sys/net/core/busy_poll")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    fn numa_count() -> usize {
        (0..).map_while(|i| {
            fs::metadata(format!("/sys/devices/system/node/node{i}")).ok()
        }).count()
    }
}

#[derive(Debug, Clone)]
pub struct KernelTuneReport {
    pub kernel: String,
    pub aspm: AspmStatus,
    pub bbr: bool,
    pub governor: String,
    pub epp: String,
    pub busy_poll: u32,
    pub usb4_driver: bool,
    pub amdxdna: bool,
    pub amdgpu: bool,
    pub numa_nodes: usize,
}

impl std::fmt::Display for KernelTuneReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ok = "✅";
        let no = "❌";

        writeln!(f, "── Kernel ──")?;
        writeln!(f, "  Release:       {}", self.kernel)?;
        writeln!(f, "  NUMA nodes:    {}", self.numa_nodes)?;
        writeln!(f)?;
        writeln!(f, "── Drivers ──")?;
        writeln!(f, "  amdxdna (NPU):  {}", if self.amdxdna { ok } else { no })?;
        writeln!(f, "  amdgpu (GPU):   {}", if self.amdgpu { ok } else { no })?;
        writeln!(f, "  thunderbolt:   {}", if self.usb4_driver { ok } else { no })?;
        writeln!(f)?;
        writeln!(f, "── USB4 Latency Tuning (paper §5.1.3) ──")?;
        writeln!(f, "  ASPM:          {:<12}  (needs pcie_aspm=off: -71% P99 RTT)", self.aspm)?;
        writeln!(f, "  TCP CC:        {:<12}  (needs bbr: avoids bufferbloat)", if self.bbr { ok } else { "cubic" })?;
        writeln!(f, "  Busy poll:     {} µs           (needs 50: reduces IRQ latency)", self.busy_poll)?;
        writeln!(f, "  Governor:      {:<12}  (needs performance: prevents USB4 controller sleep)", self.governor)?;
        writeln!(f, "  EPP:           {:<12}  (needs performance: AMD P-State tuning)", self.epp)?;
        writeln!(f)?;
        writeln!(f, "── Measured Impact (paper Table 2) ──")?;
        writeln!(f, "  P99 RTT before tuning:  97 µs")?;
        writeln!(f, "  P99 RTT after tuning:   27.85 µs (71% reduction)")?;
        writeln!(f, "  Min RTT after tuning:   14.13 µs")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum AspmStatus {
    Disabled,
    Enabled(String, String),
    Unknown(String),
}

impl std::fmt::Display for AspmStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AspmStatus::Disabled => write!(f, "✅ Disabled"),
            AspmStatus::Enabled(pol, state) => write!(f, "❌ pol={} state={}", pol, state),
            AspmStatus::Unknown(s) => write!(f, "❓ {}", s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuner_check() {
        let r = KernelTuner::check();
        println!("{}", r);
    }
}
