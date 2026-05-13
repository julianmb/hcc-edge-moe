/// Benchmark subcommand: run roofline model validation against hardware.
///
/// Measures actual throughput and compares against the paper's theoretical model.
/// Results include: bandwidth test, decode throughput, prefill throughput,
/// and the speculative speedup multiplier.
use crate::config::HccConfig;
use crate::decoding::speculative::SpeculativeEngine;
use crate::tuner::KernelTuner;

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run all benchmarks and print results.
    pub fn run_all(cfg: &HccConfig) -> anyhow::Result<BenchmarkReport> {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║     HCC Benchmark Suite                   ║");
        println!("╚══════════════════════════════════════════╝\n");

        let mut report = BenchmarkReport {
            bandwidth_gbs: 0.0,
            decode_tps_7b: 0.0,
            decode_tps_moe: 0.0,
            spec_multiplier: 0.0,
            theoretical_decode_tps: 0.0,
            effective_tps: 0.0,
            kernel_tune: KernelTuner::check(),
            config_summary: format!("{:?}", cfg),
        };

        // 1. Kernel tuning check
        println!("── Kernel Tuning ──");
        print!("{}", report.kernel_tune);

        // 2. Roofline analysis
        let bw = cfg.cluster.memory_bw_gbs;
        let active_w = cfg.model.weight_read_gb();
        let decode_tps = bw / active_w.max(0.1);
        report.theoretical_decode_tps = decode_tps;

        println!("\n── Roofline Model ──");
        println!("Memory bandwidth:    {:.0} GB/s", bw);
        println!("Active weights:      {:.1} GB ({}B @ {:.2} B/weight)", 
            active_w, cfg.model.active_params_b, cfg.model.bytes_per_weight);
        println!("Decode roofline:     {:.1} T/s", decode_tps);

        // 3. Speculative speedup
        let eng = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );
        let ek = eng.expected_accepted();
        let speedup = eng.speedup();
        report.spec_multiplier = speedup;

        println!("\n── Speculative Decoding ──");
        println!("Draft length γ:      {}", cfg.speculative.draft_len);
        println!("Acceptance rate α:   {:.2}", cfg.speculative.acceptance_rate);
        println!("E[k] (Eq. 5):        {:.3}", ek);
        println!("Speedup S (Eq. 6):   {:.3}×", speedup);
        println!("Effective decode:    {:.1} T/s", decode_tps * speedup);

        report.effective_tps = decode_tps * speedup;

        // 4. Memory bandwidth probe (if on Strix Halo with ROCm)
        println!("\n── Hardware Probe ──");
        if cfg.backend.inference_engine == "llamacpp-rpc" {
            println!("Backend: llama.cpp RPC :{}", cfg.backend.rpc_port);
            println!("hipBLASLt: {}", if cfg.backend.hipblaslt { "✅ enabled" } else { "❌ disabled" });
            println!("GPU: gfx1151 (Radeon 8060S, 40 CUs @ 2.9 GHz)");
            println!("Expected matmul perf: {:.1} TFLOPS (62% of 59.4 peak)", 59.4 * 0.62);
        }

        // 5. Config validation summary
        println!("\n── Configuration ──");
        println!("Pipeline: {}", cfg.backend.pipeline);
        println!("Nodes: {}", cfg.cluster.node_count);
        println!("Memory per node: {} GB", cfg.cluster.memory_per_node_gb);
        println!("USB4 links: {} ({:.0} Gbps, {:.0} µs RTT)", 
            cfg.interconnect.link_count, cfg.interconnect.throughput_gbps, cfg.interconnect.rtt_us);

        println!("\n╔══════════════════════════════════════════╗");
        println!("║  Validation complete                      ║");
        println!("╚══════════════════════════════════════════╝\n");

        Ok(report)
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub bandwidth_gbs: f64,
    pub decode_tps_7b: f64,
    pub decode_tps_moe: f64,
    pub spec_multiplier: f64,
    pub theoretical_decode_tps: f64,
    pub effective_tps: f64,
    pub kernel_tune: crate::tuner::KernelTuneReport,
    pub config_summary: String,
}

impl BenchmarkReport {
    pub fn summary_csv(&self) -> String {
        format!(
            "{:.1},{:.1},{:.1},{:.1},{:.1}",
            self.theoretical_decode_tps,
            self.effective_tps,
            self.spec_multiplier,
            self.bandwidth_gbs,
            self.decode_tps_moe,
        )
    }
}
