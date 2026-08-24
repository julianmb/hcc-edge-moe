/// Feasibility subcommand: evaluate configured capacity and roofline assumptions.
///
/// This command does not load a model or measure inference. Use `measure` for
/// observed prompt/decode performance.
use crate::config::HccConfig;
use crate::decoding::speculative::SpeculativeEngine;
use crate::tuner::KernelTuner;

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run all benchmarks and print results.
    pub fn run_all(cfg: &HccConfig) -> anyhow::Result<BenchmarkReport> {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║     HCC Feasibility Projection                   ║");
        println!("╚══════════════════════════════════════════╝\n");

        println!("  No model is loaded; throughput is projected.\n");

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

        // 2. Capacity check
        let cluster_memory_gb = cfg.cluster_memory_gb();
        println!("\n── Model Capacity ──");
        println!("Model:               {}", cfg.model.model_name);
        if cfg.model.total_params_b > 0.0 {
            println!("Total parameters:    {:.0}B", cfg.model.total_params_b);
        }
        println!("Active per token:    {:.1}B", cfg.model.active_params_b);
        println!("Cluster memory:      {:.1} GB gross", cluster_memory_gb);
        if let Some(headroom_gb) = cfg.checkpoint_headroom_gb() {
            println!(
                "Checkpoint size:     {:.1} GB",
                cfg.model.checkpoint_size_gb
            );
            println!(
                "Gross capacity fit:  {} ({:.1} GB before runtime + KV)",
                if headroom_gb >= 0.0 { "YES" } else { "NO" },
                headroom_gb
            );
        } else {
            println!("Checkpoint size:     unknown (set model.checkpoint_size_gb)");
        }

        // 3. Roofline analysis
        let bw = cfg.cluster.memory_bw_gbs;
        let active_w = cfg.model.weight_read_gb();
        let decode_tps = bw / active_w.max(0.001);
        report.bandwidth_gbs = bw;
        report.theoretical_decode_tps = decode_tps;
        report.decode_tps_moe = decode_tps;

        println!("\n── Decode Roofline (Projected) ──");
        println!("Memory bandwidth:    {:.0} GB/s", bw);
        println!(
            "Active weights:      {:.1} GB ({}B @ {:.2} B/weight)",
            active_w, cfg.model.active_params_b, cfg.model.bytes_per_weight
        );
        println!("Decode roofline:     {:.1} tok/s", decode_tps);

        // 4. Speculative speedup
        let has_draft = cfg.speculative.draft_params_b > 0.0;
        let (ek, speedup) = if has_draft {
            let eng = SpeculativeEngine::new(
                cfg.speculative.draft_len,
                cfg.speculative.acceptance_rate,
                cfg.speculative.draft_cost_ratio,
            );
            (eng.expected_accepted(), eng.speedup())
        } else {
            (1.0, 1.0)
        };
        report.spec_multiplier = speedup;

        println!(
            "\n── Speculative Decoding ({}) ──",
            if has_draft { "Assumed" } else { "Disabled" }
        );
        if has_draft {
            println!("Draft length γ:      {}", cfg.speculative.draft_len);
            println!(
                "Acceptance rate α:   {:.2}",
                cfg.speculative.acceptance_rate
            );
            println!("E[k] (Eq. 5):        {:.3}", ek);
            println!("Speedup S (Eq. 6):   {:.3}×", speedup);
        } else {
            println!("No draft model configured (draft_params_b = 0).");
        }
        println!("Effective decode:    {:.1} tok/s", decode_tps * speedup);

        report.effective_tps = decode_tps * speedup;

        // 5. Configured backend details. No hardware is probed here.
        println!("\n── Configured Backend ──");
        println!("Backend: {}", cfg.backend.inference_engine);
        if cfg.backend.inference_engine == "llamacpp-rpc" {
            println!("Backend: llama.cpp RPC :{}", cfg.backend.rpc_port);
            println!(
                "hipBLASLt: {}",
                if cfg.backend.hipblaslt {
                    "✅ enabled"
                } else {
                    "❌ disabled"
                }
            );
            println!("GPU: gfx1151 (Radeon 8060S, 40 CUs @ 2.9 GHz)");
            println!(
                "Expected matmul perf: {:.1} TFLOPS (62% of 59.4 peak)",
                59.4 * 0.62
            );
        }

        // 6. Config summary
        println!("\n── Configuration ──");
        println!("Pipeline: {}", cfg.backend.pipeline);
        println!("Nodes: {}", cfg.cluster.node_count);
        println!("Memory per node: {} GB", cfg.cluster.memory_per_node_gb);
        println!(
            "USB4 links: {} ({:.0} Gbps, {:.0} µs RTT)",
            cfg.interconnect.link_count, cfg.interconnect.throughput_gbps, cfg.interconnect.rtt_us
        );

        println!("\n╔══════════════════════════════════════════╗");
        println!("║  Projection complete                      ║");
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
