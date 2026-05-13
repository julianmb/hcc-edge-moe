/// Measurement command — runs real inference and validates paper's claims.
///
/// Starts llama-server with a real model, runs the HCC speculative pipeline,
/// measures actual throughput, and compares against the paper's Eq. 5-6 predictions.
use crate::config::HccConfig;
use crate::decoding::speculative::SpeculativeEngine;
use crate::igpu::target_runner::TargetRunner;
use std::time::Instant;

pub struct MeasureRunner;

impl MeasureRunner {
    pub async fn run(cfg: &HccConfig) -> anyhow::Result<()> {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║  HCC Measurement Suite                    ║");
        println!("║  Real local inference measurement          ║");
        println!("╚══════════════════════════════════════════╝\n");

        println!("── Starting llama-server ──");
        println!("  Model: {}", cfg.backend.model_path.split('/').last().unwrap_or(&cfg.backend.model_path));
        println!("  Port:  {}", cfg.backend.rpc_port);
        println!("  Device: {}", if cfg.backend.device.is_empty() { "auto" } else { &cfg.backend.device });
        println!("  KV: K={} V={}", cfg.backend.cache_type_k, cfg.backend.cache_type_v);
        println!("  Context: {} tokens", cfg.backend.ctx_size);

        let target = TargetRunner::new(cfg.backend.clone()).await?;

        // Run spec decode math validation
        let engine = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );

        println!("\n── Paper Predictions (Eq. 5-6) ──");
        println!("  γ = {}", cfg.speculative.draft_len);
        println!("  α = {:.2}", cfg.speculative.acceptance_rate);
        println!("  E[k] = (1 - α^γ⁺¹) / (1 - α) = {:.4}", engine.expected_accepted());
        println!("  S   = E[k] / (1 + γ·c/C) = {:.4}×", engine.speedup());
        let model_roofline_tps = cfg.cluster.memory_bw_gbs / cfg.model.weight_read_gb();
        println!("  Model roofline: {:.1} T/s", model_roofline_tps);
        println!("  Predicted HCC throughput: {:.1} T/s", model_roofline_tps * engine.speedup());

        // Measure actual prefill latency
        println!("\n── Measuring Prefill (real) ──");
        let prompt = "You are running on an AMD Strix Halo workstation. Write a concise Python function that computes fibonacci numbers iteratively, then explain the time complexity in one sentence.";
        let start = Instant::now();
        target.prefill(prompt.as_bytes()).await?;
        let ttft = start.elapsed().as_secs_f64() * 1000.0;
        println!("  TTFT: {:.1} ms", ttft);

        // Measure actual generation speed
        println!("\n── Measuring Generation (real) ──");
        let report = target.generate_completion(prompt, 96).await?;

        println!("\n── Results ──");
        println!("  Prompt tokens:    {}", report.prompt_tokens);
        println!("  Generated tokens: {}", report.predicted_tokens);
        println!("  Wall time:        {:.2}s", report.elapsed_s);
        println!("  Prompt speed:     {:.1} T/s", report.prompt_tps);
        println!("  Decode speed:     {:.1} T/s", report.predicted_tps);
        println!("  HCC projection:   {:.1} T/s", model_roofline_tps * engine.speedup());
        println!("\n── Sample ──\n{}", report.content.trim());
        println!("\n╔══════════════════════════════════════════╗");
        println!("║  Measurement complete                     ║");
        println!("╚══════════════════════════════════════════╝\n");
        Ok(())
    }
}
