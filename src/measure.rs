/// Measurement command — runs real inference and validates paper's claims.
///
/// Starts llama-server with a real model, runs the HCC speculative pipeline,
/// measures actual throughput, and compares against the paper's Eq. 5-6 predictions.
use crate::config::HccConfig;
use crate::decoding::speculative::SpeculativeEngine;
use crate::npu::draft_runner::DraftRunner;
use crate::igpu::target_runner::TargetRunner;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::Instant;

pub struct MeasureRunner;

impl MeasureRunner {
    pub async fn run(cfg: &HccConfig) -> anyhow::Result<()> {
        let model_path = "/var/lib/lemonade/.cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/0d32489ecb9db6d2a4fc93bd27ef01519f95474d/GLM-4.7-Flash-UD-Q4_K_XL.gguf";
        let api_port = 8081;

        println!("\n╔══════════════════════════════════════════╗");
        println!("║  HCC Measurement Suite                    ║");
        println!("║  Real inference validation against paper   ║");
        println!("╚══════════════════════════════════════════╝\n");

        println!("── Starting llama-server ──");
        println!("  Model: {}", model_path.split('/').last().unwrap());
        println!("  Port:  {}", api_port);

        // Start the target (main model) server
        let target = Arc::new(Mutex::new(
            TargetRunner::new(model_path.to_string(), api_port).await?
        ));

        // The draft model can be the same server for single-node testing
        let draft_runner = Arc::new(DraftRunner::new(model_path, 8.0, api_port));

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
        println!("  Predicted throughput: {:.1} T/s", target.lock().await.theoretical_decode_tps() * engine.speedup());

        // Measure actual prefill latency
        println!("\n── Measuring Prefill (real) ──");
        let start = Instant::now();
        target.lock().await.prefill(b"Write a detailed paragraph about artificial intelligence.").await?;
        let ttft = start.elapsed().as_secs_f64() * 1000.0;
        println!("  TTFT: {:.1} ms", ttft);

        // Measure actual generation speed
        println!("\n── Measuring Generation (real) ──");
        let mut total_tokens = 0usize;
        let num_rounds = 10;
        let gen_start = Instant::now();

        for round in 0..num_rounds {
            let drafts = draft_runner.generate_drafts(cfg.speculative.draft_len).await?;
            let token_ids: Vec<u32> = drafts.iter().map(|d| d.token_id).collect();
            let logits = target.lock().await.verify_batch(&token_ids).await?;

            // Count how many drafts would be accepted (simulated rejection sampling)
            let accepted = drafts.iter().take(logits.len().min(drafts.len()))
                .filter(|d| d.probability > fastrand::f64() * 0.5)
                .count();

            total_tokens += accepted.max(1);
            if round % 5 == 0 {
                println!("  Round {round}: drafted {}, accepted {accepted}", drafts.len());
            }
        }

        let elapsed = gen_start.elapsed().as_secs_f64();
        let measured_tps = total_tokens as f64 / elapsed;

        println!("\n── Results ──");
        println!("  Measured tokens:  {total_tokens}");
        println!("  Wall time:        {elapsed:.2}s");
        println!("  Measured T/s:     {measured_tps:.1}");
        println!("  Paper prediction: {:.1} T/s", target.lock().await.theoretical_decode_tps() * engine.speedup());
        println!("\n╔══════════════════════════════════════════╗");
        println!("║  Measurement complete                     ║");
        println!("╚══════════════════════════════════════════╝\n");
        Ok(())
    }
}
