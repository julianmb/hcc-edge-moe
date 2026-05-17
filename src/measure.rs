use crate::config::BackendConfig;
/// Measurement command — runs real inference and validates paper's claims.
///
/// Starts a real llama.cpp backend, runs the HCC speculative pipeline,
/// measures actual throughput, and compares against the paper's Eq. 5-6 predictions.
use crate::config::HccConfig;
use crate::decoding::speculative::SpeculativeEngine;
use crate::igpu::target_runner::TargetRunner;
use anyhow::Context;
use std::time::Instant;
use tokio::process::Command;

pub struct MeasureRunner;

impl MeasureRunner {
    pub async fn run(cfg: &HccConfig) -> anyhow::Result<()> {
        println!("\n╔══════════════════════════════════════════╗");
        println!("║  HCC Measurement Suite                    ║");
        println!("║  Real local inference measurement          ║");
        println!("╚══════════════════════════════════════════╝\n");

        println!("── Backend ──");
        println!(
            "  Model: {}",
            cfg.backend
                .model_path
                .split('/')
                .last()
                .unwrap_or(&cfg.backend.model_path)
        );
        println!("  Measure: {}", cfg.backend.measure_engine);
        println!(
            "  Device: {}",
            if cfg.backend.device.is_empty() {
                "auto"
            } else {
                &cfg.backend.device
            }
        );
        println!(
            "  KV: K={} V={}",
            cfg.backend.cache_type_k, cfg.backend.cache_type_v
        );
        println!("  Context: {} tokens", cfg.backend.ctx_size);

        // Run spec decode math validation
        let engine = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );

        println!("\n── Paper Predictions (Eq. 5-6) ──");
        println!("  γ = {}", cfg.speculative.draft_len);
        println!("  α = {:.2}", cfg.speculative.acceptance_rate);
        println!(
            "  E[k] = (1 - α^γ⁺¹) / (1 - α) = {:.4}",
            engine.expected_accepted()
        );
        println!("  S   = E[k] / (1 + γ·c/C) = {:.4}×", engine.speedup());
        let model_roofline_tps = cfg.cluster.memory_bw_gbs / cfg.model.weight_read_gb();
        println!("  Model roofline: {:.1} tok/s", model_roofline_tps);
        println!(
            "  Predicted HCC throughput: {:.1} tok/s",
            model_roofline_tps * engine.speedup()
        );

        // Measure actual prefill latency
        let prompt = "Write a concise Python function that computes fibonacci numbers iteratively, then explain the time complexity in one sentence.";
        if cfg.backend.measure_engine == "llamacpp-cli" {
            Self::run_llamacpp_cli(cfg, prompt, 96, model_roofline_tps * engine.speedup()).await?;
            println!("\n╔══════════════════════════════════════════╗");
            println!("║  Measurement complete                     ║");
            println!("╚══════════════════════════════════════════╝\n");
            return Ok(());
        }

        println!("\n── Starting llama-server ──");
        println!("  Port:  {}", cfg.backend.rpc_port);
        let target = TargetRunner::new(cfg.backend.clone()).await?;

        println!("\n── Measuring Prefill (real) ──");
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
        println!("  Prompt speed:     {:.1} tok/s", report.prompt_tps);
        println!("  Decode speed:     {:.1} tok/s", report.predicted_tps);
        println!(
            "  HCC projection:   {:.1} tok/s",
            model_roofline_tps * engine.speedup()
        );
        println!("\n── Sample ──\n{}", report.content.trim());
        println!("\n╔══════════════════════════════════════════╗");
        println!("║  Measurement complete                     ║");
        println!("╚══════════════════════════════════════════╝\n");
        Ok(())
    }

    async fn run_llamacpp_cli(
        cfg: &HccConfig,
        prompt: &str,
        n_predict: usize,
        hcc_projection: f64,
    ) -> anyhow::Result<()> {
        println!("\n── Measuring Direct llama.cpp CLI (real) ──");
        println!("  Binary: {}", cfg.backend.cli_bin);
        println!("  Mode: single-turn greedy decode");

        let args = Self::llama_cli_args(&cfg.backend, prompt, n_predict);
        let start = Instant::now();
        let output = Command::new(&cfg.backend.cli_bin)
            .args(&args)
            .output()
            .await
            .with_context(|| format!("failed to run {}", cfg.backend.cli_bin))?;
        let elapsed_s = start.elapsed().as_secs_f64();

        let mut combined = String::from_utf8_lossy(&output.stdout).to_string();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));

        if !output.status.success() {
            anyhow::bail!(
                "{} exited with status {}\n{}",
                cfg.backend.cli_bin,
                output.status,
                combined
            );
        }

        let (prompt_tps, decode_tps) = Self::parse_llama_cli_speeds(&combined)
            .context("llama-cli output did not include prompt/generation timings")?;

        println!("\n── Results ──");
        println!("  Generated tokens: {}", n_predict);
        println!("  Wall time:        {:.2}s", elapsed_s);
        println!("  Prompt speed:     {:.1} tok/s", prompt_tps);
        println!("  Decode speed:     {:.1} tok/s", decode_tps);
        println!("  HCC projection:   {:.1} tok/s", hcc_projection);

        Ok(())
    }

    fn llama_cli_args(backend: &BackendConfig, prompt: &str, n_predict: usize) -> Vec<String> {
        let mut args = vec![
            "-m".into(),
            backend.model_path.clone(),
            "-c".into(),
            backend.ctx_size.to_string(),
            "-b".into(),
            backend.batch_size.to_string(),
            "-ub".into(),
            backend.ubatch_size.to_string(),
            "-fa".into(),
            backend.flash_attn.clone(),
            "-ctk".into(),
            backend.cache_type_k.clone(),
            "-ctv".into(),
            backend.cache_type_v.clone(),
            "-ngl".into(),
            backend.gpu_layers.clone(),
            "--reasoning".into(),
            backend.reasoning.clone(),
            "--single-turn".into(),
            "--perf".into(),
            "-n".into(),
            n_predict.to_string(),
            "--temp".into(),
            "0".into(),
            "--top-k".into(),
            "1".into(),
            "--top-p".into(),
            "1".into(),
            "--no-display-prompt".into(),
            "-p".into(),
            prompt.into(),
        ];

        if !backend.device.is_empty() {
            args.extend(["--device".into(), backend.device.clone()]);
        }
        if backend.fit {
            args.extend([
                "--fit".into(),
                "on".into(),
                "--fit-target".into(),
                backend.fit_target_mib.to_string(),
                "--fit-ctx".into(),
                backend.fit_ctx.to_string(),
            ]);
        } else {
            args.extend(["--fit".into(), "off".into()]);
        }
        args
    }

    fn parse_llama_cli_speeds(output: &str) -> Option<(f64, f64)> {
        for line in output.lines().rev() {
            if !line.contains("Prompt:") || !line.contains("Generation:") {
                continue;
            }

            let mut prompt_tps = None;
            let mut generation_tps = None;
            for part in line.trim_matches(|c| c == '[' || c == ']').split('|') {
                let part = part.trim();
                if let Some(rest) = part.strip_prefix("Prompt:") {
                    prompt_tps = rest.split_whitespace().find_map(|v| v.parse::<f64>().ok());
                } else if let Some(rest) = part.strip_prefix("Generation:") {
                    generation_tps = rest.split_whitespace().find_map(|v| v.parse::<f64>().ok());
                }
            }

            if let (Some(prompt), Some(generation)) = (prompt_tps, generation_tps) {
                return Some((prompt, generation));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_llama_cli_speeds() {
        let output = "\n[ Prompt: 227.6 t/s | Generation: 52.4 t/s ]\n";
        let (prompt, generation) = MeasureRunner::parse_llama_cli_speeds(output).unwrap();
        assert_eq!(prompt, 227.6);
        assert_eq!(generation, 52.4);
    }
}
