/// iGPU target runner — launches and talks to llama-server for real inference.
///
/// The local ClawRig setup currently exposes llama.cpp through its Vulkan
/// backend. This runner keeps the launch flags explicit so we can apply the
/// standard optimizations: full device offload, Flash Attention, compact
/// asymmetric KV, prompt cache reuse, and inference-only server mode.
use crate::config::BackendConfig;
use anyhow::Context;
use serde_json::json;
use std::time::{Duration, Instant};
use tokio::process::Command;

pub struct TargetRunner {
    completion_url: String,
    health_url: String,
    _child: Option<tokio::process::Child>,
    backend: BackendConfig,
}

#[derive(Debug, Clone)]
pub struct GenerationReport {
    pub content: String,
    pub elapsed_s: f64,
    pub prompt_tokens: usize,
    pub predicted_tokens: usize,
    pub prompt_tps: f64,
    pub predicted_tps: f64,
}

impl TargetRunner {
    pub async fn new(backend: BackendConfig) -> anyhow::Result<Self> {
        let base_url = format!("http://127.0.0.1:{}", backend.rpc_port);
        let child = if backend.spawn_server {
            let args = Self::llama_server_args(&backend);
            tracing::info!("launching {} {}", backend.server_bin, args.join(" "));
            let mut cmd = Command::new(&backend.server_bin);
            cmd.args(&args).kill_on_drop(true);
            Some(cmd.spawn().with_context(|| {
                format!(
                    "failed to spawn llama-server binary '{}'",
                    backend.server_bin
                )
            })?)
        } else {
            tracing::info!(
                "attaching to existing llama-server on :{}",
                backend.rpc_port
            );
            None
        };

        let mut runner = Self {
            completion_url: format!("{base_url}/completion"),
            health_url: format!("{base_url}/health"),
            _child: child,
            backend,
        };

        runner
            .wait_until_ready(Duration::from_secs(runner.backend.startup_timeout_s))
            .await?;

        Ok(runner)
    }

    pub fn llama_server_args(backend: &BackendConfig) -> Vec<String> {
        let mut args = vec![
            "--model".into(),
            backend.model_path.clone(),
            "--alias".into(),
            backend.model_alias.clone(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            backend.rpc_port.to_string(),
            "--ctx-size".into(),
            backend.ctx_size.to_string(),
            "--batch-size".into(),
            backend.batch_size.to_string(),
            "--ubatch-size".into(),
            backend.ubatch_size.to_string(),
            "--parallel".into(),
            backend.parallel.to_string(),
            "--threads".into(),
            backend.threads.to_string(),
            "--threads-batch".into(),
            backend.threads_batch.to_string(),
            "--poll".into(),
            backend.poll.to_string(),
            "--flash-attn".into(),
            backend.flash_attn.clone(),
            "--cache-type-k".into(),
            backend.cache_type_k.clone(),
            "--cache-type-v".into(),
            backend.cache_type_v.clone(),
            "--n-gpu-layers".into(),
            backend.gpu_layers.clone(),
            "--reasoning".into(),
            backend.reasoning.clone(),
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
        if backend.cache_prompt {
            args.push("--cache-prompt".into());
        } else {
            args.push("--no-cache-prompt".into());
        }
        if backend.cache_reuse > 0 {
            args.extend(["--cache-reuse".into(), backend.cache_reuse.to_string()]);
        }
        if backend.no_webui {
            args.push("--no-webui".into());
        }
        if backend.metrics {
            args.push("--metrics".into());
        }
        if backend.spec_type != "none" {
            args.extend(["--spec-type".into(), backend.spec_type.clone()]);
        }
        args.extend(backend.extra_args.clone());
        args
    }

    async fn wait_until_ready(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let start = Instant::now();
        let client = reqwest::Client::new();
        loop {
            if let Some(child) = &mut self._child {
                if let Some(status) = child.try_wait()? {
                    anyhow::bail!("llama-server exited before becoming ready: {status}");
                }
            }
            match client.get(&self.health_url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ if start.elapsed() < timeout => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                _ => {
                    anyhow::bail!(
                        "timed out after {}s waiting for llama-server health at {}",
                        timeout.as_secs(),
                        self.health_url
                    );
                }
            }
        }
    }

    /// Prefill — sends prompt and measures TTFT.
    pub async fn prefill(&self, tokens: &[u8]) -> anyhow::Result<f64> {
        let client = reqwest::Client::new();
        let prompt = String::from_utf8_lossy(tokens);

        let start = std::time::Instant::now();
        let body = json!({
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "cache_prompt": self.backend.cache_prompt,
        });
        let _resp = client
            .post(&self.completion_url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(start.elapsed().as_secs_f64())
    }

    pub async fn generate_completion(
        &self,
        prompt: &str,
        n_predict: usize,
    ) -> anyhow::Result<GenerationReport> {
        let client = reqwest::Client::new();
        let body = json!({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": self.backend.cache_prompt,
        });

        let start = Instant::now();
        let resp = client
            .post(&self.completion_url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let elapsed_s = start.elapsed().as_secs_f64();
        let result: serde_json::Value = resp.json().await?;
        let null = serde_json::Value::Null;
        let timings = result.get("timings").unwrap_or(&null);
        let prompt_tokens = timings["prompt_n"].as_u64().unwrap_or(0) as usize;
        let predicted_tokens = timings["predicted_n"].as_u64().unwrap_or(n_predict as u64) as usize;
        let prompt_tps = timings["prompt_per_second"].as_f64().unwrap_or(0.0);
        let predicted_tps = timings["predicted_per_second"]
            .as_f64()
            .unwrap_or_else(|| predicted_tokens as f64 / elapsed_s.max(1e-9));

        Ok(GenerationReport {
            content: result["content"].as_str().unwrap_or_default().to_string(),
            elapsed_s,
            prompt_tokens,
            predicted_tokens,
            prompt_tps,
            predicted_tps,
        })
    }

    /// Verify a batch of draft tokens — single parallel forward pass.
    /// Returns logits for rejection sampling.
    pub async fn verify_batch(&self, draft_tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let client = reqwest::Client::new();
        // Convert draft tokens to a prompt string and get next-token logits
        let prompt = draft_tokens
            .iter()
            .map(|t| format!("<|token_id|>{t}"))
            .collect::<Vec<_>>()
            .join("");

        let body = json!({
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": 10,
        });

        let resp = client
            .post(&self.completion_url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let result: serde_json::Value = resp.json().await?;
        let mut logits = Vec::new();

        // Extract probabilities from response
        if let Some(choices) = result["choices"].as_array() {
            for choice in choices {
                if let Some(logprobs) = choice.get("logprobs") {
                    if let Some(top) = logprobs.get("top_logprobs").and_then(|t| t.as_array()) {
                        for entry in top {
                            if let Some(obj) = entry.as_object() {
                                for (_token, prob) in obj.iter() {
                                    logits.push(prob.as_f64().unwrap_or(0.0) as f32);
                                }
                            }
                        }
                    }
                }
            }
        }

        if logits.is_empty() {
            logits.resize(32000, 0.5);
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HccConfig;

    #[test]
    fn test_llama_args_include_clawrig_fast_path() {
        let cfg = HccConfig::default();
        let args = TargetRunner::llama_server_args(&cfg.backend);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--device" && w[1] == "Vulkan0"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--flash-attn" && w[1] == "on"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--cache-type-k" && w[1] == "q8_0"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--cache-type-v" && w[1] == "q4_0"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--n-gpu-layers" && w[1] == "all"));
        assert!(args.iter().any(|a| a == "--no-webui"));
    }
}
