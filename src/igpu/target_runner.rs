use crate::config::HccConfig;
use crate::decoding::speculative::VerifiedToken;

/// RDNA 3.5 iGPU target model runner — executes the main MoE.
///
/// From §6.1:
///   - Token generation: low arithmetic intensity (I ≈ 1), strictly memory-bound
///   - At 212 GB/s on RDNA 3.5 iGPU, decode bound across 40B active params
///     (19.1 GB at 0.48 bytes/weight) is ~11.1 T/s per node
///
/// From §9:
///   - Compiled via MIGraphX (ROCm 7.2.1) with matured MoE routing kernels
///   - HIP execution graph natively waits on XRT hardware event flags
pub struct TargetRunner {
    model_path: String,
    hidden_size: usize,
    num_layers: usize,
    num_experts: usize,
    top_k: usize,
    active_weight_gb: f64,
    memory_bw_gbs: f64,
    /// KV cache state.
    kv_cache: Vec<f32>,
    context: Vec<u32>,
}

impl TargetRunner {
    pub fn new(cfg: &HccConfig, model_path: String) -> Self {
        let active_weight_gb = cfg.model.active_params_b * cfg.model.bytes_per_weight;
        Self {
            model_path,
            hidden_size: cfg.model.hidden_size,
            num_layers: cfg.model.num_layers,
            num_experts: cfg.model.num_experts,
            top_k: cfg.model.top_k,
            active_weight_gb,
            memory_bw_gbs: cfg.cluster.memory_bw_gbs,
            kv_cache: Vec::new(),
            context: Vec::new(),
        }
    }

    /// Prefill phase — process prompt on iGPU.
    ///
    /// §6.1: Prefill operates on large prompt matrices with high arithmetic
    /// intensity (I ≫ 100). Node 1's iGPU handles Layers 0–38 (paper topology).
    pub async fn prefill(&mut self, tokens: &[u8]) -> anyhow::Result<()> {
        tracing::debug!(
            "iGPU: prefilling {} bytes on {}",
            tokens.len(),
            self.model_path
        );
        // In production: MIGraphX inference session, model compiled to iGPU
        self.context.extend(tokens.iter().map(|&b| b as u32));
        Ok(())
    }

    /// Verify a batch of draft tokens — single parallel forward pass.
    ///
    /// §7: "Node 2's iGPU performs a single memory-bound pass of the 40B active
    ///  weights to process all γ draft tokens in parallel."
    ///
    /// Returns logits and probabilities for rejection sampling.
    pub async fn verify_batch(&self, drafts: &[u8]) -> anyhow::Result<Vec<VerifiedToken>> {
        // In production: MIGraphX batched inference with MoE routing
        // Paper §9: HIP execution graph waits on XRT event flags
        let num_drafts = drafts.len() / (self.hidden_size * 4); // FP32 logits
        let tokens: Vec<VerifiedToken> = (0..num_drafts.max(1)).map(|_| {
            VerifiedToken {
                token_id: fastrand::u32(..) % 32000,
                probability: fastrand::f64(),
            }
        }).collect();

        Ok(tokens)
    }

    /// Insert token into KV cache.
    pub async fn insert_kv(&mut self, _token_id: u32, kv: &[f32]) -> anyhow::Result<()> {
        self.kv_cache.extend_from_slice(kv);
        Ok(())
    }

    /// Roofline model: theoretical decode throughput.
    ///
    /// §6.1: P = min(P_peak, I · BW_mem)
    /// For decode (I ≈ 1): throughput = BW_mem / weight_read
    pub fn theoretical_decode_tps(&self) -> f64 {
        if self.active_weight_gb <= 0.0 {
            return 0.0;
        }
        self.memory_bw_gbs / self.active_weight_gb
    }

    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.context.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HccConfig;

    #[test]
    fn test_theoretical_decode_tps() {
        let cfg = HccConfig::default();
        // 40B active @ 0.48 bytes/weight = 19.1 GB
        // BW = 212 GB/s → 212/19.1 ≈ 11.1 T/s
        let runner = TargetRunner::new(&cfg, "/models/test".into());
        let tps = runner.theoretical_decode_tps();
        assert!((tps - 11.1).abs() < 0.5, "tps={tps}");
    }
}
