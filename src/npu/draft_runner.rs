use crate::config::HccConfig;
use crate::decoding::speculative::DraftToken;

/// XDNA 2 NPU draft model runner.
///
/// From §2.2: XDNA 2 features AIE-ML v2 spatial dataflow:
///   - 2D grid of 32 independent compute tiles
///   - Each tile: VLIW SIMD vector processor + scalar RISC core + 64 KB SRAM
///   - 512 KB shared memory tiles
///   - Block FP16 (BFP16) for INT8-speed activations
///
/// From §12: Community implementations demonstrate 8B dense models at 43.7 T/s
/// on XDNA 2 NPU (41.5W). An 8B draft easily outpaces iGPU main model's 11.1 T/s.
pub struct DraftRunner {
    /// Name/identifier of the draft model checkpoint.
    model_id: String,
    /// Draft model parameter count (paper: 8B).
    params_b: f64,
    /// Token embedding dimension.
    hidden_size: usize,
    /// Max draft length.
    max_draft_len: usize,
    /// Current KV state (simplified).
    kv_state: Vec<f32>,
    /// Context tokens.
    context: Vec<u32>,
    /// In production: XRT context handle, Vitis AI VOE session.
    xrt_handle: Option<u64>,
}

impl DraftRunner {
    pub fn new(cfg: &HccConfig, draft_params_b: f64) -> Self {
        Self {
            model_id: format!("draft-{:.0}B", draft_params_b),
            params_b: draft_params_b,
            hidden_size: cfg.model.hidden_size,
            max_draft_len: cfg.speculative.draft_len,
            kv_state: Vec::new(),
            context: Vec::new(),
            xrt_handle: None,
        }
    }

    /// Prefill context — runs the prompt through the draft model on the NPU.
    ///
    /// §2.2: XDNA 2 processes BFP16 activations on its spatial dataflow array.
    /// §6.2: NPU handles high-arithmetic-intensity prefill at <5W.
    pub async fn prefill_context(&mut self, tokens: &[u8]) -> anyhow::Result<()> {
        let num_tokens = tokens.len() / (self.hidden_size * 2); // FP16 = 2 bytes
        tracing::debug!(
            "NPU draft: prefilling {num_tokens} tokens (model={})",
            self.model_id
        );

        // In production: XRT exec buffer write + NPU launch via Vitis AI VOE
        self.context.extend(tokens.iter().map(|&b| b as u32));
        self.kv_state = vec![0.0f32; num_tokens * self.hidden_size];

        Ok(())
    }

    /// Generate γ draft tokens on the NPU.
    ///
    /// §7, Algorithm 1 line 2: "Generate draft sequence using S on NPU."
    /// Returns γ DraftToken structs with token IDs and probabilities.
    pub async fn generate_drafts(&self, gamma: usize) -> anyhow::Result<Vec<DraftToken>> {
        tracing::trace!(
            "NPU draft: generating {gamma} tokens (model={})",
            self.model_id
        );

        // In production: NPU auto-regressive decode loop via XRT
        let mut drafts = Vec::with_capacity(gamma);
        for _ in 0..gamma {
            drafts.push(DraftToken {
                token_id: fastrand::u32(..) % 32000,       // vocab-bound
                probability: 0.5 + fastrand::f64() * 0.5, // simulated
                kv_state: vec![fastrand::f32(); self.hidden_size],
            });
        }

        Ok(drafts)
    }

    /// Accept tokens back from verification (Algorithm 1 line 5).
    pub async fn accept_tokens(
        &mut self,
        token_ids: &[u32],
        kv_deltas: &[f32],
    ) -> anyhow::Result<()> {
        self.context.extend(token_ids);
        self.kv_state.extend_from_slice(kv_deltas);
        Ok(())
    }

    /// Clear KV state for new session.
    pub fn reset(&mut self) {
        self.kv_state.clear();
        self.context.clear();
    }

    pub fn model_id(&self) -> &str { &self.model_id }
    pub fn context_len(&self) -> usize { self.context.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HccConfig;

    #[tokio::test]
    async fn test_draft_generation() {
        let cfg = HccConfig::default();
        let runner = DraftRunner::new(&cfg, 8.0);
        let drafts = runner.generate_drafts(5).await;
        assert!(drafts.is_ok());
        assert_eq!(drafts.unwrap().len(), 5);
    }
}
