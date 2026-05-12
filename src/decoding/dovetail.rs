/// Dovetail-style CPU/GPU heterogeneous speculative decoding pipeline.
///
/// Reference: Zhang et al., "Dovetail: A CPU/GPU Heterogeneous Speculative
/// Decoding for LLM Inference", EMNLP 2025.
///
/// Key insight (reverses HCC's topology):
///   - Draft model runs on GPU (fast, small-batch optimized)
///   - Target model runs on CPU (large memory, slower per-token)
///   - Dynamic Gating Fusion (DGF) merges draft features with embeddings
///
/// Dovetail achieves 1.79x–10.1x speedup on consumer GPUs.
pub struct DovetailPipeline {
    /// Number of draft tokens per speculation round.
    pub draft_len: usize,
    /// Draft model depth (Dovetail found deeper drafts = better acceptance).
    pub draft_depth: usize,
    /// Dynamic Gating Fusion weight.
    pub dgf_alpha: f64,
    acceptance_history: Vec<f64>,
}

impl DovetailPipeline {
    pub fn new(draft_len: usize) -> Self {
        Self {
            draft_len,
            draft_depth: 4,
            dgf_alpha: 0.3,
            acceptance_history: Vec::with_capacity(100),
        }
    }

    /// Dynamic Gating Fusion: merges feature map F with embedding E.
    ///
    /// F' = α · F + (1 − α) · E
    pub fn dynamic_gating_fusion(&self, features: &[f32], embedding: &[f32]) -> Vec<f32> {
        features.iter().zip(embedding)
            .map(|(f, e)| self.dgf_alpha as f32 * f + (1.0 - self.dgf_alpha as f32) * e)
            .collect()
    }

    /// Acceptance rate-adjusted throughput projection.
    ///
    /// Dovetail shows that deeper draft models improve acceptance
    /// at the cost of increased draft latency. The optimal depth
    /// balances these factors.
    pub fn projected_speedup(&self, acceptance_rate: f64, target_latency_ms: f64, draft_latency_ms: f64) -> f64 {
        let expected_k = (1.0 - acceptance_rate.powi(self.draft_len as i32 + 1)) / (1.0 - acceptance_rate);
        let total_time = target_latency_ms + draft_latency_ms * self.draft_len as f64;
        let baseline_time = target_latency_ms * self.draft_len as f64;
        baseline_time / total_time * expected_k
    }

    /// Record an acceptance event for adaptive tuning.
    pub fn record_acceptance(&mut self, rate: f64) {
        self.acceptance_history.push(rate);
        if self.acceptance_history.len() > 100 {
            self.acceptance_history.remove(0);
        }
        // Adapt DGF alpha based on recent acceptance
        let avg: f64 = self.acceptance_history.iter().sum::<f64>() / self.acceptance_history.len() as f64;
        self.dgf_alpha = 0.5 - avg * 0.3; // lower alpha when acceptance is high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgf_merges() {
        let p = DovetailPipeline::new(5);
        let f = vec![1.0; 128];
        let e = vec![0.0; 128];
        let merged = p.dynamic_gating_fusion(&f, &e);
        // α=0.3: 0.3*1.0 + 0.7*0.0 = 0.3
        assert!((merged[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_adaptation() {
        let mut p = DovetailPipeline::new(5);
        p.record_acceptance(0.7);
        // α should decrease when acceptance is high
        assert!(p.dgf_alpha < 0.3);
    }
}
