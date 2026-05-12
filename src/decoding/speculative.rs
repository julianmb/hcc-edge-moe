/// Speculative decoding engine — implements the mathematics from §7.
///
/// Core speedup equations from the paper:
/// ```
/// E[k] = (1 - α^{γ+1}) / (1 - α)           (Eq. 5)
/// S = E[k] / (1 + γ · c/C)                  (Eq. 6)
/// ```
pub struct SpeculativeEngine {
    pub draft_len: usize,
    pub acceptance_rate: f64,
    pub draft_cost_ratio: f64,
}

impl SpeculativeEngine {
    pub fn new(draft_len: usize, acceptance_rate: f64, draft_cost_ratio: f64) -> Self {
        Self {
            draft_len,
            acceptance_rate,
            draft_cost_ratio,
        }
    }

    /// Expected number of accepted tokens per network crossing — Eq. 5.
    ///
    /// E[k] = (1 - α^{γ+1}) / (1 - α)
    pub fn expected_accepted(&self) -> f64 {
        let alpha = self.acceptance_rate;
        let gamma = self.draft_len as f64;
        (1.0 - alpha.powf(gamma + 1.0)) / (1.0 - alpha)
    }

    /// Theoretical speedup of the speculative pipeline — Eq. 6.
    ///
    /// S = E[k] / (1 + γ · c/C)
    pub fn speedup(&self) -> f64 {
        let ek = self.expected_accepted();
        let gamma = self.draft_len as f64;
        let overhead = 1.0 + gamma * self.draft_cost_ratio;
        ek / overhead
    }

    /// Effective decode throughput multiplier under this spec config.
    ///
    /// Paper §10.2: T_HCC = 11.1 × 2.35 ≈ 26.1 T/s
    /// with multiplier = 1 − 0.7^6 / (1 − 0.7)(1 + 5 × 0.05) ≈ 2.35
    pub fn throughput_multiplier(&self) -> f64 {
        self.speedup()
    }

    /// Rejection sampling — accepts up to k ≤ γ tokens.
    ///
    /// Paper Algorithm 1, line 4:
    /// "Accept tokens up to position k ≤ γ using rejection sampling."
    pub fn rejection_sample(
        &self,
        drafts: &[DraftToken],
        verified: &[VerifiedToken],
    ) -> usize {
        let max_k = drafts.len().min(verified.len());
        for k in 0..max_k {
            if !self.accept_token(&drafts[k], &verified[k]) {
                return k;
            }
        }
        max_k
    }

    /// Accept a single token if p_target(x) / p_draft(x) ≥ uniform(0,1].
    fn accept_token(&self, draft: &DraftToken, target: &VerifiedToken) -> bool {
        // Standard rejection sampling from Leviathan et al.
        let ratio = target.probability / draft.probability.max(1e-30);
        ratio >= 1.0 || ratio > fastrand::f64()
    }

    /// Optimal draft length γ* that maximizes speedup.
    ///
    /// Found by solving dS/dγ = 0. We brute-force search up to max_gamma
    /// since the closed form involves α^{γ} terms.
    pub fn optimal_draft_len(&self, max_gamma: usize) -> usize {
        let mut best = 1usize;
        let mut best_s = 0.0f64;
        let base_alpha = self.acceptance_rate;
        let cost = self.draft_cost_ratio;

        for g in 1..=max_gamma {
            let ek = if base_alpha < 1.0 {
                (1.0 - base_alpha.powi(g as i32 + 1)) / (1.0 - base_alpha)
            } else {
                (g + 1) as f64
            };
            let s = ek / (1.0 + g as f64 * cost);
            if s > best_s {
                best_s = s;
                best = g;
            }
        }
        best
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DraftToken {
    pub token_id: u32,
    pub probability: f64,
    pub kv_state: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifiedToken {
    pub token_id: u32,
    pub probability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_accepted() {
        let eng = SpeculativeEngine::new(5, 0.7, 0.05);
        let ek = eng.expected_accepted();
        // E[k] = (1 - 0.7^6) / (1 - 0.7) = (1 - 0.117649) / 0.3
        // = 0.882351 / 0.3 ≈ 2.941
        assert!((ek - 2.941).abs() < 0.01, "E[k]={ek}");
    }

    #[test]
    fn test_speedup() {
        let eng = SpeculativeEngine::new(5, 0.7, 0.05);
        let s = eng.speedup();
        // S = 2.941 / (1 + 5*0.05) = 2.941 / 1.25 ≈ 2.353
        assert!((s - 2.353).abs() < 0.01, "S={s}");
    }

    #[test]
    fn test_throughput_multiplier() {
        let eng = SpeculativeEngine::new(5, 0.7, 0.05);
        let m = eng.throughput_multiplier();
        // Paper §10.2: 11.1 × 2.35 ≈ 26.1 T/s
        assert!((m - 2.35).abs() < 0.02, "multiplier={m}");
    }

    #[test]
    fn test_optimal_draft_len() {
        let eng = SpeculativeEngine::new(5, 0.7, 0.05);
        let g = eng.optimal_draft_len(20);
        assert!(g >= 3 && g <= 8, "optimal γ={g} should be in [3,8]");
    }
}
