/// Speculative decoding engine — implements the mathematics from §7.
///
/// Core speedup equations:
/// ```text
/// E[k] = (1 - α^{γ+1}) / (1 - α)           (Eq. 5)
/// S = E[k] / (v_γ + γ · c/C)               (Eq. 6, where v_γ accounts for MoE active-expert scaling)
/// ```
pub struct SpeculativeEngine {
    pub draft_len: usize,
    pub acceptance_rate: f64,
    pub draft_cost_ratio: f64,
    pub v_gamma: f64,
}

impl SpeculativeEngine {
    /// Create engine with default v_γ = 1.0 (dense baseline / paper Eq. 6).
    pub fn new(draft_len: usize, acceptance_rate: f64, draft_cost_ratio: f64) -> Self {
        Self {
            draft_len,
            acceptance_rate,
            draft_cost_ratio,
            v_gamma: 1.0,
        }
    }

    /// Create engine with explicit verification cost factor v_γ.
    pub fn with_v_gamma(
        draft_len: usize,
        acceptance_rate: f64,
        draft_cost_ratio: f64,
        v_gamma: f64,
    ) -> Self {
        Self {
            draft_len,
            acceptance_rate,
            draft_cost_ratio,
            v_gamma: v_gamma.max(0.1),
        }
    }

    /// Empirical and analytical active-expert scaling factor v_γ for MoE architectures.
    /// For dense models v_γ = 1.0. For MoE models (e.g. GLM-5.3-Flash, E=288, K=8),
    /// verifying γ draft tokens activates a larger union of experts across layers.
    pub fn moe_v_gamma(gamma: usize) -> f64 {
        match gamma {
            0 => 0.0,
            1 => 1.0,
            2 => 1.49,
            3 => 1.98,
            4 => 2.44,
            5 => 2.87,
            g => {
                let indep = (288.0 / 8.0) * (1.0 - (1.0 - 8.0 / 288.0_f64).powi(g as i32));
                1.0 + 0.75 * (indep - 1.0)
            }
        }
    }

    /// Expected number of generated tokens per step (including target bonus) — Eq. 5.
    ///
    /// E[k] = (1 - α^{γ+1}) / (1 - α)
    pub fn expected_accepted(&self) -> f64 {
        let alpha = self.acceptance_rate.clamp(0.0, 1.0);
        let gamma = self.draft_len as f64;
        if (1.0 - alpha).abs() < 1e-9 {
            gamma + 1.0
        } else {
            (1.0 - alpha.powf(gamma + 1.0)) / (1.0 - alpha)
        }
    }

    /// Theoretical speedup of the speculative pipeline:
    ///
    /// S = E[k] / (v_γ + γ · c/C)
    pub fn speedup(&self) -> f64 {
        self.speedup_with_v_gamma(self.v_gamma)
    }

    /// Speedup evaluated at an arbitrary verification overhead factor v_γ.
    pub fn speedup_with_v_gamma(&self, v_gamma: f64) -> f64 {
        let ek = self.expected_accepted();
        let gamma = self.draft_len as f64;
        let overhead = (v_gamma.max(0.001) + gamma * self.draft_cost_ratio.max(0.0)).max(1e-9);
        ek / overhead
    }

    /// Effective decode throughput multiplier under this spec config.
    pub fn throughput_multiplier(&self) -> f64 {
        self.speedup()
    }

    /// Rejection sampling — accepts up to k ≤ γ tokens.
    ///
    /// Paper Algorithm 1, line 4:
    /// "Accept tokens up to position k ≤ γ using rejection sampling."
    pub fn rejection_sample(&self, drafts: &[DraftToken], verified: &[VerifiedToken]) -> usize {
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
            let v_g = if (self.v_gamma - 1.0).abs() < 1e-6 {
                1.0
            } else {
                Self::moe_v_gamma(g)
            };
            let s = ek / (v_g + g as f64 * cost);
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
        // Paper §10.2: 11.1 × 2.35 ≈ 26.1 tok/s
        assert!((m - 2.35).abs() < 0.02, "multiplier={m}");
    }

    #[test]
    fn test_optimal_draft_len() {
        let eng = SpeculativeEngine::new(5, 0.7, 0.05);
        let g = eng.optimal_draft_len(20);
        assert!((3..=8).contains(&g), "optimal γ={g} should be in [3,8]");
    }

    #[test]
    fn test_alpha_one_is_finite() {
        let eng = SpeculativeEngine::new(5, 1.0, 0.05);
        assert!((eng.expected_accepted() - 6.0).abs() < 1e-9);
        assert!(eng.speedup().is_finite());
    }

    #[test]
    fn test_moe_v_gamma_scaling() {
        assert_eq!(SpeculativeEngine::moe_v_gamma(1), 1.0);
        assert_eq!(SpeculativeEngine::moe_v_gamma(2), 1.49);
        assert_eq!(SpeculativeEngine::moe_v_gamma(3), 1.98);
        assert_eq!(SpeculativeEngine::moe_v_gamma(5), 2.87);
    }

    #[test]
    fn test_speedup_with_moe_v_gamma() {
        // With gamma=2, alpha=0.7: E[k] = (1 - 0.7^3) / 0.3 = 0.657 / 0.3 = 2.19
        // v_2 = 1.49, c/C = 0.05
        // S = 2.19 / (1.49 + 2 * 0.05) = 2.19 / 1.59 ≈ 1.377
        let eng = SpeculativeEngine::with_v_gamma(2, 0.7, 0.05, 1.49);
        let s = eng.speedup();
        assert!((s - 1.377).abs() < 0.02, "S={s}");
        assert!(s > 1.0, "bounded gamma=2 MoE spec decode yields positive speedup");

        // With gamma=5, alpha=0.7, v_5 = 2.87:
        // E[k] = 2.941
        // S = 2.941 / (2.87 + 5 * 0.05) = 2.941 / 3.12 ≈ 0.942 (< 1.0, explaining why large gamma fails in MoE!)
        let eng_large = SpeculativeEngine::with_v_gamma(5, 0.7, 0.05, 2.87);
        let s_large = eng_large.speedup();
        assert!(s_large < 1.0, "large gamma MoE verification penalty leads to slowdown unless alpha is very high");
    }
}
