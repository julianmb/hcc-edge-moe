/// Rejection sampling utilities for speculative decoding.
///
/// Implements the standard rejection sampling scheme from Leviathan et al. (2023),
/// as referenced in §7 and Algorithm 1.
pub struct RejectionSampler;

impl RejectionSampler {
    /// Standard rejection sampling criterion.
    ///
    /// Accept if p_target(x) / p_draft(x) > uniform(0, 1].
    /// If p_target > p_draft, always accept (ratio ≥ 1).
    pub fn accept(target_prob: f64, draft_prob: f64, uniform: f64) -> bool {
        let ratio = target_prob / draft_prob.max(f64::EPSILON);
        ratio >= 1.0 || ratio > uniform
    }

    /// Batch rejection — returns the number of accepted tokens k ≤ γ.
    pub fn accept_batch(
        target_probs: &[f64],
        draft_probs: &[f64],
    ) -> usize {
        let n = target_probs.len().min(draft_probs.len());
        for k in 0..n {
            if !Self::accept(target_probs[k], draft_probs[k], fastrand::f64()) {
                return k;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_always_when_target_higher() {
        assert!(RejectionSampler::accept(0.9, 0.5, 0.999));
    }

    #[test]
    fn test_accept_batch_basic() {
        let target = vec![0.5, 0.6, 0.7];
        let draft = vec![0.4, 0.5, 0.6];
        let k = RejectionSampler::accept_batch(&target, &draft);
        assert_eq!(k, 3);
    }
}
