/// Rejection sampling — reused by SpeculativeEngine.
///
/// Implements the standard rejection sampling scheme from
/// Leviathan et al. (2023), as referenced in §7 and Algorithm 1.
///
/// Note: SpeculativeEngine::accept_token inlines this logic directly.
/// This module provides the standalone function for other consumers.
pub fn reject(target_prob: f64, draft_prob: f64, uniform: f64) -> bool {
    let ratio = target_prob / draft_prob.max(f64::EPSILON);
    ratio >= 1.0 || ratio > uniform
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_always_when_target_higher() {
        assert!(reject(0.9, 0.5, 0.999));
    }

    #[test]
    fn test_batch() {
        fn batch(target: &[f64], draft: &[f64]) -> usize {
            let n = target.len().min(draft.len());
            for k in 0..n {
                if !reject(target[k], draft[k], fastrand::f64()) {
                    return k;
                }
            }
            n
        }
        assert_eq!(batch(&[0.5, 0.6, 0.7], &[0.4, 0.5, 0.6]), 3);
    }
}
