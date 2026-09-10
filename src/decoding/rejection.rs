/// Rejection sampling — reused by SpeculativeEngine.
///
/// Implements the standard rejection sampling scheme from
/// Leviathan et al. (2023), as referenced in §7 and Algorithm 1.
///
/// Note: SpeculativeEngine::accept_token inlines this logic directly.
/// This module provides the standalone function for other consumers.
pub fn accept_token(target_prob: f64, draft_prob: f64, uniform: f64) -> bool {
    let ratio = target_prob / draft_prob.max(1e-30);
    ratio >= 1.0 || ratio > uniform
}

pub fn reject(target_prob: f64, draft_prob: f64, uniform: f64) -> bool {
    accept_token(target_prob, draft_prob, uniform)
}

/// Compute exact residual probability distribution max(0, p - q) normalized.
/// When a speculative draft token is rejected at position k, the target model
/// samples a replacement token from this normalized positive residual distribution.
pub fn residual_distribution(target_dist: &[f64], draft_dist: &[f64]) -> Vec<f64> {
    assert_eq!(
        target_dist.len(),
        draft_dist.len(),
        "distributions must match vocab size"
    );
    let mut residual: Vec<f64> = target_dist
        .iter()
        .zip(draft_dist.iter())
        .map(|(&p, &q)| (p - q).max(0.0))
        .collect();
    let sum: f64 = residual.iter().sum();
    if sum > 1e-12 {
        for p in &mut residual {
            *p /= sum;
        }
    } else if !residual.is_empty() {
        // Uniform fallback if distributions were identical
        let uniform = 1.0 / residual.len() as f64;
        residual.fill(uniform);
    }
    residual
}

/// Sample an index from a probability distribution given a uniform random value in [0, 1).
/// If `top_k` is specified, restricts candidate pool to the top-k highest probabilities.
pub fn sample_from_distribution(dist: &[f64], uniform: f64, top_k: Option<usize>) -> usize {
    if dist.is_empty() {
        return 0;
    }
    let u = uniform.clamp(0.0, 1.0 - 1e-12);

    if let Some(k) = top_k {
        if k > 0 && k < dist.len() {
            let mut indexed: Vec<(usize, f64)> = dist.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(k);
            let sum: f64 = indexed.iter().map(|(_, p)| p).sum();
            if sum > 1e-12 {
                let target = u * sum;
                let mut cum = 0.0;
                for (idx, p) in indexed {
                    cum += p;
                    if cum >= target {
                        return idx;
                    }
                }
            }
        }
    }

    let sum: f64 = dist.iter().sum();
    let target = if sum > 1e-12 { u * sum } else { u };
    let mut cum = 0.0;
    for (i, &p) in dist.iter().enumerate() {
        cum += p;
        if cum >= target {
            return i;
        }
    }
    dist.len().saturating_sub(1)
}

/// Sample the recovery token from the residual distribution max(0, p - q).
pub fn sample_residual(
    target_dist: &[f64],
    draft_dist: &[f64],
    uniform: f64,
    top_k: Option<usize>,
) -> usize {
    let residual = residual_distribution(target_dist, draft_dist);
    sample_from_distribution(&residual, uniform, top_k)
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

    #[test]
    fn test_residual_distribution() {
        let target = [0.1, 0.6, 0.3];
        let draft = [0.3, 0.2, 0.5];
        let residual = residual_distribution(&target, &draft);
        // target - draft = [-0.2, 0.4, -0.2]
        // max(0, diff) = [0.0, 0.4, 0.0]
        // normalized = [0.0, 1.0, 0.0]
        assert!((residual[0] - 0.0).abs() < 1e-6);
        assert!((residual[1] - 1.0).abs() < 1e-6);
        assert!((residual[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_with_top_k() {
        let dist = [0.05, 0.1, 0.5, 0.25, 0.1];
        // top_k = 2 should only sample from indices 2 (0.5) and 3 (0.25)
        let sampled_low = sample_from_distribution(&dist, 0.1, Some(2));
        assert_eq!(sampled_low, 2);
        let sampled_high = sample_from_distribution(&dist, 0.9, Some(2));
        assert_eq!(sampled_high, 3);
    }
}
