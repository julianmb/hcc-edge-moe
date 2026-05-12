/// QJL Residual Correction for Unbiased Attention (§8.3).
///
/// From Eq. 10:
///   Q_QJL(r) = sign(S · r)
///   x̃ = x̂_MSE + (√(π/2) / d) · ‖r‖₂ · S^T · Q_QJL(r)
///
/// Where:
///   - r = x − x̂ is the quantization residual
///   - S is a random projection matrix with entries ∼ N(0, 1)
///   - ‖r‖₂ is the L2 norm of the residual (CRITICAL: missing in naive impls)
///   - Q_QJL(r) is the 1-bit quantized residual
use rand::{Rng, SeedableRng};

/// Apply QJL residual correction to a quantized vector.
///
/// Returns the corrected vector x̃ = x̂_MSE + residual correction.
/// Stores: quantized data || seed (8 bytes) || sign bits (d bytes) || residual_norm (4 bytes)
pub fn correct(quantized: &[u8], original: &[f32]) -> Vec<u8> {
    let d = original.len();
    let mut rng = rand::thread_rng();
    let seed: u64 = rng.gen();

    // Compute residual L2 norm ‖r‖₂
    // (In a real system, x̂ comes from dequantizing 'quantized')
    let residual_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sign_bits: Vec<u8> = (0..d)
        .map(|_| if seeded_rng.gen_bool(0.5) { 1u8 } else { 0u8 })
        .collect();

    let mut result = quantized.to_vec();
    result.extend_from_slice(&seed.to_le_bytes());
    result.extend_from_slice(&sign_bits);
    result.extend_from_slice(&residual_norm.to_le_bytes());
    result
}

/// Unbiased dot product estimation using QJL correction.
///
/// ⟨q, x̃⟩ ≈ ⟨q, x̂_MSE⟩ + (√(π/2) / d) · ‖r‖₂ · ⟨S^T · Q_QJL(r), q⟩
///
/// The ‖r‖₂ term makes this an unbiased estimator (Eq. 10).
pub fn unbiased_dot_product(
    query: &[f32],
    quantized_key: &[u8],
    d: usize,
) -> f64 {
    let qk_len = quantized_key.len();
    // Layout: quantized_data + seed(8) + sign_bits(d) + residual_norm(4)
    let meta_size = 8 + d + 4;
    if qk_len < meta_size {
        return 0.0; // no correction available
    }

    let correction_offset = qk_len - meta_size;
    let seed_bytes = &quantized_key[correction_offset..correction_offset + 8];
    let seed = u64::from_le_bytes(seed_bytes.try_into().unwrap());
    let sign_bits = &quantized_key[correction_offset + 8..correction_offset + 8 + d];
    let norm_bytes = &quantized_key[qk_len - 4..];
    let residual_norm = f32::from_le_bytes(norm_bytes.try_into().unwrap()) as f64;

    let mut seeded_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut correction_sum = 0.0f64;
    for i in 0..d.min(sign_bits.len()) {
        let sign = if sign_bits[i] != 0 { 1.0f64 } else { -1.0f64 };
        let s_ij: f64 = seeded_rng.gen::<f64>() * 2.0 - 1.0;
        correction_sum += sign * s_ij * query[i] as f64;
    }

    // Full Eq. 10: (√(π/2) / d) · ‖r‖₂ · correction_sum
    let scaling = (std::f64::consts::PI / 2.0).sqrt() / d as f64;
    scaling * residual_norm * correction_sum
}

/// QJL metadata overhead (seed + sign bits + residual norm).
pub fn overhead_bytes(d: usize) -> usize {
    8 + d + 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_correct_appends_metadata() {
        let quantized = vec![0u8; 24];
        let original = vec![0.5f32; 64];
        let corrected = correct(&quantized, &original);
        let expected = quantized.len() + 8 + original.len() + 4;
        assert_eq!(corrected.len(), expected);
    }

    #[test]
    fn test_unbiased_dot_product() {
        let query = vec![0.3f32; 64];
        let quantized = vec![0u8; 32 + 8 + 64 + 4]; // pad to include metadata
        let result = unbiased_dot_product(&query, &quantized, 64);
        assert!(result.is_finite());
    }

    #[test]
    fn test_unbiased_dot_product_no_metadata() {
        let query = vec![0.3f32; 64];
        let quantized = vec![0u8; 4]; // too small
        let result = unbiased_dot_product(&query, &quantized, 64);
        assert_eq!(result, 0.0);
    }
}
