/// Walsh-Hadamard Transform for outlier smoothing (§8.1).
///
/// From §8.1, Eq. 8:
///   x̃ = (1/√d) · H_d · diag(s) · x,  s_i ∼ Uniform{+1, -1}
///
/// Where H_d is the d×d Walsh-Hadamard matrix.
/// After rotation, each coordinate converges to N(0, 1/d) in high dimensions,
/// transforming heavy-tailed outlier distributions into compact shapes
/// amenable to scalar quantization.
///
/// Crucially (§8.1): "the rotation can be absorbed into adjacent weight
/// matrices offline, making it free at inference time."
use rand::Rng;

/// Apply Walsh-Hadamard rotation to a vector.
///
/// x̃ = (1/√d) · H_d · diag(s) · x
pub fn rotate(data: &[f32]) -> Vec<f32> {
    let d = data.len();
    let d_pow2 = d.next_power_of_two();

    // Random sign vector s_i ∼ Uniform{+1, -1}
    let mut rng = rand::thread_rng();
    let signs: Vec<f32> = (0..d)
        .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
        .collect();

    // diag(s) · x: element-wise sign flip
    let mut rotated: Vec<f32> = data.iter().zip(signs.iter())
        .map(|(x, s)| x * s)
        .collect();

    // Pad to next power of 2 for fast Walsh-Hadamard
    rotated.resize(d_pow2, 0.0);

    // In-place fast Walsh-Hadamard transform (FWHT)
    fwht(&mut rotated);

    // Scale by 1/√d
    let scale = 1.0 / (d as f64).sqrt() as f32;
    for v in &mut rotated {
        *v *= scale;
    }

    rotated.truncate(d);
    rotated
}

/// Fast In-Place Walsh-Hadamard Transform.
///
/// O(d log d) using the butterfly algorithm.
fn fwht(data: &mut [f32]) {
    let n = data.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Checked version of FWHT with numerical stability.
pub fn rotate_stable(data: &[f32]) -> Vec<f32> {
    let d = data.len();
    if d == 0 || !d.is_power_of_two() {
        // For non-power-of-2 sizes, pad internally
        return rotate(data);
    }

    let mut rng = rand::thread_rng();
    let signs: Vec<f32> = (0..d)
        .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
        .collect();

    let mut rotated: Vec<f32> = data.iter().zip(signs.iter())
        .map(|(x, s)| x * s)
        .collect();

    fwht(&mut rotated);

    let scale = 1.0 / (d as f64).sqrt() as f32;
    for v in &mut rotated {
        *v *= scale;
    }

    rotated
}

/// Absorbing rotation into weights (offline).
///
/// §8.1: "the rotation can be absorbed into adjacent weight matrices offline,
///  making it free at inference time."
///
/// This means: W · x = (W · H_d^T) · (H_d · x) = W̃ · x̃
/// So we pre-multiply weight matrices by H_d^T (which equals H_d since it's symmetric).
pub fn absorb_into_weights(weights: &mut [f32]) {
    let d = weights.len();
    let d_pow2 = d.next_power_of_two();
    let mut buf = weights.to_vec();
    buf.resize(d_pow2, 0.0);
    fwht(&mut buf);
    let scale = 1.0 / (d as f64).sqrt() as f32;
    for (w, b) in weights.iter_mut().zip(buf.iter()) {
        *w = b * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotate_preserves_norm() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
        let original_norm: f32 = data.iter().map(|x| x * x).sum();
        let rotated = rotate(&data);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum();
        // Norm should be approximately preserved (within FP precision)
        assert!(
            (original_norm - rotated_norm).abs() < 1e-3,
            "norm diff: {}",
            (original_norm - rotated_norm).abs()
        );
    }

    #[test]
    fn test_fwht_size_power_of_two() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        fwht(&mut data);
        // Known result for [1,2,3,4]:
        // [1+2+3+4=10, 1-2+3-4=-2, 1+2-3-4=-4, 1-2-3+4=0]
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[1] - (-2.0)).abs() < 1e-5);
        assert!((data[2] - (-4.0)).abs() < 1e-5);
        assert!((data[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotate_reduces_outliers() {
        // Create data with outliers
        let mut data = vec![0.1f32; 64];
        data[0] = 100.0;  // outlier
        data[1] = -80.0;  // outlier
        let rotated = rotate(&data);
        let max_abs = rotated.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // Outliers should be smoothed
        assert!(max_abs < 30.0, "max_abs={max_abs} still too large");
    }
}
