/// TurboQuant: Near-Optimal Online Vector Quantization (§8.2).
///
/// From §8.2:
/// "TurboQuant extends QuaRot from round-to-nearest to optimal non-uniform
///  scalar quantization via precomputed Lloyd-Max codebooks."
///
/// MSE distortion bound (Eq. 9):
///   D_MSE ≤ √(3π/2) · 1/(4^b)
///
/// Which is within √(3π/2) ≈ 2.72 of the information-theoretic lower bound
/// at all bit-widths b.
pub struct TurboQuant {
    bits: usize,
    block_size: usize,
    codebook: Vec<f32>,
}

impl TurboQuant {
    pub fn new(bits: usize, block_size: usize) -> Self {
        let codebook = Self::lloyd_max_codebook(bits);
        Self {
            bits,
            block_size,
            codebook,
        }
    }

    /// Lloyd-Max optimal non-uniform scalar quantizer codebook.
    ///
    /// Computes centroids for a standard Gaussian, which after Walsh-Hadamard
    /// rotation approximates N(0, 1/d).
    fn lloyd_max_codebook(bits: usize) -> Vec<f32> {
        let levels = 1 << bits;
        let step = 2.0 / levels as f64;
        // Lloyd-Max iterations for Gaussian N(0,1)
        let mut centroids: Vec<f64> = (0..levels)
            .map(|i| (i as f64 + 0.5) * step - 1.0)
            .collect();

        for _ in 0..50 {
            let mut new_centroids = vec![0.0f64; levels];
            let mut counts = vec![0u64; levels];
            let samples = 100_000;

            for _ in 0..samples {
                let x: f64 = rand_distr::Distribution::sample(
                    &rand_distr::StandardNormal,
                    &mut rand::thread_rng(),
                );
                let idx = (0..levels)
                    .min_by(|&a, &b| {
                        (x - centroids[a]).abs().total_cmp(&(x - centroids[b]).abs())
                    })
                    .unwrap();
                new_centroids[idx] += x;
                counts[idx] += 1;
            }

            for i in 0..levels {
                if counts[i] > 0 {
                    centroids[i] = new_centroids[i] / counts[i] as f64;
                }
            }
        }

        centroids.iter().map(|&c| c as f32).collect()
    }

    /// Quantize a vector to b-bit using the codebook.
    pub fn quantize(&self, data: &[f32]) -> Vec<u8> {
        let mut packed = Vec::with_capacity(data.len() * self.bits / 8);

        if self.bits == 3 {
            // Pack 3-bit values: 8 values per 3 bytes
            for chunk in data.chunks(8) {
                let mut buf = 0u64;
                for (i, &v) in chunk.iter().enumerate() {
                    let idx = self.quantize_one(v);
                    buf |= (idx as u64) << (i * 3);
                }
                let bytes = (chunk.len() * 3 + 7) / 8;
                for b in 0..bytes {
                    packed.push((buf >> (b * 8)) as u8);
                }
            }
        } else {
            // Generic packing for arbitrary bit-widths
            let mut bit_offset = 0;
            for &v in data {
                let idx = self.quantize_one(v) as u16;
                let bits_remaining = self.bits;
                let mut bo = bit_offset;

                for b in 0..bits_remaining {
                    let byte_idx = bo / 8;
                    let bit_idx = bo % 8;
                    while packed.len() <= byte_idx {
                        packed.push(0);
                    }
                    if (idx >> b) & 1 == 1 {
                        packed[byte_idx] |= 1 << bit_idx;
                    }
                    bo += 1;
                }
                bit_offset += bits_remaining;
            }
        }

        packed
    }

    /// Dequantize back to FP32.
    pub fn dequantize(&self, packed: &[u8], count: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(count);

        if self.bits == 3 {
            // Unpack 3-bit values
            let mut byte_idx = 0;
            let mut bit_offset = 0;
            for _ in 0..count {
                let mut idx = 0usize;
                for b in 0..self.bits {
                    let byte = packed.get(byte_idx).copied().unwrap_or(0);
                    if (byte >> bit_offset) & 1 == 1 {
                        idx |= 1 << b;
                    }
                    bit_offset += 1;
                    if bit_offset >= 8 {
                        bit_offset = 0;
                        byte_idx += 1;
                    }
                }
                result.push(self.codebook.get(idx).copied().unwrap_or(0.0));
            }
        } else {
            let mut bit_offset = 0;
            for _ in 0..count {
                let mut idx = 0u16;
                for b in 0..self.bits {
                    let byte_idx = (bit_offset + b) / 8;
                    let bit_idx = (bit_offset + b) % 8;
                    if packed.get(byte_idx).copied().unwrap_or(0) >> bit_idx & 1 == 1 {
                        idx |= 1 << b;
                    }
                }
                bit_offset += self.bits;
                result.push(self.codebook.get(idx as usize).copied().unwrap_or(0.0));
            }
        }

        result
    }

    fn quantize_one(&self, value: f32) -> usize {
        self.codebook
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| (value - a).abs().total_cmp(&(value - b).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// MSE distortion bound — Eq. 9.
    ///
    /// D_MSE ≤ √(3π/2) · 1/(4^b)
    pub fn mse_bound(&self) -> f64 {
        (3.0 * std::f64::consts::PI / 2.0).sqrt() / (4usize.pow(self.bits as u32) as f64)
    }

    pub fn codebook(&self) -> &[f32] { &self.codebook }
    pub fn bits(&self) -> usize { self.bits }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip_3bit() {
        let tq = TurboQuant::new(3, 128);
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let packed = tq.quantize(&data);
        let unpacked = tq.dequantize(&packed, data.len());
        assert_eq!(unpacked.len(), data.len());
        // Should be roughly similar
        for (a, b) in data.iter().zip(unpacked.iter()) {
            let err = (a - b).abs();
            assert!(err < 2.0, "quantization error too large: {err}");
        }
    }

    #[test]
    fn test_mse_bound() {
        let tq = TurboQuant::new(3, 128);
        let bound = tq.mse_bound();
        // √(3π/2) / 4^3 = √(4.7124) / 64 ≈ 2.170 / 64 ≈ 0.0339
        assert!((bound - 0.034).abs() < 0.001, "mse_bound={bound}");
    }

    #[test]
    fn test_codebook_size() {
        let tq = TurboQuant::new(3, 128);
        assert_eq!(tq.codebook().len(), 8); // 2^3 = 8 levels
    }
}
