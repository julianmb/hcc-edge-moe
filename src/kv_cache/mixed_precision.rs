/// Mixed-precision TurboQuant KV cache — K 8-bit FP8, V 3-bit Lloyd-Max.
///
/// CRITICAL: K/V norms differ by 100-1200x in real models (scos-lab findings).
/// Uniform bit allocation is catastrophically wasteful.
///   - Keys:   8-bit FP8 (block-scaled) — preserves attention score fidelity
///   - Values: 3-bit Lloyd-Max (PolarQuant rotation + optimal centroids)
pub struct MixedPrecisionKVCache {
    key_cache: Vec<Fp8Block>,
    value_cache: Vec<Lm3Block>,
    mladim: usize,
    block_size: usize,
}

struct Fp8Block {
    data: Vec<u8>,
    /// Per-block FP8 scale factor.
    scale: f32,
}

struct Lm3Block {
    /// 3-bit Lloyd-Max packed values.
    packed: Vec<u8>,
    /// Norm for dequantization.
    norm: f32,
}

impl MixedPrecisionKVCache {
    pub fn new(mladim: usize) -> Self {
        Self {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            mladim,
            block_size: 128,
        }
    }

    /// Insert K and V vectors with appropriate mixed-precision quantization.
    pub fn insert(&mut self, key: &[f32], value: &[f32]) {
        let k_fp8 = self.quantize_fp8(key);
        assert!(
            (k_fp8.scale - 1.0).abs() < 1e-6 || k_fp8.scale > 0.0,
            "FP8 scale must be positive"
        );
        let v_lm3 = self.quantize_lm3(value);
        self.key_cache.push(k_fp8);
        self.value_cache.push(v_lm3);
    }

    /// Read a dequantized (Key, Value) pair by position.
    pub fn read(&self, pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let k = self.dequantize_key(pos)?;
        let v = self.dequantize_value(pos)?;
        Some((k, v))
    }

    /// Dequantize Key at position pos back to FP32.
    pub fn dequantize_key(&self, pos: usize) -> Option<Vec<f32>> {
        let block = self.key_cache.get(pos)?;
        let scale = block.scale;
        Some(
            block
                .data
                .iter()
                .map(|&b| (b as i8 as f32) * scale)
                .collect(),
        )
    }

    /// Dequantize Value at position pos back to FP32.
    pub fn dequantize_value(&self, pos: usize) -> Option<Vec<f32>> {
        let block = self.value_cache.get(pos)?;
        if block.norm == 0.0 {
            return Some(vec![0.0; self.mladim]);
        }

        let codebook = Self::lm3_codebook();
        let mut result = Vec::with_capacity(self.mladim);
        let mut bit_offset = 0usize;

        for _ in 0..self.mladim {
            let mut idx = 0usize;
            for b in 0..3 {
                let byte_idx = bit_offset / 8;
                let bit = bit_offset % 8;
                if (block.packed.get(byte_idx).copied().unwrap_or(0) >> bit) & 1 == 1 {
                    idx |= 1 << b;
                }
                bit_offset += 1;
            }
            let normalized = codebook.get(idx).copied().unwrap_or(0.0);
            result.push(normalized * block.norm);
        }

        Some(result)
    }

    /// Number of stored KV entries.
    pub fn len(&self) -> usize {
        self.key_cache.len().min(self.value_cache.len())
    }

    /// FP8 quantization: 8-bit with per-block max-abs scaling.
    fn quantize_fp8(&self, data: &[f32]) -> Fp8Block {
        let mut quantized = Vec::with_capacity(data.len());
        let mut max_abs = 0.0f32;
        for &v in data {
            let abs = v.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0
        };
        for &v in data {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            quantized.push(q as u8);
        }
        Fp8Block { data: quantized, scale }
    }

    /// 3-bit Lloyd-Max quantization for Value vectors.
    fn quantize_lm3(&self, data: &[f32]) -> Lm3Block {
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return Lm3Block {
                packed: vec![0u8; (self.mladim * 3 + 7) / 8],
                norm: 0.0,
            };
        }
        let normalized: Vec<f32> = data.iter().map(|x| x / norm).collect();
        let rotated = crate::kv_cache::walsh_hadamard::rotate(&normalized);
        let codebook = Self::lm3_codebook();

        let mut packed = Vec::with_capacity((self.mladim * 3 + 7) / 8);
        let mut bit_offset = 0usize;
        for &v in &rotated {
            let idx = codebook
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| (v - a).abs().total_cmp(&(v - b).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0);
            for b in 0..3 {
                let byte_idx = bit_offset / 8;
                let bit = bit_offset % 8;
                while packed.len() <= byte_idx {
                    packed.push(0u8);
                }
                if (idx >> b) & 1 == 1 {
                    packed[byte_idx] |= 1 << bit;
                }
                bit_offset += 1;
            }
        }
        Lm3Block { packed, norm }
    }

    /// Precomputed Lloyd-Max centroids for 3-bit Gaussian-optimized quantization.
    fn lm3_codebook() -> [f32; 8] {
        [-1.748, -1.050, -0.501, -0.067, 0.067, 0.501, 1.050, 1.748]
    }

    /// Memory savings vs FP16 baseline.
    /// K: 8-bit = 50% of FP16. V: 3-bit = 18.75%. Combined: (8+3)/(16+16) = 34.4% → 65.6% savings.
    pub fn savings_ratio(&self) -> f64 {
        1.0 - (8.0 + 3.0) / 32.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_precision_insert_read() {
        let dim = 576;
        let mut cache = MixedPrecisionKVCache::new(dim);
        let key = vec![0.5f32; dim];
        let value = vec![0.3f32; dim];
        cache.insert(&key, &value);
        assert_eq!(cache.len(), 1);
        let (k, v) = cache.read(0).unwrap();
        assert_eq!(k.len(), dim);
        assert_eq!(v.len(), dim);
    }

    #[test]
    fn test_savings_ratio() {
        let cache = MixedPrecisionKVCache::new(576);
        assert!((cache.savings_ratio() - 0.65625).abs() < 1e-5);
    }

    #[test]
    fn test_lm3_codebook_size() {
        let cb = MixedPrecisionKVCache::lm3_codebook();
        assert_eq!(cb.len(), 8);
    }

    #[test]
    fn test_fp8_scale_computed() {
        let cache = MixedPrecisionKVCache::new(128);
        let data = vec![100.0f32, -50.0, 25.0, -12.5];
        let block = cache.quantize_fp8(&data);
        assert!(block.scale > 0.0);
    }

    #[test]
    fn test_dequantize_out_of_bounds() {
        let cache = MixedPrecisionKVCache::new(128);
        assert!(cache.read(999).is_none());
    }
}
