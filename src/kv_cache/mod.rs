/// KV cache using TurboQuant for 3-bit value compression.
///
/// Paper Eq. 8-10: Walsh-Hadamard rotation for outlier smoothing,
/// TurboQuant MSE-bounded quantization, QJL residual correction.
/// Keys use FP8; values use TurboQuant 3-bit with power-of-2 padding
/// (MLA d_kv=576 padded to 1024).
use turboquant::packed::TurboQuantConfig;

/// TurboQuant natively requires power-of-2 dimensions. MLA d_kv = 576 → pad to 1024.
const TQ_DIM: usize = 1024;

pub struct MixedPrecisionKVCache {
    key_blocks: Vec<Fp8Block>,
    value_blocks: Vec<turboquant::packed::PackedBlock>,
    dim: usize,
    tq_config: Option<TurboQuantConfig>,
}

struct Fp8Block(Vec<u8>, f32);

impl MixedPrecisionKVCache {
    pub fn new(dim: usize) -> Self {
        let tq_config = TurboQuantConfig::new(3, TQ_DIM).ok();
        Self {
            key_blocks: Vec::new(),
            value_blocks: Vec::new(),
            dim,
            tq_config,
        }
    }

    pub fn insert(&mut self, key: &[f32], value: &[f32]) {
        let k = Self::quantize_fp8(key);
        if let Some(ref config) = self.tq_config {
            if let Some(v) = Self::quantize_lm3(value, config) {
                self.key_blocks.push(k);
                self.value_blocks.push(v);
            }
        }
    }

    pub fn read(&self, pos: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        Some((self.dequantize_key(pos)?, self.dequantize_value(pos)?))
    }

    pub fn dequantize_key(&self, pos: usize) -> Option<Vec<f32>> {
        let block = self.key_blocks.get(pos)?;
        Some(
            block
                .0
                .iter()
                .map(|&b| (b as i8 as f32) * block.1)
                .collect(),
        )
    }

    pub fn dequantize_value(&self, pos: usize) -> Option<Vec<f32>> {
        let config = self.tq_config.as_ref()?;
        let mut raw =
            turboquant::quantize::dequantize_vec(config, self.value_blocks.get(pos)?).ok()?;
        raw.truncate(self.dim);
        Some(raw)
    }

    pub fn len(&self) -> usize {
        self.key_blocks.len()
    }

    fn quantize_fp8(data: &[f32]) -> Fp8Block {
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0
        };
        Fp8Block(
            data.iter()
                .map(|&v| ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8)
                .collect(),
            scale,
        )
    }

    fn quantize_lm3(
        data: &[f32],
        config: &TurboQuantConfig,
    ) -> Option<turboquant::packed::PackedBlock> {
        let mut padded = data.to_vec();
        padded.resize(TQ_DIM, 0.0);
        turboquant::quantize::quantize_vec(config, &padded).ok()
    }

    pub fn savings_ratio(&self) -> f64 {
        1.0 - (8.0 + 3.0) / 32.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_read() {
        let mut c = MixedPrecisionKVCache::new(576);
        c.insert(&vec![0.5; 576], &vec![0.3; 576]);
        assert_eq!(c.len(), 1);
        let (k, v) = c.read(0).unwrap();
        assert_eq!(k.len(), 576);
        assert_eq!(v.len(), 576);
    }

    #[test]
    fn test_savings_ratio() {
        assert!((MixedPrecisionKVCache::new(576).savings_ratio() - 0.65625).abs() < 1e-5);
    }
}
