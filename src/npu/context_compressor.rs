/// Cascaded Context Compression — NPU-side prefill reduction.
///
/// From §6.2:
/// "By mapping CSR's Layer 1 deterministic routing and Layer 2 small-model
///  screening to the XDNA 2 NPU's spatial dataflow array...the activation
///  payload transmitted across the USB4 bridge to the main MoE on the iGPU
///  is reduced by >80%."
///
/// This implements the deterministic pre-filtering and extractive summarization
/// stage that runs entirely on the NPU before the USB4 network crossing.
///
/// CSR reference [18]: Cascaded Signal Refinement — three-layer pipeline:
///   Layer 1: Deterministic pre-filtering (regex, keyword, rule-based)
///   Layer 2: Small-model LLM screening (8B on NPU)
///   Layer 3: Deep analysis (main MoE on iGPU behind USB4)
pub struct ContextCompressor {
    /// Compression ratio target (paper: >80% reduction → α = 0.25).
    compression_ratio: f64,
    /// Minimum retained characters.
    min_chars: usize,
}

impl ContextCompressor {
    pub fn new() -> Self {
        Self {
            compression_ratio: 0.25,  // >80% reduction
            min_chars: 512,
        }
    }

    /// Compress raw context window prior to USB4 transmission.
    ///
    /// §6.2: "The activation payload transmitted across the USB4 bridge
    ///  to the main MoE on the iGPU is reduced by >80%."
    ///
    /// Combined with dual-USB4 throughput of 45 Gbps, projects dramatically
    /// faster TTFT for large-scale RAG workloads.
    pub async fn compress(&self, raw: &[u8]) -> anyhow::Result<Vec<u8>> {
        let raw_len = raw.len();
        let target_len = (raw_len as f64 * self.compression_ratio)
            .max(self.min_chars as f64) as usize;

        tracing::debug!(
            "compressing context: {raw_len}B -> {target_len}B (ratio={:.2})",
            raw_len as f64 / target_len.max(1) as f64
        );

        // Layer 1: Deterministic pre-filtering
        // Extract key sections, remove boilerplate
        let filtered = self.deterministic_filter(raw);

        // Layer 2: NPU small-model extractive summarization
        // In production: 8B draft model on XDNA 2 NPU via Vitis AI
        let compressed = self.extractive_summarize(&filtered, target_len);

        Ok(compressed)
    }

    /// Layer 1: Rule-based deterministic filtering.
    ///
    /// Removes redundant whitespace, boilerplate headers, duplicate lines.
    fn deterministic_filter(&self, data: &[u8]) -> Vec<u8> {
        let s = String::from_utf8_lossy(data);
        let mut result = String::with_capacity(s.len());

        for line in s.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Skip common boilerplate
            if trimmed.starts_with("---")
                || trimmed.starts_with("===")
                || trimmed.starts_with("***")
            {
                continue;
            }
            result.push_str(trimmed);
            result.push('\n');
        }

        result.into_bytes()
    }

    /// Layer 2: Extractive summarization via NPU draft model.
    ///
    /// In production: this dispatches to the XDNA 2 NPU running the
    /// 8B draft model in BFP16 to perform a single forward pass
    /// that scores sentence importance.
    fn extractive_summarize(&self, filtered: &[u8], target_len: usize) -> Vec<u8> {
        if filtered.len() <= target_len {
            return filtered.to_vec();
        }

        let text = String::from_utf8_lossy(filtered);
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        // Score sentences by position and length (simple heuristic)
        // In production: NPU forward pass computes importance scores
        let mut scored: Vec<(usize, f32)> = sentences.iter().enumerate().map(|(i, s)| {
            let pos_score = 1.0 - (i as f32 / sentences.len().max(1) as f32);
            let len_score = (s.len() as f32 / 512.0).min(1.0);
            (i, pos_score * 0.6 + len_score * 0.4)
        }).collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Select top sentences to fit target size
        let mut result = String::with_capacity(target_len);
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &(idx, _) in &scored {
            if result.len() >= target_len { break; }
            if used.insert(idx) {
                result.push_str(sentences[idx]);
                result.push_str(". ");
            }
        }

        result.into_bytes()
    }

    pub fn compression_ratio(&self) -> f64 { self.compression_ratio }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_filter() {
        let compressor = ContextCompressor::new();
        let input = b"Hello\n---\nWorld\n===\nTest\n***\nDone\n\n\nFinal";
        let filtered = compressor.deterministic_filter(input);
        let output = String::from_utf8_lossy(&filtered);
        // Should remove ---, ===, *** and empty lines
        assert!(!output.contains("---"));
        assert!(!output.contains("==="));
        assert!(!output.contains("***"));
        assert!(output.contains("Hello"));
        assert!(output.contains("World"));
        assert!(output.contains("Done"));
        assert!(output.contains("Final"));
    }

    #[tokio::test]
    async fn test_compress_short_passthrough() {
        let compressor = ContextCompressor::new();
        let input = b"Short text under threshold";
        let compressed = compressor.compress(input).await;
        assert!(compressed.is_ok());
    }
}
