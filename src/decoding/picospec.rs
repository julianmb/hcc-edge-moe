/// Draft-batch encoding and bounded in-flight bookkeeping.
use crate::decoding::speculative::DraftToken;
use std::collections::VecDeque;

/// Compress draft probabilities for USB4 transmission.
///
/// Instead of transmitting full vocab distributions (32K × fp32 = 128 KB),
/// send only token IDs + top-k probabilities (<1 KB per step).
pub struct PicoSpecRejection;

impl PicoSpecRejection {
    pub fn compress_draft(drafts: &[DraftToken], _top_k: usize) -> Vec<u8> {
        let mut compressed = Vec::with_capacity(drafts.len() * 8);
        for d in drafts {
            compressed.extend_from_slice(&d.token_id.to_le_bytes());
            compressed.extend_from_slice(&(d.probability as f32).to_le_bytes());
        }
        compressed
    }
}

/// Bounded queue for submitted draft batches awaiting verification.
pub struct AsyncDraftStage {
    pending: VecDeque<DraftBatch>,
    max_inflight: usize,
}

struct DraftBatch {
    tokens: Vec<DraftToken>,
    seq: u64,
}

impl AsyncDraftStage {
    pub fn new(max_inflight: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            max_inflight,
        }
    }

    /// Submit a new draft batch for verification.
    pub fn submit(&mut self, tokens: Vec<DraftToken>, seq: u64) -> Option<Vec<u8>> {
        if self.pending.len() >= self.max_inflight {
            return None;
        }
        let compressed = PicoSpecRejection::compress_draft(&tokens, 8);
        self.pending.push_back(DraftBatch { tokens, seq });
        Some(compressed)
    }

    /// Process verification result — release oldest pending batch.
    pub fn verify(&mut self, _accepted_count: usize) -> Option<usize> {
        self.pending.pop_front().map(|batch| {
            tracing::trace!("verified draft batch seq={}", batch.seq);
            batch.tokens.len()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoding::speculative::DraftToken;

    #[test]
    fn test_compress_draft_small() {
        let drafts = vec![DraftToken {
            token_id: 42,
            probability: 0.8,
            kv_state: vec![],
        }];
        let compressed = PicoSpecRejection::compress_draft(&drafts, 8);
        assert_eq!(compressed.len(), 8);
    }

    #[test]
    fn test_async_pipeline() {
        let mut stage = AsyncDraftStage::new(3);
        let drafts = vec![DraftToken {
            token_id: 1,
            probability: 0.5,
            kv_state: vec![],
        }];
        assert!(stage.submit(drafts, 0).is_some());
        assert!(stage
            .submit(
                vec![DraftToken {
                    token_id: 2,
                    probability: 0.6,
                    kv_state: vec![],
                }],
                1
            )
            .is_some());
    }

    #[test]
    fn test_async_pipeline_allows_exact_inflight_limit() {
        let mut stage = AsyncDraftStage::new(2);
        let draft = |token_id| {
            vec![DraftToken {
                token_id,
                probability: 0.5,
                kv_state: vec![],
            }]
        };

        assert!(stage.submit(draft(1), 0).is_some());
        assert!(stage.submit(draft(2), 1).is_some());
        assert_eq!(stage.pending.len(), 2);
        assert!(stage.submit(draft(3), 2).is_none());
        assert_eq!(stage.pending.len(), 2);
        assert_eq!(stage.verify(1), Some(1));
    }
}
