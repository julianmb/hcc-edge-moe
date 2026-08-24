use serde::{Deserialize, Serialize};

/// HCC wire protocol messages over USB4.
///
/// Defines the message types exchanged between Node 1 (NPU draft + iGPU layers 0–38)
/// and Node 2 (iGPU layers 39–77).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HccMessage {
    /// Prefill compressed context from Node 1 NPU to Node 2.
    /// §6.2: compressed activation payload after NPU summarization.
    PrefillPayload {
        tokens: Vec<u32>,
        compressed_activations: Vec<u8>,
        context_len: u32,
    },

    /// Draft batch from Node 1 NPU to Node 2 iGPU for verification.
    /// §7, Algorithm 1 line 3: "Pass sequence through M on iGPU cluster
    /// as a single parallel batch."
    DraftBatch {
        tokens: Vec<u32>,
        hidden_states: Vec<f32>,
        draft_len: u8,
    },

    /// Verification result from Node 2 iGPU back to Node 1.
    /// Accepted token indices and logits for rejection sampling.
    VerificationResult {
        accepted_prefix_len: u8,
        logits: Vec<f32>,
        probabilities: Vec<f32>,
    },

    /// KV cache sync message for context update.
    /// §7: "updating the NPU's context" — return payload is "mere bytes"
    /// over the 17 µs link, <0.02% overhead.
    ContextSync {
        accepted_tokens: Vec<u32>,
        kv_state_delta: Vec<u8>,
    },

    /// Session management.
    SessionRequest {
        session_id: u64,
        max_tokens: u32,
    },
    SessionResponse {
        session_id: u64,
        status: SessionStatus,
    },

    /// Heartbeat / health check.
    Ping(u64),
    Pong(u64),
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    Accepted,
    Rejected(String),
    Completed,
}

/// Serialization via bincode (fixed-int encoding). Payload sizes below are
/// estimates for capacity planning; the wire format is bincode-encoded.
impl HccMessage {
    pub fn msg_type(&self) -> &str {
        match self {
            Self::PrefillPayload { .. } => "prefill",
            Self::DraftBatch { .. } => "draft",
            Self::VerificationResult { .. } => "verify",
            Self::ContextSync { .. } => "context_sync",
            Self::SessionRequest { .. } => "session_req",
            Self::SessionResponse { .. } => "session_resp",
            Self::Ping(_) => "ping",
            Self::Pong(_) => "pong",
            Self::Shutdown => "shutdown",
        }
    }

    /// Size in bytes (approximate).
    pub fn byte_size(&self) -> usize {
        match self {
            Self::PrefillPayload {
                compressed_activations,
                ..
            } => 8usize.saturating_add(compressed_activations.len()),
            Self::DraftBatch { hidden_states, .. } => {
                8usize.saturating_add(hidden_states.len().saturating_mul(4))
            }
            Self::VerificationResult {
                logits,
                probabilities,
                ..
            } => 1usize
                .saturating_add(logits.len().saturating_mul(4))
                .saturating_add(probabilities.len().saturating_mul(4)),
            Self::ContextSync { kv_state_delta, .. } => 8usize.saturating_add(kv_state_delta.len()),
            _ => 64,
        }
    }
}

/// Compact binary header for fast dispatch.
#[repr(C, packed)]
pub struct HccPacketHeader {
    pub magic: [u8; 4], // b"HCC\0"
    pub msg_type: u8,
    pub seq: u64,
    pub src_node: u8,
    pub dst_node: u8,
    pub payload_len: u32,
    pub flags: u8,
    pub checksum: u32,
}

impl HccPacketHeader {
    pub const MAGIC: [u8; 4] = [b'H', b'C', b'C', 0];

    #[cfg(test)]
    const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(msg_type: u8, seq: u64, src: u8, dst: u8, payload_len: u32) -> Self {
        Self {
            magic: Self::MAGIC,
            msg_type,
            seq,
            src_node: src,
            dst_node: dst,
            payload_len,
            flags: 0,
            checksum: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_roundtrip() {
        let msg = HccMessage::DraftBatch {
            tokens: vec![42, 99, 101],
            hidden_states: vec![0.5f32; 12288],
            draft_len: 3,
        };
        let encoded = bincode::serialize(&msg).unwrap();
        let decoded: HccMessage = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded.msg_type(), "draft");
    }

    #[test]
    fn test_packet_header_size_matches_struct() {
        assert_eq!(
            HccPacketHeader::SIZE,
            std::mem::size_of::<HccPacketHeader>()
        );
    }
}
