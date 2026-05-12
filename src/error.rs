/// HCC-specific error types.
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HccError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("hardware error: {0}")]
    Hardware(String),

    #[error("MIGraphX error: {0}")]
    Migraphx(String),

    #[error("ROCm HIP error: code {0}")]
    Hip(i32),

    #[error("llama.cpp RPC error: {0}")]
    Rpc(String),

    #[error("USB4 transport error: {0}")]
    Transport(String),

    #[error("DMA-BUF allocation failed: {0}")]
    DmaBuf(String),

    #[error("network timeout after {0}ms")]
    Timeout(u64),

    #[error("model load failed: {0}")]
    ModelLoad(String),

    #[error("KV cache error: {0}")]
    KvCache(String),

    #[error("NUMA/node mismatch: expected {expected} nodes, got {actual}")]
    NodeMismatch { expected: usize, actual: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    #[error("{0}")]
    Other(String),
}

impl From<&str> for HccError {
    fn from(s: &str) -> Self {
        HccError::Other(s.to_string())
    }
}

impl From<String> for HccError {
    fn from(s: String) -> Self {
        HccError::Other(s)
    }
}

pub type HccResult<T> = Result<T, HccError>;
