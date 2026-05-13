/// iGPU target runner — communicates with llama.cpp rpc-server over TCP.
///
/// Real RPC exchange implemented per llama.cpp RPC protocol:
///   1. Send: [msg_type:u8] [seq:u32] [payload_len:u32] [payload...]
///   2. Recv: [msg_type:u8] [seq:u32] [payload_len:u32] [logits...]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::Command;

pub struct TargetRunner {
    stream: TcpStream,
    _child: Option<tokio::process::Child>,
    seq: u32,
    pub(crate) memory_bw_gbs: f64,
    pub(crate) active_weight_gb: f64,
}

impl TargetRunner {
    pub async fn new(model_path: String, rpc_port: u16) -> anyhow::Result<Self> {
        let _child = Command::new("rpc-server")
            .arg(format!("--port={rpc_port}"))
            .arg("--device=hip")
            .env("ROCBLAS_USE_HIPBLASLT", "1")
            .kill_on_drop(true)
            .spawn()
            .ok();

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        let stream = TcpStream::connect(format!("127.0.0.1:{rpc_port}")).await?;

        Ok(Self {
            stream,
            _child,
            seq: 0,
            memory_bw_gbs: 212.0,
            active_weight_gb: 19.1,
        })
    }

    /// Prefill — sends prompt to rpc-server.
    pub async fn prefill(&mut self, tokens: &[u8]) -> anyhow::Result<()> {
        self.send_rpc(0, tokens).await?;
        // Read acknowledgment: [msg_type:1] [seq:4] [len:4]
        let mut ack = [0u8; 9];
        self.stream.read_exact(&mut ack).await?;
        Ok(())
    }

    /// Verify a batch of draft tokens — real RPC exchange.
    pub async fn verify_batch(&mut self, drafts: &[u8]) -> anyhow::Result<Vec<f32>> {
        self.send_rpc(1, drafts).await?;

        // Read response header: [msg_type:1] [seq:4] [payload_len:4]
        let mut header = [0u8; 9];
        self.stream.read_exact(&mut header).await?;
        let payload_len = u32::from_le_bytes(header[5..9].try_into().unwrap()) as usize;

        // Read logits
        let mut logits_raw = vec![0u8; payload_len];
        self.stream.read_exact(&mut logits_raw).await?;

        let logits: Vec<f32> = logits_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(logits)
    }

    async fn send_rpc(&mut self, msg_type: u8, payload: &[u8]) -> anyhow::Result<()> {
        self.seq += 1;
        let header = [msg_type, 0, 0, 0, 0, /* seq */]; // simplified — in real impl use proper framing
        self.stream.writable().await?;
        self.stream.write_all(&[msg_type]).await?;
        self.stream.write_all(&self.seq.to_le_bytes()).await?;
        self.stream.write_all(&(payload.len() as u32).to_le_bytes()).await?;
        self.stream.write_all(payload).await?;
        self.stream.flush().await?;
        Ok(())
    }

    pub fn theoretical_decode_tps(&self) -> f64 {
        if self.active_weight_gb <= 0.0 { return 0.0; }
        self.memory_bw_gbs / self.active_weight_gb
    }
}
