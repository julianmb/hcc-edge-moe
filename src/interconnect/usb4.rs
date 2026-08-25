use crate::config::InterconnectConfig;
use crate::interconnect::dmabuf::DmaBufDescriptor;
use crate::interconnect::protocol::HccMessage;
use anyhow::Context;
use bincode::Options;
use serde::{Deserialize, Serialize};
use std::io::ErrorKind;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const PROTOCOL_MAGIC: [u8; 4] = *b"HCC\0";
const PROTOCOL_VERSION: u8 = 1;
const HANDSHAKE_SIZE: usize = 8;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const RX_QUEUE_CAPACITY: usize = 32;

/// Framed TCP transport over the Linux thunderbolt-net interface.
///
/// Node 1 listens on `listen_addr`; Node 0 connects to `peer_addr`. TCP is
/// full-duplex, while the internal receive queue preserves packet boundaries
/// for the orchestration loop.
pub struct Usb4Transport {
    cfg: InterconnectConfig,
    node_id: usize,
    node_count: usize,
    writer: OwnedWriteHalf,
    rx_chan: mpsc::Receiver<anyhow::Result<Vec<u8>>>,
    reader_task: JoinHandle<()>,
    bytes_sent: u64,
    accumulated_rtt_us: f64,
    packets_sent: u64,
    packets_received: u64,
    next_recv_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usb4Packet {
    pub src_node: usize,
    pub dst_node: usize,
    pub seq: u64,
    pub flags: u8,
    pub payload: Vec<u8>,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct PacketFlags: u8 {
        const NONE = 0x00;
        const DMA_BUF = 0x01;
        const DRAFT = 0x02;
        const VERIFY = 0x04;
        const PREFILL = 0x08;
        const ACK = 0x10;
        const SHUTDOWN = 0x20;
    }
}

impl Usb4Transport {
    pub async fn new(
        cfg: &InterconnectConfig,
        node_count: usize,
        node_id: usize,
    ) -> anyhow::Result<Self> {
        if node_count != 2 || node_id >= node_count {
            anyhow::bail!(
                "USB4 transport requires node_id 0 or 1 in a two-node topology; got {node_id}/{node_count}"
            );
        }

        let stream = if node_id == 1 {
            Self::accept_peer(cfg).await?
        } else {
            Self::connect_peer(cfg).await?
        };
        stream.set_nodelay(true)?;
        let stream = tokio::time::timeout(
            Duration::from_secs(cfg.connect_timeout_s),
            Self::handshake(stream, node_id),
        )
        .await
        .context("timed out during USB4 handshake")??;
        let (reader, writer) = stream.into_split();
        let (rx_tx, rx_chan) = mpsc::channel(RX_QUEUE_CAPACITY);
        let reader_task = tokio::spawn(Self::read_frames(
            reader,
            rx_tx,
            Duration::from_secs(cfg.io_timeout_s),
        ));

        Ok(Self {
            cfg: cfg.clone(),
            node_id,
            node_count,
            writer,
            rx_chan,
            reader_task,
            bytes_sent: 0,
            accumulated_rtt_us: 0.0,
            packets_sent: 0,
            packets_received: 0,
            next_recv_seq: 0,
        })
    }

    async fn accept_peer(cfg: &InterconnectConfig) -> anyhow::Result<TcpStream> {
        let listener = TcpListener::bind(&cfg.listen_addr)
            .await
            .with_context(|| format!("failed to bind USB4 listener at {}", cfg.listen_addr))?;
        tracing::info!(address = %cfg.listen_addr, "waiting for USB4 peer");
        let timeout = Duration::from_secs(cfg.connect_timeout_s);
        let (stream, peer) = tokio::time::timeout(timeout, listener.accept())
            .await
            .with_context(|| {
                format!(
                    "timed out waiting for USB4 peer after {}s",
                    cfg.connect_timeout_s
                )
            })??;
        tracing::info!(%peer, "accepted USB4 peer");
        Ok(stream)
    }

    async fn connect_peer(cfg: &InterconnectConfig) -> anyhow::Result<TcpStream> {
        let deadline = Instant::now() + Duration::from_secs(cfg.connect_timeout_s);
        let mut delay = Duration::from_millis(50);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                anyhow::bail!(
                    "failed to connect to USB4 peer {} within {}s",
                    cfg.peer_addr,
                    cfg.connect_timeout_s
                );
            }
            let attempt_timeout = remaining.min(Duration::from_secs(1));
            match tokio::time::timeout(attempt_timeout, TcpStream::connect(&cfg.peer_addr)).await {
                Ok(Ok(stream)) => {
                    tracing::info!(address = %cfg.peer_addr, "connected to USB4 peer");
                    return Ok(stream);
                }
                Ok(Err(error)) if Instant::now() < deadline => {
                    tracing::debug!(address = %cfg.peer_addr, %error, "USB4 peer not ready");
                    tokio::time::sleep(delay.min(remaining)).await;
                    delay = (delay * 2).min(Duration::from_secs(1));
                }
                Err(_) if Instant::now() < deadline => {
                    tracing::debug!(address = %cfg.peer_addr, "USB4 connect attempt timed out");
                }
                Ok(Err(error)) => {
                    return Err(error).with_context(|| {
                        format!(
                            "failed to connect to USB4 peer {} within {}s",
                            cfg.peer_addr, cfg.connect_timeout_s
                        )
                    });
                }
                Err(_) => {
                    anyhow::bail!(
                        "failed to connect to USB4 peer {} within {}s",
                        cfg.peer_addr,
                        cfg.connect_timeout_s
                    );
                }
            }
        }
    }

    async fn handshake(mut stream: TcpStream, node_id: usize) -> anyhow::Result<TcpStream> {
        let peer_id = 1 - node_id;
        let local = [
            PROTOCOL_MAGIC[0],
            PROTOCOL_MAGIC[1],
            PROTOCOL_MAGIC[2],
            PROTOCOL_MAGIC[3],
            PROTOCOL_VERSION,
            node_id as u8,
            peer_id as u8,
            0,
        ];
        stream.write_all(&local).await?;

        let mut remote = [0u8; HANDSHAKE_SIZE];
        stream.read_exact(&mut remote).await?;
        if remote[..4] != PROTOCOL_MAGIC
            || remote[4] != PROTOCOL_VERSION
            || remote[5] != peer_id as u8
            || remote[6] != node_id as u8
        {
            anyhow::bail!(
                "USB4 handshake rejected: expected peer {peer_id}, received version={} src={} dst={}",
                remote[4],
                remote[5],
                remote[6]
            );
        }
        Ok(stream)
    }

    async fn read_frames(
        mut reader: OwnedReadHalf,
        tx: mpsc::Sender<anyhow::Result<Vec<u8>>>,
        io_timeout: Duration,
    ) {
        loop {
            let mut length_bytes = [0u8; 4];
            match tokio::time::timeout(io_timeout, reader.read_exact(&mut length_bytes)).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) if error.kind() == ErrorKind::UnexpectedEof => break,
                Ok(Err(error)) => {
                    let _ = tx.send(Err(error.into())).await;
                    break;
                }
                Err(_) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!("USB4 frame header timed out")))
                        .await;
                    break;
                }
            }

            let frame_len = u32::from_be_bytes(length_bytes) as usize;
            if frame_len == 0 || frame_len > MAX_FRAME_BYTES {
                let _ = tx
                    .send(Err(anyhow::anyhow!(
                        "invalid USB4 frame length {frame_len}"
                    )))
                    .await;
                break;
            }

            let mut frame = vec![0u8; frame_len];
            match tokio::time::timeout(io_timeout, reader.read_exact(&mut frame)).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "truncated USB4 frame ({frame_len} bytes): {error}"
                        )))
                        .await;
                    break;
                }
                Err(_) => {
                    let _ = tx
                        .send(Err(anyhow::anyhow!(
                            "USB4 frame body timed out after {}s",
                            io_timeout.as_secs()
                        )))
                        .await;
                    break;
                }
            }
            if tx.send(Ok(frame)).await.is_err() {
                break;
            }
        }
    }

    pub fn transmission_time_us(cfg: &InterconnectConfig, payload_bytes: usize) -> f64 {
        let base_latency = if cfg.kernel_tune {
            cfg.base_latency_us
        } else {
            cfg.base_latency_us + 35.0
        };
        let bw_bytes_per_us = ((cfg.throughput_gbps.max(0.001) * 1e9 / 8.0) / 1e6).max(1e-9);
        let serialization = payload_bytes as f64 / bw_bytes_per_us;
        let packets = payload_bytes.div_ceil(cfg.mtu.max(1));
        base_latency + serialization + packets as f64 * cfg.tcp_overhead_us
    }

    fn tx_time(&self, payload_bytes: usize) -> f64 {
        Self::transmission_time_us(&self.cfg, payload_bytes)
    }

    pub async fn send_to_node(&mut self, dst: usize, data: &[u8]) -> anyhow::Result<Vec<u8>> {
        if dst >= self.node_count || dst == self.node_id {
            anyhow::bail!("invalid remote destination node {dst}");
        }
        let packet = Usb4Packet {
            src_node: self.node_id,
            dst_node: dst,
            seq: self.packets_sent,
            flags: PacketFlags::NONE.bits(),
            payload: data.to_vec(),
        };
        let encoded = bincode::serialize(&packet)?;
        if encoded.len() > MAX_FRAME_BYTES {
            anyhow::bail!(
                "USB4 frame {} exceeds {} byte limit",
                encoded.len(),
                MAX_FRAME_BYTES
            );
        }
        let frame_len = u32::try_from(encoded.len()).context("USB4 frame exceeds u32 length")?;
        tokio::time::timeout(Duration::from_secs(self.cfg.io_timeout_s), async {
            self.writer.write_all(&frame_len.to_be_bytes()).await?;
            self.writer.write_all(&encoded).await
        })
        .await
        .context("USB4 frame write timed out")??;

        let rtt = self.cfg.rtt_us;
        let comm_time = self.tx_time(data.len());
        self.bytes_sent += data.len() as u64;
        self.accumulated_rtt_us += rtt;
        self.packets_sent += 1;
        tracing::trace!(
            "USB4: node {} -> {}: {} bytes, comm={comm_time:.1}µs, RTT={rtt:.1}µs",
            self.node_id,
            dst,
            data.len()
        );
        Ok(encoded)
    }

    async fn recv_wire(&mut self) -> anyhow::Result<Vec<u8>> {
        self.rx_chan
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("USB4 peer disconnected"))?
    }

    pub async fn recv_dmabuf(&mut self) -> anyhow::Result<DmaBufDescriptor> {
        let bytes = self.recv_wire().await?;
        let packet = Self::decode_packet(&bytes)?;
        self.validate_inbound(&packet)?;
        self.packets_received += 1;
        let mut desc = DmaBufDescriptor::allocate(packet.payload.len().max(1))?;
        desc.write(&packet.payload)?;
        Ok(desc)
    }

    pub async fn send_recv(&mut self, data: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.send_to_node(1 - self.node_id, data).await?;
        let desc = self.recv_dmabuf().await?;
        Ok(desc.as_slice().to_vec())
    }

    pub async fn recv_packet(&mut self) -> anyhow::Result<Usb4Packet> {
        let bytes = self.recv_wire().await?;
        let packet = Self::decode_packet(&bytes)?;
        self.validate_inbound(&packet)?;
        self.packets_received += 1;
        Ok(packet)
    }

    pub fn try_recv_packet(&mut self) -> anyhow::Result<Option<Usb4Packet>> {
        let bytes = match self.rx_chan.try_recv() {
            Ok(result) => result?,
            Err(mpsc::error::TryRecvError::Empty) => return Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => {
                anyhow::bail!("USB4 peer disconnected")
            }
        };
        let packet = Self::decode_packet(&bytes)?;
        self.validate_inbound(&packet)?;
        self.packets_received += 1;
        Ok(Some(packet))
    }

    fn decode_packet(bytes: &[u8]) -> anyhow::Result<Usb4Packet> {
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(MAX_FRAME_BYTES as u64)
            .deserialize(bytes)
            .map_err(|error| anyhow::anyhow!("malformed packet on wire: {error}"))
    }

    fn validate_inbound(&mut self, packet: &Usb4Packet) -> anyhow::Result<()> {
        let expected_peer = 1 - self.node_id;
        if packet.dst_node != self.node_id || packet.src_node != expected_peer {
            anyhow::bail!(
                "misrouted USB4 packet src={} dst={} received by node {}",
                packet.src_node,
                packet.dst_node,
                self.node_id
            );
        }
        if packet.seq != self.next_recv_seq {
            anyhow::bail!(
                "USB4 packet sequence {} does not match expected {}",
                packet.seq,
                self.next_recv_seq
            );
        }
        self.next_recv_seq += 1;
        Ok(())
    }

    pub fn deserialize_msg(data: &[u8]) -> anyhow::Result<HccMessage> {
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .with_limit(MAX_FRAME_BYTES as u64)
            .deserialize(data)
            .map_err(|error| anyhow::anyhow!("malformed HCC message: {error}"))
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.writer.shutdown().await?;
        Ok(())
    }

    pub fn stats(&self) -> TransportStats {
        TransportStats {
            bytes_sent: self.bytes_sent,
            packets_sent: self.packets_sent,
            packets_received: self.packets_received,
            avg_rtt_us: if self.packets_sent > 0 {
                self.accumulated_rtt_us / self.packets_sent as f64
            } else {
                0.0
            },
        }
    }
}

impl Drop for Usb4Transport {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

#[derive(Debug, Clone)]
pub struct TransportStats {
    pub bytes_sent: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub avg_rtt_us: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cfg(address: String) -> InterconnectConfig {
        InterconnectConfig {
            link_count: 2,
            throughput_gbps: 45.0,
            rtt_us: 17.0,
            base_latency_us: 14.0,
            mtu: 9000,
            tcp_overhead_us: 1.2,
            kernel_tune: true,
            listen_addr: address.clone(),
            peer_addr: address,
            connect_timeout_s: 2,
            io_timeout_s: 2,
        }
    }

    fn unused_local_address() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().to_string()
    }

    async fn connected_pair() -> (Usb4Transport, Usb4Transport) {
        let cfg = test_cfg(unused_local_address());
        let (node_1, node_0) = tokio::try_join!(
            Usb4Transport::new(&cfg, 2, 1),
            Usb4Transport::new(&cfg, 2, 0)
        )
        .unwrap();
        (node_0, node_1)
    }

    #[test]
    fn test_transmission_time_prefill() {
        let cfg = test_cfg("127.0.0.1:50053".into());
        let t = Usb4Transport::transmission_time_us(&cfg, 1_230_000_000);
        assert!(t > 350_000.0 && t < 400_000.0, "t={t}µs");
    }

    #[test]
    fn test_transmission_time_decode() {
        let cfg = test_cfg("127.0.0.1:50053".into());
        let t = Usb4Transport::transmission_time_us(&cfg, 12_288);
        assert!(t > 10.0 && t < 30.0, "t={t}µs");
    }

    #[tokio::test]
    async fn test_send_receive_peer() {
        let (mut node_0, mut node_1) = connected_pair().await;
        node_0.send_to_node(1, b"hello HCC").await.unwrap();

        let packet = node_1.recv_packet().await.unwrap();
        assert_eq!(packet.src_node, 0);
        assert_eq!(packet.dst_node, 1);
        assert_eq!(packet.payload, b"hello HCC");
    }

    #[tokio::test]
    async fn test_outbound_packet_is_not_received_by_sender() {
        let (mut node_0, mut node_1) = connected_pair().await;
        node_0.send_to_node(1, b"peer only").await.unwrap();

        assert!(node_0.try_recv_packet().unwrap().is_none());
        assert_eq!(node_1.recv_packet().await.unwrap().payload, b"peer only");
    }

    #[tokio::test]
    async fn test_large_frame_survives_tcp_chunking() {
        let (mut node_0, mut node_1) = connected_pair().await;
        let payload = vec![0xA5; 2 * 1024 * 1024];
        node_0.send_to_node(1, &payload).await.unwrap();

        assert_eq!(node_1.recv_packet().await.unwrap().payload, payload);
    }

    #[tokio::test]
    async fn silent_handshake_times_out() {
        let address = unused_local_address();
        let mut cfg = test_cfg(address.clone());
        cfg.connect_timeout_s = 1;
        let server_cfg = cfg.clone();
        let server = tokio::spawn(async move { Usb4Transport::new(&server_cfg, 2, 1).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _silent_peer = TcpStream::connect(address).await.unwrap();

        let error = match server.await.unwrap() {
            Ok(_) => panic!("silent handshake unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("timed out during USB4 handshake"));
    }

    #[tokio::test]
    async fn partial_frame_body_times_out() {
        let (mut node_0, mut node_1) = connected_pair().await;
        node_0
            .writer
            .write_all(&100u32.to_be_bytes())
            .await
            .unwrap();
        node_0.writer.write_all(&[0xA5]).await.unwrap();

        let error = node_1.recv_packet().await.unwrap_err();
        assert!(error.to_string().contains("frame body timed out"));
    }

    #[tokio::test]
    async fn oversized_outbound_frame_is_rejected() {
        let (mut node_0, _node_1) = connected_pair().await;
        let payload = vec![0u8; MAX_FRAME_BYTES];

        let error = node_0.send_to_node(1, &payload).await.unwrap_err();
        assert!(error.to_string().contains("exceeds"));
    }
}
