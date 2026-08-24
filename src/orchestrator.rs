use crate::config::HccConfig;
use crate::decoding::picospec::AsyncDraftStage;
use crate::decoding::speculative::SpeculativeEngine;
use crate::igpu::migraphx::MIGraphXSession;
use crate::igpu::target_runner::TargetRunner;
use crate::interconnect::protocol::HccMessage;
use crate::interconnect::usb4::Usb4Transport;
use crate::kv_cache::MixedPrecisionKVCache;
use crate::npu::draft_runner::DraftRunner;
use crate::session::metrics;
use crate::session::session_manager::SessionManager;
use std::sync::Arc;
use tokio::sync::Mutex;

/// HCC orchestrator — wires the paper's dual-node speculative pipeline.
///
/// Topology (paper Algorithm 1, Section 7):
///   Node 1 (NPU):  8B draft model generates γ candidate tokens
///   USB4 bridge:   draft batch transmitted as single TCP/IP crossing
///   Node 2 (iGPU): 380B target model verifies γ tokens in parallel
///   Result:        E[k] accepted tokens per crossing, hiding ~17 µs RTT
pub struct HccOrchestrator {
    cfg: HccConfig,
    draft_runner: Option<Arc<Mutex<DraftRunner>>>,
    target_runner: Option<Arc<Mutex<TargetRunner>>>,
    migraphx: Option<MIGraphXSession>,
    transport: Arc<Mutex<Usb4Transport>>,
    kv_cache: Arc<Mutex<MixedPrecisionKVCache>>,
    speculative_engine: SpeculativeEngine,
    async_draft: AsyncDraftStage,
    session_manager: Arc<Mutex<SessionManager>>,
    step: u64,
    seq: u64,
    total_accepted: usize,
    total_drafted: usize,
}

impl HccOrchestrator {
    pub async fn new(cfg: HccConfig) -> anyhow::Result<Self> {
        cfg.validate();

        let transport = Arc::new(Mutex::new(
            Usb4Transport::new(
                &cfg.interconnect,
                cfg.cluster.node_count,
                cfg.cluster.node_id,
            )
            .await?,
        ));

        let kv_cache = Arc::new(Mutex::new(MixedPrecisionKVCache::new(
            cfg.model.kv_lora_rank + cfg.model.qk_rope_head_dim,
        )));

        let (target_runner, migraphx) = match cfg.backend.inference_engine.as_str() {
            "llamacpp-rpc" => {
                tracing::info!("Backend: llama.cpp RPC :{}", cfg.backend.rpc_port);
                let runner = TargetRunner::new(cfg.backend.clone()).await?;
                (Some(Arc::new(Mutex::new(runner))), None)
            }
            "migraphx" => {
                tracing::info!("Backend: MIGraphX (ROCm 7.2)");
                let mx = MIGraphXSession::load(&cfg.backend.model_path)?;
                (None, Some(mx))
            }
            _ => {
                tracing::warn!("Backend: SIMULATED (no hardware)");
                (None, None)
            }
        };

        let draft_runner = if cfg.speculative.draft_params_b > 0.0 {
            Some(Arc::new(Mutex::new(DraftRunner::new(
                &cfg.backend.model_path,
                cfg.speculative.draft_params_b,
                cfg.backend.rpc_port,
            ))))
        } else {
            None
        };

        let speculative_engine = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );

        let async_draft = AsyncDraftStage::new(3);
        let session_manager = Arc::new(Mutex::new(SessionManager::new(
            cfg.session.max_sessions,
            cfg.session.max_context,
            cfg.cluster.memory_per_node_gb,
            &cfg.model,
        )));

        let theoretical_tps = cfg.cluster.memory_bw_gbs / cfg.model.weight_read_gb();
        let spec_speedup = speculative_engine.speedup();
        tracing::info!("roofline decode: {theoretical_tps:.1} tok/s per node, {:.1} tok/s with speculation ({:.2}x)",
            theoretical_tps * spec_speedup, spec_speedup);
        tracing::info!(
            "KV cache: mixed-precision (K 8-bit FP8 + V 3-bit TurboQuant) via turboquant-rs"
        );

        Ok(Self {
            cfg,
            draft_runner,
            target_runner,
            migraphx,
            transport,
            kv_cache,
            speculative_engine,
            async_draft,
            session_manager,
            step: 0,
            seq: 0,
            total_accepted: 0,
            total_drafted: 0,
        })
    }

    /// Main generation loop.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        tracing::info!(
            "HCC orchestrator running ({} nodes)",
            self.cfg.cluster.node_count
        );

        loop {
            {
                let sessions = self.session_manager.lock().await;
                if !sessions.has_pending() {
                    continue;
                }
            }
            self.run_hcc().await?;
        }
    }

    /// HCC pipeline: NPU drafts on Node 1, iGPU verifies on Node 2 over USB4.
    async fn run_hcc(&mut self) -> anyhow::Result<()> {
        let node_id = self.cfg.cluster.node_id;

        if node_id == 0 {
            // Node 1: NPU-side. Send context to Node 2 for prefill.
            let raw = self.session_manager.lock().await.next_context().await;

            let ttft_start = std::time::Instant::now();
            self.transport.lock().await.send_to_node(1, &raw).await?;
            metrics::record_ttft(raw.len(), ttft_start.elapsed().as_secs_f64() * 1000.0);
        } else {
            // Node 2: iGPU-side. Receive context and run prefill.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let desc = self.transport.lock().await.recv_dmabuf().await?;
            let payload = desc.as_slice().to_vec();
            if let Some(target) = &self.target_runner {
                target.lock().await.prefill(&payload).await?;
            }
        }

        self.async_pipeline().await
    }

    /// Core speculative decode loop (paper Algorithm 1).
    async fn async_pipeline(&mut self) -> anyhow::Result<()> {
        loop {
            if self.session_manager.lock().await.all_completed() {
                break;
            }

            if let Some(draft) = &self.draft_runner {
                // Step 1: NPU generates γ draft tokens
                let tokens = draft
                    .lock()
                    .await
                    .generate_drafts(self.speculative_engine.draft_len)
                    .await?;

                self.total_drafted += tokens.len();

                metrics::record_speculative_step(
                    tokens.len(),
                    self.speculative_engine.draft_len,
                    0.0,
                );

                // Step 2: Send draft batch over USB4 as single TCP/IP crossing
                if let Some(compressed) = self.async_draft.submit(tokens, self.seq) {
                    let transport_clone = self.transport.clone();
                    let seq_clone = self.seq;
                    let compressed_clone = compressed.clone();

                    tokio::spawn(async move {
                        if let Err(err) = transport_clone
                            .lock()
                            .await
                            .send_to_node(1, &compressed_clone)
                            .await
                        {
                            tracing::warn!("failed to send draft batch seq={seq_clone}: {err:#}");
                        } else {
                            tracing::trace!("sent draft batch seq={seq_clone}");
                        }
                    });

                    self.seq += 1;
                }
            }

            // Step 3: iGPU verifies and returns accepted prefix length
            self.drain_verifications().await;
            self.step += 1;

            if self.step % 50 == 0 {
                let rate = if self.total_drafted > 0 {
                    self.total_accepted as f64 / self.total_drafted as f64
                } else {
                    0.0
                };
                tracing::debug!("step={} accepted_rate={rate:.3}", self.step);
                metrics::record_decode_throughput(self.total_accepted as f64 / 50.0);
                let kv_len = self.kv_cache.lock().await.len();
                metrics::record_kv_cache(
                    self.session_manager.lock().await.session_count(),
                    kv_len as f64,
                );
            }
        }
        Ok(())
    }

    async fn drain_verifications(&mut self) {
        let mut transport = self.transport.lock().await;
        while let Some(packet) = transport.try_recv_packet() {
            if let Some(HccMessage::VerificationResult {
                accepted_prefix_len,
                ..
            }) = Usb4Transport::deserialize_msg(&packet.payload)
            {
                let accepted = accepted_prefix_len as usize;
                self.total_accepted += accepted;
                self.async_draft.verify(accepted);
            }
        }
    }
}
