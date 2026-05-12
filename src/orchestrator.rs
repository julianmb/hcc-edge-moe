use crate::config::HccConfig;
use crate::decoding::picospec::AsyncDraftStage;
use crate::decoding::speculative::SpeculativeEngine;
use crate::igpu::target_runner::TargetRunner;
use crate::interconnect::protocol::{HccMessage, SessionStatus};
use crate::interconnect::usb4::Usb4Transport;
use crate::kv_cache::MixedPrecisionKVCache;
use crate::npu::draft_runner::DraftRunner;
use crate::npu::context_compressor::ContextCompressor;
use crate::session::session_manager::SessionManager;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Top-level HCC orchestrator — PicoSpec async pipeline + Algorithm 1.
pub struct HccOrchestrator {
    cfg: HccConfig,
    draft_runner: Arc<Mutex<DraftRunner>>,
    target_runner: Arc<Mutex<TargetRunner>>,
    transport: Arc<Mutex<Usb4Transport>>,
    kv_cache: Arc<Mutex<MixedPrecisionKVCache>>,
    compressor: Arc<Mutex<ContextCompressor>>,
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
            Usb4Transport::new(&cfg.interconnect, cfg.cluster.node_count, cfg.cluster.node_id)
                .await?,
        ));

        let kv_cache = Arc::new(Mutex::new(MixedPrecisionKVCache::new(
            cfg.model.kv_lora_rank + cfg.model.qk_rope_head_dim,
        )));

        let draft_runner = Arc::new(Mutex::new(DraftRunner::new(
            &cfg,
            cfg.speculative.draft_params_b,
        )));

        let target_runner = Arc::new(Mutex::new(TargetRunner::new(
            &cfg,
            cfg.model.checkpoint_path.clone(),
        )));

        let compressor = Arc::new(Mutex::new(ContextCompressor::new()));

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

        tracing::info!(
            "HCC: pipeline depth=3, mixed-precision KV (K 8-bit + V 3-bit), PicoSpec async SD"
        );

        Ok(Self {
            cfg,
            draft_runner,
            target_runner,
            transport,
            kv_cache,
            compressor,
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
        let node_id = self.cfg.cluster.node_id;
        tracing::info!("HCC orchestrator started (node {node_id})");

        loop {
            // Check for pending sessions (lock not held across .await)
            {
                let sessions = self.session_manager.lock().await;
                if !sessions.has_pending() {
                    continue;
                }
            }

            // Phase 1: Cascaded Context Compression (Node 1 NPU only)
            if node_id == 0 {
                let raw = self.session_manager.lock().await.next_context().await;
                let compressed = self.compressor.lock().await.compress(&raw).await?;
                self.transport.lock().await.send_to_node(1, &compressed).await?;
            } else {
                let desc = self.transport.lock().await.recv_dmabuf().await?;
                let payload = desc.as_slice().to_vec();
                self.target_runner.lock().await.prefill(&payload).await?;
            }

            // Phase 2: PicoSpec async speculative decode
            self.async_pipeline().await?;
        }
    }

    /// PicoSpec async pipeline: NPU drafts continuously while verifications
    /// arrive asynchronously. Pipeline depth = 3 in-flight batches.
    async fn async_pipeline(&mut self) -> anyhow::Result<()> {
        loop {
            // Check completion (drop lock before .await)
            if self.session_manager.lock().await.all_completed() {
                break;
            }

            // Step A: NPU generates γ draft tokens
            let drafts = self.draft_runner
                .lock()
                .await
                .generate_drafts(self.speculative_engine.draft_len)
                .await?;
            self.total_drafted += drafts.len();

            // Step B: Submit to async pipeline (sparse compressed)
            if let Some(compressed) = self.async_draft.submit(drafts, self.seq) {
                self.transport
                    .lock()
                    .await
                    .send_to_node(1, &compressed)
                    .await?;
                self.seq += 1;
            }

            // Step C: Drain all available verification results
            self.drain_verifications().await;

            self.step += 1;
            if self.step % 50 == 0 {
                let rate = if self.total_drafted > 0 {
                    self.total_accepted as f64 / self.total_drafted as f64
                } else {
                    0.0
                };
                tracing::debug!("step={} accepted_rate={rate:.3}", self.step);
            }
        }
        Ok(())
    }

    /// Drain completed verifications from the async pipeline.
    ///
    /// Deserializes incoming verification results, feeds them to
    /// `AsyncDraftStage::verify()` to release pending batches,
    /// and updates acceptance statistics.
    async fn drain_verifications(&mut self) {
        let mut transport = self.transport.lock().await;
        loop {
            match transport.try_recv_packet() {
                Some(packet) => {
                    if let Some(msg) = Usb4Transport::deserialize_msg(&packet.payload) {
                        match msg {
                            HccMessage::VerificationResult {
                                accepted_prefix_len,
                                ..
                            } => {
                                let accepted = accepted_prefix_len as usize;
                                self.total_accepted += accepted;
                                // Release oldest pending batch
                                self.async_draft.verify(accepted);
                            }
                            HccMessage::ContextSync { accepted_tokens, .. } => {
                                self.total_accepted += accepted_tokens.len();
                            }
                            _ => {}
                        }
                    }
                }
                None => break,
            }
        }
    }

    pub fn cfg(&self) -> &HccConfig {
        &self.cfg
    }
}
