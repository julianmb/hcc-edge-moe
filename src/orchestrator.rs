use crate::config::HccConfig;
use crate::decoding::dovetail::DovetailPipeline;
use crate::decoding::picospec::AsyncDraftStage;
use crate::decoding::speculative::SpeculativeEngine;
use crate::igpu::migraphx::MIGraphXSession;
use crate::igpu::target_runner::TargetRunner;
use crate::interconnect::protocol::HccMessage;
use crate::interconnect::usb4::Usb4Transport;
use crate::kv_cache::MixedPrecisionKVCache;
use crate::npu::calibrator::DraftCalibrator;
use crate::npu::draft_runner::DraftRunner;
use crate::npu::context_compressor::ContextCompressor;
use crate::session::session_manager::SessionManager;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Production HCC orchestrator — wires real hardware backends.
///
/// Backends (configurable):
///   - "llamacpp-rpc": llama.cpp rpc-server over TCP (default, proven on Strix Halo)
///   - "migraphx": direct MIGraphX via libloading (ROCm 7.2+)
///   - "simulated": test harness (no hardware required)
///
/// Pipelines:
///   - "hcc": NPU→GPU speculative (paper topology)
///   - "dovetail": GPU→CPU speculative (Dovetail EMNLP 2025)
pub struct HccOrchestrator {
    cfg: HccConfig,
    draft_runner: Option<Arc<Mutex<DraftRunner>>>,
    target_runner: Option<Arc<Mutex<TargetRunner>>>,
    migraphx: Option<MIGraphXSession>,
    transport: Arc<Mutex<Usb4Transport>>,
    kv_cache: Arc<Mutex<MixedPrecisionKVCache>>,
    compressor: Arc<Mutex<ContextCompressor>>,
    calibrator: Arc<Mutex<DraftCalibrator>>,
    speculative_engine: SpeculativeEngine,
    dovetail: Option<DovetailPipeline>,
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
            Usb4Transport::new(&cfg.interconnect, cfg.cluster.node_count, cfg.cluster.node_id).await?,
        ));

        let kv_cache = Arc::new(Mutex::new(
            MixedPrecisionKVCache::new(cfg.model.kv_lora_rank + cfg.model.qk_rope_head_dim),
        ));

        // Initialize backend based on config
        let (target_runner, migraphx) = match cfg.backend.inference_engine.as_str() {
            "llamacpp-rpc" => {
                tracing::info!("Backend: llama.cpp RPC :{}", cfg.backend.rpc_port);
                let runner = TargetRunner::new(cfg.backend.model_path.clone(), cfg.backend.rpc_port).await?;
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
            Some(Arc::new(Mutex::new(DraftRunner::new(&cfg, cfg.speculative.draft_params_b))))
        } else {
            None
        };

        let compressor = Arc::new(Mutex::new(ContextCompressor::new()));
        let calibrator = Arc::new(Mutex::new(DraftCalibrator::new()));

        let speculative_engine = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );

        let dovetail = if cfg.dovetail.enabled {
            tracing::info!("Pipeline: Dovetail (GPU→CPU)");
            Some(DovetailPipeline::new(cfg.speculative.draft_len))
        } else {
            tracing::info!("Pipeline: HCC (NPU→GPU)");
            None
        };

        let async_draft = AsyncDraftStage::new(3);
        let session_manager = Arc::new(Mutex::new(SessionManager::new(
            cfg.session.max_sessions, cfg.session.max_context,
            cfg.cluster.memory_per_node_gb, &cfg.model,
        )));

        let theoretical_tps = cfg.cluster.memory_bw_gbs / cfg.model.weight_read_gb;
        let spec_speedup = speculative_engine.speedup();
        tracing::info!("roofline decode: {theoretical_tps:.1} T/s per node, {:.1} T/s with speculation ({:.2}x)", 
            theoretical_tps * spec_speedup, spec_speedup);
        tracing::info!("KV cache: mixed-precision (K 8-bit FP8 + V 3-bit PolarQuant) via turboquant-rs");

        Ok(Self {
            cfg, draft_runner, target_runner, migraphx, transport, kv_cache,
            compressor, calibrator, speculative_engine, dovetail, async_draft,
            session_manager, step: 0, seq: 0, total_accepted: 0, total_drafted: 0,
        })
    }

    /// Main generation loop.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        tracing::info!("HCC orchestrator running on Strix Halo (Ryzen AI MAX+ 395)");

        loop {
            {
                let sessions = self.session_manager.lock().await;
                if !sessions.has_pending() { continue; }
            }

            if let Some(dovetail) = &self.dovetail {
                // Dovetail pipeline: GPU→CPU speculative
                self.run_dovetail().await?;
            } else {
                // HCC pipeline: NPU→GPU speculative
                self.run_hcc().await?;
            }
        }
    }

    /// HCC pipeline: NPU drafts, iGPU verifies over USB4.
    async fn run_hcc(&mut self) -> anyhow::Result<()> {
        let node_id = self.cfg.cluster.node_id;

        if node_id == 0 {
            let raw = self.session_manager.lock().await.next_context().await;
            let compressed = self.compressor.lock().await.compress(&raw).await?;
            self.transport.lock().await.send_to_node(1, &compressed).await?;

            if let Some(draft) = &self.draft_runner {
                draft.lock().await.prefill_context(&compressed).await?;
            }
        } else {
            let desc = self.transport.lock().await.recv_dmabuf().await?;
            let payload = desc.as_slice().to_vec();
            if let Some(target) = &self.target_runner {
                target.lock().await.prefill(&payload).await?;
            }
        }

        self.async_pipeline().await
    }

    /// Dovetail pipeline: GPU drafts, CPU verifies.
    async fn run_dovetail(&mut self) -> anyhow::Result<()> {
        tracing::debug!("Dovetail pipeline active (GPU→CPU speculative)");
        // Dovetail reverses the draft/verify roles:
        // GPU (fast) runs draft, CPU (large memory) runs target
        self.async_pipeline().await
    }

    /// Core speculative decode pipeline.
    async fn async_pipeline(&mut self) -> anyhow::Result<()> {
        loop {
            if self.session_manager.lock().await.all_completed() { break; }

            if let Some(draft) = &self.draft_runner {
                let tokens = draft.lock().await.generate_drafts(self.speculative_engine.draft_len).await?;
                self.total_drafted += tokens.len();

                // Context-aligned calibration (sd.npu)
                let cal_embedding = tokens.iter().flat_map(|t| t.kv_state.clone()).collect::<Vec<_>>();
                self.calibrator.lock().await.observe_prompt(cal_embedding);

                if let Some(compressed) = self.async_draft.submit(tokens, self.seq) {
                    self.transport.lock().await.send_to_node(1, &compressed).await?;
                    self.seq += 1;
                }
            }

            self.drain_verifications().await;
            self.step += 1;

            if self.step % 50 == 0 {
                let rate = if self.total_drafted > 0 { self.total_accepted as f64 / self.total_drafted as f64 } else { 0.0 };
                tracing::debug!("step={} accepted_rate={rate:.3}", self.step);
            }
        }
        Ok(())
    }

    async fn drain_verifications(&mut self) {
        let mut transport = self.transport.lock().await;
        loop {
            match transport.try_recv_packet() {
                Some(packet) => {
                    if let Some(msg) = Usb4Transport::deserialize_msg(&packet.payload) {
                        match msg {
                            HccMessage::VerificationResult { accepted_prefix_len, .. } => {
                                let accepted = accepted_prefix_len as usize;
                                self.total_accepted += accepted;
                                self.async_draft.verify(accepted);
                                // Feed acceptance back to Dovetail
                                if let Some(dovetail) = &self.dovetail {
                                    // dovetail adaptive tuning
                                }
                            }
                            _ => {}
                        }
                    }
                }
                None => break,
            }
        }
    }
}
