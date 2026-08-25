use crate::config::HccConfig;
use crate::decoding::picospec::AsyncDraftStage;
use crate::decoding::speculative::{DraftToken, SpeculativeEngine};
use crate::interconnect::protocol::{HccMessage, SessionStatus};
use crate::interconnect::usb4::Usb4Transport;
use crate::kv_cache::MixedPrecisionKVCache;
use crate::session::metrics;
use crate::session::session_manager::SessionManager;
use anyhow::Context;

const SESSION_ID: u64 = 1;

pub struct HccOrchestrator {
    cfg: HccConfig,
    prompt: String,
    max_tokens: usize,
    transport: Usb4Transport,
    kv_cache: MixedPrecisionKVCache,
    speculative_engine: SpeculativeEngine,
    async_draft: AsyncDraftStage,
    session_manager: SessionManager,
    seq: u64,
    total_accepted: usize,
    total_drafted: usize,
}

impl HccOrchestrator {
    pub async fn new(cfg: HccConfig, prompt: String, max_tokens: usize) -> anyhow::Result<Self> {
        cfg.validate();
        cfg.validate_hcc_topology()?;
        if max_tokens == 0 || max_tokens > cfg.session.max_context {
            anyhow::bail!(
                "max_tokens must be in 1..={}, got {max_tokens}",
                cfg.session.max_context
            );
        }
        if cfg.backend.inference_engine != "simulated" {
            anyhow::bail!(
                "hcch run currently supports only the simulated backend; use measure for real llama.cpp execution"
            );
        }

        let node_id = cfg.cluster.node_id;
        let transport =
            Usb4Transport::new(&cfg.interconnect, cfg.cluster.node_count, node_id).await?;
        let speculative_engine = SpeculativeEngine::new(
            cfg.speculative.draft_len,
            cfg.speculative.acceptance_rate,
            cfg.speculative.draft_cost_ratio,
        );
        let mut session_manager = SessionManager::new(
            cfg.session.max_sessions,
            cfg.session.max_context,
            cfg.cluster.memory_per_node_gb,
            &cfg.model,
        );
        if node_id == 0 {
            session_manager.create_session(SESSION_ID, max_tokens)?;
        }
        let kv_cache =
            MixedPrecisionKVCache::new(cfg.model.kv_lora_rank + cfg.model.qk_rope_head_dim);

        Ok(Self {
            cfg,
            prompt,
            max_tokens,
            transport,
            kv_cache,
            speculative_engine,
            async_draft: AsyncDraftStage::new(1),
            session_manager,
            seq: 0,
            total_accepted: 0,
            total_drafted: 0,
        })
    }

    pub async fn run(&mut self) -> anyhow::Result<()> {
        tracing::info!(
            node_id = self.cfg.cluster.node_id,
            "HCC orchestrator running"
        );
        if self.cfg.cluster.node_id == 0 {
            self.run_draft_node().await
        } else {
            self.run_target_node().await
        }
    }

    async fn run_draft_node(&mut self) -> anyhow::Result<()> {
        self.send_message(
            1,
            &HccMessage::SessionRequest {
                session_id: SESSION_ID,
                max_tokens: u32::try_from(self.max_tokens).context("max_tokens exceeds u32")?,
            },
        )
        .await?;
        match self.recv_message().await? {
            HccMessage::SessionResponse {
                session_id: SESSION_ID,
                status: SessionStatus::Accepted,
            } => {}
            message => anyhow::bail!("target rejected session startup: {message:?}"),
        }

        self.session_manager
            .activate_next()
            .context("missing pending draft session")?;
        let prefill = HccMessage::PrefillPayload {
            tokens: Vec::new(),
            compressed_activations: self.prompt.as_bytes().to_vec(),
            context_len: u32::try_from(self.prompt.len()).context("prompt too large")?,
        };
        self.send_message(1, &prefill).await?;

        while !self.session_manager.all_completed() {
            let drafts = self.generate_drafts().await?;
            let draft_len = u8::try_from(drafts.len()).context("draft batch exceeds u8")?;
            let draft_ids = drafts
                .iter()
                .map(|token| token.token_id)
                .collect::<Vec<_>>();
            let message = HccMessage::DraftBatch {
                seq: self.seq,
                tokens: draft_ids.clone(),
                probabilities: drafts
                    .iter()
                    .map(|token| token.probability as f32)
                    .collect(),
                draft_len,
            };
            if self.async_draft.submit(drafts, self.seq).is_none() {
                anyhow::bail!("draft pipeline rejected batch {}", self.seq);
            }
            self.send_message(1, &message).await?;

            match self.recv_message().await? {
                HccMessage::VerificationResult {
                    seq,
                    accepted_prefix_len,
                    accepted_tokens,
                    ..
                } => {
                    let remaining = self.max_tokens.saturating_sub(self.total_accepted);
                    let accepted_prefix = accepted_prefix_len as usize;
                    if seq != self.seq {
                        anyhow::bail!("verification sequence {seq} does not match {}", self.seq);
                    }
                    if accepted_prefix > draft_ids.len()
                        || accepted_tokens.len() != accepted_prefix.saturating_add(1)
                        || accepted_tokens.len() > remaining
                        || accepted_tokens[..accepted_prefix] != draft_ids[..accepted_prefix]
                    {
                        anyhow::bail!(
                            "invalid verification result for sequence {seq}: prefix={accepted_prefix}, tokens={}",
                            accepted_tokens.len(),
                        );
                    }
                    self.total_accepted += accepted_tokens.len();
                    self.async_draft.verify(accepted_prefix);
                    self.session_manager
                        .advance(SESSION_ID, accepted_tokens.len());
                    metrics::record_acceptance(accepted_prefix, self.speculative_engine.draft_len);
                    tracing::info!(
                        seq,
                        accepted_prefix,
                        generated = self.total_accepted,
                        target = self.max_tokens,
                        "verification completed"
                    );
                    self.seq += 1;
                }
                message => anyhow::bail!("expected verification result, received {message:?}"),
            }
        }

        self.send_message(1, &HccMessage::Shutdown).await?;
        self.transport.shutdown().await?;
        tracing::info!(generated = self.total_accepted, "draft session completed");
        Ok(())
    }

    async fn run_target_node(&mut self) -> anyhow::Result<()> {
        let (session_id, max_tokens) = match self.recv_message().await? {
            HccMessage::SessionRequest {
                session_id,
                max_tokens,
            } if max_tokens > 0 && max_tokens as usize <= self.cfg.session.max_context => {
                (session_id, max_tokens)
            }
            HccMessage::SessionRequest {
                session_id,
                max_tokens,
            } => {
                self.send_message(
                    0,
                    &HccMessage::SessionResponse {
                        session_id,
                        status: SessionStatus::Rejected(format!(
                            "max_tokens {max_tokens} outside target limit 1..={}",
                            self.cfg.session.max_context
                        )),
                    },
                )
                .await?;
                anyhow::bail!("rejected invalid session {session_id}");
            }
            message => anyhow::bail!("invalid session request: {message:?}"),
        };
        self.send_message(
            0,
            &HccMessage::SessionResponse {
                session_id,
                status: SessionStatus::Accepted,
            },
        )
        .await?;

        match self.recv_message().await? {
            HccMessage::PrefillPayload {
                compressed_activations,
                ..
            } => {
                let _ = compressed_activations;
            }
            message => anyhow::bail!("expected prefill payload, received {message:?}"),
        }

        let mut expected_seq = 0u64;
        let mut generated_tokens = 0usize;
        loop {
            match self.recv_message().await? {
                HccMessage::DraftBatch {
                    seq,
                    tokens,
                    probabilities,
                    draft_len,
                } => {
                    validate_draft_batch(
                        seq,
                        expected_seq,
                        &tokens,
                        &probabilities,
                        draft_len,
                        self.speculative_engine.draft_len,
                    )?;
                    let remaining = (max_tokens as usize).saturating_sub(generated_tokens);
                    let (accepted_prefix_len, accepted_tokens, target_probabilities) =
                        simulated_verification(
                            self.cfg.speculative.acceptance_rate,
                            seq,
                            &tokens,
                            remaining,
                        )?;
                    generated_tokens += accepted_tokens.len();
                    self.send_message(
                        0,
                        &HccMessage::VerificationResult {
                            seq,
                            accepted_prefix_len: u8::try_from(accepted_prefix_len)
                                .context("accepted prefix exceeds u8")?,
                            accepted_tokens,
                            logits: Vec::new(),
                            probabilities: target_probabilities,
                        },
                    )
                    .await?;
                    expected_seq += 1;
                }
                HccMessage::Shutdown => {
                    tracing::info!(session_id, max_tokens, "target session completed");
                    self.transport.shutdown().await?;
                    return Ok(());
                }
                message => anyhow::bail!("unexpected target-node message: {message:?}"),
            }
        }
    }

    async fn generate_drafts(&mut self) -> anyhow::Result<Vec<DraftToken>> {
        let drafts = (0..self.speculative_engine.draft_len)
            .map(|offset| DraftToken {
                token_id: self
                    .seq
                    .saturating_mul(self.speculative_engine.draft_len as u64)
                    .saturating_add(offset as u64) as u32,
                probability: 0.5,
                kv_state: Vec::new(),
            })
            .collect::<Vec<_>>();
        if drafts.is_empty() {
            anyhow::bail!("draft backend returned an empty batch");
        }
        self.total_drafted += drafts.len();
        Ok(drafts)
    }

    async fn send_message(&mut self, dst: usize, message: &HccMessage) -> anyhow::Result<()> {
        let payload = bincode::serialize(message)?;
        self.transport.send_to_node(dst, &payload).await?;
        Ok(())
    }

    async fn recv_message(&mut self) -> anyhow::Result<HccMessage> {
        let packet = self.transport.recv_packet().await?;
        Usb4Transport::deserialize_msg(&packet.payload)
            .with_context(|| format!("malformed HCC message in packet {}", packet.seq))
    }

    pub fn transport_stats(&self) -> crate::interconnect::usb4::TransportStats {
        self.transport.stats()
    }

    pub fn kv_cache_len(&self) -> usize {
        self.kv_cache.len()
    }
}

const fn simulated_target_token(seq: u64) -> u32 {
    0x8000_0000 | seq as u32
}

fn simulated_verification(
    acceptance_rate: f64,
    seq: u64,
    tokens: &[u32],
    remaining: usize,
) -> anyhow::Result<(usize, Vec<u32>, Vec<f32>)> {
    if tokens.is_empty() || remaining == 0 {
        anyhow::bail!("simulated verifier requires drafts and remaining token budget");
    }
    let desired_prefix =
        ((acceptance_rate * (tokens.len() + 1) as f64).floor() as usize).min(tokens.len());
    let accepted_prefix = desired_prefix.min(remaining.saturating_sub(1));
    let mut accepted_tokens = tokens[..accepted_prefix].to_vec();
    accepted_tokens.push(simulated_target_token(seq));
    Ok((
        accepted_prefix,
        accepted_tokens,
        vec![acceptance_rate as f32; tokens.len()],
    ))
}

fn validate_draft_batch(
    seq: u64,
    expected_seq: u64,
    tokens: &[u32],
    probabilities: &[f32],
    draft_len: u8,
    max_draft_len: usize,
) -> anyhow::Result<()> {
    if seq != expected_seq {
        anyhow::bail!("draft sequence {seq} does not match {expected_seq}");
    }
    if tokens.is_empty()
        || tokens.len() != probabilities.len()
        || tokens.len() != draft_len as usize
        || tokens.len() > max_draft_len
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
    {
        anyhow::bail!("malformed draft batch sequence {seq}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unused_local_address() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().to_string()
    }

    #[tokio::test]
    async fn simulated_nodes_complete_typed_speculative_session() {
        let address = unused_local_address();
        let mut node_0_cfg = HccConfig::default();
        node_0_cfg.backend.inference_engine = "simulated".into();
        node_0_cfg.interconnect.listen_addr = address.clone();
        node_0_cfg.interconnect.peer_addr = address.clone();
        node_0_cfg.interconnect.connect_timeout_s = 2;
        let mut node_1_cfg = node_0_cfg.clone();
        node_1_cfg.cluster.node_id = 1;

        let (node_1, node_0) = tokio::try_join!(
            HccOrchestrator::new(node_1_cfg, "ignored".into(), 5),
            HccOrchestrator::new(node_0_cfg, "hello GLM".into(), 5)
        )
        .unwrap();
        let (node_1_result, node_0_result) = tokio::join!(
            async move {
                let mut node = node_1;
                let result = node.run().await;
                (result, node.transport_stats())
            },
            async move {
                let mut node = node_0;
                let result = node.run().await;
                (result, node.transport_stats(), node.total_accepted)
            }
        );

        node_0_result.0.unwrap();
        node_1_result.0.unwrap();
        assert!(node_0_result.1.packets_sent >= 4);
        assert_eq!(node_0_result.2, 5);
        assert!(node_1_result.1.packets_received >= 3);
    }

    #[test]
    fn simulated_verifier_preserves_prefix_and_generates_target_token() {
        let tokens = [10, 11, 12, 13, 14];

        let zero = simulated_verification(0.0, 1, &tokens, 10).unwrap();
        assert_eq!(zero.0, 0);
        assert_eq!(zero.1, vec![simulated_target_token(1)]);

        let partial = simulated_verification(0.5, 2, &tokens, 10).unwrap();
        assert_eq!(partial.0, 3);
        assert_eq!(&partial.1[..3], &tokens[..3]);
        assert_eq!(partial.1[3], simulated_target_token(2));

        let full = simulated_verification(0.99, 3, &tokens, 10).unwrap();
        assert_eq!(full.0, tokens.len());
        assert_eq!(&full.1[..tokens.len()], tokens);
        assert_eq!(full.1[tokens.len()], simulated_target_token(3));

        let final_batch = simulated_verification(0.99, 4, &tokens, 2).unwrap();
        assert_eq!(final_batch.0, 1);
        assert_eq!(final_batch.1, vec![tokens[0], simulated_target_token(4)]);
    }

    #[tokio::test]
    async fn run_rejects_real_backend_before_connecting() {
        let mut cfg = HccConfig::default();
        cfg.backend.inference_engine = "llamacpp-rpc".into();

        let error = HccOrchestrator::new(cfg, "prompt".into(), 1)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("only the simulated backend"));
    }

    #[test]
    fn draft_batch_validation_rejects_untrusted_values() {
        assert!(validate_draft_batch(0, 0, &[], &[], 0, 5).is_err());
        assert!(validate_draft_batch(1, 0, &[1], &[0.5], 1, 5).is_err());
        assert!(validate_draft_batch(0, 0, &[1], &[f32::NAN], 1, 5).is_err());
        assert!(validate_draft_batch(0, 0, &[1], &[1.1], 1, 5).is_err());
        assert!(validate_draft_batch(0, 0, &[1], &[0.5], 1, 5).is_ok());
    }

    #[tokio::test]
    async fn target_rejects_session_above_its_context_limit() {
        let address = unused_local_address();
        let mut node_0_cfg = HccConfig::default();
        node_0_cfg.backend.inference_engine = "simulated".into();
        node_0_cfg.session.max_context = 2048;
        node_0_cfg.interconnect.listen_addr = address.clone();
        node_0_cfg.interconnect.peer_addr = address.clone();
        node_0_cfg.interconnect.connect_timeout_s = 2;
        node_0_cfg.interconnect.io_timeout_s = 2;
        let mut node_1_cfg = node_0_cfg.clone();
        node_1_cfg.cluster.node_id = 1;
        node_1_cfg.session.max_context = 1024;

        let (mut node_1, mut node_0) = tokio::try_join!(
            HccOrchestrator::new(node_1_cfg, "ignored".into(), 1),
            HccOrchestrator::new(node_0_cfg, "prompt".into(), 2048)
        )
        .unwrap();
        let (target_result, draft_result) = tokio::join!(node_1.run(), node_0.run());

        assert!(target_result
            .unwrap_err()
            .to_string()
            .contains("rejected invalid session"));
        assert!(draft_result
            .unwrap_err()
            .to_string()
            .contains("target rejected session startup"));
    }
}
