use crate::config::ModelConfig;

/// Concurrent session multiplexing manager.
///
/// From §10.3:
///   "GLM-5.1's MLA compresses the KV cache to just ~9 GB per session
///    at 200K context. With ~86 GB of free memory across the dual-node
///    cluster, this enables up to 9 concurrent sessions."
///
///   "At shorter contexts (32K), the KV cache drops to ~1.4 GB per session,
///    supporting >50 concurrent sessions."
pub struct SessionManager {
    max_sessions: usize,
    max_context: usize,
    memory_per_node_gb: f64,
    model_cfg: ModelConfig,
    sessions: Vec<Session>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub id: u64,
    pub state: SessionState,
    pub context_len: usize,
    pub tokens_generated: usize,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    Pending,
    Active,
    Completed,
}

impl SessionManager {
    pub fn new(
        max_sessions: usize,
        max_context: usize,
        memory_per_node_gb: f64,
        model_cfg: &ModelConfig,
    ) -> Self {
        Self {
            max_sessions,
            max_context,
            memory_per_node_gb,
            model_cfg: model_cfg.clone(),
            sessions: Vec::new(),
        }
    }

    /// Create a new session.
    pub fn create_session(&mut self, id: u64, max_tokens: usize) -> anyhow::Result<()> {
        if self.sessions.len() >= self.max_sessions {
            anyhow::bail!("max sessions ({}) reached", self.max_sessions);
        }
        self.sessions.push(Session {
            id,
            state: SessionState::Pending,
            context_len: 0,
            tokens_generated: 0,
            max_tokens,
        });
        Ok(())
    }

    /// Check if there are pending sessions.
    pub fn has_pending(&self) -> bool {
        self.sessions.iter().any(|s| s.state == SessionState::Pending)
    }

    /// Check if any session is active.
    pub fn has_active(&self) -> bool {
        self.sessions.iter().any(|s| s.state == SessionState::Active)
    }

    /// Check if all sessions completed.
    pub fn all_completed(&self) -> bool {
        self.sessions.is_empty() || self.sessions.iter().all(|s| s.state == SessionState::Completed)
    }

    /// Get next pending context.
    pub async fn next_context(&mut self) -> Vec<u8> {
        for session in &mut self.sessions {
            if session.state == SessionState::Pending {
                session.state = SessionState::Active;
                return vec![0u8; 1024]; // simulated context
            }
        }
        vec![]
    }

    /// Update session token count.
    pub fn advance(&mut self, session_id: u64, tokens: usize) {
        if let Some(session) = self.sessions.iter_mut().find(|s| s.id == session_id) {
            session.tokens_generated += tokens;
            session.context_len += tokens;
            if session.tokens_generated >= session.max_tokens {
                session.state = SessionState::Completed;
            }
        }
    }

    /// KV cache memory per session in GB (from §10.3).
    pub fn kv_cache_gb_per_session(&self, context_len: usize) -> f64 {
        let kv_per_token = (self.model_cfg.kv_lora_rank + self.model_cfg.qk_rope_head_dim) as f64;
        let layers_per_node = self.model_cfg.num_layers as f64 / 2.0; // 39 per node
        let bytes_per_value = 2.0; // FP16
        context_len as f64 * layers_per_node * kv_per_token * bytes_per_value / 1e9
    }

    /// Maximum concurrent sessions at given context length.
    ///
    /// From §10.3:
    ///   "With ~86 GB of free memory...this enables up to 9 concurrent sessions"
    pub fn max_sessions_at_context(&self, context_len: usize) -> usize {
        let kv_per_session_gb = self.kv_cache_gb_per_session(context_len);
        if kv_per_session_gb <= 0.0 {
            return self.max_sessions;
        }
        let free_gb = self.memory_per_node_gb * 2.0 * 0.336; // ~86 GB free after weights+OS
        (free_gb / kv_per_session_gb) as usize
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HccConfig;

    #[test]
    fn test_session_creation() {
        let cfg = HccConfig::default();
        let mut mgr = SessionManager::new(
            9, 200_000, 128.0, &cfg.model,
        );
        assert!(mgr.create_session(1, 1000).is_ok());
        assert_eq!(mgr.session_count(), 1);
        assert!(mgr.has_pending());
    }

    #[test]
    fn test_kv_cache_calculation() {
        let cfg = HccConfig::default();
        let mgr = SessionManager::new(9, 200_000, 128.0, &cfg.model);
        let gb = mgr.kv_cache_gb_per_session(200_000);
        // ~9 GB per session at 200K
        assert!((gb - 9.0).abs() < 1.0, "kv_cache_gb={gb}");
    }

    #[test]
    fn test_max_sessions_200k() {
        let cfg = HccConfig::default();
        let mgr = SessionManager::new(9, 200_000, 128.0, &cfg.model);
        let n = mgr.max_sessions_at_context(200_000);
        assert_eq!(n, 9);
    }
}
