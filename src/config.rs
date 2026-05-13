use serde::{Deserialize, Serialize};

/// Complete HCC deployment configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HccConfig {
    pub cluster: ClusterConfig,
    pub model: ModelConfig,
    pub speculative: SpeculativeConfig,
    pub interconnect: InterconnectConfig,
    pub kv_cache: KvCacheConfig,
    pub session: SessionConfig,
    pub backend: BackendConfig,
    pub dovetail: DovetailConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Total nodes in this cluster (paper: 2).
    pub node_count: usize,
    /// This node's index {0, 1}.
    pub node_id: usize,
    /// GB of unified memory per node (paper: 128).
    pub memory_per_node_gb: f64,
    /// LPDDR5x sustained bandwidth (paper: 212 GB/s).
    pub memory_bw_gbs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub checkpoint_path: String,
    /// Hidden dimension H (paper: 6144).
    pub hidden_size: usize,
    /// Number of transformer layers (paper: 78).
    pub num_layers: usize,
    /// Number of routed experts (paper: 128 after REAP).
    pub num_experts: usize,
    /// Active experts per token K (paper: 8).
    pub top_k: usize,
    /// Total active parameters in billions (paper: 40B).
    pub active_params_b: f64,
    /// Bytes per weight (paper: 0.48 for UD-Q3 K M).
    pub bytes_per_weight: f64,
    /// MLA KV dimension d_kv (paper: 576 = kv_lora_rank 512 + qk_rope_head_dim 64).
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Inference backend: "llamacpp-rpc", "migraphx", or "simulated".
    pub inference_engine: String,
    /// llama.cpp server binary.
    #[serde(default = "default_server_bin")]
    pub server_bin: String,
    /// Start the server as a child process. Set false to attach to an existing server.
    #[serde(default = "default_true")]
    pub spawn_server: bool,
    /// llama.cpp rpc-server port.
    pub rpc_port: u16,
    /// Target model path for rpc-server.
    pub model_path: String,
    /// OpenAI/server alias advertised by llama.cpp.
    #[serde(default = "default_model_alias")]
    pub model_alias: String,
    /// HIP device ID (0 = iGPU on Strix Halo).
    pub hip_device: usize,
    /// Explicit llama.cpp device selector. For the local Vulkan build: "Vulkan0".
    #[serde(default = "default_device")]
    pub device: String,
    /// Enable hipBLASLt (critical for Strix Halo perf: 36.9 vs 5.1 TFLOPS).
    pub hipblaslt: bool,
    /// ROCm version string for compatibility checks.
    pub rocm_version: String,
    /// Pipeline topology: "hcc" (NPU→GPU) or "dovetail" (GPU→CPU).
    pub pipeline: String,
    /// llama.cpp context size.
    #[serde(default = "default_ctx_size")]
    pub ctx_size: usize,
    /// Logical prompt batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Physical micro-batch size.
    #[serde(default = "default_ubatch_size")]
    pub ubatch_size: usize,
    /// Server slots.
    #[serde(default = "default_parallel")]
    pub parallel: usize,
    /// CPU threads for generation. -1 lets llama.cpp choose.
    #[serde(default = "default_threads")]
    pub threads: isize,
    /// CPU threads for prompt processing. -1 lets llama.cpp choose.
    #[serde(default = "default_threads")]
    pub threads_batch: isize,
    /// llama.cpp polling level.
    #[serde(default = "default_poll")]
    pub poll: usize,
    /// llama.cpp Flash Attention mode: "on", "off", or "auto".
    #[serde(default = "default_flash_attn")]
    pub flash_attn: String,
    /// KV cache data type for keys.
    #[serde(default = "default_cache_type_k")]
    pub cache_type_k: String,
    /// KV cache data type for values.
    #[serde(default = "default_cache_type_v")]
    pub cache_type_v: String,
    /// Number of model layers to place on the accelerator ("all", "auto", or integer).
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: String,
    /// Let llama.cpp fit context/layers to device memory.
    #[serde(default = "default_true")]
    pub fit: bool,
    /// Target free-memory margin for llama.cpp --fit.
    #[serde(default = "default_fit_target_mib")]
    pub fit_target_mib: usize,
    /// Minimum context considered by llama.cpp --fit.
    #[serde(default = "default_fit_ctx")]
    pub fit_ctx: usize,
    /// Enable prompt cache on llama-server.
    #[serde(default = "default_true")]
    pub cache_prompt: bool,
    /// Minimum chunk size for cache reuse.
    #[serde(default = "default_cache_reuse")]
    pub cache_reuse: usize,
    /// llama.cpp no-webui mode for inference-only server use.
    #[serde(default = "default_true")]
    pub no_webui: bool,
    /// Enable Prometheus metrics endpoint.
    #[serde(default)]
    pub metrics: bool,
    /// Qwen thinking mode: "on", "off", or "auto".
    #[serde(default = "default_reasoning")]
    pub reasoning: String,
    /// llama.cpp no-draft speculative mode, e.g. "none" or "ngram-cache".
    #[serde(default = "default_spec_type")]
    pub spec_type: String,
    /// Seconds to wait for llama-server to finish loading.
    #[serde(default = "default_startup_timeout_s")]
    pub startup_timeout_s: u64,
    /// Additional llama.cpp flags appended last.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_server_bin() -> String { "llama-server".into() }
fn default_model_alias() -> String { "hcc-local".into() }
fn default_device() -> String { "Vulkan0".into() }
fn default_ctx_size() -> usize { 16_384 }
fn default_batch_size() -> usize { 2048 }
fn default_ubatch_size() -> usize { 512 }
fn default_parallel() -> usize { 1 }
fn default_threads() -> isize { -1 }
fn default_poll() -> usize { 50 }
fn default_flash_attn() -> String { "on".into() }
fn default_cache_type_k() -> String { "q8_0".into() }
fn default_cache_type_v() -> String { "q4_0".into() }
fn default_gpu_layers() -> String { "all".into() }
fn default_fit_target_mib() -> usize { 4096 }
fn default_fit_ctx() -> usize { 4096 }
fn default_cache_reuse() -> usize { 256 }
fn default_reasoning() -> String { "off".into() }
fn default_spec_type() -> String { "none".into() }
fn default_startup_timeout_s() -> u64 { 240 }
fn default_true() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DovetailConfig {
    /// Enable Dovetail pipeline (alternative to HCC).
    pub enabled: bool,
    /// Draft model depth (Dovetail finding: deeper = better).
    pub draft_depth: usize,
    /// Dynamic Gating Fusion initial alpha.
    pub dgf_alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft length γ (paper: 5).
    pub draft_len: usize,
    /// Target acceptance rate α (paper: 0.7 with aligned draft).
    pub acceptance_rate: f64,
    /// Draft cost ratio c/C (paper: 0.05).
    pub draft_cost_ratio: f64,
    /// Draft model size in params (paper: 8B).
    pub draft_params_b: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterconnectConfig {
    /// USB4 link count (paper: 2 bonded).
    pub link_count: usize,
    /// Per-direction throughput Gbps (paper: 22.5 single, 45 dual).
    pub throughput_gbps: f64,
    /// Measured RTT µs (paper: 17 tuned).
    pub rtt_us: f64,
    /// Base protocol latency µs.
    pub base_latency_us: f64,
    /// MTU bytes (paper: 9000 jumbo).
    pub mtu: usize,
    /// TCP overhead per-packet µs.
    pub tcp_overhead_us: f64,
    /// Enable kernel-level optimizations (busy-poll, BBR, ASPM off).
    pub kernel_tune: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Quantization bits (paper: 3).
    pub bits: usize,
    /// Block size for BFP16 or TurboQuant.
    pub block_size: usize,
    /// Enable Walsh-Hadamard rotation (QuaRot-style).
    pub hadamard_rotate: bool,
    /// Enable QJL residual correction.
    pub qjl_correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Max concurrent sessions.
    pub max_sessions: usize,
    /// Max context length per session.
    pub max_context: usize,
    /// Enable static batching via --parallel N.
    pub static_batch: usize,
}

impl Default for HccConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterConfig {
                node_count: 1,
                node_id: 0,
                // Measured: Strix Halo 128 GB LPDDR5x-8000 systems
                memory_per_node_gb: 128.0,
                // Measured: rocm_bandwidth_test on gfx1151 = 212 GB/s (kyuz0/lhl Mar 2026)
                memory_bw_gbs: 212.0,
            },
            model: ModelConfig {
                model_name: "GLM-5.1-REAP-50".into(),
                checkpoint_path: "/models/glm-5.1-reap-50-udq3km".into(),
                hidden_size: 6144,
                num_layers: 78,
                num_experts: 128,
                top_k: 8,
                active_params_b: 40.0,
                bytes_per_weight: 0.48,
                // weight_read_gb removed: computed as active_params_b * bytes_per_weight
                kv_lora_rank: 512,
                qk_rope_head_dim: 64,
            },
            speculative: SpeculativeConfig {
                draft_len: 5,
                // Target: α ≥ 0.7 with aligned draft (EAGLE/Medusa literature)
                acceptance_rate: 0.7,
                draft_cost_ratio: 0.05,
                draft_params_b: 8.0,
            },
            interconnect: InterconnectConfig {
                link_count: 2,
                // Measured: dual USB4 bonded = 45 Gbps aggregate (paper §5.1)
                throughput_gbps: 45.0,
                // Measured: tuned USB4 P2P RTT = 17 µs avg (paper §5.1.2)
                rtt_us: 17.0,
                base_latency_us: 14.0,
                mtu: 9000,
                tcp_overhead_us: 1.2,
                kernel_tune: true,
            },
            kv_cache: KvCacheConfig {
                bits: 3,
                block_size: 128,
                hadamard_rotate: true,
                qjl_correct: true,
            },
            session: SessionConfig {
                max_sessions: 9,
                max_context: 200_000,
                static_batch: 1,
            },
            backend: BackendConfig {
                inference_engine: "llamacpp-rpc".into(),
                server_bin: default_server_bin(),
                spawn_server: true,
                rpc_port: 50052,
                model_path: "/models/glm-5.1.gguf".into(),
                model_alias: default_model_alias(),
                hip_device: 0,
                device: default_device(),
                hipblaslt: true,
                rocm_version: "7.2.3".into(),
                pipeline: "hcc".into(),
                ctx_size: default_ctx_size(),
                batch_size: default_batch_size(),
                ubatch_size: default_ubatch_size(),
                parallel: default_parallel(),
                threads: default_threads(),
                threads_batch: default_threads(),
                poll: default_poll(),
                flash_attn: default_flash_attn(),
                cache_type_k: default_cache_type_k(),
                cache_type_v: default_cache_type_v(),
                gpu_layers: default_gpu_layers(),
                fit: true,
                fit_target_mib: default_fit_target_mib(),
                fit_ctx: default_fit_ctx(),
                cache_prompt: true,
                cache_reuse: default_cache_reuse(),
                no_webui: true,
                metrics: false,
                reasoning: default_reasoning(),
                spec_type: default_spec_type(),
                startup_timeout_s: default_startup_timeout_s(),
                extra_args: Vec::new(),
            },
            dovetail: DovetailConfig {
                enabled: false,
                draft_depth: 4,
                dgf_alpha: 0.3,
            },
        }
    }
}

impl ModelConfig {
    /// Compute weight read per token in GB.
    /// Paper §6.1: 40B active params × 0.48 bytes/weight = 19.1 GB
    pub fn weight_read_gb(&self) -> f64 {
        self.active_params_b * self.bytes_per_weight
    }
}

impl HccConfig {
    /// Validate all configuration parameters. Panics with a clear message on invalid values.
    pub fn validate(&self) {
        assert!(self.cluster.node_count >= 1, "need at least 1 node");
        assert!(
            self.cluster.node_id < self.cluster.node_count,
            "node_id {} out of range [0, {})",
            self.cluster.node_id,
            self.cluster.node_count
        );
        assert!(self.cluster.memory_per_node_gb > 0.0, "memory_per_node_gb must be > 0");
        assert!(self.cluster.memory_bw_gbs > 0.0, "memory_bw_gbs must be > 0");
        assert!(self.model.hidden_size > 0, "hidden_size must be > 0");
        assert!(self.model.num_layers > 0, "num_layers must be > 0");
        assert!(
            self.model.active_params_b > 0.0 && self.model.bytes_per_weight > 0.0,
            "active_params_b and bytes_per_weight must be > 0"
        );
        assert!(
            self.speculative.draft_len > 0 && self.speculative.draft_len <= 20,
            "draft_len γ must be in (0, 20] (paper recommends 5)"
        );
        assert!(
            (0.0..1.0).contains(&self.speculative.acceptance_rate),
            "acceptance_rate α must be in [0, 1) (paper targets 0.7)"
        );
        assert!(
            self.speculative.draft_cost_ratio > 0.0 && self.speculative.draft_cost_ratio < 1.0,
            "draft_cost_ratio c/C must be in (0, 1) (paper: 0.05)"
        );
        assert!(
            self.interconnect.throughput_gbps > 0.0,
            "interconnect throughput must be > 0"
        );
        assert!(
            (3..=4).contains(&self.kv_cache.bits),
            "kv_cache bits must be 3 or 4 (paper + mixed-precision: K 8, V 3)"
        );
        assert!(self.session.max_sessions >= 1, "max_sessions must be ≥ 1");
        assert!(self.session.max_context >= 1024, "max_context must be ≥ 1K");
        assert!(
            self.backend.inference_engine == "llamacpp-rpc"
                || self.backend.inference_engine == "migraphx"
                || self.backend.inference_engine == "simulated",
            "inference_engine must be llamacpp-rpc, migraphx, or simulated"
        );
        assert!(
            matches!(self.backend.flash_attn.as_str(), "on" | "off" | "auto"),
            "backend.flash_attn must be on, off, or auto"
        );
        assert!(
            matches!(self.backend.reasoning.as_str(), "on" | "off" | "auto"),
            "backend.reasoning must be on, off, or auto"
        );
        assert!(
            matches!(
                self.backend.cache_type_k.as_str(),
                "f32" | "f16" | "bf16" | "q8_0" | "q4_0" | "q4_1" | "iq4_nl" | "q5_0" | "q5_1"
            ),
            "backend.cache_type_k is not supported by this llama.cpp build"
        );
        assert!(
            matches!(
                self.backend.cache_type_v.as_str(),
                "f32" | "f16" | "bf16" | "q8_0" | "q4_0" | "q4_1" | "iq4_nl" | "q5_0" | "q5_1"
            ),
            "backend.cache_type_v is not supported by this llama.cpp build"
        );
        assert!(
            matches!(
                self.backend.spec_type.as_str(),
                "none" | "ngram-cache" | "ngram-simple" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod"
            ),
            "backend.spec_type is not supported by this llama.cpp build"
        );
        assert!(self.backend.ctx_size >= 1024, "backend.ctx_size must be >= 1024");
        assert!(self.backend.batch_size >= 1, "backend.batch_size must be >= 1");
        assert!(self.backend.ubatch_size >= 1, "backend.ubatch_size must be >= 1");
        assert!(self.backend.parallel >= 1, "backend.parallel must be >= 1");
        // Measured roofline: 212 GB/s / 19.1 GB ≈ 11.1 T/s per node
        let theoretical_tps = self.cluster.memory_bw_gbs / self.model.weight_read_gb();
        assert!(
            theoretical_tps > 5.0,
            "theoretical decode TPS too low: {theoretical_tps:.1}. Check memory_bw_gbs and active_params_b"
        );
    }
}

pub fn load(path: &str) -> anyhow::Result<HccConfig> {
    let contents = std::fs::read_to_string(path)?;
    let cfg: HccConfig = toml::from_str(&contents)?;
    cfg.validate();
    Ok(cfg)
}
