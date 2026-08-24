/// Performance telemetry and metrics collection.
///
/// Exposes HCC runtime metrics via Prometheus endpoint (§11 validation).
/// Initialize metrics recording.
pub fn init_metrics() {
    metrics::gauge!("hcc_memory_total_gb").set(256.0);
    metrics::gauge!("hcc_nodes").set(2.0);
}

/// Record a speculative decoding step.
pub fn record_speculative_step(accepted: usize, draft_len: usize, elapsed_us: f64) {
    metrics::counter!("hcc_tokens_accepted_total").increment(accepted as u64);
    metrics::counter!("hcc_drafts_generated_total").increment(draft_len as u64);
    if elapsed_us.is_finite() && elapsed_us >= 0.0 {
        metrics::histogram!("hcc_step_duration_us").record(elapsed_us);
    }
    let rate = accepted as f64 / draft_len.max(1) as f64;
    if rate.is_finite() {
        metrics::histogram!("hcc_acceptance_rate").record(rate);
    }
}

/// Record observed per-step acceptance (verified tokens vs drafts offered).
pub fn record_acceptance(accepted: usize, drafted: usize) {
    let rate = accepted as f64 / drafted.max(1) as f64;
    if rate.is_finite() && rate >= 0.0 {
        metrics::histogram!("hcc_acceptance_rate").record(rate);
        metrics::counter!("hcc_tokens_accepted_total").increment(accepted as u64);
    }
}

/// Record USB4 transport metrics.
pub fn record_transport(bytes: u64, rtt_us: f64) {
    metrics::counter!("hcc_transport_bytes_total").increment(bytes);
    if rtt_us.is_finite() && rtt_us >= 0.0 {
        metrics::histogram!("hcc_transport_rtt_us").record(rtt_us);
    }
}

/// Record TTFT for validation (Hypothesis H1).
pub fn record_ttft(context_len: usize, ttft_ms: f64) {
    if ttft_ms.is_finite() && ttft_ms >= 0.0 {
        metrics::histogram!("hcc_ttft_ms").record(ttft_ms);
    }
    metrics::gauge!("hcc_context_len").set(context_len as f64);
}

/// Record KV cache metrics.
pub fn record_kv_cache(sessions: usize, cache_gb: f64) {
    if cache_gb.is_finite() && cache_gb >= 0.0 {
        metrics::gauge!("hcc_kv_cache_gb").set(cache_gb);
    }
    metrics::gauge!("hcc_active_sessions").set(sessions as f64);
}

/// Aggregate decode throughput in tok/s (Hypothesis H2 validation target: ≥26.1 tok/s).
pub fn record_decode_throughput(tokens_per_sec: f64) {
    if tokens_per_sec.is_finite() && tokens_per_sec >= 0.0 {
        metrics::gauge!("hcc_decode_tps").set(tokens_per_sec);
    }
}

/// Record cost-per-token metrics for TCO tracking.
pub fn record_cost_per_token(cents_per_million: f64) {
    if cents_per_million.is_finite() && cents_per_million >= 0.0 {
        metrics::gauge!("hcc_cost_per_mtok").set(cents_per_million);
    }
}
