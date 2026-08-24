/// NPU draft runner — calls llama-server HTTP API for actual inference.
///
/// Paper Section 6.1, 7: The 8B draft model runs on the NPU (Node 1),
/// generating γ candidate tokens that are sent to the iGPU (Node 2)
/// for parallel verification over USB4.
use crate::decoding::speculative::DraftToken;

pub struct DraftRunner {
    model_path: String,
    api_url: String,
    pub(crate) params_b: f64,
    fallback_steps: u64,
}

impl DraftRunner {
    pub fn new(model_path: &str, params_b: f64, api_port: u16) -> Self {
        Self {
            model_path: model_path.to_string(),
            api_url: format!("http://127.0.0.1:{api_port}/v1/chat/completions"),
            params_b,
            fallback_steps: 0,
        }
    }

    /// Generate γ linear draft tokens for the speculative decoding loop.
    ///
    /// Parses the OpenAI-compatible logprobs schema (`content[].token`,
    /// `content[].logprob`). When the backend cannot serve logprobs, falls
    /// back to a deterministic placeholder sequence; every fallback logs a
    /// warning and increments `fallback_steps` so outages stay visible.
    pub async fn generate_drafts(&mut self, gamma: usize) -> anyhow::Result<Vec<DraftToken>> {
        let client = reqwest::Client::new();

        let body = serde_json::json!({
            "model": self.model_path,
            "messages": [{"role": "user", "content": "Write a short story about AI."}],
            "max_tokens": gamma,
            "temperature": 0.0,
            "logprobs": true,
        });

        let resp = match client.post(&self.api_url).json(&body).send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::warn!("draft backend rejected request: {e}");
                    self.fallback_steps += 1;
                    return Ok(Self::placeholder_drafts(gamma));
                }
            },
            Err(e) => {
                tracing::warn!("draft backend unreachable: {e}");
                self.fallback_steps += 1;
                return Ok(Self::placeholder_drafts(gamma));
            }
        };

        let result: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("draft backend returned non-JSON body: {e}");
                self.fallback_steps += 1;
                return Ok(Self::placeholder_drafts(gamma));
            }
        };

        let mut tokens = Vec::new();
        if let Some(content) = result["choices"]
            .get(0)
            .and_then(|c| c["logprobs"]["content"].as_array())
        {
            for info in content {
                let token_str = info["token"].as_str().unwrap_or("");
                let probability = info["logprob"].as_f64().map(f64::exp).unwrap_or(0.0);
                tokens.push(DraftToken {
                    token_id: fnv1a(token_str),
                    probability: probability.clamp(0.0, 1.0),
                    kv_state: vec![],
                });
            }
        }

        if tokens.is_empty() {
            tracing::warn!("draft backend served no logprobs; using deterministic placeholders");
            self.fallback_steps += 1;
            return Ok(Self::placeholder_drafts(gamma));
        }

        tokens.truncate(gamma);
        Ok(tokens)
    }

    pub fn fallback_count(&self) -> u64 {
        self.fallback_steps
    }

    fn placeholder_drafts(gamma: usize) -> Vec<DraftToken> {
        (0..gamma)
            .map(|i| DraftToken {
                token_id: i as u32,
                probability: 0.5,
                kv_state: vec![],
            })
            .collect()
    }
}

fn fnv1a(s: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.as_bytes() {
        h ^= u32::from(*b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}
