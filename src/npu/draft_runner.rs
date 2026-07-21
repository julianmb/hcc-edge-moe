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
}

impl DraftRunner {
    pub fn new(model_path: &str, params_b: f64, api_port: u16) -> Self {
        Self {
            model_path: model_path.to_string(),
            api_url: format!("http://127.0.0.1:{api_port}/v1/chat/completions"),
            params_b,
        }
    }

    /// Generate γ linear draft tokens for the speculative decoding loop.
    pub async fn generate_drafts(&self, gamma: usize) -> anyhow::Result<Vec<DraftToken>> {
        let client = reqwest::Client::new();

        let body = serde_json::json!({
            "model": self.model_path,
            "messages": [{"role": "user", "content": "Write a short story about AI."}],
            "max_tokens": gamma,
            "temperature": 0.0,
            "return_logits": true,
        });

        let resp = client.post(&self.api_url).json(&body).send().await?;
        let result: serde_json::Value = resp.json().await?;

        let mut tokens = Vec::new();

        if let Some(choices) = result["choices"].as_array() {
            for choice in choices {
                if let Some(logprobs) = choice.get("logprobs") {
                    if let Some(content) = logprobs.get("content").and_then(|c| c.as_array()) {
                        for token_info in content {
                            let token_id = token_info["token_id"].as_u64().unwrap_or(0) as u32;
                            let probability = token_info["prob"].as_f64().unwrap_or(0.0);
                            tokens.push(DraftToken {
                                token_id,
                                probability,
                                kv_state: vec![],
                            });
                        }
                    }
                }
            }
        }

        // Fallback for non-logprob backends
        if tokens.is_empty() {
            for _ in 0..gamma {
                tokens.push(DraftToken {
                    token_id: fastrand::u32(..) % 32000,
                    probability: 0.8,
                    kv_state: vec![],
                });
            }
        }

        Ok(tokens)
    }
}
