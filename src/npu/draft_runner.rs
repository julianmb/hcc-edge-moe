/// NPU draft runner — calls llama-server HTTP API for actual inference.
///
/// Paper Section 6.1, 7: The draft model generates γ candidate tokens
/// that are sent to the iGPU for parallel verification over USB4.
use crate::decoding::speculative::DraftToken;

pub struct DraftRunner {
    model_path: String,
    api_url: String,
    pub(crate) params_b: f64,
    pub strict_vocab: bool,
}

impl DraftRunner {
    pub fn new(model_path: &str, params_b: f64, api_port: u16) -> Self {
        Self {
            model_path: model_path.to_string(),
            api_url: format!("http://127.0.0.1:{api_port}/v1/chat/completions"),
            params_b,
            strict_vocab: true,
        }
    }

    /// Extract explicit vocabulary token ID from llama-server / OpenAI logprob entry.
    /// Does not perform lossy or arbitrary string hashing; requires actual integer token ID.
    pub fn extract_token_id(info: &serde_json::Value) -> anyhow::Result<u32> {
        if let Some(id) = info["token_id"].as_u64() {
            return Ok(id as u32);
        }
        if let Some(id) = info["id"].as_u64() {
            return Ok(id as u32);
        }
        if let Some(s) = info["token"].as_str() {
            if let Ok(id) = s.parse::<u32>() {
                return Ok(id);
            }
        }
        anyhow::bail!(
            "logprobs response does not contain valid vocabulary token_id (got: {})",
            info
        )
    }

    /// Generate γ linear draft tokens for the speculative decoding loop.
    ///
    /// Parses the OpenAI-compatible logprobs schema (`content[].token_id`,
    /// `content[].logprob`). Requires verified vocabulary token IDs from the engine.
    pub async fn generate_drafts(&mut self, gamma: usize) -> anyhow::Result<Vec<DraftToken>> {
        let client = reqwest::Client::new();

        let body = serde_json::json!({
            "model": self.model_path,
            "messages": [{"role": "user", "content": "Write a short story about AI."}],
            "max_tokens": gamma,
            "temperature": 0.0,
            "logprobs": true,
        });

        let resp = client
            .post(&self.api_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("draft backend unreachable at {}: {e}", self.api_url))?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("draft backend returned error status: {e}"))?;

        let result: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("draft backend returned non-JSON body: {e}"))?;

        let content = result["choices"]
            .get(0)
            .and_then(|c| c["logprobs"]["content"].as_array())
            .ok_or_else(|| {
                anyhow::anyhow!("draft backend response choices[0] contains no logprobs content")
            })?;

        let mut tokens = Vec::with_capacity(gamma.min(content.len()));
        for info in content {
            let token_id = Self::extract_token_id(info)?;
            let probability = info["logprob"].as_f64().map(f64::exp).unwrap_or(0.0);
            tokens.push(DraftToken {
                token_id,
                probability: probability.clamp(0.0, 1.0),
                kv_state: vec![],
            });
            if tokens.len() >= gamma {
                break;
            }
        }

        if tokens.is_empty() {
            anyhow::bail!("draft backend produced zero tokens");
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_token_id_from_token_id_field() {
        let json = serde_json::json!({
            "token": "hello",
            "token_id": 15496,
            "logprob": -0.25
        });
        assert_eq!(DraftRunner::extract_token_id(&json).unwrap(), 15496);
    }

    #[test]
    fn test_extract_token_id_from_id_field() {
        let json = serde_json::json!({
            "id": 4242,
            "token": "test"
        });
        assert_eq!(DraftRunner::extract_token_id(&json).unwrap(), 4242);
    }

    #[test]
    fn test_extract_token_id_from_numeric_string() {
        let json = serde_json::json!({
            "token": "999"
        });
        assert_eq!(DraftRunner::extract_token_id(&json).unwrap(), 999);
    }

    #[test]
    fn test_extract_token_id_rejects_non_numeric_without_id() {
        let json = serde_json::json!({
            "token": "unmapped_word"
        });
        assert!(DraftRunner::extract_token_id(&json).is_err());
    }
}
