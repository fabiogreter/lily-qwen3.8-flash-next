//! Checkpoint-config types shared by the loader and the per-architecture
//! config parsers (`src/qwen4exp/config.rs` re-exports them).

use serde::Deserialize;

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerType {
    #[serde(rename = "linear_attention")]
    LinearAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

/// The `quantization` block an mlx-converted checkpoint carries at the top
/// level of `config.json`. Only the affine mode (`w = scales*q + biases`,
/// codes packed low-nibble-first into u32 along the input dim) is supported.
#[derive(Deserialize, Clone, Copy, Debug)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
}

/// `eos_token_id` appears as a scalar in some checkpoints and a list in
/// others (Qwen chat models stop on two ids).
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum EosTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl EosTokenIds {
    pub fn as_vec(&self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![*id],
            Self::Many(ids) => ids.clone(),
        }
    }
}
