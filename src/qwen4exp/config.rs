//! HF `config.json` parsing for Qwen3.8-Flash-Next, Qwen's `qwen4_exp` preview
//! architecture, as written by `tools/convert/convert_qwen38_flash_next.py`
//! (see `docs/qwen38-flash-next-checkpoint-format.md`). The file is a
//! multimodal wrapper; lily reads `text_config` plus the converter's `lily`
//! block and ignores the vision tower.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;

pub use crate::config::{EosTokenIds, LayerType, QuantizationConfig};

/// Activation applied to the Gated DeltaNet output gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateAct {
    Silu,
    Sigmoid,
}

#[derive(Deserialize, Debug)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

/// Qwen Sparse Attention indexer: `n_heads` query heads score one shared key
/// head per compressed block; the `budget / compress_ratio` best blocks plus
/// the incomplete tail block are attended.
#[derive(Clone, Copy, Debug)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl IndexerConfig {
    /// Blocks kept per query.
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// Longest context whose attention is provably dense: every complete block
    /// fits the budget, so the selection is the whole causal window.
    pub fn dense_limit(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }
}

/// Per-Layer (n-gram) Embedding hyper-parameters and the hashing constants the
/// converter copied out of the checkpoint.
#[derive(Clone, Debug)]
pub struct PleConfig {
    /// Zero-based decoder layer carrying the PLE module.
    pub layer: usize,
    pub embed_dim: usize,
    pub conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    /// Segment separator for the n-gram context (the text config's eos id).
    pub eos_token_id: u32,
    /// Odd 63-bit multipliers, one per n-gram position.
    pub layer_multipliers: Vec<u64>,
    /// Prime table size per hashed head.
    pub head_vocab_sizes: Vec<u64>,
    /// Row offset of each head's slice of the shared table.
    pub head_offsets: Vec<u64>,
    /// Row count of the concatenated table (padded).
    pub padded_vocab_size: usize,
    /// Checkpoint shards the table is split into along its rows.
    pub table_shards: usize,
    pub quantization: QuantizationConfig,
}

impl PleConfig {
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.ngram_heads()
    }

    /// Tokens of context the dilated conv keeps: `(K - 1) * dilation`.
    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.ngram_size
    }
}

#[derive(Deserialize, Debug)]
struct TextConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    layer_types: Vec<LayerType>,
    vocab_size: usize,
    rms_norm_eps: f32,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    eos_token_id: EosTokenIds,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_parameters: RopeParameters,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    output_gate_type: Option<String>,
    hidden_act: String,
    hc_count: usize,
    hc_lowrank: usize,
    #[serde(default)]
    ple_layer_ids: Vec<usize>,
    #[serde(default)]
    ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv")]
    ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    heads_per_ngram: usize,
    #[serde(default = "default_split_ngram_parts")]
    split_ngram_parts: usize,
    indexer_n_heads: usize,
    indexer_kv_heads: usize,
    indexer_head_dim: usize,
    indexer_budget: usize,
    indexer_compress_ratio: usize,
}

fn default_true() -> bool {
    true
}
fn default_ple_conv() -> usize {
    4
}
fn default_ngram_size() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_split_ngram_parts() -> usize {
    512
}

#[derive(Deserialize)]
struct QuantJson {
    bits: usize,
    group_size: usize,
}

#[derive(Deserialize)]
struct QuantBlockJson {
    default: QuantJson,
    ngram_embedding: QuantJson,
}

#[derive(Deserialize, Default)]
struct PleJson {
    #[serde(default)]
    layer_multipliers: Vec<u64>,
    #[serde(default)]
    ngram_heads_vocab_sizes: Vec<u64>,
    #[serde(default)]
    ngram_heads_offsets: Vec<u64>,
    #[serde(default)]
    padded_vocab_size: usize,
}

#[derive(Deserialize)]
struct LilyJson {
    format: String,
    quantization: QuantBlockJson,
    #[serde(default)]
    ple: PleJson,
}

#[derive(Deserialize)]
struct WrapperJson {
    model_type: String,
    text_config: TextConfigJson,
    lily: LilyJson,
}

pub const FORMAT: &str = "qwen4_exp-affine-v1";

/// The parsed, validated Qwen3.8-Flash-Next text model configuration.
#[derive(Debug)]
pub struct Qwen4ExpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: EosTokenIds,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_parameters: RopeParameters,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub gdn_gate: GateAct,
    /// Residual streams of the hyper-connections (gated residual).
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub ple: Option<PleConfig>,
    pub indexer: IndexerConfig,
    /// Default affine quantization of the linear projections.
    pub quantization: QuantizationConfig,
}

impl Qwen4ExpConfig {
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let wrapper: WrapperJson =
            serde_json::from_slice(&bytes).context("parsing config.json")?;
        ensure!(
            wrapper.model_type == "qwen4_exp",
            "unsupported model_type {:?}; this loader is for Qwen3.8-Flash-Next (qwen4_exp)",
            wrapper.model_type
        );
        ensure!(
            wrapper.lily.format == FORMAT,
            "checkpoint format {:?} != {FORMAT}; re-run tools/convert",
            wrapper.lily.format
        );
        let t = wrapper.text_config;
        ensure!(
            t.layer_types.len() == t.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            t.layer_types.len(),
            t.num_hidden_layers
        );
        ensure!(!t.tie_word_embeddings, "tied embeddings are not supported");
        ensure!(
            t.indexer_kv_heads == 1,
            "QSA requires one indexer key head, got {}",
            t.indexer_kv_heads
        );
        let gdn_gate =
            match t.output_gate_type.as_deref().unwrap_or(t.hidden_act.as_str()) {
                "silu" => GateAct::Silu,
                "sigmoid" => GateAct::Sigmoid,
                other => {
                    anyhow::bail!("unsupported GDN output gate activation {other:?}")
                }
            };
        let q = &wrapper.lily.quantization;
        let quantization = QuantizationConfig {
            group_size: q.default.group_size,
            bits: q.default.bits,
        };
        ensure!(
            quantization.bits == 4 && quantization.group_size == 64,
            "unsupported default quantization: {} bits, group {}",
            quantization.bits,
            quantization.group_size
        );

        let ple = match t.ple_layer_ids.as_slice() {
            [] => None,
            [one_indexed] => {
                let layer = one_indexed
                    .checked_sub(1)
                    .context("ple_layer_ids must be one-indexed")?;
                if layer >= t.num_hidden_layers {
                    // Truncated checkpoints may drop the PLE layer entirely.
                    None
                } else {
                    let p = wrapper.lily.ple;
                    let heads = (t.ngram_size - 1) * t.heads_per_ngram;
                    ensure!(
                        p.layer_multipliers.len() == t.ngram_size
                            && p.ngram_heads_vocab_sizes.len() == heads
                            && p.ngram_heads_offsets.len() == heads
                            && p.padded_vocab_size > 0,
                        "config.json lily.ple block is incomplete"
                    );
                    let embed_dim = t.ple_embed_dim.unwrap_or(t.hidden_size);
                    ensure!(
                        embed_dim.is_multiple_of(heads),
                        "ple_embed_dim {embed_dim} not divisible by {heads} heads"
                    );
                    Some(PleConfig {
                        layer,
                        embed_dim,
                        conv_kernel_size: t.ple_conv_kernel_size,
                        ngram_size: t.ngram_size,
                        heads_per_ngram: t.heads_per_ngram,
                        eos_token_id: t.eos_token_id.as_vec()[0],
                        layer_multipliers: p.layer_multipliers,
                        head_vocab_sizes: p.ngram_heads_vocab_sizes,
                        head_offsets: p.ngram_heads_offsets,
                        padded_vocab_size: p.padded_vocab_size,
                        table_shards: t.split_ngram_parts,
                        quantization: QuantizationConfig {
                            group_size: q.ngram_embedding.group_size,
                            bits: q.ngram_embedding.bits,
                        },
                    })
                }
            }
            more => anyhow::bail!("only one PLE layer is supported, got {more:?}"),
        };

        let config = Self {
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            layer_types: t.layer_types,
            vocab_size: t.vocab_size,
            rms_norm_eps: t.rms_norm_eps,
            max_position_embeddings: t.max_position_embeddings,
            eos_token_id: t.eos_token_id,
            num_experts: t.num_experts,
            num_experts_per_tok: t.num_experts_per_tok,
            moe_intermediate_size: t.moe_intermediate_size,
            shared_expert_intermediate_size: t.shared_expert_intermediate_size,
            norm_topk_prob: t.norm_topk_prob,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rope_parameters: t.rope_parameters,
            linear_num_key_heads: t.linear_num_key_heads,
            linear_num_value_heads: t.linear_num_value_heads,
            linear_key_head_dim: t.linear_key_head_dim,
            linear_value_head_dim: t.linear_value_head_dim,
            linear_conv_kernel_dim: t.linear_conv_kernel_dim,
            gdn_gate,
            hc_count: t.hc_count,
            hc_lowrank: t.hc_lowrank,
            ple,
            indexer: IndexerConfig {
                n_heads: t.indexer_n_heads,
                head_dim: t.indexer_head_dim,
                budget: t.indexer_budget,
                compress_ratio: t.indexer_compress_ratio,
            },
            quantization,
        };
        config.validate_flash_next()?;
        Ok(config)
    }

    /// The kernels are written for Qwen3.8-Flash-Next's exact shape; refuse
    /// anything else loudly rather than run subtly wrong.
    fn validate_flash_next(&self) -> Result<()> {
        let layer_pattern = self.layer_types.iter().enumerate().all(|(i, ty)| {
            *ty == if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        });
        ensure!(
            self.hidden_size == 2560
                && (1..=48).contains(&self.num_hidden_layers)
                && self.vocab_size == 248_320
                && self.num_experts == 512
                && self.num_experts_per_tok == 10
                && self.moe_intermediate_size == 640
                && self.shared_expert_intermediate_size == 640
                && self.num_attention_heads == 24
                && self.num_key_value_heads == 2
                && self.head_dim == 256
                && self.linear_num_key_heads == 16
                && self.linear_num_value_heads == 48
                && self.linear_key_head_dim == 128
                && self.linear_value_head_dim == 128
                && self.linear_conv_kernel_dim == 4
                && self.hc_count == 4
                && self.hc_lowrank == 320
                && self.indexer.n_heads == 4
                && self.indexer.head_dim == 128
                && self.indexer.budget == 2048
                && self.indexer.compress_ratio == 4
                && self.rotary_dim() == 64
                && layer_pattern,
            "checkpoint is not the supported Qwen3.8-Flash-Next shape"
        );
        if let Some(ple) = &self.ple {
            ensure!(
                ple.layer == 1
                    && ple.embed_dim == 2560
                    && ple.ngram_heads() == 16
                    && ple.head_dim() == 160
                    && ple.conv_kernel_size == 4
                    && ple.quantization.bits == 4
                    && ple.quantization.group_size == 32,
                "unsupported PLE configuration {ple:?}"
            );
        }
        Ok(())
    }

    /// Rotary dims of each attention head (the rest pass through unrotated).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as usize
    }

    /// GDN qkv channel count: the conv1d operates over `[q | k | v]`.
    pub fn gdn_conv_channels(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Width of the hyper-connection residual stream.
    pub fn hc_width(&self) -> usize {
        self.hc_count * self.hidden_size
    }
}
