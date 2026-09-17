//! HF `config.json` parsing for Qwen3.8-Flash-Next, Qwen's `qwen4_exp` preview
//! architecture, as written by `tools/convert/convert_qwen38_flash_next.py`
//! (see `docs/qwen38-flash-next-checkpoint-format.md`). The file is a
//! multimodal wrapper; lily reads `text_config`, the converter's `lily`
//! block, and `vision_config` plus the vision token ids when the converter
//! kept the tower (`lily.vision` says so; a conversion without it strips them).

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
    /// Multimodal RoPE: rotary pairs per (temporal, height, width) axis. For
    /// text-only prompts all three axes carry the same position, so the
    /// engine's scalar RoPE is exact; an image makes the axes differ.
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// Interleaved assignment of pairs to axes (pair `i` takes axis `i % 3`
    /// within each section's budget), as opposed to contiguous sections.
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

/// The vision tower the converter keeps in bf16 (`docs/architecture.md`,
/// "The vision tower"; `tools/reference/VISION.md` for what each field means for the
/// compute). Present only when the checkpoint carries the tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionConfig {
    /// Transformer blocks.
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    /// Pixels per patch side.
    pub patch_size: usize,
    /// Frames folded into one patch row (a still image is copied twice).
    pub temporal_patch_size: usize,
    /// Patches folded per side into one language token by the merger.
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    /// Rows of the learned position table (a square grid).
    pub num_position_embeddings: usize,
    /// Merger output width: the language model's hidden size.
    pub out_hidden_size: usize,
    pub hidden_act: String,
    /// Blocks whose states are injected into later text layers (empty here).
    pub deepstack_visual_indexes: Vec<usize>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    /// Tensors under `model.visual.` the converter copied.
    pub tensors: usize,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Input width of the patch embedding: one patch row of pixels.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Input width of the merger: one merge block of hidden states.
    pub fn merge_dim(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }

    /// Tensor count the tower's structure implies: patch embedding weight and
    /// bias, the position table, twelve per block, six in the merger.
    pub fn expected_tensors(&self) -> usize {
        3 + 12 * self.depth + 6
    }
}

/// The multi-token-prediction draft head the converter appends with
/// `--mtp-only`: `layers` trunk-style blocks (Qwen3.8 ships one, a full
/// attention block) fed by the trunk's wide residual and the next token's
/// embedding, ending in its own stream mixer before the shared LM head.
#[derive(Clone, Copy, Debug)]
pub struct MtpConfig {
    pub layers: usize,
    pub rope_theta: f32,
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
struct MtpJson {
    layers: usize,
    #[serde(default)]
    layer_types: Vec<LayerType>,
    rope_theta: f32,
}

#[derive(Deserialize)]
struct LilyVisionJson {
    dtype: String,
    tensors: usize,
}

#[derive(Deserialize)]
struct LilyJson {
    format: String,
    quantization: QuantBlockJson,
    #[serde(default)]
    ple: PleJson,
    /// Present (non-null) when the checkpoint carries the `mtp.*` tensors.
    #[serde(default)]
    mtp: Option<MtpJson>,
    /// Present (non-null) when the checkpoint carries the `model.visual.*`
    /// tensors.
    #[serde(default)]
    vision: Option<LilyVisionJson>,
    /// Tensor name prefixes the converter dropped.
    #[serde(default)]
    dropped: Vec<String>,
}

#[derive(Deserialize)]
struct VisionConfigJson {
    depth: usize,
    hidden_size: usize,
    num_heads: usize,
    intermediate_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    in_channels: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    hidden_act: String,
    #[serde(default)]
    deepstack_visual_indexes: Vec<usize>,
}

#[derive(Deserialize)]
struct WrapperJson {
    model_type: String,
    text_config: TextConfigJson,
    lily: LilyJson,
    #[serde(default)]
    vision_config: Option<VisionConfigJson>,
    #[serde(default)]
    image_token_id: Option<u32>,
    #[serde(default)]
    video_token_id: Option<u32>,
    #[serde(default)]
    vision_start_token_id: Option<u32>,
    #[serde(default)]
    vision_end_token_id: Option<u32>,
}

/// The tensor name prefix of the vision tower.
pub const VISION_PREFIX: &str = "model.visual.";

pub const FORMAT: &str = "qwen4_exp-affine-v1";

/// Rotary pairs per (temporal, height, width) axis of the text model's
/// M-RoPE, the only layout the vision path is written for (VISION.md).
pub const MROPE_SECTION: &[usize] = &[11, 11, 10];

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
    /// The draft head, when the checkpoint includes it.
    pub mtp: Option<MtpConfig>,
    /// The vision tower, when the checkpoint includes it.
    pub vision: Option<VisionConfig>,
}

impl Qwen4ExpConfig {
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&bytes)
    }

    /// Parses and validates the contents of a converted `config.json`.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let wrapper: WrapperJson =
            serde_json::from_slice(bytes).context("parsing config.json")?;
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

        let mtp = match wrapper.lily.mtp {
            None => None,
            Some(m) => {
                ensure!(
                    m.layers == 1
                        && m.layer_types.as_slice() == [LayerType::FullAttention],
                    "unsupported MTP head: {} layers of {:?} (lily runs one full-attention block)",
                    m.layers,
                    m.layer_types
                );
                ensure!(
                    m.rope_theta.is_finite() && m.rope_theta > 0.0,
                    "invalid MTP rope_theta"
                );
                Some(MtpConfig { layers: m.layers, rope_theta: m.rope_theta })
            }
        };

        let vision = match wrapper.lily.vision {
            None => None,
            Some(v) => {
                ensure!(
                    !wrapper.lily.dropped.iter().any(|p| p == VISION_PREFIX),
                    "config.json declares lily.vision but lists {VISION_PREFIX} in lily.dropped"
                );
                ensure!(
                    v.dtype == "bf16",
                    "unsupported vision tower storage dtype {:?} (lily loads bf16)",
                    v.dtype
                );
                let vc = wrapper.vision_config.context(
                    "config.json declares lily.vision but has no vision_config",
                )?;
                let token = |id: Option<u32>, name: &str| {
                    id.with_context(|| {
                        format!("config.json declares lily.vision but has no {name}")
                    })
                };
                Some(VisionConfig {
                    depth: vc.depth,
                    hidden_size: vc.hidden_size,
                    num_heads: vc.num_heads,
                    intermediate_size: vc.intermediate_size,
                    patch_size: vc.patch_size,
                    temporal_patch_size: vc.temporal_patch_size,
                    spatial_merge_size: vc.spatial_merge_size,
                    in_channels: vc.in_channels,
                    num_position_embeddings: vc.num_position_embeddings,
                    out_hidden_size: vc.out_hidden_size,
                    hidden_act: vc.hidden_act,
                    deepstack_visual_indexes: vc.deepstack_visual_indexes,
                    image_token_id: token(wrapper.image_token_id, "image_token_id")?,
                    video_token_id: token(wrapper.video_token_id, "video_token_id")?,
                    vision_start_token_id: token(
                        wrapper.vision_start_token_id,
                        "vision_start_token_id",
                    )?,
                    vision_end_token_id: token(
                        wrapper.vision_end_token_id,
                        "vision_end_token_id",
                    )?,
                    tensors: v.tensors,
                })
            }
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
            mtp,
            vision,
        };
        config.validate_flash_next()?;
        config.validate_vision()?;
        Ok(config)
    }

    /// What lily's vision path (plan items 3 to 5) is written for: tanh-GELU
    /// blocks, no deepstack injection into the trunk, a merger that lands in
    /// the text hidden size, 2 x 2 merge over 2-frame patches, and the
    /// interleaved 3-axis RoPE VISION.md describes. Names the field that is
    /// off so a new checkpoint fails clearly.
    fn validate_vision(&self) -> Result<()> {
        let Some(v) = &self.vision else {
            return Ok(());
        };
        ensure!(
            v.hidden_act == "gelu_pytorch_tanh",
            "vision_config.hidden_act {:?} is not supported (lily implements gelu_pytorch_tanh)",
            v.hidden_act
        );
        ensure!(
            v.deepstack_visual_indexes.is_empty(),
            "vision_config.deepstack_visual_indexes {:?} is not supported (lily injects \
             vision features only at the placeholder rows)",
            v.deepstack_visual_indexes
        );
        ensure!(
            v.out_hidden_size == self.hidden_size,
            "vision_config.out_hidden_size {} != text hidden_size {}",
            v.out_hidden_size,
            self.hidden_size
        );
        ensure!(
            v.spatial_merge_size == 2,
            "vision_config.spatial_merge_size {} is not supported (lily merges 2 x 2)",
            v.spatial_merge_size
        );
        ensure!(
            v.temporal_patch_size == 2,
            "vision_config.temporal_patch_size {} is not supported (lily folds 2 frames)",
            v.temporal_patch_size
        );
        ensure!(
            v.depth > 0
                && v.num_heads > 0
                && v.hidden_size.is_multiple_of(v.num_heads)
                && v.head_dim().is_multiple_of(2),
            "vision_config.hidden_size {} / num_heads {} is not an even head dim",
            v.hidden_size,
            v.num_heads
        );
        let side = (v.num_position_embeddings as f64).sqrt() as usize;
        ensure!(
            side * side == v.num_position_embeddings,
            "vision_config.num_position_embeddings {} is not a square grid",
            v.num_position_embeddings
        );
        ensure!(
            v.tensors == v.expected_tensors(),
            "lily.vision.tensors {} != {} implied by vision_config.depth {}",
            v.tensors,
            v.expected_tensors(),
            v.depth
        );
        let rope = &self.rope_parameters;
        let section = rope.mrope_section.as_deref().context(
            "text_config.rope_parameters.mrope_section is missing (required with a vision tower)",
        )?;
        ensure!(
            section == MROPE_SECTION,
            "text_config.rope_parameters.mrope_section {section:?} is not supported \
             (lily's interleaved M-RoPE is written for {MROPE_SECTION:?})"
        );
        ensure!(
            section.iter().sum::<usize>() * 2 == self.rotary_dim(),
            "text_config.rope_parameters.mrope_section {section:?} does not sum to rotary_dim / 2 = {}",
            self.rotary_dim() / 2
        );
        ensure!(
            rope.mrope_interleaved == Some(true),
            "text_config.rope_parameters.mrope_interleaved {:?} is not supported (lily implements \
             the interleaved layout)",
            rope.mrope_interleaved
        );
        Ok(())
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

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/config.rs"]
mod tests;
