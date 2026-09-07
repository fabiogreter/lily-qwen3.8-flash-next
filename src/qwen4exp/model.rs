//! The Qwen3.8-Flash-Next model graph: single-token decode steps plus a
//! batched prefill path that processes `PREFILL_CHUNK` prompt tokens per
//! command buffer.
//!
//! Compared with the Qwen3.5 graph the residual is a `hc_count`-wide stream
//! read through a gated low-rank mixer before every block and written back
//! through per-stream scalar gates (hyper-connections); decoder layer 1 adds
//! the hashed n-gram embedding; and the full-attention layers run Qwen Sparse
//! Attention once the context exceeds the indexer budget.

use std::path::Path;

use anyhow::{Result, ensure};

use crate::engine::{DecodeStateApi, LanguageModel, ScratchApi};
use crate::kernels::attention::{
    MAX_SEQ, k_norm_rope_scatter_decode, q_norm_rope_split_decode, rope_neox,
    scatter_kv, sdpa_decode, sdpa_prefill, sdpa_split_scratch_splits, split_q_gate,
};
use crate::kernels::elementwise::{
    ARGMAX_GROUPS, argmax_f32, gather_row_bf16, sigmoid_mul_bf16,
};
use crate::kernels::gdn::{
    GDN_HEAD_DIM, GDN_STATE_DTYPE, GdnGate, GdnRegscanStaging, conv1d_prefill,
    conv1d_step, gated_rmsnorm, gdn_prefill, gdn_step_gated_fused,
};
use crate::kernels::hc::{
    hc_broadcast_bf16, hc_inject_bf16, hc_mix_bf16, rmsnorm_grouped_bf16,
    silu_scaled_bf16,
};
use crate::kernels::norm::rmsnorm_bf16;
use crate::kernels::ple::{self, PleHashTables};
use crate::kernels::qsa::{self, INDEXER_D, SparseSplitScratch};
use crate::kernels::{quant, skinny};
use crate::metal::{ComputePass, MetalContext, PendingPass};
use crate::moe_ffn::{
    DecodeMoeIo, MoeDims, MoeScratch, PrefillMoeIo, PrefillMoeScratch, decode_moe,
    prefill_moe, prefix_rows, project_mat, project_stack_or_slices,
};
use crate::tensor::{DType, Tensor};

use super::config::{GateAct, Qwen4ExpConfig};
use super::weights::{
    self, AttnWeights, GdnWeights, HcWeights, Mixer, ModelWeights, PleWeights,
};

/// Every RMSNorm in this model is zero-centered: gain = 1 + weight. (The GDN
/// GatedNorm is the exception and uses a plain gain.)
const NORM_WEIGHT_BIAS: f32 = 1.0;

/// Prompt tokens processed by one prefill command buffer.
const PREFILL_CHUNK: usize = 4096;

/// Queries per sparse-attention sub-batch: bounds the `[QB, blocks]` score
/// matrix and the split partials while keeping the GPU busy.
const QSA_QUERY_BATCH: usize = 256;

fn moe_dims(cfg: &Qwen4ExpConfig) -> MoeDims {
    MoeDims {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        moe_intermediate: cfg.moe_intermediate_size,
        hidden: cfg.hidden_size,
        norm_topk_prob: cfg.norm_topk_prob,
    }
}

pub struct Qwen4ExpModel {
    pub config: Qwen4ExpConfig,
    weights: ModelWeights,
    attn_scale: f32,
    gdn_scale: f32,
    gdn_gate: GdnGate,
    ple_tables: Option<PleHashTables>,
}

enum LayerState {
    Gdn {
        /// Recurrent state, fp32 `[H, 128, 128]`.
        state: Tensor,
        /// Double-buffered conv window, bf16 `[C, KD-1]` each (see the
        /// Qwen3.5 graph for the buffering contract).
        conv_windows: [Tensor; 2],
    },
    Attn {
        /// `[KVH, max_seq, D]` bf16.
        k_cache: Tensor,
        v_cache: Tensor,
        /// Raw indexer keys, bf16 `[max_seq, INDEXER_D]`.
        idx_keys: Tensor,
        /// Normed, roped block keys, bf16 `[max_seq / ratio, INDEXER_D]`.
        blk_keys: Tensor,
    },
}

/// Per-Layer Embedding recurrent state: the two-token hash history and the
/// dilated conv window, both double-buffered like the GDN conv windows.
struct PleState {
    hist: [Tensor; 2],
    conv_windows: [Tensor; 2],
    eos: u32,
}

impl PleState {
    fn reset(&self) -> Result<()> {
        let eos = [self.eos, self.eos];
        for h in &self.hist {
            h.write_bytes(bytemuck::cast_slice(&eos))?;
        }
        for w in &self.conv_windows {
            w.zero_fill();
        }
        Ok(())
    }
}

pub struct DecodeState {
    pub pos: usize,
    max_seq: usize,
    layers: Vec<LayerState>,
    /// Which double-buffered window/history slot is current (all recurrent
    /// layers advance in lockstep): prefill chunks read it, write the other,
    /// then flip; decode steps update it in place.
    conv_slot: usize,
    ple: Option<PleState>,
}

impl DecodeState {
    /// Returns the state to position zero. KV and indexer caches need no
    /// clearing: rows past `pos` are never read and get overwritten. The GPU
    /// must be idle on this state's buffers.
    pub fn reset(&mut self) -> Result<()> {
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                state.zero_fill();
                conv_windows[0].zero_fill();
                conv_windows[1].zero_fill();
            }
        }
        if let Some(ple) = &self.ple {
            ple.reset()?;
        }
        self.pos = 0;
        self.conv_slot = 0;
        Ok(())
    }
}

/// Sparse-attention scratch shared by decode (one query) and prefill
/// sub-batches (up to `QSA_QUERY_BATCH` queries), viewed per use.
struct QsaScratch {
    /// F32 `[QB, max_blocks]`.
    scores: Tensor,
    /// U32 `[QB, k_max]`.
    sel: Tensor,
    /// U32 `[QB]`.
    n_sel: Tensor,
    /// F32 `[QB * NQ, splits, D]`.
    partials: Tensor,
    /// F32 `[QB * NQ, splits, 2]`.
    stats: Tensor,
}

impl QsaScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        max_seq: usize,
        qb: usize,
    ) -> Result<Self> {
        let idx = &cfg.indexer;
        let max_blocks = (max_seq / idx.compress_ratio).max(1);
        let k_max = idx.block_topk();
        let splits = qsa::sparse_splits(k_max, idx.compress_ratio);
        let nq = cfg.num_attention_heads;
        Ok(Self {
            scores: Tensor::zeros(ctx, &[qb, max_blocks], DType::F32)?,
            sel: Tensor::zeros(ctx, &[qb, k_max], DType::U32)?,
            n_sel: Tensor::zeros(ctx, &[qb], DType::U32)?,
            partials: Tensor::zeros(ctx, &[qb * nq, splits, cfg.head_dim], DType::F32)?,
            stats: Tensor::zeros(ctx, &[qb * nq, splits, 2], DType::F32)?,
        })
    }

    fn split_scratch(&self) -> SparseSplitScratch<'_> {
        SparseSplitScratch { partials: &self.partials, stats: &self.stats }
    }
}

/// Single-token PLE intermediates.
struct PleScratch {
    ids: Tensor,
    emb: Tensor,
    key: Tensor,
    key_n: Tensor,
    query_n: Tensor,
    value: Tensor,
    gated: Tensor,
    gated_n: Tensor,
}

impl PleScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig, rows: usize) -> Result<Self> {
        let ple = cfg.ple.as_ref().expect("PLE scratch without PLE config");
        let (h, wide) = (cfg.hidden_size, cfg.hc_width());
        let bf = DType::BF16;
        Ok(Self {
            ids: Tensor::zeros(ctx, &[rows, ple.ngram_heads()], DType::U32)?,
            emb: Tensor::zeros(ctx, &[rows, ple.embed_dim], bf)?,
            key: Tensor::zeros(ctx, &[rows, wide], bf)?,
            key_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            query_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            value: Tensor::zeros(ctx, &[rows, h], bf)?,
            gated: Tensor::zeros(ctx, &[rows, wide], bf)?,
            gated_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        Ok(Self {
            ids: prefix_rows(&self.ids, m)?,
            emb: prefix_rows(&self.emb, m)?,
            key: prefix_rows(&self.key, m)?,
            key_n: prefix_rows(&self.key_n, m)?,
            query_n: prefix_rows(&self.query_n, m)?,
            value: prefix_rows(&self.value, m)?,
            gated: prefix_rows(&self.gated, m)?,
            gated_n: prefix_rows(&self.gated_n, m)?,
        })
    }
}

/// Hyper-connection read intermediates for `rows` tokens.
struct HcScratch {
    /// `[rows, G*H]` normed streams.
    hn: Tensor,
    /// `[rows, lowrank]` read-gate logits and activation.
    down: Tensor,
    act: Tensor,
    /// `[rows, G*H]` per-element gate logits.
    up: Tensor,
    /// `[rows, H]` the block input.
    mixed: Tensor,
    /// `[rows, G]` write-gate logits.
    inj: Tensor,
}

impl HcScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig, rows: usize) -> Result<Self> {
        let (h, wide, g) = (cfg.hidden_size, cfg.hc_width(), cfg.hc_count);
        let bf = DType::BF16;
        Ok(Self {
            hn: Tensor::zeros(ctx, &[rows, wide], bf)?,
            down: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            act: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            up: Tensor::zeros(ctx, &[rows, wide], bf)?,
            mixed: Tensor::zeros(ctx, &[rows, h], bf)?,
            inj: Tensor::zeros(ctx, &[rows, g], bf)?,
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        Ok(Self {
            hn: prefix_rows(&self.hn, m)?,
            down: prefix_rows(&self.down, m)?,
            act: prefix_rows(&self.act, m)?,
            up: prefix_rows(&self.up, m)?,
            mixed: prefix_rows(&self.mixed, m)?,
            inj: prefix_rows(&self.inj, m)?,
        })
    }
}

/// Per-step intermediates, allocated once. Projection outputs that share one
/// fused matvec (`gdn_in`, `attn_qkv`, `mlp_gu`) are single buffers whose
/// named segments are views.
pub struct Scratch {
    x: Tensor,
    /// `[G*H]` the hyper-connection residual stream.
    hyper: Tensor,
    hc: HcScratch,
    branch_out: Tensor,
    gdn_in: Tensor,
    attn_qkv: Tensor,
    mlp_gu: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_gated: Tensor,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    /// `[(NH+1)*INDEXER_D]` indexer projection and its prepared queries.
    idx_qk: Tensor,
    idx_q: Tensor,
    qsa: QsaScratch,
    ple: Option<PleScratch>,
    logits: Tensor,
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
    argmax_partials: Tensor,
    /// Greedy tokens written by the in-graph argmax (`U32[2]`): two ping-pong
    /// slots so the pipelined loop can host-read step N's token while the
    /// in-flight step N+1 writes the other slot.
    pub next_token: Tensor,
    /// Prefill staging for the largest quantized projection.
    dequant: Tensor,
    moe: MoeScratch,
    /// Session-lived batched prefill intermediates, grown on demand and capped
    /// at `PREFILL_CHUNK` rows.
    prefill: Option<PrefillScratch>,
}

/// Batched prefill intermediates at a capacity that only grows; each chunk
/// uses exact row-prefix views.
struct PrefillScratch {
    m: usize,
    ids: Tensor,
    x: Tensor,
    hyper: Tensor,
    hc: HcScratch,
    branch_out: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_out: Tensor,
    gdn_gated: Tensor,
    gdn_stage: GdnStageScratch,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    idx_qk: Tensor,
    idx_q: Tensor,
    stack: Tensor,
    moe: PrefillMoeScratch,
    qsa: QsaScratch,
    ple: Option<PleScratch>,
}

struct GdnStageScratch {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
}

impl PrefillScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        capacity: usize,
        max_seq: usize,
    ) -> Result<Self> {
        ensure!(capacity > 0, "prefill scratch capacity must be nonzero");
        let m = capacity;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let hk = cfg.linear_num_key_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        Ok(Self {
            m,
            ids: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: Tensor::zeros(ctx, &[m, h], bf)?,
            hyper: Tensor::zeros(ctx, &[m, cfg.hc_width()], bf)?,
            hc: HcScratch::new(ctx, cfg, m)?,
            branch_out: Tensor::zeros(ctx, &[m, h], bf)?,
            mlp_gate: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_up: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_act: Tensor::zeros(ctx, &[m, inter], bf)?,
            qkv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            qkv_conv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            z: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            a: Tensor::zeros(ctx, &[m, heads], bf)?,
            b: Tensor::zeros(ctx, &[m, heads], bf)?,
            gdn_out: Tensor::zeros(ctx, &[m, heads, GDN_HEAD_DIM], bf)?,
            gdn_gated: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            gdn_stage: GdnStageScratch {
                qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], bf)?,
                decay: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                beta: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
            },
            qg: Tensor::zeros(ctx, &[m, nq * 2 * hd], bf)?,
            q: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            k_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            v_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            attn_o: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[m, nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[m, (nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[m, nh, INDEXER_D], bf)?,
            stack: {
                // Rows cap at the largest chunk eligible for fused projection.
                let rows = m.min(skinny::DENSE_SMALLM_THRESHOLD);
                let width = (conv_c + dim_v + 2 * heads)
                    .max((nq * 2 + 2 * nkv) * hd)
                    .max(2 * inter);
                Tensor::zeros(ctx, &[rows, width], bf)?
            },
            moe: PrefillMoeScratch::new(ctx, &moe_dims(cfg), m)?,
            qsa: QsaScratch::new(ctx, cfg, max_seq, QSA_QUERY_BATCH.min(m))?,
            ple: cfg.ple.as_ref().map(|_| PleScratch::new(ctx, cfg, m)).transpose()?,
        })
    }

    /// Views of the capacity scratch for one chunk of `tokens`, uploading the
    /// token ids (the GPU is idle here: chunks are commit_wait-synchronized).
    fn chunk(&self, tokens: &[u32]) -> Result<Self> {
        let m = tokens.len();
        ensure!(m <= self.m, "chunk of {m} tokens exceeds scratch capacity {}", self.m);
        let ids = prefix_rows(&self.ids, m)?;
        let chunk = Self {
            m,
            ids,
            x: prefix_rows(&self.x, m)?,
            hyper: prefix_rows(&self.hyper, m)?,
            hc: self.hc.rows(m)?,
            branch_out: prefix_rows(&self.branch_out, m)?,
            mlp_gate: prefix_rows(&self.mlp_gate, m)?,
            mlp_up: prefix_rows(&self.mlp_up, m)?,
            mlp_act: prefix_rows(&self.mlp_act, m)?,
            qkv: prefix_rows(&self.qkv, m)?,
            qkv_conv: prefix_rows(&self.qkv_conv, m)?,
            z: prefix_rows(&self.z, m)?,
            a: prefix_rows(&self.a, m)?,
            b: prefix_rows(&self.b, m)?,
            gdn_out: prefix_rows(&self.gdn_out, m)?,
            gdn_gated: prefix_rows(&self.gdn_gated, m)?,
            gdn_stage: GdnStageScratch {
                qk_norm: prefix_rows(&self.gdn_stage.qk_norm, m)?,
                decay: prefix_rows(&self.gdn_stage.decay, m)?,
                beta: prefix_rows(&self.gdn_stage.beta, m)?,
            },
            qg: prefix_rows(&self.qg, m)?,
            q: prefix_rows(&self.q, m)?,
            gate: prefix_rows(&self.gate, m)?,
            k_new: prefix_rows(&self.k_new, m)?,
            v_new: prefix_rows(&self.v_new, m)?,
            attn_o: prefix_rows(&self.attn_o, m)?,
            attn_gated: prefix_rows(&self.attn_gated, m)?,
            idx_qk: prefix_rows(&self.idx_qk, m)?,
            idx_q: prefix_rows(&self.idx_q, m)?,
            // Kept at full extent: `project_stack_or_slices` views the exact
            // [m, n_total] prefix per level and never fuses past the threshold.
            stack: self.stack.view(0, self.stack.shape())?,
            moe: self.moe.chunk(m)?,
            // Sub-batch views are taken per use inside the attention branch.
            qsa: QsaScratch {
                scores: self.qsa.scores.view(0, self.qsa.scores.shape())?,
                sel: self.qsa.sel.view(0, self.qsa.sel.shape())?,
                n_sel: self.qsa.n_sel.view(0, self.qsa.n_sel.shape())?,
                partials: self.qsa.partials.view(0, self.qsa.partials.shape())?,
                stats: self.qsa.stats.view(0, self.qsa.stats.shape())?,
            },
            ple: self.ple.as_ref().map(|p| p.rows(m)).transpose()?,
        };
        chunk.ids.write_bytes(bytemuck::cast_slice(tokens))?;
        Ok(chunk)
    }
}

impl Qwen4ExpModel {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        let config = Qwen4ExpConfig::from_model_dir(&dir)?;
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "GDN head dim {} unsupported (kernel is compiled for {GDN_HEAD_DIM})",
            config.linear_key_head_dim,
        );
        ensure!(
            config.indexer.head_dim == INDEXER_D,
            "indexer head dim {} unsupported (kernels are compiled for {INDEXER_D})",
            config.indexer.head_dim
        );
        let weights = weights::load(ctx, &dir, &config)?;
        let ple_tables = match &config.ple {
            Some(p) => Some(PleHashTables::new(
                ctx,
                &p.layer_multipliers,
                &p.head_vocab_sizes,
                &p.head_offsets,
                p.heads_per_ngram,
                p.eos_token_id,
            )?),
            None => None,
        };
        let gdn_gate = match config.gdn_gate {
            GateAct::Silu => GdnGate::Silu,
            GateAct::Sigmoid => GdnGate::Sigmoid,
        };
        let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
        let gdn_scale = 1.0 / (config.linear_key_head_dim as f32).sqrt();
        Ok(Self { config, weights, attn_scale, gdn_scale, gdn_gate, ple_tables })
    }

    pub fn new_state(&self, ctx: &MetalContext, max_seq: usize) -> Result<DecodeState> {
        ensure!(max_seq <= MAX_SEQ, "max_seq {max_seq} exceeds kernel limit {MAX_SEQ}");
        let cfg = &self.config;
        let c = cfg.gdn_conv_channels();
        let kd = cfg.linear_conv_kernel_dim;
        let heads = cfg.linear_num_value_heads;
        let ratio = cfg.indexer.compress_ratio;
        let layers = self
            .weights
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Mixer::Gdn(_) => Ok(LayerState::Gdn {
                    state: Tensor::zeros(
                        ctx,
                        &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )?,
                    conv_windows: [
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                    ],
                }),
                Mixer::Attn(_) => Ok(LayerState::Attn {
                    k_cache: Tensor::zeros(
                        ctx,
                        &[cfg.num_key_value_heads, max_seq, cfg.head_dim],
                        DType::BF16,
                    )?,
                    v_cache: Tensor::zeros(
                        ctx,
                        &[cfg.num_key_value_heads, max_seq, cfg.head_dim],
                        DType::BF16,
                    )?,
                    idx_keys: Tensor::zeros(ctx, &[max_seq, INDEXER_D], DType::BF16)?,
                    blk_keys: Tensor::zeros(
                        ctx,
                        &[(max_seq / ratio).max(1), INDEXER_D],
                        DType::BF16,
                    )?,
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        let ple = match &cfg.ple {
            Some(p) => {
                let s = p.conv_state_len();
                let state = PleState {
                    hist: [
                        Tensor::zeros(ctx, &[2], DType::U32)?,
                        Tensor::zeros(ctx, &[2], DType::U32)?,
                    ],
                    conv_windows: [
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                    ],
                    eos: p.eos_token_id,
                };
                state.reset()?;
                Some(state)
            }
            None => None,
        };
        Ok(DecodeState { pos: 0, max_seq, layers, conv_slot: 0, ple })
    }

    /// Scratch whose split-decode and sparse-attention buffers are sized for
    /// `capacity_tokens` of context rather than the engine's ceiling.
    pub fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        let cfg = &self.config;
        // Dense decode only ever runs up to the indexer's dense limit.
        let splits =
            sdpa_split_scratch_splits(capacity_tokens.min(cfg.indexer.dense_limit()));
        let h = cfg.hidden_size;
        let wide = cfg.hc_width();
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        let gdn_in = Tensor::zeros(ctx, &[conv_c + dim_v + 2 * heads], bf)?;
        let attn_qkv = Tensor::zeros(ctx, &[(nq * 2 + 2 * nkv) * hd], bf)?;
        let mlp_gu = Tensor::zeros(ctx, &[2 * inter], bf)?;
        // Largest dense projection the batched fallback may stage. Expert
        // stacks use native grouped Q4 kernels and never dequant in full.
        let ple_embed = cfg.ple.as_ref().map_or(0, |p| p.embed_dim);
        let dequant_numel = [
            conv_c * h,
            dim_v * h,
            heads * h,
            h * dim_v,
            nq * 2 * hd * h,
            nkv * hd * h,
            h * nq * hd,
            cfg.num_experts * h,
            cfg.moe_intermediate_size * h,
            h * cfg.moe_intermediate_size,
            inter * h,
            h * inter,
            cfg.hc_lowrank * wide,
            cfg.hc_count * wide,
            (nh + 1) * INDEXER_D * h,
            wide * ple_embed,
            h * ple_embed,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(Scratch {
            x: Tensor::zeros(ctx, &[h], bf)?,
            hyper: Tensor::zeros(ctx, &[wide], bf)?,
            hc: HcScratch::new(ctx, cfg, 1)?,
            branch_out: Tensor::zeros(ctx, &[h], bf)?,
            mlp_gate: mlp_gu.view(0, &[inter])?,
            mlp_up: mlp_gu.view(inter, &[inter])?,
            mlp_act: Tensor::zeros(ctx, &[inter], bf)?,
            qkv: gdn_in.view(0, &[conv_c])?,
            qkv_conv: Tensor::zeros(ctx, &[conv_c], bf)?,
            z: gdn_in.view(conv_c, &[dim_v])?,
            a: gdn_in.view(conv_c + dim_v, &[heads])?,
            b: gdn_in.view(conv_c + dim_v + heads, &[heads])?,
            gdn_gated: Tensor::zeros(ctx, &[dim_v], bf)?,
            qg: attn_qkv.view(0, &[nq, 2 * hd])?,
            q: Tensor::zeros(ctx, &[nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[nq, hd], bf)?,
            k_new: attn_qkv.view(nq * 2 * hd, &[nkv, hd])?,
            v_new: attn_qkv.view((nq * 2 + nkv) * hd, &[nkv, hd])?,
            attn_o: Tensor::zeros(ctx, &[nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[(nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[nh, INDEXER_D], bf)?,
            qsa: QsaScratch::new(ctx, cfg, capacity_tokens, 1)?,
            ple: cfg.ple.as_ref().map(|_| PleScratch::new(ctx, cfg, 1)).transpose()?,
            gdn_in,
            attn_qkv,
            mlp_gu,
            logits: Tensor::zeros(ctx, &[cfg.vocab_size], DType::F32)?,
            sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
            sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
            argmax_partials: Tensor::zeros(ctx, &[2 * ARGMAX_GROUPS], DType::U32)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_numel], bf)?,
            moe: MoeScratch::new(ctx, &moe_dims(cfg))?,
            prefill: None,
        })
    }

    /// Submits one decode step whose input token is read on-GPU from
    /// `next_token[slot_in]` and whose argmax lands in `next_token[slot_out]`,
    /// without waiting (the depth-2 pipelined primitive). `pos` advances at
    /// encode time.
    pub fn submit_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<PendingPass<'a>> {
        let pass = ctx.begin_concurrent()?;
        self.encode_decode_step(ctx, &pass, state, s, slot_in, slot_out)?;
        state.pos += 1;
        pass.commit()
    }

    /// Runs the prompt in batches of `PREFILL_CHUNK` tokens, one command
    /// buffer per chunk. Logits are produced for the final prompt token only.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prompt");
        let needed = tokens.len().min(PREFILL_CHUNK);
        let have = s.prefill.as_ref().map_or(0, |p| p.m);
        if have < needed {
            let target = needed.next_power_of_two().min(PREFILL_CHUNK);
            s.prefill = None;
            s.prefill =
                Some(PrefillScratch::new(ctx, &self.config, target, state.max_seq)?);
        }
        let s = &*s;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing after growth"))?;
        let mut remaining = tokens.len();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let ps = capacity.chunk(chunk)?;
            remaining -= chunk.len();
            self.encode_prefill_chunk(ctx, state, s, &ps, remaining == 0)?;
            state.pos += chunk.len();
            // The chunk's recurrent kernels wrote the other window buffers.
            state.conv_slot = 1 - state.conv_slot;
        }
        Ok(())
    }

    // --- prefill ------------------------------------------------------------

    fn encode_prefill_chunk(
        &self,
        ctx: &MetalContext,
        state: &DecodeState,
        s: &Scratch,
        ps: &PrefillScratch,
        want_logits: bool,
    ) -> Result<()> {
        let pass = ctx.begin()?;
        let m = ps.m;
        ensure!(state.pos + m <= state.max_seq, "sequence full ({})", state.max_seq);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let pos = state.pos;
        let conv_slot = state.conv_slot;

        quant::gather_rows_q4(ctx, &pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;
        hc_broadcast_bf16(ctx, &pass, &ps.x, &ps.hyper, h, g)?;

        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            if let (Some(ple_w), Some(ple_s), Some(pst)) =
                (&layer.ple, &ps.ple, &state.ple)
            {
                self.ple_prefill(ctx, &pass, ple_w, ple_s, pst, conv_slot, s, ps)?;
            }

            self.hc_read_prefill(ctx, &pass, &layer.attn_hc, s, ps)?;
            match (&layer.mixer, lstate) {
                (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    self.gdn_prefill_branch(
                        ctx,
                        &pass,
                        w,
                        s,
                        ps,
                        state,
                        conv_windows,
                        conv_slot,
                    )?;
                }
                (
                    Mixer::Attn(w),
                    LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
                ) => {
                    self.attn_prefill_branch(
                        ctx, &pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            hc_inject_bf16(ctx, &pass, &ps.hyper, &ps.branch_out, &ps.hc.inj, h, g)?;

            self.hc_read_prefill(ctx, &pass, &layer.mlp_hc, s, ps)?;
            prefill_moe(
                ctx,
                &pass,
                &moe_dims(cfg),
                &layer.ffn,
                &PrefillMoeIo {
                    x: &ps.hc.mixed,
                    out: &ps.branch_out,
                    stack: &ps.stack,
                    mlp_gate: &ps.mlp_gate,
                    mlp_up: &ps.mlp_up,
                    mlp_act: &ps.mlp_act,
                    dequant: &s.dequant,
                },
                &ps.moe,
            )?;
            hc_inject_bf16(ctx, &pass, &ps.hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        }

        if want_logits {
            // Only the last prompt token feeds decoding: pull its stream row
            // into the single-token scratch and reuse the decode head path.
            gather_row_bf16(ctx, &pass, &ps.hyper, &s.hyper, m - 1)?;
            self.hc_read_decode(ctx, &pass, &self.weights.final_mixer, s)?;
            quant::gemv_quant(
                ctx,
                &pass,
                &self.weights.lm_head,
                &s.hc.mixed,
                &s.logits,
            )?;
            // Slot 0 by convention: the pipelined loop's first decode step
            // consumes the prefill token from this slot.
            let out = s.next_token.view(0, &[1])?;
            argmax_f32(ctx, &pass, &s.logits, &s.argmax_partials, &out)?;
        }
        pass.commit_wait()
    }

    /// hyper `[m, G*H]` → `ps.hc.mixed` `[m, H]` (and the write-gate logits
    /// when the block has an inject weight).
    fn hc_read_prefill(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let inv_g = 1.0 / g as f32;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &ps.hyper,
            &hc.norm,
            &ps.hc.hn,
            h,
            g,
            cfg.rms_norm_eps,
            NORM_WEIGHT_BIAS,
        )?;
        project_mat(ctx, pass, &ps.hc.hn, &hc.down, &ps.hc.down, &s.dequant)?;
        silu_scaled_bf16(ctx, pass, &ps.hc.down, &ps.hc.act, inv_g)?;
        project_mat(ctx, pass, &ps.hc.act, &hc.up, &ps.hc.up, &s.dequant)?;
        hc_mix_bf16(ctx, pass, &ps.hc.up, &ps.hc.hn, &ps.hc.mixed, h, g)?;
        if let Some(inject) = &hc.inject {
            project_mat(ctx, pass, &ps.hc.hn, inject, &ps.hc.inj, &s.dequant)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_prefill(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let tables = self.ple_tables.as_ref().expect("PLE weights without hash tables");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        ple::ple_hash_ids(ctx, pass, tables, &ps.ids, &pst.hist[conv_slot], &p.ids)?;
        ple::ple_hist_update(
            ctx,
            pass,
            &ps.ids,
            &pst.hist[conv_slot],
            &pst.hist[1 - conv_slot],
        )?;
        ple::ple_gather_q4(ctx, pass, &w.table, &p.ids, ple.ngram_heads(), &p.emb)?;
        project_mat(ctx, pass, &p.emb, &w.key_proj, &p.key, &s.dequant)?;
        project_mat(ctx, pass, &p.emb, &w.value_proj, &p.value, &s.dequant)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &ps.hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        ple::ple_gate_value_bf16(
            ctx, pass, &p.key_n, &p.query_n, &p.value, &p.gated, h, g,
        )?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        ple::ple_conv1d_prefill(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &pst.conv_windows[1 - conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            &ps.hyper,
            ple.ngram_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_prefill_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        gdn_state: &Tensor,
        conv_windows: &[Tensor; 2],
        conv_slot: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.in_proj,
            &ps.stack,
            [
                (&w.in_proj_qkv, &ps.qkv),
                (&w.in_proj_z, &ps.z),
                (&w.in_proj_a, &ps.a),
                (&w.in_proj_b, &ps.b),
            ],
            &s.dequant,
        )?;
        conv1d_prefill(
            ctx,
            pass,
            &conv_windows[conv_slot],
            &conv_windows[1 - conv_slot],
            &ps.qkv,
            &w.conv_w,
            &ps.qkv_conv,
        )?;
        let staging = GdnRegscanStaging {
            qk_norm: &ps.gdn_stage.qk_norm,
            decay: &ps.gdn_stage.decay,
            beta: &ps.gdn_stage.beta,
        };
        gdn_prefill(
            ctx,
            pass,
            &ps.qkv_conv,
            &ps.a,
            &ps.b,
            &w.a_log,
            &w.dt_bias,
            &staging,
            gdn_state,
            &ps.gdn_out,
            self.gdn_scale,
            cfg.linear_num_key_heads,
        )?;
        gated_rmsnorm(
            ctx,
            pass,
            &ps.gdn_out,
            &ps.z,
            &w.norm_w,
            &ps.gdn_gated,
            cfg.rms_norm_eps,
            self.gdn_gate,
        )?;
        project_mat(ctx, pass, &ps.gdn_gated, &w.out_proj, &ps.branch_out, &s.dequant)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_prefill_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;
        let m = ps.m;
        let idx = &cfg.indexer;
        let (nq, hd) = (cfg.num_attention_heads, cfg.head_dim);

        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.qkv_proj,
            &ps.stack,
            [(&w.q_proj, &ps.qg), (&w.k_proj, &ps.k_new), (&w.v_proj, &ps.v_new)],
            &s.dequant,
        )?;
        split_q_gate(ctx, pass, &ps.qg, &ps.q, &ps.gate)?;
        rmsnorm_bf16(ctx, pass, &ps.q, &w.q_norm, &ps.q, eps, NORM_WEIGHT_BIAS)?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.k_new,
            &w.k_norm,
            &ps.k_new,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        rope_neox(ctx, pass, &ps.q, nq, rot, pos, theta)?;
        rope_neox(ctx, pass, &ps.k_new, cfg.num_key_value_heads, rot, pos, theta)?;
        scatter_kv(ctx, pass, k_cache, &ps.k_new, pos)?;
        scatter_kv(ctx, pass, v_cache, &ps.v_new, pos)?;

        // Indexer: queries for this chunk, raw keys into the cache, and the
        // block keys every block completed so far by this chunk.
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
            &s.dequant,
        )?;
        qsa::qsa_prep_q(
            ctx,
            pass,
            &ps.idx_qk,
            &w.indexer.q_norm,
            &ps.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &ps.idx_qk, idx_keys, idx.n_heads, pos)?;
        let first_block = pos / idx.compress_ratio;
        let complete = (pos + m) / idx.compress_ratio;
        qsa::qsa_block_keys(
            ctx,
            pass,
            idx_keys,
            &w.indexer.k_norm,
            blk_keys,
            idx.compress_ratio,
            first_block,
            complete - first_block,
            rot,
            theta,
            eps,
        )?;

        if pos + m <= idx.dense_limit() {
            // Every query sees at most the budget: the selection is the whole
            // causal window, so the dense kernel is exact.
            sdpa_prefill(
                ctx,
                pass,
                &ps.q,
                k_cache,
                v_cache,
                &ps.attn_o,
                pos,
                self.attn_scale,
            )?;
        } else {
            let qb_cap = ps.qsa.n_sel.numel();
            for q0 in (0..m).step_by(qb_cap) {
                let qb = qb_cap.min(m - q0);
                let base = pos + q0;
                let nb_max = qsa::visible_blocks(base + qb - 1, idx.compress_ratio);
                let idx_q = ps.idx_q.view(
                    q0 * idx.n_heads * INDEXER_D,
                    &[qb, idx.n_heads, INDEXER_D],
                )?;
                let q = ps.q.view(q0 * nq * hd, &[qb, nq, hd])?;
                let out = ps.attn_o.view(q0 * nq * hd, &[qb, nq, hd])?;
                qsa::qsa_scores(
                    ctx,
                    pass,
                    &idx_q,
                    blk_keys,
                    &ps.qsa.scores,
                    idx.n_heads,
                    nb_max,
                    base,
                    idx.compress_ratio,
                )?;
                qsa::qsa_select_blocks(
                    ctx,
                    pass,
                    &ps.qsa.scores,
                    &ps.qsa.sel,
                    &ps.qsa.n_sel,
                    qb,
                    nb_max,
                    base,
                    idx.compress_ratio,
                    idx.block_topk(),
                )?;
                qsa::qsa_attention(
                    ctx,
                    pass,
                    &q,
                    k_cache,
                    v_cache,
                    &ps.qsa.sel,
                    &ps.qsa.n_sel,
                    &out,
                    &ps.qsa.split_scratch(),
                    qb,
                    idx.block_topk(),
                    idx.compress_ratio,
                    base,
                    self.attn_scale,
                )?;
            }
        }
        sigmoid_mul_bf16(ctx, pass, &ps.gate, &ps.attn_o, &ps.attn_gated)?;
        project_mat(ctx, pass, &ps.attn_gated, &w.o_proj, &ps.branch_out, &s.dequant)
    }

    // --- decode -------------------------------------------------------------

    /// Encodes one concurrent-dispatch decode graph at `state.pos`. The
    /// concurrent encoder has no implicit dispatch ordering: `level_barrier`
    /// marks every true inter-level data edge.
    fn encode_decode_step(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<()> {
        ensure!(state.pos < state.max_seq, "sequence full ({})", state.max_seq);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let conv_slot = state.conv_slot;

        let ids = s.next_token.view(slot_in, &[1])?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ids, &s.x)?;
        pass.level_barrier(&[&s.x])?;
        hc_broadcast_bf16(ctx, pass, &s.x, &s.hyper, h, g)?;
        pass.level_barrier(&[&s.hyper])?;

        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            if let (Some(ple_w), Some(ple_s), Some(pst)) =
                (&layer.ple, &s.ple, &state.ple)
            {
                self.ple_decode(ctx, pass, ple_w, ple_s, pst, conv_slot, s, &ids)?;
            }

            self.hc_read_decode(ctx, pass, &layer.attn_hc, s)?;
            match (&layer.mixer, lstate) {
                (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    self.gdn_decode_branch(
                        ctx,
                        pass,
                        w,
                        s,
                        state,
                        &conv_windows[conv_slot],
                    )?;
                }
                (
                    Mixer::Attn(w),
                    LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
                ) => {
                    self.attn_decode_branch(
                        ctx, pass, w, s, k_cache, v_cache, idx_keys, blk_keys,
                        state.pos,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            pass.level_barrier(&[&s.branch_out])?;
            hc_inject_bf16(ctx, pass, &s.hyper, &s.branch_out, &s.hc.inj, h, g)?;
            pass.level_barrier(&[&s.hyper])?;

            self.hc_read_decode(ctx, pass, &layer.mlp_hc, s)?;
            decode_moe(
                ctx,
                pass,
                &moe_dims(cfg),
                &layer.ffn,
                &DecodeMoeIo {
                    x: &s.hc.mixed,
                    out: &s.branch_out,
                    mlp_gu: &s.mlp_gu,
                    mlp_gate: &s.mlp_gate,
                    mlp_up: &s.mlp_up,
                    mlp_act: &s.mlp_act,
                },
                &s.moe,
            )?;
            pass.level_barrier(&[&s.branch_out])?;
            hc_inject_bf16(ctx, pass, &s.hyper, &s.branch_out, &s.hc.inj, h, g)?;
            pass.level_barrier(&[&s.hyper])?;
        }

        self.hc_read_decode(ctx, pass, &self.weights.final_mixer, s)?;
        quant::gemv_quant(ctx, pass, &self.weights.lm_head, &s.hc.mixed, &s.logits)?;
        pass.level_barrier(&[&s.logits])?;
        let out = s.next_token.view(slot_out, &[1])?;
        argmax_f32(ctx, pass, &s.logits, &s.argmax_partials, &out)?;
        pass.level_barrier(&[&s.next_token])?;
        Ok(())
    }

    /// Single-row hyper-connection read: `s.hyper` → `s.hc.mixed` (+ `inj`).
    /// Leaves `mixed`/`inj` ordered for the caller.
    fn hc_read_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        s: &Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let inv_g = 1.0 / g as f32;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &s.hyper,
            &hc.norm,
            &s.hc.hn,
            h,
            g,
            cfg.rms_norm_eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&s.hc.hn])?;
        quant::gemv_quant(ctx, pass, &hc.down, &s.hc.hn, &s.hc.down)?;
        if let Some(inject) = &hc.inject {
            quant::gemv_quant(ctx, pass, inject, &s.hc.hn, &s.hc.inj)?;
        }
        pass.level_barrier(&[&s.hc.down, &s.hc.inj])?;
        silu_scaled_bf16(ctx, pass, &s.hc.down, &s.hc.act, inv_g)?;
        pass.level_barrier(&[&s.hc.act])?;
        quant::gemv_quant(ctx, pass, &hc.up, &s.hc.act, &s.hc.up)?;
        pass.level_barrier(&[&s.hc.up])?;
        hc_mix_bf16(ctx, pass, &s.hc.up, &s.hc.hn, &s.hc.mixed, h, g)?;
        pass.level_barrier(&[&s.hc.mixed])
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        s: &Scratch,
        token: &Tensor,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let tables = self.ple_tables.as_ref().expect("PLE weights without hash tables");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        let hist = &pst.hist[conv_slot];
        ple::ple_hash_ids(ctx, pass, tables, token, hist, &p.ids)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &s.hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.ids, &p.query_n])?;
        // The history advances in place once the hash has consumed it.
        ple::ple_hist_update(ctx, pass, token, hist, hist)?;
        ple::ple_gather_q4(ctx, pass, &w.table, &p.ids, ple.ngram_heads(), &p.emb)?;
        pass.level_barrier(&[&p.emb, hist])?;
        quant::gemv_quant(ctx, pass, &w.key_proj, &p.emb, &p.key)?;
        quant::gemv_quant(ctx, pass, &w.value_proj, &p.emb, &p.value)?;
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16(
            ctx, pass, &p.key_n, &p.query_n, &p.value, &p.gated, h, g,
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        ple::ple_conv1d_step(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            &s.hyper,
            ple.ngram_size,
        )?;
        pass.level_barrier(&[&s.hyper, &pst.conv_windows[conv_slot]])
    }

    fn gdn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        gdn_state: &Tensor,
        conv_window: &Tensor,
    ) -> Result<()> {
        quant::gemv_quant(ctx, pass, &w.in_proj, &s.hc.mixed, &s.gdn_in)?;
        pass.level_barrier(&[&s.gdn_in])?;
        conv1d_step(ctx, pass, conv_window, &s.qkv, &w.conv_w, &s.qkv_conv)?;
        pass.level_barrier(&[&s.qkv_conv, conv_window])?;
        gdn_step_gated_fused(
            ctx,
            pass,
            &s.qkv_conv,
            &s.a,
            &s.b,
            &w.a_log,
            &w.dt_bias,
            gdn_state,
            &s.z,
            &w.norm_w,
            &s.gdn_gated,
            self.gdn_scale,
            self.config.linear_num_key_heads,
            self.config.rms_norm_eps,
            self.gdn_gate,
        )?;
        pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
        quant::gemv_quant(ctx, pass, &w.out_proj, &s.gdn_gated, &s.branch_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;
        let idx = &cfg.indexer;
        let len = pos + 1;

        quant::gemv_quant(ctx, pass, &w.qkv_proj, &s.hc.mixed, &s.attn_qkv)?;
        quant::gemv_quant(ctx, pass, &w.indexer.qk_proj, &s.hc.mixed, &s.idx_qk)?;
        pass.level_barrier(&[&s.attn_qkv, &s.idx_qk])?;
        q_norm_rope_split_decode(
            ctx, pass, &s.qg, &w.q_norm, &s.q, &s.gate, rot, pos, theta, eps,
        )?;
        k_norm_rope_scatter_decode(
            ctx, pass, &s.k_new, &w.k_norm, k_cache, rot, pos, theta, eps,
        )?;
        scatter_kv(ctx, pass, v_cache, &s.v_new, pos)?;
        qsa::qsa_prep_q(
            ctx,
            pass,
            &s.idx_qk,
            &w.indexer.q_norm,
            &s.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &s.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[&s.q, &s.gate, k_cache, v_cache, &s.idx_q, idx_keys])?;
        if len.is_multiple_of(idx.compress_ratio) {
            // This token completes a block: its key becomes selectable from
            // the next position on (and this one, beyond the dense limit).
            let block = len / idx.compress_ratio - 1;
            qsa::qsa_block_keys(
                ctx,
                pass,
                idx_keys,
                &w.indexer.k_norm,
                blk_keys,
                idx.compress_ratio,
                block,
                1,
                rot,
                theta,
                eps,
            )?;
            pass.level_barrier(&[blk_keys])?;
        }

        if len <= idx.dense_limit() {
            sdpa_decode(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.attn_o,
                len,
                self.attn_scale,
                Some((&s.sdpa_partials, &s.sdpa_stats)),
            )?;
        } else {
            let nb = qsa::visible_blocks(pos, idx.compress_ratio);
            qsa::qsa_scores(
                ctx,
                pass,
                &s.idx_q,
                blk_keys,
                &s.qsa.scores,
                idx.n_heads,
                nb,
                pos,
                idx.compress_ratio,
            )?;
            pass.level_barrier(&[&s.qsa.scores])?;
            qsa::qsa_select_blocks(
                ctx,
                pass,
                &s.qsa.scores,
                &s.qsa.sel,
                &s.qsa.n_sel,
                1,
                nb,
                pos,
                idx.compress_ratio,
                idx.block_topk(),
            )?;
            pass.level_barrier(&[&s.qsa.sel, &s.qsa.n_sel])?;
            qsa::qsa_attention(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.qsa.sel,
                &s.qsa.n_sel,
                &s.attn_o,
                &s.qsa.split_scratch(),
                1,
                idx.block_topk(),
                idx.compress_ratio,
                pos,
                self.attn_scale,
            )?;
        }
        pass.level_barrier(&[&s.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &s.gate, &s.attn_o, &s.attn_gated)?;
        pass.level_barrier(&[&s.attn_gated])?;
        quant::gemv_quant(ctx, pass, &w.o_proj, &s.attn_gated, &s.branch_out)
    }
}

impl DecodeStateApi for DecodeState {
    fn pos(&self) -> usize {
        self.pos
    }

    fn reset(&mut self) -> Result<()> {
        DecodeState::reset(self)
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// The block selection of the last sparse-attention decode step (query
    /// row 0): `null` until a step ran past the dense limit.
    fn debug_json(&self) -> Result<serde_json::Value> {
        let n = self.qsa.n_sel.to_u32()?[0] as usize;
        let k_max = self.qsa.sel.shape()[1];
        let sel = self.qsa.sel.to_u32()?;
        Ok(serde_json::json!({ "selected_blocks": &sel[..n.min(k_max)] }))
    }
}

impl LanguageModel for Qwen4ExpModel {
    type State = DecodeState;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "Qwen3.8-Flash-Next";

    fn load(ctx: &MetalContext, dir: &Path) -> Result<Self> {
        Qwen4ExpModel::load(ctx, dir)
    }

    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        self.config.eos_token_id.as_vec()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn new_state(&self, ctx: &MetalContext, max_seq: usize) -> Result<DecodeState> {
        Qwen4ExpModel::new_state(self, ctx, max_seq)
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        Qwen4ExpModel::new_scratch_with_capacity(self, ctx, capacity_tokens)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        Qwen4ExpModel::prefill(self, ctx, state, scratch, tokens)
    }

    fn submit_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<PendingPass<'a>> {
        Qwen4ExpModel::submit_decode_step(self, ctx, state, scratch, slot_in, slot_out)
    }
}
