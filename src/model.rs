//! The Qwen3.5 model graph: single-token decode steps plus a batched prefill
//! path that processes `PREFILL_CHUNK` prompt tokens per dispatch round (GEMM
//! projections, in-kernel token loops for the conv/GDN recurrences, causal
//! SDPA), reading the weights once per chunk instead of once per token.

use anyhow::{Result, ensure};
use std::path::Path;

use crate::config::TextConfig;
use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi};
use crate::kernels::attention::{
    MAX_SEQ, k_norm_rope_scatter_decode, q_norm_rope_split_decode, rope_neox,
    scatter_kv, sdpa_decode, sdpa_prefill, sdpa_split_scratch_splits, split_q_gate,
};
use crate::kernels::elementwise::{
    add_bf16, gather_row_bf16, sigmoid_mul_bf16,
};
use crate::kernels::gdn::{
    GDN_HEAD_DIM, GDN_STATE_DTYPE, GdnGate, GdnRegscanStaging, conv1d_prefill,
    conv1d_step, gated_rmsnorm, gdn_prefill, gdn_step_gated_fused,
};
use crate::kernels::norm::{add_rmsnorm_bf16, rmsnorm_bf16};
use crate::kernels::{quant, skinny};
use crate::kernels::sample::{SamplerScratch, sample_f32};
use crate::metal::{BlitCopy, ComputePass, EncodedPass, MetalContext};
use crate::moe_ffn::{
    DecodeMoeIo, MoeDims, MoeScratch, PrefillMoeIo, PrefillMoeScratch, decode_moe,
    prefill_moe, prefix_rows, project_mat, project_stack_or_slices,
};
use crate::tensor::{DType, Tensor};
use crate::weights::{self, AttnWeights, GdnWeights, LayerWeights, ModelWeights};

/// Qwen3.5's standard RMSNorms are zero-centered: gain = 1 + weight. (The GDN
/// GatedNorm is the exception and uses a plain gain.)
const NORM_WEIGHT_BIAS: f32 = 1.0;

/// Prompt tokens processed by one prefill command buffer.
const PREFILL_CHUNK: usize = 4096;

fn moe_dims(cfg: &TextConfig) -> MoeDims {
    MoeDims {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        moe_intermediate: cfg.moe_intermediate_size,
        hidden: cfg.hidden_size,
        norm_topk_prob: cfg.norm_topk_prob,
    }
}

pub struct Qwen3_5Model {
    pub config: TextConfig,
    weights: ModelWeights,
    attn_scale: f32,
    gdn_scale: f32,
}

enum LayerState {
    Gdn {
        /// Recurrent state, fp32 `[H, 128, 128]`.
        state: Tensor,
        /// Double-buffered conv window, bf16 `[C, KD-1]` each: a prefill
        /// chunk's conv kernel reads one buffer and writes the other (an
        /// in-place update would race across its threadgroups), selected by
        /// `DecodeState::conv_slot`.
        conv_windows: [Tensor; 2],
    },
    Full {
        /// `[KVH, max_seq, D]` bf16.
        k_cache: Tensor,
        v_cache: Tensor,
    },
}

pub struct DecodeState {
    pub pos: usize,
    /// Tokens the KV caches hold; grows in `CAPACITY_STEP`s.
    capacity: usize,
    layers: Vec<LayerState>,
    kv_heads: usize,
    head_dim: usize,
    /// Which `conv_windows` buffer holds the current cross-chunk conv state
    /// (all GDN layers advance in lockstep): prefill chunks read it, write
    /// the other, then flip; decode steps update it in place.
    conv_slot: usize,
}

impl DecodeState {
    /// Returns the state to position zero with zeroed GDN recurrent state
    /// and conv windows, so a session slot can recycle its buffers instead
    /// of reallocating. KV needs no clearing: rows past `pos` are never
    /// read and get overwritten. The GPU must be idle on this state's buffers.
    pub fn reset(&mut self) {
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                state.zero_fill();
                conv_windows[0].zero_fill();
                conv_windows[1].zero_fill();
            }
        }
        self.pos = 0;
        self.conv_slot = 0;
    }

    fn kv_caches(ctx: &MetalContext, kv_heads: usize, head_dim: usize, capacity: usize) -> Result<LayerState> {
        Ok(LayerState::Full {
            k_cache: Tensor::zeros(ctx, &[kv_heads, capacity, head_dim], DType::BF16)?,
            v_cache: Tensor::zeros(ctx, &[kv_heads, capacity, head_dim], DType::BF16)?,
        })
    }
}

/// KV caches grow in steps of this many tokens.
const CAPACITY_STEP: usize = 8192;

fn round_capacity(tokens: usize) -> Result<usize> {
    ensure!(tokens <= MAX_SEQ, "capacity {tokens} exceeds kernel limit {MAX_SEQ}");
    Ok(tokens.max(1).div_ceil(CAPACITY_STEP).saturating_mul(CAPACITY_STEP).min(MAX_SEQ))
}

/// GDN states and conv windows at one position.
pub struct Snapshot {
    pos: usize,
    gdn: Vec<(Tensor, Tensor)>,
}

impl SnapshotApi for Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.gdn.iter().map(|(a, b)| a.byte_len() + b.byte_len()).sum()
    }
}

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let out = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    ctx.blit_copy(&[BlitCopy { src: t, src_offset: 0, dst: &out, dst_offset: 0, len: t.byte_len() }])?;
    Ok(out)
}

/// Copies the first `rows` rows of every `[heads, cap, d]` head block.
fn head_block_copies<'t>(src: &'t Tensor, dst: &'t Tensor, rows: usize, out: &mut Vec<BlitCopy<'t>>) -> Result<()> {
    let (heads, src_cap, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let dst_cap = dst.shape()[1];
    ensure!(rows <= src_cap && rows <= dst_cap && dst.shape()[0] == heads && dst.shape()[2] == d, "cache copy shape mismatch");
    let row_bytes = d * src.dtype().size();
    for h in 0..heads {
        out.push(BlitCopy {
            src,
            src_offset: h * src_cap * row_bytes,
            dst,
            dst_offset: h * dst_cap * row_bytes,
            len: rows * row_bytes,
        });
    }
    Ok(())
}

/// Per-step intermediates, allocated once. The projection outputs that share
/// one fused matvec (`gdn_in`, `attn_qkv`, `mlp_gu`) are single buffers whose
/// named segments below are views.
pub struct Scratch {
    x: Tensor,
    normed: Tensor,
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
    logits: Tensor,
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
    sampler: SamplerScratch,
    /// Tokens written by the in-graph sampler (`U32[2]`): two ping-pong
    /// slots so the pipelined loop can host-read step N's token while the
    /// in-flight step N+1 writes the other slot.
    pub next_token: Tensor,
    /// Prefill staging for the largest quantized projection.
    dequant: Tensor,
    /// Sparse-MoE decode intermediates.
    moe: MoeScratch,
    /// Session-lived batched prefill intermediates, grown on demand and capped
    /// at `PREFILL_CHUNK` rows.
    prefill: Option<PrefillScratch>,
}

/// Batched prefill intermediates, owned by [`Scratch`] at a capacity that only
/// grows. Each chunk uses exact row-prefix views of these buffers.
struct PrefillScratch {
    m: usize,
    ids: Tensor,
    /// Router logits `[m, E]` + shared-expert staging.
    moe: PrefillMoeScratch,
    x: Tensor,
    normed: Tensor,
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
    /// Register-scan staging: normalized q/k rows and precomputed per-token
    /// gates.
    gdn_stage: GdnStageScratch,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    /// The fused-stack GEMM output (`project_stack_or_slices` views
    /// `[m, n_total]` prefixes of it per level), sized for the widest
    /// stacked weight group at the largest row count the small-m route serves.
    stack: Tensor,
}

/// The two `[m, ·]` elementwise outputs the regscan prefill path stages per
/// layer·chunk (plus β, which shares the gates pass).
struct GdnStageScratch {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
}

impl PrefillScratch {
    fn new(ctx: &MetalContext, cfg: &TextConfig, capacity: usize) -> Result<Self> {
        ensure!(capacity > 0, "prefill scratch capacity must be nonzero");
        let m = capacity;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let bf = DType::BF16;
        Ok(Self {
            m,
            ids: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: Tensor::zeros(ctx, &[m, h], bf)?,
            normed: Tensor::zeros(ctx, &[m, h], bf)?,
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
            // Register-scan staging scales with the current row capacity.
            gdn_stage: {
                let hk = cfg.linear_num_key_heads;
                GdnStageScratch {
                    qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], bf)?,
                    decay: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                    beta: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                }
            },
            qg: Tensor::zeros(ctx, &[m, nq * 2 * hd], bf)?,
            q: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            k_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            v_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            attn_o: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[m, nq * hd], bf)?,
            stack: {
                // Rows cap at the largest chunk eligible for fused projection.
                let rows = m.min(skinny::DENSE_SMALLM_THRESHOLD);
                let width = (conv_c + dim_v + 2 * heads)
                    .max((nq * 2 + 2 * nkv) * hd)
                    .max(2 * inter);
                Tensor::zeros(ctx, &[rows, width], bf)?
            },
            moe: PrefillMoeScratch::new(ctx, &moe_dims(cfg), m)?,
        })
    }

    /// Views of the capacity scratch for one chunk of `tokens`, uploading the
    /// token ids (the GPU is idle here: chunks are commit_wait-synchronized).
    /// Reuse across chunks and prefill calls needs no re-zeroing: within a
    /// chunk's pass every buffer is written over the full view extent before
    /// it is read. Counts and cursors are filled explicitly, block maps are
    /// fully rewritten, and union-map reads are bounded by fresh counts.
    fn chunk(&self, tokens: &[u32]) -> Result<Self> {
        let m = tokens.len();
        ensure!(m <= self.m, "chunk of {m} tokens exceeds scratch capacity {}", self.m);
        let ids = prefix_rows(&self.ids, m)?;
        let chunk = Self {
            m,
            ids,
            x: prefix_rows(&self.x, m)?,
            normed: prefix_rows(&self.normed, m)?,
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
            // Kept at full extent (its rows cap at the routing threshold,
            // not the chunk): `project_stack_or_slices` views the exact [m, n_total]
            // prefix per level and never fuses past the threshold.
            stack: self.stack.view(0, self.stack.shape())?,
            moe: self.moe.chunk(m)?,
        };
        chunk.ids.write_bytes(bytemuck::cast_slice(tokens))?;
        Ok(chunk)
    }
}

impl Qwen3_5Model {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        let config = TextConfig::from_model_dir(&dir)?;
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "GDN head dim {} unsupported (kernel is compiled for {GDN_HEAD_DIM})",
            config.linear_key_head_dim,
        );
        let weights = weights::load(ctx, &dir, &config)?;
        let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
        let gdn_scale = 1.0 / (config.linear_key_head_dim as f32).sqrt();
        Ok(Self { config, weights, attn_scale, gdn_scale })
    }

    pub fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<DecodeState> {
        let capacity = round_capacity(capacity)?;
        let c = self.config.gdn_conv_channels();
        let kd = self.config.linear_conv_kernel_dim;
        let heads = self.config.linear_num_value_heads;
        let layers = self
            .weights
            .layers
            .iter()
            .map(|layer| match layer {
                LayerWeights::Gdn(_) => Ok(LayerState::Gdn {
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
                LayerWeights::Full(_) => DecodeState::kv_caches(
                    ctx,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    capacity,
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodeState {
            pos: 0,
            capacity,
            layers,
            kv_heads: self.config.num_key_value_heads,
            head_dim: self.config.head_dim,
            conv_slot: 0,
        })
    }

    /// Scratch whose split-decode buffers are sized for `split_capacity_tokens`
    /// of context rather than for the engine's ceiling.
    ///
    /// This avoids sizing the two `sdpa_*` buffers for the model's full
    /// declared context when the server uses a smaller limit.
    pub fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        split_capacity_tokens: usize,
    ) -> Result<Scratch> {
        // Size for 256-token split routes and the 128-block fixed route.
        let splits = sdpa_split_scratch_splits(split_capacity_tokens);
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let bf = DType::BF16;
        let gdn_in = Tensor::zeros(ctx, &[conv_c + dim_v + 2 * heads], bf)?;
        let attn_qkv = Tensor::zeros(ctx, &[(nq * 2 + 2 * nkv) * hd], bf)?;
        let mlp_gu = Tensor::zeros(ctx, &[2 * inter], bf)?;
        // Largest single dense projection that the batched fallback may stage.
        // Expert stacks use native grouped Q4 kernels and never dequant in full.
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
            h,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(Scratch {
            x: Tensor::zeros(ctx, &[h], bf)?,
            normed: Tensor::zeros(ctx, &[h], bf)?,
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
            gdn_in,
            attn_qkv,
            mlp_gu,
            logits: Tensor::zeros(ctx, &[cfg.vocab_size], DType::F32)?,
            sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
            sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
            sampler: SamplerScratch::new(ctx, self.config.vocab_size)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_numel], bf)?,
            moe: MoeScratch::new(ctx, &moe_dims(cfg))?,
            prefill: None,
        })
    }

    /// Encodes one decode step whose input token is read on-GPU from
    /// `next_token[slot_in]` and whose draw lands in `next_token[slot_out]`,
    /// without committing. This model has no host-side per-step inputs, so
    /// the caller may commit immediately; `pos` advances when it does.
    pub fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        let pass = ctx.begin_concurrent()?;
        pass.set_label("decode");
        self.encode_decode_graph(ctx, &pass, state, s, slot_in, slot_out, draw)?;
        pass.end()
    }

    /// Runs the prompt in batches of `PREFILL_CHUNK` tokens: one command
    /// buffer per chunk, GEMM-shaped projections over all its rows, and
    /// in-kernel token loops for the sequential recurrences. Logits are
    /// produced for the final prompt token only.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prompt");
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        // Power-of-two growth bounds reallocations and total zero-fill work.
        let needed = tokens.len().min(PREFILL_CHUNK);
        let have = s.prefill.as_ref().map_or(0, |p| p.m);
        if have < needed {
            let target = needed.next_power_of_two().min(PREFILL_CHUNK);
            // Drop the old buffers first so peak is max(old, new), not the sum.
            s.prefill = None;
            s.prefill = Some(PrefillScratch::new(ctx, &self.config, target)?);
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
            self.encode_prefill_chunk(ctx, state, s, &ps, if remaining == 0 { draw } else { None })?;
            state.pos += chunk.len();
            // The chunk's conv kernels wrote the other window buffer.
            state.conv_slot = 1 - state.conv_slot;
        }
        Ok(())
    }

    /// Runs one batched prefill chunk (`m = ps.m` tokens) starting at
    /// `state.pos` as a single command buffer (the MoE FFN is GPU-resident,
    /// so no mid-chunk readback splits the pass). When `want_logits`, the
    /// final row is normed and projected into `s.logits`.
    fn encode_prefill_chunk(
        &self,
        ctx: &MetalContext,
        state: &DecodeState,
        s: &Scratch,
        ps: &PrefillScratch,
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        let pass = ctx.begin()?;
        pass.set_label("prefill");
        let m = ps.m;
        ensure!(state.pos + m <= state.capacity, "sequence full ({})", state.capacity);
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let pos = state.pos;
        // `state` is shadowed by the GDN recurrent tensor in the match arms.
        let conv_slot = state.conv_slot;

        quant::gather_rows_q4(ctx, &pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;

        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            let (input_norm, post_norm, ffn) = match layer {
                LayerWeights::Gdn(w) => (&w.input_norm, &w.post_norm, &w.ffn),
                LayerWeights::Full(w) => (&w.input_norm, &w.post_norm, &w.ffn),
            };

            rmsnorm_bf16(
                ctx,
                &pass,
                &ps.x,
                input_norm,
                &ps.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            match (layer, lstate) {
                (LayerWeights::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    project_stack_or_slices(
                        ctx,
                        &pass,
                        &ps.normed,
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
                        &pass,
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
                        &pass,
                        &ps.qkv_conv,
                        &ps.a,
                        &ps.b,
                        &w.a_log,
                        &w.dt_bias,
                        &staging,
                        state,
                        &ps.gdn_out,
                        self.gdn_scale,
                        cfg.linear_num_key_heads,
                    )?;
                    gated_rmsnorm(
                        ctx,
                        &pass,
                        &ps.gdn_out,
                        &ps.z,
                        &w.norm_w,
                        &ps.gdn_gated,
                        eps,
                        GdnGate::Silu,
                    )?;
                    project_mat(
                        ctx,
                        &pass,
                        &ps.gdn_gated,
                        &w.out_proj,
                        &ps.branch_out,
                        &s.dequant,
                    )?;
                }
                (LayerWeights::Full(w), LayerState::Full { k_cache, v_cache }) => {
                    let rot = cfg.rotary_dim();
                    let theta = cfg.rope_parameters.rope_theta;
                    project_stack_or_slices(
                        ctx,
                        &pass,
                        &ps.normed,
                        &w.qkv_proj,
                        &ps.stack,
                        [
                            (&w.q_proj, &ps.qg),
                            (&w.k_proj, &ps.k_new),
                            (&w.v_proj, &ps.v_new),
                        ],
                        &s.dequant,
                    )?;
                    split_q_gate(ctx, &pass, &ps.qg, &ps.q, &ps.gate)?;
                    rmsnorm_bf16(
                        ctx,
                        &pass,
                        &ps.q,
                        &w.q_norm,
                        &ps.q,
                        eps,
                        NORM_WEIGHT_BIAS,
                    )?;
                    rmsnorm_bf16(
                        ctx,
                        &pass,
                        &ps.k_new,
                        &w.k_norm,
                        &ps.k_new,
                        eps,
                        NORM_WEIGHT_BIAS,
                    )?;
                    rope_neox(
                        ctx,
                        &pass,
                        &ps.q,
                        cfg.num_attention_heads,
                        rot,
                        pos,
                        theta,
                    )?;
                    rope_neox(
                        ctx,
                        &pass,
                        &ps.k_new,
                        cfg.num_key_value_heads,
                        rot,
                        pos,
                        theta,
                    )?;
                    scatter_kv(ctx, &pass, k_cache, &ps.k_new, pos)?;
                    scatter_kv(ctx, &pass, v_cache, &ps.v_new, pos)?;
                    sdpa_prefill(
                        ctx,
                        &pass,
                        &ps.q,
                        k_cache,
                        v_cache,
                        &ps.attn_o,
                        pos,
                        self.attn_scale,
                    )?;
                    sigmoid_mul_bf16(ctx, &pass, &ps.gate, &ps.attn_o, &ps.attn_gated)?;
                    project_mat(
                        ctx,
                        &pass,
                        &ps.attn_gated,
                        &w.o_proj,
                        &ps.branch_out,
                        &s.dequant,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            add_bf16(ctx, &pass, &ps.x, &ps.branch_out, &ps.x)?;

            rmsnorm_bf16(
                ctx,
                &pass,
                &ps.x,
                post_norm,
                &ps.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            prefill_moe(
                ctx,
                &pass,
                &moe_dims(cfg),
                ffn,
                &PrefillMoeIo {
                    x: &ps.normed,
                    out: &ps.branch_out,
                    stack: &ps.stack,
                    mlp_gate: &ps.mlp_gate,
                    mlp_up: &ps.mlp_up,
                    mlp_act: &ps.mlp_act,
                    dequant: &s.dequant,
                },
                &ps.moe,
            )?;
            add_bf16(ctx, &pass, &ps.x, &ps.branch_out, &ps.x)?;
        }

        if let Some(draw) = draw {
            // Only the last prompt token feeds decoding: pull its row into the
            // single-token scratch and reuse the decode logits path.
            gather_row_bf16(ctx, &pass, &ps.x, &s.x, m - 1)?;
            rmsnorm_bf16(
                ctx,
                &pass,
                &s.x,
                &self.weights.final_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            quant::gemv_quant(ctx, &pass, &self.weights.lm_head, &s.normed, &s.logits)?;
            // Slot 0 by convention: the pipelined loop's first decode step
            // consumes the prefill token from this slot.
            let out = s.next_token.view(0, &[1])?;
            sample_f32(ctx, &pass, &s.logits, &s.sampler, draw.params, draw.step, &out)?;
        }
        pass.commit_wait()
    }

    /// Encodes one concurrent-dispatch decode graph at `state.pos`. The input
    /// token comes from `next_token[slot_in]`; greedy argmax writes
    /// `next_token[slot_out]` for the next submitted pass.
    #[allow(clippy::too_many_arguments)]
    fn encode_decode_graph(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<()> {
        ensure!(state.pos < state.capacity, "sequence full ({})", state.capacity);
        let eps = self.config.rms_norm_eps;

        let ids = s.next_token.view(slot_in, &[1])?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ids, &s.x)?;
        pass.level_barrier(&[&s.x])?;

        // Every rmsnorm after layer 0's input norm is preceded by a residual
        // add in the graph, so the pairs run as one fused launch: the layer's
        // post-norm fuses with the branch add, and the trailing FFN add fuses
        // with the NEXT layer's input norm (or the final norm when logits are
        // output) — halving the norm-boundary launches per step.
        //
        // The concurrent encoder has no implicit dispatch ordering:
        // `level_barrier` marks every true inter-level data edge. Each branch
        // barriers between its own levels but leaves its final edge to the
        // caller, allowing independent dispatches inside a level to overlap.
        let layers = &self.weights.layers;
        // `state` is shadowed by the GDN recurrent tensor in the match arms.
        let conv_slot = state.conv_slot;
        for (idx, (layer, lstate)) in layers.iter().zip(state.layers.iter()).enumerate()
        {
            let (input_norm, post_norm, ffn) = match layer {
                LayerWeights::Gdn(w) => (&w.input_norm, &w.post_norm, &w.ffn),
                LayerWeights::Full(w) => (&w.input_norm, &w.post_norm, &w.ffn),
            };

            if idx == 0 {
                rmsnorm_bf16(
                    ctx,
                    pass,
                    &s.x,
                    input_norm,
                    &s.normed,
                    eps,
                    NORM_WEIGHT_BIAS,
                )?;
                pass.level_barrier(&[&s.normed])?;
            }
            match (layer, lstate) {
                (LayerWeights::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    self.gdn_branch(ctx, pass, w, s, state, &conv_windows[conv_slot])?;
                }
                (LayerWeights::Full(w), LayerState::Full { k_cache, v_cache }) => {
                    self.attn_branch(ctx, pass, w, s, k_cache, v_cache, state.pos)?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            pass.level_barrier(&[&s.branch_out])?;
            add_rmsnorm_bf16(
                ctx,
                pass,
                &s.x,
                &s.branch_out,
                post_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            // The fused kernel writes both: `x` is the running residual and
            // `normed` is what the FFN reads.
            pass.level_barrier(&[&s.x, &s.normed])?;

            decode_moe(
                ctx,
                pass,
                &moe_dims(&self.config),
                ffn,
                &DecodeMoeIo {
                    x: &s.normed,
                    out: &s.branch_out,
                    mlp_gu: &s.mlp_gu,
                    mlp_gate: &s.mlp_gate,
                    mlp_up: &s.mlp_up,
                    mlp_act: &s.mlp_act,
                },
                &s.moe,
            )?;
            pass.level_barrier(&[&s.branch_out])?;
            let next_norm = if idx + 1 < layers.len() {
                match &layers[idx + 1] {
                    LayerWeights::Gdn(w) => &w.input_norm,
                    LayerWeights::Full(w) => &w.input_norm,
                }
            } else {
                &self.weights.final_norm
            };
            add_rmsnorm_bf16(
                ctx,
                pass,
                &s.x,
                &s.branch_out,
                next_norm,
                &s.normed,
                eps,
                NORM_WEIGHT_BIAS,
            )?;
            pass.level_barrier(&[&s.x, &s.normed])?;
        }

        // The last layer's fused tail already normed `x` with final_norm.
        quant::gemv_quant(ctx, pass, &self.weights.lm_head, &s.normed, &s.logits)?;
        pass.level_barrier(&[&s.logits])?;
        let out = s.next_token.view(slot_out, &[1])?;
        sample_f32(ctx, pass, &s.logits, &s.sampler, draw.params, draw.step, &out)?;
        pass.level_barrier(&[&s.next_token])?;
        Ok(())
    }

    fn gdn_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        gdn_state: &Tensor,
        conv_window: &Tensor,
    ) -> Result<()> {
        // One fused matvec covers qkv | z | a | b (the scratch segments are
        // views of s.gdn_in). Five dispatches in five dependency levels — the
        // GDN branch is a pure chain with no removable edge.
        quant::gemv_quant(ctx, pass, &w.in_proj, &s.normed, &s.gdn_in)?;
        pass.level_barrier(&[&s.gdn_in])?;
        conv1d_step(ctx, pass, conv_window, &s.qkv, &w.conv_w, &s.qkv_conv)?;
        pass.level_barrier(&[&s.qkv_conv, conv_window])?;
        {
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
                GdnGate::Silu,
            )?;
        }
        pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
        quant::gemv_quant(ctx, pass, &w.out_proj, &s.gdn_gated, &s.branch_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        pos: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;

        // One fused matvec covers q(|gate) | k | v.
        quant::gemv_quant(ctx, pass, &w.qkv_proj, &s.normed, &s.attn_qkv)?;
        pass.level_barrier(&[&s.attn_qkv])?;
        {
            q_norm_rope_split_decode(
                ctx, pass, &s.qg, &w.q_norm, &s.q, &s.gate, rot, pos, theta, eps,
            )?;
            k_norm_rope_scatter_decode(
                ctx, pass, &s.k_new, &w.k_norm, k_cache, rot, pos, theta, eps,
            )?;
            scatter_kv(ctx, pass, v_cache, &s.v_new, pos)?;
            pass.level_barrier(&[&s.q, &s.gate, k_cache, v_cache])?;
        }
        sdpa_decode(
            ctx,
            pass,
            &s.q,
            k_cache,
            v_cache,
            &s.attn_o,
            pos + 1,
            self.attn_scale,
            Some((&s.sdpa_partials, &s.sdpa_stats)),
        )?;
        pass.level_barrier(&[&s.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &s.gate, &s.attn_o, &s.attn_gated)?;
        pass.level_barrier(&[&s.attn_gated])?;
        quant::gemv_quant(ctx, pass, &w.o_proj, &s.attn_gated, &s.branch_out)
    }
}

impl DecodeStateApi for DecodeState {
    type Snapshot = Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        DecodeState::reset(self);
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = round_capacity(tokens)?;
        let (kv_heads, head_dim) = (self.kv_heads, self.head_dim);
        for lstate in &mut self.layers {
            if let LayerState::Full { k_cache, v_cache } = lstate {
                let LayerState::Full { k_cache: k2, v_cache: v2 } =
                    DecodeState::kv_caches(ctx, kv_heads, head_dim, capacity)?
                else {
                    unreachable!()
                };
                let mut copies = Vec::new();
                head_block_copies(k_cache, &k2, self.pos, &mut copies)?;
                head_block_copies(v_cache, &v2, self.pos, &mut copies)?;
                ctx.blit_copy(&copies)?;
                drop(copies);
                *k_cache = k2;
                *v_cache = v2;
            }
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerState::Gdn { state, conv_windows } => {
                    state.byte_len() + conv_windows[0].byte_len() + conv_windows[1].byte_len()
                }
                LayerState::Full { k_cache, v_cache } => k_cache.byte_len() + v_cache.byte_len(),
            })
            .sum()
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Snapshot> {
        let mut gdn = Vec::new();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                gdn.push((clone_tensor(ctx, state)?, clone_tensor(ctx, &conv_windows[self.conv_slot])?));
            }
        }
        Ok(Snapshot { pos: self.pos, gdn })
    }

    fn restore(&mut self, ctx: &MetalContext, snapshot: &Snapshot) -> Result<()> {
        ensure!(snapshot.pos <= self.capacity, "snapshot position exceeds capacity");
        let mut copies = Vec::new();
        let mut saved = snapshot.gdn.iter();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                let (s, w) = saved.next().ok_or_else(|| anyhow::anyhow!("snapshot has too few GDN layers"))?;
                copies.push(BlitCopy { src: s, src_offset: 0, dst: state, dst_offset: 0, len: state.byte_len() });
                copies.push(BlitCopy { src: w, src_offset: 0, dst: &conv_windows[0], dst_offset: 0, len: w.byte_len() });
            }
        }
        ensure!(saved.next().is_none(), "snapshot has too many GDN layers");
        ctx.blit_copy(&copies)?;
        self.pos = snapshot.pos;
        self.conv_slot = 0;
        Ok(())
    }

    fn copy_prefix_from(&mut self, ctx: &MetalContext, from: &Self, tokens: usize) -> Result<()> {
        ensure!(tokens <= from.pos, "source state has fed {} tokens, {tokens} requested", from.pos);
        self.ensure_capacity(ctx, tokens)?;
        let mut copies = Vec::new();
        for (dst, src) in self.layers.iter().zip(&from.layers) {
            if let (LayerState::Full { k_cache, v_cache }, LayerState::Full { k_cache: k0, v_cache: v0 }) = (dst, src) {
                head_block_copies(k0, k_cache, tokens, &mut copies)?;
                head_block_copies(v0, v_cache, tokens, &mut copies)?;
            }
        }
        ctx.blit_copy(&copies)
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn logits(&self) -> &Tensor {
        &self.logits
    }

    fn begin_request(&self) {
        self.sampler.reset_counts();
    }
}

impl LanguageModel for Qwen3_5Model {
    type State = DecodeState;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "Qwen3.6-35B-A3B";

    fn load(ctx: &MetalContext, dir: &Path, _options: &LoadOptions) -> Result<Self> {
        Qwen3_5Model::load(ctx, dir)
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

    fn bytes_per_token(&self) -> usize {
        let full = self.weights.layers.iter().filter(|l| matches!(l, LayerWeights::Full(_))).count();
        full * 2 * self.config.num_key_value_heads * self.config.head_dim * 2
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<DecodeState> {
        Qwen3_5Model::new_state(self, ctx, capacity)
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        Qwen3_5Model::new_scratch_with_capacity(self, ctx, capacity_tokens)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        Qwen3_5Model::prefill(self, ctx, state, scratch, tokens, draw)
    }

    fn prepare_step_inputs(&self, _state: &mut DecodeState, _scratch: &Scratch, _token: u32) -> Result<()> {
        Ok(())
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        Qwen3_5Model::encode_decode_step(self, ctx, state, scratch, slot_in, slot_out, draw)
    }
}
