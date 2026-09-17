//! RoPE, KV-cache updates, and scaled dot-product attention.

use anyhow::{Result, ensure};

use crate::kernels::{MROPE_SECTION, Pos, Rope, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

/// Hard kernel limit; checkpoint and CLI limits may be lower.
pub const MAX_SEQ: usize = 262144;

/// Attention Metal source shared with compile tests.
pub const SOURCE: &str = include_str!("metal/attention.metal");

// Head dimension compiled into the tensor-ops kernel.
const FA_D: usize = 256;

// Maximum score row held by the single-pass decoder.
const DECODE_MONO_MAX_T: usize = 4096;

/// Cache positions per split-K chunk.
pub const SDPA_SPLIT: usize = 256;

/// Context length at which split decode folds four query heads per threadgroup.
pub const GQA_FOLD_MIN_CONTEXT: usize = 8192;

const SDPA_MLX_BLOCKS: usize = 128;
const SDPA_FIXED_BLOCK_CROSSOVER: usize = 32768;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SdpaSplitRoute {
    PerHead,
    Gqa,
    FixedBlock,
}

#[allow(clippy::too_many_arguments)]
fn sdpa_split_route(
    len: usize,
    d: usize,
    group: usize,
    nq: usize,
    kvh: usize,
    gqa_fold_min_context: usize,
    fixed_block_min_context: usize,
) -> SdpaSplitRoute {
    let exact_model_shape = d == 256 && group == 8 && nq == kvh * group;
    if len >= fixed_block_min_context && exact_model_shape {
        SdpaSplitRoute::FixedBlock
    } else if len >= gqa_fold_min_context
        && d == 256
        && (1..=8).contains(&group)
        && group.is_multiple_of(4)
        && nq == kvh * group
    {
        SdpaSplitRoute::Gqa
    } else {
        SdpaSplitRoute::PerHead
    }
}

/// Scratch rows required for a context capacity.
pub fn sdpa_split_scratch_splits(capacity_tokens: usize) -> usize {
    capacity_tokens
        .min(SDPA_FIXED_BLOCK_CROSSOVER)
        .div_ceil(SDPA_SPLIT)
        .max(
            usize::from(capacity_tokens >= SDPA_FIXED_BLOCK_CROSSOVER)
                * SDPA_MLX_BLOCKS,
        )
        .max(1)
}

/// In-place partial NeoX RoPE over `[M, heads, D]`; token `m` is sequence
/// index `base_pos + m` (decode is the M=1 case) and rotates at the position
/// `rope` derives from it: the index plus a delta (the scalar kernel, text
/// and generated tokens), or its own 3-axis position under the interleaved
/// M-RoPE (the `_mrope` variant, prefill rows of a prompt with an image).
#[allow(clippy::too_many_arguments)]
pub fn rope_neox<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    heads: usize,
    rot: usize,
    base_pos: impl Into<Pos<'t>>,
    theta: f32,
    rope: Rope<'_>,
) -> Result<()> {
    let base_pos = base_pos.into();
    let d = *x.shape().last().ok_or_else(|| anyhow::anyhow!("rope on 0-d tensor"))?;
    ensure!(x.numel().is_multiple_of(heads * d), "x not a multiple of heads*D");
    let m = x.numel() / (heads * d);
    ensure!(rot.is_multiple_of(2) && rot <= d, "rotary dim {rot} invalid for D {d}");
    let pairs = heads * rot / 2;
    let grid =
        Grid::Threads { grid: (pairs, m, 1), threadgroup: (256.min(pairs), 1, 1) };
    match rope {
        Rope::Delta(delta) => {
            let delta = Rope::delta_i32(delta)?.to_ne_bytes();
            let pipeline = ctx.pipeline("rope_neox_bf16", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_with(
                &pipeline,
                &[x.binding()],
                &[
                    Param::U32(d as u32),
                    Param::U32(rot as u32),
                    base_pos.param(),
                    Param::F32(theta),
                    Param::U32((heads * d) as u32),
                    Param::Bytes(&delta),
                ],
                grid,
            )
        }
        Rope::Rows { positions, base } => {
            let pos_base =
                Rope::check_rows(positions, base, base_pos.min, base_pos.max, m)?;
            let pipeline =
                ctx.pipeline("rope_neox_mrope_bf16", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_with(
                &pipeline,
                &[x.binding(), positions.binding()],
                &[
                    Param::U32(d as u32),
                    Param::U32(rot as u32),
                    base_pos.param(),
                    Param::F32(theta),
                    Param::U32((heads * d) as u32),
                    Param::U32(pos_base as u32),
                    Param::U32((3 * MROPE_SECTION[1]) as u32),
                    Param::U32((3 * MROPE_SECTION[2]) as u32),
                ],
                grid,
            )
        }
    }
}

/// Splits Qwen3.5's per-head-interleaved q_proj output (`[H, 2D]`, each head
/// `[q | gate]`) into `q` and `gate` (`[H, D]` each).
pub fn split_q_gate(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qg: &Tensor,
    q: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    let n = q.numel();
    ensure!(gate.numel() == n && qg.numel() == 2 * n, "split_q_gate size mismatch");
    let d = *q.shape().last().ok_or_else(|| anyhow::anyhow!("0-d q"))?;
    let pipeline = ctx.pipeline("split_q_gate_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qg.binding(), q.binding(), gate.binding()],
        &[&u32_bytes(d)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

/// Appends M tokens' per-head rows (`[M, KVH, D]`) into `cache`
/// (`[KVH, max_seq, D]`) at positions `base_pos..base_pos + M`.
pub fn scatter_kv<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cache: &Tensor,
    rows: &Tensor,
    base_pos: impl Into<Pos<'t>>,
) -> Result<()> {
    let base_pos = base_pos.into();
    let (kvh, max_seq, d) = (cache.shape()[0], cache.shape()[1], cache.shape()[2]);
    ensure!(
        rows.numel().is_multiple_of(kvh * d),
        "rows numel {} not [M, KVH, D]",
        rows.numel()
    );
    let m = rows.numel() / (kvh * d);
    ensure!(
        base_pos.max + m <= max_seq,
        "positions {}+{m} exceed max_seq {max_seq}",
        base_pos.max
    );
    let pipeline = ctx.pipeline("scatter_kv_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[cache.binding(), rows.binding()],
        &[
            Param::U32(d as u32),
            Param::U32(max_seq as u32),
            base_pos.param(),
            Param::U32(kvh as u32),
        ],
        Grid::Threads { grid: (m * kvh * d, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Decode-step Q prep: RMSNorm, the q/gate split and RoPE at rotary position
/// `pos + rope_delta` (`pos` is the sequence index).
#[allow(clippy::too_many_arguments)]
pub fn q_norm_rope_split_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qg: &Tensor,
    w: &Tensor,
    q: &Tensor,
    gate: &Tensor,
    rot: usize,
    pos: usize,
    theta: f32,
    eps: f32,
    rope_delta: i64,
) -> Result<()> {
    let delta = Rope::delta_i32(rope_delta)?.to_ne_bytes();
    let d = *q.shape().last().ok_or_else(|| anyhow::anyhow!("0-d q"))?;
    let heads = q.numel() / d;
    ensure!(d == 256, "decode fused Q prep requires D=256");
    ensure!(rot <= d && rot.is_multiple_of(2), "bad rotary dim {rot}");
    ensure!(qg.numel() == 2 * q.numel(), "qg size mismatch");
    ensure!(gate.numel() == q.numel(), "gate size mismatch");
    ensure!(w.numel() == d, "Q norm weight size mismatch");
    for tensor in [qg, w, q, gate] {
        ensure!(tensor.dtype() == DType::BF16, "fused Q prep requires BF16");
    }
    let pipeline =
        ctx.pipeline("q_norm_rope_split_decode_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qg.binding(), w.binding(), q.binding(), gate.binding()],
        &[
            &u32_bytes(d),
            &u32_bytes(rot),
            &u32_bytes(pos),
            &theta.to_ne_bytes(),
            &eps.to_ne_bytes(),
            &delta,
        ],
        Grid::Threadgroups { groups: (heads, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Decode-step K prep: RMSNorm, RoPE at rotary position `pos + rope_delta`
/// and the scatter into cache slot `pos` (the sequence index).
#[allow(clippy::too_many_arguments)]
pub fn k_norm_rope_scatter_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    k: &Tensor,
    w: &Tensor,
    cache: &Tensor,
    rot: usize,
    pos: usize,
    theta: f32,
    eps: f32,
    rope_delta: i64,
) -> Result<()> {
    let delta = Rope::delta_i32(rope_delta)?.to_ne_bytes();
    let (heads, max_seq, d) = (cache.shape()[0], cache.shape()[1], cache.shape()[2]);
    ensure!(d == 256, "decode fused K prep requires D=256");
    ensure!(rot <= d && rot.is_multiple_of(2), "bad rotary dim {rot}");
    ensure!(k.numel() == heads * d, "K row size mismatch");
    ensure!(w.numel() == d, "K norm weight size mismatch");
    ensure!(pos < max_seq, "position {pos} exceeds cache {max_seq}");
    for tensor in [k, w, cache] {
        ensure!(tensor.dtype() == DType::BF16, "fused K prep requires BF16");
    }
    let pipeline =
        ctx.pipeline("k_norm_rope_scatter_decode_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[k.binding(), w.binding(), cache.binding()],
        &[
            &u32_bytes(d),
            &u32_bytes(rot),
            &u32_bytes(pos),
            &theta.to_ne_bytes(),
            &eps.to_ne_bytes(),
            &u32_bytes(max_seq),
            &delta,
        ],
        Grid::Threadgroups { groups: (heads, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Causal prefill SDPA: `q`/`out` are `[M, NQ, D]`; token `m` attends to cache
/// positions `0..base_len + m + 1` (the chunk's own K/V must already be
/// scattered).
#[allow(clippy::too_many_arguments)]
pub fn sdpa_prefill<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    base_len: impl Into<Pos<'t>>,
    scale: f32,
) -> Result<()> {
    let base_len = base_len.into();
    let (kvh, max_seq, d) =
        (k_cache.shape()[0], k_cache.shape()[1], k_cache.shape()[2]);
    ensure!(q.shape().len() == 3 && q.shape()[2] == d, "q must be [M, NQ, D]");
    let (m, nq) = (q.shape()[0], q.shape()[1]);
    ensure!(out.numel() == q.numel(), "out shape mismatch");
    ensure!(nq.is_multiple_of(kvh), "NQ {nq} not a multiple of KVH {kvh}");
    ensure!(d <= 256, "head dim {d} > 256 unsupported");
    ensure!(
        base_len.max + m <= max_seq && max_seq <= MAX_SEQ,
        "chunk exceeds kernel MAX_SEQ"
    );
    ensure!(v_cache.shape() == k_cache.shape(), "k/v cache shape mismatch");
    // Prefill requires the Metal 4 tensor-ops path.
    ensure!(
        d == FA_D,
        "head dim {d} is not the compiled flash-attention head dim {FA_D}"
    );
    let pipeline = ctx.pipeline(SDPA_PREFILL_KERNEL, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[q.binding(), k_cache.binding(), v_cache.binding(), out.binding()],
        &[
            Param::U32(max_seq as u32),
            base_len.param(),
            Param::U32(m as u32),
            Param::U32(nq as u32),
            Param::U32((nq / kvh) as u32),
            Param::F32(scale),
            Param::U32(1), // parallel softmax
            Param::U32(0), // ascending query blocks
        ],
        Grid::Threadgroups {
            // Geometry must match the selected query tile and simdgroup count.
            groups: (m.div_ceil(SDPA_PREFILL_BQ), nq, 1),
            threadgroup: (SDPA_PREFILL_THREADS, 1, 1),
        },
    )
}

const SDPA_PREFILL_KERNEL: &str = "sdpa_prefill_nax_qdev128";
const SDPA_PREFILL_BQ: usize = 16;
const SDPA_PREFILL_THREADS: usize = 128;

/// Single-token SDPA over `len` cached positions with GQA head mapping.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    len: usize,
    scale: f32,
    split_scratch: Option<(&Tensor, &Tensor)>,
) -> Result<()> {
    sdpa_decode_inner(
        ctx,
        pass,
        q,
        k_cache,
        v_cache,
        out,
        len,
        scale,
        split_scratch,
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    )
}

#[allow(clippy::too_many_arguments)]
fn sdpa_decode_inner(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    len: usize,
    scale: f32,
    split_scratch: Option<(&Tensor, &Tensor)>,
    gqa_fold_min_context: usize,
    fixed_block_min_context: usize,
) -> Result<()> {
    let (kvh, max_seq, d) =
        (k_cache.shape()[0], k_cache.shape()[1], k_cache.shape()[2]);
    let nq = q.numel() / d;
    ensure!(q.numel() == nq * d && out.numel() == nq * d, "q/out shape mismatch");
    ensure!(nq.is_multiple_of(kvh), "NQ {nq} not a multiple of KVH {kvh}");
    ensure!(d <= 256, "head dim {d} > 256 unsupported");
    ensure!(len <= max_seq && max_seq <= MAX_SEQ, "len {len} exceeds kernel MAX_SEQ");
    ensure!(v_cache.shape() == k_cache.shape(), "k/v cache shape mismatch");
    let group = nq / kvh;
    // Split long contexts to expose more threadgroups.
    let split = len > SDPA_SPLIT;
    // Four-head folding requires contiguous GQA groups, D=256, and group <= 8
    // divisible by four.
    let route = sdpa_split_route(
        len,
        d,
        group,
        nq,
        kvh,
        gqa_fold_min_context,
        fixed_block_min_context,
    );
    let fixed_block = route == SdpaSplitRoute::FixedBlock;
    let gqa = route == SdpaSplitRoute::Gqa;
    if split && let Some((partials, stats)) = split_scratch {
        let splits =
            if fixed_block { SDPA_MLX_BLOCKS } else { len.div_ceil(SDPA_SPLIT) };
        ensure!(
            partials.numel() >= nq * splits * d && partials.dtype() == DType::F32,
            "split partials scratch too small"
        );
        ensure!(
            stats.numel() >= nq * splits * 2 && stats.dtype() == DType::F32,
            "split stats scratch too small"
        );

        if fixed_block {
            let stage1 =
                ctx.pipeline("sdpa_decode_mlx_b128h2r2", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_at(
                &stage1,
                &[
                    q.binding(),
                    k_cache.binding(),
                    v_cache.binding(),
                    partials.binding(),
                    stats.binding(),
                ],
                &[
                    &u32_bytes(d),
                    &u32_bytes(max_seq),
                    &u32_bytes(len),
                    &u32_bytes(splits),
                    &u32_bytes(group),
                    &scale.to_ne_bytes(),
                    &u32_bytes(0),
                ],
                Grid::Threadgroups {
                    groups: (kvh, splits, 1),
                    threadgroup: (32 * (group / 2), 1, 1),
                },
            )?;
        } else {
            let name = if gqa {
                "sdpa_decode_split_gqa_h4u_bf16"
            } else {
                "sdpa_decode_split_bf16"
            };
            let stage1 = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
            let groups_x = if gqa { kvh * (group / 4).max(1) } else { nq };
            let bindings = [
                q.binding(),
                k_cache.binding(),
                v_cache.binding(),
                partials.binding(),
                stats.binding(),
            ];
            let grid = Grid::Threadgroups {
                groups: (groups_x, splits, 1),
                threadgroup: (256, 1, 1),
            };
            if gqa {
                pass.dispatch_at(
                    &stage1,
                    &bindings,
                    &[
                        &u32_bytes(d),
                        &u32_bytes(max_seq),
                        &u32_bytes(len),
                        &u32_bytes(splits),
                        &u32_bytes(group),
                        &scale.to_ne_bytes(),
                        &u32_bytes(SDPA_SPLIT),
                    ],
                    grid,
                )?;
            } else {
                pass.dispatch_at(
                    &stage1,
                    &bindings,
                    &[
                        &u32_bytes(d),
                        &u32_bytes(max_seq),
                        &u32_bytes(len),
                        &u32_bytes(splits),
                        &u32_bytes(group),
                        &scale.to_ne_bytes(),
                    ],
                    grid,
                )?;
            }
        }

        // The combine pass consumes stage-one partials.
        pass.level_barrier(&[partials, stats])?;
        let stage2 = ctx.pipeline("sdpa_decode_combine", SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &stage2,
            &[partials.binding(), stats.binding(), out.binding()],
            &[&u32_bytes(d), &u32_bytes(splits)],
            Grid::Threadgroups { groups: (nq, 1, 1), threadgroup: (256, 1, 1) },
        );
    }
    // The single-pass kernel stores the full score row in threadgroup memory.
    ensure!(
        len <= DECODE_MONO_MAX_T,
        "the mono SDPA decode path caps context at {DECODE_MONO_MAX_T} tokens (this \
         request is at {len}); it holds a whole score row in threadgroup memory. \
         The split path handles longer contexts and needs its scratch buffers"
    );
    let pipeline = ctx.pipeline("sdpa_decode_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[q.binding(), k_cache.binding(), v_cache.binding(), out.binding()],
        &[
            &u32_bytes(d),
            &u32_bytes(max_seq),
            &u32_bytes(len),
            &u32_bytes(nq / kvh),
            &scale.to_ne_bytes(),
        ],
        Grid::Threadgroups { groups: (nq, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/attention.rs"]
mod tests;
