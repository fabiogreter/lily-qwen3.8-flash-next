//! Qwen Sparse Attention: the lightning indexer (query prep, raw-key cache,
//! block keys, block scores, top-k block selection) and the split-K sparse
//! attention over the selected tokens.

use anyhow::{Result, ensure};

use crate::kernels::attention;
use crate::kernels::{Pos, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/qsa.metal");
const TG: usize = 256;
/// Selected tokens per split of the sparse attention kernel.
pub const QSA_SPLIT: usize = 256;
/// Threads of the per-query selection threadgroup.
const SELECT_TG: usize = 1024;
/// Indexer head dimension the kernels are written for.
pub const INDEXER_D: usize = 128;
/// Attention head dimension the sparse kernel is written for.
const ATTN_D: usize = 256;

/// Complete blocks visible to a query at `pos` (its causal window is `pos+1`).
pub fn visible_blocks(pos: usize, ratio: usize) -> usize {
    (pos + 1) / ratio
}

/// Tokens a query at `pos` attends once `n_blocks` blocks were selected.
pub fn attended_tokens(pos: usize, ratio: usize, n_blocks: usize) -> usize {
    let tail_start = visible_blocks(pos, ratio) * ratio;
    n_blocks * ratio + (pos + 1 - tail_start)
}

/// Splits the sparse kernel needs for the longest possible selection.
pub fn sparse_splits(k_max: usize, ratio: usize) -> usize {
    (k_max * ratio + ratio - 1).div_ceil(QSA_SPLIT)
}

/// `q[m, h, :] = rope(rmsnorm(qk[m, h*D..]) * (1 + w))` at position `base_pos + m`.
#[allow(clippy::too_many_arguments)]
pub fn qsa_prep_q<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qk: &Tensor,
    w: &Tensor,
    q: &Tensor,
    n_heads: usize,
    rot: usize,
    base_pos: impl Into<Pos<'t>>,
    theta: f32,
    eps: f32,
) -> Result<()> {
    let base_pos = base_pos.into();
    let d = INDEXER_D;
    ensure!(
        qk.dtype() == DType::BF16 && qk.numel().is_multiple_of((n_heads + 1) * d),
        "qk must be BF16 [M, (NH+1)*D]"
    );
    let m = qk.numel() / ((n_heads + 1) * d);
    ensure!(
        w.numel() == d && w.dtype() == DType::BF16,
        "q norm weight must be BF16 [D]"
    );
    ensure!(
        q.numel() == m * n_heads * d && q.dtype() == DType::BF16,
        "q must be BF16 [M, NH, D]"
    );
    ensure!(rot.is_multiple_of(2) && rot <= d, "bad rotary dim {rot}");
    let pipeline = ctx.pipeline("qsa_prep_q_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[qk.binding(), w.binding(), q.binding()],
        &[
            Param::U32(d as u32),
            Param::U32(n_heads as u32),
            Param::U32(rot as u32),
            base_pos.param(),
            Param::F32(theta),
            Param::F32(eps),
        ],
        Grid::Threadgroups { groups: (m * n_heads, 1, 1), threadgroup: (d, 1, 1) },
    )
}

/// Appends the raw indexer keys of `qk` (`[M, (NH+1)*D]`) to `cache` (`[max_seq, D]`).
pub fn qsa_scatter_keys<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qk: &Tensor,
    cache: &Tensor,
    n_heads: usize,
    base_pos: impl Into<Pos<'t>>,
) -> Result<()> {
    let base_pos = base_pos.into();
    let d = INDEXER_D;
    ensure!(
        qk.dtype() == DType::BF16 && qk.numel().is_multiple_of((n_heads + 1) * d),
        "qk must be BF16 [M, (NH+1)*D]"
    );
    let m = qk.numel() / ((n_heads + 1) * d);
    ensure!(
        cache.shape().len() == 2
            && cache.shape()[1] == d
            && cache.dtype() == DType::BF16,
        "cache must be BF16 [max_seq, D]"
    );
    ensure!(base_pos.max + m <= cache.shape()[0], "indexer key cache overflow");
    let pipeline = ctx.pipeline("qsa_scatter_keys_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[qk.binding(), cache.binding()],
        &[Param::U32(d as u32), Param::U32(n_heads as u32), base_pos.param()],
        Grid::Threads { grid: (d, m, 1), threadgroup: (d, 1, 1) },
    )
}

/// Builds block keys `first_block .. first_block + count` from the raw-key
/// cache. Both may be GPU-supplied (bounded): the grid covers the largest
/// count and threadgroups past the actual count exit.
#[allow(clippy::too_many_arguments)]
pub fn qsa_block_keys<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cache: &Tensor,
    w: &Tensor,
    blocks: &Tensor,
    ratio: usize,
    first_block: impl Into<Pos<'t>>,
    count: impl Into<Pos<'t>>,
    rot: usize,
    theta: f32,
    eps: f32,
) -> Result<()> {
    let (first_block, count) = (first_block.into(), count.into());
    if count.max == 0 {
        return Ok(());
    }
    let d = INDEXER_D;
    ensure!(
        cache.shape().len() == 2 && cache.shape()[1] == d,
        "cache must be [max_seq, D]"
    );
    ensure!(
        blocks.shape().len() == 2
            && blocks.shape()[1] == d
            && blocks.dtype() == DType::BF16,
        "blocks must be BF16 [max_blocks, D]"
    );
    ensure!(
        (first_block.max + count.max) * ratio <= cache.shape()[0],
        "block keys read past the cache"
    );
    ensure!(first_block.max + count.max <= blocks.shape()[0], "block key store overflow");
    ensure!(
        w.numel() == d && w.dtype() == DType::BF16,
        "k norm weight must be BF16 [D]"
    );
    let pipeline = ctx.pipeline("qsa_block_keys_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[cache.binding(), w.binding(), blocks.binding()],
        &[
            Param::U32(d as u32),
            Param::U32(ratio as u32),
            first_block.param(),
            Param::U32(rot as u32),
            Param::F32(theta),
            Param::F32(eps),
            count.param(),
        ],
        Grid::Threadgroups { groups: (count.max, 1, 1), threadgroup: (d, 1, 1) },
    )
}

/// `scores[qi, b]` for every block visible to query `qi` at `base_pos + qi`;
/// `nb_max` is the score row stride (blocks visible to the last query).
#[allow(clippy::too_many_arguments)]
pub fn qsa_scores<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    blocks: &Tensor,
    scores: &Tensor,
    n_heads: usize,
    nb_max: usize,
    base_pos: impl Into<Pos<'t>>,
    ratio: usize,
) -> Result<()> {
    let base_pos = base_pos.into();
    let d = INDEXER_D;
    ensure!(n_heads <= 4, "scores kernel stages at most 4 indexer heads");
    ensure!(
        q.dtype() == DType::BF16 && q.numel().is_multiple_of(n_heads * d),
        "q must be BF16 [QB, NH, D]"
    );
    let qb = q.numel() / (n_heads * d);
    ensure!(
        nb_max > 0 && blocks.shape()[0] >= nb_max,
        "block keys shorter than nb_max"
    );
    ensure!(
        scores.numel() >= qb * nb_max && scores.dtype() == DType::F32,
        "scores must be F32 [QB, nb_max]"
    );
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let pipeline = ctx.pipeline("qsa_scores_f32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[q.binding(), blocks.binding(), scores.binding()],
        &[
            Param::U32(d as u32),
            Param::U32(n_heads as u32),
            Param::U32(nb_max as u32),
            base_pos.param(),
            Param::U32(ratio as u32),
            Param::F32(inv_sqrt_d),
        ],
        Grid::Threadgroups {
            groups: (nb_max.div_ceil(TG), qb, 1),
            threadgroup: (TG, 1, 1),
        },
    )
}

/// Top-`k_max` visible blocks per query (ascending block order) into `sel`
/// (`U32 [QB, k_max]`) with counts in `n_sel` (`U32 [QB]`).
#[allow(clippy::too_many_arguments)]
pub fn qsa_select_blocks<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    scores: &Tensor,
    sel: &Tensor,
    n_sel: &Tensor,
    qb: usize,
    nb_max: usize,
    base_pos: impl Into<Pos<'t>>,
    ratio: usize,
    k_max: usize,
) -> Result<()> {
    let base_pos = base_pos.into();
    ensure!(
        scores.numel() >= qb * nb_max && scores.dtype() == DType::F32,
        "scores must be F32 [QB, nb_max]"
    );
    ensure!(
        sel.numel() >= qb * k_max && sel.dtype() == DType::U32,
        "sel must be U32 [QB, k_max]"
    );
    ensure!(
        n_sel.numel() >= qb && n_sel.dtype() == DType::U32,
        "n_sel must be U32 [QB]"
    );
    let pipeline = ctx.pipeline("qsa_select_blocks", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[scores.binding(), sel.binding(), n_sel.binding()],
        &[
            Param::U32(nb_max as u32),
            base_pos.param(),
            Param::U32(ratio as u32),
            Param::U32(k_max as u32),
        ],
        Grid::Threadgroups { groups: (qb, 1, 1), threadgroup: (SELECT_TG, 1, 1) },
    )
}

/// Split scratch for [`qsa_attention`]: `partials` F32 `[QB*NQ, splits, D]`,
/// `stats` F32 `[QB*NQ, splits, 2]`.
pub struct SparseSplitScratch<'a> {
    pub partials: &'a Tensor,
    pub stats: &'a Tensor,
}

/// Sparse GQA attention of `qb` queries (`q`: `[QB, NQ, D]`, query `qi` at
/// position `base_pos + qi`) over their selected blocks plus tail, writing
/// `out` (`[QB, NQ, D]`).
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    sel: &Tensor,
    n_sel: &Tensor,
    out: &Tensor,
    scratch: &SparseSplitScratch<'_>,
    qb: usize,
    k_max: usize,
    ratio: usize,
    base_pos: impl Into<Pos<'t>>,
    scale: f32,
) -> Result<()> {
    let base_pos = base_pos.into();
    let (kvh, max_seq, d) =
        (k_cache.shape()[0], k_cache.shape()[1], k_cache.shape()[2]);
    ensure!(d == ATTN_D, "sparse attention is compiled for head dim {ATTN_D}, got {d}");
    ensure!(v_cache.shape() == k_cache.shape(), "k/v cache shape mismatch");
    ensure!(
        q.dtype() == DType::BF16 && q.numel().is_multiple_of(qb * d),
        "q must be BF16 [QB, NQ, D]"
    );
    let nq = q.numel() / (qb * d);
    ensure!(
        nq.is_multiple_of(kvh) && nq / kvh <= 32,
        "NQ {nq} not a small multiple of KVH {kvh}"
    );
    let group = nq / kvh;
    ensure!(out.numel() == q.numel() && out.dtype() == DType::BF16, "out must match q");
    ensure!(
        sel.numel() >= qb * k_max && sel.dtype() == DType::U32,
        "sel must be U32 [QB, k_max]"
    );
    ensure!(
        n_sel.numel() >= qb && n_sel.dtype() == DType::U32,
        "n_sel must be U32 [QB]"
    );
    ensure!(base_pos.max + qb <= max_seq, "queries exceed the cache");
    let splits = sparse_splits(k_max, ratio);
    ensure!(
        scratch.partials.numel() >= qb * nq * splits * d
            && scratch.partials.dtype() == DType::F32,
        "sparse partials scratch too small"
    );
    ensure!(
        scratch.stats.numel() >= qb * nq * splits * 2
            && scratch.stats.dtype() == DType::F32,
        "sparse stats scratch too small"
    );
    let stage1 = ctx.pipeline("qsa_attn_split_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &stage1,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            sel.binding(),
            n_sel.binding(),
            scratch.partials.binding(),
            scratch.stats.binding(),
        ],
        &[
            Param::U32(d as u32),
            Param::U32(max_seq as u32),
            Param::U32(group as u32),
            Param::U32(splits as u32),
            Param::U32(k_max as u32),
            Param::U32(ratio as u32),
            base_pos.param(),
            Param::F32(scale),
            Param::U32(nq as u32),
        ],
        Grid::Threadgroups { groups: (kvh, splits, qb), threadgroup: (TG, 1, 1) },
    )?;
    pass.level_barrier(&[scratch.partials, scratch.stats])?;
    let combine =
        ctx.pipeline("sdpa_decode_combine", attention::SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &combine,
        &[scratch.partials.binding(), scratch.stats.binding(), out.binding()],
        &[&u32_bytes(d), &u32_bytes(splits)],
        Grid::Threadgroups { groups: (qb * nq, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/qsa.rs"]
mod tests;
