//! Qwen Sparse Attention: the lightning indexer (query prep, raw-key cache,
//! block keys, block scores, top-k block selection) and the split-K sparse
//! attention over the selected tokens.

use anyhow::{Result, ensure};

use crate::kernels::attention;
use crate::kernels::{MROPE_SECTION, Pos, Rope, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/qsa.metal");
const TG: usize = 256;
/// Largest split (attended tokens per threadgroup) of the sparse attention
/// kernel: its threadgroup arrays are sized for it.
pub const QSA_SPLIT_MAX: usize = 256;
/// Split for batched sub-batches (prefill), where the query count already
/// supplies the threadgroups.
pub const QSA_SPLIT_BATCHED: usize = 256;
/// Split for small batches (decode, verify): a single query with the
/// batched split is 18 threadgroups on a 40-core GPU.
pub const QSA_SPLIT_SMALL: usize = 64;
/// Smallest split allowed (`LILY_QSA_SPLIT`); the scratch is sized for it.
pub const QSA_SPLIT_MIN: usize = 32;
/// Batches up to this many rows take the small-batch split plan.
pub const QSA_SMALL_ROWS: usize = 4;
/// Query heads the split kernel folds per K/V pass (`QSA_HPP` in the shader).
pub const QSA_HEADS_PER_PASS: usize = 4;
/// Threads of the per-query selection threadgroup.
const SELECT_TG: usize = 256;
/// Threads per tile of the selection union; must match `QSA_UNION_TG`.
const UNION_TG: usize = 1024;
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

/// Splits of `split` tokens the sparse kernel needs for the longest
/// possible selection.
pub fn sparse_splits(k_max: usize, ratio: usize, split: usize) -> usize {
    (k_max * ratio + ratio - 1).div_ceil(split)
}

/// How one [`qsa_attention`] dispatch is cut: `split` attended tokens per
/// threadgroup and `head_groups` threadgroups per KV head (each covering
/// [`QSA_HEADS_PER_PASS`] query heads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSplitPlan {
    pub split: usize,
    pub head_groups: usize,
}

impl SparseSplitPlan {
    /// The plan for `qb` queries over a GQA group of `group` heads: small
    /// batches take the fine split and the head split (`LILY_QSA_SPLIT`
    /// overrides the token count, `LILY_QSA_HEAD_SPLIT=0` the head split),
    /// batched sub-batches keep one split of [`QSA_SPLIT_BATCHED`].
    pub fn for_rows(qb: usize, group: usize) -> Self {
        if qb > QSA_SMALL_ROWS {
            return Self { split: QSA_SPLIT_BATCHED, head_groups: 1 };
        }
        static SMALL: std::sync::OnceLock<(usize, bool)> = std::sync::OnceLock::new();
        let (split, head_split) = *SMALL.get_or_init(|| {
            let split = std::env::var("LILY_QSA_SPLIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .map_or(QSA_SPLIT_SMALL, |v| v.clamp(QSA_SPLIT_MIN, QSA_SPLIT_MAX));
            let head_split =
                std::env::var("LILY_QSA_HEAD_SPLIT").map_or(true, |v| v != "0");
            (split, head_split)
        });
        let head_groups =
            if head_split { group.div_ceil(QSA_HEADS_PER_PASS) } else { 1 };
        Self { split, head_groups }
    }

    pub fn splits(self, k_max: usize, ratio: usize) -> usize {
        sparse_splits(k_max, ratio, self.split)
    }
}

/// `(query, split)` slots the split scratch of a scratch for up to `qb`
/// queries must hold: the batched plan over all of them, or the finest
/// small-batch plan over a small batch, whichever is larger.
pub fn split_scratch_slots(qb: usize, k_max: usize, ratio: usize) -> usize {
    (qb * sparse_splits(k_max, ratio, QSA_SPLIT_BATCHED))
        .max(qb.min(QSA_SMALL_ROWS) * sparse_splits(k_max, ratio, QSA_SPLIT_MIN))
}

/// `q[m, h, :] = rope(rmsnorm(qk[m, h*D..]) * (1 + w))` for sequence index
/// `base_pos + m`, at the rotary position `rope` derives from it (see
/// [`crate::kernels::attention::rope_neox`]).
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
    rope: Rope<'_>,
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
    let grid =
        Grid::Threadgroups { groups: (m * n_heads, 1, 1), threadgroup: (d, 1, 1) };
    match rope {
        Rope::Delta(delta) => {
            let delta = Rope::delta_i32(delta)?.to_ne_bytes();
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
                    Param::Bytes(&delta),
                ],
                grid,
            )
        }
        Rope::Rows { positions, base } => {
            let pos_base =
                Rope::check_rows(positions, base, base_pos.min, base_pos.max, m)?;
            let pipeline =
                ctx.pipeline("qsa_prep_q_mrope_bf16", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_with(
                &pipeline,
                &[qk.binding(), w.binding(), q.binding(), positions.binding()],
                &[
                    Param::U32(d as u32),
                    Param::U32(n_heads as u32),
                    Param::U32(rot as u32),
                    base_pos.param(),
                    Param::F32(theta),
                    Param::F32(eps),
                    Param::U32(pos_base as u32),
                    Param::U32((3 * MROPE_SECTION[1]) as u32),
                    Param::U32((3 * MROPE_SECTION[2]) as u32),
                ],
                grid,
            )
        }
    }
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
/// cache, each roped at the rotary position `rope` derives from its first
/// token's sequence index `block * ratio`. Both bounds may be GPU-supplied
/// (bounded): the grid covers the largest count and threadgroups past the
/// actual count exit.
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
    rope: Rope<'_>,
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
    ensure!(
        first_block.max + count.max <= blocks.shape()[0],
        "block key store overflow"
    );
    ensure!(
        w.numel() == d && w.dtype() == DType::BF16,
        "k norm weight must be BF16 [D]"
    );
    let grid = Grid::Threadgroups { groups: (count.max, 1, 1), threadgroup: (d, 1, 1) };
    match rope {
        Rope::Delta(delta) => {
            let delta = Rope::delta_i32(delta)?.to_ne_bytes();
            let pipeline =
                ctx.pipeline("qsa_block_keys_bf16", SOURCE, MslVersion::V3_1)?;
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
                    Param::Bytes(&delta),
                ],
                grid,
            )
        }
        Rope::Rows { positions, base } => {
            // The last block's first token must have a position row; the
            // first block's must not lie before the buffer.
            let last_first_token = (first_block.max + count.max - 1) * ratio;
            let pos_base = Rope::check_rows(
                positions,
                base,
                first_block.min * ratio,
                last_first_token,
                1,
            )?;
            let pipeline =
                ctx.pipeline("qsa_block_keys_mrope_bf16", SOURCE, MslVersion::V3_1)?;
            pass.dispatch_with(
                &pipeline,
                &[cache.binding(), w.binding(), blocks.binding(), positions.binding()],
                &[
                    Param::U32(d as u32),
                    Param::U32(ratio as u32),
                    first_block.param(),
                    Param::U32(rot as u32),
                    Param::F32(theta),
                    Param::F32(eps),
                    count.param(),
                    Param::U32(pos_base as u32),
                    Param::U32((3 * MROPE_SECTION[1]) as u32),
                    Param::U32((3 * MROPE_SECTION[2]) as u32),
                ],
                grid,
            )
        }
    }
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

/// Split scratch for [`qsa_attention`]: `partials` F32 with at least
/// `QB * NQ * splits * D` elements (indexed `[QB*NQ, splits, D]`), `stats`
/// F32 with at least `QB * NQ * splits * 2` (see [`split_scratch_slots`]).
pub struct SparseSplitScratch<'a> {
    pub partials: &'a Tensor,
    pub stats: &'a Tensor,
}

/// Sparse GQA attention of `qb` queries (`q`: `[QB, NQ, D]`, query `qi` at
/// position `base_pos + qi`) over their selected blocks plus tail, writing
/// `out` (`[QB, NQ, D]`), under the split plan for `qb` rows.
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
    let kvh = k_cache.shape()[0];
    ensure!(
        qb > 0 && q.numel().is_multiple_of(qb * ATTN_D),
        "q must be BF16 [QB, NQ, D]"
    );
    let nq = q.numel() / (qb * ATTN_D);
    ensure!(nq.is_multiple_of(kvh) && kvh > 0, "NQ {nq} not a multiple of KVH {kvh}");
    let plan = SparseSplitPlan::for_rows(qb, nq / kvh);
    qsa_attention_with(
        ctx, pass, q, k_cache, v_cache, sel, n_sel, out, scratch, qb, k_max, ratio,
        base_pos, scale, plan,
    )
}

/// [`qsa_attention`] under an explicit split plan.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_with<'t>(
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
    plan: SparseSplitPlan,
) -> Result<()> {
    qsa_attention_named(
        ctx, pass, q, k_cache, v_cache, sel, n_sel, out, scratch, qb, k_max, ratio,
        base_pos, scale, plan, None,
    )
}

/// [`qsa_attention_with`] through the given `(split, combine)` kernel names
/// (timing variants share the shipped kernels' bindings and grids); `None`
/// takes the shipped ones.
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_named<'t>(
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
    plan: SparseSplitPlan,
    names: Option<(&'static str, &'static str)>,
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
    ensure!(
        (QSA_SPLIT_MIN..=QSA_SPLIT_MAX).contains(&plan.split)
            && plan.split.is_multiple_of(32),
        "split of {} tokens: the kernel takes 32..={QSA_SPLIT_MAX} in steps of 32",
        plan.split
    );
    ensure!(
        plan.head_groups == 1 || plan.head_groups == group.div_ceil(QSA_HEADS_PER_PASS),
        "{} head groups for a GQA group of {group}: 1 or ceil(group / {QSA_HEADS_PER_PASS})",
        plan.head_groups
    );
    let splits = plan.splits(k_max, ratio);
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
    let (split_name, combine_name) =
        names.unwrap_or(("qsa_attn_split_bf16", "sdpa_decode_combine"));
    let stage1 = ctx.pipeline(split_name, SOURCE, MslVersion::V3_1)?;
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
            Param::U32(plan.split as u32),
            Param::U32(plan.head_groups as u32),
        ],
        Grid::Threadgroups {
            groups: (kvh * plan.head_groups, splits, qb),
            threadgroup: (TG, 1, 1),
        },
    )?;
    pass.level_barrier(&[scratch.partials, scratch.stats])?;
    let combine = ctx.pipeline(combine_name, attention::SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &combine,
        &[scratch.partials.binding(), scratch.stats.binding(), out.binding()],
        &[&u32_bytes(d), &u32_bytes(splits)],
        Grid::Threadgroups { groups: (qb * nq, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

// --- Tiled sparse attention (prefill past the dense limit) ----------------------

/// How the batched path attends past the dense limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseAttnRoute {
    /// The per-query split kernel (`qsa_attn_split_bf16`).
    Split,
    /// The tensor-op tile kernel over per-tile unions of selected blocks,
    /// `heads_per_pass` query heads (1, 2 or 4) per staged K/V slice.
    Tiled { heads_per_pass: usize },
}

impl SparseAttnRoute {
    /// `LILY_QSA_ROUTE`: `tile1` (the default: measured fastest at 8K and
    /// 32K, see docs/architecture.md), `tile2`, `tile4` or `split` (the
    /// per-query kernel).
    pub fn from_env() -> Self {
        match std::env::var("LILY_QSA_ROUTE").as_deref() {
            Ok("tile1") | Err(_) => Self::Tiled { heads_per_pass: 1 },
            Ok("tile2") => Self::Tiled { heads_per_pass: 2 },
            Ok("tile4") => Self::Tiled { heads_per_pass: 4 },
            Ok("split") => Self::Split,
            Ok(other) => {
                eprintln!("LILY_QSA_ROUTE={other}: unknown route, using tile1");
                Self::Tiled { heads_per_pass: 1 }
            }
        }
    }

    /// The route for a sub-batch of `rows` queries: a tile needs a full
    /// tile's worth of queries to pay for building the union, so smaller
    /// batches (the verify pass) keep the split kernel.
    pub fn for_rows(self, rows: usize) -> Self {
        match self {
            Self::Tiled { .. } if rows >= QSA_TILE_BQ => self,
            _ => Self::Split,
        }
    }
}

/// Consecutive queries per tile of the tiled sparse attention kernel.
pub const QSA_TILE_BQ: usize = 16;
/// Blocks the tail region of a tile can span (`(QSA_TILE_BQ - 1 + ratio - 1) / ratio + 1`).
pub const QSA_TILE_TAIL_BLOCKS: usize = 8;
/// Threads of the tiled attention threadgroup (four simdgroups).
const QSA_TILE_THREADS: usize = 128;

/// Union entries a tile can hold: every block below the tile's window or
/// every query's full selection, whichever is smaller.
pub fn tile_union_capacity(max_blocks: usize, k_max: usize) -> usize {
    max_blocks.min(QSA_TILE_BQ * k_max).max(1)
}

/// Scratch of [`qsa_tile_union`] and [`qsa_attention_tiled`] for up to
/// `tiles` tiles: the per-tile mask rows (`U32 [tiles, max_blocks]`, zero
/// between dispatches), the compacted unions (`U32 [tiles, cap]` block ids
/// and query masks), their counts, the tail-region masks and the running
/// union-size statistics.
pub struct SparseTileScratch {
    pub mask: Tensor,
    pub union_blk: Tensor,
    pub union_mask: Tensor,
    pub n_union: Tensor,
    pub tail_mask: Tensor,
    /// `U32 [2]`: union blocks summed over every tile built, tiles built.
    pub stats: Tensor,
}

impl SparseTileScratch {
    pub fn new(
        ctx: &MetalContext,
        qb: usize,
        max_blocks: usize,
        k_max: usize,
    ) -> Result<Self> {
        let tiles = qb.div_ceil(QSA_TILE_BQ).max(1);
        let cap = tile_union_capacity(max_blocks, k_max);
        Ok(Self {
            mask: Tensor::zeros(ctx, &[tiles, max_blocks.max(1)], DType::U32)?,
            union_blk: Tensor::zeros(ctx, &[tiles, cap], DType::U32)?,
            union_mask: Tensor::zeros(ctx, &[tiles, cap], DType::U32)?,
            n_union: Tensor::zeros(ctx, &[tiles], DType::U32)?,
            tail_mask: Tensor::zeros(ctx, &[tiles, QSA_TILE_TAIL_BLOCKS], DType::U32)?,
            stats: Tensor::zeros(ctx, &[2], DType::U32)?,
        })
    }

    /// The same buffers under fresh handles (for per-chunk scratch views).
    pub fn share(&self) -> Result<Self> {
        let v = |t: &Tensor| t.view(0, t.shape());
        Ok(Self {
            mask: v(&self.mask)?,
            union_blk: v(&self.union_blk)?,
            union_mask: v(&self.union_mask)?,
            n_union: v(&self.n_union)?,
            tail_mask: v(&self.tail_mask)?,
            stats: v(&self.stats)?,
        })
    }

    pub fn tiles(&self) -> usize {
        self.n_union.numel()
    }

    pub fn max_blocks(&self) -> usize {
        self.mask.shape()[1]
    }

    pub fn cap(&self) -> usize {
        self.union_blk.shape()[1]
    }

    /// Union blocks summed over every tile built so far, and the tile count
    /// (the mean union size per tile is their quotient). Reads the GPU
    /// buffer, so the pass that built the unions must have completed.
    pub fn union_stats(&self) -> Result<(u64, u64)> {
        let v = self.stats.to_u32()?;
        Ok((u64::from(v[0]), u64::from(v[1])))
    }
}

/// Builds each tile's union of selected blocks from `sel`/`n_sel` (the
/// [`qsa_select_blocks`] output for `qb` queries at `base_pos..`).
#[allow(clippy::too_many_arguments)]
pub fn qsa_tile_union<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sel: &Tensor,
    n_sel: &Tensor,
    tiles: &SparseTileScratch,
    qb: usize,
    k_max: usize,
    ratio: usize,
    base_pos: impl Into<Pos<'t>>,
) -> Result<()> {
    let base_pos = base_pos.into();
    ensure!(qb > 0, "no queries");
    ensure!(
        sel.numel() >= qb * k_max && sel.dtype() == DType::U32,
        "sel must be U32 [QB, k_max]"
    );
    ensure!(
        n_sel.numel() >= qb && n_sel.dtype() == DType::U32,
        "n_sel must be U32 [QB]"
    );
    let n_tiles = qb.div_ceil(QSA_TILE_BQ);
    ensure!(
        n_tiles <= tiles.tiles(),
        "tile scratch holds {} tiles, {qb} queries need {n_tiles}",
        tiles.tiles()
    );
    ensure!(
        visible_blocks(base_pos.max + qb - 1, ratio) <= tiles.max_blocks(),
        "tile mask rows shorter than the visible blocks"
    );
    let pipeline = ctx.pipeline("qsa_tile_union", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[
            sel.binding(),
            n_sel.binding(),
            tiles.mask.binding(),
            tiles.union_blk.binding(),
            tiles.union_mask.binding(),
            tiles.n_union.binding(),
            tiles.tail_mask.binding(),
            tiles.stats.binding(),
        ],
        &[
            Param::U32(qb as u32),
            Param::U32(k_max as u32),
            Param::U32(ratio as u32),
            base_pos.param(),
            Param::U32(tiles.max_blocks() as u32),
            Param::U32(tiles.cap() as u32),
        ],
        Grid::Threadgroups { groups: (n_tiles, 1, 1), threadgroup: (UNION_TG, 1, 1) },
    )
}

/// Sparse GQA attention of `qb` queries (`q`: `[QB, NQ, D]`, query `qi` at
/// position `base_pos + qi`) through the tensor-op tile kernel over the
/// unions [`qsa_tile_union`] built, writing `out` (`[QB, NQ, D]`). Each
/// threadgroup stages a K/V slice once for `heads_per_pass` query heads (1, 2
/// or 4, dividing the GQA group).
#[allow(clippy::too_many_arguments)]
pub fn qsa_attention_tiled<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    tiles: &SparseTileScratch,
    out: &Tensor,
    qb: usize,
    ratio: usize,
    base_pos: impl Into<Pos<'t>>,
    scale: f32,
    heads_per_pass: usize,
) -> Result<()> {
    let base_pos = base_pos.into();
    let (kvh, max_seq, d) =
        (k_cache.shape()[0], k_cache.shape()[1], k_cache.shape()[2]);
    ensure!(
        d == ATTN_D,
        "tiled sparse attention is compiled for head dim {ATTN_D}, got {d}"
    );
    ensure!(v_cache.shape() == k_cache.shape(), "k/v cache shape mismatch");
    ensure!(qb > 0, "no queries");
    ensure!(
        q.dtype() == DType::BF16 && q.numel().is_multiple_of(qb * d),
        "q must be BF16 [QB, NQ, D]"
    );
    let nq = q.numel() / (qb * d);
    ensure!(nq.is_multiple_of(kvh), "NQ {nq} not a multiple of KVH {kvh}");
    let group = nq / kvh;
    // `LILY_QSA_TILE_KERNEL` names an alternative instantiation of the tile
    // kernel (timing experiments).
    static FORCED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let forced = FORCED.get_or_init(|| std::env::var("LILY_QSA_TILE_KERNEL").ok());
    let name: &'static str = match heads_per_pass {
        1 => "qsa_attn_tile_nax_h1",
        2 => "qsa_attn_tile_nax_h2",
        4 => "qsa_attn_tile_nax_h4",
        other => anyhow::bail!(
            "{other} heads per pass: the tile kernel exists for 1, 2 and 4"
        ),
    };
    let name: &'static str = match forced {
        Some(f) => Box::leak(f.clone().into_boxed_str()),
        None => name,
    };
    ensure!(
        group.is_multiple_of(heads_per_pass),
        "{heads_per_pass} heads per pass do not divide the GQA group of {group}"
    );
    ensure!(out.numel() == q.numel() && out.dtype() == DType::BF16, "out must match q");
    ensure!(base_pos.max + qb <= max_seq, "queries exceed the cache");
    let n_tiles = qb.div_ceil(QSA_TILE_BQ);
    ensure!(
        n_tiles <= tiles.tiles(),
        "tile scratch holds {} tiles, {qb} queries need {n_tiles}",
        tiles.tiles()
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            tiles.union_blk.binding(),
            tiles.union_mask.binding(),
            tiles.n_union.binding(),
            tiles.tail_mask.binding(),
            out.binding(),
        ],
        &[
            Param::U32(max_seq as u32),
            base_pos.param(),
            Param::U32(qb as u32),
            Param::U32(nq as u32),
            Param::U32(group as u32),
            Param::F32(scale),
            Param::U32(tiles.cap() as u32),
            Param::U32(ratio as u32),
        ],
        Grid::Threadgroups {
            groups: (n_tiles, nq / heads_per_pass, 1),
            threadgroup: (QSA_TILE_THREADS, 1, 1),
        },
    )
}

/// [`qsa_attention_tiled`] through a tile kernel given by name (any
/// instantiation of the tile body; the name is leaked into the pipeline
/// cache). For the timing and comparison tests.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tiled_named<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    tiles: &SparseTileScratch,
    out: &Tensor,
    qb: usize,
    ratio: usize,
    base_pos: impl Into<Pos<'t>>,
    scale: f32,
    name: &str,
) -> Result<()> {
    let base_pos = base_pos.into();
    let (kvh, max_seq, d) =
        (k_cache.shape()[0], k_cache.shape()[1], k_cache.shape()[2]);
    ensure!(
        d == ATTN_D,
        "tiled sparse attention is compiled for head dim {ATTN_D}, got {d}"
    );
    ensure!(v_cache.shape() == k_cache.shape(), "k/v cache shape mismatch");
    ensure!(qb > 0, "no queries");
    ensure!(
        q.dtype() == DType::BF16 && q.numel().is_multiple_of(qb * d),
        "q must be BF16 [QB, NQ, D]"
    );
    let nq = q.numel() / (qb * d);
    ensure!(nq.is_multiple_of(kvh), "NQ {nq} not a multiple of KVH {kvh}");
    let group = nq / kvh;
    // The instantiation's heads per pass, from its name (`_h1`, `_h2`, `_h4`).
    let heads_per_pass = if name.contains("_h4") {
        4
    } else if name.contains("_h2") {
        2
    } else if name.contains("_h1") {
        1
    } else {
        anyhow::bail!(
            "tile kernel name {name} carries no _h1/_h2/_h4 heads-per-pass tag"
        )
    };
    let name: &'static str = Box::leak(name.to_string().into_boxed_str());
    ensure!(out.numel() == q.numel() && out.dtype() == DType::BF16, "out must match q");
    ensure!(base_pos.max + qb <= max_seq, "queries exceed the cache");
    let n_tiles = qb.div_ceil(QSA_TILE_BQ);
    ensure!(
        n_tiles <= tiles.tiles(),
        "tile scratch holds {} tiles, {qb} queries need {n_tiles}",
        tiles.tiles()
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            tiles.union_blk.binding(),
            tiles.union_mask.binding(),
            tiles.n_union.binding(),
            tiles.tail_mask.binding(),
            out.binding(),
        ],
        &[
            Param::U32(max_seq as u32),
            base_pos.param(),
            Param::U32(qb as u32),
            Param::U32(nq as u32),
            Param::U32(group as u32),
            Param::F32(scale),
            Param::U32(tiles.cap() as u32),
            Param::U32(ratio as u32),
        ],
        Grid::Threadgroups {
            groups: (n_tiles, nq / heads_per_pass, 1),
            threadgroup: (QSA_TILE_THREADS, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/qsa.rs"]
mod tests;
