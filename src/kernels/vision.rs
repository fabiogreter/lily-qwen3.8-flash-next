//! The vision tower's kernels: the position-embedding blend, the 2-D rotary
//! over fused qkv rows, and full bidirectional attention over one image
//! (`docs/architecture.md`, "The vision tower"; the chain is
//! `tools/reference/VISION.md`, "The tower").

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

pub const SOURCE: &str = include_str!("metal/vision.metal");

/// Head dimension the attention kernel is compiled for.
pub const ATTN_D: usize = 72;
/// Query rows per threadgroup of the attention kernel.
const ATTN_BQ: usize = 32;
const ATTN_THREADS: usize = 128;

/// Checks that `(gh, gw)` is a grid of `n` patches in 2 x 2 merge blocks.
fn check_grid(n: usize, gh: usize, gw: usize) -> Result<()> {
    ensure!(
        gh >= 2 && gw >= 2 && gh.is_multiple_of(2) && gw.is_multiple_of(2),
        "grid ({gh}, {gw}) is not made of 2 x 2 merge blocks"
    );
    ensure!(
        gh * gw == n,
        "grid ({gh}, {gw}) holds {} patches, tensor has {n}",
        gh * gw
    );
    Ok(())
}

/// In place over the patch embedding `x` `[N, H]`: adds the `side x side`
/// position `table` `[side^2, H]` resampled bilinearly (align_corners) to the
/// `(gh, gw)` grid, rows in the block-major patch order. Taps and weights are
/// computed in the kernel from the patch's grid coordinates.
pub fn patch_pos_embed_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    table: &Tensor,
    gh: usize,
    gw: usize,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [N, H], got {:?}", x.shape());
    let (n, h) = (x.shape()[0], x.shape()[1]);
    check_grid(n, gh, gw)?;
    ensure!(
        table.shape().len() == 2 && table.shape()[1] == h,
        "position table {:?} is not [side^2, {h}]",
        table.shape()
    );
    let side = (table.shape()[0] as f64).sqrt() as usize;
    ensure!(side * side == table.shape()[0], "position table rows are not a square");
    for t in [x, table] {
        ensure!(t.dtype() == DType::BF16, "patch_pos_embed expects BF16");
    }
    let pipeline = ctx.pipeline("vision_patch_pos_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), table.binding()],
        &[&u32_bytes(h), &u32_bytes(gh), &u32_bytes(gw), &u32_bytes(side)],
        Grid::Threads { grid: (h, n, 1), threadgroup: (256.min(h), 1, 1) },
    )
}

/// In place over the fused `qkv` `[N, 3H]` projection: applies the 2-D
/// rotary (`heads` heads of `H / heads`, angles from the patch's absolute
/// grid coordinates, `theta`) to the q and k thirds in f32 with
/// `rotate_half` pairing, results back to bf16.
pub fn qkv_rope_2d_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    heads: usize,
    gh: usize,
    gw: usize,
    theta: f32,
) -> Result<()> {
    ensure!(qkv.shape().len() == 2, "qkv must be [N, 3H], got {:?}", qkv.shape());
    let (n, h3) = (qkv.shape()[0], qkv.shape()[1]);
    ensure!(h3.is_multiple_of(3), "qkv width {h3} is not 3H");
    let h = h3 / 3;
    check_grid(n, gh, gw)?;
    ensure!(
        heads > 0 && h.is_multiple_of(heads),
        "H {h} not a multiple of {heads} heads"
    );
    let d = h / heads;
    ensure!(d.is_multiple_of(4), "head dim {d} must split into rotate_half quarters");
    ensure!(qkv.dtype() == DType::BF16, "rope expects BF16");
    let pipeline = ctx.pipeline("vision_qkv_rope_bf16", SOURCE, MslVersion::V3_1)?;
    // One thread per rotate_half pair of q and of k: 2 * H / 2.
    let pairs = h;
    pass.dispatch_with(
        &pipeline,
        &[qkv.binding()],
        &[
            Param::U32(h as u32),
            Param::U32(d as u32),
            Param::U32(gw as u32),
            Param::F32(theta),
        ],
        Grid::Threads { grid: (pairs, n, 1), threadgroup: (256.min(pairs), 1, 1) },
    )
}

/// Full bidirectional attention over all `N` rows of `qkv` `[N, 3H]`
/// (q | k | v, `heads` heads of [`ATTN_D`]), softmax in f32, into `out`
/// `[N, H]`. One tensor-op threadgroup per 32-query tile and head, walking
/// 64-key tiles with an online softmax.
pub fn attention_full_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    out: &Tensor,
    heads: usize,
    scale: f32,
) -> Result<()> {
    ensure!(qkv.shape().len() == 2, "qkv must be [N, 3H], got {:?}", qkv.shape());
    let (n, h3) = (qkv.shape()[0], qkv.shape()[1]);
    ensure!(h3.is_multiple_of(3), "qkv width {h3} is not 3H");
    let h = h3 / 3;
    ensure!(n > 0, "attention over zero rows");
    ensure!(
        heads * ATTN_D == h,
        "H {h} is not {heads} heads of the compiled head dim {ATTN_D}"
    );
    ensure!(out.numel() == n * h, "out numel {} != [{n}, {h}]", out.numel());
    ensure!(
        qkv.dtype() == DType::BF16 && out.dtype() == DType::BF16,
        "attention expects BF16"
    );
    let pipeline = ctx.pipeline("vision_attn_nax_d72", SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[qkv.binding(), out.binding()],
        &[Param::U32(n as u32), Param::U32(h as u32), Param::F32(scale)],
        Grid::Threadgroups {
            groups: (n.div_ceil(ATTN_BQ), heads, 1),
            threadgroup: (ATTN_THREADS, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/vision.rs"]
mod tests;
