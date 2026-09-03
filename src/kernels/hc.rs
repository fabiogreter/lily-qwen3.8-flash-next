//! Hyper-connection (gated residual) kernels: grouped RMSNorm over the residual
//! streams, the read-gate mixing, and the write-gate injection.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/hc.metal");
const TG: usize = 256;

fn bf16_rows(t: &Tensor, cols: usize, what: &str) -> Result<usize> {
    ensure!(t.dtype() == DType::BF16, "{what} must be BF16");
    ensure!(
        t.numel().is_multiple_of(cols),
        "{what} numel {} not [rows, {cols}]",
        t.numel()
    );
    Ok(t.numel() / cols)
}

/// RMSNorm over each of the `groups` segments of width `h` in every row of
/// `x` (`[rows, groups*h]`), with gain `w_bias + w[g*h + i]`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_grouped_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    h: usize,
    groups: usize,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    ensure!(groups > 0 && h > 0, "empty grouped norm");
    let rows = bf16_rows(x, groups * h, "grouped norm input")?;
    ensure!(
        w.numel() == groups * h && w.dtype() == DType::BF16,
        "weight must be BF16 [G*H]"
    );
    ensure!(out.numel() == x.numel() && out.dtype() == DType::BF16, "output mismatch");
    let pipeline = ctx.pipeline("rmsnorm_grouped_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(h), &u32_bytes(groups), &eps.to_ne_bytes(), &w_bias.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows * groups, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

/// `out = silu(x * inv_scale)`.
pub fn silu_scaled_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    out: &Tensor,
    inv_scale: f32,
) -> Result<()> {
    let n = x.numel();
    ensure!(n > 0 && out.numel() == n, "size mismatch");
    ensure!(
        x.dtype() == DType::BF16 && out.dtype() == DType::BF16,
        "silu_scaled expects BF16"
    );
    let pipeline = ctx.pipeline("silu_scaled_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), out.binding()],
        &[&inv_scale.to_ne_bytes()],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (TG.min(n), 1, 1) },
    )
}

/// `mixed[r, i] = mean_g sigmoid(up[r, g*h+i]) * hn[r, g*h+i]`.
pub fn hc_mix_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    up: &Tensor,
    hn: &Tensor,
    mixed: &Tensor,
    h: usize,
    groups: usize,
) -> Result<()> {
    let rows = bf16_rows(up, groups * h, "read-gate logits")?;
    ensure!(hn.numel() == up.numel() && hn.dtype() == DType::BF16, "hn mismatch");
    ensure!(
        mixed.numel() == rows * h && mixed.dtype() == DType::BF16,
        "mixed must be BF16 [rows, h]"
    );
    let pipeline = ctx.pipeline("hc_mix_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[up.binding(), hn.binding(), mixed.binding()],
        &[&u32_bytes(h), &u32_bytes(groups)],
        Grid::Threads { grid: (h, rows, 1), threadgroup: (TG.min(h), 1, 1) },
    )
}

/// `hyper[r, g*h+i] += branch[r, i] * 2*sigmoid(inj[r, g] / groups)`.
pub fn hc_inject_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    branch: &Tensor,
    inj: &Tensor,
    h: usize,
    groups: usize,
) -> Result<()> {
    let rows = bf16_rows(hyper, groups * h, "hyper stream")?;
    ensure!(
        branch.numel() == rows * h && branch.dtype() == DType::BF16,
        "branch must be BF16 [rows, h]"
    );
    ensure!(
        inj.numel() == rows * groups && inj.dtype() == DType::BF16,
        "inject logits must be BF16 [rows, G]"
    );
    let inv_g = 1.0 / groups as f32;
    let pipeline = ctx.pipeline("hc_inject_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[hyper.binding(), branch.binding(), inj.binding()],
        &[&u32_bytes(h), &u32_bytes(groups), &inv_g.to_ne_bytes()],
        Grid::Threads { grid: (groups * h, rows, 1), threadgroup: (TG, 1, 1) },
    )
}

/// `hyper[r, g*h+i] = x[r, i]` for every stream `g`.
pub fn hc_broadcast_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    hyper: &Tensor,
    h: usize,
    groups: usize,
) -> Result<()> {
    let rows = bf16_rows(x, h, "stream source")?;
    ensure!(
        hyper.numel() == rows * groups * h && hyper.dtype() == DType::BF16,
        "hyper mismatch"
    );
    let pipeline = ctx.pipeline("hc_broadcast_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), hyper.binding()],
        &[&u32_bytes(h), &u32_bytes(groups)],
        Grid::Threads { grid: (groups * h, rows, 1), threadgroup: (TG, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/hc.rs"]
mod tests;
