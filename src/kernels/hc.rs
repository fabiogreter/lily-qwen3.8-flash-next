//! Hyper-connection (gated residual) kernels: grouped RMSNorm over the residual
//! streams, the read-gate mixing, and the write-gate injection.

use anyhow::{Result, ensure};

use crate::kernels::quant::check_quant;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/hc.metal");
const TG: usize = 256;
/// Streams the fused read-gate kernels are instantiated for (`HC_MAX_G`).
const HC_MAX_G: usize = 8;
/// Output columns per threadgroup of the fused up kernel (one simdgroup each);
/// timing is flat from 4 to 16.
const HC_UP_COLS: usize = 8;

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

/// Validates a Q8 read-gate weight of `[n, k]` for the fused kernels.
fn check_hc_q8(w: &QuantWeights, n: usize, k: usize, what: &str) -> Result<()> {
    let (wn, wk) = check_quant(w)?;
    ensure!(w.bits == 8, "{what} must be Q8, got {} bits", w.bits);
    ensure!((wn, wk) == (n, k), "{what} is [{wn}, {wk}], expected [{n}, {k}]");
    Ok(())
}

/// Fused decode read gate, first half: `down = W_down . rmsnorm_grouped(hyper)`
/// (`[lowrank]`) and, with an inject weight, `inj = W_inject . hn` (`[G]`),
/// plus the `groups` stream `inv_rms` values (F32) the second half reuses.
/// Equivalent to `rmsnorm_grouped_bf16` followed by `gemv_quant` on the two
/// Q8 weights, except that the normalized stream is not rounded to bf16
/// before the dot (results differ at bf16 rounding level; see the kernel).
/// One threadgroup of `32 * groups` threads per output row. `hyper`/`w` are
/// `[groups*h]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_read_down_q8(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    w: &Tensor,
    down: &QuantWeights,
    inject: Option<&QuantWeights>,
    down_out: &Tensor,
    inj_out: &Tensor,
    inv_rms: &Tensor,
    h: usize,
    groups: usize,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    ensure!(
        groups > 0 && groups <= HC_MAX_G,
        "streams {groups} outside 1..={HC_MAX_G}"
    );
    ensure!(h.is_multiple_of(16), "hidden width {h} must be a multiple of 16");
    let k = groups * h;
    ensure!(
        hyper.numel() == k && hyper.dtype() == DType::BF16,
        "hyper must be BF16 [{k}]"
    );
    ensure!(
        w.numel() == k && w.dtype() == DType::BF16,
        "norm weight must be BF16 [{k}]"
    );
    let r = down.out_features();
    check_hc_q8(down, r, k, "read-gate down weight")?;
    ensure!(
        down_out.numel() == r && down_out.dtype() == DType::BF16,
        "down output must be BF16 [{r}]"
    );
    let n_inj = match inject {
        Some(inject) => {
            let n = inject.out_features();
            check_hc_q8(inject, n, k, "write-gate inject weight")?;
            ensure!(
                inject.group_size == down.group_size,
                "inject group size {} != down group size {}",
                inject.group_size,
                down.group_size
            );
            ensure!(
                inj_out.numel() == n && inj_out.dtype() == DType::BF16,
                "inject output must be BF16 [{n}]"
            );
            n
        }
        None => 0,
    };
    ensure!(
        inv_rms.numel() == groups && inv_rms.dtype() == DType::F32,
        "inv_rms must be F32 [{groups}]"
    );
    // Without an inject weight its bindings are never read; bind the down
    // weight (and output) so every slot names a resident buffer.
    let (inj_w, inj_y) = match inject {
        Some(inject) => (inject, inj_out),
        None => (down, down_out),
    };
    let pipeline = ctx.pipeline("hc_read_down_q8", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            hyper.binding(),
            w.binding(),
            down.codes.binding(),
            down.scales.binding(),
            down.biases.binding(),
            inj_w.codes.binding(),
            inj_w.scales.binding(),
            inj_w.biases.binding(),
            down_out.binding(),
            inj_y.binding(),
            inv_rms.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(groups),
            &u32_bytes(r),
            &u32_bytes(n_inj),
            &u32_bytes(down.group_size),
            &eps.to_ne_bytes(),
            &w_bias.to_ne_bytes(),
        ],
        Grid::Threadgroups {
            groups: (r + n_inj, 1, 1),
            threadgroup: (32 * groups, 1, 1),
        },
    )
}

/// Fused decode read gate, second half: `mixed[i] = mean_g sigmoid(up[g*h+i])
/// * hn[g*h+i]` with `up = W_up . silu(down / groups)` and `hn` recomputed
/// from `hyper`, `inv_rms` (from [`hc_read_down_q8`]) and the norm weight.
/// Given the same `down` and `inv_rms`, bit-identical to `silu_scaled_bf16`,
/// `gemv_quant` and `hc_mix_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn hc_read_up_mix_q8(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    up: &QuantWeights,
    down: &Tensor,
    hyper: &Tensor,
    w: &Tensor,
    inv_rms: &Tensor,
    mixed: &Tensor,
    h: usize,
    groups: usize,
    w_bias: f32,
) -> Result<()> {
    // One instantiation per stream count: the per-stream registers need a
    // compile-time bound.
    let name: &'static str = match groups {
        1 => "hc_read_up_mix_q8_g1",
        2 => "hc_read_up_mix_q8_g2",
        3 => "hc_read_up_mix_q8_g3",
        4 => "hc_read_up_mix_q8_g4",
        5 => "hc_read_up_mix_q8_g5",
        6 => "hc_read_up_mix_q8_g6",
        7 => "hc_read_up_mix_q8_g7",
        8 => "hc_read_up_mix_q8_g8",
        _ => anyhow::bail!("streams {groups} outside 1..={HC_MAX_G}"),
    };
    let k = groups * h;
    let r = down.numel();
    ensure!(down.dtype() == DType::BF16, "down logits must be BF16");
    check_hc_q8(up, k, r, "read-gate up weight")?;
    ensure!(
        hyper.numel() == k && hyper.dtype() == DType::BF16,
        "hyper must be BF16 [{k}]"
    );
    ensure!(
        w.numel() == k && w.dtype() == DType::BF16,
        "norm weight must be BF16 [{k}]"
    );
    ensure!(
        inv_rms.numel() == groups && inv_rms.dtype() == DType::F32,
        "inv_rms must be F32 [{groups}]"
    );
    ensure!(
        mixed.numel() == h && mixed.dtype() == DType::BF16,
        "mixed must be BF16 [{h}]"
    );
    let inv_g = 1.0 / groups as f32;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            down.binding(),
            hyper.binding(),
            w.binding(),
            inv_rms.binding(),
            mixed.binding(),
        ],
        &[
            &u32_bytes(h),
            &u32_bytes(r),
            &u32_bytes(up.group_size),
            &inv_g.to_ne_bytes(),
            &w_bias.to_ne_bytes(),
        ],
        Grid::Threadgroups {
            groups: (h.div_ceil(HC_UP_COLS), 1, 1),
            threadgroup: (HC_UP_COLS * 32, 1, 1),
        },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/hc.rs"]
mod tests;
