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
/// Rows per dispatch the fused small-batch read-gate kernels
/// ([`hc_read_down_q8_rows`], [`hc_read_up_mix_q8_rows`]) are instantiated
/// for (`HC_MAX_MB`): the verify pass and the draft head's catch-up rows.
pub const HC_FUSED_MAX_ROWS: usize = 4;

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

/// Whether the fused read-gate kernels can run this read: Q8 `down`, `up`
/// (and `inject`) with block-packable group sizes, a hidden width the
/// kernels' 16-element blocks tile, and a stream count they are
/// instantiated for. The batched path checks this before routing small row
/// counts to them.
pub fn fused_read_supported(
    down: &QuantWeights,
    up: &QuantWeights,
    inject: Option<&QuantWeights>,
    h: usize,
    groups: usize,
) -> bool {
    let q8 = |w: &QuantWeights| w.bits == 8 && w.group_size.is_multiple_of(16);
    (1..=HC_MAX_G).contains(&groups)
        && h.is_multiple_of(16)
        && q8(down)
        && q8(up)
        && inject.is_none_or(|w| q8(w) && w.group_size == down.group_size)
}

/// Validates the arguments of the fused down kernels for `rows` rows;
/// returns `(r, n_inj)`, the `down` and `inject` output widths.
#[allow(clippy::too_many_arguments)]
fn check_read_down(
    hyper: &Tensor,
    w: &Tensor,
    down: &QuantWeights,
    inject: Option<&QuantWeights>,
    down_out: &Tensor,
    inj_out: &Tensor,
    inv_rms: &Tensor,
    h: usize,
    groups: usize,
    rows: usize,
) -> Result<(usize, usize)> {
    ensure!(
        groups > 0 && groups <= HC_MAX_G,
        "streams {groups} outside 1..={HC_MAX_G}"
    );
    ensure!(h.is_multiple_of(16), "hidden width {h} must be a multiple of 16");
    let k = groups * h;
    ensure!(
        hyper.numel() == rows * k && hyper.dtype() == DType::BF16,
        "hyper must be BF16 [{rows}, {k}]"
    );
    ensure!(
        w.numel() == k && w.dtype() == DType::BF16,
        "norm weight must be BF16 [{k}]"
    );
    let r = down.out_features();
    check_hc_q8(down, r, k, "read-gate down weight")?;
    ensure!(
        down_out.numel() == rows * r && down_out.dtype() == DType::BF16,
        "down output must be BF16 [{rows}, {r}]"
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
                inj_out.numel() == rows * n && inj_out.dtype() == DType::BF16,
                "inject output must be BF16 [{rows}, {n}]"
            );
            n
        }
        None => 0,
    };
    ensure!(
        inv_rms.numel() == rows * groups && inv_rms.dtype() == DType::F32,
        "inv_rms must be F32 [{rows}, {groups}]"
    );
    Ok((r, n_inj))
}

/// Encodes a fused down kernel (`name`) over the stacked `down`+`inject`
/// rows; shared by the single-row and the small-batch wrappers. The
/// small-batch kernels take one more output, `act` (`[rows, r]`, the scaled
/// SiLU of `down`), and the `1/groups` it needs.
#[allow(clippy::too_many_arguments)]
fn dispatch_read_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    hyper: &Tensor,
    w: &Tensor,
    down: &QuantWeights,
    inject: Option<&QuantWeights>,
    down_out: &Tensor,
    inj_out: &Tensor,
    inv_rms: &Tensor,
    act_out: Option<&Tensor>,
    (r, n_inj): (usize, usize),
    h: usize,
    groups: usize,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    // Without an inject weight its bindings are never read; bind the down
    // weight (and output) so every slot names a resident buffer.
    let (inj_w, inj_y) = match inject {
        Some(inject) => (inject, inj_out),
        None => (down, down_out),
    };
    let mut buffers = vec![
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
    ];
    let inv_g = 1.0 / groups as f32;
    let mut params: Vec<&[u8]> = Vec::with_capacity(8);
    let (h_b, g_b, r_b, n_b, gs_b) = (
        u32_bytes(h),
        u32_bytes(groups),
        u32_bytes(r),
        u32_bytes(n_inj),
        u32_bytes(down.group_size),
    );
    let (eps_b, wb_b, ig_b) =
        (eps.to_ne_bytes(), w_bias.to_ne_bytes(), inv_g.to_ne_bytes());
    params.extend_from_slice(&[&h_b, &g_b, &r_b, &n_b, &gs_b, &eps_b, &wb_b]);
    if let Some(act) = act_out {
        buffers.push(act.binding());
        params.push(&ig_b);
    }
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &buffers,
        &params,
        Grid::Threadgroups {
            groups: (r + n_inj, 1, 1),
            threadgroup: (32 * groups, 1, 1),
        },
    )
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
    let dims = check_read_down(
        hyper, w, down, inject, down_out, inj_out, inv_rms, h, groups, 1,
    )?;
    dispatch_read_down(
        ctx,
        pass,
        "hc_read_down_q8",
        hyper,
        w,
        down,
        inject,
        down_out,
        inj_out,
        inv_rms,
        None,
        dims,
        h,
        groups,
        eps,
        w_bias,
    )
}

/// [`hc_read_down_q8`] over `rows <= HC_FUSED_MAX_ROWS` rows at once:
/// `hyper` is `[rows, groups*h]`, `down_out` `[rows, lowrank]`, `inj_out`
/// `[rows, G]`, `inv_rms` F32 `[rows, groups]`; `act_out` (`[rows, lowrank]`)
/// additionally receives `silu(down / groups)` (`silu_scaled_bf16`'s value)
/// for [`hc_read_up_mix_q8_rows`]. Row `m` of `down`, `inj` and `inv_rms` is
/// bit-identical to the single-row kernel on row `m` of `hyper`.
#[allow(clippy::too_many_arguments)]
pub fn hc_read_down_q8_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    w: &Tensor,
    down: &QuantWeights,
    inject: Option<&QuantWeights>,
    down_out: &Tensor,
    inj_out: &Tensor,
    inv_rms: &Tensor,
    act_out: &Tensor,
    h: usize,
    groups: usize,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    ensure!(
        hyper.shape().len() == 2,
        "hyper must be [rows, G*H], got {:?}",
        hyper.shape()
    );
    let rows = hyper.shape()[0];
    let name: &'static str = match rows {
        1 => "hc_read_down_q8_m1",
        2 => "hc_read_down_q8_m2",
        3 => "hc_read_down_q8_m3",
        4 => "hc_read_down_q8_m4",
        _ => anyhow::bail!("rows {rows} outside 1..={HC_FUSED_MAX_ROWS}"),
    };
    let dims = check_read_down(
        hyper, w, down, inject, down_out, inj_out, inv_rms, h, groups, rows,
    )?;
    ensure!(
        act_out.numel() == down_out.numel() && act_out.dtype() == DType::BF16,
        "act output must be BF16 [{rows}, {}]",
        dims.0
    );
    dispatch_read_down(
        ctx,
        pass,
        name,
        hyper,
        w,
        down,
        inject,
        down_out,
        inj_out,
        inv_rms,
        Some(act_out),
        dims,
        h,
        groups,
        eps,
        w_bias,
    )
}

/// Kernel names of the small-batch up kernels, `[groups - 1][rows - 1]`.
const HC_UP_MIX_ROWS_FNS: [[&str; HC_FUSED_MAX_ROWS]; HC_MAX_G] = [
    [
        "hc_read_up_mix_q8_g1_m1",
        "hc_read_up_mix_q8_g1_m2",
        "hc_read_up_mix_q8_g1_m3",
        "hc_read_up_mix_q8_g1_m4",
    ],
    [
        "hc_read_up_mix_q8_g2_m1",
        "hc_read_up_mix_q8_g2_m2",
        "hc_read_up_mix_q8_g2_m3",
        "hc_read_up_mix_q8_g2_m4",
    ],
    [
        "hc_read_up_mix_q8_g3_m1",
        "hc_read_up_mix_q8_g3_m2",
        "hc_read_up_mix_q8_g3_m3",
        "hc_read_up_mix_q8_g3_m4",
    ],
    [
        "hc_read_up_mix_q8_g4_m1",
        "hc_read_up_mix_q8_g4_m2",
        "hc_read_up_mix_q8_g4_m3",
        "hc_read_up_mix_q8_g4_m4",
    ],
    [
        "hc_read_up_mix_q8_g5_m1",
        "hc_read_up_mix_q8_g5_m2",
        "hc_read_up_mix_q8_g5_m3",
        "hc_read_up_mix_q8_g5_m4",
    ],
    [
        "hc_read_up_mix_q8_g6_m1",
        "hc_read_up_mix_q8_g6_m2",
        "hc_read_up_mix_q8_g6_m3",
        "hc_read_up_mix_q8_g6_m4",
    ],
    [
        "hc_read_up_mix_q8_g7_m1",
        "hc_read_up_mix_q8_g7_m2",
        "hc_read_up_mix_q8_g7_m3",
        "hc_read_up_mix_q8_g7_m4",
    ],
    [
        "hc_read_up_mix_q8_g8_m1",
        "hc_read_up_mix_q8_g8_m2",
        "hc_read_up_mix_q8_g8_m3",
        "hc_read_up_mix_q8_g8_m4",
    ],
];

/// Validates and encodes a fused up+mix kernel (`name`) for `rows` rows;
/// shared by the single-row wrapper (`down` holds the logits and the kernel
/// takes `1/groups` to apply the SiLU itself) and the small-batch one
/// (`down` holds the precomputed activation, no `inv_g`).
#[allow(clippy::too_many_arguments)]
fn dispatch_read_up_mix(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    up: &QuantWeights,
    down: &Tensor,
    hyper: &Tensor,
    w: &Tensor,
    inv_rms: &Tensor,
    mixed: &Tensor,
    h: usize,
    groups: usize,
    rows: usize,
    with_inv_g: bool,
    w_bias: f32,
) -> Result<()> {
    let k = groups * h;
    ensure!(down.dtype() == DType::BF16, "down logits must be BF16");
    ensure!(
        down.numel().is_multiple_of(rows) && down.numel() > 0,
        "down logits numel {} not [{rows}, lowrank]",
        down.numel()
    );
    let r = down.numel() / rows;
    check_hc_q8(up, k, r, "read-gate up weight")?;
    ensure!(
        hyper.numel() == rows * k && hyper.dtype() == DType::BF16,
        "hyper must be BF16 [{rows}, {k}]"
    );
    ensure!(
        w.numel() == k && w.dtype() == DType::BF16,
        "norm weight must be BF16 [{k}]"
    );
    ensure!(
        inv_rms.numel() == rows * groups && inv_rms.dtype() == DType::F32,
        "inv_rms must be F32 [{rows}, {groups}]"
    );
    ensure!(
        mixed.numel() == rows * h && mixed.dtype() == DType::BF16,
        "mixed must be BF16 [{rows}, {h}]"
    );
    let inv_g = 1.0 / groups as f32;
    let (h_b, r_b, gs_b) = (u32_bytes(h), u32_bytes(r), u32_bytes(up.group_size));
    let (ig_b, wb_b) = (inv_g.to_ne_bytes(), w_bias.to_ne_bytes());
    let mut params: Vec<&[u8]> = vec![&h_b, &r_b, &gs_b];
    if with_inv_g {
        params.push(&ig_b);
    }
    params.push(&wb_b);
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
        &params,
        Grid::Threadgroups {
            groups: (h.div_ceil(HC_UP_COLS), 1, 1),
            threadgroup: (HC_UP_COLS * 32, 1, 1),
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
    dispatch_read_up_mix(
        ctx, pass, name, up, down, hyper, w, inv_rms, mixed, h, groups, 1, true, w_bias,
    )
}

/// [`hc_read_up_mix_q8`] over `rows <= HC_FUSED_MAX_ROWS` rows at once, on
/// the precomputed activation: `act` is `[rows, lowrank]` (`silu(down /
/// groups)` from [`hc_read_down_q8_rows`]), `hyper` `[rows, groups*h]`,
/// `inv_rms` F32 `[rows, groups]`, `mixed` `[rows, h]`. Row `m` of `mixed`
/// computes what the single-row kernel computes on row `m` (bit-identical up
/// to the compiler's fast-math scheduling of the unrolled body; the tests
/// bound the residual mismatch to single bf16 ulps in ~1e-5 of the elements).
#[allow(clippy::too_many_arguments)]
pub fn hc_read_up_mix_q8_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    up: &QuantWeights,
    act: &Tensor,
    hyper: &Tensor,
    w: &Tensor,
    inv_rms: &Tensor,
    mixed: &Tensor,
    h: usize,
    groups: usize,
    w_bias: f32,
) -> Result<()> {
    ensure!(
        hyper.shape().len() == 2,
        "hyper must be [rows, G*H], got {:?}",
        hyper.shape()
    );
    let rows = hyper.shape()[0];
    ensure!(
        (1..=HC_FUSED_MAX_ROWS).contains(&rows),
        "rows {rows} outside 1..={HC_FUSED_MAX_ROWS}"
    );
    ensure!(
        (1..=HC_MAX_G).contains(&groups),
        "streams {groups} outside 1..={HC_MAX_G}"
    );
    let name = HC_UP_MIX_ROWS_FNS[groups - 1][rows - 1];
    dispatch_read_up_mix(
        ctx, pass, name, up, act, hyper, w, inv_rms, mixed, h, groups, rows, false,
        w_bias,
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/hc.rs"]
mod tests;
