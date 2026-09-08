//! Per-Layer (n-gram) Embedding kernels: quantized table gather (ids come
//! from the host-side hasher in `qwen4exp::ngram`),
//! the per-stream gate, and the dilated causal conv that closes the module.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/ple.metal");
const TG: usize = 256;
/// Largest conv context `(kernel - 1) * dilation` the register window holds.
pub const PLE_MAX_CONTEXT: usize = 16;

/// `out[m, j*K..]` = dequantized table row `ids[m, j]` (Q4, any group size
/// dividing the row width).
pub fn ple_gather_q4(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QuantWeights,
    ids: &Tensor,
    heads: usize,
    out: &Tensor,
) -> Result<()> {
    let k = table.in_features();
    ensure!(table.bits == 4, "n-gram table gather is 4-bit only");
    ensure!(
        k.is_multiple_of(8)
            && k.is_multiple_of(table.group_size)
            && table.group_size.is_multiple_of(8),
        "row width {k} / group {} not word-packable",
        table.group_size
    );
    ensure!(
        ids.dtype() == DType::U32 && ids.numel().is_multiple_of(heads),
        "ids must be U32 [M, heads]"
    );
    let m = ids.numel() / heads;
    ensure!(
        out.numel() == m * heads * k && out.dtype() == DType::BF16,
        "out must be BF16 [M, heads*K]"
    );
    let pipeline = ctx.pipeline("ple_gather_q4_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            table.codes.binding(),
            table.scales.binding(),
            table.biases.binding(),
            ids.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(table.group_size), &u32_bytes(heads)],
        Grid::Threads { grid: (k / 8, heads, m), threadgroup: (k / 8, 1, 1) },
    )
}

/// `gated[r, g*h+i] = sigmoid(signed_sqrt(<key_g, query_g> / sqrt(h))) * value[r, i]`.
#[allow(clippy::too_many_arguments)]
pub fn ple_gate_value_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    key: &Tensor,
    query: &Tensor,
    value: &Tensor,
    gated: &Tensor,
    h: usize,
    groups: usize,
) -> Result<()> {
    let wide = groups * h;
    ensure!(
        key.numel().is_multiple_of(wide) && key.numel() > 0,
        "key must be [rows, G*H]"
    );
    let rows = key.numel() / wide;
    for (name, t, len) in [
        ("key", key, rows * wide),
        ("query", query, rows * wide),
        ("value", value, rows * h),
        ("gated", gated, rows * wide),
    ] {
        ensure!(
            t.numel() == len && t.dtype() == DType::BF16,
            "{name} must be BF16 [{len}]"
        );
    }
    let inv_sqrt_h = 1.0 / (h as f32).sqrt();
    let pipeline = ctx.pipeline("ple_gate_value_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[key.binding(), query.binding(), value.binding(), gated.binding()],
        &[&u32_bytes(h), &u32_bytes(groups), &inv_sqrt_h.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows * groups, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

fn check_conv(w: &Tensor, c: usize, dilation: usize) -> Result<usize> {
    ensure!(
        w.numel().is_multiple_of(c) && w.dtype() == DType::BF16,
        "conv weight must be BF16 [KD, C]"
    );
    let kd = w.numel() / c;
    ensure!(kd >= 2 && dilation >= 1, "conv kernel {kd} / dilation {dilation} invalid");
    let s = (kd - 1) * dilation;
    ensure!(
        s <= PLE_MAX_CONTEXT,
        "conv context {s} exceeds register window {PLE_MAX_CONTEXT}"
    );
    Ok(s)
}

/// Dilated causal depthwise conv + SiLU over `x` (`[M, C]`), accumulated as
/// `hyper += base + silu(conv)`. Windows are double-buffered like the GDN conv.
#[allow(clippy::too_many_arguments)]
pub fn ple_conv1d_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    window_out: &Tensor,
    x: &Tensor,
    w: &Tensor,
    base: &Tensor,
    hyper: &Tensor,
    dilation: usize,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [M, C]");
    let (m, c) = (x.shape()[0], x.shape()[1]);
    let s = check_conv(w, c, dilation)?;
    let kd = w.numel() / c;
    for (name, t) in [("window_in", window_in), ("window_out", window_out)] {
        ensure!(
            t.numel() == c * s && t.dtype() == DType::BF16,
            "{name} must be BF16 [C, S]"
        );
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "windows must be distinct"
    );
    for (name, t) in [("base", base), ("hyper", hyper)] {
        ensure!(
            t.numel() == m * c && t.dtype() == DType::BF16,
            "{name} must be BF16 [M, C]"
        );
    }
    let pipeline = ctx.pipeline("ple_conv1d_prefill_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            window_in.binding(),
            window_out.binding(),
            x.binding(),
            w.binding(),
            base.binding(),
            hyper.binding(),
        ],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(dilation), &u32_bytes(m)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (TG.min(c), 1, 1) },
    )
}

/// Single-token [`ple_conv1d_prefill`]; the window is shifted in place.
#[allow(clippy::too_many_arguments)]
pub fn ple_conv1d_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    base: &Tensor,
    hyper: &Tensor,
    dilation: usize,
) -> Result<()> {
    let c = x.numel();
    let s = check_conv(w, c, dilation)?;
    let kd = w.numel() / c;
    ensure!(
        window.numel() == c * s && window.dtype() == DType::BF16,
        "window must be BF16 [C, S]"
    );
    for (name, t) in [("x", x), ("base", base), ("hyper", hyper)] {
        ensure!(t.numel() == c && t.dtype() == DType::BF16, "{name} must be BF16 [C]");
    }
    let pipeline = ctx.pipeline("ple_conv1d_step_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), base.binding(), hyper.binding()],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(dilation)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (TG.min(c), 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/ple.rs"]
mod tests;
