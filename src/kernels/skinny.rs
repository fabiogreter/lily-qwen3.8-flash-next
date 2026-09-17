//! Small-M Q4 GEMMs with staged-A and register-A variants.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/skinny.metal");

/// Weight rows per threadgroup; must match `SKINNY_SG` in skinny.metal.
const ROWS_PER_TG: usize = 4;

/// Largest row count compiled for register-A kernels.
const REG_MAX_M: usize = 8;

/// Output width from which register-A is used for every small m. Below it,
/// register-A still wins for `m <= REG_SMALL_M` rows (measured on the M5 Max
/// at Qwen3.8-Flash-Next's shapes: the staged walk is latency-bound for
/// narrow, deep projections), so the threshold only matters above that.
const WIDE_N_MIN: usize = 65536;
const REG_SMALL_M: usize = 8;

/// Simdgroups per register-A threadgroup.
const REG_SIMDGROUPS_PER_TG: usize = 2;
/// Weight rows each register-A simdgroup computes; must match
/// `SKINNY_REG_ROWS` in skinny.metal.
const REG_ROWS_PER_SG: usize = 2;

/// Selects register-A when shape and packing constraints hold.
fn reg_routes(m: usize, n: usize, block_walk_ok: bool) -> bool {
    block_walk_ok && m <= REG_MAX_M && (n >= WIDE_N_MIN || m <= REG_SMALL_M)
}

/// Requires a fused stack and every slice to use the same reduction variant.
pub fn stack_route_uniform(
    m: usize,
    n_total: usize,
    slice_ns: &[usize],
    block_walk_ok: bool,
) -> bool {
    let stack_reg = reg_routes(m, n_total, block_walk_ok);
    slice_ns.iter().all(|&n| reg_routes(m, n, block_walk_ok) == stack_reg)
}

/// Largest row count routed to the staged small-M family.
pub const DENSE_SMALLM_THRESHOLD: usize = 16;

/// Whether a nonempty dense chunk uses the staged small-M kernel.
pub fn dense_smallm_routes(m: usize) -> bool {
    m > 0 && m <= DENSE_SMALLM_THRESHOLD
}

fn staged_grid(n: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(ROWS_PER_TG), 1, 1),
        threadgroup: (32 * ROWS_PER_TG, 1, 1),
    }
}

/// Kernel names for the per-m register-A instantiations (index m - 1), by
/// output type: bf16 activations, f32 logits.
const Q4_REG_FNS: [&str; REG_MAX_M] = [
    "gemm_skinny_q4_bf16_reg_m1",
    "gemm_skinny_q4_bf16_reg_m2",
    "gemm_skinny_q4_bf16_reg_m3",
    "gemm_skinny_q4_bf16_reg_m4",
    "gemm_skinny_q4_bf16_reg_m5",
    "gemm_skinny_q4_bf16_reg_m6",
    "gemm_skinny_q4_bf16_reg_m7",
    "gemm_skinny_q4_bf16_reg_m8",
];
const Q8_REG_FNS: [&str; REG_MAX_M] = [
    "gemm_skinny_q8_bf16_reg_m1",
    "gemm_skinny_q8_bf16_reg_m2",
    "gemm_skinny_q8_bf16_reg_m3",
    "gemm_skinny_q8_bf16_reg_m4",
    "gemm_skinny_q8_bf16_reg_m5",
    "gemm_skinny_q8_bf16_reg_m6",
    "gemm_skinny_q8_bf16_reg_m7",
    "gemm_skinny_q8_bf16_reg_m8",
];
const Q4_REG_FNS_F32: [&str; REG_MAX_M] = [
    "gemm_skinny_q4_f32_reg_m1",
    "gemm_skinny_q4_f32_reg_m2",
    "gemm_skinny_q4_f32_reg_m3",
    "gemm_skinny_q4_f32_reg_m4",
    "gemm_skinny_q4_f32_reg_m5",
    "gemm_skinny_q4_f32_reg_m6",
    "gemm_skinny_q4_f32_reg_m7",
    "gemm_skinny_q4_f32_reg_m8",
];

fn reg_grid(n: usize, simdgroups: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(simdgroups * REG_ROWS_PER_SG), 1, 1),
        threadgroup: (32 * simdgroups, 1, 1),
    }
}

fn validate_q4(
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<(usize, usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    let m = a.shape()[0];
    ensure!(m > 0, "skinny GEMM needs at least one row");
    ensure!(
        m <= DENSE_SMALLM_THRESHOLD,
        "skinny GEMM is instantiated only through m={DENSE_SMALLM_THRESHOLD} (got {m})"
    );
    ensure!(w.bits == 4, "skinny quantized GEMM is 4-bit only");
    // The Metal body requires complete 64-element quantization groups.
    ensure!(
        w.group_size == 64 && k.is_multiple_of(64),
        "skinny GEMM requires group_size=64 and K % 64 == 0 (k={k}, gs={})",
        w.group_size
    );
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    ensure!(a.dtype() == DType::BF16, "skinny q4 GEMM input must be BF16");
    ensure!(
        matches!(c.dtype(), DType::BF16 | DType::F32),
        "skinny q4 GEMM output must be BF16 or F32"
    );
    Ok((m, k, n))
}

/// Whether the register-A walk can consume complete quant groups.
pub fn q4_block_walk_ok(k: usize, group_size: usize) -> bool {
    group_size == 64 && k.is_multiple_of(64)
}

fn dispatch_q4_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let fn_name = match (m <= REG_MAX_M, c.dtype()) {
        (true, DType::F32) => "gemm_skinny_q4_f32_m8",
        (false, DType::F32) => "gemm_skinny_q4_f32_m16",
        (true, _) => "gemm_skinny_q4_bf16_m8",
        (false, _) => "gemm_skinny_q4_bf16_m16",
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

fn dispatch_q4_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let names = if c.dtype() == DType::F32 { &Q4_REG_FNS_F32 } else { &Q4_REG_FNS };
    let pipeline = ctx.pipeline(names[m - 1], SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, REG_SIMDGROUPS_PER_TG),
    )
}

/// Small-M affine-Q8 GEMM (the 8-bit tensors are the narrow routers, gates
/// and stream mixers). The staged kernels have the numerics of the bf16
/// dequant + GEMM fallback they replace; register-A (`m <= 8`, bf16 out)
/// dots the raw codes like the decode GEMV.
pub fn gemm_skinny_q8_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    let (n, k) = (w.out_features(), w.in_features());
    let m = a.shape()[0];
    ensure!(
        m > 0 && m <= DENSE_SMALLM_THRESHOLD,
        "skinny q8 GEMM handles 1..={DENSE_SMALLM_THRESHOLD} rows (got {m})"
    );
    ensure!(w.bits == 8, "skinny q8 GEMM needs 8-bit weights");
    ensure!(
        w.group_size.is_multiple_of(8) && k.is_multiple_of(8),
        "skinny q8 GEMM needs group_size % 8 == 0 and K % 8 == 0 (k={k}, gs={})",
        w.group_size
    );
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    ensure!(a.dtype() == DType::BF16, "skinny q8 GEMM input must be BF16");
    // Register-A for small m (no threadgroup staging or barriers: the
    // narrow, deep mixers and routers are latency-bound in the staged walk).
    if m <= REG_MAX_M
        && c.dtype() == DType::BF16
        && w.group_size.is_multiple_of(16)
        && k.is_multiple_of(16)
    {
        let pipeline = ctx.pipeline(Q8_REG_FNS[m - 1], SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &pipeline,
            &[
                w.codes.binding(),
                w.scales.binding(),
                w.biases.binding(),
                a.binding(),
                c.binding(),
            ],
            &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
            reg_grid(n, REG_SIMDGROUPS_PER_TG),
        );
    }
    let fn_name = match (m <= REG_MAX_M, c.dtype()) {
        (true, DType::F32) => "gemm_skinny_q8_f32_m8",
        (false, DType::F32) => "gemm_skinny_q8_f32_m16",
        (true, DType::BF16) => "gemm_skinny_q8_bf16_m8",
        (false, DType::BF16) => "gemm_skinny_q8_bf16_m16",
        _ => anyhow::bail!("skinny q8 GEMM output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

/// Small-M affine-Q4 GEMM. Register-A (`m <= 8`, `group_size == 64`,
/// `K % 64 == 0`) dots the raw codes with scale and bias applied per block,
/// like the decode GEMV; the staged fallback rounds dequantized weights to
/// bf16 first. `c` may be BF16 or F32 (the latter for logits); the f32
/// result is what the bf16 variant rounds.
pub fn gemm_skinny_q4_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    let dims = validate_q4(a, w, c)?;
    let (m, k, n) = dims;
    if reg_routes(m, n, q4_block_walk_ok(k, w.group_size)) {
        return dispatch_q4_reg(ctx, pass, a, w, c, dims);
    }
    dispatch_q4_staged(ctx, pass, a, w, c, dims)
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/skinny.rs"]
mod tests;
