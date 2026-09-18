//! Gated DeltaNet recurrence, causal depthwise conv1d, and gated RMSNorm.

use anyhow::{Result, ensure};

use crate::kernels::{Pos, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

/// Compiled GDN key/value head dimension.
pub const GDN_HEAD_DIM: usize = 128;

/// Prefill conv1d tile width; must match `CONV_TILE` in gdn.metal.
pub const CONV1D_PREFILL_TILE: usize = 64;

const SOURCE: &str = include_str!("metal/gdn.metal");

/// Recurrent state is F32 because rounding compounds across tokens.
pub const GDN_STATE_DTYPE: DType = DType::F32;

/// Activation applied to the output gate `z` inside the gated RMSNorm.
/// Qwen3.5 uses SiLU; Qwen3.8-Flash-Next sets `output_gate_type = "sigmoid"`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GdnGate {
    Silu = 0,
    Sigmoid = 1,
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let k_part = num_k_heads * dim;
    let part = num_heads * dim;
    ensure!(qkv.numel() == 2 * k_part + part, "bad fused qkv size");
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    ensure!(state.numel() == num_heads * dim * dim, "bad state size");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(z.numel() == part && z.dtype() == DType::BF16, "z must be BF16 [H, dim]");
    ensure!(
        norm_w.numel() == dim && norm_w.dtype() == DType::F32,
        "norm_w must be F32 [dim]"
    );
    ensure!(out.numel() == part && out.dtype() == DType::BF16, "bad output size");
    gdn_step_gated_named(
        ctx, pass, "gdn_step_gated", qkv, a, b, a_log, dt_bias, state, z, norm_w, out,
        scale, num_k_heads, eps, gate_act,
    )
}

/// [`gdn_step_gated_fused`] through the kernel `name` (a variant with the
/// shipped kernel's bindings and grid, for the timing harness).
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    let vpk = num_heads / num_k_heads;
    let k_part = num_k_heads * dim;
    let elem = qkv.dtype().size();
    let (qkv_buf, qkv_off) = qkv.binding();
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            (qkv_buf, qkv_off),
            (qkv_buf, qkv_off + k_part * elem),
            (qkv_buf, qkv_off + 2 * k_part * elem),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            z.binding(),
            norm_w.binding(),
            out.binding(),
        ],
        &[
            &scale.to_ne_bytes(),
            &u32_bytes(vpk),
            &eps.to_ne_bytes(),
            &u32_bytes(gate_act as usize),
        ],
        Grid::Threadgroups { groups: (num_heads, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

/// Staging for normalized q/k rows and precomputed gates.
pub struct GdnRegscanStaging<'a> {
    /// BF16 `[M, 2*HK*128]` normalized q then k rows.
    pub qk_norm: &'a Tensor,
    /// F32 `[M, H]` decay gates.
    pub decay: &'a Tensor,
    /// F32 `[M, H]` sigmoid(b) gates.
    pub beta: &'a Tensor,
    /// The chunked scan's staging; without it every prefill takes the
    /// token-serial scan.
    pub chunk: Option<GdnChunkStaging<'a>>,
}

/// Tokens per chunk of the chunked prefill scan (`GDN_CHUNK` in gdn.metal).
pub const GDN_CHUNK: usize = 64;
/// State columns per `gdn_chunk_scan` threadgroup (`GDN_CHUNK_NB`).
pub const GDN_CHUNK_NB: usize = 16;
/// Rows from which a prefill takes the chunked scan; shorter batches (the
/// verify passes) stay on the token-serial scan.
pub const GDN_CHUNK_MIN_ROWS: usize = 128;

/// Per-chunk WY staging the chunked scan writes in its first pass and reads
/// in its second: [`gdn_chunk_staging_shapes`] gives the tensors' shapes.
pub struct GdnChunkStaging<'a> {
    /// BF16 `[M, H, 128]` W = T diag(beta exp(G)) K.
    pub w: &'a Tensor,
    /// BF16 `[M, H, 128]` U = T diag(beta) V.
    pub u: &'a Tensor,
    /// BF16 `[M, H, 64]` causal, decayed Q K^T within the chunk.
    pub p: &'a Tensor,
    /// F32 `[M, H]` cumulative log-decay within the chunk.
    pub g: &'a Tensor,
}

/// Shapes and dtypes of the chunk staging tensors (`w`, `u`, `p`, `g`) for
/// `m` rows and `num_heads` value heads.
pub fn gdn_chunk_staging_shapes(m: usize, num_heads: usize) -> [(Vec<usize>, DType); 4] {
    [
        (vec![m, num_heads, GDN_HEAD_DIM], DType::BF16),
        (vec![m, num_heads, GDN_HEAD_DIM], DType::BF16),
        (vec![m, num_heads, GDN_CHUNK], DType::BF16),
        (vec![m, num_heads], DType::F32),
    ]
}

/// L2-normalizes q/k into BF16 `[M, 2*HK*128]`; q is pre-scaled.
#[allow(clippy::too_many_arguments)]
fn gdn_qk_l2norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    qk_norm: &Tensor,
    scale: f32,
    num_k_heads: usize,
    num_heads: usize,
) -> Result<()> {
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a positive multiple of k-heads {num_k_heads}"
    );
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    let row_size = (2 * num_k_heads + num_heads) * dim;
    ensure!(qkv.numel().is_multiple_of(row_size), "qkv not [M, (2*HK+H)*{dim}]");
    let m = qkv.numel() / row_size;
    ensure!(m > 0, "qkv must contain at least one row");
    ensure!(
        qk_norm.numel() == m * 2 * num_k_heads * dim && qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );

    let pipeline = ctx.pipeline("gdn_qk_l2norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qkv.binding(), qk_norm.binding()],
        &[&scale.to_ne_bytes(), &u32_bytes(num_k_heads), &u32_bytes(num_heads)],
        Grid::Threads {
            grid: (32, num_k_heads, m),
            threadgroup: (32, 4.min(num_k_heads), 1),
        },
    )
}

/// Computes F32 decay and beta gates from BF16 `[M, H]` inputs.
#[allow(clippy::too_many_arguments)]
fn gdn_gates(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let num_heads = a_log.numel();
    ensure!(num_heads > 0, "a_log must contain at least one head");
    ensure!(a.dtype() == DType::BF16, "a must be BF16");
    ensure!(b.dtype() == DType::BF16, "b must be BF16");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(dt_bias.dtype() == DType::BF16, "dt_bias must be BF16");
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let n = a.numel();
    ensure!(n > 0, "a must contain at least one row");
    ensure!(b.numel() == n, "b not [M, H]");
    ensure!(dt_bias.numel() == num_heads, "dt_bias not [H]");
    ensure!(
        decay.numel() == n && decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(beta.numel() == n && beta.dtype() == DType::F32, "beta must be F32 [M, H]");

    let pipeline = ctx.pipeline("gdn_gates", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            decay.binding(),
            beta.binding(),
        ],
        &[&u32_bytes(num_heads)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

/// Scans `[q | k | v]` rows serially with F32 state held in registers.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    gdn_prefill_mid(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        staging,
        state,
        out,
        scale,
        num_k_heads,
        None,
    )
}

/// [`gdn_prefill`] that also records the running state after each of the
/// first `mid.shape()[0]` tokens into `mid` (F32 `[count, H, 128, 128]`), for
/// rolling back a speculative batch to an accepted prefix.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_mid(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
) -> Result<()> {
    // `LILY_GDN_SCAN_KERNEL` names an alternative scan (timing experiments):
    // a token-serial kernel by name, or `gdn_chunk` for the chunked scan.
    static FORCED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let forced = FORCED.get_or_init(|| std::env::var("LILY_GDN_SCAN_KERNEL").ok());
    let m = a.numel() / a_log.numel().max(1);
    let default = if mid.is_none() && staging.chunk.is_some() && m >= GDN_CHUNK_MIN_ROWS {
        GDN_CHUNK_SCAN
    } else {
        GDN_SCAN_KERNEL
    };
    let scan_name = forced.as_deref().unwrap_or(default);
    gdn_prefill_scan_named(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        staging,
        state,
        out,
        scale,
        num_k_heads,
        mid,
        scan_name,
    )
}

/// The token-serial prefill scan kernel.
const GDN_SCAN_KERNEL: &str = "gdn_prefill_regscan";
/// The name that selects the chunked scan (`gdn_chunk_wy` + `gdn_chunk_scan`).
pub const GDN_CHUNK_SCAN: &str = "gdn_chunk";

/// [`gdn_prefill_mid`] through the scan kernel given by name (the shipped
/// scan takes four value columns per simdgroup; `_c1` / `_c2` are the
/// one- and two-column instantiations).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_scan_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
    scan_name: &str,
) -> Result<()> {
    let num_heads = a_log.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let m = a.numel() / num_heads;
    ensure!(b.numel() == m * num_heads, "b not [M, H]");
    ensure!(
        qkv.numel() == m * (2 * num_k_heads + num_heads) * dim,
        "qkv not [M, (2*HK+H)*{dim}]"
    );
    ensure!(out.numel() == m * num_heads * dim, "out not [M, H, {dim}]");
    ensure!(state.numel() == num_heads * dim * dim, "state must be [H, {dim}, {dim}]");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(
        staging.qk_norm.numel() == m * 2 * num_k_heads * dim
            && staging.qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );
    ensure!(
        staging.decay.numel() == m * num_heads && staging.decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(
        staging.beta.numel() == m * num_heads && staging.beta.dtype() == DType::F32,
        "beta must be F32 [M, H]"
    );

    if scan_name == GDN_CHUNK_SCAN {
        ensure!(mid.is_none(), "the chunked scan records no mid states");
        let chunk = staging
            .chunk
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the chunked scan needs its chunk staging"))?;
        // Whole chunks through the tensor kernels; a ragged tail continues
        // from their state through the token-serial scan (its kernels read
        // whole chunks of rows, see gdn.metal).
        let m_main = m - m % GDN_CHUNK;
        if m_main == m {
            return gdn_prefill_chunked(
                ctx, pass, qkv, a, b, a_log, dt_bias, staging, chunk, state, out, scale,
                num_k_heads,
            );
        }
        let rows = |t: &Tensor, start: usize, len: usize| -> Result<Tensor> {
            let mut shape = t.shape().to_vec();
            let per_row = t.numel() / shape[0];
            shape[0] = len;
            t.view(start * per_row, &shape)
        };
        if m_main > 0 {
            let head = GdnRegscanStaging {
                qk_norm: &rows(staging.qk_norm, 0, m_main)?,
                decay: staging.decay,
                beta: staging.beta,
                chunk: Some(GdnChunkStaging {
                    w: &rows(chunk.w, 0, m_main)?,
                    u: &rows(chunk.u, 0, m_main)?,
                    p: &rows(chunk.p, 0, m_main)?,
                    g: &rows(chunk.g, 0, m_main)?,
                }),
            };
            gdn_prefill_chunked(
                ctx,
                pass,
                &rows(qkv, 0, m_main)?,
                &rows(a, 0, m_main)?,
                &rows(b, 0, m_main)?,
                a_log,
                dt_bias,
                &head,
                head.chunk.as_ref().expect("chunk staging"),
                state,
                &rows(out, 0, m_main)?,
                scale,
                num_k_heads,
            )?;
            pass.level_barrier(&[state])?;
        }
        let tail_len = m - m_main;
        let tail = GdnRegscanStaging {
            qk_norm: &rows(staging.qk_norm, m_main, tail_len)?,
            decay: &rows(staging.decay, m_main, tail_len)?,
            beta: &rows(staging.beta, m_main, tail_len)?,
            chunk: None,
        };
        return gdn_prefill_scan_named(
            ctx,
            pass,
            &rows(qkv, m_main, tail_len)?,
            &rows(a, m_main, tail_len)?,
            &rows(b, m_main, tail_len)?,
            a_log,
            dt_bias,
            &tail,
            state,
            &rows(out, m_main, tail_len)?,
            scale,
            num_k_heads,
            None,
            GDN_SCAN_KERNEL,
        );
    }
    let mid_count = match mid {
        Some(t) => {
            ensure!(
                t.dtype() == DType::F32
                    && t.shape().len() == 4
                    && t.shape()[1..] == [num_heads, dim, dim],
                "mid states must be F32 [count, H, {dim}, {dim}]"
            );
            ensure!(
                t.shape()[0] <= m,
                "more mid states ({}) than tokens ({m})",
                t.shape()[0]
            );
            t.shape()[0]
        }
        None => 0,
    };

    gdn_qk_l2norm(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
    gdn_gates(ctx, pass, a, b, a_log, dt_bias, staging.decay, staging.beta)?;
    // The scan reads the staging the two dispatches above wrote.
    pass.level_barrier(&[staging.qk_norm, staging.decay, staging.beta])?;
    let scan_name: &'static str = if scan_name == GDN_SCAN_KERNEL {
        GDN_SCAN_KERNEL
    } else {
        Box::leak(scan_name.to_string().into_boxed_str())
    };
    // Value columns per simdgroup: the shipped scan takes four, the
    // comparison instantiations carry theirs in the name.
    let cols_per_sg = ["_c16", "_c8", "_c4", "_c2", "_c1"]
        .iter()
        .find(|tag| scan_name.contains(*tag))
        .map_or(4, |tag| tag[2..].parse::<usize>().expect("column tag"));
    let scan = ctx.pipeline(scan_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            staging.decay.binding(),
            staging.beta.binding(),
            state.binding(),
            out.binding(),
            // Never written when mid_count is 0; the state buffer stands in.
            mid.unwrap_or(state).binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk), &u32_bytes(mid_count)],
        Grid::Threadgroups {
            groups: (num_heads, dim / (4 * cols_per_sg), 1),
            threadgroup: (32, 4, 1),
        },
    )
}

/// The chunked scan: `gdn_chunk_wy` over every (chunk, key head), then
/// `gdn_chunk_scan` over every (column block, value head) walking the
/// chunks in order. Same operands, state and output as the serial scan.
#[allow(clippy::too_many_arguments)]
fn gdn_prefill_chunked(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    chunk: &GdnChunkStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    let num_heads = a_log.numel();
    let dim = GDN_HEAD_DIM;
    let vpk = num_heads / num_k_heads;
    let m = a.numel() / num_heads;
    ensure!(m > 0 && m.is_multiple_of(GDN_CHUNK), "the chunked scan takes whole chunks of {GDN_CHUNK} rows, got {m}");
    for ((t, name), (shape, dtype)) in [(chunk.w, "w"), (chunk.u, "u"), (chunk.p, "p"), (chunk.g, "g")]
        .into_iter()
        .zip(gdn_chunk_staging_shapes(m, num_heads))
    {
        ensure!(
            t.numel() >= shape.iter().product::<usize>() && t.dtype() == dtype,
            "chunk staging {name} must hold {dtype:?} {shape:?}, got {:?} {:?}",
            t.dtype(),
            t.shape()
        );
    }
    ensure!(dt_bias.dtype() == DType::BF16 && a.dtype() == DType::BF16 && b.dtype() == DType::BF16,
            "a, b and dt_bias must be BF16");
    gdn_qk_l2norm(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
    pass.level_barrier(&[staging.qk_norm])?;
    // The three-heads-at-once pass covers up to three value heads per key
    // head; wider grouping takes the per-head pass.
    let (wy_name, wy_threads) = if vpk <= 3 { ("gdn_chunk_wy3", 256) } else { ("gdn_chunk_wy", 128) };
    let wy = ctx.pipeline(wy_name, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &wy,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            chunk.w.binding(),
            chunk.u.binding(),
            chunk.p.binding(),
            chunk.g.binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk)],
        Grid::Threadgroups {
            groups: (m.div_ceil(GDN_CHUNK), num_k_heads, 1),
            threadgroup: (wy_threads, 1, 1),
        },
    )?;
    pass.level_barrier(&[chunk.w, chunk.u, chunk.p, chunk.g])?;
    let scan = ctx.pipeline("gdn_chunk_scan", SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &scan,
        &[
            staging.qk_norm.binding(),
            chunk.w.binding(),
            chunk.u.binding(),
            chunk.p.binding(),
            chunk.g.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk)],
        Grid::Threadgroups {
            groups: (dim / GDN_CHUNK_NB, num_heads, 1),
            threadgroup: (32 * 4, 1, 1),
        },
    )
}

/// Rewinds a conv window to the state after only the first `n` rows of `x`
/// followed `window_in`: `window_out` (`[C, S]`, may not alias `window_in`)
/// gets the last `S` entries of `window_in ++ x[..n]`. `x` is `[M, C]` with
/// `n <= M`; `S` is the window length (GDN: `KD-1`, PLE: `(KD-1)*dilation`).
pub fn conv_window_rollback<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    x: &Tensor,
    window_out: &Tensor,
    n: impl Into<Pos<'t>>,
) -> Result<()> {
    let n = n.into();
    ensure!(
        window_in.shape().len() == 2 && x.shape().len() == 2,
        "window must be [C, S] and x [M, C]"
    );
    let (c, s) = (window_in.shape()[0], window_in.shape()[1]);
    ensure!(
        window_out.shape() == [c, s],
        "window_out shape {:?} != [{c}, {s}]",
        window_out.shape()
    );
    ensure!(
        x.shape()[1] == c && n.max <= x.shape()[0],
        "x {:?} does not hold {} rows of {c} channels",
        x.shape(),
        n.max
    );
    for t in [window_in, x, window_out] {
        ensure!(t.dtype() == DType::BF16, "conv window rollback expects BF16");
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct"
    );
    let pipeline =
        ctx.pipeline("conv_window_rollback_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[window_in.binding(), x.binding(), window_out.binding()],
        &[Param::U32(c as u32), Param::U32(s as u32), n.param()],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// Causal conv1d + SiLU over `[M, C]`; separate window buffers avoid tile races.
pub fn conv1d_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    window_out: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [M, C]");
    let (m, c) = (x.shape()[0], x.shape()[1]);
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!((2..=9).contains(&kd), "conv kernel dim {kd} outside register window");
    for (name, t) in [("window_in", window_in), ("window_out", window_out)] {
        ensure!(t.numel() == c * (kd - 1), "{name} numel {} != C*(KD-1)", t.numel());
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct (double-buffered)"
    );
    ensure!(out.numel() == m * c, "out numel {} != M*C", out.numel());
    let pipeline = ctx.pipeline("conv1d_prefill_bf16", SOURCE, MslVersion::V3_1)?;
    let tiles = m.div_ceil(CONV1D_PREFILL_TILE).max(1);
    pass.dispatch_at(
        &pipeline,
        &[
            window_in.binding(),
            window_out.binding(),
            x.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
        Grid::Threads { grid: (c, tiles, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// One causal conv1d + SiLU decode step. `window` (`[C, KD-1]`, bf16) is
/// shifted in place; `w` is tap-major `[KD, C]`.
pub fn conv1d_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let c = x.numel();
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!(kd >= 2, "conv kernel dim {kd} < 2");
    ensure!(
        window.numel() == c * (kd - 1),
        "window numel {} != C*(KD-1)",
        window.numel()
    );
    ensure!(out.numel() == c, "out numel {} != C {c}", out.numel());
    let pipeline = ctx.pipeline("conv1d_step_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(c), &u32_bytes(kd)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// Gated RMS norm over rows of size `d` with F32 weights.
#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    gate: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let d = w.numel();
    ensure!(w.dtype() == DType::F32, "gated_rmsnorm weight must be F32");
    ensure!(
        x.numel().is_multiple_of(d),
        "x numel {} not a multiple of D {d}",
        x.numel()
    );
    ensure!(gate.numel() == x.numel() && out.numel() == x.numel(), "size mismatch");
    let rows = x.numel() / d;
    let pipeline = ctx.pipeline("gated_rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), gate.binding(), w.binding(), out.binding()],
        &[&u32_bytes(d), &eps.to_ne_bytes(), &u32_bytes(gate_act as usize)],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/gdn.rs"]
mod tests;
