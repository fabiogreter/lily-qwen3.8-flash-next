const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/skinny.metal"),
    "\n",
    include_str!("../../metal/skinny_test.metal")
);

/// Test-only explicit staged variant at K-chunk `kc`.
fn gemm_skinny_q4_nt_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    kc: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    if kc == 256 {
        return dispatch_q4_staged(ctx, pass, a, w, c, (m, k, n));
    }
    ensure!(m <= 8, "K-chunk {kc} is instantiated only for m <= 8 (m = {})", m);
    let fn_name = match kc {
        512 => "gemm_skinny_q4_bf16_m8_kc512",
        1024 => "gemm_skinny_q4_bf16_m8_kc1024",
        _ => anyhow::bail!("no test skinny q4 instantiation for K-chunk {kc}"),
    };
    let pipeline = ctx.pipeline(fn_name, TEST_SOURCE, MslVersion::V3_1)?;
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

/// Test-only register-A variant with an explicit simdgroups-per-threadgroup
/// width (each simdgroup computes `Q4_REG_ROWS_PER_SG[m - 1]` rows).
fn gemm_skinny_q4_nt_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    simdgroups: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    ensure!(m <= REG_MAX_M, "register-A kernels cover m <= 8 (m = {m})");
    ensure!(
        k.is_multiple_of(32) && w.group_size.is_multiple_of(32),
        "register-A q4 needs K % 32 == 0 and group size % 32 == 0 (k = {} gs = {})",
        k,
        w.group_size
    );
    ensure!((1..=32).contains(&simdgroups), "simdgroups {simdgroups} out of 1..=32");
    let pipeline = ctx.pipeline(Q4_REG_FNS[m - 1], SOURCE, MslVersion::V3_1)?;
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
        reg_grid(n, simdgroups, Q4_REG_ROWS_PER_SG[m - 1]),
    )
}

/// Test-only dispatch of the register-A Q4 kernels as shipped before the
/// explicit-order rewrite (`tests/metal/skinny_test.metal`): two weight rows
/// per simdgroup, two simdgroups per threadgroup.
fn gemm_skinny_q4_nt_previous(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    const BF16: [&str; REG_MAX_M] = [
        "gemm_skinny_q4_bf16_reg_m1_previous",
        "gemm_skinny_q4_bf16_reg_m2_previous",
        "gemm_skinny_q4_bf16_reg_m3_previous",
        "gemm_skinny_q4_bf16_reg_m4_previous",
        "gemm_skinny_q4_bf16_reg_m5_previous",
        "gemm_skinny_q4_bf16_reg_m6_previous",
        "gemm_skinny_q4_bf16_reg_m7_previous",
        "gemm_skinny_q4_bf16_reg_m8_previous",
    ];
    const F32: [&str; REG_MAX_M] = [
        "gemm_skinny_q4_f32_reg_m1_previous",
        "gemm_skinny_q4_f32_reg_m2_previous",
        "gemm_skinny_q4_f32_reg_m3_previous",
        "gemm_skinny_q4_f32_reg_m4_previous",
        "gemm_skinny_q4_f32_reg_m5_previous",
        "gemm_skinny_q4_f32_reg_m6_previous",
        "gemm_skinny_q4_f32_reg_m7_previous",
        "gemm_skinny_q4_f32_reg_m8_previous",
    ];
    let (m, k, n) = validate_q4(a, w, c)?;
    ensure!(m <= REG_MAX_M, "register-A kernels cover m <= 8 (m = {m})");
    let names = if c.dtype() == DType::F32 { &F32 } else { &BF16 };
    let pipeline = ctx.pipeline(names[m - 1], TEST_SOURCE, MslVersion::V3_1)?;
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
        Grid::Threadgroups { groups: (n.div_ceil(4), 1, 1), threadgroup: (64, 1, 1) },
    )
}

use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

const GROUP_SIZE: usize = 64;

/// M values covering register-A and both sides of the staged-A boundary.
const M_SWEEP: [usize; 6] = [1, 2, 5, 8, 9, 16];

/// Tolerance against an f32 reference over BF16-rounded operands (the
/// staged kernels).
const ATOL: f32 = 2e-2;
const RTOL: f32 = 2e-2;
/// Tolerance of the register-A kernels against an f32 reference over the
/// unrounded dequantized weights: the bf16 output rounding (2^-9) and the
/// f32 accumulation order.
const ATOL_REG: f32 = 4e-3;
const RTOL_REG: f32 = 4e-3;

/// The f32 reference of `gemm_skinny_q4_nt` at this shape and its
/// tolerance: the register-A route (m <= 8) dots the unrounded dequantized
/// weights like the decode GEMV, the staged route rounds them to bf16
/// first.
fn q4_reference(
    a: &[f32],
    dequant: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, f32, f32) {
    let a_r = cpu_ref::round_bf16(a);
    if reg_routes(m, n, q4_block_walk_ok(k, GROUP_SIZE)) {
        (cpu_ref::gemm_nt(&a_r, dequant, m, k, n), ATOL_REG, RTOL_REG)
    } else {
        (cpu_ref::gemm_nt(&a_r, &cpu_ref::round_bf16(dequant), m, k, n), ATOL, RTOL)
    }
}

fn assert_q4_close(
    got: &[f32],
    a: &[f32],
    dequant: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let (expected, atol, rtol) = q4_reference(a, dequant, m, k, n);
    cpu_ref::assert_close(got, &expected, atol, rtol);
}

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// A random 4-bit affine weight plus its exact f32 dequant image (the
/// quant.rs test-fixture recipe).
fn random_quant(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.01f32..0.5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, n, k, gs);
    (quant_tensors(ctx, codes, &scales, &biases, n, k, gs), dequant)
}

/// Uploads CPU-side q4 arrays as a [`QuantWeights`] (scales/biases are
/// already bf16-rounded f32).
fn quant_tensors(
    ctx: &MetalContext,
    codes: Vec<u32>,
    scales: &[f32],
    biases: &[f32],
    n: usize,
    k: usize,
    gs: usize,
) -> QuantWeights {
    let to_bf16 =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, k / 8],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(scales)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(biases)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("biases"),
        group_size: gs,
        bits: 4,
    }
}

/// Deterministic Q4 weights for tests that dequantize sampled rows on demand.
fn hash_quant_raw(
    ctx: &MetalContext,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<u32>, Vec<f32>, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| {
            let h1 = (i as u32).wrapping_mul(2654435761).wrapping_add(97);
            let h2 = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(0xA5A5_A5A5);
            (h1 & 0xFFFF_0000) | (h2 >> 16)
        })
        .collect();
    let unit = |i: usize, salt: u32| -> f32 {
        let h = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(salt);
        (h >> 8) as f32 / (1u32 << 24) as f32
    };
    let scales: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(0.01 + 0.49 * unit(i, 5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(-2.0 + 2.0 * unit(i, 9)).to_f32())
        .collect();
    let w = quant_tensors(ctx, codes.clone(), &scales, &biases, n, k, gs);
    (w, codes, scales, biases)
}

/// Checks register/staged routing, N tails, and K-chunk tails against f32.
#[test]
fn gemm_skinny_q4_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(41);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (33, 320),
        (256, 64),
        (4, 256),
        (64, 256),
        (1024, 256),
        (256, 512),
        (1, 256),
        (2, 64),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in M_SWEEP {
            if m * n * k > 150_000_000 {
                continue;
            }
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
        }
    }
}

/// The 16-byte A staging path needs a 16B-aligned base; a bf16 view at an
/// odd 4-byte offset must take the scalar staging path and still agree.
#[test]
fn gemm_skinny_q4_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(43);
    let (n, k, m) = (48usize, 320usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    // Byte offset 4: valid for Metal (4B) but not 16B-aligned.
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
}

/// Samples boundary rows and a prime-stride walk across wide outputs.
fn sample_rows(n: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..16).chain(n - 16..n).collect();
    rows.extend((0..n).step_by(9973));
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Checks the wide-output register range, final-row tail, and staged fallback.
/// The f32 reference evaluates sampled rows at dispatch boundaries.
#[test]
fn gemm_skinny_q4_vocab_shape_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(44);
    let (k, gs) = (2048usize, 64usize);
    let (words, groups) = (k / 8, k / gs);
    for n in [248320usize, 248317] {
        let (w, codes, scales, biases) = hash_quant_raw(&ctx, n, k, gs);
        let rows = sample_rows(n);
        for m in 1..=9usize {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            let got_full = tc.to_f32().expect("read");
            let a_r = cpu_ref::round_bf16(&a);
            let mut got = Vec::new();
            let mut want = Vec::new();
            for &r in &rows {
                let deq = cpu_ref::dequant_q4(
                    &codes[r * words..(r + 1) * words],
                    &scales[r * groups..(r + 1) * groups],
                    &biases[r * groups..(r + 1) * groups],
                    1,
                    k,
                    gs,
                );
                let reg = reg_routes(m, n, q4_block_walk_ok(k, gs));
                let w_r = if reg { deq } else { cpu_ref::round_bf16(&deq) };
                for i in 0..m {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a_r[i * k + kk] * w_r[kk];
                    }
                    want.push(sum);
                    got.push(got_full[i * n + r]);
                }
            }
            let reg = reg_routes(m, n, q4_block_walk_ok(k, gs));
            let (atol, rtol) = if reg { (ATOL_REG, RTOL_REG) } else { (ATOL, RTOL) };
            cpu_ref::assert_close(&got, &want, atol, rtol);
        }
    }
}

/// Wide N used to exercise register-A routing.
const WIDE_N: usize = 65536;

#[test]
fn reg_route_covers_small_m_everywhere_and_wide_n_up_to_the_ceiling() {
    // Narrow layer widths take register-A for small m (the staged walk is
    // latency-bound there) and nothing past the instantiation ceiling.
    assert!(reg_routes(1, WIDE_N - 1, true));
    assert!(reg_routes(REG_SMALL_M, WIDE_N - 1, true));
    assert!(!reg_routes(REG_MAX_M + 1, WIDE_N - 1, true));
    assert!(reg_routes(1, WIDE_N, true));
    assert!(!reg_routes(1, WIDE_N, false));
}

#[test]
fn fused_stack_requires_one_route_for_stack_and_slices() {
    assert!(stack_route_uniform(1, WIDE_N - 1, &[1024, 2048], true));
    // Small m: every width routes register-A, so any stack is uniform.
    assert!(stack_route_uniform(1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(REG_MAX_M + 1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(1, WIDE_N, &[WIDE_N], false));
}

/// The route boundary itself: the register body serves m up to its
/// instantiation ceiling and nothing past it.
#[test]
fn gemm_skinny_q4_wide_route_boundary_matches_reference() {
    assert!(reg_routes(1, WIDE_N, true));
    assert!(reg_routes(REG_MAX_M, WIDE_N, true));
    assert!(
        !reg_routes(REG_MAX_M + 1, WIDE_N, true),
        "m past the register instantiations must stage"
    );
    assert!(
        !reg_routes(REG_MAX_M, WIDE_N, false),
        "a failed block-walk precondition must stage"
    );

    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(45);
    let k = 256usize;
    let check = |w: &QuantWeights, dequant: &[f32], n: usize, m: usize| {
        let a = random_vec(&mut StdRng::seed_from_u64((n + m) as u64), m * k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt(&ctx, &pass, &ta, w, &tc).expect("skinny q4");
        pass.commit_wait().expect("commit");
        assert_q4_close(&tc.to_f32().expect("read"), &a, dequant, m, k, n);
    };
    for (n, ms) in [(WIDE_N - 4, &[8usize][..]), (WIDE_N, &[1usize, 8, 9][..])] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for &m in ms {
            check(&w, &dequant, n, m);
        }
    }
}

/// The register kernel's scalar-A fallback: a bf16 view at an odd 4-byte
/// offset is not 16B-aligned, so the wide route's per-lane A loads must
/// take the element path and still agree.
#[test]
fn gemm_skinny_q4_wide_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(47);
    let (n, k, m) = (WIDE_N, 256usize, 3usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
}

/// Checks explicit staged K-chunk and register threadgroup-width variants.
#[test]
fn gemm_skinny_q4_variants_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(48);
    for (n, k) in [(1024usize, 1280usize), (64, 320)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in [1usize, 5, 8] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            // The explicit staged variants round the weights to bf16.
            let expected = cpu_ref::gemm_nt(
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dequant),
                m,
                k,
                n,
            );
            for kc in [512usize, 1024] {
                let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                gemm_skinny_q4_nt_staged(&ctx, &pass, &ta, &w, &tc, kc)
                    .expect("skinny q4 kc");
                pass.commit_wait().expect("commit");
                cpu_ref::assert_close(
                    &tc.to_f32().expect("read"),
                    &expected,
                    ATOL,
                    RTOL,
                );
            }
        }
    }
    let (n, k, m) = (WIDE_N, 256usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
    // The register-A variants dot the unrounded weights.
    let expected = cpu_ref::gemm_nt(&cpu_ref::round_bf16(&a), &dequant, m, k, n);
    for rows in [2usize, 4, 8] {
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, rows).expect("skinny q4 reg");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(
            &tc.to_f32().expect("read"),
            &expected,
            ATOL_REG,
            RTOL_REG,
        );
    }
}

/// Creates a row-range view of a Q4 weight.
fn quant_view_rows(q: &QuantWeights, start: usize, rows: usize) -> QuantWeights {
    let words = q.codes.shape()[1];
    let groups = q.scales.shape()[1];
    QuantWeights {
        codes: q.codes.view(start * words, &[rows, words]).expect("codes view"),
        scales: q.scales.view(start * groups, &[rows, groups]).expect("scales"),
        biases: q.biases.view(start * groups, &[rows, groups]).expect("biases"),
        group_size: q.group_size,
        bits: q.bits,
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: bit mismatch at {i} ({g} vs {w})"
        );
    }
}

/// Fused-stack and per-slice dispatches must be bit-identical, including
/// column splitting. Uneven widths and K=320 cover row and K-chunk tails.
#[test]
fn gemm_skinny_fused_stack_matches_per_slice_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(49);
    let k = 320usize;
    // Starts must be even: gs = 64 gives an odd group count (5), and a
    // bf16 scale view at an odd row start would not be 4-byte aligned.
    let widths = [40usize, 24, 4, 4];
    let n_total: usize = widths.iter().sum();
    let (w, _) = random_quant(&ctx, &mut rng, n_total, k, GROUP_SIZE);

    for m in [1usize, 5, 8, 16] {
        let a_vals = random_vec(&mut rng, m * k);
        let a = Tensor::from_f32_as_bf16(&ctx, &a_vals, &[m, k]).expect("a");
        // The register kernels are instantiated for m <= 8 only.
        for use_reg in [false, true] {
            if use_reg && m > REG_MAX_M {
                continue;
            }
            let name = if use_reg { "reg2" } else { "routed" };
            let run = |wq: &QuantWeights, n: usize| -> Vec<f32> {
                let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                if use_reg {
                    gemm_skinny_q4_nt_reg(&ctx, &pass, &a, wq, &c, 2)
                        .expect("reg dispatch");
                } else {
                    gemm_skinny_q4_nt(&ctx, &pass, &a, wq, &c).expect("dispatch");
                }
                pass.commit_wait().expect("commit");
                c.to_f32().expect("read")
            };
            let fused = run(&w, n_total);
            let mut start = 0usize;
            let mut slice_outs: Vec<Vec<f32>> = Vec::new();
            for &rows in &widths {
                let ws = quant_view_rows(&w, start, rows);
                slice_outs.push(run(&ws, rows));
                start += rows;
            }
            let mut off = 0usize;
            for (si, slice) in slice_outs.iter().enumerate() {
                let mut seg = Vec::with_capacity(m * widths[si]);
                for i in 0..m {
                    for j in 0..widths[si] {
                        seg.push(fused[i * n_total + off + j]);
                    }
                }
                assert_bits_eq(
                    &seg,
                    slice,
                    &format!("{name} fused vs slice {si} at m={m}"),
                );
                off += widths[si];
            }
            // The production epilogue: split the fused output back into
            // the contiguous per-projection tensors, bit-exact.
            let fused_t =
                Tensor::from_f32_as_bf16(&ctx, &fused, &[m, n_total]).expect("c");
            let dsts: Vec<Tensor> = widths
                .iter()
                .map(|&rows| Tensor::zeros(&ctx, &[m, rows], DType::BF16))
                .collect::<Result<_>>()
                .expect("dsts");
            let pass = ctx.begin().expect("pass");
            crate::kernels::elementwise::split_cols_bf16(
                &ctx,
                &pass,
                &fused_t,
                &dsts.iter().collect::<Vec<_>>(),
            )
            .expect("split");
            pass.commit_wait().expect("commit");
            for (si, (d, slice)) in dsts.iter().zip(&slice_outs).enumerate() {
                assert_bits_eq(
                    &d.to_f32().expect("read"),
                    slice,
                    &format!("{name} split segment {si} at m={m}"),
                );
            }
        }
    }
}

/// Hashed Q4 weights for the bit-identity sweeps: every code pattern, scales
/// from 2^-9 to 2^3 and biases of both signs, cheap enough for the LM head.
fn bits_quant(ctx: &MetalContext, n: usize, k: usize, salt: u32) -> QuantWeights {
    let words = k / 8;
    let groups = k / GROUP_SIZE;
    let hash = |i: usize, s: u32| -> u32 {
        let h = (i as u32 ^ s.wrapping_mul(0x85EB_CA6B)).wrapping_mul(0x9E37_79B9);
        h ^ (h >> 15)
    };
    let codes: Vec<u32> = (0..n * words)
        .map(|i| hash(i, salt).wrapping_mul(2_654_435_761).rotate_left(11))
        .collect();
    let value = |i: usize, s: u32, sign: bool| -> f32 {
        let h = hash(i, s);
        let mant = 1.0 + (h & 0xFF) as f32 / 256.0;
        let exp = ((h >> 8) % 13) as i32 - 9;
        let v = mant * 2f32.powi(exp);
        if sign && h & (1 << 20) != 0 { -v } else { v }
    };
    let scales: Vec<f32> =
        (0..n * groups).map(|i| value(i, salt ^ 0x51, false)).collect();
    let biases: Vec<f32> =
        (0..n * groups).map(|i| value(i, salt ^ 0xB1, true)).collect();
    quant_tensors(ctx, codes, &scales, &biases, n, k, GROUP_SIZE)
}

/// Activations for the bit-identity sweeps; `wide` spreads the exponents
/// over 2^-24..2^24 so a changed summation order would round differently,
/// and `zeros` makes whole 32-element blocks -0.0, mixed signed zeros or
/// values, with the first row (`k` elements) all -0.0, so a sum started
/// differently would show in the sign of a zero.
fn bits_activations(
    rng: &mut StdRng,
    len: usize,
    k: usize,
    wide: bool,
    zeros: bool,
) -> Vec<f32> {
    let mut a: Vec<f32> = (0..len)
        .map(|_| {
            let v = rng.gen_range(-1.0f32..1.0);
            if wide { v * 2f32.powi(rng.gen_range(-24i32..24)) } else { v }
        })
        .collect();
    if zeros {
        for (bi, block) in a.chunks_mut(32).enumerate() {
            let pick = if bi * 32 < k { 0 } else { rng.gen_range(0..3) };
            for v in block.iter_mut() {
                match pick {
                    0 => *v = -0.0,
                    1 => *v = if rng.gen_range(0..2) == 0 { -0.0 } else { 0.0 },
                    _ => {}
                }
            }
        }
    }
    a
}

/// The register-A Q4 kernels compute bit for bit what they computed before
/// the explicit-order rewrite (`*_previous`, main 0446016), for every m they
/// serve, both output types, the model's projection shapes (the GDN input
/// stack, the attention qkv-and-gate stack, the output projections, the LM
/// head with f32 logits), row counts that leave a partial row group, K values
/// whose last pass splits or does not, blocks of signed zeros, and an
/// activation pointer that is not 16-byte aligned.
#[test]
fn gemm_skinny_q4_reg_matches_previous_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(97);
    // (K, N, output types)
    let both = [DType::BF16, DType::F32];
    let shapes: [(usize, usize, &[DType]); 12] = [
        (2560, 16480, &[DType::BF16]),
        (2560, 13312, &[DType::BF16]),
        (6144, 2560, &[DType::BF16]),
        (2560, 248320, &[DType::F32]),
        (2560, 1029, &both),
        (6144, 37, &both),
        (64, 5, &both),
        (320, 43, &both),
        (640, 29, &both),
        (1088, 21, &both),
        (3072, 13, &both),
        (5120, 3, &both),
    ];
    for (si, &(k, n, dtypes)) in shapes.iter().enumerate() {
        let w = bits_quant(&ctx, n, k, si as u32 + 1);
        for m in 1..=REG_MAX_M {
            for (case, wide, zeros, unaligned) in [
                ("plain", false, false, false),
                ("wide", true, false, false),
                ("signed zeros", true, true, false),
                ("unaligned", true, false, true),
            ] {
                // The large shapes skip the unaligned case.
                if unaligned && n > 4096 {
                    continue;
                }
                let a = bits_activations(&mut rng, m * k, k, wide, zeros);
                let pad = if unaligned { 2 } else { 0 };
                let mut padded = vec![0.0f32; pad];
                padded.extend_from_slice(&a);
                let backing = Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + pad])
                    .expect("backing");
                let ta = backing.view(pad, &[m, k]).expect("a view");
                for &dt in dtypes {
                    let got = Tensor::zeros(&ctx, &[m, n], dt).expect("c");
                    let want = Tensor::zeros(&ctx, &[m, n], dt).expect("c");
                    let pass = ctx.begin().expect("pass");
                    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &got).expect("shipped");
                    gemm_skinny_q4_nt_previous(&ctx, &pass, &ta, &w, &want)
                        .expect("previous");
                    pass.commit_wait().expect("commit");
                    assert_bits_eq(
                        &got.to_f32().expect("read"),
                        &want.to_f32().expect("read"),
                        &format!("K={k} N={n} m={m} {case} {dt:?}"),
                    );
                }
            }
        }
    }
}

/// Checks explicit register-A kernels across regular and boundary shapes.
#[test]
fn gemm_skinny_reg_layer_shapes_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(51);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (256, 64),
        (4, 256),
        (1, 256),
        (2, 64),
        (64, 320),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in 1..=REG_MAX_M {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let expected =
                cpu_ref::gemm_nt(&cpu_ref::round_bf16(&a), &dequant, m, k, n);
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, 2).expect("q4 reg");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(
                &tc.to_f32().expect("read"),
                &expected,
                ATOL_REG,
                RTOL_REG,
            );
        }
    }
}

/// The fixed routing predicate includes its threshold.
#[test]
fn dense_smallm_routing_boundary() {
    assert!(dense_smallm_routes(1));
    assert!(dense_smallm_routes(DENSE_SMALLM_THRESHOLD));
    assert!(!dense_smallm_routes(DENSE_SMALLM_THRESHOLD + 1));
    assert!(!dense_smallm_routes(0));
}

#[test]
fn skinny_rejects_partial_group_tail() {
    let ctx = MetalContext::new().expect("metal context");
    let (m, n, k) = (1usize, 4usize, 96usize);
    let w = quant_tensors(&ctx, vec![0; n * k / 8], &[1.0; 4], &[0.0; 4], n, k, 64);
    let a = Tensor::zeros(&ctx, &[m, k], DType::BF16).expect("a");
    let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    let err = gemm_skinny_q4_nt(&ctx, &pass, &a, &w, &c).expect_err("K=96 must reject");
    assert!(err.to_string().contains("K % 64 == 0"), "{err:#}");
}

/// The f32-output variants (used for batched logits) compute exactly what the
/// bf16 variants round: same reduction order, only the final store differs.
/// Covers both the staged family and the wide register-A family.
#[test]
fn gemm_skinny_q4_f32_out_is_the_unrounded_bf16_result() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(77);
    for (n, k) in [(512usize, 256usize), (WIDE_N_MIN + 64, 256)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in [1usize, 3, 8, 12] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let c_bf = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c bf16");
            let c_f32 = Tensor::zeros(&ctx, &[m, n], DType::F32).expect("c f32");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c_bf).expect("bf16 out");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c_f32).expect("f32 out");
            pass.commit_wait().expect("commit");
            let got_f32 = c_f32.to_f32().expect("read f32");
            let got_bf = c_bf.to_f32().expect("read bf16");
            assert_bits_eq(
                &cpu_ref::round_bf16(&got_f32),
                &got_bf,
                &format!("m={m} n={n} rounded f32 vs bf16"),
            );
            assert_q4_close(&got_f32, &a, &dequant, m, k, n);
        }
    }
}

/// The Q8 staged kernel matches the CPU reference over bf16-rounded
/// dequantized weights, in bf16 and f32 output, for the narrow router/mixer
/// shapes it serves.
#[test]
fn gemm_skinny_q8_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(78);
    for (n, k) in [(512usize, 2560usize), (320, 10240), (4, 10240), (1, 2560), (33, 64)]
    {
        let gs = 64;
        let words = k / 4;
        let groups = k / gs;
        let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
        let scales: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(0.001f32..0.02)).to_f32())
            .collect();
        let biases: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
            .collect();
        let dequant = cpu_ref::dequant_affine(&codes, &scales, &biases, n, k, gs, 8);
        let to_bf16 =
            |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
        let w = QuantWeights {
            codes: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&codes),
                &[n, words],
                DType::U32,
            )
            .expect("codes"),
            scales: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&scales)),
                &[n, groups],
                DType::BF16,
            )
            .expect("scales"),
            biases: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&biases)),
                &[n, groups],
                DType::BF16,
            )
            .expect("biases"),
            group_size: gs,
            bits: 8,
        };
        for m in [1usize, 3, 4, 8, 13] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let c_bf = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let c_f32 = Tensor::zeros(&ctx, &[m, n], DType::F32).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q8_nt(&ctx, &pass, &ta, &w, &c_bf).expect("q8 bf16");
            gemm_skinny_q8_nt(&ctx, &pass, &ta, &w, &c_f32).expect("q8 f32");
            pass.commit_wait().expect("commit");
            // The f32 output takes the staged kernel (bf16-rounded weights).
            let a_r = cpu_ref::round_bf16(&a);
            let expected =
                cpu_ref::gemm_nt(&a_r, &cpu_ref::round_bf16(&dequant), m, k, n);
            let got_f32 = c_f32.to_f32().expect("read");
            cpu_ref::assert_close(&got_f32, &expected, ATOL, RTOL);
            // bf16 output takes the register-A kernel for m <= 8, which dots
            // the unrounded weights.
            let got_bf = c_bf.to_f32().expect("read");
            if m <= REG_MAX_M {
                let expected_reg = cpu_ref::gemm_nt(&a_r, &dequant, m, k, n);
                cpu_ref::assert_close(&got_bf, &expected_reg, ATOL_REG, RTOL_REG);
            } else {
                cpu_ref::assert_close(&got_bf, &expected, ATOL, RTOL);
            }
        }
    }
}

/// The verify-pass question of the optimization notes (§ 2.2): the same
/// dense bytes through the m-row register-A kernel against the decode
/// GEMV. Streams 64 weight sets per shape (past the SLC) and prints µs per
/// dispatch and the weight bandwidth for m = 1..4, on the model's attention
/// and GDN projection shapes. Run with
/// Random affine weights at `bits` (4 or 8) for timing; no CPU dequant.
fn random_quant_bits(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
    gs: usize,
    bits: usize,
) -> QuantWeights {
    let words = k * bits / 32;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let bf = |v: Vec<f32>| -> Vec<bf16> { v.into_iter().map(bf16::from_f32).collect() };
    let scales = bf((0..n * groups).map(|_| rng.gen_range(0.001f32..0.05)).collect());
    let biases = bf((0..n * groups).map(|_| rng.gen_range(-2.0f32..0.0)).collect());
    let upload = |v: &[bf16]| {
        Tensor::from_bytes(ctx, bytemuck::cast_slice(v), &[n, groups], DType::BF16)
            .expect("bf16")
    };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )
        .expect("codes"),
        scales: upload(&scales),
        biases: upload(&biases),
        group_size: gs,
        bits,
    }
}

/// Times the decode GEMV against the one-row register-A kernel on every
/// projection shape a Qwen3.8-Flash-Next decode step dispatches through
/// `gemv_quant` (Q4 unless noted; the LM head writes f32), plus register-A
/// at 2 to 4 rows. Weight sets rotate so the weights stream from memory
/// rather than the last-level cache, as in a decode step.
/// `cargo test --release -- --ignored --nocapture skinny_reg_vs_gemv_timing`.
#[test]
#[ignore = "timing only"]
fn skinny_reg_vs_gemv_timing() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(61);
    let iters = 128;
    for (bits, n, k, f32_out, what) in [
        (4usize, 248320usize, 2560usize, true, "lm_head"),
        (4, 16480, 2560, false, "gdn in_proj"),
        (4, 2560, 6144, false, "gdn out_proj / attn o_proj"),
        (4, 13312, 2560, false, "attn qkv_proj"),
        (4, 1280, 2560, false, "shared gate_up"),
        (4, 2560, 640, false, "shared down"),
        (8, 512, 2560, false, "moe gate (q8)"),
        (8, 640, 2560, false, "indexer qk_proj (q8)"),
        (8, 10240, 2560, false, "ple key_proj (q8)"),
        (8, 2560, 2560, false, "ple value_proj / mtp fc (q8)"),
        (8, 1, 2560, false, "shared_expert_gate (q8)"),
    ] {
        let bytes = n * k * bits / 8 + 2 * n * (k / GROUP_SIZE) * 2;
        let sets = (256usize << 20).div_ceil(bytes).clamp(1, 64);
        let weights: Vec<QuantWeights> = (0..sets)
            .map(|_| random_quant_bits(&ctx, &mut rng, n, k, GROUP_SIZE, bits))
            .collect();
        let out = if f32_out { DType::F32 } else { DType::BF16 };
        let x =
            Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, k), &[k]).expect("x");
        let y = Tensor::zeros(&ctx, &[n], out).expect("y");
        let time = |name: &str, f: &dyn Fn(&ComputePass<'_>, &QuantWeights)| {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let pass = ctx.begin_concurrent().expect("pass");
                for i in 0..iters {
                    f(&pass, &weights[i % sets]);
                }
                let start = std::time::Instant::now();
                pass.commit_wait().expect("commit");
                best = best.min(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
            eprintln!(
                "{what} [{n} x {k}] {name}: {best:.1} us per dispatch, {:.0} GB/s of weights",
                bytes as f64 / best / 1e3
            );
        };
        time("decode GEMV (m = 1)", &|pass, w| {
            gemv_quant(&ctx, pass, w, &x, &y).unwrap()
        });
        let rows = if f32_out { 1 } else { 4 };
        for m in 1..=rows {
            let a =
                Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * k), &[m, k])
                    .expect("a");
            let c = Tensor::zeros(&ctx, &[m, n], out).expect("c");
            time(&format!("register-A skinny (m = {m})"), &|pass, w| {
                if bits == 4 {
                    gemm_skinny_q4_nt(&ctx, pass, &a, w, &c).unwrap()
                } else {
                    gemm_skinny_q8_nt(&ctx, pass, &a, w, &c).unwrap()
                }
            });
        }
    }
}

/// The one-row register-A kernel and the decode GEMV are the same
/// construction (raw-code dot, scale and bias per block or word), so they
/// agree to the f32 accumulation order on the model's projection shapes.
#[test]
fn gemm_skinny_reg_m1_matches_decode_gemv() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(63);
    for (n, k) in [(2560usize, 2560usize), (320, 6144), (48, 2560), (7680, 2560)] {
        let (w, _) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        let x = random_vec(&mut rng, k);
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
        let ta = Tensor::from_f32_as_bf16(&ctx, &x, &[1, k]).expect("a");
        let y = Tensor::zeros(&ctx, &[n], DType::BF16).expect("y");
        let c = Tensor::zeros(&ctx, &[1, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemv_quant(&ctx, &pass, &w, &tx, &y).expect("gemv");
        gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c).expect("skinny");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(
            &c.to_f32().expect("c"),
            &y.to_f32().expect("y"),
            ATOL_REG,
            RTOL_REG,
        );
    }
}

/// Q4 weights for timing only: cheap hashed codes, plausible scales and
/// biases, no CPU image.
fn timing_quant(ctx: &MetalContext, n: usize, k: usize, salt: u32) -> QuantWeights {
    let words = k / 8;
    let groups = k / GROUP_SIZE;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| (i as u32 ^ salt).wrapping_mul(2_654_435_761).rotate_left(7))
        .collect();
    let scales: Vec<f32> =
        (0..n * groups).map(|i| 0.01 + (i % 97) as f32 * 0.004).collect();
    let biases: Vec<f32> =
        (0..n * groups).map(|i| -2.0 + (i % 89) as f32 * 0.02).collect();
    quant_tensors(ctx, codes, &scales, &biases, n, k, GROUP_SIZE)
}

/// How the shared-weight projections of a batched decode step scale with
/// its row count (`docs/architecture.md`, "Continuous batching"): every row
/// reads the same weights, so a bandwidth-bound kernel would take about as
/// long at four rows as at one. Measured 2026-10-06 in the kernel profile,
/// the two-row register-A skinny GEMM took about 44 % longer at m = 4 than
/// at m = 1 (its per-block activation work bound it); the shipped kernel
/// shares that work across more rows per simdgroup. For Qwen3.8-Flash-Next's
/// Q4 shapes (the GDN input stack, the attention qkv-and-gate stack, the two
/// output projections, the LM head with f32 logits), this times each kernel
/// variant at m = 1..8 over enough distinct copies of the matrix (1.2 GB and
/// up) that the system cache does not serve them, best and median of five
/// passes, and prints the effective weight bandwidth and the time relative
/// to m = 1. Variants: the shipped route (`gemm_skinny_q4_nt`), the two-row
/// kernel it replaced (`*_previous`, bit-identical results), both again
/// with a barrier after every dispatch (as in the model's passes), register-A
/// with 4 simdgroups per threadgroup instead of 2, the staged route
/// (activations staged in threadgroup memory, weights rounded to bf16), and
/// the decode GEMV at m = 1 as the bandwidth reference. Compare with
/// `gemv_q4_bandwidth_probe` and `memory_bandwidth_probe`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn skinny_rows_scaling_probe() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    // (name, K, N, f32 output)
    let shapes: [(&str, usize, usize, bool); 5] = [
        ("gdn in stack 2560->16480", 2560, 16480, false),
        ("attn qkv+gate 2560->13312", 2560, 13312, false),
        ("gdn out 6144->2560", 6144, 2560, false),
        ("attn out 6144->2560", 6144, 2560, false),
        ("lm head 2560->248320 f32", 2560, 248320, true),
    ];
    let target_bytes = 1.2e9;
    let mut rng = StdRng::seed_from_u64(31);
    for (name, k, n, f32_out) in shapes {
        let probe = timing_quant(&ctx, n, k, 0);
        let per_matrix = (probe.codes.byte_len()
            + probe.scales.byte_len()
            + probe.biases.byte_len()) as f64;
        drop(probe);
        let copies = ((target_bytes / per_matrix).ceil() as usize).max(2);
        let weights: Vec<QuantWeights> =
            (0..copies).map(|c| timing_quant(&ctx, n, k, c as u32 + 1)).collect();
        let total = per_matrix * copies as f64;
        eprintln!(
            "\n{name}: {copies} copies of {:.1} MB ({:.2} GB per pass)",
            per_matrix / 1e6,
            total / 1e9
        );
        let out_dtype = if f32_out { DType::F32 } else { DType::BF16 };
        let base: std::cell::Cell<Option<f64>> = std::cell::Cell::new(None);
        let time = |encode: &dyn Fn(&ComputePass<'_>) -> Result<()>| -> (f64, f64) {
            let mut times = Vec::new();
            for _ in 0..5 {
                let pass = ctx.begin_concurrent().expect("pass");
                encode(&pass).expect("encode");
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                times.push(t.gpu_end_secs - t.gpu_start_secs);
            }
            times.sort_by(f64::total_cmp);
            (times[0], times[times.len() / 2])
        };
        let report = |label: &str, m: usize, (best, median): (f64, f64)| {
            let rel = base.get().map_or(String::new(), |b| {
                format!(", x{:.2} vs shipped m=1", best / b)
            });
            eprintln!(
                "  {label:<26} m={m}: {:7.3} ms best, {:7.3} median, {:4.0} GB/s{rel}",
                best * 1e3,
                median * 1e3,
                total / best / 1e9
            );
        };
        if !f32_out {
            let x: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
            let ys: Vec<Tensor> = (0..copies)
                .map(|_| Tensor::zeros(&ctx, &[n], DType::BF16).expect("y"))
                .collect();
            let t = time(&|pass| {
                for (w, y) in weights.iter().zip(&ys) {
                    gemv_quant(&ctx, pass, w, &tx, y)?;
                }
                Ok(())
            });
            report("decode gemv (reference)", 1, t);
        }
        for m in 1..=REG_MAX_M {
            let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let cs: Vec<Tensor> = (0..copies)
                .map(|_| Tensor::zeros(&ctx, &[m, n], out_dtype).expect("c"))
                .collect();
            let shipped = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt(&ctx, pass, &ta, w, c)?;
                }
                Ok(())
            });
            if m == 1 {
                base.set(Some(shipped.0));
            }
            report("shipped route", m, shipped);
            let previous = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt_previous(&ctx, pass, &ta, w, c)?;
                }
                Ok(())
            });
            report("two-row kernel (previous)", m, previous);
            // A barrier after every dispatch, as in the model's passes: each
            // dispatch then fills the GPU on its own and pays its own tail.
            let shipped_b = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt(&ctx, pass, &ta, w, c)?;
                    pass.level_barrier(&[])?;
                }
                Ok(())
            });
            report("shipped, barriers", m, shipped_b);
            let previous_b = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt_previous(&ctx, pass, &ta, w, c)?;
                    pass.level_barrier(&[])?;
                }
                Ok(())
            });
            report("previous, barriers", m, previous_b);
            if !f32_out {
                let wide = time(&|pass| {
                    for (w, c) in weights.iter().zip(&cs) {
                        gemm_skinny_q4_nt_reg(&ctx, pass, &ta, w, c, 4)?;
                    }
                    Ok(())
                });
                report("register-A, 4 sg/tg", m, wide);
                let staged = time(&|pass| {
                    for (w, c) in weights.iter().zip(&cs) {
                        gemm_skinny_q4_nt_staged(&ctx, pass, &ta, w, c, 256)?;
                    }
                    Ok(())
                });
                report("staged (bf16 weights)", m, staged);
            }
        }
    }
}
