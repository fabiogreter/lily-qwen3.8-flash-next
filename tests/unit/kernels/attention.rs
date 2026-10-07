use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::kernels::mrope_axis;
use crate::tensor::DType;

#[test]
fn split_scratch_tracks_surviving_routes() {
    assert_eq!(sdpa_split_scratch_splits(0), 1);
    assert_eq!(sdpa_split_scratch_splits(1), 1);
    assert_eq!(sdpa_split_scratch_splits(257), 2);
    assert_eq!(sdpa_split_scratch_splits(8192), 32);
    assert_eq!(sdpa_split_scratch_splits(32767), 128);
    assert_eq!(sdpa_split_scratch_splits(32768), 128);
    assert_eq!(sdpa_split_scratch_splits(MAX_SEQ), 128);
}

#[test]
fn production_split_route_boundaries_are_pinned() {
    let route = |len| {
        sdpa_split_route(
            len,
            256,
            8,
            16,
            2,
            GQA_FOLD_MIN_CONTEXT,
            SDPA_FIXED_BLOCK_CROSSOVER,
        )
    };
    assert_eq!(route(8191), SdpaSplitRoute::PerHead);
    assert_eq!(route(8192), SdpaSplitRoute::Gqa);
    assert_eq!(route(32767), SdpaSplitRoute::Gqa);
    assert_eq!(route(32768), SdpaSplitRoute::FixedBlock);
}

#[test]
fn sdpa_decode_matches_cpu() {
    // Mono path (len below one split chunk) and the split-K path with a
    // ragged tail chunk.
    check_sdpa_decode(
        8,
        2,
        64,
        128,
        33,
        None,
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    check_sdpa_decode(
        8,
        2,
        64,
        1024,
        700,
        Some(3),
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    check_sdpa_decode(
        4,
        2,
        32,
        600,
        512,
        Some(2),
        GQA_FOLD_MIN_CONTEXT,
        SDPA_FIXED_BLOCK_CROSSOVER,
    );
    // Force the folded GQA route at a compact test shape: catches drift in
    // the grouped kernel's distinct buffer-11 chunk ABI without making the
    // GPU-heavy unit suite allocate and scan an 8192-row cache in parallel.
    check_sdpa_decode(8, 1, 256, 600, 512, Some(2), 0, SDPA_FIXED_BLOCK_CROSSOVER);
    // Force the fixed-block route at the same compact shape. Its 128
    // strided blocks cover ragged tails as well as the production 32K+
    // regime without making this CPU-oracle test scan a long cache.
    check_sdpa_decode(8, 1, 256, 600, 513, Some(SDPA_MLX_BLOCKS), usize::MAX, 0);
}

#[allow(clippy::too_many_arguments)]
fn check_sdpa_decode(
    nq: usize,
    kvh: usize,
    d: usize,
    max_seq: usize,
    len: usize,
    splits: Option<usize>,
    gqa_fold_min_context: usize,
    fixed_block_min_context: usize,
) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(20 + len as u64);
    let scale = 1.0 / (d as f32).sqrt();

    let q: Vec<f32> = (0..nq * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let mut k = vec![0.0f32; kvh * max_seq * d];
    let mut v = vec![0.0f32; kvh * max_seq * d];
    for h in 0..kvh {
        for pos in 0..len {
            for i in 0..d {
                k[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
                v[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
            }
        }
    }

    let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[nq, d]).expect("q");
    let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let out = Tensor::zeros(&ctx, &[nq, d], DType::BF16).expect("out");

    let scratch = splits.map(|s| {
        (
            Tensor::zeros(&ctx, &[nq, s, d], DType::F32).expect("partials"),
            Tensor::zeros(&ctx, &[nq, s, 2], DType::F32).expect("stats"),
        )
    });
    let pass = ctx.begin().expect("pass");
    sdpa_decode_inner(
        &ctx,
        &pass,
        &tq,
        &tk,
        &tv,
        &out,
        len,
        scale,
        scratch.as_ref().map(|(p, st)| (p, st)),
        gqa_fold_min_context,
        fixed_block_min_context,
    )
    .expect("sdpa");
    pass.commit_wait().expect("commit");

    // CPU reference.
    let (rq, rk, rv) =
        (cpu_ref::round_bf16(&q), cpu_ref::round_bf16(&k), cpu_ref::round_bf16(&v));
    let mut expected = vec![0.0f32; nq * d];
    let group = nq / kvh;
    for hq in 0..nq {
        let h = hq / group;
        let mut scores = vec![0.0f32; len];
        for (pos, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += rq[hq * d + i] * rk[(h * max_seq + pos) * d + i];
            }
            *score = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for i in 0..d {
            let mut acc = 0.0f32;
            for (pos, s) in scores.iter().enumerate() {
                acc += s * rv[(h * max_seq + pos) * d + i];
            }
            expected[hq * d + i] = acc / sum;
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn rope_batched_matches_cpu_per_token() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(21);
    let (m, heads, d, rot, base_pos) = (5, 3, 64, 16, 7);
    let theta = 1e7f32;
    let x: Vec<f32> = (0..m * heads * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, heads, d]).expect("x");
    let pass = ctx.begin().expect("pass");
    rope_neox(&ctx, &pass, &tx, heads, rot, base_pos, theta, Rope::Delta(0))
        .expect("rope");
    pass.commit_wait().expect("commit");

    let mut expected = cpu_ref::round_bf16(&x);
    for t in 0..m {
        cpu_ref::rope_neox(
            &mut expected[t * heads * d..(t + 1) * heads * d],
            d,
            rot,
            base_pos + t,
            theta,
        );
    }
    cpu_ref::assert_close(&tx.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

/// A U32 `[rows, 3]` position tensor.
fn position_rows(ctx: &MetalContext, rows: &[[u32; 3]]) -> Tensor {
    let flat: Vec<u32> = rows.iter().flatten().copied().collect();
    Tensor::from_bytes(ctx, bytemuck::cast_slice(&flat), &[rows.len(), 3], DType::U32)
        .expect("positions")
}

/// Runs `rope_neox` over a fresh bf16 copy of `x` and returns the result.
#[allow(clippy::too_many_arguments)]
fn rope_of(
    ctx: &MetalContext,
    x: &[f32],
    shape: &[usize],
    heads: usize,
    rot: usize,
    base_pos: usize,
    theta: f32,
    rope: Rope<'_>,
) -> Vec<f32> {
    let tx = Tensor::from_f32_as_bf16(ctx, x, shape).expect("x");
    let pass = ctx.begin().expect("pass");
    rope_neox(ctx, &pass, &tx, heads, rot, base_pos, theta, rope).expect("rope");
    pass.commit_wait().expect("commit");
    tx.to_f32().expect("read")
}

/// The interleaved M-RoPE variant against a CPU reference on rows whose
/// three axes differ (VISION.md "Interleaved M-RoPE"), its bit-exact
/// equality with the scalar kernel on rows whose axes agree, and the scalar
/// kernel's delta against a shifted base position, also bit-exact. The
/// position buffer starts before the chunk (`base < base_pos`), as the
/// prefill's does for the block keys.
#[test]
fn rope_mrope_matches_cpu_and_the_scalar_kernel_on_text_rows() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(23);
    let (m, heads, d, rot, base_pos, base) =
        (6usize, 3usize, 256usize, 64usize, 11usize, 8usize);
    let theta = 1e7f32;
    let shape = [m, heads, d];
    let x: Vec<f32> = (0..m * heads * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let rows = base_pos + m - base;
    let mixed: Vec<[u32; 3]> = (0..rows)
        .map(|_| {
            [
                rng.gen_range(0..400u32),
                rng.gen_range(0..400u32),
                rng.gen_range(0..400u32),
            ]
        })
        .collect();
    let t_mixed = position_rows(&ctx, &mixed);
    let got = rope_of(
        &ctx,
        &x,
        &shape,
        heads,
        rot,
        base_pos,
        theta,
        Rope::Rows { positions: &t_mixed, base },
    );
    let mut expected = cpu_ref::round_bf16(&x);
    let half = rot / 2;
    for t in 0..m {
        let pos = mixed[base_pos + t - base];
        for h in 0..heads {
            let row = (t * heads + h) * d;
            for j in 0..half {
                let inv_freq = theta.powf(-2.0 * j as f32 / rot as f32);
                let ang = pos[mrope_axis(j)] as f32 * inv_freq;
                let (sin, cos) = ang.sin_cos();
                let (lo, hi) = (expected[row + j], expected[row + half + j]);
                expected[row + j] = lo * cos - hi * sin;
                expected[row + half + j] = hi * cos + lo * sin;
            }
        }
    }
    cpu_ref::assert_close(&got, &expected, 2e-2, 2e-2);
    // Distinct axes really were exercised: the result differs from the
    // scalar kernel at the chunk's positions.
    let scalar = rope_of(&ctx, &x, &shape, heads, rot, base_pos, theta, Rope::Delta(0));
    assert_ne!(got, scalar);

    // Text rows: every axis is the sequence index; the variant is bit-exact.
    let text: Vec<[u32; 3]> = (0..rows).map(|i| [(base + i) as u32; 3]).collect();
    let t_text = position_rows(&ctx, &text);
    let via_rows = rope_of(
        &ctx,
        &x,
        &shape,
        heads,
        rot,
        base_pos,
        theta,
        Rope::Rows { positions: &t_text, base },
    );
    assert_eq!(via_rows, scalar);

    // The delta moves the angle, not the row: index 20 with delta -9 is
    // index 11 with delta 0, bit for bit.
    let shifted =
        rope_of(&ctx, &x, &shape, heads, rot, base_pos + 9, theta, Rope::Delta(-9));
    assert_eq!(shifted, scalar);

    // A buffer that does not cover the chunk is refused, not read past.
    let short = position_rows(&ctx, &text[..rows - 1]);
    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &shape).expect("x");
    let pass = ctx.begin().expect("pass");
    let err = rope_neox(
        &ctx,
        &pass,
        &tx,
        heads,
        rot,
        base_pos,
        theta,
        Rope::Rows { positions: &short, base },
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("rope positions cover"), "{err:#}");
}

fn check_sdpa_prefill(
    m: usize,
    nq: usize,
    kvh: usize,
    d: usize,
    max_seq: usize,
    base_len: usize,
    seed: u64,
) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = 1.0 / (d as f32).sqrt();

    // Cache pre-filled for base_len + m positions (the chunk's keys are
    // already scattered, as in the real prefill flow).
    let filled = base_len + m;
    let mut k = vec![0.0f32; kvh * max_seq * d];
    let mut v = vec![0.0f32; kvh * max_seq * d];
    for h in 0..kvh {
        for pos in 0..filled {
            for i in 0..d {
                k[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
                v[(h * max_seq + pos) * d + i] = rng.gen_range(-1.0f32..1.0);
            }
        }
    }
    let q: Vec<f32> = (0..m * nq * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[m, nq, d]).expect("q");
    let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let out = Tensor::zeros(&ctx, &[m, nq, d], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    sdpa_prefill(&ctx, &pass, &tq, &tk, &tv, &out, base_len, scale).expect("sdpa");
    pass.commit_wait().expect("commit");

    let (rq, rk, rv) =
        (cpu_ref::round_bf16(&q), cpu_ref::round_bf16(&k), cpu_ref::round_bf16(&v));
    let group = nq / kvh;
    let mut expected = vec![0.0f32; m * nq * d];
    for t in 0..m {
        let len = base_len + t + 1;
        for hq in 0..nq {
            let h = hq / group;
            let qv = &rq[(t * nq + hq) * d..(t * nq + hq + 1) * d];
            let mut scores = vec![0.0f32; len];
            for (pos, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for i in 0..d {
                    dot += qv[i] * rk[(h * max_seq + pos) * d + i];
                }
                *score = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for i in 0..d {
                let mut acc = 0.0f32;
                for (pos, s) in scores.iter().enumerate() {
                    acc += s * rv[(h * max_seq + pos) * d + i];
                }
                expected[(t * nq + hq) * d + i] = acc / sum;
            }
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn sdpa_prefill_matches_cpu_causal() {
    // Every case runs the flash kernel, which is compiled for FA_D, so
    // the head dim is fixed and the cases vary what the tiling has to get
    // right: tile-edge query rows, key tiles crossing the causal diagonal,
    // and a base_len that misaligns the key tiling.
    check_sdpa_prefill(6, 4, 2, FA_D, 64, 9, 22);
    check_sdpa_prefill(16, 8, 2, 256, 64, 0, 23);
    check_sdpa_prefill(40, 4, 2, 256, 128, 9, 24);
    check_sdpa_prefill(33, 8, 2, 256, 96, 47, 25);
}

#[test]
fn scatter_kv_places_rows() {
    let ctx = MetalContext::new().expect("metal context");
    let (kvh, max_seq, d) = (2, 8, 16);
    let cache = Tensor::zeros(&ctx, &[kvh, max_seq, d], DType::BF16).expect("cache");
    let row: Vec<f32> = (0..kvh * d).map(|i| i as f32).collect();
    let trow = Tensor::from_f32_as_bf16(&ctx, &row, &[kvh, d]).expect("row");

    let pass = ctx.begin().expect("pass");
    scatter_kv(&ctx, &pass, &cache, &trow, 3).expect("scatter");
    pass.commit_wait().expect("commit");

    let data = cache.to_f32().expect("read");
    for h in 0..kvh {
        for i in 0..d {
            assert_eq!(data[(h * max_seq + 3) * d + i], (h * d + i) as f32);
            assert_eq!(data[(h * max_seq + 2) * d + i], 0.0);
        }
    }
}

/// GPU time of dense attention for a few queries over a long cache, 12 layers:
/// the batched prefill kernel against the split decode kernel per query.
#[test]
#[ignore = "timing; run with --nocapture"]
fn sdpa_small_m_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let (nq, kvh, d, layers) = (24usize, 2usize, 256usize, 12usize);
    let mut rng = StdRng::seed_from_u64(9);
    for len in [1100usize, 2048] {
        let cap = 2048;
        let k = Tensor::from_f32_as_bf16(
            &ctx,
            &(0..kvh * cap * d)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            &[kvh, cap, d],
        )
        .expect("k");
        let v = Tensor::from_f32_as_bf16(
            &ctx,
            &(0..kvh * cap * d)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect::<Vec<_>>(),
            &[kvh, cap, d],
        )
        .expect("v");
        let splits = sdpa_split_scratch_splits(cap);
        let partials =
            Tensor::zeros(&ctx, &[nq, splits, d], DType::F32).expect("partials");
        let stats = Tensor::zeros(&ctx, &[nq, splits, 2], DType::F32).expect("stats");
        for m in [1usize, 2, 4] {
            let q = Tensor::from_f32_as_bf16(
                &ctx,
                &(0..m * nq * d)
                    .map(|_| rng.gen_range(-1.0f32..1.0))
                    .collect::<Vec<_>>(),
                &[m, nq, d],
            )
            .expect("q");
            let out = Tensor::zeros(&ctx, &[m, nq, d], DType::BF16).expect("out");
            for variant in ["prefill", "decode-per-row"] {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let pass = ctx.begin_concurrent().expect("pass");
                    for _ in 0..layers {
                        if variant == "prefill" {
                            sdpa_prefill(
                                &ctx,
                                &pass,
                                &q,
                                &k,
                                &v,
                                &out,
                                len - m,
                                0.0625,
                            )
                            .expect("prefill");
                        } else {
                            for r in 0..m {
                                let qr = q.view(r * nq * d, &[nq, d]).expect("q row");
                                let outr =
                                    out.view(r * nq * d, &[nq, d]).expect("out row");
                                sdpa_decode(
                                    &ctx,
                                    &pass,
                                    &qr,
                                    &k,
                                    &v,
                                    &outr,
                                    len - m + r + 1,
                                    0.0625,
                                    Some((&partials, &stats)),
                                )
                                .expect("decode");
                                pass.level_barrier(&[&outr]).expect("barrier");
                            }
                        }
                        pass.level_barrier(&[&out]).expect("barrier");
                    }
                    let done =
                        pass.commit().expect("commit").wait_retain().expect("wait");
                    let t = done.timing().expect("timing");
                    best = best.min(t.gpu_end_secs - t.gpu_start_secs);
                }
                eprintln!(
                    "sdpa {layers} layers len={len} m={m} {variant}: {:.2} ms",
                    best * 1e3
                );
            }
        }
    }
}

// --- the q8 K/V cache -----------------------------------------------------------

/// Random K/V-like rows: uniform values with a few outlier channels (as
/// RoPE'd keys have), rounded to bf16, the values the writers receive.
pub(crate) fn kv_like_rows(rng: &mut StdRng, rows: usize, d: usize) -> Vec<f32> {
    let x: Vec<f32> = (0..rows * d)
        .map(|i| {
            let v = rng.gen_range(-1.0f32..1.0);
            if i % d % 37 == 5 { v * 24.0 } else { v }
        })
        .collect();
    cpu_ref::round_bf16(&x)
}

/// A q8 cache `[kvh, max_seq, d]` whose first `len` positions hold `x`'s
/// rows (`[kvh, max_seq, d]` f32, bf16-rounded) quantized by the GPU writer,
/// and its bf16 dequantization (the staging kernel, every position).
pub(crate) fn q8_cache_of(
    ctx: &MetalContext,
    x: &[f32],
    (kvh, max_seq, d): (usize, usize, usize),
    len: usize,
) -> (KvCache, Tensor) {
    let cache = KvCache::zeros(ctx, KvFormat::Q8, kvh, max_seq, d).expect("q8 cache");
    // scatter_kv takes [M, KVH, D] rows.
    let mut rows = vec![0.0f32; len * kvh * d];
    for h in 0..kvh {
        for t in 0..len {
            rows[(t * kvh + h) * d..(t * kvh + h + 1) * d]
                .copy_from_slice(&x[(h * max_seq + t) * d..(h * max_seq + t + 1) * d]);
        }
    }
    let rows = Tensor::from_f32_as_bf16(ctx, &rows, &[len, kvh, d]).expect("rows");
    let staged = Tensor::zeros(ctx, &[kvh, max_seq, d], DType::BF16).expect("staged");
    let pass = ctx.begin().expect("pass");
    scatter_kv(ctx, &pass, &cache, &rows, 0).expect("scatter");
    pass.level_barrier(&[&cache.values]).expect("barrier");
    kv_dequant_q8(ctx, &pass, &cache, &staged, max_seq).expect("dequant");
    pass.commit_wait().expect("commit");
    (cache, staged)
}

/// Asserts a GPU-written q8 row equals the CPU q8_0 of `x`: the scales
/// exactly, the values within one step (a quotient that lands on .5 may
/// round either way under fast math).
fn assert_q8_row(values: &[i8], scales: &[half::f16], x: &[f32], at: &str) {
    let (want_v, want_s) = cpu_ref::quantize_q8(x, KV_Q8_GROUP);
    assert_eq!(scales, &want_s[..], "{at}: scales");
    for (i, (g, w)) in values.iter().zip(&want_v).enumerate() {
        assert!(
            (i16::from(*g) - i16::from(*w)).abs() <= 1,
            "{at}: value {i}: {g} vs {w}"
        );
    }
    // Every value within half a step of the input (plus the one-step slack),
    // except in a saturated group (scale at half's maximum).
    for (i, ((&q, &v), s)) in values
        .iter()
        .zip(x)
        .zip(scales.iter().flat_map(|s| std::iter::repeat_n(s.to_f32(), KV_Q8_GROUP)))
        .enumerate()
    {
        if s < 65504.0 {
            assert!((f32::from(q) * s - v).abs() <= 1.5 * s + 1e-6, "{at}: value {i}");
        }
    }
}

fn q8_parts(cache: &KvCache) -> (Vec<i8>, Vec<half::f16>) {
    let values = cache.values.raw_bytes().iter().map(|&b| b as i8).collect();
    let scales =
        bytemuck::cast_slice(cache.scales.as_ref().expect("scales").raw_bytes())
            .to_vec();
    (values, scales)
}

/// The scatter writer quantizes each row as q8_0 does, at the right slots,
/// leaving the others zero; the staging kernel reads back value x scale.
#[test]
fn q8_scatter_matches_the_cpu_quantizer_and_dequantizes() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(81);
    let (kvh, max_seq, d, m, base) = (2usize, 16usize, 256usize, 5usize, 3usize);
    let rows = kv_like_rows(&mut rng, m * kvh, d);
    // A zero group: scale 0, values 0. A group of tiny values: a subnormal
    // half scale. A group past 127 x 65504: the scale saturates at half's
    // maximum instead of overflowing to infinity (whose zeros would
    // dequantize to NaN).
    let mut rows = rows;
    rows[..KV_Q8_GROUP].fill(0.0);
    let g = KV_Q8_GROUP;
    for (i, v) in rows[g..2 * g].iter_mut().enumerate() {
        *v = (i as f32 - 15.5) * 1e-6;
    }
    for (i, v) in rows[2 * g..3 * g].iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.6e7 } else { -3.0 };
    }
    let rows = cpu_ref::round_bf16(&rows);
    let cache = KvCache::zeros(&ctx, KvFormat::Q8, kvh, max_seq, d).expect("cache");
    let t_rows = Tensor::from_f32_as_bf16(&ctx, &rows, &[m, kvh, d]).expect("rows");
    let staged = Tensor::zeros(&ctx, &[kvh, max_seq, d], DType::BF16).expect("staged");
    let pass = ctx.begin().expect("pass");
    scatter_kv(&ctx, &pass, &cache, &t_rows, base).expect("scatter");
    pass.level_barrier(&[]).expect("barrier");
    kv_dequant_q8(&ctx, &pass, &cache, &staged, max_seq).expect("dequant");
    pass.commit_wait().expect("commit");

    let (values, scales) = q8_parts(&cache);
    let staged = staged.to_f32().expect("staged");
    let g = d / KV_Q8_GROUP;
    for h in 0..kvh {
        for t in 0..max_seq {
            let slot = h * max_seq + t;
            let (v, s) =
                (&values[slot * d..(slot + 1) * d], &scales[slot * g..(slot + 1) * g]);
            if (base..base + m).contains(&t) {
                let x =
                    &rows[((t - base) * kvh + h) * d..((t - base) * kvh + h + 1) * d];
                assert_q8_row(v, s, x, &format!("head {h} slot {t}"));
            } else {
                assert!(
                    v.iter().all(|&b| b == 0) && s.iter().all(|s| s.to_f32() == 0.0)
                );
            }
            for i in 0..d {
                let want =
                    half::bf16::from_f32(f32::from(v[i]) * s[i / KV_Q8_GROUP].to_f32());
                assert_eq!(staged[slot * d + i], want.to_f32(), "dequant {h} {t} {i}");
            }
        }
    }
    let (v0, s0) = (&values[base * d..base * d + KV_Q8_GROUP], scales[base * g]);
    assert!(v0.iter().all(|&b| b == 0) && s0.to_f32() == 0.0, "the zero group");
    let tiny = scales[base * g + 1];
    assert!(!tiny.is_normal() && tiny.to_f32() > 0.0, "subnormal scale {tiny}");
    let big = scales[base * g + 2];
    assert_eq!(big.to_f32(), 65504.0, "saturated scale");
    assert!(staged.iter().all(|x| x.is_finite()), "a dequantized value is not finite");
}

/// The fused decode K prep into a q8 cache stores the q8_0 of exactly the
/// row the bf16 kernel stores.
#[test]
fn q8_decode_k_prep_quantizes_the_bf16_kernels_row() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(82);
    let (kvh, max_seq, d, rot, pos) = (2usize, 64usize, 256usize, 64usize, 41usize);
    let k = kv_like_rows(&mut rng, kvh, d);
    let w: Vec<f32> = (0..d).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
    let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, d]).expect("k");
    let t_w = Tensor::from_f32_as_bf16(&ctx, &w, &[d]).expect("w");
    let bf = Tensor::zeros(&ctx, &[kvh, max_seq, d], DType::BF16).expect("bf16 cache");
    let q8 = KvCache::zeros(&ctx, KvFormat::Q8, kvh, max_seq, d).expect("q8 cache");
    let pass = ctx.begin().expect("pass");
    k_norm_rope_scatter_decode(&ctx, &pass, &t_k, &t_w, &bf, rot, pos, 1e7, 1e-6, 3)
        .expect("bf16 prep");
    k_norm_rope_scatter_decode(&ctx, &pass, &t_k, &t_w, &q8, rot, pos, 1e7, 1e-6, 3)
        .expect("q8 prep");
    pass.commit_wait().expect("commit");
    let bf = bf.to_f32().expect("bf16");
    let (values, scales) = q8_parts(&q8);
    let g = d / KV_Q8_GROUP;
    for h in 0..kvh {
        let slot = h * max_seq + pos;
        assert_q8_row(
            &values[slot * d..(slot + 1) * d],
            &scales[slot * g..(slot + 1) * g],
            &bf[slot * d..(slot + 1) * d],
            &format!("head {h}"),
        );
    }
}

/// Dense decode over a q8 cache against the bf16 kernel over the same cache
/// dequantized: one split, a full one, ragged tails, up to the dense limit.
/// Only the bf16 rounding of the dequantized operands differs.
#[test]
fn q8_sdpa_decode_matches_bf16_over_the_dequantized_cache() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(83);
    // The model's GQA shape: 24 query heads over 2 KV heads.
    let (nq, kvh, d, max_seq) = (24usize, 2usize, 256usize, 2056usize);
    let scale = 1.0 / (d as f32).sqrt();
    let k = kv_like_rows(&mut rng, kvh * max_seq, d);
    let v = kv_like_rows(&mut rng, kvh * max_seq, d);
    let (kq, ks) = q8_cache_of(&ctx, &k, (kvh, max_seq, d), max_seq);
    let (vq, vs) = q8_cache_of(&ctx, &v, (kvh, max_seq, d), max_seq);
    let splits = max_seq.div_ceil(SDPA_SPLIT);
    let partials = Tensor::zeros(&ctx, &[nq, splits, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[nq, splits, 2], DType::F32).expect("stats");
    for len in [1usize, 33, 256, 257, 700, 2051] {
        let q = cpu_ref::round_bf16(
            &(0..nq * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect::<Vec<_>>(),
        );
        let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[nq, d]).expect("q");
        let out_q8 = Tensor::zeros(&ctx, &[nq, d], DType::BF16).expect("out");
        let out_bf = Tensor::zeros(&ctx, &[nq, d], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        sdpa_decode(
            &ctx,
            &pass,
            &t_q,
            &kq,
            &vq,
            &out_q8,
            len,
            scale,
            Some((&partials, &stats)),
        )
        .expect("q8 decode");
        pass.level_barrier(&[]).expect("barrier");
        sdpa_decode(
            &ctx,
            &pass,
            &t_q,
            &ks,
            &vs,
            &out_bf,
            len,
            scale,
            Some((&partials, &stats)),
        )
        .expect("bf16 decode");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(
            &out_q8.to_f32().expect("q8"),
            &out_bf.to_f32().expect("bf16"),
            2e-2,
            2e-2,
        );
    }
}

/// A q8 cache is never read as bf16 (or the other way round): the dense
/// prefill kernel, a decode without its split scratch, mismatched K and V
/// formats, and a split route past the dense lengths all refuse.
#[test]
fn q8_caches_are_refused_where_no_q8_kernel_reads_them() {
    let ctx = MetalContext::new().expect("metal context");
    let (nq, kvh, d, max_seq) = (16usize, 2usize, 256usize, 9000usize);
    let q8 = KvCache::zeros(&ctx, KvFormat::Q8, kvh, max_seq, d).expect("q8");
    let bf = KvCache::zeros(&ctx, KvFormat::Bf16, kvh, max_seq, d).expect("bf16");
    let q = Tensor::zeros(&ctx, &[1, nq, d], DType::BF16).expect("q");
    let out = Tensor::zeros(&ctx, &[1, nq, d], DType::BF16).expect("out");
    let partials = Tensor::zeros(&ctx, &[nq, 64, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[nq, 64, 2], DType::F32).expect("stats");
    let scratch = Some((&partials, &stats));
    let pass = ctx.begin().expect("pass");
    assert!(
        sdpa_prefill(&ctx, &pass, &q, &q8.values, &q8.values, &out, 0, 1.0).is_err()
    );
    assert!(sdpa_decode(&ctx, &pass, &q, &q8, &q8, &out, 10, 1.0, None).is_err());
    assert!(sdpa_decode(&ctx, &pass, &q, &q8, &bf, &out, 10, 1.0, scratch).is_err());
    assert!(sdpa_decode(&ctx, &pass, &q, &q8, &q8, &out, 8192, 1.0, scratch).is_err());
    // Values without their scales are not a cache.
    let bare = Kv { values: &q8.values, scales: None };
    assert!(sdpa_decode(&ctx, &pass, &q, bare, bare, &out, 10, 1.0, scratch).is_err());
    pass.commit_wait().expect("commit");
}
