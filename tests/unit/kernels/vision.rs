use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

/// Grid coordinates of block-major patch row `r` (VISION.md "Patchify").
fn grid_coords(r: usize, gw: usize) -> (usize, usize) {
    let blocks_w = gw / 2;
    let iw = r % 2;
    let ih = (r / 2) % 2;
    let bw = (r / 4) % blocks_w;
    let bh = r / (4 * blocks_w);
    (bh * 2 + ih, bw * 2 + iw)
}

/// One axis of the align_corners bilinear resample, as vision_utils computes
/// it in float32: taps and weights for target index `i` on an axis of `n`.
fn axis_taps(i: usize, n: usize, side: usize) -> [(usize, f32); 2] {
    let src = i as f32 * (side - 1) as f32 / (n.max(2) - 1) as f32;
    let f = src.floor();
    let t0 = f as i64;
    let clamp = |t: i64| t.clamp(0, side as i64 - 1) as usize;
    [
        (clamp(t0), (1.0 - (src - f).abs()).max(0.0)),
        (clamp(t0 + 1), (1.0 - (src - f - 1.0).abs()).max(0.0)),
    ]
}

#[test]
fn patch_pos_embed_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(31);
    // A grid larger than the table on one axis and smaller on the other, so
    // both the interpolating and the extrapolating (clamped) taps run.
    let (gh, gw, side, h) = (8, 4, 6, 24);
    let n = gh * gw;
    let x: Vec<f32> = (0..n * h).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let table: Vec<f32> =
        (0..side * side * h).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[n, h]).expect("x");
    let tt = Tensor::from_f32_as_bf16(&ctx, &table, &[side * side, h]).expect("table");
    let pass = ctx.begin().expect("pass");
    patch_pos_embed_bf16(&ctx, &pass, &tx, &tt, gh, gw).expect("pos embed");
    pass.commit_wait().expect("commit");

    let (rx, rt) = (cpu_ref::round_bf16(&x), cpu_ref::round_bf16(&table));
    let mut expected = vec![0.0f32; n * h];
    for r in 0..n {
        let (row, col) = grid_coords(r, gw);
        let ht = axis_taps(row, gh, side);
        let wt = axis_taps(col, gw, side);
        for c in 0..h {
            let mut blend = 0.0f32;
            for (hi, hw) in ht {
                for (wi, ww) in wt {
                    blend += hw * ww * rt[(hi * side + wi) * h + c];
                }
            }
            expected[r * h + c] = rx[r * h + c] + blend;
        }
    }
    // Two bf16 roundings (blend, final add) on values of order 1.
    cpu_ref::assert_close(&tx.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

/// The 2-D rotary of one head (`d` wide) at grid position `(row, col)`.
fn rope_2d(x: &mut [f32], row: usize, col: usize, theta: f32) {
    let d = x.len();
    let half = d / 2;
    let quarter = half / 2;
    for i in 0..half {
        let fi = i % quarter;
        let inv_freq = 1.0 / theta.powf(2.0 * fi as f32 / half as f32);
        let pos = if i < quarter { row } else { col } as f32;
        let (sin, cos) = (pos * inv_freq).sin_cos();
        let lo = x[i];
        let hi = x[i + half];
        x[i] = lo * cos - hi * sin;
        x[i + half] = hi * cos + lo * sin;
    }
}

#[test]
fn qkv_rope_2d_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(32);
    let (gh, gw, heads, d) = (6, 4, 3, 8);
    let n = gh * gw;
    let h = heads * d;
    let theta = 10000.0f32;
    let qkv: Vec<f32> = (0..n * 3 * h).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let t = Tensor::from_f32_as_bf16(&ctx, &qkv, &[n, 3 * h]).expect("qkv");
    let pass = ctx.begin().expect("pass");
    qkv_rope_2d_bf16(&ctx, &pass, &t, heads, gh, gw, theta).expect("rope");
    pass.commit_wait().expect("commit");

    // v (the last third) must come out untouched.
    let mut expected = cpu_ref::round_bf16(&qkv);
    for r in 0..n {
        let (row, col) = grid_coords(r, gw);
        for which in 0..2 {
            for head in 0..heads {
                let start = r * 3 * h + which * h + head * d;
                rope_2d(&mut expected[start..start + d], row, col, theta);
            }
        }
    }
    cpu_ref::assert_close(&t.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

fn check_attention_full(n: usize, heads: usize, seed: u64) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(seed);
    let d = ATTN_D;
    let h = heads * d;
    let scale = 1.0 / (d as f32).sqrt();
    let qkv: Vec<f32> = (0..n * 3 * h).map(|_| rng.gen_range(-1.0f32..1.0)).collect();

    let t = Tensor::from_f32_as_bf16(&ctx, &qkv, &[n, 3 * h]).expect("qkv");
    let out = Tensor::zeros(&ctx, &[n, h], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    attention_full_bf16(&ctx, &pass, &t, &out, heads, scale).expect("attention");
    pass.commit_wait().expect("commit");

    let r = cpu_ref::round_bf16(&qkv);
    let at = |row: usize, which: usize, head: usize, i: usize| {
        r[row * 3 * h + which * h + head * d + i]
    };
    let mut expected = vec![0.0f32; n * h];
    for head in 0..heads {
        for q in 0..n {
            let mut scores: Vec<f32> = (0..n)
                .map(|k| {
                    (0..d).map(|i| at(q, 0, head, i) * at(k, 1, head, i)).sum::<f32>()
                })
                .map(|s| s * scale)
                .collect();
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for i in 0..d {
                let acc: f32 = (0..n).map(|k| scores[k] * at(k, 2, head, i)).sum();
                expected[q * h + head * d + i] = acc / sum;
            }
        }
    }
    // bf16 probabilities and a bf16 output over unit-scale values.
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn attention_full_matches_cpu() {
    // Query rows short of one tile, one exact key tile, and a ragged key
    // tail with several query tiles; two heads so the head offset into the
    // fused rows is exercised.
    check_attention_full(20, 2, 40);
    check_attention_full(128, 1, 41);
    check_attention_full(300, 2, 42);
}

#[test]
fn attention_is_bidirectional() {
    // A key far past the query must influence the output: with all keys
    // equal except the last, the last row's value shows up in row 0.
    let ctx = MetalContext::new().expect("metal context");
    let (n, heads) = (40, 1);
    let h = heads * ATTN_D;
    let mut qkv = vec![0.0f32; n * 3 * h];
    for r in 0..n {
        qkv[r * 3 * h] = 1.0; // q[0] = 1 for every row
        qkv[r * 3 * h + h] = if r == n - 1 { 4.0 } else { 0.0 }; // k[0]
        qkv[r * 3 * h + 2 * h] = if r == n - 1 { 1.0 } else { 0.0 }; // v[0]
    }
    let t = Tensor::from_f32_as_bf16(&ctx, &qkv, &[n, 3 * h]).expect("qkv");
    let out = Tensor::zeros(&ctx, &[n, h], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    attention_full_bf16(&ctx, &pass, &t, &out, heads, 1.0).expect("attention");
    pass.commit_wait().expect("commit");
    let got = out.to_f32().expect("read");
    let p_last = 4.0f32.exp() / (4.0f32.exp() + (n - 1) as f32);
    assert!(
        (got[0] - p_last).abs() < 1e-2,
        "row 0 saw the last key: {} vs {p_last}",
        got[0]
    );
}
