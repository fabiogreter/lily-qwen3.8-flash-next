use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::kernels::quant::gemv_quant;
use crate::weights::QuantWeights;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn random(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

/// Root mean square of a vector: the scale for an absolute tolerance on a
/// long dot product, where cancellation makes elementwise relative bounds
/// meaningless.
fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

#[test]
fn grouped_rmsnorm_matches_per_segment_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(21);
    let (rows, g, h) = (3, 4, 512);
    let x = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -2.0, 2.0));
    let w = cpu_ref::round_bf16(&random(&mut rng, g * h, -0.5, 0.5));
    let eps = 1e-6;

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[rows, g * h]).expect("x");
    let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[g * h]).expect("w");
    let out = Tensor::zeros(&ctx, &[rows, g * h], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    rmsnorm_grouped_bf16(&ctx, &pass, &tx, &tw, &out, h, g, eps, 1.0).expect("norm");
    pass.commit_wait().expect("commit");

    // Each (row, segment) is an independent RMSNorm with its own weight slice.
    let mut expected = vec![0.0f32; rows * g * h];
    for r in 0..rows {
        for s in 0..g {
            let seg = &x[(r * g + s) * h..(r * g + s + 1) * h];
            let gain: Vec<f32> =
                w[s * h..(s + 1) * h].iter().map(|v| 1.0 + v).collect();
            let normed = cpu_ref::rmsnorm(seg, &gain, h, eps);
            expected[(r * g + s) * h..(r * g + s + 1) * h].copy_from_slice(&normed);
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn mix_and_inject_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(22);
    let (rows, g, h) = (2, 4, 640);
    let up = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -4.0, 4.0));
    let hn = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -2.0, 2.0));
    let hyper = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -2.0, 2.0));
    let branch = cpu_ref::round_bf16(&random(&mut rng, rows * h, -2.0, 2.0));
    let inj = cpu_ref::round_bf16(&random(&mut rng, rows * g, -6.0, 6.0));

    let t_up = Tensor::from_f32_as_bf16(&ctx, &up, &[rows, g * h]).expect("up");
    let t_hn = Tensor::from_f32_as_bf16(&ctx, &hn, &[rows, g * h]).expect("hn");
    let t_mixed = Tensor::zeros(&ctx, &[rows, h], DType::BF16).expect("mixed");
    let t_hyper =
        Tensor::from_f32_as_bf16(&ctx, &hyper, &[rows, g * h]).expect("hyper");
    let t_branch = Tensor::from_f32_as_bf16(&ctx, &branch, &[rows, h]).expect("branch");
    let t_inj = Tensor::from_f32_as_bf16(&ctx, &inj, &[rows, g]).expect("inj");

    let pass = ctx.begin().expect("pass");
    hc_mix_bf16(&ctx, &pass, &t_up, &t_hn, &t_mixed, h, g).expect("mix");
    hc_inject_bf16(&ctx, &pass, &t_hyper, &t_branch, &t_inj, h, g).expect("inject");
    pass.commit_wait().expect("commit");

    let mut mixed = vec![0.0f32; rows * h];
    let mut injected = hyper.clone();
    for r in 0..rows {
        for i in 0..h {
            let mut acc = 0.0f32;
            for s in 0..g {
                let at = r * g * h + s * h + i;
                acc += sigmoid(up[at]) * hn[at];
                let weight = 2.0 * sigmoid(inj[r * g + s] / g as f32);
                injected[at] += branch[r * h + i] * weight;
            }
            mixed[r * h + i] = acc / g as f32;
        }
    }
    cpu_ref::assert_close(&t_mixed.to_f32().expect("mixed"), &mixed, 2e-2, 2e-2);
    cpu_ref::assert_close(&t_hyper.to_f32().expect("hyper"), &injected, 3e-2, 2e-2);
}

#[test]
fn broadcast_and_scaled_silu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(23);
    let (rows, g, h) = (2, 4, 96);
    let x = cpu_ref::round_bf16(&random(&mut rng, rows * h, -3.0, 3.0));
    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[rows, h]).expect("x");
    let hyper = Tensor::zeros(&ctx, &[rows, g * h], DType::BF16).expect("hyper");
    let act = Tensor::zeros(&ctx, &[rows, h], DType::BF16).expect("act");
    let pass = ctx.begin().expect("pass");
    hc_broadcast_bf16(&ctx, &pass, &tx, &hyper, h, g).expect("broadcast");
    silu_scaled_bf16(&ctx, &pass, &tx, &act, 0.25).expect("silu");
    pass.commit_wait().expect("commit");

    let got = hyper.to_f32().expect("hyper");
    for r in 0..rows {
        for s in 0..g {
            assert_eq!(
                &got[(r * g + s) * h..(r * g + s + 1) * h],
                &x[r * h..(r + 1) * h]
            );
        }
    }
    let expected: Vec<f32> = x.iter().map(|v| cpu_ref::silu(v * 0.25)).collect();
    cpu_ref::assert_close(&act.to_f32().expect("act"), &expected, 1e-2, 1e-2);
}

/// Random Q8 `[n, k]` weight (group size 64) with its dequantized image.
fn random_q8(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
) -> (QuantWeights, Vec<f32>) {
    const GS: usize = 64;
    let words = k / 4;
    let groups = k / GS;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.0005f32..0.01)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_affine(&codes, &scales, &biases, n, k, GS, 8);
    let bf =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    let w = QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&bf(&scales)),
            &[n, groups],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&bf(&biases)),
            &[n, groups],
            DType::BF16,
        )
        .expect("biases"),
        group_size: GS,
        bits: 8,
    };
    (w, dequant)
}

/// Normalized stream as the kernels compute it, `bf16(x * inv_rms[g] * gain)`
/// with `gain = w_bias + w`: two f32 multiplies and no fusable add, so the
/// CPU value is bit-identical.
fn hn_bf16(
    hyper: &[f32],
    norm_w: &[f32],
    inv_rms: &[f32],
    h: usize,
    w_bias: f32,
) -> Vec<f32> {
    hyper
        .iter()
        .zip(norm_w)
        .enumerate()
        .map(|(e, (x, w))| bf16::from_f32(x * inv_rms[e / h] * (w_bias + w)).to_f32())
        .collect()
}

/// The fused decode read gate (`hc_read_down_q8` + `hc_read_up_mix_q8`)
/// against the six unfused kernels it replaces and an f32 CPU chain, on the
/// same random Q8 weights. The down half keeps the normalized stream in f32
/// (the unfused path rounds it to bf16), so `down`, `inj` and `mixed` agree
/// within bf16 tolerance; `inv_rms` is an f32 reduction in a different order
/// and agrees to ~1e-6. Covers the model's shape (H=2560, G=4, R=320), a
/// small shape whose down GEMV leaves lanes idle, and the inject-less final
/// mixer. The up half's exactness has its own test.
#[test]
fn fused_read_gate_matches_unfused_kernels() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(24);
    let eps = 1e-6f32;
    let w_bias = 1.0f32;
    for (h, g, r, with_inject) in [
        (2560, 4, 320, true),
        (2560, 4, 320, false),
        (192, 3, 128, true),
        (64, 8, 64, true),
    ] {
        let k = g * h;
        let hyper = cpu_ref::round_bf16(&random(&mut rng, k, -3.0, 3.0));
        let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
        let (down_w, down_deq) = random_q8(&ctx, &mut rng, r, k);
        let (up_w, up_deq) = random_q8(&ctx, &mut rng, k, r);
        let inject = with_inject.then(|| random_q8(&ctx, &mut rng, g, k));

        let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper, &[k]).expect("hyper");
        let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
        let zeros = |shape: &[usize], dtype| {
            Tensor::zeros(&ctx, shape, dtype).expect("scratch")
        };

        // Unfused reference chain on the GPU.
        let hn = zeros(&[k], DType::BF16);
        let down = zeros(&[r], DType::BF16);
        let inj = zeros(&[g], DType::BF16);
        let act = zeros(&[r], DType::BF16);
        let up = zeros(&[k], DType::BF16);
        let mixed = zeros(&[h], DType::BF16);
        let pass = ctx.begin().expect("pass");
        rmsnorm_grouped_bf16(&ctx, &pass, &t_hyper, &t_norm, &hn, h, g, eps, w_bias)
            .expect("norm");
        gemv_quant(&ctx, &pass, &down_w, &hn, &down).expect("down");
        if let Some((inj_w, _)) = &inject {
            gemv_quant(&ctx, &pass, inj_w, &hn, &inj).expect("inject");
        }
        silu_scaled_bf16(&ctx, &pass, &down, &act, 1.0 / g as f32).expect("silu");
        gemv_quant(&ctx, &pass, &up_w, &act, &up).expect("up");
        hc_mix_bf16(&ctx, &pass, &up, &hn, &mixed, h, g).expect("mix");
        pass.commit_wait().expect("commit");

        // Fused pair.
        let f_down = zeros(&[r], DType::BF16);
        let f_inj = zeros(&[g], DType::BF16);
        let f_inv_rms = zeros(&[g], DType::F32);
        let f_mixed = zeros(&[h], DType::BF16);
        let pass = ctx.begin().expect("pass");
        hc_read_down_q8(
            &ctx,
            &pass,
            &t_hyper,
            &t_norm,
            &down_w,
            inject.as_ref().map(|(w, _)| w),
            &f_down,
            &f_inj,
            &f_inv_rms,
            h,
            g,
            eps,
            w_bias,
        )
        .expect("fused down");
        hc_read_up_mix_q8(
            &ctx, &pass, &up_w, &f_down, &t_hyper, &t_norm, &f_inv_rms, &f_mixed, h, g,
            w_bias,
        )
        .expect("fused up");
        pass.commit_wait().expect("commit");

        // Against an f32 CPU chain with the fused kernels' rounding points
        // (f32 hn for the down half; bf16 down, act, up and hn for the up
        // half): differences are f32 accumulation order plus the bf16 output
        // rounding.
        let mut c_hn = vec![0.0f32; k];
        let mut c_inv_rms = vec![0.0f32; g];
        for s in 0..g {
            let seg = &hyper[s * h..(s + 1) * h];
            let gain: Vec<f32> =
                norm_w[s * h..(s + 1) * h].iter().map(|v| w_bias + v).collect();
            c_hn[s * h..(s + 1) * h]
                .copy_from_slice(&cpu_ref::rmsnorm(seg, &gain, h, eps));
            let mean_sq = seg.iter().map(|v| v * v).sum::<f32>() / h as f32;
            c_inv_rms[s] = 1.0 / (mean_sq + eps).sqrt();
        }
        let c_down = cpu_ref::round_bf16(&cpu_ref::gemm_nt(&c_hn, &down_deq, 1, k, r));
        let c_act: Vec<f32> = cpu_ref::round_bf16(
            &c_down.iter().map(|v| cpu_ref::silu(v / g as f32)).collect::<Vec<_>>(),
        );
        let c_up = cpu_ref::round_bf16(&cpu_ref::gemm_nt(&c_act, &up_deq, 1, r, k));
        let c_hn_bf16 = hn_bf16(&hyper, &norm_w, &c_inv_rms, h, w_bias);
        let c_mixed: Vec<f32> = (0..h)
            .map(|i| {
                (0..g)
                    .map(|s| sigmoid(c_up[s * h + i]) * c_hn_bf16[s * h + i])
                    .sum::<f32>()
                    / g as f32
            })
            .collect();
        cpu_ref::assert_close(
            &f_inv_rms.to_f32().expect("inv_rms"),
            &c_inv_rms,
            1e-6,
            1e-5,
        );
        cpu_ref::assert_close(
            &f_down.to_f32().expect("down"),
            &c_down,
            1e-2 * rms(&c_down),
            1e-2,
        );
        if let Some((inj_w, inj_deq)) = &inject {
            let c_inj = cpu_ref::gemm_nt(&c_hn, inj_deq, 1, k, inj_w.out_features());
            cpu_ref::assert_close(
                &f_inj.to_f32().expect("inj"),
                &c_inj,
                1e-2 * rms(&c_inj),
                1e-2,
            );
        }
        cpu_ref::assert_close(
            &f_mixed.to_f32().expect("mixed"),
            &c_mixed,
            1e-2 * rms(&c_mixed),
            1e-2,
        );

        // Against the unfused GPU chain. The unfused path rounds hn to bf16
        // before a K = G*H dot, whose error is a random walk of about
        // 2^-9 * sqrt(K) * |term| and shows up in cancelling outputs, so the
        // bound on the logits is 1% of the vector's RMS plus 1% relative. The
        // gates then amplify that difference (these random logits are O(100),
        // far larger than the model's), so `mixed` only gets a coarse bound;
        // its precise checks are the CPU chain above and the exactness test.
        let close = |got: &Tensor, want: &Tensor, frac: f32| {
            let want = want.to_f32().expect("reference");
            cpu_ref::assert_close(
                &got.to_f32().expect("fused"),
                &want,
                frac * rms(&want),
                frac,
            );
        };
        close(&f_down, &down, 1e-2);
        if with_inject {
            close(&f_inj, &inj, 1e-2);
        } else {
            assert!(
                f_inj.to_f32().expect("inj").iter().all(|v| *v == 0.0),
                "inj untouched"
            );
        }
        close(&f_mixed, &mixed, 1e-1);
    }
}

/// `hc_read_up_mix_q8` is bit-identical to the unfused `silu_scaled_bf16`,
/// `gemv_quant` and `hc_mix_bf16` sequence given the same `down` logits and
/// `inv_rms`: the unfused side gets `hn` built on the CPU with the kernel's
/// own expression.
#[test]
fn fused_up_mix_is_bit_identical_to_unfused_kernels() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(26);
    let w_bias = 1.0f32;
    for (h, g, r) in [(2560, 4, 320), (192, 3, 128), (64, 8, 64), (256, 1, 64)] {
        let k = g * h;
        let hyper = cpu_ref::round_bf16(&random(&mut rng, k, -3.0, 3.0));
        let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
        let down = cpu_ref::round_bf16(&random(&mut rng, r, -6.0, 6.0));
        let inv_rms: Vec<f32> = random(&mut rng, g, 0.3, 1.2);
        let (up_w, _) = random_q8(&ctx, &mut rng, k, r);
        let hn = hn_bf16(&hyper, &norm_w, &inv_rms, h, w_bias);

        let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper, &[k]).expect("hyper");
        let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
        let t_down = Tensor::from_f32_as_bf16(&ctx, &down, &[r]).expect("down");
        let t_inv_rms = Tensor::from_f32(&ctx, &inv_rms, &[g]).expect("inv_rms");
        let t_hn = Tensor::from_f32_as_bf16(&ctx, &hn, &[k]).expect("hn");
        let zeros = |shape: &[usize], dtype| {
            Tensor::zeros(&ctx, shape, dtype).expect("scratch")
        };
        let act = zeros(&[r], DType::BF16);
        let up = zeros(&[k], DType::BF16);
        let mixed = zeros(&[h], DType::BF16);
        let f_mixed = zeros(&[h], DType::BF16);

        let pass = ctx.begin().expect("pass");
        silu_scaled_bf16(&ctx, &pass, &t_down, &act, 1.0 / g as f32).expect("silu");
        gemv_quant(&ctx, &pass, &up_w, &act, &up).expect("up");
        hc_mix_bf16(&ctx, &pass, &up, &t_hn, &mixed, h, g).expect("mix");
        hc_read_up_mix_q8(
            &ctx, &pass, &up_w, &t_down, &t_hyper, &t_norm, &t_inv_rms, &f_mixed, h, g,
            w_bias,
        )
        .expect("fused up");
        pass.commit_wait().expect("commit");
        assert_eq!(
            f_mixed.to_f32().expect("mixed"),
            mixed.to_f32().expect("mixed"),
            "h={h} g={g} r={r}"
        );
    }
}

/// Kernel-level timing of the fused decode read gate against the unfused
/// six-kernel sequence at the model's shape, on a concurrent pass with a
/// level barrier after every stage (the decode graph's shape), cycling
/// through enough weight sets that every read streams its weights from DRAM
/// as the model does. Prints µs per read; run with
/// `cargo test --release -- --ignored --nocapture fused_read_gate_timing`.
#[test]
#[ignore = "timing only"]
fn fused_read_gate_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(25);
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    let sets = 64; // 64 * 6.6 MB = 420 MB of weights, well past the SLC
    let hyper = cpu_ref::round_bf16(&random(&mut rng, k, -3.0, 3.0));
    let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
    let weights: Vec<(QuantWeights, QuantWeights, QuantWeights)> = (0..sets)
        .map(|_| {
            (
                random_q8(&ctx, &mut rng, r, k).0,
                random_q8(&ctx, &mut rng, k, r).0,
                random_q8(&ctx, &mut rng, g, k).0,
            )
        })
        .collect();
    let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper, &[k]).expect("hyper");
    let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
    let zeros =
        |shape: &[usize], dtype| Tensor::zeros(&ctx, shape, dtype).expect("scratch");
    let hn = zeros(&[k], DType::BF16);
    let down = zeros(&[r], DType::BF16);
    let inj = zeros(&[g], DType::BF16);
    let act = zeros(&[r], DType::BF16);
    let up = zeros(&[k], DType::BF16);
    let mixed = zeros(&[h], DType::BF16);
    let inv_rms = zeros(&[g], DType::F32);
    let iters = 256;
    type Read<'a> =
        dyn Fn(&ComputePass<'_>, &QuantWeights, &QuantWeights, &QuantWeights) + 'a;
    let time = |name: &str, f: &Read<'_>| {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..iters {
                let (d, u, j) = &weights[i % sets];
                f(&pass, d, u, j);
            }
            let start = std::time::Instant::now();
            pass.commit_wait().expect("commit");
            best = best.min(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        eprintln!("{name}: {best:.1} us per read (best of 3)");
    };
    time("unfused (6 dispatches)", &|pass, down_w, up_w, inj_w| {
        rmsnorm_grouped_bf16(&ctx, pass, &t_hyper, &t_norm, &hn, h, g, 1e-6, 1.0)
            .unwrap();
        pass.level_barrier(&[&hn]).unwrap();
        gemv_quant(&ctx, pass, down_w, &hn, &down).unwrap();
        gemv_quant(&ctx, pass, inj_w, &hn, &inj).unwrap();
        pass.level_barrier(&[&down, &inj]).unwrap();
        silu_scaled_bf16(&ctx, pass, &down, &act, 0.25).unwrap();
        pass.level_barrier(&[&act]).unwrap();
        gemv_quant(&ctx, pass, up_w, &act, &up).unwrap();
        pass.level_barrier(&[&up]).unwrap();
        hc_mix_bf16(&ctx, pass, &up, &hn, &mixed, h, g).unwrap();
        pass.level_barrier(&[&mixed]).unwrap();
    });
    time("fused (2 dispatches)", &|pass, down_w, up_w, inj_w| {
        hc_read_down_q8(
            &ctx,
            pass,
            &t_hyper,
            &t_norm,
            down_w,
            Some(inj_w),
            &down,
            &inj,
            &inv_rms,
            h,
            g,
            1e-6,
            1.0,
        )
        .unwrap();
        pass.level_barrier(&[&down, &inj, &inv_rms]).unwrap();
        hc_read_up_mix_q8(
            &ctx, pass, up_w, &down, &t_hyper, &t_norm, &inv_rms, &mixed, h, g, 1.0,
        )
        .unwrap();
        pass.level_barrier(&[&mixed]).unwrap();
    });
    time("fused down only", &|pass, down_w, _, inj_w| {
        hc_read_down_q8(
            &ctx,
            pass,
            &t_hyper,
            &t_norm,
            down_w,
            Some(inj_w),
            &down,
            &inj,
            &inv_rms,
            h,
            g,
            1e-6,
            1.0,
        )
        .unwrap();
        pass.level_barrier(&[&down, &inj, &inv_rms]).unwrap();
    });
    time("fused up+mix only", &|pass, _, up_w, _| {
        hc_read_up_mix_q8(
            &ctx, pass, up_w, &down, &t_hyper, &t_norm, &inv_rms, &mixed, h, g, 1.0,
        )
        .unwrap();
        pass.level_barrier(&[&mixed]).unwrap();
    });
    time("unfused norm + down + inject gemv", &|pass, down_w, _, inj_w| {
        rmsnorm_grouped_bf16(&ctx, pass, &t_hyper, &t_norm, &hn, h, g, 1e-6, 1.0)
            .unwrap();
        pass.level_barrier(&[&hn]).unwrap();
        gemv_quant(&ctx, pass, down_w, &hn, &down).unwrap();
        gemv_quant(&ctx, pass, inj_w, &hn, &inj).unwrap();
        pass.level_barrier(&[&down, &inj]).unwrap();
    });
    time("unfused silu + up gemv + mix", &|pass, _, up_w, _| {
        silu_scaled_bf16(&ctx, pass, &down, &act, 0.25).unwrap();
        pass.level_barrier(&[&act]).unwrap();
        gemv_quant(&ctx, pass, up_w, &act, &up).unwrap();
        pass.level_barrier(&[&up]).unwrap();
        hc_mix_bf16(&ctx, pass, &up, &hn, &mixed, h, g).unwrap();
        pass.level_barrier(&[&mixed]).unwrap();
    });
}
