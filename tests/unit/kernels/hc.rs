use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::kernels::quant::gemv_quant;
use crate::kernels::skinny::gemm_skinny_q8_nt;
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
    // Short passes, many of them, and the minimum: display work and other
    // GPU clients interrupt some passes, and the clean ones show the kernel.
    let iters = 64;
    let passes = 16;
    type Read<'a> =
        dyn Fn(&ComputePass<'_>, &QuantWeights, &QuantWeights, &QuantWeights) + 'a;
    let time = |name: &str, f: &Read<'_>| {
        let mut best = f64::MAX;
        for p in 0..passes {
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..iters {
                let (d, u, j) = &weights[(p * iters + i) % sets];
                f(&pass, d, u, j);
            }
            let start = std::time::Instant::now();
            pass.commit_wait().expect("commit");
            best = best.min(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        eprintln!("{name}: {best:.1} us per read (best of {passes} passes of {iters})");
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

/// The small-batch fused kernels (`hc_read_down_q8_rows`,
/// `hc_read_up_mix_q8_rows`) against the single-row fused kernels run on each
/// row alone: the same mapping and per-lane operation order, so `down`,
/// `inj`, `inv_rms` and `mixed` are bit-identical row by row, for every
/// instantiated row count and stream count (including the inject-less final
/// mixer); `act` equals `silu_scaled_bf16(down)`. The residual fast-math
/// scheduling mismatch of the up kernel (one bf16 ulp in ~1e-5 of the
/// elements for MB >= 3) has its own bound in
/// `batched_fused_up_mix_matches_unfused_kernels`; these inputs do not hit it.
#[test]
fn batched_fused_read_is_bit_identical_to_decode_fused_kernels_per_row() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(27);
    let eps = 1e-6f32;
    let w_bias = 1.0f32;
    for (h, g, r, with_inject) in [
        (2560, 4, 320, true),
        (2560, 4, 320, false),
        (192, 3, 128, true),
        (64, 8, 64, true),
        (256, 1, 64, true),
    ] {
        let k = g * h;
        let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
        let (down_w, _) = random_q8(&ctx, &mut rng, r, k);
        let (up_w, _) = random_q8(&ctx, &mut rng, k, r);
        let inject = with_inject.then(|| random_q8(&ctx, &mut rng, g, k).0);
        let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
        let n_inj = inject.as_ref().map_or(0, |w| w.out_features());
        for m in 1..=HC_FUSED_MAX_ROWS {
            let hyper = cpu_ref::round_bf16(&random(&mut rng, m * k, -3.0, 3.0));
            let t_hyper =
                Tensor::from_f32_as_bf16(&ctx, &hyper, &[m, k]).expect("hyper");
            let zeros = |shape: &[usize], dtype| {
                Tensor::zeros(&ctx, shape, dtype).expect("scratch")
            };
            // Single-row kernels, one row at a time, into row views (a
            // separate tensor per row for `inj`: an odd G makes a bf16 row
            // view misaligned).
            let d_down = zeros(&[m, r], DType::BF16);
            let d_inj: Vec<Tensor> =
                (0..m).map(|_| zeros(&[n_inj.max(1)], DType::BF16)).collect();
            let d_inv_rms = zeros(&[m, g], DType::F32);
            let d_mixed = zeros(&[m, h], DType::BF16);
            let pass = ctx.begin().expect("pass");
            for (row, inj_row) in d_inj.iter().enumerate() {
                let hyper_row = t_hyper.view(row * k, &[k]).expect("row");
                let down_row = d_down.view(row * r, &[r]).expect("row");
                let inv_rms_row = d_inv_rms.view(row * g, &[g]).expect("row");
                let mixed_row = d_mixed.view(row * h, &[h]).expect("row");
                hc_read_down_q8(
                    &ctx,
                    &pass,
                    &hyper_row,
                    &t_norm,
                    &down_w,
                    inject.as_ref(),
                    &down_row,
                    inj_row,
                    &inv_rms_row,
                    h,
                    g,
                    eps,
                    w_bias,
                )
                .expect("decode down");
                hc_read_up_mix_q8(
                    &ctx,
                    &pass,
                    &up_w,
                    &down_row,
                    &hyper_row,
                    &t_norm,
                    &inv_rms_row,
                    &mixed_row,
                    h,
                    g,
                    w_bias,
                )
                .expect("decode up");
            }
            pass.commit_wait().expect("commit");

            // Small-batch kernels over all rows at once.
            let b_down = zeros(&[m, r], DType::BF16);
            let b_inj = zeros(&[m, n_inj.max(1)], DType::BF16);
            let b_inv_rms = zeros(&[m, g], DType::F32);
            let b_act = zeros(&[m, r], DType::BF16);
            let b_mixed = zeros(&[m, h], DType::BF16);
            let d_act = zeros(&[m, r], DType::BF16);
            let pass = ctx.begin().expect("pass");
            hc_read_down_q8_rows(
                &ctx,
                &pass,
                &t_hyper,
                &t_norm,
                &down_w,
                inject.as_ref(),
                &b_down,
                &b_inj,
                &b_inv_rms,
                &b_act,
                h,
                g,
                eps,
                w_bias,
            )
            .expect("batched down");
            hc_read_up_mix_q8_rows(
                &ctx, &pass, &up_w, &b_act, &t_hyper, &t_norm, &b_inv_rms, &b_mixed, h,
                g, w_bias,
            )
            .expect("batched up");
            silu_scaled_bf16(&ctx, &pass, &d_down, &d_act, 1.0 / g as f32)
                .expect("silu");
            pass.commit_wait().expect("commit");

            let same = |name: &str, a: &Tensor, b: &Tensor| {
                assert_eq!(
                    a.to_f32().expect(name),
                    b.to_f32().expect(name),
                    "{name}: h={h} g={g} r={r} inject={with_inject} m={m}"
                );
            };
            same("down", &b_down, &d_down);
            let d_inj_all: Vec<f32> =
                d_inj.iter().flat_map(|t| t.to_f32().expect("inj")).collect();
            assert_eq!(
                b_inj.to_f32().expect("inj"),
                d_inj_all,
                "inj: h={h} g={g} r={r} inject={with_inject} m={m}"
            );
            same("inv_rms", &b_inv_rms, &d_inv_rms);
            same("act", &b_act, &d_act);
            same("mixed", &b_mixed, &d_mixed);
            assert!(
                b_inv_rms
                    .to_f32()
                    .expect("inv_rms")
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0),
                "inv_rms written for every row"
            );
        }
    }
}

/// `hc_read_up_mix_q8_rows` against the unfused `gemv_quant` (per row) and
/// `hc_mix_bf16` on the same `silu_scaled_bf16` activation and `inv_rms`.
/// The single-row kernel is bit-identical to that chain; the row-templated
/// bodies compute the same f32 expressions, but the compiler's fast-math
/// scheduling of the unrolled MB = 3 and 4 bodies differs at f32 rounding
/// level in a way the bf16 output rounding exposes as a single-ulp flip in
/// about 1e-5 of the elements (a 24-seed sweep at the model's shape found 5
/// in 184K for MB = 3 and 1 in 246K for MB = 4, none for MB = 1 and 2). This
/// asserts exactly that: every element within one bf16 ulp, at most 5e-4 of
/// them differing, none for MB = 1 and 2.
#[test]
fn batched_fused_up_mix_matches_unfused_kernels() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(28);
    let w_bias = 1.0f32;
    for (h, g, r) in [(2560, 4, 320), (192, 3, 128), (64, 8, 64), (256, 1, 64)] {
        let k = g * h;
        let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
        let (up_w, _) = random_q8(&ctx, &mut rng, k, r);
        let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
        for m in 1..=HC_FUSED_MAX_ROWS {
            let hyper = cpu_ref::round_bf16(&random(&mut rng, m * k, -3.0, 3.0));
            let down = cpu_ref::round_bf16(&random(&mut rng, m * r, -6.0, 6.0));
            let inv_rms: Vec<f32> = random(&mut rng, m * g, 0.3, 1.2);
            let hn: Vec<f32> = (0..m)
                .flat_map(|row| {
                    hn_bf16(
                        &hyper[row * k..(row + 1) * k],
                        &norm_w,
                        &inv_rms[row * g..(row + 1) * g],
                        h,
                        w_bias,
                    )
                })
                .collect();

            let t_hyper =
                Tensor::from_f32_as_bf16(&ctx, &hyper, &[m, k]).expect("hyper");
            let t_down = Tensor::from_f32_as_bf16(&ctx, &down, &[m, r]).expect("down");
            let t_inv_rms = Tensor::from_f32(&ctx, &inv_rms, &[m, g]).expect("inv_rms");
            let t_hn = Tensor::from_f32_as_bf16(&ctx, &hn, &[m, k]).expect("hn");
            let zeros = |shape: &[usize], dtype| {
                Tensor::zeros(&ctx, shape, dtype).expect("scratch")
            };
            let act = zeros(&[m, r], DType::BF16);
            let up = zeros(&[m, k], DType::BF16);
            let mixed = zeros(&[m, h], DType::BF16);
            let f_mixed = zeros(&[m, h], DType::BF16);

            let pass = ctx.begin().expect("pass");
            silu_scaled_bf16(&ctx, &pass, &t_down, &act, 1.0 / g as f32).expect("silu");
            for row in 0..m {
                let act_row = act.view(row * r, &[r]).expect("row");
                let up_row = up.view(row * k, &[k]).expect("row");
                gemv_quant(&ctx, &pass, &up_w, &act_row, &up_row).expect("up");
            }
            hc_mix_bf16(&ctx, &pass, &up, &t_hn, &mixed, h, g).expect("mix");
            hc_read_up_mix_q8_rows(
                &ctx, &pass, &up_w, &act, &t_hyper, &t_norm, &t_inv_rms, &f_mixed, h,
                g, w_bias,
            )
            .expect("fused up");
            pass.commit_wait().expect("commit");
            let (got, want) =
                (f_mixed.to_f32().expect("mixed"), mixed.to_f32().expect("mixed"));
            let mismatches: Vec<(usize, f32, f32)> = got
                .iter()
                .zip(&want)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect();
            for (i, a, b) in &mismatches {
                // One bf16 ulp of the larger magnitude.
                let ulp = bf16::from_f32(a.abs().max(b.abs())).to_f32() * 2f32.powi(-7);
                assert!(
                    (a - b).abs() <= ulp,
                    "h={h} g={g} r={r} m={m}: element {i} {a} vs {b} beyond one bf16 ulp"
                );
            }
            assert!(
                mismatches.len() as f64 <= 5e-4 * got.len() as f64,
                "h={h} g={g} r={r} m={m}: {} of {} elements differ",
                mismatches.len(),
                got.len()
            );
            if m <= 2 {
                assert!(
                    mismatches.is_empty(),
                    "h={h} g={g} r={r} m={m}: {mismatches:?}"
                );
            }
        }
    }
}

/// The small-batch fused read against the six-kernel skinny chain it
/// replaces in `hc_read_batched` (grouped norm, `gemm_skinny_q8_nt` for down,
/// inject and up, scaled SiLU, mix), at the model's shape with and without
/// the inject weight. Not bit-identical: the skinny GEMM rounds each
/// dequantized weight to bf16 and consumes a bf16 `hn`, the fused kernels
/// dot f32-dequantized weights against an f32 `hn` (down half) and a bf16
/// `hn` (up half). Bounds as in `fused_read_gate_matches_unfused_kernels`
/// (the mixed bound is coarse: these random logits are O(100) and the gates
/// amplify the bf16-level logit differences).
#[test]
fn batched_fused_read_matches_skinny_chain() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(29);
    let eps = 1e-6f32;
    let w_bias = 1.0f32;
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    for with_inject in [true, false] {
        let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
        let (down_w, _) = random_q8(&ctx, &mut rng, r, k);
        let (up_w, _) = random_q8(&ctx, &mut rng, k, r);
        let inject = with_inject.then(|| random_q8(&ctx, &mut rng, g, k).0);
        let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
        for m in 1..=HC_FUSED_MAX_ROWS {
            let hyper = cpu_ref::round_bf16(&random(&mut rng, m * k, -3.0, 3.0));
            let t_hyper =
                Tensor::from_f32_as_bf16(&ctx, &hyper, &[m, k]).expect("hyper");
            let zeros = |shape: &[usize], dtype| {
                Tensor::zeros(&ctx, shape, dtype).expect("scratch")
            };
            let hn = zeros(&[m, k], DType::BF16);
            let down = zeros(&[m, r], DType::BF16);
            let inj = zeros(&[m, g], DType::BF16);
            let act = zeros(&[m, r], DType::BF16);
            let up = zeros(&[m, k], DType::BF16);
            let mixed = zeros(&[m, h], DType::BF16);
            let pass = ctx.begin().expect("pass");
            rmsnorm_grouped_bf16(
                &ctx, &pass, &t_hyper, &t_norm, &hn, h, g, eps, w_bias,
            )
            .expect("norm");
            gemm_skinny_q8_nt(&ctx, &pass, &hn, &down_w, &down).expect("down");
            if let Some(inj_w) = &inject {
                gemm_skinny_q8_nt(&ctx, &pass, &hn, inj_w, &inj).expect("inject");
            }
            silu_scaled_bf16(&ctx, &pass, &down, &act, 1.0 / g as f32).expect("silu");
            gemm_skinny_q8_nt(&ctx, &pass, &act, &up_w, &up).expect("up");
            hc_mix_bf16(&ctx, &pass, &up, &hn, &mixed, h, g).expect("mix");
            pass.commit_wait().expect("commit");

            let f_down = zeros(&[m, r], DType::BF16);
            let f_inj = zeros(&[m, g], DType::BF16);
            let f_inv_rms = zeros(&[m, g], DType::F32);
            let f_act = zeros(&[m, r], DType::BF16);
            let f_mixed = zeros(&[m, h], DType::BF16);
            let pass = ctx.begin().expect("pass");
            hc_read_down_q8_rows(
                &ctx,
                &pass,
                &t_hyper,
                &t_norm,
                &down_w,
                inject.as_ref(),
                &f_down,
                &f_inj,
                &f_inv_rms,
                &f_act,
                h,
                g,
                eps,
                w_bias,
            )
            .expect("fused down");
            hc_read_up_mix_q8_rows(
                &ctx, &pass, &up_w, &f_act, &t_hyper, &t_norm, &f_inv_rms, &f_mixed, h,
                g, w_bias,
            )
            .expect("fused up");
            pass.commit_wait().expect("commit");

            let close = |name: &str, got: &Tensor, want: &Tensor, frac: f32| {
                let want = want.to_f32().expect("reference");
                let got = got.to_f32().expect("fused");
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "skinny chain vs fused, m={m} inject={with_inject} {name}: max |diff| {worst:.4} (rms {:.3})",
                    rms(&want)
                );
                cpu_ref::assert_close(&got, &want, frac * rms(&want), frac);
            };
            close("down", &f_down, &down, 1e-2);
            if with_inject {
                close("inj", &f_inj, &inj, 1e-2);
            } else {
                assert!(
                    f_inj.to_f32().expect("inj").iter().all(|v| *v == 0.0),
                    "inj untouched"
                );
            }
            close("mixed", &f_mixed, &mixed, 2e-1);
        }
    }
}

/// Mismatch-rate sweep behind the bound in
/// `batched_fused_up_mix_matches_unfused_kernels`: the row-templated kernels
/// against the single-row fused kernels over 24 seeds at the model's shape,
/// counting elements that differ, per row count. Prints the counts; run with
/// `cargo test --release -- --ignored --nocapture batched_fused_mismatch_rate_sweep`.
/// Last run: up 0/61K, 0/123K, 5/184K, 2/246K for MB = 1..4 (single bf16
/// ulps); down 0 throughout.
#[test]
#[ignore = "diagnostic sweep"]
fn batched_fused_mismatch_rate_sweep() {
    let ctx = MetalContext::new().expect("metal context");
    let w_bias = 1.0f32;
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    for m in 1..=HC_FUSED_MAX_ROWS {
        let mut mism_up = 0usize;
        let mut mism_down = 0usize;
        let mut total = 0usize;
        for seed in 0..24u64 {
            let mut rng = StdRng::seed_from_u64(1000 + seed);
            let norm_w = cpu_ref::round_bf16(&random(&mut rng, k, -0.5, 0.5));
            let (up_w, _) = random_q8(&ctx, &mut rng, k, r);
            let (down_w, _) = random_q8(&ctx, &mut rng, r, k);
            let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
            let hyper = cpu_ref::round_bf16(&random(&mut rng, m * k, -3.0, 3.0));
            let down = cpu_ref::round_bf16(&random(&mut rng, m * r, -6.0, 6.0));
            let inv_rms: Vec<f32> = random(&mut rng, m * g, 0.3, 1.2);
            let t_hyper =
                Tensor::from_f32_as_bf16(&ctx, &hyper, &[m, k]).expect("hyper");
            let t_down = Tensor::from_f32_as_bf16(&ctx, &down, &[m, r]).expect("down");
            let t_inv_rms = Tensor::from_f32(&ctx, &inv_rms, &[m, g]).expect("inv_rms");
            let zeros = |shape: &[usize], dtype| {
                Tensor::zeros(&ctx, shape, dtype).expect("scratch")
            };
            let a = zeros(&[m, h], DType::BF16);
            let b = zeros(&[m, h], DType::BF16);
            let da = zeros(&[m, r], DType::BF16);
            let db = zeros(&[m, r], DType::BF16);
            let ia = zeros(&[m, g], DType::BF16);
            let ib = zeros(&[m, g], DType::BF16);
            let ra = zeros(&[m, g], DType::F32);
            let rb = zeros(&[m, g], DType::F32);
            let act = zeros(&[m, r], DType::BF16);
            let actb = zeros(&[m, r], DType::BF16);
            let pass = ctx.begin().expect("pass");
            silu_scaled_bf16(&ctx, &pass, &t_down, &act, 1.0 / g as f32).unwrap();
            for row in 0..m {
                hc_read_up_mix_q8(
                    &ctx,
                    &pass,
                    &up_w,
                    &t_down.view(row * r, &[r]).unwrap(),
                    &t_hyper.view(row * k, &[k]).unwrap(),
                    &t_norm,
                    &t_inv_rms.view(row * g, &[g]).unwrap(),
                    &a.view(row * h, &[h]).unwrap(),
                    h,
                    g,
                    w_bias,
                )
                .unwrap();
                hc_read_down_q8(
                    &ctx,
                    &pass,
                    &t_hyper.view(row * k, &[k]).unwrap(),
                    &t_norm,
                    &down_w,
                    None,
                    &da.view(row * r, &[r]).unwrap(),
                    &ia.view(row * g, &[g]).unwrap(),
                    &ra.view(row * g, &[g]).unwrap(),
                    h,
                    g,
                    1e-6,
                    w_bias,
                )
                .unwrap();
            }
            hc_read_up_mix_q8_rows(
                &ctx, &pass, &up_w, &act, &t_hyper, &t_norm, &t_inv_rms, &b, h, g,
                w_bias,
            )
            .unwrap();
            hc_read_down_q8_rows(
                &ctx, &pass, &t_hyper, &t_norm, &down_w, None, &db, &ib, &rb, &actb, h,
                g, 1e-6, w_bias,
            )
            .unwrap();
            pass.commit_wait().expect("commit");
            let (a, b) = (a.to_f32().unwrap(), b.to_f32().unwrap());
            mism_up += a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let (da, db) = (da.to_f32().unwrap(), db.to_f32().unwrap());
            mism_down += da.iter().zip(&db).filter(|(x, y)| x != y).count();
            total += a.len();
        }
        eprintln!(
            "MB={m}: up mismatches {mism_up} / {total}; down mismatches {mism_down} / {}",
            total * r / h
        );
    }
}

/// Kernel-level timing of the small-batch read gate at the model's shape for
/// m = 1, 3, 4 rows (the draft head's chain rows, a 2-draft verify pass, the
/// largest verify pass): the six-kernel skinny chain `hc_read_batched` used,
/// the two row-templated fused kernels, and `m` concurrent single-row fused
/// dispatches per half (the alternative mapping, a grid over rows). Same
/// setup as `fused_read_gate_timing`: concurrent pass, level barrier after
/// every stage, DRAM-streamed weights. Run with
/// `cargo test --release -- --ignored --nocapture fused_batched_read_gate_timing`.
#[test]
#[ignore = "timing only"]
fn fused_batched_read_gate_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(30);
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    let sets = 64; // 64 * 6.6 MB = 420 MB of weights, well past the SLC
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
    let t_norm = Tensor::from_f32_as_bf16(&ctx, &norm_w, &[k]).expect("norm");
    let iters = 256;
    for m in [1usize, 3, 4] {
        let hyper = cpu_ref::round_bf16(&random(&mut rng, m * k, -3.0, 3.0));
        let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper, &[m, k]).expect("hyper");
        let zeros = |shape: &[usize], dtype| {
            Tensor::zeros(&ctx, shape, dtype).expect("scratch")
        };
        let hn = zeros(&[m, k], DType::BF16);
        let down = zeros(&[m, r], DType::BF16);
        let inj = zeros(&[m, g], DType::BF16);
        let act = zeros(&[m, r], DType::BF16);
        let up = zeros(&[m, k], DType::BF16);
        let mixed = zeros(&[m, h], DType::BF16);
        let inv_rms = zeros(&[m, g], DType::F32);
        let f_act = zeros(&[m, r], DType::BF16);
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
            eprintln!("m={m} {name}: {best:.1} us per read (best of 3)");
        };
        time("unfused skinny chain (6 dispatches)", &|pass, down_w, up_w, inj_w| {
            rmsnorm_grouped_bf16(&ctx, pass, &t_hyper, &t_norm, &hn, h, g, 1e-6, 1.0)
                .unwrap();
            pass.level_barrier(&[&hn]).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &hn, down_w, &down).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &hn, inj_w, &inj).unwrap();
            pass.level_barrier(&[&down, &inj]).unwrap();
            silu_scaled_bf16(&ctx, pass, &down, &act, 0.25).unwrap();
            pass.level_barrier(&[&act]).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &act, up_w, &up).unwrap();
            pass.level_barrier(&[&up]).unwrap();
            hc_mix_bf16(&ctx, pass, &up, &hn, &mixed, h, g).unwrap();
            pass.level_barrier(&[&mixed]).unwrap();
        });
        time("fused rows (2 dispatches)", &|pass, down_w, up_w, inj_w| {
            hc_read_down_q8_rows(
                &ctx,
                pass,
                &t_hyper,
                &t_norm,
                down_w,
                Some(inj_w),
                &down,
                &inj,
                &inv_rms,
                &f_act,
                h,
                g,
                1e-6,
                1.0,
            )
            .unwrap();
            pass.level_barrier(&[&down, &inj, &inv_rms, &f_act]).unwrap();
            hc_read_up_mix_q8_rows(
                &ctx, pass, up_w, &f_act, &t_hyper, &t_norm, &inv_rms, &mixed, h, g,
                1.0,
            )
            .unwrap();
            pass.level_barrier(&[&mixed]).unwrap();
        });
        time("fused rows, down only", &|pass, down_w, _, inj_w| {
            hc_read_down_q8_rows(
                &ctx,
                pass,
                &t_hyper,
                &t_norm,
                down_w,
                Some(inj_w),
                &down,
                &inj,
                &inv_rms,
                &f_act,
                h,
                g,
                1e-6,
                1.0,
            )
            .unwrap();
            pass.level_barrier(&[&down, &inj, &inv_rms, &f_act]).unwrap();
        });
        time("fused rows, up+mix only", &|pass, _, up_w, _| {
            hc_read_up_mix_q8_rows(
                &ctx, pass, up_w, &f_act, &t_hyper, &t_norm, &inv_rms, &mixed, h, g,
                1.0,
            )
            .unwrap();
            pass.level_barrier(&[&mixed]).unwrap();
        });
        time("m x single-row fused (2m dispatches)", &|pass, down_w, up_w, inj_w| {
            for row in 0..m {
                hc_read_down_q8(
                    &ctx,
                    pass,
                    &t_hyper.view(row * k, &[k]).unwrap(),
                    &t_norm,
                    down_w,
                    Some(inj_w),
                    &down.view(row * r, &[r]).unwrap(),
                    &inj.view(row * g, &[g]).unwrap(),
                    &inv_rms.view(row * g, &[g]).unwrap(),
                    h,
                    g,
                    1e-6,
                    1.0,
                )
                .unwrap();
            }
            pass.level_barrier(&[&down, &inj, &inv_rms]).unwrap();
            for row in 0..m {
                hc_read_up_mix_q8(
                    &ctx,
                    pass,
                    up_w,
                    &down.view(row * r, &[r]).unwrap(),
                    &t_hyper.view(row * k, &[k]).unwrap(),
                    &t_norm,
                    &inv_rms.view(row * g, &[g]).unwrap(),
                    &mixed.view(row * h, &[h]).unwrap(),
                    h,
                    g,
                    1.0,
                )
                .unwrap();
            }
            pass.level_barrier(&[&mixed]).unwrap();
        });
    }
}

/// Per-dispatch GPU timestamps of the fused read kernels on the profile
/// transport (one command buffer per dispatch), reported as minimum and
/// percentiles over the samples, so GPU contention (display work, other
/// clients) that hits some samples does not decide a comparison. To A/B a
/// variant, add it to `names` under its own kernel name: every entry
/// streams its own weight set and the order rotates.
#[test]
#[ignore = "timing only"]
fn fused_read_kernels_dispatch_timing() {
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    let mut rng = StdRng::seed_from_u64(25);
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    let sets = 64;
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
    let down = zeros(&[r], DType::BF16);
    let inj = zeros(&[g], DType::BF16);
    let mixed = zeros(&[h], DType::BF16);
    let inv_rms = zeros(&[g], DType::F32);
    // (kernel name, is a down kernel)
    let names = [("hc_read_down_q8", true), ("hc_read_up_mix_q8_g4", false)];
    let iters = 512;
    crate::metal::profile::take();
    let pass = ctx.begin().expect("pass");
    for i in 0..iters {
        for slot in 0..names.len() {
            let which = (slot + i) % names.len();
            let (name, is_down) = names[which];
            let (d, u, j) = &weights[(names.len() * i + which) % sets];
            if is_down {
                let dims = check_read_down(
                    &t_hyper,
                    &t_norm,
                    d,
                    Some(j),
                    &down,
                    &inj,
                    &inv_rms,
                    h,
                    g,
                    1,
                )
                .unwrap();
                dispatch_read_down(
                    &ctx,
                    &pass,
                    name,
                    &t_hyper,
                    &t_norm,
                    d,
                    Some(j),
                    &down,
                    &inj,
                    &inv_rms,
                    None,
                    dims,
                    h,
                    g,
                    1e-6,
                    1.0,
                )
                .unwrap();
            } else {
                dispatch_read_up_mix(
                    &ctx, &pass, name, u, &down, &t_hyper, &t_norm, &inv_rms, &mixed,
                    h, g, 1, true, 1.0,
                )
                .unwrap();
            }
        }
    }
    pass.commit_wait().expect("commit");
    let passes = crate::metal::profile::take();
    for (name, _) in names {
        let mut us: Vec<f64> = passes
            .iter()
            .flat_map(|p| p.kernels.iter())
            .filter(|s| s.name == name)
            .map(|s| s.gpu_secs * 1e6)
            .collect();
        us.sort_by(|a, b| a.total_cmp(b));
        let n = us.len();
        eprintln!(
            "{name}: min {:.1} us, p10 {:.1}, median {:.1}, p90 {:.1} over {n} dispatches",
            us[0],
            us[n / 10],
            us[n / 2],
            us[n * 9 / 10]
        );
    }
}

/// The decode read gate as decode runs it: a chain of (down, barrier, up,
/// barrier) over 64 distinct weight sets in one concurrent pass, plus the
/// down-only and up-only chains, for every `down:up` kernel pair in
/// `LILY_HC_KERNELS` (comma separated; default the shipped pair). Prints
/// microseconds per level, best of several passes, order rotated. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn hc_read_chain_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(25);
    let (h, g, r) = (2560usize, 4usize, 320usize);
    let k = g * h;
    let sets = 64;
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
    let down = zeros(&[r], DType::BF16);
    let inj = zeros(&[g], DType::BF16);
    let mixed = zeros(&[h], DType::BF16);
    let inv_rms = zeros(&[g], DType::F32);
    let pairs: Vec<(&'static str, &'static str)> = std::env::var("LILY_HC_KERNELS")
        .map(|v| {
            v.split(',')
                .map(|pair| {
                    let (d, u) = pair.split_once(':').expect("down:up");
                    (
                        &*Box::leak(d.to_string().into_boxed_str()),
                        &*Box::leak(u.to_string().into_boxed_str()),
                    )
                })
                .collect()
        })
        .unwrap_or_else(|_| vec![("hc_read_down_q8", "hc_read_up_mix_q8_g4")]);
    // (label, run down, run up)
    let modes = [("pair", true, true), ("down", true, false), ("up", false, true)];
    let mut best = vec![[f64::INFINITY; 3]; pairs.len()];
    for round in 0..6 {
        for k_pair in 0..pairs.len() {
            let which = (round + k_pair) % pairs.len();
            let (dname, uname) = pairs[which];
            for (m, &(_, run_down, run_up)) in modes.iter().enumerate() {
                let pass = ctx.begin_concurrent().expect("pass");
                for (d, u, j) in &weights {
                    if run_down {
                        let dims = check_read_down(
                            &t_hyper, &t_norm, d, Some(j), &down, &inj, &inv_rms, h, g, 1,
                        )
                        .unwrap();
                        dispatch_read_down(
                            &ctx, &pass, dname, &t_hyper, &t_norm, d, Some(j), &down, &inj,
                            &inv_rms, None, dims, h, g, 1e-6, 1.0,
                        )
                        .unwrap();
                        pass.level_barrier(&[&down]).unwrap();
                    }
                    if run_up {
                        dispatch_read_up_mix(
                            &ctx, &pass, uname, u, &down, &t_hyper, &t_norm, &inv_rms, &mixed,
                            h, g, 1, true, 1.0,
                        )
                        .unwrap();
                        pass.level_barrier(&[&mixed]).unwrap();
                    }
                }
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                if round > 0 {
                    best[which][m] =
                        best[which][m].min((t.gpu_end_secs - t.gpu_start_secs) / sets as f64);
                }
            }
        }
    }
    let mb = |bytes: usize| bytes as f64 / 1e6;
    let down_bytes = mb(r * k + g * k + 2 * (r + g) * k / 64 * 2);
    let up_bytes = mb(k * r + 2 * k * r / 64 * 2);
    for ((dname, uname), b) in pairs.iter().zip(&best) {
        eprintln!(
            "{dname} + {uname}: pair {:.2} us per layer | down alone {:.2} us ({:.0} GB/s) | up alone {:.2} us ({:.0} GB/s)",
            b[0] * 1e6,
            b[1] * 1e6,
            down_bytes / b[1] / 1e3,
            b[2] * 1e6,
            up_bytes / b[2] / 1e3,
        );
    }
}
