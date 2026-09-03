use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn random(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
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
            let gain: Vec<f32> = w[s * h..(s + 1) * h].iter().map(|v| 1.0 + v).collect();
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
    let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper, &[rows, g * h]).expect("hyper");
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
            assert_eq!(&got[(r * g + s) * h..(r * g + s + 1) * h], &x[r * h..(r + 1) * h]);
        }
    }
    let expected: Vec<f32> = x.iter().map(|v| cpu_ref::silu(v * 0.25)).collect();
    cpu_ref::assert_close(&act.to_f32().expect("act"), &expected, 1e-2, 1e-2);
}
