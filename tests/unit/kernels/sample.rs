use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;

/// Reference sampler: same semantics as the kernel, in plain f32 on the host.
/// Returns the chosen id and the kept candidate ids (sorted by probability).
fn cpu_sample(
    logits: &[f32],
    counts: &[u32],
    p: &SamplingParams,
    step: u32,
) -> (u32, Vec<u32>) {
    let (temperature, k) = p.effective();
    let adjusted: Vec<f32> = logits
        .iter()
        .zip(counts)
        .map(|(&l, &c)| {
            let mut l = l;
            if p.uses_penalties() && c > 0 {
                l = if l > 0.0 { l / p.repetition_penalty } else { l * p.repetition_penalty };
                l -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
            l / temperature
        })
        .collect();
    let mut order: Vec<usize> = (0..adjusted.len()).collect();
    // Descending by value, ascending by id on ties (the kernel takes the
    // lowest ids among tied threshold keys).
    order.sort_by(|&a, &b| adjusted[b].total_cmp(&adjusted[a]).then(a.cmp(&b)));
    let cands = &order[..k.min(order.len())];
    let top = adjusted[cands[0]];
    let probs: Vec<f32> = cands.iter().map(|&i| (adjusted[i] - top).exp()).collect();
    let total: f32 = probs.iter().sum();
    let mut kept = Vec::new();
    let mut cum = 0.0f32;
    for (rank, (&id, &prob)) in cands.iter().zip(&probs).enumerate() {
        if rank > 0 && !(cum < p.top_p * total && prob >= p.min_p) {
            break;
        }
        cum += prob;
        kept.push((id as u32, cum));
    }
    let target = uniform_for(p.seed, step) * kept.last().unwrap().1;
    let chosen = kept
        .iter()
        .find(|(_, c)| target < *c)
        .map_or(kept.last().unwrap().0, |(id, _)| *id);
    (chosen, kept.into_iter().map(|(id, _)| id).collect())
}

fn run_kernel(
    ctx: &MetalContext,
    logits: &[f32],
    counts: &[u32],
    p: &SamplingParams,
    step: usize,
) -> u32 {
    let v = logits.len();
    let t_logits = Tensor::from_f32(ctx, logits, &[v]).expect("logits");
    let scratch = SamplerScratch::new(ctx, v).expect("scratch");
    scratch.counts.write_bytes(bytemuck::cast_slice(counts)).expect("counts");
    let out = Tensor::zeros(ctx, &[1], DType::U32).expect("out");
    let pass = ctx.begin().expect("pass");
    sample_f32(ctx, &pass, &t_logits, &scratch, p, step, &out).expect("sample");
    pass.commit_wait().expect("run");
    out.to_u32().expect("read")[0]
}

#[test]
fn greedy_matches_argmax_with_lowest_id_on_ties() {
    let ctx = MetalContext::new().expect("metal context");
    let mut logits = vec![0.0f32; 5000];
    logits[1234] = 7.5;
    logits[4321] = 7.5;
    let counts = vec![0u32; logits.len()];
    let got = run_kernel(&ctx, &logits, &counts, &SamplingParams::greedy(), 0);
    assert_eq!(got, 1234);
}

#[test]
fn draws_match_the_cpu_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(7);
    let v = 6007;
    let logits: Vec<f32> = (0..v).map(|_| rng.gen_range(-8.0..8.0)).collect();
    let counts = vec![0u32; v];
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.02,
        seed: 0xDEAD_BEEF,
        ..SamplingParams::greedy()
    };
    let mut agree = 0;
    let steps = 64;
    for step in 0..steps {
        let got = run_kernel(&ctx, &logits, &counts, &params, step);
        let (want, kept) = cpu_sample(&logits, &counts, &params, step as u32);
        assert!(kept.contains(&got), "step {step}: {got} outside the kept set {kept:?}");
        if got == want {
            agree += 1;
        }
    }
    // Prefix sums accumulate in a different order on the GPU, so a draw that
    // lands within float noise of a boundary may pick the neighbour.
    assert!(agree >= steps - 2, "only {agree}/{steps} draws matched the reference");
}

#[test]
fn full_vocabulary_top_p_and_seed_replay() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    // Smaller than the candidate cap: every id is a candidate.
    let v = 700;
    let logits: Vec<f32> = (0..v).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let counts = vec![0u32; v];
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.5,
        seed: 99,
        ..SamplingParams::greedy()
    };
    let a = run_kernel(&ctx, &logits, &counts, &params, 3);
    let b = run_kernel(&ctx, &logits, &counts, &params, 3);
    assert_eq!(a, b, "same seed and step must replay");
    let (want, kept) = cpu_sample(&logits, &counts, &params, 3);
    assert!(kept.contains(&a));
    assert!(kept.len() < v, "top_p must truncate");
    let _ = want;
}

#[test]
fn penalties_suppress_repeated_ids_and_update_counts() {
    let ctx = MetalContext::new().expect("metal context");
    let v = 3000;
    let mut logits = vec![0.0f32; v];
    logits[10] = 5.0;
    logits[20] = 4.0;
    let mut counts = vec![0u32; v];
    counts[10] = 3;
    let params = SamplingParams {
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        repetition_penalty: 1.5,
        ..SamplingParams::greedy()
    };
    // 5.0 / 1.5 - 0.5 - 1.5 = 1.33 < 4.0: the repeated id loses.
    let t_logits = Tensor::from_f32(&ctx, &logits, &[v]).expect("logits");
    let scratch = SamplerScratch::new(&ctx, v).expect("scratch");
    scratch.counts.write_bytes(bytemuck::cast_slice(&counts)).expect("counts");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");
    let pass = ctx.begin().expect("pass");
    sample_f32(&ctx, &pass, &t_logits, &scratch, &params, 0, &out).expect("sample");
    pass.commit_wait().expect("run");
    assert_eq!(out.to_u32().expect("read")[0], 20);
    let after = scratch.counts.to_u32().expect("counts");
    assert_eq!(after[20], 1, "the chosen id is counted");
    assert_eq!(after[10], 3);
    let adjusted = scratch.adjusted.to_f32().expect("adjusted");
    assert!((adjusted[10] - (5.0 / 1.5 - 0.5 - 1.5)).abs() < 1e-5);
    assert_eq!(adjusted[20], 4.0);
}

/// Kernel time per draw at the real vocabulary size; run with
/// `cargo test --release --lib sampler_timing -- --ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn sampler_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(3);
    let v = 248_320;
    let logits: Vec<f32> = (0..v).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let t_logits = Tensor::from_f32(&ctx, &logits, &[v]).expect("logits");
    let scratch = SamplerScratch::new(&ctx, v).expect("scratch");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");
    for (label, params) in [
        ("greedy", SamplingParams::greedy()),
        ("top_k 20", SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, ..SamplingParams::greedy() }),
        ("top_k 0 (cap)", SamplingParams { temperature: 1.0, top_k: 0, top_p: 0.95, ..SamplingParams::greedy() }),
    ] {
        // Warm the pipeline, then time 50 back-to-back dispatches.
        let pass = ctx.begin().expect("pass");
        sample_f32(&ctx, &pass, &t_logits, &scratch, &params, 0, &out).expect("sample");
        pass.commit_wait().expect("run");
        let n = 50;
        let pass = ctx.begin().expect("pass");
        for step in 0..n {
            sample_f32(&ctx, &pass, &t_logits, &scratch, &params, step, &out).expect("sample");
        }
        let started = std::time::Instant::now();
        pass.commit_wait().expect("run");
        let per = started.elapsed().as_secs_f64() / n as f64;
        eprintln!("{label}: {:.1} us per draw", per * 1e6);
    }
}
