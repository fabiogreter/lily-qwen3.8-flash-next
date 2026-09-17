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
                l = if l > 0.0 {
                    l / p.repetition_penalty
                } else {
                    l * p.repetition_penalty
                };
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
        assert!(
            kept.contains(&got),
            "step {step}: {got} outside the kept set {kept:?}"
        );
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
        (
            "top_k 20",
            SamplingParams {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95,
                ..SamplingParams::greedy()
            },
        ),
        (
            "top_k 0 (cap)",
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.95,
                ..SamplingParams::greedy()
            },
        ),
    ] {
        // Warm the pipeline, then time 50 back-to-back dispatches.
        let pass = ctx.begin().expect("pass");
        sample_f32(&ctx, &pass, &t_logits, &scratch, &params, 0, &out).expect("sample");
        pass.commit_wait().expect("run");
        let n = 50;
        let pass = ctx.begin().expect("pass");
        for step in 0..n {
            sample_f32(&ctx, &pass, &t_logits, &scratch, &params, step, &out)
                .expect("sample");
        }
        let started = std::time::Instant::now();
        pass.commit_wait().expect("run");
        let per = started.elapsed().as_secs_f64() / n as f64;
        eprintln!("{label}: {:.1} us per draw", per * 1e6);
    }
}

/// The kept candidates of the CPU reference with their probabilities
/// normalised over the kept mass, in the kernel's (probability, id) order.
fn cpu_kept(logits: &[f32], p: &SamplingParams) -> Vec<(u32, f32)> {
    let counts = vec![0u32; logits.len()];
    let (_, kept) = cpu_sample(logits, &counts, p, 0);
    let (temperature, _) = p.effective();
    let top = logits[kept[0] as usize] / temperature;
    let w: Vec<f32> =
        kept.iter().map(|&i| (logits[i as usize] / temperature - top).exp()).collect();
    let mass: f32 = w.iter().sum();
    kept.iter().zip(&w).map(|(&id, &x)| (id, x / mass)).collect()
}

/// Speculative sampling on the host: accept `d` on the first uniform with
/// probability p(d) / q(d), otherwise draw from the residual on the second.
fn cpu_spec(p: &[(u32, f32)], q: &[(u32, f32)], d: u32, seed: u64, step: u32) -> u32 {
    let at = |dist: &[(u32, f32)], id: u32| {
        dist.iter().find(|(i, _)| *i == id).map_or(0.0, |(_, x)| *x)
    };
    let (p_d, q_d) = (at(p, d), at(q, d));
    if uniform_for(seed, step) * q_d < p_d {
        return d;
    }
    let r: Vec<f32> = p.iter().map(|(id, x)| (x - at(q, *id)).max(0.0)).collect();
    let total: f32 = r.iter().sum();
    let target = uniform_for(seed, step | 0x8000_0000) * total;
    let mut cum = 0.0;
    for ((id, _), x) in p.iter().zip(&r) {
        cum += x;
        if target < cum {
            return *id;
        }
    }
    p.last().unwrap().0
}

fn draft_and_spec(
    ctx: &MetalContext,
    trunk: &[f32],
    head: &[f32],
    params: &SamplingParams,
    step: usize,
) -> (u32, u32, Vec<(u32, f32)>) {
    let v = trunk.len();
    let t_trunk = Tensor::from_f32(ctx, trunk, &[v]).expect("trunk");
    let t_head = Tensor::from_f32(ctx, head, &[v]).expect("head");
    let scratch = SamplerScratch::new(ctx, v).expect("scratch");
    let dists = DraftDists::new(ctx, 2).expect("dists");
    let dist = dists.slot(1).expect("slot");
    let drafted = Tensor::zeros(ctx, &[1], DType::U32).expect("drafted");
    let out = Tensor::zeros(ctx, &[1], DType::U32).expect("out");
    let pass = ctx.begin().expect("pass");
    let dp = draft_params(params);
    sample_draft_f32(
        ctx,
        &pass,
        &t_head,
        &scratch,
        &dp,
        draft_step(step),
        &drafted,
        &dist,
    )
    .expect("draft");
    sample_spec_f32(
        ctx, &pass, &t_trunk, &scratch, params, step, &dist, &drafted, &out,
    )
    .expect("spec");
    pass.commit_wait().expect("run");
    let n = dist.count.to_u32().expect("n")[0] as usize;
    let ids = dist.ids.to_u32().expect("ids");
    let probs = dist.probs.to_f32().expect("probs");
    let exported = ids[..n].iter().copied().zip(probs[..n].iter().copied()).collect();
    (drafted.to_u32().expect("d")[0], out.to_u32().expect("out")[0], exported)
}

#[test]
fn draft_draw_exports_the_kept_distribution() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(21);
    let v = 5003;
    let head: Vec<f32> = (0..v).map(|_| rng.gen_range(-8.0..8.0)).collect();
    let trunk: Vec<f32> = head.iter().map(|x| x + rng.gen_range(-1.0..1.0)).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        seed: 5,
        ..SamplingParams::greedy()
    };
    let (d, _, exported) = draft_and_spec(&ctx, &trunk, &head, &params, 4);
    let want = cpu_kept(&head, &draft_params(&params));
    assert_eq!(exported.len(), want.len());
    for ((id, q), (wid, wq)) in exported.iter().zip(&want) {
        assert_eq!(id, wid);
        assert!((q - wq).abs() < 1e-5, "q({id}) = {q}, reference {wq}");
    }
    assert!(
        exported.iter().any(|(id, _)| *id == d),
        "the proposal is drawn from the export"
    );
}

#[test]
fn speculative_draw_matches_the_cpu_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(23);
    let v = 4099;
    let head: Vec<f32> = (0..v).map(|_| rng.gen_range(-6.0..6.0)).collect();
    // A trunk that disagrees with the head often enough to exercise both
    // acceptance and the residual draw.
    let trunk: Vec<f32> = head.iter().map(|x| x + rng.gen_range(-2.5..2.5)).collect();
    for (name, params) in [
        (
            "server defaults",
            SamplingParams {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95,
                seed: 77,
                ..SamplingParams::greedy()
            },
        ),
        (
            "wide",
            SamplingParams {
                temperature: 1.3,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.001,
                seed: 78,
                ..SamplingParams::greedy()
            },
        ),
    ] {
        let p = cpu_kept(&trunk, &params);
        let q = cpu_kept(&head, &draft_params(&params));
        let (mut agree, mut accepted) = (0, 0);
        let steps = 96;
        for step in 0..steps {
            let (d, got, _) = draft_and_spec(&ctx, &trunk, &head, &params, step);
            let want = cpu_spec(&p, &q, d, params.seed, step as u32);
            assert!(
                p.iter().any(|(id, _)| *id == got),
                "{name}: {got} outside p's support"
            );
            if got == want {
                agree += 1;
            }
            if got == d {
                accepted += 1;
            }
        }
        // Float noise at interval boundaries may pick a neighbour.
        assert!(agree >= steps - 3, "{name}: {agree}/{steps} matched the reference");
        assert!(
            accepted > 0 && accepted < steps,
            "{name}: {accepted}/{steps} accepted, both outcomes must occur"
        );
    }
}

/// The marginal of the speculative draw is the trunk's distribution, and
/// the acceptance rate is the overlap sum(min(p, q)): 8 192 draws over a
/// 48-token vocabulary against the exact p.
#[test]
fn speculative_draw_is_distributed_as_the_trunk() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(25);
    let v = 48;
    let head: Vec<f32> = (0..v).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let trunk: Vec<f32> = head.iter().map(|x| x + rng.gen_range(-1.5..1.5)).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        seed: 31,
        ..SamplingParams::greedy()
    };
    let p = cpu_kept(&trunk, &params);
    let q = cpu_kept(&head, &params);
    let overlap: f32 = p
        .iter()
        .map(|(id, x)| x.min(q.iter().find(|(i, _)| i == id).map_or(0.0, |(_, y)| *y)))
        .sum();
    let n = 8192usize;
    let t_trunk = Tensor::from_f32(&ctx, &trunk, &[v]).expect("trunk");
    let t_head = Tensor::from_f32(&ctx, &head, &[v]).expect("head");
    let scratch = SamplerScratch::new(&ctx, v).expect("scratch");
    let dists = DraftDists::new(&ctx, 1).expect("dists");
    let dist = dists.slot(0).expect("slot");
    let drafted = Tensor::zeros(&ctx, &[n], DType::U32).expect("drafted");
    let outs = Tensor::zeros(&ctx, &[n], DType::U32).expect("outs");
    let pass = ctx.begin().expect("pass");
    for step in 0..n {
        let d = drafted.view(step, &[1]).expect("d");
        let out = outs.view(step, &[1]).expect("out");
        sample_draft_f32(
            &ctx,
            &pass,
            &t_head,
            &scratch,
            &params,
            draft_step(step),
            &d,
            &dist,
        )
        .expect("draft");
        sample_spec_f32(
            &ctx, &pass, &t_trunk, &scratch, &params, step, &dist, &d, &out,
        )
        .expect("spec");
    }
    pass.commit_wait().expect("run");
    let d = drafted.to_u32().expect("d");
    let got = outs.to_u32().expect("outs");
    let mut hist = vec![0usize; v];
    for &t in &got {
        hist[t as usize] += 1;
    }
    let accepted = d.iter().zip(&got).filter(|(a, b)| a == b).count();
    let mut chi2 = 0.0f64;
    for (id, x) in &p {
        let expected = *x as f64 * n as f64;
        let observed = hist[*id as usize] as f64;
        chi2 += (observed - expected).powi(2) / expected;
        assert!(
            (observed / n as f64 - *x as f64).abs() < 0.02,
            "token {id}: {observed} of {n} against p = {x}"
        );
    }
    // 47 degrees of freedom: the 99.9th percentile is about 82.
    assert!(chi2 < 82.0, "chi-square {chi2:.1} over {} tokens", p.len());
    let rate = accepted as f64 / n as f64;
    assert!(
        (rate - overlap as f64).abs() < 0.03,
        "accepted {rate:.3} of proposals, overlap sum(min(p, q)) = {overlap:.3}"
    );
}
