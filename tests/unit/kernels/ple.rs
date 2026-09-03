use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

fn random(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

/// The HF `Qwen4ExpTextNGramEmbedding` hashing, token by token, for a history
/// starting two eos tokens before `tokens[0]`.
fn cpu_hash_ids(
    tokens: &[u32],
    hist: [u32; 2],
    mult: &[u64],
    sizes: &[u64],
    offsets: &[u64],
    hpn: usize,
    eos: u32,
) -> Vec<u32> {
    let heads = sizes.len();
    let mut out = Vec::with_capacity(tokens.len() * heads);
    for r in 0..tokens.len() {
        let t0 = tokens[r];
        let p1 = if r >= 1 { tokens[r - 1] } else { hist[1] };
        let p2 = if r >= 2 { tokens[r - 2] } else if r == 1 { hist[1] } else { hist[0] };
        let s1 = if p1 == eos { eos } else { p1 };
        let s2 = if p1 == eos || p2 == eos { eos } else { p2 };
        for j in 0..heads {
            let mut mixed = t0 as u64 * mult[0] ^ s1 as u64 * mult[1];
            if j / hpn >= 1 {
                mixed ^= s2 as u64 * mult[2];
            }
            out.push((mixed % sizes[j]) as u32 + offsets[j] as u32);
        }
    }
    out
}

#[test]
fn hash_ids_match_cpu_and_respect_eos_segments() {
    let ctx = MetalContext::new().expect("metal context");
    // The real checkpoint constants (first four heads) keep the modulo honest.
    let mult = [23_703_573_157_769u64, 20_109_073_645_365, 8_052_911_324_071];
    let sizes: Vec<u64> = (0..16).map(|j| 20_000_003 + 10 * j as u64).collect();
    let offsets: Vec<u64> = (0..16).map(|j| j as u64 * 20_000_100).collect();
    let eos = 248_044u32;
    let tables = PleHashTables::new(&ctx, &mult, &sizes, &offsets, 8, eos).expect("tables");

    let tokens = vec![17u32, 248_319, eos, 5, 6, eos, eos, 9];
    let hist = [eos, 42];
    let t_tokens = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&tokens), &[tokens.len()], DType::U32).expect("tokens");
    let t_hist = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&hist), &[2], DType::U32).expect("hist");
    let ids = Tensor::zeros(&ctx, &[tokens.len(), 16], DType::U32).expect("ids");
    let hist_out = Tensor::zeros(&ctx, &[2], DType::U32).expect("hist_out");
    let pass = ctx.begin().expect("pass");
    ple_hash_ids(&ctx, &pass, &tables, &t_tokens, &t_hist, &ids).expect("hash");
    ple_hist_update(&ctx, &pass, &t_tokens, &t_hist, &hist_out).expect("hist");
    pass.commit_wait().expect("commit");

    let expected = cpu_hash_ids(&tokens, hist, &mult, &sizes, &offsets, 8, eos);
    assert_eq!(ids.to_u32().expect("ids"), expected);
    assert_eq!(hist_out.to_u32().expect("hist"), vec![eos, 9]);

    // A one-token step keeps the previous last token as the older entry.
    let one = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&[77u32]), &[1], DType::U32).expect("one");
    let pass = ctx.begin().expect("pass");
    ple_hist_update(&ctx, &pass, &one, &hist_out, &hist_out).expect("hist step");
    pass.commit_wait().expect("commit");
    assert_eq!(hist_out.to_u32().expect("hist"), vec![9, 77]);
}

#[test]
fn gather_q4_group32_matches_dequant() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(31);
    let (rows, k, gs, heads, m) = (64usize, 160usize, 32usize, 16usize, 3usize);
    let groups = k / gs;
    let codes: Vec<u32> = (0..rows * k / 8).map(|_| rng.r#gen()).collect();
    let scales = cpu_ref::round_bf16(&random(&mut rng, rows * groups, 0.001, 0.02));
    let biases = cpu_ref::round_bf16(&random(&mut rng, rows * groups, -0.1, 0.1));
    let table = QuantWeights {
        codes: Tensor::from_bytes(&ctx, bytemuck::cast_slice(&codes), &[rows, k / 8], DType::U32).expect("codes"),
        scales: Tensor::from_f32_as_bf16(&ctx, &scales, &[rows, groups]).expect("scales"),
        biases: Tensor::from_f32_as_bf16(&ctx, &biases, &[rows, groups]).expect("biases"),
        group_size: gs,
        bits: 4,
    };
    let ids: Vec<u32> = (0..m * heads).map(|_| rng.gen_range(0..rows as u32)).collect();
    let t_ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[m, heads], DType::U32).expect("ids");
    let out = Tensor::zeros(&ctx, &[m, heads * k], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    ple_gather_q4(&ctx, &pass, &table, &t_ids, heads, &out).expect("gather");
    pass.commit_wait().expect("commit");

    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, rows, k, gs);
    let mut expected = Vec::with_capacity(m * heads * k);
    for &id in &ids {
        expected.extend_from_slice(&dequant[id as usize * k..(id as usize + 1) * k]);
    }
    cpu_ref::assert_close(&out.to_f32().expect("out"), &cpu_ref::round_bf16(&expected), 1e-3, 1e-2);
}

#[test]
fn gate_value_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(32);
    let (rows, g, h) = (2usize, 4usize, 512usize);
    let key = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -1.0, 1.0));
    let query = cpu_ref::round_bf16(&random(&mut rng, rows * g * h, -1.0, 1.0));
    let value = cpu_ref::round_bf16(&random(&mut rng, rows * h, -2.0, 2.0));
    let t_key = Tensor::from_f32_as_bf16(&ctx, &key, &[rows, g * h]).expect("key");
    let t_query = Tensor::from_f32_as_bf16(&ctx, &query, &[rows, g * h]).expect("query");
    let t_value = Tensor::from_f32_as_bf16(&ctx, &value, &[rows, h]).expect("value");
    let gated = Tensor::zeros(&ctx, &[rows, g * h], DType::BF16).expect("gated");
    let pass = ctx.begin().expect("pass");
    ple_gate_value_bf16(&ctx, &pass, &t_key, &t_query, &t_value, &gated, h, g).expect("gate");
    pass.commit_wait().expect("commit");

    let mut expected = vec![0.0f32; rows * g * h];
    for r in 0..rows {
        for s in 0..g {
            let base = (r * g + s) * h;
            let dot: f32 = (0..h).map(|i| key[base + i] * query[base + i]).sum();
            let gate = dot / (h as f32).sqrt();
            let signed = gate.abs().max(1e-6).sqrt() * gate.signum();
            let sig = 1.0 / (1.0 + (-signed).exp());
            for i in 0..h {
                expected[base + i] = sig * value[r * h + i];
            }
        }
    }
    cpu_ref::assert_close(&gated.to_f32().expect("gated"), &expected, 2e-2, 2e-2);
}

/// Dilated causal conv reference over a whole sequence with an initial window.
fn cpu_dilated_conv(
    window: &[f32],
    x: &[f32],
    w: &[f32],
    c: usize,
    kd: usize,
    dil: usize,
) -> (Vec<f32>, Vec<f32>) {
    let s = (kd - 1) * dil;
    let m = x.len() / c;
    let mut out = vec![0.0f32; m * c];
    let mut window_out = vec![0.0f32; c * s];
    for ch in 0..c {
        // Full timeline for this channel: the window then the chunk.
        let mut line: Vec<f32> = (0..s).map(|i| window[ch * s + i]).collect();
        line.extend((0..m).map(|t| x[t * c + ch]));
        for t in 0..m {
            let now = s + t;
            let mut acc = 0.0f32;
            for tap in 0..kd {
                acc += w[tap * c + ch] * line[now - (kd - 1 - tap) * dil];
            }
            out[t * c + ch] = cpu_ref::silu(acc);
        }
        for i in 0..s {
            window_out[ch * s + i] = line[line.len() - s + i];
        }
    }
    (out, window_out)
}

#[test]
fn dilated_conv_prefill_matches_steps_and_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(33);
    let (c, kd, dil, m) = (96usize, 4usize, 3usize, 7usize);
    let s = (kd - 1) * dil;
    let window = cpu_ref::round_bf16(&random(&mut rng, c * s, -1.0, 1.0));
    let x = cpu_ref::round_bf16(&random(&mut rng, m * c, -1.0, 1.0));
    let w = cpu_ref::round_bf16(&random(&mut rng, kd * c, -0.5, 0.5));
    let base = cpu_ref::round_bf16(&random(&mut rng, m * c, -1.0, 1.0));
    let hyper0 = cpu_ref::round_bf16(&random(&mut rng, m * c, -1.0, 1.0));
    let t_w = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");

    // Batched prefill.
    let win_in = Tensor::from_f32_as_bf16(&ctx, &window, &[c, s]).expect("win_in");
    let win_out = Tensor::zeros(&ctx, &[c, s], DType::BF16).expect("win_out");
    let t_x = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
    let t_base = Tensor::from_f32_as_bf16(&ctx, &base, &[m, c]).expect("base");
    let t_hyper = Tensor::from_f32_as_bf16(&ctx, &hyper0, &[m, c]).expect("hyper");
    let pass = ctx.begin().expect("pass");
    ple_conv1d_prefill(&ctx, &pass, &win_in, &win_out, &t_x, &t_w, &t_base, &t_hyper, dil).expect("prefill");
    pass.commit_wait().expect("commit");

    let (conv, expected_window) = cpu_dilated_conv(&window, &x, &w, c, kd, dil);
    let expected: Vec<f32> = (0..m * c).map(|i| hyper0[i] + base[i] + conv[i]).collect();
    cpu_ref::assert_close(&t_hyper.to_f32().expect("hyper"), &expected, 2e-2, 2e-2);
    cpu_ref::assert_close(&win_out.to_f32().expect("win"), &expected_window, 0.0, 0.0);

    // Token-by-token steps reproduce the batched result.
    let step_window = Tensor::from_f32_as_bf16(&ctx, &window, &[c, s]).expect("step window");
    let mut stepped = Vec::with_capacity(m * c);
    for t in 0..m {
        let xt = Tensor::from_f32_as_bf16(&ctx, &x[t * c..(t + 1) * c], &[c]).expect("xt");
        let bt = Tensor::from_f32_as_bf16(&ctx, &base[t * c..(t + 1) * c], &[c]).expect("bt");
        let ht = Tensor::from_f32_as_bf16(&ctx, &hyper0[t * c..(t + 1) * c], &[c]).expect("ht");
        let pass = ctx.begin().expect("pass");
        ple_conv1d_step(&ctx, &pass, &step_window, &xt, &t_w, &bt, &ht, dil).expect("step");
        pass.commit_wait().expect("commit");
        stepped.extend(ht.to_f32().expect("ht"));
    }
    assert_eq!(stepped, t_hyper.to_f32().expect("hyper"));
    assert_eq!(step_window.to_f32().expect("step window"), win_out.to_f32().expect("win"));
}
