use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::kernels::mrope_axis;

fn random(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

/// rmsnorm(+1) then partial NeoX RoPE on one head row, rounding like the kernel.
fn prep_head(
    x: &[f32],
    w: &[f32],
    rot: usize,
    pos: usize,
    theta: f32,
    eps: f32,
) -> Vec<f32> {
    prep_head_mrope(x, w, rot, [pos as u32; 3], theta, eps)
}

/// `prep_head` with a 3-axis position: pair `j` rotates by the axis
/// `mrope_axis` gives it (equal axes make this the scalar rule).
fn prep_head_mrope(
    x: &[f32],
    w: &[f32],
    rot: usize,
    pos: [u32; 3],
    theta: f32,
    eps: f32,
) -> Vec<f32> {
    let gain: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
    let mut normed = cpu_ref::round_bf16(&cpu_ref::rmsnorm(x, &gain, x.len(), eps));
    // The kernel rounds cos/sin, both products and the sum to bf16, like the
    // torch reference does on bf16 tensors.
    let bf = |v: f32| cpu_ref::round_bf16(&[v])[0];
    let half = rot / 2;
    for j in 0..half {
        let angle =
            pos[mrope_axis(j)] as f32 * theta.powf(-2.0 * j as f32 / rot as f32);
        let (c, s) = (bf(angle.cos()), bf(angle.sin()));
        let (lo, hi) = (normed[j], normed[half + j]);
        normed[j] = bf(bf(lo * c) - bf(hi * s));
        normed[half + j] = bf(bf(hi * c) + bf(lo * s));
    }
    normed
}

fn position_rows(ctx: &MetalContext, rows: &[[u32; 3]]) -> Tensor {
    let flat: Vec<u32> = rows.iter().flatten().copied().collect();
    Tensor::from_bytes(ctx, bytemuck::cast_slice(&flat), &[rows.len(), 3], DType::U32)
        .expect("positions")
}

/// Runs the indexer's query prep and block keys for the chunk at `base_pos`
/// with `rope` and returns `(q, blocks)`.
#[allow(clippy::too_many_arguments)]
fn prep_and_blocks(
    ctx: &MetalContext,
    qk: &[f32],
    wq: &[f32],
    wk: &[f32],
    cache_init: &[f32],
    m: usize,
    nh: usize,
    rot: usize,
    ratio: usize,
    base_pos: usize,
    rope: Rope<'_>,
) -> (Vec<f32>, Vec<f32>) {
    let d = INDEXER_D;
    let (theta, eps) = (1.0e7f32, 1e-6f32);
    let max_seq = cache_init.len() / d;
    let t_qk = Tensor::from_f32_as_bf16(ctx, qk, &[m, (nh + 1) * d]).expect("qk");
    let t_wq = Tensor::from_f32_as_bf16(ctx, wq, &[d]).expect("wq");
    let t_wk = Tensor::from_f32_as_bf16(ctx, wk, &[d]).expect("wk");
    let q = Tensor::zeros(ctx, &[m, nh, d], DType::BF16).expect("q");
    let cache =
        Tensor::from_f32_as_bf16(ctx, cache_init, &[max_seq, d]).expect("cache");
    let blocks =
        Tensor::zeros(ctx, &[max_seq / ratio, d], DType::BF16).expect("blocks");
    let first_block = base_pos / ratio;
    let count = (base_pos + m) / ratio - first_block;
    let pass = ctx.begin().expect("pass");
    qsa_prep_q(ctx, &pass, &t_qk, &t_wq, &q, nh, rot, base_pos, theta, eps, rope)
        .expect("prep q");
    qsa_scatter_keys(ctx, &pass, &t_qk, &cache, nh, base_pos).expect("scatter");
    qsa_block_keys(
        ctx,
        &pass,
        &cache,
        &t_wk,
        &blocks,
        ratio,
        first_block,
        count,
        rot,
        theta,
        eps,
        rope,
    )
    .expect("blocks");
    pass.commit_wait().expect("commit");
    (q.to_f32().expect("q"), blocks.to_f32().expect("blocks"))
}

/// The indexer's M-RoPE variants (queries at their own 3-axis positions,
/// block keys at their first token's) against the CPU rule, bit-exact
/// against the scalar kernels when the axes agree, and the scalar kernels'
/// delta against a shifted chunk, bit-exact. The position buffer starts at
/// the first block's first token, before the chunk, as the prefill's does.
#[test]
fn prep_q_and_block_keys_mrope_match_cpu_and_the_scalar_kernels() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(47);
    let (m, nh, d, rot, ratio, base_pos) =
        (7usize, 4usize, INDEXER_D, 64usize, 4usize, 9usize);
    let (theta, eps) = (1.0e7f32, 1e-6f32);
    let qk = cpu_ref::round_bf16(&random(&mut rng, m * (nh + 1) * d, -2.0, 2.0));
    let wq = cpu_ref::round_bf16(&random(&mut rng, d, -0.5, 0.5));
    let wk = cpu_ref::round_bf16(&random(&mut rng, d, -0.5, 0.5));
    let max_seq = 32usize;
    let cache_init = cpu_ref::round_bf16(&random(&mut rng, max_seq * d, -2.0, 2.0));
    let base = base_pos / ratio * ratio;
    let rows = base_pos + m - base;
    let mixed: Vec<[u32; 3]> = (0..rows)
        .map(|_| {
            [
                rng.gen_range(0..300u32),
                rng.gen_range(0..300u32),
                rng.gen_range(0..300u32),
            ]
        })
        .collect();
    let t_mixed = position_rows(&ctx, &mixed);
    let (q, blocks) = prep_and_blocks(
        &ctx,
        &qk,
        &wq,
        &wk,
        &cache_init,
        m,
        nh,
        rot,
        ratio,
        base_pos,
        Rope::Rows { positions: &t_mixed, base },
    );

    let mut expected_q = Vec::with_capacity(m * nh * d);
    let mut cache_ref = cache_init.clone();
    for r in 0..m {
        let row = &qk[r * (nh + 1) * d..(r + 1) * (nh + 1) * d];
        for h in 0..nh {
            expected_q.extend(prep_head_mrope(
                &row[h * d..(h + 1) * d],
                &wq,
                rot,
                mixed[base_pos + r - base],
                theta,
                eps,
            ));
        }
        cache_ref[(base_pos + r) * d..(base_pos + r + 1) * d]
            .copy_from_slice(&row[nh * d..]);
    }
    cpu_ref::assert_close(&q, &expected_q, 2e-2, 2e-2);
    let first_block = base_pos / ratio;
    let complete = (base_pos + m) / ratio;
    assert!(complete > first_block + 1, "the chunk completes several blocks");
    for b in first_block..complete {
        let mut pooled = vec![0.0f32; d];
        for i in 0..ratio {
            for (p, v) in pooled
                .iter_mut()
                .zip(&cache_ref[(b * ratio + i) * d..(b * ratio + i + 1) * d])
            {
                *p += v;
            }
        }
        let pooled: Vec<f32> = cpu_ref::round_bf16(
            &pooled.iter().map(|v| v / ratio as f32).collect::<Vec<_>>(),
        );
        // The block key takes the 3-axis position of its first token, which
        // for the first block lies before the chunk.
        let expected =
            prep_head_mrope(&pooled, &wk, rot, mixed[b * ratio - base], theta, eps);
        cpu_ref::assert_close(&blocks[b * d..(b + 1) * d], &expected, 2e-2, 2e-2);
    }
    assert!(blocks[complete * d..].iter().all(|&v| v == 0.0));

    // Text rows: bit-exact against the scalar kernels.
    let (q_scalar, blocks_scalar) = prep_and_blocks(
        &ctx,
        &qk,
        &wq,
        &wk,
        &cache_init,
        m,
        nh,
        rot,
        ratio,
        base_pos,
        Rope::Delta(0),
    );
    assert_ne!((&q, &blocks), (&q_scalar, &blocks_scalar));
    let text: Vec<[u32; 3]> = (0..rows).map(|i| [(base + i) as u32; 3]).collect();
    let t_text = position_rows(&ctx, &text);
    let (q_text, blocks_text) = prep_and_blocks(
        &ctx,
        &qk,
        &wq,
        &wk,
        &cache_init,
        m,
        nh,
        rot,
        ratio,
        base_pos,
        Rope::Rows { positions: &t_text, base },
    );
    assert_eq!(q_text, q_scalar);
    assert_eq!(blocks_text, blocks_scalar);

    // The delta on the scalar kernels: a chunk `ratio` positions later with
    // delta `-ratio` ropes identically; the blocks made of chunk rows alone
    // come out equal one slot later (the first block also pools cache rows
    // before the chunk, which differ between the two runs).
    let (q_shift, blocks_shift) = prep_and_blocks(
        &ctx,
        &qk,
        &wq,
        &wk,
        &cache_init,
        m,
        nh,
        rot,
        ratio,
        base_pos + ratio,
        Rope::Delta(-(ratio as i64)),
    );
    assert_eq!(q_shift, q_scalar);
    let inner = base_pos.div_ceil(ratio);
    assert!(inner < complete, "a block lies entirely inside the chunk");
    assert_eq!(
        blocks_shift[(inner + 1) * d..(complete + 1) * d],
        blocks_scalar[inner * d..complete * d]
    );
}

#[test]
fn prep_q_and_block_keys_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(41);
    let (m, nh, d, rot, ratio, base_pos) =
        (6usize, 4usize, INDEXER_D, 64usize, 4usize, 9usize);
    let (theta, eps) = (1.0e7f32, 1e-6f32);
    let qk = cpu_ref::round_bf16(&random(&mut rng, m * (nh + 1) * d, -2.0, 2.0));
    let wq = cpu_ref::round_bf16(&random(&mut rng, d, -0.5, 0.5));
    let wk = cpu_ref::round_bf16(&random(&mut rng, d, -0.5, 0.5));
    let max_seq = 32usize;
    let cache_init = cpu_ref::round_bf16(&random(&mut rng, max_seq * d, -2.0, 2.0));

    let t_qk = Tensor::from_f32_as_bf16(&ctx, &qk, &[m, (nh + 1) * d]).expect("qk");
    let t_wq = Tensor::from_f32_as_bf16(&ctx, &wq, &[d]).expect("wq");
    let t_wk = Tensor::from_f32_as_bf16(&ctx, &wk, &[d]).expect("wk");
    let q = Tensor::zeros(&ctx, &[m, nh, d], DType::BF16).expect("q");
    let cache =
        Tensor::from_f32_as_bf16(&ctx, &cache_init, &[max_seq, d]).expect("cache");
    let blocks =
        Tensor::zeros(&ctx, &[max_seq / ratio, d], DType::BF16).expect("blocks");

    // Blocks completed by this chunk: those whose four tokens all lie below
    // base_pos + m, starting at the first block touching the chunk.
    let first_block = base_pos / ratio;
    let complete = (base_pos + m) / ratio;
    let count = complete - first_block;
    let pass = ctx.begin().expect("pass");
    qsa_prep_q(
        &ctx,
        &pass,
        &t_qk,
        &t_wq,
        &q,
        nh,
        rot,
        base_pos,
        theta,
        eps,
        Rope::Delta(0),
    )
    .expect("prep q");
    qsa_scatter_keys(&ctx, &pass, &t_qk, &cache, nh, base_pos).expect("scatter");
    qsa_block_keys(
        &ctx,
        &pass,
        &cache,
        &t_wk,
        &blocks,
        ratio,
        first_block,
        count,
        rot,
        theta,
        eps,
        Rope::Delta(0),
    )
    .expect("blocks");
    pass.commit_wait().expect("commit");

    let mut expected_q = Vec::with_capacity(m * nh * d);
    let mut cache_ref = cache_init.clone();
    for r in 0..m {
        let row = &qk[r * (nh + 1) * d..(r + 1) * (nh + 1) * d];
        for h in 0..nh {
            expected_q.extend(prep_head(
                &row[h * d..(h + 1) * d],
                &wq,
                rot,
                base_pos + r,
                theta,
                eps,
            ));
        }
        cache_ref[(base_pos + r) * d..(base_pos + r + 1) * d]
            .copy_from_slice(&row[nh * d..]);
    }
    cpu_ref::assert_close(&q.to_f32().expect("q"), &expected_q, 2e-2, 2e-2);
    assert_eq!(cache.to_f32().expect("cache"), cache_ref);

    let got_blocks = blocks.to_f32().expect("blocks");
    for b in first_block..complete {
        let mut pooled = vec![0.0f32; d];
        for i in 0..ratio {
            for (p, v) in pooled
                .iter_mut()
                .zip(&cache_ref[(b * ratio + i) * d..(b * ratio + i + 1) * d])
            {
                *p += v;
            }
        }
        let pooled: Vec<f32> = cpu_ref::round_bf16(
            &pooled.iter().map(|v| v / ratio as f32).collect::<Vec<_>>(),
        );
        let expected = prep_head(&pooled, &wk, rot, b * ratio, theta, eps);
        cpu_ref::assert_close(&got_blocks[b * d..(b + 1) * d], &expected, 2e-2, 2e-2);
    }
    // Blocks outside the range stay untouched (zero).
    assert!(got_blocks[complete * d..].iter().all(|&v| v == 0.0));
}

fn cpu_scores(q: &[f32], blocks: &[f32], nh: usize, d: usize, nb: usize) -> Vec<f32> {
    (0..nb)
        .map(|b| {
            let mut s = 0.0f32;
            for h in 0..nh {
                let dot: f32 = (0..d).map(|i| q[h * d + i] * blocks[b * d + i]).sum();
                s += dot.max(0.0);
            }
            s / (d as f32).sqrt()
        })
        .collect()
}

#[test]
fn scores_and_selection_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(42);
    let (nh, d, ratio, k_max) = (4usize, INDEXER_D, 4usize, 5usize);
    // Positions 40..43 see 10 or 11 complete blocks: more than k_max.
    let (qb, base_pos) = (4usize, 40usize);
    let nb_max = visible_blocks(base_pos + qb - 1, ratio);
    let q = cpu_ref::round_bf16(&random(&mut rng, qb * nh * d, -1.0, 1.0));
    // Quantized block keys make exact score ties likely, which exercises the
    // deterministic tie break.
    let blocks: Vec<f32> =
        (0..nb_max * d).map(|_| (rng.gen_range(-2i32..3)) as f32 * 0.25).collect();
    let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nh, d]).expect("q");
    let t_blocks =
        Tensor::from_f32_as_bf16(&ctx, &blocks, &[nb_max, d]).expect("blocks");
    let scores = Tensor::zeros(&ctx, &[qb, nb_max], DType::F32).expect("scores");
    let sel = Tensor::zeros(&ctx, &[qb, k_max], DType::U32).expect("sel");
    let n_sel = Tensor::zeros(&ctx, &[qb], DType::U32).expect("n_sel");
    let pass = ctx.begin().expect("pass");
    qsa_scores(&ctx, &pass, &t_q, &t_blocks, &scores, nh, nb_max, base_pos, ratio)
        .expect("scores");
    qsa_select_blocks(
        &ctx, &pass, &scores, &sel, &n_sel, qb, nb_max, base_pos, ratio, k_max,
    )
    .expect("select");
    pass.commit_wait().expect("commit");

    let got_scores = scores.to_f32().expect("scores");
    let got_sel = sel.to_u32().expect("sel");
    let got_n = n_sel.to_u32().expect("n_sel");
    for qi in 0..qb {
        let nb = visible_blocks(base_pos + qi, ratio);
        let expected =
            cpu_scores(&q[qi * nh * d..(qi + 1) * nh * d], &blocks, nh, d, nb);
        let row = &got_scores[qi * nb_max..qi * nb_max + nb];
        cpu_ref::assert_close(row, &expected, 1e-3, 1e-3);
        assert!(
            got_scores[qi * nb_max + nb..(qi + 1) * nb_max]
                .iter()
                .all(|v| *v == f32::NEG_INFINITY)
        );

        // Reference selection on the kernel's own scores: sort by (score desc,
        // index asc), keep k_max, report ascending.
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
        let mut expected_sel: Vec<u32> =
            order[..k_max.min(nb)].iter().map(|&b| b as u32).collect();
        expected_sel.sort_unstable();
        assert_eq!(got_n[qi] as usize, k_max.min(nb));
        assert_eq!(
            &got_sel[qi * k_max..qi * k_max + got_n[qi] as usize],
            &expected_sel[..]
        );
    }
}

#[test]
fn selection_keeps_everything_below_budget() {
    let ctx = MetalContext::new().expect("metal context");
    let (ratio, k_max, qb, base_pos) = (4usize, 8usize, 3usize, 5usize);
    let nb_max = visible_blocks(base_pos + qb - 1, ratio);
    let scores = Tensor::from_f32(&ctx, &vec![0.5f32; qb * nb_max], &[qb, nb_max])
        .expect("scores");
    let sel = Tensor::zeros(&ctx, &[qb, k_max], DType::U32).expect("sel");
    let n_sel = Tensor::zeros(&ctx, &[qb], DType::U32).expect("n_sel");
    let pass = ctx.begin().expect("pass");
    qsa_select_blocks(
        &ctx, &pass, &scores, &sel, &n_sel, qb, nb_max, base_pos, ratio, k_max,
    )
    .expect("select");
    pass.commit_wait().expect("commit");
    let got_sel = sel.to_u32().expect("sel");
    for (qi, n) in n_sel.to_u32().expect("n").into_iter().enumerate() {
        let nb = visible_blocks(base_pos + qi, ratio);
        assert_eq!(n as usize, nb);
        assert_eq!(
            &got_sel[qi * k_max..qi * k_max + nb],
            &(0..nb as u32).collect::<Vec<_>>()[..]
        );
    }
}

#[test]
fn sparse_attention_matches_dense_over_selected_tokens() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(43);
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 3usize);
    let nq = kvh * group;
    let (qb, base_pos, max_seq) = (3usize, 20usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = cpu_ref::round_bf16(&random(&mut rng, qb * nq * d, -1.0, 1.0));
    let k = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    let v = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    // Hand-picked ascending selections (k_max blocks each).
    let sel: Vec<u32> = vec![0, 2, 4, 1, 2, 3, 0, 3, 5];
    let n_sel: Vec<u32> = vec![3, 3, 3];

    let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
    let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let t_v = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let t_sel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[qb, k_max], DType::U32)
            .expect("sel");
    let t_n = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
        .expect("n");
    let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
    let slots = split_scratch_slots(qb, k_max, ratio);
    let partials = Tensor::zeros(&ctx, &[slots * nq, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[slots * nq, 2], DType::F32).expect("stats");
    let pass = ctx.begin().expect("pass");
    qsa_attention(
        &ctx,
        &pass,
        &t_q,
        &t_k,
        &t_v,
        &t_sel,
        &t_n,
        &out,
        &SparseSplitScratch { partials: &partials, stats: &stats },
        qb,
        k_max,
        ratio,
        base_pos,
        scale,
    )
    .expect("sparse attention");
    pass.commit_wait().expect("commit");

    let mut expected = vec![0.0f32; qb * nq * d];
    for qi in 0..qb {
        let pos = base_pos + qi;
        let tail_start = visible_blocks(pos, ratio) * ratio;
        let mut tokens: Vec<usize> = Vec::new();
        for &b in &sel[qi * k_max..qi * k_max + n_sel[qi] as usize] {
            tokens.extend((0..ratio).map(|i| b as usize * ratio + i));
        }
        tokens.extend(tail_start..=pos);
        assert_eq!(tokens.len(), attended_tokens(pos, ratio, n_sel[qi] as usize));
        for hq in 0..nq {
            let kh = hq / group;
            let qrow = &q[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            let logits: Vec<f32> = tokens
                .iter()
                .map(|&t| {
                    let krow = &k[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                    qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
            let sum: f32 = weights.iter().sum();
            let o = &mut expected[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            for (w, &t) in weights.iter().zip(&tokens) {
                let vrow = &v[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                for i in 0..d {
                    o[i] += w / sum * vrow[i];
                }
            }
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("out"), &expected, 2e-2, 2e-2);
}

/// The production shape puts the incomplete tail block alone in the last
/// split (2048 selected tokens fill exactly eight splits); reproduce that with
/// a 64-block budget over 256 tokens and tails of 0..3 tokens.
#[test]
fn sparse_attention_tail_in_its_own_split_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(44);
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 64usize);
    let nq = kvh * group;
    let (qb, base_pos, max_seq) = (4usize, 299usize, 320usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = cpu_ref::round_bf16(&random(&mut rng, qb * nq * d, -1.0, 1.0));
    let k = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    let v = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    // Every query selects 64 distinct blocks out of its visible 75 or 76.
    let mut sel: Vec<u32> = Vec::new();
    for qi in 0..qb {
        let nb = visible_blocks(base_pos + qi, ratio);
        let mut blocks: Vec<u32> = (0..nb as u32).collect();
        for i in (1..blocks.len()).rev() {
            let j = rng.gen_range(0..=i);
            blocks.swap(i, j);
        }
        let mut chosen = blocks[..k_max].to_vec();
        chosen.sort_unstable();
        sel.extend(chosen);
    }
    let n_sel = vec![k_max as u32; qb];

    let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
    let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let t_v = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let t_sel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[qb, k_max], DType::U32)
            .expect("sel");
    let t_n = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
        .expect("n");
    let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
    let plan = SparseSplitPlan { split: QSA_SPLIT_BATCHED, head_groups: 1 };
    assert_eq!(plan.splits(k_max, ratio), 2, "the tail must fall into a second split");
    let slots = split_scratch_slots(qb, k_max, ratio);
    let partials = Tensor::zeros(&ctx, &[slots * nq, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[slots * nq, 2], DType::F32).expect("stats");
    let run = |plan: SparseSplitPlan| {
        let pass = ctx.begin().expect("pass");
        qsa_attention_with(
            &ctx,
            &pass,
            &t_q,
            &t_k,
            &t_v,
            &t_sel,
            &t_n,
            &out,
            &SparseSplitScratch { partials: &partials, stats: &stats },
            qb,
            k_max,
            ratio,
            base_pos,
            scale,
            plan,
        )
        .expect("sparse attention");
        pass.commit_wait().expect("commit");
    };
    run(plan);

    let mut expected = vec![0.0f32; qb * nq * d];
    for qi in 0..qb {
        let pos = base_pos + qi;
        let tail_start = visible_blocks(pos, ratio) * ratio;
        let mut tokens: Vec<usize> = Vec::new();
        for &b in &sel[qi * k_max..(qi + 1) * k_max] {
            tokens.extend((0..ratio).map(|i| b as usize * ratio + i));
        }
        tokens.extend(tail_start..=pos);
        assert_eq!(tokens.len(), k_max * ratio + (pos + 1 - tail_start));
        for hq in 0..nq {
            let kh = hq / group;
            let qrow = &q[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            let logits: Vec<f32> = tokens
                .iter()
                .map(|&t| {
                    let krow = &k[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                    qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
            let sum: f32 = weights.iter().sum();
            let o = &mut expected[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            for (w, &t) in weights.iter().zip(&tokens) {
                let vrow = &v[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                for i in 0..d {
                    o[i] += w / sum * vrow[i];
                }
            }
        }
    }
    cpu_ref::assert_close(&out.to_f32().expect("out"), &expected, 2e-2, 2e-2);

    // The small-batch plans (finer splits, the head split) and the
    // finest split allowed give the same result, and the scratch a scratch
    // for `qb` queries allocates holds every one of them.
    for plan in [
        SparseSplitPlan { split: QSA_SPLIT_SMALL, head_groups: 1 },
        SparseSplitPlan {
            split: QSA_SPLIT_SMALL,
            head_groups: group.div_ceil(QSA_HEADS_PER_PASS),
        },
        SparseSplitPlan {
            split: QSA_SPLIT_MIN,
            head_groups: group.div_ceil(QSA_HEADS_PER_PASS),
        },
        SparseSplitPlan::for_rows(qb, group),
    ] {
        out.zero_fill();
        run(plan);
        let got = out.to_f32().expect("out");
        assert!(got.iter().all(|x| x.is_finite()), "{plan:?}: non-finite output");
        cpu_ref::assert_close(&got, &expected, 2e-2, 2e-2);
    }
}

// --- Per-query sparse attention -------------------------------------------------

/// The definition: a complete block's rows if selected, the tail causally.
fn attends_direct(sel_row: &[u32], pos: usize, ratio: usize, token: usize) -> bool {
    let b = token / ratio;
    if b < visible_blocks(pos, ratio) {
        sel_row.contains(&(b as u32))
    } else {
        token <= pos
    }
}

/// Random ascending selections with the overlap of neighbouring queries
/// controlled: `hot` blocks every query prefers, the rest drawn at random.
fn random_selections(
    rng: &mut StdRng,
    qb: usize,
    k_max: usize,
    base_pos: usize,
    ratio: usize,
    hot: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut sel = vec![0u32; qb * k_max];
    let mut n_sel = vec![0u32; qb];
    for qi in 0..qb {
        let nb = visible_blocks(base_pos + qi, ratio);
        let n = k_max.min(nb);
        let mut chosen: Vec<u32> = Vec::new();
        for &h in hot {
            if (h as usize) < nb && chosen.len() < n && rng.gen_bool(0.8) {
                chosen.push(h);
            }
        }
        while chosen.len() < n {
            let b = rng.gen_range(0..nb) as u32;
            if !chosen.contains(&b) {
                chosen.push(b);
            }
        }
        chosen.sort_unstable();
        sel[qi * k_max..qi * k_max + n].copy_from_slice(&chosen);
        n_sel[qi] = n as u32;
    }
    (sel, n_sel)
}

/// The selection against a CPU sort on decode-sized rows: every
/// blocks-per-thread variant of the kernel (8K, 16K and 32K contexts), a
/// two-chunk row past 32K tokens, three queries with different visible
/// counts, and coarse scores so the k-th key ties across many blocks.
#[test]
fn selection_matches_cpu_on_long_contexts() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    let (ratio, k_max, qb) = (4usize, 512usize, 3usize);
    for base_pos in [2100usize, 8192, 16384, 32768, 40000] {
        let nb_max = visible_blocks(base_pos + qb - 1, ratio);
        let scores: Vec<f32> =
            (0..qb * nb_max).map(|_| rng.gen_range(0i32..64) as f32 * 0.125).collect();
        let t_scores = Tensor::from_f32(&ctx, &scores, &[qb, nb_max]).expect("scores");
        let sel = Tensor::zeros(&ctx, &[qb, k_max], DType::U32).expect("sel");
        let n_sel = Tensor::zeros(&ctx, &[qb], DType::U32).expect("n_sel");
        let pass = ctx.begin().expect("pass");
        qsa_select_blocks(
            &ctx, &pass, &t_scores, &sel, &n_sel, qb, nb_max, base_pos, ratio, k_max,
        )
        .expect("select");
        pass.commit_wait().expect("commit");
        let got_sel = sel.to_u32().expect("sel");
        let got_n = n_sel.to_u32().expect("n_sel");
        for qi in 0..qb {
            let nb = visible_blocks(base_pos + qi, ratio);
            let row = &scores[qi * nb_max..qi * nb_max + nb];
            let mut order: Vec<usize> = (0..nb).collect();
            order
                .sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let mut expected: Vec<u32> =
                order[..k_max].iter().map(|&b| b as u32).collect();
            expected.sort_unstable();
            assert_eq!(got_n[qi] as usize, k_max, "base_pos {base_pos} query {qi}");
            assert_eq!(
                &got_sel[qi * k_max..(qi + 1) * k_max],
                &expected[..],
                "base_pos {base_pos} query {qi}"
            );
        }
    }
}

/// Latency of the per-query block selection on the decode path (one
/// serial dispatch after another, as between the layers of a step) at the
/// decode and verify row counts on 8K and 32K contexts, and on a prefill
/// chunk. Scores are non-negative like the indexer's ReLU'd dot products,
/// so the keys crowd the top radix digits.
/// `cargo test --release -- --ignored --nocapture select_blocks_timing`.
#[test]
#[ignore = "timing only"]
fn select_blocks_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(7);
    let (ratio, k_max) = (4usize, 512usize);
    for (qb, base_pos, what) in [
        (1usize, 8192usize, "warm-up"),
        (1, 64, "dispatch floor (16 blocks, no select)"),
        (1, 8192, "decode 8K"),
        (3, 8192, "verify 8K"),
        (1, 32768, "decode 32K"),
        (3, 32768, "verify 32K"),
        (64, 8192, "chunk 8K"),
    ] {
        let nb_max = visible_blocks(base_pos + qb - 1, ratio);
        let scores: Vec<f32> =
            (0..qb * nb_max).map(|_| rng.gen_range(0.0f32..8.0)).collect();
        let t_scores = Tensor::from_f32(&ctx, &scores, &[qb, nb_max]).expect("scores");
        let sel = Tensor::zeros(&ctx, &[qb, k_max], DType::U32).expect("sel");
        let n_sel = Tensor::zeros(&ctx, &[qb], DType::U32).expect("n_sel");
        let iters = 64;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let pass = ctx.begin().expect("pass");
            for _ in 0..iters {
                qsa_select_blocks(
                    &ctx, &pass, &t_scores, &sel, &n_sel, qb, nb_max, base_pos, ratio,
                    k_max,
                )
                .expect("select");
            }
            let start = std::time::Instant::now();
            pass.commit_wait().expect("commit");
            best = best.min(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        eprintln!("{what} [qb {qb}, {nb_max} blocks]: {best:.1} us per dispatch");
    }
}

/// Decode-shape inputs for the sparse split kernels: `layers` K/V caches of
/// `max_seq` positions, one query at `pos` with 512 random ascending blocks
/// selected (the production budget).
struct SparseDecodeSetup {
    q: Tensor,
    caches: Vec<(Tensor, Tensor)>,
    sel: Tensor,
    n_sel: Tensor,
    out: Tensor,
    partials: Tensor,
    stats: Tensor,
}

fn sparse_decode_setup(
    ctx: &MetalContext,
    rng: &mut StdRng,
    layers: usize,
    pos: usize,
) -> SparseDecodeSetup {
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 512usize);
    let nq = kvh * group;
    let max_seq = pos + 1;
    let q = Tensor::from_f32_as_bf16(ctx, &random(rng, nq * d, -1.0, 1.0), &[1, nq, d])
        .expect("q");
    let caches = (0..layers)
        .map(|_| {
            let k = Tensor::from_f32_as_bf16(
                ctx,
                &random(rng, kvh * max_seq * d, -1.0, 1.0),
                &[kvh, max_seq, d],
            )
            .expect("k");
            let v = Tensor::from_f32_as_bf16(
                ctx,
                &random(rng, kvh * max_seq * d, -1.0, 1.0),
                &[kvh, max_seq, d],
            )
            .expect("v");
            (k, v)
        })
        .collect();
    let nb = visible_blocks(pos, ratio);
    let mut pool: Vec<u32> = (0..nb as u32).collect();
    for i in 0..k_max {
        let pick = rng.gen_range(i..nb);
        pool.swap(i, pick);
    }
    let mut sel_v = pool[..k_max].to_vec();
    sel_v.sort_unstable();
    let sel =
        Tensor::from_bytes(ctx, bytemuck::cast_slice(&sel_v), &[1, k_max], DType::U32)
            .expect("sel");
    let n_sel = Tensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&[k_max as u32]),
        &[1],
        DType::U32,
    )
    .expect("n_sel");
    let out = Tensor::zeros(ctx, &[1, nq, d], DType::BF16).expect("out");
    let slots = split_scratch_slots(1, k_max, ratio);
    let partials = Tensor::zeros(ctx, &[slots * nq, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(ctx, &[slots * nq, 2], DType::F32).expect("stats");
    SparseDecodeSetup { q, caches, sel, n_sel, out, partials, stats }
}

fn sparse_decode_dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    s: &SparseDecodeSetup,
    layer: usize,
    pos: usize,
    names: (&'static str, &'static str),
) {
    let (k, v) = &s.caches[layer];
    let plan = SparseSplitPlan::for_rows(1, 12);
    qsa_attention_named(
        ctx,
        pass,
        &s.q,
        k,
        v,
        &s.sel,
        &s.n_sel,
        &s.out,
        &SparseSplitScratch { partials: &s.partials, stats: &s.stats },
        1,
        512,
        4,
        pos,
        1.0 / 16.0,
        plan,
        Some(names),
    )
    .expect("sparse attention");
}

/// The sparse decode attention as decode runs it: per layer the split
/// kernel, a barrier, the combine, a barrier, over 12 layers with their own
/// 8K K/V caches, one query with 512 blocks selected. Prints microseconds
/// per level for every `split:combine` pair in `LILY_QSA_DECODE_KERNELS`
/// (default the shipped pair). Run with `--ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn sparse_decode_chain_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(62);
    let layers = 12;
    let pos: usize = std::env::var("LILY_QSA_DECODE_POS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8191);
    let s = sparse_decode_setup(&ctx, &mut rng, layers, pos);
    let pairs: Vec<(&'static str, &'static str)> =
        std::env::var("LILY_QSA_DECODE_KERNELS")
            .map(|v| {
                v.split(',')
                    .map(|pair| {
                        let (a, b) = pair.split_once(':').expect("split:combine");
                        (
                            &*Box::leak(a.to_string().into_boxed_str()),
                            &*Box::leak(b.to_string().into_boxed_str()),
                        )
                    })
                    .collect()
            })
            .unwrap_or_else(|_| vec![("qsa_attn_split_bf16", "sdpa_decode_combine")]);
    let mut best = vec![f64::INFINITY; pairs.len()];
    // The GPU clock ramps over the first few hundred milliseconds of work:
    // untimed rounds until `LILY_CHAIN_WARMUP_MS` (300) have passed, then
    // six timed rounds, the first discarded.
    let warm = chain_warmup();
    let mut round = 0usize;
    loop {
        let timed = warm.elapsed() >= chain_warmup_ms();
        for k in 0..pairs.len() {
            let which = (round + k) % pairs.len();
            let pass = ctx.begin_concurrent().expect("pass");
            for layer in 0..layers {
                sparse_decode_dispatch(&ctx, &pass, &s, layer, pos, pairs[which]);
                pass.level_barrier(&[&s.out]).expect("barrier");
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if timed && round > 0 {
                best[which] = best[which]
                    .min((t.gpu_end_secs - t.gpu_start_secs) / layers as f64);
            }
        }
        if timed {
            round += 1;
            if round == 6 {
                break;
            }
        }
    }
    for ((a, b), secs) in pairs.iter().zip(&best) {
        eprintln!(
            "{a} + {b}: {:.2} us per layer (split + combine, two levels)",
            secs * 1e6
        );
    }
}

/// Start of a chain harness's warm-up window (see [`chain_warmup_ms`]).
pub(crate) fn chain_warmup() -> std::time::Instant {
    std::time::Instant::now()
}

/// How long a chain harness runs untimed before measuring: the GPU clock
/// ramps over the first few hundred milliseconds of work, and a harness
/// that measures cold reads up to twice the warm figure.
pub(crate) fn chain_warmup_ms() -> std::time::Duration {
    let ms = std::env::var("LILY_CHAIN_WARMUP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300u64);
    std::time::Duration::from_millis(ms)
}

/// The attention of each query over its own selection plus tail, on the CPU
/// (the definition every sparse route must reproduce).
#[allow(clippy::too_many_arguments)]
fn cpu_sparse_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sel: &[u32],
    n_sel: &[u32],
    (qb, nq, kvh, d): (usize, usize, usize, usize),
    (max_seq, k_max, ratio, base_pos): (usize, usize, usize, usize),
    scale: f32,
) -> Vec<f32> {
    let group = nq / kvh;
    let mut expected = vec![0.0f32; qb * nq * d];
    for qi in 0..qb {
        let pos = base_pos + qi;
        let row = &sel[qi * k_max..qi * k_max + n_sel[qi] as usize];
        let tokens: Vec<usize> =
            (0..=pos).filter(|&t| attends_direct(row, pos, ratio, t)).collect();
        for hq in 0..nq {
            let kh = hq / group;
            let qrow = &q[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            let logits: Vec<f32> = tokens
                .iter()
                .map(|&t| {
                    let krow = &k[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                    qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
            let sum: f32 = weights.iter().sum();
            let o = &mut expected[(qi * nq + hq) * d..(qi * nq + hq + 1) * d];
            for (w, &t) in weights.iter().zip(&tokens) {
                let vrow = &v[(kh * max_seq + t) * d..(kh * max_seq + t + 1) * d];
                for i in 0..d {
                    o[i] += w / sum * vrow[i];
                }
            }
        }
    }
    expected
}

/// The per-query tensor-op route against the definition, for both step
/// widths: a small budget where most queries attend a handful of blocks
/// (totals far from a step multiple, a tail of one to four rows), and the
/// production budget of 512 blocks at a 4K context with the attended count
/// just past a step boundary for some queries.
#[test]
fn query_attention_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(53);
    let (kvh, group, d, ratio) = (2usize, 12usize, 256usize, 4usize);
    let nq = kvh * group;
    let scale = 1.0 / (d as f32).sqrt();
    for (qb, base_pos, k_max) in [(37usize, 61usize, 6usize), (9, 4093, 512)] {
        let max_seq = base_pos + qb + 3;
        let q = cpu_ref::round_bf16(&random(&mut rng, qb * nq * d, -1.0, 1.0));
        let k = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
        let v = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
        let hot: Vec<u32> = vec![2, 5, 11];
        let (sel, n_sel) =
            random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);
        let expected = cpu_sparse_attention(
            &q,
            &k,
            &v,
            &sel,
            &n_sel,
            (qb, nq, kvh, d),
            (max_seq, k_max, ratio, base_pos),
            scale,
        );
        let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
        let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
        let t_v = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
        let t_sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[qb, k_max],
            DType::U32,
        )
        .expect("sel");
        let t_n =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
                .expect("n");
        let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        qsa_attention_query(
            &ctx, &pass, &t_q, &t_k, &t_v, &t_sel, &t_n, &out, qb, k_max, ratio,
            base_pos, scale,
        )
        .expect("query attention");
        pass.commit_wait().expect("commit");
        let got = out.to_f32().expect("out");
        assert!(got.iter().all(|x| x.is_finite()), "non-finite output");
        cpu_ref::assert_close(&got, &expected, 2e-2, 2e-2);
    }
}

/// Per-dispatch GPU time of the per-query sparse attention against the
/// split kernel (split pass plus combine) on a prefill
/// sub-batch (256 queries, 24 heads over 2 KV heads, 512-block budget) at
/// 8K, 32K and 64K, on the profile transport: minimum and median over the
/// dispatches of each kernel. The tiled route it replaced measured, gather
/// and attention, 0.55 and 1.95 ms at 8K, 1.19 and 3.30 at 32K, 1.51 and
/// 3.67 at 64K here, against 0.67, 0.69 and 0.71 for this kernel.
/// `LILY_QSA_QUERY_QB` sets the sub-batch, `LILY_QSA_QUERY_CONTEXTS` the
/// contexts (comma separated).
/// `cargo test --release -- --ignored --nocapture query_attention_timing`.
#[test]
#[ignore = "timing only"]
fn query_attention_timing() {
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    let mut rng = StdRng::seed_from_u64(59);
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 512usize);
    let nq = kvh * group;
    let qb: usize = std::env::var("LILY_QSA_QUERY_QB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let scale = 1.0 / (d as f32).sqrt();
    let contexts: Vec<usize> = std::env::var("LILY_QSA_QUERY_CONTEXTS")
        .map(|v| v.split(',').map(|x| x.parse().expect("context")).collect())
        .unwrap_or_else(|_| vec![8192, 32768, 65536]);
    for base_pos in contexts {
        let max_seq = base_pos + qb;
        let q = random(&mut rng, qb * nq * d, -1.0, 1.0);
        let k = random(&mut rng, kvh * max_seq * d, -1.0, 1.0);
        let v = random(&mut rng, kvh * max_seq * d, -1.0, 1.0);
        let nb = visible_blocks(base_pos, ratio);
        // Neighbouring queries share about half their blocks (a hot set
        // drawn with probability 0.8, the rest at random).
        let hot: Vec<u32> = (0..320).map(|_| rng.gen_range(0..nb as u32)).collect();
        let (sel, n_sel) =
            random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);
        let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
        let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
        let t_v = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
        let t_sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[qb, k_max],
            DType::U32,
        )
        .expect("sel");
        let t_n =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
                .expect("n");
        let slots = split_scratch_slots(qb, k_max, ratio);
        let partials =
            Tensor::zeros(&ctx, &[slots * nq, d], DType::F32).expect("partials");
        let stats = Tensor::zeros(&ctx, &[slots * nq, 2], DType::F32).expect("stats");
        let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
        crate::metal::profile::take();
        for _ in 0..8 {
            let pass = ctx.begin().expect("pass");
            qsa_attention(
                &ctx,
                &pass,
                &t_q,
                &t_k,
                &t_v,
                &t_sel,
                &t_n,
                &out,
                &SparseSplitScratch { partials: &partials, stats: &stats },
                qb,
                k_max,
                ratio,
                base_pos,
                scale,
            )
            .expect("split");
            pass.commit_wait().expect("commit");
            let pass = ctx.begin().expect("pass");
            qsa_attention_query(
                &ctx, &pass, &t_q, &t_k, &t_v, &t_sel, &t_n, &out, qb, k_max, ratio,
                base_pos, scale,
            )
            .expect("query");
            pass.commit_wait().expect("commit");
        }
        let passes = crate::metal::profile::take();
        for name in ["qsa_attn_split_bf16", "sdpa_decode_combine", "qsa_attn_gqa_nax"] {
            let mut ms: Vec<f64> = passes
                .iter()
                .map(|p| {
                    p.kernels
                        .iter()
                        .filter(|s| s.name == name)
                        .map(|s| s.gpu_secs * 1e3)
                        .sum::<f64>()
                })
                .filter(|&t| t > 0.0)
                .collect();
            ms.sort_by(|a, b| a.total_cmp(b));
            let n = ms.len();
            eprintln!(
                "base {base_pos} {name}: min {:.3} ms, median {:.3} ms over {n} passes",
                ms[0],
                ms[n / 2]
            );
        }
    }
}

/// The scores route boundary: 15 rows take the scalar kernel, 16 and more
/// the tensor-op one; both agree with the CPU on every visible block and
/// write -inf past it, on ragged query and block counts (a last group of
/// fewer than four queries, a last block group of fewer than 32).
#[test]
fn scores_on_both_routes_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(61);
    let (nh, d, ratio) = (4usize, INDEXER_D, 4usize);
    for (qb, base_pos) in [(15usize, 700usize), (16, 700), (37, 1229), (256, 4093)] {
        let nb_max = visible_blocks(base_pos + qb - 1, ratio);
        let q = cpu_ref::round_bf16(&random(&mut rng, qb * nh * d, -1.0, 1.0));
        let blocks = cpu_ref::round_bf16(&random(&mut rng, nb_max * d, -1.0, 1.0));
        let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nh, d]).expect("q");
        let t_blocks =
            Tensor::from_f32_as_bf16(&ctx, &blocks, &[nb_max, d]).expect("blocks");
        let scores = Tensor::zeros(&ctx, &[qb, nb_max], DType::F32).expect("scores");
        let pass = ctx.begin().expect("pass");
        qsa_scores(&ctx, &pass, &t_q, &t_blocks, &scores, nh, nb_max, base_pos, ratio)
            .expect("scores");
        pass.commit_wait().expect("commit");
        let got = scores.to_f32().expect("scores");
        for qi in 0..qb {
            let nb = visible_blocks(base_pos + qi, ratio);
            let expected =
                cpu_scores(&q[qi * nh * d..(qi + 1) * nh * d], &blocks, nh, d, nb);
            cpu_ref::assert_close(
                &got[qi * nb_max..qi * nb_max + nb],
                &expected,
                1e-4,
                1e-4,
            );
            assert!(
                got[qi * nb_max + nb..(qi + 1) * nb_max]
                    .iter()
                    .all(|v| *v == f32::NEG_INFINITY),
                "qb {qb} query {qi}: blocks past the visible ones must be -inf"
            );
        }
    }
}

// --- the q8 K/V cache -----------------------------------------------------------

/// Both sparse kernels over a q8 cache against their bf16 twins over the
/// same cache dequantized (`kv_dequant_q8`, the values rounded to bf16):
/// the per-query tensor-op kernel rounds its dequantized operands to bf16
/// exactly as the staging does, so it must match bit for bit; the split
/// kernel dots the unrounded products, so it matches within that rounding.
#[test]
fn q8_sparse_attention_matches_bf16_over_the_dequantized_cache() {
    use crate::kernels::attention::tests::{kv_like_rows, q8_cache_of};
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(84);
    // The model's GQA group of 12: the tensor-op tile's second row
    // fragment and the split kernel's third head group both hold heads.
    let (kvh, group, d, ratio) = (2usize, 12usize, 256usize, 4usize);
    let nq = kvh * group;
    let scale = 1.0 / (d as f32).sqrt();
    for (qb, base_pos, k_max) in
        [(37usize, 61usize, 6usize), (20, 4093, 512), (1, 5000, 512)]
    {
        let max_seq = base_pos + qb + 3;
        let q = cpu_ref::round_bf16(&random(&mut rng, qb * nq * d, -1.0, 1.0));
        let k = kv_like_rows(&mut rng, kvh * max_seq, d);
        let v = kv_like_rows(&mut rng, kvh * max_seq, d);
        let (kq, ks) = q8_cache_of(&ctx, &k, (kvh, max_seq, d), max_seq);
        let (vq, vs) = q8_cache_of(&ctx, &v, (kvh, max_seq, d), max_seq);
        let hot: Vec<u32> = vec![2, 5, 11];
        let (sel, n_sel) =
            random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);
        let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
        let t_sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[qb, k_max],
            DType::U32,
        )
        .expect("sel");
        let t_n =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
                .expect("n");
        let out = |_: ()| Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
        let (query_q8, query_bf, split_q8, split_bf) =
            (out(()), out(()), out(()), out(()));
        let slots = split_scratch_slots(qb, k_max, ratio);
        let partials =
            Tensor::zeros(&ctx, &[slots * nq, d], DType::F32).expect("partials");
        let stats = Tensor::zeros(&ctx, &[slots * nq, 2], DType::F32).expect("stats");
        let scratch = SparseSplitScratch { partials: &partials, stats: &stats };
        let pass = ctx.begin().expect("pass");
        qsa_attention_query(
            &ctx, &pass, &t_q, &kq, &vq, &t_sel, &t_n, &query_q8, qb, k_max, ratio,
            base_pos, scale,
        )
        .expect("q8 query attention");
        qsa_attention_query(
            &ctx, &pass, &t_q, &ks, &vs, &t_sel, &t_n, &query_bf, qb, k_max, ratio,
            base_pos, scale,
        )
        .expect("bf16 query attention");
        qsa_attention(
            &ctx, &pass, &t_q, &kq, &vq, &t_sel, &t_n, &split_q8, &scratch, qb, k_max,
            ratio, base_pos, scale,
        )
        .expect("q8 split attention");
        pass.level_barrier(&[]).expect("barrier");
        qsa_attention(
            &ctx, &pass, &t_q, &ks, &vs, &t_sel, &t_n, &split_bf, &scratch, qb, k_max,
            ratio, base_pos, scale,
        )
        .expect("bf16 split attention");
        pass.commit_wait().expect("commit");
        let at = format!("{qb} queries at {base_pos}, {k_max} blocks");
        let query_q8 = query_q8.to_f32().expect("q8");
        assert!(query_q8.iter().all(|x| x.is_finite()), "{at}: non-finite output");
        assert_eq!(query_q8, query_bf.to_f32().expect("bf16"), "{at}: query route");
        cpu_ref::assert_close(
            &split_q8.to_f32().expect("q8"),
            &split_bf.to_f32().expect("bf16"),
            2e-2,
            2e-2,
        );
        // And the q8 output is close to the unquantized attention overall
        // (relative RMS error): per element the outlier channels (x24) are
        // off by more than a step, since their group's coarse K steps also
        // move the softmax weights their large values are averaged with.
        // The data is harsh on purpose: 7 of a row's 8 groups hold an
        // outlier, so their step is 24 times coarser (about 2 % here); the
        // model's own error is measured on its activations, not here.
        let expected = cpu_sparse_attention(
            &q,
            &k,
            &v,
            &sel,
            &n_sel,
            (qb, nq, kvh, d),
            (max_seq, k_max, ratio, base_pos),
            scale,
        );
        let rms = |x: &mut dyn Iterator<Item = f32>| {
            let (sum, n) =
                x.fold((0.0f64, 0usize), |(s, n), v| (s + f64::from(v * v), n + 1));
            (sum / n as f64).sqrt()
        };
        let err = rms(&mut query_q8.iter().zip(&expected).map(|(a, b)| a - b));
        let rel = err / rms(&mut expected.iter().copied());
        println!("{at}: q8 vs unquantized relative RMS error {rel:.4}");
        assert!(rel < 0.04, "{at}: relative RMS error {rel}");
    }
}
