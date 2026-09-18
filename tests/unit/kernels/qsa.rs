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

// --- Tiled sparse attention ------------------------------------------------------

/// CPU model of `qsa_tile_union`: per tile, the ascending (block, query
/// mask) pairs below the tile's window plus the tail-block masks.
struct CpuTile {
    blocks: Vec<(u32, u32)>,
    tail: [u32; QSA_TILE_TAIL_BLOCKS],
}

fn cpu_tile_union(
    sel: &[u32],
    n_sel: &[u32],
    qb: usize,
    k_max: usize,
    base_pos: usize,
    ratio: usize,
) -> Vec<CpuTile> {
    (0..qb.div_ceil(QSA_TILE_BQ))
        .map(|t| {
            let q0 = t * QSA_TILE_BQ;
            let qn = QSA_TILE_BQ.min(qb - q0);
            let vb0 = visible_blocks(base_pos + q0, ratio) as u32;
            let mut mask = std::collections::BTreeMap::<u32, u32>::new();
            let mut tail = [0u32; QSA_TILE_TAIL_BLOCKS];
            for i in 0..qn {
                let qi = q0 + i;
                for &b in &sel[qi * k_max..qi * k_max + n_sel[qi] as usize] {
                    if b < vb0 {
                        *mask.entry(b).or_default() |= 1 << i;
                    } else {
                        tail[(b - vb0) as usize] |= 1 << i;
                    }
                }
            }
            CpuTile { blocks: mask.into_iter().collect(), tail }
        })
        .collect()
}

/// Whether query `i` of tile `t` attends cache row `token`, read off the
/// tile structures the way the kernel does.
fn cpu_tile_attends(
    tile: &CpuTile,
    base_pos: usize,
    t: usize,
    i: usize,
    ratio: usize,
    token: usize,
) -> bool {
    let p0 = base_pos + t * QSA_TILE_BQ;
    let p = p0 + i;
    let vb0 = visible_blocks(p0, ratio);
    let b = token / ratio;
    if b < vb0 {
        tile.blocks
            .binary_search_by_key(&(b as u32), |x| x.0)
            .map(|k| (tile.blocks[k].1 >> i) & 1 == 1)
            .unwrap_or(false)
    } else {
        token <= p
            && (token >= visible_blocks(p, ratio) * ratio
                || (tile.tail[b - vb0] >> i) & 1 == 1)
    }
}

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

/// CPU only: the union structures reproduce the attended set of every query
/// exactly, including the tail region where a block is complete for some
/// queries of a tile and not for others, and tiles short of BQ queries.
#[test]
fn tile_union_reference_matches_direct_definition() {
    let mut rng = StdRng::seed_from_u64(45);
    for &(qb, k_max, base_pos, ratio) in &[
        (37usize, 6usize, 61usize, 4usize),
        (16, 3, 0, 4),
        (50, 4, 7, 4),
        (5, 2, 13, 2),
    ] {
        let hot: Vec<u32> = (0..4).map(|_| rng.gen_range(0..8)).collect();
        let (sel, n_sel) =
            random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);
        let tiles = cpu_tile_union(&sel, &n_sel, qb, k_max, base_pos, ratio);
        for (t, tile) in tiles.iter().enumerate() {
            let q0 = t * QSA_TILE_BQ;
            for i in 0..QSA_TILE_BQ.min(qb - q0) {
                let qi = q0 + i;
                let pos = base_pos + qi;
                let row = &sel[qi * k_max..qi * k_max + n_sel[qi] as usize];
                for token in 0..base_pos + qb + ratio {
                    let direct = token <= pos && attends_direct(row, pos, ratio, token);
                    let tiled = cpu_tile_attends(tile, base_pos, t, i, ratio, token);
                    assert_eq!(
                        tiled, direct,
                        "qb {qb} k_max {k_max} base {base_pos}: tile {t} query {i} token {token}"
                    );
                }
            }
        }
    }
}

#[test]
fn tile_union_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(46);
    let (qb, k_max, base_pos, ratio, max_blocks) =
        (37usize, 6usize, 61usize, 4usize, 64usize);
    let hot: Vec<u32> = vec![1, 4, 9];
    let (sel, n_sel) = random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);
    let expected = cpu_tile_union(&sel, &n_sel, qb, k_max, base_pos, ratio);

    let t_sel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[qb, k_max], DType::U32)
            .expect("sel");
    let t_n = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
        .expect("n");
    let tiles =
        SparseTileScratch::new(&ctx, qb, max_blocks, k_max, 2, ratio, 512 << 20)
            .expect("tiles");
    let pass = ctx.begin().expect("pass");
    qsa_tile_union(&ctx, &pass, &t_sel, &t_n, &tiles, qb, k_max, ratio, base_pos)
        .expect("union");
    pass.commit_wait().expect("commit");

    let cap = tiles.cap();
    let n_union = tiles.n_union.to_u32().expect("n_union");
    let union_blk = tiles.union_blk.to_u32().expect("union_blk");
    let union_mask = tiles.union_mask.to_u32().expect("union_mask");
    let tail_mask = tiles.tail_mask.to_u32().expect("tail_mask");
    for (t, tile) in expected.iter().enumerate() {
        assert_eq!(n_union[t] as usize, tile.blocks.len(), "tile {t} union size");
        for (r, &(b, m)) in tile.blocks.iter().enumerate() {
            assert_eq!(union_blk[t * cap + r], b, "tile {t} entry {r} block");
            assert_eq!(union_mask[t * cap + r], m, "tile {t} entry {r} mask");
        }
        assert_eq!(
            &tail_mask[t * QSA_TILE_TAIL_BLOCKS..(t + 1) * QSA_TILE_TAIL_BLOCKS],
            &tile.tail[..],
            "tile {t} tail masks"
        );
    }
    // The mask rows are clean for the next sub-batch, and the statistics
    // count this dispatch.
    assert!(tiles.mask.to_u32().expect("mask").iter().all(|&m| m == 0));
    let total: u64 = expected.iter().map(|t| t.blocks.len() as u64).sum();
    assert_eq!(tiles.union_stats().expect("stats"), (total, expected.len() as u64));
}

/// The tile kernel agrees with the split kernel and the CPU definition over
/// the same selections, for every heads-per-pass variant, with a partial
/// last tile (37 queries) and overlapping selections.
#[test]
fn tiled_attention_matches_split_kernel_and_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(47);
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 6usize);
    let nq = kvh * group;
    let (qb, base_pos, max_seq) = (37usize, 61usize, 128usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = cpu_ref::round_bf16(&random(&mut rng, qb * nq * d, -1.0, 1.0));
    let k = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    let v = cpu_ref::round_bf16(&random(&mut rng, kvh * max_seq * d, -1.0, 1.0));
    let hot: Vec<u32> = vec![2, 5, 11];
    let (sel, n_sel) = random_selections(&mut rng, qb, k_max, base_pos, ratio, &hot);

    let t_q = Tensor::from_f32_as_bf16(&ctx, &q, &[qb, nq, d]).expect("q");
    let t_k = Tensor::from_f32_as_bf16(&ctx, &k, &[kvh, max_seq, d]).expect("k");
    let t_v = Tensor::from_f32_as_bf16(&ctx, &v, &[kvh, max_seq, d]).expect("v");
    let t_sel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[qb, k_max], DType::U32)
            .expect("sel");
    let t_n = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&n_sel), &[qb], DType::U32)
        .expect("n");

    // Reference: the split kernel.
    let out_split = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
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
        &out_split,
        &SparseSplitScratch { partials: &partials, stats: &stats },
        qb,
        k_max,
        ratio,
        base_pos,
        scale,
    )
    .expect("split attention");
    pass.commit_wait().expect("commit");
    let got_split = out_split.to_f32().expect("out");

    // Reference: the definition on the CPU.
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
    cpu_ref::assert_close(&got_split, &expected, 2e-2, 2e-2);

    // A row budget that holds every tile of the batch, and one that holds
    // a single tile so the batch is gathered and attended in groups.
    for budget in [1100usize << 20, 1 << 20] {
        let tiles = SparseTileScratch::new(
            &ctx,
            qb,
            max_seq / ratio,
            k_max,
            kvh,
            ratio,
            budget,
        )
        .expect("tiles");
        assert_eq!(
            tiles.row_group(),
            if budget > 1 << 20 { 3 } else { 1 },
            "row group at {budget} bytes"
        );
        let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        qsa_tile_union(&ctx, &pass, &t_sel, &t_n, &tiles, qb, k_max, ratio, base_pos)
            .expect("union");
        pass.level_barrier(&[
            &tiles.union_blk,
            &tiles.union_mask,
            &tiles.n_union,
            &tiles.tail_mask,
        ])
        .expect("barrier");
        qsa_attention_tiled(
            &ctx, &pass, &t_q, &t_k, &t_v, &tiles, &out, qb, ratio, base_pos, scale,
        )
        .expect("tiled attention");
        pass.commit_wait().expect("commit");
        let got = out.to_f32().expect("out");
        assert!(got.iter().all(|x| x.is_finite()), "non-finite output");
        cpu_ref::assert_close(&got, &expected, 2e-2, 2e-2);
        cpu_ref::assert_close(&got, &got_split, 2e-2, 2e-2);
    }
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

/// Per-dispatch GPU time of the tiled sparse attention on a prefill
/// sub-batch shape (256 queries, 24 heads over 2 KV heads, 512-block
/// budget, about 1.8x selection overlap per 16-query tile) at 8K and 32K,
/// on the profile transport: the minimum and median over the dispatches
/// of each kernel named in `LILY_QSA_TILE_KERNELS` (comma separated; the
/// shipped one-head kernel and the plain one-head body by default).
/// `cargo test --release -- --ignored --nocapture tiled_attention_timing`.
#[test]
#[ignore = "timing only"]
fn tiled_attention_timing() {
    // `LILY_QSA_TIMING_WALL=1`: production-shape passes timed by the clock
    // (dispatches on one level overlap) instead of per-kernel GPU times.
    let wall = std::env::var("LILY_QSA_TIMING_WALL").is_ok();
    let ctx = MetalContext::new_with_profile(!wall).expect("metal context");
    let mut rng = StdRng::seed_from_u64(49);
    let (kvh, group, d, ratio, k_max) = (2usize, 12usize, 256usize, 4usize, 512usize);
    let nq = kvh * group;
    let qb = 256usize;
    let scale = 1.0 / (d as f32).sqrt();
    let kernels: Vec<String> = std::env::var("LILY_QSA_TILE_KERNELS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| vec!["qsa_attn_rows_nax_h1".to_string()]);
    for base_pos in [8192usize, 32768] {
        let max_seq = base_pos + qb;
        let q = random(&mut rng, qb * nq * d, -1.0, 1.0);
        let k = random(&mut rng, kvh * max_seq * d, -1.0, 1.0);
        let v = random(&mut rng, kvh * max_seq * d, -1.0, 1.0);
        let nb = visible_blocks(base_pos, ratio);
        let hot: Vec<u32> = (0..300).map(|_| rng.gen_range(0..nb as u32)).collect();
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
        let tiles = SparseTileScratch::new(
            &ctx,
            qb,
            max_seq / ratio,
            k_max,
            kvh,
            ratio,
            1100 << 20,
        )
        .expect("tiles");
        let out = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        qsa_tile_union(&ctx, &pass, &t_sel, &t_n, &tiles, qb, k_max, ratio, base_pos)
            .expect("union");
        pass.commit_wait().expect("union");
        let (blocks, built) = tiles.union_stats().expect("stats");
        eprintln!(
            "base {base_pos}: union {:.0} blocks per tile ({:.2}x one selection)",
            blocks as f64 / built.max(1) as f64,
            (QSA_TILE_BQ * k_max) as f64 / (blocks as f64 / built.max(1) as f64)
        );
        crate::metal::profile::take();
        let rounds = 12;
        let mut walls: std::collections::HashMap<String, Vec<f64>> = Default::default();
        for round in 0..rounds {
            for k in 0..kernels.len() {
                let name = &kernels[(k + round) % kernels.len()];
                let pass = ctx.begin().expect("pass");
                dispatch_tiled_named(
                    &ctx, &pass, &t_q, &t_k, &t_v, &tiles, &out, qb, ratio, base_pos,
                    scale, name,
                )
                .expect("tiled attention");
                let t0 = std::time::Instant::now();
                pass.commit_wait().expect("commit");
                walls
                    .entry(name.clone())
                    .or_default()
                    .push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        if wall {
            for name in &kernels {
                let mut ms = walls.remove(name).unwrap_or_default();
                ms.sort_by(|a, b| a.total_cmp(b));
                let n = ms.len();
                eprintln!(
                    "base {base_pos} {name}: pass wall min {:.2} ms, median {:.2} ms over {n} passes",
                    ms[0],
                    ms[n / 2]
                );
            }
            continue;
        }
        let passes = crate::metal::profile::take();
        let mut names: Vec<String> = kernels.clone();
        if names.iter().any(|n| n.starts_with("qsa_attn_rows")) {
            names.push("qsa_tile_gather".to_string());
        }
        for name in &names {
            let mut ms: Vec<f64> = passes
                .iter()
                .flat_map(|p| p.kernels.iter())
                .filter(|s| s.name == name.as_str())
                .map(|s| s.gpu_secs * 1e3)
                .collect();
            ms.sort_by(|a, b| a.total_cmp(b));
            let n = ms.len();
            eprintln!(
                "base {base_pos} {name}: min {:.2} ms, median {:.2} ms over {n} dispatches",
                ms[0],
                ms[n / 2]
            );
        }
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
