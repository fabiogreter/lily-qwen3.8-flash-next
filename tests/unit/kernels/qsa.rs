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
    let splits = sparse_splits(k_max, ratio);
    let partials =
        Tensor::zeros(&ctx, &[qb * nq, splits, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[qb * nq, splits, 2], DType::F32).expect("stats");
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
    let splits = sparse_splits(k_max, ratio);
    assert_eq!(splits, 2, "the tail must fall into a second split");
    let partials =
        Tensor::zeros(&ctx, &[qb * nq, splits, d], DType::F32).expect("partials");
    let stats = Tensor::zeros(&ctx, &[qb * nq, splits, 2], DType::F32).expect("stats");
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
}
