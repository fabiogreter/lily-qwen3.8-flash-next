const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/gdn.metal"),
    "\n",
    include_str!("../../metal/gdn_test.metal")
);

/// One GDN decode step over all v-heads. `state` is fp32 `[H, 128, 128]` and
/// is updated in place; `a_log`/`w` in the checkpoint are f32, everything
/// else bf16. GVA: q/k carry `num_k_heads` heads; v-head `h` uses q/k head
/// `h / (H / num_k_heads)`.
#[allow(clippy::too_many_arguments)]
fn gdn_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    for (name, t, len) in [
        ("q", q, num_k_heads * dim),
        ("k", k, num_k_heads * dim),
        ("v", v, num_heads * dim),
        ("b", b, num_heads),
        ("dt_bias", dt_bias, num_heads),
        ("out", out, num_heads * dim),
    ] {
        ensure!(t.numel() == len, "{name} numel {} != {len}", t.numel());
        ensure!(t.dtype() == DType::BF16, "{name} must be BF16");
    }
    ensure!(
        a_log.numel() == num_heads && a_log.dtype() == DType::F32,
        "a_log must be F32 [H]"
    );
    ensure!(state.numel() == num_heads * dim * dim, "state must be [H, {dim}, {dim}]");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    let pipeline = ctx.pipeline("gdn_step", TEST_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.binding(),
            k.binding(),
            v.binding(),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&scale.to_ne_bytes(), &u32_bytes(vpk)],
        Grid::Threadgroups { groups: (num_heads, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

fn random_vec(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(lo..hi)).collect()
}

/// Checks blockwise and sequential CPU recurrence across block boundaries.
#[test]
fn gdn_blockwise_matches_sequential_reference() {
    let mut rng = StdRng::seed_from_u64(9);
    let (dim, tokens) = (32, 45);
    let q = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let k = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let v = random_vec(&mut rng, tokens * dim, -1.0, 1.0);
    let decay = random_vec(&mut rng, tokens, 0.05, 0.995);
    let beta = random_vec(&mut rng, tokens, 0.05, 0.95);
    let scale = 1.0 / (dim as f32).sqrt();

    let mut st_seq = vec![0.0f32; dim * dim];
    let mut out_seq = Vec::with_capacity(tokens * dim);
    for t in 0..tokens {
        out_seq.extend(cpu_ref::gdn_step(
            &q[t * dim..(t + 1) * dim],
            &k[t * dim..(t + 1) * dim],
            &v[t * dim..(t + 1) * dim],
            &mut st_seq,
            &decay[t..t + 1],
            &beta[t..t + 1],
            scale,
            1,
            1,
            dim,
            dim,
        ));
    }

    for block in [1, 16, 64] {
        let mut st = vec![0.0f32; dim * dim];
        let out = cpu_ref::gdn_blockwise(
            &q, &k, &v, &mut st, &decay, &beta, scale, dim, block,
        );
        cpu_ref::assert_close(&out, &out_seq, 2e-3, 2e-3);
        cpu_ref::assert_close(&st, &st_seq, 2e-3, 2e-3);
    }
}

#[test]
fn gdn_step_matches_cpu_over_sequence() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let steps = 64;

    // nk == nv, plus GVA layouts (v-heads a multiple of k-heads), including
    // the 1:3 ratio the Qwen3.8-Flash-Next checkpoint uses (16 key heads,
    // 48 value heads).
    for (seed, hk, h) in [(10, 4, 4), (20, 2, 4), (21, 4, 8), (22, 2, 6)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_state = Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state");
        let mut ref_state = vec![0.0f32; h * dim * dim];

        for step in 0..steps {
            let q = random_vec(&mut rng, hk * dim, -1.0, 1.0);
            let k = random_vec(&mut rng, hk * dim, -1.0, 1.0);
            let v = random_vec(&mut rng, h * dim, -1.0, 1.0);
            let a = random_vec(&mut rng, h, -1.0, 1.0);
            let b = random_vec(&mut rng, h, -1.0, 1.0);

            let tq = Tensor::from_f32_as_bf16(&ctx, &q, &[hk, dim]).expect("q");
            let tk = Tensor::from_f32_as_bf16(&ctx, &k, &[hk, dim]).expect("k");
            let tv = Tensor::from_f32_as_bf16(&ctx, &v, &[h, dim]).expect("v");
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[h]).expect("a");
            let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[h]).expect("b");
            let out = Tensor::zeros(&ctx, &[h, dim], DType::BF16).expect("out");

            let pass = ctx.begin().expect("pass");
            gdn_step(
                &ctx, &pass, &tq, &tk, &tv, &ta, &tb, &t_a_log, &t_dt_bias, &t_state,
                &out, scale, hk,
            )
            .expect("gdn_step");
            pass.commit_wait().expect("commit");

            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dt_bias),
                &cpu_ref::round_bf16(&b),
            );
            let expected = cpu_ref::gdn_step(
                &cpu_ref::round_bf16(&q),
                &cpu_ref::round_bf16(&k),
                &cpu_ref::round_bf16(&v),
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            );

            let actual = out.to_f32().expect("read out");
            cpu_ref::assert_close(&actual, &expected, 2e-2, 2e-2);
            // The state is fp32 on both sides; only op order differs, so
            // drift across a long sequence must stay tiny.
            if step == steps - 1 {
                let state_actual = t_state.to_f32().expect("read state");
                cpu_ref::assert_close(&state_actual, &ref_state, 5e-3, 5e-3);
            }
        }
    }
}

#[test]
fn gdn_prefill_matches_looped_step() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let m = 33;

    for (seed, hk, h) in [(13, 4, 4), (23, 2, 4)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
        let t_state = Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state");
        let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");

        // The default route is the register scan and needs staging.
        let st = staging_tensors(&ctx, m, hk, h);
        let staging = st.staging();
        let pass = ctx.begin().expect("pass");
        gdn_prefill(
            &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging, &t_state,
            &out, scale, hk,
        )
        .expect("gdn_prefill");
        pass.commit_wait().expect("commit");

        // Reference: the single-token step looped over the sequence.
        let (rqkv, ra, rb) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
        );
        let rdt = cpu_ref::round_bf16(&dt_bias);
        let mut ref_state = vec![0.0f32; h * dim * dim];
        let mut expected = Vec::new();
        for t in 0..m {
            let row = &rqkv[t * c..(t + 1) * c];
            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &ra[t * h..(t + 1) * h],
                &rdt,
                &rb[t * h..(t + 1) * h],
            );
            expected.extend(cpu_ref::gdn_step(
                &row[..hk * dim],
                &row[hk * dim..2 * hk * dim],
                &row[2 * hk * dim..],
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            ));
        }
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        cpu_ref::assert_close(
            &t_state.to_f32().expect("state"),
            &ref_state,
            5e-3,
            5e-3,
        );
    }
}

#[test]
fn gdn_prefill_nonzero_state_matches_looped_step() {
    // Cover nonzero-state resume across tail and multi-block lengths.
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();

    for (seed, hk, h, m) in
        [(31, 2, 4, 7), (33, 4, 4, 16), (35, 4, 4, 33), (39, 2, 4, 64)]
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);
        let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
        let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
        let t_state =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");

        // The default route is the register scan and needs staging.
        let st = staging_tensors(&ctx, m, hk, h);
        let staging = st.staging();
        let pass = ctx.begin().expect("pass");
        gdn_prefill(
            &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging, &t_state,
            &out, scale, hk,
        )
        .expect("gdn_prefill");
        pass.commit_wait().expect("commit");

        let (rqkv, ra, rb) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
        );
        let rdt = cpu_ref::round_bf16(&dt_bias);
        let mut ref_state = init_state.clone();
        let mut expected = Vec::new();
        for t in 0..m {
            let row = &rqkv[t * c..(t + 1) * c];
            let (decay, beta) = cpu_ref::gdn_gates(
                &a_log,
                &ra[t * h..(t + 1) * h],
                &rdt,
                &rb[t * h..(t + 1) * h],
            );
            expected.extend(cpu_ref::gdn_step(
                &row[..hk * dim],
                &row[hk * dim..2 * hk * dim],
                &row[2 * hk * dim..],
                &mut ref_state,
                &decay,
                &beta,
                scale,
                h,
                hk,
                dim,
                dim,
            ));
        }
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        cpu_ref::assert_close(
            &t_state.to_f32().expect("state"),
            &ref_state,
            5e-3,
            5e-3,
        );
    }
}

/// Fresh regscan staging tensors for one chunk shape (the default route
/// needs them, like the production per-chunk scratch views).
fn staging_tensors(
    ctx: &MetalContext,
    m: usize,
    hk: usize,
    h: usize,
) -> StagingTensors {
    let [w, u, p, g] = gdn_chunk_staging_shapes(m, h);
    StagingTensors {
        qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], DType::BF16)
            .expect("qk_norm"),
        decay: Tensor::zeros(ctx, &[m, h], DType::F32).expect("decay"),
        beta: Tensor::zeros(ctx, &[m, h], DType::F32).expect("beta"),
        w: Tensor::zeros(ctx, &w.0, w.1).expect("w"),
        u: Tensor::zeros(ctx, &u.0, u.1).expect("u"),
        p: Tensor::zeros(ctx, &p.0, p.1).expect("p"),
        g: Tensor::zeros(ctx, &g.0, g.1).expect("g"),
    }
}

/// Owned staging for both scans; `staging()` borrows it the way the model's
/// scratch does.
struct StagingTensors {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
    w: Tensor,
    u: Tensor,
    p: Tensor,
    g: Tensor,
}

impl StagingTensors {
    fn staging(&self) -> GdnRegscanStaging<'_> {
        GdnRegscanStaging {
            qk_norm: &self.qk_norm,
            decay: &self.decay,
            beta: &self.beta,
            chunk: Some(GdnChunkStaging {
                w: &self.w,
                u: &self.u,
                p: &self.p,
                g: &self.g,
            }),
        }
    }
}

#[test]
fn gdn_regscan_staging_stages_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(40);
    let (m, hk, h, dim) = (7, 2, 4, GDN_HEAD_DIM);
    let c = (2 * hk + h) * dim;
    let scale = 1.0 / (dim as f32).sqrt();
    let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
    let a = random_vec(&mut rng, m * h, -1.0, 1.0);
    let b = random_vec(&mut rng, m * h, -1.0, 1.0);
    let a_log = random_vec(&mut rng, h, -2.0, 0.5);
    let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);

    let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
    let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
    let t_dt_bias = Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
    let st = staging_tensors(&ctx, m, hk, h);

    let pass = ctx.begin().expect("pass");
    gdn_qk_l2norm(&ctx, &pass, &t_qkv, &st.qk_norm, scale, hk, h)
        .expect("gdn_qk_l2norm");
    gdn_gates(&ctx, &pass, &ta, &tb, &t_a_log, &t_dt_bias, &st.decay, &st.beta)
        .expect("gdn_gates");
    pass.commit_wait().expect("commit");

    let rounded_qkv = cpu_ref::round_bf16(&qkv);
    let mut expected_qk = vec![0.0f32; m * 2 * hk * dim];
    for token in 0..m {
        let row = &rounded_qkv[token * c..(token + 1) * c];
        let out_row = token * 2 * hk * dim;
        for head in 0..hk {
            let mut q = row[head * dim..(head + 1) * dim].to_vec();
            let mut k = row[(hk + head) * dim..(hk + head + 1) * dim].to_vec();
            cpu_ref::l2_normalize(&mut q);
            cpu_ref::l2_normalize(&mut k);
            for value in &mut q {
                *value *= scale;
            }
            expected_qk[out_row + head * dim..out_row + (head + 1) * dim]
                .copy_from_slice(&q);
            expected_qk[out_row + (hk + head) * dim..out_row + (hk + head + 1) * dim]
                .copy_from_slice(&k);
        }
    }
    let expected_qk = cpu_ref::round_bf16(&expected_qk);
    cpu_ref::assert_close(
        &st.qk_norm.to_f32().expect("read qk_norm"),
        &expected_qk,
        2e-2,
        2e-2,
    );

    let (rounded_a, rounded_b, rounded_dt_bias) = (
        cpu_ref::round_bf16(&a),
        cpu_ref::round_bf16(&b),
        cpu_ref::round_bf16(&dt_bias),
    );
    let mut expected_decay = Vec::with_capacity(m * h);
    let mut expected_beta = Vec::with_capacity(m * h);
    for token in 0..m {
        let (token_decay, token_beta) = cpu_ref::gdn_gates(
            &a_log,
            &rounded_a[token * h..(token + 1) * h],
            &rounded_dt_bias,
            &rounded_b[token * h..(token + 1) * h],
        );
        expected_decay.extend(token_decay);
        expected_beta.extend(token_beta);
    }
    cpu_ref::assert_close(
        &st.decay.to_f32().expect("read decay"),
        &expected_decay,
        1e-5,
        1e-5,
    );
    cpu_ref::assert_close(
        &st.beta.to_f32().expect("read beta"),
        &expected_beta,
        1e-5,
        1e-5,
    );
}

/// One regscan prefill dispatch over host data; returns (out, state).
/// Staging tensors are allocated per call, like the production per-chunk
/// scratch views.
#[allow(clippy::too_many_arguments)]
fn run_regscan(
    ctx: &MetalContext,
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    scale: f32,
    hk: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim = GDN_HEAD_DIM;
    let m = a.len() / h;
    let c = (2 * hk + h) * dim;
    let t_qkv = Tensor::from_f32_as_bf16(ctx, qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(ctx, a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(ctx, b, &[m, h]).expect("b");
    let st = staging_tensors(ctx, m, hk, h);
    let staging = st.staging();
    let out = Tensor::zeros(ctx, &[m, h, dim], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    // The token-serial scan by name: the default route takes the chunked
    // scan from `GDN_CHUNK_MIN_ROWS` rows.
    gdn_prefill_scan_named(
        ctx,
        &pass,
        &t_qkv,
        &ta,
        &tb,
        a_log,
        dt_bias,
        &staging,
        state,
        &out,
        scale,
        hk,
        None,
        "gdn_prefill_regscan",
    )
    .expect("gdn_prefill");
    pass.commit_wait().expect("commit");
    (out.to_f32().expect("read out"), state.to_f32().expect("read state"))
}

#[test]
fn gdn_prefill_regscan_rejects_bf16_state() {
    let ctx = MetalContext::new().expect("metal context");
    let (m, hk, h, dim) = (1, 1, 1, GDN_HEAD_DIM);
    let qkv = Tensor::zeros(&ctx, &[m, (2 * hk + h) * dim], DType::BF16).expect("qkv");
    let a = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("a");
    let b = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("b");
    let a_log = Tensor::zeros(&ctx, &[h], DType::F32).expect("a_log");
    let dt_bias = Tensor::zeros(&ctx, &[h], DType::BF16).expect("dt_bias");
    let st = staging_tensors(&ctx, m, hk, h);
    let staging = st.staging();
    let state = Tensor::zeros(&ctx, &[h, dim, dim], DType::BF16).expect("state");
    let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");

    let err = gdn_prefill(
        &ctx, &pass, &qkv, &a, &b, &a_log, &dt_bias, &staging, &state, &out, 1.0, hk,
    )
    .expect_err("BF16 state must be rejected before dispatch");
    assert!(err.to_string().contains("state must be F32"), "unexpected error: {err:#}");
}

#[test]
fn gdn_fused_decode_f32_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let (steps, hk, h, dim) = (16usize, 1usize, 2usize, GDN_HEAD_DIM);
    let scale = 1.0 / (dim as f32).sqrt();
    let eps = 1e-6f32;
    let mut rng = StdRng::seed_from_u64(0xf32dec0de);
    let a_log = random_vec(&mut rng, h, -2.0, 0.5);
    let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
    let norm_w = random_vec(&mut rng, dim, 0.5, 1.5);
    let init_state = random_vec(&mut rng, h * dim * dim, -0.25, 0.25);
    let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
    let t_dt_bias = Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
    let t_norm_w = Tensor::from_f32(&ctx, &norm_w, &[dim]).expect("norm_w");
    let t_state = Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
    let mut ref_state = init_state;

    for _ in 0..steps {
        let qkv = random_vec(&mut rng, (2 * hk + h) * dim, -1.0, 1.0);
        let a = random_vec(&mut rng, h, -1.0, 1.0);
        let b = random_vec(&mut rng, h, -1.0, 1.0);
        let z = random_vec(&mut rng, h * dim, -1.0, 1.0);
        let t_qkv =
            Tensor::from_f32_as_bf16(&ctx, &qkv, &[2 * hk + h, dim]).expect("qkv");
        let t_a = Tensor::from_f32_as_bf16(&ctx, &a, &[h]).expect("a");
        let t_b = Tensor::from_f32_as_bf16(&ctx, &b, &[h]).expect("b");
        let t_z = Tensor::from_f32_as_bf16(&ctx, &z, &[h, dim]).expect("z");
        let out = Tensor::zeros(&ctx, &[h, dim], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        gdn_step_gated_fused(
            &ctx,
            &pass,
            &t_qkv,
            &t_a,
            &t_b,
            &t_a_log,
            &t_dt_bias,
            &t_state,
            &t_z,
            &t_norm_w,
            &out,
            scale,
            hk,
            eps,
            GdnGate::Silu,
        )
        .expect("F32 fused decode");
        pass.commit_wait().expect("commit");

        let (rqkv, ra, rb, rz, rdt) = (
            cpu_ref::round_bf16(&qkv),
            cpu_ref::round_bf16(&a),
            cpu_ref::round_bf16(&b),
            cpu_ref::round_bf16(&z),
            cpu_ref::round_bf16(&dt_bias),
        );
        let (decay, beta) = cpu_ref::gdn_gates(&a_log, &ra, &rdt, &rb);
        let raw = cpu_ref::gdn_step(
            &rqkv[..hk * dim],
            &rqkv[hk * dim..2 * hk * dim],
            &rqkv[2 * hk * dim..],
            &mut ref_state,
            &decay,
            &beta,
            scale,
            h,
            hk,
            dim,
            dim,
        );
        let mut expected = vec![0.0f32; h * dim];
        for head in 0..h {
            let row = &raw[head * dim..(head + 1) * dim];
            let inv_rms = 1.0
                / (row.iter().map(|value| value * value).sum::<f32>() / dim as f32
                    + eps)
                    .sqrt();
            for d in 0..dim {
                let gate = rz[head * dim + d];
                expected[head * dim + d] =
                    norm_w[d] * row[d] * inv_rms * gate / (1.0 + (-gate).exp());
            }
        }
        cpu_ref::assert_close(&out.to_f32().expect("out"), &expected, 4e-2, 4e-2);
    }
    cpu_ref::assert_close(&t_state.to_f32().expect("state"), &ref_state, 2e-5, 2e-5);
}

/// Correctness matrix: the register scan vs the per-token cpu_ref from a
/// NONZERO initial state, over
/// M in {1, 7, 16, 33, 200, 2048} x vpk in {1, 2}, at the existing
/// tolerances. The full head geometry uses a shorter M list; small-head cells
/// cover the complete sequence-length matrix.
#[test]
fn gdn_prefill_regscan_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let m_full: &[usize] = &[1, 7, 16, 33, 200, 2048];
    let m_shape: &[usize] = &[33, 200];
    for (seed, hk, h, ms) in [
        (41u64, 4usize, 4usize, m_full),
        (43, 2, 4, m_full),
        (45, 16, 16, m_shape),
        (47, 16, 32, m_shape),
    ] {
        for &m in ms {
            let mut rng = StdRng::seed_from_u64(seed + m as u64);
            let c = (2 * hk + h) * dim;

            let a_log = random_vec(&mut rng, h, -2.0, 0.5);
            let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
            let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
            let a = random_vec(&mut rng, m * h, -1.0, 1.0);
            let b = random_vec(&mut rng, m * h, -1.0, 1.0);
            let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

            let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
            let t_dt_bias =
                Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
            let t_state =
                Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
            let (out_reg, state_reg) = run_regscan(
                &ctx, &qkv, &a, &b, &t_a_log, &t_dt_bias, &t_state, scale, hk, h,
            );

            // Per-token cpu_ref reference over the same bf16-rounded
            // inputs.
            let (rqkv, ra, rb) = (
                cpu_ref::round_bf16(&qkv),
                cpu_ref::round_bf16(&a),
                cpu_ref::round_bf16(&b),
            );
            let rdt = cpu_ref::round_bf16(&dt_bias);
            let mut ref_state = init_state.clone();
            let mut expected = Vec::new();
            for t in 0..m {
                let row = &rqkv[t * c..(t + 1) * c];
                let (decay, beta) = cpu_ref::gdn_gates(
                    &a_log,
                    &ra[t * h..(t + 1) * h],
                    &rdt,
                    &rb[t * h..(t + 1) * h],
                );
                expected.extend(cpu_ref::gdn_step(
                    &row[..hk * dim],
                    &row[hk * dim..2 * hk * dim],
                    &row[2 * hk * dim..],
                    &mut ref_state,
                    &decay,
                    &beta,
                    scale,
                    h,
                    hk,
                    dim,
                    dim,
                ));
            }
            cpu_ref::assert_close(&out_reg, &expected, 2e-2, 2e-2);
            cpu_ref::assert_close(&state_reg, &ref_state, 5e-3, 5e-3);
        }
    }
}

/// The chunked scan against the per-token CPU reference and against the
/// serial scan, at the model's head shape and at small ones, with chunk
/// counts of one, several and a ragged last chunk, from a nonzero state.
/// It reads the state and the pseudo values through bf16 copies, so the
/// tolerance is looser than the serial scan's.
#[test]
fn gdn_prefill_chunked_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    for (seed, hk, h, ms) in [
        (61u64, 4usize, 4usize, &[64usize, 100, 200, 333][..]),
        (63, 2, 6, &[128, 250][..]),
        (65, 16, 48, &[200, 1024][..]),
    ] {
        for &m in ms {
            let mut rng = StdRng::seed_from_u64(seed + m as u64);
            let c = (2 * hk + h) * dim;
            let a_log = random_vec(&mut rng, h, -2.0, 0.5);
            let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
            let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
            let a = random_vec(&mut rng, m * h, -1.0, 1.0);
            let b = random_vec(&mut rng, m * h, -1.0, 1.0);
            let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

            let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
            let t_dt_bias =
                Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
            // The inputs and the normalized rows sit in front of 64 rows of
            // NaN: a kernel reading past its rows shows in the results.
            let guarded = |data: &[f32], cols: usize| {
                let mut padded = data.to_vec();
                padded.extend(std::iter::repeat_n(f32::NAN, 64 * cols));
                Tensor::from_f32_as_bf16(&ctx, &padded, &[m + 64, cols])
                    .expect("padded")
                    .view(0, &[m, cols])
                    .expect("view")
            };
            let t_qkv = guarded(&qkv, c);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
            let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
            let st = staging_tensors(&ctx, m, hk, h);
            let qk_guard = guarded(&vec![0.0f32; m * 2 * hk * dim], 2 * hk * dim);
            let run = |name: &str| {
                let staging = GdnRegscanStaging { qk_norm: &qk_guard, ..st.staging() };
                let state =
                    Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
                let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
                let pass = ctx.begin().expect("pass");
                gdn_prefill_scan_named(
                    &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging,
                    &state, &out, scale, hk, None, name,
                )
                .expect("scan");
                pass.commit_wait().expect("commit");
                (out.to_f32().expect("out"), state.to_f32().expect("state"))
            };
            let (out_chunk, state_chunk) = run(GDN_CHUNK_SCAN);
            let (out_reg, state_reg) = run("gdn_prefill_regscan");

            let (rqkv, ra, rb) = (
                cpu_ref::round_bf16(&qkv),
                cpu_ref::round_bf16(&a),
                cpu_ref::round_bf16(&b),
            );
            let rdt = cpu_ref::round_bf16(&dt_bias);
            let mut ref_state = init_state.clone();
            let mut expected = Vec::new();
            for t in 0..m {
                let row = &rqkv[t * c..(t + 1) * c];
                let (decay, beta) = cpu_ref::gdn_gates(
                    &a_log,
                    &ra[t * h..(t + 1) * h],
                    &rdt,
                    &rb[t * h..(t + 1) * h],
                );
                expected.extend(cpu_ref::gdn_step(
                    &row[..hk * dim],
                    &row[hk * dim..2 * hk * dim],
                    &row[2 * hk * dim..],
                    &mut ref_state,
                    &decay,
                    &beta,
                    scale,
                    h,
                    hk,
                    dim,
                    dim,
                ));
            }
            let max_err = |x: &[f32], y: &[f32]| {
                x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
            };
            let scale_of = |x: &[f32]| x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "hk {hk} h {h} m {m}: out |chunk - cpu| {:.4}, |serial - cpu| {:.4} (scale {:.2}); state |chunk - cpu| {:.5}, |serial - cpu| {:.5} (scale {:.2})",
                max_err(&out_chunk, &expected),
                max_err(&out_reg, &expected),
                scale_of(&expected),
                max_err(&state_chunk, &ref_state),
                max_err(&state_reg, &ref_state),
                scale_of(&ref_state),
            );
            assert!(out_chunk.iter().all(|x| x.is_finite()), "non-finite output");
            cpu_ref::assert_close(&out_chunk, &expected, 5e-2, 5e-2);
            cpu_ref::assert_close(&state_chunk, &ref_state, 2e-2, 2e-2);
        }
    }
}

/// Chunk-boundary state carry through the regscan kernel: splitting a
/// sequence into chunks only round-trips the fp32 state through device
/// memory (which the kernel does at entry/exit anyway), so outputs and
/// the final state must be BIT-identical to the single-shot run.
#[test]
fn gdn_prefill_regscan_chunk_carry_bit_identical() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let scale = 1.0 / (dim as f32).sqrt();
    let (m, chunk) = (200usize, 64usize);

    for (seed, hk, h) in [(51u64, 4usize, 4usize), (53, 2, 4)] {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = (2 * hk + h) * dim;

        let a_log = random_vec(&mut rng, h, -2.0, 0.5);
        let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
        let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
        let a = random_vec(&mut rng, m * h, -1.0, 1.0);
        let b = random_vec(&mut rng, m * h, -1.0, 1.0);
        let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

        let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
        let t_dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");

        let state_full =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let (out_full, final_full) = run_regscan(
            &ctx,
            &qkv,
            &a,
            &b,
            &t_a_log,
            &t_dt_bias,
            &state_full,
            scale,
            hk,
            h,
        );

        let state_chunked =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let mut out_chunked = Vec::new();
        let mut final_chunked = Vec::new();
        for t0 in (0..m).step_by(chunk) {
            let t1 = (t0 + chunk).min(m);
            let (out, state) = run_regscan(
                &ctx,
                &qkv[t0 * c..t1 * c],
                &a[t0 * h..t1 * h],
                &b[t0 * h..t1 * h],
                &t_a_log,
                &t_dt_bias,
                &state_chunked,
                scale,
                hk,
                h,
            );
            out_chunked.extend(out);
            final_chunked = state;
        }
        assert_eq!(out_full, out_chunked, "chunked outputs diverged");
        assert_eq!(final_full, final_chunked, "chunked final state diverged");
    }
}

/// Chunk sequences covering single tokens, tile boundaries, a full
/// prefill chunk, and window carry across chunks including trailing
/// chunks shorter than the window (M < KD-1).
fn conv1d_chunk_cases() -> Vec<Vec<usize>> {
    let tile = CONV1D_PREFILL_TILE;
    vec![
        vec![1],
        vec![3],
        vec![4],
        vec![16],
        vec![tile - 1],
        vec![tile],
        vec![tile + 1],
        vec![2048],
        vec![tile + 1, 2],
        vec![2, 2],
        vec![5, tile],
    ]
}

#[test]
fn conv1d_prefill_matches_looped_step() {
    let ctx = MetalContext::new().expect("metal context");

    for kd in [4usize, 9] {
        for chunks in conv1d_chunk_cases() {
            let mut rng = StdRng::seed_from_u64(14 + kd as u64);
            let c = 96;
            let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
            let win0 = random_vec(&mut rng, c * (kd - 1), -1.0, 1.0);
            // The inactive buffer starts as garbage: the final tile must
            // overwrite every slot of it.
            let junk = random_vec(&mut rng, c * (kd - 1), -9.0, 9.0);
            let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
            let windows = [
                Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("window a"),
                Tensor::from_f32_as_bf16(&ctx, &junk, &[c, kd - 1]).expect("window b"),
            ];
            let mut slot = 0;
            let rw = cpu_ref::round_bf16(&w);
            let mut ref_window = cpu_ref::round_bf16(&win0);

            for m in chunks {
                let x = random_vec(&mut rng, m * c, -1.0, 1.0);
                let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
                let out = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("out");

                let pass = ctx.begin().expect("pass");
                conv1d_prefill(
                    &ctx,
                    &pass,
                    &windows[slot],
                    &windows[1 - slot],
                    &tx,
                    &tw,
                    &out,
                )
                .expect("conv1d_prefill");
                pass.commit_wait().expect("commit");
                slot = 1 - slot;

                let rx = cpu_ref::round_bf16(&x);
                let mut expected = Vec::new();
                for t in 0..m {
                    expected.extend(cpu_ref::conv1d_step(
                        &mut ref_window,
                        &rx[t * c..(t + 1) * c],
                        &rw,
                        c,
                        kd,
                    ));
                }
                cpu_ref::assert_close(
                    &out.to_f32().expect("read"),
                    &expected,
                    2e-2,
                    2e-2,
                );
                cpu_ref::assert_close(
                    &windows[slot].to_f32().expect("window"),
                    &ref_window,
                    2e-2,
                    2e-2,
                );
            }
        }
    }
}

/// Serial prefill conv oracle; tiled output and carried window must match bits.
const CONV1D_SERIAL_REF: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv1d_prefill_serial_ref(device bfloat*       window [[buffer(0)]],
                                      device const bfloat* x      [[buffer(1)]],
                                      device const bfloat* w      [[buffer(2)]],
                                      device bfloat*       out    [[buffer(3)]],
                                      constant uint&       C      [[buffer(4)]],
                                      constant uint&       KD     [[buffer(5)]],
                                      constant uint&       M      [[buffer(6)]],
                                      uint c [[thread_position_in_grid]]) {
    uint taps = KD - 1;
    float win[8];
    for (uint t = 0; t < taps; ++t) {
        win[t] = float(window[c * taps + t]);
    }
    for (uint m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (uint t = 0; t < taps; ++t) {
            acc += win[t] * float(w[t * C + c]);
        }
        float xc = float(x[(ulong)m * C + c]);
        acc += xc * float(w[taps * C + c]);
        out[(ulong)m * C + c] = bfloat(acc / (1.0f + exp(-acc)));
        for (uint t = 0; t + 1 < taps; ++t) {
            win[t] = win[t + 1];
        }
        win[taps - 1] = xc;
    }
    for (uint t = 0; t < taps; ++t) {
        window[c * taps + t] = bfloat(win[t]);
    }
}
"#;

#[test]
fn conv1d_prefill_bitwise_matches_serial_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let tile = CONV1D_PREFILL_TILE;
    let ref_pipeline = ctx
        .pipeline("conv1d_prefill_serial_ref", CONV1D_SERIAL_REF, MslVersion::V3_1)
        .expect("ref pipeline");

    for (seed, c, kd, chunks) in [
        (31u64, 96usize, 4usize, vec![2048]),
        (32, 96, 4, vec![tile + 1, 2]),
        (33, 300, 4, vec![tile, 1, 3]),
        (34, 96, 2, vec![tile - 1, 16]),
        (35, 96, 9, vec![2, tile + 1, 4]),
    ] {
        let mut rng = StdRng::seed_from_u64(seed);
        let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
        let win0 = random_vec(&mut rng, c * (kd - 1), -1.0, 1.0);
        let junk = random_vec(&mut rng, c * (kd - 1), -9.0, 9.0);
        let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
        let win_new = [
            Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("win a"),
            Tensor::from_f32_as_bf16(&ctx, &junk, &[c, kd - 1]).expect("win b"),
        ];
        let mut slot = 0;
        let win_ref =
            Tensor::from_f32_as_bf16(&ctx, &win0, &[c, kd - 1]).expect("win ref");

        for m in chunks {
            let x = random_vec(&mut rng, m * c, -1.0, 1.0);
            let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
            let out_new = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("out");
            let out_ref = Tensor::zeros(&ctx, &[m, c], DType::BF16).expect("ref");

            let pass = ctx.begin().expect("pass");
            conv1d_prefill(
                &ctx,
                &pass,
                &win_new[slot],
                &win_new[1 - slot],
                &tx,
                &tw,
                &out_new,
            )
            .expect("conv1d_prefill");
            pass.dispatch_at(
                &ref_pipeline,
                &[win_ref.binding(), tx.binding(), tw.binding(), out_ref.binding()],
                &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
                Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
            )
            .expect("serial ref dispatch");
            pass.commit_wait().expect("commit");
            slot = 1 - slot;

            assert_eq!(
                out_new.to_f32().expect("read out"),
                out_ref.to_f32().expect("read ref out"),
                "output diverged from the serial kernel (c={c} kd={kd} m={m})"
            );
            // The freshly written (now active) slot must carry exactly
            // the serial kernel's in-place window.
            assert_eq!(
                win_new[slot].to_f32().expect("read window"),
                win_ref.to_f32().expect("read ref window"),
                "window diverged from the serial kernel (c={c} kd={kd} m={m})"
            );
        }
    }
}

#[test]
fn conv1d_step_matches_cpu_over_steps() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(11);
    let (c, kd) = (96, 4);

    let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
    let tw = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
    let t_window = Tensor::zeros(&ctx, &[c, kd - 1], DType::BF16).expect("window");
    let mut ref_window = vec![0.0f32; c * (kd - 1)];

    for _ in 0..8 {
        let x = random_vec(&mut rng, c, -1.0, 1.0);
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[c]).expect("x");
        let out = Tensor::zeros(&ctx, &[c], DType::BF16).expect("out");

        let pass = ctx.begin().expect("pass");
        conv1d_step(&ctx, &pass, &t_window, &tx, &tw, &out).expect("conv1d_step");
        pass.commit_wait().expect("commit");

        let expected = cpu_ref::conv1d_step(
            &mut ref_window,
            &cpu_ref::round_bf16(&x),
            &cpu_ref::round_bf16(&w),
            c,
            kd,
        );
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn gated_rmsnorm_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(12);
    let (rows, d) = (16, 128);
    let eps = 1e-6;

    let x = random_vec(&mut rng, rows * d, -2.0, 2.0);
    let gate = random_vec(&mut rng, rows * d, -2.0, 2.0);
    let w = random_vec(&mut rng, d, 0.5, 1.5);

    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[rows, d]).expect("x");
    let tg = Tensor::from_f32_as_bf16(&ctx, &gate, &[rows, d]).expect("gate");
    let tw = Tensor::from_f32(&ctx, &w, &[d]).expect("w");
    let out = Tensor::zeros(&ctx, &[rows, d], DType::BF16).expect("out");

    let pass = ctx.begin().expect("pass");
    gated_rmsnorm(&ctx, &pass, &tx, &tg, &tw, &out, eps, GdnGate::Silu)
        .expect("gated_rmsnorm");
    pass.commit_wait().expect("commit");

    let expected = cpu_ref::gated_rmsnorm(
        &cpu_ref::round_bf16(&x),
        &cpu_ref::round_bf16(&gate),
        &w,
        d,
        eps,
    );
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

#[test]
fn gated_rmsnorm_sigmoid_gate_matches_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(77);
    let (rows, d) = (6, 128);
    let eps = 1e-6;
    let x = random_vec(&mut rng, rows * d, -2.0, 2.0);
    let gate = random_vec(&mut rng, rows * d, -3.0, 3.0);
    let w = random_vec(&mut rng, d, 0.5, 1.5);
    let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[rows, d]).expect("x");
    let tg = Tensor::from_f32_as_bf16(&ctx, &gate, &[rows, d]).expect("gate");
    let tw = Tensor::from_f32(&ctx, &w, &[d]).expect("w");
    let out = Tensor::zeros(&ctx, &[rows, d], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    gated_rmsnorm(&ctx, &pass, &tx, &tg, &tw, &out, eps, GdnGate::Sigmoid)
        .expect("gated");
    pass.commit_wait().expect("commit");
    let expected = cpu_ref::gated_rmsnorm_with(
        &cpu_ref::round_bf16(&x),
        &cpu_ref::round_bf16(&gate),
        &w,
        d,
        eps,
        cpu_ref::sigmoid,
    );
    cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
}

/// The intermediate states the regscan records after each token are exactly
/// the states a scan over that many tokens would leave behind (the register
/// values are the same; only the store differs), so rolling back to one of
/// them is bit-identical to never having fed the rejected tokens.
#[test]
fn gdn_prefill_regscan_mid_states_match_prefix_scans() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let (hk, h, m) = (2usize, 4usize, 4usize);
    let scale = 1.0 / (dim as f32).sqrt();
    let mut rng = StdRng::seed_from_u64(91);
    let c = (2 * hk + h) * dim;
    let a_log = Tensor::from_f32(&ctx, &random_vec(&mut rng, h, -2.0, 0.5), &[h])
        .expect("a_log");
    let dt_bias =
        Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, h, -0.5, 0.5), &[h])
            .expect("dt_bias");
    let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
    let a = random_vec(&mut rng, m * h, -1.0, 1.0);
    let b = random_vec(&mut rng, m * h, -1.0, 1.0);
    let init_state = random_vec(&mut rng, h * dim * dim, -0.5, 0.5);

    // Full scan with mid capture of the first m-1 tokens.
    let state = Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
    let mid = Tensor::zeros(&ctx, &[m - 1, h, dim, dim], DType::F32).expect("mid");
    let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
    let st = staging_tensors(&ctx, m, hk, h);
    let staging = st.staging();
    let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
    let pass = ctx.begin().expect("pass");
    gdn_prefill_mid(
        &ctx,
        &pass,
        &t_qkv,
        &ta,
        &tb,
        &a_log,
        &dt_bias,
        &staging,
        &state,
        &out,
        scale,
        hk,
        Some(&mid),
    )
    .expect("gdn_prefill_mid");
    pass.commit_wait().expect("commit");
    let mid_all = mid.to_f32().expect("read mid");
    let per = h * dim * dim;

    for n in 1..m {
        let prefix_state =
            Tensor::from_f32(&ctx, &init_state, &[h, dim, dim]).expect("state");
        let (_, got) = run_regscan(
            &ctx,
            &qkv[..n * c],
            &a[..n * h],
            &b[..n * h],
            &a_log,
            &dt_bias,
            &prefix_state,
            scale,
            hk,
            h,
        );
        assert_eq!(got, mid_all[(n - 1) * per..n * per], "mid state after {n} tokens");
    }
    // A caller that passes more mid slots than tokens is rejected.
    let too_many = Tensor::zeros(&ctx, &[m + 1, h, dim, dim], DType::F32).expect("mid");
    let pass = ctx.begin().expect("pass");
    assert!(
        gdn_prefill_mid(
            &ctx,
            &pass,
            &t_qkv,
            &ta,
            &tb,
            &a_log,
            &dt_bias,
            &staging,
            &state,
            &out,
            scale,
            hk,
            Some(&too_many)
        )
        .is_err()
    );
}

/// Rolling a conv window back to `n` rows equals running the conv over just
/// those rows, for GDN (S = KD-1) and PLE-like (S = 9) window lengths.
#[test]
fn conv_window_rollback_matches_prefix_window() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(93);
    for s in [3usize, 9] {
        let (c, m) = (40usize, 5usize);
        let kd = s + 1; // a plain conv with KD-1 = s taps keeps an S-long window (GPU cross-check only for KD <= 9)
        let win0 = random_vec(&mut rng, c * s, -1.0, 1.0);
        let x = random_vec(&mut rng, m * c, -1.0, 1.0);
        let w = random_vec(&mut rng, kd * c, -1.0, 1.0);
        let t_win0 = Tensor::from_f32_as_bf16(&ctx, &win0, &[c, s]).expect("win0");
        let t_x = Tensor::from_f32_as_bf16(&ctx, &x, &[m, c]).expect("x");
        let t_w = Tensor::from_f32_as_bf16(&ctx, &w, &[kd, c]).expect("w");
        let (r_win0, r_x) =
            (t_win0.to_f32().expect("read"), t_x.to_f32().expect("read"));
        for n in 0..=m {
            // Host reference: the last S entries of win0 ++ x[..n] per channel.
            let host: Vec<f32> = (0..c)
                .flat_map(|ch| {
                    let seq: Vec<f32> = (0..s)
                        .map(|i| r_win0[ch * s + i])
                        .chain((0..n).map(|r| r_x[r * c + ch]))
                        .collect();
                    seq[seq.len() - s..].to_vec()
                })
                .collect();
            let expected = if n == 0 || kd > 9 {
                host
            } else {
                let win_out =
                    Tensor::zeros(&ctx, &[c, s], DType::BF16).expect("win_out");
                let out = Tensor::zeros(&ctx, &[n, c], DType::BF16).expect("out");
                let xn = t_x.view(0, &[n, c]).expect("prefix");
                let pass = ctx.begin().expect("pass");
                conv1d_prefill(&ctx, &pass, &t_win0, &win_out, &xn, &t_w, &out)
                    .expect("conv1d_prefill");
                pass.commit_wait().expect("commit");
                let gpu = win_out.to_f32().expect("read");
                assert_eq!(
                    gpu, host,
                    "host window reference vs conv1d_prefill, S={s} n={n}"
                );
                gpu
            };
            let rolled = Tensor::zeros(&ctx, &[c, s], DType::BF16).expect("rolled");
            let pass = ctx.begin().expect("pass");
            conv_window_rollback(&ctx, &pass, &t_win0, &t_x, &rolled, n)
                .expect("rollback");
            pass.commit_wait().expect("commit");
            assert_eq!(rolled.to_f32().expect("read"), expected, "S={s} n={n}");
        }
    }
}

/// GPU time of the two recurrence kernels at Qwen3.8-Flash-Next's shape over
/// 36 layers, for small token counts: the register scan (prefill) against
/// the coalesced single-token step kernel looped per token.
#[test]
#[ignore = "timing; run with --nocapture"]
fn gdn_small_m_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let dim = GDN_HEAD_DIM;
    let (hk, h, layers) = (16usize, 48usize, 36usize);
    let scale = 1.0 / (dim as f32).sqrt();
    let mut rng = StdRng::seed_from_u64(5);
    let c = (2 * hk + h) * dim;
    let a_log = Tensor::from_f32(&ctx, &random_vec(&mut rng, h, -2.0, 0.5), &[h])
        .expect("a_log");
    let dt_bias =
        Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, h, -0.5, 0.5), &[h])
            .expect("dt_bias");
    let norm_w = Tensor::from_f32(&ctx, &vec![1.0; dim], &[dim]).expect("norm_w");
    let states: Vec<Tensor> = (0..layers)
        .map(|_| {
            Tensor::from_f32(
                &ctx,
                &random_vec(&mut rng, h * dim * dim, -0.5, 0.5),
                &[h, dim, dim],
            )
            .expect("state")
        })
        .collect();
    for m in [1usize, 2, 4] {
        let qkv = Tensor::from_f32_as_bf16(
            &ctx,
            &random_vec(&mut rng, m * c, -1.0, 1.0),
            &[m, c],
        )
        .expect("qkv");
        let a = Tensor::from_f32_as_bf16(
            &ctx,
            &random_vec(&mut rng, m * h, -1.0, 1.0),
            &[m, h],
        )
        .expect("a");
        let b = Tensor::from_f32_as_bf16(
            &ctx,
            &random_vec(&mut rng, m * h, -1.0, 1.0),
            &[m, h],
        )
        .expect("b");
        let z = Tensor::from_f32_as_bf16(
            &ctx,
            &random_vec(&mut rng, m * h * dim, -1.0, 1.0),
            &[m, h * dim],
        )
        .expect("z");
        let st = staging_tensors(&ctx, m, hk, h);
        let staging = st.staging();
        let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
        let gated = Tensor::zeros(&ctx, &[h * dim], DType::BF16).expect("gated");
        for variant in ["regscan", "step-loop"] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let pass = ctx.begin_concurrent().expect("pass");
                for state in &states {
                    if variant == "regscan" {
                        gdn_prefill(
                            &ctx, &pass, &qkv, &a, &b, &a_log, &dt_bias, &staging,
                            state, &out, scale, hk,
                        )
                        .expect("regscan");
                    } else {
                        for t in 0..m {
                            let row = |x: &Tensor, w: usize| {
                                x.view(t * w, &[w]).expect("row")
                            };
                            gdn_step_gated_fused(
                                &ctx,
                                &pass,
                                &row(&qkv, c),
                                &row(&a, h),
                                &row(&b, h),
                                &a_log,
                                &dt_bias,
                                state,
                                &row(&z, h * dim),
                                &norm_w,
                                &gated,
                                scale,
                                hk,
                                1e-6,
                                GdnGate::Sigmoid,
                            )
                            .expect("step");
                            pass.level_barrier(&[state]).expect("barrier");
                        }
                    }
                    pass.level_barrier(&[state]).expect("barrier");
                }
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                best = best.min(t.gpu_end_secs - t.gpu_start_secs);
            }
            eprintln!("gdn {layers} layers m={m} {variant}: {:.2} ms", best * 1e3);
        }
    }
}

/// Per-dispatch GPU time of the prefill scan at the model's chunk shape
/// (4096 tokens, 48 value heads over 16 key heads) on the profile
/// transport, for the kernels `LILY_GDN_SCAN_KERNELS` names (comma
/// separated, rotated per round so they share the GPU's conditions; the
/// shipped scan and the single-column one by default): minimum and median
/// over the dispatches.
#[test]
#[ignore = "timing only"]
fn gdn_prefill_scan_timing() {
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    let mut rng = StdRng::seed_from_u64(61);
    let (dim, hk, h, m) = (GDN_HEAD_DIM, 16usize, 48usize, 4096usize);
    let scale = 1.0 / (dim as f32).sqrt();
    let c = (2 * hk + h) * dim;
    let a_log = random_vec(&mut rng, h, -2.0, 0.5);
    let dt_bias = random_vec(&mut rng, h, -0.5, 0.5);
    let qkv = random_vec(&mut rng, m * c, -1.0, 1.0);
    let a = random_vec(&mut rng, m * h, -1.0, 1.0);
    let b = random_vec(&mut rng, m * h, -1.0, 1.0);
    let t_a_log = Tensor::from_f32(&ctx, &a_log, &[h]).expect("a_log");
    let t_dt_bias = Tensor::from_f32_as_bf16(&ctx, &dt_bias, &[h]).expect("dt_bias");
    let t_qkv = Tensor::from_f32_as_bf16(&ctx, &qkv, &[m, c]).expect("qkv");
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, h]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[m, h]).expect("b");
    let st = staging_tensors(&ctx, m, hk, h);
    let staging = st.staging();
    let state = Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state");
    let out = Tensor::zeros(&ctx, &[m, h, dim], DType::BF16).expect("out");
    let names: Vec<String> = std::env::var("LILY_GDN_SCAN_KERNELS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| {
            vec![
                "gdn_prefill_regscan".to_string(),
                "gdn_prefill_regscan_c1".to_string(),
            ]
        });
    // The GPU clock ramps over the first hundred milliseconds of work, so
    // warm it up before the measured rounds.
    let rounds = 8;
    let warmup = std::env::var("LILY_GDN_SCAN_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40usize);
    for round in 0..warmup + rounds {
        if round == warmup {
            crate::metal::profile::take();
        }
        for k in 0..names.len() {
            let name = &names[(k + round) % names.len()];
            let pass = ctx.begin().expect("pass");
            gdn_prefill_scan_named(
                &ctx, &pass, &t_qkv, &ta, &tb, &t_a_log, &t_dt_bias, &staging, &state,
                &out, scale, hk, None, name,
            )
            .expect("gdn_prefill");
            pass.commit_wait().expect("commit");
        }
    }
    let passes = crate::metal::profile::take();
    for name in &names {
        // The chunked scan is two dispatches per pass; report their sum.
        let matches = |kernel: &str| {
            kernel == name.as_str()
                || (name == GDN_CHUNK_SCAN && kernel.starts_with("gdn_chunk"))
        };
        let mut ms: Vec<f64> = passes
            .iter()
            .filter_map(|p| {
                let s: f64 = p
                    .kernels
                    .iter()
                    .filter(|s| matches(s.name))
                    .map(|s| s.gpu_secs * 1e3)
                    .sum();
                (s > 0.0).then_some(s)
            })
            .collect();
        ms.sort_by(|x, y| x.total_cmp(y));
        let n = ms.len();
        assert!(n > 0, "no dispatches of {name} recorded");
        eprintln!(
            "{name}: min {:.1} ms, median {:.1} ms over {n} dispatches",
            ms[0],
            ms[n / 2]
        );
        if name == GDN_CHUNK_SCAN {
            for part in ["gdn_chunk_wy", "gdn_chunk_scan"] {
                let mut ms: Vec<f64> = passes
                    .iter()
                    .flat_map(|p| p.kernels.iter())
                    .filter(|s| s.name.starts_with(part))
                    .map(|s| s.gpu_secs * 1e3)
                    .collect();
                ms.sort_by(|x, y| x.total_cmp(y));
                if !ms.is_empty() {
                    eprintln!(
                        "  {part}: min {:.2} ms, median {:.2} ms",
                        ms[0],
                        ms[ms.len() / 2]
                    );
                }
            }
        }
    }
}

/// The decode step kernel as decode runs it: a chain of one dispatch per
/// GDN layer (36 distinct states), a barrier between, in one concurrent
/// pass. Prints the microseconds per dispatch (best of several passes) and
/// the state traffic it implies for every kernel in `LILY_GDN_STEP_KERNELS`
/// (comma separated; default: the shipped kernel), the order rotated per
/// pass; `LILY_GDN_STEP_HEADS` sets the head count (default 48). Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn gdn_step_chain_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let h: usize = std::env::var("LILY_GDN_STEP_HEADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    let (hk, dim, layers) = (h / 3, GDN_HEAD_DIM, 36usize);
    let scale = 1.0 / (dim as f32).sqrt();
    let mut rng = StdRng::seed_from_u64(7);
    let names: Vec<String> = std::env::var("LILY_GDN_STEP_KERNELS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_else(|_| vec!["gdn_step_gated".to_string()]);
    let names: Vec<&'static str> =
        names.into_iter().map(|n| &*Box::leak(n.into_boxed_str())).collect();
    let t_a_log = Tensor::from_f32(&ctx, &random_vec(&mut rng, h, -2.0, 0.5), &[h])
        .expect("a_log");
    let t_dt_bias =
        Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, h, -0.5, 0.5), &[h])
            .expect("dt");
    let t_norm_w = Tensor::from_f32(&ctx, &random_vec(&mut rng, dim, 0.5, 1.5), &[dim])
        .expect("norm_w");
    let t_qkv = Tensor::from_f32_as_bf16(
        &ctx,
        &random_vec(&mut rng, (2 * hk + h) * dim, -1.0, 1.0),
        &[2 * hk + h, dim],
    )
    .expect("qkv");
    let t_a = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, h, -1.0, 1.0), &[h])
        .expect("a");
    let t_b = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, h, -1.0, 1.0), &[h])
        .expect("b");
    let t_z = Tensor::from_f32_as_bf16(
        &ctx,
        &random_vec(&mut rng, h * dim, -1.0, 1.0),
        &[h, dim],
    )
    .expect("z");
    let out = Tensor::zeros(&ctx, &[h, dim], DType::BF16).expect("out");
    let states: Vec<Tensor> = (0..layers)
        .map(|_| Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state"))
        .collect();
    let rounds = 7;
    let mut best = vec![f64::INFINITY; names.len()];
    for round in 0..rounds {
        for k in 0..names.len() {
            let which = (round + k) % names.len();
            let pass = ctx.begin_concurrent().expect("pass");
            for state in &states {
                gdn_step_gated_named(
                    &ctx,
                    &pass,
                    names[which],
                    &t_qkv,
                    &t_a,
                    &t_b,
                    &t_a_log,
                    &t_dt_bias,
                    state,
                    &t_z,
                    &t_norm_w,
                    &out,
                    scale,
                    hk,
                    1e-6,
                    GdnGate::Sigmoid,
                )
                .expect(names[which]);
                pass.level_barrier(&[&out]).expect("barrier");
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            let secs = t.gpu_end_secs - t.gpu_start_secs;
            if round > 0 {
                best[which] = best[which].min(secs / layers as f64);
            }
        }
    }
    let bytes = 2.0 * (h * dim * dim * 4) as f64;
    for (name, secs) in names.iter().zip(&best) {
        eprintln!(
            "{name}: {:.2} us per dispatch in the chain ({:.0} GB/s of state read+write)",
            secs * 1e6,
            bytes / secs / 1e9
        );
    }
}

/// The bandwidth the decode step's state traffic (read + write of 36 f32
/// `[48, 128, 128]` states, one dispatch per layer with a barrier between)
/// reaches for a range of grid shapes, through a streaming kernel with no
/// compute: threadgroups of `C * RG` threads, one per (head, column group
/// of `C` columns), each thread walking `128 / RG` rows. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing only"]
fn gdn_state_stream_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let (h, dim, layers) = (48usize, GDN_HEAD_DIM, 36usize);
    let states: Vec<Tensor> = (0..layers)
        .map(|_| Tensor::zeros(&ctx, &[h, dim, dim], DType::F32).expect("state"))
        .collect();
    let out = Tensor::zeros(&ctx, &[dim], DType::F32).expect("out");
    let pipeline = ctx
        .pipeline("gdn_state_stream", TEST_SOURCE, MslVersion::V3_1)
        .expect("kernel");
    let shapes: [(usize, usize); 10] = [
        (128, 1),
        (128, 2),
        (128, 4),
        (128, 8),
        (64, 2),
        (64, 4),
        (32, 1),
        (32, 4),
        (32, 16),
        (16, 8),
    ];
    let decay = 0.99f32;
    let mut best = vec![f64::INFINITY; shapes.len()];
    for round in 0..6 {
        for k in 0..shapes.len() {
            let which = (round + k) % shapes.len();
            let (c, rg) = shapes[which];
            let pass = ctx.begin_concurrent().expect("pass");
            for state in &states {
                pass.dispatch_at(
                    &pipeline,
                    &[state.binding(), out.binding()],
                    &[&u32_bytes(c), &u32_bytes(rg), &decay.to_ne_bytes()],
                    Grid::Threadgroups {
                        groups: (h * (dim / c), 1, 1),
                        threadgroup: (c * rg, 1, 1),
                    },
                )
                .expect("dispatch");
                pass.level_barrier(&[&out]).expect("barrier");
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if round > 0 {
                best[which] = best[which]
                    .min((t.gpu_end_secs - t.gpu_start_secs) / layers as f64);
            }
        }
    }
    let bytes = 2.0 * (h * dim * dim * 4) as f64;
    for ((c, rg), secs) in shapes.iter().zip(&best) {
        eprintln!(
            "C={c:>3} RG={rg:>2} ({:>4} threadgroups x {:>4} threads): {:.2} us per dispatch, {:.0} GB/s",
            h * (dim / c),
            c * rg,
            secs * 1e6,
            bytes / secs / 1e9
        );
    }
}
