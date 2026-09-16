//! Speculation internals against the 4-layer checkpoint (`LILY_MODEL_DIR_FLASH`).

use std::path::Path;

use crate::engine::{LanguageModel, LoadOptions};
use crate::metal::MetalContext;

use super::*;

fn model_dir() -> Option<String> {
    std::env::var("LILY_MODEL_DIR_FLASH").ok()
}

fn copy_state(ctx: &MetalContext, model: &Qwen4ExpModel, src: &DecodeState, tokens: usize, capacity: usize) -> DecodeState {
    let mut bytes = Vec::new();
    src.write_prefix(tokens, &mut bytes).expect("prefix");
    let snap = src.snapshot(ctx).expect("snapshot");
    let mut dst = model.new_state(ctx, capacity).expect("state");
    dst.read_prefix(ctx, tokens, tokens, &mut std::io::Cursor::new(&bytes)).expect("read prefix");
    dst.restore(ctx, &snap).expect("restore");
    dst
}

/// Runs one verify pass the host-driven way (stage the n-gram rows, encode,
/// wait, advance the bookkeeping) and returns its draws.
fn host_verify(ctx: &MetalContext, model: &Qwen4ExpModel, state: &mut DecodeState, s: &mut Scratch, tokens: &[u32]) -> Vec<u32> {
    let m = tokens.len();
    model.ensure_room(ctx, state, s, m + 8).expect("room");
    let greedy = SamplingParams::greedy();
    let s = &*s;
    let capacity = s.prefill.as_ref().expect("prefill scratch");
    let ps = capacity.chunk(tokens).expect("chunk");
    let ple_w = model.weights.layers.iter().find_map(|l| l.ple.as_deref());
    if let (Some(w), Some(p), Some(pst)) = (ple_w, &ps.ple, &state.ple) {
        model.stage_ngram(w, p, tokens, pst.hist).expect("stage n-gram rows");
    }
    model
        .encode_batch(ctx, state, s, &ps, BatchMode::Verify { params: &greedy, step0: 0, park: None })
        .expect("encode")
        .commit()
        .expect("commit")
        .wait()
        .expect("verify");
    state.pos += m;
    if let Some(pst) = &mut state.ple {
        pst.hist = NgramHasher::advance(pst.hist, tokens);
    }
    state.conv_slot = 1 - state.conv_slot;
    s.spec.as_ref().expect("spec").verify_tokens.to_u32().expect("draws")[..m].to_vec()
}

/// Runs the host-driven draft pass after [`host_verify`] with the accepted
/// count known: rollback to it, head catch-up over all `m` verified rows (the
/// same rows the GPU-selected pass feeds, so the projections take the same
/// kernel routes), chain from row `a`. Returns the proposals and leaves the
/// host bookkeeping as `finish_speculation` does.
#[allow(clippy::too_many_arguments)]
fn host_draft(ctx: &MetalContext, model: &Qwen4ExpModel, state: &mut DecodeState, s: &Scratch, pos_before: usize, hist_before: Option<[u32; 2]>, tokens: &[u32], sampled: &[u32], chain: usize) -> Vec<u32> {
    let wide = model.config.hc_width();
    let m = tokens.len();
    let a = sampled.iter().zip(&tokens[1..]).take_while(|(x, y)| x == y).count();
    let capacity_scratch = s.prefill.as_ref().expect("prefill scratch");
    let hidden = capacity_scratch.hyper.view(0, &[m, wide]).expect("hidden");
    let keep = capacity_scratch.hyper.view(a * wide, &[wide]).expect("keep");
    let rollback = (a + 1 < m).then_some((a, m));
    let encoded = model.encode_draft(ctx, state, s, &hidden, m, a, pos_before, Some(&keep), rollback, chain, None).expect("encode draft");
    s.spec.as_ref().expect("spec").mtp_ids.view(0, &[m]).expect("ids").write_bytes(bytemuck::cast_slice(sampled)).expect("write ids");
    encoded.commit().expect("commit").wait().expect("draft");
    state.pos = pos_before + a + 1;
    if let (Some(pst), Some(before)) = (&mut state.ple, hist_before) {
        pst.hist = NgramHasher::advance(before, &tokens[..=a]);
    }
    s.spec.as_ref().expect("spec").draft_tokens.to_u32().expect("drafts")[..chain].to_vec()
}

/// Positions below `pos` whose draft-head cache rows differ between two states.
fn head_cache_diff(g: &DecodeState, b: &DecodeState, pos: usize) -> Vec<String> {
    let (gm, bm) = (g.mtp.as_ref().expect("mtp"), b.mtp.as_ref().expect("mtp"));
    let mut out = Vec::new();
    let (LayerState::Attn { k_cache: gk, v_cache: gv, idx_keys: gi, blk_keys: gb }, LayerState::Attn { k_cache: bk, v_cache: bv, idx_keys: bi, blk_keys: bb }) = (&gm.layer, &bm.layer) else {
        panic!("head is not attention")
    };
    for (name, x, y) in [("k", gk, bk), ("v", gv, bv)] {
        let (kvh, max_seq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (xb, yb) = (x.raw_bytes(), y.raw_bytes());
        for h in 0..kvh {
            for p in 0..pos {
                let o = ((h * max_seq + p) * d) * 2;
                if xb[o..o + d * 2] != yb[o..o + d * 2] {
                    out.push(format!("{name}[head {h}, pos {p}]"));
                }
            }
        }
    }
    let d = gi.shape()[1];
    let (xb, yb) = (gi.raw_bytes(), bi.raw_bytes());
    for p in 0..pos {
        if xb[p * d * 2..(p + 1) * d * 2] != yb[p * d * 2..(p + 1) * d * 2] {
            out.push(format!("idx[pos {p}]"));
        }
    }
    let (xb, yb) = (gb.raw_bytes(), bb.raw_bytes());
    for blk in 0..pos / 4 {
        if xb[blk * d * 2..(blk + 1) * d * 2] != yb[blk * d * 2..(blk + 1) * d * 2] {
            out.push(format!("blk[{blk}]"));
        }
    }
    if gm.hidden.raw_bytes() != bm.hidden.raw_bytes() {
        out.push("hidden".to_string());
    }
    out
}

/// Over a sequence of speculative steps with controlled accepted counts, the
/// GPU-selected draft pass proposes exactly what the host-driven one
/// (rollback to the known count, chain from the accepted row at host
/// positions) proposes, leaves the head's caches identical, and the trunk's
/// draws stay identical. Both paths are fed the same drafts (chosen from
/// probes of the greedy continuation, so every accepted count occurs; the
/// 4-layer head's own proposals would never be accepted).
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn gpu_selected_drafts_equal_host_driven_drafts() {
    let Some(dir) = model_dir() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(&ctx, Path::new(&dir), &LoadOptions { mtp_drafts: 3, ..LoadOptions::default() }).expect("load");
    let greedy = SamplingParams::greedy();
    let chain = 2usize;
    let pattern = [0usize, 0, 1, 0, 2, 1, 1, 0, 2, 0, 1, 2];
    for prompt_len in [300usize, 33000] {
        let prompt: Vec<u32> = (0..prompt_len).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
        let capacity = prompt_len + 128;
        let mut s = model.new_scratch_with_capacity(&ctx, capacity).expect("scratch");
        let mut b = model.new_state(&ctx, capacity).expect("state");
        LanguageModel::prefill(&model, &ctx, &mut b, &mut s, &prompt, None).expect("prefill");
        let mut g = copy_state(&ctx, &model, &b, prompt_len, capacity);
        let mut pending = 4242u32;
        for (step, &want) in pattern.iter().enumerate() {
            // Probe the greedy continuation from the host state.
            let fed = b.pos;
            let mut probe = copy_state(&ctx, &model, &b, fed, capacity);
            let d0 = host_verify(&ctx, &model, &mut probe, &mut s, &[pending, 0, 0])[0];
            let mut probe = copy_state(&ctx, &model, &b, fed, capacity);
            let d1 = host_verify(&ctx, &model, &mut probe, &mut s, &[pending, d0, 0])[1];
            let drafts = match want {
                0 => vec![d0 + 1, d1],
                1 => vec![d0, d1 + 1],
                _ => vec![d0, d1],
            };
            let tokens: Vec<u32> = std::iter::once(pending).chain(drafts.iter().copied()).collect();

            // Host-driven step.
            let pos_before = b.pos;
            let hist_before = b.ple.as_ref().map(|p| p.hist);
            let sampled = host_verify(&ctx, &model, &mut b, &mut s, &tokens);
            let a = sampled.iter().zip(&drafts).take_while(|(x, y)| x == y).count();
            assert_eq!(a, want, "step {step} at {prompt_len}: probe did not yield the wanted accepted count");
            let host_proposals = host_draft(&ctx, &model, &mut b, &s, pos_before, hist_before, &tokens, &sampled, chain);

            // GPU-selected step, without parking (the fed drafts are not the
            // head's proposals, so a parked next pass would verify the wrong
            // rows and advance the state wrongly).
            let (sampled_g, draft) = model.verify(&ctx, &mut g, &mut s, pending, &drafts, &greedy, 0, None, chain).expect("verify");
            assert_eq!(sampled_g, sampled, "step {step} at {prompt_len}: draws differ (trunk states diverged)");
            model.finish_speculation(&ctx, &mut g, &mut s, a, None, draft).expect("finish");
            let gpu_proposals = s.spec.as_ref().expect("spec").draft_tokens.to_u32().expect("drafts")[..chain].to_vec();
            let diff = head_cache_diff(&g, &b, b.pos + 3);
            eprintln!("step {step} a={a} at {prompt_len}: draws {sampled:?}, host {host_proposals:?}, gpu {gpu_proposals:?}, head cache rows differing: {diff:?}");
            assert_eq!(gpu_proposals, host_proposals, "step {step} a={a} at {prompt_len}: GPU-selected drafts differ from the host-driven ones");
            assert!(diff.is_empty(), "step {step} a={a} at {prompt_len}: head caches differ at {diff:?}");
            assert_eq!(g.pos, b.pos);
            pending = sampled[a];
        }
    }
}

/// Documents, without asserting, how a row's result through the head's input
/// projection depends on how many rows the batch has: the projection sees
/// `rows * hc_count` streams, and the skinny GEMM changes kernel (and
/// reduction order) past 8 streams. The GPU-selected draft pass always feeds
/// all verified rows, the host-driven one of earlier commits fed the
/// accepted rows only, so their drafts could differ on a near-tie (outputs
/// never do). Run with `--nocapture` to see the counts.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn probe_head_row_result_vs_batch_size() {
    let Some(dir) = model_dir() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(&ctx, Path::new(&dir), &LoadOptions { mtp_drafts: 3, ..LoadOptions::default() }).expect("load");
    let cfg = &model.config;
    let wide = cfg.hc_width();
    let (h, groups, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
    let capacity = prompt.len() + 64;
    let mut s = model.new_scratch_with_capacity(&ctx, capacity).expect("scratch");
    let mut state = model.new_state(&ctx, capacity).expect("state");
    LanguageModel::prefill(&model, &ctx, &mut state, &mut s, &prompt, None).expect("prefill");
    host_verify(&ctx, &model, &mut state, &mut s, &[4242, 17, 99]);
    let mtp = model.weights.mtp.as_ref().expect("draft head");
    let s = &s;
    let cap = s.prefill.as_ref().expect("prefill scratch");
    let mut outs: Vec<[Vec<f32>; 3]> = Vec::new();
    for rows in [1usize, 3] {
        let ps = cap.rows(rows).expect("rows");
        let hidden = cap.hyper.view(0, &[rows, wide]).expect("hidden");
        let hyper = ps.mtp_hyper.as_ref().expect("head scratch");
        let pass = ctx.begin_concurrent().expect("pass");
        crate::kernels::hc::rmsnorm_grouped_bf16(&ctx, &pass, &hidden, &mtp.norm_hidden, &ps.hc.hn, h, groups, eps, super::super::model::NORM_WEIGHT_BIAS).expect("norm");
        pass.level_barrier(&[&ps.hc.hn]).expect("barrier");
        let hn_streams = ps.hc.hn.view(0, &[rows * groups, h]).expect("streams");
        let hyper_streams = hyper.view(0, &[rows * groups, h]).expect("streams");
        project_mat(&ctx, &pass, &hn_streams, &mtp.fc_hidden, &hyper_streams, &s.dequant).expect("project");
        pass.commit_wait().expect("run");
        outs.push([
            ps.hc.hn.view(0, &[wide]).expect("row").to_f32().expect("read"),
            hyper.view(0, &[wide]).expect("row").to_f32().expect("read"),
            Vec::new(),
        ]);
    }
    for (i, name) in ["rmsnorm_grouped_bf16", "fc_hidden project_mat (skinny Q4, 4 vs 12 streams)"].iter().enumerate() {
        let n = outs[0][i].iter().zip(&outs[1][i]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        eprintln!("{name}: row 0 differs in {n} of {} elements between a 1-row and a 3-row run", outs[0][i].len());
    }
}
