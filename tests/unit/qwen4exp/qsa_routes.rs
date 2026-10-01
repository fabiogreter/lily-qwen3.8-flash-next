//! The per-query tensor-op sparse-attention route against the 4-layer
//! checkpoint (`LILY_MODEL_DIR_FLASH`): agreement with the split route, and
//! the prefill chunk size under it.

use std::path::Path;

use crate::engine::{Draw, LanguageModel, LoadOptions};
use crate::kernels::qsa::SparseAttnRoute;
use crate::metal::MetalContext;

use super::*;

fn model_dir() -> Option<String> {
    std::env::var("LILY_MODEL_DIR_FLASH").ok()
}

fn set_route(s: &mut Scratch, route: SparseAttnRoute) {
    s.prefill.as_mut().expect("prefill scratch").qsa.route = route;
}

/// Prefills `prompt` under `route` and returns the last row's logits.
fn prefill_logits(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
    route: SparseAttnRoute,
) -> Vec<f32> {
    prefill_then_decode(ctx, model, prompt, route, 0).0
}

/// Prefills `prompt` under `route`, then decodes `steps` greedy tokens
/// synchronously: the last prefill row's logits and the decoded tokens
/// (which exercise the state the prefill left behind).
fn prefill_then_decode(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
    route: SparseAttnRoute,
    steps: usize,
) -> (Vec<f32>, Vec<u32>) {
    let capacity = prompt.len() + 64;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    model.ensure_prefill_scratch(ctx, &mut s, prompt.len()).expect("prefill scratch");
    set_route(&mut s, route);
    let greedy = SamplingParams::greedy();
    LanguageModel::prefill(
        model,
        ctx,
        &mut state,
        &mut s,
        prompt,
        Some(Draw { params: &greedy, step: 0 }),
    )
    .expect("prefill");
    let logits = s.logits.to_f32().expect("logits");
    let mut tokens =
        vec![s.next_token().view(0, &[1]).expect("view").to_u32().expect("token")[0]];
    let mut slot = 0usize;
    for step in 1..=steps {
        let input = *tokens.last().expect("prefill draw");
        let encoded = model
            .encode_decode_step(
                ctx,
                &state,
                &s,
                slot,
                1 - slot,
                Draw { params: &greedy, step },
                None,
            )
            .expect("encode");
        model.prepare_step_inputs(&mut state, &s, input).expect("inputs");
        let pending = encoded.commit().expect("commit");
        state.advance(1);
        pending.wait().expect("wait");
        slot = 1 - slot;
        tokens.push(
            s.next_token().view(slot, &[1]).expect("view").to_u32().expect("token")[0],
        );
    }
    (logits, tokens)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
}

/// The query route reproduces the split route's prefill: same top token
/// and logits within the attention path's bf16 rounding, on a prompt that
/// crosses the dense limit inside its first chunk and on one with whole
/// chunks past it.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn query_prefill_matches_split_route() {
    let Some(dir) = model_dir() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    for prompt_len in [3000usize, 9000] {
        let prompt: Vec<u32> =
            (0..prompt_len).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
        let reference = prefill_logits(&ctx, &model, &prompt, SparseAttnRoute::Split);
        let top = argmax(&reference);
        {
            let got = prefill_logits(&ctx, &model, &prompt, SparseAttnRoute::Query);
            let max_abs = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = reference.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "{prompt_len} tokens: top {} vs {top}, max |dlogit| {max_abs:.4} (logit scale {scale:.2})",
                argmax(&got)
            );
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{prompt_len} tokens: non-finite logits"
            );
            assert!(
                max_abs <= 0.05 * scale,
                "{prompt_len} tokens: logits differ by {max_abs} (scale {scale})"
            );
            assert_eq!(argmax(&got), top, "{prompt_len} tokens: top token differs");
        }
    }
}

/// An 8 192-token prefill chunk (what the expert cache uses) reproduces
/// the 4 096-token chunking: same top token and logits within the
/// attention path's rounding on a prompt of two default chunks plus a
/// tail, and bit-identical on a prompt within one chunk either way.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn prefill_chunk_8192_matches_4096() {
    let Some(dir) = model_dir() else {
        eprintln!("LILY_MODEL_DIR_FLASH unset; skipped");
        return;
    };
    let ctx = MetalContext::new().expect("metal context");
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    for (prompt_len, exact) in [(3000usize, true), (9000, false)] {
        let prompt: Vec<u32> =
            (0..prompt_len).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
        let route = SparseAttnRoute::Query;
        model.set_prefill_chunk(4096).expect("chunk");
        let (reference, tokens_a) =
            prefill_then_decode(&ctx, &model, &prompt, route, 12);
        model.set_prefill_chunk(8192).expect("chunk");
        let (got, tokens_b) = prefill_then_decode(&ctx, &model, &prompt, route, 12);
        eprintln!("{prompt_len} tokens: decoded {tokens_a:?} against {tokens_b:?}");
        assert_eq!(
            tokens_a, tokens_b,
            "{prompt_len} tokens: the decode after the prefill differs"
        );
        let top = argmax(&reference);
        let max_abs = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = reference.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        eprintln!(
            "{prompt_len} tokens: top {} vs {top}, max |dlogit| {max_abs:.4} (logit scale {scale:.2})",
            argmax(&got)
        );
        assert!(
            got.iter().all(|x| x.is_finite()),
            "{prompt_len} tokens: non-finite logits"
        );
        if exact {
            assert_eq!(
                got, reference,
                "{prompt_len} tokens: one chunk either way must match exactly"
            );
        } else {
            assert!(
                max_abs <= 0.05 * scale,
                "{prompt_len} tokens: logits differ by {max_abs} (scale {scale})"
            );
            assert_eq!(argmax(&got), top, "{prompt_len} tokens: top token differs");
        }
    }
}
