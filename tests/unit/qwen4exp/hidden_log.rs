//! The final hidden-state log against the 4-layer checkpoint
//! (`LILY_MODEL_DIR_FLASH`): over a two-chunk prefill, every prompt row is
//! written and none after it; the last prompt row is the hidden state the
//! head path computes for that position on its own; enabling the log does
//! not change what the model computes; and the rows do not depend on how
//! the prompt was chunked, a tail on the fused read route included.

use std::path::Path;

use crate::engine::{Draw, LanguageModel, LoadOptions};
use crate::metal::MetalContext;

use super::*;

/// Prefills `prompt` with a greedy draw, with or without the hidden log:
/// the draw, the logits, the head path's hidden state for the last prompt
/// position, and the log.
fn run(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
    log: bool,
) -> (u32, Vec<u8>, Vec<f32>, Option<Vec<f32>>) {
    let capacity = prompt.len() + 8;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    if log {
        model.enable_hidden_log(ctx, &mut s, capacity).expect("enable");
    }
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
    let token = s.next_token().view(0, &[1]).expect("view").to_u32().expect("token")[0];
    let head = s.hc.mixed.to_f32().expect("read");
    let logged =
        log.then(|| model.hidden_log(&s).expect("log").to_f32().expect("read"));
    (token, s.logits.raw_bytes().to_vec(), head, logged)
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn the_hidden_log_holds_every_prompt_row_and_changes_nothing() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    // A full chunk and a tail of 44 rows.
    model.set_prefill_chunk(256).expect("chunk");
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
    let (plain_token, plain_logits, _, _) = run(&ctx, &model, &prompt, false);
    let (token, logits, head, logged) = run(&ctx, &model, &prompt, true);
    assert_eq!(plain_token, token, "the log changed the draw");
    assert!(plain_logits == logits, "the log changed the logits");
    let logged = logged.expect("log");
    let h = model.config.hidden_size;
    let n = prompt.len();
    let rows: Vec<&[f32]> = logged.chunks(h).collect();
    for (p, row) in rows.iter().enumerate() {
        let written = row.iter().any(|&v| v != 0.0);
        assert_eq!(written, p < n, "row {p}: written {written}");
        assert!(row.iter().all(|v| v.is_finite()), "row {p} not finite");
    }
    // The rows' final read and the head path's single-row read are two
    // kernels on the same streams: equal up to bf16 rounding.
    let last = rows[n - 1];
    let dot: f32 = last.iter().zip(&head).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm(last) * norm(&head));
    assert!(cosine > 0.999, "last prompt row vs the head path: cosine {cosine}");
    // And not some other row's state.
    let prev = rows[n - 2];
    let dot_prev: f32 = prev.iter().zip(&head).map(|(a, b)| a * b).sum();
    assert!(
        dot_prev / (norm(prev) * norm(&head)) < cosine,
        "the row before matches the head path as well"
    );
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn the_hidden_log_does_not_depend_on_the_chunking() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    // 258 tokens: one chunk of 512 rows, or 256 rows and a tail of 2, which
    // takes the fused read (at most HC_FUSED_MAX_ROWS rows).
    let prompt: Vec<u32> = (0..258).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
    let h = model.config.hidden_size;
    model.set_prefill_chunk(512).expect("chunk");
    let (_, _, _, one) = run(&ctx, &model, &prompt, true);
    model.set_prefill_chunk(256).expect("chunk");
    let (_, _, _, two) = run(&ctx, &model, &prompt, true);
    let (one, two) = (one.expect("log"), two.expect("log"));
    for p in 0..prompt.len() {
        let (a, b) = (&one[p * h..(p + 1) * h], &two[p * h..(p + 1) * h]);
        let c = cosine(a, b);
        assert!(c > 0.999, "row {p}: cosine {c} between the chunkings");
    }
}
