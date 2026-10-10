//! The mid-trunk log against the 4-layer checkpoint (`LILY_MODEL_DIR_FLASH`):
//! enabling it does not change what the model computes; each layer's top
//! router picks begin with the routing the layer took; and the reads after a
//! cut do not depend on how the prompt was chunked.

use std::path::Path;

use crate::engine::{Draw, LanguageModel, LoadOptions};
use crate::metal::MetalContext;
use crate::qwen4exp::route_predict::top_by_score;

use super::*;

/// What one greedy prefill of `prompt` leaves: the draw, the logits, the
/// routing log, and (with `cuts`) the trunk log's reads, router picks and
/// scores.
struct Run {
    token: u32,
    logits: Vec<u8>,
    routing: Vec<u32>,
    trunk: Option<(Vec<f32>, Vec<u32>, Vec<f32>)>,
}

const WIDTH: usize = 32;

fn run(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
    cuts: Option<&[usize]>,
) -> Run {
    let capacity = prompt.len() + 8;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    model.enable_expert_log(ctx, &mut s, capacity).expect("routing log");
    if let Some(cuts) = cuts {
        model.enable_trunk_log(ctx, &mut s, capacity, cuts, WIDTH).expect("enable");
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
    let routing = model.expert_log(&s).expect("log").to_u32().expect("read");
    let trunk = cuts.map(|_| {
        let reads = model.trunk_reads(&s).expect("reads").to_f32().expect("read");
        let (ids, scores) = model.trunk_router(&s).expect("router");
        (reads, ids.to_u32().expect("read"), scores.to_f32().expect("read"))
    });
    Run { token, logits: s.logits.raw_bytes().to_vec(), routing, trunk }
}

fn load(dir: &str) -> (MetalContext, Qwen4ExpModel) {
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(dir),
        &LoadOptions::default(),
    )
    .expect("load");
    (ctx, model)
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn the_trunk_log_changes_nothing_and_its_router_picks_begin_with_the_routing() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let (ctx, mut model) = load(&dir);
    // A full chunk and a tail of 44 rows.
    model.set_prefill_chunk(256).expect("chunk");
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
    let plain = run(&ctx, &model, &prompt, None);
    let logged = run(&ctx, &model, &prompt, Some(&[0, 2]));
    assert_eq!(plain.token, logged.token, "the log changed the draw");
    assert!(plain.logits == logged.logits, "the log changed the logits");
    assert_eq!(plain.routing, logged.routing, "the log changed the routing");
    let (reads, ids, scores) = logged.trunk.expect("trunk");
    let (layers, _, top_k) = model.moe_shape();
    let (n, capacity) = (prompt.len(), prompt.len() + 8);
    let mut row = Vec::new();
    for layer in 0..layers {
        for p in 0..n {
            let at = (layer * capacity + p) * WIDTH;
            top_by_score(
                &ids[at..at + WIDTH],
                &scores[at..at + WIDTH],
                top_k,
                &mut row,
            );
            row.sort_unstable();
            let r = (layer * capacity + p) * top_k;
            let mut routed = plain.routing[r..r + top_k].to_vec();
            routed.sort_unstable();
            assert_eq!(row, routed, "layer {layer} position {p}");
        }
    }
    // Every prompt row of both cuts is written, none after.
    let h = model.config.hidden_size;
    for (c, cut) in reads.chunks(capacity * h).enumerate() {
        for (p, r) in cut.chunks(h).enumerate() {
            let written = r.iter().any(|&v| v != 0.0);
            assert_eq!(written, p < n, "cut {c} row {p}: written {written}");
            assert!(r.iter().all(|v| v.is_finite()), "cut {c} row {p} not finite");
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn the_trunk_reads_do_not_depend_on_the_chunking() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let (ctx, mut model) = load(&dir);
    // 258 tokens: one chunk of 512 rows, or 256 rows and a tail of 2, which
    // takes the fused read (at most HC_FUSED_MAX_ROWS rows).
    let prompt: Vec<u32> = (0..258).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
    let (h, capacity) = (model.config.hidden_size, prompt.len() + 8);
    model.set_prefill_chunk(512).expect("chunk");
    let one = run(&ctx, &model, &prompt, Some(&[1])).trunk.expect("trunk").0;
    model.set_prefill_chunk(256).expect("chunk");
    let two = run(&ctx, &model, &prompt, Some(&[1])).trunk.expect("trunk").0;
    assert_eq!(one.len(), capacity * h);
    for p in 0..prompt.len() {
        let c = cosine(&one[p * h..(p + 1) * h], &two[p * h..(p + 1) * h]);
        assert!(c > 0.999, "row {p}: cosine {c} between the chunkings");
    }
}
