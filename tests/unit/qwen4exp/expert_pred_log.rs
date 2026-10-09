//! The cross-layer prediction log against the 4-layer checkpoint
//! (`LILY_MODEL_DIR_FLASH`): lookahead 0 of both variants (A: each layer's
//! router rerun on its own gate input; B: each layer's read-in and router
//! rerun on the streams its real read consumed, all through the prediction
//! path's own scratch) must reproduce the routing the layer itself logged,
//! bit for bit, over a two-chunk prefill and plain decode steps: the top-k
//! by score of the wider logged selection (`EXPERT_PRED_WIDTH * top_k`) is
//! the routed set, and the whole selection contains it; and enabling the
//! logs must not change what the model computes.

use std::path::Path;

use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions};
use crate::metal::MetalContext;

use super::*;
use crate::qwen4exp::route_predict::top_by_score;

const STEPS: usize = 6;

struct Run {
    tokens: Vec<u32>,
    logits: Vec<u8>,
    /// The routing log, the prediction log's ids and scores.
    logs: Option<(Vec<u32>, Vec<u32>, Vec<f32>)>,
}

/// Prefills `prompt` (two chunks at the 256-row chunk the test sets), then
/// decodes `STEPS` greedy tokens with the plain decode step, with or
/// without the logs.
fn run(ctx: &MetalContext, model: &Qwen4ExpModel, prompt: &[u32], logs: bool) -> Run {
    let capacity = prompt.len() + STEPS + 1;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    if logs {
        model.enable_expert_prediction(ctx, &mut s, capacity).expect("enable");
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
    let mut tokens =
        vec![s.next_token().view(0, &[1]).expect("view").to_u32().expect("token")[0]];
    let mut slot = 0usize;
    for step in 1..=STEPS {
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
    let logs = logs.then(|| {
        (
            model.expert_log(&s).expect("log").to_u32().expect("read"),
            model.expert_pred_log(&s).expect("pred log").to_u32().expect("read"),
            model.expert_pred_scores(&s).expect("scores").to_f32().expect("read"),
        )
    });
    Run { tokens, logits: s.logits.raw_bytes().to_vec(), logs }
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn lookahead_zero_reproduces_the_routing() {
    let Some(dir) = std::env::var("LILY_MODEL_DIR_FLASH").ok() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let mut model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    // A full chunk on the grouped route and a tail of 44 rows.
    model.set_prefill_chunk(256).expect("chunk");
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37 % 5000) as u32).collect();

    let plain = run(&ctx, &model, &prompt, false);
    let logged = run(&ctx, &model, &prompt, true);
    assert_eq!(plain.tokens, logged.tokens, "the logs changed the draws");
    assert!(plain.logits == logged.logits, "the logs changed the logits");

    let (layers, experts, k) = model.moe_shape();
    let capacity = prompt.len() + STEPS + 1;
    // Every position the passes ran: the prompt, then each decode step's
    // input (the prefill draw and the first STEPS - 1 decoded tokens).
    let positions = prompt.len() + STEPS;
    let (actual, pred, scores) = logged.logs.expect("logs");
    let w = EXPERT_PRED_WIDTH * k;
    assert_eq!(
        pred.len(),
        EXPERT_PRED_VARIANTS * EXPERT_PRED_LOOKAHEADS * layers * capacity * w
    );
    assert_eq!(scores.len(), pred.len());
    let row = |l: usize, p: usize| {
        let at = (l * capacity + p) * k;
        actual[at..at + k].to_vec()
    };
    // `plane` = variant * EXPERT_PRED_LOOKAHEADS + lookahead: the whole
    // selection with its scores, and its top-k by score.
    let wide = |plane: usize, l: usize, p: usize| {
        let at = ((plane * layers + l) * capacity + p) * w;
        (pred[at..at + w].to_vec(), scores[at..at + w].to_vec())
    };
    let predicted_top = |plane: usize, l: usize, p: usize| {
        let (ids, sc) = wide(plane, l, p);
        let mut top = Vec::new();
        top_by_score(&ids, &sc, k, &mut top);
        top
    };
    let set = |ids: &[u32]| {
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        sorted
    };
    let distinct_set = |ids: &[u32], n: usize| {
        let mut sorted = set(ids);
        sorted.dedup();
        sorted.len() == n && sorted.iter().all(|&e| (e as usize) < experts)
    };
    let mut same_as_source = 0usize;
    let mut b_differs = 0usize;
    for p in 0..positions {
        for l in 0..layers {
            let act = row(l, p);
            assert!(
                distinct_set(&act, k),
                "layer {l}, position {p}: not a top-{k} set: {act:?}"
            );
            for v in 0..EXPERT_PRED_VARIANTS {
                let plane = v * EXPERT_PRED_LOOKAHEADS;
                let tag = format!("variant {v}, lookahead 0, layer {l}, position {p}");
                assert_eq!(set(&predicted_top(plane, l, p)), set(&act), "{tag}");
                let (all, sc) = wide(plane, l, p);
                assert!(act.iter().all(|e| all.contains(e)), "{tag}: top-{w}");
                assert!(sc.windows(2).all(|s| s[0] >= s[1]), "{tag}: scores {sc:?}");
                for d in 1..EXPERT_PRED_LOOKAHEADS {
                    let (all, _) = wide(plane + d, l, p);
                    if l < d {
                        assert!(
                            all.iter().all(|&e| e == 0),
                            "written without a source"
                        );
                        continue;
                    }
                    assert!(
                        distinct_set(&all, w),
                        "variant {v}, lookahead {d}, layer {l}, position {p}: not written: \
                         {all:?}"
                    );
                    let predicted = set(&predicted_top(plane + d, l, p));
                    same_as_source += (predicted == set(&row(l - d, p))) as usize;
                    if v == 1 {
                        b_differs +=
                            (predicted != set(&predicted_top(d, l, p))) as usize;
                    }
                }
            }
        }
    }
    // A prediction made with the source layer's router instead of the
    // target's would reproduce the source layer's routing everywhere.
    let predictions = EXPERT_PRED_VARIANTS
        * (1..EXPERT_PRED_LOOKAHEADS).map(|d| layers - d).sum::<usize>()
        * positions;
    assert!(
        same_as_source < predictions / 2,
        "{same_as_source} of {predictions} predictions equal the source layer's routing"
    );
    // B reads the streams through the target's read-in, A takes the source's
    // gate input as is: the two cannot agree everywhere.
    assert!(b_differs > 0, "variant B never differs from A");
    // Positions past the run were never written.
    for l in 0..layers {
        assert!(row(l, positions).iter().all(|&e| e == 0));
    }
}
