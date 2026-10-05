//! The batched decode step across sessions against the 4-layer checkpoint
//! (`LILY_MODEL_DIR_FLASH`). Written with the draft, not run yet.

use std::path::Path;

use crate::engine::{
    BatchRow, DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
};
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;

use super::*;

const GREEDY: SamplingParams = SamplingParams::greedy();

fn load(ctx: &MetalContext) -> Option<Qwen4ExpModel> {
    let dir = std::env::var("LILY_MODEL_DIR_FLASH").ok()?;
    Some(
        <Qwen4ExpModel as LanguageModel>::load(
            ctx,
            Path::new(&dir),
            &LoadOptions::default(),
        )
        .expect("load"),
    )
}

fn prompt(len: usize, salt: usize) -> Vec<u32> {
    (0..len).map(|i| 1000 + ((i * 37 + salt * 101) % 5000) as u32).collect()
}

/// A state that has fed `prompt` and its first greedy draw (not fed).
fn prefilled(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    s: &mut Scratch,
    prompt: &[u32],
) -> (DecodeState, u32) {
    let mut state = model.new_state(ctx, prompt.len() + 64).expect("state");
    model
        .prefill(ctx, &mut state, s, prompt, Some(Draw { params: &GREEDY, step: 0 }))
        .expect("prefill");
    let first = s.next_token().view(0, &[1]).expect("view").to_u32().expect("read")[0];
    (state, first)
}

/// A state's per-token caches and recurrent state, serialized.
type Fingerprint = (Vec<u8>, Vec<u8>);

/// What a decode leaves behind that later steps read: the per-token caches
/// and the recurrent state.
fn fingerprint(ctx: &MetalContext, state: &DecodeState) -> Fingerprint {
    let mut caches = Vec::new();
    state.write_prefix(state.pos(), &mut caches).expect("prefix");
    let mut recurrent = Vec::new();
    state.snapshot(ctx).expect("snapshot").write_to(&mut recurrent).expect("write");
    (caches, recurrent)
}

/// A row of a two-row step computes what the same row computes in a
/// one-row step: same draws, same caches, same recurrent state, the draft
/// head's included. This is the exact check of the batched graph's
/// per-row bindings (a row reading another row's cache, position or state
/// breaks it); it relies on the batched kernels being row-count invariant
/// per row, which a failure here would also expose (then compare with a
/// tolerance and look at which part moved). Prompts of different lengths,
/// so the rows sit at different positions and complete indexer blocks at
/// different steps.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_row_beside_another_decodes_as_it_does_alone() {
    let ctx = MetalContext::new().expect("metal context");
    let Some(model) = load(&ctx) else { return };
    let steps = 24;
    let prompts = [prompt(41, 1), prompt(70, 2)];
    let alone: Vec<(Vec<u32>, Fingerprint)> = prompts
        .iter()
        .map(|p| {
            let mut s = model.new_scratch_with_capacity(&ctx, 256).expect("scratch");
            let (mut state, first) = prefilled(&ctx, &model, &mut s, p);
            let mut tokens = vec![first];
            for _ in 0..steps {
                let mut rows = [BatchRow {
                    token: *tokens.last().expect("drawn"),
                    draw: Draw { params: &GREEDY, step: tokens.len() },
                    slot: 0,
                    state: &mut state,
                }];
                tokens
                    .push(model.decode_rows(&ctx, &mut s, &mut rows).expect("step")[0]);
            }
            let print = fingerprint(&ctx, &state);
            (tokens, print)
        })
        .collect();

    let mut s = model.new_scratch_with_capacity(&ctx, 256).expect("scratch");
    let (mut a, first_a) = prefilled(&ctx, &model, &mut s, &prompts[0]);
    let (mut b, first_b) = prefilled(&ctx, &model, &mut s, &prompts[1]);
    let (mut ta, mut tb) = (vec![first_a], vec![first_b]);
    for _ in 0..steps {
        let mut rows = [
            BatchRow {
                token: *ta.last().expect("drawn"),
                draw: Draw { params: &GREEDY, step: ta.len() },
                slot: 0,
                state: &mut a,
            },
            BatchRow {
                token: *tb.last().expect("drawn"),
                draw: Draw { params: &GREEDY, step: tb.len() },
                slot: 1,
                state: &mut b,
            },
        ];
        let draws = model.decode_rows(&ctx, &mut s, &mut rows).expect("step");
        ta.push(draws[0]);
        tb.push(draws[1]);
    }
    assert_eq!(ta, alone[0].0, "row 0's draws");
    assert_eq!(tb, alone[1].0, "row 1's draws");
    assert!(fingerprint(&ctx, &a) == alone[0].1, "row 0's caches or state differ");
    assert!(fingerprint(&ctx, &b) == alone[1].1, "row 1's caches or state differ");
}

/// A one-row batched step against the production decode step: the same
/// model, different reductions (skinny GEMMs and the small-m MoE against
/// the decode GEMVs), so only near-ties may differ. The four-layer model's
/// logits are nearly tied, so this prints the agreement and asserts only
/// the first step.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_batched_row_agrees_with_the_decode_step() {
    let ctx = MetalContext::new().expect("metal context");
    let Some(model) = load(&ctx) else { return };
    let steps = 32;
    let p = prompt(57, 3);

    let mut s = model.new_scratch_with_capacity(&ctx, 256).expect("scratch");
    let (mut state, first) = prefilled(&ctx, &model, &mut s, &p);
    let mut decoded = vec![first];
    for _ in 0..steps {
        let token = *decoded.last().expect("drawn");
        // Slot 0 holds the input token, as the prefill left it.
        s.next_token()
            .view(0, &[1])
            .expect("view")
            .write_bytes(bytemuck::cast_slice(&[token]))
            .expect("write");
        let pass = model
            .encode_decode_step(
                &ctx,
                &state,
                &s,
                0,
                1,
                Draw { params: &GREEDY, step: decoded.len() },
                None,
            )
            .expect("encode");
        model.prepare_step_inputs(&mut state, &s, token).expect("stage");
        pass.commit().expect("commit").wait().expect("run");
        state.advance(1);
        decoded.push(
            s.next_token().view(1, &[1]).expect("view").to_u32().expect("read")[0],
        );
    }

    let mut s = model.new_scratch_with_capacity(&ctx, 256).expect("scratch");
    let (mut state, first) = prefilled(&ctx, &model, &mut s, &p);
    let mut batched = vec![first];
    for _ in 0..steps {
        let mut rows = [BatchRow {
            token: *batched.last().expect("drawn"),
            draw: Draw { params: &GREEDY, step: batched.len() },
            slot: 0,
            state: &mut state,
        }];
        batched.push(model.decode_rows(&ctx, &mut s, &mut rows).expect("step")[0]);
    }
    let agree = decoded.iter().zip(&batched).take_while(|(a, b)| a == b).count();
    eprintln!(
        "batched row agrees with the decode step for {agree} of {} draws",
        steps + 1
    );
    assert!(agree >= 2, "the first batched draw differs from the decode step's");
}
