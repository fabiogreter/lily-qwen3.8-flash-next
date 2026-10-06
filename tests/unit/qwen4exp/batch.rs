//! The batched decode step across sessions against the 4-layer checkpoint
//! (`LILY_MODEL_DIR_FLASH`).

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

/// A row of a two-row step does not depend on the row beside it: the same
/// session paired with two different neighbours (different prompts and
/// lengths, so different positions, and indexer blocks completing at
/// different steps), in either position, draws the same tokens and leaves
/// the same caches and recurrent state, the draft head's included. This is
/// the exact check of the batched graph's per-row bindings: a row reading
/// another row's cache, position or state breaks it. It is not compared
/// with the session decoded in a one-row step: the batched kernels are not
/// row-count invariant (a skinny GEMM at two rows reduces differently than
/// at one), so that agrees only up to near-ties, which the four-layer
/// model's logits are full of (measured 2026-10-05: row 0 alone and paired
/// differ by the same amounts whatever the neighbour, and flip at a top-2
/// gap of 0.07 after 20 equal draws).
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_row_does_not_depend_on_the_row_beside_it() {
    let ctx = MetalContext::new().expect("metal context");
    let Some(model) = load(&ctx) else { return };
    let steps = 24;
    let own = prompt(41, 1);
    // Runs `own` at `at` (0 or 1) beside `other`; returns own's draws and
    // fingerprint.
    let paired = |other: &[u32], at: usize| -> (Vec<u32>, Fingerprint) {
        let mut s = model.new_scratch_with_capacity(&ctx, 256).expect("scratch");
        let (mut a, first_a) = prefilled(&ctx, &model, &mut s, &own);
        let (mut b, first_b) = prefilled(&ctx, &model, &mut s, other);
        let (mut ta, mut tb) = (vec![first_a], vec![first_b]);
        for _ in 0..steps {
            let row_a = BatchRow {
                token: *ta.last().expect("drawn"),
                draw: Draw { params: &GREEDY, step: ta.len() },
                slot: at,
                state: &mut a,
            };
            let row_b = BatchRow {
                token: *tb.last().expect("drawn"),
                draw: Draw { params: &GREEDY, step: tb.len() },
                slot: 1 - at,
                state: &mut b,
            };
            let mut rows = if at == 0 { [row_a, row_b] } else { [row_b, row_a] };
            let draws = model.decode_rows(&ctx, &mut s, &mut rows).expect("step");
            ta.push(draws[at]);
            tb.push(draws[1 - at]);
        }
        let print = fingerprint(&ctx, &a);
        (ta, print)
    };
    for at in [0, 1] {
        let (draws_b, print_b) = paired(&prompt(70, 2), at);
        let (draws_c, print_c) = paired(&prompt(23, 3), at);
        assert_eq!(draws_b, draws_c, "row {at}'s draws depend on its neighbour");
        assert!(
            print_b == print_c,
            "row {at}'s caches or state depend on its neighbour"
        );
    }
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

/// The server's sampling defaults with a presence penalty: draws that depend
/// on the draw index (the RNG counter) and on each slot's penalty counts.
const SAMPLED: SamplingParams = SamplingParams {
    temperature: 1.0,
    top_k: 20,
    top_p: 0.95,
    presence_penalty: 1.5,
    seed: 7,
    ..SamplingParams::greedy()
};

/// The rows of a step over `states`, row `r` in slot `slots[r]`, feeding
/// each row's last draw and drawing its next one; `ahead` more for a step
/// parked behind the one whose draws are not taken yet (its token is then
/// not read).
fn rows_of<'r>(
    states: &'r mut [DecodeState],
    drawn: &[Vec<u32>],
    slots: &[usize],
    params: &'r SamplingParams,
    ahead: usize,
) -> Vec<BatchRow<'r, DecodeState>> {
    states
        .iter_mut()
        .zip(drawn)
        .zip(slots)
        .map(|((state, d), &slot)| BatchRow {
            token: *d.last().expect("drawn"),
            draw: Draw { params, step: d.len() + ahead },
            slot,
            state,
        })
        .collect()
}

/// One way of running batched steps: `steps` of them over `states`, each
/// draw appended to its row.
type Runner = fn(
    &MetalContext,
    &Qwen4ExpModel,
    &mut Scratch,
    &mut [DecodeState],
    &mut [Vec<u32>],
    &[usize],
    &SamplingParams,
    usize,
);

/// `steps` unparked batched steps (`decode_rows`, one at a time).
#[allow(clippy::too_many_arguments)]
fn run_unparked(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    s: &mut Scratch,
    states: &mut [DecodeState],
    drawn: &mut [Vec<u32>],
    slots: &[usize],
    params: &SamplingParams,
    steps: usize,
) {
    for _ in 0..steps {
        let mut rows = rows_of(states, drawn, slots, params, 0);
        let draws = model.decode_rows(ctx, s, &mut rows).expect("step");
        for (d, token) in drawn.iter_mut().zip(draws) {
            d.push(token);
        }
    }
}

/// `steps` batched steps as the scheduler runs them parked: the first
/// committed unparked, every later one committed behind the step before it
/// and released once that step's draws were taken. The last step is not
/// followed by a parked one, so the GPU is idle on return.
#[allow(clippy::too_many_arguments)]
fn run_parked(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    s: &mut Scratch,
    states: &mut [DecodeState],
    drawn: &mut [Vec<u32>],
    slots: &[usize],
    params: &SamplingParams,
    steps: usize,
) {
    let mut current = model
        .commit_rows(ctx, s, &mut rows_of(states, drawn, slots, params, 0), false)
        .expect("commit");
    for i in 0..steps {
        let next = (i + 1 < steps).then(|| {
            model
                .park_rows(
                    ctx,
                    s,
                    &mut rows_of(states, drawn, slots, params, 1),
                    &current,
                    false,
                )
                .expect("park")
        });
        let (draws, _) = model.finish_rows(s, current).expect("finish");
        for (d, token) in drawn.iter_mut().zip(draws) {
            d.push(token);
        }
        let Some(mut next) = next else { break };
        model
            .release_rows(s, &mut next, &mut rows_of(states, drawn, slots, params, 0))
            .expect("release");
        current = next;
    }
}

/// Prefilled states for `prompts` (each with its first draw) on a fresh
/// scratch.
fn prefilled_all(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompts: &[Vec<u32>],
) -> (Scratch, Vec<DecodeState>, Vec<Vec<u32>>) {
    let mut s = model.new_scratch_with_capacity(ctx, 256).expect("scratch");
    let (states, drawn) = prompts
        .iter()
        .map(|p| {
            let (state, first) = prefilled(ctx, model, &mut s, p);
            (state, vec![first])
        })
        .unzip();
    (s, states, drawn)
}

/// Parking changes when the host stages a step's inputs, never what the
/// step computes: 2 and 4 rows (different prompt lengths, so different
/// positions and indexer blocks completing at different steps), greedy and
/// sampled with a presence penalty, decoded parked and unparked from the
/// same states draw bit-identical tokens and leave bit-identical caches and
/// recurrent state, the draft head's included.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_parked_batched_step_computes_what_an_unparked_one_does() {
    let ctx = MetalContext::new().expect("metal context");
    let Some(model) = load(&ctx) else { return };
    let steps = 24;
    let all = [prompt(41, 1), prompt(70, 2), prompt(23, 3), prompt(56, 4)];
    for m in [2, 4] {
        for params in [&GREEDY, &SAMPLED] {
            let prompts = &all[..m];
            let slots: Vec<usize> = (0..m).collect();
            let run = |go: Runner| {
                let (mut s, mut states, mut drawn) =
                    prefilled_all(&ctx, &model, prompts);
                go(
                    &ctx,
                    &model,
                    &mut s,
                    &mut states,
                    &mut drawn,
                    &slots,
                    params,
                    steps,
                );
                let prints: Vec<Fingerprint> =
                    states.iter().map(|st| fingerprint(&ctx, st)).collect();
                (drawn, prints)
            };
            let (drawn_u, prints_u) = run(run_unparked);
            let (drawn_p, prints_p) = run(run_parked);
            let sampled = !params.is_greedy();
            assert_eq!(drawn_u, drawn_p, "{m} rows (sampled {sampled}): draws differ");
            for (r, (u, p)) in prints_u.iter().zip(&prints_p).enumerate() {
                assert!(
                    u == p,
                    "{m} rows (sampled {sampled}): row {r}'s caches or state differ"
                );
            }
        }
    }
}

/// A row that finishes on what step `k` drew while step `k + 1` is already
/// parked: the parked step runs anyway, released with every row's draw, so
/// the finished row's state holds its final token (fed == drawn, as after a
/// parked single-session step) and its extra draw is dropped; the other rows
/// take theirs as an ordinary step and go on (parked again) without it.
/// Against the same schedule unparked, with the finishing row taking part
/// in step `k + 1`: the finished row's state and every other row's draws,
/// caches and state are bit-identical.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn a_row_finishing_under_a_parked_step_keeps_its_final_token() {
    let ctx = MetalContext::new().expect("metal context");
    let Some(model) = load(&ctx) else { return };
    // Row 0 finishes on step k's draw; rows 1 and 2 decode `more` steps
    // after the drain.
    let (k, more) = (9, 10);
    let prompts = [prompt(41, 1), prompt(70, 2), prompt(23, 3)];
    let slots = [0, 1, 2];
    let params = &GREEDY;

    // Unparked: k + 2 steps of all three rows (the last feeds row 0's final
    // token), then the other two alone.
    let (mut s, mut states, mut drawn) = prefilled_all(&ctx, &model, &prompts);
    run_unparked(&ctx, &model, &mut s, &mut states, &mut drawn, &slots, params, k + 2);
    let (finished_draws, finished_print) =
        (drawn[0].clone(), fingerprint(&ctx, &states[0]));
    run_unparked(
        &ctx,
        &model,
        &mut s,
        &mut states[1..],
        &mut drawn[1..],
        &slots[1..],
        params,
        more,
    );
    let rest_u: Vec<(Vec<u32>, Fingerprint)> = states[1..]
        .iter()
        .zip(&drawn[1..])
        .map(|(st, d)| (d.clone(), fingerprint(&ctx, st)))
        .collect();

    // Parked: steps 0 to k - 1, then step k with step k + 1 parked behind
    // it. Row 0 finishes on step k's draw; step k + 1 is released with row
    // 0's final token and drained, row 0's extra draw dropped; then a new
    // parked stretch of the other two.
    let (mut s, mut states, mut drawn) = prefilled_all(&ctx, &model, &prompts);
    run_parked(&ctx, &model, &mut s, &mut states, &mut drawn, &slots, params, k);
    let step_k = model
        .commit_rows(
            &ctx,
            &mut s,
            &mut rows_of(&mut states, &drawn, &slots, params, 0),
            false,
        )
        .expect("commit step k");
    let mut parked = model
        .park_rows(
            &ctx,
            &s,
            &mut rows_of(&mut states, &drawn, &slots, params, 1),
            &step_k,
            false,
        )
        .expect("park step k + 1");
    let (draws, _) = model.finish_rows(&s, step_k).expect("finish step k");
    for (d, token) in drawn.iter_mut().zip(draws) {
        d.push(token);
    }
    // Row 0 is done; the parked step still feeds its final draw.
    model
        .release_rows(
            &s,
            &mut parked,
            &mut rows_of(&mut states, &drawn, &slots, params, 0),
        )
        .expect("release step k + 1");
    let (draws, _) = model.finish_rows(&s, parked).expect("drain step k + 1");
    for (d, token) in drawn[1..].iter_mut().zip(&draws[1..]) {
        d.push(*token);
    }
    assert_eq!(
        states[0].pos(),
        prompts[0].len() + drawn[0].len(),
        "the finished row's state holds every draw, the final one included"
    );
    assert_eq!(drawn[0], finished_draws[..drawn[0].len()], "row 0's draws");
    assert_eq!(
        draws[0],
        finished_draws[drawn[0].len()],
        "the dropped draw is what the unparked step drew"
    );
    assert!(
        fingerprint(&ctx, &states[0]) == finished_print,
        "row 0's caches or state differ from the unparked run's"
    );
    run_parked(
        &ctx,
        &model,
        &mut s,
        &mut states[1..],
        &mut drawn[1..],
        &slots[1..],
        params,
        more,
    );
    for (r, ((st, d), (du, pu))) in
        states[1..].iter().zip(&drawn[1..]).zip(&rest_u).enumerate()
    {
        assert_eq!(d, du, "row {}'s draws", r + 1);
        assert!(fingerprint(&ctx, st) == *pu, "row {}'s caches or state", r + 1);
    }
}
