//! Step-by-step forward probes of the Qwen3.8-Flash-Next graph for the
//! reference comparisons (`tools/reference/compare.py`, comparison 4 of
//! `docs/architecture.md`, "How the tower was verified"): the top logits a prompt produces at chosen
//! positions and over a greedy continuation, with or without an image in the
//! prompt. Shared by `lily-vision-probe --forward` and the model-gated test,
//! so the check also runs from `cargo test` without Python.
//!
//! Logits at a prompt position come from a fresh prefill of the prompt up to
//! and including it, in one chunk, which is exactly how the engine computes
//! a prompt of that length (and, for the last position, exactly what
//! `lily-probe` records). The greedy continuation runs the single-token
//! decode graph, as `lily-probe` does.

use std::time::Instant;

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::engine::{DecodeStateApi, Draw, ScratchApi};
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;

use super::model::{Qwen4ExpModel, VisionInput};

/// The logits of one position, in the `lily-probe` record layout.
#[derive(Clone, Debug, Serialize)]
pub struct ProbeStep {
    /// Sequence index the logits belong to.
    pub position: usize,
    /// The argmax (the greedy draw).
    pub chosen: u32,
    /// The `top` ids by logit, descending (ties by id).
    pub ids: Vec<u32>,
    pub logits: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct ForwardProbe {
    /// Ascending by position: the requested prompt positions, the last
    /// prompt token, then the greedy continuation.
    pub steps: Vec<ProbeStep>,
    /// The full prompt's prefill (host wall time, one call).
    pub prefill_seconds: f64,
    /// The greedy decode steps together.
    pub decode_seconds: f64,
}

/// The `k` largest logits, descending, ties broken by the lower id.
pub fn top_k(logits: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).expect("finite").then(a.cmp(&b))
    });
    let ids: Vec<u32> = order[..k.min(order.len())].iter().map(|&i| i as u32).collect();
    let values = ids.iter().map(|&i| logits[i as usize]).collect();
    (ids, values)
}

/// Runs `tokens` through `model` with `vision` (the prompt's positions and
/// image rows; `None` for text): a fresh prefill up to each of `positions`
/// (prompt indices below `tokens.len() - 1`), then the whole prompt with a
/// greedy draw followed by `greedy_steps` decode steps. Every step records
/// the `top` logits and the argmax.
pub fn forward_probe(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    tokens: &[u32],
    vision: Option<&VisionInput<'_>>,
    positions: &[usize],
    greedy_steps: usize,
    top: usize,
) -> Result<ForwardProbe> {
    let n = tokens.len();
    ensure!(n > 0, "empty prompt");
    let greedy = SamplingParams::greedy();
    let capacity = n + greedy_steps + 1;
    let mut state = model.new_state(ctx, capacity)?;
    let mut scratch = model.new_scratch_with_capacity(ctx, capacity)?;

    let read_step = |scratch: &super::Scratch, slot: usize, position: usize| {
        let chosen = scratch.next_token().view(slot, &[1])?.to_u32()?[0];
        let logits = scratch.logits().to_f32()?;
        let (ids, values) = top_k(&logits, top);
        Ok::<_, anyhow::Error>(ProbeStep { position, chosen, ids, logits: values })
    };

    let mut wanted: Vec<usize> =
        positions.iter().copied().filter(|&p| p < n - 1).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let mut steps = Vec::with_capacity(wanted.len() + 1 + greedy_steps);
    for &p in &wanted {
        state.reset()?;
        scratch.begin_request();
        model.prefill_with_vision(
            ctx,
            &mut state,
            &mut scratch,
            &tokens[..=p],
            Some(Draw { params: &greedy, step: 0 }),
            vision,
        )?;
        steps.push(read_step(&scratch, 0, p)?);
    }

    state.reset()?;
    scratch.begin_request();
    let started = Instant::now();
    model.prefill_with_vision(
        ctx,
        &mut state,
        &mut scratch,
        tokens,
        Some(Draw { params: &greedy, step: 0 }),
        vision,
    )?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    steps.push(read_step(&scratch, 0, n - 1)?);

    let started = Instant::now();
    let mut slot = 0usize;
    for step in 1..=greedy_steps {
        // Synchronous steps: each one's logits are read before the next runs.
        let input = steps.last().map(|s| s.chosen).expect("prefill step");
        let encoded = model.encode_decode_step(
            ctx,
            &state,
            &scratch,
            slot,
            1 - slot,
            Draw { params: &greedy, step },
            None,
        )?;
        model.prepare_step_inputs(&mut state, &scratch, input)?;
        let pending = encoded.commit()?;
        state.advance(1);
        pending.wait()?;
        slot = 1 - slot;
        steps.push(read_step(&scratch, slot, n - 1 + step)?);
    }
    let decode_seconds = started.elapsed().as_secs_f64();
    Ok(ForwardProbe { steps, prefill_seconds, decode_seconds })
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/probe.rs"]
mod tests;
