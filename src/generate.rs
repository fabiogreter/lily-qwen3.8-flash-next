//! Checkpoint tokenization and the pipelined decode loop.
//!
//! The loop keeps one decode pass in flight while it encodes the next, and
//! commits the encoded pass only once the host has seen the token it consumes
//! (some models stage per-token host inputs, see
//! [`LanguageModel::prepare_step_inputs`]). Tokens are delivered to a callback
//! as they are drawn, so callers stream them and can stop early.

use std::path::Path;

use anyhow::{Result, ensure};

use crate::chat::Conversation;
use crate::engine::{DecodeStateApi, Draw, LanguageModel, NextStep, ScratchApi};
use crate::kernels::sample::SamplingParams;
use std::time::Instant;

use crate::metal::{EncodedPass, MetalContext, Pacer, PendingPass};
pub use crate::tokenizer::Thinking;
use crate::tokenizer::Tokenizer;

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A configured stop token was drawn (it is the last token returned).
    StopToken,
    /// `max_tokens` tokens were drawn.
    Length,
    /// The token callback asked to stop (stop string, client gone, ...).
    Callback,
}

pub struct Generation {
    pub tokens: Vec<u32>,
    pub finish: FinishReason,
    /// Number of prompt and generated tokens actually fed into the state
    /// (the final generated token is drawn but never fed).
    pub fed: usize,
    /// Speculative decoding: draft tokens proposed and confirmed.
    pub drafted: usize,
    pub accepted: usize,
}

fn ends_with_stop_token(tokens: &[u32], stop_tokens: &[u32]) -> bool {
    tokens.last().is_some_and(|token| stop_tokens.contains(token))
}

/// One generation's settings.
pub struct GenerateOptions<'a> {
    pub max_tokens: usize,
    pub sampling: &'a SamplingParams,
    /// Ids that end the generation, in addition to the tokenizer's.
    pub stop_tokens: &'a [u32],
    /// Draft tokens per speculative step (capped by the model; `0` decodes
    /// one token per step).
    pub drafts: usize,
}

/// What [`speculate`] reports.
pub struct Speculated {
    pub finish: FinishReason,
    pub drafted: usize,
    pub accepted: usize,
}

/// Speculative decoding through a model's draft head, from a state that has
/// fed its prompt and drawn `tokens[0]` (already delivered). Each step
/// verifies the pending token plus the current drafts in one pass, emits the
/// confirmed prefix and one fresh draw, then rolls back and proposes again.
/// Tokens reach `on_token` as they are confirmed; `is_stop` ends the
/// generation at that token (which is still pushed).
#[allow(clippy::too_many_arguments)]
pub fn speculate<M: LanguageModel>(
    ctx: &MetalContext,
    model: &M,
    state: &mut M::State,
    scratch: &mut M::Scratch,
    params: &SamplingParams,
    drafts: usize,
    max_tokens: usize,
    tokens: &mut Vec<u32>,
    is_stop: &dyn Fn(u32) -> bool,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
) -> Result<Speculated> {
    let k = drafts.min(model.max_drafts()).max(1);
    ensure!(tokens.len() == 1, "speculation starts right after the first draw");
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut proposals = model.draft_initial(ctx, state, scratch, tokens[0], k)?;
    // The next verify pass, committed by finish_speculation and parked on the
    // GPU until verify stages its n-gram rows.
    let mut parked: Option<PendingPass<'_>> = None;
    loop {
        let pending = *tokens.last().expect("tokens is never empty");
        let step0 = tokens.len();
        let (sampled, draft) = model.verify(
            ctx,
            state,
            scratch,
            pending,
            &proposals,
            params,
            step0,
            parked.take(),
            k,
        )?;
        if std::env::var_os("LILY_TRACE_SPEC").is_some() {
            eprintln!(
                "trace step0={step0} pending={pending} proposals={proposals:?} sampled={sampled:?}"
            );
        }
        // Row j confirms draft j when its draw equals it; the first row that
        // does not (or the row after the last draft) supplies the fresh token.
        let mut kept = 0usize;
        let mut finish = None;
        for (j, &token) in sampled.iter().enumerate() {
            tokens.push(token);
            if is_stop(token) {
                finish = Some(FinishReason::StopToken);
            } else if !on_token(token)? {
                finish = Some(FinishReason::Callback);
            } else if tokens.len() >= max_tokens {
                finish = Some(FinishReason::Length);
            }
            kept = j;
            if finish.is_some() || j >= proposals.len() || proposals[j] != token {
                break;
            }
        }
        drafted += proposals.len();
        accepted += kept;
        match finish {
            Some(finish) => {
                model.finish_speculation(ctx, state, scratch, kept, None, draft)?;
                return Ok(Speculated { finish, drafted, accepted });
            }
            None => {
                let next =
                    NextStep { token: sampled[kept], params, step0: tokens.len() };
                let (next_proposals, next_parked) = model.finish_speculation(
                    ctx,
                    state,
                    scratch,
                    kept,
                    Some(next),
                    draft,
                )?;
                proposals = next_proposals;
                parked = next_parked;
            }
        }
    }
}

pub struct Generator {
    tokenizer: Tokenizer,
    stop_tokens: Vec<u32>,
}

impl Generator {
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_model_dir(dir)?;
        let stop_tokens = tokenizer.stop_tokens().to_vec();
        Ok(Self { tokenizer, stop_tokens })
    }

    pub fn add_stop_tokens(&mut self, ids: &[u32]) {
        for &id in ids {
            if !self.stop_tokens.contains(&id) {
                self.stop_tokens.push(id);
            }
        }
    }

    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn encode_chat(
        &self,
        messages: &Conversation,
        thinking: Thinking,
    ) -> Result<Vec<u32>> {
        ensure!(!messages.is_empty(), "empty conversation");
        self.tokenizer.encode(&self.tokenizer.render_chat(messages, thinking)?)
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
    }

    /// Feeds `prompt_ids` (the not yet cached suffix) and generates greedily
    /// or by sampling, delivering each token to `on_token`; a `false` return
    /// stops after that token.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &mut M::Scratch,
        prompt_ids: &[u32],
        options: &GenerateOptions<'_>,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<Generation> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        ensure!(options.max_tokens > 0, "max_tokens must be positive");
        let pos_before = state.pos();
        scratch.begin_request();
        let params = options.sampling;
        model.prefill(
            ctx,
            state,
            scratch,
            prompt_ids,
            Some(Draw { params, step: 0 }),
        )?;

        let is_stop =
            |t: u32| self.stop_tokens.contains(&t) || options.stop_tokens.contains(&t);
        let read_slot = |scratch: &M::Scratch, slot: usize| -> Result<u32> {
            Ok(scratch.next_token().view(slot, &[1])?.to_u32()?[0])
        };

        let mut tokens = Vec::with_capacity(options.max_tokens.min(4096));
        let first = read_slot(scratch, 0)?;
        tokens.push(first);
        let mut finish = FinishReason::Length;
        let (mut drafted, mut accepted) = (0usize, 0usize);
        if is_stop(first) {
            finish = FinishReason::StopToken;
        } else if !on_token(first)? {
            finish = FinishReason::Callback;
        } else if tokens.len() < options.max_tokens {
            if options.drafts > 0 && model.max_drafts() > 0 {
                let outcome = speculate(
                    ctx,
                    model,
                    state,
                    scratch,
                    params,
                    options.drafts,
                    options.max_tokens,
                    &mut tokens,
                    &is_stop,
                    on_token,
                )?;
                finish = outcome.finish;
                drafted = outcome.drafted;
                accepted = outcome.accepted;
            } else {
                finish = self.decode_loop(
                    ctx,
                    model,
                    state,
                    scratch,
                    options,
                    &mut tokens,
                    on_token,
                )?;
            }
        }
        let fed = state
            .pos()
            .checked_sub(pos_before)
            .ok_or_else(|| anyhow::anyhow!("decode state moved backwards"))?;
        // The last drawn token is fed only when a parked step consumed it.
        let drawn = prompt_ids.len() + tokens.len();
        ensure!(
            fed == drawn - 1 || fed == drawn,
            "state fed {fed} tokens for {} prompt and {} drawn",
            prompt_ids.len(),
            tokens.len()
        );
        Ok(Generation { tokens, finish, fed, drafted, accepted })
    }

    /// The pipelined loop proper. `tokens` holds the tokens drawn so far, the
    /// last of which is the input of the next step; returns why it stopped.
    ///
    /// Two protocols, chosen by [`LanguageModel::supports_parking`]. Parking:
    /// step N+1 is encoded and committed while step N runs and waits on the
    /// GPU for its host inputs; once N's token is read and staged it is
    /// released, so the command-buffer submission never sits between two
    /// steps. When the generation stops with a step parked, that step runs
    /// anyway (fed with the final token, which a continued conversation wants
    /// in the state). Without parking the next step is only encoded ahead and
    /// committed after staging.
    #[allow(clippy::too_many_arguments)]
    fn decode_loop<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &M::Scratch,
        options: &GenerateOptions<'_>,
        tokens: &mut Vec<u32>,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<FinishReason> {
        let params = options.sampling;
        let is_stop =
            |t: u32| self.stop_tokens.contains(&t) || options.stop_tokens.contains(&t);
        let read_slot = |slot: usize| -> Result<u32> {
            Ok(scratch.next_token().view(slot, &[1])?.to_u32()?[0])
        };
        let parking = model.supports_parking();
        // On any error a parked pass must not be left blocking the queue.
        let _release = ReleaseOnExit { model, scratch };
        // The slot holding the next step's input token.
        let mut slot_in = 0usize;
        // A step encoded ahead of time (reads `slot_in`, writes the other
        // slot, at the state's current position): parked and committed, or
        // merely encoded.
        let mut parked: Option<PendingPass<'_>> = None;
        let mut ahead: Option<EncodedPass<'_>> = None;
        // Sleep-then-poll waits: the step interval is regular enough to
        // predict, and polling wakes ~0.1 ms sooner than a blocked thread.
        let mut pacer = Pacer::default();
        while tokens.len() < options.max_tokens {
            let input = *tokens.last().expect("tokens holds the prefill draw");
            // The GPU is idle here (every committed pass has been waited on
            // and nothing was parked past the capacity), so the caches may
            // grow. An encoded-ahead pass would reference the old buffers
            // and is discarded.
            if state.pos() >= state.capacity() {
                ensure!(
                    parked.is_none(),
                    "parked step beyond the state's capacity (position {}, capacity {}, {} of {} tokens drawn)",
                    state.pos(),
                    state.capacity(),
                    tokens.len(),
                    options.max_tokens
                );
                ahead = None;
                state.ensure_capacity(ctx, state.pos() + 1)?;
            }
            let step = tokens.len();
            let pending = match parked.take() {
                Some(pass) => {
                    model.prepare_step_inputs(state, scratch, input)?;
                    model.release_parked(scratch)?;
                    pass
                }
                None => {
                    let encoded = match ahead.take() {
                        Some(pass) => pass,
                        None => model.encode_decode_step(
                            ctx,
                            state,
                            scratch,
                            slot_in,
                            1 - slot_in,
                            Draw { params, step },
                        )?,
                    };
                    model.prepare_step_inputs(state, scratch, input)?;
                    let pass = encoded.commit()?;
                    state.advance(1);
                    pacer.begin(Instant::now());
                    pass
                }
            };
            // Encode the following step while this one runs, when there will
            // be one and the caches have room for it and for the step after
            // it: a parked step is committed against the current cache
            // buffers, and the top of the loop must be able to grow the
            // caches without one outstanding. Prompts that end within a few
            // hundred tokens below a capacity step hit this every time;
            // skipping the pipelining for one step there costs nothing.
            if tokens.len() + 1 < options.max_tokens
                && state.pos() + 1 < state.capacity()
            {
                let draw = Draw { params, step: step + 1 };
                if parking {
                    let pass = model
                        .encode_parked_step(
                            ctx,
                            state,
                            scratch,
                            1 - slot_in,
                            slot_in,
                            draw,
                        )?
                        .commit()?;
                    state.advance(1);
                    parked = Some(pass);
                } else {
                    ahead = Some(model.encode_decode_step(
                        ctx,
                        state,
                        scratch,
                        1 - slot_in,
                        slot_in,
                        draw,
                    )?);
                }
            }
            pending.wait_paced(&mut pacer)?;
            // A parked step started running the moment this one finished.
            if parked.is_some() {
                pacer.begin(Instant::now());
            }
            slot_in = 1 - slot_in;
            let drawn = read_slot(slot_in)?;
            tokens.push(drawn);
            let finish = if is_stop(drawn) {
                Some(FinishReason::StopToken)
            } else if !on_token(drawn)? {
                Some(FinishReason::Callback)
            } else {
                None
            };
            if let Some(finish) = finish {
                if let Some(pass) = parked.take() {
                    // Already committed and counted as fed: let it consume
                    // the final token rather than leave the queue blocked.
                    model.prepare_step_inputs(state, scratch, drawn)?;
                    model.release_parked(scratch)?;
                    pass.wait()?;
                }
                return Ok(finish);
            }
        }
        Ok(FinishReason::Length)
    }
}

/// Releases a parked decode step when the loop exits early (an error), so
/// the committed pass cannot hold the command queue forever.
struct ReleaseOnExit<'m, M: LanguageModel> {
    model: &'m M,
    scratch: &'m M::Scratch,
}

impl<M: LanguageModel> Drop for ReleaseOnExit<'_, M> {
    fn drop(&mut self) {
        let _ = self.model.release_parked(self.scratch);
    }
}

/// Whether `tokens` ends in one of `stop_tokens` (the pre-streaming check
/// kept for the unit test and for callers that collect first).
pub fn stopped_by_token(tokens: &[u32], stop_tokens: &[u32]) -> bool {
    ends_with_stop_token(tokens, stop_tokens)
}

#[cfg(test)]
#[path = "../tests/unit/generate.rs"]
mod tests;
