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
use crate::engine::{DecodeStateApi, Draw, LanguageModel, ScratchApi};
use crate::kernels::sample::SamplingParams;
use crate::metal::MetalContext;
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
        model.prefill(ctx, state, scratch, prompt_ids, Some(Draw { params, step: 0 }))?;

        let is_stop = |t: u32| self.stop_tokens.contains(&t) || options.stop_tokens.contains(&t);
        let read_slot = |scratch: &M::Scratch, slot: usize| -> Result<u32> {
            Ok(scratch.next_token().view(slot, &[1])?.to_u32()?[0])
        };

        let mut tokens = Vec::with_capacity(options.max_tokens.min(4096));
        let first = read_slot(scratch, 0)?;
        tokens.push(first);
        let mut finish = FinishReason::Length;
        if is_stop(first) {
            finish = FinishReason::StopToken;
        } else if !on_token(first)? {
            finish = FinishReason::Callback;
        } else if tokens.len() < options.max_tokens {
            finish = self.decode_loop(ctx, model, state, scratch, options, &mut tokens, on_token)?;
        }
        let fed = state
            .pos()
            .checked_sub(pos_before)
            .ok_or_else(|| anyhow::anyhow!("decode state moved backwards"))?;
        debug_assert_eq!(fed, prompt_ids.len() + tokens.len() - 1);
        Ok(Generation { tokens, finish, fed })
    }

    /// The pipelined loop proper. `tokens` holds the tokens drawn so far, the
    /// last of which is the input of the next step; returns why it stopped.
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
        let is_stop = |t: u32| self.stop_tokens.contains(&t) || options.stop_tokens.contains(&t);
        let read_slot = |slot: usize| -> Result<u32> {
            Ok(scratch.next_token().view(slot, &[1])?.to_u32()?[0])
        };
        // The slot holding the next step's input token.
        let mut slot_in = 0usize;
        // A step encoded ahead of time: reads `slot_in`, writes the other
        // slot, at the state's current position.
        let mut ahead = None;
        while tokens.len() < options.max_tokens {
            let input = *tokens.last().expect("tokens holds the prefill draw");
            // The GPU is idle here (every committed pass has been waited on),
            // so the caches may grow. An encoded-ahead pass would reference
            // the old buffers and is discarded.
            if state.pos() >= state.capacity() {
                ahead = None;
                state.ensure_capacity(ctx, state.pos() + 1)?;
            }
            let step = tokens.len();
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
            let pending = encoded.commit()?;
            state.advance(1);
            // Encode the following step while this one runs, when there will
            // be one and the caches already have room for it.
            if tokens.len() + 1 < options.max_tokens && state.pos() < state.capacity() {
                ahead = Some(model.encode_decode_step(
                    ctx,
                    state,
                    scratch,
                    1 - slot_in,
                    slot_in,
                    Draw { params, step: step + 1 },
                )?);
            }
            pending.wait()?;
            slot_in = 1 - slot_in;
            let drawn = read_slot(slot_in)?;
            tokens.push(drawn);
            if is_stop(drawn) {
                return Ok(FinishReason::StopToken);
            }
            if !on_token(drawn)? {
                return Ok(FinishReason::Callback);
            }
        }
        Ok(FinishReason::Length)
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
