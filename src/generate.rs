//! Checkpoint tokenization and the pipelined decode loop.
//!
//! The loop keeps one decode pass in flight while it encodes the next, and
//! commits the encoded pass only once the host has seen the token it consumes
//! (some models stage per-token host inputs, see
//! [`LanguageModel::prepare_step_inputs`]). Tokens are delivered to a callback
//! as they are drawn, so callers stream them and can stop early.
//!
//! A caller can also have recurrent-state checkpoints taken along the way
//! ([`DecodeCheckpointer`], [`Generator::generate_checkpointed`]): when one
//! is due, the loop lets its pipeline drain for one step, so that nothing is
//! in flight and no speculative step is pending, and hands the caller the
//! state at rest.
//!
//! A generation can carry thinking controls ([`crate::thinking`]): tokens
//! the control inserts are emitted and fed like drawn ones. Where it acts,
//! the loop comes to rest as for a checkpoint (nothing in flight, no
//! speculative step pending), feeds what was inserted the way a prompt is
//! fed ([`LanguageModel::prefill`]), and carries on from the last inserted
//! token, re-proposing with the draft head as after the prefill.

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Result, ensure};

use crate::chat::Conversation;
use crate::engine::{DecodeStateApi, Draw, LanguageModel, NextStep, ScratchApi};
use crate::kernels::sample::SamplingParams;
use std::time::Instant;

use crate::metal::{EncodedPass, MetalContext, Pacer, PendingPass};
use crate::thinking::{Action, ThinkingControl};
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
    /// The request's thinking controls; `None` decodes exactly as without
    /// them. Behind a `RefCell` because the control outlives one loop: a
    /// batch row moves between the single-session loop and batched steps
    /// with the same control.
    ///
    /// A draw is keyed by `(seed, index)`, and the index is the token's
    /// place in the generation, inserted tokens included: an insertion of
    /// `n` tokens skips `n` indices of the seed's stream. Every loop (plain,
    /// speculative, batched) counts the same way, so a seeded request with
    /// the same controls draws the same tokens whichever loop runs it; it
    /// draws differently from the same request without them past the first
    /// insertion, as it should.
    pub thinking: Option<&'a RefCell<ThinkingControl>>,
}

/// A generation's thinking control with the tokenizer that gives it the
/// text of each token.
pub struct ThinkingHook<'a> {
    pub control: &'a RefCell<ThinkingControl>,
    pub tokenizer: &'a Tokenizer,
}

impl ThinkingHook<'_> {
    /// [`ThinkingControl::decide`] on `token` as it is about to be emitted.
    pub fn decide(&self, token: u32) -> Result<Action> {
        // Once the block is closed nothing acts: no decode per token.
        if !self.control.borrow().is_open() {
            return Ok(Action::Keep);
        }
        let text = self.tokenizer.decode(&[token], false)?;
        Ok(self.control.borrow_mut().decide(token, &text))
    }

    /// [`ThinkingControl::may_act_next`].
    pub fn may_act_next(&self) -> bool {
        self.control.borrow().may_act_next()
    }
}

/// Pushes `group`, tokens the thinking control inserted (with, for a close
/// in front of a draw, that draw last), delivering each. A group is emitted
/// whole: a callback that asks to stop (a stop string, a departed client,
/// the batch scheduler's preemption, which resumes the request later) ends
/// the generation after the group, never inside it, so a resumed request
/// never finds half a close. Only `max_tokens` cuts it short. Returns why
/// the generation ends there, if it does.
pub fn push_group(
    tokens: &mut Vec<u32>,
    group: &[u32],
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
) -> Result<Option<FinishReason>> {
    let mut finish = None;
    for &token in group {
        if tokens.len() >= max_tokens {
            break;
        }
        tokens.push(token);
        if !on_token(token)? {
            finish.get_or_insert(FinishReason::Callback);
        }
    }
    if tokens.len() >= max_tokens {
        finish.get_or_insert(FinishReason::Length);
    }
    Ok(finish)
}

/// Emits `drawn`, a draw the thinking control acted on (`action` is not
/// [`Action::Keep`]), with what the control inserts: the close and then the
/// draw ([`Action::Before`]), or the draw and then the close
/// ([`Action::After`]). Returns why the generation ends, if it does.
pub fn emit_acted(
    action: Action,
    drawn: u32,
    tokens: &mut Vec<u32>,
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
) -> Result<Option<FinishReason>> {
    match action {
        Action::Keep => anyhow::bail!("emit_acted with nothing to insert"),
        Action::Before(mut group) => {
            group.push(drawn);
            push_group(tokens, &group, max_tokens, on_token)
        }
        Action::After(group) => {
            tokens.push(drawn);
            let stop = !on_token(drawn)?;
            let rest = push_group(tokens, &group, max_tokens, on_token)?;
            Ok(if stop { Some(FinishReason::Callback) } else { rest })
        }
    }
}

/// Feeds the tokens of `tokens` the state has not fed yet, all but the last
/// (the next step's input), as a prefill without a draw; `base` is the
/// state's position of `tokens[0]`. The state must be at rest. This is how
/// inserted tokens reach the state: exactly as a prompt's would, the draft
/// head's caches included, so the head can propose from the last one
/// ([`LanguageModel::draft_initial`]) as it does after the prefill.
pub fn feed_inserted<M: LanguageModel>(
    ctx: &MetalContext,
    model: &M,
    state: &mut M::State,
    scratch: &mut M::Scratch,
    tokens: &[u32],
    base: usize,
) -> Result<()> {
    let fed = state
        .pos()
        .checked_sub(base)
        .ok_or_else(|| anyhow::anyhow!("decode state behind its generation"))?;
    let rest = tokens.len().saturating_sub(1);
    ensure!(
        fed <= rest,
        "decode state fed {fed} of {} generated tokens before an insertion",
        tokens.len()
    );
    if fed < rest {
        model.prefill(ctx, state, scratch, &tokens[fed..rest], None)?;
    }
    ensure!(
        state.pos() == base + rest,
        "inserted tokens left the state at {} instead of {}",
        state.pos(),
        base + rest
    );
    Ok(())
}

/// Recurrent-state checkpoints taken during a generation (the session
/// cache's decode checkpoints). Before the loop pipelines the step that
/// would carry the state past a point of rest, it asks [`Self::due`] with
/// the number of tokens the state holds there; on a yes it lets the pipeline
/// drain at that point (no pass in flight, no step parked, no speculative
/// step pending) and calls [`Self::take`].
pub trait DecodeCheckpointer<S: DecodeStateApi> {
    /// Whether a checkpoint is due once the state holds `pos` tokens. It may
    /// be asked more than once for the same `pos` before [`Self::take`] and
    /// must give the same answer each time.
    fn due(&mut self, pos: usize) -> bool;
    /// Takes the checkpoint: `state` is at rest at a position [`Self::due`]
    /// accepted, holding the prompt and every drawn token but the last one
    /// (the next step's input).
    fn take(&mut self, ctx: &MetalContext, state: &S) -> Result<()>;
}

/// What [`speculate`] reports.
pub struct Speculated {
    pub finish: FinishReason,
    pub drafted: usize,
    pub accepted: usize,
}

/// What [`Generator::resume`] reports.
pub struct Resumed {
    pub finish: FinishReason,
    /// Speculative decoding: draft tokens proposed and confirmed.
    pub drafted: usize,
    pub accepted: usize,
    /// The decode loop ended with a step parked, which then ran on the final
    /// draw: the state has fed every token of `tokens`, and this is what
    /// that step drew (a valid next draw, already counted by the sampler).
    /// `None` when the final draw is not fed.
    pub parked_draw: Option<u32>,
}

/// Speculative decoding through a model's draft head, from a state that has
/// fed everything but the last of `tokens` (the draws so far, all already
/// delivered; the prefill's single draw for a new request). Each step
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
    speculate_checkpointed(
        ctx, model, state, scratch, params, drafts, max_tokens, tokens, is_stop, None,
        None, on_token,
    )
}

/// [`speculate`] with decode checkpoints. A step after which one is due
/// completes its verify pass without committing the next one
/// ([`LanguageModel::finish_speculation`] without a next step, as at the end
/// of a generation), so the state is at rest with the accepted rows fed; the
/// checkpoint is taken there and the head proposes afresh for the fresh draw
/// ([`LanguageModel::draft_initial`], as after the prefill). The emitted
/// tokens are the trunk's draws either way; only the proposals after the
/// restart can differ, which under sampling changes which draws a seed
/// realises, not their distribution.
///
/// With a thinking control, every confirmed row's token is put to it before
/// it is emitted. Where the control acts at row `j`, the step ends there as
/// a generation would (the state keeps the rows before `j`, and row `j`'s
/// token stays the unfed last one when the insertion follows it), the
/// inserted tokens are emitted and fed ([`feed_inserted`]), and the head
/// proposes afresh from the last of them, as after a checkpoint. A sampled
/// `<tool_call>` the control closes thinking in front of is thus fed after
/// the close, never before it.
#[allow(clippy::too_many_arguments)]
fn speculate_checkpointed<M: LanguageModel>(
    ctx: &MetalContext,
    model: &M,
    state: &mut M::State,
    scratch: &mut M::Scratch,
    params: &SamplingParams,
    drafts: usize,
    max_tokens: usize,
    tokens: &mut Vec<u32>,
    is_stop: &dyn Fn(u32) -> bool,
    mut checkpoints: Option<&mut dyn DecodeCheckpointer<M::State>>,
    thinking: Option<&ThinkingHook<'_>>,
    on_token: &mut dyn FnMut(u32) -> Result<bool>,
) -> Result<Speculated> {
    let k = drafts.min(model.max_drafts()).max(1);
    // The last drawn token is the one not yet fed: the prefill's draw for a
    // new request, or where a resumed request left off.
    let last =
        *tokens.last().ok_or_else(|| anyhow::anyhow!("speculation without a draw"))?;
    // The state holds the prompt and every draw but the last, here and at
    // every later point of rest: `base` is where the prompt ends.
    let base = state
        .pos()
        .checked_sub(tokens.len() - 1)
        .ok_or_else(|| anyhow::anyhow!("speculation from a state behind its draws"))?;
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut proposals =
        model.draft_initial(ctx, state, scratch, last, k, params, tokens.len())?;
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
        // Tokens the thinking control inserts where the step ends.
        let mut group: Option<Vec<u32>> = None;
        let callbacks_began = std::time::Instant::now();
        for (j, &token) in sampled.iter().enumerate() {
            let action = match thinking {
                Some(hook) if !is_stop(token) => hook.decide(token)?,
                _ => Action::Keep,
            };
            if let Action::Before(mut close) = action {
                // The state keeps the rows before this one; the token is
                // emitted, and fed, after the close.
                close.push(token);
                group = Some(close);
                kept = j;
                break;
            }
            tokens.push(token);
            if is_stop(token) {
                finish = Some(FinishReason::StopToken);
            } else if !on_token(token)? {
                finish = Some(FinishReason::Callback);
            } else if tokens.len() >= max_tokens {
                finish = Some(FinishReason::Length);
            }
            kept = j;
            if let Action::After(close) = action {
                // Emitted whole even when the callback asked to stop (see
                // `push_group`); only the length leaves no room for it.
                if finish != Some(FinishReason::Length) {
                    group = Some(close);
                }
                break;
            }
            if finish.is_some() || j >= proposals.len() || proposals[j] != token {
                break;
            }
        }
        drafted += proposals.len();
        accepted += kept;
        if std::env::var_os("LILY_PROFILE").is_some() {
            eprintln!(
                "profile host callbacks: {:.2} ms for {} tokens",
                callbacks_began.elapsed().as_secs_f64() * 1e3,
                kept + 1
            );
        }
        if let Some(group) = group {
            // At rest with the kept rows fed, as at the end of a generation.
            model.finish_speculation(ctx, state, scratch, kept, None, draft)?;
            if let Some(end) = push_group(tokens, &group, max_tokens, on_token)? {
                finish.get_or_insert(end);
            }
            feed_inserted(ctx, model, state, scratch, tokens, base)?;
            if let Some(finish) = finish {
                return Ok(Speculated { finish, drafted, accepted });
            }
            if let Some(c) = checkpoints.as_mut()
                && c.due(state.pos())
            {
                c.take(ctx, &*state)?;
            }
            let last = *tokens.last().expect("an inserted group is never empty");
            proposals = model.draft_initial(
                ctx,
                state,
                scratch,
                last,
                k,
                params,
                tokens.len(),
            )?;
            // Nothing is parked: `verify` encodes and commits the next pass.
            continue;
        }
        match finish {
            Some(finish) => {
                model.finish_speculation(ctx, state, scratch, kept, None, draft)?;
                return Ok(Speculated { finish, drafted, accepted });
            }
            None => {
                let rest = base + tokens.len() - 1;
                if checkpoints.as_mut().is_some_and(|c| c.due(rest)) {
                    model.finish_speculation(ctx, state, scratch, kept, None, draft)?;
                    ensure!(
                        state.pos() == rest,
                        "speculation came to rest at {} instead of {rest}",
                        state.pos()
                    );
                    if let Some(c) = checkpoints.as_mut() {
                        c.take(ctx, &*state)?;
                    }
                    proposals = model.draft_initial(
                        ctx,
                        state,
                        scratch,
                        sampled[kept],
                        k,
                        params,
                        tokens.len(),
                    )?;
                    // Nothing is parked: the next verify pass is encoded and
                    // committed by `verify` itself.
                    continue;
                }
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
        self.tokenizer.encode_conversation(messages, thinking)
    }

    pub fn decode_text(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, true)
    }

    /// Whether `token` ends a generation under `options`.
    pub fn is_stop(&self, token: u32, options: &GenerateOptions<'_>) -> bool {
        self.stop_tokens.contains(&token) || options.stop_tokens.contains(&token)
    }

    /// The thinking control of `options` with this tokenizer, if it has one.
    pub fn thinking_hook<'a>(
        &'a self,
        options: &GenerateOptions<'a>,
    ) -> Option<ThinkingHook<'a>> {
        options
            .thinking
            .map(|control| ThinkingHook { control, tokenizer: &self.tokenizer })
    }

    /// The start of a request: clears the per-request sampler state, feeds
    /// `prompt_ids` (the not yet cached suffix) and returns the first draw
    /// (draw 0 of the request, not fed). What [`Self::generate`] does before
    /// its loops, for a caller that runs the loops itself (the server's
    /// batch scheduler).
    pub fn begin<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &mut M::Scratch,
        prompt_ids: &[u32],
        params: &SamplingParams,
    ) -> Result<u32> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        scratch.begin_request();
        model.prefill(
            ctx,
            state,
            scratch,
            prompt_ids,
            Some(Draw { params, step: 0 }),
        )?;
        Ok(scratch.next_token().view(0, &[1])?.to_u32()?[0])
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
        self.generate_checkpointed(
            ctx, model, state, scratch, prompt_ids, options, None, on_token,
        )
    }

    /// [`Self::generate`], taking a checkpoint of the state whenever
    /// `checkpoints` says one is due ([`DecodeCheckpointer`]).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_checkpointed<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &mut M::Scratch,
        prompt_ids: &[u32],
        options: &GenerateOptions<'_>,
        checkpoints: Option<&mut dyn DecodeCheckpointer<M::State>>,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<Generation> {
        ensure!(!prompt_ids.is_empty(), "empty prompt");
        ensure!(options.max_tokens > 0, "max_tokens must be positive");
        let pos_before = state.pos();
        let first =
            self.begin(ctx, model, state, scratch, prompt_ids, options.sampling)?;

        let mut tokens = Vec::with_capacity(options.max_tokens.min(4096));
        let (mut drafted, mut accepted) = (0usize, 0usize);
        let action = match self.thinking_hook(options) {
            Some(hook) if !self.is_stop(first, options) => hook.decide(first)?,
            _ => Action::Keep,
        };
        let end = if action == Action::Keep {
            tokens.push(first);
            if self.is_stop(first, options) {
                Some(FinishReason::StopToken)
            } else if !on_token(first)? {
                Some(FinishReason::Callback)
            } else if tokens.len() >= options.max_tokens {
                Some(FinishReason::Length)
            } else {
                None
            }
        } else {
            // Thinking ends at the prefill's draw (a tool call right after
            // the generation prompt's `<think>\n`, or a budget of 1).
            let end =
                emit_acted(action, first, &mut tokens, options.max_tokens, on_token)?;
            feed_inserted(
                ctx,
                model,
                state,
                scratch,
                &tokens,
                pos_before + prompt_ids.len(),
            )?;
            end
        };
        let finish = match end {
            Some(end) => end,
            None => {
                let resumed = self.resume(
                    ctx,
                    model,
                    state,
                    scratch,
                    &mut tokens,
                    options,
                    checkpoints,
                    on_token,
                )?;
                drafted = resumed.drafted;
                accepted = resumed.accepted;
                resumed.finish
            }
        };
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

    /// Continues a generation from a state that has fed everything but the
    /// last of `tokens` (the draws so far, all delivered): the speculative
    /// loop when `options.drafts` and the model allow it, the pipelined decode
    /// loop otherwise, until a stop token, `options.max_tokens` draws in all
    /// or `on_token` returning false. New draws are appended to `tokens`.
    /// [`Self::generate`] runs this after the first draw; the server's batch
    /// scheduler runs it for a request that decodes alone, also after a
    /// batched stretch (the draw indices continue from `tokens.len()`).
    #[allow(clippy::too_many_arguments)]
    pub fn resume<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &mut M::Scratch,
        tokens: &mut Vec<u32>,
        options: &GenerateOptions<'_>,
        checkpoints: Option<&mut dyn DecodeCheckpointer<M::State>>,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
    ) -> Result<Resumed> {
        ensure!(!tokens.is_empty(), "a generation resumes after a draw");
        if tokens.len() >= options.max_tokens {
            return Ok(Resumed {
                finish: FinishReason::Length,
                drafted: 0,
                accepted: 0,
                parked_draw: None,
            });
        }
        let hook = self.thinking_hook(options);
        if options.drafts > 0 && model.max_drafts() > 0 {
            let is_stop = |t: u32| self.is_stop(t, options);
            let outcome = speculate_checkpointed(
                ctx,
                model,
                state,
                scratch,
                options.sampling,
                options.drafts,
                options.max_tokens,
                tokens,
                &is_stop,
                checkpoints,
                hook.as_ref(),
                on_token,
            )?;
            Ok(Resumed {
                finish: outcome.finish,
                drafted: outcome.drafted,
                accepted: outcome.accepted,
                parked_draw: None,
            })
        } else {
            // The state's position of `tokens[0]`: it holds all but the last.
            let base =
                (state.pos() + 1).checked_sub(tokens.len()).ok_or_else(|| {
                    anyhow::anyhow!("decoding from a state behind its draws")
                })?;
            let mut checkpoints = checkpoints;
            let mut parked_draw = None;
            let finish = loop {
                // The loop's first step reads its input from slot 0, where
                // the prefill's draw left it; anything since (another
                // request's prefill, batched steps, an insertion's prefill)
                // may have overwritten it. The GPU is idle here. After a
                // prefill with a draw this rewrites the value it holds.
                let last = *tokens.last().expect("checked above");
                scratch
                    .next_token()
                    .view(0, &[1])?
                    .write_bytes(bytemuck::cast_slice(&[last]))?;
                let exit = self.decode_loop(
                    ctx,
                    model,
                    state,
                    scratch,
                    options,
                    tokens,
                    checkpoints
                        .as_mut()
                        .map(|c| &mut **c as &mut dyn DecodeCheckpointer<M::State>),
                    hook.as_ref(),
                    on_token,
                    &mut parked_draw,
                )?;
                match exit {
                    LoopExit::Finished(finish) => break finish,
                    LoopExit::Inserted(end) => {
                        feed_inserted(ctx, model, state, scratch, tokens, base)?;
                        if let Some(finish) = end {
                            break finish;
                        }
                    }
                }
            };
            Ok(Resumed { finish, drafted: 0, accepted: 0, parked_draw })
        }
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
    ///
    /// When a decode checkpoint is due at the position the current step
    /// leaves the state at, the following step is not encoded ahead: once
    /// the current step completes nothing is in flight, and the top of the
    /// loop takes the checkpoint before it encodes the next step unparked.
    ///
    /// Likewise when the thinking control may act on the current step's
    /// draw ([`ThinkingHook::may_act_next`]): a step committed behind it
    /// would feed the draw before anything the control inserts. When the
    /// control acts, the loop emits the draw and the inserted tokens and
    /// returns [`LoopExit::Inserted`] with nothing in flight, for the caller
    /// to feed them and come back.
    #[allow(clippy::too_many_arguments)]
    fn decode_loop<M: LanguageModel>(
        &self,
        ctx: &MetalContext,
        model: &M,
        state: &mut M::State,
        scratch: &M::Scratch,
        options: &GenerateOptions<'_>,
        tokens: &mut Vec<u32>,
        mut checkpoints: Option<&mut dyn DecodeCheckpointer<M::State>>,
        thinking: Option<&ThinkingHook<'_>>,
        on_token: &mut dyn FnMut(u32) -> Result<bool>,
        parked_draw: &mut Option<u32>,
    ) -> Result<LoopExit> {
        let params = options.sampling;
        let is_stop =
            |t: u32| self.stop_tokens.contains(&t) || options.stop_tokens.contains(&t);
        let read_slot = |slot: usize| -> Result<u32> {
            Ok(scratch.next_token().view(slot, &[1])?.to_u32()?[0])
        };
        let parking = model.supports_parking();
        // The slot holding the next step's input token.
        let mut slot_in = 0usize;
        // A step encoded ahead of time (reads `slot_in`, writes the other
        // slot, at the state's current position): parked and committed, or
        // merely encoded.
        let mut parked: Option<PendingPass<'_>> = None;
        let mut ahead: Option<EncodedPass<'_>> = None;
        // On any error a parked pass must not be left blocking the queue.
        // Declared after `parked` so that it drops first: dropping a pending
        // pass waits for it, and a parked one never finishes unreleased.
        let _release = ReleaseOnExit { model, scratch };
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
            // Nothing in flight and nothing encoded: the state is at rest,
            // holding every drawn token but `input`.
            if parked.is_none() && ahead.is_none() {
                let pos = state.pos();
                if let Some(c) = checkpoints.as_mut()
                    && c.due(pos)
                {
                    c.take(ctx, &*state)?;
                }
            }
            let step = tokens.len();
            let pending = match parked.take() {
                Some(pass) => {
                    // Released before a staging error propagates: `pass`
                    // would wait for itself when dropped.
                    let staged = model.prepare_step_inputs(state, scratch, input);
                    model.release_parked(scratch)?;
                    staged?;
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
            let pos = state.pos();
            let rest_due = checkpoints.as_mut().is_some_and(|c| c.due(pos))
                || thinking.is_some_and(ThinkingHook::may_act_next);
            if tokens.len() + 1 < options.max_tokens
                && state.pos() + 1 < state.capacity()
                && !rest_due
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
            if let Some(hook) = thinking
                && !is_stop(drawn)
            {
                let action = hook.decide(drawn)?;
                if action != Action::Keep {
                    ensure!(
                        parked.is_none() && ahead.is_none(),
                        "the thinking control acted with a step committed ahead"
                    );
                    let end = emit_acted(
                        action,
                        drawn,
                        tokens,
                        options.max_tokens,
                        on_token,
                    )?;
                    return Ok(LoopExit::Inserted(end));
                }
            }
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
                    // the final token rather than leave the queue blocked
                    // (released before a staging error propagates, as above).
                    let staged = model.prepare_step_inputs(state, scratch, drawn);
                    model.release_parked(scratch)?;
                    staged?;
                    pass.wait()?;
                    // It drew the token after `drawn` (draw `tokens.len()`)
                    // into the other slot: a caller that continues the
                    // generation later takes it as its next draw.
                    *parked_draw = Some(read_slot(1 - slot_in)?);
                }
                return Ok(LoopExit::Finished(finish));
            }
        }
        Ok(LoopExit::Finished(FinishReason::Length))
    }
}

/// How [`Generator::decode_loop`] returned.
enum LoopExit {
    Finished(FinishReason),
    /// The thinking control acted: the draw and the inserted tokens were
    /// emitted, the state is at rest without them, and the generation ends
    /// once they are fed when this says why.
    Inserted(Option<FinishReason>),
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
