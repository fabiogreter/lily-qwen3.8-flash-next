//! The model-agnostic surface the generator, session cache, and API server
//! drive. Each supported architecture implements it over its own state and
//! scratch types; `serve::run` picks the implementation from the checkpoint's
//! `model_type`.

use std::path::Path;

use anyhow::Result;

use crate::kernels::sample::SamplingParams;
use crate::metal::{EncodedPass, MetalContext};
use crate::tensor::Tensor;

/// A copy of the recurrent part of a decode state (everything that is not a
/// per-token cache) at one position. Together with the per-token caches that
/// are still in place up to that position it lets a state resume from there.
pub trait SnapshotApi {
    /// Tokens fed when the snapshot was taken.
    fn pos(&self) -> usize;
    /// GPU bytes the snapshot holds.
    fn bytes(&self) -> usize;
}

/// Per-session recurrent/cache state.
pub trait DecodeStateApi: Sized {
    type Snapshot: SnapshotApi;

    /// Tokens fed so far.
    fn pos(&self) -> usize;

    /// Records that `n` more tokens were fed (the caller committed the
    /// passes that write them).
    fn advance(&mut self, n: usize);

    /// Returns to position zero so the buffers can be recycled. The GPU must
    /// be idle on this state's buffers.
    fn reset(&mut self) -> Result<()>;

    /// Tokens the per-token caches can hold before [`Self::ensure_capacity`]
    /// has to grow them.
    fn capacity(&self) -> usize;

    /// Grows the per-token caches to hold at least `tokens`, copying the
    /// live prefix. The GPU must be idle on this state's buffers.
    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()>;

    /// GPU bytes this state holds (caches at capacity plus recurrent state).
    fn bytes(&self) -> usize;

    /// Copies the recurrent state at the current position. GPU idle.
    fn snapshot(&self, ctx: &MetalContext) -> Result<Self::Snapshot>;

    /// Rewinds to `snapshot`'s position: recurrent state from the snapshot,
    /// per-token caches kept (they are valid up to that position). GPU idle.
    fn restore(&mut self, ctx: &MetalContext, snapshot: &Self::Snapshot) -> Result<()>;

    /// Copies the first `tokens` entries of every per-token cache from
    /// `from` (which must have fed at least that many). GPU idle; the caller
    /// follows up with [`Self::restore`] to set the recurrent part and position.
    fn copy_prefix_from(&mut self, ctx: &MetalContext, from: &Self, tokens: usize) -> Result<()>;
}

/// Per-engine intermediates.
pub trait ScratchApi {
    /// `U32[2]`: the ping-pong slots the in-graph sampler writes tokens to.
    fn next_token(&self) -> &Tensor;
    /// `F32[vocab]`: the logits of the most recent step (prefill leaves the
    /// last prompt token's). Host reads need an idle GPU.
    fn logits(&self) -> &Tensor;
    /// Clears per-request sampler state (the repetition histogram). GPU idle.
    fn begin_request(&self);
    /// Model-specific diagnostics of the most recent step for probes
    /// (e.g. the sparse-attention block selection). Host reads need an idle GPU.
    fn debug_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
}

/// Engine-wide load options; architectures ignore what does not apply.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    /// Where the Qwen3.8-Flash-Next n-gram table lives.
    pub ngram_storage: crate::qwen4exp::NgramStorage,
}

/// Which draw a decode pass ends with.
#[derive(Clone, Copy, Debug)]
pub struct Draw<'p> {
    pub params: &'p SamplingParams,
    /// Index of this draw within the request (the RNG counter).
    pub step: usize,
}

pub trait LanguageModel: Sized {
    type State: DecodeStateApi;
    type Scratch: ScratchApi;

    /// The id the OpenAI-compatible API exposes.
    const MODEL_ID: &'static str;

    fn load(ctx: &MetalContext, dir: &Path, options: &LoadOptions) -> Result<Self>;

    /// Checkpoint-declared context window; zero means unspecified.
    fn max_position_embeddings(&self) -> usize;

    /// Stop ids the checkpoint config names (the tokenizer adds its own).
    fn eos_token_ids(&self) -> Vec<u32>;

    fn vocab_size(&self) -> usize;

    /// Bytes the per-token caches take per token of capacity, for budgeting.
    fn bytes_per_token(&self) -> usize;

    /// Warms weights served from disk (the paged n-gram table) and returns
    /// the bytes touched; models without such weights return 0.
    fn warm_storage(&self) -> Result<u64> {
        Ok(0)
    }

    /// A state with capacity for `capacity` tokens (grown later on demand).
    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Self::State>;

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Self::Scratch>;

    /// Feeds `tokens` and waits. With `draw`, the last token's logits are
    /// sampled into `next_token[0]`; without, no logits are produced (used to
    /// fill a cache prefix). Grows the state's capacity as needed.
    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()>;

    /// Host work for the step that consumes `token` at the current position
    /// (e.g. staging its n-gram rows). Must run after the pass that produced
    /// `token` completed and before the consuming step is committed.
    fn prepare_step_inputs(
        &self,
        state: &mut Self::State,
        scratch: &Self::Scratch,
        token: u32,
    ) -> Result<()>;

    /// Encodes one decode step at the state's position reading
    /// `next_token[slot_in]`, drawing into `next_token[slot_out]`, without
    /// committing. The caller commits it after [`Self::prepare_step_inputs`]
    /// and then calls [`DecodeStateApi::advance`]. The state must have
    /// capacity for one more token.
    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Self::State,
        scratch: &Self::Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>>;
}
