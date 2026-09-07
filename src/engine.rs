//! The model-agnostic surface the generator, session cache, and API server
//! drive. Each supported architecture implements it over its own state and
//! scratch types; `serve::run` picks the implementation from the checkpoint's
//! `model_type`.

use std::path::Path;

use anyhow::Result;

use crate::metal::{MetalContext, PendingPass};
use crate::tensor::Tensor;

/// Per-session recurrent/cache state.
pub trait DecodeStateApi {
    /// Tokens fed so far.
    fn pos(&self) -> usize;
    /// Returns to position zero so the buffers can be recycled. The GPU must
    /// be idle on this state's buffers.
    fn reset(&mut self) -> Result<()>;
}

/// Per-engine intermediates.
pub trait ScratchApi {
    /// `U32[2]`: the ping-pong slots the in-graph argmax writes greedy tokens to.
    fn next_token(&self) -> &Tensor;
    /// `F32[vocab]`: the logits of the most recent step (prefill leaves the
    /// last prompt token's). Host reads need an idle GPU.
    fn logits(&self) -> &Tensor;
    /// Model-specific diagnostics of the most recent step for probes
    /// (e.g. the sparse-attention block selection). Host reads need an idle GPU.
    fn debug_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
}

pub trait LanguageModel: Sized {
    type State: DecodeStateApi;
    type Scratch: ScratchApi;

    /// The id the OpenAI-compatible API exposes.
    const MODEL_ID: &'static str;

    fn load(ctx: &MetalContext, dir: &Path) -> Result<Self>;

    /// Checkpoint-declared context window; zero means unspecified.
    fn max_position_embeddings(&self) -> usize;

    /// Stop ids the checkpoint config names (the tokenizer adds its own).
    fn eos_token_ids(&self) -> Vec<u32>;

    fn vocab_size(&self) -> usize;

    fn new_state(&self, ctx: &MetalContext, max_seq: usize) -> Result<Self::State>;

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Self::Scratch>;

    /// Feeds `tokens`, leaving the greedy token for the last one in
    /// `next_token[0]`.
    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        tokens: &[u32],
    ) -> Result<()>;

    /// Submits one decode step reading `next_token[slot_in]` and writing
    /// `next_token[slot_out]`, without waiting.
    fn submit_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Self::State,
        scratch: &Self::Scratch,
        slot_in: usize,
        slot_out: usize,
    ) -> Result<PendingPass<'a>>;
}
