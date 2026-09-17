//! The model-agnostic surface the generator, session cache, and API server
//! drive. Each supported architecture implements it over its own state and
//! scratch types; `serve::run` picks the implementation from the checkpoint's
//! `model_type`.

use std::path::Path;

use anyhow::Result;

use crate::kernels::sample::SamplingParams;
use crate::metal::{EncodedPass, MetalContext, PendingPass, SharedEvent};
use crate::tensor::Tensor;

/// A copy of the recurrent part of a decode state (everything that is not a
/// per-token cache) at one position. Together with the per-token caches that
/// are still in place up to that position it lets a state resume from there.
pub trait SnapshotApi {
    /// Tokens fed when the snapshot was taken.
    fn pos(&self) -> usize;
    /// GPU bytes the snapshot holds.
    fn bytes(&self) -> usize;
    /// Serializes the snapshot (for the on-disk session tier; models that
    /// support it also implement [`LanguageModel::read_snapshot`]). GPU idle.
    fn write_to(&self, w: &mut dyn std::io::Write) -> Result<()> {
        let _ = w;
        anyhow::bail!("this model does not persist sessions")
    }
}

/// Per-session recurrent/cache state.
pub trait DecodeStateApi: Sized {
    type Snapshot: SnapshotApi;

    /// Tokens fed so far.
    fn pos(&self) -> usize;

    /// Records that `n` more tokens were fed (the caller committed the
    /// passes that write them).
    fn advance(&mut self, n: usize);

    /// Sets what every token fed from now on adds to its sequence index to
    /// get its rotary position: the prompt's `rope_delta`
    /// ([`crate::qwen4exp::Positions`]), 0 for text. A pure function of the
    /// prompt, so the engine sets it whenever it acquires a session, and a
    /// resumed session decodes at the right positions whatever state it was
    /// restored from. Models without image positions accept only 0.
    fn set_rope_delta(&mut self, delta: i64) -> Result<()> {
        anyhow::ensure!(
            delta == 0,
            "this model has no image positions (rope delta {delta})"
        );
        Ok(())
    }

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
    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()>;

    /// Streams the first `tokens` entries of every per-token cache to `w`, in
    /// the layout [`Self::read_prefix`] expects. GPU idle.
    fn write_prefix(&self, tokens: usize, w: &mut dyn std::io::Write) -> Result<()> {
        let _ = (tokens, w);
        anyhow::bail!("this model does not persist sessions")
    }

    /// Fills the first `tokens` entries of every per-token cache from `r`,
    /// which holds what [`Self::write_prefix`] wrote for `written` tokens
    /// (`tokens <= written`): the layout is region by region, so a reader
    /// that wants a shorter prefix has to skip each region's tail rather
    /// than stop early. Grows the capacity as needed. GPU idle; follow up
    /// with [`Self::restore`].
    fn read_prefix(
        &mut self,
        ctx: &MetalContext,
        written: usize,
        tokens: usize,
        r: &mut dyn std::io::Read,
    ) -> Result<()> {
        let _ = (ctx, written, tokens, r);
        anyhow::bail!("this model does not persist sessions")
    }
}

/// Per-engine intermediates.
pub trait ScratchApi {
    /// Diagnostics: an event the GPU raises when a parked step reaches its
    /// wait (`LILY_PROBE_ARRIVAL`), if the model provides one.
    fn arrival_probe(&self) -> Option<&SharedEvent> {
        None
    }

    /// Diagnostics: an event the GPU raises right after that wait.
    fn resumed_probe(&self) -> Option<&SharedEvent> {
        None
    }

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

/// Whether to load a checkpoint's vision tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VisionMode {
    /// Load the tower when the checkpoint carries one (default).
    #[default]
    Auto,
    /// Leave it on disk and save its memory; image requests are refused.
    Off,
}

impl std::str::FromStr for VisionMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            other => anyhow::bail!("unknown vision mode {other:?}; use auto or off"),
        }
    }
}

/// What became of the vision tower at load, for the startup log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionTower {
    /// The checkpoint does not carry one.
    Absent,
    /// The checkpoint carries one and [`VisionMode::Off`] skipped it.
    Off,
    /// Resident on the GPU.
    Loaded { bytes: usize, blocks: usize },
}

/// Engine-wide load options; architectures ignore what does not apply.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Where the Qwen3.8-Flash-Next n-gram table lives.
    pub ngram_storage: crate::qwen4exp::NgramStorage,
    /// Draft tokens per speculative step; `0` leaves the draft head unloaded.
    pub mtp_drafts: usize,
    /// Whether to load the vision tower when the checkpoint has one.
    pub vision: VisionMode,
    /// Routed experts kept on the GPU at once, for machines that cannot hold
    /// them all (`docs/low-ram-experts.md`); `None` loads every expert as
    /// its layer's own stack. `LILY_EXPERT_SLOTS` overrides it.
    pub expert_slots: Option<usize>,
    /// The usage counts (`lily-experts` output) that place experts in the
    /// cache; `None` looks for `expert-usage.json` next to the checkpoint
    /// and falls back to uniform. `LILY_EXPERT_USAGE` overrides it.
    pub expert_usage: Option<std::path::PathBuf>,
}

/// Which draw a decode pass ends with.
#[derive(Clone, Copy, Debug)]
pub struct Draw<'p> {
    pub params: &'p SamplingParams,
    /// Index of this draw within the request (the RNG counter).
    pub step: usize,
}

pub trait LanguageModel: Sized {
    /// Distinct expert lookups and misses of an expert cache, when the
    /// model serves its experts from one (`LoadOptions::expert_slots`).
    fn expert_cache_stats(&self) -> Option<(u64, u64)> {
        None
    }

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

    /// A tag identifying the on-disk layout of this model's sessions (model,
    /// cache shapes, optional heads); `None` when sessions cannot be
    /// persisted. Files written under a different tag are never read.
    fn persistence_format(&self) -> Option<String> {
        None
    }

    /// Reads a snapshot [`SnapshotApi::write_to`] wrote.
    fn read_snapshot(
        &self,
        ctx: &MetalContext,
        r: &mut dyn std::io::Read,
    ) -> Result<<Self::State as DecodeStateApi>::Snapshot> {
        let _ = (ctx, r);
        anyhow::bail!("this model does not persist sessions")
    }

    /// Warms weights served from disk (the paged n-gram table), pinning them
    /// in memory when `lock` is set, and returns the bytes found resident;
    /// models without such weights return 0.
    fn warm_storage(&self, lock: bool) -> Result<u64> {
        let _ = lock;
        Ok(0)
    }

    /// Bytes of weights served from the page cache instead of GPU memory
    /// (the paged n-gram table). They are not in the device's allocated
    /// size, yet they want to stay resident and so compete with the session
    /// cache for physical memory; 0 for models without such weights.
    fn paged_storage_bytes(&self) -> usize {
        0
    }

    /// The vision tower's fate at load; `None` for architectures whose
    /// checkpoints lily reads text-only.
    fn vision_tower(&self) -> Option<VisionTower> {
        None
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

    /// [`Self::prefill`] for a prompt that may carry images: `vision` holds
    /// the whole prompt's per-token rotary positions and the images' merged
    /// rows (from [`Self::encode_image`]) that replace the placeholder rows
    /// the fed range covers. `None` is exactly [`Self::prefill`], so a text
    /// request takes the text path untouched. Models without a vision path
    /// refuse a `Some`.
    fn prefill_with_vision(
        &self,
        ctx: &MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&crate::qwen4exp::VisionInput<'_>>,
    ) -> Result<()> {
        match vision {
            None => self.prefill(ctx, state, scratch, tokens, draw),
            Some(_) => anyhow::bail!("this model has no vision path"),
        }
    }

    /// Runs the vision tower over one preprocessed image
    /// (`[grid_h * grid_w, patch_dim]` f32 rows in block-major patch order,
    /// [`crate::qwen4exp::image::preprocess`]) and returns its merged rows
    /// as an owned bf16 `[grid_h * grid_w / 4, hidden]` tensor, one row per
    /// `<|image_pad|>` of the image's span, valid for as long as the caller
    /// keeps it. Waits for the GPU. Models without a loaded tower refuse.
    fn encode_image(
        &self,
        ctx: &MetalContext,
        scratch: &mut Self::Scratch,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let _ = (ctx, scratch, pixels, grid_h, grid_w);
        anyhow::bail!("this model has no vision tower loaded")
    }

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

    /// Whether [`Self::encode_parked_step`] is available: the model can take a
    /// decode step that is committed before its per-token host inputs exist
    /// and blocks on the GPU until [`Self::release_parked`].
    fn supports_parking(&self) -> bool {
        false
    }

    /// Like [`Self::encode_decode_step`], but meant to be committed right
    /// away: the pass parks on the GPU where it first reads what
    /// [`Self::prepare_step_inputs`] stages, until [`Self::release_parked`].
    /// This takes the command-buffer submission latency off the per-token
    /// critical path. One parked step at a time; the caller commits it and
    /// calls [`DecodeStateApi::advance`], later stages the inputs and
    /// releases it. A committed parked pass holds the queue until released.
    fn encode_parked_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &Self::State,
        scratch: &Self::Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        let _ = (ctx, state, scratch, slot_in, slot_out, draw);
        anyhow::bail!("this model cannot park decode steps")
    }

    /// Lets the parked pass continue; its staged inputs must be in place.
    /// No-op when nothing is parked.
    fn release_parked(&self, scratch: &Self::Scratch) -> Result<()> {
        let _ = scratch;
        Ok(())
    }

    // --- speculative decoding (models with a draft head) ------------------

    /// Draft tokens per step the model can propose; `0` when it has no draft
    /// head loaded, in which case the remaining methods are never called.
    fn max_drafts(&self) -> usize {
        0
    }

    /// The first proposals of a request: up to `drafts` tokens following
    /// `first`, the token the prefill drew, proposed under the request's
    /// sampler (`step0` is the draw index of the verify pass that will check
    /// them). Waits for the GPU.
    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        first: u32,
        drafts: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<u32>> {
        let _ = (ctx, state, scratch, first, drafts, params, step0);
        anyhow::bail!("this model has no draft head")
    }

    /// Feeds `pending` and `drafts` in one batched pass and draws one token
    /// per row (draw `step0 + row` of the request), waiting for the GPU. The
    /// state is then mid-step: it has fed every row, and must be completed
    /// with [`Self::finish_speculation`] before anything else. `parked` is the
    /// pass [`Self::finish_speculation`] may have committed for exactly these
    /// arguments, parked on the GPU until its host inputs exist; the model
    /// stages them and releases it. Returns the draws and the draft pass the
    /// model committed behind the verify pass: it decides the accepted count
    /// on the GPU, rolls the state back to it and proposes up to
    /// `next_drafts` tokens following the accepted draw. Hand it to
    /// [`Self::finish_speculation`].
    #[allow(clippy::too_many_arguments)]
    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        parked: Option<PendingPass<'a>>,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        let _ =
            (ctx, state, scratch, pending, drafts, params, step0, parked, next_drafts);
        anyhow::bail!("this model has no draft head")
    }

    /// Completes a verify pass: keeps `pending` plus the first `accepted`
    /// drafts as fed. `accepted` may not exceed the number of drafts the
    /// draws confirmed (which is what `draft` rolled back to); it is smaller
    /// when the generation ends at a confirmed draft, and the model then
    /// rolls back further. With `next` given (the accepted draw and the
    /// sampler settings of the following step) returns the draft pass's
    /// proposals and, when it could, the next verify pass, already committed
    /// and parked behind the draft pass, to hand back to [`Self::verify`].
    /// Waits for the GPU.
    fn finish_speculation<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        accepted: usize,
        next: Option<NextStep<'_>>,
        draft: PendingPass<'a>,
    ) -> Result<(Vec<u32>, Option<PendingPass<'a>>)> {
        let _ = (ctx, state, scratch, accepted, next, draft);
        anyhow::bail!("this model has no draft head")
    }
}

/// What the following speculative step will verify with: the fresh token and
/// the sampler settings of its pass.
#[derive(Clone, Copy)]
pub struct NextStep<'p> {
    pub token: u32,
    pub params: &'p SamplingParams,
    /// Draw index of the pass's first row.
    pub step0: usize,
}
