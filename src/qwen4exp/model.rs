//! The Qwen3.8-Flash-Next model graph: single-token decode steps plus a
//! batched prefill path that processes `PREFILL_CHUNK` prompt tokens per
//! command buffer.
//!
//! Compared with the Qwen3.5 graph the residual is a `hc_count`-wide stream
//! read through a gated low-rank mixer before every block and written back
//! through per-stream scalar gates (hyper-connections); decoder layer 1 adds
//! the hashed n-gram embedding; and the full-attention layers run Qwen Sparse
//! Attention once the context exceeds the indexer budget.

use std::path::Path;

use anyhow::{Result, ensure};

use std::cell::Cell;
use std::rc::Rc;

use crate::engine::{
    DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi, SnapshotApi,
    VisionMode, VisionTower,
};
use crate::kernels::attention::{
    MAX_SEQ, k_norm_rope_scatter_decode, q_norm_rope_split_decode, rope_neox,
    scatter_kv, sdpa_decode, sdpa_prefill, sdpa_split_scratch_splits, split_q_gate,
};
use crate::kernels::elementwise::{
    add_bf16, copy_words, gather_row_bf16, sigmoid_mul_bf16,
};
use crate::kernels::gdn::{
    GDN_HEAD_DIM, GDN_STATE_DTYPE, GdnGate, GdnRegscanStaging, conv1d_prefill,
    conv1d_step, gated_rmsnorm, gdn_prefill_mid, gdn_step_gated_fused,
};
use crate::kernels::hc::{
    HC_FUSED_MAX_ROWS, fused_read_supported, hc_broadcast_bf16, hc_inject_bf16,
    hc_mix_bf16, hc_read_down_q8, hc_read_down_q8_rows, hc_read_up_mix_q8,
    hc_read_up_mix_q8_rows, rmsnorm_grouped_bf16, silu_scaled_bf16,
};
use crate::kernels::norm::rmsnorm_bf16;
use crate::kernels::ple;
use crate::kernels::qsa::{
    self, INDEXER_D, SparseAttnRoute, SparseSplitScratch, SparseTileScratch,
};
use crate::kernels::sample::{
    DraftDists, SamplerScratch, SamplingParams, sample_f32, sample_spec_f32,
};
use crate::kernels::spec::ctrl_words;
use crate::kernels::{Arg, Pos, Rope};
use crate::kernels::{quant, skinny};
use crate::metal::{
    BlitCopy, ComputePass, DetachedPass, EncodedPass, MetalContext, Pacer, PendingPass,
    SharedEvent,
};
use crate::moe_ffn::{
    DecodeMoeIo, MoeDims, MoeScratch, PrefillMoeIo, PrefillMoeScratch, decode_moe,
    prefill_moe, prefix_rows, project_mat, project_stack_or_slices,
};
use crate::tensor::{DType, Tensor};

use super::vision::{self, VisionScratch};

use super::config::{GateAct, Qwen4ExpConfig};
use super::ngram::{NgramHasher, NgramStorage, NgramTable, StagedRows};
use super::positions::{ImageSpan, Positions};
use super::weights::{
    self, AttnWeights, GdnWeights, HcWeights, LayerWeights, Mixer, ModelWeights,
    MtpWeights, PleWeights,
};

/// Every RMSNorm in this model is zero-centered: gain = 1 + weight. (The GDN
/// GatedNorm is the exception and uses a plain gain.)
pub(super) const NORM_WEIGHT_BIAS: f32 = 1.0;

/// Prompt tokens processed by one prefill command buffer.
const PREFILL_CHUNK: usize = 4096;

/// Queries per sparse-attention sub-batch: bounds the `[QB, blocks]` score
/// matrix and the split partials while keeping the GPU busy.
const QSA_QUERY_BATCH: usize = 256;

/// Per-token caches grow in steps of this many tokens (192 MiB of KV plus
/// indexer keys per step), so a session's footprint follows its length.
const CAPACITY_STEP: usize = 8192;

/// Most draft tokens a speculative step may verify at once (the spec scratch
/// is sized for it).
pub const MAX_DRAFTS: usize = 3;
// Every verify pass (MAX_DRAFTS + 1 rows) and every draft-head batch takes
// the fused hyper-connection read.
const _: () = assert!(MAX_DRAFTS < HC_FUSED_MAX_ROWS);

/// How the batched (multi-row) graph is encoded: the serial encoder orders
/// every dispatch implicitly; the concurrent one runs a dependency level's
/// dispatches together and relies on the explicit barriers in this file.
/// Both give bit-identical results. Measured on the M5 Max: the concurrent
/// encoder wins for a few rows (53 ms → 22 ms at one row, where 2 229
/// dispatches each waited for the previous one), the serial encoder for big
/// chunks (8 000 tokens: 1 366 against 882 tok/s, where the coarse
/// level barriers stall more than the driver's own hazard tracking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoder {
    Serial,
    Concurrent,
}

/// Row count up to which the batched graph is encoded concurrently.
const CONCURRENT_ROWS_MAX: usize = 16;

fn encoder_for(rows: usize) -> Encoder {
    if let Some(forced) = std::env::var_os("LILY_BATCHED_ENCODER") {
        return if forced == "serial" { Encoder::Serial } else { Encoder::Concurrent };
    }
    if rows <= CONCURRENT_ROWS_MAX { Encoder::Concurrent } else { Encoder::Serial }
}

/// What a batched pass computes after its layers.
#[derive(Clone, Copy)]
pub(super) enum BatchMode<'p> {
    /// Prompt prefill: logits (and a draw) for the last row only, plus the
    /// draft head's catch-up over the chunk when the model has one. `vision`
    /// is set for a prompt with images: per-row rotary positions and the
    /// vision rows that replace the placeholders' embeddings.
    Prefill { draw: Option<Draw<'p>>, vision: Option<&'p PrefillVision<'p>> },
    /// Speculative verification: a draw per row (`step0 + row` indexes the
    /// request's draws), recurrent states after each row recorded for
    /// rollback, no draft-head work (that follows once acceptance is known).
    /// With `park`, the pass waits on the scratch's step sync for that value
    /// before its first host-staged input (the n-gram rows), so it can be
    /// committed before the host has them.
    Verify { params: &'p SamplingParams, step0: usize, park: Option<u64> },
}

/// One image of a prompt: its placeholder span and the tower's merged rows.
pub struct ImageEmbeds<'t> {
    pub span: ImageSpan,
    /// bf16 `[span.len, hidden_size]`: the tower's merged output, one row per
    /// placeholder in order (what the reference `masked_scatter`s into the
    /// token embeddings).
    pub rows: &'t Tensor,
}

/// What a prompt with images adds to its tokens for
/// [`Qwen4ExpModel::prefill_with_vision`].
pub struct VisionInput<'t> {
    /// Positions of the whole prompt from sequence index 0, not only of the
    /// tokens being fed: the kernels also rope the head's catch-up row one
    /// before the chunk and the first token of every indexer block the chunk
    /// completes, which can lie before it.
    pub positions: &'t Positions,
    pub images: &'t [ImageEmbeds<'t>],
}

/// A prefill chunk's share of a [`VisionInput`]: the position rows its
/// kernels read (U32 `[rows, 3]`, row `i` for sequence index `base + i`) and
/// the images whose rows may fall into the chunk.
pub(super) struct PrefillVision<'p> {
    positions: Tensor,
    base: usize,
    images: &'p [ImageEmbeds<'p>],
}

impl PrefillVision<'_> {
    fn rope(&self) -> Rope<'_> {
        Rope::Rows { positions: &self.positions, base: self.base }
    }
}

/// Vision rows replacing the embedding rows of one batch: the batch row the
/// run starts at and a `[rows, hidden]` bf16 view of the merged rows.
pub(super) struct RowOverride {
    row: usize,
    rows: Tensor,
}

/// The overrides of a batch whose row `j` holds token `first + j`, `rows`
/// rows long: every image span overlapping that range, clipped to it. Empty
/// for a batch without placeholders.
fn row_overrides(
    images: &[ImageEmbeds<'_>],
    first: usize,
    rows: usize,
    h: usize,
) -> Result<Vec<RowOverride>> {
    let mut out = Vec::new();
    for image in images {
        let a = image.span.start.max(first);
        let b = image.span.end().min(first + rows);
        if a >= b {
            continue;
        }
        ensure!(
            image.rows.shape() == [image.span.len, h]
                && image.rows.dtype() == DType::BF16,
            "image rows are {:?} {:?}, the span at {} needs bf16 [{}, {h}]",
            image.rows.shape(),
            image.rows.dtype(),
            image.span.start,
            image.span.len
        );
        out.push(RowOverride {
            row: a - first,
            rows: image.rows.view((a - image.span.start) * h, &[b - a, h])?,
        });
    }
    Ok(out)
}

/// Writes `overrides` into `x` (`[rows, h]` bf16, the gathered embeddings).
/// The caller orders this after the gather and before the next reader.
fn override_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    overrides: &[RowOverride],
    h: usize,
) -> Result<()> {
    for o in overrides {
        copy_words(ctx, pass, &o.rows, &x.view(o.row * h, o.rows.shape())?)?;
    }
    Ok(())
}

/// Host/GPU handshake for parked passes: a pass committed before its
/// per-token host inputs exist waits for `event >= value`; the host stages
/// the inputs and releases it. Values only grow, one parked pass at a time.
pub(super) struct StepSync {
    pub(super) event: SharedEvent,
    /// Diagnostics (`LILY_PROBE_ARRIVAL`): the GPU raises this to the parked
    /// value right before it waits, so a host poller can time the arrival.
    pub arrival: Option<SharedEvent>,
    /// Diagnostics: raised right after the wait, when the GPU resumed.
    pub resumed: Option<SharedEvent>,
    /// Value the outstanding parked pass waits for; 0 when none.
    armed: Cell<u64>,
    last: Cell<u64>,
    /// Completion predictors for the speculative loop's verify and draft
    /// passes (sleep-then-poll waits).
    pub(super) pace_verify: Cell<Pacer>,
    pub(super) pace_draft: Cell<Pacer>,
    /// Raised by the GPU at the end of every decode, verify and draft pass
    /// (`done_last` is the last value handed out), for paced waits.
    done: SharedEvent,
    done_last: Cell<u64>,
}

impl StepSync {
    fn new(ctx: &MetalContext) -> Result<Self> {
        let probe = std::env::var_os("LILY_PROBE_ARRIVAL").is_some();
        let arrival = if probe { Some(ctx.new_shared_event()?) } else { None };
        let resumed = if probe { Some(ctx.new_shared_event()?) } else { None };
        Ok(Self {
            event: ctx.new_shared_event()?,
            arrival,
            resumed,
            armed: Cell::new(0),
            last: Cell::new(0),
            pace_verify: Cell::new(Pacer::default()),
            pace_draft: Cell::new(Pacer::default()),
            done: ctx.new_shared_event()?,
            done_last: Cell::new(0),
        })
    }

    /// Encodes the pass's done signal as its last command.
    pub(super) fn signal_done(&self, pass: &ComputePass<'_>) -> Result<()> {
        let value = self.done_last.get() + 1;
        self.done_last.set(value);
        pass.signal_done(&self.done, value)
    }

    /// Encodes the parked wait (and the arrival probe when enabled).
    pub(super) fn encode_wait(&self, pass: &ComputePass<'_>, value: u64) -> Result<()> {
        if let Some(arrival) = &self.arrival {
            pass.signal_event(arrival, value)?;
        }
        pass.wait_event(&self.event, value)?;
        if let Some(resumed) = &self.resumed {
            pass.signal_event(resumed, value)?;
        }
        Ok(())
    }

    /// The value [`Self::arm`] will hand out next (for encoding a pass ahead
    /// of its commit).
    pub(super) fn peek(&self) -> u64 {
        self.last.get() + 1
    }

    /// Claims the value the next parked pass waits for.
    pub(super) fn arm(&self) -> Result<u64> {
        ensure!(self.armed.get() == 0, "a decode step is already parked");
        let value = self.last.get() + 1;
        self.last.set(value);
        self.armed.set(value);
        Ok(value)
    }

    /// Releases the parked pass, if any.
    pub(super) fn release(&self) -> Result<()> {
        let value = self.armed.take();
        if value != 0 {
            self.event.signal(value);
        }
        Ok(())
    }

    /// Releases whatever is parked when dropped: for error paths, so a
    /// committed pass never blocks the queue forever.
    pub(super) fn release_on_drop(&self) -> ReleaseOnDrop<'_> {
        ReleaseOnDrop(self)
    }
}

pub(super) struct ReleaseOnDrop<'s>(&'s StepSync);

impl Drop for ReleaseOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.0.release();
    }
}

fn moe_dims(cfg: &Qwen4ExpConfig) -> MoeDims {
    MoeDims {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        moe_intermediate: cfg.moe_intermediate_size,
        hidden: cfg.hidden_size,
        norm_topk_prob: cfg.norm_topk_prob,
    }
}

pub struct Qwen4ExpModel {
    pub config: Qwen4ExpConfig,
    pub(super) weights: ModelWeights,
    pub(super) attn_scale: f32,
    gdn_scale: f32,
    gdn_gate: GdnGate,
    pub(super) hasher: Option<NgramHasher>,
}

pub(super) enum LayerState {
    Gdn {
        /// Recurrent state, fp32 `[H, 128, 128]`.
        state: Tensor,
        /// Double-buffered conv window, bf16 `[C, KD-1]` each (see the
        /// Qwen3.5 graph for the buffering contract).
        conv_windows: [Tensor; 2],
    },
    Attn {
        /// `[KVH, max_seq, D]` bf16.
        k_cache: Tensor,
        v_cache: Tensor,
        /// Raw indexer keys, bf16 `[max_seq, INDEXER_D]`.
        idx_keys: Tensor,
        /// Normed, roped block keys, bf16 `[max_seq / ratio, INDEXER_D]`.
        blk_keys: Tensor,
    },
}

/// Per-Layer Embedding recurrent state: the two-token hash history (host
/// side, since the host hashes) and the dilated conv window, double-buffered
/// like the GDN conv windows.
pub(super) struct PleState {
    pub(super) hist: [u32; 2],
    pub(super) conv_windows: [Tensor; 2],
    eos: u32,
}

/// The draft head's per-session state: its attention caches (positions
/// aligned with the trunk's) and the trunk's wide residual for the last fed
/// token, which pairs with the next token to form the head's next input.
pub(super) struct MtpState {
    pub(super) layer: LayerState,
    /// bf16 `[hc_count * h]`.
    pub(super) hidden: Tensor,
}

/// A verify pass whose acceptance is still pending: enough to roll the state
/// back to any accepted prefix of the verified tokens.
pub(super) struct SpecPending {
    pub(super) pos_before: usize,
    pub(super) hist_before: Option<[u32; 2]>,
    /// The verified tokens (pending token first, then the drafts).
    pub(super) tokens: Vec<u32>,
    /// The trunk's draw per row.
    pub(super) sampled: Vec<u32>,
    pub(super) uses_penalties: bool,
    /// Proposals the draft pass behind the verify pass chains.
    pub(super) chain: usize,
    /// Next verify passes encoded while the verify pass ran, one per possible
    /// accepted count (index = accepted). Empty when there was no room to
    /// encode them ahead.
    pub(super) prepared: Vec<Prepared>,
    /// GPU end time of the verify pass (`LILY_PROFILE` only), for the gap to
    /// the draft pass.
    pub(super) verify_gpu_end: Option<f64>,
}

/// The verify pass that follows a draft pass, encoded ahead for one accepted
/// count. Its ids are written on the GPU by the draft pass; it only needs
/// committing (parked on the step sync).
pub(super) struct Prepared {
    pub(super) verify: DetachedPass,
    /// The step-sync value the verify pass waits for.
    pub(super) park: u64,
    /// Drafts the pair was encoded for.
    pub(super) drafts: usize,
}

impl PleState {
    fn reset(&mut self) {
        self.hist = [self.eos, self.eos];
        for w in &self.conv_windows {
            w.zero_fill();
        }
    }
}

pub struct DecodeState {
    pub pos: usize,
    /// What every token fed from now on adds to its sequence index to get
    /// its rotary position on all three axes (VISION.md `rope_deltas`): 0
    /// for a text prompt, negative after an image. Set from the prompt's
    /// [`Positions`] by [`Qwen4ExpModel::prefill_with_vision`] (the engine
    /// sets it when it acquires a session); a pure function of the prompt,
    /// so it is not part of a snapshot or the disk tier and a restore leaves
    /// it alone.
    pub rope_delta: i64,
    /// Tokens the per-token caches hold; grows in `CAPACITY_STEP`s.
    capacity: usize,
    pub(super) layers: Vec<LayerState>,
    /// Which double-buffered window slot is current (all recurrent layers
    /// advance in lockstep): prefill chunks read it, write the other, then
    /// flip; decode steps update it in place.
    pub(super) conv_slot: usize,
    pub(super) ple: Option<PleState>,
    pub(super) mtp: Option<MtpState>,
    pub(super) spec: Option<SpecPending>,
    kv_heads: usize,
    head_dim: usize,
    ratio: usize,
}

/// The recurrent part of a [`DecodeState`] at one position: GDN states and
/// conv windows, the PLE conv window and hash history, the draft head's
/// last hidden.
pub struct Snapshot {
    pos: usize,
    /// Per GDN layer, in layer order: `(state, conv_window)`.
    gdn: Vec<(Tensor, Tensor)>,
    ple: Option<([u32; 2], Tensor)>,
    mtp_hidden: Option<Tensor>,
}

impl SnapshotApi for Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.gdn.iter().map(|(a, b)| a.byte_len() + b.byte_len()).sum::<usize>()
            + self.ple.as_ref().map_or(0, |(_, w)| w.byte_len())
            + self.mtp_hidden.as_ref().map_or(0, Tensor::byte_len)
    }

    /// Layout: position (u64 LE); per GDN layer the state then the conv
    /// window; the PLE hash history (2 x u32 LE) and window; the draft head's
    /// hidden. Shapes come from the model, so nothing else is written.
    fn write_to(&self, w: &mut dyn std::io::Write) -> Result<()> {
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        for (state, window) in &self.gdn {
            state.write_to(w)?;
            window.write_to(w)?;
        }
        if let Some((hist, window)) = &self.ple {
            w.write_all(&hist[0].to_le_bytes())?;
            w.write_all(&hist[1].to_le_bytes())?;
            window.write_to(w)?;
        }
        if let Some(hidden) = &self.mtp_hidden {
            hidden.write_to(w)?;
        }
        Ok(())
    }
}

/// Visits an attention layer's cache regions holding the first `tokens`
/// entries, in the fixed persistence order: K per head, V per head, indexer
/// keys, block keys of the blocks those tokens start. `written` is how many
/// tokens the persisted layout was produced for (`>= tokens`); `f` gets each
/// region's view of `tokens` rows and the bytes the layout holds after them
/// for that region, so a reader of a shorter prefix can skip them. Writers
/// pass `written == tokens` and get zero.
fn attn_prefix_regions(
    lstate: &LayerState,
    tokens: usize,
    written: usize,
    ratio: usize,
    f: &mut dyn FnMut(&Tensor, usize) -> Result<()>,
) -> Result<()> {
    let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = lstate else {
        return Ok(());
    };
    ensure!(tokens <= written, "prefix of {tokens} tokens from a layout of {written}");
    for cache in [k_cache, v_cache] {
        let (heads, cap, d) = (cache.shape()[0], cache.shape()[1], cache.shape()[2]);
        ensure!(tokens <= cap, "prefix of {tokens} exceeds capacity {cap}");
        let tail = (written - tokens) * d * cache.dtype().size();
        for h in 0..heads {
            f(&cache.view(h * cap * d, &[tokens, d])?, tail)?;
        }
    }
    let row = INDEXER_D * idx_keys.dtype().size();
    f(&idx_keys.view(0, &[tokens, INDEXER_D])?, (written - tokens) * row)?;
    let blocks = tokens.div_ceil(ratio).min(blk_keys.shape()[0]);
    let written_blocks = written.div_ceil(ratio).min(blk_keys.shape()[0]);
    f(
        &blk_keys.view(0, &[blocks, INDEXER_D])?,
        (written_blocks - blocks) * INDEXER_D * blk_keys.dtype().size(),
    )
}

fn clone_tensor(ctx: &MetalContext, t: &Tensor) -> Result<Tensor> {
    let out = Tensor::zeros(ctx, t.shape(), t.dtype())?;
    ctx.blit_copy(&[BlitCopy {
        src: t,
        src_offset: 0,
        dst: &out,
        dst_offset: 0,
        len: t.byte_len(),
    }])?;
    Ok(out)
}

/// Copies the first `rows` rows of every `[heads, cap, d]` head block from
/// `src` to `dst` (both bf16, possibly different capacities).
fn head_block_copies<'t>(
    src: &'t Tensor,
    dst: &'t Tensor,
    rows: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    let (heads, src_cap, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let dst_cap = dst.shape()[1];
    ensure!(
        rows <= src_cap
            && rows <= dst_cap
            && dst.shape()[0] == heads
            && dst.shape()[2] == d,
        "cache copy shape mismatch"
    );
    let row_bytes = d * src.dtype().size();
    for h in 0..heads {
        out.push(BlitCopy {
            src,
            src_offset: h * src_cap * row_bytes,
            dst,
            dst_offset: h * dst_cap * row_bytes,
            len: rows * row_bytes,
        });
    }
    Ok(())
}

/// Copies the first `rows` rows of a `[cap, d]` store.
fn row_copy<'t>(
    src: &'t Tensor,
    dst: &'t Tensor,
    rows: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    let rows = rows.min(src.shape()[0]).min(dst.shape()[0]);
    let row_bytes = src.shape()[1] * src.dtype().size();
    out.push(BlitCopy {
        src,
        src_offset: 0,
        dst,
        dst_offset: 0,
        len: rows * row_bytes,
    });
    Ok(())
}

impl DecodeState {
    /// Returns the state to position zero. KV and indexer caches need no
    /// clearing: rows past `pos` are never read and get overwritten. The GPU
    /// must be idle on this state's buffers.
    pub fn reset(&mut self) -> Result<()> {
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                state.zero_fill();
                conv_windows[0].zero_fill();
                conv_windows[1].zero_fill();
            }
        }
        if let Some(ple) = &mut self.ple {
            ple.reset();
        }
        if let Some(mtp) = &self.mtp {
            mtp.hidden.zero_fill();
        }
        self.spec = None;
        self.pos = 0;
        self.rope_delta = 0;
        self.conv_slot = 0;
        Ok(())
    }

    fn attn_caches(
        ctx: &MetalContext,
        kv_heads: usize,
        head_dim: usize,
        ratio: usize,
        capacity: usize,
    ) -> Result<LayerState> {
        Ok(LayerState::Attn {
            k_cache: Tensor::zeros(ctx, &[kv_heads, capacity, head_dim], DType::BF16)?,
            v_cache: Tensor::zeros(ctx, &[kv_heads, capacity, head_dim], DType::BF16)?,
            idx_keys: Tensor::zeros(ctx, &[capacity, INDEXER_D], DType::BF16)?,
            blk_keys: Tensor::zeros(
                ctx,
                &[(capacity / ratio).max(1), INDEXER_D],
                DType::BF16,
            )?,
        })
    }

    /// Per-token cache bytes for `capacity` tokens across all attention
    /// layers, the draft head's included.
    fn cache_bytes(&self, capacity: usize) -> usize {
        let attn_layers =
            self.layers.iter().filter(|l| matches!(l, LayerState::Attn { .. })).count()
                + usize::from(self.mtp.is_some());
        let per_token = 2 * self.kv_heads * self.head_dim * 2 + INDEXER_D * 2;
        let blocks = (capacity / self.ratio).max(1) * INDEXER_D * 2;
        attn_layers * (capacity * per_token + blocks)
    }

    /// Replaces an attention layer's caches with ones of `capacity` tokens,
    /// copying the first `pos` entries. GPU idle.
    fn grow_attn(
        &self,
        ctx: &MetalContext,
        lstate: &mut LayerState,
        capacity: usize,
    ) -> Result<()> {
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = lstate else {
            return Ok(());
        };
        let LayerState::Attn { k_cache: k2, v_cache: v2, idx_keys: i2, blk_keys: b2 } =
            DecodeState::attn_caches(
                ctx,
                self.kv_heads,
                self.head_dim,
                self.ratio,
                capacity,
            )?
        else {
            unreachable!()
        };
        let mut copies = Vec::new();
        head_block_copies(k_cache, &k2, self.pos, &mut copies)?;
        head_block_copies(v_cache, &v2, self.pos, &mut copies)?;
        row_copy(idx_keys, &i2, self.pos, &mut copies)?;
        row_copy(blk_keys, &b2, self.pos.div_ceil(self.ratio), &mut copies)?;
        ctx.blit_copy(&copies)?;
        drop(copies);
        *k_cache = k2;
        *v_cache = v2;
        *idx_keys = i2;
        *blk_keys = b2;
        Ok(())
    }
}

/// Copies the first `tokens` entries of one attention layer's caches.
fn attn_prefix_copies<'t>(
    dst: &'t LayerState,
    src: &'t LayerState,
    tokens: usize,
    ratio: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    if let (
        LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
        LayerState::Attn { k_cache: k0, v_cache: v0, idx_keys: i0, blk_keys: b0 },
    ) = (dst, src)
    {
        head_block_copies(k0, k_cache, tokens, out)?;
        head_block_copies(v0, v_cache, tokens, out)?;
        row_copy(i0, idx_keys, tokens, out)?;
        row_copy(b0, blk_keys, tokens.div_ceil(ratio), out)?;
    }
    Ok(())
}

/// Sparse-attention scratch shared by decode (one query) and prefill
/// sub-batches (up to `QSA_QUERY_BATCH` queries), viewed per use.
struct QsaScratch {
    /// F32 `[QB, max_blocks]`.
    scores: Tensor,
    /// U32 `[QB, k_max]`.
    sel: Tensor,
    /// U32 `[QB]`.
    n_sel: Tensor,
    /// F32 `[slots * NQ, D]`, indexed `[QB*NQ, splits, D]` per dispatch
    /// (`qsa::split_scratch_slots`).
    partials: Tensor,
    /// F32 `[slots * NQ, 2]`.
    stats: Tensor,
    /// Per-tile unions for the tiled route.
    tiles: SparseTileScratch,
    /// How sub-batches past the dense limit attend (`LILY_QSA_ROUTE`).
    route: SparseAttnRoute,
}

impl QsaScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        max_seq: usize,
        qb: usize,
    ) -> Result<Self> {
        let idx = &cfg.indexer;
        let max_blocks = (max_seq / idx.compress_ratio).max(1);
        let k_max = idx.block_topk();
        let slots = qsa::split_scratch_slots(qb, k_max, idx.compress_ratio);
        let nq = cfg.num_attention_heads;
        Ok(Self {
            scores: Tensor::zeros(ctx, &[qb, max_blocks], DType::F32)?,
            sel: Tensor::zeros(ctx, &[qb, k_max], DType::U32)?,
            n_sel: Tensor::zeros(ctx, &[qb], DType::U32)?,
            partials: Tensor::zeros(ctx, &[slots * nq, cfg.head_dim], DType::F32)?,
            stats: Tensor::zeros(ctx, &[slots * nq, 2], DType::F32)?,
            tiles: SparseTileScratch::new(ctx, qb, max_blocks, k_max)?,
            route: SparseAttnRoute::from_env(),
        })
    }

    fn split_scratch(&self) -> SparseSplitScratch<'_> {
        SparseSplitScratch { partials: &self.partials, stats: &self.stats }
    }
}

/// Single-token PLE intermediates.
pub(super) struct PleScratch {
    /// U32 `[rows, heads]`: hashed row ids (resident table) or the
    /// sequential ids of the staged rows (paged table).
    ids: Tensor,
    /// Gathered rows for the paged table, at full capacity.
    stage: Option<Rc<StagedRows>>,
    emb: Tensor,
    key: Tensor,
    key_n: Tensor,
    query_n: Tensor,
    value: Tensor,
    gated: Tensor,
    gated_n: Tensor,
}

impl PleScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        rows: usize,
        table: &NgramTable,
    ) -> Result<Self> {
        let ple = cfg.ple.as_ref().expect("PLE scratch without PLE config");
        let (h, wide) = (cfg.hidden_size, cfg.hc_width());
        let bf = DType::BF16;
        let heads = ple.ngram_heads();
        let (ids, stage) = match table {
            NgramTable::Resident(_) => {
                (Tensor::zeros(ctx, &[rows, heads], DType::U32)?, None)
            }
            NgramTable::Paged(paged) => {
                // Staged rows are gathered with their own sequential ids.
                let seq: Vec<u32> = (0..(rows * heads) as u32).collect();
                let ids = Tensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&seq),
                    &[rows, heads],
                    DType::U32,
                )?;
                (ids, Some(Rc::new(StagedRows::new(ctx, paged, rows * heads)?)))
            }
        };
        Ok(Self {
            ids,
            stage,
            emb: Tensor::zeros(ctx, &[rows, ple.embed_dim], bf)?,
            key: Tensor::zeros(ctx, &[rows, wide], bf)?,
            key_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            query_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            value: Tensor::zeros(ctx, &[rows, h], bf)?,
            gated: Tensor::zeros(ctx, &[rows, wide], bf)?,
            gated_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        Ok(Self {
            ids: prefix_rows(&self.ids, m)?,
            stage: self.stage.clone(),
            emb: prefix_rows(&self.emb, m)?,
            key: prefix_rows(&self.key, m)?,
            key_n: prefix_rows(&self.key_n, m)?,
            query_n: prefix_rows(&self.query_n, m)?,
            value: prefix_rows(&self.value, m)?,
            gated: prefix_rows(&self.gated, m)?,
            gated_n: prefix_rows(&self.gated_n, m)?,
        })
    }
}

/// Hyper-connection read intermediates for `rows` tokens. The fused decode
/// read (`hc_read_decode`) only touches `down`, `inj`, `inv_rms` and
/// `mixed`; the fused small-batch read (`hc_read_batched`, up to
/// `HC_FUSED_MAX_ROWS` rows) also writes `act`; the unfused batched read of
/// larger chunks writes every buffer.
pub(super) struct HcScratch {
    /// `[rows, G*H]` normed streams.
    pub(super) hn: Tensor,
    /// `[rows, lowrank]` read-gate logits and activation.
    down: Tensor,
    act: Tensor,
    /// `[rows, G*H]` per-element gate logits.
    pub(super) up: Tensor,
    /// `[rows, H]` the block input.
    pub(super) mixed: Tensor,
    /// `[rows, G]` write-gate logits.
    inj: Tensor,
    /// `[rows, G]` F32 stream RMS reciprocals handed from the fused down
    /// kernel to the fused up kernel.
    inv_rms: Tensor,
}

impl HcScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig, rows: usize) -> Result<Self> {
        let (h, wide, g) = (cfg.hidden_size, cfg.hc_width(), cfg.hc_count);
        let bf = DType::BF16;
        Ok(Self {
            hn: Tensor::zeros(ctx, &[rows, wide], bf)?,
            down: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            act: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            up: Tensor::zeros(ctx, &[rows, wide], bf)?,
            mixed: Tensor::zeros(ctx, &[rows, h], bf)?,
            inj: Tensor::zeros(ctx, &[rows, g], bf)?,
            inv_rms: Tensor::zeros(ctx, &[rows, g], DType::F32)?,
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        Ok(Self {
            hn: prefix_rows(&self.hn, m)?,
            down: prefix_rows(&self.down, m)?,
            act: prefix_rows(&self.act, m)?,
            up: prefix_rows(&self.up, m)?,
            mixed: prefix_rows(&self.mixed, m)?,
            inj: prefix_rows(&self.inj, m)?,
            inv_rms: prefix_rows(&self.inv_rms, m)?,
        })
    }
}

/// Per-step intermediates, allocated once. Projection outputs that share one
/// fused matvec (`gdn_in`, `attn_qkv`, `mlp_gu`) are single buffers whose
/// named segments are views.
pub struct Scratch {
    x: Tensor,
    /// `[G*H]` the hyper-connection residual stream.
    hyper: Tensor,
    hc: HcScratch,
    branch_out: Tensor,
    gdn_in: Tensor,
    attn_qkv: Tensor,
    mlp_gu: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_gated: Tensor,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    /// `[(NH+1)*INDEXER_D]` indexer projection and its prepared queries.
    idx_qk: Tensor,
    idx_q: Tensor,
    qsa: QsaScratch,
    ple: Option<PleScratch>,
    logits: Tensor,
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
    pub(super) sampler: SamplerScratch,
    /// Tokens written by the in-graph sampler (`U32[2]`): two ping-pong
    /// slots so the pipelined loop can host-read step N's token while the
    /// in-flight step N+1 writes the other slot.
    pub next_token: Tensor,
    /// Prefill staging for the largest quantized projection.
    pub(super) dequant: Tensor,
    moe: MoeScratch,
    /// Session-lived batched prefill intermediates, grown on demand and capped
    /// at `PREFILL_CHUNK` rows.
    pub(super) prefill: Option<PrefillScratch>,
    /// Speculative-decoding intermediates (models with a draft head).
    pub(super) spec: Option<SpecScratch>,
    /// Handshake for passes committed ahead of their host inputs.
    pub(super) sync: StepSync,
    /// The vision tower's intermediates, allocated by the first image and
    /// grown to the largest patch count seen; `None` until then.
    pub(super) vision: Option<VisionScratch>,
    /// Routing log for measurement (`Qwen4ExpModel::enable_expert_log`):
    /// `U32 [layers, capacity, top_k]`, the experts each prefilled position
    /// was routed to, per MoE layer.
    pub(super) expert_log: Option<Tensor>,
}

impl Scratch {
    /// The arrival probe event (`LILY_PROBE_ARRIVAL`), for benchmarks.
    pub fn arrival_probe(&self) -> Option<&SharedEvent> {
        self.sync.arrival.as_ref()
    }

    /// The resumed probe event (`LILY_PROBE_ARRIVAL`), for benchmarks.
    pub fn resumed_probe(&self) -> Option<&SharedEvent> {
        self.sync.resumed.as_ref()
    }
}

/// What a verify pass records so a draft pass can roll the recurrent state
/// back to the accepted prefix, plus the draft head's small buffers.
pub(super) struct SpecScratch {
    /// Per GDN layer: F32 `[MAX_DRAFTS, H, 128, 128]`, the state after each
    /// verified row but the last.
    pub(super) mid: Vec<Tensor>,
    /// Per GDN layer: bf16 `[MAX_DRAFTS + 1, C]`, the verified rows' conv inputs.
    pub(super) conv_in: Vec<Tensor>,
    /// bf16 `[MAX_DRAFTS + 1, G*H]`: the PLE conv inputs.
    pub(super) ple_conv_in: Option<Tensor>,
    /// F32 `[MAX_DRAFTS + 1, vocab]`.
    pub(super) logits: Tensor,
    /// U32 `[MAX_DRAFTS + 1]`: the trunk's draw per verified row.
    pub(super) verify_tokens: Tensor,
    /// U32 `[MAX_DRAFTS]`: the draft head's proposals.
    pub(super) draft_tokens: Tensor,
    /// The distributions the proposals were drawn from under a sampling
    /// request (one slot per draft), for the verify pass's speculative draws.
    pub(super) dists: DraftDists,
    /// U32 `[MAX_DRAFTS + 1]`: host-written tokens for the head's catch-up rows.
    pub(super) mtp_ids: Tensor,
    /// U32 control block the GPU fills with the accepted count and what
    /// follows from it (`kernels::spec`), read by the draft pass's dispatches.
    pub(super) ctrl: Tensor,
    /// bf16 `[1, G*H]`: the head residual row the chain continues from.
    pub(super) chain_in: Tensor,
}

impl SpecScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig) -> Result<Self> {
        let heads = cfg.linear_num_value_heads;
        let c = cfg.gdn_conv_channels();
        let gdn_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, super::config::LayerType::LinearAttention))
            .count();
        let rows = MAX_DRAFTS + 1;
        Ok(Self {
            mid: (0..gdn_layers)
                .map(|_| {
                    Tensor::zeros(
                        ctx,
                        &[MAX_DRAFTS, heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )
                })
                .collect::<Result<_>>()?,
            conv_in: (0..gdn_layers)
                .map(|_| Tensor::zeros(ctx, &[rows, c], DType::BF16))
                .collect::<Result<_>>()?,
            ple_conv_in: cfg
                .ple
                .as_ref()
                .map(|_| Tensor::zeros(ctx, &[rows, cfg.hc_width()], DType::BF16))
                .transpose()?,
            logits: Tensor::zeros(ctx, &[rows, cfg.vocab_size], DType::F32)?,
            verify_tokens: Tensor::zeros(ctx, &[rows], DType::U32)?,
            draft_tokens: Tensor::zeros(ctx, &[MAX_DRAFTS], DType::U32)?,
            dists: DraftDists::new(ctx, MAX_DRAFTS)?,
            mtp_ids: Tensor::zeros(ctx, &[rows], DType::U32)?,
            ctrl: Tensor::zeros(ctx, &[ctrl_words(MAX_DRAFTS)], DType::U32)?,
            chain_in: Tensor::zeros(ctx, &[1, cfg.hc_width()], DType::BF16)?,
        })
    }
}

/// Where a batched attention pass runs: the position of its first row and,
/// when the GPU supplies that position, the indexer block the pass completes
/// (block index and 0/1 count), written earlier in the same pass. A
/// GPU-supplied position is limited to single-row passes whose position range
/// spans less than one indexer block, so at most one block completes.
#[derive(Clone, Copy)]
pub(super) struct AttnPos<'t> {
    pub(super) pos: Pos<'t>,
    block: Option<(&'t Tensor, &'t Tensor)>,
}

impl<'t> AttnPos<'t> {
    pub(super) fn host(pos: usize) -> Self {
        Self { pos: Pos::host(pos), block: None }
    }

    /// `pos` (a U32 word in `min..=max`), `block` and `count`: the words a
    /// `kernels::spec::spec_accept` dispatch wrote for this row.
    pub(super) fn gpu(
        pos: &'t Tensor,
        min: usize,
        max: usize,
        block: &'t Tensor,
        count: &'t Tensor,
    ) -> Self {
        Self { pos: Pos::gpu(pos, min, max), block: Some((block, count)) }
    }
}

/// Batched prefill intermediates at a capacity that only grows; each chunk
/// uses exact row-prefix views.
pub(super) struct PrefillScratch {
    pub(super) m: usize,
    pub(super) ids: Tensor,
    x: Tensor,
    pub(super) hyper: Tensor,
    pub(super) hc: HcScratch,
    branch_out: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_out: Tensor,
    gdn_gated: Tensor,
    gdn_stage: GdnStageScratch,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    idx_qk: Tensor,
    idx_q: Tensor,
    stack: Tensor,
    moe: PrefillMoeScratch,
    qsa: QsaScratch,
    pub(super) ple: Option<PleScratch>,
    /// Draft head: its residual stream and the trunk hiddens feeding it,
    /// bf16 `[m, G*H]` each (models with a draft head only).
    pub(super) mtp_hyper: Option<Tensor>,
    mtp_hidden_in: Option<Tensor>,
    /// U32 `[m + ratio, 3]`: the 3-axis rotary positions a chunk of a prompt
    /// with an image reads (its rows, the head's row before it and the first
    /// token of every block it completes, up to `ratio` rows earlier).
    positions: Tensor,
}

struct GdnStageScratch {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
}

impl PrefillScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        capacity: usize,
        max_seq: usize,
        table: Option<&NgramTable>,
        with_mtp: bool,
    ) -> Result<Self> {
        ensure!(capacity > 0, "prefill scratch capacity must be nonzero");
        let m = capacity;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let hk = cfg.linear_num_key_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        Ok(Self {
            m,
            ids: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: Tensor::zeros(ctx, &[m, h], bf)?,
            hyper: Tensor::zeros(ctx, &[m, cfg.hc_width()], bf)?,
            hc: HcScratch::new(ctx, cfg, m)?,
            branch_out: Tensor::zeros(ctx, &[m, h], bf)?,
            mlp_gate: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_up: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_act: Tensor::zeros(ctx, &[m, inter], bf)?,
            qkv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            qkv_conv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            z: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            a: Tensor::zeros(ctx, &[m, heads], bf)?,
            b: Tensor::zeros(ctx, &[m, heads], bf)?,
            gdn_out: Tensor::zeros(ctx, &[m, heads, GDN_HEAD_DIM], bf)?,
            gdn_gated: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            gdn_stage: GdnStageScratch {
                qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], bf)?,
                decay: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
                beta: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
            },
            qg: Tensor::zeros(ctx, &[m, nq * 2 * hd], bf)?,
            q: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            k_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            v_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            attn_o: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[m, nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[m, (nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[m, nh, INDEXER_D], bf)?,
            stack: {
                // Rows cap at the largest chunk eligible for fused projection.
                let rows = m.min(skinny::DENSE_SMALLM_THRESHOLD);
                let width = (conv_c + dim_v + 2 * heads)
                    .max((nq * 2 + 2 * nkv) * hd)
                    .max(2 * inter);
                Tensor::zeros(ctx, &[rows, width], bf)?
            },
            moe: PrefillMoeScratch::new(ctx, &moe_dims(cfg), m)?,
            qsa: QsaScratch::new(ctx, cfg, max_seq, QSA_QUERY_BATCH.min(m))?,
            ple: table.map(|t| PleScratch::new(ctx, cfg, m, t)).transpose()?,
            mtp_hyper: with_mtp
                .then(|| Tensor::zeros(ctx, &[m, cfg.hc_width()], bf))
                .transpose()?,
            mtp_hidden_in: with_mtp
                .then(|| Tensor::zeros(ctx, &[m, cfg.hc_width()], bf))
                .transpose()?,
            positions: Tensor::zeros(
                ctx,
                &[m + cfg.indexer.compress_ratio, 3],
                DType::U32,
            )?,
        })
    }

    /// Views of the capacity scratch for one chunk of `tokens`, uploading the
    /// token ids (the GPU is idle here: chunks are commit_wait-synchronized).
    pub(super) fn chunk(&self, tokens: &[u32]) -> Result<Self> {
        let chunk = self.rows(tokens.len())?;
        chunk.ids.write_bytes(bytemuck::cast_slice(tokens))?;
        Ok(chunk)
    }

    /// Row-prefix views for `m` rows without touching the ids.
    pub(super) fn rows(&self, m: usize) -> Result<Self> {
        ensure!(m <= self.m, "chunk of {m} tokens exceeds scratch capacity {}", self.m);
        let ids = prefix_rows(&self.ids, m)?;
        Ok(Self {
            m,
            ids,
            x: prefix_rows(&self.x, m)?,
            hyper: prefix_rows(&self.hyper, m)?,
            hc: self.hc.rows(m)?,
            branch_out: prefix_rows(&self.branch_out, m)?,
            mlp_gate: prefix_rows(&self.mlp_gate, m)?,
            mlp_up: prefix_rows(&self.mlp_up, m)?,
            mlp_act: prefix_rows(&self.mlp_act, m)?,
            qkv: prefix_rows(&self.qkv, m)?,
            qkv_conv: prefix_rows(&self.qkv_conv, m)?,
            z: prefix_rows(&self.z, m)?,
            a: prefix_rows(&self.a, m)?,
            b: prefix_rows(&self.b, m)?,
            gdn_out: prefix_rows(&self.gdn_out, m)?,
            gdn_gated: prefix_rows(&self.gdn_gated, m)?,
            gdn_stage: GdnStageScratch {
                qk_norm: prefix_rows(&self.gdn_stage.qk_norm, m)?,
                decay: prefix_rows(&self.gdn_stage.decay, m)?,
                beta: prefix_rows(&self.gdn_stage.beta, m)?,
            },
            qg: prefix_rows(&self.qg, m)?,
            q: prefix_rows(&self.q, m)?,
            gate: prefix_rows(&self.gate, m)?,
            k_new: prefix_rows(&self.k_new, m)?,
            v_new: prefix_rows(&self.v_new, m)?,
            attn_o: prefix_rows(&self.attn_o, m)?,
            attn_gated: prefix_rows(&self.attn_gated, m)?,
            idx_qk: prefix_rows(&self.idx_qk, m)?,
            idx_q: prefix_rows(&self.idx_q, m)?,
            // Kept at full extent: `project_stack_or_slices` views the exact
            // [m, n_total] prefix per level and never fuses past the threshold.
            stack: self.stack.view(0, self.stack.shape())?,
            moe: self.moe.chunk(m)?,
            // Sub-batch views are taken per use inside the attention branch.
            qsa: QsaScratch {
                scores: self.qsa.scores.view(0, self.qsa.scores.shape())?,
                sel: self.qsa.sel.view(0, self.qsa.sel.shape())?,
                n_sel: self.qsa.n_sel.view(0, self.qsa.n_sel.shape())?,
                partials: self.qsa.partials.view(0, self.qsa.partials.shape())?,
                stats: self.qsa.stats.view(0, self.qsa.stats.shape())?,
                tiles: self.qsa.tiles.share()?,
                route: self.qsa.route,
            },
            ple: self.ple.as_ref().map(|p| p.rows(m)).transpose()?,
            mtp_hyper: self
                .mtp_hyper
                .as_ref()
                .map(|t| prefix_rows(t, m))
                .transpose()?,
            mtp_hidden_in: self
                .mtp_hidden_in
                .as_ref()
                .map(|t| prefix_rows(t, m))
                .transpose()?,
            // Full extent: the chunk's slice is viewed at upload.
            positions: self.positions.view(0, self.positions.shape())?,
        })
    }
}

/// What a verify pass records per layer for rollback: the GDN states after
/// each row but the last, and the rows' conv inputs (GDN and PLE) so the conv
/// windows can be rewound.
pub(super) struct Capture {
    pub(super) mid: Option<Tensor>,
    pub(super) conv_in: Tensor,
    pub(super) ple_conv_in: Option<Tensor>,
}

impl Qwen4ExpModel {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(ctx, dir, NgramStorage::default(), true, VisionMode::Auto)
    }

    /// Loads with the n-gram table `storage` of choice, the draft head when
    /// `with_mtp` and the checkpoint has one, and the vision tower when
    /// `vision` allows it and the checkpoint has one.
    pub fn load_with(
        ctx: &MetalContext,
        dir: impl AsRef<Path>,
        storage: NgramStorage,
        with_mtp: bool,
        vision: VisionMode,
    ) -> Result<Self> {
        let config = Qwen4ExpConfig::from_model_dir(&dir)?;
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "GDN head dim {} unsupported (kernel is compiled for {GDN_HEAD_DIM})",
            config.linear_key_head_dim,
        );
        ensure!(
            config.indexer.head_dim == INDEXER_D,
            "indexer head dim {} unsupported (kernels are compiled for {INDEXER_D})",
            config.indexer.head_dim
        );
        let weights = weights::load(
            ctx,
            &dir,
            &config,
            storage,
            with_mtp,
            vision == VisionMode::Auto,
        )?;
        let hasher = match &config.ple {
            Some(p) => Some(NgramHasher::new(
                &p.layer_multipliers,
                &p.head_vocab_sizes,
                &p.head_offsets,
                p.heads_per_ngram,
                p.eos_token_id,
            )?),
            None => None,
        };
        let gdn_gate = match config.gdn_gate {
            GateAct::Silu => GdnGate::Silu,
            GateAct::Sigmoid => GdnGate::Sigmoid,
        };
        let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
        let gdn_scale = 1.0 / (config.linear_key_head_dim as f32).sqrt();
        Ok(Self { config, weights, attn_scale, gdn_scale, gdn_gate, hasher })
    }

    /// Runs the vision tower over one preprocessed image and returns its
    /// merged rows as an owned bf16 `[grid_h * grid_w / 4, hidden_size]`
    /// tensor (the tower's output lives in the scratch only until the next
    /// image, so the engine's copy is what a prompt with several images
    /// hands to [`Self::prefill_with_vision`]). Waits for the GPU.
    pub fn encode_image(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let (Some(config), Some(weights)) = (&self.config.vision, &self.weights.vision)
        else {
            anyhow::bail!("the vision tower is not loaded");
        };
        let out = vision::forward_with(
            ctx,
            config,
            weights,
            &mut s.vision,
            pixels,
            grid_h,
            grid_w,
            config.depth,
        )?;
        let shape = [grid_h * grid_w / 4, self.config.hidden_size];
        ensure!(
            out.merged.shape() == shape,
            "the tower produced {:?} rows for a ({grid_h}, {grid_w}) grid, expected {shape:?}",
            out.merged.shape()
        );
        Tensor::from_bytes(ctx, out.merged.raw_bytes(), &shape, DType::BF16)
    }

    /// Whether the draft head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.weights.mtp.is_some()
    }

    /// Whether the vision tower is loaded.
    pub fn has_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    fn begin_batched<'a>(
        &self,
        ctx: &'a MetalContext,
        rows: usize,
    ) -> Result<ComputePass<'a>> {
        match encoder_for(rows) {
            Encoder::Serial => ctx.begin(),
            Encoder::Concurrent => ctx.begin_concurrent(),
        }
    }

    /// The n-gram table, when the checkpoint has a PLE layer.
    fn ple_table(&self) -> Option<&NgramTable> {
        self.weights.layers.iter().find_map(|l| l.ple.as_ref()).map(|p| &p.table)
    }

    /// Stages the n-gram rows for `tokens` following `hist` into `p`: hashed
    /// ids for a resident table, the rows themselves for a paged one. The GPU
    /// must not be reading `p`.
    pub(super) fn stage_ngram(
        &self,
        w: &PleWeights,
        p: &PleScratch,
        tokens: &[u32],
        hist: [u32; 2],
    ) -> Result<()> {
        let hasher = self.hasher.as_ref().expect("PLE weights without hasher");
        let mut ids = Vec::with_capacity(tokens.len() * hasher.heads());
        hasher.ids(tokens, hist, &mut ids);
        match &w.table {
            NgramTable::Resident(_) => p.ids.write_bytes(bytemuck::cast_slice(&ids)),
            NgramTable::Paged(table) => {
                let stage = p.stage.as_ref().expect("paged table without staging");
                stage.fill(table, &ids)
            }
        }
    }

    /// Encodes the table gather for `rows` tokens into `p.emb`.
    fn gather_ngram(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        rows: usize,
    ) -> Result<()> {
        let heads = self.config.ple.as_ref().expect("PLE config").ngram_heads();
        match &w.table {
            NgramTable::Resident(table) => {
                ple::ple_gather_q4(ctx, pass, table, &p.ids, heads, &p.emb)
            }
            NgramTable::Paged(_) => {
                let stage = p.stage.as_ref().expect("paged table without staging");
                let staged = stage.as_table(rows * heads)?;
                ple::ple_gather_q4(ctx, pass, &staged, &p.ids, heads, &p.emb)
            }
        }
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<DecodeState> {
        let capacity = round_capacity(capacity)?;
        let cfg = &self.config;
        let c = cfg.gdn_conv_channels();
        let kd = cfg.linear_conv_kernel_dim;
        let heads = cfg.linear_num_value_heads;
        let ratio = cfg.indexer.compress_ratio;
        let layers = self
            .weights
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Mixer::Gdn(_) => Ok(LayerState::Gdn {
                    state: Tensor::zeros(
                        ctx,
                        &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )?,
                    conv_windows: [
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                    ],
                }),
                Mixer::Attn(_) => DecodeState::attn_caches(
                    ctx,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    ratio,
                    capacity,
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        let ple = cfg
            .ple
            .as_ref()
            .map(|p| {
                let s = p.conv_state_len();
                Ok::<_, anyhow::Error>(PleState {
                    hist: [p.eos_token_id, p.eos_token_id],
                    conv_windows: [
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                    ],
                    eos: p.eos_token_id,
                })
            })
            .transpose()?;
        let mtp = match &self.weights.mtp {
            Some(_) => Some(MtpState {
                layer: DecodeState::attn_caches(
                    ctx,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    ratio,
                    capacity,
                )?,
                hidden: Tensor::zeros(ctx, &[cfg.hc_width()], DType::BF16)?,
            }),
            None => None,
        };
        Ok(DecodeState {
            pos: 0,
            rope_delta: 0,
            capacity,
            layers,
            conv_slot: 0,
            ple,
            mtp,
            spec: None,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            ratio,
        })
    }

    /// Scratch whose split-decode and sparse-attention buffers are sized for
    /// `capacity_tokens` of context rather than the engine's ceiling.
    /// Records, during prefill, which experts every position of every MoE
    /// layer was routed to, into `U32 [layers, capacity, top_k]` scratch
    /// (measurement: expert usage skew for weight offloading). The log
    /// covers positions below the scratch's capacity; the draft head's own
    /// routing is not logged.
    pub fn enable_expert_log(&self, ctx: &MetalContext, scratch: &mut Scratch, capacity: usize) -> Result<()> {
        let layers = self.weights.layers.len();
        let top_k = self.config.num_experts_per_tok;
        scratch.expert_log = Some(Tensor::zeros(ctx, &[layers, capacity, top_k], DType::U32)?);
        Ok(())
    }

    /// The routing log enabled by [`Self::enable_expert_log`].
    pub fn expert_log<'s>(&self, scratch: &'s Scratch) -> Option<&'s Tensor> {
        scratch.expert_log.as_ref()
    }

    /// Experts and top-k of the routed FFN (for reading the log).
    pub fn moe_shape(&self) -> (usize, usize, usize) {
        (self.weights.layers.len(), self.config.num_experts, self.config.num_experts_per_tok)
    }

    pub fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        let cfg = &self.config;
        // Dense decode only ever runs up to the indexer's dense limit.
        let splits =
            sdpa_split_scratch_splits(capacity_tokens.min(cfg.indexer.dense_limit()));
        let h = cfg.hidden_size;
        let wide = cfg.hc_width();
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        let gdn_in = Tensor::zeros(ctx, &[conv_c + dim_v + 2 * heads], bf)?;
        let attn_qkv = Tensor::zeros(ctx, &[(nq * 2 + 2 * nkv) * hd], bf)?;
        let mlp_gu = Tensor::zeros(ctx, &[2 * inter], bf)?;
        // Largest dense projection the batched fallback may stage. Expert
        // stacks use native grouped Q4 kernels and never dequant in full.
        let ple_embed = cfg.ple.as_ref().map_or(0, |p| p.embed_dim);
        let dequant_numel = [
            conv_c * h,
            dim_v * h,
            heads * h,
            h * dim_v,
            nq * 2 * hd * h,
            nkv * hd * h,
            h * nq * hd,
            cfg.num_experts * h,
            cfg.moe_intermediate_size * h,
            h * cfg.moe_intermediate_size,
            inter * h,
            h * inter,
            cfg.hc_lowrank * wide,
            cfg.hc_count * wide,
            (nh + 1) * INDEXER_D * h,
            wide * ple_embed,
            h * ple_embed,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(Scratch {
            sync: StepSync::new(ctx)?,
            x: Tensor::zeros(ctx, &[h], bf)?,
            hyper: Tensor::zeros(ctx, &[wide], bf)?,
            hc: HcScratch::new(ctx, cfg, 1)?,
            branch_out: Tensor::zeros(ctx, &[h], bf)?,
            mlp_gate: mlp_gu.view(0, &[inter])?,
            mlp_up: mlp_gu.view(inter, &[inter])?,
            mlp_act: Tensor::zeros(ctx, &[inter], bf)?,
            qkv: gdn_in.view(0, &[conv_c])?,
            qkv_conv: Tensor::zeros(ctx, &[conv_c], bf)?,
            z: gdn_in.view(conv_c, &[dim_v])?,
            a: gdn_in.view(conv_c + dim_v, &[heads])?,
            b: gdn_in.view(conv_c + dim_v + heads, &[heads])?,
            gdn_gated: Tensor::zeros(ctx, &[dim_v], bf)?,
            qg: attn_qkv.view(0, &[nq, 2 * hd])?,
            q: Tensor::zeros(ctx, &[nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[nq, hd], bf)?,
            k_new: attn_qkv.view(nq * 2 * hd, &[nkv, hd])?,
            v_new: attn_qkv.view((nq * 2 + nkv) * hd, &[nkv, hd])?,
            attn_o: Tensor::zeros(ctx, &[nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[(nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[nh, INDEXER_D], bf)?,
            qsa: QsaScratch::new(ctx, cfg, capacity_tokens, 1)?,
            ple: self
                .ple_table()
                .map(|t| PleScratch::new(ctx, cfg, 1, t))
                .transpose()?,
            gdn_in,
            attn_qkv,
            mlp_gu,
            logits: Tensor::zeros(ctx, &[cfg.vocab_size], DType::F32)?,
            sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
            sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
            sampler: SamplerScratch::new(ctx, cfg.vocab_size)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_numel], bf)?,
            moe: MoeScratch::new(ctx, &moe_dims(cfg))?,
            prefill: None,
            vision: None,
            expert_log: None,
            spec: self
                .weights
                .mtp
                .as_ref()
                .map(|_| SpecScratch::new(ctx, cfg))
                .transpose()?,
        })
    }

    /// Makes sure `s.prefill` can hold chunks of `needed` rows.
    pub(super) fn ensure_prefill_scratch(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        needed: usize,
    ) -> Result<()> {
        let needed = needed.min(PREFILL_CHUNK);
        let have = s.prefill.as_ref().map_or(0, |p| p.m);
        if have < needed {
            let target = needed.next_power_of_two().min(PREFILL_CHUNK);
            s.prefill = None;
            s.prefill = Some(PrefillScratch::new(
                ctx,
                &self.config,
                target,
                MAX_SEQ,
                self.ple_table(),
                self.weights.mtp.is_some(),
            )?);
        }
        Ok(())
    }

    /// Encodes one decode step reading `next_token[slot_in]` and drawing
    /// into `next_token[slot_out]`, without committing. Without `park` the
    /// caller stages the token's n-gram rows first, then commits and advances
    /// `pos`. With `park` (a value from [`StepSync::arm`]) the pass waits on
    /// the step sync right before the n-gram gather, so it can be committed
    /// at once and released after staging.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
        park: Option<u64>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.spec.is_none(), "decode step during a pending speculative step");
        let pass = ctx.begin_concurrent()?;
        pass.set_label("decode");
        self.encode_decode_graph(ctx, &pass, state, s, slot_in, slot_out, draw, park)?;
        pass.end()
    }

    /// Host work for the step consuming `token`: stage its n-gram rows and
    /// advance the hash history.
    pub fn prepare_step_inputs(
        &self,
        state: &mut DecodeState,
        s: &Scratch,
        token: u32,
    ) -> Result<()> {
        if let (Some(pst), Some(p)) = (&mut state.ple, &s.ple) {
            let w = self
                .weights
                .layers
                .iter()
                .find_map(|l| l.ple.as_deref())
                .expect("PLE state without weights");
            self.stage_ngram(w, p, &[token], pst.hist)?;
            pst.hist = NgramHasher::advance(pst.hist, &[token]);
        }
        Ok(())
    }

    /// Runs the prompt in batches of `PREFILL_CHUNK` tokens, one command
    /// buffer per chunk. With `draw`, the final token's logits are sampled
    /// into `next_token[0]`. Text only: the rotary position of every row is
    /// its sequence index plus the state's `rope_delta`.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        self.prefill_with_vision(ctx, state, s, tokens, draw, None)
    }

    /// [`Self::prefill`] for a prompt that may carry images: `tokens` are fed
    /// at `state.pos` as sequence indices into `vision.positions`, every
    /// row takes its own 3-axis rotary position, the placeholders' embedding
    /// rows are replaced by the images' merged rows (in the trunk and in the
    /// draft head's catch-up), and the state's `rope_delta` becomes the
    /// prompt's. Without `vision` this is the text path, untouched.
    pub fn prefill_with_vision(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&VisionInput<'_>>,
    ) -> Result<()> {
        ensure!(!tokens.is_empty(), "empty prompt");
        ensure!(state.spec.is_none(), "prefill during a pending speculative step");
        let h = self.config.hidden_size;
        if let Some(v) = vision {
            let n = v.positions.len();
            ensure!(
                state.pos + tokens.len() <= n,
                "the prompt's positions cover {n} tokens, the prefill feeds {}..{}",
                state.pos,
                state.pos + tokens.len()
            );
            for image in v.images {
                ensure!(
                    image.span.end() <= n,
                    "image span {}..{} exceeds the prompt of {n} tokens",
                    image.span.start,
                    image.span.end()
                );
                ensure!(
                    image.rows.shape() == [image.span.len, h]
                        && image.rows.dtype() == DType::BF16,
                    "image rows are {:?} {:?}, the span at {} needs bf16 [{}, {h}]",
                    image.rows.shape(),
                    image.rows.dtype(),
                    image.span.start,
                    image.span.len
                );
            }
            state.rope_delta = v.positions.rope_delta;
        }
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        self.ensure_prefill_scratch(ctx, s, tokens.len())?;
        let s = &*s;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing after growth"))?;
        let ple_w = self.weights.layers.iter().find_map(|l| l.ple.as_deref());
        let ratio = self.config.indexer.compress_ratio;
        let mut remaining = tokens.len();
        let profile = std::env::var_os("LILY_PROFILE").is_some();
        for chunk in tokens.chunks(PREFILL_CHUNK) {
            let ps = capacity.chunk(chunk)?;
            if let (Some(w), Some(p), Some(pst)) = (ple_w, &ps.ple, &state.ple) {
                self.stage_ngram(w, p, chunk, pst.hist)?;
            }
            remaining -= chunk.len();
            let chunk_vision = match vision {
                Some(v) => {
                    // The rows this chunk's kernels rope: its own, the head's
                    // catch-up row one before it, and the first token of
                    // every block the chunk or the head completes, which lies
                    // at most `ratio - 1` before that row.
                    let pos = state.pos;
                    let lo = pos.saturating_sub(1) / ratio * ratio;
                    let hi = pos + chunk.len();
                    let positions = ps.positions.view(0, &[hi - lo, 3])?;
                    positions
                        .write_bytes(bytemuck::cast_slice(&v.positions.flat(lo..hi)))?;
                    Some(PrefillVision { positions, base: lo, images: v.images })
                }
                None => None,
            };
            let mode = BatchMode::Prefill {
                draw: if remaining == 0 { draw } else { None },
                vision: chunk_vision.as_ref(),
            };
            let started = std::time::Instant::now();
            let encoded = self.encode_batch(ctx, state, s, &ps, mode)?;
            let encoded_at = std::time::Instant::now();
            let completed = encoded.commit()?.wait_retain()?;
            if profile {
                let gpu = completed
                    .timing()
                    .map(|t| (t.gpu_end_secs - t.gpu_start_secs) * 1e3)
                    .unwrap_or(-1.0);
                eprintln!(
                    "profile prefill m={}: encode {:.2} ms, gpu {:.2} ms, total {:.2} ms",
                    chunk.len(),
                    (encoded_at - started).as_secs_f64() * 1e3,
                    gpu,
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            state.pos += chunk.len();
            if let Some(pst) = &mut state.ple {
                pst.hist = NgramHasher::advance(pst.hist, chunk);
            }
            // The chunk's recurrent kernels wrote the other window buffers.
            state.conv_slot = 1 - state.conv_slot;
        }
        Ok(())
    }

    // --- prefill ------------------------------------------------------------

    /// Encodes the batched graph for the chunk `ps` at `state.pos` without
    /// committing: every layer over all rows, then the head according to
    /// `mode`. Barriers mark every inter-level data edge so the same code runs
    /// on the serial and the concurrent encoder.
    pub(super) fn encode_batch<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        ps: &PrefillScratch,
        mode: BatchMode<'_>,
    ) -> Result<EncodedPass<'a>> {
        let m = ps.m;
        let pass = self.begin_batched(ctx, m)?;
        ensure!(state.pos + m <= state.capacity, "sequence full ({})", state.capacity);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let pos = state.pos;
        let conv_slot = state.conv_slot;
        let verify = matches!(mode, BatchMode::Verify { .. });
        pass.set_label(if verify { "verify" } else { "prefill" });
        if verify {
            ensure!(
                m <= MAX_DRAFTS + 1,
                "verify batch of {m} rows exceeds {}",
                MAX_DRAFTS + 1
            );
        }
        let spec = if verify { s.spec.as_ref() } else { None };
        let vision = match mode {
            BatchMode::Prefill { vision, .. } => vision,
            BatchMode::Verify { .. } => None,
        };
        // Rotary positions: each row's own 3-axis position for a chunk of a
        // prompt with an image, otherwise the sequence index plus the
        // state's delta (0 for text).
        let rope = match vision {
            Some(v) => v.rope(),
            None => Rope::Delta(state.rope_delta),
        };

        quant::gather_rows_q4(ctx, &pass, &self.weights.embed_tokens, &ps.ids, &ps.x)?;
        if let Some(v) = vision {
            // The images' merged rows replace the placeholders' embeddings
            // before the streams are initialised from them (the reference's
            // `masked_scatter` on `inputs_embeds`).
            let overrides = row_overrides(v.images, pos, m, h)?;
            if !overrides.is_empty() {
                pass.level_barrier(&[&ps.x])?;
                override_rows(ctx, &pass, &ps.x, &overrides, h)?;
            }
        }
        pass.level_barrier(&[&ps.x])?;
        hc_broadcast_bf16(ctx, &pass, &ps.x, &ps.hyper, h, g)?;
        pass.level_barrier(&[&ps.hyper])?;

        let mut gdn_index = 0usize;
        let mut layer_index = 0usize;
        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            let ple = match (&layer.ple, &ps.ple, &state.ple) {
                (Some(w), Some(p), Some(pst)) => Some((w.as_ref(), p, pst)),
                _ => None,
            };
            let capture = match (spec, lstate) {
                (Some(sp), LayerState::Gdn { .. }) => {
                    let mid = (m > 1)
                        .then(|| {
                            sp.mid[gdn_index].view(
                                0,
                                &[
                                    m - 1,
                                    cfg.linear_num_value_heads,
                                    GDN_HEAD_DIM,
                                    GDN_HEAD_DIM,
                                ],
                            )
                        })
                        .transpose()?;
                    Some(Capture {
                        mid,
                        conv_in: prefix_rows(&sp.conv_in[gdn_index], m)?,
                        ple_conv_in: sp
                            .ple_conv_in
                            .as_ref()
                            .map(|t| prefix_rows(t, m))
                            .transpose()?,
                    })
                }
                (Some(sp), LayerState::Attn { .. }) => Some(Capture {
                    mid: None,
                    conv_in: prefix_rows(&sp.conv_in[0], 0)?,
                    ple_conv_in: sp
                        .ple_conv_in
                        .as_ref()
                        .map(|t| prefix_rows(t, m))
                        .transpose()?,
                }),
                _ => None,
            };
            if matches!(lstate, LayerState::Gdn { .. }) {
                gdn_index += 1;
            }
            if let (Some(_), BatchMode::Verify { park: Some(value), .. }) =
                (&ple, &mode)
            {
                // The n-gram rows are the only host-staged input; everything
                // before this point ran while the host staged them.
                s.sync.encode_wait(&pass, *value)?;
            }
            self.block_batched(
                ctx,
                &pass,
                layer,
                lstate,
                &ps.hyper,
                pos,
                conv_slot,
                s,
                ps,
                ple,
                capture.as_ref(),
                rope,
            )?;
            if let Some(log) = &s.expert_log {
                // Keep this layer's routing before the next layer's router
                // overwrites the shared indices.
                let top_k = cfg.num_experts_per_tok;
                let capacity = log.shape()[1];
                ensure!(pos + m <= capacity, "expert log holds {capacity} positions");
                copy_words(
                    ctx,
                    &pass,
                    &ps.moe.indices.view(0, &[m * top_k])?,
                    &log.view((layer_index * capacity + pos) * top_k, &[m * top_k])?,
                )?;
                pass.level_barrier(&[log])?;
            }
            layer_index += 1;
        }

        match mode {
            BatchMode::Prefill { draw, vision } => {
                if let Some(draw) = draw {
                    // Only the last prompt token feeds decoding: pull its stream
                    // row into the single-token scratch and reuse the decode
                    // head path.
                    gather_row_bf16(ctx, &pass, &ps.hyper, &s.hyper, m - 1)?;
                    pass.level_barrier(&[&s.hyper])?;
                    self.hc_read_decode(ctx, &pass, &self.weights.final_mixer, s)?;
                    quant::gemv_quant(
                        ctx,
                        &pass,
                        &self.weights.lm_head,
                        &s.hc.mixed,
                        &s.logits,
                    )?;
                    pass.level_barrier(&[&s.logits])?;
                    // Slot 0 by convention: the pipelined loop's first decode step
                    // consumes the prefill token from this slot.
                    let out = s.next_token.view(0, &[1])?;
                    sample_f32(
                        ctx,
                        &pass,
                        &s.logits,
                        &s.sampler,
                        draw.params,
                        draw.step,
                        &out,
                    )?;
                    pass.level_barrier(&[&s.next_token])?;
                }
                if let (Some(mtp), Some(mst)) = (&self.weights.mtp, &state.mtp) {
                    let images = vision.map_or(&[][..], |v| v.images);
                    self.mtp_catch_up(ctx, &pass, mtp, mst, pos, s, ps, rope, images)?;
                }
            }
            BatchMode::Verify { params, step0, .. } => {
                let sp = s.spec.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("verify pass without spec scratch")
                })?;
                self.hc_read_batched(
                    ctx,
                    &pass,
                    &self.weights.final_mixer,
                    &ps.hyper,
                    s,
                    ps,
                )?;
                let logits = prefix_rows(&sp.logits, m)?;
                project_mat(
                    ctx,
                    &pass,
                    &ps.hc.mixed,
                    &self.weights.lm_head,
                    &logits,
                    &s.dequant,
                )?;
                pass.level_barrier(&[&logits])?;
                // One draw per row; the rows share the sampler scratch (and
                // the penalty counts, which must see earlier rows' draws).
                // Row j's draw checks proposal j (the row after it in the
                // pass): a speculative draw against the distribution the
                // draft head drew it from. The last row has no proposal.
                for j in 0..m {
                    let row = sp.logits.view(j * cfg.vocab_size, &[cfg.vocab_size])?;
                    let out = sp.verify_tokens.view(j, &[1])?;
                    if j + 1 < m {
                        sample_spec_f32(
                            ctx,
                            &pass,
                            &row,
                            &s.sampler,
                            params,
                            step0 + j,
                            &sp.dists.slot(j)?,
                            &ps.ids.view(j + 1, &[1])?,
                            &out,
                        )?;
                    } else {
                        sample_f32(
                            ctx,
                            &pass,
                            &row,
                            &s.sampler,
                            params,
                            step0 + j,
                            &out,
                        )?;
                    }
                    pass.level_barrier(&[&sp.verify_tokens])?;
                }
            }
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }

    /// One decoder block over the rows of `ps` on the residual `hyper`
    /// (`[m, G*H]`): the optional PLE, the mixer branch, the MoE, each read
    /// and written through its hyper-connection. `capture` (verify passes)
    /// records what a rollback needs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        layer: &LayerWeights,
        lstate: &LayerState,
        hyper: &Tensor,
        pos: usize,
        conv_slot: usize,
        s: &Scratch,
        ps: &PrefillScratch,
        ple: Option<(&PleWeights, &PleScratch, &PleState)>,
        capture: Option<&Capture>,
        rope: Rope<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        if let Some((w, p, pst)) = ple {
            self.ple_batched(
                ctx,
                pass,
                w,
                p,
                pst,
                conv_slot,
                hyper,
                s,
                ps,
                capture.and_then(|c| c.ple_conv_in.as_ref()),
            )?;
        }

        self.hc_read_batched(ctx, pass, &layer.attn_hc, hyper, s, ps)?;
        match (&layer.mixer, lstate) {
            (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                self.gdn_batched(
                    ctx,
                    pass,
                    w,
                    s,
                    ps,
                    state,
                    conv_windows,
                    conv_slot,
                    capture,
                )?;
            }
            (
                Mixer::Attn(w),
                LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
            ) => {
                self.attn_batched(
                    ctx, pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos,
                    rope,
                )?;
            }
            _ => anyhow::bail!("layer/state kind mismatch"),
        }
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])?;

        self.hc_read_batched(ctx, pass, &layer.mlp_hc, hyper, s, ps)?;
        prefill_moe(
            ctx,
            pass,
            &moe_dims(cfg),
            &layer.ffn,
            &PrefillMoeIo {
                x: &ps.hc.mixed,
                out: &ps.branch_out,
                stack: &ps.stack,
                mlp_gate: &ps.mlp_gate,
                mlp_up: &ps.mlp_up,
                mlp_act: &ps.mlp_act,
                dequant: &s.dequant,
            },
            &ps.moe,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])
    }

    /// `hyper` `[m, G*H]` → `ps.hc.mixed` `[m, H]` (and the write-gate logits
    /// when the block has an inject weight). Up to `HC_FUSED_MAX_ROWS` rows
    /// (the verify pass, the draft head) take the two fused kernels of the
    /// decode read, so those rows compute the read exactly as decode does;
    /// larger chunks (prefill) take the six-kernel skinny/GEMM chain. Leaves
    /// `mixed`/`inj` ordered.
    pub(super) fn hc_read_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        hyper: &Tensor,
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        if ps.m <= HC_FUSED_MAX_ROWS
            && fused_read_supported(&hc.down, &hc.up, hc.inject.as_ref(), h, g)
        {
            hc_read_down_q8_rows(
                ctx,
                pass,
                hyper,
                &hc.norm,
                &hc.down,
                hc.inject.as_ref(),
                &ps.hc.down,
                &ps.hc.inj,
                &ps.hc.inv_rms,
                &ps.hc.act,
                h,
                g,
                cfg.rms_norm_eps,
                NORM_WEIGHT_BIAS,
            )?;
            pass.level_barrier(&[&ps.hc.down, &ps.hc.inj, &ps.hc.inv_rms, &ps.hc.act])?;
            hc_read_up_mix_q8_rows(
                ctx,
                pass,
                &hc.up,
                &ps.hc.act,
                hyper,
                &hc.norm,
                &ps.hc.inv_rms,
                &ps.hc.mixed,
                h,
                g,
                NORM_WEIGHT_BIAS,
            )?;
            return pass.level_barrier(&[&ps.hc.mixed]);
        }
        let inv_g = 1.0 / g as f32;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hyper,
            &hc.norm,
            &ps.hc.hn,
            h,
            g,
            cfg.rms_norm_eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&ps.hc.hn])?;
        project_mat(ctx, pass, &ps.hc.hn, &hc.down, &ps.hc.down, &s.dequant)?;
        if let Some(inject) = &hc.inject {
            project_mat(ctx, pass, &ps.hc.hn, inject, &ps.hc.inj, &s.dequant)?;
        }
        pass.level_barrier(&[&ps.hc.down, &ps.hc.inj])?;
        silu_scaled_bf16(ctx, pass, &ps.hc.down, &ps.hc.act, inv_g)?;
        pass.level_barrier(&[&ps.hc.act])?;
        project_mat(ctx, pass, &ps.hc.act, &hc.up, &ps.hc.up, &s.dequant)?;
        pass.level_barrier(&[&ps.hc.up])?;
        hc_mix_bf16(ctx, pass, &ps.hc.up, &ps.hc.hn, &ps.hc.mixed, h, g)?;
        pass.level_barrier(&[&ps.hc.mixed])
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        hyper: &Tensor,
        s: &Scratch,
        ps: &PrefillScratch,
        conv_capture: Option<&Tensor>,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        self.gather_ngram(ctx, pass, w, p, ps.m)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.emb, &p.query_n])?;
        project_mat(ctx, pass, &p.emb, &w.key_proj, &p.key, &s.dequant)?;
        project_mat(ctx, pass, &p.emb, &w.value_proj, &p.value, &s.dequant)?;
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16(
            ctx, pass, &p.key_n, &p.query_n, &p.value, &p.gated, h, g,
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        if let Some(keep) = conv_capture {
            copy_words(ctx, pass, &p.gated_n, keep)?;
        }
        ple::ple_conv1d_prefill(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &pst.conv_windows[1 - conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            hyper,
            ple.ngram_size,
        )?;
        pass.level_barrier(&[hyper, &pst.conv_windows[1 - conv_slot]])
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        gdn_state: &Tensor,
        conv_windows: &[Tensor; 2],
        conv_slot: usize,
        capture: Option<&Capture>,
    ) -> Result<()> {
        let cfg = &self.config;
        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.in_proj,
            &ps.stack,
            [
                (&w.in_proj_qkv, &ps.qkv),
                (&w.in_proj_z, &ps.z),
                (&w.in_proj_a, &ps.a),
                (&w.in_proj_b, &ps.b),
            ],
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.qkv, &ps.z, &ps.a, &ps.b])?;
        if let Some(c) = capture {
            copy_words(ctx, pass, &ps.qkv, &c.conv_in)?;
        }
        conv1d_prefill(
            ctx,
            pass,
            &conv_windows[conv_slot],
            &conv_windows[1 - conv_slot],
            &ps.qkv,
            &w.conv_w,
            &ps.qkv_conv,
        )?;
        pass.level_barrier(&[&ps.qkv_conv, &conv_windows[1 - conv_slot]])?;
        let staging = GdnRegscanStaging {
            qk_norm: &ps.gdn_stage.qk_norm,
            decay: &ps.gdn_stage.decay,
            beta: &ps.gdn_stage.beta,
        };
        gdn_prefill_mid(
            ctx,
            pass,
            &ps.qkv_conv,
            &ps.a,
            &ps.b,
            &w.a_log,
            &w.dt_bias,
            &staging,
            gdn_state,
            &ps.gdn_out,
            self.gdn_scale,
            cfg.linear_num_key_heads,
            capture.and_then(|c| c.mid.as_ref()),
        )?;
        pass.level_barrier(&[&ps.gdn_out, gdn_state])?;
        gated_rmsnorm(
            ctx,
            pass,
            &ps.gdn_out,
            &ps.z,
            &w.norm_w,
            &ps.gdn_gated,
            cfg.rms_norm_eps,
            self.gdn_gate,
        )?;
        pass.level_barrier(&[&ps.gdn_gated])?;
        project_mat(ctx, pass, &ps.gdn_gated, &w.out_proj, &ps.branch_out, &s.dequant)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
        rope: Rope<'_>,
    ) -> Result<()> {
        self.attn_batched_theta(
            ctx,
            pass,
            w,
            s,
            ps,
            k_cache,
            v_cache,
            idx_keys,
            blk_keys,
            AttnPos::host(pos),
            self.config.rope_parameters.rope_theta,
            rope,
        )
    }

    /// The attention branch over `ps.hc.mixed` at sequence indices
    /// `pos..pos+m` against the given caches, with an explicit RoPE base (the
    /// draft head declares its own) and the rule turning an index into a
    /// rotary position (`rope`). Cache slots and indexer blocks always take
    /// the sequence index.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_batched_theta(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        at: AttnPos<'_>,
        theta: f32,
        rope: Rope<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let m = ps.m;
        let idx = &cfg.indexer;
        let (nq, hd) = (cfg.num_attention_heads, cfg.head_dim);
        let pos = at.pos;

        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.qkv_proj,
            &ps.stack,
            [(&w.q_proj, &ps.qg), (&w.k_proj, &ps.k_new), (&w.v_proj, &ps.v_new)],
            &s.dequant,
        )?;
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.qg, &ps.k_new, &ps.v_new, &ps.idx_qk])?;
        split_q_gate(ctx, pass, &ps.qg, &ps.q, &ps.gate)?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.k_new,
            &w.k_norm,
            &ps.k_new,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        // Indexer: queries for this chunk, raw keys into the cache.
        qsa::qsa_prep_q(
            ctx,
            pass,
            &ps.idx_qk,
            &w.indexer.q_norm,
            &ps.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
            rope,
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &ps.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[&ps.q, &ps.gate, &ps.k_new, &ps.idx_q, idx_keys])?;
        rmsnorm_bf16(ctx, pass, &ps.q, &w.q_norm, &ps.q, eps, NORM_WEIGHT_BIAS)?;
        rope_neox(
            ctx,
            pass,
            &ps.k_new,
            cfg.num_key_value_heads,
            rot,
            pos,
            theta,
            rope,
        )?;
        // Block keys for every block this chunk completes.
        let ratio = idx.compress_ratio;
        match (pos.arg, at.block) {
            (Arg::Const(p), _) => {
                let first_block = p / ratio;
                let complete = (p + m) / ratio;
                if complete > first_block {
                    qsa::qsa_block_keys(
                        ctx,
                        pass,
                        idx_keys,
                        &w.indexer.k_norm,
                        blk_keys,
                        ratio,
                        first_block,
                        complete - first_block,
                        rot,
                        theta,
                        eps,
                        rope,
                    )?;
                }
            }
            (Arg::Gpu(_), Some((block, count))) => {
                // The GPU decided whether this row completes a block; the
                // dispatch is sized for one block and gated by its count word.
                ensure!(
                    m == 1 && pos.max - pos.min < ratio,
                    "a GPU-supplied position needs a single row within one indexer block"
                );
                let first_min = pos.min / ratio;
                if (pos.max + 1) / ratio > first_min {
                    qsa::qsa_block_keys(
                        ctx,
                        pass,
                        idx_keys,
                        &w.indexer.k_norm,
                        blk_keys,
                        ratio,
                        Pos::gpu(block, first_min, pos.max / ratio),
                        Pos::gpu(count, 0, 1),
                        rot,
                        theta,
                        eps,
                        rope,
                    )?;
                }
            }
            (Arg::Gpu(_), None) => {
                anyhow::bail!("a GPU-supplied position needs its block words")
            }
        }
        pass.level_barrier(&[&ps.q, &ps.k_new, blk_keys])?;
        rope_neox(ctx, pass, &ps.q, nq, rot, pos, theta, rope)?;
        scatter_kv(ctx, pass, k_cache, &ps.k_new, pos)?;
        scatter_kv(ctx, pass, v_cache, &ps.v_new, pos)?;
        pass.level_barrier(&[&ps.q, k_cache, v_cache])?;

        if pos.max + m <= idx.dense_limit() {
            // Every query sees at most the budget: the selection is the whole
            // causal window, so the dense kernel is exact.
            sdpa_prefill(
                ctx,
                pass,
                &ps.q,
                k_cache,
                v_cache,
                &ps.attn_o,
                pos,
                self.attn_scale,
            )?;
        } else {
            // Rows whose causal window still fits the budget select every
            // visible block, so the dense kernel is exact for them: the
            // chunk's prefix up to the dense limit (empty when the chunk
            // starts past it). The rest goes through the indexer.
            let dense_rows = idx.dense_limit().saturating_sub(pos.max).min(m);
            if dense_rows > 0 {
                let q = ps.q.view(0, &[dense_rows, nq, hd])?;
                let out = ps.attn_o.view(0, &[dense_rows, nq, hd])?;
                sdpa_prefill(
                    ctx,
                    pass,
                    &q,
                    k_cache,
                    v_cache,
                    &out,
                    pos,
                    self.attn_scale,
                )?;
            }
            let qb_cap = ps.qsa.n_sel.numel();
            for q0 in (dense_rows..m).step_by(qb_cap) {
                let qb = qb_cap.min(m - q0);
                let base = pos.offset(q0)?;
                let nb_max = qsa::visible_blocks(base.max + qb - 1, idx.compress_ratio);
                let idx_q = ps.idx_q.view(
                    q0 * idx.n_heads * INDEXER_D,
                    &[qb, idx.n_heads, INDEXER_D],
                )?;
                let q = ps.q.view(q0 * nq * hd, &[qb, nq, hd])?;
                let out = ps.attn_o.view(q0 * nq * hd, &[qb, nq, hd])?;
                qsa::qsa_scores(
                    ctx,
                    pass,
                    &idx_q,
                    blk_keys,
                    &ps.qsa.scores,
                    idx.n_heads,
                    nb_max,
                    base,
                    idx.compress_ratio,
                )?;
                pass.level_barrier(&[&ps.qsa.scores])?;
                qsa::qsa_select_blocks(
                    ctx,
                    pass,
                    &ps.qsa.scores,
                    &ps.qsa.sel,
                    &ps.qsa.n_sel,
                    qb,
                    nb_max,
                    base,
                    idx.compress_ratio,
                    idx.block_topk(),
                )?;
                pass.level_barrier(&[&ps.qsa.sel, &ps.qsa.n_sel])?;
                match ps.qsa.route.for_rows(qb) {
                    SparseAttnRoute::Split => qsa::qsa_attention(
                        ctx,
                        pass,
                        &q,
                        k_cache,
                        v_cache,
                        &ps.qsa.sel,
                        &ps.qsa.n_sel,
                        &out,
                        &ps.qsa.split_scratch(),
                        qb,
                        idx.block_topk(),
                        idx.compress_ratio,
                        base,
                        self.attn_scale,
                    )?,
                    SparseAttnRoute::Tiled { heads_per_pass } => {
                        let tiles = &ps.qsa.tiles;
                        qsa::qsa_tile_union(
                            ctx,
                            pass,
                            &ps.qsa.sel,
                            &ps.qsa.n_sel,
                            tiles,
                            qb,
                            idx.block_topk(),
                            idx.compress_ratio,
                            base,
                        )?;
                        pass.level_barrier(&[
                            &tiles.union_blk,
                            &tiles.union_mask,
                            &tiles.n_union,
                            &tiles.tail_mask,
                        ])?;
                        qsa::qsa_attention_tiled(
                            ctx,
                            pass,
                            &q,
                            k_cache,
                            v_cache,
                            tiles,
                            &out,
                            qb,
                            idx.compress_ratio,
                            base,
                            self.attn_scale,
                            heads_per_pass,
                        )?;
                    }
                }
                // The next sub-batch reuses the score/selection/union scratch.
                pass.level_barrier(&[&out])?;
            }
        }
        pass.level_barrier(&[&ps.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &ps.gate, &ps.attn_o, &ps.attn_gated)?;
        pass.level_barrier(&[&ps.attn_gated])?;
        project_mat(ctx, pass, &ps.attn_gated, &w.o_proj, &ps.branch_out, &s.dequant)
    }

    // --- draft head -----------------------------------------------------------

    /// Runs the draft head over the chunk during prefill so its caches keep up
    /// with the trunk: row `j` pairs the trunk hidden of token `pos+j-1` with
    /// token `pos+j` (the previous chunk's last hidden comes from the state;
    /// at position 0 there is no earlier hidden, so that row is skipped).
    /// Afterwards the state's hidden holds the chunk's last row. Row `j`'s
    /// token is `pos+j`; when that is an image placeholder the head is fed
    /// the image's row instead of the placeholder's embedding (`images`), as
    /// the trunk was.
    #[allow(clippy::too_many_arguments)]
    fn mtp_catch_up(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        mst: &MtpState,
        pos: usize,
        s: &Scratch,
        ps: &PrefillScratch,
        rope: Rope<'_>,
        images: &[ImageEmbeds<'_>],
    ) -> Result<()> {
        let wide = self.config.hc_width();
        let m = ps.m;
        let hidden_in = ps
            .mtp_hidden_in
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without prefill scratch"))?;
        let (rows, pos0) = if pos > 0 { (m, pos - 1) } else { (m - 1, 0) };
        if rows > 0 {
            if pos > 0 {
                copy_words(ctx, pass, &mst.hidden, &hidden_in.view(0, &[wide])?)?;
                if m > 1 {
                    copy_words(
                        ctx,
                        pass,
                        &ps.hyper.view(0, &[m - 1, wide])?,
                        &hidden_in.view(wide, &[m - 1, wide])?,
                    )?;
                }
            } else {
                copy_words(
                    ctx,
                    pass,
                    &ps.hyper.view(0, &[m - 1, wide])?,
                    &hidden_in.view(0, &[m - 1, wide])?,
                )?;
            }
            let (ids, first_token) = if pos > 0 {
                (ps.ids.view(0, &[m])?, pos)
            } else {
                (ps.ids.view(1, &[m - 1])?, 1)
            };
            let overrides =
                row_overrides(images, first_token, rows, self.config.hidden_size)?;
            pass.level_barrier(&[hidden_in])?;
            let ps_rows = ps.rows(rows)?;
            self.mtp_block(
                ctx,
                pass,
                mtp,
                mst,
                &hidden_in.view(0, &[rows, wide])?,
                &ids,
                AttnPos::host(pos0),
                s,
                &ps_rows,
                rope,
                &overrides,
            )?;
        }
        // The chunk's last trunk hidden pairs with the next token, whenever
        // it arrives. (Ordered after the reads above by the block's barriers,
        // or trivially when no row ran.)
        copy_words(ctx, pass, &ps.hyper.view((m - 1) * wide, &[wide])?, &mst.hidden)?;
        pass.level_barrier(&[&mst.hidden])
    }

    /// The draft head's block over `rows` inputs: `hidden` (`[rows, G*H]`
    /// trunk or head residuals) and `ids` (`[rows]`, the tokens following
    /// them) become the head's residual in `ps.mtp_hyper`, and the block runs
    /// at head positions `pos0..pos0+rows` against the head's caches, roped
    /// by `rope`; `overrides` are the image rows standing in for placeholder
    /// ids among `ids`. Leaves `ps.mtp_hyper` ordered.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_block(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        mst: &MtpState,
        hidden: &Tensor,
        ids: &Tensor,
        pos0: AttnPos<'_>,
        s: &Scratch,
        ps: &PrefillScratch,
        rope: Rope<'_>,
        overrides: &[RowOverride],
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        let rows = ps.m;
        ensure!(
            hidden.shape() == [rows, g * h] && ids.numel() == rows,
            "draft head input shape mismatch"
        );
        let hyper = ps
            .mtp_hyper
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without scratch"))?;
        let theta = cfg.mtp.map_or(cfg.rope_parameters.rope_theta, |m| m.rope_theta);

        // Per stream: fc_hidden(norm(stream)); shared: fc_embedding(norm(emb)).
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hidden,
            &mtp.norm_hidden,
            &ps.hc.hn,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, ids, &ps.x)?;
        if !overrides.is_empty() {
            pass.level_barrier(&[&ps.x])?;
            override_rows(ctx, pass, &ps.x, overrides, h)?;
        }
        pass.level_barrier(&[&ps.hc.hn, &ps.x])?;
        let hn_streams = ps.hc.hn.view(0, &[rows * g, h])?;
        let hyper_streams = hyper.view(0, &[rows * g, h])?;
        project_mat(
            ctx,
            pass,
            &hn_streams,
            &mtp.fc_hidden,
            &hyper_streams,
            &s.dequant,
        )?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.x,
            &mtp.norm_embedding,
            &ps.x,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[hyper, &ps.x])?;
        project_mat(ctx, pass, &ps.x, &mtp.fc_embedding, &ps.branch_out, &s.dequant)?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_broadcast_bf16(ctx, pass, &ps.branch_out, &ps.hc.up, h, g)?;
        pass.level_barrier(&[&ps.hc.up])?;
        add_bf16(ctx, pass, hyper, &ps.hc.up, hyper)?;
        pass.level_barrier(&[hyper])?;

        // The block itself: a trunk-style attention layer with its own RoPE base.
        let Mixer::Attn(w) = &mtp.layer.mixer else {
            anyhow::bail!("draft head block is not attention")
        };
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = &mst.layer
        else {
            anyhow::bail!("draft head state is not attention")
        };
        self.hc_read_batched(ctx, pass, &mtp.layer.attn_hc, hyper, s, ps)?;
        self.attn_batched_theta(
            ctx, pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos0, theta,
            rope,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])?;
        self.hc_read_batched(ctx, pass, &mtp.layer.mlp_hc, hyper, s, ps)?;
        prefill_moe(
            ctx,
            pass,
            &moe_dims(cfg),
            &mtp.layer.ffn,
            &PrefillMoeIo {
                x: &ps.hc.mixed,
                out: &ps.branch_out,
                stack: &ps.stack,
                mlp_gate: &ps.mlp_gate,
                mlp_up: &ps.mlp_up,
                mlp_act: &ps.mlp_act,
                dequant: &s.dequant,
            },
            &ps.moe,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(ctx, pass, hyper, &ps.branch_out, &ps.hc.inj, h, g)?;
        pass.level_barrier(&[hyper])
    }

    // --- decode -------------------------------------------------------------

    /// Encodes one concurrent-dispatch decode graph at `state.pos`. The
    /// concurrent encoder has no implicit dispatch ordering: `level_barrier`
    /// marks every true inter-level data edge.
    #[allow(clippy::too_many_arguments)]
    fn encode_decode_graph(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
        park: Option<u64>,
    ) -> Result<()> {
        ensure!(state.pos < state.capacity, "sequence full ({})", state.capacity);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let conv_slot = state.conv_slot;

        let ids = s.next_token.view(slot_in, &[1])?;
        quant::gather_rows_q4(ctx, pass, &self.weights.embed_tokens, &ids, &s.x)?;
        pass.level_barrier(&[&s.x])?;
        hc_broadcast_bf16(ctx, pass, &s.x, &s.hyper, h, g)?;
        pass.level_barrier(&[&s.hyper])?;

        for (layer, lstate) in self.weights.layers.iter().zip(state.layers.iter()) {
            if let (Some(ple_w), Some(ple_s), Some(pst)) =
                (&layer.ple, &s.ple, &state.ple)
            {
                if let Some(value) = park {
                    // First use of host-staged data: the n-gram rows. The
                    // embedding and the layers above ran while the host
                    // read the token and staged them.
                    s.sync.encode_wait(pass, value)?;
                }
                self.ple_decode(ctx, pass, ple_w, ple_s, pst, conv_slot, s)?;
            }

            self.hc_read_decode(ctx, pass, &layer.attn_hc, s)?;
            match (&layer.mixer, lstate) {
                (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows }) => {
                    self.gdn_decode_branch(
                        ctx,
                        pass,
                        w,
                        s,
                        state,
                        &conv_windows[conv_slot],
                    )?;
                }
                (
                    Mixer::Attn(w),
                    LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
                ) => {
                    self.attn_decode_branch(
                        ctx,
                        pass,
                        w,
                        s,
                        k_cache,
                        v_cache,
                        idx_keys,
                        blk_keys,
                        state.pos,
                        state.rope_delta,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            pass.level_barrier(&[&s.branch_out])?;
            hc_inject_bf16(ctx, pass, &s.hyper, &s.branch_out, &s.hc.inj, h, g)?;
            pass.level_barrier(&[&s.hyper])?;

            self.hc_read_decode(ctx, pass, &layer.mlp_hc, s)?;
            decode_moe(
                ctx,
                pass,
                &moe_dims(cfg),
                &layer.ffn,
                &DecodeMoeIo {
                    x: &s.hc.mixed,
                    out: &s.branch_out,
                    mlp_gu: &s.mlp_gu,
                    mlp_gate: &s.mlp_gate,
                    mlp_up: &s.mlp_up,
                    mlp_act: &s.mlp_act,
                },
                &s.moe,
            )?;
            pass.level_barrier(&[&s.branch_out])?;
            hc_inject_bf16(ctx, pass, &s.hyper, &s.branch_out, &s.hc.inj, h, g)?;
            pass.level_barrier(&[&s.hyper])?;
        }

        self.hc_read_decode(ctx, pass, &self.weights.final_mixer, s)?;
        quant::gemv_quant(ctx, pass, &self.weights.lm_head, &s.hc.mixed, &s.logits)?;
        pass.level_barrier(&[&s.logits])?;
        let out = s.next_token.view(slot_out, &[1])?;
        sample_f32(ctx, pass, &s.logits, &s.sampler, draw.params, draw.step, &out)?;
        pass.level_barrier(&[&s.next_token])?;
        s.sync.signal_done(pass)
    }

    /// Single-row hyper-connection read: `s.hyper` → `s.hc.mixed` (+ `inj`).
    /// Two fused dispatches (norm + down/inject GEMV; SiLU + up GEMV + mix)
    /// standing in for the six-kernel batched read; the down half keeps the
    /// normalized stream in f32 where the batched path rounds it to bf16, so
    /// the two differ at bf16 rounding level. Leaves `mixed`/`inj` ordered
    /// for the caller.
    fn hc_read_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        s: &Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        hc_read_down_q8(
            ctx,
            pass,
            &s.hyper,
            &hc.norm,
            &hc.down,
            hc.inject.as_ref(),
            &s.hc.down,
            &s.hc.inj,
            &s.hc.inv_rms,
            h,
            g,
            cfg.rms_norm_eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&s.hc.down, &s.hc.inj, &s.hc.inv_rms])?;
        hc_read_up_mix_q8(
            ctx,
            pass,
            &hc.up,
            &s.hc.down,
            &s.hyper,
            &hc.norm,
            &s.hc.inv_rms,
            &s.hc.mixed,
            h,
            g,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&s.hc.mixed])
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        s: &Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        self.gather_ngram(ctx, pass, w, p, 1)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &s.hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.emb, &p.query_n])?;
        quant::gemv_quant(ctx, pass, &w.key_proj, &p.emb, &p.key)?;
        quant::gemv_quant(ctx, pass, &w.value_proj, &p.emb, &p.value)?;
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16(
            ctx, pass, &p.key_n, &p.query_n, &p.value, &p.gated, h, g,
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            NORM_WEIGHT_BIAS,
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        ple::ple_conv1d_step(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            &s.hyper,
            ple.ngram_size,
        )?;
        pass.level_barrier(&[&s.hyper, &pst.conv_windows[conv_slot]])
    }

    fn gdn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        gdn_state: &Tensor,
        conv_window: &Tensor,
    ) -> Result<()> {
        quant::gemv_quant(ctx, pass, &w.in_proj, &s.hc.mixed, &s.gdn_in)?;
        pass.level_barrier(&[&s.gdn_in])?;
        conv1d_step(ctx, pass, conv_window, &s.qkv, &w.conv_w, &s.qkv_conv)?;
        pass.level_barrier(&[&s.qkv_conv, conv_window])?;
        gdn_step_gated_fused(
            ctx,
            pass,
            &s.qkv_conv,
            &s.a,
            &s.b,
            &w.a_log,
            &w.dt_bias,
            gdn_state,
            &s.z,
            &w.norm_w,
            &s.gdn_gated,
            self.gdn_scale,
            self.config.linear_num_key_heads,
            self.config.rms_norm_eps,
            self.gdn_gate,
        )?;
        pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
        quant::gemv_quant(ctx, pass, &w.out_proj, &s.gdn_gated, &s.branch_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        k_cache: &Tensor,
        v_cache: &Tensor,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
        rope_delta: i64,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;
        let idx = &cfg.indexer;
        let len = pos + 1;
        // The token sits at cache slot `pos` and rotates at `pos + delta`
        // (equal for text; VISION.md `rope_deltas` after an image).
        ensure!(
            pos as i64 + rope_delta >= 0,
            "rotary position {} + {rope_delta} is negative",
            pos
        );

        quant::gemv_quant(ctx, pass, &w.qkv_proj, &s.hc.mixed, &s.attn_qkv)?;
        quant::gemv_quant(ctx, pass, &w.indexer.qk_proj, &s.hc.mixed, &s.idx_qk)?;
        pass.level_barrier(&[&s.attn_qkv, &s.idx_qk])?;
        q_norm_rope_split_decode(
            ctx, pass, &s.qg, &w.q_norm, &s.q, &s.gate, rot, pos, theta, eps,
            rope_delta,
        )?;
        k_norm_rope_scatter_decode(
            ctx, pass, &s.k_new, &w.k_norm, k_cache, rot, pos, theta, eps, rope_delta,
        )?;
        scatter_kv(ctx, pass, v_cache, &s.v_new, pos)?;
        qsa::qsa_prep_q(
            ctx,
            pass,
            &s.idx_qk,
            &w.indexer.q_norm,
            &s.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
            Rope::Delta(rope_delta),
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &s.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[&s.q, &s.gate, k_cache, v_cache, &s.idx_q, idx_keys])?;
        if len.is_multiple_of(idx.compress_ratio) {
            // This token completes a block: its key becomes selectable from
            // the next position on (and this one, beyond the dense limit).
            let block = len / idx.compress_ratio - 1;
            qsa::qsa_block_keys(
                ctx,
                pass,
                idx_keys,
                &w.indexer.k_norm,
                blk_keys,
                idx.compress_ratio,
                block,
                1,
                rot,
                theta,
                eps,
                Rope::Delta(rope_delta),
            )?;
            pass.level_barrier(&[blk_keys])?;
        }

        if len <= idx.dense_limit() {
            sdpa_decode(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.attn_o,
                len,
                self.attn_scale,
                Some((&s.sdpa_partials, &s.sdpa_stats)),
            )?;
        } else {
            let nb = qsa::visible_blocks(pos, idx.compress_ratio);
            qsa::qsa_scores(
                ctx,
                pass,
                &s.idx_q,
                blk_keys,
                &s.qsa.scores,
                idx.n_heads,
                nb,
                pos,
                idx.compress_ratio,
            )?;
            pass.level_barrier(&[&s.qsa.scores])?;
            qsa::qsa_select_blocks(
                ctx,
                pass,
                &s.qsa.scores,
                &s.qsa.sel,
                &s.qsa.n_sel,
                1,
                nb,
                pos,
                idx.compress_ratio,
                idx.block_topk(),
            )?;
            pass.level_barrier(&[&s.qsa.sel, &s.qsa.n_sel])?;
            qsa::qsa_attention(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.qsa.sel,
                &s.qsa.n_sel,
                &s.attn_o,
                &s.qsa.split_scratch(),
                1,
                idx.block_topk(),
                idx.compress_ratio,
                pos,
                self.attn_scale,
            )?;
        }
        pass.level_barrier(&[&s.attn_o])?;
        sigmoid_mul_bf16(ctx, pass, &s.gate, &s.attn_o, &s.attn_gated)?;
        pass.level_barrier(&[&s.attn_gated])?;
        quant::gemv_quant(ctx, pass, &w.o_proj, &s.attn_gated, &s.branch_out)
    }
}

/// Rounds a requested capacity up to the growth step, within the kernel limit.
fn round_capacity(tokens: usize) -> Result<usize> {
    ensure!(tokens <= MAX_SEQ, "capacity {tokens} exceeds kernel limit {MAX_SEQ}");
    Ok(tokens.max(1).div_ceil(CAPACITY_STEP).saturating_mul(CAPACITY_STEP).min(MAX_SEQ))
}

impl DecodeStateApi for DecodeState {
    fn set_rope_delta(&mut self, delta: i64) -> Result<()> {
        self.rope_delta = delta;
        Ok(())
    }

    type Snapshot = Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        DecodeState::reset(self)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = round_capacity(tokens)?;
        let mut layers = std::mem::take(&mut self.layers);
        for lstate in &mut layers {
            self.grow_attn(ctx, lstate, capacity)?;
        }
        self.layers = layers;
        if let Some(mut mtp) = self.mtp.take() {
            self.grow_attn(ctx, &mut mtp.layer, capacity)?;
            self.mtp = Some(mtp);
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        let recurrent: usize = self
            .layers
            .iter()
            .map(|l| match l {
                LayerState::Gdn { state, conv_windows } => {
                    state.byte_len()
                        + conv_windows[0].byte_len()
                        + conv_windows[1].byte_len()
                }
                LayerState::Attn { .. } => 0,
            })
            .sum();
        let ple = self.ple.as_ref().map_or(0, |p| 2 * p.conv_windows[0].byte_len());
        let mtp = self.mtp.as_ref().map_or(0, |m| m.hidden.byte_len());
        recurrent + ple + mtp + self.cache_bytes(self.capacity)
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Snapshot> {
        let slot = self.conv_slot;
        let mut gdn = Vec::new();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                gdn.push((
                    clone_tensor(ctx, state)?,
                    clone_tensor(ctx, &conv_windows[slot])?,
                ));
            }
        }
        let ple = match &self.ple {
            Some(p) => Some((p.hist, clone_tensor(ctx, &p.conv_windows[slot])?)),
            None => None,
        };
        let mtp_hidden = match &self.mtp {
            Some(m) => Some(clone_tensor(ctx, &m.hidden)?),
            None => None,
        };
        Ok(Snapshot { pos: self.pos, gdn, ple, mtp_hidden })
    }

    fn restore(&mut self, ctx: &MetalContext, snapshot: &Snapshot) -> Result<()> {
        ensure!(
            snapshot.pos <= self.capacity,
            "snapshot position {} exceeds capacity {}",
            snapshot.pos,
            self.capacity
        );
        let mut copies = Vec::new();
        let mut saved = snapshot.gdn.iter();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows } = lstate {
                let (s, w) = saved.next().ok_or_else(|| {
                    anyhow::anyhow!("snapshot has too few GDN layers")
                })?;
                copies.push(BlitCopy {
                    src: s,
                    src_offset: 0,
                    dst: state,
                    dst_offset: 0,
                    len: state.byte_len(),
                });
                copies.push(BlitCopy {
                    src: w,
                    src_offset: 0,
                    dst: &conv_windows[0],
                    dst_offset: 0,
                    len: w.byte_len(),
                });
            }
        }
        ensure!(saved.next().is_none(), "snapshot has too many GDN layers");
        match (&mut self.ple, &snapshot.ple) {
            (Some(p), Some((hist, w))) => {
                p.hist = *hist;
                copies.push(BlitCopy {
                    src: w,
                    src_offset: 0,
                    dst: &p.conv_windows[0],
                    dst_offset: 0,
                    len: w.byte_len(),
                });
            }
            (None, None) => {}
            _ => anyhow::bail!("snapshot PLE state mismatch"),
        }
        match (&self.mtp, &snapshot.mtp_hidden) {
            (Some(m), Some(h)) => {
                copies.push(BlitCopy {
                    src: h,
                    src_offset: 0,
                    dst: &m.hidden,
                    dst_offset: 0,
                    len: h.byte_len(),
                });
            }
            (None, None) => {}
            _ => anyhow::bail!("snapshot draft-head state mismatch"),
        }
        ctx.blit_copy(&copies)?;
        self.pos = snapshot.pos;
        self.conv_slot = 0;
        self.spec = None;
        Ok(())
    }

    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()> {
        ensure!(
            tokens <= from.pos,
            "source state has fed {} tokens, {tokens} requested",
            from.pos
        );
        self.ensure_capacity(ctx, tokens)?;
        let mut copies = Vec::new();
        for (dst, src) in self.layers.iter().zip(&from.layers) {
            attn_prefix_copies(dst, src, tokens, self.ratio, &mut copies)?;
        }
        match (&self.mtp, &from.mtp) {
            (Some(dst), Some(src)) => attn_prefix_copies(
                &dst.layer,
                &src.layer,
                tokens,
                self.ratio,
                &mut copies,
            )?,
            (None, None) => {}
            _ => anyhow::bail!("draft-head state mismatch between sessions"),
        }
        ctx.blit_copy(&copies)
    }

    fn write_prefix(&self, tokens: usize, w: &mut dyn std::io::Write) -> Result<()> {
        ensure!(
            tokens <= self.pos,
            "state has fed {} tokens, {tokens} requested",
            self.pos
        );
        let mut sink = |t: &Tensor, _: usize| t.write_to(w);
        for lstate in self.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)) {
            attn_prefix_regions(lstate, tokens, tokens, self.ratio, &mut sink)?;
        }
        Ok(())
    }

    /// Reads a `tokens`-token prefix out of a layout written for `written`
    /// tokens. Each region's rows past `tokens` are skipped, not stopped at:
    /// the regions follow one another, so stopping early would read the next
    /// head's rows as this one's (a checkpoint hit inside a longer disk entry
    /// once did exactly that).
    fn read_prefix(
        &mut self,
        ctx: &MetalContext,
        written: usize,
        tokens: usize,
        r: &mut dyn std::io::Read,
    ) -> Result<()> {
        ensure!(
            tokens <= written,
            "prefix of {tokens} tokens from a layout of {written}"
        );
        self.ensure_capacity(ctx, tokens)?;
        let mut source = |t: &Tensor, tail: usize| {
            t.fill_from(r)?;
            if tail > 0 {
                let skipped = std::io::copy(
                    &mut std::io::Read::take(&mut *r, tail as u64),
                    &mut std::io::sink(),
                )?;
                ensure!(
                    skipped == tail as u64,
                    "prefix file ends {} bytes early",
                    tail as u64 - skipped
                );
            }
            Ok(())
        };
        for lstate in self.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)) {
            attn_prefix_regions(lstate, tokens, written, self.ratio, &mut source)?;
        }
        Ok(())
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn arrival_probe(&self) -> Option<&SharedEvent> {
        Scratch::arrival_probe(self)
    }

    fn resumed_probe(&self) -> Option<&SharedEvent> {
        Scratch::resumed_probe(self)
    }

    fn logits(&self) -> &Tensor {
        &self.logits
    }

    fn begin_request(&self) {
        self.sampler.reset_counts();
    }

    /// The block selection of the last sparse-attention decode step (query
    /// row 0): `null` until a step ran past the dense limit.
    fn debug_json(&self) -> Result<serde_json::Value> {
        let n = self.qsa.n_sel.to_u32()?[0] as usize;
        let k_max = self.qsa.sel.shape()[1];
        let sel = self.qsa.sel.to_u32()?;
        // The score row is valid up to the query's visible block count, which
        // the reader derives from the position; cap the dump for sanity.
        let scores = self.qsa.scores.to_f32()?;
        let dumped = scores.len().min(self.qsa.scores.shape()[1]).min(65536);
        Ok(serde_json::json!({
            "selected_blocks": &sel[..n.min(k_max)],
            "block_scores": &scores[..dumped],
        }))
    }
}

impl LanguageModel for Qwen4ExpModel {
    type State = DecodeState;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "Qwen3.8-Flash-Next";

    fn load(ctx: &MetalContext, dir: &Path, options: &LoadOptions) -> Result<Self> {
        Qwen4ExpModel::load_with(
            ctx,
            dir,
            options.ngram_storage,
            options.mtp_drafts > 0,
            options.vision,
        )
    }

    fn vision_tower(&self) -> Option<VisionTower> {
        Some(match (&self.config.vision, &self.weights.vision) {
            (None, _) => VisionTower::Absent,
            (Some(_), None) => VisionTower::Off,
            (Some(v), Some(w)) => {
                VisionTower::Loaded { bytes: w.bytes(), blocks: v.depth }
            }
        })
    }

    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        self.config.eos_token_id.as_vec()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn persistence_format(&self) -> Option<String> {
        let cfg = &self.config;
        Some(format!(
            "{};layers={};kv={}x{};indexer={}/{};mtp={}",
            super::config::FORMAT,
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            INDEXER_D,
            cfg.indexer.compress_ratio,
            self.weights.mtp.is_some()
        ))
    }

    fn read_snapshot(
        &self,
        ctx: &MetalContext,
        r: &mut dyn std::io::Read,
    ) -> Result<Snapshot> {
        let cfg = &self.config;
        let mut pos = [0u8; 8];
        r.read_exact(&mut pos)?;
        let pos = usize::try_from(u64::from_le_bytes(pos))?;
        let heads = cfg.linear_num_value_heads;
        let c = cfg.gdn_conv_channels();
        let kd = cfg.linear_conv_kernel_dim;
        let mut gdn = Vec::new();
        for _ in cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, super::config::LayerType::LinearAttention))
        {
            let state = Tensor::zeros(
                ctx,
                &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                GDN_STATE_DTYPE,
            )?;
            state.fill_from(r)?;
            let window = Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?;
            window.fill_from(r)?;
            gdn.push((state, window));
        }
        let ple = match &cfg.ple {
            Some(p) => {
                let mut hist = [0u8; 8];
                r.read_exact(&mut hist)?;
                let h = [
                    u32::from_le_bytes(hist[..4].try_into()?),
                    u32::from_le_bytes(hist[4..].try_into()?),
                ];
                let window = Tensor::zeros(
                    ctx,
                    &[cfg.hc_width(), p.conv_state_len()],
                    DType::BF16,
                )?;
                window.fill_from(r)?;
                Some((h, window))
            }
            None => None,
        };
        let mtp_hidden = match &self.weights.mtp {
            Some(_) => {
                let hidden = Tensor::zeros(ctx, &[cfg.hc_width()], DType::BF16)?;
                hidden.fill_from(r)?;
                Some(hidden)
            }
            None => None,
        };
        Ok(Snapshot { pos, gdn, ple, mtp_hidden })
    }

    fn bytes_per_token(&self) -> usize {
        let cfg = &self.config;
        let attn_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, super::config::LayerType::FullAttention))
            .count()
            + usize::from(self.weights.mtp.is_some());
        attn_layers
            * (2 * cfg.num_key_value_heads * cfg.head_dim * 2
                + INDEXER_D * 2
                + INDEXER_D * 2 / cfg.indexer.compress_ratio)
    }

    fn warm_storage(&self, lock: bool) -> Result<u64> {
        match self.ple_table() {
            Some(NgramTable::Paged(table)) => table.preload(lock),
            _ => Ok(0),
        }
    }

    fn paged_storage_bytes(&self) -> usize {
        match self.ple_table() {
            Some(NgramTable::Paged(table)) => table.bytes() as usize,
            _ => 0,
        }
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<DecodeState> {
        Qwen4ExpModel::new_state(self, ctx, capacity)
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        Qwen4ExpModel::new_scratch_with_capacity(self, ctx, capacity_tokens)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        Qwen4ExpModel::prefill(self, ctx, state, scratch, tokens, draw)
    }

    fn prefill_with_vision(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&VisionInput<'_>>,
    ) -> Result<()> {
        Qwen4ExpModel::prefill_with_vision(
            self, ctx, state, scratch, tokens, draw, vision,
        )
    }

    fn encode_image(
        &self,
        ctx: &MetalContext,
        scratch: &mut Scratch,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        Qwen4ExpModel::encode_image(self, ctx, scratch, pixels, grid_h, grid_w)
    }

    fn prepare_step_inputs(
        &self,
        state: &mut DecodeState,
        scratch: &Scratch,
        token: u32,
    ) -> Result<()> {
        Qwen4ExpModel::prepare_step_inputs(self, state, scratch, token)
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        Qwen4ExpModel::encode_decode_step(
            self, ctx, state, scratch, slot_in, slot_out, draw, None,
        )
    }

    fn supports_parking(&self) -> bool {
        true
    }

    fn encode_parked_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        let value = scratch.sync.arm()?;
        match Qwen4ExpModel::encode_decode_step(
            self,
            ctx,
            state,
            scratch,
            slot_in,
            slot_out,
            draw,
            Some(value),
        ) {
            Ok(pass) => Ok(pass),
            Err(e) => {
                // Nothing was committed; free the claimed value.
                scratch.sync.release()?;
                Err(e)
            }
        }
    }

    fn release_parked(&self, scratch: &Scratch) -> Result<()> {
        scratch.sync.release()
    }

    fn max_drafts(&self) -> usize {
        Qwen4ExpModel::max_drafts(self)
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        first: u32,
        drafts: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<u32>> {
        Qwen4ExpModel::draft_initial(
            self, ctx, state, scratch, first, drafts, params, step0,
        )
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        parked: Option<PendingPass<'a>>,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        Qwen4ExpModel::verify(
            self,
            ctx,
            state,
            scratch,
            pending,
            drafts,
            params,
            step0,
            parked,
            next_drafts,
        )
    }

    fn finish_speculation<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        accepted: usize,
        next: Option<crate::engine::NextStep<'_>>,
        draft: PendingPass<'a>,
    ) -> Result<(Vec<u32>, Option<PendingPass<'a>>)> {
        Qwen4ExpModel::finish_speculation(
            self, ctx, state, scratch, accepted, next, draft,
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/qsa_tiles.rs"]
mod qsa_tile_tests;
