//! The model-agnostic surface the generator, session cache, and API server
//! drive. Each supported architecture implements it over its own state and
//! scratch types; `serve::run` picks the implementation from the checkpoint's
//! `model_type`.

use std::path::Path;

use anyhow::Result;

use crate::kernels::sample::SamplingParams;
use crate::metal::{EncodedPass, MetalContext, PassTiming, PendingPass, SharedEvent};
use crate::tensor::Tensor;

/// One piece of a persisted layout (a disk-tier file): bytes the layout
/// computes, such as a position header, or a range of a shared-storage
/// buffer that holds cache rows or recurrent state.
///
/// A layout lets the session cache write a parked session from another
/// thread: the writer gets addresses and lengths, never the buffers'
/// handles (which are `Rc`s owned by the engine thread), and writes them
/// with [`write_layout`] under that function's contract.
pub enum Segment {
    Bytes(Vec<u8>),
    Host(HostRange),
}

/// An address and a length inside a shared-storage buffer.
#[derive(Clone, Copy)]
pub struct HostRange {
    ptr: *const u8,
    len: usize,
}

// SAFETY: a plain address and length; reading through it is `write_layout`'s
// contract, which the caller upholds on whichever thread it runs.
unsafe impl Send for HostRange {}

impl Segment {
    /// The bytes of `t` (honouring a view's offset), by address.
    pub fn tensor(t: &Tensor) -> Self {
        Self::host(t.contents())
    }

    /// `bytes` by address: the segment reads them again when written.
    pub fn host(bytes: &[u8]) -> Self {
        Self::Host(HostRange { ptr: bytes.as_ptr(), len: bytes.len() })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Host(range) => range.len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Bytes a layout writes, total.
pub fn layout_len(segments: &[Segment]) -> usize {
    segments.iter().map(Segment::len).sum()
}

/// Writes `segments` to `w` in order, in pieces of at most 8 MB, stopping
/// with an error between pieces once `cancel` is set.
///
/// # Safety
///
/// Every [`Segment::Host`] range must stay allocated, and neither the GPU
/// nor the host may write it, until this returns: the owner of the buffers
/// (a session parked for the write, or the state or snapshot a caller holds
/// borrowed for the call) keeps them alive and idle meanwhile.
pub unsafe fn write_layout(
    segments: &[Segment],
    w: &mut dyn std::io::Write,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    const PIECE: usize = 8 << 20;
    for segment in segments {
        let bytes: &[u8] = match segment {
            Segment::Bytes(bytes) => bytes,
            // SAFETY: the caller's contract above.
            Segment::Host(r) => unsafe { std::slice::from_raw_parts(r.ptr, r.len) },
        };
        for piece in bytes.chunks(PIECE) {
            if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed)) {
                anyhow::bail!("cancelled");
            }
            w.write_all(piece)?;
        }
    }
    Ok(())
}

/// Whether two layouts write the same bytes (however they are split into
/// segments).
///
/// # Safety
///
/// [`write_layout`]'s, for both.
pub unsafe fn layouts_equal(a: &[Segment], b: &[Segment]) -> bool {
    fn bytes(s: &Segment) -> &[u8] {
        match s {
            Segment::Bytes(bytes) => bytes,
            // SAFETY: the caller's contract.
            Segment::Host(r) => unsafe { std::slice::from_raw_parts(r.ptr, r.len) },
        }
    }
    let (mut ia, mut ib) = (a.iter().map(bytes), b.iter().map(bytes));
    let (mut ca, mut cb): (&[u8], &[u8]) = (&[], &[]);
    loop {
        while ca.is_empty() {
            match ia.next() {
                Some(s) => ca = s,
                None => break,
            }
        }
        while cb.is_empty() {
            match ib.next() {
                Some(s) => cb = s,
                None => break,
            }
        }
        if ca.is_empty() || cb.is_empty() {
            return ca.is_empty() && cb.is_empty();
        }
        let n = ca.len().min(cb.len());
        if ca[..n] != cb[..n] {
            return false;
        }
        (ca, cb) = (&ca[n..], &cb[n..]);
    }
}

/// A copy of the recurrent part of a decode state (everything that is not a
/// per-token cache) at one position. Together with the per-token caches that
/// are still in place up to that position it lets a state resume from there.
pub trait SnapshotApi {
    /// Tokens fed when the snapshot was taken.
    fn pos(&self) -> usize;
    /// GPU bytes the snapshot holds.
    fn bytes(&self) -> usize;
    /// The serialized snapshot as a layout (for the on-disk session tier;
    /// models that support it also implement
    /// [`LanguageModel::read_snapshot`]). The host ranges point into the
    /// snapshot's own buffers.
    fn layout(&self) -> Result<Vec<Segment>> {
        anyhow::bail!("this model does not persist sessions")
    }
    /// Serializes the snapshot: [`Self::layout`], written. GPU idle.
    fn write_to(&self, w: &mut dyn std::io::Write) -> Result<()> {
        let layout = self.layout()?;
        // SAFETY: the ranges point into `self`'s buffers, borrowed for the
        // call; nothing writes a snapshot after it was taken.
        unsafe { write_layout(&layout, w, None) }
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

    /// The first `tokens` entries of every per-token cache as a layout, in
    /// the order [`Self::read_prefix`] expects; the host ranges point into
    /// this state's buffers.
    fn prefix_layout(&self, tokens: usize) -> Result<Vec<Segment>> {
        let _ = tokens;
        anyhow::bail!("this model does not persist sessions")
    }

    /// The recurrent state at the current position as a layout, exactly the
    /// bytes a [`Self::snapshot`] taken now would write, read from this
    /// state's own buffers instead of a copy: a state at rest that nothing
    /// feeds can be persisted without allocating a snapshot.
    fn live_layout(&self) -> Result<Vec<Segment>> {
        anyhow::bail!("this model does not persist sessions")
    }

    /// Streams the first `tokens` entries of every per-token cache to `w`:
    /// [`Self::prefix_layout`], written. GPU idle.
    fn write_prefix(&self, tokens: usize, w: &mut dyn std::io::Write) -> Result<()> {
        let layout = self.prefix_layout(tokens)?;
        // SAFETY: the ranges point into `self`'s buffers, borrowed for the
        // call, and the GPU is idle on them (the contract above).
        unsafe { write_layout(&layout, w, None) }
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

/// `--memory-limit-gb` in bytes for `LoadOptions::memory_limit`, refusing
/// the old `--memory-gb` (`retired`), which meant the machine's memory and
/// is now `LILY_MEMORY_GB`, instead of reading it as a limit.
pub fn memory_limit_bytes(
    limit_gb: Option<f64>,
    retired: Option<f64>,
) -> anyhow::Result<Option<u64>> {
    if let Some(gb) = retired {
        anyhow::bail!(
            "--memory-gb {gb} is gone: --memory-limit-gb {gb} caps what lily uses (the plan still keeps \
             its 12 GB reserve free), LILY_MEMORY_GB={gb} plans as if the machine had {gb} GB \
             (trying the small-machine mode on a bigger machine)"
        );
    }
    limit_gb
        .map(|gb| {
            anyhow::ensure!(
                gb.is_finite() && gb > 0.0,
                "--memory-limit-gb {gb}: a positive size in GB"
            );
            Ok((gb * (1u64 << 30) as f64) as u64)
        })
        .transpose()
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
    /// The most the process should take, in bytes (`--memory-limit-gb`);
    /// `None` plans from the machine's memory alone. It only lowers what
    /// the plan would use: below what the checkpoint needs, the routed
    /// experts are cached and served from disk (`docs/low-ram-experts.md`).
    /// `LILY_MEMORY_GB` stands in for the machine's memory, not for this.
    pub memory_limit: Option<u64>,
    /// Where the expert cache persists the usage it measures while serving
    /// (the loader prefers this file, when present, over the shipped
    /// ranking); `None` persists nothing. `LILY_EXPERT_USAGE_OUT` overrides
    /// it.
    pub expert_usage_out: Option<std::path::PathBuf>,
    /// The session the server keeps room for when the expert cache engages
    /// (its `--max-seq` and checkpoints per session): the plan reserves one
    /// full session and the server budgets exactly that. `None` (the bench,
    /// the probes) reserves nothing.
    pub session_context: Option<crate::qwen4exp::weights::SessionContext>,
    /// The attention K/V caches' element format (`--kv-cache`).
    pub kv_format: crate::kernels::attention::KvFormat,
}

/// Which draw a decode pass ends with.
#[derive(Clone, Copy, Debug)]
pub struct Draw<'p> {
    pub params: &'p SamplingParams,
    /// Index of this draw within the request (the RNG counter).
    pub step: usize,
}

/// Where the wall time of one batched decode step went
/// ([`LanguageModel::decode_rows_timed`], or a step committed with `timed`
/// set): host marks and the pass's GPU span, all in seconds on the clock of
/// [`crate::metal::host_secs`], so marks and GPU times subtract. A
/// diagnostic for `lily-bench --batch-rows`; the server never asks for it.
///
/// A step committed unparked passes the marks in field order. A parked step
/// ([`LanguageModel::park_rows`]) is encoded and committed (`began` to
/// `committed`) while the step before it runs, and its n-gram rows are
/// staged only once that step's draws were read (`parked_staging` to
/// `staged`, the release).
#[derive(Clone, Copy, Debug)]
pub struct RowsStepTiming {
    /// The step began (before any host input was written; for a parked
    /// step, before it was encoded).
    pub began: f64,
    /// The rows' token ids and n-gram inputs were staged; for a parked step,
    /// its n-gram rows were staged and it was released.
    pub staged: f64,
    /// The pass was encoded.
    pub encoded: f64,
    /// The pass was committed.
    pub committed: f64,
    pub gpu: PassTiming,
    /// The host woke from the wait for the pass.
    pub woke: f64,
    /// The draws were read back: the step is over.
    pub ended: f64,
    /// A parked step only: the host began staging its n-gram rows (after
    /// reading the previous step's draws). `None` for an unparked step.
    pub parked_staging: Option<f64>,
}

/// [`RowsStepTiming`] as durations in milliseconds. For an unparked step
/// each phase is one interval of the step, so they add up to its wall time,
/// except that the submission gap and the wake-up are measured against the
/// GPU's clock marks and come out slightly negative when the GPU started
/// before `commit` returned. For a parked step the encoding and the commit
/// overlap the step before it, so the phases do not add up to anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowsStepPhases {
    /// Writing the ids and staging the n-gram rows (paged: copied by the
    /// host out of the page cache); for a parked step, staging its n-gram
    /// rows and releasing it.
    pub stage_ms: f64,
    /// Encoding the pass on the host (serial: nothing runs on the GPU
    /// unless a step is in flight ahead of it).
    pub encode_ms: f64,
    pub commit_ms: f64,
    /// From `commit` returning to the GPU starting the pass. For a parked
    /// step, from its release to the GPU starting it: negative, since the
    /// pass starts once the step ahead of it ends and runs up to its wait
    /// while the host reads that step's draws and stages.
    pub submit_ms: f64,
    /// The pass's GPU span; for a parked step it includes any time it
    /// waited for its release.
    pub gpu_ms: f64,
    /// From the GPU's end to the host waking.
    pub wake_ms: f64,
    /// Reading the draws back.
    pub finish_ms: f64,
}

impl RowsStepTiming {
    pub fn phases(&self) -> RowsStepPhases {
        let ms = |from: f64, to: f64| (to - from) * 1e3;
        let (stage_ms, encode_ms, submit_ms) = match self.parked_staging {
            None => (
                ms(self.began, self.staged),
                ms(self.staged, self.encoded),
                ms(self.committed, self.gpu.gpu_start_secs),
            ),
            Some(staging) => (
                ms(staging, self.staged),
                ms(self.began, self.encoded),
                ms(self.staged, self.gpu.gpu_start_secs),
            ),
        };
        RowsStepPhases {
            stage_ms,
            encode_ms,
            commit_ms: ms(self.encoded, self.committed),
            submit_ms,
            gpu_ms: ms(self.gpu.gpu_start_secs, self.gpu.gpu_end_secs),
            wake_ms: ms(self.gpu.gpu_end_secs, self.woke),
            finish_ms: ms(self.woke, self.ended),
        }
    }

    /// The step's wall time in milliseconds, `began` to `ended` (for a
    /// parked step this overlaps the step before it).
    pub fn wall_ms(&self) -> f64 {
        (self.ended - self.began) * 1e3
    }
}

/// A batched decode step committed and not yet waited for
/// ([`LanguageModel::commit_rows`], [`LanguageModel::park_rows`]); hand it
/// to [`LanguageModel::finish_rows`]. A parked step must be released
/// ([`LanguageModel::release_rows`]) first.
///
/// Dropping one waits for its pass. A parked step that is dropped
/// unreleased (an error path) is released first, without its inputs, so the
/// queue never waits for it forever; the model's own bookkeeping of the
/// parked step is then cleared by [`LanguageModel::release_parked`].
pub struct RowsInFlight<'a> {
    /// The pass, committed; taken by [`LanguageModel::finish_rows`]. Fields
    /// drop after `Drop::drop` ran, so a parked pass is released before its
    /// own drop waits for it.
    pub(crate) pass: Option<PendingPass<'a>>,
    /// The draw buffer the step writes (models ping-pong between two so the
    /// next step can be committed before this one's draws are read).
    pub(crate) out: usize,
    /// The rows, in order: what the next step over them is checked against.
    pub(crate) rows: Vec<RowKey>,
    /// The event and value a parked step waits for; `None` once released
    /// (or for an unparked step).
    pub(crate) park: Option<(SharedEvent, u64)>,
    /// The host marks so far when the step is timed.
    pub(crate) marks: Option<RowsStepTiming>,
}

/// Which row a [`RowsInFlight`] step ran: the state (by address), its
/// position once the step's token is fed, and its batch slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowKey {
    pub(crate) state: usize,
    pub(crate) pos: usize,
    pub(crate) slot: usize,
}

impl RowKey {
    /// `row`'s key once its state has advanced past the step's token.
    pub(crate) fn of<S: DecodeStateApi>(row: &BatchRow<'_, S>) -> Self {
        Self {
            state: std::ptr::from_ref::<S>(&*row.state) as usize,
            pos: row.state.pos(),
            slot: row.slot,
        }
    }
}

impl RowsInFlight<'_> {
    /// Rows the step runs.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Whether the step still waits for its release.
    pub fn is_parked(&self) -> bool {
        self.park.is_some()
    }

    /// Checks that `rows` are the step's rows, in order, with their states
    /// where the step left them.
    pub(crate) fn check_rows<S: DecodeStateApi>(
        &self,
        rows: &[BatchRow<'_, S>],
    ) -> Result<()> {
        anyhow::ensure!(
            rows.len() == self.rows.len(),
            "{} rows for a step of {}",
            rows.len(),
            self.rows.len()
        );
        for (r, (row, key)) in rows.iter().zip(&self.rows).enumerate() {
            anyhow::ensure!(
                RowKey::of(row) == *key,
                "row {r} is not the step's row {key:?}: {:?}",
                RowKey::of(row)
            );
        }
        Ok(())
    }
}

impl Drop for RowsInFlight<'_> {
    fn drop(&mut self) {
        if let Some((event, value)) = self.park.take() {
            event.signal(value);
        }
    }
}

/// One session's row of a batched decode step
/// ([`LanguageModel::decode_rows`]): its state, the token the step feeds
/// at the state's position (drawn by the row's previous step, not yet fed)
/// and the draw the step ends with.
pub struct BatchRow<'r, S> {
    pub state: &'r mut S,
    pub token: u32,
    pub draw: Draw<'r>,
    /// The batch slot whose sampler scratch holds this row's penalty counts
    /// (`0..max_batch_rows`), see [`CountsSlot`].
    pub slot: usize,
}

/// Where a request's penalty counts live: the engine's own sampler (the
/// single-session paths: prefill draws, the decode loop, speculation) or
/// the sampler of a batch slot while the request decodes in a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CountsSlot {
    Engine,
    Batch(usize),
}

pub trait LanguageModel: Sized {
    /// The expert cache's counters, when the model serves its experts from
    /// one (`LoadOptions::expert_slots`).
    fn expert_cache_stats(&self) -> Option<crate::qwen4exp::ExpertCacheStats> {
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

    /// The attention K/V caches' element format the model runs with.
    fn kv_format(&self) -> crate::kernels::attention::KvFormat {
        crate::kernels::attention::KvFormat::Bf16
    }

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

    /// Warms weights served from disk (the paged n-gram table) before
    /// returning, pinning them in memory when `lock` is set, and returns the
    /// bytes found resident; models without such weights return 0. The
    /// server preloads in the background instead ([`Self::paged_table`]);
    /// the bench wants the table warm before it measures.
    fn warm_storage(&self, lock: bool) -> Result<u64> {
        let _ = lock;
        Ok(0)
    }

    /// The table served from disk, for a preload that runs while the engine
    /// serves ([`crate::qwen4exp::ngram::PagedTable::preload_in_background`]);
    /// `None` for models without one.
    fn paged_table(
        &self,
    ) -> Option<std::sync::Arc<crate::qwen4exp::ngram::PagedTable>> {
        None
    }

    /// Bytes of weights served from the page cache instead of GPU memory
    /// (the paged n-gram table). They are not in the device's allocated
    /// size, yet they want to stay resident and so compete with the session
    /// cache for physical memory; 0 for models without such weights.
    fn paged_storage_bytes(&self) -> usize {
        0
    }

    /// The GPU buffers that hold the model's weights (the vision tower's and
    /// the draft head's included), without an expert cache's slab and slot
    /// tables, weights served from the page cache, or anything allocated
    /// after the load: what the server's `--pin-weights` locks in memory.
    /// Empty for models that do not track them, which are then never pinned.
    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        Vec::new()
    }

    /// The machine memory the load planned for (`LILY_MEMORY_GB`, else the
    /// physical memory); `None` when unknown.
    fn planned_memory(&self) -> Option<u64> {
        None
    }

    /// The footprint the load kept to under `LoadOptions::memory_limit`,
    /// never more than the machine's memory less the plan's reserve; `None`
    /// without a limit.
    fn memory_limit(&self) -> Option<u64> {
        None
    }

    /// What the expert cache's plan kept free for the session cache (one
    /// full session, `LoadOptions::session_context`); `None` when no plan
    /// reserved anything.
    fn session_reserve(&self) -> Option<u64> {
        None
    }

    /// GPU bytes one session holds with caches for `capacity` tokens and
    /// `checkpoints` recurrent snapshots, from the model's shapes; `None`
    /// for architectures that do not compute it.
    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        let _ = (capacity, checkpoints);
        None
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

    /// [`Self::prefill_with_vision`] without a draw (it fills a cache
    /// prefix), stopping at the first chunk boundary at which `stop` returns
    /// true, before that chunk is committed. Returns how many of `tokens`
    /// were fed. A stopped state sits at a chunk boundary exactly as if the
    /// prefix fed so far had been the whole call, so it can be kept and
    /// extended later. Architectures that do not implement it never stop
    /// early.
    fn prefill_until(
        &self,
        ctx: &MetalContext,
        state: &mut Self::State,
        scratch: &mut Self::Scratch,
        tokens: &[u32],
        vision: Option<&crate::qwen4exp::VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        let _ = stop;
        self.prefill_with_vision(ctx, state, scratch, tokens, None, vision)?;
        Ok(tokens.len())
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

    // --- batched decode across sessions (`--max-batch`) -------------------

    /// Rows [`Self::decode_rows`] takes at once; `0` when the model cannot
    /// batch sessions (the server then serves one request at a time).
    fn max_batch_rows(&self) -> usize {
        0
    }

    /// One decode step over independent sessions: row `r` feeds `token` at
    /// its own state's position and draws with its own sampler settings
    /// into slot `slot`'s sampler. Stages every row's per-token host inputs,
    /// commits, waits, and advances every state by one. Returns the draws in
    /// row order. Every state must be at rest (no pending speculative step,
    /// nothing parked) with room for one more token. With a draft head
    /// loaded the head is caught up on every row, so a row can return to
    /// speculative decoding afterwards. [`Self::commit_rows`] followed by
    /// [`Self::finish_rows`].
    fn decode_rows(
        &self,
        ctx: &MetalContext,
        scratch: &mut Self::Scratch,
        rows: &mut [BatchRow<'_, Self::State>],
    ) -> Result<Vec<u32>> {
        let step = self.commit_rows(ctx, scratch, rows, false)?;
        Ok(self.finish_rows(scratch, step)?.0)
    }

    /// [`Self::decode_rows`], also reporting where the step's wall time went
    /// when the model can measure it (`None` otherwise). It keeps the
    /// completed pass to read its GPU span, which the server's path does not
    /// pay for; a benchmark diagnostic only.
    fn decode_rows_timed(
        &self,
        ctx: &MetalContext,
        scratch: &mut Self::Scratch,
        rows: &mut [BatchRow<'_, Self::State>],
    ) -> Result<(Vec<u32>, Option<RowsStepTiming>)> {
        let step = self.commit_rows(ctx, scratch, rows, true)?;
        self.finish_rows(scratch, step)
    }

    /// The first half of [`Self::decode_rows`]: stages the rows' host
    /// inputs, encodes and commits the step, and advances every state by one
    /// (the step's token counts as fed once committed), without waiting.
    /// Nothing may be in flight; every state must be at rest with room for
    /// one more token. With `timed`, [`Self::finish_rows`] reports the
    /// step's [`RowsStepTiming`].
    fn commit_rows<'a>(
        &self,
        ctx: &'a MetalContext,
        scratch: &mut Self::Scratch,
        rows: &mut [BatchRow<'_, Self::State>],
        timed: bool,
    ) -> Result<RowsInFlight<'a>> {
        let _ = (ctx, scratch, rows, timed);
        anyhow::bail!("this model cannot batch decode steps across sessions")
    }

    /// Waits for a step [`Self::commit_rows`] or [`Self::park_rows`]
    /// committed (a parked one must have been released) and returns its
    /// draws in row order, with its timing when it was committed `timed`.
    fn finish_rows(
        &self,
        scratch: &Self::Scratch,
        step: RowsInFlight<'_>,
    ) -> Result<(Vec<u32>, Option<RowsStepTiming>)> {
        let _ = (scratch, step);
        anyhow::bail!("this model cannot batch decode steps across sessions")
    }

    /// Whether [`Self::park_rows`] is available.
    fn supports_rows_parking(&self) -> bool {
        false
    }

    /// Encodes and commits the batched step that follows `after` over the
    /// same rows, before `after`'s draws are known: the batched counterpart
    /// of [`Self::encode_parked_step`]. `rows` are `after`'s rows in the
    /// same order and slots, their states where `after` left them (advanced
    /// past its token); each row's `draw` is this step's, and its `token`
    /// is not read: the step feeds `after`'s draws, read on the GPU. The
    /// pass parks where it first reads host-staged inputs until
    /// [`Self::release_rows`], which needs `after`'s draws. Every state
    /// advances by one; each must have room for this step's token, and the
    /// caller must not grow a cache, take a checkpoint or run anything else
    /// on the GPU, the scratch or the states until the step finished. One
    /// parked step at a time (batched or not).
    fn park_rows<'a>(
        &self,
        ctx: &'a MetalContext,
        scratch: &Self::Scratch,
        rows: &mut [BatchRow<'_, Self::State>],
        after: &RowsInFlight<'a>,
        timed: bool,
    ) -> Result<RowsInFlight<'a>> {
        let _ = (ctx, scratch, rows, after, timed);
        anyhow::bail!("this model cannot park batched decode steps")
    }

    /// Stages a parked step's per-token host inputs and lets it run. `rows`
    /// are its rows, each with `token` set to what the step feeds: the draw
    /// of the step before it, which must have finished.
    fn release_rows(
        &self,
        scratch: &Self::Scratch,
        step: &mut RowsInFlight<'_>,
        rows: &mut [BatchRow<'_, Self::State>],
    ) -> Result<()> {
        let _ = (scratch, step, rows);
        anyhow::bail!("this model cannot park batched decode steps")
    }

    /// Copies a request's penalty counts from one sampler to another (a
    /// request entering or leaving a batch). The GPU must be idle on both.
    fn move_sampler_counts(
        &self,
        ctx: &MetalContext,
        scratch: &mut Self::Scratch,
        from: CountsSlot,
        to: CountsSlot,
    ) -> Result<()> {
        let _ = (ctx, scratch, from, to);
        anyhow::bail!("this model cannot batch decode steps across sessions")
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
    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;
