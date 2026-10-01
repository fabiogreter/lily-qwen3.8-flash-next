//! The session cache: token lineages with resumable positions under a byte
//! budget.
//!
//! A session owns one decode state (per-token caches valid up to `pos`) and a
//! few checkpoints of the recurrent part at earlier positions. A new prompt
//! resumes from the largest position that is both a checkpoint (or the live
//! end) and a common prefix of the session's tokens and the prompt, capped at
//! `prompt_len - 1` so at least one token is fed to produce logits.
//!
//! Extending the live end reuses the session in place. Anything else forks:
//! the per-token caches up to the resume position are copied into a fresh
//! state and the checkpoint restored, so the original lineage survives (a
//! parallel conversation sharing only the system prompt must not destroy a
//! long context). Sessions are evicted least-recently-used under the budget.
//!
//! Below the GPU tier sits an optional disk tier ([`DiskStore`]): sessions
//! evicted from GPU memory are written out (per-token caches plus their
//! checkpoints and a snapshot of the live end) and a later prompt that shares
//! a prefix reads them back instead of recomputing it. A disk hit at the live
//! end moves the session back to the GPU (the file copy is dropped); a hit at
//! an earlier checkpoint forks from the file and leaves it in place.
//!
//! The disk tier also holds **durable prefix entries**: a prefix that two
//! prompts shared (the `agreement`) but that no lineage could resume from,
//! because checkpoints sit at the end of served prompts and two agent runs
//! diverge before that. The engine materialises such a boundary once, as a
//! disk entry whose live end is the boundary, and every later prompt with the
//! same preamble resumes there. These entries live only on disk, never as
//! resident sessions or extra checkpoints: they are hit far more rarely than
//! the live conversation's cache and must not compete with it for the GPU
//! budget. A hit never consumes them.
//!
//! **Writing ahead.** An eviction that has to write its session first puts
//! that write on the request that needed the room, 0.5 to 1 s of time to
//! first token. So between requests, when the next evictions would need
//! writes, the store copies the least recently used sessions to the disk tier
//! on a background thread while they stay resident ([`SessionStore::write_ahead`]).
//! A session being written is parked: out of reach of every lookup, counted
//! in the budget, its buffers untouched, and its memory released only by a
//! later eviction, never by the write. Evicting a session whose copy is on
//! disk is then only a drop. A prompt that resumes a parked session cancels
//! the write and takes the session back (a partly written entry is never
//! indexed); an eviction that reaches it waits for the write to finish; one
//! that reaches a session without a copy writes it synchronously, as before.
//!
//! **Images** (`docs/architecture.md`, "The session cache"). An image is a run of
//! identical `<|image_pad|>` tokens, so two prompts with different
//! screenshots have identical tokens there. Tokens stay the identity, and
//! every lineage (a session, a disk entry, a durable entry) also carries its
//! image spans, each with the position, the length, the patch grid and a
//! digest of the preprocessed pixel rows. Wherever two lineages are matched
//! ([`shared_prefix`]), the token prefix they share is cut back to the start
//! of the first span one has that the other does not match exactly. A
//! text-only lineage has no spans and matches exactly as before.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};

use super::disk::{self, DiskStore, NewEntry, Reserved};
use crate::engine::{
    DecodeStateApi, LanguageModel, Segment, SnapshotApi, write_layout,
};
use crate::metal::MetalContext;
use crate::qwen4exp::ImageSpan;

type SnapshotOf<M> = <<M as LanguageModel>::State as DecodeStateApi>::Snapshot;

/// Sessions shorter than this are not worth a round trip to disk.
const MIN_DISK_TOKENS: usize = 256;

/// Tokens of the best-agreeing lineage returned past the agreement, enough
/// for the divergence diagnostic to show a line of text.
const DIVERGENT_TAIL_TOKENS: usize = 16;

/// New sessions whose size sets the write-ahead headroom (the largest of
/// them): enough to follow what the client does now, few enough that one
/// odd request stops mattering soon.
const RECENT_NEW_SESSIONS: usize = 8;

/// One image of a lineage, as the caches identify it: where its
/// placeholders sit, what the tower was given (the grid) and what the pixel
/// rows were (a SHA-256 over the preprocessed rows and the grid). Two
/// lineages share an image only when all of it is equal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedImage {
    /// Sequence index of the first `<|image_pad|>`.
    pub start: usize,
    /// Placeholders, `grid_h * grid_w / 4`.
    pub len: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// SHA-256 of the preprocessed f32 pixel rows followed by the grid.
    pub digest: [u8; 32],
}

impl CachedImage {
    pub fn new(span: ImageSpan, digest: [u8; 32]) -> Self {
        Self {
            start: span.start,
            len: span.len,
            grid_h: span.grid_h,
            grid_w: span.grid_w,
            digest,
        }
    }

    /// Sequence index one past the last placeholder.
    pub fn end(&self) -> usize {
        self.start + self.len
    }

    pub fn span(&self) -> ImageSpan {
        ImageSpan {
            start: self.start,
            len: self.len,
            grid_h: self.grid_h,
            grid_w: self.grid_w,
        }
    }
}

/// The spans of `images` that lie entirely within the first `n` tokens.
pub fn images_within(images: &[CachedImage], n: usize) -> Vec<CachedImage> {
    images.iter().filter(|i| i.end() <= n).cloned().collect()
}

/// How many leading tokens two lineages share, images included: the token
/// prefix they have in common, cut back to the start of the first span in
/// either that the other does not match exactly (same start, length, grid
/// and digest), so a placeholder run is only ever shared between requests
/// carrying the same preprocessed image. Text-only lineages share their
/// token prefix, as before images existed.
pub fn shared_prefix(
    tokens_a: &[u32],
    images_a: &[CachedImage],
    tokens_b: &[u32],
    images_b: &[CachedImage],
) -> usize {
    let mut shared = common_prefix_len(tokens_a, tokens_b);
    for (mine, theirs) in [(images_a, images_b), (images_b, images_a)] {
        for image in mine {
            if image.start < shared && !theirs.contains(image) {
                shared = image.start;
            }
        }
    }
    shared
}

/// How far `prompt` (with `images`) agrees with the closest of `lineages`:
/// the longest [`shared_prefix`] with any of them, capped at
/// `prompt.len() - 1` like every resume position (the last prompt token is
/// always fed). Also returns up to [`DIVERGENT_TAIL_TOKENS`] of that
/// lineage's tokens from the agreement on, which is what the prompt would
/// have had to continue with to keep matching; empty when nothing agrees.
pub fn agreement<'a>(
    prompt: &[u32],
    images: &[CachedImage],
    lineages: impl IntoIterator<Item = (&'a [u32], &'a [CachedImage])>,
) -> (usize, Vec<u32>) {
    let cap = prompt.len().saturating_sub(1);
    let mut best: Option<(usize, &[u32])> = None;
    for (tokens, lineage_images) in lineages {
        let lcp = shared_prefix(tokens, lineage_images, prompt, images).min(cap);
        if lcp > 0 && best.is_none_or(|(b, _)| lcp > b) {
            best = Some((lcp, tokens));
        }
    }
    match best {
        Some((lcp, tokens)) => {
            (lcp, tokens[lcp..tokens.len().min(lcp + DIVERGENT_TAIL_TOKENS)].to_vec())
        }
        None => (0, Vec::new()),
    }
}

/// Where a durable prefix entry should be materialised for a prompt that
/// agreed with a cached lineage for `agreement` tokens but resumed at
/// `reused`: the agreement itself when it reaches at least `min_tokens`
/// beyond where the run resumed (0 turns the feature off) and leaves at least
/// one token to feed. Two real prompts shared that prefix, so a third is
/// likely. Measuring from the resume position keeps ordinary forks inside one
/// conversation out of the disk tier: a turn that resumes at the previous
/// request's end and diverges a few hundred tokens later (a regenerated
/// answer, a re-tokenized message) has nothing a later run would reuse.
pub fn boundary_position(
    agreement: usize,
    reused: usize,
    prompt_len: usize,
    min_tokens: usize,
) -> Option<usize> {
    let worth_it = min_tokens > 0 && agreement.saturating_sub(reused) >= min_tokens;
    (worth_it && agreement <= prompt_len.checked_sub(1)?).then_some(agreement)
}

/// The best position to resume `prompt` from given a lineage's tokens,
/// images and resumable positions: the live end when the lineage is a
/// strict prefix of the prompt (and `live_end` is resumable), else the
/// latest resumable position within the [`shared_prefix`]. Never
/// `>= prompt.len()`.
fn resume_position(
    tokens: &[u32],
    images: &[CachedImage],
    checkpoints: &[usize],
    live_end: bool,
    prompt: &[u32],
    prompt_images: &[CachedImage],
) -> Option<usize> {
    let lcp = shared_prefix(tokens, images, prompt, prompt_images);
    let limit = lcp.min(prompt.len().checked_sub(1)?);
    if tokens.len() <= limit {
        if live_end {
            return (!tokens.is_empty()).then_some(tokens.len());
        }
        return checkpoints.iter().rev().copied().find(|&p| p > 0 && p <= limit);
    }
    checkpoints.iter().rev().copied().find(|&p| p > 0 && p <= limit)
}

pub struct Session<M: LanguageModel> {
    /// Tokens fed into `state`; `state.pos() == tokens.len()` when at rest.
    pub tokens: Vec<u32>,
    /// The images among `tokens`, in order; every span ends within them.
    pub images: Vec<CachedImage>,
    pub state: M::State,
    /// Recurrent-state checkpoints, ascending by position, all `<= tokens.len()`.
    checkpoints: Vec<Arc<SnapshotOf<M>>>,
    cache_key: Option<String>,
    last_used: u64,
    /// What the disk tier holds of this session as it is now.
    disk: DiskCopy,
    /// Created by the request that holds it (not an extension of a resident
    /// session), so its size at release counts toward the headroom.
    born: bool,
}

/// Whether a resident session's current contents are on disk already.
#[derive(Clone, Debug, PartialEq, Eq)]
enum DiskCopy {
    /// No: evicting it means writing it.
    None,
    /// Written ahead as this disk entry, which the tier may have deleted
    /// since (budget, age): checked before it is relied on.
    Written(String),
    /// The tier refused it (too big for the budget, the volume short of
    /// space): evicting it drops it rather than writing it again.
    Refused,
}

impl<M: LanguageModel> Session<M> {
    fn new(state: M::State) -> Self {
        Self {
            tokens: Vec::new(),
            images: Vec::new(),
            state,
            checkpoints: Vec::new(),
            cache_key: None,
            last_used: 0,
            disk: DiskCopy::None,
            born: true,
        }
    }

    /// Records a checkpoint of the state's current position. Duplicates by
    /// position are dropped.
    pub fn add_checkpoint(&mut self, snapshot: SnapshotOf<M>) {
        let pos = snapshot.pos();
        if self.checkpoints.iter().any(|c| c.pos() == pos) {
            return;
        }
        let at = self.checkpoints.partition_point(|c| c.pos() < pos);
        self.checkpoints.insert(at, Arc::new(snapshot));
    }

    /// The lineage after a prefill that stopped early: the request resumed
    /// at `reused` and fed `prompt[reused..at]`, which the state holds, so
    /// the session is the prompt's first `at` tokens and its live end is
    /// resumable there like any other. Nothing else changes: its
    /// checkpoints are the ones it was acquired with, all at or below
    /// `reused`, since a request adds its own only after the prefill.
    pub fn stop_at(&mut self, prompt: &[u32], reused: usize, at: usize) -> Result<()> {
        ensure!(
            reused <= at && at <= prompt.len() && self.state.pos() == at,
            "a prefill stopped at {at} (resumed at {reused}, {} prompt tokens) left the state at {}",
            prompt.len(),
            self.state.pos()
        );
        self.tokens.truncate(reused);
        self.tokens.extend_from_slice(&prompt[reused..at]);
        Ok(())
    }

    /// GPU bytes the session holds.
    pub fn bytes(&self) -> usize {
        self.state.bytes() + self.checkpoints.iter().map(|c| c.bytes()).sum::<usize>()
    }

    /// The best position to resume `prompt` from: the live end when the
    /// session is a strict prefix of the prompt, else the latest checkpoint
    /// within the shared prefix. Never `>= prompt.len()`.
    fn resume_position(&self, prompt: &[u32], images: &[CachedImage]) -> Option<usize> {
        let positions: Vec<usize> = self.checkpoints.iter().map(|c| c.pos()).collect();
        resume_position(&self.tokens, &self.images, &positions, true, prompt, images)
    }

    fn checkpoint_at(&self, pos: usize) -> Option<&Arc<SnapshotOf<M>>> {
        self.checkpoints.iter().find(|c| c.pos() == pos)
    }
}

/// What [`SessionStore::acquire`] hands out.
pub struct Acquired<M: LanguageModel> {
    pub session: Session<M>,
    /// Tokens of the prompt already fed into `session.state`.
    pub reused: usize,
    /// Whether the session was forked from a cached lineage (diagnostics).
    pub forked: bool,
    /// Whether the reused prefix was read from the disk tier, and how long
    /// that took.
    pub from_disk: Option<std::time::Duration>,
    /// The longest common prefix between the prompt and any lineage the store
    /// knows (resident or on disk), capped at `prompt.len() - 1`. Always
    /// `>= reused`; the gap is a prefix that was shared but not resumable,
    /// which is what a durable prefix entry fixes (see [`boundary_position`]).
    pub agreement: usize,
    /// The best-agreeing lineage's tokens from `agreement` on (at most
    /// [`DIVERGENT_TAIL_TOKENS`]), for the divergence diagnostic. Empty when
    /// nothing agreed.
    pub divergent_tail: Vec<u32>,
    /// What making room for the session cost (part of the session phase).
    pub evictions: Evictions,
}

/// What evicting sessions cost one acquire or release: how many went, how
/// many of those were on disk already (a drop), how many had to be written
/// on the spot and how long that took, how long an eviction waited for a
/// write ahead to finish, and how many writes ahead were cancelled because
/// the prompt resumed the session being written.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Evictions {
    pub evicted: usize,
    /// Of `evicted`, the ones written ahead: dropped without I/O.
    pub written_ahead: usize,
    /// Of `evicted`, the ones written synchronously (the fallback).
    pub spilled: usize,
    pub spill_secs: f64,
    /// Waiting for a write ahead that an eviction reached before it finished.
    pub waited_secs: f64,
    pub cancelled: usize,
}

/// A session copied to the disk tier on a background thread. The session is
/// parked here for the write: no lookup sees it and nothing encodes work on
/// its buffers, so the ranges the writer reads stay valid and unchanged.
/// Dropping the handle cancels the write and joins the thread before the
/// session (and its buffers) goes.
struct Writing<M: LanguageModel> {
    /// Taken out only after the thread was joined.
    session: Option<Session<M>>,
    reserved: Option<Reserved>,
    positions: Vec<usize>,
    cancel: Arc<AtomicBool>,
    thread: Option<JoinHandle<Result<u64>>>,
    started: Instant,
}

impl<M: LanguageModel> Writing<M> {
    fn finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Waits for the thread and returns what it wrote.
    fn join(&mut self) -> Result<u64> {
        match self.thread.take() {
            Some(thread) => thread
                .join()
                .unwrap_or_else(|_| Err(anyhow::anyhow!("writer panicked"))),
            None => Err(anyhow::anyhow!("already joined")),
        }
    }
}

impl<M: LanguageModel> Drop for Writing<M> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if let Some(reserved) = self.reserved.take() {
            let _ = std::fs::remove_dir_all(reserved.path());
        }
    }
}

/// A session's disk entry as a layout: the per-token caches, then one file
/// per checkpoint position, the live end's from the state's own recurrent
/// buffers. Every host range points into the session's buffers.
struct Plan {
    positions: Vec<usize>,
    prefix: Vec<Segment>,
    checkpoints: Vec<(usize, Vec<Segment>)>,
}

impl Plan {
    fn of<M: LanguageModel>(session: &Session<M>) -> Result<Self> {
        let n = session.tokens.len();
        let mut checkpoints = Vec::new();
        for c in &session.checkpoints {
            let pos = c.pos();
            if pos > 0 && pos < n {
                checkpoints.push((pos, c.layout()?));
            }
        }
        checkpoints.push((n, session.state.live_layout()?));
        Ok(Self {
            positions: checkpoints.iter().map(|(p, _)| *p).collect(),
            prefix: session.state.prefix_layout(n)?,
            checkpoints,
        })
    }

    /// Writes the files into `path`.
    ///
    /// # Safety
    ///
    /// [`write_layout`]'s: the session the plan was made of must stay alive
    /// and untouched until this returns.
    unsafe fn write(
        &self,
        path: &std::path::Path,
        cancel: Option<&AtomicBool>,
    ) -> Result<u64> {
        disk::write_files(
            path,
            &self.positions,
            // SAFETY: the caller's contract.
            &mut |w| unsafe { write_layout(&self.prefix, w, cancel) },
            &mut |pos, w| {
                let (_, layout) = self
                    .checkpoints
                    .iter()
                    .find(|(p, _)| *p == pos)
                    .context("checkpoint vanished")?;
                // SAFETY: the caller's contract.
                unsafe { write_layout(layout, w, cancel) }
            },
        )
    }
}

/// Puts the calling thread at utility QoS with utility disk I/O: the write
/// ahead yields the CPU and the SSD to the request that is running (its
/// n-gram page reads above all) and to interactive work, yet finishes in
/// about the time it would take in the foreground on an idle machine.
/// Best effort.
fn lower_writer_priority() {
    const QOS_CLASS_UTILITY: u32 = 0x11;
    const IOPOL_TYPE_DISK: std::ffi::c_int = 0;
    const IOPOL_SCOPE_THREAD: std::ffi::c_int = 1;
    const IOPOL_UTILITY: std::ffi::c_int = 4;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(
            qos: u32,
            relative: std::ffi::c_int,
        ) -> std::ffi::c_int;
        fn setiopolicy_np(
            iotype: std::ffi::c_int,
            scope: std::ffi::c_int,
            policy: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
    // SAFETY: both calls only change the calling thread's own policy.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_UTILITY);
    }
}

impl<M: LanguageModel> Acquired<M> {
    /// A session nothing was reused for; the other fields are filled in by
    /// the path that built it.
    fn fresh(session: Session<M>) -> Self {
        Self {
            session,
            reused: 0,
            forked: false,
            from_disk: None,
            agreement: 0,
            divergent_tail: Vec::new(),
            evictions: Evictions::default(),
        }
    }
}

pub struct SessionStore<M: LanguageModel> {
    entries: Vec<Session<M>>,
    /// The session being written ahead, parked (see [`Writing`]).
    writing: Option<Writing<M>>,
    budget_bytes: usize,
    max_sessions: usize,
    max_checkpoints: usize,
    clock: u64,
    disk: Option<DiskStore>,
    /// Shortest shared prefix worth a durable disk entry (0: never).
    durable_min_tokens: usize,
    /// The least room the write ahead keeps free of writes (`None`: off).
    write_ahead_floor: Option<usize>,
    /// Bytes of the most recent new sessions at their release.
    recent_new: VecDeque<usize>,
    /// Evictions of the acquire or release in progress.
    evictions: Evictions,
    /// Tests: while set, a writer thread waits before writing (cancellable).
    #[cfg(test)]
    write_gate: Option<Arc<AtomicBool>>,
}

impl<M: LanguageModel> SessionStore<M> {
    pub fn new(
        budget_bytes: usize,
        max_sessions: usize,
        max_checkpoints: usize,
    ) -> Self {
        Self {
            entries: Vec::new(),
            writing: None,
            budget_bytes,
            max_sessions,
            max_checkpoints: max_checkpoints.max(1),
            clock: 0,
            disk: None,
            durable_min_tokens: 0,
            write_ahead_floor: None,
            recent_new: VecDeque::with_capacity(RECENT_NEW_SESSIONS),
            evictions: Evictions::default(),
            #[cfg(test)]
            write_gate: None,
        }
    }

    /// Attaches the disk tier.
    pub fn with_disk(mut self, disk: DiskStore) -> Self {
        self.disk = Some(disk);
        self
    }

    /// Sets the shortest shared prefix worth a durable disk entry (0 turns
    /// durable entries off). Only meaningful with a disk tier attached.
    pub fn with_durable_min_tokens(mut self, min_tokens: usize) -> Self {
        self.durable_min_tokens = min_tokens;
        self
    }

    /// Turns the write ahead on ([`Self::write_ahead`]): between requests
    /// the store keeps the evictions that the next new session would cause
    /// free of writes, for a new session as large as the largest of the last
    /// few, and at least `floor_bytes`. Only meaningful with a disk tier.
    pub fn with_write_ahead(mut self, floor_bytes: usize) -> Self {
        self.write_ahead_floor = Some(floor_bytes);
        self
    }

    /// Whether the store writes sessions ahead.
    pub fn writes_ahead(&self) -> bool {
        self.write_ahead_floor.is_some() && self.disk.is_some()
    }

    /// Whether a write ahead is in flight.
    pub fn writing(&self) -> bool {
        self.writing.is_some()
    }

    pub fn durable_min_tokens(&self) -> usize {
        self.durable_min_tokens
    }

    pub fn disk(&self) -> Option<&DiskStore> {
        self.disk.as_ref()
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// GPU bytes of every session held, the one being written ahead included.
    pub fn used_bytes(&self) -> usize {
        self.entries.iter().chain(self.parked()).map(Session::bytes).sum()
    }

    /// Sessions held, the one being written ahead included.
    pub fn len(&self) -> usize {
        self.entries.len() + usize::from(self.writing.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn parked(&self) -> Option<&Session<M>> {
        self.writing.as_ref().and_then(|w| w.session.as_ref())
    }

    /// Checks out the session that can resume `prompt` (with `images` at
    /// their spans) from the furthest position, forking when that position
    /// is not the live end. `cache_key` only breaks ties between equally
    /// good candidates. Also reports how far the prompt agreed with any
    /// lineage at all (`agreement`), which the engine compares with `reused`
    /// to decide on a durable prefix entry, and what making room cost.
    pub fn acquire(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
    ) -> Result<Acquired<M>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        self.evictions = Evictions::default();
        self.collect_finished();
        self.reclaim_for(prompt, images);
        if let Some(disk) = self.disk.as_mut() {
            disk.expire();
        }
        let (agreement, divergent_tail) = agreement(
            prompt,
            images,
            self.entries
                .iter()
                .chain(self.parked())
                .map(|s| (s.tokens.as_slice(), s.images.as_slice()))
                .chain(self.disk.iter().flat_map(|d| {
                    d.entries()
                        .iter()
                        .map(|e| (e.tokens.as_slice(), e.images.as_slice()))
                })),
        );
        let mut acquired =
            self.acquire_resumable(ctx, model, prompt, images, cache_key)?;
        debug_assert!(acquired.reused <= agreement, "resumed past the agreement");
        acquired.agreement = agreement;
        acquired.divergent_tail = divergent_tail;
        acquired.evictions = std::mem::take(&mut self.evictions);
        Ok(acquired)
    }

    /// [`Self::acquire`] without the agreement: picks the lineage and builds
    /// the session. `agreement` and `divergent_tail` are left at their
    /// defaults for the caller to fill in.
    fn acquire_resumable(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
    ) -> Result<Acquired<M>> {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.resume_position(prompt, images).map(|p| (i, p)))
            .max_by_key(|&(i, p)| {
                let entry = &self.entries[i];
                let key_match =
                    cache_key.is_some_and(|k| entry.cache_key.as_deref() == Some(k));
                (p, key_match, entry.last_used)
            });

        // The disk tier competes on resume position; ties go to the GPU.
        let best_disk = self.disk.as_ref().and_then(|disk| {
            disk.entries()
                .iter()
                .filter_map(|e| {
                    resume_position(
                        &e.tokens,
                        &e.images,
                        &e.checkpoints,
                        true,
                        prompt,
                        images,
                    )
                    .map(|p| (e.id.clone(), p, e.tokens.len()))
                })
                .max_by_key(|(id, p, _)| {
                    let key_match = cache_key.is_some_and(|k| {
                        disk.entries()
                            .iter()
                            .any(|e| &e.id == id && e.cache_key.as_deref() == Some(k))
                    });
                    (*p, key_match)
                })
        });
        if let Some((id, pos, len)) = best_disk
            && best.is_none_or(|(_, gpu_pos)| pos > gpu_pos)
        {
            return self.acquire_from_disk(ctx, model, prompt, &id, pos, len);
        }

        let Some((index, resume_at)) = best else {
            let state = model.new_state(ctx, prompt.len())?;
            let session = Session::new(state);
            self.trim(ctx, session.bytes());
            return Ok(Acquired::fresh(session));
        };

        if resume_at == self.entries[index].tokens.len() {
            // Pure extension of the live end: take the session as is.
            let session = self.entries.swap_remove(index);
            ensure!(
                session.state.pos() == resume_at,
                "cached decode state out of step with its tokens"
            );
            return Ok(Acquired { reused: resume_at, ..Acquired::fresh(session) });
        }

        // Fork: copy the per-token prefix, restore the recurrent checkpoint.
        // The new state briefly exceeds the budget until `trim` runs after
        // the copy (the source must survive until then).
        let mut state = model.new_state(ctx, prompt.len())?;
        let source = &self.entries[index];
        let checkpoint = source
            .checkpoint_at(resume_at)
            .ok_or_else(|| anyhow::anyhow!("resume position without checkpoint"))?
            .clone();
        state.copy_prefix_from(ctx, &source.state, resume_at)?;
        state.restore(ctx, &checkpoint)?;
        let mut session = Session::new(state);
        session.tokens = source.tokens[..resume_at].to_vec();
        session.images = images_within(&source.images, resume_at);
        session.checkpoints.push(checkpoint);
        self.trim(ctx, session.bytes());
        Ok(Acquired { reused: resume_at, forked: true, ..Acquired::fresh(session) })
    }

    /// Builds a session from disk entry `id` resumed at `pos`: the per-token
    /// caches up to `pos` and the checkpoint there. A live-end hit consumes
    /// the entry; a checkpoint hit forks from it and leaves it on disk. A
    /// durable prefix entry is forked from even at its live end: it exists to
    /// be hit again, and every hit is a lineage of its own.
    fn acquire_from_disk(
        &mut self,
        ctx: &MetalContext,
        model: &M,
        prompt: &[u32],
        id: &str,
        pos: usize,
        len: usize,
    ) -> Result<Acquired<M>> {
        let started = Instant::now();
        let disk = self.disk.as_mut().expect("disk tier");
        let entry = disk
            .entries()
            .iter()
            .find(|e| e.id == id)
            .context("disk entry vanished")?;
        let (tokens, durable) = (entry.tokens[..pos].to_vec(), entry.durable);
        let images = images_within(&entry.images, pos);
        let mut state = model.new_state(ctx, prompt.len().max(pos))?;
        let result = (|| -> Result<SnapshotOf<M>> {
            let mut prefix = disk.open_prefix(id)?;
            state.read_prefix(ctx, len, pos, &mut prefix)?;
            let mut ckpt = disk.open_checkpoint(id, pos)?;
            let snapshot = model.read_snapshot(ctx, &mut ckpt)?;
            ensure!(
                snapshot.pos() == pos,
                "checkpoint file at {pos} holds position {}",
                snapshot.pos()
            );
            state.restore(ctx, &snapshot)?;
            Ok(snapshot)
        })();
        let snapshot = match result {
            Ok(snapshot) => snapshot,
            Err(error) => {
                // A damaged entry must not poison every later request.
                eprintln!(
                    "session cache: dropping unreadable disk entry {id}: {error:#}"
                );
                disk.remove(id);
                let state = model.new_state(ctx, prompt.len())?;
                let session = Session::new(state);
                self.trim(ctx, session.bytes());
                return Ok(Acquired::fresh(session));
            }
        };
        let forked = pos != len || durable;
        if forked {
            disk.touch(id);
        } else {
            disk.remove(id);
        }
        let mut session = Session::new(state);
        session.tokens = tokens;
        session.images = images;
        session.checkpoints.push(Arc::new(snapshot));
        self.trim(ctx, session.bytes());
        Ok(Acquired {
            reused: pos,
            forked,
            from_disk: Some(started.elapsed()),
            ..Acquired::fresh(session)
        })
    }

    /// Writes a durable prefix entry for `tokens`, whose per-token caches
    /// `state` holds and whose recurrent state at `tokens.len()` is
    /// `snapshot`: the boundary [`boundary_position`] found, materialised so
    /// later prompts with the same prefix resume there. Of `images` (the
    /// prompt's) the entry keeps the spans that lie within the boundary.
    /// Only the disk tier keeps it (see the module docs for why). Returns the
    /// entry id, `None` when the tier did not take it, or the write error;
    /// failures only cost the entry, so the caller logs and carries on.
    pub fn store_durable(
        &mut self,
        tokens: &[u32],
        images: &[CachedImage],
        cache_key: Option<&str>,
        state: &M::State,
        snapshot: &SnapshotOf<M>,
    ) -> Result<Option<String>> {
        let disk = self.disk.as_mut().context("no disk tier")?;
        let n = tokens.len();
        ensure!(
            n > 0 && snapshot.pos() == n,
            "durable snapshot at {} for {n} tokens",
            snapshot.pos()
        );
        ensure!(
            state.pos() >= n,
            "decode state at {} holds no caches for {n} tokens",
            state.pos()
        );
        disk.store_durable(
            tokens,
            &images_within(images, n),
            cache_key,
            &[n],
            &mut |w| state.write_prefix(n, w),
            &mut |_, w| snapshot.write_to(w),
        )
    }

    /// Evicts least-recently-used sessions until `extra` more bytes fit the
    /// budget (or the store is empty): dropped when written ahead, else
    /// spilled to the disk tier on the spot.
    fn trim(&mut self, ctx: &MetalContext, extra: usize) {
        while !self.is_empty() && self.used_bytes() + extra > self.budget_bytes {
            if !self.evict_lru(ctx) {
                break;
            }
        }
    }

    /// Evicts the least recently used session, the one being written ahead
    /// included (the eviction then waits for its write). Returns whether
    /// there was one.
    fn evict_lru(&mut self, ctx: &MetalContext) -> bool {
        let lru = self.lru_index();
        let parked_first = match (self.parked(), lru) {
            (Some(parked), Some(i)) => parked.last_used < self.entries[i].last_used,
            (Some(_), None) => true,
            (None, _) => false,
        };
        let session = if parked_first {
            // A write that already finished costs only its commit.
            let running = self.writing.as_ref().is_some_and(|w| !w.finished());
            let started = Instant::now();
            let Some(session) = self.finish_write(false) else { return false };
            if running {
                let waited = started.elapsed().as_secs_f64();
                self.evictions.waited_secs += waited;
                eprintln!(
                    "session cache: waited {waited:.2}s for the write ahead of {} tokens to finish",
                    session.tokens.len()
                );
            }
            session
        } else {
            match lru {
                Some(i) => self.entries.swap_remove(i),
                None => return false,
            }
        };
        self.evict(ctx, session);
        true
    }

    /// Takes `session` out of GPU memory: a drop when its copy is on disk
    /// already (the entry is marked used, as a spill would have stamped it),
    /// else a synchronous spill. Returns whether the session is on disk.
    fn evict(&mut self, ctx: &MetalContext, session: Session<M>) -> bool {
        self.evictions.evicted += 1;
        match (&session.disk, self.disk.as_mut()) {
            (DiskCopy::Written(id), Some(disk)) if disk.contains(id) => {
                disk.touch(id);
                self.evictions.written_ahead += 1;
                eprintln!(
                    "session cache: evicted {} tokens, written ahead as {id}",
                    session.tokens.len()
                );
                true
            }
            (DiskCopy::Refused, _) => false,
            _ => {
                let started = Instant::now();
                let attempted = self.spillable(ctx, &session);
                let on_disk = self.spill(ctx, session);
                if attempted {
                    self.evictions.spilled += 1;
                    self.evictions.spill_secs += started.elapsed().as_secs_f64();
                }
                on_disk
            }
        }
    }

    /// Between requests: collects a write ahead that finished and, unless
    /// one is still running, starts the next one when the evictions a new
    /// session would cause are not all drops. The room that counts is the
    /// free budget plus the sessions, in eviction order, whose eviction
    /// writes nothing (written ahead, or too short to keep); it should hold
    /// a new session as large as the largest of the last few that were
    /// released (at least the floor). Short of that, the first session in
    /// eviction order that would need a write is written. The most recently
    /// used session is never written ahead: it is the conversation that just
    /// ran, whose next request would make the copy stale at once. Nothing
    /// is allocated for the write, so the resident bytes do not change.
    /// Returns whether a write is in flight.
    pub fn write_ahead(&mut self, ctx: &MetalContext) -> bool {
        self.collect_finished();
        if self.writing.is_some() {
            return true;
        }
        let Some(floor) = self.write_ahead_floor else { return false };
        let Some(disk) = self.disk.as_ref() else { return false };
        if ctx.fault().is_some() {
            return false;
        }
        // Copies the tier has deleted since (budget, age) no longer count.
        for session in &mut self.entries {
            if let DiskCopy::Written(id) = &session.disk
                && !disk.contains(id)
            {
                session.disk = DiskCopy::None;
            }
        }
        let target = self.headroom_target(floor);
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by_key(|&i| self.entries[i].last_used);
        order.pop();
        let mut room = self.budget_bytes.saturating_sub(self.used_bytes());
        for i in order {
            if room >= target {
                return false;
            }
            if !Self::needs_write(&self.entries[i]) {
                room += self.entries[i].bytes();
                continue;
            }
            self.start_write(i);
            return self.writing.is_some();
        }
        false
    }

    /// The room [`Self::write_ahead`] keeps free of writes.
    fn headroom_target(&self, floor: usize) -> usize {
        let recent = self.recent_new.iter().copied().max().unwrap_or(0);
        recent.max(floor).min(self.budget_bytes)
    }

    /// Whether evicting `session` would write it.
    fn needs_write(session: &Session<M>) -> bool {
        session.disk == DiskCopy::None
            && session.tokens.len() >= MIN_DISK_TOKENS
            && session.state.pos() == session.tokens.len()
    }

    /// Parks resident session `index` and writes it to the disk tier on a
    /// background thread.
    fn start_write(&mut self, index: usize) {
        let mut session = self.entries.swap_remove(index);
        let plan = match Plan::of(&session) {
            Ok(plan) => plan,
            Err(error) => {
                eprintln!("session cache: cannot write a session ahead: {error:#}");
                session.disk = DiskCopy::Refused;
                self.entries.push(session);
                return;
            }
        };
        let Some(disk) = self.disk.as_mut() else {
            self.entries.push(session);
            return;
        };
        let reserved = disk.reserve();
        let path = reserved.path().to_path_buf();
        let positions = plan.positions.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        #[cfg(test)]
        let gate = self.write_gate.clone();
        let spawned = std::thread::Builder::new()
            .name("lily-write-ahead".into())
            .spawn(move || {
                lower_writer_priority();
                #[cfg(test)]
                while gate.as_ref().is_some_and(|g| g.load(Ordering::Acquire))
                    && !stop.load(Ordering::Relaxed)
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                // SAFETY: the session the plan points into is parked in
                // `Writing` until this thread is joined (its `Drop` joins
                // before the session goes), and nothing encodes work on, or
                // writes, a parked session.
                unsafe { plan.write(&path, Some(&stop)) }
            });
        match spawned {
            Ok(thread) => {
                self.writing = Some(Writing {
                    session: Some(session),
                    reserved: Some(reserved),
                    positions,
                    cancel,
                    thread: Some(thread),
                    started: Instant::now(),
                });
            }
            Err(error) => {
                eprintln!("session cache: cannot start a write ahead: {error}");
                disk.abandon(reserved);
                self.entries.push(session);
            }
        }
    }

    /// Joins the write ahead, cancelling it first when `cancel`, indexes
    /// what it wrote (or deletes it) and returns the session with its copy
    /// recorded. `None` when nothing was being written.
    fn finish_write(&mut self, cancel: bool) -> Option<Session<M>> {
        let mut writing = self.writing.take()?;
        if cancel {
            writing.cancel.store(true, Ordering::Relaxed);
        }
        let result = writing.join();
        let reserved = writing.reserved.take()?;
        let mut session = writing.session.take()?;
        let secs = writing.started.elapsed().as_secs_f64();
        let n = session.tokens.len();
        let Some(disk) = self.disk.as_mut() else {
            let _ = std::fs::remove_dir_all(reserved.path());
            return Some(session);
        };
        session.disk = match (cancel, result) {
            (true, _) => {
                disk.abandon(reserved);
                self.evictions.cancelled += 1;
                eprintln!(
                    "session cache: cancelled the write ahead of {n} tokens after {secs:.2}s: the request resumes that session"
                );
                DiskCopy::None
            }
            (false, Ok(bytes)) => {
                let new = NewEntry {
                    tokens: &session.tokens,
                    images: &session.images,
                    cache_key: session.cache_key.as_deref(),
                    checkpoints: &writing.positions,
                    durable: false,
                };
                match disk.commit(reserved, &new, bytes) {
                    Ok(Some(id)) => {
                        eprintln!(
                            "session cache: wrote {n} tokens ahead to disk as {id} in {secs:.2}s ({} entries, {:.1}/{:.1} GB on disk)",
                            disk.len(),
                            disk.used_bytes() as f64 / 1e9,
                            disk.budget_bytes() as f64 / 1e9
                        );
                        DiskCopy::Written(id)
                    }
                    Ok(None) => DiskCopy::Refused,
                    Err(error) => {
                        eprintln!("session cache: writing ahead failed: {error:#}");
                        DiskCopy::Refused
                    }
                }
            }
            (false, Err(error)) => {
                disk.abandon(reserved);
                eprintln!("session cache: writing ahead failed: {error:#}");
                DiskCopy::Refused
            }
        };
        Some(session)
    }

    /// Returns a finished write ahead's session to the resident set.
    fn collect_finished(&mut self) {
        if self.writing.as_ref().is_some_and(Writing::finished)
            && let Some(session) = self.finish_write(false)
        {
            self.entries.push(session);
        }
    }

    /// Takes the parked session back when `prompt` resumes it at least as
    /// far as any resident session or disk entry would: the write is
    /// cancelled (its files deleted, nothing was indexed) and the lookup
    /// that follows finds the session resident. A parked session is at rest,
    /// so this is exact: nothing of it was changed or lost.
    fn reclaim_for(&mut self, prompt: &[u32], images: &[CachedImage]) {
        let Some(pos) = self.parked().and_then(|s| s.resume_position(prompt, images))
        else {
            return;
        };
        let resident =
            self.entries.iter().filter_map(|s| s.resume_position(prompt, images)).max();
        let on_disk = self.disk.iter().flat_map(|d| d.entries()).filter_map(|e| {
            resume_position(&e.tokens, &e.images, &e.checkpoints, true, prompt, images)
        });
        if resident.into_iter().chain(on_disk).any(|p| p > pos) {
            return;
        }
        if let Some(session) = self.finish_write(true) {
            self.entries.push(session);
        }
    }

    /// Moves every resident session to the disk tier and empties the store
    /// (a write ahead in flight is finished first); sessions the tier does
    /// not take (too short, or no tier at all) are lost. Returns how many
    /// are on disk and how many were dropped.
    pub fn spill_all(&mut self, ctx: &MetalContext) -> (usize, usize) {
        let (mut spilled, mut dropped) = (0, 0);
        if let Some(session) = self.finish_write(false) {
            self.entries.push(session);
        }
        let entries = std::mem::take(&mut self.entries);
        if self.disk.is_none() && !entries.is_empty() {
            eprintln!(
                "session cache: no disk tier; {} resident sessions are lost",
                entries.len()
            );
        }
        for session in entries {
            if self.evict(ctx, session) {
                spilled += 1;
            } else {
                dropped += 1;
            }
        }
        (spilled, dropped)
    }

    /// Empties the store without writing anything to disk, for a context
    /// whose GPU state can no longer be trusted or read (a fault); a write
    /// ahead in flight is cancelled and its files deleted. Returns how many
    /// sessions were dropped; the disk tier's earlier copies stay.
    pub fn drop_all(&mut self) -> usize {
        let dropped = self.len();
        // `Writing`'s drop cancels, joins and deletes.
        self.writing = None;
        self.entries.clear();
        dropped
    }

    /// Whether [`Self::spill`] would write `session`.
    fn spillable(&self, ctx: &MetalContext, session: &Session<M>) -> bool {
        self.disk.is_some()
            && session.tokens.len() >= MIN_DISK_TOKENS
            && session.state.pos() == session.tokens.len()
            && ctx.fault().is_none()
    }

    /// Writes an evicted session to the disk tier (when there is one and the
    /// session is long enough to be worth it). Failures only cost the copy.
    /// Returns whether the session is now on disk.
    fn spill(&mut self, ctx: &MetalContext, session: Session<M>) -> bool {
        // A faulted context's caches are not trustworthy; the engine is about
        // to drop every session and reports that once, so do not log a
        // failure per session here.
        if !self.spillable(ctx, &session) {
            return false;
        }
        let Some(disk) = self.disk.as_mut() else { return false };
        let started = Instant::now();
        let n = session.tokens.len();
        let result = Plan::of(&session).and_then(|plan| {
            let reserved = disk.reserve();
            // SAFETY: `session` is owned here and at rest (the GPU is idle
            // on it), so the ranges stay valid and unchanged for the write.
            match unsafe { plan.write(reserved.path(), None) } {
                Ok(bytes) => disk.commit(
                    reserved,
                    &NewEntry {
                        tokens: &session.tokens,
                        images: &session.images,
                        cache_key: session.cache_key.as_deref(),
                        checkpoints: &plan.positions,
                        durable: false,
                    },
                    bytes,
                ),
                Err(error) => {
                    disk.abandon(reserved);
                    Err(error)
                }
            }
        });
        match result {
            Ok(Some(id)) => {
                eprintln!(
                    "session cache: spilled {n} tokens to disk as {id} in {:.2}s ({} entries, {:.1}/{:.1} GB on disk)",
                    started.elapsed().as_secs_f64(),
                    disk.len(),
                    disk.used_bytes() as f64 / 1e9,
                    disk.budget_bytes() as f64 / 1e9
                );
                true
            }
            Ok(None) => false,
            Err(error) => {
                eprintln!("session cache: spilling to disk failed: {error:#}");
                false
            }
        }
    }

    fn lru_index(&self) -> Option<usize> {
        self.entries.iter().enumerate().min_by_key(|(_, s)| s.last_used).map(|(i, _)| i)
    }

    /// Returns a session to the cache after a request. Its state must sit at
    /// `tokens.len()`, and `images` are the spans of the prompt it just
    /// served (which are now the lineage's: every earlier span the prompt
    /// kept is among them). Old checkpoints beyond the per-session cap are
    /// dropped (the newest are the ones the next request most likely resumes
    /// from), and the store is trimmed to budget and count. A copy written
    /// ahead of what the session was before the request is stale and is
    /// deleted, as a live-end disk hit consumes its entry. Returns what the
    /// trim evicted.
    pub fn release(
        &mut self,
        ctx: &MetalContext,
        mut session: Session<M>,
        images: &[CachedImage],
        cache_key: Option<&str>,
    ) -> Evictions {
        self.evictions = Evictions::default();
        self.collect_finished();
        if session.state.pos() != session.tokens.len() || session.tokens.is_empty() {
            return std::mem::take(&mut self.evictions);
        }
        if let DiskCopy::Written(id) =
            std::mem::replace(&mut session.disk, DiskCopy::None)
            && let Some(disk) = self.disk.as_mut()
        {
            disk.remove(&id);
        }
        session.images = images_within(images, session.tokens.len());
        session.checkpoints.retain(|c| c.pos() <= session.tokens.len());
        while session.checkpoints.len() > self.max_checkpoints {
            session.checkpoints.remove(0);
        }
        if std::mem::take(&mut session.born) {
            if self.recent_new.len() == RECENT_NEW_SESSIONS {
                self.recent_new.pop_front();
            }
            self.recent_new.push_back(session.bytes());
        }
        self.clock = self.clock.wrapping_add(1);
        session.last_used = self.clock;
        session.cache_key = cache_key.map(str::to_owned);
        self.entries.push(session);
        while self.len() > self.max_sessions
            || (self.used_bytes() > self.budget_bytes && self.len() > 1)
        {
            if !self.evict_lru(ctx) {
                break;
            }
        }
        std::mem::take(&mut self.evictions)
    }
}

pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "../../tests/unit/serve/session.rs"]
mod tests;
