//! Decode-time expert prefetch from predictions computed offline
//! (`LILY_EXPERT_PREFETCH_FILE`): the expert cache's side of what
//! `lily-cache-replay --prefetch 0:TOP@CUT+CUT..[r]` replays on the shared
//! drive model, so the stall a replay says a predictor saves can be measured
//! on the real drive and cache before the predictor runs on the GPU
//! (`docs/low-ram-experts.md`, "Prefetch from a prediction file").
//!
//! The file has [`cache_replay::Predictions`]' layout: `[planes][layers]
//! [positions][width]` little-endian `u16` expert ids, each row best first
//! and ending at its first [`NO_PREDICTION`]; plane 0 is the token start's,
//! plane `1 + i` the `i`-th cut's. A decode pass at position `p` (its input
//! token's, as a trace numbers it) reads row `p` of each plane; a verify
//! pass reads its first row's only.
//!
//! The schedule follows the replay's (`CacheSim::session` under
//! `SsdModel::Shared`), on the service thread:
//! - At a decode pass's first request (layer 0), taken from the queue before
//!   the GPU signals it, which is the token's start: plane 0's `top` picks
//!   for every layer in turn, those not resident, each take an LRU slot
//!   ([`SlotPolicy::reserve`], stamped with that request's sequence number,
//!   as the replay stamps a prefetch with the next request's tick, so that
//!   request's misses cannot take it) and queue a read of it.
//! - Once a cut layer's request is resolved (its `ready` signalled), the
//!   cut's plane does the same for the layers after it; with `replace` it
//!   first cancels the queued reads for those layers that it does not pick.
//! - Reads start in queue order, at most `depth` in flight, each expert's
//!   nine regions at once on a pool of their own; demand reads never wait
//!   for a prefetch to start, and the drive serves both as it will.
//! - At a layer's request, a routed expert whose read is still queued is
//!   read with the demand misses, into the slot it took; one in flight is
//!   waited for; the layer's other queued reads are cancelled and their
//!   slots emptied ([`SlotPolicy::forget`]). A read once started is never
//!   cancelled.
//!
//! Where the server cannot be the replay:
//! - The planes come a little earlier: plane 0 when layer 0's request is
//!   taken (the GPU is still finishing the pass before), a cut's when its
//!   `ready` is signalled (the GPU has that layer's experts still to run);
//!   the replay issues both at a layer's end.
//! - A prefetch never evicts a slot used by the last request signalled or
//!   the one being resolved, which a pass on the GPU may still read; the
//!   replay's only spares the next request's.
//! - A lookup that evicts a prefetch's slot cancels the prefetch if it is
//!   queued and waits for it if it is in flight, so no two reads ever write
//!   one slot; the replay keeps the queued read.
//!
//! A slot table entry points at a prefetched slot only once the read has
//! landed, set by the service thread before it signals a `ready`, so the GPU
//! never reads a slot whose data is still on its way.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{Context as _, Result, ensure};

use super::cache_replay::{NO_PREDICTION, PrefetchReplay};
use super::expert_store::{Lookup, SlotPolicy};

/// Picks per row when `LILY_EXPERT_PREFETCH_WIDTH` is not set: what the
/// offline selection of the trained predictors writes.
pub const DEFAULT_WIDTH: usize = 24;

/// Precomputed predictions ([`cache_replay::Predictions`]' layout), read
/// whole.
pub struct PredictionFile {
    ids: Vec<u16>,
    planes: usize,
    layers: usize,
    positions: usize,
    width: usize,
}

impl PredictionFile {
    /// `ids` as `planes` x `layers` x positions x `width`; the positions
    /// follow from the length, which must be a whole number of them.
    pub fn new(
        ids: Vec<u16>,
        planes: usize,
        layers: usize,
        width: usize,
    ) -> Result<Self> {
        ensure!(
            planes > 0 && layers > 0 && width > 0,
            "a prediction file needs planes, layers and a width"
        );
        let per_position = planes * layers * width;
        ensure!(
            !ids.is_empty() && ids.len().is_multiple_of(per_position),
            "{} ids are not {planes} planes x {layers} layers x positions x {width}",
            ids.len()
        );
        Ok(Self { positions: ids.len() / per_position, ids, planes, layers, width })
    }

    /// Reads `path` (little-endian `u16`s) as [`Self::new`] lays it out.
    pub fn open(
        path: &Path,
        planes: usize,
        layers: usize,
        width: usize,
    ) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        ensure!(bytes.len().is_multiple_of(2), "{}: not u16s", path.display());
        let ids =
            bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        Self::new(ids, planes, layers, width)
            .with_context(|| format!("predictions {}", path.display()))
    }

    pub fn planes(&self) -> usize {
        self.planes
    }

    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Plane `plane`'s picks for `layer` at `pos`: up to `top`, best first,
    /// ending before the first [`NO_PREDICTION`]; none past the file's
    /// positions.
    pub fn picks(&self, plane: usize, layer: usize, pos: usize, top: usize) -> &[u16] {
        if pos >= self.positions {
            return &[];
        }
        let at = ((plane * self.layers + layer) * self.positions + pos) * self.width;
        let row = &self.ids[at..at + top.min(self.width)];
        let end = row.iter().position(|&e| e == NO_PREDICTION).unwrap_or(row.len());
        &row[..end]
    }

    /// The largest expert id the file names.
    fn max_id(&self) -> Option<u16> {
        self.ids.iter().copied().filter(|&e| e != NO_PREDICTION).max()
    }
}

/// A prefetch plan with its predictions, checked against the model.
pub struct PrefetchSetup {
    pub plan: PrefetchReplay,
    /// Prefetch reads in flight at most.
    pub depth: usize,
    pub file: PredictionFile,
}

impl PrefetchSetup {
    /// A token-start plan (lookahead 0, cuts optional, no per-layer budget)
    /// over a file with a plane for the token start and each cut, for
    /// `layers` layers of `experts` experts.
    pub fn new(
        plan: PrefetchReplay,
        depth: usize,
        file: PredictionFile,
        layers: usize,
        experts: usize,
    ) -> Result<Self> {
        ensure!(
            plan.lookahead == 0 && plan.budget.is_none() && plan.top > 0,
            "the expert cache prefetches at the token's start: 0:TOP[@CUT+CUT..[r]]"
        );
        ensure!(
            plan.cuts.checked_shr(layers as u32 - 1).unwrap_or(0) == 0,
            "every cut needs a layer after it"
        );
        ensure!(depth >= 1, "a prefetch depth of at least 1");
        let planes = 1 + plan.cuts.count_ones() as usize;
        ensure!(
            file.planes >= planes && file.layers == layers,
            "the prediction file has {} planes of {} layers, {planes} of {layers} needed",
            file.planes,
            file.layers
        );
        ensure!(
            file.max_id().is_none_or(|e| usize::from(e) < experts),
            "a predicted id is past the model's {experts} experts"
        );
        Ok(Self { plan, depth, file })
    }

    /// From the environment, or `None` without `LILY_EXPERT_PREFETCH_FILE`:
    /// the file, `LILY_EXPERT_PREFETCH` (the plan in `lily-cache-replay
    /// --prefetch`'s syntax, required), `LILY_EXPERT_PREFETCH_PLANES` (the
    /// file's planes, by default one per plane the plan reads),
    /// `LILY_EXPERT_PREFETCH_WIDTH` ([`DEFAULT_WIDTH`]) and
    /// `LILY_EXPERT_PREFETCH_DEPTH` (1).
    pub fn from_env(layers: usize, experts: usize) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("LILY_EXPERT_PREFETCH_FILE") else {
            return Ok(None);
        };
        let spec = std::env::var("LILY_EXPERT_PREFETCH").context(
            "LILY_EXPERT_PREFETCH_FILE needs LILY_EXPERT_PREFETCH (e.g. 0:20@3+7+15+23r)",
        )?;
        let plan = PrefetchReplay::parse(&spec)?
            .context("LILY_EXPERT_PREFETCH=off with a prediction file")?;
        let var = |name: &str| -> Result<Option<usize>> {
            std::env::var(name)
                .ok()
                .map(|v| v.parse().with_context(|| format!("{name}={v}")))
                .transpose()
        };
        let planes = var("LILY_EXPERT_PREFETCH_PLANES")?
            .unwrap_or(1 + plan.cuts.count_ones() as usize);
        let width = var("LILY_EXPERT_PREFETCH_WIDTH")?.unwrap_or(DEFAULT_WIDTH);
        let depth = var("LILY_EXPERT_PREFETCH_DEPTH")?.unwrap_or(1);
        let file = PredictionFile::open(Path::new(&path), planes, layers, width)?;
        Self::new(plan, depth, file, layers, experts).map(Some)
    }

    /// For the load's log line.
    pub fn describe(&self) -> String {
        let p = &self.plan;
        let cuts: Vec<String> =
            (0..64).filter(|&k| p.is_cut(k)).map(|k| k.to_string()).collect();
        format!(
            "top {} at the token start{}{}, {} in flight, {} positions",
            p.top,
            if cuts.is_empty() {
                String::new()
            } else {
                format!(", cuts {}", cuts.join("+"))
            },
            if p.replace { " replacing" } else { "" },
            self.depth,
            self.file.positions
        )
    }
}

/// What the prefetch did, in `lily-cache-replay`'s terms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchCounts {
    /// Decode passes whose position the file covers.
    pub tokens: u64,
    /// Reads issued, less those cancelled before they started.
    pub reads: u64,
    /// Of those, the ones whose layer then routed to them, still resident.
    pub useful: u64,
    /// Routed while still queued (read with the demand misses) or in flight
    /// (waited for).
    pub late: u64,
    /// Cancelled before they started: their layer passed without routing to
    /// them, a later plane did not pick them, or a lookup took their slot.
    pub cancelled: u64,
    /// Picks given up because no LRU slot was free.
    pub full: u64,
    /// Reads landed and their summed time, microseconds.
    pub landed: u64,
    pub read_us: u64,
}

impl PrefetchCounts {
    /// The counts since `earlier`.
    pub fn since(self, earlier: PrefetchCounts) -> PrefetchCounts {
        PrefetchCounts {
            tokens: self.tokens - earlier.tokens,
            reads: self.reads - earlier.reads,
            useful: self.useful - earlier.useful,
            late: self.late - earlier.late,
            cancelled: self.cancelled - earlier.cancelled,
            full: self.full - earlier.full,
            landed: self.landed - earlier.landed,
            read_us: self.read_us - earlier.read_us,
        }
    }

    /// `prefetch: r reads (x/token), u useful, l late, c cancelled, f full,
    /// read m ms`.
    pub fn describe(&self) -> String {
        let per = |n: u64| n as f64 / self.tokens.max(1) as f64;
        format!(
            "prefetch over {} tokens: {} reads ({:.2}/token), {} useful ({:.2}/token), {} late, {} cancelled, {} full, {:.2} ms per read",
            self.tokens,
            self.reads,
            per(self.reads),
            self.useful,
            per(self.useful),
            self.late,
            self.cancelled,
            self.full,
            self.read_us as f64 / 1e3 / self.landed.max(1) as f64
        )
    }
}

/// One prefetch read: an expert into the slot it took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Read {
    pub layer: usize,
    pub expert: u32,
    pub slot: u32,
}

/// Starts a read; whatever runs it calls [`Reads::landed`] once the slot
/// holds the expert and is no longer written.
pub(crate) type Launch = Box<dyn Fn(&Arc<Reads>, Read) + Send + Sync>;

/// The reads queued and in flight, shared by the service thread and the
/// jobs that read.
pub(crate) struct Reads {
    state: Mutex<ReadState>,
    depth: usize,
    launch: Launch,
    /// The length of [`ReadState::landed`], so the service thread can skip
    /// the lock when nothing landed.
    unpublished: AtomicUsize,
    landed: AtomicU64,
    read_ns: AtomicU64,
}

#[derive(Default)]
struct ReadState {
    queue: VecDeque<Read>,
    in_flight: Vec<Read>,
    /// Landed, table entries not set yet.
    landed: Vec<Read>,
}

impl Reads {
    pub(crate) fn new(depth: usize, launch: Launch) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(ReadState::default()),
            depth,
            launch,
            unpublished: AtomicUsize::new(0),
            landed: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
        })
    }

    fn lock(&self) -> MutexGuard<'_, ReadState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queues `read` and starts it if the depth has room.
    fn push(self: &Arc<Self>, read: Read) {
        self.lock().queue.push_back(read);
        self.start_queued();
    }

    /// Starts queued reads while fewer than the depth are in flight.
    fn start_queued(self: &Arc<Self>) {
        loop {
            let read = {
                let mut s = self.lock();
                if s.in_flight.len() >= self.depth {
                    return;
                }
                let Some(read) = s.queue.pop_front() else { return };
                s.in_flight.push(read);
                read
            };
            (self.launch)(self, read);
        }
    }

    /// `read` has landed after `took`: its slot holds the expert and nothing
    /// writes it any more. Starts the next queued read.
    pub(crate) fn landed(self: &Arc<Self>, read: Read, took: Duration) {
        {
            let mut s = self.lock();
            if let Some(i) = s.in_flight.iter().position(|r| *r == read) {
                s.in_flight.swap_remove(i);
            }
            s.landed.push(read);
            self.unpublished.fetch_add(1, Ordering::Release);
        }
        self.landed.fetch_add(1, Ordering::Relaxed);
        self.read_ns.fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
        self.start_queued();
    }

    fn in_flight(&self, layer: usize, expert: u32) -> bool {
        self.lock().in_flight.iter().any(|r| r.layer == layer && r.expert == expert)
    }

    /// Spins until no read of `layer`'s `expert` is in flight. The GPU, or
    /// a lookup about to reuse the slot, waits on it: a blocking wait would
    /// wake tens of microseconds late.
    fn wait(&self, layer: usize, expert: u32) {
        let mut spins = 0u32;
        while self.in_flight(layer, expert) {
            spins += 1;
            if spins.is_multiple_of(1024) {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

/// The service thread's half: issues the planes, settles a layer's
/// prefetches at its request, and points the tables at landed reads.
pub(crate) struct Prefetcher {
    plan: PrefetchReplay,
    file: PredictionFile,
    layers: usize,
    experts: usize,
    reads: Arc<Reads>,
    /// `[layer * experts + expert]`: issued, and its table entry not set yet
    /// nor the read cancelled or its slot taken.
    outstanding: Vec<bool>,
    /// The position of the decode pass being served, `None` between them.
    pos: Option<usize>,
    /// Per layer, what this token's planes read for it (`CacheSim`'s
    /// `prefetched`), for counting the useful reads.
    issued: Vec<Vec<u32>>,
    counts: PrefetchCounts,
    /// Scratch: the reads taken from [`ReadState::landed`], a plane's picks.
    landed: Vec<Read>,
    picks: Vec<u16>,
}

impl Prefetcher {
    pub(crate) fn new(
        setup: PrefetchSetup,
        layers: usize,
        experts: usize,
        launch: Launch,
    ) -> Self {
        Self {
            reads: Reads::new(setup.depth, launch),
            plan: setup.plan,
            file: setup.file,
            layers,
            experts,
            outstanding: vec![false; layers * experts],
            pos: None,
            issued: vec![Vec::new(); layers],
            counts: PrefetchCounts::default(),
            landed: Vec::new(),
            picks: Vec::new(),
        }
    }

    pub(crate) fn counts(&self) -> PrefetchCounts {
        PrefetchCounts {
            landed: self.reads.landed.load(Ordering::Relaxed),
            read_us: self.reads.read_ns.load(Ordering::Relaxed) / 1000,
            ..self.counts
        }
    }

    /// A request was taken from the queue, before the GPU signalled it:
    /// `layer` of a pass at `pos` (`None`: not a decode pass, or no position
    /// known). A decode pass's layer 0 starts its token: plane 0 is issued,
    /// stamped `tick` (the request's sequence number), sparing slots used at
    /// `busy_from` or later. Anything but a decode pass cancels what is
    /// queued.
    pub(crate) fn taken(
        &mut self,
        layer: usize,
        pos: Option<usize>,
        policy: &mut SlotPolicy,
        tick: u64,
        busy_from: u64,
    ) {
        match pos {
            Some(p) if layer == 0 => {
                // Leftovers of a pass that never asked for its layers.
                self.cancel_queued(policy, |_| true);
                for read in &mut self.issued {
                    read.clear();
                }
                self.pos = Some(p);
                if p < self.file.positions() {
                    self.counts.tokens += 1;
                }
                for target in 0..self.layers {
                    self.issue(0, target, p, policy, tick, busy_from);
                }
            }
            Some(_) => {}
            None => {
                if self.pos.take().is_some() {
                    self.cancel_queued(policy, |_| true);
                }
            }
        }
    }

    /// Before `layer`'s lookups of `ids` (sorted, distinct): counts the
    /// reads this token issued for it that it routes to and that are still
    /// resident.
    pub(crate) fn before_lookups(
        &mut self,
        layer: usize,
        ids: &[u32],
        policy: &SlotPolicy,
    ) {
        let read = &mut self.issued[layer];
        if read.is_empty() {
            return;
        }
        self.counts.useful += read
            .iter()
            .filter(|&&e| {
                ids.binary_search(&e).is_ok()
                    && policy.slot_of(layer, e as usize).is_some()
            })
            .count() as u64;
        read.clear();
    }

    /// After `layer`'s lookups of `ids` (sorted, distinct), which missed
    /// `misses` (`(slot, layer, expert)`, to be read) and took the slots of
    /// `evicted`: frees the evicted slots of prefetches (cancelled if
    /// queued, waited for if in flight), adds the routed experts whose
    /// prefetch is still queued to `misses`, puts those in flight in
    /// `awaited`, and cancels the layer's other queued reads.
    pub(crate) fn after_lookups(
        &mut self,
        layer: usize,
        ids: &[u32],
        misses: &mut Vec<(u32, usize, usize)>,
        evicted: &[(usize, usize)],
        policy: &mut SlotPolicy,
        awaited: &mut Vec<Read>,
    ) {
        for &(l, e) in evicted {
            self.release(l, e);
        }
        let experts = self.experts;
        if ids.iter().any(|&e| self.outstanding[layer * experts + e as usize]) {
            let mut s = self.reads.lock();
            for &e in ids {
                let flag = &mut self.outstanding[layer * experts + e as usize];
                if !*flag {
                    continue;
                }
                if let Some(i) =
                    s.queue.iter().position(|r| r.layer == layer && r.expert == e)
                {
                    let read = s.queue.remove(i).expect("position in the queue");
                    misses.push((read.slot, layer, e as usize));
                    *flag = false;
                    self.counts.late += 1;
                } else if let Some(read) =
                    s.in_flight.iter().find(|r| r.layer == layer && r.expert == e)
                {
                    awaited.push(*read);
                    self.counts.late += 1;
                }
                // Landed: the table entry is set before `ready`.
            }
        }
        self.cancel_queued(policy, |r| r.layer == layer);
    }

    /// Waits for the reads in `awaited` to land.
    pub(crate) fn wait(&self, awaited: &[Read]) {
        for r in awaited {
            self.reads.wait(r.layer, r.expert);
        }
    }

    /// Points the tables at the slots of the reads that landed and still
    /// hold their expert, through `set(layer, expert, slot)`. Called before
    /// every `ready`.
    pub(crate) fn publish(
        &mut self,
        policy: &SlotPolicy,
        mut set: impl FnMut(usize, usize, u32),
    ) {
        if self.reads.unpublished.load(Ordering::Acquire) == 0 {
            return;
        }
        {
            let mut s = self.reads.lock();
            std::mem::swap(&mut self.landed, &mut s.landed);
            self.reads.unpublished.store(0, Ordering::Release);
        }
        for r in self.landed.drain(..) {
            let (layer, expert) = (r.layer, r.expert as usize);
            let flag = &mut self.outstanding[layer * self.experts + expert];
            // Not outstanding: a lookup took the slot (and waited for it).
            if std::mem::take(flag) && policy.slot_of(layer, expert) == Some(r.slot) {
                set(layer, expert, r.slot);
            }
        }
    }

    /// `layer`'s request was resolved and signalled: after a cut, its plane
    /// for the layers after it, stamped `tick` (the next request's sequence
    /// number), sparing slots used at `busy_from` or later.
    pub(crate) fn resolved(
        &mut self,
        layer: usize,
        policy: &mut SlotPolicy,
        tick: u64,
        busy_from: u64,
    ) {
        let Some(p) = self.pos else { return };
        if !self.plan.is_cut(layer) {
            return;
        }
        let plane = self.plan.cut_plane(layer);
        for target in layer + 1..self.layers {
            if self.plan.replace {
                let keep = self.file.picks(plane, target, p, self.plan.top);
                let keep: Vec<u32> = keep.iter().map(|&e| u32::from(e)).collect();
                self.cancel_queued(policy, |r| {
                    r.layer == target && !keep.contains(&r.expert)
                });
            }
            self.issue(plane, target, p, policy, tick, busy_from);
        }
    }

    /// Reads `plane`'s picks for `target` at `pos` that are not resident.
    fn issue(
        &mut self,
        plane: usize,
        target: usize,
        pos: usize,
        policy: &mut SlotPolicy,
        tick: u64,
        busy_from: u64,
    ) {
        let mut picks = std::mem::take(&mut self.picks);
        picks.clear();
        picks.extend_from_slice(self.file.picks(plane, target, pos, self.plan.top));
        for &expert in &picks {
            let e = usize::from(expert);
            if policy.slot_of(target, e).is_some() {
                continue;
            }
            match policy.reserve(target, e, tick, busy_from) {
                Lookup::Miss { slot, evicted } => {
                    if let Some((l, x)) = evicted {
                        self.release(l, x);
                    }
                    self.outstanding[target * self.experts + e] = true;
                    self.issued[target].push(u32::from(expert));
                    self.counts.reads += 1;
                    self.reads.push(Read {
                        layer: target,
                        expert: u32::from(expert),
                        slot,
                    });
                }
                Lookup::Full => {
                    self.counts.full += 1;
                    break;
                }
                Lookup::Hit(_) => unreachable!("reserve of a resident expert"),
            }
        }
        self.picks = picks;
    }

    /// A lookup took the slot of `expert` of `layer`: if a prefetch of it is
    /// queued it is cancelled, if one is in flight it is waited for, so the
    /// slot's new owner is not overwritten.
    fn release(&mut self, layer: usize, expert: usize) {
        let flag = &mut self.outstanding[layer * self.experts + expert];
        if !std::mem::take(flag) {
            return;
        }
        let e = expert as u32;
        self.issued[layer].retain(|&x| x != e);
        let mut s = self.reads.lock();
        if let Some(i) = s.queue.iter().position(|r| r.layer == layer && r.expert == e)
        {
            s.queue.remove(i);
            self.counts.reads -= 1;
            self.counts.cancelled += 1;
        } else if s.in_flight.iter().any(|r| r.layer == layer && r.expert == e) {
            drop(s);
            self.reads.wait(layer, e);
        }
    }

    /// Cancels the queued reads `which` selects: their slots are emptied.
    fn cancel_queued(
        &mut self,
        policy: &mut SlotPolicy,
        which: impl Fn(&Read) -> bool,
    ) {
        let mut dropped = Vec::new();
        self.reads.lock().queue.retain(|r| {
            let drop = which(r);
            if drop {
                dropped.push(*r);
            }
            !drop
        });
        for r in dropped {
            let e = r.expert as usize;
            self.outstanding[r.layer * self.experts + e] = false;
            self.issued[r.layer].retain(|&x| x != r.expert);
            policy.forget(r.layer, e);
            self.counts.reads -= 1;
            self.counts.cancelled += 1;
        }
    }
}

impl Drop for Prefetcher {
    /// Nothing may write the slab after the service stops: the queue is
    /// dropped and the reads in flight are waited for.
    fn drop(&mut self) {
        self.reads.lock().queue.clear();
        loop {
            if self.reads.lock().in_flight.is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/expert_prefetch.rs"]
mod tests;
