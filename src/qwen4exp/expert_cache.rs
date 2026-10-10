//! The expert cache for machines whose memory cannot hold every routed
//! expert (`docs/low-ram-experts.md`): one slab of expert slots on the GPU
//! that every MoE layer's kernels read through a per-layer slot table, and
//! the host side that fills slots from the checkpoint files and resolves
//! misses while a pass waits.
//!
//! A slot holds one expert's three projections in the layout the gather
//! kernels and the grouped GEMM read for a layer's own stacked experts, so
//! the slab is those stacks with `n_slots` experts instead of `E`, shared
//! by all layers: layer `l`'s `MoeWeights` views the slab and carries the
//! table `slot_of[l]` (`U32 [E]`, [`NONE`] where the expert is not
//! resident) that `moe_ffn` remaps the routed ids through.
//!
//! The protocol per cached MoE layer of a pass: after the router's top-k
//! the pass signals `routed` to a sequence number and waits on `ready` for
//! the same number before the remap. A service thread takes the requests
//! in order, waits for the GPU's signal, reads the routed ids from the
//! shared indices buffer, loads every missing expert into a slot chosen by
//! [`SlotPolicy`] (writing the evicted expert's table entry to [`NONE`]
//! and the loaded one's to its slot), and signals `ready`. The GPU is
//! stalled at the wait meanwhile, so no kernel reads a slot being
//! refilled.
//!
//! The service thread also keeps [`LiveUsage`]: every resolved expert is
//! counted (decode-time and prefill-time apart), every
//! [`PROMOTE_EVERY`] resolutions the cold experts that out-earned the
//! least-used pinned ones take over their pinned role (no data moves),
//! and the counts, merged with the loaded ranking, are written to the
//! usage file every [`PERSIST_SECS`] and at drop, so the next load places
//! experts by what this machine ran. `LILY_EXPERT_ADAPT=0` keeps the
//! counts but never promotes.

use std::cell::Cell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{Result, ensure};

use super::expert_store::{ExpertStore, LiveUsage, Lookup, REGIONS, SlotPolicy};
use crate::config::QuantizationConfig;
use crate::metal::{MetalContext, SharedEvent};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

/// The slot-table entry of an expert that is not resident.
pub const NONE: u32 = u32::MAX;

/// One layer's request for a pass: the routed ids sit at `ids` (a shared
/// buffer the GPU wrote before signalling `seq`).
#[derive(Clone, Copy)]
struct Request {
    seq: u64,
    layer: usize,
    ids: usize,
    count: usize,
}

struct Service {
    requests: Mutex<VecDeque<Request>>,
    wake: Condvar,
    /// Sequence ranges (inclusive) of passes that were encoded but will
    /// never run (`QueuedRequests`), for the request the thread may already
    /// hold; queued ones are removed at cancellation. Pruned as served.
    cancelled: Mutex<Vec<(u64, u64)>>,
    stop: AtomicBool,
    lookups: AtomicU64,
    misses: AtomicU64,
    decode_lookups: AtomicU64,
    decode_misses: AtomicU64,
    promotions: AtomicU64,
    stale: AtomicU64,
    /// Resolutions the GPU did not wait for (see `serve`).
    released: AtomicU64,
    /// Requests skipped because their pass was dropped (`QueuedRequests`),
    /// counted at the thread, so the ones the queue purge did not catch.
    skipped: AtomicU64,
}

impl Service {
    /// Forgets the requests with sequence numbers in `from..=to`: those
    /// still queued are dropped, and the one the thread may hold is skipped
    /// when it checks [`Self::is_cancelled`].
    fn cancel(&self, from: u64, to: u64) {
        let mut queue = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        queue.retain(|r| r.seq < from || r.seq > to);
        drop(queue);
        let mut cancelled = self.cancelled.lock().unwrap_or_else(|e| e.into_inner());
        cancelled.push((from, to));
    }

    /// Whether `seq` was cancelled. Requests are served in rising order, so
    /// ranges entirely below `seq` are pruned here.
    fn is_cancelled(&self, seq: u64) -> bool {
        let mut cancelled = self.cancelled.lock().unwrap_or_else(|e| e.into_inner());
        cancelled.retain(|&(_, to)| to >= seq);
        cancelled.iter().any(|&(from, to)| from <= seq && seq <= to)
    }
}

/// The requests a pass enqueued while it was encoded, held by a pass that
/// may never be committed (a verify pass prepared for one of several
/// outcomes). Dropped, it cancels them, so the service thread never resolves
/// a layer the GPU will not run with whatever the indices buffer holds at
/// the time; [`Self::keep`] lets them stand once the pass is committed.
pub struct QueuedRequests {
    service: Arc<Service>,
    from: u64,
    to: u64,
    armed: bool,
}

impl QueuedRequests {
    /// The pass was committed: its requests are served as encoded.
    pub fn keep(mut self) {
        self.armed = false;
    }
}

impl Drop for QueuedRequests {
    fn drop(&mut self) {
        if self.armed {
            self.service.cancel(self.from, self.to);
        }
    }
}

/// Counters of the cache's service so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertCacheStats {
    /// Distinct expert lookups and the misses among them, all passes.
    pub lookups: u64,
    pub misses: u64,
    /// The same for decode-class passes (batches of a few rows).
    pub decode_lookups: u64,
    pub decode_misses: u64,
    /// Cold experts promoted into the pinned set.
    pub promotions: u64,
    /// Requests of dropped passes the thread skipped (after the queue purge).
    pub skipped: u64,
}

impl ExpertCacheStats {
    /// The counters since `earlier`.
    pub fn since(self, earlier: ExpertCacheStats) -> ExpertCacheStats {
        ExpertCacheStats {
            lookups: self.lookups - earlier.lookups,
            misses: self.misses - earlier.misses,
            decode_lookups: self.decode_lookups - earlier.decode_lookups,
            decode_misses: self.decode_misses - earlier.decode_misses,
            promotions: self.promotions - earlier.promotions,
            skipped: self.skipped - earlier.skipped,
        }
    }

    fn rate(misses: u64, lookups: u64) -> f64 {
        100.0 * misses as f64 / lookups.max(1) as f64
    }

    /// `misses of lookups (x%), decode m of n (y%), p promotions, s skipped`.
    pub fn describe(&self) -> String {
        format!(
            "{} misses of {} lookups ({:.1}%), decode {} of {} ({:.1}%), {} promotions, {} skipped",
            self.misses,
            self.lookups,
            Self::rate(self.misses, self.lookups),
            self.decode_misses,
            self.decode_lookups,
            Self::rate(self.decode_misses, self.decode_lookups),
            self.promotions,
            self.skipped
        )
    }
}

/// Resolutions between promotion passes: about ten decoded tokens at 48
/// cached layers, or ten prefill chunks; a pass over the slots costs a
/// few tens of microseconds. The replays (`expert_replay`, `cache_replay`)
/// use it too.
pub(crate) const PROMOTE_EVERY: u64 = 480;
/// Seconds between writes of the usage file while serving.
const PERSIST_SECS: u64 = 300;
/// Routed ids per resolution up to which a pass counts as decode-class
/// (a step, a verify pass of a few rows), above it as prefill. The replays
/// (`expert_replay`, `cache_replay`) classify their requests by it too.
pub(crate) const DECODE_IDS: usize = 256;

/// What a cached layer's `MoeWeights` holds: the events of the protocol
/// and the request queue the service thread drains.
pub struct ExpertCacheLink {
    pub routed: SharedEvent,
    pub ready: SharedEvent,
    next_seq: Cell<u64>,
    service: Arc<Service>,
}

impl ExpertCacheLink {
    /// Registers the routing of `layer` held in `ids` (`count` `U32`s) and
    /// returns the sequence number the pass must signal `routed` and wait
    /// `ready` with. Sequence numbers rise in encode order, which is the
    /// queue's execution order.
    pub fn enqueue(&self, layer: usize, ids: &Tensor, count: usize) -> u64 {
        let seq = self.next_seq.get() + 1;
        self.next_seq.set(seq);
        let request = Request { seq, layer, ids: ids.contents_ptr() as usize, count };
        self.service
            .requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(request);
        self.service.wake.notify_one();
        seq
    }

    /// The sequence number of the last request enqueued (0 before any).
    pub fn last_seq(&self) -> u64 {
        self.next_seq.get()
    }

    /// A handle on the requests enqueued since `first` (the value
    /// [`Self::last_seq`] returned plus one before the pass was encoded):
    /// `None` when the pass enqueued nothing.
    pub fn queued_since(&self, first: u64) -> Option<QueuedRequests> {
        let to = self.next_seq.get();
        (to >= first).then(|| QueuedRequests {
            service: self.service.clone(),
            from: first,
            to,
            armed: true,
        })
    }

    /// The counters so far.
    pub fn stats(&self) -> ExpertCacheStats {
        let s = &self.service;
        ExpertCacheStats {
            lookups: s.lookups.load(Ordering::Relaxed),
            misses: s.misses.load(Ordering::Relaxed),
            decode_lookups: s.decode_lookups.load(Ordering::Relaxed),
            decode_misses: s.decode_misses.load(Ordering::Relaxed),
            promotions: s.promotions.load(Ordering::Relaxed),
            skipped: s.skipped.load(Ordering::Relaxed),
        }
    }
}

/// The slab's host addresses, for the service thread.
#[derive(Clone, Copy)]
struct SlabPointers {
    regions: [usize; REGIONS],
    lens: [usize; REGIONS],
}

pub struct ExpertCache {
    gate: QuantWeights,
    up: QuantWeights,
    down: QuantWeights,
    /// Per layer, `U32 [E]`.
    slot_of: Vec<Tensor>,
    n_slots: usize,
    experts: usize,
    inter: usize,
    hidden: usize,
    quant: QuantizationConfig,
    link: Rc<ExpertCacheLink>,
    worker: Option<JoinHandle<LiveUsage>>,
    /// Where the live usage is persisted (`None`: not persisted).
    usage_out: Option<PathBuf>,
}

impl ExpertCache {
    /// A slab of `n_slots` empty slots for `layers` MoE layers of `experts`
    /// experts with intermediate size `inter` over hidden size `hidden`,
    /// every slot table set to [`NONE`], with the protocol's events.
    pub fn new(
        ctx: &MetalContext,
        n_slots: usize,
        layers: usize,
        experts: usize,
        inter: usize,
        hidden: usize,
        quant: QuantizationConfig,
    ) -> Result<Self> {
        ensure!(n_slots > 0, "an expert cache needs at least one slot");
        ensure!(quant.bits == 4, "the expert cache is written for Q4 experts");
        let stack = |rows: usize, k: usize| -> Result<QuantWeights> {
            let words = k * quant.bits / 32;
            let groups = k / quant.group_size;
            Ok(QuantWeights {
                codes: Tensor::zeros(ctx, &[rows, words], DType::U32)?,
                scales: Tensor::zeros(ctx, &[rows, groups], DType::BF16)?,
                biases: Tensor::zeros(ctx, &[rows, groups], DType::BF16)?,
                group_size: quant.group_size,
                bits: quant.bits,
            })
        };
        let none = vec![NONE; experts];
        let slot_of = (0..layers)
            .map(|_| {
                Tensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&none),
                    &[experts],
                    DType::U32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let link = Rc::new(ExpertCacheLink {
            routed: ctx.new_shared_event()?,
            ready: ctx.new_shared_event()?,
            next_seq: Cell::new(0),
            service: Arc::new(Service {
                requests: Mutex::new(VecDeque::new()),
                wake: Condvar::new(),
                cancelled: Mutex::new(Vec::new()),
                stop: AtomicBool::new(false),
                lookups: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                decode_lookups: AtomicU64::new(0),
                decode_misses: AtomicU64::new(0),
                promotions: AtomicU64::new(0),
                stale: AtomicU64::new(0),
                released: AtomicU64::new(0),
                skipped: AtomicU64::new(0),
            }),
        });
        Ok(Self {
            gate: stack(n_slots * inter, hidden)?,
            up: stack(n_slots * inter, hidden)?,
            down: stack(n_slots * hidden, inter)?,
            slot_of,
            n_slots,
            experts,
            inter,
            hidden,
            quant,
            link,
            worker: None,
            usage_out: None,
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn layers(&self) -> usize {
        self.slot_of.len()
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// The GPU addresses of the buffers the cache allocated (the slab's
    /// nine stacks and the slot tables), so the model load can tell them
    /// from its own weights: `--pin-weights` never pins the slab.
    pub fn buffer_addresses(&self) -> Vec<u64> {
        [&self.gate, &self.up, &self.down]
            .into_iter()
            .flat_map(|q| [&q.codes, &q.scales, &q.biases])
            .chain(&self.slot_of)
            .map(|t| t.binding().0.address())
            .collect()
    }

    /// The protocol handle the cached layers' `MoeWeights` share.
    pub fn link(&self) -> Rc<ExpertCacheLink> {
        self.link.clone()
    }

    /// Bytes of each of a slot's nine regions (the order of
    /// `ExpertStore::read_into`).
    pub fn region_lens(&self) -> [usize; REGIONS] {
        let (i, h, gs, bits) =
            (self.inter, self.hidden, self.quant.group_size, self.quant.bits);
        let gu_codes = i * (h * bits / 32) * 4;
        let gu_sb = i * (h / gs) * 2;
        let dn_codes = h * (i * bits / 32) * 4;
        let dn_sb = h * (i / gs) * 2;
        [gu_codes, gu_sb, gu_sb, gu_codes, gu_sb, gu_sb, dn_codes, dn_sb, dn_sb]
    }

    /// Layer `layer`'s view of the slab: its expert stacks (the whole slab)
    /// and its slot table.
    pub fn layer_weights(
        &self,
        layer: usize,
    ) -> Result<(QuantWeights, QuantWeights, QuantWeights, Tensor)> {
        ensure!(layer < self.slot_of.len(), "layer {layer} has no slot table");
        Ok((
            self.gate.view_rows(0, self.n_slots * self.inter)?,
            self.up.view_rows(0, self.n_slots * self.inter)?,
            self.down.view_rows(0, self.n_slots * self.hidden)?,
            self.slot_of[layer].view(0, &[self.experts])?,
        ))
    }

    fn pointers(&self) -> SlabPointers {
        let tensors = [
            &self.gate.codes,
            &self.gate.scales,
            &self.gate.biases,
            &self.up.codes,
            &self.up.scales,
            &self.up.biases,
            &self.down.codes,
            &self.down.scales,
            &self.down.biases,
        ];
        let mut regions = [0usize; REGIONS];
        for (r, t) in regions.iter_mut().zip(tensors) {
            *r = t.contents_ptr() as usize;
        }
        SlabPointers { regions, lens: self.region_lens() }
    }

    /// Fills the slab per `policy` from `store` (every pinned slice and
    /// the LRU region's prefill), points the tables at the slots, and
    /// starts the service thread that resolves misses, counts `usage`,
    /// promotes when `adapt` and persists to `usage_out`. Must run before
    /// any pass reads the slab.
    pub fn fill_and_serve(
        &mut self,
        store: ExpertStore,
        mut policy: SlotPolicy,
        mut usage: LiveUsage,
        adapt: bool,
        usage_out: Option<PathBuf>,
    ) -> Result<()> {
        ensure!(self.worker.is_none(), "the expert cache is already being served");
        ensure!(
            policy.n_slots() == self.n_slots
                && store.layers() == self.slot_of.len()
                && store.experts() == self.experts
                && store.region_lens() == self.region_lens(),
            "expert store, policy and slab disagree on the layout"
        );
        let slab = self.pointers();
        let tables: Vec<usize> =
            self.slot_of.iter().map(|t| t.contents_ptr() as usize).collect();
        let started = std::time::Instant::now();
        let fill: Vec<(u32, usize, usize)> = policy.initial_fill().collect();
        let filled = fill.len();
        let pool = read_pool()?;
        // SAFETY: nothing reads the slab before this function returns, and
        // every entry names a distinct slot and expert.
        unsafe {
            load_slots(&pool, &store, &slab, &tables, self.experts, &fill, None)?
        };
        eprintln!(
            "expert cache: {filled} of {} slots filled ({} pinned) in {:.1}s",
            self.n_slots,
            policy.pinned(),
            started.elapsed().as_secs_f64()
        );
        let service = self.link.service.clone();
        let (routed, ready) = (self.link.routed.clone(), self.link.ready.clone());
        let experts = self.experts;
        self.usage_out = usage_out.clone();
        self.worker =
            Some(std::thread::Builder::new().name("expert-cache".into()).spawn(
                move || {
                    serve(
                        pool,
                        store,
                        &mut policy,
                        &mut usage,
                        adapt,
                        usage_out.as_deref(),
                        slab,
                        tables,
                        experts,
                        routed,
                        ready,
                        service,
                    );
                    usage
                },
            )?);
        Ok(())
    }
}

/// The kernel [`Warmer`] runs while the GPU waits for a resolution: one
/// SIMD group polling a host-written word until it reaches the resolution's
/// generation, or `cap` polls pass.
const WARM_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void expert_cache_warm(device atomic_uint* flag [[buffer(0)]],
                              constant uint& gen [[buffer(1)]],
                              constant uint& cap [[buffer(2)]]) {
    for (uint i = 0; i < cap; i++) {
        if (int(atomic_load_explicit(flag, memory_order_relaxed) - gen) >= 0) {
            break;
        }
    }
}
"#;

/// Keeps the GPU busy while it waits for misses to be read. A GPU that
/// idles at the handshake for half a millisecond or more at a time drops
/// its clock, and every layer after runs slower, hits included; one SIMD
/// group polling on a second queue for the length of the wait keeps the
/// clock up. The service thread calls [`Self::start`] before a resolution's
/// reads and [`Self::release`] before it signals `ready`; a thread of its
/// own (with its own Metal context, so its submissions never wait behind
/// the model's) commits the polling pass.
struct Warmer {
    /// The generation the GPU-side poll ends at, written by the host.
    flag: usize,
    /// The generation a pass was asked for (0: none yet).
    want: Arc<AtomicU64>,
    thread: std::thread::Thread,
    generation: u32,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Warmer {
    /// Polls per warm pass at most: a few milliseconds, so a lost release
    /// never holds a GPU core for long.
    const CAP: u32 = 1 << 16;

    fn spawn() -> Result<Self> {
        let want = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<Result<usize>>();
        let (w, st) = (want.clone(), stop.clone());
        let handle = std::thread::Builder::new().name("expert-warm".into()).spawn(
            move || {
                let setup = (|| -> Result<_> {
                    let ctx = MetalContext::new()?;
                    let kernel = ctx.pipeline(
                        "expert_cache_warm",
                        WARM_SRC,
                        crate::metal::MslVersion::V3_1,
                    )?;
                    let flag = Tensor::zeros(&ctx, &[1], DType::U32)?;
                    Ok((ctx, kernel, flag))
                })();
                let (ctx, kernel, flag) = match setup {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                let _ = tx.send(Ok(flag.contents_ptr() as usize));
                let mut done = 0u64;
                let mut stats = WarmStats::default();
                while !st.load(Ordering::Acquire) {
                    let generation = w.load(Ordering::Acquire);
                    if generation == done {
                        std::thread::park();
                        continue;
                    }
                    done = generation;
                    let run = || -> Result<crate::metal::CompletedPass> {
                        let pass = ctx.begin()?;
                        pass.dispatch_at(
                            &kernel,
                            &[flag.binding()],
                            &[
                                &(generation as u32).to_ne_bytes(),
                                &Self::CAP.to_ne_bytes(),
                            ],
                            crate::metal::Grid::Threads {
                                grid: (32, 1, 1),
                                threadgroup: (32, 1, 1),
                            },
                        )?;
                        pass.commit()?.wait_retain()
                    };
                    match run() {
                        Ok(pass) => stats.record(pass.timing().ok()),
                        Err(e) => {
                            eprintln!(
                                "expert cache: warm pass failed, warming off: {e:#}"
                            );
                            return;
                        }
                    }
                }
                if stats.passes > 0 && std::env::var_os("LILY_EXPERT_TRACE").is_some() {
                    eprintln!(
                        "expert cache: {} warm passes, GPU {:.3} ms each on average",
                        stats.passes,
                        stats.gpu_secs * 1e3 / stats.passes as f64
                    );
                }
            },
        )?;
        let flag = rx.recv().map_err(|_| anyhow::anyhow!("warm thread ended"))??;
        Ok(Self {
            flag,
            want,
            thread: handle.thread().clone(),
            generation: 0,
            stop,
            handle: Some(handle),
        })
    }

    /// Asks for a warm pass for the resolution about to read.
    fn start(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.want.store(u64::from(self.generation), Ordering::Release);
        self.thread.unpark();
    }

    /// Ends the current warm pass's poll.
    fn release(&self) {
        // SAFETY: a live shared U32 buffer the warm thread keeps until it
        // is stopped, which happens only after the service thread is done.
        let flag = unsafe { &*(self.flag as *const std::sync::atomic::AtomicU32) };
        flag.store(self.generation, Ordering::Release);
    }
}

impl Drop for Warmer {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::Release);
        self.thread.unpark();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[derive(Default)]
struct WarmStats {
    passes: u64,
    gpu_secs: f64,
}

impl WarmStats {
    fn record(&mut self, timing: Option<crate::metal::PassTiming>) {
        self.passes += 1;
        if let Some(t) = timing {
            self.gpu_secs += t.gpu_end_secs - t.gpu_start_secs;
        }
    }
}

/// Writes the merged usage to `path`, logging rather than failing.
fn persist(usage: &LiveUsage, path: &std::path::Path) {
    match usage
        .merged()
        .save(path, "lily expert cache, live usage over the loaded ranking")
    {
        Ok(()) => {}
        Err(e) => eprintln!("expert cache: writing usage to {}: {e:#}", path.display()),
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        let stats = self.link.stats();
        if stats.lookups > 0 {
            eprintln!(
                "expert cache: {}, {} stale reads, {} timed-out waits",
                stats.describe(),
                self.link.service.stale.load(Ordering::Relaxed),
                self.link.service.released.load(Ordering::Relaxed)
            );
        }
        self.link.service.stop.store(true, Ordering::Release);
        self.link.service.wake.notify_all();
        if let Some(worker) = self.worker.take()
            && let Ok(usage) = worker.join()
            && let Some(path) = &self.usage_out
            && stats.lookups > 0
        {
            persist(&usage, path);
        }
    }
}

/// Loads every `(slot, layer, expert)` of `entries`: the reads (nine
/// regions per expert) are spread over `pool`'s threads ([`LOAD_THREADS`],
/// started once with the cache), so a resolution with a few misses pays
/// about one region's read latency, then the tables are pointed at the
/// slots.
///
/// # Safety
///
/// As [`load_slot`], for every entry; the entries name distinct slots and
/// distinct experts.
unsafe fn load_slots(
    pool: &rayon::ThreadPool,
    store: &ExpertStore,
    slab: &SlabPointers,
    tables: &[usize],
    experts: usize,
    entries: &[(u32, usize, usize)],
    trace: Option<(&MissTrace, u64)>,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let read = |k: usize| -> Result<()> {
        let start = trace.map(|(t, _)| t.now());
        let (i, r) = claimed_read(k, entries.len());
        let (slot, layer, expert) = entries[i];
        // SAFETY: shared-storage slab region of n_slots * len bytes; the
        // caller guarantees no reader and distinct slots per entry, and
        // each (entry, region) pair is read by one thread.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                (slab.regions[r] as *mut u8).add(slot as usize * slab.lens[r]),
                slab.lens[r],
            )
        };
        store.read_region(layer, expert, r, dst)?;
        if let (Some((t, seq)), Some(start)) = (trace, start) {
            t.read(seq, k, start);
        }
        Ok(())
    };
    run_claimed(pool, entries.len() * REGIONS, read)?;
    for &(slot, layer, expert) in entries {
        // SAFETY: the table is a shared-storage U32 [E] buffer.
        unsafe { set_entry(tables, experts, layer, expert, slot) };
    }
    Ok(())
}

/// Reads each `(layer, expert)` of `experts` into its buffer in `dst`
/// ([`ExpertStore::slice_bytes`] long, the nine regions back to back in
/// `REGION_NAMES` order) the way [`load_slots`] reads a resolution's misses:
/// one task per region over `pool` and the calling thread
/// ([`run_claimed`]). The server reads into the slab; this serves
/// `lily-ssd-probe`, which times that path without a model.
// TODO: this mirrors `load_slots`' region split; share one helper with it
// when the server's read path is next touched.
pub fn read_experts(
    pool: &rayon::ThreadPool,
    store: &ExpertStore,
    experts: &[(usize, usize)],
    dst: &mut [Vec<u8>],
) -> Result<()> {
    let bytes = store.slice_bytes();
    ensure!(
        dst.len() >= experts.len()
            && dst[..experts.len()].iter().all(|b| b.len() == bytes),
        "{} destinations of {bytes} bytes needed",
        experts.len()
    );
    let lens = store.region_lens();
    let mut starts = [0usize; REGIONS];
    for r in 1..REGIONS {
        starts[r] = starts[r - 1] + lens[r - 1];
    }
    let bases: Vec<usize> =
        dst.iter_mut().take(experts.len()).map(|b| b.as_mut_ptr() as usize).collect();
    let read = |k: usize| -> Result<()> {
        let (i, r) = claimed_read(k, experts.len());
        let (layer, expert) = experts[i];
        // SAFETY: `dst[i]` is `bytes` long and borrowed mutably for the
        // call; each (expert, region) range is written by one task only.
        let dst = unsafe {
            core::slice::from_raw_parts_mut(
                (bases[i] as *mut u8).add(starts[r]),
                lens[r],
            )
        };
        store.read_region(layer, expert, r, dst)
    };
    run_claimed(pool, experts.len() * REGIONS, read)
}

/// The order a resolution's reads are claimed in, as (expert, region) for
/// task `k` of `REGIONS * n`. Up to [`BIG_FIRST_MAX`] experts, every
/// expert's large code regions go first: claims start a few microseconds
/// apart as pool threads wake, and the read that starts last ends the
/// resolution; with regions in file order, an expert's third code region
/// (the down projection's) started some 60 microseconds after the first and
/// finished last in four single misses of five. Larger batches keep each
/// expert's regions together, which `lily-ssd-probe` measured faster from
/// 16 experts on (and the initial fill 10% faster), as a queue of small
/// reads at the end draws less from the drive.
fn claimed_read(k: usize, n: usize) -> (usize, usize) {
    if n <= BIG_FIRST_MAX {
        (k % n, REGIONS_BY_SIZE[k / n])
    } else {
        (k / REGIONS, k % REGIONS)
    }
}

/// The most experts a resolution reads code regions first for
/// ([`claimed_read`]).
const BIG_FIRST_MAX: usize = 8;

/// The regions largest first: the three code tensors, then the scales and
/// biases.
const REGIONS_BY_SIZE: [usize; REGIONS] = [0, 3, 6, 1, 2, 4, 5, 7, 8];

/// Threads in the read pool; the service thread reads alongside them.
pub const LOAD_THREADS: usize = 16;

/// The read pool the cache's service thread loads slots with:
/// [`LOAD_THREADS`] threads, started once.
pub fn read_pool() -> Result<rayon::ThreadPool> {
    Ok(rayon::ThreadPoolBuilder::new()
        .num_threads(LOAD_THREADS)
        .thread_name(|i| format!("expert-read-{i}"))
        .build()?)
}

/// Runs `task(0)` to `task(tasks - 1)`, each once, on the calling thread and
/// up to `pool`'s thread count of pool jobs, every one claiming the next task
/// from a shared counter until none is left, and returns when all have
/// finished: the first error, or an error for a panicking task. The caller
/// starts on the first task at once instead of waiting for pool threads to
/// wake; a parallel iterator's work stealing, or a broadcast that blocks the
/// caller, measured no faster on a few blocking reads than the threads this
/// pool replaced ("The read threads" in `docs/low-ram-experts.md`).
pub fn run_claimed(
    pool: &rayon::ThreadPool,
    tasks: usize,
    task: impl Fn(usize) -> Result<()> + Sync,
) -> Result<()> {
    let next = AtomicUsize::new(0);
    let failed: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let claim = || {
        loop {
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= tasks {
                return;
            }
            if let Err(e) = task(k) {
                failed.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert(e);
                return;
            }
        }
    };
    // One pool job is spawned, and it spawns the others, so the calling
    // thread starts its first read after one wake-up call rather than after
    // all of them (about 18 microseconds for eight).
    let helpers = tasks.min(pool.current_num_threads() + 1).saturating_sub(1);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.in_place_scope(|scope| {
            if helpers > 0 {
                scope.spawn(|scope| {
                    for _ in 1..helpers {
                        scope.spawn(|_| claim());
                    }
                    claim();
                });
            }
            claim();
        })
    }))
    .map_err(|_| anyhow::anyhow!("an expert read panicked"))?;
    match failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// # Safety
///
/// `tables[layer]` is a live `U32 [experts]` shared buffer.
unsafe fn set_entry(
    tables: &[usize],
    experts: usize,
    layer: usize,
    expert: usize,
    slot: u32,
) {
    debug_assert!(expert < experts);
    // SAFETY: as documented.
    unsafe { (tables[layer] as *mut u32).add(expert).write_volatile(slot) };
}

/// Timing of the service thread's resolutions, a diagnostic
/// (`LILY_EXPERT_TRACE=<path>`): per resolution when the thread took the
/// request, saw the GPU's signal, had looked every id up, had read the
/// misses and had signalled `ready`; per region read its start, end and
/// reading thread. Written as `<path>.layers.csv` and `<path>.reads.csv`
/// when the service stops; nanoseconds since the service started. Off, the
/// thread pays one branch per resolution and per read.
struct MissTrace {
    path: PathBuf,
    base: std::time::Instant,
    layers: Vec<LayerTiming>,
    reads: Mutex<Vec<ReadTiming>>,
}

struct LayerTiming {
    seq: u64,
    layer: usize,
    count: usize,
    ids: usize,
    misses: usize,
    popped: u64,
    seen: u64,
    resolved: u64,
    loaded: u64,
    ready: u64,
}

struct ReadTiming {
    seq: u64,
    task: usize,
    /// The pool thread's index, -1 for the service thread.
    thread: i64,
    start: u64,
    end: u64,
}

impl MissTrace {
    fn from_env() -> Option<Self> {
        let path = std::env::var_os("LILY_EXPERT_TRACE")?;
        Some(Self {
            path: PathBuf::from(path),
            base: std::time::Instant::now(),
            layers: Vec::with_capacity(1 << 20),
            reads: Mutex::new(Vec::with_capacity(1 << 20)),
        })
    }

    fn now(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }

    /// Records a region read of resolution `seq` that began at `start`.
    fn read(&self, seq: u64, task: usize, start: u64) {
        let end = self.now();
        let thread = rayon::current_thread_index().map_or(-1, |i| i as i64);
        self.reads.lock().unwrap_or_else(|e| e.into_inner()).push(ReadTiming {
            seq,
            task,
            thread,
            start,
            end,
        });
    }

    fn write(&self) -> std::io::Result<()> {
        use std::io::Write as _;
        let with = |suffix: &str| {
            let mut p = self.path.clone().into_os_string();
            p.push(suffix);
            PathBuf::from(p)
        };
        let mut out =
            std::io::BufWriter::new(std::fs::File::create(with(".layers.csv"))?);
        writeln!(out, "seq,layer,count,ids,misses,popped,seen,resolved,loaded,ready")?;
        for t in &self.layers {
            writeln!(
                out,
                "{},{},{},{},{},{},{},{},{},{}",
                t.seq,
                t.layer,
                t.count,
                t.ids,
                t.misses,
                t.popped,
                t.seen,
                t.resolved,
                t.loaded,
                t.ready
            )?;
        }
        out.flush()?;
        let mut out =
            std::io::BufWriter::new(std::fs::File::create(with(".reads.csv"))?);
        writeln!(out, "seq,task,thread,start,end")?;
        for r in self.reads.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            writeln!(out, "{},{},{},{},{}", r.seq, r.task, r.thread, r.start, r.end)?;
        }
        out.flush()
    }
}

impl Drop for MissTrace {
    fn drop(&mut self) {
        match self.write() {
            Ok(()) => eprintln!(
                "expert cache: {} resolutions traced to {}.*.csv",
                self.layers.len(),
                self.path.display()
            ),
            Err(e) => eprintln!("expert cache: writing the trace: {e}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serve(
    pool: rayon::ThreadPool,
    store: ExpertStore,
    policy: &mut SlotPolicy,
    usage: &mut LiveUsage,
    adapt: bool,
    usage_out: Option<&std::path::Path>,
    slab: SlabPointers,
    tables: Vec<usize>,
    experts: usize,
    routed: SharedEvent,
    ready: SharedEvent,
    service: Arc<Service>,
) {
    let mut ids: Vec<u32> = Vec::new();
    let mut misses: Vec<(u32, usize, usize)> = Vec::new();
    let debug = std::env::var_os("LILY_EXPERT_CACHE_DEBUG").is_some();
    let mut trace = MissTrace::from_env();
    // `LILY_EXPERT_WARM=0` leaves the GPU idle while misses are read.
    let mut warmer = match std::env::var("LILY_EXPERT_WARM").as_deref() {
        Ok("0") => None,
        _ => match Warmer::spawn() {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("expert cache: no GPU warming: {e:#}");
                None
            }
        },
    };
    let mut served = 0u64;
    let mut persisted = std::time::Instant::now();
    loop {
        let request = {
            let mut queue = service.requests.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if service.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(r) = queue.pop_front() {
                    break r;
                }
                queue = service.wake.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
        };
        let popped = trace.as_ref().map_or(0, MissTrace::now);
        // A pass encoded ahead that was dropped instead of committed: its
        // layers never signal, and the ids at its buffer belong to whatever
        // pass runs next.
        if service.is_cancelled(request.seq) {
            service.skipped.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // The GPU is stalled behind this request from the moment it
        // signals, so the wait spins: a blocking wait wakes tens of
        // microseconds late and there are 48 of these per token.
        let mut spins = 0u32;
        let mut cancelled = false;
        while routed.signaled_value() < request.seq {
            spins += 1;
            if spins.is_multiple_of(4096) {
                if service.stop.load(Ordering::Acquire) {
                    return;
                }
                if service.is_cancelled(request.seq) {
                    cancelled = true;
                    break;
                }
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // The signal that ended the spin may have come from the pass that
        // cancelled this request (its first layer runs ahead of its parked
        // wait), so check once more.
        if cancelled || service.is_cancelled(request.seq) {
            service.skipped.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let seen = trace.as_ref().map_or(0, MissTrace::now);
        let seen_at = std::time::Instant::now();
        // SAFETY: the pass wrote `count` ids at `ids` before signalling.
        let routed_ids = unsafe {
            core::slice::from_raw_parts(request.ids as *const u32, request.count)
        };
        ids.clear();
        ids.extend_from_slice(routed_ids);
        if debug {
            // Visibility check: the ids must not change once the signal
            // was seen.
            let first: Vec<u32> = ids.clone();
            let until =
                std::time::Instant::now() + std::time::Duration::from_micros(300);
            while std::time::Instant::now() < until {
                std::hint::spin_loop();
            }
            std::sync::atomic::fence(Ordering::SeqCst);
            let again: Vec<u32> = routed_ids
                .iter()
                .map(|p| unsafe { std::ptr::read_volatile(p) })
                .collect();
            if first != again {
                let changed = first.iter().zip(&again).filter(|(a, b)| a != b).count();
                service.stale.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "expert cache: seq {} layer {}: {changed} of {} ids changed after the signal",
                    request.seq,
                    request.layer,
                    again.len()
                );
                ids.clear();
                ids.extend_from_slice(&again);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        let decode = request.count <= DECODE_IDS;
        service.lookups.fetch_add(ids.len() as u64, Ordering::Relaxed);
        if decode {
            service.decode_lookups.fetch_add(ids.len() as u64, Ordering::Relaxed);
        }
        misses.clear();
        for &e in &ids {
            let expert = e as usize;
            if expert >= experts {
                eprintln!(
                    "expert cache: routed id {e} out of range; the pass is corrupt"
                );
                std::process::abort();
            }
            usage.record(request.layer, expert, decode);
            match policy.lookup(request.layer, expert, request.seq) {
                Lookup::Hit(_) => {}
                Lookup::Miss { slot, evicted } => {
                    if let Some((l2, e2)) = evicted {
                        // SAFETY: the GPU waits at this layer; the evicted
                        // expert's readers have completed.
                        unsafe { set_entry(&tables, experts, l2, e2, NONE) };
                    }
                    misses.push((slot, request.layer, expert));
                }
                Lookup::Full => {
                    eprintln!(
                        "expert cache: no slot free for layer {} expert {expert} (LRU region too small)",
                        request.layer
                    );
                    std::process::abort();
                }
            }
        }
        if !misses.is_empty()
            && let Some(w) = warmer.as_mut()
        {
            w.start();
        }
        let resolved = trace.as_ref().map_or(0, MissTrace::now);
        // SAFETY: the GPU waits at this layer; the slots were just taken
        // from experts whose readers have completed, and are distinct.
        let traced = trace.as_ref().map(|t| (t, request.seq));
        if let Err(e) = unsafe {
            load_slots(&pool, &store, &slab, &tables, experts, &misses, traced)
        } {
            eprintln!("expert cache: loading layer {}: {e:#}", request.layer);
            std::process::abort();
        }
        service.misses.fetch_add(misses.len() as u64, Ordering::Relaxed);
        if decode {
            service.decode_misses.fetch_add(misses.len() as u64, Ordering::Relaxed);
        }
        // The GPU waits at this layer for `ready`, so it cannot have
        // signalled a later layer yet; if it has, a queue-level wait timed
        // out (Metal 4 then releases the queue silently) and the pass ran on
        // with experts that were not loaded.
        if routed.signaled_value() > request.seq
            && service.released.fetch_add(1, Ordering::Relaxed) == 0
        {
            eprintln!(
                "expert cache: layer {} of a {}-id pass ran on before its experts were loaded (a GPU wait timed out after {:.0} ms); that pass computed with missing experts",
                request.layer,
                request.count,
                (std::time::Instant::now() - seen_at).as_secs_f64() * 1e3
            );
        }
        let loaded = trace.as_ref().map_or(0, MissTrace::now);
        if !misses.is_empty()
            && let Some(w) = warmer.as_ref()
        {
            w.release();
        }
        ready.signal(request.seq);
        if let Some(t) = trace.as_mut() {
            let ready = t.now();
            t.layers.push(LayerTiming {
                seq: request.seq,
                layer: request.layer,
                count: request.count,
                ids: ids.len(),
                misses: misses.len(),
                popped,
                seen,
                resolved,
                loaded,
                ready,
            });
        }
        // Off the GPU's critical path: the pass has its slots.
        served += 1;
        if adapt && served.is_multiple_of(PROMOTE_EVERY) {
            let swaps = usage.promote(policy, request.seq);
            if swaps > 0 {
                service.promotions.fetch_add(swaps as u64, Ordering::Relaxed);
            }
        }
        if let Some(path) = usage_out
            && persisted.elapsed().as_secs() >= PERSIST_SECS
        {
            persist(usage, path);
            persisted = std::time::Instant::now();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/expert_cache.rs"]
mod tests;
