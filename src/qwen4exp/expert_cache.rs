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

use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{Result, ensure};

use super::expert_store::{ExpertStore, Lookup, REGIONS, SlotPolicy};
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
    stop: AtomicBool,
    lookups: AtomicU64,
    misses: AtomicU64,
}

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

    /// Distinct expert lookups and misses served so far.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.service.lookups.load(Ordering::Relaxed),
            self.service.misses.load(Ordering::Relaxed),
        )
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
    worker: Option<JoinHandle<()>>,
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
                Tensor::from_bytes(ctx, bytemuck::cast_slice(&none), &[experts], DType::U32)
            })
            .collect::<Result<Vec<_>>>()?;
        let link = Rc::new(ExpertCacheLink {
            routed: ctx.new_shared_event()?,
            ready: ctx.new_shared_event()?,
            next_seq: Cell::new(0),
            service: Arc::new(Service {
                requests: Mutex::new(VecDeque::new()),
                wake: Condvar::new(),
                stop: AtomicBool::new(false),
                lookups: AtomicU64::new(0),
                misses: AtomicU64::new(0),
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

    /// The protocol handle the cached layers' `MoeWeights` share.
    pub fn link(&self) -> Rc<ExpertCacheLink> {
        self.link.clone()
    }

    /// Bytes of each of a slot's nine regions (the order of
    /// `ExpertStore::read_into`).
    pub fn region_lens(&self) -> [usize; REGIONS] {
        let (i, h, gs, bits) = (self.inter, self.hidden, self.quant.group_size, self.quant.bits);
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
    /// starts the service thread that resolves misses. Must run before any
    /// pass reads the slab.
    pub fn fill_and_serve(&mut self, store: ExpertStore, mut policy: SlotPolicy) -> Result<()> {
        ensure!(self.worker.is_none(), "the expert cache is already being served");
        ensure!(
            policy.n_slots() == self.n_slots
                && store.layers() == self.slot_of.len()
                && store.experts() == self.experts
                && store.region_lens() == self.region_lens(),
            "expert store, policy and slab disagree on the layout"
        );
        let slab = self.pointers();
        let tables: Vec<usize> = self.slot_of.iter().map(|t| t.contents_ptr() as usize).collect();
        let started = std::time::Instant::now();
        let fill: Vec<(u32, usize, usize)> = policy.initial_fill().collect();
        let filled = fill.len();
        // SAFETY: nothing reads the slab before this function returns, and
        // every entry names a distinct slot and expert.
        unsafe { load_slots(&store, &slab, &tables, self.experts, &fill)? };
        eprintln!(
            "expert cache: {filled} of {} slots filled ({} pinned) in {:.1}s",
            self.n_slots,
            policy.pinned(),
            started.elapsed().as_secs_f64()
        );
        let service = self.link.service.clone();
        let (routed, ready) = (self.link.routed.clone(), self.link.ready.clone());
        let experts = self.experts;
        self.worker = Some(std::thread::Builder::new().name("expert-cache".into()).spawn(
            move || serve(store, &mut policy, slab, tables, experts, routed, ready, service),
        )?);
        Ok(())
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        let (lookups, misses) = self.link.stats();
        if lookups > 0 {
            eprintln!(
                "expert cache: {lookups} distinct expert lookups, {misses} misses ({:.2}%)",
                100.0 * misses as f64 / lookups as f64
            );
        }
        self.link.service.stop.store(true, Ordering::Release);
        self.link.service.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Loads every `(slot, layer, expert)` of `entries`: the reads (nine
/// regions per expert) are spread over up to [`LOAD_THREADS`] threads, so
/// a resolution with a few misses pays about one region's read latency,
/// then the tables are pointed at the slots.
///
/// # Safety
///
/// As [`load_slot`], for every entry; the entries name distinct slots and
/// distinct experts.
unsafe fn load_slots(
    store: &ExpertStore,
    slab: &SlabPointers,
    tables: &[usize],
    experts: usize,
    entries: &[(u32, usize, usize)],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    let tasks: Vec<(usize, usize)> = (0..entries.len())
        .flat_map(|i| (0..REGIONS).map(move |r| (i, r)))
        .collect();
    let threads = tasks.len().min(LOAD_THREADS);
    let per = tasks.len().div_ceil(threads);
    let read = |&(i, r): &(usize, usize)| -> Result<()> {
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
        store.read_region(layer, expert, r, dst)
    };
    if threads == 1 {
        for task in &tasks {
            read(task)?;
        }
    } else {
        std::thread::scope(|scope| {
            let workers: Vec<_> = tasks
                .chunks(per)
                .map(|chunk| scope.spawn(move || chunk.iter().try_for_each(read)))
                .collect();
            for worker in workers {
                worker.join().map_err(|_| anyhow::anyhow!("expert load thread panicked"))??;
            }
            Ok::<(), anyhow::Error>(())
        })?;
    }
    for &(slot, layer, expert) in entries {
        // SAFETY: the table is a shared-storage U32 [E] buffer.
        unsafe { set_entry(tables, experts, layer, expert, slot) };
    }
    Ok(())
}

/// Threads a batch of expert reads is spread over.
const LOAD_THREADS: usize = 16;

/// # Safety
///
/// `tables[layer]` is a live `U32 [experts]` shared buffer.
unsafe fn set_entry(tables: &[usize], experts: usize, layer: usize, expert: usize, slot: u32) {
    debug_assert!(expert < experts);
    // SAFETY: as documented.
    unsafe { (tables[layer] as *mut u32).add(expert).write_volatile(slot) };
}

#[allow(clippy::too_many_arguments)]
fn serve(
    store: ExpertStore,
    policy: &mut SlotPolicy,
    slab: SlabPointers,
    tables: Vec<usize>,
    experts: usize,
    routed: SharedEvent,
    ready: SharedEvent,
    service: Arc<Service>,
) {
    let mut ids: Vec<u32> = Vec::new();
    let mut misses: Vec<(u32, usize, usize)> = Vec::new();
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
        // The GPU is stalled behind this request from the moment it
        // signals, so the wait spins: a blocking wait wakes tens of
        // microseconds late and there are 48 of these per token.
        let mut spins = 0u32;
        while routed.signaled_value() < request.seq {
            spins += 1;
            if spins % 4096 == 0 {
                if service.stop.load(Ordering::Acquire) {
                    return;
                }
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: the pass wrote `count` ids at `ids` before signalling.
        let routed_ids = unsafe {
            core::slice::from_raw_parts(request.ids as *const u32, request.count)
        };
        ids.clear();
        ids.extend_from_slice(routed_ids);
        ids.sort_unstable();
        ids.dedup();
        service.lookups.fetch_add(ids.len() as u64, Ordering::Relaxed);
        misses.clear();
        for &e in &ids {
            let expert = e as usize;
            if expert >= experts {
                eprintln!("expert cache: routed id {e} out of range; the pass is corrupt");
                std::process::abort();
            }
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
        // SAFETY: the GPU waits at this layer; the slots were just taken
        // from experts whose readers have completed, and are distinct.
        if let Err(e) = unsafe { load_slots(&store, &slab, &tables, experts, &misses) } {
            eprintln!("expert cache: loading layer {}: {e:#}", request.layer);
            std::process::abort();
        }
        service.misses.fetch_add(misses.len() as u64, Ordering::Relaxed);
        ready.signal(request.seq);
    }
}
