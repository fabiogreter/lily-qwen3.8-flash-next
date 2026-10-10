//! Replays logged expert routing through the expert cache's own policy code
//! ([`SlotPolicy`], [`LiveUsage`]), on the host and without the model: what
//! the cache would have missed on a recorded workload, for any slot count
//! and LRU share. Used by `lily-cache-replay`
//! on routing that `lily-trace-routing` logged for real agent sessions
//! (`docs/low-ram-experts.md`, "Replaying agent sessions").
//!
//! The request stream is the service thread's: one request per MoE layer
//! per pass. A prefill pass covers up to a prefill chunk of a request's new
//! tokens and resolves the union of their routed experts; a decode pass is
//! one token's ten. A request is decode-class when it routes at most
//! `DECODE_IDS` ids, as in the service, so a suffix of 25 tokens or fewer
//! counts as decode there too. Decoding is replayed one token per pass, or
//! with [`ReplayConfig::speculation`] as verify passes of the pending token
//! and its drafts ([`SpecReplay`]), one request per layer for all their
//! rows; the rows of drafts that were not accepted, which a trace does not
//! hold, are made up as [`RejectedRows`] says.
//!
//! A decode-time prefetch can be replayed from logged predictions
//! (`lily-trace-routing --predict`), as `expert_replay` replays it for one
//! prompt: after a decoded token's request for layer `l`, the predicted
//! experts for layer `l + d` that are not resident are looked up at the
//! next request's tick, taking LRU slots as a real load would. Reads land
//! at once unless the replay is timed against a modelled SSD
//! ([`SsdTiming`]).

use std::ops::Range;

use anyhow::{Context as _, Result, ensure};

use super::expert_cache::{DECODE_IDS, PROMOTE_EVERY};
use super::expert_store::{LiveUsage, Lookup, SlotPolicy, UsageRanking};

/// One request of a recorded session: the tokens it prefilled (the suffix
/// after the session's cached prefix) and the tokens it generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRequest {
    pub prefill: Range<usize>,
    pub decode: Range<usize>,
}

/// The cache's configuration for a replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayConfig {
    pub slots: usize,
    pub lru_share: f64,
    /// Live promotion (`LILY_EXPERT_ADAPT`).
    pub adapt: bool,
    /// Prompt tokens per prefill pass.
    pub prefill_chunk: usize,
    /// The decode-time prefetch; `None`: off.
    pub prefetch: Option<PrefetchReplay>,
    /// [`LiveUsage::with_half_life`], in decoded tokens; `None`: no decay.
    pub half_life: Option<u64>,
    /// Times the decode against a modelled SSD; `None`: counts only.
    pub timing: Option<SsdTiming>,
    /// Replays decoding as speculative verify passes; `None`: one token per
    /// pass.
    pub speculation: Option<SpecReplay>,
}

/// Decoding as speculative verify passes: each pass runs the pending token
/// and up to `drafts` drafted tokens through every layer together (one
/// request per layer for the union of their experts) and accepts a prefix
/// of the drafts, each with probability `accept` in turn. The pass emits
/// the accepted drafts plus one token of its own, so the next pass starts
/// at the first token not accepted. A trace holds the routing of the
/// tokens that were kept only: rows for drafts that were not accepted
/// route as `rejected` says. The draws are seeded by `seed` and the
/// session's index ([`CacheSim::starting_at_session`]), so a replay
/// repeats and every session draws its own, from two streams, acceptances
/// and [`RejectedRows::Random`]'s rows: both kinds of rejected rows replay
/// the same acceptances, session by session. Prefetches: the
/// token start's plane covers the pending token only (nothing is known
/// about the drafts' rows before the pass); cuts and lookaheads cover every
/// row whose routing is known, as each row runs through the cut layers
/// itself. Stall and tokens count per accepted token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecReplay {
    pub drafts: usize,
    pub accept: f64,
    pub rejected: RejectedRows,
    pub seed: u64,
}

/// What a draft row that was not accepted routes to, which no trace holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectedRows {
    /// The routing of the token the trace kept at that position (as if the
    /// draft had been right: optimistic, the experts are those the next
    /// passes need anyway), and its predictions.
    Next,
    /// `top_k` experts uniformly at random per layer, no predictions
    /// (pessimistic: unrelated experts, cold more often than routed ones).
    Random,
}

/// SplitMix64, for the replay's seeded draws.
struct Rng(u64);

/// SplitMix64's output function: a bijective mix of all 64 bits.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Stream `stream` of session `session` under `seed`: hashed apart, so
    /// no two streams start near each other in SplitMix's sequence (seeds
    /// one increment apart would replay each other shifted by a draw).
    fn stream(seed: u64, session: u64, stream: u64) -> Self {
        Self(mix(mix(mix(seed) ^ session) ^ stream))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// Uniform in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The decode timeline of a timed replay. A decode pass runs its layers in
/// order, each taking `layer_ms` when nothing is missing (the warm
/// per-layer time under the cache, handshake included) whatever the pass's
/// row count: a verify pass of the pending token and its drafts is charged
/// one token's time per layer, the GPU time its extra rows add is not
/// modelled. A layer's request comes `mixer_share` into it, after its
/// attention or DeltaNet mixer. A layer waits for its missed experts and
/// for prefetched ones still in flight; a prefetch still queued when its
/// layer asks becomes a demand read, and the ones its layer did not route
/// to are cancelled and their slots emptied ([`SlotPolicy::forget`]),
/// never a routed expert's; a prefetch once started is not cancelled.
/// Prefill is not timed.
///
/// The SSD is [`SsdModel::Serial`] unless [`CacheSim::with_ssd`] says
/// otherwise: one expert at a time in `read_ms` (bandwidth bound: 2.76 MB
/// at about 6.5 GB/s), demand reads before queued prefetches, a read once
/// started not interrupted. `read_ms` is that model's only parameter;
/// [`SsdModel::Shared`] carries its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsdTiming {
    pub layer_ms: f64,
    pub mixer_share: f64,
    pub read_ms: f64,
}

/// How a timed replay's SSD serves reads ([`SsdTiming`]).
///
/// A probe of the real read path (`lily-ssd-probe`: q4-xl, 2.76 MB experts
/// read uncached through the 16-thread read pool) found the serial model
/// wrong in shape. A batch of `k` experts asked for at once took about
/// 0.28 + 0.225 k ms (p50 0.51 ms at k = 1, 2.17 at 8, 7.43 at 32; peak
/// 4 450 experts/s, 12.3 GB/s), so reads overlap rather than queue; and
/// with `D` background reads kept in flight, one demand read took 0.62,
/// 1.04, 1.89 and 3.59 ms at `D` = 1, 4, 8 and 16: the drive shares its
/// bandwidth among the reads in flight and gives demand reads no priority.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SsdModel {
    /// One read at a time in [`SsdTiming::read_ms`], demand first.
    Serial,
    Shared(SharedSsd),
}

/// [`SsdModel::Shared`]: each read waits `latency_ms` (using no bandwidth),
/// then transfers `transfer_ms` worth of work at full bandwidth, the
/// bandwidth shared equally among all reads in their transfer
/// (processor sharing). Demand reads start at once. Prefetches start in
/// issue order, no earlier than issued, at most `prefetch_depth` in flight;
/// one its layer waits for stays a prefetch (it holds its place in the
/// depth until it lands).
///
/// [`SharedSsd::PROBED`] is the best fit to the probe's p50s (an expert
/// as one read): demand batches of 1 to 32 within 10%, and one demand read
/// against 1 to 16 background reads within 7%. The shape misses where many
/// demand reads meet many background reads: a batch of 8 took the probe
/// 4.8, 8.2 and 15.5 ms against 4, 8 and 16 background reads, the model
/// 43 to 66% less. The probe's background lanes had all nine regions of an
/// expert in flight while the read pool holds a batch to 17 regions at
/// once, and the drive shares by request, not by expert; per-expert
/// sharing is optimistic for large demand batches under a deep prefetch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedSsd {
    pub latency_ms: f64,
    pub transfer_ms: f64,
    pub prefetch_depth: usize,
}

impl SharedSsd {
    /// Fitted to the probe ([`SsdModel`]). Four prefetches in flight: the
    /// probe's background reached 4 420 of its 4 450 experts/s peak at four
    /// (2 160 at one, 3 600 at two), while a demand read's time grew from
    /// 0.62 ms at one to 1.04 at four and 1.89 at eight.
    pub const PROBED: SharedSsd =
        SharedSsd { latency_ms: 0.33, transfer_ms: 0.217, prefetch_depth: 4 };
}

/// The SSD and the clock of a timed replay ([`SsdTiming`]).
struct Clock {
    timing: SsdTiming,
    now: f64,
    /// [`SsdModel::Serial`]: when the read in service ends.
    ssd_free: f64,
    /// Prefetches not started: `(layer, expert, issued at)`, in issue order.
    queue: std::collections::VecDeque<(usize, u32, f64)>,
    /// [`SsdModel::Serial`]: prefetches started: `(layer, expert)` -> when
    /// they land.
    landing: std::collections::HashMap<(usize, u32), f64>,
    /// [`SsdModel::Shared`]'s reads in flight; `None`: serial.
    shared: Option<SharedState>,
}

impl Clock {
    fn new(timing: SsdTiming, model: SsdModel) -> Self {
        Self {
            timing,
            now: 0.0,
            ssd_free: 0.0,
            queue: Default::default(),
            landing: Default::default(),
            shared: match model {
                SsdModel::Serial => None,
                SsdModel::Shared(m) => Some(SharedState::new(m)),
            },
        }
    }

    /// Waits for the SSD to finish what it has started (or queued): a
    /// request starts decoding on an idle SSD.
    fn idle(&mut self) {
        match &mut self.shared {
            None => self.now = self.now.max(self.ssd_free),
            Some(s) => {
                s.run_until(&mut self.queue, self.now);
                s.run_while(&mut self.queue, |s| !s.reads.is_empty());
                self.now = self.now.max(s.at);
            }
        }
    }

    /// Starts queued prefetches the SSD reaches before `until`.
    fn run_queue(&mut self, until: f64) {
        while let Some(&(layer, expert, issued)) = self.queue.front() {
            let start = self.ssd_free.max(issued);
            if start >= until {
                break;
            }
            self.queue.pop_front();
            self.ssd_free = start + self.timing.read_ms;
            self.landing.insert((layer, expert), self.ssd_free);
        }
    }

    /// One demand read asked for at `t`: when it lands.
    fn demand(&mut self, t: f64) -> f64 {
        self.ssd_free = self.ssd_free.max(t) + self.timing.read_ms;
        self.ssd_free
    }

    /// `layer`'s request at `t` with its routed experts (distinct) and the
    /// ones the policy missed: the stall, how many prefetched experts had
    /// not landed, and (into `cancelled`) the prefetches for `layer` still
    /// queued, which no longer serve any purpose. A routed expert queued
    /// and missed (its slot evicted after the prefetch took it) is among
    /// them, though its slot is the demand read's now: the caller must not
    /// empty a routed expert's slot.
    fn resolve(
        &mut self,
        layer: usize,
        t: f64,
        routed: &[u32],
        missed: &[u32],
        cancelled: &mut Vec<u32>,
    ) -> (f64, u64) {
        if let Some(s) = &mut self.shared {
            return s.resolve(&mut self.queue, layer, t, routed, missed, cancelled);
        }
        self.run_queue(t);
        let (mut done, mut late) = (t, 0);
        for &e in routed {
            if missed.contains(&e) {
                done = done.max(self.demand(t));
            } else if let Some(i) =
                self.queue.iter().position(|&(l, x, _)| l == layer && x == e)
            {
                // Queued before in flight: a read still in flight for the
                // same expert is an older one, for a slot since evicted.
                self.queue.remove(i);
                done = done.max(self.demand(t));
                late += 1;
            } else if let Some(land) = self.landing.get(&(layer, e)).copied()
                && land > t
            {
                done = done.max(land);
                late += 1;
            }
        }
        self.landing.retain(|_, &mut land| land > t);
        cancel_queued(&mut self.queue, layer, cancelled);
        (done - t, late)
    }
}

/// Moves `layer`'s queued prefetches from `queue` to `cancelled`.
fn cancel_queued(
    queue: &mut std::collections::VecDeque<(usize, u32, f64)>,
    layer: usize,
    cancelled: &mut Vec<u32>,
) {
    cancelled.clear();
    queue.retain(|&(l, e, _)| {
        let stale = l == layer;
        if stale {
            cancelled.push(e);
        }
        !stale
    });
}

/// One read in flight under [`SsdModel::Shared`].
struct Read {
    id: u64,
    /// The prefetch's `(layer, expert)`; `None`: a demand read.
    prefetch: Option<(usize, u32)>,
    /// In its latency until then; `None`: transferring.
    ready: Option<f64>,
    /// Transfer work left, in ms at full bandwidth.
    left: f64,
}

/// [`SsdModel::Shared`]'s state, simulated event by event: between events
/// every transferring read progresses at `1 / n` of full bandwidth.
struct SharedState {
    model: SharedSsd,
    /// The time the state is at.
    at: f64,
    reads: Vec<Read>,
    /// Prefetches among `reads`.
    prefetching: usize,
    next_id: u64,
    /// The reads a resolving layer waits for.
    awaited: Vec<u64>,
}

/// Transfer work below which a read has landed, ms: rounding only.
const LANDED_MS: f64 = 1e-6;

impl SharedState {
    fn new(model: SharedSsd) -> Self {
        Self {
            model,
            at: 0.0,
            reads: Vec::new(),
            prefetching: 0,
            next_id: 0,
            awaited: Vec::new(),
        }
    }

    /// Starts a read now: its id.
    fn start(&mut self, prefetch: Option<(usize, u32)>) -> u64 {
        self.next_id += 1;
        self.prefetching += usize::from(prefetch.is_some());
        self.reads.push(Read {
            id: self.next_id,
            prefetch,
            ready: Some(self.at + self.model.latency_ms),
            left: self.model.transfer_ms,
        });
        self.next_id
    }

    /// Starts the queued prefetches that are issued and have room.
    fn start_queued(
        &mut self,
        queue: &mut std::collections::VecDeque<(usize, u32, f64)>,
    ) {
        while self.prefetching < self.model.prefetch_depth
            && let Some(&(layer, expert, issued)) = queue.front()
            && issued <= self.at
        {
            queue.pop_front();
            self.start(Some((layer, expert)));
        }
    }

    /// The next event: a latency ending, a transfer landing, or a queued
    /// prefetch's issue with room for it; infinity for none.
    fn next_event(&self, queue: &std::collections::VecDeque<(usize, u32, f64)>) -> f64 {
        let mut next = f64::INFINITY;
        let (mut n, mut least) = (0usize, f64::INFINITY);
        for r in &self.reads {
            match r.ready {
                Some(ready) => next = next.min(ready),
                None => {
                    n += 1;
                    least = least.min(r.left);
                }
            }
        }
        if n > 0 {
            next = next.min(self.at + n as f64 * least);
        }
        if self.prefetching < self.model.prefetch_depth
            && let Some(&(_, _, issued)) = queue.front()
        {
            next = next.min(issued.max(self.at));
        }
        next
    }

    /// Moves the state to `to` (no event before it): transfers progress,
    /// latencies that end start transferring, transfers done land.
    fn progress(&mut self, to: f64) {
        let n = self.reads.iter().filter(|r| r.ready.is_none()).count();
        if n > 0 {
            let done = (to - self.at) / n as f64;
            for r in self.reads.iter_mut().filter(|r| r.ready.is_none()) {
                r.left -= done;
            }
        }
        self.at = self.at.max(to);
        let at = self.at;
        let mut landed_prefetches = 0;
        self.reads.retain_mut(|r| match r.ready {
            Some(ready) => {
                if ready <= at {
                    r.ready = None;
                }
                true
            }
            None => {
                let landed = r.left <= LANDED_MS;
                if landed && r.prefetch.is_some() {
                    landed_prefetches += 1;
                }
                !landed
            }
        });
        self.prefetching -= landed_prefetches;
    }

    /// Runs the SSD to `until` (not back: a time already passed is now).
    fn run_until(
        &mut self,
        queue: &mut std::collections::VecDeque<(usize, u32, f64)>,
        until: f64,
    ) {
        let until = until.max(self.at);
        loop {
            self.start_queued(queue);
            let next = self.next_event(queue);
            if next > until {
                self.progress(until);
                return;
            }
            self.progress(next);
        }
    }

    /// Runs the SSD event by event while `pending` holds.
    fn run_while(
        &mut self,
        queue: &mut std::collections::VecDeque<(usize, u32, f64)>,
        pending: impl Fn(&Self) -> bool,
    ) {
        loop {
            self.start_queued(queue);
            if !pending(self) {
                return;
            }
            let next = self.next_event(queue);
            debug_assert!(next.is_finite(), "waiting on an idle SSD");
            if !next.is_finite() {
                return;
            }
            self.progress(next);
        }
    }

    /// [`Clock::resolve`] under [`SsdModel::Shared`].
    fn resolve(
        &mut self,
        queue: &mut std::collections::VecDeque<(usize, u32, f64)>,
        layer: usize,
        t: f64,
        routed: &[u32],
        missed: &[u32],
        cancelled: &mut Vec<u32>,
    ) -> (f64, u64) {
        self.run_until(queue, t);
        let mut late = 0;
        let mut awaited = std::mem::take(&mut self.awaited);
        awaited.clear();
        for &e in routed {
            if missed.contains(&e) {
                awaited.push(self.start(None));
            } else if let Some(i) =
                queue.iter().position(|&(l, x, _)| l == layer && x == e)
            {
                // As in [`Clock::resolve`]: the queued read is the current
                // slot's, whatever older one is in flight.
                queue.remove(i);
                awaited.push(self.start(None));
                late += 1;
            } else if let Some(r) =
                self.reads.iter().rfind(|r| r.prefetch == Some((layer, e)))
            {
                // The latest started, should an older one still be in flight.
                awaited.push(r.id);
                late += 1;
            }
        }
        cancel_queued(queue, layer, cancelled);
        self.run_while(queue, |s| s.reads.iter().any(|r| awaited.contains(&r.id)));
        self.awaited = awaited;
        (self.at - t, late)
    }
}

/// A prefetch during decode: after layer `l`'s request, read the `top`
/// best-scored predictions for layer `l + lookahead` that are not resident,
/// at most `budget` per layer. Lookahead 0 reads for every layer at the
/// token's start instead, from the predictions' first plane (a whole-token
/// predictor such as [`TokenRouting`]); with `cuts` it then reads again
/// when each cut layer ends, for the layers after it, from the cut's own
/// plane (a predictor that has seen the token's first layers). With
/// `replace`, a cut also cancels the reads still queued for its layers that
/// it does not pick itself (timed replays only: untimed reads land at once).
/// `budget` caps a layer's reads over the token start and every cut
/// together, so without `replace` a cut adds nothing to a layer whose
/// budget the token start has used up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchReplay {
    /// 0 (token start, every layer), 1 or 2.
    pub lookahead: usize,
    pub top: usize,
    pub budget: Option<usize>,
    /// Lookahead 0 only: bit `k` set reads again after layer `k`, from plane
    /// 1 + the number of cuts below `k`.
    pub cuts: u64,
    pub replace: bool,
}

impl PrefetchReplay {
    /// Parses `off` (`None`) or `LOOKAHEAD:TOP[:BUDGET][@CUT+CUT..[r]]`,
    /// the syntax of `lily-cache-replay --prefetch` and of the server's
    /// `LILY_EXPERT_PREFETCH` (`expert_prefetch`).
    pub fn parse(spec: &str) -> Result<Option<Self>> {
        if spec == "off" {
            return Ok(None);
        }
        let (head, cuts, replace) = match spec.split_once('@') {
            Some((head, cuts)) => {
                let (cuts, replace) = match cuts.strip_suffix('r') {
                    Some(c) => (c, true),
                    None => (cuts, false),
                };
                let mut mask = 0u64;
                for c in cuts.split('+') {
                    let k: u32 =
                        c.parse().with_context(|| format!("prefetch {spec}"))?;
                    ensure!(k < 63, "prefetch {spec}: cut {k} out of range");
                    mask |= 1 << k;
                }
                (head, mask, replace)
            }
            None => (spec, 0, false),
        };
        let v: Vec<usize> = head
            .split(':')
            .map(|x| x.parse().with_context(|| format!("prefetch {spec}")))
            .collect::<Result<_>>()?;
        ensure!(
            (2..=3).contains(&v.len()),
            "prefetch {spec}: LOOKAHEAD:TOP[:BUDGET][@CUT]"
        );
        Ok(Some(Self {
            lookahead: v[0],
            top: v[1],
            budget: v.get(2).copied(),
            cuts,
            replace,
        }))
    }

    /// The plane a cut after layer `k` reads.
    pub fn cut_plane(&self, k: usize) -> usize {
        1 + (self.cuts & ((1u64 << k) - 1)).count_ones() as usize
    }

    /// Whether a cut follows layer `layer`.
    pub fn is_cut(&self, layer: usize) -> bool {
        layer < 64 && self.cuts >> layer & 1 == 1
    }
}

/// Ends a row of [`Predictions`] early: a predictor that picks fewer than
/// `width` experts for a layer (none, for a layer it leaves alone).
pub const NO_PREDICTION: u16 = u16::MAX;

/// Logged predictions of one session (`lily-trace-routing --predict`):
/// `[planes][layers][tokens][width]` expert ids, row-wise best first, a row
/// ending at its first [`NO_PREDICTION`]; plane `d - 1` holds lookahead
/// `d` (for lookahead 0, plane 0 the token start's and the next ones the
/// cuts', in layer order, [`PrefetchReplay`]).
#[derive(Clone, Copy, Debug)]
pub struct Predictions<'a> {
    pub ids: &'a [u16],
    pub planes: usize,
    pub width: usize,
}

/// What a replay counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    pub prefill_tokens: u64,
    pub prefill_lookups: u64,
    pub prefill_misses: u64,
    pub decode_tokens: u64,
    /// Decode passes: as many as tokens, fewer under speculation.
    pub decode_passes: u64,
    pub decode_lookups: u64,
    pub decode_misses: u64,
    /// Summed over decode passes (one per decoded token without
    /// speculation, one per verify pass with it): the layers with at least
    /// one miss.
    pub decode_layers_missed: u64,
    /// Lookups the policy could not place (the LRU region too small).
    pub full: u64,
    pub promotions: u64,
    /// Prefetch loads issued.
    pub prefetch_reads: u64,
    /// Of those, the ones the target layer's request then routed to.
    pub prefetch_useful: u64,
    /// Prefetches given up because no LRU slot was free at the tick, or the
    /// slots taken for the next request left only as many as its pass routes
    /// ids.
    pub prefetch_full: u64,
    /// Timed replay: prefetches still queued when their layer had passed,
    /// cancelled unread (not in `prefetch_reads`).
    pub prefetch_cancelled: u64,
    /// Timed replay: decode stall, in microseconds, and prefetched experts
    /// a layer had to wait for.
    pub decode_stall_us: u64,
    pub prefetch_late: u64,
    /// Decode misses whose expert the same layer routed to within the last
    /// 64 (1 024) positions of the session: what a predictor from recent
    /// routing could have known. Over the rows the trace holds only: the
    /// experts of rows [`RejectedRows::Random`] makes up neither count
    /// here nor mark an expert as recently routed.
    pub decode_misses_recent_64: u64,
    pub decode_misses_recent_1024: u64,
}

impl ReplayCounts {
    pub fn plus(&mut self, o: &ReplayCounts) {
        self.prefill_tokens += o.prefill_tokens;
        self.prefill_lookups += o.prefill_lookups;
        self.prefill_misses += o.prefill_misses;
        self.decode_tokens += o.decode_tokens;
        self.decode_passes += o.decode_passes;
        self.decode_lookups += o.decode_lookups;
        self.decode_misses += o.decode_misses;
        self.decode_layers_missed += o.decode_layers_missed;
        self.full += o.full;
        self.promotions += o.promotions;
        self.prefetch_reads += o.prefetch_reads;
        self.prefetch_useful += o.prefetch_useful;
        self.prefetch_full += o.prefetch_full;
        self.prefetch_cancelled += o.prefetch_cancelled;
        self.decode_stall_us += o.decode_stall_us;
        self.prefetch_late += o.prefetch_late;
        self.decode_misses_recent_64 += o.decode_misses_recent_64;
        self.decode_misses_recent_1024 += o.decode_misses_recent_1024;
    }
}

/// The cache's host state, driven one request at a time as the service
/// thread drives it.
pub struct CacheSim {
    config: ReplayConfig,
    layers: usize,
    experts: usize,
    /// Per layer, over everything replayed: decode misses and useful
    /// prefetch reads.
    layer_misses: Vec<u64>,
    layer_useful: Vec<u64>,
    /// The last request's missed experts.
    missed: Vec<u32>,
    /// `[layer * experts + expert]` -> 1 + the last position of the current
    /// session that routed to it, 0 for none.
    last_seen: Vec<usize>,
    policy: SlotPolicy,
    usage: LiveUsage,
    tick: u64,
    served: u64,
    ids: Vec<u32>,
    /// [`Self::log_misses`]: the last session's decode misses.
    miss_log: Option<Vec<u8>>,
    /// [`Self::with_ssd`].
    ssd: SsdModel,
    /// The index of the next session replayed, which seeds its draws
    /// ([`Self::starting_at_session`]).
    session_index: u64,
    /// The routed ids of the decode pass being replayed (rows x top-k): the
    /// LRU slots prefetches leave free for each of its requests.
    keep_free: usize,
    /// The tick the last prefetches were stamped with, and how many.
    stamped: (u64, usize),
}

impl CacheSim {
    pub fn new(
        ranking: &UsageRanking,
        top_k: usize,
        config: ReplayConfig,
    ) -> Result<Self> {
        let (layers, experts) = (ranking.layers, ranking.experts);
        ensure!(config.slots >= 2 * experts, "at least two layers' worth of slots");
        let lru_share =
            SlotPolicy::lru_share_for(config.lru_share, experts, config.slots);
        Ok(Self {
            config,
            layers,
            experts,
            layer_misses: vec![0; layers],
            layer_useful: vec![0; layers],
            missed: Vec::new(),
            last_seen: vec![0; layers * experts],
            policy: SlotPolicy::new(config.slots, layers, experts, ranking, lru_share)?,
            usage: LiveUsage::new(ranking, config.slots, top_k)
                .with_half_life(config.half_life),
            tick: 0,
            served: 0,
            ids: Vec::new(),
            miss_log: None,
            ssd: SsdModel::Serial,
            session_index: 0,
            keep_free: 0,
            stamped: (0, 0),
        })
    }

    /// Numbers the next session replayed `index` (each later one the next):
    /// a session's speculation draws ([`SpecReplay`]) are seeded by its
    /// index, so a cache started fresh for session `i` draws what session
    /// `i` of one cache across sessions does.
    pub fn starting_at_session(mut self, index: u64) -> Self {
        self.session_index = index;
        self
    }

    /// Times the decode ([`ReplayConfig::timing`]) against `ssd` rather
    /// than [`SsdModel::Serial`].
    pub fn with_ssd(mut self, ssd: SsdModel) -> Self {
        self.ssd = ssd;
        self
    }

    /// Keeps, per session, which of each decoded position's routed experts
    /// missed ([`Self::take_miss_log`]): labels for a predictor trained on
    /// what the cache actually lacked.
    pub fn log_misses(mut self) -> Self {
        self.miss_log = Some(Vec::new());
        self
    }

    /// The last session's miss log ([`Self::log_misses`]): `[layers][tokens]
    /// [top_k]`, aligned with its routing, 1 where a decoded position's
    /// routed expert missed, 0 elsewhere (prefill positions included).
    pub fn take_miss_log(&mut self) -> Option<Vec<u8>> {
        self.miss_log.as_mut().map(std::mem::take)
    }

    /// The usage the cache would persist now ([`LiveUsage::merged`]).
    pub fn merged_usage(&self) -> UsageRanking {
        self.usage.merged()
    }

    /// Decode misses and useful prefetch reads per layer, summed over
    /// everything replayed.
    pub fn by_layer(&self) -> (&[u64], &[u64]) {
        (&self.layer_misses, &self.layer_useful)
    }

    /// The pinned slots (after the loader's clamp of the LRU share).
    pub fn pinned(&self) -> usize {
        self.policy.pinned()
    }

    /// One request: `layer`'s routed ids for one pass (with repeats, as the
    /// GPU wrote them). Returns the misses.
    pub fn request(
        &mut self,
        layer: usize,
        routed: &[u32],
        counts: &mut ReplayCounts,
    ) -> u64 {
        self.tick += 1;
        let decode = routed.len() <= DECODE_IDS;
        self.ids.clear();
        self.ids.extend_from_slice(routed);
        self.ids.sort_unstable();
        self.ids.dedup();
        let mut misses = 0;
        self.missed.clear();
        for i in 0..self.ids.len() {
            let expert = self.ids[i] as usize;
            self.usage.record(layer, expert, decode);
            match self.policy.lookup(layer, expert, self.tick) {
                Lookup::Hit(_) => {}
                Lookup::Miss { .. } => {
                    misses += 1;
                    self.missed.push(self.ids[i]);
                }
                Lookup::Full => {
                    misses += 1;
                    self.missed.push(self.ids[i]);
                    counts.full += 1;
                }
            }
        }
        if decode {
            counts.decode_lookups += self.ids.len() as u64;
            counts.decode_misses += misses;
        } else {
            counts.prefill_lookups += self.ids.len() as u64;
            counts.prefill_misses += misses;
        }
        self.served += 1;
        if self.config.adapt && self.served.is_multiple_of(PROMOTE_EVERY) {
            counts.promotions += self.usage.promote(&mut self.policy, self.tick) as u64;
        }
        misses
    }

    /// The picks of plane `plane` for layer `target` at `pos` (up to `top`,
    /// ending at [`NO_PREDICTION`]), resident or not.
    fn picks(
        &self,
        p: PrefetchReplay,
        pred: &Predictions<'_>,
        (plane, target, pos, tokens): (usize, usize, usize, usize),
    ) -> Vec<u32> {
        let at = ((plane * self.layers + target) * tokens + pos) * pred.width;
        pred.ids[at..at + p.top.min(pred.width)]
            .iter()
            .take_while(|&&e| e != NO_PREDICTION)
            .map(|&e| u32::from(e))
            .collect()
    }

    /// The prefetch for layer `target` of the decoded token at `pos` from
    /// prediction plane `plane`, after the request of layer `target -
    /// lookahead` (or at the token's start, or after the cut): the experts
    /// read go to `prefetched`.
    fn prefetch(
        &mut self,
        p: PrefetchReplay,
        pred: &Predictions<'_>,
        (plane, target, pos, tokens): (usize, usize, usize, usize),
        prefetched: &mut Vec<u32>,
        counts: &mut ReplayCounts,
    ) -> Result<()> {
        let at = ((plane * self.layers + target) * tokens + pos) * pred.width;
        // The read happens at the next request's tick, so that request's own
        // misses cannot take the slot, as they cannot take one a pass reads.
        let tick = self.tick + 1;
        if self.stamped.0 != tick {
            self.stamped = (tick, 0);
        }
        // The slots stamped `tick` leave that request's misses as many LRU
        // slots as its pass routes ids, as in the server
        // (`expert_prefetch::Prefetcher`), where a request that finds none
        // free cannot go on.
        let cap = (self.policy.n_slots() - self.policy.pinned())
            .saturating_sub(self.keep_free);
        for &e in &pred.ids[at..at + p.top.min(pred.width)] {
            if e == NO_PREDICTION || p.budget.is_some_and(|b| prefetched.len() >= b) {
                break;
            }
            let expert = usize::from(e);
            ensure!(expert < self.experts, "predicted id {e} out of range");
            if self.policy.slot_of(target, expert).is_some() {
                continue;
            }
            if self.stamped.1 >= cap {
                counts.prefetch_full += 1;
                break;
            }
            match self.policy.lookup(target, expert, tick) {
                Lookup::Miss { .. } => {
                    self.stamped.1 += 1;
                    counts.prefetch_reads += 1;
                    prefetched.push(u32::from(e));
                }
                Lookup::Hit(_) => unreachable!("checked not resident"),
                Lookup::Full => {
                    counts.prefetch_full += 1;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Replays one recorded session: `routing` is `[layers][tokens][top_k]`
    /// (the layout `lily-trace-routing` writes), `requests` its requests in
    /// order. Prefill passes cover up to a prefill chunk of a request's new
    /// tokens each; decode passes one token each, or under
    /// [`ReplayConfig::speculation`] the pending token and its drafts.
    /// `predicted` drives the prefetch, and is required when the
    /// configuration has one.
    pub fn session(
        &mut self,
        routing: &[u32],
        predicted: Option<&Predictions<'_>>,
        tokens: usize,
        top_k: usize,
        requests: &[TraceRequest],
        counts: &mut ReplayCounts,
    ) -> Result<()> {
        ensure!(
            routing.len() == self.layers * tokens * top_k,
            "routing holds {} ids, {} layers x {tokens} tokens x {top_k} expected",
            routing.len(),
            self.layers
        );
        let prefetch = match (self.config.prefetch, predicted) {
            (None, _) => None,
            (Some(p), Some(pred)) => {
                ensure!(
                    p.lookahead <= 2,
                    "lookahead {} (0, token start, or 1 or 2)",
                    p.lookahead
                );
                ensure!(
                    p.cuts == 0
                        || (p.lookahead == 0
                            && p.cuts.checked_shr(self.layers as u32 - 1).unwrap_or(0)
                                == 0),
                    "cuts ({:#x}) need lookahead 0 and a layer after each",
                    p.cuts
                );
                let planes_needed = if p.lookahead == 0 {
                    1 + p.cuts.count_ones() as usize
                } else {
                    p.lookahead
                };
                ensure!(
                    pred.planes >= planes_needed,
                    "{} prediction planes, {planes_needed} needed",
                    pred.planes
                );
                ensure!(
                    pred.ids.len() == pred.planes * self.layers * tokens * pred.width,
                    "predictions hold {} ids, {} x {} layers x {tokens} tokens x {} \
                     expected",
                    pred.ids.len(),
                    pred.planes,
                    self.layers,
                    pred.width
                );
                Some((p, *pred))
            }
            (Some(_), None) => anyhow::bail!("a prefetch needs the predictions"),
        };
        let at = |layer: usize, pos: usize| (layer * tokens + pos) * top_k;
        let mut pass: Vec<u32> = Vec::new();
        let mut prefetched: Vec<Vec<u32>> = vec![Vec::new(); self.layers];
        self.last_seen.fill(0);
        if let Some(log) = &mut self.miss_log {
            log.clear();
            log.resize(routing.len(), 0);
        }
        let ssd = self.ssd;
        let mut clock = self.config.timing.map(|t| Clock::new(t, ssd));
        let seed = self.config.speculation.map_or(0, |sp| sp.seed);
        let (mut accept_rng, mut rows_rng) = (
            Rng::stream(seed, self.session_index, 0),
            Rng::stream(seed, self.session_index, 1),
        );
        self.session_index += 1;
        let mut cancelled: Vec<u32> = Vec::new();
        let experts = self.experts;
        for r in requests {
            ensure!(
                r.prefill.start <= r.prefill.end
                    && r.prefill.end <= r.decode.start
                    && r.decode.start <= r.decode.end
                    && r.decode.end <= tokens,
                "request {r:?} outside the session's {tokens} tokens"
            );
            let mut pos = r.prefill.start;
            while pos < r.prefill.end {
                let end = (pos + self.config.prefill_chunk).min(r.prefill.end);
                for layer in 0..self.layers {
                    pass.clear();
                    pass.extend_from_slice(&routing[at(layer, pos)..at(layer, end)]);
                    self.request(layer, &pass, counts);
                    for (row, ids) in pass.chunks(top_k).enumerate() {
                        for &e in ids {
                            self.last_seen[layer * experts + e as usize] =
                                pos + row + 1;
                        }
                    }
                }
                counts.prefill_tokens += (end - pos) as u64;
                pos = end;
            }
            // Prefill is not timed: the SSD is taken to be idle when the
            // request starts decoding.
            if let Some(c) = &mut clock {
                c.idle();
            }
            let mut pos = r.decode.start;
            while pos < r.decode.end {
                // The pass's rows: the pending token at `pos`, then the
                // drafts; rows `0..=accepted` hold routing the trace kept.
                let (rows, accepted) = match self.config.speculation {
                    None => (1, 0),
                    Some(sp) => {
                        let drafts = sp.drafts.min(r.decode.end - pos - 1);
                        let mut a = 0;
                        while a < drafts && accept_rng.next_f64() < sp.accept {
                            a += 1;
                        }
                        (1 + drafts, a)
                    }
                };
                self.keep_free = rows * top_k;
                // Rows with routing (and predictions) from the trace.
                let known = match self.config.speculation {
                    Some(SpecReplay { rejected: RejectedRows::Random, .. }) => {
                        accepted + 1
                    }
                    _ => rows,
                };
                let mut missed = 0;
                if let Some((p, pred)) = &prefetch
                    && p.lookahead == 0
                {
                    for (target, read) in prefetched.iter_mut().enumerate() {
                        let before = read.len();
                        self.prefetch(
                            *p,
                            pred,
                            (0, target, pos, tokens),
                            read,
                            counts,
                        )?;
                        if let Some(c) = &mut clock {
                            let now = c.now;
                            c.queue.extend(
                                read[before..].iter().map(|&e| (target, e, now)),
                            );
                        }
                    }
                }
                for layer in 0..self.layers {
                    pass.clear();
                    pass.extend_from_slice(
                        &routing[at(layer, pos)..at(layer, pos + known)],
                    );
                    for _ in known..rows {
                        for _ in 0..top_k {
                            pass.push((rows_rng.next_u64() % experts as u64) as u32);
                        }
                    }
                    let read = &mut prefetched[layer];
                    if !read.is_empty() {
                        pass.sort_unstable();
                        pass.dedup();
                        // Read for this layer, routed to and still resident.
                        let policy = &self.policy;
                        let useful = read
                            .iter()
                            .filter(|&&e| {
                                pass.contains(&e)
                                    && policy.slot_of(layer, e as usize).is_some()
                            })
                            .count() as u64;
                        counts.prefetch_useful += useful;
                        self.layer_useful[layer] += useful;
                        read.clear();
                    }
                    let misses = self.request(layer, &pass, counts);
                    self.layer_misses[layer] += misses;
                    if let Some(log) = &mut self.miss_log
                        && misses > 0
                    {
                        // The rows the trace kept.
                        let (a, z) = (at(layer, pos), at(layer, pos + accepted + 1));
                        for (flag, e) in log[a..z].iter_mut().zip(&routing[a..z]) {
                            *flag = u8::from(self.missed.contains(e));
                        }
                    }
                    let mut layer_end = 0.0;
                    if let Some(c) = &mut clock {
                        let t = c.now + c.timing.mixer_share * c.timing.layer_ms;
                        let (stall, late) = c.resolve(
                            layer,
                            t,
                            &self.ids,
                            &self.missed,
                            &mut cancelled,
                        );
                        for &e in &cancelled {
                            // A queued duplicate of a routed expert the
                            // policy missed: the demand read fills its slot.
                            if self.ids.binary_search(&e).is_err() {
                                self.policy.forget(layer, e as usize);
                            }
                        }
                        counts.prefetch_reads -= cancelled.len() as u64;
                        counts.prefetch_cancelled += cancelled.len() as u64;
                        counts.decode_stall_us += (stall * 1000.0).round() as u64;
                        counts.prefetch_late += late;
                        layer_end = t
                            + stall
                            + (1.0 - c.timing.mixer_share) * c.timing.layer_ms;
                        c.now = layer_end;
                    }
                    // Recency over the rows the trace holds only: rows made
                    // up for rejected drafts have no history to be recent in.
                    let known_ids = &routing[at(layer, pos)..at(layer, pos + known)];
                    for &e in &self.missed {
                        if !known_ids.contains(&e) {
                            continue;
                        }
                        let last = self.last_seen[layer * experts + e as usize];
                        if last > 0 && pos + 1 - last <= 64 {
                            counts.decode_misses_recent_64 += 1;
                        }
                        if last > 0 && pos + 1 - last <= 1024 {
                            counts.decode_misses_recent_1024 += 1;
                        }
                    }
                    for &e in known_ids {
                        self.last_seen[layer * experts + e as usize] = pos + 1;
                    }
                    if misses > 0 {
                        missed += 1;
                    }
                    if let Some((p, pred)) = &prefetch
                        && p.is_cut(layer)
                    {
                        // The cut's plane, ready when the cut layer ends,
                        // queued behind what is still queued.
                        let plane = p.cut_plane(layer);
                        for (target, read) in
                            prefetched.iter_mut().enumerate().skip(layer + 1)
                        {
                            if p.replace
                                && let Some(c) = &mut clock
                            {
                                let keep: Vec<u32> = (0..known)
                                    .flat_map(|j| {
                                        self.picks(
                                            *p,
                                            pred,
                                            (plane, target, pos + j, tokens),
                                        )
                                    })
                                    .collect();
                                let mut dropped = Vec::new();
                                c.queue.retain(|&(l, e, _)| {
                                    let stale = l == target && !keep.contains(&e);
                                    if stale {
                                        dropped.push(e);
                                    }
                                    !stale
                                });
                                for e in dropped {
                                    self.policy.forget(target, e as usize);
                                    read.retain(|&x| x != e);
                                    counts.prefetch_reads -= 1;
                                    counts.prefetch_cancelled += 1;
                                }
                            }
                            let before = read.len();
                            for j in 0..known {
                                let at = (plane, target, pos + j, tokens);
                                self.prefetch(*p, pred, at, read, counts)?;
                            }
                            if let Some(c) = &mut clock {
                                c.queue.extend(
                                    read[before..]
                                        .iter()
                                        .map(|&e| (target, e, layer_end)),
                                );
                            }
                        }
                    }
                    if let Some((p, pred)) = &prefetch
                        && p.lookahead > 0
                    {
                        let target = layer + p.lookahead;
                        if target < self.layers {
                            let mut read = std::mem::take(&mut prefetched[target]);
                            let before = read.len();
                            for j in 0..known {
                                self.prefetch(
                                    *p,
                                    pred,
                                    (p.lookahead - 1, target, pos + j, tokens),
                                    &mut read,
                                    counts,
                                )?;
                            }
                            // The prediction is ready when layer `l` ends.
                            if let Some(c) = &mut clock {
                                c.queue.extend(
                                    read[before..]
                                        .iter()
                                        .map(|&e| (target, e, layer_end)),
                                );
                            }
                            prefetched[target] = read;
                        }
                    }
                }
                counts.decode_tokens += accepted as u64 + 1;
                counts.decode_passes += 1;
                counts.decode_layers_missed += missed;
                pos += accepted + 1;
            }
        }
        Ok(())
    }
}

/// Routing counted by input token id, the whole-token predictor
/// candidate: how often a position whose input was token `t` routed to
/// expert `e` at layer `l`. Known when a pass starts (the sampled token, or
/// a draft), so every layer's reads could be issued before the pass reaches
/// it. Rows exist for the tokens given at construction only.
pub struct TokenRouting {
    layers: usize,
    experts: usize,
    rows: std::collections::HashMap<u32, usize>,
    /// `[row][layer][expert]`.
    counts: Vec<u32>,
}

impl TokenRouting {
    /// Rows for `tokens` (the ids predictions will be asked for).
    pub fn new(
        layers: usize,
        experts: usize,
        tokens: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut rows = std::collections::HashMap::new();
        for t in tokens {
            let n = rows.len();
            rows.entry(t).or_insert(n);
        }
        let counts = vec![0; rows.len() * layers * experts];
        Self { layers, experts, rows, counts }
    }

    /// Counts (`add`) or uncounts every position of a session: `ids` its
    /// tokens, `routing` `[layers][tokens][top_k]`. Tokens without a row
    /// are skipped.
    pub fn count(&mut self, ids: &[u32], routing: &[u32], top_k: usize, add: bool) {
        let (layers, experts, tokens) = (self.layers, self.experts, ids.len());
        for (p, t) in ids.iter().enumerate() {
            let Some(&row) = self.rows.get(t) else { continue };
            for l in 0..layers {
                let base = (row * layers + l) * experts;
                for &e in &routing[(l * tokens + p) * top_k..][..top_k] {
                    let c = &mut self.counts[base + e as usize];
                    *c = if add { *c + 1 } else { *c - 1 };
                }
            }
        }
    }

    /// Predictions for `positions` of a session in the layout
    /// [`Predictions`] reads, in its first plane (lookahead 0): for each
    /// layer, the `width` experts the position's token routed to most often,
    /// best first, ties to the lower id; zero for a token never counted.
    pub fn predict(&self, ids: &[u32], positions: &[usize], width: usize) -> Vec<u16> {
        let (layers, experts, tokens) = (self.layers, self.experts, ids.len());
        let mut out = vec![0u16; 2 * layers * tokens * width];
        let mut order: Vec<u16> = Vec::with_capacity(experts);
        for &p in positions {
            let Some(&row) = self.rows.get(&ids[p]) else { continue };
            for l in 0..layers {
                let c = &self.counts[(row * layers + l) * experts..][..experts];
                if c.iter().all(|&x| x == 0) {
                    continue;
                }
                order.clear();
                order.extend(0..experts as u16);
                let key = |&e: &u16| (std::cmp::Reverse(c[e as usize]), e);
                order.select_nth_unstable_by_key(width - 1, key);
                order.truncate(width);
                order.sort_unstable_by_key(key);
                let at = (l * tokens + p) * width;
                out[at..at + width].copy_from_slice(&order);
            }
        }
        out
    }
}

/// Cross-layer transition counts learned from routing, the host-side
/// predictor candidate: how often a position routed to expert `e` at
/// layer `l` routed to `f` at layer `l + d`, for `d` 1 and 2. A service
/// thread holding layer `l`'s routed ids at its handshake could predict
/// layers `l + 1` and `l + 2` from them without any GPU work.
#[derive(Clone)]
pub struct Transitions {
    layers: usize,
    experts: usize,
    /// `[d - 1][l][e][f]`.
    counts: Vec<u32>,
    /// `[d - 1][l][e]`: the positions that routed to `e` at layer `l`
    /// with a layer `l + d`.
    from: Vec<u32>,
}

impl Transitions {
    pub fn new(layers: usize, experts: usize) -> Self {
        Self {
            layers,
            experts,
            counts: vec![0; 2 * layers * experts * experts],
            from: vec![0; 2 * layers * experts],
        }
    }

    /// Counts (`add`) or uncounts every position of `routing`
    /// (`[layers][tokens][top_k]`).
    pub fn count(&mut self, routing: &[u32], tokens: usize, top_k: usize, add: bool) {
        let (layers, experts) = (self.layers, self.experts);
        let at = |l: usize, p: usize| (l * tokens + p) * top_k;
        for d in 1..=2 {
            for l in 0..layers.saturating_sub(d) {
                let base = (d - 1) * layers + l;
                for p in 0..tokens {
                    let src = &routing[at(l, p)..at(l, p) + top_k];
                    let dst = &routing[at(l + d, p)..at(l + d, p) + top_k];
                    for &e in src {
                        let e = e as usize;
                        let row = (base * experts + e) * experts;
                        let from = &mut self.from[base * experts + e];
                        *from = if add { *from + 1 } else { *from - 1 };
                        for &f in dst {
                            let c = &mut self.counts[row + f as usize];
                            *c = if add { *c + 1 } else { *c - 1 };
                        }
                    }
                }
            }
        }
    }

    /// Predictions for `positions` of `routing` in the layout
    /// [`Predictions`] reads (`[2][layers][tokens][width]`, zero elsewhere):
    /// for layer `t`, the `width` experts with the highest
    /// `sum over e routed at t - d of P(f at t | e at t - d)`, best first,
    /// ties to the lower id.
    pub fn predict(
        &self,
        routing: &[u32],
        tokens: usize,
        top_k: usize,
        positions: &[usize],
        width: usize,
    ) -> Vec<u16> {
        let (layers, experts) = (self.layers, self.experts);
        let probs: Vec<f32> = (0..self.counts.len())
            .map(|i| {
                let from = self.from[i / experts];
                if from == 0 { 0.0 } else { self.counts[i] as f32 / from as f32 }
            })
            .collect();
        let mut out = vec![0u16; 2 * layers * tokens * width];
        let mut score = vec![0f32; experts];
        let mut order: Vec<u16> = Vec::with_capacity(experts);
        for d in 1..=2 {
            for t in d..layers {
                let l = t - d;
                let base = (d - 1) * layers + l;
                for &p in positions {
                    score.fill(0.0);
                    let src = &routing[(l * tokens + p) * top_k..][..top_k];
                    for &e in src {
                        let row = &probs[(base * experts + e as usize) * experts..]
                            [..experts];
                        for (s, &q) in score.iter_mut().zip(row) {
                            *s += q;
                        }
                    }
                    order.clear();
                    order.extend(0..experts as u16);
                    let key =
                        |&f: &u16| (std::cmp::Reverse(score[f as usize].to_bits()), f);
                    order.select_nth_unstable_by_key(width - 1, key);
                    order.truncate(width);
                    order.sort_unstable_by_key(key);
                    let at = (((d - 1) * layers + t) * tokens + p) * width;
                    out[at..at + width].copy_from_slice(&order);
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/cache_replay.rs"]
mod tests;
