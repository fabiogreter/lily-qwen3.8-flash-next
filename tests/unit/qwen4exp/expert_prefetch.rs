//! The prefetch from a prediction file, host side: the file's layout, the
//! plan's checks, and the schedule the service thread runs, driven as the
//! service thread drives it with reads that land when a test says so (or
//! at once), against `cache_replay`'s replay of the same plan.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::qwen4exp::cache_replay::{
    CacheSim, Predictions, ReplayConfig, ReplayCounts, TraceRequest,
};
use crate::qwen4exp::expert_cache::{Resolution, look_up};
use crate::qwen4exp::expert_store::{LiveUsage, UsageRanking};

const LAYERS: usize = 4;
const EXPERTS: usize = 16;
const NONE: u16 = NO_PREDICTION;

/// Every layer ranks low expert ids first.
fn ranking() -> UsageRanking {
    UsageRanking {
        layers: LAYERS,
        experts: EXPERTS,
        counts: vec![(0..EXPERTS as u64).rev().map(|c| c + 1).collect(); LAYERS],
    }
}

/// 40 slots for 64 slices, half of them LRU (the loader's floor for 16
/// experts): experts 0 to 4 pinned in every layer, 5 to 9 in the LRU region,
/// 10 to 15 cold.
fn policy() -> SlotPolicy {
    let share = SlotPolicy::lru_share_for(0.1, EXPERTS, 40);
    SlotPolicy::new(40, LAYERS, EXPERTS, &ranking(), share).expect("policy")
}

/// A file of `planes` planes over `positions` positions, width 4, every row
/// empty but those `rows` sets: `(plane, layer, pos, picks)`.
fn file(
    planes: usize,
    positions: usize,
    rows: &[(usize, usize, usize, &[u16])],
) -> PredictionFile {
    let width = 4;
    let mut ids = vec![NONE; planes * LAYERS * positions * width];
    for &(plane, layer, pos, picks) in rows {
        let at = ((plane * LAYERS + layer) * positions + pos) * width;
        ids[at..at + picks.len()].copy_from_slice(picks);
    }
    PredictionFile::new(ids, planes, LAYERS, width).expect("file")
}

fn setup(spec: &str, depth: usize, f: PredictionFile) -> PrefetchSetup {
    let plan = PrefetchReplay::parse(spec).expect("spec").expect("a plan");
    PrefetchSetup::new(plan, depth, f, LAYERS, EXPERTS).expect("setup")
}

/// A launcher that only records the reads started, with their lanes; the
/// test lands them.
#[allow(clippy::type_complexity)]
fn recording() -> (Launch, Arc<Mutex<Vec<(Read, usize)>>>) {
    let started = Arc::new(Mutex::new(Vec::new()));
    let log = started.clone();
    let launch: Launch =
        Box::new(move |_, read, lane| log.lock().unwrap().push((read, lane)));
    (launch, started)
}

/// A launcher whose reads land at once.
fn instant() -> Launch {
    Box::new(|reads, read, _| reads.landed(read, Duration::ZERO))
}

fn started(log: &Arc<Mutex<Vec<(Read, usize)>>>) -> Vec<(usize, u32)> {
    log.lock().unwrap().iter().map(|(r, _)| (r.layer, r.expert)).collect()
}

/// The token start of the pass at `pos` (one row of top-10), stamped `tick`,
/// sparing slots used at `busy_from`; the table writes are dropped.
fn start(
    pf: &mut Prefetcher,
    policy: &mut SlotPolicy,
    pos: Option<usize>,
    tick: u64,
    busy_from: u64,
) {
    pf.taken(0, pos, 10, policy, tick, busy_from, &mut |_, _, _| {});
}

fn queued(pf: &Prefetcher) -> Vec<(usize, u32)> {
    pf.reads.lock().queue.iter().map(|r| (r.layer, r.expert)).collect()
}

/// Lands the read of `layer`'s `expert` that is in flight.
fn land(pf: &Prefetcher, layer: usize, expert: u32) {
    let (read, _) = *pf
        .reads
        .lock()
        .in_flight
        .iter()
        .find(|(r, _)| r.layer == layer && r.expert == expert)
        .expect("in flight");
    pf.reads.landed(read, Duration::from_micros(500));
}

/// Lands every read the test left in flight and drops what is queued, so
/// the prefetcher's drop (which waits for reads in flight) returns.
fn finish(pf: Prefetcher) {
    pf.reads.lock().queue.clear();
    loop {
        let Some((read, _)) = pf.reads.lock().in_flight.first().copied() else {
            break;
        };
        pf.reads.landed(read, Duration::ZERO);
    }
    drop(pf);
}

/// The service thread's side of one request, as `expert_cache::serve` runs
/// it (the lookups through the same function), the reads of misses aside:
/// the policy's misses, the demand reads (prefetches still queued
/// included) and the prefetches the layer waited for. `table` gets every
/// entry written: `NONE` for evicted experts, slots for published reads.
struct Served {
    missed: Vec<u32>,
    demand: Vec<u32>,
    awaited: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn request(
    pf: &mut Prefetcher,
    policy: &mut SlotPolicy,
    usage: &mut LiveUsage,
    layer: usize,
    pos: Option<usize>,
    routed: &[u32],
    seq: u64,
    last_ready: &mut u64,
    table: &mut Vec<(usize, usize, u32)>,
) -> Served {
    request_sparing(
        pf,
        policy,
        usage,
        (layer, pos, routed, seq),
        last_ready,
        table,
        true,
    )
}

/// [`request`]; with `spare_busy` false the prefetches spare only the slots
/// the next request stamps, as the replay's do, not the last signalled
/// request's too.
#[allow(clippy::too_many_arguments)]
fn request_sparing(
    pf: &mut Prefetcher,
    policy: &mut SlotPolicy,
    usage: &mut LiveUsage,
    (layer, pos, routed, seq): (usize, Option<usize>, &[u32], u64),
    last_ready: &mut u64,
    table: &mut Vec<(usize, usize, u32)>,
    spare_busy: bool,
) -> Served {
    let mut set = |l: usize, e: usize, s: u32| table.push((l, e, s));
    let decode = routed.len() <= crate::qwen4exp::expert_cache::DECODE_IDS;
    let pos = pos.filter(|_| decode);
    let busy = if spare_busy { *last_ready } else { seq };
    pf.taken(layer, pos, routed.len(), policy, seq, busy, &mut set);
    let mut ids = routed.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let mut res = Resolution::default();
    let missed = look_up(
        policy,
        usage,
        Some(&mut *pf),
        (layer, seq, decode),
        &ids,
        &mut res,
        &mut set,
    )
    .expect("a slot for every miss");
    let missed_ids = res.misses[..missed].iter().map(|m| m.2 as u32).collect();
    let awaited_ids = res.awaited.iter().map(|r| r.expert).collect();
    pf.wait(&res.awaited);
    pf.publish(policy, &mut set);
    *last_ready = seq;
    let busy = if spare_busy { seq } else { seq + 1 };
    pf.resolved(layer, routed.len(), policy, (seq + 1, busy), &mut set);
    Served {
        missed: missed_ids,
        demand: res.misses.iter().map(|m| m.2 as u32).collect(),
        awaited: awaited_ids,
    }
}

fn usage() -> LiveUsage {
    LiveUsage::new(&ranking(), 40, 2)
}

#[test]
fn the_file_reads_rows_by_plane_layer_and_position() {
    let f = file(2, 3, &[(0, 1, 2, &[12, 13, 14, 15]), (1, 3, 0, &[11, NONE, 9])]);
    assert_eq!(f.positions(), 3);
    assert_eq!(f.picks(0, 1, 2, 20), &[12, 13, 14, 15]);
    assert_eq!(f.picks(0, 1, 2, 2), &[12, 13], "top 2 of the row");
    assert_eq!(f.picks(1, 3, 0, 20), &[11], "a row ends at its first NONE");
    assert!(f.picks(0, 1, 1, 20).is_empty());
    assert!(f.picks(0, 1, 3, 20).is_empty(), "past the file's positions");
    // Not a whole number of positions.
    assert!(PredictionFile::new(vec![0; 2 * LAYERS * 4 + 1], 2, LAYERS, 4).is_err());
    assert!(PredictionFile::new(Vec::new(), 2, LAYERS, 4).is_err());
}

#[test]
fn the_file_reads_little_endian_u16s() {
    let dir =
        std::env::temp_dir().join(format!("lily-prefetch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("p.pred.u16");
    let ids: Vec<u16> = (0..LAYERS as u16 * 2).map(|i| 0x0100 + i).collect();
    std::fs::write(&path, ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let f = PredictionFile::open(&path, 1, LAYERS, 2).expect("open");
    assert_eq!(f.positions(), 1);
    assert_eq!(f.picks(0, 3, 0, 2), &[0x0106, 0x0107]);
    assert!(PredictionFile::open(&path, 1, LAYERS, 3).is_err(), "a wrong width");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_plan_must_start_at_the_token_and_fit_the_file() {
    let plan = |s: &str| PrefetchReplay::parse(s).unwrap().unwrap();
    let p = plan("0:20@1+2r");
    assert_eq!((p.lookahead, p.top, p.cuts, p.replace), (0, 20, 0b110, true));
    assert!(p.is_cut(1) && p.is_cut(2) && !p.is_cut(0) && !p.is_cut(3));
    assert_eq!((p.cut_plane(1), p.cut_plane(2)), (1, 2));
    assert!(PrefetchReplay::parse("off").unwrap().is_none());
    let ok = |s: &str, planes: usize| {
        PrefetchSetup::new(plan(s), 1, file(planes, 1, &[]), LAYERS, EXPERTS).is_ok()
    };
    assert!(ok("0:20@1+2r", 3));
    assert!(ok("0:20", 3), "more planes than read are fine");
    assert!(!ok("0:20@1+2r", 2), "a plane per cut");
    assert!(!ok("1:20", 3), "a lookahead");
    assert!(!ok("0:20:4", 3), "a budget");
    assert!(!ok("0:20@3", 3), "a cut after the last layer");
    let bad = file(1, 1, &[(0, 0, 0, &[EXPERTS as u16])]);
    assert!(PrefetchSetup::new(plan("0:4"), 1, bad, LAYERS, EXPERTS).is_err());
    assert!(
        PrefetchSetup::new(plan("0:4"), 0, file(1, 1, &[]), LAYERS, EXPERTS).is_err()
    );
}

#[test]
fn the_token_start_reads_every_layer_in_order_one_at_a_time() {
    // Position 0: layer 0 predicts 11, 3 (resident), 12; layer 2 predicts 13.
    let f = file(1, 2, &[(0, 0, 0, &[11, 3, 12]), (0, 2, 0, &[13, 14])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    // The resident pick is skipped; every other one has a slot already.
    for (l, e) in [(0, 11), (0, 12), (2, 13), (2, 14)] {
        assert!(policy.slot_of(l, e).is_some(), "layer {l} expert {e} reserved");
    }
    assert_eq!(started(&log), vec![(0, 11)], "depth 1");
    assert_eq!(queued(&pf), vec![(0, 12), (2, 13), (2, 14)]);
    assert_eq!(pf.counts().reads(), 4);
    assert_eq!(pf.counts().tokens, 1);
    // Landing starts the next; the table is untouched until published.
    land(&pf, 0, 11);
    assert_eq!(started(&log), vec![(0, 11), (0, 12)]);
    let mut table = Vec::new();
    pf.publish(&policy, |l, e, s| table.push((l, e, s)));
    assert_eq!(table, vec![(0, 11, policy.slot_of(0, 11).unwrap())]);
    table.clear();
    pf.publish(&policy, |l, e, s| table.push((l, e, s)));
    assert!(table.is_empty(), "published once");
    finish(pf);
}

#[test]
fn a_layer_reads_its_queued_prefetches_waits_for_one_in_flight_and_cancels_the_rest() {
    // Layer 0 predicts 11, 12, 13, 14; the token routes 11 and 13.
    let f = file(1, 1, &[(0, 0, 0, &[11, 12, 13, 14])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    assert_eq!(started(&log), vec![(0, 11)]);
    let slot13 = policy.slot_of(0, 13).unwrap();
    // 11 is in flight: land it from another thread while the layer waits.
    let reads = pf.reads.clone();
    let lander = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        let (read, _) = reads.lock().in_flight[0];
        reads.landed(read, Duration::from_millis(20));
    });
    let ids = [11, 13];
    pf.before_lookups(0, &ids, &policy);
    for &e in &ids {
        assert!(matches!(policy.lookup(0, e as usize, 10), Lookup::Hit(_)));
    }
    let (mut misses, mut awaited) = (Vec::new(), Vec::new());
    pf.after_lookups(0, &ids, &mut misses, &[], &mut policy, &mut awaited);
    // 13 was queued: read on demand into the slot it took.
    assert_eq!(misses, vec![(slot13, 0, 13)]);
    assert_eq!(awaited.iter().map(|r| r.expert).collect::<Vec<_>>(), vec![11]);
    // 12 and 14 were queued and not routed: cancelled, their slots emptied.
    assert_eq!(queued(&pf), vec![]);
    assert!(policy.slot_of(0, 12).is_none() && policy.slot_of(0, 14).is_none());
    pf.wait(&awaited);
    lander.join().unwrap();
    let mut table = Vec::new();
    pf.publish(&policy, |l, e, s| table.push((l, e, s)));
    assert_eq!(table, vec![(0, 11, policy.slot_of(0, 11).unwrap())]);
    let c = pf.counts();
    assert_eq!((c.reads(), c.useful, c.late, c.cancelled), (2, 2, 2, 2));
    assert_eq!((c.landed, c.read_us), (1, 20_000));
    finish(pf);
}

#[test]
fn a_cut_replaces_the_queued_reads_it_does_not_pick() {
    // Plane 0 predicts 11 and 12 for layers 2 and 3; the cut after layer 1
    // keeps 12 for layer 2 and picks 13 and 14 for layer 3.
    let f = file(
        2,
        1,
        &[
            (0, 2, 0, &[11, 12]),
            (0, 3, 0, &[11, 12]),
            (1, 2, 0, &[12]),
            (1, 3, 0, &[13, 14]),
        ],
    );
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20@1r", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    let (mut last, mut table, mut usage) = (9, Vec::new(), usage());
    let u = &mut usage;
    request(&mut pf, &mut policy, u, 0, Some(0), &[0, 1], 10, &mut last, &mut table);
    assert_eq!(started(&log), vec![(2, 11)]);
    assert_eq!(queued(&pf), vec![(2, 12), (3, 11), (3, 12)]);
    request(&mut pf, &mut policy, u, 1, Some(0), &[0, 1], 11, &mut last, &mut table);
    // (2, 11) is in flight and stays; (2, 12) is picked again and stays
    // queued; layer 3's two are replaced by the cut's.
    assert_eq!(queued(&pf), vec![(2, 12), (3, 13), (3, 14)]);
    assert!(policy.slot_of(3, 11).is_none() && policy.slot_of(3, 12).is_none());
    let c = pf.counts();
    assert_eq!((c.reads(), c.cancelled), (4, 2));
    finish(pf);
}

/// Stamps every resident LRU slot but those of `except` at `tick`, so the
/// slots of `except` (stamped earlier) are the next victims.
fn age_all_but(policy: &mut SlotPolicy, except: &[(usize, usize)], tick: u64) {
    let lru: Vec<(usize, usize)> = (0..policy.n_slots() as u32)
        .filter(|&s| !policy.is_pinned(s))
        .filter_map(|s| policy.slice_in(s))
        .collect();
    for (l, e) in lru {
        if !except.contains(&(l, e)) {
            policy.lookup(l, e, tick);
        }
    }
}

/// `layer`'s lookups of `ids` at `seq` through the service thread's own
/// function, the table writes into `table`.
fn look_up_layer(
    pf: &mut Prefetcher,
    policy: &mut SlotPolicy,
    (layer, ids, seq): (usize, &[u32], u64),
    table: &mut Vec<(usize, usize, u32)>,
) -> Resolution {
    let mut res = Resolution::default();
    let mut set = |l: usize, e: usize, s: u32| table.push((l, e, s));
    look_up(
        policy,
        &mut usage(),
        Some(pf),
        (layer, seq, true),
        ids,
        &mut res,
        &mut set,
    )
    .expect("a slot");
    res
}

#[test]
fn a_lookup_that_takes_a_queued_prefetchs_slot_cancels_it() {
    let f = file(1, 1, &[(0, 3, 0, &[11, 12])]);
    let (launch, _log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    assert_eq!(queued(&pf), vec![(3, 12)]);
    let slot = policy.slot_of(3, 12).unwrap();
    // (3, 12)'s slot is the LRU's oldest when layer 1 misses expert 15.
    age_all_but(&mut policy, &[(3, 12)], 11);
    let mut table = Vec::new();
    let res = look_up_layer(&mut pf, &mut policy, (1, &[15], 12), &mut table);
    assert_eq!(res.misses, vec![(slot, 1, 15)], "the demand read only");
    assert_eq!(res.evicted, vec![(3, 12)]);
    assert_eq!(table, vec![(3, 12, SlotPolicy::NONE)], "the entry cleared");
    assert!(queued(&pf).is_empty());
    let c = pf.counts();
    assert_eq!((c.reads(), c.cancelled), (1, 1));
    finish(pf);
}

#[test]
fn a_prefill_request_cancels_what_is_queued() {
    let f = file(1, 1, &[(0, 2, 0, &[11, 12, 13])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    start(&mut pf, &mut policy, None, 11, 10);
    assert!(queued(&pf).is_empty());
    assert_eq!(started(&log), vec![(2, 11)], "the read in flight stays");
    assert!(policy.slot_of(2, 11).is_some());
    assert!(policy.slot_of(2, 12).is_none() && policy.slot_of(2, 13).is_none());
    // Past the file's positions nothing is read.
    start(&mut pf, &mut policy, Some(5), 12, 11);
    assert!(queued(&pf).is_empty());
    assert_eq!(pf.counts().tokens, 1);
    finish(pf);
}

#[test]
fn a_prefetch_spares_the_slots_a_pass_may_still_read() {
    // Every LRU slot used, the first ten at tick 8 and the others at 9.
    let mut p = policy();
    let lru: Vec<(usize, usize)> =
        (0..40).filter(|&s| !p.is_pinned(s)).filter_map(|s| p.slice_in(s)).collect();
    assert_eq!(lru.len(), 20);
    for (i, &(l, e)) in lru.iter().enumerate() {
        assert!(matches!(p.lookup(l, e, if i < 10 { 8 } else { 9 }), Lookup::Hit(_)));
    }
    // With a pass possibly still reading tick 9's slots, a prefetch takes
    // one of tick 8's.
    match p.reserve(3, 15, 10, 9) {
        Lookup::Miss { evicted: Some(v), .. } => assert!(lru[..10].contains(&v)),
        other => panic!("{other:?}"),
    }
    // Once every LRU slot was used at the busy tick, none is free to it,
    // though a demand lookup would still find one.
    let mut q = policy();
    for &(l, e) in &lru {
        q.lookup(l, e, 9);
    }
    assert_eq!(q.reserve(3, 15, 10, 9), Lookup::Full);
    assert!(matches!(q.clone().lookup(3, 15, 10), Lookup::Miss { .. }));
    assert!(matches!(q.reserve(3, 15, 10, 10), Lookup::Miss { .. }));
}

#[test]
fn dropping_waits_for_the_reads_in_flight() {
    let f = file(1, 1, &[(0, 2, 0, &[11, 12])]);
    let (launch, _log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    let reads = pf.reads.clone();
    let landed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = landed.clone();
    let lander = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        let (read, _) = reads.lock().in_flight[0];
        reads.landed(read, Duration::ZERO);
    });
    drop(pf);
    assert!(landed.load(std::sync::atomic::Ordering::SeqCst));
    lander.join().unwrap();
}

/// A pseudo-random `[layers][tokens][2]` routing with distinct pairs.
fn synthetic_routing(tokens: usize) -> Vec<u32> {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut r = Vec::new();
    for _ in 0..LAYERS * tokens {
        // Skewed to low ids, as a ranking would see it.
        let a = (next() % 16).min(next() % 16) as u32;
        let mut b = (next() % 16) as u32;
        if b == a {
            b = (a + 1) % 16;
        }
        r.extend_from_slice(&[a, b]);
    }
    r
}

/// Predictions for every position: plane 0 the routing two positions back
/// plus a fixed guess, plane 1 the routing itself, best first.
fn synthetic_predictions(routing: &[u32], tokens: usize) -> Vec<u16> {
    let width = 4;
    let mut ids = vec![NONE; 2 * LAYERS * tokens * width];
    for l in 0..LAYERS {
        for p in 0..tokens {
            let at = |plane: usize| ((plane * LAYERS + l) * tokens + p) * width;
            let back = p.saturating_sub(2);
            let r = &routing[(l * tokens + back) * 2..][..2];
            let row = [r[0] as u16, r[1] as u16, 13, 14];
            ids[at(0)..at(0) + 4].copy_from_slice(&row);
            let r = &routing[(l * tokens + p) * 2..][..2];
            ids[at(1)..at(1) + 2].copy_from_slice(&[r[0] as u16, r[1] as u16]);
        }
    }
    ids
}

/// Replays `spec` over the synthetic session with `ids` (`planes` x width
/// `width` predictions) untimed, serves the same requests with reads that
/// land at once (sparing the busy slots or not, [`request_sparing`]), and
/// with `exact` checks the two count alike: both sides' counts.
fn against_the_replay(
    spec: &str,
    routing: &[u32],
    ids: &[u16],
    (planes, width): (usize, usize),
    (spare_busy, exact): (bool, bool),
) -> (ReplayCounts, PrefetchCounts) {
    let (prefill, tokens) = (6, routing.len() / (LAYERS * 2));
    // The replay: reads land at once without a timing.
    let plan = PrefetchReplay::parse(spec).unwrap().unwrap();
    let config = ReplayConfig {
        slots: 40,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(plan),
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&ranking(), 2, config).expect("sim");
    let mut want = ReplayCounts::default();
    let pred = Predictions { ids, planes, width };
    let req = [TraceRequest { prefill: 0..prefill, decode: prefill..tokens }];
    sim.session(routing, Some(&pred), tokens, 2, &req, &mut want).expect("replay");
    // The service thread's schedule over the same requests.
    let f = PredictionFile::new(ids.to_vec(), planes, LAYERS, width).unwrap();
    let mut pf = Prefetcher::new(setup(spec, 1, f), LAYERS, EXPERTS, instant());
    let mut policy = policy();
    let (mut seq, mut last, mut table, mut usage) = (0u64, 0u64, Vec::new(), usage());
    let (mut decode_misses, mut prefill_misses) = (0usize, 0usize);
    for layer in 0..LAYERS {
        seq += 1;
        let pass = &routing[(layer * tokens) * 2..(layer * tokens + prefill) * 2];
        let at = (layer, None, pass, seq);
        let s = request_sparing(
            &mut pf,
            &mut policy,
            &mut usage,
            at,
            &mut last,
            &mut table,
            spare_busy,
        );
        prefill_misses += s.missed.len();
    }
    for pos in prefill..tokens {
        for layer in 0..LAYERS {
            seq += 1;
            let routed = &routing[(layer * tokens + pos) * 2..][..2];
            let at = (layer, Some(pos), routed, seq);
            let s = request_sparing(
                &mut pf,
                &mut policy,
                &mut usage,
                at,
                &mut last,
                &mut table,
                spare_busy,
            );
            assert!(
                s.awaited.is_empty() && s.demand == s.missed,
                "reads landed at once"
            );
            decode_misses += s.missed.len();
            for &e in routed {
                assert!(policy.slot_of(layer, e as usize).is_some());
            }
        }
    }
    let c = pf.counts();
    finish(pf);
    if !exact {
        return (want, c);
    }
    // A prefill pass of 12 ids is decode-class to the replay, as to the
    // service.
    assert_eq!(want.prefill_lookups, 0);
    assert_eq!(want.decode_misses as usize, prefill_misses + decode_misses);
    assert_eq!(
        (want.prefetch_reads, want.prefetch_useful, want.prefetch_full),
        (c.reads(), c.useful, c.full)
    );
    assert_eq!((c.late, c.cancelled, c.tokens), (0, 0, (tokens - prefill) as u64));
    // Every read was published once, into the slot its expert holds or held;
    // every other entry written cleared an evicted expert.
    let published = table.iter().filter(|&&(_, _, s)| s != SlotPolicy::NONE).count();
    assert_eq!(published as u64, c.reads());
    (want, c)
}

#[test]
fn reads_that_land_at_once_count_as_the_untimed_replay_does() {
    let routing = synthetic_routing(40);
    let ids = synthetic_predictions(&routing, 40);
    let (_, c) = against_the_replay("0:3@1", &routing, &ids, (2, 4), (true, true));
    assert!(c.reads() > 0 && c.useful > 0, "the predictions read something");
    assert_eq!(c.full, 0);
}

#[test]
fn the_replay_caps_a_plane_where_the_server_does() {
    // Every position's plane 0 picks every cold expert of every layer: 24
    // picks against 20 LRU slots less a pass's 2 routed ids.
    let routing = synthetic_routing(40);
    let mut ids = Vec::new();
    for _ in 0..LAYERS * 40 {
        ids.extend_from_slice(&[10, 11, 12, 13, 14, 15]);
    }
    // Sparing what the replay spares, every count agrees.
    let (_, c) = against_the_replay("0:6", &routing, &ids, (1, 6), (false, true));
    assert!(c.full > 0 && c.reads() > 0, "{c:?}");
    // Sparing the busy slots too, the server can take other victims once
    // the region runs out (a miss more or less here), but caps alike.
    let (want, c) = against_the_replay("0:6", &routing, &ids, (1, 6), (true, false));
    assert_eq!(want.prefetch_full, c.full);
    assert!(c.full > 0);
}

/// Plane 0 picking every cold expert (10 to 15) of every layer: 24 picks
/// for 20 LRU slots.
fn every_cold_expert() -> PredictionFile {
    let width = 6;
    let mut ids = vec![NONE; LAYERS * width];
    for l in 0..LAYERS {
        ids[l * width..(l + 1) * width].copy_from_slice(&[10, 11, 12, 13, 14, 15]);
    }
    PredictionFile::new(ids, 1, LAYERS, width).unwrap()
}

#[test]
fn a_plane_leaves_its_request_a_slot_for_every_routed_id() {
    let (launch, _log) = recording();
    let mut pf =
        Prefetcher::new(setup("0:6", 1, every_cold_expert()), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    let mut cleared = Vec::new();
    // A pass of one row of top-10: 20 LRU slots less 10 for its misses.
    pf.taken(0, Some(0), 10, &mut policy, 10, 9, &mut |l, e, s| {
        cleared.push((l, e, s))
    });
    let c = pf.counts();
    assert_eq!(c.issued, 10);
    assert_eq!(c.full, 3, "layer 1's fifth pick, then layers 2 and 3");
    // Every reservation evicted an LRU expert, whose entry was cleared.
    assert_eq!(cleared.len(), 10);
    assert!(
        cleared.iter().all(|&(l, e, s)| {
            s == SlotPolicy::NONE && policy.slot_of(l, e).is_none()
        })
    );
    // The request at that tick still finds a slot for each of ten misses.
    for e in 0..10 {
        let layer = 2 + e / 5;
        let expert = 10 + e % 5;
        assert!(
            matches!(policy.lookup(layer, expert, 10), Lookup::Miss { .. }),
            "miss {e} found no slot"
        );
    }
    finish(pf);
}

#[test]
fn a_demand_miss_that_takes_a_prefetch_in_flight_waits_for_it() {
    let f = file(1, 1, &[(0, 3, 0, &[11])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    assert_eq!(started(&log), vec![(3, 11)]);
    let slot = policy.slot_of(3, 11).unwrap();
    // (3, 11)'s slot is the LRU's oldest when layer 1 misses expert 15.
    age_all_but(&mut policy, &[(3, 11)], 11);
    let reads = pf.reads.clone();
    let landed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = landed.clone();
    let lander = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        let (read, _) = reads.lock().in_flight[0];
        reads.landed(read, Duration::ZERO);
    });
    let mut table = Vec::new();
    let res = look_up_layer(&mut pf, &mut policy, (1, &[15], 12), &mut table);
    assert!(
        landed.load(std::sync::atomic::Ordering::SeqCst),
        "the demand read waits for the prefetch writing its slot"
    );
    lander.join().unwrap();
    assert_eq!(res.misses, vec![(slot, 1, 15)]);
    assert!(res.awaited.is_empty());
    assert_eq!(table, vec![(3, 11, SlotPolicy::NONE)], "the entry cleared");
    // The landed read is not published: its slot is the demand miss's now.
    table.clear();
    pf.publish(&policy, |l, e, s| table.push((l, e, s)));
    assert!(table.is_empty());
    finish(pf);
}

#[test]
fn a_cut_that_takes_plane_zeros_slots_cancels_or_waits_for_them() {
    // Plane 0 reserves three slots for layer 3; the cut after layer 1 needs
    // three for layer 2, and only those three are old enough to take.
    let f = file(2, 1, &[(0, 3, 0, &[11, 12, 13]), (1, 2, 0, &[11, 12, 13])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20@1", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    // Every LRU slot used at tick 8.
    let lru: Vec<(usize, usize)> = (0..40)
        .filter(|&s| !policy.is_pinned(s))
        .filter_map(|s| policy.slice_in(s))
        .collect();
    for &(l, e) in &lru {
        policy.lookup(l, e, 8);
    }
    start(&mut pf, &mut policy, Some(0), 10, 9);
    assert_eq!(started(&log), vec![(3, 11)]);
    assert_eq!(queued(&pf), vec![(3, 12), (3, 13)]);
    // The other LRU slots used at tick 11.
    for &(l, e) in &lru {
        if policy.slot_of(l, e).is_some() {
            policy.lookup(l, e, 11);
        }
    }
    // Lands whatever is in flight every few milliseconds: a read the cut
    // waits for lands, and the queued one it starts next can land too.
    let (reads, stop) =
        (pf.reads.clone(), Arc::new(std::sync::atomic::AtomicBool::new(false)));
    let halt = stop.clone();
    let lander = std::thread::spawn(move || {
        while !halt.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
            let first = reads.lock().in_flight.first().copied();
            if let Some((read, _)) = first {
                reads.landed(read, Duration::ZERO);
            }
        }
    });
    let mut cleared = Vec::new();
    pf.resolved(1, 10, &mut policy, (12, 11), &mut |l, e, s| cleared.push((l, e, s)));
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    lander.join().unwrap();
    // Plane 0's reads lost their slots to the cut's: cancelled while queued,
    // waited for while in flight (which ones depends on when the lander
    // starts the next).
    let c = pf.counts();
    assert_eq!((c.issued, c.full), (6, 0));
    assert!(c.cancelled <= 2, "{} cancelled", c.cancelled);
    assert!(queued(&pf).iter().all(|&(l, _)| l == 2));
    for e in 11..14 {
        assert!(policy.slot_of(3, e).is_none() && policy.slot_of(2, e).is_some());
    }
    assert_eq!(cleared.len(), 3);
    assert!(cleared.iter().all(|&(l, _, s)| l == 3 && s == SlotPolicy::NONE));
    let mut table = Vec::new();
    pf.publish(&policy, |l, e, s| table.push((l, e, s)));
    assert!(
        table.iter().all(|&(l, _, _)| l == 2),
        "plane 0's landed reads lost their slots: {table:?}"
    );
    finish(pf);
}

#[test]
fn promotion_leaves_a_reservation_alone_until_it_lands() {
    let f = file(1, 1, &[(0, 2, 0, &[11, 12])]);
    let (launch, _log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 1, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    assert_eq!(queued(&pf), vec![(2, 12)]);
    let (pinned, slot) = (policy.pinned(), policy.slot_of(2, 12).unwrap());
    // (2, 12) has out-earned every pinned expert.
    let mut usage = usage();
    for _ in 0..100_000 {
        usage.record(2, 12, true);
    }
    // Plain promotion would pin the reservation, and its cancellation would
    // then empty a pinned slot for good.
    let mut p = policy.clone();
    assert_eq!(usage.clone().promote(&mut p, 10), 1);
    assert!(p.is_pinned(slot));
    p.forget(2, 12);
    assert_eq!(p.pinned(), pinned - 1);
    // The service thread's promotion leaves it in the LRU region.
    let swaps = usage.promote_except(&mut policy, 10, |l, e| pf.is_outstanding(l, e));
    assert_eq!(swaps, 0);
    assert!(!policy.is_pinned(slot));
    start(&mut pf, &mut policy, None, 11, 10);
    assert!(policy.slot_of(2, 12).is_none());
    assert_eq!(policy.pinned(), pinned, "the pinned set kept its size");
    // Landed and published, it is a candidate like any other.
    land(&pf, 2, 11);
    pf.publish(&policy, |_, _, _| {});
    for _ in 0..100_000 {
        usage.record(2, 11, true);
    }
    let swaps = usage.promote_except(&mut policy, 12, |l, e| pf.is_outstanding(l, e));
    assert_eq!(swaps, 1);
    assert!(policy.is_pinned(policy.slot_of(2, 11).unwrap()));
    finish(pf);
}

#[test]
fn reads_in_flight_each_have_a_lane_of_their_own() {
    let f = file(1, 1, &[(0, 1, 0, &[11, 12, 13])]);
    let (launch, log) = recording();
    let mut pf = Prefetcher::new(setup("0:20", 2, f), LAYERS, EXPERTS, launch);
    let mut policy = policy();
    start(&mut pf, &mut policy, Some(0), 10, 9);
    let lanes = |log: &Arc<Mutex<Vec<(Read, usize)>>>| -> Vec<(u32, usize)> {
        log.lock().unwrap().iter().map(|(r, l)| (r.expert, *l)).collect()
    };
    assert_eq!(lanes(&log), vec![(11, 0), (12, 1)]);
    land(&pf, 1, 11);
    assert_eq!(lanes(&log), vec![(11, 0), (12, 1), (13, 0)], "the lane freed");
    finish(pf);
}
