//! The replay's request stream and its policy calls, on a synthetic session.

use super::*;

/// Two layers of eight experts, top-2; ranking prefers low expert ids.
fn ranking() -> UsageRanking {
    UsageRanking {
        layers: 2,
        experts: 8,
        counts: vec![vec![80, 70, 60, 50, 40, 30, 20, 10]; 2],
    }
}

/// 16 slots for 16 slices: everything resident, so only the stream's shape
/// is under test.
fn config() -> ReplayConfig {
    ReplayConfig {
        slots: 16,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4,
        prefetch: None,
        half_life: None,
        timing: None,
        speculation: None,
    }
}

/// `[layers][tokens][top_k]` with every token of layer `l` routed to
/// `(a, b)`.
fn routing(tokens: usize, pairs: &[(u32, u32)]) -> Vec<u32> {
    let mut r = Vec::new();
    for l in 0..2 {
        for t in 0..tokens {
            let (a, b) = pairs[(l * tokens + t) % pairs.len()];
            r.extend_from_slice(&[a, b]);
        }
    }
    r
}

#[test]
fn passes_follow_the_service_stream() {
    let mut sim = CacheSim::new(&ranking(), 2, config()).expect("sim");
    let mut c = ReplayCounts::default();
    // 10 prefilled tokens in chunks of 4 (3 passes) and 3 decoded tokens,
    // two layers each.
    let req = [TraceRequest { prefill: 0..10, decode: 10..13 }];
    sim.session(&routing(13, &[(0, 1)]), None, 13, 2, &req, &mut c).expect("replay");
    assert_eq!((c.prefill_tokens, c.decode_tokens), (10, 3));
    // A prefill pass of 4 tokens routes 8 ids, at most DECODE_IDS: in the
    // service it counts as decode-class, and so it does here.
    assert_eq!(c.prefill_lookups, 0);
    assert_eq!(
        c.decode_lookups,
        (3 + 3) * 2 * 2,
        "6 passes x 2 layers x 2 distinct ids"
    );
    assert_eq!(c.decode_misses, 0, "the whole model fits");
}

#[test]
fn a_small_slab_misses_and_counts_layers() {
    // 3 layers x 64 experts = 192 slices in 128 slots (64 pinned, 64 LRU):
    // the ranking's worst experts, 62 and 63, are not resident in any layer.
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: None,
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let mut c = ReplayCounts::default();
    // Two decoded tokens routed to (62, 63) in every layer: the first misses
    // both in all three layers, the second finds them in the LRU region.
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63, 62, 63]).collect();
    let req = [TraceRequest { prefill: 0..0, decode: 0..2 }];
    sim.session(&routing, None, 2, 2, &req, &mut c).expect("replay");
    assert_eq!((c.decode_tokens, c.decode_lookups), (2, 12));
    assert_eq!((c.decode_misses, c.decode_layers_missed), (6, 3));
    // A routing log of the wrong size is refused.
    assert!(sim.session(&routing[1..], None, 2, 2, &req, &mut c).is_err());
}

/// One decoded token over three layers of 64 experts in 128 slots, routed
/// to (62, 63) everywhere, which no layer holds; predictions for every
/// layer they cover say `[62, 63, 0, 1]`.
fn prefetched(prefetch: PrefetchReplay) -> ReplayCounts {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(prefetch),
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let mut ids = Vec::new();
    for d in 1..=2 {
        for layer in 0..3 {
            ids.extend_from_slice(if layer < d { &[0u16; 4] } else { &[62, 63, 0, 1] });
        }
    }
    let pred = Predictions { ids: &ids, planes: 2, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    // Without predictions, or with the wrong count, a prefetch is refused.
    assert!(sim.session(&routing, None, 1, 2, &req, &mut c.clone()).is_err());
    let short = Predictions { ids: &ids[1..], planes: 2, width: 4 };
    assert!(sim.session(&routing, Some(&short), 1, 2, &req, &mut c.clone()).is_err());
    c
}

#[test]
fn a_prefetch_reads_the_next_layers_predicted_experts() {
    // Layer 0 has no source and misses both; layers 1 and 2 find theirs
    // read ahead. The resident picks 0 and 1 are not read.
    let c = prefetched(PrefetchReplay {
        lookahead: 1,
        top: 4,
        budget: None,
        cuts: 0,
        replace: false,
    });
    assert_eq!((c.decode_misses, c.decode_layers_missed), (2, 1));
    assert_eq!((c.prefetch_reads, c.prefetch_useful, c.prefetch_full), (4, 4, 0));
    // One read per layer: each later layer misses its second expert.
    let c = prefetched(PrefetchReplay {
        lookahead: 1,
        top: 4,
        budget: Some(1),
        cuts: 0,
        replace: false,
    });
    assert_eq!((c.decode_misses, c.decode_layers_missed), (4, 3));
    assert_eq!((c.prefetch_reads, c.prefetch_useful), (2, 2));
    // Two layers ahead: only layer 2 has a source.
    let c = prefetched(PrefetchReplay {
        lookahead: 2,
        top: 4,
        budget: None,
        cuts: 0,
        replace: false,
    });
    assert_eq!((c.decode_misses, c.decode_layers_missed), (4, 2));
    assert_eq!((c.prefetch_reads, c.prefetch_useful), (2, 2));
}

#[test]
fn transitions_predict_what_followed_and_uncount_exactly() {
    // Three layers of 8 experts, top-1, four positions: expert e at layer
    // 0 is followed by e + 1 at layer 1 and e + 2 at layer 2.
    let (layers, experts, tokens) = (3, 8, 4);
    let routing: Vec<u32> = (0..layers as u32)
        .flat_map(|l| (0..tokens as u32).map(move |p| p + l))
        .collect();
    let mut t = Transitions::new(layers, experts);
    t.count(&routing, tokens, 1, true);
    let pred = t.predict(&routing, tokens, 1, &[2], 2);
    let at = |d: usize, l: usize, p: usize| ((d * layers + l) * tokens + p) * 2;
    // Lookahead 1 for layer 1 at position 2 (routed to 2 at layer 0): 3
    // first, then the lowest id among the zero scores.
    assert_eq!(&pred[at(0, 1, 2)..at(0, 1, 2) + 2], &[3, 0]);
    assert_eq!(&pred[at(0, 2, 2)..at(0, 2, 2) + 2], &[4, 0]);
    assert_eq!(&pred[at(1, 2, 2)..at(1, 2, 2) + 2], &[4, 0]);
    // Rows without a source layer and positions not asked for stay zero.
    assert_eq!(&pred[at(1, 1, 2)..at(1, 1, 2) + 2], &[0, 0]);
    assert_eq!(&pred[at(0, 1, 1)..at(0, 1, 1) + 2], &[0, 0]);
    // Uncounting what was counted leaves nothing to predict from.
    t.count(&routing, tokens, 1, false);
    let empty = t.predict(&routing, tokens, 1, &[2], 2);
    assert_eq!(&empty[at(0, 1, 2)..at(0, 1, 2) + 2], &[0, 1]);
}

#[test]
fn token_routing_predicts_what_a_token_routed_to() {
    // Two layers of 8 experts, top-1; token 7 routes to 3 then 5, token 9
    // to 1 then 2.
    let ids = [7u32, 9, 7, 7];
    let routing: Vec<u32> = vec![3, 1, 3, 4, /* layer 1 */ 5, 2, 5, 5];
    let mut t = TokenRouting::new(2, 8, [7, 9, 11]);
    t.count(&ids, &routing, 1, true);
    let pred = t.predict(&[7, 11], &[0, 1], 2);
    // Position 0 (token 7): layer 0 saw 3 twice and 4 once; layer 1 saw 5.
    assert_eq!(&pred[0..2], &[3, 4]);
    assert_eq!(&pred[4..6], &[5, 0]);
    // Position 1 (token 11, never counted) stays zero.
    assert_eq!(&pred[2..4], &[0, 0]);
    // Uncounting leaves no prediction.
    t.count(&ids, &routing, 1, false);
    assert!(t.predict(&[7], &[0], 2).iter().all(|&e| e == 0));
}

#[test]
fn a_token_start_prefetch_reads_every_layer_before_the_pass() {
    // As `prefetched`, but lookahead 0 from the first plane: all three
    // layers find (62, 63) read ahead, layer 0 included.
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(PrefetchReplay {
            lookahead: 0,
            top: 4,
            budget: None,
            cuts: 0,
            replace: false,
        }),
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let mut ids: Vec<u16> = (0..3).flat_map(|_| [62u16, 63, 0, 1]).collect();
    ids.extend([0u16; 12]);
    let pred = Predictions { ids: &ids, planes: 2, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    assert_eq!((c.decode_misses, c.prefetch_reads, c.prefetch_useful), (0, 6, 6));
}

/// One decoded token over three layers routed to (62, 63), none resident,
/// with the token start predicting `start` for layer 0 only and the cut
/// after layer 0 predicting (62, 63) for layers 1 and 2; rows end at
/// [`NO_PREDICTION`].
fn after_a_cut(start: [u16; 4], timing: Option<SsdTiming>) -> ReplayCounts {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(PrefetchReplay {
            lookahead: 0,
            top: 4,
            budget: None,
            cuts: 1,
            replace: false,
        }),
        half_life: None,
        timing,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let none = [NO_PREDICTION; 4];
    let ids: Vec<u16> =
        [start, none, none, none, [62, 63, NO_PREDICTION, 0], [62, 63, 0, 1]].concat();
    let pred = Predictions { ids: &ids, planes: 2, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    c
}

#[test]
fn a_cut_reads_the_second_plane_for_the_layers_after_it() {
    // Layer 0 from the token start, layers 1 and 2 after the cut; layer 1's
    // row ends after two picks, layer 2's reads all four.
    let c = after_a_cut([62, 63, NO_PREDICTION, 0], None);
    assert_eq!((c.decode_misses, c.prefetch_reads, c.prefetch_useful), (0, 6, 6));
    // An empty token-start row reads nothing: layer 0 misses both.
    let c = after_a_cut([NO_PREDICTION; 4], None);
    assert_eq!((c.decode_misses, c.prefetch_reads), (2, 4));
    // Timed (1 ms layers, 0.25 ms reads): layer 0's reads land by 0.5 ms
    // when it asks; the cut's queue at its end (1.0 ms) and lands by 2.0,
    // in time for layer 1 at 1.5 ms (62 at 1.25, 63 at 1.5) and layer 2.
    let timing = SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 0.25 };
    let c = after_a_cut([62, 63, NO_PREDICTION, 0], Some(timing));
    assert_eq!((c.decode_misses, c.decode_stall_us, c.prefetch_late), (0, 0, 0));
}

#[test]
fn the_miss_log_flags_the_routed_experts_a_decoded_position_missed() {
    // Experts 0 and 1 are pinned everywhere, 62 and 63 never resident: the
    // first decoded position routes (0, 62) and misses 62; the second
    // routes (62, 1) and finds 62 in the LRU region.
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: None,
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim").log_misses();
    // [layers][tokens][top_k], three tokens, the first prefilled.
    let routing: Vec<u32> = (0..3).flat_map(|_| [63, 0, 0, 62, 62, 1]).collect();
    let req = [TraceRequest { prefill: 0..1, decode: 1..3 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, None, 3, 2, &req, &mut c).expect("replay");
    let log = sim.take_miss_log().expect("log");
    // The prefilled position's miss of 63 is not logged, though a one-token
    // prefill counts as a decode-class request, as in the service.
    assert_eq!(log, [0, 0, 0, 1, 0, 0].repeat(3));
    assert_eq!(c.decode_misses, 6);
}

/// One decoded token over four layers routed to (62, 63), none resident,
/// timed with 1 ms layers against `ssd` (serial: 1 ms reads). The token
/// start predicts (50, 51) for every layer (wrong, not resident); cuts
/// after layers 0 and 1 predict (62, 63) from planes 1 and 2 for the layers
/// after them.
fn two_cuts(replace: bool, ssd: SsdModel) -> ReplayCounts {
    let r = UsageRanking {
        layers: 4,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 4],
    };
    let cfg = ReplayConfig {
        slots: 160,
        lru_share: 0.2,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(PrefetchReplay {
            lookahead: 0,
            top: 2,
            budget: None,
            cuts: 0b11,
            replace,
        }),
        half_life: None,
        timing: Some(SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 }),
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim").with_ssd(ssd);
    let routing: Vec<u32> = (0..4).flat_map(|_| [62, 63]).collect();
    let none = [NO_PREDICTION; 2];
    let start: Vec<u16> = (0..4).flat_map(|_| [50u16, 51]).collect();
    let cut0: Vec<u16> = [none, [62, 63], [62, 63], [62, 63]].concat();
    let cut1: Vec<u16> = [none, none, [62, 63], [62, 63]].concat();
    let ids = [start, cut0, cut1].concat();
    let pred = Predictions { ids: &ids, planes: 3, width: 2 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    c
}

#[test]
fn cuts_read_their_own_planes_and_can_replace_queued_reads() {
    // Without replacing, the eight wrong token-start reads queue first and
    // the cuts' reads wait behind them; replacing cancels the wrong ones
    // still queued when the first cut ends, so its reads start sooner.
    let kept = two_cuts(false, SsdModel::Serial);
    let replaced = two_cuts(true, SsdModel::Serial);
    assert_eq!(kept.prefetch_useful, replaced.prefetch_useful);
    assert!(replaced.prefetch_cancelled > kept.prefetch_cancelled, "{replaced:?}");
    assert!(replaced.decode_stall_us < kept.decode_stall_us, "{kept:?} {replaced:?}");
}

#[test]
fn cuts_replace_queued_reads_on_the_shared_ssd() {
    // As above against the shared SSD, two prefetches in flight at most:
    // the cuts' reads queue behind the token start's wrong ones there too,
    // and replacing them drops the wrong ones still queued.
    let ssd = SsdModel::Shared(SharedSsd {
        latency_ms: 0.5,
        transfer_ms: 0.5,
        prefetch_depth: 2,
    });
    let kept = two_cuts(false, ssd);
    let replaced = two_cuts(true, ssd);
    // Every pick is issued either way, read or cancelled unread.
    for c in [&kept, &replaced] {
        assert_eq!(c.prefetch_reads + c.prefetch_cancelled, 14, "{c:?}");
    }
    assert_eq!(kept.prefetch_useful, replaced.prefetch_useful);
    assert!(replaced.prefetch_cancelled > kept.prefetch_cancelled, "{replaced:?}");
    assert!(replaced.decode_stall_us < kept.decode_stall_us, "{kept:?} {replaced:?}");
}

/// Three decoded tokens over three layers, each routed to its own pair of
/// cold experts (token q: 56 + 2q, 57 + 2q), replayed as verify passes of
/// the pending token and two drafts, with `prefetch` from `pred` (a cut
/// after layer 0 predicting every token's own pair for layers 1 and 2).
fn speculated(
    accept: f64,
    rejected: RejectedRows,
    prefetch: Option<PrefetchReplay>,
) -> ReplayCounts {
    speculated_on(accept, rejected, prefetch, None)
}

/// [`speculated`], timed with 1 ms layers against `ssd` when given.
fn speculated_on(
    accept: f64,
    rejected: RejectedRows,
    prefetch: Option<PrefetchReplay>,
    ssd: Option<SharedSsd>,
) -> ReplayCounts {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 160,
        lru_share: 0.2,
        adapt: false,
        prefill_chunk: 4096,
        prefetch,
        half_life: None,
        timing: ssd.map(|_| SsdTiming {
            layer_ms: 1.0,
            mixer_share: 0.5,
            read_ms: 1.0,
        }),
        speculation: Some(SpecReplay { drafts: 2, accept, rejected, seed: 7 }),
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    if let Some(ssd) = ssd {
        sim = sim.with_ssd(SsdModel::Shared(ssd));
    }
    let pair = |q: u32| [56 + 2 * q, 57 + 2 * q];
    // [layers][tokens][top_k]
    let routing: Vec<u32> = (0..3).flat_map(|_| (0..3).flat_map(pair)).collect();
    let none = [NO_PREDICTION; 2];
    let start: Vec<u16> = (0..9).flat_map(|_| none).collect();
    let cut: Vec<u16> = (0..3)
        .flat_map(|l| {
            (0..3u16)
                .flat_map(move |q| if l == 0 { none } else { [56 + 2 * q, 57 + 2 * q] })
        })
        .collect();
    let ids = [start, cut].concat();
    let pred = Predictions { ids: &ids, planes: 2, width: 2 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..3 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 3, 2, &req, &mut c).expect("replay");
    c
}

#[test]
fn verify_passes_resolve_their_rows_together() {
    // Every draft accepted: one pass for all three tokens, each layer
    // missing all six experts at once.
    let c = speculated(1.0, RejectedRows::Next, None);
    assert_eq!((c.decode_tokens, c.decode_passes), (3, 1));
    assert_eq!((c.decode_misses, c.decode_layers_missed), (18, 3));
    // None accepted: three passes; the first already routes the next two
    // tokens' experts (`Next`), so the later passes find them resident.
    let c = speculated(0.0, RejectedRows::Next, None);
    assert_eq!((c.decode_tokens, c.decode_passes), (3, 3));
    assert_eq!(c.decode_misses, 18);
    assert_eq!(c.decode_layers_missed, 3);
    // Rejected rows routing at random miss elsewhere instead.
    let c = speculated(0.0, RejectedRows::Random, None);
    assert_eq!(c.decode_passes, 3);
    assert!(c.decode_layers_missed > 3, "{c:?}");
    // Their experts are no history: every routed pair is new to its layer
    // when the trace's own row asks for it, so no miss counts as recent.
    assert_eq!((c.decode_misses_recent_64, c.decode_misses_recent_1024), (0, 0));
}

#[test]
fn a_verify_pass_takes_one_layer_time_on_the_shared_ssd() {
    // All accepted: one pass, each layer missing all six experts at once,
    // read together (0.1 ms latency, then 0.6 ms of shared transfer); the
    // pass's three rows take one layer time, not three.
    let ssd = SharedSsd { latency_ms: 0.1, transfer_ms: 0.1, prefetch_depth: 2 };
    let c = speculated_on(1.0, RejectedRows::Next, None, Some(ssd));
    assert_eq!((c.decode_tokens, c.decode_passes, c.decode_misses), (3, 1, 18));
    assert_eq!(c.decode_stall_us, 3 * 700);
}

/// `sessions` replays of one session of 64 decoded tokens (two layers,
/// everything resident) through one cache, as verify passes of two drafts
/// accepted at even odds, rejected rows `rejected`: each session's passes.
fn spec_passes(rejected: RejectedRows, sessions: usize, first: u64) -> Vec<u64> {
    let cfg = ReplayConfig {
        speculation: Some(SpecReplay { drafts: 2, accept: 0.5, rejected, seed: 7 }),
        ..config()
    };
    let mut sim =
        CacheSim::new(&ranking(), 2, cfg).expect("sim").starting_at_session(first);
    let req = [TraceRequest { prefill: 0..0, decode: 0..64 }];
    let routing = routing(64, &[(0, 1), (2, 3), (4, 5)]);
    (0..sessions)
        .map(|_| {
            let mut c = ReplayCounts::default();
            sim.session(&routing, None, 64, 2, &req, &mut c).expect("replay");
            assert_eq!(c.decode_tokens, 64);
            c.decode_passes
        })
        .collect()
}

#[test]
fn speculation_draws_per_session_and_pairs_the_rejected_rows() {
    // Random rows draw from their own stream: both kinds of rejected rows
    // replay the same acceptances, session by session.
    let next = spec_passes(RejectedRows::Next, 4, 0);
    assert_eq!(spec_passes(RejectedRows::Random, 4, 0), next);
    // Each session draws its own (64 tokens at even odds: the pass counts
    // of four sessions all equal would be a reused seed).
    assert!(next.iter().any(|&p| p != next[0]), "{next:?}");
    // A cache started at session 2 draws what session 2 of one cache did.
    assert_eq!(spec_passes(RejectedRows::Next, 2, 2), next[2..]);
}

#[test]
fn a_cut_reads_for_every_row_of_a_verify_pass() {
    // All accepted: the cut after layer 0 reads all three rows' pairs for
    // layers 1 and 2, so only layer 0 misses.
    let cut =
        PrefetchReplay { lookahead: 0, top: 2, budget: None, cuts: 1, replace: false };
    let c = speculated(1.0, RejectedRows::Next, Some(cut));
    assert_eq!((c.prefetch_reads, c.prefetch_useful), (12, 12));
    assert_eq!((c.decode_misses, c.decode_layers_missed), (6, 1));
}

#[test]
fn misses_are_classed_by_how_recently_the_layer_used_the_expert() {
    // Experts 62 and 63 are not resident in any layer and were never
    // routed before, so their misses at position 0 are not recent.
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: None,
        half_life: None,
        timing: None,
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    // Two decoded positions routed to (62, 63) in every layer: the first
    // misses all six, the second finds them in the LRU region.
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63, 62, 63]).collect();
    let req = [TraceRequest { prefill: 0..0, decode: 0..2 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, None, 2, 2, &req, &mut c).expect("replay");
    assert_eq!(c.decode_misses, 6);
    assert_eq!((c.decode_misses_recent_64, c.decode_misses_recent_1024), (0, 0));
}

/// One decoded token over three layers of 64 experts, routed to (62, 63)
/// everywhere, which no layer holds, timed: every layer waits for two reads
/// in a row; a prefetch from the token's start has them landed for the
/// later layers but not for layer 0.
fn timed(prefetch: Option<PrefetchReplay>) -> ReplayCounts {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch,
        half_life: None,
        timing: Some(SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 0.25 }),
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let mut ids: Vec<u16> = (0..3).flat_map(|_| [62u16, 63, 0, 1]).collect();
    ids.extend([0u16; 12]);
    let pred = Predictions { ids: &ids, planes: 2, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    c
}

#[test]
fn a_timed_replay_charges_reads_and_credits_landed_prefetches() {
    // Without a prefetch: three layers, two reads of 0.25 ms each.
    let c = timed(None);
    assert_eq!((c.decode_misses, c.decode_stall_us, c.prefetch_late), (6, 1500, 0));
    // From the token's start, the six reads queue in layer order at t = 0:
    // layer 0 asks at 0.5 ms with both of its reads landed (0.25, 0.5),
    // layer 1 at 1.5 ms, layer 2 at 2.5 ms: no stall at all.
    let c = timed(Some(PrefetchReplay {
        lookahead: 0,
        top: 4,
        budget: None,
        cuts: 0,
        replace: false,
    }));
    assert_eq!((c.decode_misses, c.prefetch_reads, c.decode_stall_us), (0, 6, 0));
    // One layer ahead: layer 0 misses both (0.5 ms of stall) and ends at
    // 1.5 ms, when layer 1's prefetch is issued; layer 1 asks at 2 ms, its
    // reads land at 1.75 and 2.0: no stall. Layer 2 likewise.
    let c = timed(Some(PrefetchReplay {
        lookahead: 1,
        top: 4,
        budget: None,
        cuts: 0,
        replace: false,
    }));
    assert_eq!((c.decode_misses, c.decode_stall_us, c.prefetch_late), (2, 500, 0));
}

/// As [`timed`], with 1 ms reads and layers, and every layer's prediction
/// `row`, from the token's start.
fn timed_slow(row: [u16; 4]) -> ReplayCounts {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(PrefetchReplay {
            lookahead: 0,
            top: 4,
            budget: None,
            cuts: 0,
            replace: false,
        }),
        half_life: None,
        timing: Some(SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 }),
        speculation: None,
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let mut ids: Vec<u16> = (0..3).flat_map(|_| row).collect();
    ids.extend([0u16; 12]);
    let pred = Predictions { ids: &ids, planes: 2, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    c
}

#[test]
fn a_timed_replay_waits_for_late_prefetches_and_cancels_stale_ones() {
    // Six reads queued at t = 0. Layer 0 asks at 0.5 ms: its first read
    // lands at 1.0 (late), its second, still queued, becomes a demand read
    // landing at 2.0 (late): 1.5 ms of stall, the layer ends at 2.5. Layer
    // 1 asks at 3.0: one read started at 2.0 has landed, the other becomes
    // a demand read to 4.0. Layer 2 likewise: 1.0 ms each. 3.5 ms, 4 late.
    let c = timed_slow([62, 63, 0, 1]);
    assert_eq!(c.decode_misses, 0);
    assert_eq!((c.prefetch_reads, c.prefetch_cancelled), (6, 0));
    assert_eq!((c.decode_stall_us, c.prefetch_late), (3500, 4));
    // Two wrong picks per layer (50 and 51, not resident) are read too:
    // layer 0's are still queued when it has passed, so they are cancelled,
    // and their slots emptied, rather than read.
    let c = timed_slow([62, 63, 50, 51]);
    assert!(c.prefetch_cancelled >= 2, "{c:?}");
    assert_eq!(c.prefetch_reads + c.prefetch_cancelled, 12);
    assert_eq!(c.decode_misses, 0);
}

/// The shared SSD alone: `k` demand reads asked for together on an idle
/// SSD, as the probe's demand batches were (`SsdModel`): when the last
/// lands, ms.
fn demand_batch(m: SharedSsd, k: u32) -> f64 {
    let timing = SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 };
    let mut c = Clock::new(timing, SsdModel::Shared(m));
    let ids: Vec<u32> = (0..k).collect();
    c.resolve(0, 1.0, &ids, &ids, &mut Vec::new()).0
}

/// A background of prefetches for another layer, `d` in flight
/// (`prefetch_depth`), issued `gap_ms` apart so the queue never runs dry
/// and the first `d` do not start in step; demand batches of `k` every
/// 5 ms (back to back when one takes longer) from 50 ms on, as the probe's
/// contention phase ran: the median batch time, ms, and the background's
/// experts per second with no demand (`k` 0).
fn contended(m: SharedSsd, d: usize, k: u32, gap_ms: f64) -> f64 {
    let timing = SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 };
    let mut c =
        Clock::new(timing, SsdModel::Shared(SharedSsd { prefetch_depth: d, ..m }));
    let total = 40_000;
    c.queue.extend((0..total).map(|i| (1, i, i as f64 * gap_ms)));
    let s = c.shared.as_mut().expect("shared");
    s.run_until(&mut c.queue, 50.0);
    let landed = |c: &Clock| {
        total as usize - c.queue.len() - c.shared.as_ref().map_or(0, |s| s.prefetching)
    };
    if k == 0 {
        let before = landed(&c);
        let s = c.shared.as_mut().expect("shared");
        s.run_until(&mut c.queue, 1050.0);
        return (landed(&c) - before) as f64;
    }
    let ids: Vec<u32> = (0..k).collect();
    let mut t = 50.0;
    let mut times = Vec::new();
    for _ in 0..300 {
        let (stall, _) = c.resolve(0, t, &ids, &ids, &mut Vec::new());
        times.push(stall);
        t = (t + 5.0).max(t + stall);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

#[test]
fn the_shared_ssd_reproduces_the_probe() {
    let m = SharedSsd::PROBED;
    let near = |model: f64, probe: f64, cell: &str| {
        let off = model / probe - 1.0;
        assert!(
            off.abs() <= 0.15,
            "{cell}: model {model:.3} ms, probe {probe} ({off:+.3})"
        );
    };
    // Demand batches on an idle drive (probe p50s).
    for (k, probe) in [(1, 0.508), (2, 0.840), (4, 1.284), (8, 2.169), (32, 7.434)] {
        near(demand_batch(m, k), probe, &format!("k {k}"));
    }
    // One demand read against `d` background reads, and eight against one.
    for (d, k, probe) in
        [(1, 1, 0.62), (4, 1, 1.04), (8, 1, 1.89), (16, 1, 3.59), (1, 8, 2.67)]
    {
        near(contended(m, d, k, 0.15), probe, &format!("d {d} k {k}"));
    }
    // The background alone, experts per second (the model 1 828, 3 257 and
    // 4 609), within 20%: the fit is to demand times, this is a cross-check.
    // At depth 1 the model is 15% slow: one read at a time takes a whole
    // latency plus a transfer (1 / 0.547 ms), where the probe's lane read
    // an expert's nine regions at once.
    for (d, probe) in [(1, 2159.0), (2, 3604.0), (4, 4416.0)] {
        let model = contended(m, d, 0, 0.15);
        let off = model / probe - 1.0;
        assert!(
            off.abs() <= 0.2,
            "d {d} alone: model {model}/s, probe {probe} ({off:+.3})"
        );
    }
    // Where the shape misses (`SharedSsd`): eight demand reads against four
    // or more background reads took the probe much longer than per-read
    // sharing gives. The model stays optimistic there.
    for (d, probe) in [(4, 4.83), (8, 8.16), (16, 15.47)] {
        let model = contended(m, d, 8, 0.15);
        assert!(model < probe, "d {d} k 8: model {model:.3} ms, probe {probe}");
    }
}

#[test]
fn the_shared_ssd_overlaps_reads_and_caps_prefetches() {
    let m = SharedSsd { latency_ms: 1.0, transfer_ms: 1.0, prefetch_depth: 2 };
    let timing = SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 };
    // Two demand reads together: 1 ms latency, then 2 ms of shared transfer.
    let mut c = Clock::new(timing, SsdModel::Shared(m));
    assert_eq!(c.resolve(0, 0.0, &[1, 2], &[1, 2], &mut Vec::new()), (3.0, 0));
    // Three prefetches for layer 1 queued at 0: two start, the third waits
    // for a slot. Layer 1 asks at 2.5 for all three: the started ones land
    // at 3 (both transferring from 1 at half speed), the queued one becomes
    // a demand read at 2.5, transferring alone from 3.5 to 4.5.
    let mut c = Clock::new(timing, SsdModel::Shared(m));
    c.queue.extend([(1, 5, 0.0), (1, 6, 0.0), (1, 7, 0.0)]);
    let (stall, late) = c.resolve(1, 2.5, &[5, 6, 7], &[], &mut Vec::new());
    assert_eq!(late, 3);
    assert!((stall - 2.0).abs() < 1e-9, "{stall}");
    assert!(c.queue.is_empty());
    // One prefetch in flight at most: layer 1 asks at 0.5 for an expert it
    // missed and none of its prefetches. The queued one (6) is cancelled;
    // the started one (5) is not: it transfers alone from 1, the demand
    // read joins it at 1.5, they share to 2.5, when the prefetch lands, and
    // the demand read lands at 3: 2.5 ms of stall where alone it was 2.
    let mut c =
        Clock::new(timing, SsdModel::Shared(SharedSsd { prefetch_depth: 1, ..m }));
    c.queue.extend([(1, 5, 0.0), (1, 6, 0.0)]);
    let mut cancelled = Vec::new();
    assert_eq!(c.resolve(1, 0.5, &[9], &[9], &mut cancelled), (2.5, 0));
    assert_eq!(cancelled, [6]);
}

#[test]
fn a_timed_replay_runs_on_the_shared_ssd() {
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let run = |prefetch: Option<PrefetchReplay>| {
        let cfg = ReplayConfig {
            slots: 128,
            lru_share: 0.1,
            adapt: false,
            prefill_chunk: 4096,
            prefetch,
            half_life: None,
            timing: Some(SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 0.25 }),
            speculation: None,
        };
        let ssd = SharedSsd { latency_ms: 0.1, transfer_ms: 0.1, prefetch_depth: 2 };
        let mut sim =
            CacheSim::new(&r, 2, cfg).expect("sim").with_ssd(SsdModel::Shared(ssd));
        let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
        let mut ids: Vec<u16> = (0..3).flat_map(|_| [62u16, 63, 0, 1]).collect();
        ids.extend([0u16; 12]);
        let pred = Predictions { ids: &ids, planes: 2, width: 4 };
        let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
        let mut c = ReplayCounts::default();
        sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
        c
    };
    // Each layer's two misses read together: 0.1 ms latency, then 0.2 ms of
    // shared transfer (serially, as `timed`, 0.5 ms).
    let c = run(None);
    assert_eq!((c.decode_misses, c.decode_stall_us), (6, 900));
    // From the token's start, two prefetches at a time: layer 0's land at
    // 0.3, layer 1's at 0.6, layer 2's at 0.9, all before their layers ask.
    let c = run(Some(PrefetchReplay {
        lookahead: 0,
        top: 4,
        budget: None,
        cuts: 0,
        replace: false,
    }));
    assert_eq!((c.prefetch_reads, c.decode_stall_us, c.prefetch_late), (6, 0, 0));
}

#[test]
fn the_ssd_is_idle_when_the_next_request_decodes() {
    // Two requests of one decoded token each over three layers on a slow
    // shared SSD (1 ms latency, 2 ms transfers, two prefetches in flight).
    // The first routes the pinned (0, 1) everywhere and prefetches the
    // wrong (50, 51) for every layer from its start: layer 0's two start at
    // once and land at 5 ms (1 ms latency, then 4 ms shared), after the
    // token ends at 3 ms with no stall; the other layers' are cancelled
    // still queued. The second request decodes once they have landed: each
    // layer misses (62, 63), 1 ms latency then 4 ms of shared transfer, 5
    // ms of stall alone (more, had the first request's reads still been in
    // flight).
    let r = UsageRanking {
        layers: 3,
        experts: 64,
        counts: vec![(0..64).rev().collect(); 3],
    };
    let cfg = ReplayConfig {
        slots: 128,
        lru_share: 0.1,
        adapt: false,
        prefill_chunk: 4096,
        prefetch: Some(PrefetchReplay {
            lookahead: 0,
            top: 2,
            budget: None,
            cuts: 0,
            replace: false,
        }),
        half_life: None,
        timing: Some(SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 }),
        speculation: None,
    };
    let ssd = SharedSsd { latency_ms: 1.0, transfer_ms: 2.0, prefetch_depth: 2 };
    let mut sim =
        CacheSim::new(&r, 2, cfg).expect("sim").with_ssd(SsdModel::Shared(ssd));
    let routing: Vec<u32> = (0..3).flat_map(|_| [0, 1, 62, 63]).collect();
    let none = [NO_PREDICTION; 2];
    let ids: Vec<u16> = (0..3).flat_map(|_| [[50u16, 51], none]).flatten().collect();
    let pred = Predictions { ids: &ids, planes: 1, width: 2 };
    let req = [
        TraceRequest { prefill: 0..0, decode: 0..1 },
        TraceRequest { prefill: 1..1, decode: 1..2 },
    ];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 2, 2, &req, &mut c).expect("replay");
    assert_eq!((c.prefetch_reads, c.prefetch_cancelled, c.decode_misses), (2, 4, 6));
    assert_eq!((c.decode_stall_us, c.prefetch_late), (15_000, 0));
}

#[test]
fn a_queued_read_serves_its_expert_over_an_older_one_in_flight() {
    // Expert 5 of layer 1 was prefetched, its slot evicted while the read
    // was in flight, and prefetched again: the old read lands at 3 ms, the
    // new one is still queued behind it. Layer 1 routes 5 at 0.5 ms and
    // finds it resident: the queued read is the one for its slot, so it
    // becomes a demand read (landing at 4 behind the old one) and is not
    // cancelled, which would empty the slot of an expert just routed.
    let timing = SsdTiming { layer_ms: 1.0, mixer_share: 0.5, read_ms: 1.0 };
    let mut c = Clock::new(timing, SsdModel::Serial);
    c.ssd_free = 3.0;
    c.landing.insert((1, 5), 3.0);
    c.queue.push_back((1, 5, 0.0));
    let mut cancelled = Vec::new();
    assert_eq!(c.resolve(1, 0.5, &[5], &[], &mut cancelled), (3.5, 1));
    assert!(cancelled.is_empty() && c.queue.is_empty(), "{cancelled:?}");
    // The shared SSD likewise, one prefetch in flight at most: the queued
    // one becomes a demand read at 0.5 (latency to 1.5), sharing with the
    // old one from there; the old one lands at 2.5, the demand at 3.
    let m = SharedSsd { latency_ms: 1.0, transfer_ms: 1.0, prefetch_depth: 1 };
    let mut c = Clock::new(timing, SsdModel::Shared(m));
    c.queue.extend([(1, 5, 0.0), (1, 5, 0.0)]);
    let mut cancelled = Vec::new();
    assert_eq!(c.resolve(1, 0.5, &[5], &[], &mut cancelled), (2.5, 1));
    assert!(cancelled.is_empty() && c.queue.is_empty(), "{cancelled:?}");
    // Missed (its slot evicted after the prefetch took it), the queued
    // duplicate is cancelled behind the demand read: the replay counts it
    // and keeps the slot (`CacheSim::session` never empties a routed one).
    let mut c = Clock::new(timing, SsdModel::Serial);
    c.ssd_free = 3.0;
    c.queue.push_back((1, 5, 0.0));
    assert_eq!(c.resolve(1, 0.5, &[5], &[5], &mut cancelled), (3.5, 0));
    assert_eq!(cancelled, [5]);
}
