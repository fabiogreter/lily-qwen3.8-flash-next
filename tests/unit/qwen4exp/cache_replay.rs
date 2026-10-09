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
    };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63]).collect();
    let mut ids = Vec::new();
    for d in 1..=2 {
        for layer in 0..3 {
            ids.extend_from_slice(if layer < d { &[0u16; 4] } else { &[62, 63, 0, 1] });
        }
    }
    let pred = Predictions { ids: &ids, width: 4 };
    let req = [TraceRequest { prefill: 0..0, decode: 0..1 }];
    let mut c = ReplayCounts::default();
    sim.session(&routing, Some(&pred), 1, 2, &req, &mut c).expect("replay");
    // Without predictions, or with the wrong count, a prefetch is refused.
    assert!(sim.session(&routing, None, 1, 2, &req, &mut c.clone()).is_err());
    let short = Predictions { ids: &ids[1..], width: 4 };
    assert!(sim.session(&routing, Some(&short), 1, 2, &req, &mut c.clone()).is_err());
    c
}

#[test]
fn a_prefetch_reads_the_next_layers_predicted_experts() {
    // Layer 0 has no source and misses both; layers 1 and 2 find theirs
    // read ahead. The resident picks 0 and 1 are not read.
    let c = prefetched(PrefetchReplay { lookahead: 1, top: 4, budget: None });
    assert_eq!((c.decode_misses, c.decode_layers_missed), (2, 1));
    assert_eq!((c.prefetch_reads, c.prefetch_useful, c.prefetch_full), (4, 4, 0));
    // One read per layer: each later layer misses its second expert.
    let c = prefetched(PrefetchReplay { lookahead: 1, top: 4, budget: Some(1) });
    assert_eq!((c.decode_misses, c.decode_layers_missed), (4, 3));
    assert_eq!((c.prefetch_reads, c.prefetch_useful), (2, 2));
    // Two layers ahead: only layer 2 has a source.
    let c = prefetched(PrefetchReplay { lookahead: 2, top: 4, budget: None });
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
