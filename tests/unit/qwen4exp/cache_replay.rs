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
    ReplayConfig { slots: 16, lru_share: 0.1, adapt: false, prefill_chunk: 4 }
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
    sim.session(&routing(13, &[(0, 1)]), 13, 2, &req, &mut c).expect("replay");
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
    let cfg =
        ReplayConfig { slots: 128, lru_share: 0.1, adapt: false, prefill_chunk: 4096 };
    let mut sim = CacheSim::new(&r, 2, cfg).expect("sim");
    let mut c = ReplayCounts::default();
    // Two decoded tokens routed to (62, 63) in every layer: the first misses
    // both in all three layers, the second finds them in the LRU region.
    let routing: Vec<u32> = (0..3).flat_map(|_| [62, 63, 62, 63]).collect();
    let req = [TraceRequest { prefill: 0..0, decode: 0..2 }];
    sim.session(&routing, 2, 2, &req, &mut c).expect("replay");
    assert_eq!((c.decode_tokens, c.decode_lookups), (2, 12));
    assert_eq!((c.decode_misses, c.decode_layers_missed), (6, 3));
    // A routing log of the wrong size is refused.
    assert!(sim.session(&routing[1..], 2, 2, &req, &mut c).is_err());
}
