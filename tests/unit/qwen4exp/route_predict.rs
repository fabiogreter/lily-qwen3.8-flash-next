//! The prediction scoring over a hand-made log (3 layers, 16 experts,
//! top-2, 2 positions), and the residency sets over a one-layer ranking.

use super::*;

const LAYERS: usize = 3;
const EXPERTS: usize = 16;
const K: usize = 2;
const CAP: usize = 2;
/// Rows without a source layer: ids a scorer must ignore.
const JUNK: [u32; 4] = [15, 14, 13, 12];

/// `actual[layer][pos]` = `[4 * layer + 2 * pos, 4 * layer + 2 * pos + 1]`.
fn actual() -> Vec<u32> {
    (0..(LAYERS * CAP * K) as u32).collect()
}

/// Both variants' lookahead 0 reproduce `actual`; 1 and 2 hit, half-hit and
/// miss on purpose.
fn predicted() -> Vec<u32> {
    let mut log = actual();
    // A, lookahead 1: layer 1 [5, 4] (exact, other order) and [6, 0] (one
    // hit), layer 2 [1, 2] (none) and [11, 10] (exact).
    log.extend(JUNK);
    log.extend([5, 4, 6, 0]);
    log.extend([1, 2, 11, 10]);
    // A, lookahead 2: layer 2 [9, 8] (exact) and [10, 3] (one hit).
    log.extend(JUNK);
    log.extend(JUNK);
    log.extend([9, 8, 10, 3]);
    // B: lookahead 0 exact, lookahead 1 exact everywhere, lookahead 2
    // layer 2 [8, 0] (one hit) and [10, 11] (exact).
    log.extend(actual());
    log.extend(JUNK);
    log.extend([4, 5, 6, 7]);
    log.extend([8, 9, 10, 11]);
    log.extend(JUNK);
    log.extend(JUNK);
    log.extend([8, 0, 10, 11]);
    assert_eq!(
        log.len(),
        EXPERT_PRED_VARIANTS * EXPERT_PRED_LOOKAHEADS * LAYERS * CAP * K
    );
    log
}

/// Outside the resident set: layer 1's expert 7, layer 2's experts 1, 8, 9.
fn cold() -> Vec<bool> {
    let mut cold = vec![false; LAYERS * EXPERTS];
    for (layer, e) in [(1, 7), (2, 1), (2, 8), (2, 9)] {
        cold[layer * EXPERTS + e] = true;
    }
    cold
}

const A: usize = 0;
const B: usize = 1;

#[test]
fn overlap_coverage_and_histograms() {
    let mut t = PredictionTally::new(LAYERS, EXPERTS, K, None);
    t.add(&actual(), &predicted(), CAP, 0..CAP).expect("add");
    assert_eq!(t.tokens(), 2);

    for v in [A, B] {
        let own = t.all_layers(v, 0);
        assert_eq!(own, LayerTally { tokens: 6, hits: 12, exact: 6, covered: 6 });
        assert_eq!(t.missing_pairs(v, 0)[0], 2);
    }

    assert_eq!(
        t.layer(A, 1, 1),
        LayerTally { tokens: 2, hits: 3, exact: 1, covered: 1 }
    );
    assert_eq!(
        t.layer(A, 1, 2),
        LayerTally { tokens: 2, hits: 2, exact: 1, covered: 1 }
    );
    assert_eq!(t.layer(A, 1, 0), LayerTally::default(), "no source layer");
    assert_eq!(
        t.all_layers(A, 1),
        LayerTally { tokens: 4, hits: 5, exact: 2, covered: 2 }
    );
    // Position 0 misses both of layer 2's experts, position 1 one of
    // layer 1's: one layer each.
    assert_eq!(t.missing_pairs(A, 1), &[0, 1, 1, 0, 0, 0, 0]);
    assert_eq!(t.missing_layers(A, 1), &[0, 2, 0, 0]);
    assert_eq!(
        t.all_layers(A, 2),
        LayerTally { tokens: 2, hits: 3, exact: 1, covered: 1 }
    );
    assert_eq!(t.missing_pairs(A, 2), &[1, 1, 0, 0, 0, 0, 0]);
    assert_eq!(t.missing_layers(A, 2), &[1, 1, 0, 0]);

    assert_eq!(
        t.all_layers(B, 1),
        LayerTally { tokens: 4, hits: 8, exact: 4, covered: 4 }
    );
    assert_eq!(t.missing_pairs(B, 1)[0], 2);
    assert_eq!(
        t.all_layers(B, 2),
        LayerTally { tokens: 2, hits: 3, exact: 1, covered: 1 }
    );

    let json = t.to_json();
    assert_eq!(json["tokens"], 2);
    assert_eq!(json["self_check"]["A"]["exact"], 1.0);
    assert_eq!(json["self_check"]["B"]["rows"], 6);
    let one = &json["A"]["1"];
    assert_eq!(one["applicable_layers"], 2);
    assert!(one["per_layer"][0].is_null());
    assert_eq!(one["per_layer"][1]["overlap"], 0.75);
    assert_eq!(one["all_layers"]["overlap"], 0.625);
    assert_eq!(one["all_layers"]["covered"], 0.5);
    assert_eq!(
        one["missing_pairs_per_token"]["histogram"],
        serde_json::json!([0, 1, 1])
    );
    assert_eq!(one["missing_pairs_per_token"]["mean"], 1.5);
    assert_eq!(one["missing_layers_per_token"]["histogram"], serde_json::json!([0, 2]));
    assert!(one.get("prefetch").is_none(), "no resident set given");
    let two = &json["A"]["2"];
    assert!(two["per_layer"][1].is_null());
    assert_eq!(two["missing_layers_per_token"]["mean"], 0.5);
    assert_eq!(json["B"]["1"]["all_layers"]["exact"], 1.0);
    assert!(json["A"].get("0").is_none(), "lookahead 0 is the self-check");
}

#[test]
fn prefetch_reads_and_cold_recall() {
    let mut t = PredictionTally::new(LAYERS, EXPERTS, K, Some(cold()));
    t.add(&actual(), &predicted(), CAP, 0..CAP).expect("add");
    // A, lookahead 1: position 0 reads layer 2's expert 1 (useless) and
    // misses its cold 8 and 9; position 1 misses layer 1's cold 7.
    let a = t.prefetch(A, 1);
    assert_eq!((a.reads, a.cold_actual, a.cold_covered), (1, 3, 0));
    assert_eq!(&a.reads_histogram[..3], &[1, 1, 0]);
    // B, lookahead 1: every cold slice read, nothing else.
    let b = t.prefetch(B, 1);
    assert_eq!((b.reads, b.cold_actual, b.cold_covered), (3, 3, 3));
    assert_eq!(&b.reads_histogram[..3], &[0, 1, 1]);
    // Lookahead 0 reads exactly the cold actual slices.
    let own = t.prefetch(A, 0);
    assert_eq!((own.reads, own.cold_actual, own.cold_covered), (3, 3, 3));

    let json = t.to_json();
    let fa = &json["A"]["1"]["prefetch"];
    assert_eq!(fa["reads_per_token"]["mean"], 0.5);
    assert_eq!(fa["reads_per_layer"], 0.25);
    assert_eq!(fa["cold_actual_per_token"], 1.5);
    assert_eq!(fa["cold_recall"], 0.0);
    assert_eq!(fa["read_precision"], 0.0);
    let fb = &json["B"]["1"]["prefetch"];
    assert_eq!(fb["cold_recall"], 1.0);
    assert_eq!(fb["read_precision"], 1.0);
    assert_eq!(fb["cold_covered_per_token"], 1.5);
}

#[test]
fn positions_select_the_scored_rows() {
    let mut t = PredictionTally::new(LAYERS, EXPERTS, K, None);
    t.add(&actual(), &predicted(), CAP, 1..CAP).expect("add");
    assert_eq!(t.tokens(), 1);
    assert_eq!(
        t.all_layers(A, 1),
        LayerTally { tokens: 2, hits: 3, exact: 1, covered: 1 }
    );
    assert_eq!(t.missing_pairs(A, 1)[1], 1);
    assert!(t.add(&actual(), &predicted(), CAP, 0..CAP + 1).is_err());
    assert!(t.add(&actual()[1..], &predicted(), CAP, 0..1).is_err());
    let mut bad = predicted();
    bad[0] = EXPERTS as u32;
    assert!(t.add(&actual(), &bad, CAP, 0..1).is_err(), "expert id out of range");
}

/// One layer of four experts, two of them resident.
#[test]
fn residency_sets_and_shares() {
    let global = vec![vec![14, 0, 1, 5]];
    // The log: one layer, capacity 2, top-2: positions route [1, 3], [1, 0].
    let log = [1u32, 3, 1, 0];
    let own = routing_counts(&log, 1, 4, 2, 2, 0..2);
    assert_eq!(own, vec![vec![1, 2, 0, 1]]);
    // Ties in the prompt's own counts (0 and 3, one each) go by the global
    // order (0 before 3).
    let [own_set, global_set, mix_set] = residency_sets(&own, &global, 2);
    assert_eq!(own_set, [true, true, false, false]);
    assert_eq!(global_set, [true, false, false, true]);
    // Mix: 0.5 * own / 4 + 0.5 * global / 20 = [0.475, 0.25, 0.025, 0.25];
    // the tie between 1 and 3 goes to 3, earlier in the global order.
    assert_eq!(mix_set, [true, false, false, true]);

    // Three different sets.
    let own = vec![vec![0, 3, 1, 1]];
    let [own_set, global_set, mix_set] = residency_sets(&own, &global, 2);
    // Own: 1, then the tie between 2 and 3 by global order (3 first).
    assert_eq!(own_set, [false, true, false, true]);
    assert_eq!(global_set, [true, false, false, true]);
    // Mix: [0.35, 0.3, 0.125, 0.225].
    assert_eq!(mix_set, [true, true, false, false]);
    let share = |mask: &[bool]| share_inside(mask, &log, 1, 4, 2, 2, 0..2);
    assert_eq!(share(&own_set), 0.75);
    assert_eq!(share(&global_set), 0.5);
    assert_eq!(share(&mix_set), 0.75);
    assert_eq!(share_inside(&own_set, &log, 1, 4, 2, 2, 1..2), 0.5);
}

/// The top picks by score: equal scores keep the kernel's order, so the
/// first k of a wide selection are the k-wide selection.
#[test]
fn top_by_score_and_narrowing() {
    let mut out = Vec::new();
    // Unordered scores; 7 and 2 tie and keep their order.
    top_by_score(&[9, 7, 2, 4], &[0.1, 0.3, 0.3, 0.5], 3, &mut out);
    assert_eq!(out, [4, 7, 2]);
    top_by_score(&[9, 7], &[0.1, 0.3], 5, &mut out);
    assert_eq!(out, [7, 9], "fewer picks than asked for");

    let ids = [1u32, 2, 3, 4, 5, 6];
    let scores = [0.2f32, 0.5, 0.3, 0.0, 0.0, 0.0];
    let narrow = narrow_predictions(&ids, &scores, 3, 2).expect("narrow");
    // Row 0 by score: 2, 3; row 1 all tied (never written rows are zero):
    // kernel order.
    assert_eq!(narrow, [2, 3, 4, 5]);
    assert!(narrow_predictions(&ids, &scores[..5], 3, 2).is_err());
    assert!(narrow_predictions(&ids, &scores, 4, 2).is_err());
    assert!(narrow_predictions(&ids, &scores, 3, 4).is_err());
}
