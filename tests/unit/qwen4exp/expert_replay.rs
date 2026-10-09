//! The replay driver over hand-made traces: 2 layers of 8 experts, top-2,
//! a cache of 4 or 6 slots with half of them LRU. The global ranking pins
//! (0, 0) and (0, 1) (then (1, 0) with 6 slots) and fills the LRU region
//! with the next ranked slices; an untouched LRU slot is evicted before a
//! stamped one, the worst-ranked first.

use super::*;
use crate::qwen4exp::route_predict::residency_sets;

const LAYERS: usize = 2;
const EXPERTS: usize = 8;
const K: usize = 2;

/// Ranked (0, 0), (0, 1), (1, 0), (1, 1), then the zero counts in (layer,
/// expert) order: (0, 2), (0, 3), ...
fn global() -> UsageRanking {
    let mut counts = vec![vec![0u64; EXPERTS]; LAYERS];
    counts[0][0] = 10;
    counts[0][1] = 9;
    counts[1][0] = 8;
    counts[1][1] = 7;
    UsageRanking { layers: LAYERS, experts: EXPERTS, counts }
}

fn config(slots: usize, chunk: usize) -> ReplayConfig {
    ReplayConfig {
        layers: LAYERS,
        experts: EXPERTS,
        top_k: K,
        slots,
        lru_share: 0.5,
        prefill_chunk: chunk,
        adapt: true,
        carry: false,
    }
}

/// A trace from per-layer rows (`rows[layer][pos]`), prefill first, and
/// one predictor's rows for layer 1 per decode token (`ids`, `scores`).
fn trace(
    rows: [&[[u32; K]]; LAYERS],
    prefill: usize,
    layer1_predictions: &[(&[u32], &[f32])],
) -> PromptTrace {
    let positions = rows[0].len();
    let mut actual = Vec::new();
    for layer_rows in rows {
        assert_eq!(layer_rows.len(), positions);
        for r in layer_rows {
            actual.extend_from_slice(r);
        }
    }
    let decode = positions - prefill;
    let width = layer1_predictions.first().map_or(1, |(ids, _)| ids.len());
    let mut predicted = PredictedRows {
        width,
        ids: vec![0; LAYERS * decode * width],
        scores: vec![0.0; LAYERS * decode * width],
    };
    for (t, (ids, scores)) in layer1_predictions.iter().enumerate() {
        let at = (decode + t) * width;
        predicted.ids[at..at + width].copy_from_slice(ids);
        predicted.scores[at..at + width].copy_from_slice(scores);
    }
    PromptTrace {
        prefill_tokens: prefill,
        decode_tokens: decode,
        actual,
        predicted: vec![predicted],
    }
}

fn replay(
    cfg: ReplayConfig,
    policies: &[Policy],
    trace: &PromptTrace,
) -> Vec<PromptOutcome> {
    let global = global();
    let named = policies.iter().map(|p| (format!("{p:?}"), *p)).collect();
    let mut r = Replay::new(cfg, &global, named).expect("replay");
    r.add_prompt(trace).expect("add");
    r.outcomes().iter().map(|o| o[0].clone()).collect()
}

/// Prefill requests carry the union of a chunk's rows: 3 prompt tokens in
/// chunks of 2 are 2 requests per layer; chunk 0 of layer 0 unions [0, 1]
/// and [1, 2] into 3 lookups, (0, 2) misses into the untouched LRU slot 3
/// (evicting (1, 1)), whose own miss in layer 1 then takes slot 3 back (slot
/// 2 was used at that tick). Decode: token 0 hits everywhere; token 1 misses
/// (0, 3) and then (1, 2), one layer each.
#[test]
fn prefill_unions_chunks_and_live_counts_misses() {
    let rows0: &[[u32; K]] = &[[0, 1], [1, 2], [1, 0], [0, 1], [0, 3]];
    let rows1: &[[u32; K]] = &[[0, 1], [1, 0], [0, 1], [1, 0], [1, 2]];
    let t = trace([rows0, rows1], 3, &[]);
    let [live] = replay(config(4, 2), &[Policy::Live], &t).try_into().expect("one");
    assert_eq!(
        live.prefill,
        PhaseTally { tokens: 3, requests: 4, lookups: 9, misses: 2 }
    );
    assert_eq!(
        live.decode.phase,
        PhaseTally { tokens: 2, requests: 4, lookups: 8, misses: 2 }
    );
    assert_eq!(live.decode.layers_with_miss, vec![1, 0, 1]);
    assert_eq!((live.decode.reads, live.decode.useful), (0, 0));
    assert_eq!(live.promotions, 0);

    // A chunk of the whole prompt: one request per layer.
    let [one] = replay(config(4, 8), &[Policy::Live], &t).try_into().expect("one");
    assert_eq!((one.prefill.requests, one.prefill.lookups), (2, 5));
}

/// The re-rank's ranking pins what `residency_sets` would keep.
#[test]
fn rerank_ranking_orders_like_the_residency_sets() {
    let global = global();
    let mut own = vec![vec![0u64; EXPERTS]; LAYERS];
    own[0][4] = 3;
    own[0][0] = 2;
    own[0][1] = 1;
    own[1][5] = 3;
    own[1][0] = 3;
    for n in 1..=LAYERS * EXPERTS {
        let [own_set, _, mix_set] = residency_sets(&own, &global.counts, n);
        for (w, set) in [(Weighting::Own, own_set), (Weighting::Mix, mix_set)] {
            let ranking = rerank_ranking(&own, &global, w).expect("ranking");
            let mut mask = vec![false; LAYERS * EXPERTS];
            for &(l, e) in ranking.ranked().iter().take(n) {
                mask[l * EXPERTS + e] = true;
            }
            assert_eq!(mask, set, "{w:?}, top {n}");
        }
    }
}

/// The prefill routes (0, 4) and (1, 5) heavily; live ends it holding
/// (0, 0), (0, 1), (1, 0), (1, 5). Own ranks (1, 0), (0, 4) | (1, 5),
/// (0, 0): one pinned read. Mix ranks (1, 0), (0, 0) | (0, 1), (0, 4): one
/// LRU read and none pinned. The decode token routes (0, 4) and (1, 5):
/// live misses once in each layer, own not at all.
#[test]
fn rerank_counts_its_one_off_reads() {
    let rows0: &[[u32; K]] = &[[4, 0], [4, 0], [4, 1], [4, 0]];
    let rows1: &[[u32; K]] = &[[5, 0], [5, 0], [5, 0], [5, 0]];
    let t = trace([rows0, rows1], 3, &[]);
    let policies =
        [Policy::Live, Policy::Rerank(Weighting::Own), Policy::Rerank(Weighting::Mix)];
    let [live, own, mix] =
        replay(config(4, 2), &policies, &t).try_into().expect("three");
    assert_eq!(live.prefill, own.prefill, "the prefill is shared");
    assert_eq!((live.rerank_pinned_reads, live.rerank_lru_reads), (0, 0));
    assert_eq!((own.rerank_pinned_reads, own.rerank_lru_reads), (1, 0));
    assert_eq!((mix.rerank_pinned_reads, mix.rerank_lru_reads), (0, 1));
    assert_eq!(live.prefill.misses, 5);
    assert_eq!(live.decode.phase.misses, 2);
    assert_eq!(live.decode.layers_with_miss, vec![0, 0, 1]);
    assert_eq!(own.decode.phase.misses, 0);
    assert_eq!(own.decode.layers_with_miss, vec![1, 0, 0]);
}

/// Six slots: (0, 0), (0, 1), (1, 0) pinned; (1, 1), (0, 2), (0, 3) in the
/// LRU region. The decode token routes layer 1 to [1, 3]: live misses
/// (1, 3). The prediction for layer 1 (unordered, by score: 3, 6, 1, 0)
/// read ahead turns that miss into a hit; reading (1, 6) too is wasted.
#[test]
fn prefetch_hides_a_miss_and_counts_waste() {
    let rows0: &[[u32; K]] = &[[0, 1], [0, 1]];
    let rows1: &[[u32; K]] = &[[0, 1], [1, 3]];
    let ids: &[u32] = &[6, 3, 1, 0];
    let scores: &[f32] = &[0.3, 0.5, 0.1, 0.1];
    let t = trace([rows0, rows1], 1, &[(ids, scores)]);
    let prefetch = |set, budget, from_layer| {
        Policy::Prefetch(Prefetch { predictor: 0, set, budget, from_layer })
    };
    let policies = [
        Policy::Live,
        prefetch(PredictedSet::Top(2), None, 1),
        prefetch(PredictedSet::Top(2), Some(1), 1),
        prefetch(PredictedSet::AtLeast(0.25), None, 1),
        prefetch(PredictedSet::Top(4), None, 1),
        // A threshold above every score: nothing read.
        prefetch(PredictedSet::AtLeast(0.6), None, 1),
        // Deep layers only: layer 1 is below.
        prefetch(PredictedSet::Top(2), None, 2),
    ];
    let out = replay(config(6, 4), &policies, &t);
    let live = &out[0].decode;
    assert_eq!((live.phase.misses, live.reads), (1, 0));
    assert_eq!(live.layers_with_miss, vec![0, 1, 0]);

    let top2 = &out[1].decode;
    assert_eq!(
        (top2.phase.misses, top2.reads, top2.useful, top2.wasted()),
        (0, 2, 1, 1)
    );
    assert_eq!(top2.layers_with_miss, vec![1, 0, 0]);
    assert_eq!((top2.predicted, top2.predicted_rows), (2, 1));

    let budget1 = &out[2].decode;
    assert_eq!((budget1.phase.misses, budget1.reads, budget1.useful), (0, 1, 1));

    let threshold = &out[3].decode;
    assert_eq!((threshold.phase.misses, threshold.reads, threshold.useful), (0, 2, 1));
    assert_eq!(threshold.predicted, 2);

    // Resident picks cost no read: (1, 1) and (1, 0) are skipped.
    let top4 = &out[4].decode;
    assert_eq!((top4.phase.misses, top4.reads, top4.useful), (0, 2, 1));
    assert_eq!(top4.predicted, 4);

    let none = &out[5].decode;
    assert_eq!(
        (none.phase.misses, none.reads, none.useful, none.wasted()),
        (1, 0, 0, 0)
    );
    assert_eq!(none.predicted, 0);

    let deep = &out[6].decode;
    assert_eq!((deep.phase.misses, deep.reads, deep.predicted_rows), (1, 0, 0));

    let json = {
        let global = global();
        let named = policies.iter().map(|p| (format!("{p:?}"), *p)).collect();
        let mut r = Replay::new(config(6, 4), &global, named).expect("replay");
        r.add_prompt(&t).expect("add");
        r.to_json()
    };
    assert_eq!(json[1]["decode"]["prefetch"]["useful"], 1);
    assert_eq!(json[1]["decode"]["prefetch"]["wasted_per_token"], 1.0);
    assert_eq!(json[1]["decode"]["misses_per_token"], 0.0);
    assert_eq!(
        json[0]["decode"]["layers_with_miss_per_token"]["histogram"],
        serde_json::json!([0, 1])
    );
    assert!(json[0]["decode"].get("prefetch").is_none());
}

/// A wrong prefetch lands at a slot the real request then cannot use.
#[test]
fn a_wrong_prefetch_is_wasted() {
    let rows0: &[[u32; K]] = &[[0, 1], [0, 1]];
    let rows1: &[[u32; K]] = &[[0, 1], [1, 3]];
    let ids: &[u32] = &[6, 5];
    let scores: &[f32] = &[0.6, 0.4];
    let t = trace([rows0, rows1], 1, &[(ids, scores)]);
    let policy = Policy::Prefetch(Prefetch {
        predictor: 0,
        set: PredictedSet::Top(1),
        budget: None,
        from_layer: 1,
    });
    let [out] = replay(config(6, 4), &[policy], &t).try_into().expect("one");
    let d = &out.decode;
    assert_eq!((d.phase.misses, d.reads, d.useful, d.wasted()), (1, 1, 0, 1));
}

/// With `carry`, the second prompt starts from the first one's slab: the
/// same prompt twice misses only the first time.
#[test]
fn carry_keeps_the_slab_across_prompts() {
    let rows0: &[[u32; K]] = &[[4, 0], [4, 0]];
    let rows1: &[[u32; K]] = &[[5, 0], [5, 0]];
    let t = trace([rows0, rows1], 1, &[]);
    let global = global();
    for carry in [false, true] {
        let cfg = ReplayConfig { carry, ..config(6, 4) };
        let mut r = Replay::new(cfg, &global, vec![("live".into(), Policy::Live)])
            .expect("replay");
        r.add_prompt(&t).expect("first");
        r.add_prompt(&t).expect("second");
        let o = &r.outcomes()[0];
        assert_eq!(o[0].prefill.misses, 2);
        assert_eq!(o[1].prefill.misses, if carry { 0 } else { 2 }, "carry {carry}");
    }
}

/// The trace cut out of the model's logs: positions past the run dropped,
/// lookahead 1 of the requested variants kept.
#[test]
fn trace_from_logs() {
    let (layers, k, capacity, lookaheads, width) = (2, 2, 4, 3, 3);
    let actual: Vec<u32> = (0..(layers * capacity * k) as u32).collect();
    let n = 2 * lookaheads * layers * capacity * width;
    let ids: Vec<u32> = (0..n as u32).collect();
    let scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let t = PromptTrace::from_logs(
        &actual,
        &ids,
        &scores,
        (layers, k, capacity),
        (lookaheads, width),
        2,
        1,
        &[1, 0],
    )
    .expect("trace");
    assert_eq!(t.actual, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]);
    assert_eq!(t.predicted.len(), 2);
    // Variant 1, lookahead 1, layer 0, position 2: plane (1 * 3 + 1) * 2.
    let at = ((4 * layers) * capacity + 2) * width;
    assert_eq!(t.predicted[0].ids[..width], ids[at..at + width]);
    let at1 = ((4 * layers + 1) * capacity + 2) * width;
    assert_eq!(t.predicted[0].ids[width..], ids[at1..at1 + width]);
    let at0 = ((layers + 1) * capacity + 2) * width;
    assert_eq!(t.predicted[1].scores[width..], scores[at0..at0 + width]);
    assert!(
        PromptTrace::from_logs(
            &actual,
            &ids,
            &scores,
            (layers, k, capacity),
            (lookaheads, width),
            3,
            2,
            &[0],
        )
        .is_err(),
        "past the capacity"
    );
}
