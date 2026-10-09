//! Host-side scoring of the cross-layer routing prediction log
//! (`Qwen4ExpModel::enable_expert_prediction`): how well each predictor
//! variant with lookahead `d` predicts layer `l`'s actual top-k from layer
//! `l - d`, what a prefetcher driven by it would read beyond a resident set,
//! and whether a resident set ranked from a prompt's own prefill would serve
//! its decode better than the global ranking. `lily-experts --predict-out`
//! writes the JSON.

use std::ops::Range;

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};

use super::model::{EXPERT_PRED_LOOKAHEADS, EXPERT_PRED_VARIANTS};

/// Variant names in the order of the log's variant axis.
pub const VARIANT_NAMES: [&str; EXPERT_PRED_VARIANTS] = ["A", "B"];

/// Per (variant, lookahead, layer) counts over the scored tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerTally {
    pub tokens: u64,
    /// Sum over tokens of `|pred ∩ actual|`.
    pub hits: u64,
    /// Tokens whose predicted set equals the actual one.
    pub exact: u64,
    /// Tokens whose actual set lies inside the predicted one (with sets of
    /// equal size that is `exact`; kept apart for wider predictions).
    pub covered: u64,
}

impl LayerTally {
    fn plus(self, t: LayerTally) -> LayerTally {
        LayerTally {
            tokens: self.tokens + t.tokens,
            hits: self.hits + t.hits,
            exact: self.exact + t.exact,
            covered: self.covered + t.covered,
        }
    }
}

/// Per (variant, lookahead): what a prefetcher issuing a read for every
/// predicted slice outside the resident set would do, summed over tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefetchTally {
    /// Predicted slices outside the resident set (the reads issued).
    pub reads: u64,
    /// Actual slices outside the resident set.
    pub cold_actual: u64,
    /// Of those, the ones the prediction contained.
    pub cold_covered: u64,
    /// Tokens by reads issued, index = count.
    pub reads_histogram: Vec<u64>,
}

/// One lookahead plane's per-token tallies.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Plane {
    /// `[layers]`; layers `< d` stay empty (no source layer: not
    /// applicable, not a miss).
    per_layer: Vec<LayerTally>,
    /// Tokens by their count of (layer, expert) pairs routed to but not
    /// predicted, summed over the applicable layers.
    missing_pairs: Vec<u64>,
    /// Tokens by their count of applicable layers with such a pair.
    missing_layers: Vec<u64>,
    prefetch: PrefetchTally,
}

/// Accumulates prediction accuracy for one phase (prefill or decode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictionTally {
    layers: usize,
    experts: usize,
    top_k: usize,
    /// `[layer * experts + expert]`: outside the resident set the prefetch
    /// tallies are measured against; `None` skips them.
    cold: Option<Vec<bool>>,
    /// `[variant][lookahead]`.
    planes: Vec<Vec<Plane>>,
}

impl PredictionTally {
    pub fn new(
        layers: usize,
        experts: usize,
        top_k: usize,
        cold: Option<Vec<bool>>,
    ) -> Self {
        let plane = Plane {
            per_layer: vec![LayerTally::default(); layers],
            missing_pairs: vec![0; layers * top_k + 1],
            missing_layers: vec![0; layers + 1],
            prefetch: PrefetchTally {
                reads_histogram: vec![0; layers * top_k + 1],
                ..PrefetchTally::default()
            },
        };
        Self {
            layers,
            experts,
            top_k,
            cold,
            planes: vec![vec![plane; EXPERT_PRED_LOOKAHEADS]; EXPERT_PRED_VARIANTS],
        }
    }

    /// Scores `positions` of the logs as the model lays them out: `actual`
    /// `[layers, capacity, top_k]`, `predicted` `[EXPERT_PRED_VARIANTS,
    /// EXPERT_PRED_LOOKAHEADS, layers, capacity, top_k]`.
    pub fn add<'a>(
        &mut self,
        actual: &'a [u32],
        predicted: &'a [u32],
        capacity: usize,
        positions: Range<usize>,
    ) -> Result<()> {
        let (layers, k, experts) = (self.layers, self.top_k, self.experts);
        ensure!(actual.len() == layers * capacity * k, "actual log shape");
        ensure!(
            predicted.len()
                == EXPERT_PRED_VARIANTS
                    * EXPERT_PRED_LOOKAHEADS
                    * layers
                    * capacity
                    * k,
            "prediction log shape"
        );
        ensure!(positions.end <= capacity, "positions past the log's capacity");
        ensure!(
            actual.iter().chain(predicted).all(|&e| (e as usize) < experts),
            "expert id out of range"
        );
        let row = |log: &'a [u32], plane: usize, layer: usize, pos: usize| {
            let at = ((plane * layers + layer) * capacity + pos) * k;
            &log[at..at + k]
        };
        let cold = |layer: usize, e: u32| {
            self.cold.as_ref().is_some_and(|c| c[layer * experts + e as usize])
        };
        let (mut a, mut p) = (Vec::with_capacity(k), Vec::with_capacity(k));
        for pos in positions {
            for v in 0..EXPERT_PRED_VARIANTS {
                for d in 0..EXPERT_PRED_LOOKAHEADS {
                    let (mut pairs, mut missing_layers) = (0usize, 0usize);
                    let (mut reads, mut cold_actual, mut cold_covered) = (0, 0, 0);
                    for layer in d..layers {
                        let act = row(actual, 0, layer, pos);
                        let pred =
                            row(predicted, v * EXPERT_PRED_LOOKAHEADS + d, layer, pos);
                        let hits = act.iter().filter(|e| pred.contains(e)).count();
                        a.clear();
                        a.extend_from_slice(act);
                        a.sort_unstable();
                        p.clear();
                        p.extend_from_slice(pred);
                        p.sort_unstable();
                        let plane = &mut self.planes[v][d];
                        let t = &mut plane.per_layer[layer];
                        t.tokens += 1;
                        t.hits += hits as u64;
                        let covered = hits == k;
                        t.covered += covered as u64;
                        t.exact += (a == p) as u64;
                        pairs += k - hits;
                        missing_layers += !covered as usize;
                        reads += pred.iter().filter(|&&e| cold(layer, e)).count();
                        for &e in act.iter().filter(|&&e| cold(layer, e)) {
                            cold_actual += 1;
                            cold_covered += pred.contains(&e) as usize;
                        }
                    }
                    let plane = &mut self.planes[v][d];
                    plane.missing_pairs[pairs] += 1;
                    plane.missing_layers[missing_layers] += 1;
                    let f = &mut plane.prefetch;
                    f.reads += reads as u64;
                    f.cold_actual += cold_actual as u64;
                    f.cold_covered += cold_covered as u64;
                    f.reads_histogram[reads] += 1;
                }
            }
        }
        Ok(())
    }

    /// Tokens scored so far.
    pub fn tokens(&self) -> u64 {
        self.planes[0][0].missing_pairs.iter().sum()
    }

    pub fn layer(&self, variant: usize, lookahead: usize, layer: usize) -> LayerTally {
        self.planes[variant][lookahead].per_layer[layer]
    }

    /// The plane's counts summed over its applicable layers.
    pub fn all_layers(&self, variant: usize, lookahead: usize) -> LayerTally {
        self.planes[variant][lookahead].per_layer[lookahead..]
            .iter()
            .fold(LayerTally::default(), |acc, &t| acc.plus(t))
    }

    /// Tokens by missing (layer, expert) pairs, index = count.
    pub fn missing_pairs(&self, variant: usize, lookahead: usize) -> &[u64] {
        &self.planes[variant][lookahead].missing_pairs
    }

    /// Tokens by layers with at least one missing pair, index = count.
    pub fn missing_layers(&self, variant: usize, lookahead: usize) -> &[u64] {
        &self.planes[variant][lookahead].missing_layers
    }

    pub fn prefetch(&self, variant: usize, lookahead: usize) -> &PrefetchTally {
        &self.planes[variant][lookahead].prefetch
    }

    fn rates(&self, t: LayerTally) -> Value {
        let n = t.tokens.max(1) as f64;
        json!({
            "tokens": t.tokens,
            "overlap": t.hits as f64 / (n * self.top_k as f64),
            "exact": t.exact as f64 / n,
            "covered": t.covered as f64 / n,
        })
    }

    fn histogram(counts: &[u64]) -> Value {
        let n: u64 = counts.iter().sum();
        let mean = counts.iter().enumerate().map(|(i, &c)| i as u64 * c).sum::<u64>()
            as f64
            / n.max(1) as f64;
        let len = counts.iter().rposition(|&c| c > 0).map_or(1, |i| i + 1);
        json!({ "mean": mean, "histogram": &counts[..len] })
    }

    /// One plane's summary: per-layer rates (`null` where the layer has no
    /// source layer), the rates over all applicable layers, the two
    /// per-token miss histograms, and the prefetch tallies.
    pub fn plane_json(&self, variant: usize, lookahead: usize) -> Value {
        let plane = &self.planes[variant][lookahead];
        let applicable = self.layers.saturating_sub(lookahead);
        let per_layer: Vec<Value> = (0..self.layers)
            .map(|l| {
                if l < lookahead { Value::Null } else { self.rates(plane.per_layer[l]) }
            })
            .collect();
        let mut out = json!({
            "applicable_layers": applicable,
            "all_layers": self.rates(self.all_layers(variant, lookahead)),
            "per_layer": per_layer,
            "missing_pairs_per_token": Self::histogram(&plane.missing_pairs),
            "missing_layers_per_token": Self::histogram(&plane.missing_layers),
        });
        if self.cold.is_some() {
            let f = &plane.prefetch;
            let n = self.tokens().max(1) as f64;
            out["prefetch"] = json!({
                "reads_per_token": Self::histogram(&f.reads_histogram),
                "reads_per_layer": f.reads as f64 / n / applicable.max(1) as f64,
                "cold_actual_per_token": f.cold_actual as f64 / n,
                "cold_covered_per_token": f.cold_covered as f64 / n,
                "cold_recall": f.cold_covered as f64 / f.cold_actual.max(1) as f64,
                "read_precision": f.cold_covered as f64 / f.reads.max(1) as f64,
            });
        }
        out
    }

    /// The phase's summary: per variant, lookaheads 1 and up, plus each
    /// variant's lookahead 0 as the self-check (its `exact` must be 1).
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("tokens".into(), json!(self.tokens()));
        let mut check = Map::new();
        for (v, name) in VARIANT_NAMES.iter().enumerate() {
            let own = self.all_layers(v, 0);
            check.insert(
                (*name).into(),
                json!({
                    "rows": own.tokens,
                    "exact": own.exact as f64 / own.tokens.max(1) as f64,
                }),
            );
            let mut lookahead = Map::new();
            for d in 1..EXPERT_PRED_LOOKAHEADS {
                lookahead.insert(d.to_string(), self.plane_json(v, d));
            }
            out.insert((*name).into(), Value::Object(lookahead));
        }
        out.insert("self_check".into(), Value::Object(check));
        Value::Object(out)
    }
}

/// The `n` highest-scored picks of one prediction-log row (`ids` and
/// `scores` as the router kernel wrote them, `EXPERT_PRED_WIDTH * top_k`
/// of each), into `out`, by score descending. The sort is stable, so equal
/// scores keep the kernel's order, which is its selection order (logit
/// descending, ties to the lowest id): the first `top_k` of a wider
/// selection are then exactly what a top-k selection picks, which the
/// lookahead-0 check relies on.
pub fn top_by_score(ids: &[u32], scores: &[f32], n: usize, out: &mut Vec<u32>) {
    debug_assert_eq!(ids.len(), scores.len());
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    out.clear();
    out.extend(order.iter().take(n).map(|&i| ids[i]));
}

/// The wide prediction log (`[.., width]` rows of ids and their scores)
/// narrowed to each row's top `k` by score ([`top_by_score`]), in the
/// layout [`PredictionTally::add`] scores: `[.., k]`.
pub fn narrow_predictions(
    ids: &[u32],
    scores: &[f32],
    width: usize,
    k: usize,
) -> Result<Vec<u32>> {
    ensure!(
        ids.len() == scores.len() && width > 0 && ids.len().is_multiple_of(width),
        "prediction ids and scores are not rows of {width}"
    );
    ensure!(k <= width, "top {k} of rows of {width}");
    let mut out = Vec::with_capacity(ids.len() / width * k);
    let mut row = Vec::with_capacity(k);
    for (ids, scores) in ids.chunks_exact(width).zip(scores.chunks_exact(width)) {
        top_by_score(ids, scores, k, &mut row);
        out.extend_from_slice(&row);
    }
    Ok(out)
}

/// Every (layer, expert) slice by `score` descending, ties by `tie`
/// ascending; the first `n` as a `[layer * experts + expert]` mask.
fn top_n_mask(score: &[f64], tie: &[usize], n: usize) -> Vec<bool> {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(tie[a].cmp(&tie[b])));
    let mut mask = vec![false; score.len()];
    for &i in order.iter().take(n) {
        mask[i] = true;
    }
    mask
}

/// The three resident sets of `n` slices the residency test compares,
/// as `[layer * experts + expert]` masks: (a) the prompt's own prefill
/// counts, (b) the global counts, (c) the two normalised to sum 1 and mixed
/// 50/50. The global ranking orders ties as `UsageRanking::ranked` does
/// (count descending, then layer and expert ascending); the other two
/// break their ties by the global order, so the many slices a single prompt
/// never routes to fill (a) in global order.
pub fn residency_sets(
    own: &[Vec<u64>],
    global: &[Vec<u64>],
    n: usize,
) -> [Vec<bool>; 3] {
    let flat = |c: &[Vec<u64>]| -> Vec<f64> {
        c.iter().flatten().map(|&x| x as f64).collect()
    };
    let (own, global) = (flat(own), flat(global));
    let identity: Vec<usize> = (0..global.len()).collect();
    let global_mask = top_n_mask(&global, &identity, n);
    let mut global_rank = vec![0usize; global.len()];
    let mut order = identity.clone();
    order.sort_by(|&a, &b| global[b].total_cmp(&global[a]).then(a.cmp(&b)));
    for (rank, &i) in order.iter().enumerate() {
        global_rank[i] = rank;
    }
    let (own_sum, global_sum) =
        (own.iter().sum::<f64>().max(1.0), global.iter().sum::<f64>().max(1.0));
    let mix: Vec<f64> = own
        .iter()
        .zip(&global)
        .map(|(o, g)| 0.5 * o / own_sum + 0.5 * g / global_sum)
        .collect();
    [top_n_mask(&own, &global_rank, n), global_mask, top_n_mask(&mix, &global_rank, n)]
}

/// Per-layer routing counts of `positions` in the actual log
/// (`[layers, capacity, top_k]`): `counts[layer][expert]`.
pub fn routing_counts(
    actual: &[u32],
    layers: usize,
    experts: usize,
    top_k: usize,
    capacity: usize,
    positions: Range<usize>,
) -> Vec<Vec<u64>> {
    let mut counts = vec![vec![0u64; experts]; layers];
    for (layer, row) in counts.iter_mut().enumerate() {
        let base = layer * capacity * top_k;
        for &e in &actual[base + positions.start * top_k..base + positions.end * top_k]
        {
            row[e as usize] += 1;
        }
    }
    counts
}

/// The share of the routed (layer, expert) lookups of `positions` that
/// fall inside `mask`.
pub fn share_inside(
    mask: &[bool],
    actual: &[u32],
    layers: usize,
    experts: usize,
    top_k: usize,
    capacity: usize,
    positions: Range<usize>,
) -> f64 {
    let (mut inside, mut total) = (0u64, 0u64);
    for layer in 0..layers {
        let base = layer * capacity * top_k;
        for &e in &actual[base + positions.start * top_k..base + positions.end * top_k]
        {
            inside += mask[layer * experts + e as usize] as u64;
            total += 1;
        }
    }
    inside as f64 / total.max(1) as f64
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/route_predict.rs"]
mod tests;
