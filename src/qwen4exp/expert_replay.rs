//! Host-side replay of the expert cache's live policy over logged routing
//! (`docs/low-ram-experts.md`, "Replay of the live policy"): what the cache
//! as shipped misses on a prompt, and what re-ranking the pinned set at the
//! end of the prefill or a cross-layer prefetch would change.
//!
//! The replay feeds [`SlotPolicy`] and [`LiveUsage`], the service thread's
//! own policy code, the request stream the service thread would see for the
//! logged routing of one plain (draft-free) run: per prefill chunk one
//! request per MoE layer with the union of the chunk's rows' routed ids,
//! per decoded token one request per layer with its `top_k` ids; ids
//! deduplicated, every resolved slice recorded as decode- or prefill-class
//! by the request's routed-id count ([`DECODE_IDS`]), the tick rising by
//! one per request (the service's sequence number), a promotion pass every
//! [`PROMOTE_EVERY`] requests. `lily-experts --replay-out` drives it.

use anyhow::{Result, bail, ensure};
use serde_json::{Value, json};

use super::expert_cache::{DECODE_IDS, PROMOTE_EVERY};
use super::expert_store::{LiveUsage, Lookup, SlotPolicy, UsageRanking};
use super::route_predict::{routing_counts, top_by_score};

/// One predictor's decode-time predictions for a prompt: for every layer
/// `l` and decoded token `t`, the picks predicting layer `l` from layer
/// `l - 1` (lookahead 1) and their router scores, `[layers, decode_tokens,
/// width]` each, in the router kernel's order. Layer 0's rows are unused.
#[derive(Clone, Debug, Default)]
pub struct PredictedRows {
    pub width: usize,
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}

/// One prompt's logged routing, and its decode predictions per predictor.
#[derive(Clone, Debug)]
pub struct PromptTrace {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    /// `[layers, prefill_tokens + decode_tokens, top_k]`: prefill rows, then
    /// each decode step's.
    pub actual: Vec<u32>,
    /// Indexed by [`Prefetch::predictor`].
    pub predicted: Vec<PredictedRows>,
}

impl PromptTrace {
    /// Cuts a prompt's trace out of the model's logs: `actual` `[layers,
    /// capacity, top_k]` (prefill at positions `0..prefill_tokens`, decode
    /// after), `pred_ids`/`pred_scores` `[variants, lookaheads, layers,
    /// capacity, width]`; `variants` picks the predictors, each at
    /// lookahead 1.
    #[allow(clippy::too_many_arguments)]
    pub fn from_logs(
        actual: &[u32],
        pred_ids: &[u32],
        pred_scores: &[f32],
        (layers, top_k, capacity): (usize, usize, usize),
        (lookaheads, width): (usize, usize),
        prefill_tokens: usize,
        decode_tokens: usize,
        variants: &[usize],
    ) -> Result<Self> {
        let n = prefill_tokens + decode_tokens;
        ensure!(n <= capacity, "{n} positions in a log of {capacity}");
        ensure!(actual.len() == layers * capacity * top_k, "routing log shape");
        ensure!(
            pred_ids.len() == pred_scores.len()
                && pred_ids
                    .len()
                    .is_multiple_of(lookaheads * layers * capacity * width),
            "prediction log shape"
        );
        ensure!(lookaheads > 1, "the prediction log has no lookahead 1");
        let n_variants = pred_ids.len() / (lookaheads * layers * capacity * width);
        let mut routed = Vec::with_capacity(layers * n * top_k);
        for l in 0..layers {
            let at = l * capacity * top_k;
            routed.extend_from_slice(&actual[at..at + n * top_k]);
        }
        let mut predicted = Vec::with_capacity(variants.len());
        for &v in variants {
            ensure!(v < n_variants, "variant {v} of {n_variants}");
            let mut rows = PredictedRows {
                width,
                ids: Vec::with_capacity(layers * decode_tokens * width),
                scores: Vec::with_capacity(layers * decode_tokens * width),
            };
            for l in 0..layers {
                let plane = (v * lookaheads + 1) * layers + l;
                let at = (plane * capacity + prefill_tokens) * width;
                let span = at..at + decode_tokens * width;
                rows.ids.extend_from_slice(&pred_ids[span.clone()]);
                rows.scores.extend_from_slice(&pred_scores[span]);
            }
            predicted.push(rows);
        }
        Ok(Self { prefill_tokens, decode_tokens, actual: routed, predicted })
    }

    fn positions(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// The `top_k` routed ids of `layer` at position `pos`.
    fn routed(&self, layer: usize, pos: usize, top_k: usize) -> &[u32] {
        let at = (layer * self.positions() + pos) * top_k;
        &self.actual[at..at + top_k]
    }

    /// Sum and count, over the decode rows of layers 1 and up, of the
    /// `k`-th highest score of `predictor`'s picks (the score thresholds
    /// are multiples of its mean).
    pub fn kth_score_sum(
        &self,
        predictor: usize,
        layers: usize,
        k: usize,
    ) -> (f64, u64) {
        let rows = &self.predicted[predictor];
        let (mut sum, mut n) = (0.0f64, 0u64);
        let mut sorted = Vec::with_capacity(rows.width);
        for l in 1..layers {
            for t in 0..self.decode_tokens {
                let at = (l * self.decode_tokens + t) * rows.width;
                sorted.clear();
                sorted.extend_from_slice(&rows.scores[at..at + rows.width]);
                sorted.sort_by(|a, b| b.total_cmp(a));
                if let Some(&s) = sorted.get(k - 1) {
                    sum += f64::from(s);
                    n += 1;
                }
            }
        }
        (sum, n)
    }
}

/// The cache the replay reproduces.
#[derive(Clone, Debug)]
pub struct ReplayConfig {
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    pub slots: usize,
    /// As [`SlotPolicy::new`] takes it; the loader's clamp
    /// ([`SlotPolicy::lru_share_for`]) is the caller's to apply.
    pub lru_share: f64,
    /// Prompt tokens per prefill pass (`Qwen4ExpModel::prefill_chunk`).
    pub prefill_chunk: usize,
    /// Promotion passes on (`LILY_EXPERT_ADAPT`, on by default).
    pub adapt: bool,
    /// Keep each policy's slab and usage across prompts instead of a fresh
    /// fill per prompt.
    pub carry: bool,
}

/// How the re-ranked pinned set orders the slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    /// The prompt's own prefill counts, ties by the global order.
    Own,
    /// Own and global counts each normalised to sum 1 and mixed 50/50,
    /// ties by the global order (`route_predict::residency_sets`).
    Mix,
}

/// Which of a prediction row's picks a prefetch reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PredictedSet {
    /// The `n` highest-scored.
    Top(usize),
    /// Those scoring at least this.
    AtLeast(f32),
}

/// A cross-layer prefetch: after a decode token's request for layer `l`
/// is resolved, the non-resident picks of `predictor`'s prediction for
/// layer `l + 1` are looked up at that layer's tick (taking LRU slots as a
/// real load would), highest score first, at most `budget` per layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prefetch {
    pub predictor: usize,
    pub set: PredictedSet,
    pub budget: Option<usize>,
    /// The lowest layer prefetched for (at least 1: layer 0 has no source).
    pub from_layer: usize,
}

/// A policy the replay compares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    /// The cache as shipped.
    Live,
    /// At the first decode step, a fresh [`SlotPolicy`] (same slots and
    /// share) and [`LiveUsage`] from the prompt's prefill counts with the
    /// global ranking, then live.
    Rerank(Weighting),
    /// Live plus a cross-layer prefetch during decode.
    Prefetch(Prefetch),
}

/// Request counts of one phase.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseTally {
    pub tokens: u64,
    pub requests: u64,
    /// Distinct (layer, expert) per request, summed.
    pub lookups: u64,
    pub misses: u64,
}

impl PhaseTally {
    fn plus(&mut self, t: &PhaseTally) {
        self.tokens += t.tokens;
        self.requests += t.requests;
        self.lookups += t.lookups;
        self.misses += t.misses;
    }

    fn to_json(&self) -> Value {
        let n = self.tokens.max(1) as f64;
        json!({
            "tokens": self.tokens,
            "requests": self.requests,
            "lookups": self.lookups,
            "misses": self.misses,
            "misses_per_token": self.misses as f64 / n,
            "miss_rate": self.misses as f64 / self.lookups.max(1) as f64,
        })
    }
}

/// Decode counts, with the prefetch's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeTally {
    pub phase: PhaseTally,
    /// Tokens by their count of layers whose request missed, index =
    /// count.
    pub layers_with_miss: Vec<u64>,
    /// Prefetch loads issued.
    pub reads: u64,
    /// Of those, the ones the layer's real request for that token routed
    /// to.
    pub useful: u64,
    /// Predicted-set sizes summed over the (token, layer) rows prefetched
    /// for, and those rows.
    pub predicted: u64,
    pub predicted_rows: u64,
    /// Prefetches given up because no LRU slot was free at the tick.
    pub full: u64,
}

impl DecodeTally {
    fn new(layers: usize) -> Self {
        Self {
            phase: PhaseTally::default(),
            layers_with_miss: vec![0; layers + 1],
            reads: 0,
            useful: 0,
            predicted: 0,
            predicted_rows: 0,
            full: 0,
        }
    }

    fn plus(&mut self, t: &DecodeTally) {
        self.phase.plus(&t.phase);
        for (a, b) in self.layers_with_miss.iter_mut().zip(&t.layers_with_miss) {
            *a += b;
        }
        self.reads += t.reads;
        self.useful += t.useful;
        self.predicted += t.predicted;
        self.predicted_rows += t.predicted_rows;
        self.full += t.full;
    }

    /// Prefetch loads whose slice the layer's request did not route to.
    pub fn wasted(&self) -> u64 {
        self.reads - self.useful
    }

    fn to_json(&self, prefetch: bool) -> Value {
        let mut out = self.phase.to_json();
        let hist = &self.layers_with_miss;
        let n: u64 = hist.iter().sum();
        let mean = hist.iter().enumerate().map(|(i, &c)| i as u64 * c).sum::<u64>()
            as f64
            / n.max(1) as f64;
        let len = hist.iter().rposition(|&c| c > 0).map_or(1, |i| i + 1);
        out["layers_with_miss_per_token"] =
            json!({ "mean": mean, "histogram": &hist[..len] });
        if prefetch {
            let t = self.phase.tokens.max(1) as f64;
            out["prefetch"] = json!({
                "reads": self.reads,
                "reads_per_token": self.reads as f64 / t,
                "useful": self.useful,
                "useful_per_token": self.useful as f64 / t,
                "wasted": self.wasted(),
                "wasted_per_token": self.wasted() as f64 / t,
                "precision": self.useful as f64 / self.reads.max(1) as f64,
                "mean_set_size":
                    self.predicted as f64 / self.predicted_rows.max(1) as f64,
                "gave_up_full": self.full,
            });
        }
        out
    }
}

/// One policy's counts over one prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptOutcome {
    pub prefill: PhaseTally,
    pub decode: DecodeTally,
    /// Re-rank only: slices of the new pinned set, and of its LRU region's
    /// initial contents, not resident in the slab before (one-off reads).
    pub rerank_pinned_reads: u64,
    pub rerank_lru_reads: u64,
    pub promotions: u64,
}

/// The service thread's state: the policy, the usage, the last request's
/// tick and the requests served.
#[derive(Clone)]
struct CacheState {
    policy: SlotPolicy,
    usage: LiveUsage,
    tick: u64,
    served: u64,
}

impl CacheState {
    fn fresh(cfg: &ReplayConfig, ranking: &UsageRanking) -> Result<Self> {
        Ok(Self {
            policy: SlotPolicy::new(
                cfg.slots,
                cfg.layers,
                cfg.experts,
                ranking,
                cfg.lru_share,
            )?,
            usage: LiveUsage::new(ranking, cfg.slots, cfg.top_k),
            tick: 0,
            served: 0,
        })
    }

    /// One request as `expert_cache::serve` resolves it: `ids` deduplicated,
    /// `count` the routed ids the pass wrote (which decides decode- or
    /// prefill-class). Returns the misses and the hits on `prefetched`.
    fn request(
        &mut self,
        cfg: &ReplayConfig,
        layer: usize,
        ids: &[u32],
        count: usize,
        prefetched: &[u32],
    ) -> Result<(u64, u64)> {
        self.tick += 1;
        let decode = count <= DECODE_IDS;
        let (mut misses, mut useful) = (0u64, 0u64);
        for &e in ids {
            let expert = e as usize;
            ensure!(expert < cfg.experts, "routed id {e} out of range");
            self.usage.record(layer, expert, decode);
            match self.policy.lookup(layer, expert, self.tick) {
                Lookup::Hit(_) => useful += prefetched.contains(&e) as u64,
                Lookup::Miss { .. } => misses += 1,
                Lookup::Full => bail!(
                    "no slot free for layer {layer} expert {expert} (LRU region too \
                     small; the service thread aborts here)"
                ),
            }
        }
        self.served += 1;
        if cfg.adapt && self.served.is_multiple_of(PROMOTE_EVERY) {
            self.usage.promote(&mut self.policy, self.tick);
        }
        Ok((misses, useful))
    }

    /// The prompt's prefill: one request per layer per chunk, with the
    /// union of the chunk's rows' ids.
    fn prefill(
        &mut self,
        cfg: &ReplayConfig,
        trace: &PromptTrace,
    ) -> Result<PhaseTally> {
        let mut tally =
            PhaseTally { tokens: trace.prefill_tokens as u64, ..PhaseTally::default() };
        let mut ids = Vec::new();
        let mut start = 0;
        while start < trace.prefill_tokens {
            let end = (start + cfg.prefill_chunk).min(trace.prefill_tokens);
            for layer in 0..cfg.layers {
                ids.clear();
                for pos in start..end {
                    ids.extend_from_slice(trace.routed(layer, pos, cfg.top_k));
                }
                ids.sort_unstable();
                ids.dedup();
                let count = (end - start) * cfg.top_k;
                let (misses, _) = self.request(cfg, layer, &ids, count, &[])?;
                tally.requests += 1;
                tally.lookups += ids.len() as u64;
                tally.misses += misses;
            }
            start = end;
        }
        Ok(tally)
    }

    /// [`Self::prefill`] and the promotions it made.
    fn prefill_counted(
        &mut self,
        cfg: &ReplayConfig,
        trace: &PromptTrace,
    ) -> Result<(PhaseTally, u64)> {
        let before = self.usage.promotions();
        let tally = self.prefill(cfg, trace)?;
        Ok((tally, self.usage.promotions() - before))
    }

    /// The re-rank at the end of the prefill: returns the one-off reads of
    /// the new pinned set and of its LRU region's initial contents.
    fn rerank(
        &mut self,
        cfg: &ReplayConfig,
        global: &UsageRanking,
        trace: &PromptTrace,
        weighting: Weighting,
    ) -> Result<(u64, u64)> {
        let own = routing_counts(
            &trace.actual,
            cfg.layers,
            cfg.experts,
            cfg.top_k,
            trace.positions(),
            0..trace.prefill_tokens,
        );
        let ranking = rerank_ranking(&own, global, weighting)?;
        let policy = SlotPolicy::new(
            cfg.slots,
            cfg.layers,
            cfg.experts,
            &ranking,
            cfg.lru_share,
        )?;
        let (mut pinned, mut lru) = (0u64, 0u64);
        for (slot, layer, expert) in policy.initial_fill() {
            if self.policy.slot_of(layer, expert).is_none() {
                if policy.is_pinned(slot) {
                    pinned += 1;
                } else {
                    lru += 1;
                }
            }
        }
        self.policy = policy;
        self.usage = LiveUsage::new(&ranking, cfg.slots, cfg.top_k);
        Ok((pinned, lru))
    }

    /// The prompt's decode, one request per layer per token, with the
    /// prefetch when `prefetch` is set.
    fn decode(
        &mut self,
        cfg: &ReplayConfig,
        trace: &PromptTrace,
        prefetch: Option<&Prefetch>,
    ) -> Result<DecodeTally> {
        let (layers, k) = (cfg.layers, cfg.top_k);
        let mut tally = DecodeTally::new(layers);
        let mut ids = Vec::with_capacity(k);
        let mut prefetched: Vec<u32> = Vec::new();
        let mut picks = Vec::new();
        for t in 0..trace.decode_tokens {
            let pos = trace.prefill_tokens + t;
            let mut missed_layers = 0usize;
            prefetched.clear();
            for layer in 0..layers {
                ids.clear();
                ids.extend_from_slice(trace.routed(layer, pos, k));
                ids.sort_unstable();
                ids.dedup();
                let (misses, useful) =
                    self.request(cfg, layer, &ids, k, &prefetched)?;
                tally.phase.requests += 1;
                tally.phase.lookups += ids.len() as u64;
                tally.phase.misses += misses;
                tally.useful += useful;
                missed_layers += (misses > 0) as usize;
                prefetched.clear();
                let next = layer + 1;
                let Some(p) = prefetch else { continue };
                if next >= layers || next < p.from_layer.max(1) {
                    continue;
                }
                let rows = &trace.predicted[p.predictor];
                let at = (next * trace.decode_tokens + t) * rows.width;
                let (row_ids, row_scores) =
                    (&rows.ids[at..at + rows.width], &rows.scores[at..at + rows.width]);
                match p.set {
                    PredictedSet::Top(n) => {
                        top_by_score(row_ids, row_scores, n, &mut picks)
                    }
                    PredictedSet::AtLeast(min) => {
                        top_by_score(row_ids, row_scores, rows.width, &mut picks);
                        let keep = row_scores.iter().filter(|&&s| s >= min).count();
                        picks.truncate(keep);
                    }
                }
                tally.predicted += picks.len() as u64;
                tally.predicted_rows += 1;
                // The prefetch runs at the next request's tick: its slots
                // are then protected from that request's own misses, as a
                // slot the pass is about to read would be.
                let tick = self.tick + 1;
                for &e in &picks {
                    if p.budget.is_some_and(|b| prefetched.len() >= b) {
                        break;
                    }
                    let expert = e as usize;
                    ensure!(expert < cfg.experts, "predicted id {e} out of range");
                    if self.policy.slot_of(next, expert).is_some() {
                        continue;
                    }
                    match self.policy.lookup(next, expert, tick) {
                        Lookup::Miss { .. } => {
                            tally.reads += 1;
                            prefetched.push(e);
                        }
                        Lookup::Hit(_) => unreachable!("checked not resident"),
                        Lookup::Full => {
                            tally.full += 1;
                            break;
                        }
                    }
                }
            }
            tally.phase.tokens += 1;
            tally.layers_with_miss[missed_layers] += 1;
        }
        Ok(tally)
    }
}

/// The ranking a re-rank pins by, as counts whose [`UsageRanking::ranked`]
/// order is the weighting's (both break ties by the global ranking's
/// order) and whose magnitudes serve as the new [`LiveUsage`] prior: own,
/// `own * (N + 1) + (N - global rank)`; mix, the 50/50 mix scaled by 2^40
/// and rounded, times `N + 1`, plus the same tie term (`N` the slices).
pub fn rerank_ranking(
    own: &[Vec<u64>],
    global: &UsageRanking,
    weighting: Weighting,
) -> Result<UsageRanking> {
    let (layers, experts) = (global.layers, global.experts);
    ensure!(
        own.len() == layers && own.iter().all(|r| r.len() == experts),
        "own counts are not {layers} x {experts}"
    );
    let n = (layers * experts) as u64;
    let mut tie = vec![0u64; layers * experts];
    for (rank, (l, e)) in global.ranked().into_iter().enumerate() {
        tie[l * experts + e] = n - rank as u64;
    }
    let own_flat: Vec<u64> = own.iter().flatten().copied().collect();
    let primary: Vec<u64> = match weighting {
        Weighting::Own => own_flat,
        Weighting::Mix => {
            let own_sum = own_flat.iter().sum::<u64>().max(1) as f64;
            let global_sum = global.counts.iter().flatten().sum::<u64>().max(1) as f64;
            own_flat
                .iter()
                .zip(global.counts.iter().flatten())
                .map(|(&o, &g)| {
                    let mix = 0.5 * o as f64 / own_sum + 0.5 * g as f64 / global_sum;
                    (mix * (1u64 << 40) as f64).round() as u64
                })
                .collect()
        }
    };
    let counts = (0..layers)
        .map(|l| {
            (0..experts)
                .map(|e| {
                    let i = l * experts + e;
                    primary[i] * (n + 1) + tie[i]
                })
                .collect()
        })
        .collect();
    Ok(UsageRanking { layers, experts, counts })
}

/// Replays prompts through a set of named policies.
pub struct Replay<'a> {
    cfg: ReplayConfig,
    global: &'a UsageRanking,
    policies: Vec<(String, Policy)>,
    /// Per policy, the state carried to the next prompt (`carry`).
    carried: Vec<Option<CacheState>>,
    /// `[policy][prompt]`.
    outcomes: Vec<Vec<PromptOutcome>>,
}

impl<'a> Replay<'a> {
    pub fn new(
        cfg: ReplayConfig,
        global: &'a UsageRanking,
        policies: Vec<(String, Policy)>,
    ) -> Result<Self> {
        ensure!(
            global.layers == cfg.layers && global.experts == cfg.experts,
            "global ranking is {} x {}, the replay {} x {}",
            global.layers,
            global.experts,
            cfg.layers,
            cfg.experts
        );
        ensure!(cfg.prefill_chunk > 0 && cfg.top_k > 0, "empty chunk or top-k");
        let n = policies.len();
        Ok(Self {
            cfg,
            global,
            policies,
            carried: vec![None; n],
            outcomes: vec![Vec::new(); n],
        })
    }

    /// Per policy, per prompt so far.
    pub fn outcomes(&self) -> &[Vec<PromptOutcome>] {
        &self.outcomes
    }

    /// Replays one prompt through every policy (in parallel; each policy's
    /// replay is sequential and deterministic).
    pub fn add_prompt(&mut self, trace: &PromptTrace) -> Result<()> {
        let cfg = &self.cfg;
        ensure!(
            trace.actual.len() == cfg.layers * trace.positions() * cfg.top_k,
            "trace routing shape"
        );
        for (_, policy) in &self.policies {
            if let Policy::Prefetch(p) = policy {
                let rows = trace.predicted.get(p.predictor);
                ensure!(
                    rows.is_some_and(|r| r.width > 0
                        && r.ids.len() == cfg.layers * trace.decode_tokens * r.width
                        && r.scores.len() == r.ids.len()),
                    "trace has no predictions for predictor {}",
                    p.predictor
                );
            }
        }
        // A fresh fill per prompt: the prefill is the same for every
        // policy, so it runs once.
        let shared = if cfg.carry {
            None
        } else {
            let mut state = CacheState::fresh(cfg, self.global)?;
            let prefill = state.prefill_counted(cfg, trace)?;
            Some((state, prefill))
        };
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let global = self.global;
        let jobs: Vec<(usize, Policy, Option<CacheState>)> = self
            .policies
            .iter()
            .enumerate()
            .map(|(i, (_, p))| (i, *p, self.carried[i].take()))
            .collect();
        let per = jobs.len().div_ceil(threads).max(1);
        let results: Vec<Result<(usize, PromptOutcome, CacheState)>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = jobs
                    .chunks(per)
                    .map(|chunk| {
                        let shared = shared.as_ref();
                        scope.spawn(move || {
                            chunk
                                .iter()
                                .map(|(i, policy, carried)| {
                                    let (state, prefill) = match (shared, carried) {
                                        (Some((s, p)), _) => {
                                            (s.clone(), Some(p.clone()))
                                        }
                                        (None, Some(s)) => (s.clone(), None),
                                        (None, None) => {
                                            (CacheState::fresh(cfg, global)?, None)
                                        }
                                    };
                                    run_policy(
                                        cfg, global, trace, policy, state, prefill,
                                    )
                                    .map(|(o, s)| (*i, o, s))
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("replay thread panicked"))
                    .collect()
            });
        for r in results {
            let (i, outcome, state) = r?;
            self.outcomes[i].push(outcome);
            if self.cfg.carry {
                self.carried[i] = Some(state);
            }
        }
        Ok(())
    }

    /// The summary: per policy, the counts over all prompts and per
    /// prompt.
    pub fn to_json(&self) -> Value {
        let policies: Vec<Value> = self
            .policies
            .iter()
            .zip(&self.outcomes)
            .map(|((name, policy), outcomes)| {
                let prefetch = matches!(policy, Policy::Prefetch(_));
                let rerank = matches!(policy, Policy::Rerank(_));
                let mut prefill = PhaseTally::default();
                let mut decode = DecodeTally::new(self.cfg.layers);
                let (mut pinned, mut lru, mut promotions) = (0u64, 0u64, 0u64);
                for o in outcomes {
                    prefill.plus(&o.prefill);
                    decode.plus(&o.decode);
                    pinned += o.rerank_pinned_reads;
                    lru += o.rerank_lru_reads;
                    promotions += o.promotions;
                }
                let prompts = outcomes.len().max(1) as f64;
                let mean_misses = outcomes
                    .iter()
                    .map(|o| {
                        o.decode.phase.misses as f64
                            / o.decode.phase.tokens.max(1) as f64
                    })
                    .sum::<f64>()
                    / prompts;
                let mut decode_json = decode.to_json(prefetch);
                decode_json["mean_misses_per_token_over_prompts"] = json!(mean_misses);
                let mut out = json!({
                    "name": name,
                    "policy": describe(policy),
                    "prefill": prefill.to_json(),
                    "decode": decode_json,
                    "promotions": promotions,
                });
                if rerank {
                    out["rerank"] = json!({
                        "pinned_reads": pinned,
                        "pinned_reads_per_prompt": pinned as f64 / prompts,
                        "lru_fill_reads": lru,
                        "lru_fill_reads_per_prompt": lru as f64 / prompts,
                    });
                }
                out["per_prompt"] = outcomes
                    .iter()
                    .map(|o| {
                        let mut p = json!({
                            "prefill": o.prefill.to_json(),
                            "decode": o.decode.to_json(prefetch),
                            "promotions": o.promotions,
                        });
                        if rerank {
                            p["rerank"] = json!({
                                "pinned_reads": o.rerank_pinned_reads,
                                "lru_fill_reads": o.rerank_lru_reads,
                            });
                        }
                        p
                    })
                    .collect();
                out
            })
            .collect();
        Value::Array(policies)
    }
}

/// The policy's parameters as JSON.
fn describe(policy: &Policy) -> Value {
    match policy {
        Policy::Live => json!({ "kind": "live" }),
        Policy::Rerank(w) => json!({
            "kind": "rerank",
            "weighting": match w { Weighting::Own => "own", Weighting::Mix => "mix" },
        }),
        Policy::Prefetch(p) => json!({
            "kind": "prefetch",
            "predictor": p.predictor,
            "set": match p.set {
                PredictedSet::Top(n) => json!({ "top": n }),
                PredictedSet::AtLeast(s) => json!({ "min_score": s }),
            },
            "budget_per_layer": p.budget,
            "from_layer": p.from_layer,
        }),
    }
}

/// One policy over one prompt from `state`: the prefill (unless `prefill`
/// already holds its counts and promotions, replayed on a shared state),
/// the re-rank if any, the decode.
fn run_policy(
    cfg: &ReplayConfig,
    global: &UsageRanking,
    trace: &PromptTrace,
    policy: &Policy,
    mut state: CacheState,
    prefill: Option<(PhaseTally, u64)>,
) -> Result<(PromptOutcome, CacheState)> {
    let (prefill, mut promotions) = match prefill {
        Some(done) => done,
        None => state.prefill_counted(cfg, trace)?,
    };
    let (mut pinned, mut lru) = (0, 0);
    if let Policy::Rerank(w) = policy {
        (pinned, lru) = state.rerank(cfg, global, trace, *w)?;
    }
    // The re-rank replaces the usage and its promotion count.
    let before_decode = state.usage.promotions();
    let prefetch = match policy {
        Policy::Prefetch(p) => Some(p),
        _ => None,
    };
    let decode = state.decode(cfg, trace, prefetch)?;
    promotions += state.usage.promotions() - before_decode;
    Ok((
        PromptOutcome {
            prefill,
            decode,
            rerank_pinned_reads: pinned,
            rerank_lru_reads: lru,
            promotions,
        },
        state,
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/expert_replay.rs"]
mod tests;
