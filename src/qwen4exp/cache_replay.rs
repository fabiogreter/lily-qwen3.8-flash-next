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
//! counts as decode there too. Decoding is replayed one token per pass:
//! speculative verify passes (the pending token plus drafts) route a few
//! more experts per pass, rejected drafts included, which a trace does not
//! hold.
//!
//! A decode-time prefetch can be replayed from logged predictions
//! (`lily-trace-routing --predict`), as `expert_replay` replays it for one
//! prompt: after a decoded token's request for layer `l`, the predicted
//! experts for layer `l + d` that are not resident are looked up at the
//! next request's tick, taking LRU slots as a real load would. Reads land
//! instantly here; whether they would in time is for a live run.

use std::ops::Range;

use anyhow::{Result, ensure};

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
}

/// A prefetch during decode: after layer `l`'s request, read the `top`
/// best-scored predictions for layer `l + lookahead` that are not resident,
/// at most `budget` per layer. Lookahead 0 reads for every layer at the
/// token's start instead, from the predictions' first plane (a whole-token
/// predictor such as [`TokenRouting`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchReplay {
    /// 0 (token start, every layer), 1 or 2.
    pub lookahead: usize,
    pub top: usize,
    pub budget: Option<usize>,
}

/// Logged predictions of one session (`lily-trace-routing --predict`):
/// `[2][layers][tokens][width]` expert ids, row-wise best first; plane
/// `d - 1` holds lookahead `d`.
#[derive(Clone, Copy, Debug)]
pub struct Predictions<'a> {
    pub ids: &'a [u16],
    pub width: usize,
}

/// What a replay counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    pub prefill_tokens: u64,
    pub prefill_lookups: u64,
    pub prefill_misses: u64,
    pub decode_tokens: u64,
    pub decode_lookups: u64,
    pub decode_misses: u64,
    /// Summed over decoded tokens: the layers with at least one miss.
    pub decode_layers_missed: u64,
    /// Lookups the policy could not place (the LRU region too small).
    pub full: u64,
    pub promotions: u64,
    /// Prefetch loads issued.
    pub prefetch_reads: u64,
    /// Of those, the ones the target layer's request then routed to.
    pub prefetch_useful: u64,
    /// Prefetches given up because no LRU slot was free at the tick.
    pub prefetch_full: u64,
    /// Decode misses whose expert the same layer routed to within the last
    /// 64 (1 024) positions of the session: what a predictor from recent
    /// routing could have known.
    pub decode_misses_recent_64: u64,
    pub decode_misses_recent_1024: u64,
}

impl ReplayCounts {
    pub fn plus(&mut self, o: &ReplayCounts) {
        self.prefill_tokens += o.prefill_tokens;
        self.prefill_lookups += o.prefill_lookups;
        self.prefill_misses += o.prefill_misses;
        self.decode_tokens += o.decode_tokens;
        self.decode_lookups += o.decode_lookups;
        self.decode_misses += o.decode_misses;
        self.decode_layers_missed += o.decode_layers_missed;
        self.full += o.full;
        self.promotions += o.promotions;
        self.prefetch_reads += o.prefetch_reads;
        self.prefetch_useful += o.prefetch_useful;
        self.prefetch_full += o.prefetch_full;
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
            usage: LiveUsage::new(ranking, config.slots, top_k),
            tick: 0,
            served: 0,
            ids: Vec::new(),
        })
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

    /// The prefetch for layer `target` of the decoded token at `pos`, after
    /// the request of layer `target - lookahead`: the experts read go to
    /// `prefetched`.
    fn prefetch(
        &mut self,
        p: PrefetchReplay,
        pred: &Predictions<'_>,
        (target, pos, tokens): (usize, usize, usize),
        prefetched: &mut Vec<u32>,
        counts: &mut ReplayCounts,
    ) -> Result<()> {
        let plane = p.lookahead.max(1) - 1;
        let at = ((plane * self.layers + target) * tokens + pos) * pred.width;
        // The read happens at the next request's tick, so that request's own
        // misses cannot take the slot, as they cannot take one a pass reads.
        let tick = self.tick + 1;
        for &e in &pred.ids[at..at + p.top.min(pred.width)] {
            if p.budget.is_some_and(|b| prefetched.len() >= b) {
                break;
            }
            let expert = usize::from(e);
            ensure!(expert < self.experts, "predicted id {e} out of range");
            if self.policy.slot_of(target, expert).is_some() {
                continue;
            }
            match self.policy.lookup(target, expert, tick) {
                Lookup::Miss { .. } => {
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
    /// tokens each; decode passes one token each. `predicted` drives the
    /// prefetch, and is required when the configuration has one.
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
                    pred.ids.len() == 2 * self.layers * tokens * pred.width,
                    "predictions hold {} ids, 2 x {} layers x {tokens} tokens x {} \
                     expected",
                    pred.ids.len(),
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
            for pos in r.decode.clone() {
                let mut missed = 0;
                if let Some((p, pred)) = &prefetch
                    && p.lookahead == 0
                {
                    for (target, read) in prefetched.iter_mut().enumerate() {
                        self.prefetch(*p, pred, (target, pos, tokens), read, counts)?;
                    }
                }
                for layer in 0..self.layers {
                    let ids = &routing[at(layer, pos)..at(layer, pos + 1)];
                    pass.clear();
                    pass.extend_from_slice(ids);
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
                    for &e in &self.missed {
                        let last = self.last_seen[layer * experts + e as usize];
                        if last > 0 && pos + 1 - last <= 64 {
                            counts.decode_misses_recent_64 += 1;
                        }
                        if last > 0 && pos + 1 - last <= 1024 {
                            counts.decode_misses_recent_1024 += 1;
                        }
                    }
                    for &e in &pass {
                        self.last_seen[layer * experts + e as usize] = pos + 1;
                    }
                    if misses > 0 {
                        missed += 1;
                    }
                    if let Some((p, pred)) = &prefetch
                        && p.lookahead > 0
                    {
                        let target = layer + p.lookahead;
                        if target < self.layers {
                            let mut read = std::mem::take(&mut prefetched[target]);
                            self.prefetch(
                                *p,
                                pred,
                                (target, pos, tokens),
                                &mut read,
                                counts,
                            )?;
                            prefetched[target] = read;
                        }
                    }
                }
                counts.decode_tokens += 1;
                counts.decode_layers_missed += missed;
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
