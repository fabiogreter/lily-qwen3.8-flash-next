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
    }
}

/// The cache's host state, driven one request at a time as the service
/// thread drives it.
pub struct CacheSim {
    config: ReplayConfig,
    layers: usize,
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
            config.lru_share.max((experts + 64) as f64 / config.slots as f64).min(0.5);
        Ok(Self {
            config,
            layers,
            policy: SlotPolicy::new(config.slots, layers, experts, ranking, lru_share)?,
            usage: LiveUsage::new(ranking, config.slots, top_k),
            tick: 0,
            served: 0,
            ids: Vec::new(),
        })
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
        for i in 0..self.ids.len() {
            let expert = self.ids[i] as usize;
            self.usage.record(layer, expert, decode);
            match self.policy.lookup(layer, expert, self.tick) {
                Lookup::Hit(_) => {}
                Lookup::Miss { .. } => misses += 1,
                Lookup::Full => {
                    misses += 1;
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

    /// Replays one recorded session: `routing` is `[layers][tokens][top_k]`
    /// (the layout `lily-trace-routing` writes), `requests` its requests in
    /// order. Prefill passes cover up to a prefill chunk of a request's new
    /// tokens each; decode passes one token each.
    pub fn session(
        &mut self,
        routing: &[u32],
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
        let at = |layer: usize, pos: usize| (layer * tokens + pos) * top_k;
        let mut pass: Vec<u32> = Vec::new();
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
                }
                counts.prefill_tokens += (end - pos) as u64;
                pos = end;
            }
            for pos in r.decode.clone() {
                let mut missed = 0;
                for layer in 0..self.layers {
                    let ids = &routing[at(layer, pos)..at(layer, pos + 1)];
                    pass.clear();
                    pass.extend_from_slice(ids);
                    if self.request(layer, &pass, counts) > 0 {
                        missed += 1;
                    }
                }
                counts.decode_tokens += 1;
                counts.decode_layers_missed += missed;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/cache_replay.rs"]
mod tests;
