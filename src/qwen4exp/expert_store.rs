//! The host side of the expert cache (`docs/low-ram-experts.md`): serves one
//! routed expert's nine weight regions (gate, up and down projections, each a
//! `{weight, scales, biases}` triple) straight from the checkpoint shards
//! with positioned reads, and decides which expert lives in which of a fixed
//! number of GPU slots. Nothing here touches Metal; the model wires the
//! slots, the slot tables and the round trip per MoE layer.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use serde::Deserialize;

use super::config::Qwen4ExpConfig;
use crate::safetensors::{Checkpoint, SafetensorsDType};

/// Weight regions per expert: three projections times weight, scales, biases.
pub const REGIONS: usize = 9;

/// Tensor name suffixes of the nine regions under
/// `model.language_model.layers.{L}.mlp.experts.`, in region order.
pub const REGION_NAMES: [&str; REGIONS] = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
];

/// Tensor name prefix of the language model's decoder layers.
const LAYER_PREFIX: &str = "model.language_model.layers.";

/// One contiguous byte range of an expert inside a checkpoint shard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Region {
    /// Index into [`ExpertStore::shard_path`].
    pub shard: usize,
    /// Absolute byte offset in the shard file.
    pub offset: u64,
    /// Bytes.
    pub len: usize,
}

/// Where one expert's nine regions live, in [`REGION_NAMES`] order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertSlice {
    pub regions: [Region; REGIONS],
}

/// One expert tensor's location: its shard, the start of the tensor data,
/// and the byte count of one expert's rows (expert `e` starts at
/// `start + e * per_expert`).
#[derive(Clone, Copy, Debug)]
struct TensorLoc {
    shard: usize,
    start: u64,
    per_expert: usize,
}

/// Serves individual expert slices of the routed experts from the checkpoint
/// on demand. Every shard holding expert tensors is opened once at
/// construction; reads are plain `pread`s into caller memory (typically a
/// shared-storage Metal buffer), never a mapping.
pub struct ExpertStore {
    shards: Vec<(PathBuf, File)>,
    /// `[layer][region]`.
    tensors: Vec<[TensorLoc; REGIONS]>,
    experts: usize,
    region_lens: [usize; REGIONS],
}

impl ExpertStore {
    /// Locates the expert tensors of every language-model layer (the MTP
    /// head's experts are not served) and checks their shapes against the
    /// config: `[E, I, h/8]` U32 codes and `[E, I, h/group]` BF16 scales and
    /// biases for gate and up, `[E, h, I/8]` and `[E, h, I/group]` for down.
    pub fn open(ckpt: &Checkpoint, config: &Qwen4ExpConfig) -> Result<Self> {
        let (e, i, h) = (
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
        );
        let group = config.quantization.group_size;
        ensure!(
            config.quantization.bits == 4,
            "expert store expects 4-bit codes, checkpoint has {} bits",
            config.quantization.bits
        );
        // Expected `[shape[1], shape[2]]` and dtype per region.
        let expected: [(usize, usize, SafetensorsDType); REGIONS] = [
            (i, h / 8, SafetensorsDType::U32),
            (i, h / group, SafetensorsDType::BF16),
            (i, h / group, SafetensorsDType::BF16),
            (i, h / 8, SafetensorsDType::U32),
            (i, h / group, SafetensorsDType::BF16),
            (i, h / group, SafetensorsDType::BF16),
            (h, i / 8, SafetensorsDType::U32),
            (h, i / group, SafetensorsDType::BF16),
            (h, i / group, SafetensorsDType::BF16),
        ];

        let mut shards: Vec<(PathBuf, File)> = Vec::new();
        let mut shard_index: HashMap<PathBuf, usize> = HashMap::new();
        let mut region_lens = [0usize; REGIONS];
        let mut tensors = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let mut locs = [TensorLoc { shard: 0, start: 0, per_expert: 0 }; REGIONS];
            for (r, suffix) in REGION_NAMES.iter().enumerate() {
                let name = format!("{LAYER_PREFIX}{layer}.mlp.experts.{suffix}");
                let meta = ckpt
                    .meta(&name)
                    .with_context(|| format!("tensor {name} not in checkpoint"))?;
                let (rows, cols, ref dtype) = expected[r];
                ensure!(
                    meta.shape.as_slice() == [e, rows, cols] && meta.dtype == *dtype,
                    "{name}: expected {:?} [{e}, {rows}, {cols}], got {:?} {:?}",
                    dtype,
                    meta.dtype,
                    meta.shape
                );
                let elem = meta
                    .dtype
                    .size()
                    .with_context(|| format!("{name}: unsupported dtype"))?;
                let per_expert = rows * cols * elem;
                ensure!(
                    meta.byte_len() == e * per_expert,
                    "{name}: {} bytes for {e} experts of {per_expert}",
                    meta.byte_len()
                );
                if layer == 0 {
                    region_lens[r] = per_expert;
                }
                ensure!(
                    region_lens[r] == per_expert,
                    "{name}: {per_expert} bytes per expert, layer 0 has {}",
                    region_lens[r]
                );
                let shard = match shard_index.get(meta.shard()) {
                    Some(&idx) => idx,
                    None => {
                        let file = File::open(meta.shard())
                            .with_context(|| format!("opening {}", meta.shard().display()))?;
                        shards.push((meta.shard().to_path_buf(), file));
                        let idx = shards.len() - 1;
                        shard_index.insert(meta.shard().to_path_buf(), idx);
                        idx
                    }
                };
                locs[r] = TensorLoc { shard, start: meta.start(), per_expert };
            }
            tensors.push(locs);
        }
        Ok(Self { shards, tensors, experts: e, region_lens })
    }

    /// Language-model layers served.
    pub fn layers(&self) -> usize {
        self.tensors.len()
    }

    /// Routed experts per layer.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes of each of one expert's nine regions, the same for every layer
    /// and expert.
    pub fn region_lens(&self) -> [usize; REGIONS] {
        self.region_lens
    }

    /// Bytes of one whole expert slice.
    pub fn slice_bytes(&self) -> usize {
        self.region_lens.iter().sum()
    }

    /// Shard files opened by the store, addressed by [`Region::shard`].
    pub fn shard_path(&self, shard: usize) -> &Path {
        &self.shards[shard].0
    }

    /// Where the nine regions of `expert` in `layer` live.
    pub fn slice(&self, layer: usize, expert: usize) -> ExpertSlice {
        debug_assert!(layer < self.layers() && expert < self.experts);
        let locs = &self.tensors[layer];
        let regions = std::array::from_fn(|r| Region {
            shard: locs[r].shard,
            offset: locs[r].start + (expert * locs[r].per_expert) as u64,
            len: locs[r].per_expert,
        });
        ExpertSlice { regions }
    }

    /// Reads the nine regions of `expert` in `layer` into `dst`, one
    /// destination per region in [`REGION_NAMES`] order; each destination
    /// must be exactly its region's length. Positioned reads, no
    /// intermediate buffer.
    pub fn read_into(
        &self,
        layer: usize,
        expert: usize,
        dst: [&mut [u8]; REGIONS],
    ) -> Result<()> {
        ensure!(
            layer < self.layers() && expert < self.experts,
            "expert ({layer}, {expert}) outside {} x {}",
            self.layers(),
            self.experts
        );
        let slice = self.slice(layer, expert);
        for (r, buf) in dst.into_iter().enumerate() {
            let region = slice.regions[r];
            ensure!(
                buf.len() == region.len,
                "destination for layer {layer} expert {expert} {} is {} bytes, region is {}",
                REGION_NAMES[r],
                buf.len(),
                region.len
            );
            let (path, file) = &self.shards[region.shard];
            file.read_exact_at(buf, region.offset).with_context(|| {
                format!(
                    "reading layer {layer} expert {expert} {} from {}",
                    REGION_NAMES[r],
                    path.display()
                )
            })?;
        }
        Ok(())
    }
}

/// Routing counts per (layer, expert), as `lily-experts` writes them, from
/// which the slot placement is derived.
#[derive(Clone, Deserialize, Debug)]
pub struct UsageRanking {
    pub layers: usize,
    pub experts: usize,
    /// `counts[layer][expert]`.
    pub counts: Vec<Vec<u64>>,
}

impl UsageRanking {
    /// Reads a `lily-experts` JSON file and checks its dimensions.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let ranking: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        ranking.validate()?;
        Ok(ranking)
    }

    /// A ranking with no information: every count zero, so [`Self::ranked`]
    /// is layer-major, expert-minor order.
    pub fn uniform(layers: usize, experts: usize) -> Self {
        Self { layers, experts, counts: vec![vec![0; experts]; layers] }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.counts.len() == self.layers
                && self.counts.iter().all(|row| row.len() == self.experts),
            "usage counts are not {} x {}",
            self.layers,
            self.experts
        );
        Ok(())
    }

    /// Every (layer, expert) by count descending, ties by (layer, expert)
    /// ascending.
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let mut all: Vec<(u64, usize, usize)> = self
            .counts
            .iter()
            .enumerate()
            .flat_map(|(layer, row)| {
                row.iter().enumerate().map(move |(expert, &n)| (n, layer, expert))
            })
            .collect();
        all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        all.into_iter().map(|(_, layer, expert)| (layer, expert)).collect()
    }
}

/// What [`SlotPolicy::lookup`] found for an expert.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lookup {
    /// The expert is resident in this slot.
    Hit(u32),
    /// The expert was assigned `slot`; the caller must read it in. `evicted`
    /// is the (layer, expert) the slot held before, if any.
    Miss { slot: u32, evicted: Option<(usize, usize)> },
    /// Every LRU slot was already used at this tick (or there are none), so
    /// no slot can be reassigned without breaking another expert routed in
    /// the same step. The caller must fail the step or grow the LRU region;
    /// the mirrors are unchanged.
    Full,
}

/// Placement of expert slices into `n_slots` GPU slots: the top of a usage
/// ranking is pinned into slots `0..pinned`, the rest form an LRU region
/// for cold experts, which prefill's sweep over every expert then cannot
/// evict from the pinned set. Host mirrors of the GPU slot tables live
/// here; the model uploads [`Self::slot_table`] after each change.
pub struct SlotPolicy {
    experts: usize,
    pinned: usize,
    /// `[layer * experts + expert]` -> slot or [`Self::NONE`].
    slot_of: Vec<u32>,
    /// `[slot]` -> resident (layer, expert).
    slice_in: Vec<Option<(usize, usize)>>,
    /// `[slot]` -> tick of the last lookup that touched it; `None` since the
    /// initial fill.
    last_used: Vec<Option<u64>>,
}

impl SlotPolicy {
    /// Slot table entry of an expert that is not resident.
    pub const NONE: u32 = u32::MAX;

    /// Pins the top `n_slots * (1 - lru_share)` (rounded) ranked slices into
    /// slots `0..pinned` and prefills the LRU slots with the next ranked
    /// ones, in ranking order.
    pub fn new(
        n_slots: usize,
        layers: usize,
        experts: usize,
        ranking: &UsageRanking,
        lru_share: f64,
    ) -> Result<Self> {
        ensure!(
            ranking.layers == layers && ranking.experts == experts,
            "usage ranking is {} x {}, model has {layers} x {experts}",
            ranking.layers,
            ranking.experts
        );
        ranking.validate()?;
        ensure!(
            (0.0..=1.0).contains(&lru_share),
            "lru_share {lru_share} outside [0, 1]"
        );
        ensure!(
            n_slots < Self::NONE as usize,
            "{n_slots} slots do not fit a u32 slot table"
        );
        let ranked = ranking.ranked();
        let pinned = ((n_slots as f64) * (1.0 - lru_share)).round() as usize;
        let pinned = pinned.min(n_slots).min(ranked.len());
        let mut policy = Self {
            experts,
            pinned,
            slot_of: vec![Self::NONE; layers * experts],
            slice_in: vec![None; n_slots],
            last_used: vec![None; n_slots],
        };
        for (slot, &(layer, expert)) in ranked.iter().take(n_slots).enumerate() {
            policy.assign(slot, layer, expert);
        }
        Ok(policy)
    }

    /// GPU slots.
    pub fn n_slots(&self) -> usize {
        self.slice_in.len()
    }

    /// Slots `0..pinned` never change after construction.
    pub fn pinned(&self) -> usize {
        self.pinned
    }

    /// The slot holding `expert` of `layer`, if resident.
    pub fn slot_of(&self, layer: usize, expert: usize) -> Option<u32> {
        match self.slot_of[layer * self.experts + expert] {
            Self::NONE => None,
            slot => Some(slot),
        }
    }

    /// One layer's slot table (`[experts]`, [`Self::NONE`] for cold
    /// experts), the host mirror of what the GPU remap reads.
    pub fn slot_table(&self, layer: usize) -> &[u32] {
        &self.slot_of[layer * self.experts..(layer + 1) * self.experts]
    }

    /// The (layer, expert) resident in `slot`.
    pub fn slice_in(&self, slot: u32) -> Option<(usize, usize)> {
        self.slice_in[slot as usize]
    }

    /// Every occupied slot after construction: what to read in before the
    /// first step, as `(slot, layer, expert)`.
    pub fn initial_fill(&self) -> impl Iterator<Item = (u32, usize, usize)> + '_ {
        self.slice_in
            .iter()
            .enumerate()
            .filter_map(|(slot, held)| held.map(|(l, e)| (slot as u32, l, e)))
    }

    /// Resolves `expert` of `layer` for the step `tick`. A hit stamps the
    /// slot; a miss takes an LRU slot not used at `tick`, preferring an
    /// empty one, then one untouched since the initial fill (worst-ranked
    /// first), then the one with the oldest stamp (lowest slot on ties),
    /// assigns the expert to it and updates the mirrors. `tick` must not
    /// decrease between calls.
    pub fn lookup(&mut self, layer: usize, expert: usize, tick: u64) -> Lookup {
        if let Some(slot) = self.slot_of(layer, expert) {
            self.last_used[slot as usize] = Some(tick);
            return Lookup::Hit(slot);
        }
        let n_slots = self.n_slots();
        let victim = (self.pinned..n_slots)
            .filter(|&s| self.last_used[s] != Some(tick))
            .min_by_key(|&s| match (self.slice_in[s], self.last_used[s]) {
                (None, _) => (0u8, 0u64, s),
                (Some(_), None) => (1, 0, n_slots - s),
                (Some(_), Some(t)) => (2, t, s),
            });
        let Some(slot) = victim else {
            return Lookup::Full;
        };
        let evicted = self.slice_in[slot];
        if let Some((l, e)) = evicted {
            self.slot_of[l * self.experts + e] = Self::NONE;
        }
        self.assign(slot, layer, expert);
        self.last_used[slot] = Some(tick);
        Lookup::Miss { slot: slot as u32, evicted }
    }

    fn assign(&mut self, slot: usize, layer: usize, expert: usize) {
        self.slot_of[layer * self.experts + expert] = slot as u32;
        self.slice_in[slot] = Some((layer, expert));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/expert_store.rs"]
mod tests;
