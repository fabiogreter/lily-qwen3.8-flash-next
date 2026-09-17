//! The expert cache for machines whose memory cannot hold every routed
//! expert (`docs/low-ram-experts.md`): one slab of expert slots on the GPU
//! that every MoE layer's kernels read through a per-layer slot table, and
//! the host side that fills slots from the checkpoint files.
//!
//! A slot holds one expert's three projections in the layout the gather
//! kernels and the grouped GEMM read for a layer's own stacked experts, so
//! the slab is those stacks with `n_slots` experts instead of `E`, shared
//! by all layers: layer `l`'s `MoeWeights` views the slab and carries the
//! table `slot_of[l]` (`U32 [E]`, [`NONE`] where the expert is not
//! resident) that `moe_ffn` remaps the routed ids through.

use anyhow::{Result, ensure};

use crate::metal::MetalContext;
use crate::safetensors::Checkpoint;
use crate::tensor::{DType, Tensor};
use crate::config::QuantizationConfig;
use crate::weights::QuantWeights;

/// The slot-table entry of an expert that is not resident.
pub const NONE: u32 = u32::MAX;

/// Byte regions of one slot, in the order gate codes, gate scales, gate
/// biases, up codes, up scales, up biases, down codes, down scales, down
/// biases (the order `ExpertStore` reads them in).
pub const REGIONS: usize = 9;

pub struct ExpertCache {
    gate: QuantWeights,
    up: QuantWeights,
    down: QuantWeights,
    /// Per layer, `U32 [E]`.
    slot_of: Vec<Tensor>,
    n_slots: usize,
    experts: usize,
    inter: usize,
    hidden: usize,
    quant: QuantizationConfig,
}

impl ExpertCache {
    /// A slab of `n_slots` empty slots for `layers` MoE layers of `experts`
    /// experts with intermediate size `inter` over hidden size `hidden`,
    /// every slot table set to [`NONE`].
    pub fn new(
        ctx: &MetalContext,
        n_slots: usize,
        layers: usize,
        experts: usize,
        inter: usize,
        hidden: usize,
        quant: QuantizationConfig,
    ) -> Result<Self> {
        ensure!(n_slots > 0, "an expert cache needs at least one slot");
        ensure!(quant.bits == 4, "the expert cache is written for Q4 experts");
        let stack = |rows: usize, k: usize| -> Result<QuantWeights> {
            let words = k * quant.bits / 32;
            let groups = k / quant.group_size;
            Ok(QuantWeights {
                codes: Tensor::zeros(ctx, &[rows, words], DType::U32)?,
                scales: Tensor::zeros(ctx, &[rows, groups], DType::BF16)?,
                biases: Tensor::zeros(ctx, &[rows, groups], DType::BF16)?,
                group_size: quant.group_size,
                bits: quant.bits,
            })
        };
        let none = vec![NONE; experts];
        let slot_of = (0..layers)
            .map(|_| {
                Tensor::from_bytes(ctx, bytemuck::cast_slice(&none), &[experts], DType::U32)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            gate: stack(n_slots * inter, hidden)?,
            up: stack(n_slots * inter, hidden)?,
            down: stack(n_slots * hidden, inter)?,
            slot_of,
            n_slots,
            experts,
            inter,
            hidden,
            quant,
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn layers(&self) -> usize {
        self.slot_of.len()
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Bytes of each of a slot's nine regions.
    pub fn region_lens(&self) -> [usize; REGIONS] {
        let (i, h, gs, bits) = (self.inter, self.hidden, self.quant.group_size, self.quant.bits);
        let gu_codes = i * (h * bits / 32) * 4;
        let gu_sb = i * (h / gs) * 2;
        let dn_codes = h * (i * bits / 32) * 4;
        let dn_sb = h * (i / gs) * 2;
        [gu_codes, gu_sb, gu_sb, gu_codes, gu_sb, gu_sb, dn_codes, dn_sb, dn_sb]
    }

    /// Layer `layer`'s view of the slab: its expert stacks (the whole slab)
    /// and its slot table.
    pub fn layer_weights(
        &self,
        layer: usize,
    ) -> Result<(QuantWeights, QuantWeights, QuantWeights, Tensor)> {
        ensure!(layer < self.slot_of.len(), "layer {layer} has no slot table");
        Ok((
            self.gate.view_rows(0, self.n_slots * self.inter)?,
            self.up.view_rows(0, self.n_slots * self.inter)?,
            self.down.view_rows(0, self.n_slots * self.hidden)?,
            self.slot_of[layer].view(0, &[self.experts])?,
        ))
    }

    /// Host-writable bytes of slot `slot`'s nine regions. The caller must
    /// know the GPU is not reading the slot (a slot being refilled is one
    /// the resolve protocol has just evicted while the pass waits).
    ///
    /// # Safety
    ///
    /// The slices alias the shared GPU buffers; no pass reading the slot may
    /// be in flight while they are written.
    pub unsafe fn slot_regions(&self, slot: usize) -> Result<[&mut [u8]; REGIONS]> {
        ensure!(slot < self.n_slots, "slot {slot} of {}", self.n_slots);
        let lens = self.region_lens();
        let region = |t: &Tensor, len: usize| -> &mut [u8] {
            // SAFETY: the slab buffers are shared-storage allocations of
            // n_slots * len bytes; the caller upholds the no-concurrent-reader
            // contract.
            unsafe { core::slice::from_raw_parts_mut(t.contents_ptr().add(slot * len), len) }
        };
        Ok([
            region(&self.gate.codes, lens[0]),
            region(&self.gate.scales, lens[1]),
            region(&self.gate.biases, lens[2]),
            region(&self.up.codes, lens[3]),
            region(&self.up.scales, lens[4]),
            region(&self.up.biases, lens[5]),
            region(&self.down.codes, lens[6]),
            region(&self.down.scales, lens[7]),
            region(&self.down.biases, lens[8]),
        ])
    }

    /// Points layer `layer`'s expert `expert` at `slot` ([`NONE`] to unmap)
    /// in the GPU table. Visible to the next pass the host commits or
    /// releases after the write.
    pub fn set_slot(&self, layer: usize, expert: usize, slot: u32) -> Result<()> {
        ensure!(layer < self.slot_of.len() && expert < self.experts, "no such expert");
        // SAFETY: the table is a shared-storage U32 [E] buffer.
        unsafe {
            self.slot_of[layer].contents_ptr().add(expert * 4).cast::<u32>().write_volatile(slot);
        }
        Ok(())
    }

    /// Fills slots `slot0..slot0 + E` with every expert of the layer at
    /// `prefix` (`...layers.{l}.`) straight from the checkpoint, one read per
    /// tensor, and maps the layer's table to them: the whole-checkpoint
    /// layout used to validate the cache path on machines that fit it.
    pub fn fill_layer_from_checkpoint(
        &self,
        ckpt: &Checkpoint,
        prefix: &str,
        layer: usize,
        slot0: usize,
    ) -> Result<()> {
        ensure!(slot0 + self.experts <= self.n_slots, "layer {layer} does not fit the slab");
        let lens = self.region_lens();
        let stacks = [
            ("gate_proj", &self.gate),
            ("up_proj", &self.up),
            ("down_proj", &self.down),
        ];
        for (si, (name, stack)) in stacks.iter().enumerate() {
            let tensors = [
                (&stack.codes, "weight", lens[si * 3]),
                (&stack.scales, "scales", lens[si * 3 + 1]),
                (&stack.biases, "biases", lens[si * 3 + 2]),
            ];
            for (t, suffix, len) in tensors {
                let full = format!("{prefix}mlp.experts.{name}.{suffix}");
                ckpt.read_with(&full, |meta| {
                    ensure!(
                        meta.byte_len() == len * self.experts,
                        "{full}: {} bytes, the slab expects {}",
                        meta.byte_len(),
                        len * self.experts
                    );
                    // SAFETY: shared-storage slab of n_slots * len bytes; the
                    // range [slot0, slot0 + E) * len lies inside it (checked
                    // above) and nothing reads it during loading.
                    Ok(unsafe {
                        core::slice::from_raw_parts_mut(
                            t.contents_ptr().add(slot0 * len),
                            len * self.experts,
                        )
                    })
                })?;
            }
        }
        for e in 0..self.experts {
            self.set_slot(layer, e, (slot0 + e) as u32)?;
        }
        Ok(())
    }
}
