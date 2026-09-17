//! The model-agnostic sparse-MoE FFN graph: quantized projection dispatch,
//! the decode gather-GEMV path, and the GPU-resident batched prefill path
//! (router → top-k → counting sort → grouped GEMMs → combine), plus the
//! scratch buffers both paths own.

use anyhow::{Result, ensure};

use crate::kernels::elementwise::{
    gather_rows_bf16, silu_mul_bf16, silu_mul_rows_bf16, split_cols_bf16,
};
use crate::kernels::{moe, quant, skinny};
use crate::metal::{ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};
use crate::weights::{LinearWeights, MoeWeights};

/// The FFN shape a model hands the shared graph.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeDims {
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub hidden: usize,
    pub norm_topk_prob: bool,
}

/// Projects `a` `[m, K]` through an already-stacked weight group as one
/// skinny GEMM into `stack_c` when eligible, returning the `[m, n_total]`
/// view holding the stacked output; `None` when the stack must be projected
/// slice by slice (the caller then dispatches the per-slice projections).
/// `widths` are the slice widths in stack order.
pub(crate) fn project_stack_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    stack_w: &LinearWeights,
    stack_c: &Tensor,
    widths: &[usize],
) -> Result<Option<Tensor>> {
    let m = a.shape()[0];
    let n_total = stack_w.out_features();
    let walk_ok = skinny::q4_block_walk_ok(stack_w.in_features(), stack_w.group_size);
    let fused = widths.iter().sum::<usize>() == n_total
        && stack_w.bits == 4
        && skinny::dense_smallm_routes(m)
        && skinny::stack_route_uniform(m, n_total, widths, walk_ok);
    if !fused {
        return Ok(None);
    }
    // Fusing is bit-identical only when the stack and every slice choose
    // the same skinny variant; staged and register-A reduce in different
    // f32 orders.
    let c = stack_c.view(0, &[m, n_total])?;
    skinny::gemm_skinny_q4_nt(ctx, pass, a, stack_w, &c)?;
    Ok(Some(c))
}

/// Projects an already-stacked weight group as one skinny GEMM when eligible
/// and splits the result into the slice outputs; otherwise dispatches the
/// corresponding per-slice projections. Keeping both arms here prevents the
/// fused and fallback projection lists from drifting.
pub(crate) fn project_stack_or_slices<const N: usize>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    stack_w: &LinearWeights,
    stack_c: &Tensor,
    projections: [(&LinearWeights, &Tensor); N],
    scratch: &Tensor,
) -> Result<()> {
    let m = a.shape()[0];
    let widths: [usize; N] = std::array::from_fn(|i| projections[i].1.numel() / m);
    if let Some(c) = project_stack_fused(ctx, pass, a, stack_w, stack_c, &widths)? {
        pass.level_barrier(&[&c])?;
        let outs: [&Tensor; N] = std::array::from_fn(|i| projections[i].1);
        split_cols_bf16(ctx, pass, &c, &outs)?;
        return Ok(());
    }

    for (w, out) in projections {
        project_mat(ctx, pass, a, w, out, scratch)?;
    }
    Ok(())
}

/// Dispatches a GEMM on the projection's storage precision; quantized weights
/// stage a bf16 dequant into `scratch` first (see `quant::gemm_q4_bf16_nt`),
/// unless the site routes to a skinny small-m kernel (4- and 8-bit). Every
/// route ends with one dispatch writing `c`; ordering after it is the
/// caller's (the dequant staging carries its own internal barrier).
pub(crate) fn project_mat(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &LinearWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    let m = a.shape()[0];
    if skinny::dense_smallm_routes(m) {
        if w.bits == 4 {
            return skinny::gemm_skinny_q4_nt(ctx, pass, a, w, c);
        }
        if w.bits == 8
            && w.group_size.is_multiple_of(8)
            && w.in_features().is_multiple_of(8)
        {
            return skinny::gemm_skinny_q8_nt(ctx, pass, a, w, c);
        }
    }
    quant::gemm_quant_bf16_nt(ctx, pass, a, w, c, scratch)
}

/// Decode-side MoE intermediates: the router distribution and the top-k
/// expert projections, all device-resident (no readback per step).
pub(crate) struct MoeScratch {
    pub router_logits: Tensor,
    pub indices: Tensor,
    /// The routed ids as expert-cache slots (`MoeWeights::slot_of`); aliases
    /// `indices` when the experts are resident.
    pub slots: Tensor,
    pub scores: Tensor,
    pub act: Tensor,
    pub shared_out: Tensor,
    pub shared_gate: Tensor,
}

impl MoeScratch {
    pub fn new(ctx: &MetalContext, dims: &MoeDims) -> Result<Self> {
        let (e, k, i, h) =
            (dims.num_experts, dims.top_k, dims.moe_intermediate, dims.hidden);
        Ok(Self {
            router_logits: Tensor::zeros(ctx, &[e], DType::F32)?,
            indices: Tensor::zeros(ctx, &[k], DType::U32)?,
            slots: Tensor::zeros(ctx, &[k], DType::U32)?,
            scores: Tensor::zeros(ctx, &[k], DType::F32)?,
            act: Tensor::zeros(ctx, &[k, i], DType::BF16)?,
            shared_out: Tensor::zeros(ctx, &[h], DType::BF16)?,
            shared_gate: Tensor::zeros(ctx, &[1], DType::F32)?,
        })
    }
}

/// GPU-resident MoE prefill intermediates (capacity-sized, viewed per chunk):
/// the router distribution, the counting-sort state, the sentinel-padded block
/// maps, and the expert-slot activations (`S = m·top_k` rows).
pub(crate) struct PrefillMoeScratch {
    pub router_logits: Tensor,
    pub shared_out: Tensor,
    pub shared_gate: Tensor,
    pub indices: Tensor,
    /// The routed ids as expert-cache slots for the small-m kernels
    /// (`MoeWeights::slot_of`; the grouped route remaps in its block map).
    pub slots: Tensor,
    pub scores: Tensor,
    pub counts: Tensor,
    pub cursors: Tensor,
    pub offsets: Tensor,
    pub tile_offsets: Tensor,
    pub ids_sorted: Tensor,
    pub slot_of: Tensor,
    pub blocks_gu: Tensor,
    pub blocks_dn: Tensor,
    pub gx: Tensor,
    /// `[S, inter]` expert gate / up outputs (grouped route only).
    pub eg: Tensor,
    pub eu: Tensor,
    /// `[S, inter]` expert activations `silu(gate) * up`, one row per
    /// routed pair; both routes' down input.
    pub ea: Tensor,
    /// `[S, h]` expert down outputs (grouped route; the small-m route
    /// combines in-kernel).
    pub ed: Tensor,
    /// Row-tile capacity bound `ceil(S/T) + min(E, S)` at `tile.rows()` —
    /// covers any expert split without a readback; block-map slots past the
    /// real count carry the sentinel the grouped GEMM kernels skip.
    pub tile_capacity: usize,
    /// The grouped-GEMM row-tile config `tile_capacity` was computed at: the
    /// policy's alloc tile for the backing scratch (the sizing bound), the
    /// policy's per-chunk answer for a chunk view.
    pub tile: quant::MoeTile,
    /// The chunk's grouped-MoE execution route: block-mapped grouped GEMM at
    /// `tile`, or the small-m GEMV fast path. Set together with
    /// `tile`/`tile_capacity` in [`Self::chunk`]; `prefill_moe` re-checks it
    /// against the fixed shipped policy (the desync guard).
    pub route: quant::MoeRoute,
}

/// A row-prefix view of `t` (same buffer, leading dimension shrunk to `m`).
pub(crate) fn prefix_rows(t: &Tensor, m: usize) -> Result<Tensor> {
    let mut shape = t.shape().to_vec();
    shape[0] = m;
    t.view(0, &shape)
}

impl PrefillMoeScratch {
    /// Capacity-sized buffers for chunks of up to `m` rows.
    pub fn new(ctx: &MetalContext, dims: &MoeDims, m: usize) -> Result<Self> {
        let (e, k, mi, h) =
            (dims.num_experts, dims.top_k, dims.moe_intermediate, dims.hidden);
        let bf = DType::BF16;
        let u32t = DType::U32;
        let moe_policy = quant::moe_tile_mroute_table();
        let s_total = m * k;
        // Sized at the policy's alloc tile: cap is monotone nonincreasing in
        // tile rows, so the smallest tile the policy can emit bounds every
        // m-routed chunk's map.
        let alloc_tile = moe_policy.alloc_tile();
        // Tail chunks have S < E: bounding the partial-tile term by min(E, S)
        // keeps the sentinel share sane on short tails.
        let cap = s_total.div_ceil(alloc_tile.rows()) + e.min(s_total);
        Ok(Self {
            router_logits: Tensor::zeros(ctx, &[m, e], bf)?,
            shared_out: Tensor::zeros(ctx, &[m, h], bf)?,
            shared_gate: Tensor::zeros(ctx, &[m], bf)?,
            indices: Tensor::zeros(ctx, &[m, k], u32t)?,
            slots: Tensor::zeros(ctx, &[m, k], u32t)?,
            scores: Tensor::zeros(ctx, &[m, k], DType::F32)?,
            counts: Tensor::zeros(ctx, &[e], u32t)?,
            cursors: Tensor::zeros(ctx, &[e], u32t)?,
            offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
            tile_offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
            ids_sorted: Tensor::zeros(ctx, &[s_total], u32t)?,
            slot_of: Tensor::zeros(ctx, &[m, k], u32t)?,
            blocks_gu: Tensor::zeros(ctx, &[cap * (mi / 64) * 4], u32t)?,
            blocks_dn: Tensor::zeros(ctx, &[cap * (h / 64) * 4], u32t)?,
            gx: Tensor::zeros(ctx, &[s_total, h], bf)?,
            eg: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            eu: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            ea: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            ed: Tensor::zeros(ctx, &[s_total, h], bf)?,
            tile_capacity: cap,
            tile: alloc_tile,
            route: quant::MoeRoute::Grouped(alloc_tile),
        })
    }

    /// Per-chunk views (`S = m·top_k` slot rows, the chunk's own row-tile
    /// capacity bound). The block-map views are sized to the chunk's bound so
    /// `moe_build_blocks` (grid = view capacity) rewrites exactly the entries
    /// the grouped GEMMs read — stale tiles a larger previous chunk wrote
    /// past this extent are unreachable.
    pub fn chunk(&self, m: usize) -> Result<Self> {
        let e = self.counts.numel();
        let k = self.indices.shape()[1];
        let s_total = m * k;
        // Set route, tile, and capacity together. The GEMV route retains
        // bounded unused block-map views.
        let policy = quant::moe_tile_mroute_table();
        let route = policy.route_for(m);
        let tile = match route {
            quant::MoeRoute::Grouped(tile) => tile,
            quant::MoeRoute::SmallmGemv => policy.alloc_tile(),
        };
        let cap = s_total.div_ceil(tile.rows()) + e.min(s_total);
        let slot_rows = |t: &Tensor| prefix_rows(t, s_total);
        let block_view = |t: &Tensor| {
            let words_per_tile = t.numel() / self.tile_capacity;
            t.view(0, &[cap * words_per_tile])
        };
        Ok(Self {
            router_logits: prefix_rows(&self.router_logits, m)?,
            shared_out: prefix_rows(&self.shared_out, m)?,
            shared_gate: prefix_rows(&self.shared_gate, m)?,
            indices: prefix_rows(&self.indices, m)?,
            slots: prefix_rows(&self.slots, m)?,
            scores: prefix_rows(&self.scores, m)?,
            counts: self.counts.view(0, self.counts.shape())?,
            cursors: self.cursors.view(0, self.cursors.shape())?,
            offsets: self.offsets.view(0, self.offsets.shape())?,
            tile_offsets: self.tile_offsets.view(0, self.tile_offsets.shape())?,
            ids_sorted: self.ids_sorted.view(0, &[s_total])?,
            slot_of: prefix_rows(&self.slot_of, m)?,
            blocks_gu: block_view(&self.blocks_gu)?,
            blocks_dn: block_view(&self.blocks_dn)?,
            gx: slot_rows(&self.gx)?,
            eg: slot_rows(&self.eg)?,
            eu: slot_rows(&self.eu)?,
            ea: slot_rows(&self.ea)?,
            ed: slot_rows(&self.ed)?,
            tile_capacity: cap,
            tile,
            route,
        })
    }
}

/// The per-chunk tensors the prefill FFN reads and writes.
pub(crate) struct PrefillMoeIo<'a> {
    /// `[m, h]` FFN input.
    pub x: &'a Tensor,
    /// `[m, h]` FFN output (routed experts plus the gated shared expert).
    pub out: &'a Tensor,
    /// Fused-stack GEMM staging (see `project_stack_or_slices`).
    pub stack: &'a Tensor,
    pub mlp_gate: &'a Tensor,
    pub mlp_up: &'a Tensor,
    pub mlp_act: &'a Tensor,
    /// Dequant staging for the batched fallback GEMM.
    pub dequant: &'a Tensor,
}

/// The prefill MoE FFN, GPU-resident end to end: router GEMM → per-row
/// softmax/top-k → then either (grouped route, m > 8) counting sort
/// (histogram, serial scans, atomic scatter) → GPU-built block map
/// (sentinel-padded to a readback-free capacity bound) → grouped expert GEMMs
/// → per-row combine → gated shared-expert add, or (small-m route, m <= 8)
/// the two fused small-batch expert kernels (gate + up + SwiGLU over the
/// union of the routed experts; down + combine + shared add per row), the
/// batched shape of the decode gathers. No host round-trip and no per-layer
/// allocations, so the whole chunk stays in one command buffer.
pub(crate) fn prefill_moe(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    moe_w: &MoeWeights,
    io: &PrefillMoeIo<'_>,
    ms: &PrefillMoeScratch,
) -> Result<()> {
    let m = io.x.shape()[0];
    let (e, top_k, inter) = (dims.num_experts, dims.top_k, dims.moe_intermediate);
    let h = dims.hidden;
    let policy = quant::moe_tile_mroute_table();
    // Desync guard, extended to the route: the chunk view and this function
    // must agree on the policy's answer — a mismatch would consume block maps
    // scanned at another tile (dropping rows SILENTLY) or run the GEMV path
    // over unsorted state the grouped kernels needed.
    let s_slots = m * top_k;
    ensure!(
        ms.route == policy.route_for(m),
        "prefill MoE route desync: chunk m={m} carries route={} but the policy routes {}",
        ms.route.label(),
        policy.route_for(m).label(),
    );
    if let quant::MoeRoute::Grouped(tile) = ms.route {
        ensure!(
            ms.tile == tile
                && ms.tile_capacity
                    == s_slots.div_ceil(ms.tile.rows()) + e.min(s_slots),
            "prefill MoE tile desync: chunk m={m} carries tile={} cap={} but the policy routes {}",
            ms.tile.label(),
            ms.tile_capacity,
            tile.label(),
        );
    }
    // Level structure for a concurrent encoder (serial encoders ignore the
    // barriers): the shared expert depends only on `x`, so its projections
    // are encoded alongside the router and the two chains meet at the final
    // gated add.
    let shared = &moe_w.shared;
    project_mat(ctx, pass, io.x, &moe_w.gate, &ms.router_logits, io.dequant)?;
    // The shared expert's gate|up as one stacked GEMM when eligible: its
    // SwiGLU then reads the two halves in place (no split copy).
    let inter_s = io.mlp_gate.numel() / m;
    let stacked_gu = project_stack_fused(
        ctx,
        pass,
        io.x,
        &shared.gate_up_proj,
        io.stack,
        &[inter_s, inter_s],
    )?;
    if stacked_gu.is_none() {
        project_mat(ctx, pass, io.x, &shared.gate_proj, io.mlp_gate, io.dequant)?;
        project_mat(ctx, pass, io.x, &shared.up_proj, io.mlp_up, io.dequant)?;
    }
    project_mat(ctx, pass, io.x, &moe_w.shared_gate, &ms.shared_gate, io.dequant)?;
    pass.level_barrier(&[
        &ms.router_logits,
        io.stack,
        io.mlp_gate,
        io.mlp_up,
        &ms.shared_gate,
    ])?;
    moe::moe_router_topk_rows(
        ctx,
        pass,
        &ms.router_logits,
        &ms.indices,
        &ms.scores,
        dims.norm_topk_prob,
    )?;
    match &stacked_gu {
        Some(gu) => silu_mul_rows_bf16(ctx, pass, gu, io.mlp_act)?,
        None => silu_mul_bf16(ctx, pass, io.mlp_gate, io.mlp_up, io.mlp_act)?,
    }
    pass.level_barrier(&[&ms.indices, &ms.scores, io.mlp_act])?;
    let slots = match &moe_w.slot_of {
        Some(table) => {
            moe::moe_remap_slots(ctx, pass, &ms.indices, table, &ms.slots, s_slots)?;
            pass.level_barrier(&[&ms.slots])?;
            &ms.slots
        }
        None => &ms.indices,
    };
    project_mat(ctx, pass, io.mlp_act, &shared.down_proj, &ms.shared_out, io.dequant)?;

    let tile = match ms.route {
        quant::MoeRoute::Grouped(tile) => tile,
        quant::MoeRoute::SmallmGemv => {
            ensure!(
                moe_w.expert_gate.bits == 4
                    && moe_w.expert_up.bits == 4
                    && moe_w.expert_down.bits == 4,
                "small-m route requires q4 experts"
            );
            // The fused kernels consume the router output directly in
            // natural (row, k) pair order: no union map, no counting sort,
            // no gather, no block maps, and the combine and the shared add
            // ride in the down kernel.
            moe::moe_smallm_gate_up(
                ctx,
                pass,
                &moe_w.expert_gate,
                &moe_w.expert_up,
                inter,
                io.x,
                slots,
                &ms.ea,
                top_k,
            )?;
            pass.level_barrier(&[&ms.ea, &ms.shared_out])?;
            return moe::moe_smallm_down_combine(
                ctx,
                pass,
                &moe_w.expert_down,
                h,
                &ms.ea,
                slots,
                &ms.scores,
                &ms.shared_out,
                &ms.shared_gate,
                io.out,
                top_k,
            );
        }
    };

    moe::fill_zero_u32(ctx, pass, &ms.counts)?;
    moe::fill_zero_u32(ctx, pass, &ms.cursors)?;
    pass.level_barrier(&[&ms.counts, &ms.cursors])?;
    moe::moe_sort_slots(
        ctx,
        pass,
        &moe::MoeSortBuffers {
            indices: &ms.indices,
            counts: &ms.counts,
            cursors: &ms.cursors,
            offsets: &ms.offsets,
            tile_offsets: &ms.tile_offsets,
            ids_sorted: &ms.ids_sorted,
            slot_of: &ms.slot_of,
        },
        e,
        top_k,
        tile.rows(),
    )?;
    pass.level_barrier(&[&ms.ids_sorted, &ms.offsets, &ms.tile_offsets, &ms.slot_of])?;
    gather_rows_bf16(ctx, pass, io.x, &ms.ids_sorted, &ms.gx)?;
    moe::moe_build_blocks(
        ctx,
        pass,
        &ms.offsets,
        &ms.tile_offsets,
        &ms.blocks_gu,
        moe_w.slot_of.as_ref(),
        e,
        inter,
        tile.rows(),
    )?;
    moe::moe_build_blocks(
        ctx,
        pass,
        &ms.offsets,
        &ms.tile_offsets,
        &ms.blocks_dn,
        moe_w.slot_of.as_ref(),
        e,
        h,
        tile.rows(),
    )?;
    pass.level_barrier(&[&ms.gx, &ms.blocks_gu, &ms.blocks_dn])?;

    let nb_gu = ms.tile_capacity * (inter / 64);
    let nb_dn = ms.tile_capacity * (h / 64);
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.gx,
        &moe_w.expert_gate,
        &ms.eg,
        &ms.blocks_gu,
        nb_gu,
        tile,
    )?;
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.gx,
        &moe_w.expert_up,
        &ms.eu,
        &ms.blocks_gu,
        nb_gu,
        tile,
    )?;
    pass.level_barrier(&[&ms.eg, &ms.eu])?;
    silu_mul_bf16(ctx, pass, &ms.eg, &ms.eu, &ms.ea)?;
    pass.level_barrier(&[&ms.ea])?;
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.ea,
        &moe_w.expert_down,
        &ms.ed,
        &ms.blocks_dn,
        nb_dn,
        tile,
    )?;
    pass.level_barrier(&[&ms.ed])?;
    moe::moe_combine_rows(ctx, pass, &ms.ed, &ms.slot_of, &ms.scores, io.out, top_k)?;

    // Both chains are complete: the routed sum is in `out`, the shared
    // expert in `shared_out`.
    pass.level_barrier(&[io.out, &ms.shared_out])?;
    moe::moe_row_gate_add(ctx, pass, &ms.shared_out, &ms.shared_gate, io.out)
}

/// The single-token tensors the decode FFN reads and writes.
pub(crate) struct DecodeMoeIo<'a> {
    /// `[h]` FFN input.
    pub x: &'a Tensor,
    /// `[h]` FFN output.
    pub out: &'a Tensor,
    /// `[2*inter]` shared-expert gate|up matvec output; `mlp_gate`/`mlp_up`
    /// are its two halves.
    pub mlp_gu: &'a Tensor,
    pub mlp_gate: &'a Tensor,
    pub mlp_up: &'a Tensor,
    pub mlp_act: &'a Tensor,
}

/// The decode MoE FFN, fully on-GPU: router GEMV + softmax/top-k kernel,
/// gather-GEMVs over the stacked experts through the device-resident indices,
/// the shared expert through the dense-MLP scratch, and the score-weighted
/// combine into `io.out`. Encoded for a concurrent pass: every inter-level
/// data edge is marked with a barrier except the final write, which the
/// caller orders.
pub(crate) fn decode_moe(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    moe_w: &MoeWeights,
    io: &DecodeMoeIo<'_>,
    ms: &MoeScratch,
) -> Result<()> {
    let inter = dims.moe_intermediate;
    // The shared expert reads only `x` and writes only its own scratch, so
    // its whole chain is independent of the router and of the expert gathers;
    // encoding it interleaved puts the two on the same dependency levels.
    let shared = &moe_w.shared;

    // Level 0 — everything whose only input is `x`.
    quant::gemv_quant(ctx, pass, &moe_w.gate, io.x, &ms.router_logits)?;
    quant::gemv_quant(ctx, pass, &shared.gate_up_proj, io.x, io.mlp_gu)?;
    quant::gemv_quant(ctx, pass, &moe_w.shared_gate, io.x, &ms.shared_gate)?;
    pass.level_barrier(&[&ms.router_logits, io.mlp_gu, &ms.shared_gate])?;

    // Level 1 — the routing decision, and the shared expert's activation.
    moe::moe_router_topk(
        ctx,
        pass,
        &ms.router_logits,
        &ms.indices,
        &ms.scores,
        dims.norm_topk_prob,
    )?;
    silu_mul_bf16(ctx, pass, io.mlp_gate, io.mlp_up, io.mlp_act)?;
    pass.level_barrier(&[&ms.indices, &ms.scores, io.mlp_act])?;
    let slots = match &moe_w.slot_of {
        Some(table) => {
            moe::moe_remap_slots(ctx, pass, &ms.indices, table, &ms.slots, dims.top_k)?;
            pass.level_barrier(&[&ms.slots])?;
            &ms.slots
        }
        None => &ms.indices,
    };

    moe::moe_gather_gemv_gate_up(
        ctx,
        pass,
        &moe_w.expert_gate,
        &moe_w.expert_up,
        inter,
        io.x,
        slots,
        &ms.act,
    )?;
    quant::gemv_quant(ctx, pass, &shared.down_proj, io.mlp_act, &ms.shared_out)?;
    pass.level_barrier(&[&ms.act, &ms.shared_out])?;

    moe::moe_gather_gemv_down_combine(
        ctx,
        pass,
        &moe_w.expert_down,
        dims.hidden,
        &ms.act,
        slots,
        &ms.scores,
        Some((&ms.shared_out, &ms.shared_gate)),
        io.out,
    )
}
