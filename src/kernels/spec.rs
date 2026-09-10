//! GPU-side control of the speculative step: the accepted count and the
//! values derived from it, written into a control block that later dispatches
//! of the same pass read as inline arguments ([`crate::kernels::Arg::Gpu`]).

use anyhow::{Result, ensure};

use crate::kernels::Arg;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/spec.metal");

/// Words between control slots (one 16-byte slot per value, so every slot is
/// a valid inline-argument address on its own).
pub const CTRL_STRIDE: usize = 4;

/// Slot holding the accepted count `a`.
pub const SLOT_ACCEPTED: usize = 0;
/// Slot holding `a + 1`, the rows the rollback keeps.
pub const SLOT_KEEP: usize = 1;

/// Slot holding the position of chain row `i` (`pos0 + a + 1 + i`).
pub fn slot_pos(i: usize) -> usize {
    2 + 3 * i
}

/// Slot holding the indexer block chain row `i` completes.
pub fn slot_block(i: usize) -> usize {
    3 + 3 * i
}

/// Slot holding 1 when chain row `i` completes an indexer block, else 0.
pub fn slot_count(i: usize) -> usize {
    4 + 3 * i
}

/// Words a control block describing `chain` rows needs.
pub fn ctrl_words(chain: usize) -> usize {
    (2 + 3 * chain) * CTRL_STRIDE
}

/// The one-word view of `slot` in a control block.
pub fn ctrl_word(ctrl: &Tensor, slot: usize) -> Result<Tensor> {
    ctrl.view(slot * CTRL_STRIDE, &[1])
}

/// Compares the trunk's draws (`draws`, U32 `[m]`) with the verified tokens
/// (`ids`, U32 `[m]`: the pending token then the drafts) and writes the
/// control block for a draft pass whose chain has `chain` rows at
/// `pos0 + a + 1 + i`, with indexer blocks of `ratio` positions.
#[allow(clippy::too_many_arguments)]
pub fn spec_accept(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    draws: &Tensor,
    ids: &Tensor,
    ctrl: &Tensor,
    pos0: usize,
    ratio: usize,
    chain: usize,
) -> Result<()> {
    let m = draws.numel();
    ensure!(m > 0 && ids.numel() == m, "draws and ids must both hold the verified rows");
    ensure!(draws.dtype() == DType::U32 && ids.dtype() == DType::U32 && ctrl.dtype() == DType::U32, "spec_accept works on U32 tensors");
    ensure!(ctrl.numel() >= ctrl_words(chain), "control block too small for {chain} chain rows");
    ensure!(ratio > 0, "indexer block ratio must be nonzero");
    let pipeline = ctx.pipeline("spec_accept", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[draws.binding(), ids.binding(), ctrl.binding()],
        &[
            Param::U32(m as u32),
            Param::U32(pos0 as u32),
            Param::U32(ratio as u32),
            Param::U32(chain as u32),
            Param::U32(CTRL_STRIDE as u32),
        ],
        Grid::Threads { grid: (1, 1, 1), threadgroup: (1, 1, 1) },
    )
}

/// Copies row `row` of `src` (`[rows, ...]`, word-sized elements) into `dst`
/// (one row's worth of words). The row index may be GPU-supplied; a row at
/// or past `rows` copies nothing.
pub fn copy_row<'t>(ctx: &MetalContext, pass: &ComputePass<'_>, src: &Tensor, row: impl Into<Arg<'t>>, dst: &Tensor) -> Result<()> {
    let row = row.into();
    let rows = *src.shape().first().ok_or_else(|| anyhow::anyhow!("copy_row from a 0-d tensor"))?;
    ensure!(rows > 0 && src.byte_len().is_multiple_of(4 * rows), "src must be [rows, ...] of word-sized elements");
    let words = src.byte_len() / 4 / rows;
    ensure!(dst.byte_len() == words * 4 && words > 0, "dst must hold one row ({words} words)");
    if let Some(r) = row.constant() {
        ensure!(r < rows, "row {r} out of {rows}");
    }
    let pipeline = ctx.pipeline("copy_row_u32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[src.binding(), dst.binding()],
        &[Param::U32(words as u32), Param::U32(rows as u32), row.param()],
        Grid::Threads { grid: (words, 1, 1), threadgroup: (256.min(words), 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/spec.rs"]
mod tests;
