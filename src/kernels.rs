//! Runtime-compiled Metal kernels and Rust dispatch wrappers.

use crate::metal::Param;
use crate::tensor::Tensor;

pub mod attention;
pub mod elementwise;
pub mod gdn;
pub mod gemm;
pub mod hc;
pub mod moe;
pub mod norm;
pub mod ple;
pub mod qsa;
pub mod quant;
pub mod sample;
pub mod skinny;
pub mod spec;
pub mod vision;

pub fn u32_bytes(v: usize) -> [u8; 4] {
    (v as u32).to_ne_bytes()
}

pub fn f32_bytes(v: f32) -> [u8; 4] {
    v.to_ne_bytes()
}

/// A kernel's `uint` argument: a host constant, or the first word of a U32
/// tensor the pass itself writes before the dispatch runs (ordered by the
/// pass's barriers), read by the GPU when the kernel executes.
#[derive(Clone, Copy)]
pub enum Arg<'t> {
    Const(usize),
    Gpu(&'t Tensor),
}

impl<'t> Arg<'t> {
    pub fn param(self) -> Param<'t> {
        match self {
            Arg::Const(v) => Param::U32(v as u32),
            Arg::Gpu(t) => {
                let (buf, offset) = t.binding();
                Param::Gpu(buf, offset)
            }
        }
    }

    /// The constant, or `None` when the GPU supplies the value.
    pub fn constant(self) -> Option<usize> {
        match self {
            Arg::Const(v) => Some(v),
            Arg::Gpu(_) => None,
        }
    }
}

/// A position argument together with the range the host knows it lies in.
/// Everything the host decides from a position (grid sizes, dense versus
/// sparse attention, block counts, bounds checks) is sized for the range, so
/// a GPU-supplied position anywhere in it is handled exactly.
#[derive(Clone, Copy)]
pub struct Pos<'t> {
    pub arg: Arg<'t>,
    pub min: usize,
    pub max: usize,
}

impl<'t> From<usize> for Arg<'t> {
    fn from(v: usize) -> Self {
        Arg::Const(v)
    }
}

impl<'t> From<usize> for Pos<'t> {
    fn from(pos: usize) -> Self {
        Pos::host(pos)
    }
}

impl<'t> From<Arg<'t>> for Pos<'t> {
    /// A GPU word with no tighter bound than the whole `u32` range is not
    /// useful for sizing, so only constants convert; use [`Pos::gpu`] otherwise.
    fn from(arg: Arg<'t>) -> Self {
        match arg {
            Arg::Const(v) => Pos::host(v),
            Arg::Gpu(t) => Pos::gpu(t, 0, u32::MAX as usize),
        }
    }
}

impl<'t> Pos<'t> {
    /// The position `by` rows later. A GPU-supplied position cannot be
    /// offset (there is no word holding the sum), so `by` must be 0 for it.
    pub fn offset(self, by: usize) -> anyhow::Result<Self> {
        match self.arg {
            Arg::Const(p) => Ok(Pos::host(p + by)),
            Arg::Gpu(_) => {
                anyhow::ensure!(
                    by == 0,
                    "a GPU-supplied position cannot be offset by {by} rows"
                );
                Ok(self)
            }
        }
    }

    pub fn host(pos: usize) -> Self {
        Self { arg: Arg::Const(pos), min: pos, max: pos }
    }

    /// A position the GPU wrote to `word` (U32, one element), in `min..=max`.
    pub fn gpu(word: &'t Tensor, min: usize, max: usize) -> Self {
        Self { arg: Arg::Gpu(word), min, max }
    }

    pub fn param(self) -> Param<'t> {
        self.arg.param()
    }
}

/// Rotary pairs per (temporal, height, width) axis of Qwen3.8-Flash-Next's
/// interleaved M-RoPE over its 32 pairs, the one layout the kernels are
/// written for (`tools/reference/VISION.md`, "Interleaved M-RoPE").
pub const MROPE_SECTION: [usize; 3] = [11, 11, 10];

/// The axis rotary pair `pair` takes under [`MROPE_SECTION`]: pair `i` takes
/// the temporal axis (0) when `i % 3 == 0`, the height axis (1) when
/// `i % 3 == 1` and `i < 3 * 11`, the width axis (2) when `i % 3 == 2` and
/// `i < 3 * 10`, and the temporal axis for anything past a section's budget.
/// The Metal kernels (`mrope_axis` in `attention.metal` and `qsa.metal`)
/// apply the same rule; the kernel tests pin the two to each other.
pub fn mrope_axis(pair: usize) -> usize {
    let axis = pair % 3;
    if axis == 0 || pair >= 3 * MROPE_SECTION[axis] { 0 } else { axis }
}

/// How a kernel turns a row's sequence index into its rotary position.
/// Cache slots and indexer blocks always use the sequence index; only the
/// rotary angle takes this.
#[derive(Clone, Copy)]
pub enum Rope<'t> {
    /// `sequence index + delta` on every axis: text prompts (delta 0) and
    /// every token generated after a prompt with an image (VISION.md:
    /// `rope_deltas`). The delta is added in the kernel, so it applies to a
    /// GPU-supplied index too.
    Delta(i64),
    /// Per-token 3-axis positions, for the prefill rows of a prompt with an
    /// image: row `i` of `positions` (U32 `[rows, 3]`, temporal, height,
    /// width) belongs to sequence index `base + i`, and each rotary pair
    /// reads the axis [`mrope_axis`] gives it.
    Rows { positions: &'t Tensor, base: usize },
}

impl Rope<'_> {
    /// The delta as the kernels' `int` parameter.
    pub fn delta_i32(delta: i64) -> anyhow::Result<i32> {
        i32::try_from(delta)
            .map_err(|_| anyhow::anyhow!("rope delta {delta} out of range"))
    }

    /// Checks that `positions` covers sequence indices `first..first + count`
    /// (`first` may be a range for a GPU-supplied index) and returns the
    /// kernel's `pos_base`.
    pub fn check_rows(
        positions: &Tensor,
        base: usize,
        first_min: usize,
        first_max: usize,
        count: usize,
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(
            positions.dtype() == crate::tensor::DType::U32
                && positions.shape().len() == 2
                && positions.shape()[1] == 3,
            "rope positions must be U32 [rows, 3]"
        );
        let rows = positions.shape()[0];
        anyhow::ensure!(
            base <= first_min && first_max + count <= base + rows,
            "rope positions cover {base}..{}, rows need {first_min}..{}",
            base + rows,
            first_max + count
        );
        Ok(base)
    }
}
