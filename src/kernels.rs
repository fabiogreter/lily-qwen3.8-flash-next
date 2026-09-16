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
