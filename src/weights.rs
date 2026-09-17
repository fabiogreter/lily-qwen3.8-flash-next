//! The checkpoint loader the architectures share: it maps tensor names onto
//! device tensors, reading each byte range straight into a shared-storage
//! Metal buffer, and holds the weight structs (affine packed
//! `{weight, scales, biases}` triples) the kernels consume. Every tensor in
//! the file must be consumed or explicitly skip-listed by the caller — a
//! name-scheme drift fails loudly at load instead of silently dropping
//! weights.

use std::cell::RefCell;
use std::collections::HashSet;

use anyhow::{Context as _, Result, ensure};
use half::bf16;
use objc2_metal::MTLBuffer;

use crate::config::QuantizationConfig;
use crate::metal::MetalContext;
use crate::safetensors::{Checkpoint, SafetensorsDType};
use crate::tensor::{DType, Tensor};

/// One linear layer's weights in MLX affine form: `codes` packs `32 / bits`
/// elements per u32 along the input dim (low element first), dequantized per
/// `group_size` input elements as `w = scale * q + bias`. `bits` is 4 for
/// everything except MoE routers and shared-expert gates, which mlx keeps at
/// 8 bits.
pub struct QuantWeights {
    /// U32 `[out, in * bits / 32]`.
    pub codes: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub scales: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub biases: Tensor,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantWeights {
    pub fn out_features(&self) -> usize {
        self.codes.shape()[0]
    }

    pub fn in_features(&self) -> usize {
        self.codes.shape()[1] * (32 / self.bits)
    }

    /// A row-range view (`[rows, in]` starting at `start_row`) sharing the
    /// underlying buffers — how fused projections hand out their segments.
    pub fn view_rows(&self, start_row: usize, rows: usize) -> Result<Self> {
        let words = self.codes.shape()[1];
        let groups = self.scales.shape()[1];
        Ok(Self {
            codes: self.codes.view(start_row * words, &[rows, words])?,
            scales: self.scales.view(start_row * groups, &[rows, groups])?,
            biases: self.biases.view(start_row * groups, &[rows, groups])?,
            group_size: self.group_size,
            bits: self.bits,
        })
    }

    pub(crate) fn expect_features(
        &self,
        out: usize,
        inp: usize,
        name: &str,
    ) -> Result<()> {
        ensure!(
            self.out_features() == out && self.in_features() == inp,
            "{name} is [{}, {}], expected [{out}, {inp}]",
            self.out_features(),
            self.in_features(),
        );
        Ok(())
    }
}

pub type LinearWeights = QuantWeights;

pub struct MlpWeights {
    /// `[2*inter, h]` — gate rows then up rows; `gate_proj`/`up_proj` are
    /// row-range views of it, so prefill's separate GEMMs and decode's single
    /// fused matvec share the same bytes.
    pub gate_up_proj: LinearWeights,
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

/// Sparse-MoE FFN: a router over stacked expert projections plus the
/// always-on shared expert with a sigmoid gate. Expert stacks are flattened
/// `[E * rows, cols]` so expert `e` is the row range `e*rows .. (e+1)*rows`
/// (`view_rows` for prefill's per-expert GEMMs, base-offset arithmetic for
/// the decode gather kernels).
pub struct MoeWeights {
    /// Router `[num_experts, h]` (8-bit in mlx-quantized checkpoints).
    pub gate: LinearWeights,
    /// `[E * inter, h]`.
    pub expert_gate: LinearWeights,
    /// `[E * inter, h]`.
    pub expert_up: LinearWeights,
    /// `[E * h, inter]`.
    pub expert_down: LinearWeights,
    /// With an expert cache (machines the experts do not fit): the three
    /// expert stacks above are the cache's slab, shared by every layer, and
    /// this `U32 [E]` table maps the layer's expert ids to slab slots
    /// (`ExpertCache::NONE` for experts not resident). `None`: the stacks
    /// are the layer's own experts.
    pub slot_of: Option<Tensor>,
    /// With a served cache: the protocol handle and this layer's index, so
    /// a pass can have misses resolved before it reads the slab.
    pub cache: Option<(std::rc::Rc<crate::qwen4exp::expert_cache::ExpertCacheLink>, usize)>,
    pub shared: MlpWeights,
    /// `[1, h]`, sigmoid-gating the shared expert's output.
    pub shared_gate: LinearWeights,
}

fn to_dtype(dtype: &SafetensorsDType) -> Result<DType> {
    match dtype {
        SafetensorsDType::BF16 => Ok(DType::BF16),
        SafetensorsDType::F32 => Ok(DType::F32),
        SafetensorsDType::U32 => Ok(DType::U32),
        SafetensorsDType::Other(s) => anyhow::bail!("unsupported checkpoint dtype {s}"),
    }
}

/// The bit width a projection is expected to be stored at, keyed by the
/// tensor bases being loaded. Each model layout has its own policy.
pub(crate) type BitsPolicy = fn(&[&str]) -> usize;

/// Checkpoint access that records every consumed tensor name so `finish` can
/// verify nothing in the file was silently ignored.
pub(crate) struct Loader<'a> {
    ctx: &'a MetalContext,
    ckpt: Checkpoint,
    quant: QuantizationConfig,
    skip_prefixes: &'static [&'static str],
    expected_bits: BitsPolicy,
    consumed: RefCell<HashSet<String>>,
}

impl<'a> Loader<'a> {
    pub(crate) fn new(
        ctx: &'a MetalContext,
        ckpt: Checkpoint,
        quant: QuantizationConfig,
        skip_prefixes: &'static [&'static str],
        expected_bits: BitsPolicy,
    ) -> Self {
        Self {
            ctx,
            ckpt,
            quant,
            skip_prefixes,
            expected_bits,
            consumed: RefCell::new(HashSet::new()),
        }
    }
}

impl Loader<'_> {
    pub(crate) fn checkpoint(&self) -> &Checkpoint {
        &self.ckpt
    }

    /// Records a tensor as used without reading it (for weights served from
    /// the checkpoint files at runtime).
    pub(crate) fn mark_consumed(&self, name: &str) {
        self.consumed.borrow_mut().insert(name.to_string());
    }

    /// Reads one tensor straight into a fresh shared-storage Metal buffer.
    pub(crate) fn tensor(&self, name: &str) -> Result<Tensor> {
        self.consumed.borrow_mut().insert(name.to_string());
        let meta = self
            .ckpt
            .meta(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        let dtype = to_dtype(&meta.dtype)?;
        let shape = meta.shape.clone();
        let buf = self.ctx.new_buffer(meta.byte_len())?;
        self.ckpt.read_with(name, |meta| {
            // SAFETY: freshly allocated shared buffer of exactly byte_len bytes.
            Ok(unsafe {
                core::slice::from_raw_parts_mut(
                    buf.contents().as_ptr().cast::<u8>(),
                    meta.byte_len(),
                )
            })
        })?;
        Tensor::from_buffer(buf, &shape, dtype)
    }

    /// Like [`Self::tensor`] but upcasts BF16 to F32 host-side — for the few
    /// small parameters whose kernels want f32 while mlx checkpoints store
    /// bf16 (the GDN GatedNorm weight).
    pub(crate) fn tensor_f32(&self, name: &str) -> Result<Tensor> {
        let t = self.tensor(name)?;
        match t.dtype() {
            DType::F32 => Ok(t),
            DType::BF16 => Tensor::from_f32(self.ctx, &t.to_f32()?, t.shape()),
            DType::U32 => anyhow::bail!("{name} is U32, expected float"),
        }
    }

    /// Loads several `[n_i, k]` row-major tensors of one dtype into a single
    /// `[sum(n_i), k]` buffer (each read lands at its row offset), so decode
    /// can run one fused matvec while per-segment views keep the original
    /// tensors addressable.
    fn concat_rows(&self, names: &[String], k: usize) -> Result<Tensor> {
        let mut rows = 0usize;
        let mut dtype: Option<DType> = None;
        for name in names {
            self.consumed.borrow_mut().insert(name.clone());
            let meta = self
                .ckpt
                .meta(name)
                .with_context(|| format!("tensor {name} not in checkpoint"))?;
            // Stacked expert tensors are [E, I, k]; flatten leading dims.
            ensure!(
                meta.shape.len() >= 2 && *meta.shape.last().expect("shape") == k,
                "{name} shape {:?} not [.., {k}]",
                meta.shape
            );
            let d = to_dtype(&meta.dtype)?;
            ensure!(
                dtype.is_none() || dtype == Some(d),
                "{name} dtype mismatch in fuse"
            );
            dtype = Some(d);
            rows += meta.shape[..meta.shape.len() - 1].iter().product::<usize>();
        }
        let dtype = dtype.ok_or_else(|| anyhow::anyhow!("empty fuse list"))?;
        let buf = self.ctx.new_buffer(rows * k * dtype.size())?;
        let mut offset = 0usize;
        for name in names {
            self.ckpt.read_with(name, |meta| {
                let start = offset;
                // SAFETY: freshly allocated shared buffer; [start,
                // start+byte_len) is this tensor's disjoint row range.
                Ok(unsafe {
                    core::slice::from_raw_parts_mut(
                        buf.contents().as_ptr().cast::<u8>().add(start),
                        meta.byte_len(),
                    )
                })
            })?;
            let meta =
                self.ckpt.meta(name).with_context(|| format!("{name} missing"))?;
            offset += meta.byte_len();
        }
        Tensor::from_buffer(buf, &[rows, k], dtype)
    }

    /// Loads one quantized projection, or several fused along the output dim,
    /// at the loader's default group size.
    pub(crate) fn linear(&self, bases: &[&str], k: usize) -> Result<LinearWeights> {
        self.linear_grouped(bases, k, self.quant.group_size)
    }

    /// [`Self::linear`] with an explicit quantization group size, for the few
    /// tensors whose row width is not a multiple of the default group.
    pub(crate) fn linear_grouped(
        &self,
        bases: &[&str],
        k: usize,
        group_size: usize,
    ) -> Result<LinearWeights> {
        let weights: Vec<String> =
            bases.iter().map(|b| format!("{b}.weight")).collect();
        ensure!(
            self.ckpt.meta(&format!("{}.scales", bases[0])).is_some(),
            "{} is not an MLX quantized projection",
            bases[0]
        );
        let q = QuantizationConfig { group_size, bits: self.quant.bits };
        ensure!(
            k.is_multiple_of(q.group_size) && k.is_multiple_of(8),
            "in_features {k} not divisible by group size {} / packing",
            q.group_size
        );
        // Per-tensor bit width is inferred from the packed width so router /
        // shared-expert-gate tensors that mlx keeps at 8 bits load correctly
        // regardless of the config's global `bits`.
        let meta = self
            .ckpt
            .meta(&weights[0])
            .with_context(|| format!("tensor {} not in checkpoint", weights[0]))?;
        let cols = *meta.shape.last().expect("shape");
        ensure!(
            k.is_multiple_of(cols) && matches!(32 * cols / k, 4 | 8),
            "cannot infer bit width for {} ({cols} packed cols, in={k})",
            weights[0]
        );
        let bits = 32 * cols / k;
        let expected_bits = (self.expected_bits)(bases);
        ensure!(
            bits == expected_bits,
            "{} uses {bits}-bit storage; expected {expected_bits}-bit for this projection",
            weights[0]
        );
        let scales: Vec<String> = bases.iter().map(|b| format!("{b}.scales")).collect();
        let biases: Vec<String> = bases.iter().map(|b| format!("{b}.biases")).collect();
        let codes = self.concat_rows(&weights, cols)?;
        let scales = self.concat_rows(&scales, k / q.group_size)?;
        let biases = self.concat_rows(&biases, k / q.group_size)?;
        ensure!(codes.dtype() == DType::U32, "quantized codes must be U32");
        ensure!(
            scales.dtype() == DType::BF16 && biases.dtype() == DType::BF16,
            "quantized scales/biases must be BF16"
        );
        Ok(QuantWeights { codes, scales, biases, group_size: q.group_size, bits })
    }

    /// Loads the conv1d weight, accepting the HF `[C, 1, KD]` or mlx
    /// `[C, KD, 1]` layout and transposing to the tap-major `[KD, C]` the conv
    /// kernel consumes.
    pub(crate) fn conv_weight(&self, name: &str) -> Result<Tensor> {
        self.consumed.borrow_mut().insert(name.to_string());
        let bytes = self.ckpt.read(name)?;
        let meta =
            self.ckpt.meta(name).with_context(|| format!("tensor {name} missing"))?;
        ensure!(meta.dtype == SafetensorsDType::BF16, "conv weight must be BF16");
        ensure!(
            meta.shape.len() == 3 && (meta.shape[1] == 1 || meta.shape[2] == 1),
            "conv weight shape {:?} != [C, 1, KD] or [C, KD, 1]",
            meta.shape
        );
        let (c, kd) = (meta.shape[0], meta.shape[1].max(meta.shape[2]));
        let src: &[bf16] = bytemuck::cast_slice(&bytes);
        let mut transposed = vec![bf16::from_f32(0.0); kd * c];
        for ch in 0..c {
            for t in 0..kd {
                transposed[t * c + ch] = src[ch * kd + t];
            }
        }
        Tensor::from_bytes(
            self.ctx,
            bytemuck::cast_slice(&transposed),
            &[kd, c],
            DType::BF16,
        )
    }

    /// Fails if the checkpoint holds tensors lily neither consumed nor
    /// skip-listed — the guard against silent name-scheme drift.
    pub(crate) fn finish(&self) -> Result<()> {
        let consumed = self.consumed.borrow();
        let mut unconsumed: Vec<&str> = self
            .ckpt
            .names()
            .filter(|n| {
                !consumed.contains(*n)
                    && !self.skip_prefixes.iter().any(|p| n.starts_with(p))
            })
            .collect();
        unconsumed.sort_unstable();
        ensure!(
            unconsumed.is_empty(),
            "{} checkpoint tensors were not consumed (name-scheme drift?): {:?}{}",
            unconsumed.len(),
            &unconsumed[..unconsumed.len().min(8)],
            if unconsumed.len() > 8 { " ..." } else { "" }
        );
        Ok(())
    }
}

pub(crate) fn expect_shape(t: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    ensure!(t.shape() == shape, "{name} shape {:?} != expected {shape:?}", t.shape());
    Ok(())
}
