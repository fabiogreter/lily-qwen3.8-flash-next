//! The vision tower on the GPU (`docs/vision-support-plan.md` item 3): from
//! preprocessed pixel rows to the merged embeddings the language model
//! replaces its `<|image_pad|>` rows with. The chain is
//! `tools/reference/VISION.md`, "The tower", followed step by step:
//!
//! 1. patch embedding: the pixel rows cast to bf16, one dense GEMM with the
//!    flattened conv weight and its bias, then the bilinearly resampled
//!    position table ([`crate::kernels::vision::patch_pos_embed_bf16`]);
//! 2. `depth` blocks of `x += proj(attn(rope(qkv(LayerNorm1(x)))))` and
//!    `x += fc2(gelu_tanh(fc1(LayerNorm2(x))))`: LayerNorm with bias in f32,
//!    the fused qkv GEMM, the 2-D rotary in place, full bidirectional
//!    attention over every patch of the image, the projections as GEMMs;
//!    every Linear is the GEMM with its bias in the epilogue, so its output
//!    is rounded to bf16 once, as the reference's is (a separate bias pass
//!    rounded twice and measurably widened the gap to the golden);
//! 3. the merger: LayerNorm per patch, four consecutive rows (one 2 x 2 merge
//!    block) read as one `[N / 4, 4 H]` row, `fc1`, the exact erf GELU, `fc2`
//!    into the text model's hidden size.
//!
//! Attention is a fused online-softmax kernel (design (b) of the plan) rather
//! than a score matrix through the GEMM: the engine already had a tensor-op
//! flash kernel for the text path to derive it from, and materialising
//! `[N, N]` scores per head costs about 0.5 GB of traffic per head and layer
//! at 8 192 patches, 230 GB for the tower, which alone would take the
//! latency budget. The kernel is a copy with the causal limit and the KV
//! cache removed, so the text path's numerics are untouched.
//!
//! Everything runs in one serial pass per image; intermediates live in a
//! [`VisionScratch`] that grows to the largest patch count seen.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use half::bf16;

use crate::kernels::elementwise::{Gelu, add_bf16, gelu_bf16};
use crate::kernels::gemm::gemm_bf16_nt_bias;
use crate::kernels::norm::layernorm_bf16;
use crate::kernels::vision::{
    ATTN_D, attention_full_bf16, patch_pos_embed_bf16, qkv_rope_2d_bf16,
};
use crate::metal::MetalContext;
use crate::safetensors::Checkpoint;
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;

use super::config::Qwen4ExpConfig;
use super::config::VisionConfig;
use super::vision_weights::{self, VisionWeights};

/// The rotary base of `Qwen4ExpVisionRotaryEmbedding`, a constructor default
/// the checkpoint's config does not carry (VISION.md "Rotary").
const ROPE_THETA: f32 = 10000.0;
/// `nn.LayerNorm(..., eps=1e-6)` throughout the tower.
const LAYERNORM_EPS: f32 = 1e-6;

/// Intermediates for up to `capacity` patches; every forward uses exact
/// row-prefix views. Under 0.3 GB at the 8 192-patch cap.
pub struct VisionScratch {
    capacity: usize,
    /// `[n, patch_dim]` bf16: the pixel rows cast on upload.
    pixels: Tensor,
    /// `[n, H]` bf16: the residual stream (the pre-merger states at the end).
    x: Tensor,
    /// `[n, H]` bf16: the normed input of a sub-block; read as `[n / 4, 4 H]`
    /// by the merger.
    h: Tensor,
    /// `[n, 3 H]` bf16.
    qkv: Tensor,
    /// `[n, H]` bf16: attention output before the projection.
    attn: Tensor,
    /// `[n, H]` bf16: a projection's output before its bias and residual add.
    tmp: Tensor,
    /// `[n, I]` bf16: the MLP intermediate; the merger's `fc1` output fits in
    /// its first `n H` elements.
    mlp: Tensor,
    /// `[n / 4, out_hidden]` bf16.
    merged: Tensor,
}

impl VisionScratch {
    fn new(ctx: &MetalContext, v: &VisionConfig, capacity: usize) -> Result<Self> {
        let (h, i) = (v.hidden_size, v.intermediate_size);
        // The merger's fc1 output, `[n / 4, merge_dim]`, is `n * hidden`
        // elements and reuses the `[n, intermediate]` MLP scratch.
        ensure!(h <= i, "the merger intermediate does not fit the MLP scratch");
        let z = |shape: &[usize]| Tensor::zeros(ctx, shape, DType::BF16);
        Ok(Self {
            capacity,
            pixels: z(&[capacity, v.patch_dim()])?,
            x: z(&[capacity, h])?,
            h: z(&[capacity, h])?,
            qkv: z(&[capacity, 3 * h])?,
            attn: z(&[capacity, h])?,
            tmp: z(&[capacity, h])?,
            mlp: z(&[capacity, i])?,
            merged: z(&[capacity / 4, v.out_hidden_size])?,
        })
    }

    /// GPU bytes held.
    pub fn bytes(&self) -> usize {
        [
            &self.pixels,
            &self.x,
            &self.h,
            &self.qkv,
            &self.attn,
            &self.tmp,
            &self.mlp,
            &self.merged,
        ]
        .iter()
        .map(|t| t.byte_len())
        .sum()
    }
}

/// One image's tower outputs, as views into the tower's scratch: valid until
/// the next [`VisionTower::forward`].
pub struct VisionOutput {
    /// `[N / 4, out_hidden]` bf16: one row per 2 x 2 merge block in raster
    /// order, what replaces the prompt's image placeholder rows.
    pub merged: Tensor,
    /// `[N, H]` bf16: the residual stream after the last block, before the
    /// merger (`last_hidden_state`; diagnostics only).
    pub pre_merger: Tensor,
    /// The pass's GPU span.
    pub gpu_secs: f64,
    /// Host time from the call to the pass's completion (the cast and
    /// upload, encoding, the wait).
    pub host_secs: f64,
}

pub struct VisionTower {
    config: VisionConfig,
    weights: VisionWeights,
    scratch: Option<VisionScratch>,
}

impl VisionTower {
    pub fn new(config: VisionConfig, weights: VisionWeights) -> Result<Self> {
        ensure!(
            config.head_dim() == ATTN_D,
            "vision head dim {} is not the compiled attention head dim {ATTN_D}",
            config.head_dim()
        );
        Ok(Self { config, weights, scratch: None })
    }

    /// Loads only the tower from a converted checkpoint that carries one,
    /// without the language model: for the probe and the tests. The loader's
    /// consumption check is not run, since the text tensors are left on disk
    /// on purpose.
    pub fn load_from_dir(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Qwen4ExpConfig::from_model_dir(dir)?;
        let vision = config.vision.clone().with_context(|| {
            format!("{} carries no vision tower (no lily.vision block)", dir.display())
        })?;
        let ckpt = Checkpoint::open(dir)?;
        let loader = Loader::new(ctx, ckpt, config.quantization, &[], |_| 4);
        let weights = vision_weights::load(&loader, &vision)?;
        Self::new(vision, weights)
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn weights(&self) -> &VisionWeights {
        &self.weights
    }

    /// Scratch bytes currently held (0 before the first forward).
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.as_ref().map_or(0, VisionScratch::bytes)
    }

    /// Makes sure `slot` holds a scratch for `n` patches, growing to `n`
    /// rounded up to a multiple of 4 (never shrinking).
    fn ensure_scratch(
        ctx: &MetalContext,
        slot: &mut Option<VisionScratch>,
        v: &VisionConfig,
        n: usize,
    ) -> Result<()> {
        let have = slot.as_ref().map_or(0, |s| s.capacity);
        if have < n {
            *slot = None;
            *slot = Some(VisionScratch::new(ctx, v, n.div_ceil(4) * 4)?);
        }
        Ok(())
    }

    /// Runs the tower over one image: `pixels` is the preprocessed
    /// `[gh * gw, patch_dim]` float32 array in block-major patch order (cast
    /// to bf16 here, as the reference casts to the tower's dtype), `(gh, gw)`
    /// its patch grid. Blocks until the GPU is done.
    pub fn forward(
        &mut self,
        ctx: &MetalContext,
        pixels: &[f32],
        gh: usize,
        gw: usize,
    ) -> Result<VisionOutput> {
        self.forward_blocks(ctx, pixels, gh, gw, self.config.depth)
    }

    /// [`Self::forward`] through the first `blocks` transformer blocks only
    /// (the merger still runs on that residual): a diagnostic for comparing
    /// intermediate states against the reference block by block.
    pub fn forward_blocks(
        &mut self,
        ctx: &MetalContext,
        pixels: &[f32],
        gh: usize,
        gw: usize,
        blocks: usize,
    ) -> Result<VisionOutput> {
        let started = Instant::now();
        ensure!(
            blocks <= self.config.depth,
            "{blocks} blocks exceed depth {}",
            self.config.depth
        );
        let v = &self.config;
        let (hid, patch_dim, heads) = (v.hidden_size, v.patch_dim(), v.num_heads);
        let merge = v.spatial_merge_size * v.spatial_merge_size;
        let n = gh * gw;
        ensure!(
            gh >= 2 && gw >= 2 && gh.is_multiple_of(2) && gw.is_multiple_of(2),
            "grid ({gh}, {gw}) is not made of 2 x 2 merge blocks"
        );
        ensure!(
            pixels.len() == n * patch_dim,
            "pixel_values has {} elements, grid ({gh}, {gw}) x {patch_dim} needs {}",
            pixels.len(),
            n * patch_dim
        );
        let scale = (v.head_dim() as f32).powf(-0.5);
        let depth = blocks;
        let merge_dim = v.merge_dim();
        let out_hidden = v.out_hidden_size;
        Self::ensure_scratch(ctx, &mut self.scratch, &self.config, n)?;
        let s = self.scratch.as_ref().expect("scratch allocated");
        let w = &self.weights;

        let px = s.pixels.view(0, &[n, patch_dim])?;
        let x = s.x.view(0, &[n, hid])?;
        let h = s.h.view(0, &[n, hid])?;
        let qkv = s.qkv.view(0, &[n, 3 * hid])?;
        let attn = s.attn.view(0, &[n, hid])?;
        let tmp = s.tmp.view(0, &[n, hid])?;
        let mlp = s.mlp.view(0, &[n, v.intermediate_size])?;
        let h4 = s.h.view(0, &[n / merge, merge_dim])?;
        let mlp4 = s.mlp.view(0, &[n / merge, merge_dim])?;
        let merged = s.merged.view(0, &[n / merge, out_hidden])?;

        // The cast is the reference's `hidden_states.to(target_dtype)`.
        let cast: Vec<bf16> = pixels.iter().map(|&p| bf16::from_f32(p)).collect();
        px.write_bytes(bytemuck::cast_slice(&cast))?;

        let pass = ctx.begin()?;
        pass.set_label("vision");
        gemm_bf16_nt_bias(ctx, &pass, &px, &w.patch_proj_w, &w.patch_proj_b, &x)?;
        patch_pos_embed_bf16(ctx, &pass, &x, &w.pos_embed, gh, gw)?;
        for b in &w.blocks[..depth] {
            layernorm_bf16(ctx, &pass, &x, &b.norm1_w, &b.norm1_b, &h, LAYERNORM_EPS)?;
            gemm_bf16_nt_bias(ctx, &pass, &h, &b.qkv_w, &b.qkv_b, &qkv)?;
            qkv_rope_2d_bf16(ctx, &pass, &qkv, heads, gh, gw, ROPE_THETA)?;
            attention_full_bf16(ctx, &pass, &qkv, &attn, heads, scale)?;
            gemm_bf16_nt_bias(ctx, &pass, &attn, &b.proj_w, &b.proj_b, &tmp)?;
            add_bf16(ctx, &pass, &x, &tmp, &x)?;
            layernorm_bf16(ctx, &pass, &x, &b.norm2_w, &b.norm2_b, &h, LAYERNORM_EPS)?;
            gemm_bf16_nt_bias(ctx, &pass, &h, &b.fc1_w, &b.fc1_b, &mlp)?;
            gelu_bf16(ctx, &pass, &mlp, Gelu::Tanh)?;
            gemm_bf16_nt_bias(ctx, &pass, &mlp, &b.fc2_w, &b.fc2_b, &tmp)?;
            add_bf16(ctx, &pass, &x, &tmp, &x)?;
        }
        // Merger: the per-patch norm lands in `h`, whose four consecutive
        // rows per merge block are one `[4 H]` row of `h4` by layout.
        layernorm_bf16(
            ctx,
            &pass,
            &x,
            &w.merger_norm_w,
            &w.merger_norm_b,
            &h,
            LAYERNORM_EPS,
        )?;
        gemm_bf16_nt_bias(ctx, &pass, &h4, &w.merger_fc1_w, &w.merger_fc1_b, &mlp4)?;
        gelu_bf16(ctx, &pass, &mlp4, Gelu::Erf)?;
        gemm_bf16_nt_bias(
            ctx,
            &pass,
            &mlp4,
            &w.merger_fc2_w,
            &w.merger_fc2_b,
            &merged,
        )?;
        let done = pass.commit()?.wait_retain()?;
        let host_secs = started.elapsed().as_secs_f64();
        let timing = done.timing()?;
        Ok(VisionOutput {
            merged,
            pre_merger: x,
            gpu_secs: timing.gpu_end_secs - timing.gpu_start_secs,
            host_secs,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/vision.rs"]
mod tests;
