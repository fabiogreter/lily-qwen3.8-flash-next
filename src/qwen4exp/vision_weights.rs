//! The vision tower's weights (`model.visual.*`), resident bf16 on the GPU
//! exactly as the converter copied them (`docs/qwen38-flash-next-checkpoint-format.md`).
//! Every shape is checked against `VisionConfig`, so a checkpoint whose
//! `config.json` and tensors disagree fails at load. Loading only: the
//! compute is `docs/vision-support-plan.md` item 3, and the layouts here are
//! the ones it will consume (`tools/reference/VISION.md`, "The tower").

use anyhow::{Result, ensure};

use crate::tensor::{DType, Tensor};
use crate::weights::{Loader, expect_shape};

use super::config::{VISION_PREFIX, VisionConfig};

/// One transformer block: pre-norm attention and pre-norm MLP, both
/// LayerNorm with bias, every projection with bias.
pub struct VisionBlockWeights {
    /// `[hidden]` each.
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    /// `[3 * hidden, hidden]`: q | k | v rows.
    pub qkv_w: Tensor,
    /// `[3 * hidden]`.
    pub qkv_b: Tensor,
    /// `[hidden, hidden]`.
    pub proj_w: Tensor,
    pub proj_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    /// `[intermediate, hidden]`.
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    /// `[hidden, intermediate]`.
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

pub struct VisionWeights {
    /// The patch embedding's `Conv3d` weight `[hidden, C, T, P, P]` viewed as
    /// the `[hidden, C * T * P * P]` linear map it is when the kernel takes one
    /// step per patch. Row-major flattening of the trailing dims is the
    /// (C, T, H, W) order of a preprocessed pixel row, so this is a reshape of
    /// the checkpoint bytes, not a permutation.
    pub patch_proj_w: Tensor,
    /// `[hidden]`.
    pub patch_proj_b: Tensor,
    /// `[num_position_embeddings, hidden]`: the learned table, resampled to
    /// the image grid at run time.
    pub pos_embed: Tensor,
    pub blocks: Vec<VisionBlockWeights>,
    /// `[hidden]` each: the pre-shuffle LayerNorm of the merger.
    pub merger_norm_w: Tensor,
    pub merger_norm_b: Tensor,
    /// `[merge_dim, merge_dim]` with `merge_dim = hidden * merge * merge`.
    pub merger_fc1_w: Tensor,
    pub merger_fc1_b: Tensor,
    /// `[out_hidden, merge_dim]`: lands in the text model's hidden size.
    pub merger_fc2_w: Tensor,
    pub merger_fc2_b: Tensor,
    bytes: usize,
}

impl VisionWeights {
    /// GPU bytes the tower occupies.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Reads every tower tensor into GPU memory, checking dtype and shape.
pub(crate) fn load(loader: &Loader<'_>, v: &VisionConfig) -> Result<VisionWeights> {
    let h = v.hidden_size;
    let bytes = std::cell::Cell::new(0usize);
    let tensor = |name: &str, shape: &[usize]| -> Result<Tensor> {
        let full = format!("{VISION_PREFIX}{name}");
        let t = loader.tensor(&full)?;
        ensure!(
            t.dtype() == DType::BF16,
            "{full} is {:?}, expected BF16 (the tower is stored unquantized)",
            t.dtype()
        );
        expect_shape(&t, shape, &full)?;
        bytes.set(bytes.get() + t.byte_len());
        Ok(t)
    };

    let conv = tensor(
        "patch_embed.proj.weight",
        &[h, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size],
    )?;
    let patch_proj_w = conv.view(0, &[h, v.patch_dim()])?;
    let patch_proj_b = tensor("patch_embed.proj.bias", &[h])?;
    let pos_embed = tensor("pos_embed.weight", &[v.num_position_embeddings, h])?;

    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let p = format!("blocks.{i}.");
        blocks.push(VisionBlockWeights {
            norm1_w: tensor(&format!("{p}norm1.weight"), &[h])?,
            norm1_b: tensor(&format!("{p}norm1.bias"), &[h])?,
            qkv_w: tensor(&format!("{p}attn.qkv.weight"), &[3 * h, h])?,
            qkv_b: tensor(&format!("{p}attn.qkv.bias"), &[3 * h])?,
            proj_w: tensor(&format!("{p}attn.proj.weight"), &[h, h])?,
            proj_b: tensor(&format!("{p}attn.proj.bias"), &[h])?,
            norm2_w: tensor(&format!("{p}norm2.weight"), &[h])?,
            norm2_b: tensor(&format!("{p}norm2.bias"), &[h])?,
            fc1_w: tensor(
                &format!("{p}mlp.linear_fc1.weight"),
                &[v.intermediate_size, h],
            )?,
            fc1_b: tensor(&format!("{p}mlp.linear_fc1.bias"), &[v.intermediate_size])?,
            fc2_w: tensor(
                &format!("{p}mlp.linear_fc2.weight"),
                &[h, v.intermediate_size],
            )?,
            fc2_b: tensor(&format!("{p}mlp.linear_fc2.bias"), &[h])?,
        });
    }

    let m = v.merge_dim();
    let merger_norm_w = tensor("merger.norm.weight", &[h])?;
    let merger_norm_b = tensor("merger.norm.bias", &[h])?;
    let merger_fc1_w = tensor("merger.linear_fc1.weight", &[m, m])?;
    let merger_fc1_b = tensor("merger.linear_fc1.bias", &[m])?;
    let merger_fc2_w = tensor("merger.linear_fc2.weight", &[v.out_hidden_size, m])?;
    let merger_fc2_b = tensor("merger.linear_fc2.bias", &[v.out_hidden_size])?;

    Ok(VisionWeights {
        patch_proj_w,
        patch_proj_b,
        pos_embed,
        blocks,
        merger_norm_w,
        merger_norm_b,
        merger_fc1_w,
        merger_fc1_b,
        merger_fc2_w,
        merger_fc2_b,
        bytes: bytes.get(),
    })
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/vision_weights.rs"]
mod tests;
