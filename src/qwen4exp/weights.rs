//! Checkpoint loading for Qwen3.8-Flash-Next in lily's `qwen4_exp-affine-v1`
//! layout (`docs/qwen38-flash-next-checkpoint-format.md`). Tensor names are
//! the Hugging Face names; every linear projection and both embedding tables
//! are affine `{weight, scales, biases}` triples. Every tensor in the file must
//! be consumed or explicitly skip-listed so a name-scheme drift fails loudly.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::metal::MetalContext;
use crate::safetensors::Checkpoint;
use crate::tensor::Tensor;
use crate::weights::{LinearWeights, Loader, MlpWeights, MoeWeights, expect_shape};

use super::config::{LayerType, Qwen4ExpConfig};
use super::ngram::{self, NgramStorage, NgramTable, PagedTable};

const PREFIX: &str = "model.language_model.";
/// Skipped tensors: the vision tower is never converted, and the draft head
/// (`mtp.*`) is only read when the caller asks for it.
const SKIP_PREFIXES: &[&str] = &["model.visual.", "mtp."];
const MTP_PREFIX: &str = "mtp.";

/// One hyper-connection (gated residual) block: a grouped norm over the
/// `hc_count` streams, the low-rank read gate, and the per-stream write gate
/// (absent on the model-level mixer).
pub struct HcWeights {
    /// `[hc_count * h]` zero-centered norm weight.
    pub norm: Tensor,
    /// `[hc_lowrank, hc_count * h]`.
    pub down: LinearWeights,
    /// `[hc_count * h, hc_lowrank]`.
    pub up: LinearWeights,
    /// `[hc_count, hc_count * h]`.
    pub inject: Option<LinearWeights>,
}

/// Per-layer n-gram embedding module.
pub struct PleWeights {
    /// `[padded_vocab, head_dim]` hashed n-gram table: resident in one GPU
    /// buffer, or paged from the checkpoint files.
    pub table: NgramTable,
    /// `[hc_count * h, embed_dim]`.
    pub key_proj: LinearWeights,
    /// `[h, embed_dim]`.
    pub value_proj: LinearWeights,
    pub norm_key: Tensor,
    pub norm_query: Tensor,
    pub norm_conv: Tensor,
    /// Tap-major `[kernel, hc_count * h]` dilated depthwise conv.
    pub conv_w: Tensor,
}

/// QSA indexer: fused `[q heads | key] ` projection plus the two head norms.
pub struct IndexerWeights {
    pub qk_proj: LinearWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub struct AttnWeights {
    /// `[nq*2*hd + 2*nkv*hd, h]` — q(|gate) | k | v rows fused.
    pub qkv_proj: LinearWeights,
    pub q_proj: LinearWeights,
    pub k_proj: LinearWeights,
    pub v_proj: LinearWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub o_proj: LinearWeights,
    pub indexer: IndexerWeights,
}

pub struct GdnWeights {
    /// `[conv_c + dim_v + 2*heads, h]` — qkv | z | a | b rows fused.
    pub in_proj: LinearWeights,
    pub in_proj_qkv: LinearWeights,
    pub in_proj_z: LinearWeights,
    pub in_proj_a: LinearWeights,
    pub in_proj_b: LinearWeights,
    /// Tap-major `[kernel_dim, conv_channels]`.
    pub conv_w: Tensor,
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub norm_w: Tensor,
    pub out_proj: LinearWeights,
}

pub enum Mixer {
    Gdn(Box<GdnWeights>),
    Attn(Box<AttnWeights>),
}

pub struct LayerWeights {
    pub ple: Option<Box<PleWeights>>,
    pub attn_hc: HcWeights,
    pub mixer: Mixer,
    pub mlp_hc: HcWeights,
    pub ffn: Box<MoeWeights>,
}

/// The multi-token-prediction draft head: the trunk's wide residual and the
/// next token's embedding are normed, projected and summed per stream into a
/// fresh residual, one trunk-style attention+MoE block runs on it, and its own
/// stream mixer feeds the shared LM head.
pub struct MtpWeights {
    /// `[h]` zero-centered norm of the token embedding.
    pub norm_embedding: Tensor,
    /// `[hc_count * h]` zero-centered grouped norm of the incoming residual.
    pub norm_hidden: Tensor,
    /// `[h, h]`, applied to the normed embedding (8-bit).
    pub fc_embedding: LinearWeights,
    /// `[h, h]`, applied to each normed stream (8-bit).
    pub fc_hidden: LinearWeights,
    /// The block: always a full-attention layer with its own indexer.
    pub layer: LayerWeights,
    /// The head-level read that collapses the streams before the LM head.
    pub mixer: HcWeights,
}

pub struct ModelWeights {
    /// `[vocab, h]` token table.
    pub embed_tokens: LinearWeights,
    /// Untied LM head.
    pub lm_head: LinearWeights,
    /// The model-level read that collapses the streams before the LM head.
    pub final_mixer: HcWeights,
    pub layers: Vec<LayerWeights>,
    /// The draft head, when the checkpoint has it and the caller wanted it.
    pub mtp: Option<Box<MtpWeights>>,
}

/// The converter's storage policy: routers, gates and the small mixing
/// projections are 8-bit, everything else 4-bit.
fn expected_bits(bases: &[&str]) -> usize {
    let q8 = bases.len() == 1 && {
        let b = bases[0];
        b.ends_with(".mlp.gate")
            || b.ends_with(".mlp.shared_expert_gate")
            || b.ends_with(".input_mix_weight_down")
            || b.ends_with(".input_mix_weight_up")
            || b.ends_with(".block_inject_weight")
            || b.ends_with(".ple.key_proj")
            || b.ends_with(".ple.value_proj")
            || b.ends_with(".indexer.index_qk_proj")
            || b == "mtp.fc_embedding"
            || b == "mtp.fc_hidden"
    };
    if q8 { 8 } else { 4 }
}

fn load_mtp(loader: &Loader<'_>, config: &Qwen4ExpConfig) -> Result<MtpWeights> {
    let h = config.hidden_size;
    let wide = config.hc_width();
    let norm_embedding = loader.tensor(&format!("{MTP_PREFIX}pre_fc_norm_embedding.weight"))?;
    expect_shape(&norm_embedding, &[h], "mtp pre_fc_norm_embedding")?;
    let norm_hidden = loader.tensor(&format!("{MTP_PREFIX}pre_fc_norm_hidden.weight"))?;
    expect_shape(&norm_hidden, &[wide], "mtp pre_fc_norm_hidden")?;
    let fc_embedding = loader.linear(&[&format!("{MTP_PREFIX}fc_embedding")], h)?;
    fc_embedding.expect_features(h, h, "mtp fc_embedding")?;
    let fc_hidden = loader.linear(&[&format!("{MTP_PREFIX}fc_hidden")], h)?;
    fc_hidden.expect_features(h, h, "mtp fc_hidden")?;
    let p = format!("{MTP_PREFIX}layers.0.");
    let attn_hc = load_hc(loader, &format!("{p}attn_hyper_connection."), config, true)?;
    let mixer_w = Mixer::Attn(Box::new(load_attn(loader, &p, config)?));
    let mlp_hc = load_hc(loader, &format!("{p}mlp_hyper_connection."), config, true)?;
    let ffn = load_ffn(loader, &p, config)?;
    let mixer = load_hc(loader, &format!("{MTP_PREFIX}hyper_connection_mixer."), config, false)?;
    Ok(MtpWeights {
        norm_embedding,
        norm_hidden,
        fc_embedding,
        fc_hidden,
        layer: LayerWeights { ple: None, attn_hc, mixer: mixer_w, mlp_hc, ffn },
        mixer,
    })
}

fn load_hc(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
    with_inject: bool,
) -> Result<HcWeights> {
    let wide = config.hc_width();
    let norm = loader.tensor(&format!("{p}hc_norm.weight"))?;
    expect_shape(&norm, &[wide], "hc_norm")?;
    let down = loader.linear(&[&format!("{p}input_mix_weight_down")], wide)?;
    down.expect_features(config.hc_lowrank, wide, "input_mix_weight_down")?;
    let up = loader.linear(&[&format!("{p}input_mix_weight_up")], config.hc_lowrank)?;
    up.expect_features(wide, config.hc_lowrank, "input_mix_weight_up")?;
    let inject = if with_inject {
        let w = loader.linear(&[&format!("{p}block_inject_weight")], wide)?;
        w.expect_features(config.hc_count, wide, "block_inject_weight")?;
        Some(w)
    } else {
        None
    };
    Ok(HcWeights { norm, down, up, inject })
}

fn load_mlp(
    loader: &Loader<'_>,
    prefix: &str,
    h: usize,
    i: usize,
) -> Result<MlpWeights> {
    let gate_up_proj = loader
        .linear(&[&format!("{prefix}gate_proj"), &format!("{prefix}up_proj")], h)?;
    gate_up_proj.expect_features(2 * i, h, "shared gate_up_proj")?;
    let gate_proj = gate_up_proj.view_rows(0, i)?;
    let up_proj = gate_up_proj.view_rows(i, i)?;
    let down_proj = loader.linear(&[&format!("{prefix}down_proj")], i)?;
    down_proj.expect_features(h, i, "shared down_proj")?;
    Ok(MlpWeights { gate_up_proj, gate_proj, up_proj, down_proj })
}

fn load_ffn(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
) -> Result<Box<MoeWeights>> {
    let h = config.hidden_size;
    let (e, i) = (config.num_experts, config.moe_intermediate_size);
    let gate = loader.linear(&[&format!("{p}mlp.gate")], h)?;
    gate.expect_features(e, h, "router gate")?;
    let expert_gate = loader.linear(&[&format!("{p}mlp.experts.gate_proj")], h)?;
    expert_gate.expect_features(e * i, h, "expert gate_proj")?;
    let expert_up = loader.linear(&[&format!("{p}mlp.experts.up_proj")], h)?;
    expert_up.expect_features(e * i, h, "expert up_proj")?;
    let expert_down = loader.linear(&[&format!("{p}mlp.experts.down_proj")], i)?;
    expert_down.expect_features(e * h, i, "expert down_proj")?;
    let shared = load_mlp(
        loader,
        &format!("{p}mlp.shared_expert."),
        h,
        config.shared_expert_intermediate_size,
    )?;
    let shared_gate = loader.linear(&[&format!("{p}mlp.shared_expert_gate")], h)?;
    shared_gate.expect_features(1, h, "shared_expert_gate")?;
    Ok(Box::new(MoeWeights {
        gate,
        expert_gate,
        expert_up,
        expert_down,
        shared,
        shared_gate,
    }))
}

fn load_gdn(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
) -> Result<GdnWeights> {
    let h = config.hidden_size;
    let la = format!("{p}linear_attn.");
    let heads = config.linear_num_value_heads;
    let dim_v = heads * config.linear_value_head_dim;
    let conv_c = config.gdn_conv_channels();

    let in_proj = loader.linear(
        &[
            &format!("{la}in_proj_qkv"),
            &format!("{la}in_proj_z"),
            &format!("{la}in_proj_a"),
            &format!("{la}in_proj_b"),
        ],
        h,
    )?;
    in_proj.expect_features(conv_c + dim_v + 2 * heads, h, "in_proj")?;
    let in_proj_qkv = in_proj.view_rows(0, conv_c)?;
    let in_proj_z = in_proj.view_rows(conv_c, dim_v)?;
    let in_proj_a = in_proj.view_rows(conv_c + dim_v, heads)?;
    let in_proj_b = in_proj.view_rows(conv_c + dim_v + heads, heads)?;
    let conv_w = loader.conv_weight(&format!("{la}conv1d.weight"))?;
    let a_log = loader.tensor_f32(&format!("{la}A_log"))?;
    let dt_bias = loader.tensor(&format!("{la}dt_bias"))?;
    let norm_w = loader.tensor_f32(&format!("{la}norm.weight"))?;
    let out_proj = loader.linear(&[&format!("{la}out_proj")], dim_v)?;

    expect_shape(&conv_w, &[config.linear_conv_kernel_dim, conv_c], "conv_w")?;
    expect_shape(&a_log, &[heads], "A_log")?;
    expect_shape(&dt_bias, &[heads], "dt_bias")?;
    expect_shape(&norm_w, &[config.linear_value_head_dim], "gdn norm")?;
    out_proj.expect_features(h, dim_v, "out_proj")?;

    Ok(GdnWeights {
        in_proj,
        in_proj_qkv,
        in_proj_z,
        in_proj_a,
        in_proj_b,
        conv_w,
        a_log,
        dt_bias,
        norm_w,
        out_proj,
    })
}

fn load_attn(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
) -> Result<AttnWeights> {
    let h = config.hidden_size;
    let sa = format!("{p}self_attn.");
    let (hd, nq, nkv) =
        (config.head_dim, config.num_attention_heads, config.num_key_value_heads);
    // The q projection carries a per-head output gate: [q | gate] per head.
    let q_rows = 2 * nq * hd;

    let qkv_proj = loader.linear(
        &[&format!("{sa}q_proj"), &format!("{sa}k_proj"), &format!("{sa}v_proj")],
        h,
    )?;
    qkv_proj.expect_features(q_rows + 2 * nkv * hd, h, "qkv_proj")?;
    let q_proj = qkv_proj.view_rows(0, q_rows)?;
    let k_proj = qkv_proj.view_rows(q_rows, nkv * hd)?;
    let v_proj = qkv_proj.view_rows(q_rows + nkv * hd, nkv * hd)?;
    let q_norm = loader.tensor(&format!("{sa}q_norm.weight"))?;
    let k_norm = loader.tensor(&format!("{sa}k_norm.weight"))?;
    let o_proj = loader.linear(&[&format!("{sa}o_proj")], nq * hd)?;
    expect_shape(&q_norm, &[hd], "q_norm")?;
    expect_shape(&k_norm, &[hd], "k_norm")?;
    o_proj.expect_features(h, nq * hd, "o_proj")?;

    let ix = format!("{sa}indexer.");
    let idx = &config.indexer;
    let qk_proj = loader.linear(&[&format!("{ix}index_qk_proj")], h)?;
    qk_proj.expect_features((idx.n_heads + 1) * idx.head_dim, h, "index_qk_proj")?;
    let iq_norm = loader.tensor(&format!("{ix}q_layernorm.weight"))?;
    let ik_norm = loader.tensor(&format!("{ix}k_layernorm.weight"))?;
    expect_shape(&iq_norm, &[idx.head_dim], "indexer q_layernorm")?;
    expect_shape(&ik_norm, &[idx.head_dim], "indexer k_layernorm")?;

    Ok(AttnWeights {
        qkv_proj,
        q_proj,
        k_proj,
        v_proj,
        q_norm,
        k_norm,
        o_proj,
        indexer: IndexerWeights { qk_proj, q_norm: iq_norm, k_norm: ik_norm },
    })
}

fn load_ple(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
    storage: NgramStorage,
) -> Result<PleWeights> {
    let ple = config.ple.as_ref().expect("PLE layer without PLE config");
    let h = config.hidden_size;
    let wide = config.hc_width();
    let pp = format!("{p}ple.");
    let shard_names = ngram::shard_bases(&pp, ple.table_shards);
    let table = match storage {
        NgramStorage::Resident => {
            let shard_refs: Vec<&str> = shard_names.iter().map(String::as_str).collect();
            let table = loader.linear_grouped(
                &shard_refs,
                ple.head_dim(),
                ple.quantization.group_size,
            )?;
            table.expect_features(ple.padded_vocab_size, ple.head_dim(), "ngram table")?;
            ensure!(table.bits == ple.quantization.bits, "ngram table bit width mismatch");
            NgramTable::Resident(Box::new(table))
        }
        NgramStorage::Paged => {
            ensure!(
                ple.quantization.bits == PagedTable::BITS,
                "paged n-gram table supports {}-bit tables only",
                PagedTable::BITS
            );
            for base in &shard_names {
                loader.mark_consumed(&format!("{base}.weight"));
                loader.mark_consumed(&format!("{base}.scales"));
                loader.mark_consumed(&format!("{base}.biases"));
            }
            let table =
                PagedTable::open(loader.checkpoint(), &shard_names, ple.quantization.group_size)?;
            ensure!(
                table.rows() == ple.padded_vocab_size && table.width() == ple.head_dim(),
                "ngram table is [{}, {}], expected [{}, {}]",
                table.rows(),
                table.width(),
                ple.padded_vocab_size,
                ple.head_dim()
            );
            NgramTable::Paged(Arc::new(table))
        }
    };

    let key_proj = loader.linear(&[&format!("{pp}key_proj")], ple.embed_dim)?;
    key_proj.expect_features(wide, ple.embed_dim, "ple key_proj")?;
    let value_proj = loader.linear(&[&format!("{pp}value_proj")], ple.embed_dim)?;
    value_proj.expect_features(h, ple.embed_dim, "ple value_proj")?;
    let norm_key = loader.tensor(&format!("{pp}norm_key.weight"))?;
    let norm_query = loader.tensor(&format!("{pp}norm_query.weight"))?;
    let norm_conv = loader.tensor(&format!("{pp}norm_conv.weight"))?;
    for (name, t) in [
        ("norm_key", &norm_key),
        ("norm_query", &norm_query),
        ("norm_conv", &norm_conv),
    ] {
        expect_shape(t, &[wide], name)?;
    }
    let conv_w = loader.conv_weight(&format!("{pp}conv1d.weight"))?;
    expect_shape(&conv_w, &[ple.conv_kernel_size, wide], "ple conv_w")?;

    Ok(PleWeights {
        table,
        key_proj,
        value_proj,
        norm_key,
        norm_query,
        norm_conv,
        conv_w,
    })
}

/// Loads the trunk, and the draft head when `with_mtp` is set and the
/// checkpoint declares one (the `mtp.*` tensors are skipped otherwise).
pub fn load(
    ctx: &MetalContext,
    dir: impl AsRef<Path>,
    config: &Qwen4ExpConfig,
    storage: NgramStorage,
    with_mtp: bool,
) -> Result<ModelWeights> {
    let ckpt = Checkpoint::open(&dir)?;
    ensure!(
        ckpt.meta(&format!("{PREFIX}embed_tokens.weight")).is_some(),
        "unsupported checkpoint layout; expected lily's qwen4_exp-affine-v1"
    );
    let loader =
        Loader::new(ctx, ckpt, config.quantization, SKIP_PREFIXES, expected_bits);
    let h = config.hidden_size;

    let embed_tokens = loader.linear(&[&format!("{PREFIX}embed_tokens")], h)?;
    embed_tokens.expect_features(config.vocab_size, h, "embed_tokens")?;
    let lm_head = loader.linear(&["lm_head"], h)?;
    lm_head.expect_features(config.vocab_size, h, "lm_head")?;
    let final_mixer =
        load_hc(&loader, &format!("{PREFIX}hyper_connection_mixer."), config, false)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (idx, layer_type) in config.layer_types.iter().enumerate() {
        let p = format!("{PREFIX}layers.{idx}.");
        let ple = match &config.ple {
            Some(ple) if ple.layer == idx => {
                Some(Box::new(load_ple(&loader, &p, config, storage)?))
            }
            _ => None,
        };
        let attn_hc =
            load_hc(&loader, &format!("{p}attn_hyper_connection."), config, true)?;
        let mixer = match layer_type {
            LayerType::LinearAttention => {
                Mixer::Gdn(Box::new(load_gdn(&loader, &p, config)?))
            }
            LayerType::FullAttention => {
                Mixer::Attn(Box::new(load_attn(&loader, &p, config)?))
            }
        };
        let mlp_hc =
            load_hc(&loader, &format!("{p}mlp_hyper_connection."), config, true)?;
        let ffn = load_ffn(&loader, &p, config)?;
        layers.push(LayerWeights { ple, attn_hc, mixer, mlp_hc, ffn });
    }

    let mtp = match (&config.mtp, with_mtp) {
        (Some(_), true) => Some(Box::new(load_mtp(&loader, config)?)),
        _ => None,
    };

    loader.finish()?;
    Ok(ModelWeights { embed_tokens, lm_head, final_mixer, layers, mtp })
}
