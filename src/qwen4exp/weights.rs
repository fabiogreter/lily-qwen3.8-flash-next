//! Checkpoint loading for Qwen3.8-Flash-Next in lily's `qwen4_exp-affine-v1`
//! layout (`docs/qwen38-flash-next-checkpoint-format.md`). Tensor names are
//! the Hugging Face names; every linear projection and both embedding tables
//! are affine `{weight, scales, biases}` triples, the vision tower stays bf16.
//! Every tensor in the file must be consumed or explicitly skip-listed so a
//! name-scheme drift fails loudly.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::metal::{Buffer, MetalContext};
use crate::safetensors::Checkpoint;
use crate::tensor::Tensor;
use std::path::PathBuf;

use anyhow::Context as _;

use super::expert_cache::ExpertCache;
use super::expert_store::{ExpertStore, LiveUsage, SlotPolicy, UsageRanking};
use crate::weights::{LinearWeights, Loader, MlpWeights, MoeWeights, expect_shape};

use super::config::{LayerType, Q8Group, Qwen4ExpConfig, StoragePolicy, VISION_PREFIX};
use super::ngram::{self, NgramStorage, NgramTable, PagedTable};
use super::vision_weights::{self, VisionWeights};

const PREFIX: &str = "model.language_model.";
/// Skipped tensors: the draft head (`mtp.*`) is only read when the caller
/// asks for it, and the vision tower only when it is loaded.
const SKIP_PREFIXES_TEXT_ONLY: &[&str] = &[VISION_PREFIX, "mtp."];
const SKIP_PREFIXES_WITH_VISION: &[&str] = &["mtp."];
const MTP_PREFIX: &str = "mtp.";
/// The draft head's own 4-bit copy of `lm_head` (`--draft-q4`).
const MTP_LM_HEAD: &str = "mtp.lm_head";

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
    /// The head's own 4-bit LM head (`mtp.lm_head`), when the conversion
    /// stored the trunk's at 8 bits with `--draft-q4`
    /// (`StoragePolicy::draft_head_copy`); `None`: the head reads the trunk's.
    pub lm_head: Option<LinearWeights>,
}

impl MtpWeights {
    /// The LM head the draft head's logits use: its own copy, else the
    /// trunk's `lm_head`.
    pub fn head<'a>(&'a self, trunk: &'a LinearWeights) -> &'a LinearWeights {
        self.lm_head.as_ref().unwrap_or(trunk)
    }
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
    /// The vision tower, when the checkpoint has it and the caller wanted it.
    pub vision: Option<Box<VisionWeights>>,
    /// The expert cache when the experts are served from a slab of slots
    /// (`LoadOptions::expert_slots`); the layers' `MoeWeights` view it.
    pub expert_cache: Option<ExpertCache>,
    /// Every GPU buffer the load allocated for the weights above (views
    /// share them), without the expert cache's slab and slot tables and
    /// without the paged n-gram table, which is a file mapping: what the
    /// server's `--pin-weights` locks. Handles only; the memory is the same.
    pub buffers: Vec<Buffer>,
    /// The memory the load planned for: the budget it was given, else the
    /// machine's physical memory (`None` when neither is known), exactly
    /// what `auto_expert_slots` sized the expert cache from.
    pub planned_memory: Option<u64>,
    /// What the expert cache's plan kept free for the server's session
    /// cache: one full session at `SessionContext::max_seq`. `None` when the
    /// checkpoint fits, when the slots were given explicitly, or when the
    /// load had no session context (the bench, the probes).
    pub session_reserve: Option<u64>,
}

/// The session the server keeps room for under an expert cache: its context
/// length and how many recurrent checkpoints a session holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionContext {
    pub max_seq: usize,
    pub checkpoints: usize,
}

/// The converter's storage policy: routers, gates and the small mixing
/// projections are always 8-bit; the projections of each `Q8Group` are
/// whatever the checkpoint's config records (`StoragePolicy`, 4-bit unless
/// converted with `--q8`), in the draft head (`mtp.*`, its LM head copy
/// included) at `StoragePolicy::draft_bits`; the routed experts are 4-bit. A
/// fused stack (q|k|v, the GDN `in_proj_*`, the shared gate|up) is one width:
/// every base must be in the same group, and the loader's row concatenation
/// rejects slices of different packed widths anyway.
fn expected_bits(storage: StoragePolicy, bases: &[&str]) -> usize {
    let fixed_q8 = bases.len() == 1 && {
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
    if fixed_q8 {
        return 8;
    }
    let first = bases.first().and_then(|b| q8_group(b));
    match first {
        Some(g) if bases.iter().all(|b| q8_group(b) == Some(g)) => {
            if bases.iter().all(|b| b.starts_with(MTP_PREFIX)) {
                storage.draft_bits(g)
            } else {
                storage.bits(g)
            }
        }
        _ => 4,
    }
}

/// The `Q8Group` a tensor base belongs to, if any.
fn q8_group(base: &str) -> Option<Q8Group> {
    const SUFFIXES: &[(&str, Q8Group)] = &[
        (".self_attn.q_proj", Q8Group::Attn),
        (".self_attn.k_proj", Q8Group::Attn),
        (".self_attn.v_proj", Q8Group::Attn),
        (".self_attn.o_proj", Q8Group::Attn),
        (".linear_attn.in_proj_qkv", Q8Group::Gdn),
        (".linear_attn.in_proj_z", Q8Group::Gdn),
        (".linear_attn.in_proj_a", Q8Group::Gdn),
        (".linear_attn.in_proj_b", Q8Group::Gdn),
        (".linear_attn.out_proj", Q8Group::Gdn),
        (".mlp.shared_expert.gate_proj", Q8Group::Shared),
        (".mlp.shared_expert.up_proj", Q8Group::Shared),
        (".mlp.shared_expert.down_proj", Q8Group::Shared),
    ];
    if base == "lm_head" || base == MTP_LM_HEAD {
        return Some(Q8Group::Head);
    }
    if base.strip_prefix(PREFIX) == Some("embed_tokens") {
        return Some(Q8Group::Embed);
    }
    SUFFIXES.iter().find(|(s, _)| base.ends_with(s)).map(|&(_, g)| g)
}

fn load_mtp(loader: &Loader<'_>, config: &Qwen4ExpConfig) -> Result<MtpWeights> {
    let h = config.hidden_size;
    let wide = config.hc_width();
    let norm_embedding =
        loader.tensor(&format!("{MTP_PREFIX}pre_fc_norm_embedding.weight"))?;
    expect_shape(&norm_embedding, &[h], "mtp pre_fc_norm_embedding")?;
    let norm_hidden =
        loader.tensor(&format!("{MTP_PREFIX}pre_fc_norm_hidden.weight"))?;
    expect_shape(&norm_hidden, &[wide], "mtp pre_fc_norm_hidden")?;
    let fc_embedding = loader.linear(&[&format!("{MTP_PREFIX}fc_embedding")], h)?;
    fc_embedding.expect_features(h, h, "mtp fc_embedding")?;
    let fc_hidden = loader.linear(&[&format!("{MTP_PREFIX}fc_hidden")], h)?;
    fc_hidden.expect_features(h, h, "mtp fc_hidden")?;
    let p = format!("{MTP_PREFIX}layers.0.");
    let attn_hc = load_hc(loader, &format!("{p}attn_hyper_connection."), config, true)?;
    let mixer_w = Mixer::Attn(Box::new(load_attn(loader, &p, config)?));
    let mlp_hc = load_hc(loader, &format!("{p}mlp_hyper_connection."), config, true)?;
    let ffn = load_ffn(loader, &p, config, None)?;
    let mixer = load_hc(
        loader,
        &format!("{MTP_PREFIX}hyper_connection_mixer."),
        config,
        false,
    )?;
    // Without the copy a stray `mtp.lm_head` stays unconsumed and fails the
    // load, so the config and the files agree in both directions.
    let lm_head = if config.storage.draft_head_copy() {
        let w = loader.linear(&[MTP_LM_HEAD], h)?;
        w.expect_features(config.vocab_size, h, "mtp lm_head")?;
        Some(w)
    } else {
        None
    };
    Ok(MtpWeights {
        norm_embedding,
        norm_hidden,
        fc_embedding,
        fc_hidden,
        layer: LayerWeights { ple: None, attn_hc, mixer: mixer_w, mlp_hc, ffn },
        mixer,
        lm_head,
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

/// Physical memory of this machine in bytes (`hw.memsize`).
pub(crate) fn physical_memory() -> Option<u64> {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: sysctlbyname writes at most `len` bytes into `size`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut size as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<u64>()).then_some(size)
}

/// The arithmetic of [`auto_expert_slots`] on its sizes: `ram` the planned
/// memory, `experts` and `other` the checkpoint's expert and resident
/// bytes, `slices` the (layer, expert) count, `session` the session
/// reserve. `None` when everything fits.
fn plan_expert_slots(
    ram: u64,
    experts: u64,
    other: u64,
    slices: u64,
    num_experts: usize,
    session: u64,
) -> Option<usize> {
    const GB: u64 = 1 << 30;
    // Measured with 16 384 slots: 52.9 GB of process footprint for a
    // 45.3 GB slab and 3.1 GB of resident weights, so scratch, caches and
    // pipelines take about 4.5 GB (a figure that still included the
    // 0.5 GB gathered-row scratch of the tiled sparse attention, since
    // removed; the 5 GB below keeps that margin). The reserve keeps 12 GB (or a sixth of memory) for the OS, other apps
    // and the page cache.
    let scratch = 5 * GB;
    let reserve = (12 * GB).max(ram / 6);
    if experts + other + scratch + reserve <= ram {
        return None;
    }
    let slice = experts / slices.max(1);
    let budget = ram.saturating_sub(other + scratch + session + reserve);
    Some(((budget / slice.max(1)) as usize).max(2 * num_experts).min(slices as usize))
}

/// How many expert slots this machine can afford: `None` when the whole
/// checkpoint fits its memory with room to work (the current footprint),
/// otherwise the slots left after the resident weights, about 4.5 GB of
/// scratch, `session` (the server's session cache, one full session at its
/// context length) and a 12 GB reserve for the OS, other apps and the page
/// cache, at least two layers' worth. A 64 GB machine keeps about 13 GB
/// free; the session takes 4.2 GB of what would be experts at a 131 072-token
/// context and 7.9 GB at 262 144. `session` does not count toward whether the checkpoint fits:
/// a machine that holds it keeps its usual session budget.
fn auto_expert_slots(
    ckpt: &Checkpoint,
    config: &Qwen4ExpConfig,
    storage: NgramStorage,
    memory_budget: Option<u64>,
    session: u64,
) -> Option<usize> {
    const GB: u64 = 1 << 30;
    let ram = memory_budget.or_else(physical_memory)?;
    let (mut experts, mut other) = (0u64, 0u64);
    for name in ckpt.names() {
        let bytes = ckpt.meta(name)?.byte_len() as u64;
        if name.starts_with(PREFIX) && name.contains(".mlp.experts.") {
            experts += bytes;
        } else if storage == NgramStorage::Paged
            && name.contains("ngram_embedding.shard_")
        {
            // Read from the files on demand; the page cache holds what it can.
        } else {
            other += bytes;
        }
    }
    let slices = (config.num_hidden_layers * config.num_experts) as u64;
    let slice = experts / slices.max(1);
    let slots =
        plan_expert_slots(ram, experts, other, slices, config.num_experts, session)?;
    eprintln!(
        "expert cache: {:.1} GB of memory holds {:.1} GB of resident weights, {:.1} GB for one full session and {} of {slices} experts ({:.1} of {:.1} GB); the rest is served from the checkpoint",
        ram as f64 / GB as f64,
        other as f64 / GB as f64,
        session as f64 / GB as f64,
        slots,
        slots as f64 * slice as f64 / GB as f64,
        experts as f64 / GB as f64
    );
    if session > 0 {
        eprintln!(
            "expert cache: the session reserve costs {} slots ({:.1} GB); a shorter --max-seq keeps more experts resident",
            (session / slice.max(1)).min(slices),
            session as f64 / GB as f64
        );
    }
    Some(slots)
}

fn load_ffn(
    loader: &Loader<'_>,
    p: &str,
    config: &Qwen4ExpConfig,
    cache: Option<(&ExpertCache, usize)>,
) -> Result<Box<MoeWeights>> {
    let h = config.hidden_size;
    let (e, i) = (config.num_experts, config.moe_intermediate_size);
    let gate = loader.linear(&[&format!("{p}mlp.gate")], h)?;
    gate.expect_features(e, h, "router gate")?;
    let (expert_gate, expert_up, expert_down, slot_of, cache_link) = match cache {
        Some((cache, layer)) => {
            // The layer's experts live in the cache's slab (filled by the
            // caller); its tensors are consumed by the cache, not the loader.
            for name in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    loader.mark_consumed(&format!("{p}mlp.experts.{name}.{suffix}"));
                }
            }
            let (g, u, d, t) = cache.layer_weights(layer)?;
            (g, u, d, Some(t), Some((cache.link(), layer)))
        }
        None => {
            let expert_gate =
                loader.linear(&[&format!("{p}mlp.experts.gate_proj")], h)?;
            expert_gate.expect_features(e * i, h, "expert gate_proj")?;
            let expert_up = loader.linear(&[&format!("{p}mlp.experts.up_proj")], h)?;
            expert_up.expect_features(e * i, h, "expert up_proj")?;
            let expert_down =
                loader.linear(&[&format!("{p}mlp.experts.down_proj")], i)?;
            expert_down.expect_features(e * h, i, "expert down_proj")?;
            (expert_gate, expert_up, expert_down, None, None)
        }
    };
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
        slot_of,
        cache: cache_link,
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
            let shard_refs: Vec<&str> =
                shard_names.iter().map(String::as_str).collect();
            let table = loader.linear_grouped(
                &shard_refs,
                ple.head_dim(),
                ple.quantization.group_size,
            )?;
            table.expect_features(
                ple.padded_vocab_size,
                ple.head_dim(),
                "ngram table",
            )?;
            ensure!(
                table.bits == ple.quantization.bits,
                "ngram table bit width mismatch"
            );
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
            let table = PagedTable::open(
                loader.checkpoint(),
                &shard_names,
                ple.quantization.group_size,
            )?;
            ensure!(
                table.rows() == ple.padded_vocab_size
                    && table.width() == ple.head_dim(),
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

/// Loads the trunk, the draft head when `with_mtp` is set and the checkpoint
/// declares one (the `mtp.*` tensors are skipped otherwise), and the vision
/// tower when `with_vision` is set and the checkpoint declares one.
#[allow(clippy::too_many_arguments)]
pub fn load(
    ctx: &MetalContext,
    dir: impl AsRef<Path>,
    config: &Qwen4ExpConfig,
    storage: NgramStorage,
    with_mtp: bool,
    with_vision: bool,
    expert_slots: Option<usize>,
    expert_usage: Option<PathBuf>,
    memory_budget: Option<u64>,
    expert_usage_out: Option<PathBuf>,
    session_context: Option<SessionContext>,
) -> Result<ModelWeights> {
    // Every process that loads the model passes here (the server, the
    // bench, the probes, the tests with a real checkpoint): one per machine,
    // held until the process exits (`crate::instance`).
    crate::instance::acquire()?;
    // Every buffer allocated from here on that is still alive at the end is
    // a weight, or the expert cache's (filtered out below).
    let recording = ctx.record_buffers()?;
    let ckpt = Checkpoint::open(&dir)?;
    ensure!(
        ckpt.meta(&format!("{PREFIX}embed_tokens.weight")).is_some(),
        "unsupported checkpoint layout; expected lily's qwen4_exp-affine-v1"
    );
    // The config must match the weights in both directions: a tower in the
    // files that config.json does not declare would otherwise be skipped
    // silently; a declared tower whose tensors are missing fails below.
    if config.vision.is_none() {
        let stray = ckpt.names().filter(|n| n.starts_with(VISION_PREFIX)).count();
        ensure!(
            stray == 0,
            "checkpoint holds {stray} {VISION_PREFIX}* tensors but config.json declares no \
             lily.vision block; re-run the converter's --vision-only or fix the config"
        );
    }
    let load_vision = with_vision && config.vision.is_some();
    let skip =
        if load_vision { SKIP_PREFIXES_WITH_VISION } else { SKIP_PREFIXES_TEXT_ONLY };
    let policy = config.storage;
    let loader = Loader::new(
        ctx,
        ckpt,
        config.quantization,
        skip,
        Box::new(move |bases| expected_bits(policy, bases)),
    );
    let h = config.hidden_size;

    let embed_tokens = loader.linear(&[&format!("{PREFIX}embed_tokens")], h)?;
    embed_tokens.expect_features(config.vocab_size, h, "embed_tokens")?;
    let lm_head = loader.linear(&["lm_head"], h)?;
    lm_head.expect_features(config.vocab_size, h, "lm_head")?;
    let final_mixer =
        load_hc(&loader, &format!("{PREFIX}hyper_connection_mixer."), config, false)?;

    // The expert cache: every layer's experts served from one slab of
    // slots, placed by the usage ranking next to the checkpoint (or the
    // one named), uniform without one. Asked for explicitly, or sized from
    // the machine's memory when the checkpoint does not fit it.
    // The server's session cache under an expert cache: one full session at
    // its context length, which the plan keeps free and the server then
    // budgets exactly (`ModelWeights::session_reserve`). The draft head only
    // stays loaded under the cache when asked for (below), so only then does
    // its cache count.
    let drafts_under_cache = with_mtp
        && config.mtp.is_some()
        && std::env::var_os("LILY_EXPERT_CACHE_DRAFTS").is_some();
    let session = match session_context {
        Some(s) => super::model::session_bytes(
            config,
            drafts_under_cache,
            s.max_seq,
            s.checkpoints,
        )?,
        None => 0,
    };
    let planned = expert_slots.is_none();
    let expert_slots = expert_slots.or_else(|| {
        auto_expert_slots(loader.checkpoint(), config, storage, memory_budget, session)
    });
    let session_reserve =
        (planned && expert_slots.is_some() && session_context.is_some())
            .then_some(session);
    let expert_cache = match expert_slots {
        Some(n_slots) => {
            let (layers, e) = (config.num_hidden_layers, config.num_experts);
            let store = ExpertStore::open(loader.checkpoint(), config)?;
            // The usage the cache persists between runs (`LoadOptions::
            // expert_usage_out`, `LILY_EXPERT_USAGE_OUT`); when the file
            // exists it is the ranking, unless one was named explicitly;
            // the shipped one next to the checkpoint comes after; uniform
            // without any.
            let usage_out = std::env::var_os("LILY_EXPERT_USAGE_OUT")
                .map(PathBuf::from)
                .or(expert_usage_out);
            let usage = expert_usage
                .or_else(|| usage_out.clone().filter(|p| p.exists()))
                .or_else(|| {
                    let next_to_it = dir.as_ref().join("expert-usage.json");
                    next_to_it.exists().then_some(next_to_it)
                });
            let ranking = match &usage {
                Some(path) => UsageRanking::load(path)
                    .with_context(|| format!("expert usage {}", path.display()))?,
                None => UsageRanking::uniform(layers, e),
            };
            let adapt = std::env::var("LILY_EXPERT_ADAPT").map_or(true, |v| v != "0");
            let live = LiveUsage::new(&ranking, n_slots, config.num_experts_per_tok);
            ensure!(
                n_slots >= 2 * e,
                "an expert cache needs at least {} slots (two layers' worth), got {n_slots}",
                2 * e
            );
            // The LRU region must hold a whole layer's routing (prefill
            // routes every expert of a layer) plus a decode step's.
            // `LILY_EXPERT_LRU_SHARE` overrides the share for measurement.
            let lru_share = std::env::var("LILY_EXPERT_LRU_SHARE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.1)
                .max((e + 64) as f64 / n_slots as f64)
                .min(0.5);
            let policy = SlotPolicy::new(n_slots, layers, e, &ranking, lru_share)?;
            let mut cache = ExpertCache::new(
                ctx,
                n_slots,
                layers,
                e,
                config.moe_intermediate_size,
                h,
                config.quantization,
            )?;
            eprintln!(
                "expert cache: {n_slots} slots for {} experts, usage {}, live promotion {}, persisted to {}",
                layers * e,
                usage
                    .as_ref()
                    .map_or("uniform".to_string(), |p| p.display().to_string()),
                if adapt { "on" } else { "off (LILY_EXPERT_ADAPT=0)" },
                usage_out
                    .as_ref()
                    .map_or("nowhere".to_string(), |p| p.display().to_string())
            );
            cache.fill_and_serve(store, policy, live, adapt, usage_out)?;
            Some(cache)
        }
        None => None,
    };
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
        let ffn =
            load_ffn(&loader, &p, config, expert_cache.as_ref().map(|c| (c, idx)))?;
        layers.push(LayerWeights { ple, attn_hc, mixer, mlp_hc, ffn });
    }

    // Under an expert cache every trunk pass pays a handshake per MoE
    // layer plus its misses, and a speculative step runs three trunk passes
    // (two drafts, one verify) where plain decode runs one: measured 14 to
    // 15 tok/s against 55 with the reads cold (docs/low-ram-experts.md).
    // So the draft head stays unloaded there unless asked for.
    let with_mtp = with_mtp
        && (expert_cache.is_none()
            || std::env::var_os("LILY_EXPERT_CACHE_DRAFTS").is_some());
    if expert_cache.is_some() && config.mtp.is_some() && !with_mtp {
        eprintln!(
            "expert cache: speculative decoding off (plain decode is faster here; LILY_EXPERT_CACHE_DRAFTS=1 keeps it)"
        );
    }
    let mtp = match (&config.mtp, with_mtp) {
        (Some(_), true) => Some(Box::new(load_mtp(&loader, config)?)),
        _ => None,
    };
    let vision = match (&config.vision, load_vision) {
        (Some(v), true) => Some(Box::new(vision_weights::load(&loader, v)?)),
        _ => None,
    };

    loader.finish()?;
    let slab =
        expert_cache.as_ref().map(ExpertCache::buffer_addresses).unwrap_or_default();
    let buffers = recording
        .finish()
        .into_iter()
        .filter(|b| !slab.contains(&b.address()))
        .collect();
    Ok(ModelWeights {
        embed_tokens,
        lm_head,
        final_mixer,
        layers,
        mtp,
        vision,
        expert_cache,
        buffers,
        planned_memory: memory_budget.or_else(physical_memory),
        session_reserve,
    })
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4exp/weights.rs"]
mod tests;
