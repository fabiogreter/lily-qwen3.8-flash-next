//! Lily's OpenAI-compatible API server.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use lily::qwen4exp::NgramStorage;
use lily::serve::{SamplingOverrides, ServeOptions, parse_duration_secs};

#[derive(Parser)]
#[command(
    name = "lily",
    about = "Qwen3.6-35B-A3B / Qwen3.8-Flash-Next inference server"
)]
struct Cli {
    /// Checkpoint directory: an MLX affine 4-bit Qwen3.6-35B-A3B export, or
    /// a Qwen3.8-Flash-Next conversion from `tools/convert`.
    #[arg(long)]
    model: PathBuf,

    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: String,

    /// Maximum prompt plus completion length per request.
    #[arg(long, default_value_t = 131072)]
    max_seq: usize,

    /// GPU memory for cached sessions (KV caches and recurrent checkpoints),
    /// e.g. `24G`. Default: what the device's recommended working set leaves
    /// after the weights, the paged n-gram table (32 GB of page cache with
    /// `--ngram-table paged`) and 8 GiB of headroom for other applications,
    /// but at least 8 GiB; the log states the derivation.
    #[arg(long)]
    cache_bytes: Option<String>,

    /// Most sessions kept in the cache.
    #[arg(long, default_value_t = 16)]
    max_sessions: usize,

    /// Directory for the disk tier of the session cache (sessions evicted
    /// from GPU memory are written here and read back when a prompt shares
    /// their prefix). Default: ~/Library/Caches/lily/sessions.
    #[arg(long)]
    disk_cache_dir: Option<PathBuf>,

    /// Most bytes the disk tier may hold, e.g. `100G`; `0` disables it.
    /// Least recently used sessions go first.
    #[arg(long, default_value = "100G")]
    disk_cache_bytes: String,

    /// How long an evicted session may sit unused on disk before it is
    /// deleted, e.g. `3d`, `12h`, `90m`; `0` keeps entries until the budget
    /// evicts them.
    #[arg(long, default_value = "3d")]
    disk_cache_ttl: String,

    /// A prefix at least this many tokens long that two prompts shared but
    /// that no cached session could resume from (two agent runs with the
    /// same preamble diverge before any checkpoint) is written to the disk
    /// tier as a durable prefix entry, so later runs resume from it instead
    /// of prefilling it again. Needs the disk tier; `0` turns it off.
    #[arg(long, default_value_t = 1024)]
    durable_min_tokens: usize,

    /// Unload the model (weights, caches, the n-gram table) after this long
    /// without a request, e.g. `30m`, `2h`; resident sessions go to the disk
    /// tier first and the next request reloads it while it waits. `0` keeps
    /// the model loaded.
    #[arg(long, default_value = "0")]
    idle_unload: String,

    /// Where the Qwen3.8-Flash-Next n-gram table lives: `paged` reads rows
    /// from the checkpoint files through the page cache (32 GB less GPU
    /// memory), `resident` uploads it whole.
    #[arg(long, default_value = "paged")]
    ngram_table: NgramStorage,

    /// Read the whole paged n-gram table at startup so the first requests do
    /// not pay cold reads (32 GB of evictable page cache).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    ngram_preload: bool,

    /// Pin the preloaded table in memory with mlock so it never goes cold
    /// (32 GB that other applications can no longer reclaim).
    #[arg(long)]
    ngram_lock: bool,

    /// Draft tokens per step for speculative decoding with the checkpoint's
    /// multi-token-prediction head (Qwen3.8-Flash-Next conversions that
    /// include it); 0 turns the head off.
    #[arg(long, default_value_t = 2)]
    mtp_drafts: usize,

    /// Chat prompts open a reasoning block unless the request says otherwise.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    thinking: bool,

    /// Template reasoning effort when thinking is on: low, medium or xhigh
    /// (the template's default is xhigh).
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Requests waiting for the engine before new ones get 503.
    #[arg(long, default_value_t = 32)]
    queue: usize,

    /// Default sampling overrides (the checkpoint's generation_config.json
    /// supplies the rest).
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    min_p: Option<f32>,
    #[arg(long)]
    presence_penalty: Option<f32>,
    #[arg(long)]
    frequency_penalty: Option<f32>,
    #[arg(long)]
    repetition_penalty: Option<f32>,

    /// Testing only: record a Metal fault on the N-th request the engine
    /// serves (counted from 1 across reloads) so the fault recovery path can
    /// be exercised end to end.
    #[arg(long, hide = true)]
    debug_inject_metal_fault: Option<u64>,
}

fn parse_bytes(text: &str) -> Result<usize> {
    let text = text.trim();
    let (digits, unit) = text
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .map_or((text, ""), |i| text.split_at(i));
    let value: f64 = digits.parse().with_context(|| format!("invalid byte size {text:?}"))?;
    let scale = match unit.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1.0,
        "K" | "KB" | "KIB" => 1024.0,
        "M" | "MB" | "MIB" => 1024.0 * 1024.0,
        "G" | "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" | "TIB" => 1024.0f64.powi(4),
        other => anyhow::bail!("unknown byte unit {other:?}"),
    };
    Ok((value * scale) as usize)
}

fn default_disk_cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("Library").join("Caches").join("lily").join("sessions")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let options = ServeOptions {
        bind: cli.bind,
        max_seq: cli.max_seq,
        cache_bytes: cli.cache_bytes.as_deref().map(parse_bytes).transpose()?,
        max_sessions: cli.max_sessions,
        disk_cache_dir: Some(cli.disk_cache_dir.unwrap_or_else(default_disk_cache_dir)),
        disk_cache_bytes: parse_bytes(&cli.disk_cache_bytes)? as u64,
        disk_cache_ttl_secs: parse_duration_secs(&cli.disk_cache_ttl)?,
        durable_min_tokens: cli.durable_min_tokens,
        ngram_storage: cli.ngram_table,
        ngram_preload: cli.ngram_preload,
        ngram_lock: cli.ngram_lock,
        mtp_drafts: cli.mtp_drafts,
        thinking: cli.thinking,
        reasoning_effort: cli.reasoning_effort,
        queue: cli.queue,
        sampling: SamplingOverrides {
            temperature: cli.temperature,
            top_k: cli.top_k,
            top_p: cli.top_p,
            min_p: cli.min_p,
            presence_penalty: cli.presence_penalty,
            frequency_penalty: cli.frequency_penalty,
            repetition_penalty: cli.repetition_penalty,
        },
        idle_unload_secs: parse_duration_secs(&cli.idle_unload)?,
        inject_metal_fault: cli.debug_inject_metal_fault,
    };
    lily::serve::run(&cli.model, options)
}
