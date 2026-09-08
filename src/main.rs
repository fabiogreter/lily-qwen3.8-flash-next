//! Lily's OpenAI-compatible API server.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use lily::qwen4exp::NgramStorage;
use lily::serve::{SamplingOverrides, ServeOptions};

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
    /// after the weights, minus a safety margin.
    #[arg(long)]
    cache_bytes: Option<String>,

    /// Most sessions kept in the cache.
    #[arg(long, default_value_t = 16)]
    max_sessions: usize,

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

fn main() -> Result<()> {
    let cli = Cli::parse();
    let options = ServeOptions {
        bind: cli.bind,
        max_seq: cli.max_seq,
        cache_bytes: cli.cache_bytes.as_deref().map(parse_bytes).transpose()?,
        max_sessions: cli.max_sessions,
        ngram_storage: cli.ngram_table,
        ngram_preload: cli.ngram_preload,
        ngram_lock: cli.ngram_lock,
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
    };
    lily::serve::run(&cli.model, options)
}
