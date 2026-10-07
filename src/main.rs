//! Lily's OpenAI-compatible API server.

use std::path::PathBuf;

use anyhow::{Context as _, Result};
use clap::Parser;
use lily::engine::VisionMode;
use lily::qwen4exp::NgramStorage;
use lily::serve::api::{ThinkingBudgets, ThinkingDefaults};
use lily::serve::{SamplingOverrides, ServeOptions, parse_duration_secs};

/// The thinking budget by reasoning effort unless `--thinking-budget` says
/// otherwise: on the 2026-10-07 agent replay (a 56K-token opencode turn at
/// `low`), 8000 tokens with nudges ended every runaway block; xhigh gets
/// twice that, unmeasured.
const DEFAULT_THINKING_BUDGET: &str = "low=8000,medium=8000,xhigh=16000";

#[derive(Parser)]
#[command(
    name = "lily",
    about = "Qwen3.8-Flash-Next inference server",
    after_help = "Only one lily instance runs at a time: a server, lily-bench or lily-probe started \
                  while another one holds ~/Library/Caches/lily/instance.lock refuses to start \
                  and exits with status 75 (EX_TEMPFAIL). Stop the background service first \
                  with tools/service/lily-service.sh stop."
)]
struct Cli {
    /// Checkpoint directory: a Qwen3.8-Flash-Next conversion from
    /// `tools/convert`.
    #[arg(long)]
    model: PathBuf,

    /// HTTP listen address.
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: String,

    /// Maximum prompt plus completion length per request. With the expert
    /// cache (a machine that cannot hold the checkpoint, or `--memory-gb`)
    /// it also sizes the session reserve the plan keeps free: one full
    /// session with q8 K/V caches, 2.7 GB at 131072 and 4.9 GB at 262144,
    /// taken from the expert slots. 131072 is the sensible choice on 64 GB.
    #[arg(long, default_value_t = 131072)]
    max_seq: usize,

    /// GPU memory for cached sessions (KV caches and recurrent checkpoints),
    /// e.g. `24G`. Default: what the device's recommended working set leaves
    /// after the weights, the paged n-gram table (32 GB of page cache with
    /// `--ngram-table paged`) and 8 GiB of headroom for other applications,
    /// but at least 5 GiB with q8 K/V caches (8 GiB with bf16); with the
    /// expert cache exactly one full session at `--max-seq`, which its plan
    /// kept free. The log states the derivation. With `--kv-cache bf16` a
    /// value below 8 GiB refuses to start.
    #[arg(long)]
    cache_bytes: Option<String>,

    /// Element format of the attention K/V caches: `q8` (the default: int8
    /// with one f16 scale per 32 values, llama.cpp's q8_0, about 40 % less
    /// memory per token of context than bf16 at the same speed and a small
    /// loss of precision) or `bf16` (what the model computes; not with the
    /// expert cache, which runs q8 only). Each format keeps its own disk tier directory,
    /// each with the full `--disk-cache-bytes`. The `LILY_KV_CACHE`
    /// environment variable (for tests and measurement) overrides this flag;
    /// the log's memory line states the format in effect.
    #[arg(long)]
    kv_cache: Option<lily::kernels::attention::KvFormat>,

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

    /// A prefix that two prompts shared for at least this many tokens beyond
    /// where the request could resume (two agent runs with the same preamble
    /// diverge before any checkpoint) is written to the disk
    /// tier as a durable prefix entry, so later runs resume from it instead
    /// of prefilling it again. Needs the disk tier; `0` turns it off.
    #[arg(long, default_value_t = 1024)]
    durable_min_tokens: usize,

    /// Take a recurrent-state checkpoint every this many generated tokens
    /// (at most four per session, thinned evenly for longer answers), so a
    /// next prompt that diverges inside the answer (a client re-sending it
    /// as text, re-tokenized) resumes near the divergence instead of at the
    /// end of the previous prompt. Each costs one snapshot (113 MB on the
    /// full model) in the session budget. Off under the expert cache; `0`
    /// turns it off.
    #[arg(long, default_value_t = 2048)]
    decode_checkpoint_tokens: usize,

    /// Unload the model (weights, caches, the n-gram table) after this long
    /// without a request, e.g. `30m`, `2h`; resident sessions go to the disk
    /// tier first, a preload still running stops, and the next request
    /// reloads the model while it waits (weights and warm-up; the table's
    /// preload runs in the background again). `0` keeps the model loaded.
    #[arg(long, default_value = "0")]
    idle_unload: String,

    /// Lock the model's weights in memory (`mlock`) while requests come, so
    /// macOS does not compress them while an agent runs a tool and the next
    /// request does not wait seconds for them to come back. The first
    /// request after a quiet spell pins them before its prefill (about 3 s
    /// for the full model); `--pin-hold` after the last request, under
    /// memory pressure (warning or worse) and before an idle unload the pin
    /// is released. `auto` pins only when the whole model is resident (no
    /// expert cache) and leaves the plan's reserve and a margin below the
    /// wire limit free; `always` pins whenever it fits below the wire limit,
    /// on a small machine too (the expert cache's slab is never pinned);
    /// `off` never pins. A failed pin is logged and the server serves
    /// unpinned.
    #[arg(long, default_value = "auto")]
    pin_weights: lily::serve::pin::PinMode,

    /// How long the pin is held after the last request finished, e.g. `1m`,
    /// `30s`; `0` holds it until memory pressure or the unload.
    #[arg(long, default_value = "1m")]
    pin_hold: String,

    /// Where the Qwen3.8-Flash-Next n-gram table lives: `paged` reads rows
    /// from the checkpoint files through the page cache (32 GB less GPU
    /// memory), `resident` uploads it whole.
    #[arg(long, default_value = "paged")]
    ngram_table: NgramStorage,

    /// Read the whole paged n-gram table into the page cache after every load
    /// (32 GB, evictable), in a background thread at low CPU and I/O priority
    /// while the server already answers requests; rows a request needs
    /// before the preload reaches them are read on demand. Parts already
    /// resident are skipped.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    ngram_preload: bool,

    /// Pin the preloaded table in memory with mlock so it never goes cold
    /// (32 GB that other applications can no longer reclaim). Each part is
    /// locked right after the background preload has read it; implies the
    /// preload.
    #[arg(long)]
    ngram_lock: bool,

    /// Draft tokens per step for speculative decoding with the checkpoint's
    /// multi-token-prediction head (Qwen3.8-Flash-Next conversions that
    /// include it); 0 turns the head off.
    #[arg(long, default_value_t = 2)]
    mtp_drafts: usize,

    /// Memory the engine may plan for, in GB (default: the machine's
    /// physical memory). Below what the checkpoint needs, the routed experts
    /// are cached in memory and the rest served from the checkpoint files
    /// as they are routed to, and speculative decoding is off; see
    /// docs/low-ram-experts.md. Also how to try that mode on a big machine.
    #[arg(long)]
    memory_gb: Option<f64>,

    /// The vision tower of a Qwen3.8-Flash-Next conversion that carries it:
    /// `auto` loads it (0.9 GB), `off` leaves it on disk.
    #[arg(long, default_value = "auto")]
    vision: VisionMode,

    /// An image larger than this many pixels is scaled down to fit before
    /// it reaches the tower (32 x 32 pixels per prompt token: the default is
    /// 2 048 tokens; a 1920 x 1080 screenshot passes untouched).
    #[arg(long, default_value_t = lily::qwen4exp::image::DEFAULT_MAX_PIXELS)]
    image_max_pixels: usize,

    /// An image smaller than this many pixels is scaled up to reach it.
    #[arg(long, default_value_t = lily::qwen4exp::image::DEFAULT_MIN_PIXELS)]
    image_min_pixels: usize,

    /// Chat prompts open a reasoning block unless the request says otherwise.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    thinking: bool,

    /// Template reasoning effort when thinking is on: low, medium or xhigh
    /// (the template's default is xhigh).
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Thinking budget of chat requests, by the template's reasoning
    /// effort: `low=8000,medium=8000,xhigh=16000` (`high` is `xhigh`), one
    /// number for all (positive counts), or `off`. Once the reasoning block
    /// holds that many tokens it is closed at the next line end, with a
    /// short transition text, and an end of turn drawn inside the block is
    /// replaced by the close. A request's `thinking_budget` overrides it.
    #[arg(long, default_value = DEFAULT_THINKING_BUDGET)]
    thinking_budget: String,

    /// Scales the default budget of a turn whose last message is a tool
    /// result.
    #[arg(long, default_value_t = 1.0)]
    thinking_budget_tool_turn_factor: f64,

    /// Tokens the budget's close (and a nudge) waits for a line end, then
    /// as long again for a sentence end, before it is forced.
    #[arg(long, default_value_t = lily::thinking::DEFAULT_GRACE)]
    thinking_budget_grace: usize,

    /// Insert graded nudges into the reasoning at 50, 75 and 90 % of the
    /// budget by default (request field `thinking_nudges`).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    thinking_nudges: bool,

    /// A `<tool_call>` at a line start inside the reasoning block ends the
    /// block (`</think>` is inserted before it) by default, for chat
    /// requests with tools (request field `tool_call_ends_thinking`).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    tool_call_ends_thinking: bool,

    /// JSON file replacing the texts the thinking controls insert:
    /// `{"close": ["..."], "nudges": [{"at": 0.5, "texts": ["..."]}]}`.
    #[arg(long)]
    thinking_texts: Option<PathBuf>,

    /// Requests waiting for the engine before new ones get 503.
    #[arg(long, default_value_t = 32)]
    queue: usize,

    /// Requests that decode together in one batched step when several
    /// arrive at once (continuous batching, up to 4). A request decoding
    /// alone keeps speculative decoding; requests sharing a step decode one
    /// token each. 1 serves one request at a time. Default 4; the expert
    /// cache serves one request at a time and refuses a value above 1.
    #[arg(long)]
    max_batch: Option<usize>,

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
    let value: f64 =
        digits.parse().with_context(|| format!("invalid byte size {text:?}"))?;
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
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join("Library").join("Caches").join("lily").join("sessions")
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`), 1
/// on any other error.
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
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
        decode_checkpoint_tokens: cli.decode_checkpoint_tokens,
        ngram_storage: cli.ngram_table,
        ngram_preload: cli.ngram_preload,
        ngram_lock: cli.ngram_lock,
        mtp_drafts: cli.mtp_drafts,
        memory_budget: cli.memory_gb.map(|gb| (gb * (1u64 << 30) as f64) as u64),
        vision: cli.vision,
        image_max_pixels: cli.image_max_pixels,
        image_min_pixels: cli.image_min_pixels,
        thinking: cli.thinking,
        reasoning_effort: cli.reasoning_effort,
        thinking_controls: ThinkingDefaults {
            budgets: ThinkingBudgets::parse(&cli.thinking_budget)?,
            tool_turn_factor: {
                let f = cli.thinking_budget_tool_turn_factor;
                anyhow::ensure!(
                    f.is_finite() && f > 0.0,
                    "--thinking-budget-tool-turn-factor must be a positive number"
                );
                f
            },
            nudges: cli.thinking_nudges,
            tool_call_ends_thinking: cli.tool_call_ends_thinking,
            grace: cli.thinking_budget_grace,
        },
        thinking_texts: cli.thinking_texts,
        queue: cli.queue,
        max_batch: cli.max_batch,
        kv_format: cli.kv_cache,
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
        pin_weights: cli.pin_weights,
        pin_hold_secs: parse_duration_secs(&cli.pin_hold)?,
        inject_metal_fault: cli.debug_inject_metal_fault,
    };
    lily::serve::run(&cli.model, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lily::kernels::attention::KvFormat;

    /// `--kv-cache` and `--max-batch` stay unset unless given, so the server
    /// tells an explicit value (which may refuse to start) from the default.
    #[test]
    fn kv_cache_and_max_batch_are_unset_unless_given() {
        let cli = Cli::try_parse_from(["lily", "--model", "m"]).expect("parse");
        assert_eq!((cli.kv_cache, cli.max_batch), (None, None));
        let cli = Cli::try_parse_from([
            "lily",
            "--model",
            "m",
            "--kv-cache",
            "bf16",
            "--max-batch",
            "1",
        ])
        .expect("parse");
        assert_eq!((cli.kv_cache, cli.max_batch), (Some(KvFormat::Bf16), Some(1)));
        assert!(
            Cli::try_parse_from(["lily", "--model", "m", "--kv-cache", "q4"]).is_err()
        );
    }
}
