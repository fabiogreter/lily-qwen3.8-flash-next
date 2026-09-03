//! Lily's minimal OpenAI-compatible API server.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

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

    /// Maximum prompt plus completion length.
    #[arg(long, default_value_t = 4096)]
    max_seq: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    lily::serve::run(&cli.model, &cli.bind, cli.max_seq)
}
