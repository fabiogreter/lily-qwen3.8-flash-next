//! Renders recorded chat conversations as token traces, the input of
//! `lily-trace-routing` and `lily-cache-replay`; tokenizer only, no GPU.
//!
//! Each input is a JSON file holding an OpenAI-style chat request: its
//! `messages` (system, user, assistant, tool; assistant messages may carry
//! `reasoning_content` and `tool_calls`) and optional `tools`. The whole
//! conversation is rendered once through the server's template path
//! (`lily::serve::transcript`) and written to `--out-dir` under the input's
//! stem as `<stem>.u32` (little-endian `u32` ids) and `<stem>.json`:
//! `{"tokens": N, "requests": [[p0, p1, p2], ...]}`, one request per
//! assistant turn, prefilling `p0..p1` and generating `p1..p2`.
//!
//! Only counts are printed: conversations can carry anything.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::serve::api::ChatRequest;
use lily::serve::transcript::{Transcript, render_transcript};
use lily::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(
    name = "lily-render-chat",
    about = "Recorded chat conversations as token traces"
)]
struct Cli {
    /// Checkpoint directory with tokenizer.json and the chat template.
    #[arg(long)]
    model: PathBuf,
    /// Where `<stem>.u32` and `<stem>.json` go for each input.
    #[arg(long)]
    out_dir: PathBuf,
    /// Render with thinking off (the server's `enable_thinking: false`).
    #[arg(long)]
    no_thinking: bool,
    /// Template reasoning effort with thinking on (the template's default
    /// is xhigh).
    #[arg(long, value_parser = ["low", "medium", "xhigh"])]
    reasoning_effort: Option<String>,
    /// Chat request JSON files, one conversation each.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let tokenizer = Tokenizer::from_model_dir(&cli.model)?;
    std::fs::create_dir_all(&cli.out_dir)
        .with_context(|| format!("creating {}", cli.out_dir.display()))?;
    // Refuse up front what would overwrite an input (the recorded chat may be
    // the only copy) or another input's trace.
    let out_dir = std::fs::canonicalize(&cli.out_dir)
        .with_context(|| format!("resolving {}", cli.out_dir.display()))?;
    let mut stems = std::collections::HashSet::new();
    for input in &cli.inputs {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("{} has no usable file name", input.display()))?;
        ensure!(stems.insert(stem.to_string()), "two inputs share the stem {stem}");
        let input = std::fs::canonicalize(input)
            .with_context(|| format!("resolving {}", input.display()))?;
        for ext in ["u32", "json"] {
            ensure!(
                out_dir.join(format!("{stem}.{ext}")) != input,
                "{}: the trace would overwrite this input",
                input.display()
            );
        }
    }
    let mut stdout = std::io::stdout().lock();
    for input in &cli.inputs {
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .with_context(|| format!("{} has no usable file name", input.display()))?;
        let request: ChatRequest = serde_json::from_slice(
            &std::fs::read(input)
                .with_context(|| format!("reading {}", input.display()))?,
        )
        .with_context(|| format!("parsing {}", input.display()))?;
        let transcript = render_transcript(
            request,
            &tokenizer,
            !cli.no_thinking,
            cli.reasoning_effort.as_deref(),
        )
        .with_context(|| format!("rendering {}", input.display()))?;
        write_trace(&cli.out_dir, stem, &transcript)?;
        writeln!(
            stdout,
            "{stem}\ttokens {}\trequests {}\tgenerated {}",
            transcript.tokens.len(),
            transcript.requests.len(),
            transcript.generated()
        )?;
    }
    Ok(())
}

fn write_trace(dir: &Path, stem: &str, transcript: &Transcript) -> Result<()> {
    let n = transcript.tokens.len();
    ensure!(
        transcript.requests.iter().all(|r| r[0] <= r[1] && r[1] <= r[2] && r[2] <= n),
        "{stem}: a request lies outside the sequence"
    );
    let bytes: Vec<u8> =
        transcript.tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
    let path = dir.join(format!("{stem}.u32"));
    std::fs::write(&path, bytes)
        .with_context(|| format!("writing {}", path.display()))?;
    let meta = serde_json::json!({"tokens": n, "requests": transcript.requests});
    let path = dir.join(format!("{stem}.json"));
    std::fs::write(&path, serde_json::to_vec(&meta)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
