//! Measures expert usage: prefills consecutive real-text prompts and counts,
//! per MoE layer, how often each expert is routed to. The skew decides
//! whether cold experts can be offloaded on smaller machines.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use lily::engine::{LanguageModel, LoadOptions};
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;
use lily::tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "lily-experts", about = "Expert usage over real text")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    /// Text file, tokenized with the checkpoint's tokenizer and cut into
    /// consecutive prompts of `--prompt-len` tokens.
    #[arg(long)]
    text: PathBuf,
    #[arg(long, default_value_t = 8192)]
    prompt_len: usize,
    /// How many prompts to run (all that fit by default).
    #[arg(long)]
    prompts: Option<usize>,
    /// Where to write the per-layer counts (JSON: `counts[layer][expert]`).
    #[arg(long)]
    json_out: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let text = std::fs::read_to_string(&cli.text)?;
    let started = Instant::now();
    let ids = Tokenizer::from_model_dir(&cli.model)?.encode(&text)?;
    let n_prompts = (ids.len() / cli.prompt_len).min(cli.prompts.unwrap_or(usize::MAX));
    ensure!(n_prompts > 0, "{} tokens: fewer than one prompt", ids.len());
    eprintln!(
        "{} tokens in {:.1}s: {n_prompts} prompts of {}",
        ids.len(),
        started.elapsed().as_secs_f64(),
        cli.prompt_len
    );

    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        &cli.model,
        &LoadOptions::default(),
    )?;
    let (layers, experts, top_k) = model.moe_shape();
    let capacity = cli.prompt_len + 1;
    let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
    model.enable_expert_log(&ctx, &mut scratch, capacity)?;
    let mut counts = vec![vec![0u64; experts]; layers];
    let started = Instant::now();
    for p in 0..n_prompts {
        let prompt = &ids[p * cli.prompt_len..(p + 1) * cli.prompt_len];
        let mut state = model.new_state(&ctx, capacity)?;
        LanguageModel::prefill(&model, &ctx, &mut state, &mut scratch, prompt, None)?;
        let log = model.expert_log(&scratch).expect("enabled").to_u32()?;
        for (layer, row) in counts.iter_mut().enumerate() {
            let base = layer * capacity * top_k;
            for &e in &log[base..base + cli.prompt_len * top_k] {
                row[e as usize] += 1;
            }
        }
        if (p + 1) % 10 == 0 || p + 1 == n_prompts {
            eprintln!(
                "{}/{n_prompts} prompts, {:.0} tok/s",
                p + 1,
                ((p + 1) * cli.prompt_len) as f64 / started.elapsed().as_secs_f64()
            );
        }
    }

    // Skew summary: per layer, the share of routed slots the busiest
    // quarter, half and three quarters of the experts account for, and
    // globally the share of (layer, expert) slices needed for 90/95/99% of
    // all routing.
    let total: u64 = counts.iter().flatten().sum();
    let mut all: Vec<u64> = counts.iter().flatten().copied().collect();
    all.sort_unstable_by(|a, b| b.cmp(a));
    let mut cum = 0u64;
    let mut needed = [0usize; 3];
    let targets = [0.90, 0.95, 0.99];
    for (i, &c) in all.iter().enumerate() {
        cum += c;
        for (t, &target) in targets.iter().enumerate() {
            if needed[t] == 0 && cum as f64 >= target * total as f64 {
                needed[t] = i + 1;
            }
        }
    }
    let n_slices = layers * experts;
    println!(
        "{n_prompts} prompts x {} tokens, {layers} layers x {experts} experts, top-{top_k}",
        cli.prompt_len
    );
    println!(
        "slices for 90 / 95 / 99% of routing: {} / {} / {} of {n_slices} ({:.0}% / {:.0}% / {:.0}%)",
        needed[0],
        needed[1],
        needed[2],
        100.0 * needed[0] as f64 / n_slices as f64,
        100.0 * needed[1] as f64 / n_slices as f64,
        100.0 * needed[2] as f64 / n_slices as f64
    );
    let never: usize = all.iter().filter(|&&c| c == 0).count();
    println!("experts never routed to: {never} of {n_slices}");
    println!("layer | top 25% experts | top 50% | top 75% | max share | min count");
    for (layer, row) in counts.iter().enumerate() {
        let mut sorted = row.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let sum: u64 = sorted.iter().sum();
        let share = |frac: f64| -> f64 {
            let n = (experts as f64 * frac) as usize;
            sorted[..n].iter().sum::<u64>() as f64 / sum as f64
        };
        println!(
            "{layer:5} | {:14.1}% | {:6.1}% | {:6.1}% | {:8.2}% | {}",
            100.0 * share(0.25),
            100.0 * share(0.5),
            100.0 * share(0.75),
            100.0 * sorted[0] as f64 / sum as f64,
            sorted[experts - 1]
        );
    }
    let record = serde_json::json!({
        "model": cli.model.display().to_string(),
        "text": cli.text.display().to_string(),
        "prompt_len": cli.prompt_len,
        "prompts": n_prompts,
        "layers": layers, "experts": experts, "top_k": top_k,
        "counts": counts,
    });
    std::fs::write(&cli.json_out, serde_json::to_vec(&record)?)?;
    Ok(())
}
