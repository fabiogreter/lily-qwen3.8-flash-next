//! Runs one prompt through a checkpoint step by step and records the top
//! logits of every step, for comparison against a reference implementation
//! (`tools/reference/hf_reference.py`) and for quick manual inspection.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::chat::Message;
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::kernels::sample::SamplingParams;
use lily::generate::{Generator, Thinking};
use lily::metal::MetalContext;
use lily::model::Qwen3_5Model;
use lily::qwen4exp::Qwen4ExpModel;
use lily::serve::checkpoint_model_type;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "lily-probe", about = "Step-by-step greedy probe with top logits")]
struct Cli {
    /// Checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// User message, rendered through the checkpoint's chat template with
    /// thinking disabled (what the API server does).
    #[arg(long, conflicts_with = "tokens")]
    prompt: Option<String>,
    /// JSON file holding a list of prompt token ids (bypasses the template).
    #[arg(long)]
    tokens: Option<PathBuf>,
    /// Greedy steps after the prompt.
    #[arg(long, default_value_t = 8)]
    max_tokens: usize,
    /// Logit entries recorded per step.
    #[arg(long, default_value_t = 8)]
    top: usize,
    /// Where to write the JSON record.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Serialize)]
struct Step {
    /// Prompt position the logits belong to (`prompt_len - 1 + step`).
    position: usize,
    chosen: u32,
    ids: Vec<u32>,
    logits: Vec<f32>,
    debug: serde_json::Value,
}

#[derive(Serialize)]
struct Record {
    model_id: &'static str,
    prompt_token_ids: Vec<u32>,
    steps: Vec<Step>,
    text: String,
    load_seconds: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
}

fn top_k(logits: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).expect("finite").then(a.cmp(&b))
    });
    let ids: Vec<u32> = order[..k.min(order.len())].iter().map(|&i| i as u32).collect();
    let values = ids.iter().map(|&i| logits[i as usize]).collect();
    (ids, values)
}

fn probe<M: LanguageModel>(cli: &Cli) -> Result<Record> {
    let ctx = MetalContext::new()?;
    let started = Instant::now();
    let model = M::load(&ctx, &cli.model, &LoadOptions::default())?;
    let load_seconds = started.elapsed().as_secs_f64();
    let mut generator = Generator::from_model_dir(&cli.model)?;
    generator.add_stop_tokens(&model.eos_token_ids());

    let prompt: Vec<u32> = match (&cli.prompt, &cli.tokens) {
        (Some(text), None) => generator
            .encode_chat(&vec![Message::new_user(text.clone())], Thinking::Disabled)?,
        (None, Some(path)) => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_slice(&bytes).context("parsing token list")?
        }
        _ => anyhow::bail!("pass exactly one of --prompt or --tokens"),
    };
    ensure!(!prompt.is_empty(), "empty prompt");
    let max_seq = prompt.len() + cli.max_tokens + 1;
    let mut state = model.new_state(&ctx, max_seq)?;
    let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;

    let started = Instant::now();
    let greedy = SamplingParams::greedy();
    model.prefill(&ctx, &mut state, &mut scratch, &prompt, Some(Draw { params: &greedy, step: 0 }))?;
    let prefill_seconds = started.elapsed().as_secs_f64();

    let read_step = |scratch: &M::Scratch,
                     slot: usize,
                     position: usize|
     -> Result<Step> {
        let chosen = scratch.next_token().view(slot, &[1])?.to_u32()?[0];
        let logits = scratch.logits().to_f32()?;
        let (ids, values) = top_k(&logits, cli.top);
        Ok(Step { position, chosen, ids, logits: values, debug: scratch.debug_json()? })
    };

    let mut steps = vec![read_step(&scratch, 0, prompt.len() - 1)?];
    let started = Instant::now();
    let mut slot = 0usize;
    for step in 1..=cli.max_tokens {
        // Synchronous steps: each one's logits are read before the next runs.
        let input = steps.last().map(|s: &Step| s.chosen).expect("prefill step");
        let encoded = model.encode_decode_step(
            &ctx,
            &state,
            &scratch,
            slot,
            1 - slot,
            Draw { params: &greedy, step },
        )?;
        model.prepare_step_inputs(&mut state, &scratch, input)?;
        let pending = encoded.commit()?;
        state.advance(1);
        pending.wait()?;
        slot = 1 - slot;
        steps.push(read_step(&scratch, slot, prompt.len() - 1 + step)?);
    }
    let decode_seconds = started.elapsed().as_secs_f64();
    let tokens: Vec<u32> = steps.iter().map(|s| s.chosen).collect();
    let text = generator.decode_text(&tokens)?;
    Ok(Record {
        model_id: M::MODEL_ID,
        prompt_token_ids: prompt,
        steps,
        text,
        load_seconds,
        prefill_seconds,
        decode_seconds,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let record = match checkpoint_model_type(&cli.model)?.as_str() {
        "qwen3_5_moe" => probe::<Qwen3_5Model>(&cli)?,
        "qwen4_exp" => probe::<Qwen4ExpModel>(&cli)?,
        other => anyhow::bail!("unsupported model_type {other:?}"),
    };
    eprintln!(
        "{}: {} prompt tokens, load {:.1}s, prefill {:.3}s, {} decode steps in {:.3}s",
        record.model_id,
        record.prompt_token_ids.len(),
        record.load_seconds,
        record.prefill_seconds,
        record.steps.len() - 1,
        record.decode_seconds
    );
    eprintln!(
        "chosen: {:?}",
        record.steps.iter().map(|s| s.chosen).collect::<Vec<_>>()
    );
    eprintln!("text: {:?}", record.text);
    if let Some(out) = &cli.out {
        std::fs::write(out, serde_json::to_vec_pretty(&record)?)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}
