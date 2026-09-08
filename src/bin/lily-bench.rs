use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::generate::speculate;
use lily::kernels::sample::SamplingParams;
use lily::metal::PendingPass;
use lily::kernels::attention::MAX_SEQ;
use lily::metal::MetalContext;
use lily::model::Qwen3_5Model;
use lily::qwen4exp::Qwen4ExpModel;
use lily::serve::checkpoint_model_type;

#[derive(Parser)]
#[command(name = "lily-bench", about = "In-process production generation benchmark")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_len: usize,
    #[arg(long, default_value_t = 64)]
    decode_steps: usize,
    #[arg(long, default_value_t = false)]
    gpu_timing: bool,
    /// Stream the paged n-gram table through the page cache before measuring.
    #[arg(long, default_value_t = false)]
    ngram_preload: bool,
    /// Draft tokens per step: measures speculative decoding through the
    /// checkpoint's draft head instead of the pipelined one-token loop.
    #[arg(long, default_value_t = 0)]
    drafts: usize,
    #[arg(long)]
    json_out: PathBuf,
}

fn fnv1a(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf29ce484222325u64, |hash, token| {
        (hash ^ u64::from(*token)).wrapping_mul(0x100000001b3)
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match checkpoint_model_type(&cli.model)?.as_str() {
        "qwen3_5_moe" => bench::<Qwen3_5Model>(&cli),
        "qwen4_exp" => bench::<Qwen4ExpModel>(&cli),
        other => anyhow::bail!("unsupported model_type {other:?}"),
    }
}

const GREEDY: SamplingParams = SamplingParams::greedy();

fn draw(step: usize) -> Draw<'static> {
    Draw { params: &GREEDY, step }
}

/// Encodes, prepares and commits one step (no lookahead): the warm-up cadence.
fn submit_step<'a, M: LanguageModel>(
    model: &M,
    ctx: &'a MetalContext,
    state: &mut M::State,
    scratch: &M::Scratch,
    slot_in: usize,
    slot_out: usize,
    step: usize,
) -> Result<PendingPass<'a>> {
    let token = scratch.next_token().view(slot_in, &[1])?.to_u32()?[0];
    let encoded = model.encode_decode_step(ctx, state, scratch, slot_in, slot_out, draw(step))?;
    model.prepare_step_inputs(state, scratch, token)?;
    let pending = encoded.commit()?;
    state.advance(1);
    Ok(pending)
}

fn bench<M: LanguageModel>(cli: &Cli) -> Result<()> {
    ensure!(cli.prompt_len > 0, "--prompt-len must be positive");
    ensure!(cli.decode_steps > 0, "--decode-steps must be positive");
    let max_seq = cli
        .prompt_len
        .checked_add(cli.decode_steps)
        .and_then(|tokens| tokens.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("benchmark token budget overflow"))?;
    ensure!(max_seq <= MAX_SEQ, "benchmark exceeds model window {MAX_SEQ}");

    let ctx = MetalContext::new()?;
    let model = M::load(&ctx, &cli.model, &LoadOptions { mtp_drafts: cli.drafts, ..LoadOptions::default() })?;
    if cli.drafts > 0 {
        return bench_speculative(cli, &ctx, &model);
    }
    if cli.ngram_preload {
        let started = Instant::now();
        let bytes = model.warm_storage(false)?;
        eprintln!("paged weights: {:.1} GB resident after preload in {:.1}s", bytes as f64 / 1e9, started.elapsed().as_secs_f64());
    }
    let vocab = u32::try_from(model.vocab_size())?;
    let token = |index: usize| (index as u32).wrapping_mul(2_654_435_761) % vocab;
    let prompt: Vec<u32> = (0..cli.prompt_len).map(token).collect();
    let mut scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;

    // Warm the exact prompt shape and production cadence: one delivered decode
    // step plus its async lookahead. This grows scratch and compiles every
    // shape-specialized Metal pipeline before the measured state is created.
    {
        let mut warm_state = model.new_state(&ctx, cli.prompt_len + 2)?;
        model.prefill(&ctx, &mut warm_state, &mut scratch, &prompt, Some(draw(0)))?;
        let first = submit_step(&model, &ctx, &mut warm_state, &scratch, 0, 1, 1)?;
        let lookahead = submit_step(&model, &ctx, &mut warm_state, &scratch, 1, 0, 2)?;
        first.wait()?;
        lookahead.wait()?;
    }

    let mut state = model.new_state(&ctx, max_seq)?;
    let prefill_start = Instant::now();
    model.prefill(&ctx, &mut state, &mut scratch, &prompt, Some(draw(0)))?;

    // Production cadence: the next step is encoded while the previous one
    // runs and committed as soon as its input token has been read back.
    // Submit the first decode before token delivery and cadence timing.
    let mut pending = submit_step(&model, &ctx, &mut state, &scratch, 0, 1, 1)?;
    let first_token_id = scratch.next_token().view(0, &[1])?.to_u32()?[0];
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut previous_delivery = decode_start;
    let mut decode_intervals = Vec::with_capacity(cli.decode_steps);
    let mut completed_passes =
        if cli.gpu_timing { Vec::with_capacity(cli.decode_steps) } else { Vec::new() };
    let mut token_ids = Vec::with_capacity(cli.decode_steps);
    let mut prepare_secs = Vec::with_capacity(cli.decode_steps);
    for index in 1..cli.decode_steps {
        let encoded = model.encode_decode_step(
            &ctx,
            &state,
            &scratch,
            index % 2,
            (index + 1) % 2,
            draw(index + 1),
        )?;
        let completed = if cli.gpu_timing {
            Some(pending.wait_retain()?)
        } else {
            pending.wait()?;
            None
        };
        let token = scratch.next_token().view(index % 2, &[1])?.to_u32()?[0];
        let prepare_started = Instant::now();
        model.prepare_step_inputs(&mut state, &scratch, token)?;
        prepare_secs.push(prepare_started.elapsed().as_secs_f64());
        let next = encoded.commit()?;
        state.advance(1);
        token_ids.push(token);
        let delivered = Instant::now();
        decode_intervals
            .push(delivered.duration_since(previous_delivery).as_secs_f64());
        if let Some(completed) = completed {
            completed_passes.push(completed);
        }
        previous_delivery = delivered;
        pending = next;
    }
    // Encode the final lookahead inside the cadence timer, then drain it outside.
    let encoded = model.encode_decode_step(
        &ctx,
        &state,
        &scratch,
        cli.decode_steps % 2,
        (cli.decode_steps + 1) % 2,
        draw(cli.decode_steps + 1),
    )?;
    let completed = if cli.gpu_timing {
        Some(pending.wait_retain()?)
    } else {
        pending.wait()?;
        None
    };
    let token = scratch.next_token().view(cli.decode_steps % 2, &[1])?.to_u32()?[0];
    model.prepare_step_inputs(&mut state, &scratch, token)?;
    let lookahead = encoded.commit()?;
    state.advance(1);
    token_ids.push(token);
    let delivered = Instant::now();
    decode_intervals.push(delivered.duration_since(previous_delivery).as_secs_f64());
    if let Some(completed) = completed {
        completed_passes.push(completed);
    }
    let decode_secs = delivered.duration_since(decode_start).as_secs_f64();
    let lookahead_completed = if cli.gpu_timing {
        Some(lookahead.wait_retain()?)
    } else {
        lookahead.wait()?;
        None
    };

    // Diagnostic timestamp queries are intentionally outside the production
    // cadence timer. The default path retains no completed command buffers.
    let mut gpu_passes = Vec::with_capacity(completed_passes.len());
    for completed in completed_passes {
        let gpu = completed.timing()?;
        gpu_passes.push(serde_json::json!({
            "start_secs": gpu.gpu_start_secs,
            "end_secs": gpu.gpu_end_secs,
            "wall_secs": gpu.gpu_end_secs - gpu.gpu_start_secs,
        }));
    }
    let lookahead_gpu =
        lookahead_completed.map(|completed| completed.timing()).transpose()?;
    let token_digest = fnv1a(&token_ids);

    let report = serde_json::json!({
        "schema_version": 1,
        "meta": {
            "engine": "lily",
            "model_id": M::MODEL_ID,
            "source_id": option_env!("LILY_BENCH_SOURCE_ID").unwrap_or("unknown"),
            "crate_version": env!("CARGO_PKG_VERSION"),
            "harness": "src/bin/lily-bench.rs",
        },
        "workload": {
            "prompt_len": cli.prompt_len,
            "prompt_kind": "u32_golden_ratio_hash_mod_vocab",
            "decode_steps": cli.decode_steps,
            "decode_mode": "production_depth2_concurrent",
            "gpu_timing_diagnostic": cli.gpu_timing,
        },
        "results": {
            "prefill": {
                "wall_secs": prefill_secs,
                "tok_s": cli.prompt_len as f64 / prefill_secs,
                "first_token_id": first_token_id,
            },
            "decode": {
                "wall_secs": decode_secs,
                "tok_s": cli.decode_steps as f64 / decode_secs,
                "step_wall_secs": decode_intervals,
                "gpu_passes": gpu_passes,
                "lookahead_gpu_pass": lookahead_gpu.map(|gpu| serde_json::json!({
                    "start_secs": gpu.gpu_start_secs,
                    "end_secs": gpu.gpu_end_secs,
                    "wall_secs": gpu.gpu_end_secs - gpu.gpu_start_secs,
                })),
                "token_digest": format!("{token_digest:016x}"),
                "token_ids": token_ids,
            },
        },
    });
    std::fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    prepare_secs.sort_by(f64::total_cmp);
    if let Some(median) = prepare_secs.get(prepare_secs.len() / 2) {
        eprintln!("host prepare per step: median {:.3} ms, max {:.3} ms", median * 1e3, prepare_secs.last().copied().unwrap_or(0.0) * 1e3);
    }
    eprintln!(
        "prefill: {} tok in {:.6}s ({:.1} tok/s) | decode: {} steps in {:.6}s ({:.1} tok/s) | digest={token_digest:016x}",
        cli.prompt_len,
        prefill_secs,
        cli.prompt_len as f64 / prefill_secs,
        cli.decode_steps,
        decode_secs,
        cli.decode_steps as f64 / decode_secs,
    );
    Ok(())
}

/// Greedy speculative decoding: prefill, then `decode_steps` tokens through
/// the draft head, reporting tokens per second and the acceptance rate.
fn bench_speculative<M: LanguageModel>(cli: &Cli, ctx: &MetalContext, model: &M) -> Result<()> {
    ensure!(model.max_drafts() > 0, "this checkpoint has no draft head");
    let max_seq = cli.prompt_len + cli.decode_steps + 2 * cli.drafts + 2;
    let vocab = u32::try_from(model.vocab_size())?;
    let token = |index: usize| (index as u32).wrapping_mul(2_654_435_761) % vocab;
    let prompt: Vec<u32> = (0..cli.prompt_len).map(token).collect();
    let mut scratch = model.new_scratch_with_capacity(ctx, max_seq)?;
    let never_stop = |_: u32| false;
    let mut run = |steps: usize| -> Result<(f64, f64, usize, usize, Vec<u32>)> {
        let mut state = model.new_state(ctx, max_seq)?;
        let prefill_start = Instant::now();
        model.prefill(ctx, &mut state, &mut scratch, &prompt, Some(draw(0)))?;
        let first = scratch.next_token().view(0, &[1])?.to_u32()?[0];
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let mut tokens = vec![first];
        let decode_start = Instant::now();
        let outcome = speculate(ctx, model, &mut state, &mut scratch, &GREEDY, cli.drafts, steps + 1, &mut tokens, &never_stop, &mut |_| Ok(true))?;
        let decode_secs = decode_start.elapsed().as_secs_f64();
        Ok((prefill_secs, decode_secs, outcome.drafted, outcome.accepted, tokens))
    };
    // Warm-up compiles the pipelines for every shape the loop uses.
    run(4)?;
    let (prefill_secs, decode_secs, drafted, accepted, tokens) = run(cli.decode_steps)?;
    let generated = tokens.len() - 1;
    let digest = fnv1a(&tokens[1..]);
    let report = serde_json::json!({
        "schema_version": 1,
        "meta": {"engine": "lily", "model_id": M::MODEL_ID, "harness": "src/bin/lily-bench.rs", "crate_version": env!("CARGO_PKG_VERSION")},
        "workload": {"prompt_len": cli.prompt_len, "decode_steps": cli.decode_steps, "decode_mode": format!("speculative_{}_drafts", cli.drafts)},
        "results": {
            "prefill": {"wall_secs": prefill_secs, "tok_s": cli.prompt_len as f64 / prefill_secs},
            "decode": {
                "wall_secs": decode_secs,
                "tokens": generated,
                "tok_s": generated as f64 / decode_secs,
                "drafted": drafted,
                "accepted": accepted,
                "token_digest": format!("{digest:016x}"),
                "token_ids": &tokens[1..],
            },
        },
    });
    std::fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "prefill: {} tok in {:.3}s ({:.1} tok/s) | speculative decode ({} drafts): {} tokens in {:.3}s ({:.1} tok/s), {}/{} drafts accepted ({:.1}%) | digest={digest:016x}",
        cli.prompt_len,
        prefill_secs,
        cli.prompt_len as f64 / prefill_secs,
        cli.drafts,
        generated,
        decode_secs,
        generated as f64 / decode_secs,
        accepted,
        drafted,
        100.0 * accepted as f64 / drafted.max(1) as f64,
    );
    Ok(())
}
