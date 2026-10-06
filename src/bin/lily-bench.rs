use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use lily::engine::{
    BatchRow, DecodeStateApi, Draw, LanguageModel, LoadOptions, RowsStepPhases,
    RowsStepTiming, ScratchApi,
};
use lily::generate::speculate;
use lily::kernels::attention::MAX_SEQ;
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::metal::profile::{self, PassProfile};
use lily::metal::{EncodedPass, Pacer, PendingPass, host_secs};
use lily::qwen4exp::Qwen4ExpModel;
use lily::serve::checkpoint_model_type;

#[derive(Parser)]
#[command(name = "lily-bench", about = "In-process production generation benchmark")]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt_len: usize,
    /// A text file: its first `--prompt-len` tokens under the checkpoint's
    /// tokenizer are the prompt, instead of the synthetic token sequence.
    /// Real text is what draft acceptance and expert routing depend on.
    #[arg(long)]
    prompt_text: Option<PathBuf>,
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
    /// Per-kernel GPU profile: runs the transport in its profile mode (one
    /// command buffer per dispatch) and prints, per pass label, GPU ms per
    /// pass by kernel. Wall-clock results under this flag are not comparable
    /// to a normal run; the per-kernel times are the point.
    #[arg(long, default_value_t = false)]
    kernel_profile: bool,
    /// Sample every draw with the server's defaults for this checkpoint
    /// (temperature 1.0, top-k 20, top-p 0.95) instead of drawing greedily;
    /// with `--drafts`, this measures draft acceptance under sampling.
    #[arg(long, default_value_t = false)]
    sample: bool,
    /// Sampler seed under `--sample`.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Memory the engine may plan for, in GB (default: the machine's): below
    /// what the checkpoint needs the experts are cached (docs/low-ram-experts.md).
    #[arg(long)]
    memory_gb: Option<f64>,
    /// Where the expert cache writes the usage it measured (the next load
    /// prefers that file over the shipped ranking); nothing by default.
    #[arg(long)]
    expert_usage_out: Option<PathBuf>,
    /// Measures the server's batched decode step instead
    /// (`LanguageModel::decode_rows`): this many sessions, each with its own
    /// prompt of `--prompt-len` tokens (the synthetic sequence at a different
    /// offset per row, or consecutive slices of `--prompt-text`), decode
    /// `--decode-steps` steps together. With `--gpu-timing`, every step
    /// reports where its wall time went. `--drafts` then only loads the draft
    /// head, which the server's batched step catches up on every row.
    #[arg(long)]
    batch_rows: Option<usize>,
    /// With `--batch-rows`: wait for every batched step before committing
    /// the next one, instead of committing it parked behind the current one
    /// as the server's scheduler does (`LanguageModel::park_rows`). The A/B
    /// of parking on one binary.
    #[arg(long, default_value_t = false)]
    no_park: bool,
    #[arg(long)]
    json_out: PathBuf,
}

/// The prompt: the synthetic sequence (token `i` is `i * 2654435761 mod
/// vocab`) or the first `prompt_len` tokens of `--prompt-text`.
fn prompt_tokens(cli: &Cli, vocab: u32) -> Result<Vec<u32>> {
    prompt_tokens_of(cli, vocab, cli.prompt_len)
}

/// [`prompt_tokens`] for `len` tokens.
fn prompt_tokens_of(cli: &Cli, vocab: u32, len: usize) -> Result<Vec<u32>> {
    let Some(path) = &cli.prompt_text else {
        let token = |index: usize| (index as u32).wrapping_mul(2_654_435_761) % vocab;
        return Ok((0..len).map(token).collect());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let tokenizer = lily::tokenizer::Tokenizer::from_model_dir(&cli.model)?;
    // Tokenize a prefix long enough for the prompt (about four bytes per
    // token in code and prose), growing it if the estimate falls short.
    let mut bytes = (len * 6).min(text.len());
    loop {
        while bytes < text.len() && !text.is_char_boundary(bytes) {
            bytes += 1;
        }
        let mut ids = tokenizer.encode(&text[..bytes])?;
        if ids.len() >= len {
            ids.truncate(len);
            return Ok(ids);
        }
        ensure!(
            bytes < text.len(),
            "{} holds {} tokens, fewer than the {len} needed",
            path.display(),
            ids.len(),
        );
        bytes = (bytes * 2).min(text.len());
    }
}

fn fnv1a(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf29ce484222325u64, |hash, token| {
        (hash ^ u64::from(*token)).wrapping_mul(0x100000001b3)
    })
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`).
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let _activity = lily::activity::Activity::begin("lily-bench");
    let params =
        if cli.sample { server_sampling(cli.seed) } else { SamplingParams::greedy() };
    SAMPLER.set(params).expect("sampler set once");
    match checkpoint_model_type(&cli.model)?.as_str() {
        "qwen4_exp" => bench::<Qwen4ExpModel>(&cli),
        other => anyhow::bail!("unsupported model_type {other:?}"),
    }
}

/// The sampler every draw in this process uses, set once from the CLI.
static SAMPLER: OnceLock<SamplingParams> = OnceLock::new();

fn sampler() -> &'static SamplingParams {
    SAMPLER.get().expect("sampler set before the model loads")
}

/// The server's defaults for the Qwen checkpoints (`generation_config.json`).
const fn server_sampling(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        seed,
        ..SamplingParams::greedy()
    }
}

fn draw(step: usize) -> Draw<'static> {
    Draw { params: sampler(), step }
}

/// Aggregates the recorded pass profiles by label and kernel: calls and GPU
/// ms per pass, share of the summed kernel time (sorted by time), then the
/// summed kernel time and the mean pass span per pass. Every label has one
/// pass per step of its kind (decode, verify, draft), so "per pass" reads as
/// "per step". Prints the tables and returns them for the JSON report.
fn kernel_profile_report(passes: &[PassProfile]) -> serde_json::Value {
    let mut labels: Vec<&'static str> = Vec::new();
    for pass in passes {
        if !labels.contains(&pass.label) {
            labels.push(pass.label);
        }
    }
    let mut tables = Vec::with_capacity(labels.len());
    for label in labels {
        let group: Vec<&PassProfile> =
            passes.iter().filter(|pass| pass.label == label).collect();
        let count = group.len() as f64;
        let mut by_kernel: HashMap<&'static str, (usize, f64)> = HashMap::new();
        let (mut dispatches, mut kernel_secs, mut span_secs) = (0usize, 0.0f64, 0.0f64);
        for pass in &group {
            span_secs += pass.span_secs;
            for sample in &pass.kernels {
                dispatches += 1;
                kernel_secs += sample.gpu_secs;
                let entry = by_kernel.entry(sample.name).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += sample.gpu_secs;
            }
        }
        let mut rows: Vec<(&str, usize, f64)> = by_kernel
            .into_iter()
            .map(|(name, (calls, secs))| (name, calls, secs))
            .collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(b.0)));
        let share = |secs: f64| {
            if kernel_secs > 0.0 { 100.0 * secs / kernel_secs } else { 0.0 }
        };
        eprintln!(
            "kernel profile [{label}]: {} passes, {:.1} dispatches/pass",
            group.len(),
            dispatches as f64 / count
        );
        eprintln!("  {:>9}  {:>6}  {:>10}  kernel", "ms/pass", "share", "calls/pass");
        for (name, calls, secs) in &rows {
            eprintln!(
                "  {:>9.3}  {:>5.1}%  {:>10.1}  {name}",
                secs / count * 1e3,
                share(*secs),
                *calls as f64 / count
            );
        }
        eprintln!(
            "  kernels sum {:.3} ms/pass | mean pass span {:.3} ms/pass | between command buffers {:.3} ms/pass",
            kernel_secs / count * 1e3,
            span_secs / count * 1e3,
            (span_secs - kernel_secs) / count * 1e3
        );
        // Per pass (a prefill label has one pass per chunk, in completion
        // order): the kernels' GPU ms, so growth with context depth shows.
        let per_pass: Vec<serde_json::Value> = group
            .iter()
            .map(|pass| {
                let mut ms: HashMap<&'static str, f64> = HashMap::new();
                for sample in &pass.kernels {
                    *ms.entry(sample.name).or_insert(0.0) += sample.gpu_secs * 1e3;
                }
                serde_json::json!(ms)
            })
            .collect();
        if group.len() > 1 {
            let (first, last) = (&per_pass[0], &per_pass[group.len() - 1]);
            let ms = |pass: &serde_json::Value, name: &str| {
                pass.get(name).and_then(serde_json::Value::as_f64).unwrap_or(0.0)
            };
            eprintln!("  first vs last pass, ms (kernels over 1 ms in either):");
            for (name, _, _) in &rows {
                let (a, b) = (ms(first, name), ms(last, name));
                if a.max(b) >= 1.0 {
                    eprintln!("  {a:>9.3}  {b:>9.3}  {:>+8.3}  {name}", b - a);
                }
            }
        }
        tables.push(serde_json::json!({
            "label": label,
            "passes": group.len(),
            "per_pass_ms": per_pass,
            "dispatches_per_pass": dispatches as f64 / count,
            "kernel_sum_ms_per_pass": kernel_secs / count * 1e3,
            "pass_span_ms_per_pass": span_secs / count * 1e3,
            "kernels": rows.iter().map(|(name, calls, secs)| serde_json::json!({
                "kernel": name,
                "calls_per_pass": *calls as f64 / count,
                "ms_per_pass": secs / count * 1e3,
                "share_pct": share(*secs),
            })).collect::<Vec<_>>(),
        }));
    }
    serde_json::Value::Array(tables)
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
    let encoded =
        model.encode_decode_step(ctx, state, scratch, slot_in, slot_out, draw(step))?;
    model.prepare_step_inputs(state, scratch, token)?;
    let pending = encoded.commit()?;
    state.advance(1);
    Ok(pending)
}

/// Stages step `index` (reading slot `index % 2`): committed and parked, or
/// merely encoded, depending on the protocol.
#[allow(clippy::type_complexity)]
fn stage_next<'a, M: LanguageModel>(
    model: &M,
    ctx: &'a MetalContext,
    state: &mut M::State,
    scratch: &M::Scratch,
    index: usize,
    parking: bool,
) -> Result<(Option<PendingPass<'a>>, Option<EncodedPass<'a>>)> {
    let draw = draw(index + 1);
    if parking {
        let pass = model
            .encode_parked_step(ctx, state, scratch, index % 2, (index + 1) % 2, draw)?
            .commit()?;
        state.advance(1);
        Ok((Some(pass), None))
    } else {
        Ok((
            None,
            Some(model.encode_decode_step(
                ctx,
                state,
                scratch,
                index % 2,
                (index + 1) % 2,
                draw,
            )?),
        ))
    }
}

/// Reads the token of the completed step `index`, stages its inputs and lets
/// the next step go: releases the parked pass or commits the encoded one.
fn deliver<'a, M: LanguageModel>(
    model: &M,
    state: &mut M::State,
    scratch: &M::Scratch,
    index: usize,
    parked: Option<PendingPass<'a>>,
    encoded: Option<EncodedPass<'a>>,
    prepare_secs: &mut Vec<f64>,
) -> Result<(u32, PendingPass<'a>)> {
    let token = scratch.next_token().view(index % 2, &[1])?.to_u32()?[0];
    let prepare_started = Instant::now();
    model.prepare_step_inputs(state, scratch, token)?;
    prepare_secs.push(prepare_started.elapsed().as_secs_f64());
    let next = match (parked, encoded) {
        (Some(pass), _) => {
            model.release_parked(scratch)?;
            pass
        }
        (None, Some(encoded)) => {
            let pass = encoded.commit()?;
            state.advance(1);
            pass
        }
        (None, None) => anyhow::bail!("no next step staged"),
    };
    Ok((token, next))
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

    let ctx = if cli.kernel_profile {
        MetalContext::new_with_profile(true)?
    } else {
        MetalContext::new()?
    };
    let model = M::load(
        &ctx,
        &cli.model,
        &LoadOptions {
            mtp_drafts: cli.drafts,
            memory_budget: cli.memory_gb.map(|gb| (gb * (1u64 << 30) as f64) as u64),
            expert_usage_out: cli.expert_usage_out.clone(),
            ..LoadOptions::default()
        },
    )?;
    if let Some(rows) = cli.batch_rows {
        return bench_batched(cli, &ctx, &model, rows);
    }
    if cli.drafts > 0 {
        return bench_speculative(cli, &ctx, &model);
    }
    if cli.ngram_preload {
        let started = Instant::now();
        let bytes = model.warm_storage(false)?;
        eprintln!(
            "paged weights: {:.1} GB resident after preload in {:.1}s",
            bytes as f64 / 1e9,
            started.elapsed().as_secs_f64()
        );
    }
    let vocab = u32::try_from(model.vocab_size())?;
    let prompt = prompt_tokens(cli, vocab)?;
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
    if ctx.profiling() {
        // The tables cover the measured passes only.
        profile::take();
    }

    let mut state = model.new_state(&ctx, max_seq)?;
    let cache_before = model.expert_cache_stats();
    let prefill_start = Instant::now();
    model.prefill(&ctx, &mut state, &mut scratch, &prompt, Some(draw(0)))?;
    let cache_after_prefill = model.expert_cache_stats();

    // Production cadence: the next step is encoded while the previous one
    // runs. With parking (the server's protocol) it is also committed right
    // away and released once its input token has been read back and staged;
    // otherwise it is committed at that point.
    // Submit the first decode before token delivery and cadence timing.
    let parking = model.supports_parking();
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
    // (woke after the previous pass, released/committed the next one), Metal clock.
    let mut host_marks: Vec<(f64, f64)> = Vec::with_capacity(cli.decode_steps);
    // LILY_PROBE_ARRIVAL: a poller records when each parked pass reaches its
    // wait and when it resumed (the model raises two events around the wait).
    let arrivals: Arc<Mutex<Vec<(u64, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let stop_poller = Arc::new(AtomicBool::new(false));
    let resumes: Arc<Mutex<Vec<(u64, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let poller = scratch
        .arrival_probe()
        .cloned()
        .zip(scratch.resumed_probe().cloned())
        .map(|(arrival, resumed)| {
            let (arrivals, resumes, stop) =
                (Arc::clone(&arrivals), Arc::clone(&resumes), Arc::clone(&stop_poller));
            std::thread::spawn(move || {
                let (mut last_a, mut last_r) =
                    (arrival.signaled_value(), resumed.signaled_value());
                while !stop.load(Ordering::Relaxed) {
                    let a = arrival.signaled_value();
                    if a != last_a {
                        arrivals.lock().expect("arrivals").push((a, host_secs()));
                        last_a = a;
                    }
                    let r = resumed.signaled_value();
                    if r != last_r {
                        resumes.lock().expect("resumes").push((r, host_secs()));
                        last_r = r;
                    }
                    std::hint::spin_loop();
                }
            })
        });
    let mut pacer = Pacer::default();
    pacer.begin(Instant::now());
    for index in 1..cli.decode_steps {
        let levels_before = profile::levels();
        let (parked, encoded) =
            stage_next(&model, &ctx, &mut state, &scratch, index, parking)?;
        if cli.gpu_timing && index == 1 {
            eprintln!("levels per step: {}", profile::levels() - levels_before);
        }
        let completed = if cli.gpu_timing {
            Some(pending.wait_retain_paced(&mut pacer)?)
        } else {
            pending.wait_paced(&mut pacer)?;
            None
        };
        pacer.begin(Instant::now());
        let woke = if cli.gpu_timing { host_secs() } else { 0.0 };
        let (token, next) = deliver(
            &model,
            &mut state,
            &scratch,
            index,
            parked,
            encoded,
            &mut prepare_secs,
        )?;
        if cli.gpu_timing {
            host_marks.push((woke, host_secs()));
        }
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
    // Stage the final lookahead inside the cadence timer, then drain it outside.
    let (parked, encoded) =
        stage_next(&model, &ctx, &mut state, &scratch, cli.decode_steps, parking)?;
    let completed = if cli.gpu_timing {
        Some(pending.wait_retain()?)
    } else {
        pending.wait()?;
        None
    };
    let (token, lookahead) = deliver(
        &model,
        &mut state,
        &scratch,
        cli.decode_steps,
        parked,
        encoded,
        &mut prepare_secs,
    )?;
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
    stop_poller.store(true, Ordering::Relaxed);
    if let Some(poller) = poller {
        let _ = poller.join();
    }
    let arrival_marks: Vec<(u64, f64)> = arrivals.lock().expect("arrivals").clone();
    let resume_marks: Vec<(u64, f64)> = resumes.lock().expect("resumes").clone();

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
    let kernel_profile =
        ctx.profiling().then(|| kernel_profile_report(&profile::take()));
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
            "prompt": cli.prompt_text.as_ref().map_or("synthetic".to_string(), |p| p.display().to_string()),
            "prompt_kind": "u32_golden_ratio_hash_mod_vocab",
            "decode_steps": cli.decode_steps,
            "decode_mode": if parking { "production_depth2_parked" } else { "production_depth2_concurrent" }, "sampling": if cli.sample { "server_defaults" } else { "greedy" }, "seed": cli.seed,
            "gpu_timing_diagnostic": cli.gpu_timing,
            "kernel_profile_diagnostic": ctx.profiling(),
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
                "host_marks": host_marks.iter().map(|(woke, committed)| serde_json::json!({
                    "woke_secs": woke,
                    "committed_secs": committed,
                })).collect::<Vec<_>>(),
                "arrival_marks": arrival_marks.iter().map(|(value, secs)| serde_json::json!({
                    "value": value,
                    "secs": secs,
                })).collect::<Vec<_>>(),
                "resume_marks": resume_marks.iter().map(|(value, secs)| serde_json::json!({
                    "value": value,
                    "secs": secs,
                })).collect::<Vec<_>>(),
                "lookahead_gpu_pass": lookahead_gpu.map(|gpu| serde_json::json!({
                    "start_secs": gpu.gpu_start_secs,
                    "end_secs": gpu.gpu_end_secs,
                    "wall_secs": gpu.gpu_end_secs - gpu.gpu_start_secs,
                })),
                "token_digest": format!("{token_digest:016x}"),
                "token_ids": token_ids,
                "kernel_profile": kernel_profile,
            },
        },
    });
    std::fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    if let (Some(s0), Some(s1), Some(s2)) =
        (cache_before, cache_after_prefill, model.expert_cache_stats())
    {
        eprintln!(
            "expert cache: prefill: {}; decode: {}",
            s1.since(s0).describe(),
            s2.since(s1).describe()
        );
    }
    prepare_secs.sort_by(f64::total_cmp);
    if let Some(median) = prepare_secs.get(prepare_secs.len() / 2) {
        eprintln!(
            "host prepare per step: median {:.3} ms, max {:.3} ms",
            median * 1e3,
            prepare_secs.last().copied().unwrap_or(0.0) * 1e3
        );
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
fn bench_speculative<M: LanguageModel>(
    cli: &Cli,
    ctx: &MetalContext,
    model: &M,
) -> Result<()> {
    ensure!(model.max_drafts() > 0, "this checkpoint has no draft head");
    let max_seq = cli.prompt_len + cli.decode_steps + 2 * cli.drafts + 2;
    let vocab = u32::try_from(model.vocab_size())?;
    let prompt = prompt_tokens(cli, vocab)?;
    let mut scratch = model.new_scratch_with_capacity(ctx, max_seq)?;
    let never_stop = |_: u32| false;
    let mut run = |steps: usize| -> Result<(f64, f64, usize, usize, Vec<u32>)> {
        let mut state = model.new_state(ctx, max_seq)?;
        let cache_before = model.expert_cache_stats();
        let prefill_start = Instant::now();
        model.prefill(ctx, &mut state, &mut scratch, &prompt, Some(draw(0)))?;
        let first = scratch.next_token().view(0, &[1])?.to_u32()?[0];
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        let cache_after_prefill = model.expert_cache_stats();
        let mut tokens = vec![first];
        let decode_start = Instant::now();
        let outcome = speculate(
            ctx,
            model,
            &mut state,
            &mut scratch,
            sampler(),
            cli.drafts,
            steps + 1,
            &mut tokens,
            &never_stop,
            &mut |_| Ok(true),
        )?;
        let decode_secs = decode_start.elapsed().as_secs_f64();
        if let (Some(s0), Some(s1), Some(s2)) =
            (cache_before, cache_after_prefill, model.expert_cache_stats())
        {
            eprintln!(
                "expert cache: prefill: {}; decode: {}",
                s1.since(s0).describe(),
                s2.since(s1).describe()
            );
        }
        Ok((prefill_secs, decode_secs, outcome.drafted, outcome.accepted, tokens))
    };
    // Warm-up compiles the pipelines for every shape the loop uses.
    run(4)?;
    if ctx.profiling() {
        // The tables cover the measured run only.
        profile::take();
    }
    let (prefill_secs, decode_secs, drafted, accepted, tokens) = run(cli.decode_steps)?;
    let kernel_profile =
        ctx.profiling().then(|| kernel_profile_report(&profile::take()));
    let generated = tokens.len() - 1;
    let digest = fnv1a(&tokens[1..]);
    let report = serde_json::json!({
        "schema_version": 1,
        "meta": {"engine": "lily", "model_id": M::MODEL_ID, "harness": "src/bin/lily-bench.rs", "crate_version": env!("CARGO_PKG_VERSION")},
        "workload": {"prompt_len": cli.prompt_len, "prompt": cli.prompt_text.as_ref().map_or("synthetic".to_string(), |p| p.display().to_string()), "decode_steps": cli.decode_steps, "decode_mode": format!("speculative_{}_drafts", cli.drafts), "sampling": if cli.sample { "server_defaults" } else { "greedy" }, "seed": cli.seed, "kernel_profile_diagnostic": ctx.profiling()},
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
                "kernel_profile": kernel_profile,
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

/// The median of `values` (0 for none).
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// The rows of a batched bench step, row `r` in slot `r`, each feeding its
/// last draw; with `ahead` 1, of the step parked behind the one whose draws
/// are not taken yet (the token is then not read).
fn bench_rows<'r, S>(
    states: &'r mut [S],
    tokens: &[Vec<u32>],
    ahead: usize,
) -> Vec<BatchRow<'r, S>> {
    states
        .iter_mut()
        .zip(tokens)
        .enumerate()
        .map(|(slot, (state, drawn))| BatchRow {
            state,
            token: *drawn.last().expect("a row has drawn"),
            draw: draw(drawn.len() + ahead),
            slot,
        })
        .collect()
}

/// The server's batched decode step over `rows` sessions
/// (`--batch-rows`): prefill each, then `--decode-steps` steps of all rows
/// together, greedy or `--sample`d, each step after the first committed
/// parked behind the one before it as the server's scheduler does within a
/// stretch, or with `--no-park` each step waited for before the next one is
/// staged. Reports aggregate and per-row tokens per second, the time from
/// one step's draws to the next one's and, with `--gpu-timing`, the median
/// of every phase of a step (`RowsStepPhases`) and of the GPU's idle gap
/// between two passes; with `--kernel-profile`, the per-kernel table of the
/// `decode rows` pass.
fn bench_batched<M: LanguageModel>(
    cli: &Cli,
    ctx: &MetalContext,
    model: &M,
    rows: usize,
) -> Result<()> {
    ensure!(
        (1..=model.max_batch_rows()).contains(&rows),
        "--batch-rows takes 1 to {} for this model",
        model.max_batch_rows()
    );
    if cli.ngram_preload {
        let started = Instant::now();
        let bytes = model.warm_storage(false)?;
        eprintln!(
            "paged weights: {:.1} GB resident after preload in {:.1}s",
            bytes as f64 / 1e9,
            started.elapsed().as_secs_f64()
        );
    }
    let max_seq = cli.prompt_len + cli.decode_steps + 2;
    let vocab = u32::try_from(model.vocab_size())?;
    // Row `r`'s prompt: tokens `r * prompt_len ..` of one long sequence.
    let all = prompt_tokens_of(cli, vocab, cli.prompt_len * rows)?;
    let prompts: Vec<&[u32]> = all.chunks(cli.prompt_len).collect();
    let mut scratch = model.new_scratch_with_capacity(ctx, max_seq)?;
    // Each prefill draws into next_token[0]: read it before the next one.
    let prefill_all =
        |scratch: &mut M::Scratch| -> Result<(Vec<M::State>, Vec<u32>, f64)> {
            let started = Instant::now();
            let mut states = Vec::with_capacity(rows);
            let mut firsts = Vec::with_capacity(rows);
            for prompt in &prompts {
                let mut state = model.new_state(ctx, max_seq)?;
                model.prefill(ctx, &mut state, scratch, prompt, Some(draw(0)))?;
                firsts.push(scratch.next_token().view(0, &[1])?.to_u32()?[0]);
                states.push(state);
            }
            Ok((states, firsts, started.elapsed().as_secs_f64()))
        };
    let park = !cli.no_park;
    ensure!(
        !park || model.supports_rows_parking(),
        "this model cannot park batched steps; pass --no-park"
    );
    // `steps` batched steps over `states`, each draw appended to its row:
    // parked behind each other (the first committed unparked, the last not
    // followed by a parked one, as a stretch of the server's scheduler), or
    // one at a time. Returns, per step, the time from the previous step's
    // draws (or the call) to its own, and the steps' timings under
    // `--gpu-timing`.
    let run_steps = |scratch: &mut M::Scratch,
                     states: &mut [M::State],
                     tokens: &mut [Vec<u32>],
                     steps: usize|
     -> Result<(Vec<f64>, Vec<RowsStepTiming>)> {
        let timed = cli.gpu_timing;
        let batch = bench_rows::<M::State>;
        let take = |tokens: &mut [Vec<u32>], draws: Vec<u32>| {
            for (drawn, token) in tokens.iter_mut().zip(draws) {
                drawn.push(token);
            }
        };
        let mut walls = Vec::with_capacity(steps);
        let mut timings = Vec::new();
        let mut last = Instant::now();
        let mut lap = |walls: &mut Vec<f64>| {
            let now = Instant::now();
            walls.push((now - last).as_secs_f64());
            last = now;
        };
        if !park {
            for _ in 0..steps {
                let (draws, timing) = if timed {
                    model.decode_rows_timed(
                        ctx,
                        scratch,
                        &mut batch(states, tokens, 0),
                    )?
                } else {
                    (
                        model.decode_rows(
                            ctx,
                            scratch,
                            &mut batch(states, tokens, 0),
                        )?,
                        None,
                    )
                };
                take(tokens, draws);
                timings.extend(timing);
                lap(&mut walls);
            }
            return Ok((walls, timings));
        }
        let mut current =
            model.commit_rows(ctx, scratch, &mut batch(states, tokens, 0), timed)?;
        for i in 0..steps {
            let next = if i + 1 < steps {
                let rows = &mut batch(states, tokens, 1);
                Some(model.park_rows(ctx, scratch, rows, &current, timed)?)
            } else {
                None
            };
            let (draws, timing) = model.finish_rows(scratch, current)?;
            take(tokens, draws);
            timings.extend(timing);
            lap(&mut walls);
            let Some(mut next) = next else { break };
            model.release_rows(scratch, &mut next, &mut batch(states, tokens, 0))?;
            current = next;
        }
        Ok((walls, timings))
    };

    // Warm-up: the same row count compiles every shape the steps use.
    {
        let (mut states, firsts, _) = prefill_all(&mut scratch)?;
        let mut tokens: Vec<Vec<u32>> = firsts.into_iter().map(|t| vec![t]).collect();
        run_steps(&mut scratch, &mut states, &mut tokens, 2)?;
    }
    if ctx.profiling() {
        profile::take();
    }

    let (mut states, firsts, prefill_secs) = prefill_all(&mut scratch)?;
    let mut tokens: Vec<Vec<u32>> = firsts.into_iter().map(|t| vec![t]).collect();
    let decode_start = Instant::now();
    let (step_wall, timings) =
        run_steps(&mut scratch, &mut states, &mut tokens, cli.decode_steps)?;
    let decode_secs = decode_start.elapsed().as_secs_f64();
    let kernel_profile =
        ctx.profiling().then(|| kernel_profile_report(&profile::take()));

    let generated = rows * cli.decode_steps;
    let phases: Vec<RowsStepPhases> = timings.iter().map(|t| t.phases()).collect();
    let med = |f: fn(&RowsStepPhases) -> f64| median(phases.iter().map(f).collect());
    // Unparked: host time between one step's end and the next one's start
    // (the scheduler's own bookkeeping, here the loop's). Parked steps
    // begin before the one ahead of them ends.
    let between_ms = (!park).then(|| {
        median(timings.windows(2).map(|w| (w[1].began - w[0].ended) * 1e3).collect())
    });
    // The GPU's idle time between two consecutive passes: the host round
    // trip unparked, what is left of it parked.
    let gpu_gap_ms = median(
        timings
            .windows(2)
            .map(|w| (w[1].gpu.gpu_start_secs - w[0].gpu.gpu_end_secs) * 1e3)
            .collect(),
    );
    let phase_medians = (!phases.is_empty()).then(|| {
        serde_json::json!({
            "stage_ms": med(|p| p.stage_ms),
            "encode_ms": med(|p| p.encode_ms),
            "commit_ms": med(|p| p.commit_ms),
            "submit_ms": med(|p| p.submit_ms),
            "gpu_ms": med(|p| p.gpu_ms),
            "wake_ms": med(|p| p.wake_ms),
            "finish_ms": med(|p| p.finish_ms),
            "between_steps_ms": between_ms,
            "gpu_gap_ms": gpu_gap_ms,
            // `began` to `ended`; parked steps overlap the step before.
            "step_wall_ms": median(timings.iter().map(|t| t.wall_ms()).collect()),
        })
    });
    let digests: Vec<String> =
        tokens.iter().map(|t| format!("{:016x}", fnv1a(&t[1..]))).collect();
    let report = serde_json::json!({
        "schema_version": 1,
        "meta": {"engine": "lily", "model_id": M::MODEL_ID, "harness": "src/bin/lily-bench.rs", "crate_version": env!("CARGO_PKG_VERSION")},
        "workload": {
            "prompt_len": cli.prompt_len,
            "prompt": cli.prompt_text.as_ref().map_or("synthetic".to_string(), |p| p.display().to_string()),
            "decode_steps": cli.decode_steps,
            "decode_mode": "batched_rows",
            "batch_rows": rows,
            "parked": park,
            "draft_head_loaded": cli.drafts > 0,
            "sampling": if cli.sample { "server_defaults" } else { "greedy" },
            "seed": cli.seed,
            "gpu_timing_diagnostic": cli.gpu_timing,
            "kernel_profile_diagnostic": ctx.profiling(),
        },
        "results": {
            "prefill": {"wall_secs": prefill_secs, "tok_s": (cli.prompt_len * rows) as f64 / prefill_secs},
            "decode": {
                "wall_secs": decode_secs,
                "tokens": generated,
                "tok_s": generated as f64 / decode_secs,
                "tok_s_per_row": cli.decode_steps as f64 / decode_secs,
                "step_wall_secs": step_wall,
                "step_phases_median": phase_medians,
                "step_phases": phases.iter().map(|p| serde_json::json!({
                    "stage_ms": p.stage_ms, "encode_ms": p.encode_ms, "commit_ms": p.commit_ms,
                    "submit_ms": p.submit_ms, "gpu_ms": p.gpu_ms, "wake_ms": p.wake_ms,
                    "finish_ms": p.finish_ms,
                })).collect::<Vec<_>>(),
                "token_digests": digests,
                "kernel_profile": kernel_profile,
            },
        },
    });
    std::fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "batched decode ({}): {rows} rows x {} steps in {decode_secs:.3}s = {:.1} tok/s ({:.1} per row), step median {:.3} ms | digests {}",
        if park { "parked" } else { "unparked" },
        cli.decode_steps,
        generated as f64 / decode_secs,
        cli.decode_steps as f64 / decode_secs,
        median(step_wall.clone()) * 1e3,
        digests.join(","),
    );
    if let Some(m) = &phase_medians {
        eprintln!("step phases (median ms): {m}");
    }
    Ok(())
}
