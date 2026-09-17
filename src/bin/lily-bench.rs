use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Result, ensure};
use clap::Parser;
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::generate::speculate;
use lily::kernels::attention::MAX_SEQ;
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::metal::profile::{self, PassProfile};
use lily::metal::{EncodedPass, Pacer, PendingPass};
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
    let params =
        if cli.sample { server_sampling(cli.seed) } else { SamplingParams::greedy() };
    SAMPLER.set(params).expect("sampler set once");
    match checkpoint_model_type(&cli.model)?.as_str() {
        "qwen3_5_moe" => bench::<Qwen3_5Model>(&cli),
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

/// Host time in seconds on the clock Metal's `GPUStartTime`/`GPUEndTime`
/// use (mach_absolute_time), so host and GPU marks can be subtracted.
fn host_secs() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let mut tb = Timebase { numer: 0, denom: 0 };
    // SAFETY: plain libSystem calls with a valid out-pointer.
    let ticks = unsafe {
        mach_timebase_info(&mut tb);
        mach_absolute_time()
    };
    ticks as f64 * tb.numer as f64 / tb.denom as f64 * 1e-9
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
        tables.push(serde_json::json!({
            "label": label,
            "passes": group.len(),
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
        &LoadOptions { mtp_drafts: cli.drafts, ..LoadOptions::default() },
    )?;
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
    if ctx.profiling() {
        // The tables cover the measured passes only.
        profile::take();
    }

    let mut state = model.new_state(&ctx, max_seq)?;
    let prefill_start = Instant::now();
    model.prefill(&ctx, &mut state, &mut scratch, &prompt, Some(draw(0)))?;

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
        let (parked, encoded) =
            stage_next(&model, &ctx, &mut state, &scratch, index, parking)?;
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
        "workload": {"prompt_len": cli.prompt_len, "decode_steps": cli.decode_steps, "decode_mode": format!("speculative_{}_drafts", cli.drafts), "sampling": if cli.sample { "server_defaults" } else { "greedy" }, "seed": cli.seed, "kernel_profile_diagnostic": ctx.profiling()},
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
