//! Measures expert usage: prefills consecutive real-text prompts and counts,
//! per MoE layer, how often each expert is routed to. The skew decides
//! whether cold experts can be offloaded on smaller machines.
//!
//! With `--predict-out`, also measures cross-layer expert prediction
//! (`Qwen4ExpModel::enable_expert_prediction`, `docs/low-ram-experts.md`
//! "Cross-layer prediction"): after each prompt's prefill it decodes
//! `--predict-decode-tokens` greedy tokens with the plain decode step, and
//! scores, per phase, predictor variant and lookahead, how well the routing
//! of layer `l + d` is predicted from layer `l`; what a prefetcher would
//! read beyond the global top `--resident-slices` slices; and, per prompt,
//! which resident set (the prompt's own prefill ranking, the global one, a
//! mix) serves its decode lookups best. The counts output is the same with
//! or without it (prefill routing only). The prediction log keeps each
//! predictor's top `2 * top_k` with router scores; the scores above use
//! its top `top_k` by score.
//!
//! With `--replay-out` (needs `--predict-out`), also replays the expert
//! cache's live policy (`SlotPolicy`, `LiveUsage`, `expert_replay`) over
//! each prompt's logged routing, as the service thread would see it for a
//! plain run with `--replay-slots` slots, and compares it with re-ranking
//! the pinned set at the end of the prefill and with cross-layer prefetch
//! fed by the measured predictors: misses per token at prefill and decode,
//! prefetch reads, useful and wasted, layers with a miss per token.
//!
//! With `--predict-raw <dir>`, writes per prompt the token ids, the routing
//! log and the prediction log's ids and scores as little-endian binary
//! files with a JSON sidecar describing them.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions, ScratchApi};
use lily::kernels::sample::SamplingParams;
use lily::metal::MetalContext;
use lily::qwen4exp::expert_replay::{
    Policy, PredictedSet, Prefetch, PromptTrace, Replay, ReplayConfig, Weighting,
};
use lily::qwen4exp::expert_store::{SlotPolicy, UsageRanking};
use lily::qwen4exp::route_predict::{
    PredictionTally, VARIANT_NAMES, narrow_predictions, residency_sets, routing_counts,
    share_inside,
};
use lily::qwen4exp::{EXPERT_PRED_LOOKAHEADS, EXPERT_PRED_VARIANTS, Qwen4ExpModel};
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
    /// Prompts of the text to skip before the first one run (to keep
    /// prompts a ranking was measured on out of a measurement against it).
    #[arg(long, default_value_t = 0)]
    skip_prompts: usize,
    /// Where to write the per-layer counts (JSON: `counts[layer][expert]`).
    #[arg(long)]
    json_out: PathBuf,
    /// Also measure cross-layer expert prediction and write its summary
    /// here (JSON).
    #[arg(long)]
    predict_out: Option<PathBuf>,
    /// Greedy tokens decoded after each prompt with `--predict-out` (fewer
    /// when the model draws an end-of-sequence token).
    #[arg(long, default_value_t = 256)]
    predict_decode_tokens: usize,
    /// The global usage ranking (`lily-experts` counts JSON) the prefetch
    /// and residency figures are measured against; `expert-usage.json`
    /// next to the checkpoint by default.
    #[arg(long)]
    global_usage: Option<PathBuf>,
    /// Slices in the resident set (the global top N; also N for the
    /// per-prompt residency test).
    #[arg(long, default_value_t = 16384)]
    resident_slices: usize,
    /// Replay the expert cache's policy over each prompt's logged routing
    /// and write the comparison here (JSON).
    #[arg(long, requires = "predict_out")]
    replay_out: Option<PathBuf>,
    /// Slots of the replayed cache (16 246: the 64 GB plan).
    #[arg(long, default_value_t = 16246)]
    replay_slots: usize,
    /// LRU share of the replayed cache, clamped as the loader clamps it.
    #[arg(long, default_value_t = 0.10)]
    replay_lru_share: f64,
    /// Keep each replayed policy's slab and usage across prompts instead of
    /// a fresh fill per prompt (as a fresh process would have).
    #[arg(long)]
    replay_carry: bool,
    /// Score thresholds of the replayed prefetch, as multiples of the
    /// predictor's mean `top_k`-th highest score over the decode rows.
    #[arg(long, value_delimiter = ',', default_values_t = [0.5, 1.0, 2.0])]
    replay_thresholds: Vec<f64>,
    /// The lowest layer the deep-layers-only prefetch reads for.
    #[arg(long, default_value_t = 36)]
    replay_deep_from: usize,
    /// Write each prompt's token ids, routing log and prediction log (ids
    /// and scores) into this directory as little-endian binaries with a
    /// JSON sidecar.
    #[arg(long, requires = "predict_out")]
    predict_raw: Option<PathBuf>,
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`).
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

/// The prediction run's state across prompts.
struct Predict {
    out: PathBuf,
    usage_path: PathBuf,
    global: UsageRanking,
    /// `[layer * experts + expert]`: outside the global top N.
    cold: Vec<bool>,
    prefill: PredictionTally,
    decode: PredictionTally,
    /// Per prompt: decoded tokens and the share of their lookups inside the
    /// own / global / mixed top N.
    residency: Vec<(usize, [f64; 3])>,
    eos_stops: usize,
    /// Per prompt, for the replay: the routing and the lookahead-1
    /// predictions of variants B and A (predictor 0 and 1).
    traces: Vec<PromptTrace>,
}

/// The replay's predictors, as variant indices of the prediction log.
const REPLAY_PREDICTORS: [usize; 2] = [1, 0];

fn run() -> Result<()> {
    let cli = Cli::parse();
    let text = std::fs::read_to_string(&cli.text)?;
    let started = Instant::now();
    let ids = Tokenizer::from_model_dir(&cli.model)?.encode(&text)?;
    let n_prompts = (ids.len() / cli.prompt_len)
        .saturating_sub(cli.skip_prompts)
        .min(cli.prompts.unwrap_or(usize::MAX));
    ensure!(
        n_prompts > 0,
        "{} tokens: fewer than one prompt after skipping {}",
        ids.len(),
        cli.skip_prompts
    );
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
    let mut predict = match &cli.predict_out {
        Some(out) => {
            let usage_path = cli
                .global_usage
                .clone()
                .unwrap_or_else(|| cli.model.join("expert-usage.json"));
            let global = UsageRanking::load(&usage_path).with_context(|| {
                "the global usage ranking (--global-usage, or expert-usage.json \
                 next to the checkpoint)"
            })?;
            ensure!(
                global.layers == layers && global.experts == experts,
                "{}: {} x {} counts for a {layers} x {experts} model",
                usage_path.display(),
                global.layers,
                global.experts
            );
            ensure!(
                cli.resident_slices <= layers * experts,
                "--resident-slices {} exceeds the {} slices",
                cli.resident_slices,
                layers * experts
            );
            let none = vec![vec![0u64; experts]; layers];
            let [_, resident, _] =
                residency_sets(&none, &global.counts, cli.resident_slices);
            let cold: Vec<bool> = resident.iter().map(|&r| !r).collect();
            Some(Predict {
                out: out.clone(),
                usage_path,
                global,
                prefill: PredictionTally::new(
                    layers,
                    experts,
                    top_k,
                    Some(cold.clone()),
                ),
                decode: PredictionTally::new(
                    layers,
                    experts,
                    top_k,
                    Some(cold.clone()),
                ),
                cold,
                residency: Vec::new(),
                eos_stops: 0,
                traces: Vec::new(),
            })
        }
        None => None,
    };
    let decode_tokens = if predict.is_some() { cli.predict_decode_tokens } else { 0 };
    let capacity = cli.prompt_len + decode_tokens + 1;
    let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
    if predict.is_some() {
        model.enable_expert_prediction(&ctx, &mut scratch, capacity)?;
    } else {
        model.enable_expert_log(&ctx, &mut scratch, capacity)?;
    }
    let eos = model.eos_token_ids();
    let greedy = SamplingParams::greedy();
    let mut counts = vec![vec![0u64; experts]; layers];
    let started = Instant::now();
    for p in 0..n_prompts {
        let first = (cli.skip_prompts + p) * cli.prompt_len;
        let prompt = &ids[first..first + cli.prompt_len];
        let mut state = model.new_state(&ctx, capacity)?;
        // The draw only feeds the decode below; the routing is the same.
        let draw = predict.is_some().then_some(Draw { params: &greedy, step: 0 });
        LanguageModel::prefill(&model, &ctx, &mut state, &mut scratch, prompt, draw)?;
        let mut decoded = 0usize;
        // The token at every logged position: the prompt, then each decode
        // step's input.
        let mut tokens = prompt.to_vec();
        if let Some(pr) = &mut predict {
            let mut slot = 0usize;
            while decoded < decode_tokens {
                let input = scratch.next_token().view(slot, &[1])?.to_u32()?[0];
                if eos.contains(&input) {
                    pr.eos_stops += 1;
                    break;
                }
                tokens.push(input);
                let encoded = LanguageModel::encode_decode_step(
                    &model,
                    &ctx,
                    &state,
                    &scratch,
                    slot,
                    1 - slot,
                    Draw { params: &greedy, step: decoded + 1 },
                )?;
                model.prepare_step_inputs(&mut state, &scratch, input)?;
                let pending = encoded.commit()?;
                state.advance(1);
                pending.wait()?;
                slot = 1 - slot;
                decoded += 1;
            }
        }
        let log = model.expert_log(&scratch).expect("enabled").to_u32()?;
        for (layer, row) in counts.iter_mut().enumerate() {
            let base = layer * capacity * top_k;
            for &e in &log[base..base + cli.prompt_len * top_k] {
                row[e as usize] += 1;
            }
        }
        if let Some(pr) = &mut predict {
            let wide_log = model.expert_pred_log(&scratch).expect("enabled");
            let width = wide_log.shape()[4];
            let wide = wide_log.to_u32()?;
            let scores =
                model.expert_pred_scores(&scratch).expect("enabled").to_f32()?;
            // The summary scores each prediction's top-k by score.
            let pred = narrow_predictions(&wide, &scores, width, top_k)?;
            let prefill = 0..cli.prompt_len;
            let decode = cli.prompt_len..cli.prompt_len + decoded;
            pr.prefill.add(&log, &pred, capacity, prefill.clone())?;
            pr.decode.add(&log, &pred, capacity, decode.clone())?;
            if let Some(dir) = &cli.predict_raw {
                let raw = RawLogs {
                    tokens: &tokens,
                    routing: &log,
                    pred_ids: &wide,
                    pred_scores: &scores,
                    capacity,
                    width,
                };
                write_raw(
                    dir,
                    p,
                    first,
                    raw,
                    (layers, experts, top_k),
                    cli.prompt_len,
                )?;
            }
            if cli.replay_out.is_some() {
                pr.traces.push(PromptTrace::from_logs(
                    &log,
                    &wide,
                    &scores,
                    (layers, top_k, capacity),
                    (EXPERT_PRED_LOOKAHEADS, width),
                    cli.prompt_len,
                    decoded,
                    &REPLAY_PREDICTORS,
                )?);
            }
            if decoded > 0 {
                let own =
                    routing_counts(&log, layers, experts, top_k, capacity, prefill);
                let sets = residency_sets(&own, &pr.global.counts, cli.resident_slices);
                let shares = sets.map(|mask| {
                    share_inside(
                        &mask,
                        &log,
                        layers,
                        experts,
                        top_k,
                        capacity,
                        decode.clone(),
                    )
                });
                pr.residency.push((decoded, shares));
            }
        }
        if (p + 1) % 10 == 0 || p + 1 == n_prompts || predict.is_some() {
            eprintln!(
                "{}/{n_prompts} prompts, {:.0} tok/s{}",
                p + 1,
                ((p + 1) * cli.prompt_len) as f64 / started.elapsed().as_secs_f64(),
                if predict.is_some() {
                    format!(", {decoded} decoded")
                } else {
                    String::new()
                }
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
    if let Some(pr) = predict {
        write_prediction(&cli, &pr, n_prompts, (layers, experts, top_k))?;
        if let Some(out) = &cli.replay_out {
            let chunk = model.prefill_chunk();
            write_replay(&cli, out, &pr, chunk, (layers, experts, top_k))?;
        }
    }
    Ok(())
}

/// One prompt's logs for `--predict-raw`.
struct RawLogs<'a> {
    tokens: &'a [u32],
    /// `[layers, capacity, top_k]`.
    routing: &'a [u32],
    /// `[variants, lookaheads, layers, capacity, width]` each.
    pred_ids: &'a [u32],
    pred_scores: &'a [f32],
    capacity: usize,
    width: usize,
}

/// Writes `values` as little-endian 4-byte words.
fn write_words<T: Copy>(
    path: &Path,
    values: impl Iterator<Item = T>,
    le: fn(T) -> [u8; 4],
) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    for v in values {
        w.write_all(&le(v))?;
    }
    w.flush().with_context(|| format!("writing {}", path.display()))
}

/// `--predict-raw`: prompt `p`'s token ids, routing and predictions, cut
/// to the positions the passes ran, plus `prompt-NNN.json` describing them.
fn write_raw(
    dir: &Path,
    p: usize,
    first_token: usize,
    raw: RawLogs<'_>,
    (layers, experts, top_k): (usize, usize, usize),
    prompt_len: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    let n = raw.tokens.len();
    let (cap, w) = (raw.capacity, raw.width);
    let stem = format!("prompt-{p:03}");
    let file = |suffix: &str| format!("{stem}-{suffix}");
    write_words(
        &dir.join(file("tokens.u32")),
        raw.tokens.iter().copied(),
        u32::to_le_bytes,
    )?;
    let routing =
        (0..layers).flat_map(|l| raw.routing[l * cap * top_k..][..n * top_k].iter());
    write_words(&dir.join(file("routing.u32")), routing.copied(), u32::to_le_bytes)?;
    let planes = EXPERT_PRED_VARIANTS * EXPERT_PRED_LOOKAHEADS * layers;
    let cut = |plane: usize| plane * cap * w..plane * cap * w + n * w;
    let ids = (0..planes).flat_map(|pl| raw.pred_ids[cut(pl)].iter());
    write_words(&dir.join(file("pred-ids.u32")), ids.copied(), u32::to_le_bytes)?;
    let scores = (0..planes).flat_map(|pl| raw.pred_scores[cut(pl)].iter());
    write_words(&dir.join(file("pred-scores.f32")), scores.copied(), f32::to_le_bytes)?;
    let pred_shape = [EXPERT_PRED_VARIANTS, EXPERT_PRED_LOOKAHEADS, layers, n, w];
    let sidecar = serde_json::json!({
        "prompt": p,
        "first_token": first_token,
        "prefill_tokens": prompt_len,
        "decode_tokens": n - prompt_len,
        "positions": n,
        "layers": layers, "experts": experts, "top_k": top_k, "width": w,
        "byte_order": "little-endian, C order (last axis fastest)",
        "files": {
            "tokens": {
                "path": file("tokens.u32"), "dtype": "u32", "shape": [n],
                "meaning": "the token at each position: the prompt, then each decode \
                            step's input",
            },
            "routing": {
                "path": file("routing.u32"), "dtype": "u32", "shape": [layers, n, top_k],
                "meaning": "the routed experts of each MoE layer at each position, in \
                            the router kernel's order (score descending)",
            },
            "pred_ids": {
                "path": file("pred-ids.u32"), "dtype": "u32", "shape": pred_shape,
                "meaning": "[variant, lookahead, layer, position, pick]: the experts \
                            variant v predicts for layer l from layer l - d, in the \
                            router kernel's order; rows with l < d are zero (no source)",
            },
            "pred_scores": {
                "path": file("pred-scores.f32"), "dtype": "f32", "shape": pred_shape,
                "meaning": "the router scores of pred_ids (renormalised over the \
                            width picks when the model renormalises)",
            },
        },
        "variants": {
            "0": "A: router of layer l+d on layer l's gate input",
            "1": "B: read-in and router of layer l+d on the streams after layer l's \
                  MoE inject",
        },
        "lookaheads": "0 (self-check, reproduces the routing), 1, 2",
    });
    let path = dir.join(format!("{stem}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&sidecar)?)
        .with_context(|| format!("writing {}", path.display()))
}

/// `--replay-out`: replays every prompt's trace through the live policy,
/// the two re-ranks and the prefetch grid, prints the headline rows and
/// writes the JSON.
fn write_replay(
    cli: &Cli,
    out: &Path,
    pr: &Predict,
    prefill_chunk: usize,
    (layers, experts, top_k): (usize, usize, usize),
) -> Result<()> {
    ensure!(
        cli.replay_slots >= 2 * experts && cli.replay_slots <= layers * experts,
        "--replay-slots {} outside {}..={} (the loader needs two layers' worth)",
        cli.replay_slots,
        2 * experts,
        layers * experts
    );
    let lru_share =
        SlotPolicy::lru_share_for(cli.replay_lru_share, experts, cli.replay_slots);
    let cfg = ReplayConfig {
        layers,
        experts,
        top_k,
        slots: cli.replay_slots,
        lru_share,
        prefill_chunk,
        adapt: true,
        carry: cli.replay_carry,
    };
    // Per predictor, the mean top_k-th score the thresholds scale.
    let kth: Vec<f64> = (0..REPLAY_PREDICTORS.len())
        .map(|i| {
            let (sum, n) = pr.traces.iter().fold((0.0, 0u64), |(s, n), t| {
                let (ts, tn) = t.kth_score_sum(i, layers, top_k);
                (s + ts, n + tn)
            });
            sum / n.max(1) as f64
        })
        .collect();
    let mut policies = vec![
        ("live".to_string(), Policy::Live),
        ("rerank-own".into(), Policy::Rerank(Weighting::Own)),
        ("rerank-mix".into(), Policy::Rerank(Weighting::Mix)),
    ];
    let mut thresholds = Vec::new();
    for (i, &variant) in REPLAY_PREDICTORS.iter().enumerate() {
        let name = VARIANT_NAMES[variant];
        let mut sets = vec![
            (format!("top{top_k}"), PredictedSet::Top(top_k)),
            (format!("top{}", 2 * top_k), PredictedSet::Top(2 * top_k)),
        ];
        for &tau in &cli.replay_thresholds {
            let min = (tau * kth[i]) as f32;
            sets.push((format!("score>={tau}x"), PredictedSet::AtLeast(min)));
            thresholds.push(serde_json::json!({
                "variant": name, "tau": tau, "min_score": min,
            }));
        }
        for (set_name, set) in &sets {
            for (budget_name, budget) in
                [("unlimited", None), ("2/layer", Some(2)), ("1/layer", Some(1))]
            {
                for (layers_name, from_layer) in
                    [("all", 1), ("deep", cli.replay_deep_from.max(1))]
                {
                    policies.push((
                        format!(
                            "prefetch-{name}-{set_name}-{budget_name}-{layers_name}"
                        ),
                        Policy::Prefetch(Prefetch {
                            predictor: i,
                            set: *set,
                            budget,
                            from_layer,
                        }),
                    ));
                }
            }
        }
    }
    let started = Instant::now();
    let mut replay = Replay::new(cfg.clone(), &pr.global, policies)?;
    for (p, trace) in pr.traces.iter().enumerate() {
        replay.add_prompt(trace)?;
        eprintln!(
            "replay: {}/{} prompts, {:.1}s",
            p + 1,
            pr.traces.len(),
            started.elapsed().as_secs_f64()
        );
    }
    let results = replay.to_json();
    println!(
        "replay of the live policy: {} slots ({} LRU share), {} prompts, {}",
        cfg.slots,
        lru_share,
        pr.traces.len(),
        if cfg.carry { "slab carried across prompts" } else { "fresh fill per prompt" }
    );
    println!(
        "policy                                         | prefill miss/tok | decode miss/tok | \
         layers missed/tok | reads/tok | useful/tok | wasted/tok | set size | one-off reads"
    );
    for p in results.as_array().into_iter().flatten() {
        let d = &p["decode"];
        let f = &d["prefetch"];
        let num = |v: &serde_json::Value| v.as_f64().unwrap_or(0.0);
        println!(
            "{:46} | {:16.2} | {:15.2} | {:17.2} | {:9.2} | {:10.2} | {:10.2} | {:8.1} | {}",
            p["name"].as_str().unwrap_or(""),
            num(&p["prefill"]["misses_per_token"]),
            num(&d["misses_per_token"]),
            num(&d["layers_with_miss_per_token"]["mean"]),
            num(&f["reads_per_token"]),
            num(&f["useful_per_token"]),
            num(&f["wasted_per_token"]),
            num(&f["mean_set_size"]),
            p.get("rerank").map_or(String::new(), |r| format!(
                "{:.0} pinned + {:.0} LRU per prompt",
                num(&r["pinned_reads_per_prompt"]),
                num(&r["lru_fill_reads_per_prompt"])
            )),
        );
    }
    let record = serde_json::json!({
        "model": cli.model.display().to_string(),
        "text": cli.text.display().to_string(),
        "prompts": pr.traces.len(),
        "prompt_len": cli.prompt_len,
        "decode_tokens": pr.traces.iter().map(|t| t.decode_tokens).collect::<Vec<_>>(),
        "global_usage": pr.usage_path.display().to_string(),
        "layers": layers, "experts": experts, "top_k": top_k,
        "slots": cfg.slots,
        "lru_share": lru_share,
        "pinned": ((cfg.slots as f64) * (1.0 - lru_share)).round() as usize,
        "prefill_chunk": prefill_chunk,
        "carry": cfg.carry,
        "deep_from": cli.replay_deep_from,
        "predictors": REPLAY_PREDICTORS.iter().map(|&v| VARIANT_NAMES[v]).collect::<Vec<_>>(),
        "mean_kth_score": REPLAY_PREDICTORS
            .iter()
            .zip(&kth)
            .map(|(&v, &s)| (VARIANT_NAMES[v].to_string(), serde_json::json!(s)))
            .collect::<serde_json::Map<_, _>>(),
        "thresholds": thresholds,
        "policies": results,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("writing {}", out.display()))
}

/// Prints the headline figures and writes the `--predict-out` JSON.
fn write_prediction(
    cli: &Cli,
    pr: &Predict,
    n_prompts: usize,
    (layers, experts, top_k): (usize, usize, usize),
) -> Result<()> {
    let prefill = pr.prefill.to_json();
    let decode = pr.decode.to_json();
    for (phase, json) in [("prefill", &prefill), ("decode", &decode)] {
        for name in VARIANT_NAMES {
            let exact = json["self_check"][name]["exact"].as_f64().unwrap_or(0.0);
            if exact != 1.0 {
                eprintln!(
                    "WARNING: {phase} variant {name} lookahead 0 reproduces only {:.6} of \
                     the routing; the prediction path is not the real one",
                    exact
                );
            }
        }
    }
    println!(
        "prediction, {} decoded tokens (resident set: global top {} of {}):",
        pr.decode.tokens(),
        cli.resident_slices,
        layers * experts
    );
    println!(
        "phase   | variant | ahead | overlap | covered | missing layers/token | \
         reads/token | cold recall | read precision"
    );
    for (phase, json) in [("prefill", &prefill), ("decode", &decode)] {
        for name in VARIANT_NAMES {
            for d in ["1", "2"] {
                let p = &json[name][d];
                let f = &p["prefetch"];
                println!(
                    "{phase:7} | {name:7} | {d:5} | {:6.1}% | {:6.1}% | {:20.2} | {:11.2} | \
                     {:10.1}% | {:13.1}%",
                    100.0 * p["all_layers"]["overlap"].as_f64().unwrap_or(0.0),
                    100.0 * p["all_layers"]["covered"].as_f64().unwrap_or(0.0),
                    p["missing_layers_per_token"]["mean"].as_f64().unwrap_or(0.0),
                    f["reads_per_token"]["mean"].as_f64().unwrap_or(0.0),
                    100.0 * f["cold_recall"].as_f64().unwrap_or(0.0),
                    100.0 * f["read_precision"].as_f64().unwrap_or(0.0),
                );
            }
        }
    }
    let n = pr.residency.len().max(1) as f64;
    let mean: [f64; 3] = std::array::from_fn(|i| {
        pr.residency.iter().map(|(_, s)| s[i]).sum::<f64>() / n
    });
    println!(
        "decode lookups inside the top {} (mean over {} prompts): own prefill {:.1}%, \
         global {:.1}%, mix {:.1}%",
        cli.resident_slices,
        pr.residency.len(),
        100.0 * mean[0],
        100.0 * mean[1],
        100.0 * mean[2]
    );
    let per_prompt: Vec<serde_json::Value> = pr
        .residency
        .iter()
        .map(|(decoded, s)| {
            serde_json::json!({
                "decode_tokens": decoded, "own": s[0], "global": s[1], "mix": s[2],
            })
        })
        .collect();
    let record = serde_json::json!({
        "model": cli.model.display().to_string(),
        "text": cli.text.display().to_string(),
        "prompts": n_prompts,
        "skip_prompts": cli.skip_prompts,
        "prompt_len": cli.prompt_len,
        "decode_tokens_per_prompt": cli.predict_decode_tokens,
        "decode_tokens": pr.decode.tokens(),
        "decode_eos_stops": pr.eos_stops,
        "layers": layers, "experts": experts, "top_k": top_k,
        "global_usage": pr.usage_path.display().to_string(),
        "resident_slices": cli.resident_slices,
        "cold_slices": pr.cold.iter().filter(|&&c| c).count(),
        "variants": {
            "A": "router of layer l+d on layer l's gate input (its hyper-connection read)",
            "B": "read-in (hyper-connection read and norm) and router of layer l+d on \
                  the streams after layer l's MoE inject",
        },
        "prefill": prefill,
        "decode": decode,
        "residency": {
            "top_n": cli.resident_slices,
            "mean": { "own": mean[0], "global": mean[1], "mix": mean[2] },
            "per_prompt": per_prompt,
        },
    });
    std::fs::write(&pr.out, serde_json::to_vec_pretty(&record)?)?;
    Ok(())
}
