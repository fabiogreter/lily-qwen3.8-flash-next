//! Replays recorded sessions' expert routing through the expert cache's
//! policy (`lily::qwen4exp::cache_replay`) for a sweep of LRU shares and
//! decode-time prefetches; host only, no model.
//!
//! Each `--trace STEM` names `STEM.routing.u32` (from `lily-trace-routing`)
//! and `STEM.json`: `{"tokens": N, "requests": [[p0, p1, p2], ...]}`, a
//! request prefilling `p0..p1` and generating `p1..p2`. Sessions replay in
//! the order given through one cache, as one server would serve them,
//! unless `--fresh` starts every session from the loaded ranking. A
//! prefetch reads `STEM.pred.u16` (`lily-trace-routing --predict`), or with
//! `--predictor transitions` predicts from cross-layer transition counts
//! (`cache_replay::Transitions`) learned on the other sessions given.

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::qwen4exp::cache_replay::{
    CacheSim, Predictions, PrefetchReplay, ReplayConfig, ReplayCounts, TraceRequest,
    Transitions,
};
use lily::qwen4exp::expert_store::UsageRanking;

#[derive(Parser)]
#[command(
    name = "lily-cache-replay",
    about = "Expert cache policy over recorded routing"
)]
struct Cli {
    /// The loaded usage ranking (`lily-experts` JSON).
    #[arg(long)]
    usage: PathBuf,
    /// Session stems (`STEM.routing.u32` and `STEM.json`), in serving order.
    #[arg(long, required = true)]
    trace: Vec<PathBuf>,
    #[arg(long, default_value_t = 16246)]
    slots: usize,
    /// LRU shares to sweep (the loader's floor and ceiling still apply).
    #[arg(long, value_delimiter = ',', default_value = "0.05,0.1,0.2,0.3,0.4")]
    lru_shares: Vec<f64>,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 4096)]
    prefill_chunk: usize,
    /// Prefetches to sweep: `off`, or `LOOKAHEAD:TOP[:BUDGET]` (predictor
    /// B's `TOP` best for the layer `LOOKAHEAD` ahead, at most `BUDGET`
    /// reads per layer).
    #[arg(long, value_delimiter = ',', default_value = "off")]
    prefetch: Vec<String>,
    /// What the prefetch predicts from: `b` (the logged predictor B) or
    /// `transitions` (each session's decode predicted from the transition
    /// counts of all the other sessions' positions).
    #[arg(long, default_value = "b")]
    predictor: String,
    /// Picks kept per row by `--predictor transitions`.
    #[arg(long, default_value_t = 20)]
    learned_width: usize,
    /// Start every session from the loaded ranking.
    #[arg(long, default_value_t = false)]
    fresh: bool,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

/// One configuration's replay: the pinned slots, each session's counts, and
/// decode misses and useful prefetch reads per layer over all of them.
struct Replayed {
    pinned: usize,
    per: Vec<ReplayCounts>,
    misses: Vec<u64>,
    useful: Vec<u64>,
}

struct Trace {
    name: String,
    routing: Vec<u32>,
    /// `STEM.pred.u16`, when a prefetch is swept.
    predicted: Vec<u16>,
    tokens: usize,
    requests: Vec<TraceRequest>,
}

fn parse_prefetch(spec: &str) -> Result<Option<PrefetchReplay>> {
    if spec == "off" {
        return Ok(None);
    }
    let v: Vec<usize> = spec
        .split(':')
        .map(|x| x.parse().with_context(|| format!("prefetch {spec}")))
        .collect::<Result<_>>()?;
    ensure!((2..=3).contains(&v.len()), "prefetch {spec}: LOOKAHEAD:TOP[:BUDGET]");
    Ok(Some(PrefetchReplay { lookahead: v[0], top: v[1], budget: v.get(2).copied() }))
}

fn label(p: Option<PrefetchReplay>) -> String {
    match p {
        None => "off".into(),
        Some(p) => match p.budget {
            Some(b) => format!("{}:{}:{b}", p.lookahead, p.top),
            None => format!("{}:{}", p.lookahead, p.top),
        },
    }
}

/// Fills each trace's predictions from the transition counts of all the
/// other traces (leave one session out), for its decoded positions.
fn learn_predictions(
    traces: &mut [Trace],
    ranking: &UsageRanking,
    top_k: usize,
    width: usize,
) -> Result<()> {
    let (layers, experts) = (ranking.layers, ranking.experts);
    ensure!((1..=experts).contains(&width), "--learned-width 1 to {experts}");
    let mut all = Transitions::new(layers, experts);
    for t in traces.iter() {
        ensure!(
            t.routing.len() == layers * t.tokens * top_k,
            "{}: routing size",
            t.name
        );
        all.count(&t.routing, t.tokens, top_k, true);
    }
    std::thread::scope(|scope| {
        let all = &all;
        let handles: Vec<_> = traces
            .iter_mut()
            .map(|t| {
                scope.spawn(move || {
                    let mut own = all.clone();
                    own.count(&t.routing, t.tokens, top_k, false);
                    let positions: Vec<usize> =
                        t.requests.iter().flat_map(|r| r.decode.clone()).collect();
                    t.predicted =
                        own.predict(&t.routing, t.tokens, top_k, &positions, width);
                })
            })
            .collect();
        for h in handles {
            h.join().map_err(|_| anyhow::anyhow!("a prediction thread panicked"))?;
        }
        Ok(())
    })
}

fn load_trace(stem: &std::path::Path, predicted: bool) -> Result<Trace> {
    let with = |ext: &str| PathBuf::from(format!("{}.{ext}", stem.display()));
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(with("json"))
            .with_context(|| format!("{}.json", stem.display()))?,
    )?;
    let tokens = meta["tokens"].as_u64().context("tokens")? as usize;
    let requests = meta["requests"]
        .as_array()
        .context("requests")?
        .iter()
        .map(|r| -> Result<TraceRequest> {
            let v: Vec<usize> = r
                .as_array()
                .context("request")?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).context("bound"))
                .collect::<Result<_>>()?;
            ensure!(v.len() == 3, "a request is [p0, p1, p2]");
            Ok(TraceRequest { prefill: v[0]..v[1], decode: v[1]..v[2] })
        })
        .collect::<Result<Vec<_>>>()?;
    let bytes = std::fs::read(with("routing.u32"))
        .with_context(|| format!("{}.routing.u32", stem.display()))?;
    ensure!(bytes.len().is_multiple_of(4), "routing is not u32s");
    let routing = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let predicted = if predicted {
        let bytes = std::fs::read(with("pred.u16"))
            .with_context(|| format!("{}.pred.u16", stem.display()))?;
        ensure!(bytes.len().is_multiple_of(2), "predictions are not u16s");
        bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
    } else {
        Vec::new()
    };
    let name = stem.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    Ok(Trace { name, routing, predicted, tokens, requests })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ranking = UsageRanking::load(&cli.usage)?;
    let prefetches =
        cli.prefetch.iter().map(|p| parse_prefetch(p)).collect::<Result<Vec<_>>>()?;
    let predicted = prefetches.iter().any(Option::is_some);
    ensure!(
        matches!(cli.predictor.as_str(), "b" | "transitions"),
        "--predictor b or transitions"
    );
    let learned = cli.predictor == "transitions";
    let mut traces = cli
        .trace
        .iter()
        .map(|t| load_trace(t, predicted && !learned))
        .collect::<Result<Vec<_>>>()?;
    if predicted && learned {
        learn_predictions(&mut traces, &ranking, cli.top_k, cli.learned_width)?;
    }
    let mut configs = Vec::new();
    for &lru_share in &cli.lru_shares {
        for &prefetch in &prefetches {
            configs.push(ReplayConfig {
                slots: cli.slots,
                lru_share,
                adapt: true,
                prefill_chunk: cli.prefill_chunk,
                prefetch,
            });
        }
    }
    let results: Vec<Result<Replayed>> = std::thread::scope(|scope| {
        let handles: Vec<_> = configs
            .iter()
            .map(|&config| {
                let (ranking, traces) = (&ranking, &traces);
                scope.spawn(move || -> Result<Replayed> {
                    let mut sim = CacheSim::new(ranking, cli.top_k, config)?;
                    let pinned = sim.pinned();
                    let mut per = Vec::with_capacity(traces.len());
                    let layers = ranking.layers;
                    let (mut misses, mut useful) = (vec![0; layers], vec![0; layers]);
                    let mut take = |sim: &CacheSim| {
                        let (m, u) = sim.by_layer();
                        misses.iter_mut().zip(m).for_each(|(a, b)| *a += b);
                        useful.iter_mut().zip(u).for_each(|(a, b)| *a += b);
                    };
                    for t in traces {
                        if cli.fresh {
                            take(&sim);
                            sim = CacheSim::new(ranking, cli.top_k, config)?;
                        }
                        let mut c = ReplayCounts::default();
                        let pred = (!t.predicted.is_empty()).then(|| Predictions {
                            ids: &t.predicted,
                            width: t.predicted.len() / (2 * layers * t.tokens.max(1)),
                        });
                        sim.session(
                            &t.routing,
                            pred.as_ref(),
                            t.tokens,
                            cli.top_k,
                            &t.requests,
                            &mut c,
                        )?;
                        per.push(c);
                    }
                    take(&sim);
                    Ok(Replayed { pinned, per, misses, useful })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().map_err(|_| anyhow::anyhow!("a replay thread panicked"))?
            })
            .collect()
    });
    let rate = |n: u64, d: u64| n as f64 / d.max(1) as f64;
    println!(
        "{} sessions, {} slots, {}",
        traces.len(),
        cli.slots,
        if cli.fresh { "fresh per session" } else { "one cache across sessions" }
    );
    println!(
        "lru share | pinned | prefetch | decode miss/tok | layers missed/tok | prefetch reads/tok | useful/tok | prefill miss/tok | promotions | full"
    );
    let mut json = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        let Replayed { pinned, per, misses, useful } = result?;
        let mut all = ReplayCounts::default();
        for c in &per {
            all.plus(c);
        }
        println!(
            "{:9.2} | {:6} | {:8} | {:15.2} | {:17.2} | {:18.2} | {:10.2} | {:16.3} | {:10} | {}",
            config.lru_share,
            pinned,
            label(config.prefetch),
            rate(all.decode_misses, all.decode_tokens),
            rate(all.decode_layers_missed, all.decode_tokens),
            rate(all.prefetch_reads, all.decode_tokens),
            rate(all.prefetch_useful, all.decode_tokens),
            rate(all.prefill_misses, all.prefill_tokens),
            all.promotions,
            all.full
        );
        let counts = |c: &ReplayCounts| {
            serde_json::json!({
                "prefill_tokens": c.prefill_tokens, "prefill_lookups": c.prefill_lookups,
                "prefill_misses": c.prefill_misses, "decode_tokens": c.decode_tokens,
                "decode_lookups": c.decode_lookups, "decode_misses": c.decode_misses,
                "decode_layers_missed": c.decode_layers_missed, "full": c.full,
                "promotions": c.promotions, "prefetch_reads": c.prefetch_reads,
                "prefetch_useful": c.prefetch_useful, "prefetch_full": c.prefetch_full,
            })
        };
        json.push(serde_json::json!({
            "lru_share": config.lru_share,
            "pinned": pinned,
            "decode_misses_by_layer": misses,
            "prefetch_useful_by_layer": useful,
            "prefetch": label(config.prefetch),
            "all": counts(&all),
            "sessions": traces.iter().zip(&per)
                .map(|(t, c)| serde_json::json!({"name": t.name, "counts": counts(c)}))
                .collect::<Vec<_>>(),
        }));
    }
    if let Some(path) = &cli.json_out {
        let record = serde_json::json!({
            "usage": cli.usage.display().to_string(),
            "slots": cli.slots,
            "fresh": cli.fresh,
            "predictor": cli.predictor,
            "results": json,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
    }
    Ok(())
}
