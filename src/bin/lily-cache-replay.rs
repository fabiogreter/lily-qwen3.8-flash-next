//! Replays recorded sessions' expert routing through the expert cache's
//! policy (`lily::qwen4exp::cache_replay`) for a sweep of LRU shares and
//! decode-time prefetches; host only, no model.
//!
//! Each `--trace STEM` names `STEM.routing.u32` (from `lily-trace-routing`)
//! and `STEM.json`: `{"tokens": N, "requests": [[p0, p1, p2], ...]}`, a
//! request prefilling `p0..p1` and generating `p1..p2`. Sessions replay in
//! the order given through one cache, as one server would serve them,
//! unless `--fresh` starts every session from the loaded ranking (or with
//! `--persist` from the usage the previous session's cache would have
//! persisted, as a server restarted between sessions would). A
//! prefetch reads `STEM.pred.u16` (`lily-trace-routing --predict`), or with
//! `--predictor transitions` predicts from cross-layer transition counts
//! (`cache_replay::Transitions`) learned on the other sessions given, or
//! with `--predictor token` from the routing of each input token id
//! (`cache_replay::TokenRouting`, reading `STEM.u32`), for a prefetch at
//! the token's start (`0:TOP`).

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::qwen4exp::cache_replay::{
    CacheSim, Predictions, PrefetchReplay, ReplayConfig, ReplayCounts, SsdTiming,
    TokenRouting, TraceRequest, Transitions,
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
    /// Prefetches to sweep: `off`, or `LOOKAHEAD:TOP[:BUDGET]` (the
    /// predictor's `TOP` best for the layer `LOOKAHEAD` ahead, at most
    /// `BUDGET` reads per layer; lookahead 0 reads every layer at the
    /// token's start).
    #[arg(long, value_delimiter = ',', default_value = "off")]
    prefetch: Vec<String>,
    /// What the prefetch predicts from: `b` (the logged predictor B),
    /// `transitions` (each session's decode predicted from the transition
    /// counts of all the other sessions' positions) or `token` (from the
    /// other sessions' routing per input token id).
    #[arg(long, default_value = "b")]
    predictor: String,
    /// Picks kept per row by the learned predictors.
    #[arg(long, default_value_t = 20)]
    learned_width: usize,
    /// Start every session from the loaded ranking.
    #[arg(long, default_value_t = false)]
    fresh: bool,
    /// With `--fresh`: start each session after the first from the usage
    /// the previous one's cache would persist (`LiveUsage::merged`).
    #[arg(long, default_value_t = false, requires = "fresh")]
    persist: bool,
    /// Usage half-lives to sweep, in decoded tokens (`off`: no decay); the
    /// default is the server's (`LiveUsage::HALF_LIFE_TOKENS`).
    #[arg(long, value_delimiter = ',', default_value = "500")]
    half_lives: Vec<String>,
    /// Time the decode against a modelled SSD (`cache_replay::SsdTiming`).
    #[arg(long, default_value_t = false)]
    timed: bool,
    /// Per-layer decode time with nothing missing, ms (15.1 ms warm token
    /// under the cache over 48 layers).
    #[arg(long, default_value_t = 0.315)]
    layer_ms: f64,
    /// Where in a layer its request comes (after the mixer).
    #[arg(long, default_value_t = 0.5)]
    mixer_share: f64,
    /// One expert read, ms (2.76 MB at about 6.5 GB/s).
    #[arg(long, default_value_t = 0.42)]
    read_ms: f64,
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

/// Fills each trace's predictions (lookahead 0) from the routing per input
/// token id of all the other traces (leave one session out), for its
/// decoded positions.
fn learn_token_predictions(
    stems: &[PathBuf],
    traces: &mut [Trace],
    ranking: &UsageRanking,
    top_k: usize,
    width: usize,
) -> Result<()> {
    let (layers, experts) = (ranking.layers, ranking.experts);
    ensure!((1..=experts).contains(&width), "--learned-width 1 to {experts}");
    let ids = stems
        .iter()
        .map(|stem| -> Result<Vec<u32>> {
            let path = PathBuf::from(format!("{}.u32", stem.display()));
            let bytes =
                std::fs::read(&path).with_context(|| format!("{}", path.display()))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;
    let decoded = |t: &Trace| -> Vec<usize> {
        t.requests.iter().flat_map(|r| r.decode.clone()).collect()
    };
    let wanted: Vec<u32> = traces
        .iter()
        .zip(&ids)
        .flat_map(|(t, ids)| decoded(t).into_iter().map(|p| ids[p]).collect::<Vec<_>>())
        .collect();
    let mut table = TokenRouting::new(layers, experts, wanted);
    for (t, ids) in traces.iter().zip(&ids) {
        ensure!(
            ids.len() == t.tokens,
            "{}: {} token ids for {} positions",
            t.name,
            ids.len(),
            t.tokens
        );
        table.count(ids, &t.routing, top_k, true);
    }
    for (t, ids) in traces.iter_mut().zip(&ids) {
        table.count(ids, &t.routing, top_k, false);
        t.predicted = table.predict(ids, &decoded(t), width);
        table.count(ids, &t.routing, top_k, true);
    }
    Ok(())
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
        matches!(cli.predictor.as_str(), "b" | "transitions" | "token"),
        "--predictor b, transitions or token"
    );
    let learned = cli.predictor != "b";
    let mut traces = cli
        .trace
        .iter()
        .map(|t| load_trace(t, predicted && !learned))
        .collect::<Result<Vec<_>>>()?;
    if predicted && cli.predictor == "transitions" {
        learn_predictions(&mut traces, &ranking, cli.top_k, cli.learned_width)?;
    }
    if predicted && cli.predictor == "token" {
        learn_token_predictions(
            &cli.trace,
            &mut traces,
            &ranking,
            cli.top_k,
            cli.learned_width,
        )?;
    }
    let half_lives = cli
        .half_lives
        .iter()
        .map(|h| -> Result<Option<u64>> {
            if h == "off" {
                return Ok(None);
            }
            Ok(Some(h.parse().with_context(|| format!("half-life {h}"))?))
        })
        .collect::<Result<Vec<_>>>()?;
    let timing = cli.timed.then_some(SsdTiming {
        layer_ms: cli.layer_ms,
        mixer_share: cli.mixer_share,
        read_ms: cli.read_ms,
    });
    let mut configs = Vec::new();
    for &lru_share in &cli.lru_shares {
        for &prefetch in &prefetches {
            for &half_life in &half_lives {
                configs.push(ReplayConfig {
                    slots: cli.slots,
                    lru_share,
                    adapt: true,
                    prefill_chunk: cli.prefill_chunk,
                    prefetch,
                    half_life,
                    timing,
                });
            }
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
                            let next = if cli.persist {
                                sim.merged_usage()
                            } else {
                                ranking.clone()
                            };
                            sim = CacheSim::new(&next, cli.top_k, config)?;
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
        match (cli.fresh, cli.persist) {
            (false, _) => "one cache across sessions",
            (true, false) => "fresh per session from the loaded ranking",
            (true, true) => "fresh per session from the previous session's usage",
        }
    );
    println!(
        "lru share | pinned | prefetch | half-life | decode miss/tok | layers missed/tok | prefetch reads/tok | useful/tok | stall ms/tok | late/tok | recent 64/1024 | prefill miss/tok | promotions | full"
    );
    let mut json = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        let Replayed { pinned, per, misses, useful } = result?;
        let mut all = ReplayCounts::default();
        for c in &per {
            all.plus(c);
        }
        println!(
            "{:9.2} | {:6} | {:8} | {:9} | {:15.2} | {:17.2} | {:18.2} | {:10.2} | {:12.3} | {:8.2} | {:6.2} {:6.2} | {:16.3} | {:10} | {}",
            config.lru_share,
            pinned,
            label(config.prefetch),
            config.half_life.map_or("off".into(), |h| h.to_string()),
            rate(all.decode_misses, all.decode_tokens),
            rate(all.decode_layers_missed, all.decode_tokens),
            rate(all.prefetch_reads, all.decode_tokens),
            rate(all.prefetch_useful, all.decode_tokens),
            rate(all.decode_stall_us, all.decode_tokens) / 1000.0,
            rate(all.prefetch_late, all.decode_tokens),
            rate(all.decode_misses_recent_64, all.decode_tokens),
            rate(all.decode_misses_recent_1024, all.decode_tokens),
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
                "prefetch_cancelled": c.prefetch_cancelled,
                "decode_misses_recent_64": c.decode_misses_recent_64,
                "decode_misses_recent_1024": c.decode_misses_recent_1024,
                "decode_stall_us": c.decode_stall_us, "prefetch_late": c.prefetch_late,
            })
        };
        json.push(serde_json::json!({
            "lru_share": config.lru_share,
            "pinned": pinned,
            "decode_misses_by_layer": misses,
            "prefetch_useful_by_layer": useful,
            "prefetch": label(config.prefetch),
            "half_life": config.half_life,
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
            "persist": cli.persist,
            "timing": timing.map(|t| serde_json::json!({
                "layer_ms": t.layer_ms, "mixer_share": t.mixer_share, "read_ms": t.read_ms,
            })),
            "predictor": cli.predictor,
            "results": json,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
    }
    Ok(())
}
