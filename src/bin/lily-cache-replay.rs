//! Replays recorded sessions' expert routing through the expert cache's
//! policy (`lily::qwen4exp::cache_replay`) for a sweep of LRU shares; host
//! only, no model.
//!
//! Each `--trace STEM` names `STEM.routing.u32` (from `lily-trace-routing`)
//! and `STEM.json`: `{"tokens": N, "requests": [[p0, p1, p2], ...]}`, a
//! request prefilling `p0..p1` and generating `p1..p2`. Sessions replay in
//! the order given through one cache, as one server would serve them,
//! unless `--fresh` starts every session from the loaded ranking.

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::qwen4exp::cache_replay::{
    CacheSim, ReplayConfig, ReplayCounts, TraceRequest,
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
    /// Start every session from the loaded ranking.
    #[arg(long, default_value_t = false)]
    fresh: bool,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

struct Trace {
    name: String,
    routing: Vec<u32>,
    tokens: usize,
    requests: Vec<TraceRequest>,
}

fn load_trace(stem: &std::path::Path) -> Result<Trace> {
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
    let name = stem.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
    Ok(Trace { name, routing, tokens, requests })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ranking = UsageRanking::load(&cli.usage)?;
    let traces = cli.trace.iter().map(|t| load_trace(t)).collect::<Result<Vec<_>>>()?;
    let configs: Vec<ReplayConfig> = cli
        .lru_shares
        .iter()
        .map(|&lru_share| ReplayConfig {
            slots: cli.slots,
            lru_share,
            adapt: true,
            prefill_chunk: cli.prefill_chunk,
        })
        .collect();
    let results: Vec<Result<(usize, Vec<ReplayCounts>)>> =
        std::thread::scope(|scope| {
            let handles: Vec<_> = configs
                .iter()
                .map(|&config| {
                    let (ranking, traces) = (&ranking, &traces);
                    scope.spawn(move || -> Result<(usize, Vec<ReplayCounts>)> {
                        let mut sim = CacheSim::new(ranking, cli.top_k, config)?;
                        let pinned = sim.pinned();
                        let mut per = Vec::with_capacity(traces.len());
                        for t in traces {
                            if cli.fresh {
                                sim = CacheSim::new(ranking, cli.top_k, config)?;
                            }
                            let mut c = ReplayCounts::default();
                            sim.session(
                                &t.routing,
                                t.tokens,
                                cli.top_k,
                                &t.requests,
                                &mut c,
                            )?;
                            per.push(c);
                        }
                        Ok((pinned, per))
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
        "lru share | pinned | decode miss/tok | layers missed/tok | prefill miss/tok | promotions | full"
    );
    let mut json = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        let (pinned, per) = result?;
        let mut all = ReplayCounts::default();
        for c in &per {
            all.plus(c);
        }
        println!(
            "{:9.2} | {:6} | {:15.2} | {:17.2} | {:16.3} | {:10} | {}",
            config.lru_share,
            pinned,
            rate(all.decode_misses, all.decode_tokens),
            rate(all.decode_layers_missed, all.decode_tokens),
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
                "promotions": c.promotions,
            })
        };
        json.push(serde_json::json!({
            "lru_share": config.lru_share,
            "pinned": pinned,
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
            "results": json,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
    }
    Ok(())
}
