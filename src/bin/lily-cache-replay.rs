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
//! prefetch reads `STEM.pred.u16` (`lily-trace-routing --predict`, or
//! `tools/bench/expert_predictor.py` for a token-start prefetch), or with
//! `--predictor transitions` predicts from cross-layer transition counts
//! (`cache_replay::Transitions`) learned on the other sessions given, or
//! with `--predictor token` from the routing of each input token id
//! (`cache_replay::TokenRouting`, reading `STEM.u32`), for a prefetch at
//! the token's start (`0:TOP`).

use std::path::PathBuf;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::qwen4exp::cache_replay::{
    CacheSim, Predictions, PrefetchReplay, RejectedRows, ReplayConfig, ReplayCounts,
    SharedSsd, SpecReplay, SsdModel, SsdTiming, TokenRouting, TraceRequest,
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
    /// Prefetches to sweep: `off`, or `LOOKAHEAD:TOP[:BUDGET]` (the
    /// predictor's `TOP` best for the layer `LOOKAHEAD` ahead, at most
    /// `BUDGET` reads per layer; lookahead 0 reads every layer at the
    /// token's start, and with `@CUT+CUT..` again after each layer `CUT`
    /// for the layers after it, from the cut's own plane; a trailing `r`
    /// lets each cut cancel queued reads for its layers that it does not
    /// pick).
    #[arg(long, value_delimiter = ',', default_value = "off")]
    prefetch: Vec<String>,
    /// What the prefetch predicts from: `b` (the logged predictor B, or any
    /// `STEM.pred.u16`), `transitions` (each session's decode predicted from
    /// the transition counts of all the other sessions' positions), `token`
    /// (from the other sessions' routing per input token id) or `oracle`
    /// (the routing itself, every plane: the timing's ceiling).
    #[arg(long, default_value = "b")]
    predictor: String,
    /// Replay decoding as speculative verify passes, `DRAFTS:ACCEPT`
    /// (`cache_replay::SpecReplay`, e.g. `2:0.79`); stall and rates are then
    /// per accepted token, and `layers missed/tok` counts a pass's missed
    /// layers once for all the tokens it accepts.
    #[arg(long)]
    speculate: Option<String>,
    /// Under `--speculate`: what rows of drafts that were not accepted route
    /// to, `next` (the kept token's routing, optimistic; the default) or
    /// `random`.
    #[arg(long)]
    rejected: Option<String>,
    /// Planes in each `STEM.pred.u16` (`cache_replay::Predictions`).
    #[arg(long, default_value_t = 2)]
    planes: usize,
    /// `--predictor b`: picks per row of each `STEM.pred.u16`. Default:
    /// twice `--top-k`, what `lily-trace-routing --predict` writes, and for
    /// its two planes only; other files (`tools/bench/expert_predictor.py`
    /// writes 40) and any other `--planes` need it given, so a wrong layout
    /// is refused rather than read.
    #[arg(long)]
    pred_width: Option<usize>,
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
    /// `--timed`: the SSD model (`cache_replay::SsdModel`), `serial` (the
    /// default: one read at a time in `--read-ms`, demand first) or
    /// `shared` (each read a latency, then a transfer sharing the bandwidth
    /// with every read in flight; fitted to `lily-ssd-probe`).
    #[arg(long)]
    ssd: Option<String>,
    /// `--timed --ssd serial`: one expert read, ms (default 0.42: 2.76 MB
    /// at about 6.5 GB/s).
    #[arg(long)]
    read_ms: Option<f64>,
    /// `--timed --ssd shared`: a read's latency before its transfer, ms
    /// (default `SharedSsd::PROBED`'s).
    #[arg(long)]
    latency_ms: Option<f64>,
    /// `--timed --ssd shared`: one expert's transfer at full bandwidth, ms
    /// (default `SharedSsd::PROBED`'s).
    #[arg(long)]
    transfer_ms: Option<f64>,
    /// `--timed --ssd shared`: prefetches in flight at most (more wait
    /// queued; default `SharedSsd::PROBED`'s 4, which all but saturated the
    /// probed drive's bandwidth).
    #[arg(long)]
    prefetch_depth: Option<usize>,
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Write each session's decode misses to `DIR/<name>.miss.u8`
    /// (`CacheSim::take_miss_log`); one configuration only.
    #[arg(long)]
    miss_log: Option<PathBuf>,
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
    let (head, cuts, replace) = match spec.split_once('@') {
        Some((head, cuts)) => {
            let (cuts, replace) = match cuts.strip_suffix('r') {
                Some(c) => (c, true),
                None => (cuts, false),
            };
            let mut mask = 0u64;
            for c in cuts.split('+') {
                let k: u32 = c.parse().with_context(|| format!("prefetch {spec}"))?;
                ensure!(k < 63, "prefetch {spec}: cut {k} out of range");
                mask |= 1 << k;
            }
            (head, mask, replace)
        }
        None => (spec, 0, false),
    };
    let v: Vec<usize> = head
        .split(':')
        .map(|x| x.parse().with_context(|| format!("prefetch {spec}")))
        .collect::<Result<_>>()?;
    ensure!(
        (2..=3).contains(&v.len()),
        "prefetch {spec}: LOOKAHEAD:TOP[:BUDGET][@CUT]"
    );
    Ok(Some(PrefetchReplay {
        lookahead: v[0],
        top: v[1],
        budget: v.get(2).copied(),
        cuts,
        replace,
    }))
}

fn label(p: Option<PrefetchReplay>) -> String {
    match p {
        None => "off".into(),
        Some(p) => {
            let mut s = format!("{}:{}", p.lookahead, p.top);
            if let Some(b) = p.budget {
                s += &format!(":{b}");
            }
            if p.cuts != 0 {
                let cuts: Vec<String> = (0..64)
                    .filter(|k| p.cuts >> k & 1 == 1)
                    .map(|k| k.to_string())
                    .collect();
                s += &format!("@{}", cuts.join("+"));
                if p.replace {
                    s += "r";
                }
            }
            s
        }
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

/// The routing itself as predictions, in every plane, `top_k` wide: what a
/// perfect predictor would read, under the same timing.
/// The routing's layout, `[layers][tokens][top_k]`, is one such plane.
fn oracle_predictions(routing: &[u32], planes: usize) -> Vec<u16> {
    let plane = routing.iter().map(|&e| e as u16);
    plane.cycle().take(planes * routing.len()).collect()
}

/// `--ssd serial`'s default read, ms: 2.76 MB at about 6.5 GB/s.
const SERIAL_READ_MS: f64 = 0.42;

/// The row width of `--predictor b`'s `STEM.pred.u16`: `--pred-width`, or
/// by default `lily-trace-routing --predict`'s, twice the top-k over its
/// two planes. With another plane count the default would be a guess, and
/// a guess that divides the file would misread it: refused.
fn logged_width(
    given: Option<usize>,
    planes: usize,
    top_k: usize,
    experts: usize,
) -> Result<usize> {
    let width = match given {
        Some(w) => w,
        None => {
            ensure!(
                planes == 2,
                "--planes {planes} needs --pred-width (the default, {}, is \
                 lily-trace-routing --predict's, over two planes)",
                2 * top_k
            );
            2 * top_k
        }
    };
    ensure!((1..=experts).contains(&width), "--pred-width {width}: 1 to {experts}");
    Ok(width)
}

/// Refuses a logged `STEM.pred.u16` that does not hold exactly `planes` x
/// layers x tokens x `width` ids, the layout it is read as.
fn check_predictions(
    t: &Trace,
    planes: usize,
    layers: usize,
    width: usize,
) -> Result<()> {
    let expected = planes * layers * t.tokens * width;
    ensure!(
        t.predicted.len() == expected,
        "{}.pred.u16 holds {} ids, not {planes} planes x {layers} layers x {} tokens x \
         {width} (--planes, --pred-width)",
        t.name,
        t.predicted.len(),
        t.tokens
    );
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
        matches!(cli.predictor.as_str(), "b" | "transitions" | "token" | "oracle"),
        "--predictor b, transitions, token or oracle"
    );
    let learned = cli.predictor != "b";
    // The learned predictors build two planes (token start, lookahead 1 and
    // 2) and fill rows they cannot predict with expert 0, so a cut would
    // read those as picks; the logged and oracle predictions take `--planes`.
    let learned_tables = matches!(cli.predictor.as_str(), "transitions" | "token");
    ensure!(cli.planes >= 1, "--planes 1 or more");
    ensure!(
        !learned_tables || prefetches.iter().flatten().all(|p| p.cuts == 0),
        "cuts need logged or oracle predictions, not --predictor {}",
        cli.predictor
    );
    let planes = if learned_tables { 2 } else { cli.planes };
    ensure!(
        cli.pred_width.is_none() || (predicted && !learned),
        "--pred-width is the layout of --predictor b's STEM.pred.u16, read for a prefetch"
    );
    // Picks per prediction row: the logged file's, the oracle's top-k, or
    // what the learned predictors keep.
    let width = match cli.predictor.as_str() {
        _ if !predicted => 0,
        "b" => logged_width(cli.pred_width, planes, cli.top_k, ranking.experts)?,
        "oracle" => cli.top_k,
        _ => cli.learned_width,
    };
    let names: std::collections::HashSet<_> =
        cli.trace.iter().filter_map(|t| t.file_name()).collect();
    ensure!(
        cli.miss_log.is_none() || names.len() == cli.trace.len(),
        "--miss-log needs distinct trace file names"
    );
    let mut traces = cli
        .trace
        .iter()
        .map(|t| load_trace(t, predicted && !learned))
        .collect::<Result<Vec<_>>>()?;
    if predicted && !learned {
        for t in &traces {
            check_predictions(t, planes, ranking.layers, width)?;
        }
    }
    if predicted && cli.predictor == "transitions" {
        learn_predictions(&mut traces, &ranking, cli.top_k, cli.learned_width)?;
    }
    if predicted && cli.predictor == "oracle" {
        for t in &mut traces {
            t.predicted = oracle_predictions(&t.routing, cli.planes);
        }
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
    // Flags that would change nothing are refused rather than ignored.
    let shared_flags = cli.latency_ms.is_some()
        || cli.transfer_ms.is_some()
        || cli.prefetch_depth.is_some();
    ensure!(
        cli.timed || (cli.ssd.is_none() && cli.read_ms.is_none() && !shared_flags),
        "--ssd, --read-ms, --latency-ms, --transfer-ms and --prefetch-depth model the \
         SSD of a timed replay: they need --timed"
    );
    let ssd = match cli.ssd.as_deref().unwrap_or("serial") {
        "serial" => {
            ensure!(
                !shared_flags,
                "--latency-ms, --transfer-ms and --prefetch-depth are --ssd shared's"
            );
            SsdModel::Serial
        }
        "shared" => {
            ensure!(cli.read_ms.is_none(), "--read-ms is --ssd serial's");
            let p = SharedSsd::PROBED;
            let m = SharedSsd {
                latency_ms: cli.latency_ms.unwrap_or(p.latency_ms),
                transfer_ms: cli.transfer_ms.unwrap_or(p.transfer_ms),
                prefetch_depth: cli.prefetch_depth.unwrap_or(p.prefetch_depth),
            };
            ensure!(
                m.latency_ms >= 0.0 && m.transfer_ms > 0.0 && m.prefetch_depth >= 1,
                "--ssd shared: --latency-ms 0 or more, --transfer-ms above 0, \
                 --prefetch-depth 1 or more"
            );
            SsdModel::Shared(m)
        }
        other => anyhow::bail!("--ssd {other}: serial or shared"),
    };
    let timing = cli.timed.then_some(SsdTiming {
        layer_ms: cli.layer_ms,
        mixer_share: cli.mixer_share,
        read_ms: cli.read_ms.unwrap_or(SERIAL_READ_MS),
    });
    ensure!(
        cli.rejected.is_none() || cli.speculate.is_some(),
        "--rejected needs --speculate"
    );
    let speculation = match &cli.speculate {
        None => None,
        Some(spec) => {
            let (d, a) = spec
                .split_once(':')
                .with_context(|| format!("--speculate {spec}: DRAFTS:ACCEPT"))?;
            let accept: f64 =
                a.parse().with_context(|| format!("--speculate {spec}"))?;
            ensure!(
                (0.0..=1.0).contains(&accept),
                "--speculate {spec}: ACCEPT in 0..=1"
            );
            Some(SpecReplay {
                drafts: d.parse().with_context(|| format!("--speculate {spec}"))?,
                accept,
                rejected: match cli.rejected.as_deref().unwrap_or("next") {
                    "next" => RejectedRows::Next,
                    "random" => RejectedRows::Random,
                    other => anyhow::bail!("--rejected {other}: next or random"),
                },
                seed: 1,
            })
        }
    };
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
                    speculation,
                });
            }
        }
    }
    ensure!(
        cli.miss_log.is_none() || configs.len() == 1,
        "--miss-log takes one configuration, not {}",
        configs.len()
    );
    if let Some(dir) = &cli.miss_log {
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
    }
    let miss_dir = cli.miss_log.as_deref();
    let results: Vec<Result<Replayed>> = std::thread::scope(|scope| {
        let handles: Vec<_> = configs
            .iter()
            .map(|&config| {
                let (ranking, traces) = (&ranking, &traces);
                scope.spawn(move || -> Result<Replayed> {
                    // A fresh cache for session `i` numbers it `i`, so it
                    // draws what one cache across sessions would.
                    let new_sim = |r: &UsageRanking, i: usize| -> Result<CacheSim> {
                        let sim = CacheSim::new(r, cli.top_k, config)?
                            .with_ssd(ssd)
                            .starting_at_session(i as u64);
                        Ok(if miss_dir.is_some() { sim.log_misses() } else { sim })
                    };
                    let mut sim = new_sim(ranking, 0)?;
                    let pinned = sim.pinned();
                    let mut per = Vec::with_capacity(traces.len());
                    let layers = ranking.layers;
                    let (mut misses, mut useful) = (vec![0; layers], vec![0; layers]);
                    let mut take = |sim: &CacheSim| {
                        let (m, u) = sim.by_layer();
                        misses.iter_mut().zip(m).for_each(|(a, b)| *a += b);
                        useful.iter_mut().zip(u).for_each(|(a, b)| *a += b);
                    };
                    for (i, t) in traces.iter().enumerate() {
                        if cli.fresh {
                            take(&sim);
                            let next = if cli.persist {
                                sim.merged_usage()
                            } else {
                                ranking.clone()
                            };
                            sim = new_sim(&next, i)?;
                        }
                        let mut c = ReplayCounts::default();
                        let pred = (!t.predicted.is_empty()).then(|| Predictions {
                            ids: &t.predicted,
                            planes,
                            width,
                        });
                        sim.session(
                            &t.routing,
                            pred.as_ref(),
                            t.tokens,
                            cli.top_k,
                            &t.requests,
                            &mut c,
                        )?;
                        if let (Some(dir), Some(log)) = (miss_dir, sim.take_miss_log())
                        {
                            let dst = dir.join(format!("{}.miss.u8", t.name));
                            std::fs::write(&dst, log)
                                .with_context(|| format!("{}", dst.display()))?;
                        }
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
    if let Some(t) = timing {
        match ssd {
            SsdModel::Serial => println!(
                "timed: layer {} ms, mixer share {}, serial SSD, read {} ms",
                t.layer_ms, t.mixer_share, t.read_ms
            ),
            SsdModel::Shared(s) => println!(
                "timed: layer {} ms, mixer share {}, shared SSD, latency {} ms, transfer {} ms, prefetch depth {}",
                t.layer_ms,
                t.mixer_share,
                s.latency_ms,
                s.transfer_ms,
                s.prefetch_depth
            ),
        }
    }
    println!(
        "lru share | pinned | prefetch | half-life | decode miss/tok | layers missed/tok | prefetch reads/tok | useful/tok | stall ms/tok | late/tok | recent 64/1024 | prefill miss/tok | promotions | full | passes/tok"
    );
    let mut json = Vec::new();
    for (config, result) in configs.iter().zip(results) {
        let Replayed { pinned, per, misses, useful } = result?;
        let mut all = ReplayCounts::default();
        for c in &per {
            all.plus(c);
        }
        println!(
            "{:9.2} | {:6} | {:8} | {:9} | {:15.2} | {:17.2} | {:18.2} | {:10.2} | {:12.3} | {:8.2} | {:6.2} {:6.2} | {:16.3} | {:10} | {:4} | {:.3}",
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
            all.full,
            rate(all.decode_passes, all.decode_tokens)
        );
        let counts = |c: &ReplayCounts| {
            serde_json::json!({
                "prefill_tokens": c.prefill_tokens, "prefill_lookups": c.prefill_lookups,
                "prefill_misses": c.prefill_misses, "decode_tokens": c.decode_tokens,
                "decode_passes": c.decode_passes,
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
            "timing": timing.map(|t| {
                let mut j = serde_json::json!({
                    "layer_ms": t.layer_ms, "mixer_share": t.mixer_share,
                });
                match ssd {
                    SsdModel::Serial => {
                        j["ssd"] = "serial".into();
                        j["read_ms"] = t.read_ms.into();
                    }
                    SsdModel::Shared(s) => {
                        j["ssd"] = "shared".into();
                        j["latency_ms"] = s.latency_ms.into();
                        j["transfer_ms"] = s.transfer_ms.into();
                        j["prefetch_depth"] = s.prefetch_depth.into();
                    }
                }
                j
            }),
            "predictor": cli.predictor,
            "results": json,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A session of `tokens` positions over `layers` layers whose
    /// `STEM.pred.u16` holds `ids` ids.
    fn trace(layers: usize, tokens: usize, ids: usize) -> Trace {
        Trace {
            name: "s".into(),
            routing: vec![0; layers * tokens * 10],
            predicted: vec![0; ids],
            tokens,
            requests: Vec::new(),
        }
    }

    #[test]
    fn the_logged_layout_is_explicit() {
        // lily-trace-routing --predict's output: two planes of twice top-10.
        let width = logged_width(None, 2, 10, 512).expect("default");
        assert_eq!(width, 20);
        let t = trace(48, 7, 2 * 48 * 7 * 20);
        check_predictions(&t, 2, 48, width).expect("its own layout");
        // Read as four planes, the file divides into rows of 10: inferring
        // the width would have misread it. The default width is only for
        // two planes, and four planes of 20 do not match the file.
        assert!(logged_width(None, 4, 10, 512).is_err());
        let w = logged_width(Some(20), 4, 10, 512).expect("given");
        assert!(check_predictions(&t, 4, 48, w).is_err());
        assert!(check_predictions(&t, 3, 48, w).is_err());
        // Another producer's width is given, and checked against the file.
        let t = trace(48, 7, 5 * 48 * 7 * 24);
        check_predictions(&t, 5, 48, 24).expect("five planes of 24");
        assert!(check_predictions(&t, 6, 48, 24).is_err());
        assert!(check_predictions(&t, 5, 48, 20).is_err());
        // Wider than the experts there are, or empty: refused.
        assert!(logged_width(Some(513), 2, 10, 512).is_err());
        assert!(logged_width(Some(0), 2, 10, 512).is_err());
    }
}
