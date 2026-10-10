//! Logs which experts every position of every MoE layer routes to, for
//! recorded token sequences: the input of `lily-cache-replay`.
//!
//! Each `--tokens` file is a sequence of little-endian `u32` token ids (a
//! whole session as the server's disk tier stores it, chat template and
//! generated turns included). The sequence is prefilled once, resident,
//! and the routing log is written to `--out-dir` as `<stem>.routing.u32`:
//! `[layers][tokens][top_k]` little-endian `u32`. Prefill routes a position
//! as the decode step would up to rounding (near-ties can pick another
//! expert), which is close enough for cache statistics.
//!
//! With `--predict`, the cross-layer prediction log is enabled as well
//! (`Qwen4ExpModel::enable_expert_prediction`) and predictor B's picks for
//! lookaheads 1 and 2 go to `<stem>.pred.u16`: `[2][layers][tokens][width]`
//! little-endian `u16` expert ids, each row ordered by router score, best
//! first (`route_predict::top_by_score`), `width` twice the top-k. Entry
//! `(d - 1, l, p)` is what B predicts for layer `l` at position `p` from
//! layer `l - d`; rows with `l < d` are zero. Predicted in the prefill
//! pass, with the prefill router's bf16 logits where decode's are f32.
//!
//! With `--hidden`, every position's final hidden state (the LM head's
//! input, `Qwen4ExpModel::enable_hidden_log`) goes to `<stem>.hidden.bf16`:
//! `[tokens][hidden]` little-endian bf16, the input of a whole-token expert
//! predictor trained offline.
//!
//! With `--cuts`, every position's read after each listed layer `k` (layer
//! `k + 1`'s MoE read-in over the streams right after layer `k`,
//! `Qwen4ExpModel::enable_trunk_log`) goes to `<stem>.trunk.bf16`:
//! `[cuts][tokens][hidden]` little-endian bf16, cuts ascending. With
//! `--router-top W`, every layer's own top `W` router picks go to
//! `<stem>.router.u16` (`[layers][tokens][W]` little-endian `u16` ids, best
//! first) and their scores to `<stem>.router-p.bf16` (same layout, bf16,
//! renormalised over the `W` picks when the model renormalises its top-k):
//! soft targets for that predictor.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::engine::{LanguageModel, LoadOptions};
use lily::metal::MetalContext;
use lily::qwen4exp::Qwen4ExpModel;
use lily::qwen4exp::model::{EXPERT_PRED_LOOKAHEADS, EXPERT_PRED_VARIANTS};
use lily::qwen4exp::route_predict::top_by_score;

/// Predictor B on the log's variant axis (`route_predict::VARIANT_NAMES`).
const VARIANT_B: usize = 1;

#[derive(Parser)]
#[command(
    name = "lily-trace-routing",
    about = "Expert routing of recorded token sequences"
)]
struct Cli {
    #[arg(long)]
    model: PathBuf,
    /// Token files (little-endian u32 ids), one session each.
    #[arg(long, required = true)]
    tokens: Vec<PathBuf>,
    /// Where `<stem>.routing.u32` goes for each token file.
    #[arg(long)]
    out_dir: PathBuf,
    /// Also write predictor B's lookahead 1 and 2 picks, `<stem>.pred.u16`.
    #[arg(long, default_value_t = false)]
    predict: bool,
    /// Also write every position's final hidden state, `<stem>.hidden.bf16`.
    #[arg(long, default_value_t = false)]
    hidden: bool,
    /// Also write the read after each of these layers, `<stem>.trunk.bf16`.
    #[arg(long, value_delimiter = ',')]
    cuts: Vec<usize>,
    /// Also write each layer's top W router picks and scores,
    /// `<stem>.router.u16` and `<stem>.router-p.bf16`.
    #[arg(long, default_value_t = 0)]
    router_top: usize,
}

/// Exits 75 when another lily instance holds the lock (`lily::instance`).
fn main() -> std::process::ExitCode {
    lily::instance::exit_code(run())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    std::fs::create_dir_all(&cli.out_dir)
        .with_context(|| format!("creating {}", cli.out_dir.display()))?;
    let ctx = MetalContext::new()?;
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        &cli.model,
        &LoadOptions::default(),
    )?;
    let (layers, _experts, top_k) = model.moe_shape();
    let vocab = u32::try_from(model.vocab_size())?;
    for path in &cli.tokens {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(4),
            "{}: {} bytes is not a sequence of u32 ids",
            path.display(),
            bytes.len()
        );
        let ids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        ensure!(
            ids.iter().all(|&t| t < vocab),
            "{}: a token id is outside the vocabulary of {vocab}",
            path.display()
        );
        let n = ids.len();
        let capacity = n + 1;
        let mut scratch = model.new_scratch_with_capacity(&ctx, capacity)?;
        if cli.predict {
            model.enable_expert_prediction(&ctx, &mut scratch, capacity)?;
        } else {
            model.enable_expert_log(&ctx, &mut scratch, capacity)?;
        }
        if cli.hidden {
            model.enable_hidden_log(&ctx, &mut scratch, capacity)?;
        }
        if !cli.cuts.is_empty() || cli.router_top > 0 {
            model.enable_trunk_log(
                &ctx,
                &mut scratch,
                capacity,
                &cli.cuts,
                cli.router_top,
            )?;
        }
        let mut state = model.new_state(&ctx, capacity)?;
        let started = Instant::now();
        LanguageModel::prefill(&model, &ctx, &mut state, &mut scratch, &ids, None)?;
        let secs = started.elapsed().as_secs_f64();
        let log = model.expert_log(&scratch).expect("enabled").to_u32()?;
        let mut out = Vec::with_capacity(layers * n * top_k * 4);
        for layer in 0..layers {
            let base = layer * capacity * top_k;
            for &id in &log[base..base + n * top_k] {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .context("token file without a name")?;
        let dst = cli.out_dir.join(format!("{stem}.routing.u32"));
        std::fs::write(&dst, &out)
            .with_context(|| format!("writing {}", dst.display()))?;
        if cli.hidden {
            let log = model.hidden_log(&scratch).expect("enabled");
            let row = log.shape()[1] * log.dtype().size();
            let dst = cli.out_dir.join(format!("{stem}.hidden.bf16"));
            std::fs::write(&dst, &log.contents()[..n * row])
                .with_context(|| format!("writing {}", dst.display()))?;
        }
        if !cli.cuts.is_empty() {
            let reads = model.trunk_reads(&scratch).expect("enabled");
            let row = reads.shape()[2] * reads.dtype().size();
            let mut out = Vec::with_capacity(cli.cuts.len() * n * row);
            for c in 0..cli.cuts.len() {
                let at = c * capacity * row;
                out.extend_from_slice(&reads.contents()[at..at + n * row]);
            }
            let dst = cli.out_dir.join(format!("{stem}.trunk.bf16"));
            std::fs::write(&dst, &out)
                .with_context(|| format!("writing {}", dst.display()))?;
        }
        if cli.router_top > 0 {
            let (ids, scores) = model.trunk_router(&scratch).expect("enabled");
            let (ids, scores) = (ids.to_u32()?, scores.to_f32()?);
            let w = cli.router_top;
            let mut out_ids = Vec::with_capacity(layers * n * w * 2);
            let mut out_p = Vec::with_capacity(layers * n * w * 2);
            let mut order: Vec<usize> = Vec::with_capacity(w);
            for layer in 0..layers {
                for pos in 0..n {
                    let at = (layer * capacity + pos) * w;
                    let (i, sc) = (&ids[at..at + w], &scores[at..at + w]);
                    // By score, best first; stable, as `top_by_score`.
                    order.clear();
                    order.extend(0..w);
                    order.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]));
                    for &k in &order {
                        out_ids.extend_from_slice(&u16::try_from(i[k])?.to_le_bytes());
                        out_p.extend_from_slice(&bf16_bits(sc[k]).to_le_bytes());
                    }
                }
            }
            for (ext, out) in [("router.u16", &out_ids), ("router-p.bf16", &out_p)] {
                let dst = cli.out_dir.join(format!("{stem}.{ext}"));
                std::fs::write(&dst, out)
                    .with_context(|| format!("writing {}", dst.display()))?;
            }
        }
        if cli.predict {
            let pred = model.expert_pred_log(&scratch).expect("enabled");
            let width = pred.shape()[4];
            let ids = pred.to_u32()?;
            let scores =
                model.expert_pred_scores(&scratch).expect("enabled").to_f32()?;
            ensure!(
                ids.len()
                    == EXPERT_PRED_VARIANTS
                        * EXPERT_PRED_LOOKAHEADS
                        * layers
                        * capacity
                        * width,
                "prediction log of {} entries",
                ids.len()
            );
            let mut out = Vec::with_capacity(2 * layers * n * width * 2);
            let mut row = Vec::with_capacity(width);
            for lookahead in 1..=2 {
                for layer in 0..layers {
                    let plane = (VARIANT_B * EXPERT_PRED_LOOKAHEADS + lookahead)
                        * layers
                        + layer;
                    for pos in 0..n {
                        let at = (plane * capacity + pos) * width;
                        if layer < lookahead {
                            out.resize(out.len() + width * 2, 0);
                            continue;
                        }
                        top_by_score(
                            &ids[at..at + width],
                            &scores[at..at + width],
                            width,
                            &mut row,
                        );
                        for &e in &row {
                            out.extend_from_slice(&u16::try_from(e)?.to_le_bytes());
                        }
                    }
                }
            }
            let dst = cli.out_dir.join(format!("{stem}.pred.u16"));
            std::fs::write(&dst, &out)
                .with_context(|| format!("writing {}", dst.display()))?;
        }
        eprintln!(
            "{stem}: {n} tokens in {secs:.1}s ({:.0} tok/s) -> {}",
            n as f64 / secs,
            dst.display()
        );
    }
    Ok(())
}

/// `x` as bf16 bits, rounded to nearest even (finite inputs).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
