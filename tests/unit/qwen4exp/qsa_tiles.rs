//! The tiled sparse-attention route against the 4-layer checkpoint
//! (`LILY_MODEL_DIR_FLASH`): agreement with the split route, and the
//! measurement that sizes its gain (how much a tile's union exceeds one
//! query's selection).

use std::path::Path;

use crate::engine::{Draw, LanguageModel, LoadOptions};
use crate::kernels::qsa::{QSA_TILE_BQ, SparseAttnRoute};
use crate::metal::MetalContext;
use crate::tokenizer::Tokenizer;

use super::*;

fn model_dir() -> Option<String> {
    std::env::var("LILY_MODEL_DIR_FLASH").ok()
}

fn set_route(s: &mut Scratch, route: SparseAttnRoute) {
    s.prefill.as_mut().expect("prefill scratch").qsa.route = route;
}

/// Prefills `prompt` under `route` and returns the last row's logits.
fn prefill_logits(
    ctx: &MetalContext,
    model: &Qwen4ExpModel,
    prompt: &[u32],
    route: SparseAttnRoute,
) -> Vec<f32> {
    let capacity = prompt.len() + 64;
    let mut s = model.new_scratch_with_capacity(ctx, capacity).expect("scratch");
    let mut state = model.new_state(ctx, capacity).expect("state");
    model.ensure_prefill_scratch(ctx, &mut s, prompt.len()).expect("prefill scratch");
    set_route(&mut s, route);
    let greedy = SamplingParams::greedy();
    LanguageModel::prefill(
        model,
        ctx,
        &mut state,
        &mut s,
        prompt,
        Some(Draw { params: &greedy, step: 0 }),
    )
    .expect("prefill");
    s.logits.to_f32().expect("logits")
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, &x)| if x > v[best] { i } else { best })
}

/// Every tile variant reproduces the split route's prefill: same top token
/// and logits within the attention path's bf16 rounding, on a prompt that
/// crosses the dense limit inside its first chunk and on one with whole
/// chunks past it.
#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH"]
fn tiled_prefill_matches_split_route() {
    let Some(dir) = model_dir() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    for prompt_len in [3000usize, 9000] {
        let prompt: Vec<u32> =
            (0..prompt_len).map(|i| 1000 + (i * 37 % 5000) as u32).collect();
        let reference = prefill_logits(&ctx, &model, &prompt, SparseAttnRoute::Split);
        let top = argmax(&reference);
        for heads_per_pass in [1usize, 2, 4] {
            let got = prefill_logits(
                &ctx,
                &model,
                &prompt,
                SparseAttnRoute::Tiled { heads_per_pass },
            );
            let max_abs = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = reference.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            eprintln!(
                "{prompt_len} tokens, {heads_per_pass} heads per pass: top {} vs {top}, max |dlogit| {max_abs:.4} (logit scale {scale:.2})",
                argmax(&got)
            );
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{prompt_len} tokens, hpp {heads_per_pass}: non-finite logits"
            );
            assert!(
                max_abs <= 0.05 * scale,
                "{prompt_len} tokens, hpp {heads_per_pass}: logits differ by {max_abs} (scale {scale})"
            );
            assert_eq!(
                argmax(&got),
                top,
                "{prompt_len} tokens, hpp {heads_per_pass}: top token differs"
            );
        }
    }
}

/// The text the overlap measurement prefills: `LILY_OVERLAP_TEXT` (a file),
/// else this repository's own documentation.
fn measurement_text() -> String {
    if let Ok(path) = std::env::var("LILY_OVERLAP_TEXT") {
        return std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {path}: {e}"));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<_> = std::fs::read_dir(root.join("docs"))
        .expect("docs dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    let mut text = String::new();
    for p in
        [root.join("README.md"), root.join("CONTRIBUTING.md")].into_iter().chain(files)
    {
        if let Ok(t) = std::fs::read_to_string(&p) {
            text.push_str(&t);
            text.push('\n');
        }
    }
    text
}

/// Measures, on real text, how many blocks a 16-query tile's union holds
/// against the 512 one query selects: the work multiplier of the tile
/// kernel, and with the 16x bound of disjoint selections the overlap that
/// the § 3.3 upside assumes. Prints per prompt length; run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "measurement; requires LILY_MODEL_DIR_FLASH"]
fn measure_tile_union_overlap() {
    let Some(dir) = model_dir() else { return };
    let ctx = MetalContext::new().expect("metal context");
    let model = <Qwen4ExpModel as LanguageModel>::load(
        &ctx,
        Path::new(&dir),
        &LoadOptions::default(),
    )
    .expect("load");
    let tokenizer = Tokenizer::from_model_dir(Path::new(&dir)).expect("tokenizer");
    let tokens = tokenizer.encode(&measurement_text()).expect("encode");
    eprintln!("measurement text: {} tokens", tokens.len());
    let k_max = model.config.indexer.block_topk();
    for want in [8192usize, 32768] {
        let n = want.min(tokens.len());
        if n < model.config.indexer.dense_limit() + QSA_TILE_BQ {
            eprintln!("{n} tokens: too short to cross the dense limit, skipped");
            continue;
        }
        let prompt = &tokens[..n];
        let capacity = n + 64;
        let mut s = model.new_scratch_with_capacity(&ctx, capacity).expect("scratch");
        let mut state = model.new_state(&ctx, capacity).expect("state");
        model.ensure_prefill_scratch(&ctx, &mut s, n).expect("prefill scratch");
        set_route(&mut s, SparseAttnRoute::Tiled { heads_per_pass: 1 });
        s.prefill.as_ref().expect("prefill scratch").qsa.tiles.stats.zero_fill();
        LanguageModel::prefill(&model, &ctx, &mut state, &mut s, prompt, None)
            .expect("prefill");
        let (blocks, tiles) = s
            .prefill
            .as_ref()
            .expect("prefill scratch")
            .qsa
            .tiles
            .union_stats()
            .expect("stats");
        assert!(tiles > 0, "no tiles were built: did the tiled route run?");
        let mean = blocks as f64 / tiles as f64;
        eprintln!(
            "{n} tokens{}: {tiles} tiles, mean union {mean:.1} blocks per tile = {:.2}x one query's {k_max} (disjoint selections would give {}x); per-query work in the tile kernel is {:.2}x the split kernel's",
            if n < want { " (text exhausted)" } else { "" },
            mean / k_max as f64,
            QSA_TILE_BQ,
            mean / k_max as f64,
        );
    }
}
