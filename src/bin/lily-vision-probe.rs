//! Runs lily's image preprocessing and the vision tower alone and writes
//! their outputs in the golden format, for `tools/reference/compare_vision.py`
//! (docs/architecture.md, "How the tower was verified", comparisons 1 and 2) and for timing them.
//!
//! ```sh
//! # Comparison 2 on the reference's own pixel_values:
//! cargo run --release --bin lily-vision-probe -- \
//!     --model <dir>-l4 \
//!     --golden tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json \
//!     --out /tmp/lily_tower_333x777.json
//! .venv/bin/python tools/reference/compare_vision.py /tmp/lily_tower_333x777.json \
//!     tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json
//!
//! # Comparison 1 alone (no GPU): lily's preprocessing of an image file.
//! cargo run --release --bin lily-vision-probe -- \
//!     --image tools/reference/images/333x777.png \
//!     --preprocess-golden tools/reference/goldens/hf_vision_preprocess_333x777_cap2097152.json \
//!     --out /tmp/lily_pre_333x777.json
//!
//! # Comparisons 1 and 2 end to end: the tower on lily's own pixel rows.
//! cargo run --release --bin lily-vision-probe -- --model <dir>-l4 \
//!     --image tools/reference/images/333x777.png \
//!     --golden tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json \
//!     --out /tmp/lily_tower_333x777.json      # + /tmp/lily_tower_333x777.preprocess.json
//! ```
//!
//! Without `--image` the grid comes from the tower golden and the input is
//! the matching `goldens/large/preprocess_<img>_cap<max>.pixel_values.npy`
//! unless `--pixels` names another. Every record carries the golden's own
//! sample indices with the candidate's values at them, so the comparison
//! works with or without the `.npy` files.
//!
//! ```sh
//! # Comparisons 3 and 4: the language model's logits with the image in the
//! # prompt (positions, the tower's rows in place of the placeholders), in
//! # the lily-probe record format for compare.py; --image is preprocessed
//! # with the golden's cap. Without --image the golden must be the text-only
//! # control.
//! cargo run --release --bin lily-vision-probe -- --model <dir>-l4 \
//!     --forward tools/reference/goldens/hf_l4_vision_333x777_dequant.json \
//!     --image tools/reference/images/333x777.png --out /tmp/lily_l4_vision_333x777.json
//! .venv/bin/python tools/reference/compare.py /tmp/lily_l4_vision_333x777.json \
//!     tools/reference/goldens/hf_l4_vision_333x777_dequant.json
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result, bail, ensure};
use clap::Parser;
use lily::engine::VisionMode;
use lily::generate::Generator;
use lily::metal::MetalContext;
use lily::metal::profile;
use lily::npy;
use lily::qwen4exp::image::{self, ImageLimits, PixelValues};
use lily::qwen4exp::probe::{ProbeStep, forward_probe};
use lily::qwen4exp::vision::VisionTower;
use lily::qwen4exp::{
    ImageEmbeds, ImageSpan, NgramStorage, Qwen4ExpModel, VisionInput,
    positions_for_prompt,
};
use lily::sha256::{hex, sha256};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "lily-vision-probe", about = "Vision preprocessing and tower probe")]
struct Cli {
    /// Checkpoint directory carrying the tower (the language model is not
    /// loaded). Without it only the preprocessing runs.
    #[arg(long)]
    model: Option<PathBuf>,
    /// A `hf_vision_tower_<img>_cap<max>.json` golden: grid and sample
    /// indices for the tower record. Required with `--model` unless
    /// `--forward` is given.
    #[arg(long)]
    golden: Option<PathBuf>,
    /// A `hf_l4_vision_*_dequant.json` forward golden: run its prompt
    /// through the language model (with `--image` preprocessed at the
    /// golden's cap and fed through the tower when the golden has an image,
    /// or text only) and write a `lily-probe` record for `compare.py`.
    #[arg(long, conflicts_with = "golden")]
    forward: Option<PathBuf>,
    /// An image file (PNG or JPEG) to preprocess with lily's own code; the
    /// tower then runs on those pixel rows.
    #[arg(long)]
    image: Option<PathBuf>,
    /// A `hf_vision_preprocess_<img>_cap<max>.json` golden for the
    /// preprocess record's sample indices; defaults to the one next to
    /// `--golden` for the image and cap.
    #[arg(long)]
    preprocess_golden: Option<PathBuf>,
    /// Pixel cap for `smart_resize` (VISION.md "The pixel cap").
    #[arg(long, default_value_t = image::DEFAULT_MAX_PIXELS)]
    max_pixels: usize,
    /// Pixel floor for `smart_resize`.
    #[arg(long, default_value_t = image::DEFAULT_MIN_PIXELS)]
    min_pixels: usize,
    /// The float32 `[N, 1536]` pixel_values `.npy`; defaults to the golden's
    /// matching preprocess file in `goldens/large/`. Ignored with `--image`.
    #[arg(long)]
    pixels: Option<PathBuf>,
    /// Where to write the candidate JSON; the `.npy` outputs sit next to it.
    /// With `--model` this is the tower record and the preprocess record is
    /// `<out>.preprocess.json`; without, it is the preprocess record.
    #[arg(long)]
    out: PathBuf,
    /// Where to write the preprocess record instead of the default.
    #[arg(long)]
    preprocess_out: Option<PathBuf>,
    /// Timed runs after one warm-up.
    #[arg(long, default_value_t = 3)]
    repeat: usize,
    /// Per-kernel GPU profile of the last run (one command buffer per
    /// dispatch; wall time under it is not comparable, the kernel times are).
    #[arg(long, default_value_t = false)]
    kernel_profile: bool,
    /// Run only the first N transformer blocks (diagnostics: the pre-merger
    /// output is then the residual after N blocks; the merged output is not
    /// meaningful). Default: all.
    #[arg(long)]
    blocks: Option<usize>,
}

/// The forward record (`lily-probe`'s layout plus the positions block
/// `compare.py` checks exactly, comparison 3).
#[derive(Serialize)]
struct ForwardRecord {
    model_id: &'static str,
    prompt_token_ids: Vec<u32>,
    steps: Vec<ProbeStep>,
    text: String,
    positions: Option<PositionsRecord>,
    load_seconds: f64,
    /// The full prompt's prefill.
    prefill_seconds: f64,
    decode_seconds: f64,
    image: Option<serde_json::Value>,
    golden: String,
}

#[derive(Serialize)]
struct PositionsRecord {
    image_span: ImageSpan,
    /// `[3][n]`: the temporal, height and width axis of every prompt token.
    position_ids: Vec<Vec<u32>>,
    rope_deltas: i64,
}

/// One tensor in the golden format (`vision_golden.tensor_record`).
#[derive(Serialize)]
struct TensorRecord {
    name: &'static str,
    shape: Vec<usize>,
    dtype: &'static str,
    sha256: String,
    stats: Stats,
    sample: Sample,
    npy: String,
}

#[derive(Serialize)]
struct Stats {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    abs_mean: f64,
}

#[derive(Serialize)]
struct Sample {
    seed: u64,
    indices: Vec<usize>,
    values: Vec<f32>,
    /// Where the indices came from: the golden, or an even stride when no
    /// golden was given (the comparison then needs the `.npy` files).
    source: String,
}

#[derive(Serialize)]
struct Record {
    kind: &'static str,
    golden: String,
    cap: serde_json::Value,
    input: serde_json::Value,
    run: Run,
    tokens: usize,
    merged: TensorRecord,
    pre_merger: TensorRecord,
}

#[derive(Serialize)]
struct Run {
    device: &'static str,
    dtype: &'static str,
    attention: &'static str,
    patches: usize,
    grid: [usize; 2],
    /// Transformer blocks run (the depth unless `--blocks` cut it).
    blocks: usize,
    load_seconds: f64,
    /// GPU span of each timed run, milliseconds.
    gpu_ms: Vec<f64>,
    /// Host time of each timed run (upload, encode, wait), milliseconds.
    host_ms: Vec<f64>,
    scratch_bytes: usize,
    kernel_profile_ms: Option<BTreeMap<String, f64>>,
}

/// The preprocess record (`hf_vision_reference.py preprocess`, comparison 1).
#[derive(Serialize)]
struct PreprocessRecord {
    kind: &'static str,
    golden: Option<String>,
    image: serde_json::Value,
    cap: serde_json::Value,
    resized_hw: [usize; 2],
    image_grid_thw: [[usize; 3]; 1],
    tokens: usize,
    pixel_values: TensorRecord,
    run: PreprocessRun,
}

#[derive(Serialize)]
struct PreprocessRun {
    /// Decode, resize and patchify, milliseconds, one run.
    preprocess_ms: f64,
    /// Repeated timings after the first, milliseconds.
    repeat_ms: Vec<f64>,
    header: String,
}

fn stats(data: &[f32]) -> Stats {
    let n = data.len() as f64;
    let (mut sum, mut abs, mut min, mut max) =
        (0.0f64, 0.0f64, f64::INFINITY, f64::NEG_INFINITY);
    for &v in data {
        let v = v as f64;
        sum += v;
        abs += v.abs();
        min = min.min(v);
        max = max.max(v);
    }
    let mean = sum / n;
    let var = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    Stats { mean, std: var.sqrt(), min, max, abs_mean: abs / n }
}

/// The sample indices of a golden's tensor record, or an even stride over
/// `len` elements when there is no golden.
fn sample_indices(
    golden_sample: Option<&serde_json::Value>,
    len: usize,
) -> Result<(Vec<usize>, u64, String)> {
    match golden_sample {
        Some(sample) => {
            let indices: Vec<usize> = sample["indices"]
                .as_array()
                .context("golden sample indices")?
                .iter()
                .map(|v| v.as_u64().map(|v| v as usize).context("sample index"))
                .collect::<Result<_>>()?;
            ensure!(
                indices.iter().all(|&i| i < len),
                "a golden sample index exceeds the {len} elements"
            );
            Ok((indices, sample["seed"].as_u64().unwrap_or(0), "golden".to_string()))
        }
        None => {
            let n = 4096.min(len);
            let indices = (0..n).map(|i| i * len / n).collect();
            Ok((indices, 0, "even stride, no golden: compare on the .npy".to_string()))
        }
    }
}

fn record(
    name: &'static str,
    shape: &[usize],
    data: &[f32],
    golden_sample: Option<&serde_json::Value>,
    npy_path: &Path,
) -> Result<TensorRecord> {
    let bytes = npy::to_bytes_f32(shape, data)?;
    std::fs::write(npy_path, &bytes)
        .with_context(|| format!("writing {}", npy_path.display()))?;
    let (indices, seed, source) = sample_indices(golden_sample, data.len())?;
    Ok(TensorRecord {
        name,
        shape: shape.to_vec(),
        dtype: "float32",
        sha256: hex(&sha256(&f32_bytes(data))),
        stats: stats(data),
        sample: Sample {
            seed,
            values: indices.iter().map(|&i| data[i]).collect(),
            indices,
            source,
        },
        npy: npy_path.file_name().context("npy name")?.to_string_lossy().into_owned(),
    })
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}

/// Runs lily's preprocessing over `--image`, writes its record and returns
/// the pixel rows for the tower.
fn preprocess(cli: &Cli, image_path: &Path, out: &Path) -> Result<PixelValues> {
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("reading {}", image_path.display()))?;
    let limits = ImageLimits {
        max_pixels: cli.max_pixels,
        min_pixels: cli.min_pixels,
        ..ImageLimits::default()
    };
    let header = image::read_header(&bytes)?;
    let started = Instant::now();
    let pv = image::preprocess(&bytes, &limits)?;
    let preprocess_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut repeat_ms = Vec::new();
    for _ in 0..cli.repeat {
        let started = Instant::now();
        let again = image::preprocess(&bytes, &limits)?;
        repeat_ms.push(started.elapsed().as_secs_f64() * 1e3);
        ensure!(again == pv, "preprocessing is not deterministic");
    }
    eprintln!(
        "{}: {} {} x {} -> {} x {}, grid ({}, {}), {} patches, {} tokens; {preprocess_ms:.1} ms (repeats {:?})",
        image_path.display(),
        header.format,
        header.width,
        header.height,
        pv.resized.0,
        pv.resized.1,
        pv.grid_h,
        pv.grid_w,
        pv.patches(),
        pv.tokens(),
        repeat_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
    );

    // The preprocess golden: given, or derived from the tower golden's
    // directory, the image stem and the cap.
    let stem =
        image_path.file_stem().context("image name")?.to_string_lossy().into_owned();
    let golden_path = cli.preprocess_golden.clone().or_else(|| {
        cli.golden.as_ref().map(|g| {
            g.parent()
                .unwrap_or(Path::new("."))
                .join(format!("hf_vision_preprocess_{stem}_cap{}.json", cli.max_pixels))
        })
    });
    let golden = match &golden_path {
        Some(p) if p.exists() => {
            let g = read_json(p)?;
            ensure!(
                g["kind"] == "preprocess",
                "{} is not a preprocess golden",
                p.display()
            );
            Some(g)
        }
        Some(p) => {
            eprintln!(
                "note: no preprocess golden at {}; sampling at an even stride",
                p.display()
            );
            None
        }
        None => None,
    };
    let out_dir = out.parent().map(Path::to_path_buf).unwrap_or_default();
    let base = out.file_stem().context("out name")?.to_string_lossy().into_owned();
    let record = PreprocessRecord {
        kind: "preprocess",
        golden: golden_path
            .as_ref()
            .filter(|p| p.exists())
            .map(|p| p.display().to_string()),
        image: serde_json::json!({
            "file": image_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "sha256": hex(&sha256(&bytes)),
            "width": header.width,
            "height": header.height,
            "format": header.format.to_string(),
            "channels": header.channels,
            "bit_depth": header.bit_depth,
        }),
        cap: serde_json::json!({
            "min_pixels": cli.min_pixels,
            "max_pixels": cli.max_pixels,
            "as_size": {"shortest_edge": cli.min_pixels, "longest_edge": cli.max_pixels},
        }),
        resized_hw: [pv.resized.1, pv.resized.0],
        image_grid_thw: [[1, pv.grid_h, pv.grid_w]],
        tokens: pv.tokens(),
        pixel_values: record(
            "pixel_values",
            &[pv.patches(), image::PATCH_DIM],
            &pv.data,
            golden.as_ref().map(|g| &g["pixel_values"]["sample"]),
            &out_dir.join(format!("{base}.pixel_values.npy")),
        )?,
        run: PreprocessRun { preprocess_ms, repeat_ms, header: format!("{header:?}") },
    };
    std::fs::write(out, serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("writing {}", out.display()))?;
    eprintln!("wrote {}", out.display());
    Ok(pv)
}

/// Comparisons 3 and 4: the prompt of a forward golden through the language
/// model, with its image (lily's preprocessing at the golden's cap, the
/// tower, the positions, the placeholder override) or as the text-only
/// control, recorded at the golden's top-8 positions and over its greedy
/// continuation.
fn forward(cli: &Cli, model_dir: &Path, golden_path: &Path) -> Result<()> {
    let golden = read_json(golden_path)?;
    ensure!(
        golden["kind"] == "forward",
        "{} is not a forward golden",
        golden_path.display()
    );
    let tokens: Vec<u32> = golden["prompt_token_ids"]
        .as_array()
        .context("prompt_token_ids")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as u32).context("token id"))
        .collect::<Result<_>>()?;
    let ints = |v: &serde_json::Value| -> Result<Vec<usize>> {
        v.as_array()
            .context("integer list")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).context("integer"))
            .collect()
    };
    let dump_positions: Vec<usize> = golden["top8_last"]
        .as_array()
        .context("top8_last")?
        .iter()
        .map(|e| e["position"].as_u64().map(|p| p as usize).context("position"))
        .collect::<Result<_>>()?;
    let greedy_steps = golden["greedy"].as_array().map_or(0, Vec::len);

    let ctx = MetalContext::new()?;
    let started = Instant::now();
    // The model without its tower copy; the tower is loaded on its own below
    // so the probe drives it directly.
    let model = Qwen4ExpModel::load_with(
        &ctx,
        model_dir,
        NgramStorage::default(),
        true,
        VisionMode::Off,
    )?;
    let generator = Generator::from_model_dir(model_dir)?;

    let (probe, positions_record, image_info, load_seconds) = if golden["positions"]
        .is_null()
    {
        ensure!(
            cli.image.is_none(),
            "{} is the text-only control; drop --image",
            golden_path.display()
        );
        let load_seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "model loaded in {load_seconds:.2}s; text-only prompt of {} tokens",
            tokens.len()
        );
        let probe = forward_probe(
            &ctx,
            &model,
            &tokens,
            None,
            &dump_positions,
            greedy_steps,
            8,
        )?;
        (probe, None, None, load_seconds)
    } else {
        let image_path = cli.image.as_ref().with_context(|| {
            format!("{} has an image; pass --image <file>", golden_path.display())
        })?;
        let pb = &golden["positions"];
        let grid = ints(&pb["image_grid_thw"][0])?;
        ensure!(
            grid.len() == 3 && grid[0] == 1,
            "image_grid_thw {grid:?} is not one still image"
        );
        let span = ImageSpan {
            start: pb["image_span"]["start"].as_u64().context("image_span.start")?
                as usize,
            len: pb["image_span"]["length"].as_u64().context("image_span.length")?
                as usize,
            grid_h: grid[1],
            grid_w: grid[2],
        };
        let cap = &golden["meta"]["cap"];
        let limits = ImageLimits {
            max_pixels: cap["max_pixels"].as_u64().context("meta.cap.max_pixels")?
                as usize,
            min_pixels: cap["min_pixels"].as_u64().context("meta.cap.min_pixels")?
                as usize,
            ..ImageLimits::default()
        };
        let bytes = std::fs::read(image_path)
            .with_context(|| format!("reading {}", image_path.display()))?;
        let header = image::read_header(&bytes)?;
        let pv = image::preprocess(&bytes, &limits)?;
        ensure!(
            (pv.grid_h, pv.grid_w) == (span.grid_h, span.grid_w),
            "lily's grid ({}, {}) does not match the golden's ({}, {})",
            pv.grid_h,
            pv.grid_w,
            span.grid_h,
            span.grid_w
        );
        let mut tower = VisionTower::load_from_dir(&ctx, model_dir)?;
        let load_seconds = started.elapsed().as_secs_f64();
        let out = tower.forward(&ctx, &pv.data, pv.grid_h, pv.grid_w)?;
        eprintln!(
            "model and tower loaded in {load_seconds:.2}s; {} {} x {} -> grid ({}, {}), {} image tokens, tower {:.1} ms; prompt of {} tokens",
            header.format,
            header.width,
            header.height,
            pv.grid_h,
            pv.grid_w,
            span.len,
            out.gpu_secs * 1e3,
            tokens.len()
        );

        // Comparison 3, exactly, before the forward: the positions the
        // reference's language model saw.
        let positions = positions_for_prompt(&tokens, &[span])?;
        let axes: Vec<Vec<usize>> = pb["position_ids"]
            .as_array()
            .context("position_ids")?
            .iter()
            .map(ints)
            .collect::<Result<_>>()?;
        let position_ids: Vec<Vec<u32>> =
            (0..3).map(|a| positions.rows.iter().map(|r| r[a]).collect()).collect();
        let same_ids = axes.len() == 3
            && (0..3).all(|a| {
                axes[a].len() == position_ids[a].len()
                    && axes[a]
                        .iter()
                        .zip(&position_ids[a])
                        .all(|(x, y)| *x == *y as usize)
            });
        let same_delta = pb["rope_deltas"].as_i64() == Some(positions.rope_delta);
        eprintln!(
            "positions: {} against the golden, rope_delta {} ({})",
            if same_ids { "exact" } else { "DIFFERENT" },
            positions.rope_delta,
            if same_delta { "same" } else { "DIFFERENT" }
        );
        ensure!(same_ids && same_delta, "lily's positions differ from the reference's");

        let images = [ImageEmbeds { span, rows: &out.merged }];
        let vision = VisionInput { positions: &positions, images: &images };
        let probe = forward_probe(
            &ctx,
            &model,
            &tokens,
            Some(&vision),
            &dump_positions,
            greedy_steps,
            8,
        )?;
        let record = PositionsRecord {
            image_span: span,
            position_ids,
            rope_deltas: positions.rope_delta,
        };
        let info = serde_json::json!({
            "file": image_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "sha256": hex(&sha256(&bytes)),
            "width": header.width,
            "height": header.height,
            "resized_wh": [pv.resized.0, pv.resized.1],
            "image_grid_thw": [[1, pv.grid_h, pv.grid_w]],
            "cap": {"min_pixels": limits.min_pixels, "max_pixels": limits.max_pixels},
            "tower_gpu_ms": out.gpu_secs * 1e3,
        });
        (probe, Some(record), Some(info), load_seconds)
    };
    let chosen: Vec<u32> = probe.steps.iter().map(|s| s.chosen).collect();
    let text = generator.decode_text(&chosen[chosen.len() - greedy_steps - 1..])?;
    eprintln!(
        "prefill of the full prompt {:.3}s, {greedy_steps} greedy steps {:.3}s; chosen {:?}; text {text:?}",
        probe.prefill_seconds, probe.decode_seconds, chosen
    );
    let record = ForwardRecord {
        model_id: "Qwen3.8-Flash-Next",
        prompt_token_ids: tokens,
        steps: probe.steps,
        text,
        positions: positions_record,
        load_seconds,
        prefill_seconds: probe.prefill_seconds,
        decode_seconds: probe.decode_seconds,
        image: image_info,
        golden: golden_path.display().to_string(),
    };
    std::fs::write(&cli.out, serde_json::to_vec_pretty(&record)?)
        .with_context(|| format!("writing {}", cli.out.display()))?;
    eprintln!("wrote {}", cli.out.display());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.image.is_some() || cli.model.is_some(),
        "nothing to do: pass --image to preprocess, --model to run the tower, or both"
    );
    if let Some(golden) = &cli.forward {
        let model = cli.model.as_ref().context("--forward needs --model")?;
        return forward(&cli, model, golden);
    }
    let out_dir = cli.out.parent().map(Path::to_path_buf).unwrap_or_default();
    let base = cli.out.file_stem().context("out name")?.to_string_lossy().into_owned();

    // Comparison 1: lily's own preprocessing.
    let preprocessed = match &cli.image {
        Some(image_path) => {
            let pre_out = cli.preprocess_out.clone().unwrap_or_else(|| {
                if cli.model.is_some() {
                    out_dir.join(format!("{base}.preprocess.json"))
                } else {
                    cli.out.clone()
                }
            });
            Some(preprocess(&cli, image_path, &pre_out)?)
        }
        None => None,
    };
    let Some(model) = &cli.model else {
        return Ok(());
    };

    // Comparison 2: the tower, on lily's pixel rows or the golden's.
    let Some(golden_path) = &cli.golden else {
        bail!(
            "--model needs --golden (a hf_vision_tower_<img>_cap<max>.json) for the grid and sample indices"
        );
    };
    let golden = read_json(golden_path)?;
    ensure!(
        golden["kind"] == "tower",
        "{} is not a tower golden",
        golden_path.display()
    );
    let grid = &golden["input"]["image_grid_thw"][0];
    let (gh, gw) = (
        grid[1].as_u64().context("grid height")? as usize,
        grid[2].as_u64().context("grid width")? as usize,
    );
    let (pixels_data, pixels_shape, pixels_source) = match &preprocessed {
        Some(pv) => {
            ensure!(
                (pv.grid_h, pv.grid_w) == (gh, gw),
                "lily's grid ({}, {}) does not match the golden's ({gh}, {gw}); is the cap the golden's?",
                pv.grid_h,
                pv.grid_w
            );
            (
                pv.data.clone(),
                vec![pv.patches(), image::PATCH_DIM],
                format!("lily preprocess of {}", cli.image.as_ref().unwrap().display()),
            )
        }
        None => {
            let pixels_path = match &cli.pixels {
                Some(p) => p.clone(),
                None => {
                    let stem = golden_path
                        .file_stem()
                        .context("golden name")?
                        .to_string_lossy();
                    let rest = stem.strip_prefix("hf_vision_tower_").with_context(|| {
                        format!("{stem} is not hf_vision_tower_<img>_cap<max>; pass --pixels")
                    })?;
                    golden_path
                        .parent()
                        .context("golden dir")?
                        .join("large")
                        .join(format!("preprocess_{rest}.pixel_values.npy"))
                }
            };
            let pixels = npy::read_f32(&pixels_path).with_context(|| {
                format!(
                    "{}: regenerate with hf_vision_reference.py preprocess (see tools/README.md)",
                    pixels_path.display()
                )
            })?;
            ensure!(
                pixels.shape == [gh * gw, pixels.shape.get(1).copied().unwrap_or(0)],
                "pixel_values shape {:?} does not match the golden grid ({gh}, {gw})",
                pixels.shape
            );
            (pixels.data, pixels.shape, pixels_path.display().to_string())
        }
    };
    let digest = hex(&sha256(&f32_bytes(&pixels_data)));
    if golden["input"]["pixel_values_sha256"] != digest {
        eprintln!(
            "note: pixel_values differ from the golden's input (sha256 {digest})"
        );
    }

    let ctx = MetalContext::new_with_profile(cli.kernel_profile)?;
    let started = Instant::now();
    let mut tower = VisionTower::load_from_dir(&ctx, model)?;
    let load_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "tower loaded in {load_seconds:.2}s ({:.2} GB); grid ({gh}, {gw}), {} patches, {} tokens",
        tower.weights().bytes() as f64 / 1e9,
        gh * gw,
        gh * gw / 4
    );

    let blocks = cli.blocks.unwrap_or(tower.config().depth);
    // Warm-up compiles the pipelines and allocates the scratch.
    tower.forward_blocks(&ctx, &pixels_data, gh, gw, blocks)?;
    let _ = profile::take();
    let (mut gpu_ms, mut host_ms) = (Vec::new(), Vec::new());
    let mut last = None;
    for _ in 0..cli.repeat.max(1) {
        let out = tower.forward_blocks(&ctx, &pixels_data, gh, gw, blocks)?;
        gpu_ms.push(out.gpu_secs * 1e3);
        host_ms.push(out.host_secs * 1e3);
        last = Some(out);
    }
    let out = last.expect("at least one run");
    let kernel_profile_ms = cli.kernel_profile.then(|| {
        let mut by_kernel: BTreeMap<String, f64> = BTreeMap::new();
        if let Some(pass) = profile::take().last() {
            for k in &pass.kernels {
                *by_kernel.entry(k.name.to_string()).or_default() += k.gpu_secs * 1e3;
            }
        }
        by_kernel
    });
    let merged = out.merged.to_f32()?;
    let pre = out.pre_merger.to_f32()?;
    eprintln!(
        "gpu ms {:?}, host ms {:?}, scratch {:.0} MB",
        gpu_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(),
        host_ms.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>(),
        tower.scratch_bytes() as f64 / 1e6
    );
    if let Some(prof) = &kernel_profile_ms {
        let total: f64 = prof.values().sum();
        let mut rows: Vec<_> = prof.iter().collect();
        rows.sort_by(|a, b| b.1.total_cmp(a.1));
        for (name, ms) in rows {
            eprintln!("  {name:<28} {ms:8.2} ms  {:5.1}%", ms / total * 100.0);
        }
        eprintln!("  {:<28} {total:8.2} ms", "total");
    }

    let record = Record {
        kind: "tower",
        golden: golden_path.display().to_string(),
        cap: golden["cap"].clone(),
        input: serde_json::json!({
            "pixel_values_sha256": digest,
            "pixel_values_shape": pixels_shape,
            "image_grid_thw": [[1, gh, gw]],
            "source": pixels_source,
        }),
        run: Run {
            device: "metal",
            dtype: "bf16",
            attention: "fused online softmax (vision_attn_nax_d72)",
            patches: gh * gw,
            grid: [gh, gw],
            blocks,
            load_seconds,
            gpu_ms,
            host_ms,
            scratch_bytes: tower.scratch_bytes(),
            kernel_profile_ms,
        },
        tokens: gh * gw / 4,
        merged: record(
            "merged",
            out.merged.shape(),
            &merged,
            Some(&golden["merged"]["sample"]),
            &out_dir.join(format!("{base}.merged.npy")),
        )?,
        pre_merger: record(
            "pre_merger",
            out.pre_merger.shape(),
            &pre,
            Some(&golden["pre_merger"]["sample"]),
            &out_dir.join(format!("{base}.pre_merger.npy")),
        )?,
    };
    std::fs::write(&cli.out, serde_json::to_vec_pretty(&record)?)?;
    eprintln!("wrote {}", cli.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_sample_covers_the_tensor_when_there_is_no_golden() {
        let (indices, seed, source) = sample_indices(None, 10_000).unwrap();
        assert_eq!(indices.len(), 4096);
        assert_eq!(indices[0], 0);
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
        assert!(*indices.last().unwrap() < 10_000);
        assert_eq!(seed, 0);
        assert!(source.contains("stride"));
        let (small, ..) = sample_indices(None, 10).unwrap();
        assert_eq!(small, (0..10).collect::<Vec<_>>());
    }
}
