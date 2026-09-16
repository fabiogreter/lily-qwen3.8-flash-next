//! Runs the vision tower alone over a reference `pixel_values` input and
//! writes its outputs in the golden format, for
//! `tools/reference/compare_vision.py` (docs/vision-support-plan.md,
//! comparison 2) and for timing it.
//!
//! ```sh
//! cargo run --release --bin lily-vision-probe -- \
//!     --model <dir>-l4 \
//!     --golden tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json \
//!     --out /tmp/lily_tower_333x777.json
//! .venv/bin/python tools/reference/compare_vision.py /tmp/lily_tower_333x777.json \
//!     tools/reference/goldens/hf_vision_tower_333x777_cap2097152.json
//! ```
//!
//! The grid comes from the golden; the input is the matching
//! `goldens/large/preprocess_<img>_cap<max>.pixel_values.npy` unless
//! `--pixels` names another. The record carries the golden's own sample
//! indices with the candidate's values at them, so the comparison works with
//! or without the `.npy` files.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use lily::metal::MetalContext;
use lily::metal::profile;
use lily::npy;
use lily::qwen4exp::vision::VisionTower;
use serde::Serialize;

#[derive(Parser)]
#[command(name = "lily-vision-probe", about = "Vision tower probe against a golden")]
struct Cli {
    /// Checkpoint directory carrying the tower (the language model is not loaded).
    #[arg(long)]
    model: PathBuf,
    /// A `hf_vision_tower_<img>_cap<max>.json` golden: grid and sample indices.
    #[arg(long)]
    golden: PathBuf,
    /// The float32 `[N, 1536]` pixel_values `.npy`; defaults to the golden's
    /// matching preprocess file in `goldens/large/`.
    #[arg(long)]
    pixels: Option<PathBuf>,
    /// Where to write the candidate JSON; the `.npy` outputs sit next to it.
    #[arg(long)]
    out: PathBuf,
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

fn record(
    name: &'static str,
    shape: &[usize],
    data: &[f32],
    golden_sample: &serde_json::Value,
    npy_path: &Path,
) -> Result<TensorRecord> {
    let bytes = npy::to_bytes_f32(shape, data)?;
    std::fs::write(npy_path, &bytes)
        .with_context(|| format!("writing {}", npy_path.display()))?;
    let indices: Vec<usize> = golden_sample["indices"]
        .as_array()
        .context("golden sample indices")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).context("sample index"))
        .collect::<Result<_>>()?;
    ensure!(
        indices.iter().all(|&i| i < data.len()),
        "a golden sample index exceeds the {} elements of {name}",
        data.len()
    );
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(TensorRecord {
        name,
        shape: shape.to_vec(),
        dtype: "float32",
        sha256: hex(&sha256(&raw)),
        stats: stats(data),
        sample: Sample {
            seed: golden_sample["seed"].as_u64().unwrap_or(0),
            values: indices.iter().map(|&i| data[i]).collect(),
            indices,
        },
        npy: npy_path.file_name().context("npy name")?.to_string_lossy().into_owned(),
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&cli.golden)
            .with_context(|| format!("reading {}", cli.golden.display()))?,
    )?;
    ensure!(
        golden["kind"] == "tower",
        "{} is not a tower golden",
        cli.golden.display()
    );
    let grid = &golden["input"]["image_grid_thw"][0];
    let (gh, gw) = (
        grid[1].as_u64().context("grid height")? as usize,
        grid[2].as_u64().context("grid width")? as usize,
    );
    let pixels_path = match &cli.pixels {
        Some(p) => p.clone(),
        None => {
            let stem = cli.golden.file_stem().context("golden name")?.to_string_lossy();
            let rest = stem.strip_prefix("hf_vision_tower_").with_context(|| {
                format!("{stem} is not hf_vision_tower_<img>_cap<max>; pass --pixels")
            })?;
            cli.golden
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
    let raw: Vec<u8> = pixels.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let digest = hex(&sha256(&raw));
    if golden["input"]["pixel_values_sha256"] != digest {
        eprintln!(
            "note: pixel_values differ from the golden's input (sha256 {digest})"
        );
    }

    let ctx = MetalContext::new_with_profile(cli.kernel_profile)?;
    let started = std::time::Instant::now();
    let mut tower = VisionTower::load_from_dir(&ctx, &cli.model)?;
    let load_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "tower loaded in {load_seconds:.2}s ({:.2} GB); grid ({gh}, {gw}), {} patches, {} tokens",
        tower.weights().bytes() as f64 / 1e9,
        gh * gw,
        gh * gw / 4
    );

    let blocks = cli.blocks.unwrap_or(tower.config().depth);
    // Warm-up compiles the pipelines and allocates the scratch.
    tower.forward_blocks(&ctx, &pixels.data, gh, gw, blocks)?;
    let _ = profile::take();
    let (mut gpu_ms, mut host_ms) = (Vec::new(), Vec::new());
    let mut last = None;
    for _ in 0..cli.repeat.max(1) {
        let out = tower.forward_blocks(&ctx, &pixels.data, gh, gw, blocks)?;
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

    let out_dir = cli.out.parent().map(Path::to_path_buf).unwrap_or_default();
    let base = cli.out.file_stem().context("out name")?.to_string_lossy().into_owned();
    let record = Record {
        kind: "tower",
        golden: cli.golden.display().to_string(),
        cap: golden["cap"].clone(),
        input: serde_json::json!({
            "pixel_values_sha256": digest,
            "pixel_values_shape": pixels.shape,
            "image_grid_thw": [[1, gh, gw]],
            "npy": pixels_path.display().to_string(),
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
            &golden["merged"]["sample"],
            &out_dir.join(format!("{base}.merged.npy")),
        )?,
        pre_merger: record(
            "pre_merger",
            out.pre_merger.shape(),
            &pre,
            &golden["pre_merger"]["sample"],
            &out_dir.join(format!("{base}.pre_merger.npy")),
        )?,
    };
    std::fs::write(&cli.out, serde_json::to_vec_pretty(&record)?)?;
    eprintln!("wrote {}", cli.out.display());
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 (FIPS 180-4), for the goldens' `sha256` fields; no crate carries
/// it and the dependency list is deliberately short.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    let mut w = [0u32; 64];
    for block in msg.chunks_exact(64) {
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    let mut out = [0u8; 32];
    for (chunk, v) in out.chunks_exact_mut(4).zip(h) {
        chunk.copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_fips_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two blocks, with the length field crossing into the second.
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(&sha256(long)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
}
