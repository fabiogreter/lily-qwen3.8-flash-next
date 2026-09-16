//! The tower against the reference golden (`docs/vision-support-plan.md`,
//! comparison 2): the four-layer conversion with the tower
//! (`LILY_MODEL_DIR_FLASH`) runs the 333 x 777 `pixel_values` from
//! `tools/reference/goldens/large/` and the committed JSON golden's 4 096
//! sampled merged values are checked within `TOWER_TOLERANCE`, so the check
//! runs from `cargo test` without Python. The `.npy` input is untracked;
//! the test says so and returns when it is absent.

use std::path::Path;

use super::*;
use crate::metal::MetalContext;
use crate::npy;

/// The metrics `compare_vision.py` judges, over one sample.
struct Metrics {
    max_abs: f64,
    max_rel: f64,
    cosine: f64,
    rel_l2: f64,
    fraction_within: f64,
}

fn metrics(candidate: &[f32], golden: &[f64], atol: f64, rtol: f64) -> Metrics {
    assert_eq!(candidate.len(), golden.len());
    let (mut max_abs, mut max_rel, mut within) = (0.0f64, 0.0f64, 0usize);
    let (mut dot, mut cn, mut gn, mut dn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&c, &g) in candidate.iter().zip(golden) {
        let c = c as f64;
        let diff = (c - g).abs();
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(diff / g.abs().max(1e-12));
        within += usize::from(diff <= atol + rtol * g.abs());
        dot += c * g;
        cn += c * c;
        gn += g * g;
        dn += diff * diff;
    }
    Metrics {
        max_abs,
        max_rel,
        cosine: dot / (cn.sqrt() * gn.sqrt()),
        rel_l2: dn.sqrt() / gn.sqrt(),
        fraction_within: within as f64 / golden.len() as f64,
    }
}

fn sample(record: &serde_json::Value) -> (Vec<usize>, Vec<f64>) {
    let s = &record["sample"];
    let indices = s["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let values =
        s["values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    (indices, values)
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH with the vision tower and goldens/large/*.npy"]
fn the_tower_reproduces_the_333x777_golden_sample() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let goldens = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/reference/goldens");
    let pixels_path =
        goldens.join("large/preprocess_333x777_cap2097152.pixel_values.npy");
    if !pixels_path.exists() {
        eprintln!(
            "skipping: {} is absent (untracked); regenerate it with \
             .venv/bin/python tools/reference/hf_vision_reference.py preprocess \
             --src <Qwen3.8-Flash-Next> --image tools/reference/images/333x777.png",
            pixels_path.display()
        );
        return;
    }
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(goldens.join("hf_vision_tower_333x777_cap2097152.json"))
            .expect("golden"),
    )
    .expect("golden json");
    let grid = &golden["input"]["image_grid_thw"][0];
    let (gh, gw) =
        (grid[1].as_u64().unwrap() as usize, grid[2].as_u64().unwrap() as usize);
    let pixels = npy::read_f32(&pixels_path).expect("pixel_values");
    assert_eq!(pixels.shape, vec![gh * gw, 1536]);

    let ctx = MetalContext::new().expect("metal");
    let mut tower = VisionTower::load_from_dir(&ctx, &dir).expect("tower loads");
    let out = tower.forward(&ctx, &pixels.data, gh, gw).expect("forward");
    assert_eq!(out.merged.shape(), &[gh * gw / 4, 2560]);
    assert_eq!(out.pre_merger.shape(), &[gh * gw, 1152]);
    let merged = out.merged.to_f32().expect("merged");
    let pre = out.pre_merger.to_f32().expect("pre-merger");

    let tol = &golden["tolerance"];
    let (atol, rtol) = (tol["atol"].as_f64().unwrap(), tol["rtol"].as_f64().unwrap());
    let (idx, want) = sample(&golden["merged"]);
    let got: Vec<f32> = idx.iter().map(|&i| merged[i]).collect();
    let m = metrics(&got, &want, atol, rtol);
    println!(
        "merged sample: max|err| {:.4}, max rel {:.3}, cosine {:.5}, rel L2 {:.4}, within {:.4}, \
         gpu {:.1} ms, host {:.1} ms",
        m.max_abs,
        m.max_rel,
        m.cosine,
        m.rel_l2,
        m.fraction_within,
        out.gpu_secs * 1e3,
        out.host_secs * 1e3
    );
    let (idx, want) = sample(&golden["pre_merger"]);
    let got: Vec<f32> = idx.iter().map(|&i| pre[i]).collect();
    let p = metrics(&got, &want, atol, rtol);
    println!(
        "pre-merger sample (reported): max|err| {:.3}, cosine {:.5}, rel L2 {:.4}, within {:.4}",
        p.max_abs, p.cosine, p.rel_l2, p.fraction_within
    );

    // TOWER_TOLERANCE (vision_golden.py): the within-fraction threshold is the
    // bf16 reference's own fraction against f32 on this image minus a margin
    // for its run-to-run spread. `vision_tolerance_floors.json`,
    // `tower_bf16_vs_f32.images.333x777.merged_bf16_vs_f32.fraction_within`
    // is 0.9995442708333333; the margin is TOWER_TOLERANCE["frac_margin"],
    // 0.002 (the spread measured there is at most 0.0008).
    const FLOOR_WITHIN_333X777: f64 = 0.9995442708333333;
    const FRAC_MARGIN: f64 = 0.002;
    let min_frac = FLOOR_WITHIN_333X777 - FRAC_MARGIN;
    assert!(
        m.fraction_within >= min_frac,
        "within {} < threshold {min_frac} (floor {FLOOR_WITHIN_333X777} - {FRAC_MARGIN})",
        m.fraction_within
    );
    assert!(m.cosine >= tol["min_cosine"].as_f64().unwrap(), "cosine {}", m.cosine);
    assert!(m.rel_l2 <= tol["max_rel_l2"].as_f64().unwrap(), "rel L2 {}", m.rel_l2);
}
