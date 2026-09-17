//! Comparison 4 of `docs/architecture.md`, "How the tower was verified", without Python: the
//! four-layer conversion with the tower (`LILY_MODEL_DIR_FLASH`) runs the
//! 333 x 777 prompt through the probe path (lily's own preprocessing, the
//! tower, the positions, the prefill with the override) and its argmax must
//! agree with `hf_l4_vision_333x777_dequant.json` at every compared position
//! (the golden's top-8 positions and its greedy continuation); the text-only
//! control `hf_l4_vision_textonly_dequant.json` goes through the same path.
//! Top-8 overlap and the logit gap are reported, as `compare.py` does.
//!
//! One allowance: a disagreement where the golden's own top two logits are
//! within [`NEAR_TIE`] of each other is reported, not failed. The reference
//! rounds its logits to bf16 and breaks exact ties by the lower id; the text
//! control has such a tie at its last prompt position (17896 and 163670 both
//! at 3.5312), which the text path resolved the other way before this work
//! and, being byte-identical, still does. The greedy continuation is judged
//! only while every earlier position agreed, since after a divergence the
//! two continuations are different prompts.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::*;
use crate::engine::VisionMode;
use crate::metal::MetalContext;
use crate::qwen4exp::image::{self, ImageLimits};
use crate::qwen4exp::model::ImageEmbeds;
use crate::qwen4exp::positions::{ImageSpan, positions_for_prompt};
use crate::qwen4exp::vision::VisionTower;
use crate::qwen4exp::{NgramStorage, Qwen4ExpModel};

fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/reference")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("golden file")).expect("json")
}

fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// Golden top-2 logit gap below which a disagreement is a rounding-level
/// near tie (bf16 at these magnitudes has an ulp of 0.0156).
const NEAR_TIE: f32 = 0.05;

/// `compare.py`'s metrics over the steps a golden can judge.
#[derive(Debug, Default)]
pub(super) struct Agreement {
    pub compared: usize,
    pub agreed: usize,
    /// `(position, lily, golden)` of every disagreement.
    pub mismatches: Vec<(usize, u32, u32)>,
    /// Disagreements at a golden top-2 gap below [`NEAR_TIE`].
    pub near_ties: Vec<usize>,
    /// Greedy steps not judged because an earlier position disagreed.
    pub unjudged_greedy: usize,
    pub min_top8_overlap: Option<usize>,
    pub worst_gap: f32,
}

impl Agreement {
    /// Whether every disagreement is a near tie.
    pub fn passes(&self) -> bool {
        self.mismatches.len() == self.near_ties.len()
    }
}

pub(super) fn compare(steps: &[ProbeStep], golden: &Value) -> Agreement {
    let argmax = ints(&golden["argmax"]);
    let greedy =
        golden.get("greedy").filter(|g| g.is_array()).map(ints).unwrap_or_default();
    let mut top8: std::collections::BTreeMap<usize, (Vec<usize>, Vec<f32>)> =
        Default::default();
    for e in golden["top8_last"].as_array().unwrap() {
        let ids = ints(&e["ids"]);
        let logits =
            e["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32);
        top8.insert(e["position"].as_u64().unwrap() as usize, (ids, logits.collect()));
    }
    let mut a = Agreement::default();
    for step in steps {
        let pos = step.position;
        let want = if pos < argmax.len() {
            argmax[pos]
        } else if pos + 1 - argmax.len() < greedy.len() {
            // The golden's greedy continuation: token k follows position
            // n - 1 + k, as lily's step k does when every earlier one agreed.
            if !a.mismatches.is_empty() {
                a.unjudged_greedy += 1;
                continue;
            }
            greedy[pos + 1 - argmax.len()]
        } else {
            continue;
        };
        a.compared += 1;
        if step.chosen as usize == want {
            a.agreed += 1;
        } else {
            a.mismatches.push((pos, step.chosen, want as u32));
            if let Some((_, logits)) = top8.get(&pos)
                && logits.len() >= 2
                && (logits[0] - logits[1]).abs() < NEAR_TIE
            {
                a.near_ties.push(pos);
            }
        }
        if let Some((ids, logits)) = top8.get(&pos) {
            let shared: Vec<usize> = step
                .ids
                .iter()
                .map(|&i| i as usize)
                .filter(|i| ids.contains(i))
                .collect();
            a.min_top8_overlap =
                Some(a.min_top8_overlap.map_or(shared.len(), |m| m.min(shared.len())));
            for i in shared {
                let ours = step.logits
                    [step.ids.iter().position(|&x| x as usize == i).unwrap()];
                let theirs = logits[ids.iter().position(|&x| x == i).unwrap()];
                a.worst_gap = a.worst_gap.max((ours - theirs).abs());
            }
        }
    }
    a
}

fn top8_positions(golden: &Value) -> Vec<usize> {
    golden["top8_last"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["position"].as_u64().unwrap() as usize)
        .collect()
}

#[test]
#[ignore = "requires LILY_MODEL_DIR_FLASH with the vision tower"]
fn the_image_forward_and_the_text_control_agree_with_the_goldens() {
    let Ok(dir) = std::env::var("LILY_MODEL_DIR_FLASH") else { return };
    let reference = reference_dir();
    let ctx = MetalContext::new().expect("metal");
    let model = Qwen4ExpModel::load_with(
        &ctx,
        &dir,
        NgramStorage::default(),
        true,
        VisionMode::Off,
        None,
        None,
    )
    .expect("model");
    assert!(model.has_mtp(), "the check covers the draft head's catch-up too");

    // The image prompt.
    let golden =
        read_json(&reference.join("goldens/hf_l4_vision_333x777_dequant.json"));
    let tokens: Vec<u32> =
        ints(&golden["prompt_token_ids"]).into_iter().map(|t| t as u32).collect();
    let pb = &golden["positions"];
    let grid = ints(&pb["image_grid_thw"][0]);
    let span = ImageSpan {
        start: pb["image_span"]["start"].as_u64().unwrap() as usize,
        len: pb["image_span"]["length"].as_u64().unwrap() as usize,
        grid_h: grid[1],
        grid_w: grid[2],
    };
    let cap = &golden["meta"]["cap"];
    let limits = ImageLimits {
        max_pixels: cap["max_pixels"].as_u64().unwrap() as usize,
        min_pixels: cap["min_pixels"].as_u64().unwrap() as usize,
        ..ImageLimits::default()
    };
    let image_file = golden["meta"]["image"]["file"].as_str().unwrap();
    let bytes =
        std::fs::read(reference.join("images").join(image_file)).expect("image");
    let pv = image::preprocess(&bytes, &limits).expect("preprocess");
    assert_eq!((pv.grid_h, pv.grid_w), (span.grid_h, span.grid_w), "grid");
    let mut tower = VisionTower::load_from_dir(&ctx, &dir).expect("tower");
    let out =
        tower.forward(&ctx, &pv.data, pv.grid_h, pv.grid_w).expect("tower forward");
    assert_eq!(out.merged.shape(), &[span.len, model.config.hidden_size]);

    // Comparison 3 on the way: the positions the forward golden recorded.
    let positions = positions_for_prompt(&tokens, &[span]).expect("positions");
    let axes: Vec<Vec<usize>> =
        pb["position_ids"].as_array().unwrap().iter().map(ints).collect();
    for (i, row) in positions.rows.iter().enumerate() {
        assert_eq!(
            *row,
            [axes[0][i] as u32, axes[1][i] as u32, axes[2][i] as u32],
            "position of token {i}"
        );
    }
    assert_eq!(positions.rope_delta, pb["rope_deltas"].as_i64().unwrap());
    assert_eq!(positions.rope_delta, golden["model_rope_deltas"].as_i64().unwrap());

    let images = [ImageEmbeds { span, rows: &out.merged }];
    let vision = VisionInput { positions: &positions, images: &images };
    let greedy_steps = golden["greedy"].as_array().unwrap().len();
    let probe = forward_probe(
        &ctx,
        &model,
        &tokens,
        Some(&vision),
        &top8_positions(&golden),
        greedy_steps,
        8,
    )
    .expect("image forward");
    let a = compare(&probe.steps, &golden);
    println!(
        "333 x 777: argmax agreement {}/{}, min top-8 overlap {:?}, worst shared-id logit gap {:.3}, prefill {:.3} s, {} greedy steps in {:.3} s; mismatches {:?} (near ties {:?})",
        a.agreed,
        a.compared,
        a.min_top8_overlap,
        a.worst_gap,
        probe.prefill_seconds,
        greedy_steps,
        probe.decode_seconds,
        a.mismatches,
        a.near_ties
    );
    // The golden's greedy list starts at the last prompt position, which the
    // argmax array already covers, so one fewer greedy step is judged.
    assert_eq!(a.compared, top8_positions(&golden).len() + greedy_steps - 1);
    assert!(a.passes(), "argmax disagreements at {:?}", a.mismatches);

    // The text-only control through the same path.
    let golden =
        read_json(&reference.join("goldens/hf_l4_vision_textonly_dequant.json"));
    assert!(golden["positions"].is_null());
    let tokens: Vec<u32> =
        ints(&golden["prompt_token_ids"]).into_iter().map(|t| t as u32).collect();
    let greedy_steps = golden["greedy"].as_array().unwrap().len();
    let probe = forward_probe(
        &ctx,
        &model,
        &tokens,
        None,
        &top8_positions(&golden),
        greedy_steps,
        8,
    )
    .expect("text forward");
    let a = compare(&probe.steps, &golden);
    println!(
        "text control: argmax agreement {}/{}, min top-8 overlap {:?}, worst shared-id logit gap {:.3}; mismatches {:?} (near ties {:?}), {} greedy steps not judged",
        a.agreed,
        a.compared,
        a.min_top8_overlap,
        a.worst_gap,
        a.mismatches,
        a.near_ties,
        a.unjudged_greedy
    );
    assert_eq!(
        a.compared + a.unjudged_greedy,
        top8_positions(&golden).len() + greedy_steps - 1
    );
    assert!(a.passes(), "argmax disagreements at {:?}", a.mismatches);
}

#[test]
fn top_k_orders_by_logit_then_id() {
    let (ids, logits) = top_k(&[0.5, 2.0, 2.0, -1.0, 3.0], 3);
    assert_eq!(ids, vec![4, 1, 2]);
    assert_eq!(logits, vec![3.0, 2.0, 2.0]);
}

#[test]
fn compare_judges_prompt_positions_and_the_greedy_continuation() {
    let golden = serde_json::json!({
        "argmax": [5, 6, 7],
        "greedy": [7, 8, 9],
        "top8_last": [
            {"position": 1, "ids": [6, 4, 2], "logits": [3.0, 2.99, 1.0]},
            {"position": 2, "ids": [7, 1, 2], "logits": [3.0, 2.0, 1.0]},
        ],
    });
    let step = |position, chosen| ProbeStep {
        position,
        chosen,
        ids: vec![chosen, 1, 9],
        logits: vec![3.25, 1.5, 0.0],
    };
    // Everything agrees but the last greedy step: judged, failed.
    let a =
        compare(&[step(1, 6), step(2, 7), step(3, 8), step(4, 0), step(5, 1)], &golden);
    assert_eq!((a.compared, a.agreed, a.unjudged_greedy), (4, 3, 0));
    assert_eq!(a.mismatches, vec![(4, 0, 9)]);
    assert!(a.near_ties.is_empty() && !a.passes());
    assert_eq!(a.min_top8_overlap, Some(1));
    assert!((a.worst_gap - 0.5).abs() < 1e-6);
    // A near tie at a prompt position passes; the continuation after it is
    // not judged.
    let a = compare(&[step(1, 4), step(2, 7), step(3, 8), step(4, 9)], &golden);
    assert_eq!((a.compared, a.agreed, a.unjudged_greedy), (2, 1, 2));
    assert_eq!(a.mismatches, vec![(1, 4, 6)]);
    assert_eq!(a.near_ties, vec![1]);
    assert!(a.passes());
    // A clear disagreement at a prompt position fails.
    let a = compare(&[step(1, 6), step(2, 1), step(3, 8)], &golden);
    assert_eq!((a.compared, a.unjudged_greedy), (2, 1));
    assert!(!a.passes());
}
