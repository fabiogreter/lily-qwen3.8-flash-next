//! Comparison 3 of `docs/vision-support-plan.md`: lily's token numbering for
//! a prompt with an image against the reference's, exactly, on all four
//! `hf_vision_positions_*` goldens (input ids, the three axes, `rope_deltas`),
//! plus the text-only identity and the span validation.

use std::path::Path;

use serde_json::Value;

use super::*;
use crate::kernels::{MROPE_SECTION, mrope_axis};

fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools/reference/goldens")
        .join(name);
    serde_json::from_slice(&std::fs::read(&path).expect("golden file")).expect("json")
}

fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// The golden's span and grid as an [`ImageSpan`], and its positions as the
/// rows lily produces.
fn golden_case(g: &Value) -> (Vec<u32>, ImageSpan, Positions) {
    let ids: Vec<u32> = ints(&g["input_ids"]).into_iter().map(|t| t as u32).collect();
    let grid = ints(&g["image_grid_thw"][0]);
    assert_eq!(grid[0], 1, "still images have one temporal step");
    let span = ImageSpan {
        start: g["image_span"]["start"].as_u64().unwrap() as usize,
        len: g["image_span"]["length"].as_u64().unwrap() as usize,
        grid_h: grid[1],
        grid_w: grid[2],
    };
    let axes: Vec<Vec<usize>> =
        g["position_ids"].as_array().unwrap().iter().map(ints).collect();
    assert_eq!(axes.len(), 3);
    let rows = (0..ids.len())
        .map(|i| [axes[0][i] as u32, axes[1][i] as u32, axes[2][i] as u32])
        .collect();
    let rope_delta = g["rope_deltas"].as_i64().unwrap();
    (ids, span, Positions { rows, rope_delta })
}

#[test]
fn positions_match_the_reference_on_every_golden() {
    for (name, grid, delta) in [
        ("hf_vision_positions_333x777_cap2097152.json", (48, 20), -216),
        ("hf_vision_positions_640x480_cap2097152.json", (30, 40), -280),
        ("hf_vision_positions_1920x1080_cap2097152.json", (68, 120), -1980),
        ("hf_vision_positions_3840x2160_cap2097152.json", (66, 120), -1920),
    ] {
        let g = golden(name);
        let (ids, span, want) = golden_case(&g);
        assert_eq!((span.grid_h, span.grid_w), grid, "{name}: grid");
        assert_eq!(span.len, span.grid_h * span.grid_w / 4, "{name}: span length");
        // The placeholders are exactly the span, and the golden's own
        // `mm_token_type_ids` agree.
        let pads: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == 248056)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(pads, (span.start..span.end()).collect::<Vec<_>>(), "{name}: pads");
        let types = ints(&g["mm_token_type_ids"]);
        for (i, t) in types.iter().enumerate() {
            assert_eq!(
                *t == 1,
                span.start <= i && i < span.end(),
                "{name}: type at {i}"
            );
        }

        let got = positions_for_prompt(&ids, &[span]).expect("positions");
        assert_eq!(got.rows.len(), ids.len(), "{name}: rows");
        for (i, (g, w)) in got.rows.iter().zip(&want.rows).enumerate() {
            assert_eq!(g, w, "{name}: position of token {i}");
        }
        assert_eq!(got.rope_delta, want.rope_delta, "{name}: rope_deltas");
        assert_eq!(got.rope_delta, delta, "{name}: rope_deltas against VISION.md");
        assert_eq!(
            got.rows.iter().flatten().copied().max().unwrap() as u64,
            g["max_position"].as_u64().unwrap(),
            "{name}: max position"
        );
        assert_eq!(
            ids.len() as i64 + got.rope_delta,
            g["next_position"].as_i64().unwrap(),
            "{name}: the first generated token's position"
        );
        assert!(!got.is_identity(), "{name}: an image prompt is not the identity");
    }
}

#[test]
fn the_333x777_golden_follows_the_documented_shape() {
    // VISION.md: text 0..19, `<|vision_start|>` at 19, image rows 20..259
    // with t = 20, h = 20..43, w = 20..29, `<|vision_end|>` at 260 with
    // position 44, max position 65, rope_deltas -216.
    let g = golden("hf_vision_positions_333x777_cap2097152.json");
    let (ids, span, _) = golden_case(&g);
    let p = positions_for_prompt(&ids, &[span]).unwrap();
    assert_eq!(p.rows[19], [19, 19, 19]);
    assert_eq!(p.rows[20], [20, 20, 20]);
    assert_eq!(p.rows[29], [20, 20, 29]);
    assert_eq!(p.rows[30], [20, 21, 20]);
    assert_eq!(p.rows[259], [20, 43, 29]);
    assert_eq!(p.rows[260], [44, 44, 44]);
    assert_eq!(p.rows[281], [65, 65, 65]);
    assert_eq!(p.rope_delta, -216);
    assert_eq!(p.flat(19..21), vec![19, 19, 19, 20, 20, 20]);
}

#[test]
fn a_text_prompt_is_the_identity_with_delta_zero() {
    let g = golden("l4_vision_textonly_tokens.json");
    let ids: Vec<u32> = ints(&g).into_iter().map(|t| t as u32).collect();
    let p = positions_for_prompt(&ids, &[]).unwrap();
    assert_eq!(p.len(), 40);
    assert!(p.is_identity());
    assert_eq!(p.rope_delta, 0);
    assert_eq!(p.rows[39], [39, 39, 39]);

    let empty = positions_for_prompt(&[], &[]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.rope_delta, 0);
}

#[test]
fn two_images_number_like_two_visits_of_the_rule() {
    // text(2) image(4 x 4 -> 4 rows) text(1) image(2 x 6 -> 3 rows) text(1).
    let tokens = vec![1u32; 11];
    let a = ImageSpan { start: 2, len: 4, grid_h: 4, grid_w: 4 };
    let b = ImageSpan { start: 7, len: 3, grid_h: 2, grid_w: 6 };
    let p = positions_for_prompt(&tokens, &[a, b]).unwrap();
    assert_eq!(&p.rows[..2], &[[0, 0, 0], [1, 1, 1]]);
    assert_eq!(&p.rows[2..6], &[[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]);
    // Text resumes at 2 + max(4, 4) / 2 = 4.
    assert_eq!(p.rows[6], [4, 4, 4]);
    assert_eq!(&p.rows[7..10], &[[5, 5, 5], [5, 5, 6], [5, 5, 7]]);
    // Then at 5 + max(2, 6) / 2 = 8.
    assert_eq!(p.rows[10], [8, 8, 8]);
    assert_eq!(p.rope_delta, 9 - 11);
}

#[test]
fn malformed_spans_are_refused() {
    let tokens = vec![0u32; 20];
    let ok = ImageSpan { start: 5, len: 4, grid_h: 4, grid_w: 4 };
    assert!(positions_for_prompt(&tokens, &[ok]).is_ok());
    let err = |spans: &[ImageSpan]| {
        format!("{:#}", positions_for_prompt(&tokens, spans).unwrap_err())
    };
    assert!(err(&[ok, ImageSpan { start: 8, ..ok }]).contains("overlaps"));
    assert!(err(&[ImageSpan { start: 18, ..ok }]).contains("exceeds"));
    assert!(err(&[ImageSpan { len: 5, ..ok }]).contains("expected 4"));
    assert!(err(&[ImageSpan { grid_h: 3, ..ok }]).contains("2 x 2"));
    assert!(err(&[ImageSpan { grid_w: 0, ..ok }]).contains("2 x 2"));
}

#[test]
fn the_interleaved_axis_rule_is_visions_md() {
    // T on 0, 3, ..., 30 (11 pairs); H on 1, 4, ..., 31 (11); W on 2, 5,
    // ..., 29 (10); 32 pairs in all.
    let axes: Vec<usize> = (0..32).map(mrope_axis).collect();
    for (i, &axis) in axes.iter().enumerate() {
        let want = match i % 3 {
            0 => 0,
            1 => usize::from(i < 33),
            _ => {
                if i < 30 {
                    2
                } else {
                    0
                }
            }
        };
        assert_eq!(axis, want, "pair {i}");
    }
    for (axis, &pairs) in MROPE_SECTION.iter().enumerate() {
        assert_eq!(axes.iter().filter(|&&a| a == axis).count(), pairs);
    }
    assert_eq!(MROPE_SECTION.iter().sum::<usize>(), 32);
    // The rule is the same one the config validates the checkpoint against.
    assert_eq!(super::super::config::MROPE_SECTION, &MROPE_SECTION[..]);
}
